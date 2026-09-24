// The MIT License (MIT)
//
// Copyright (c) 2026 Xiaoyang Chen
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software
// and associated documentation files (the "Software"), to deal in the Software without
// restriction, including without limitation the rights to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all copies or
// substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
// BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
// DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

//! The names a model can be asked for, and fetching what they name.
//!
//! `-m` takes either a path to a manifest or a name like `sdxl:base`. A name is looked up in the
//! table below, fetched if it is not in the cache already, and what comes back is a path -- so
//! everything past [`resolve`] works on a file the same way it always did.
//!
//! Every model is published twice, to Hugging Face and to ModelScope, under the same repository
//! name and byte for byte the same files. Which one a fetch uses is [`mirror`]'s to decide, and
//! `WAIFU_MIRROR` overrides it.
//!
//! Only the manifest is named in the table, and it is the first thing fetched: it is a couple of
//! kilobytes, it names every package the weights are in, and reading it is what turns a name into
//! a list of files to go and get. The packages are not written down here as well, because a table
//! that repeated them could come to disagree with the model.
//!
//! This lives in the command line tool rather than in the library: a name is a convenience for
//! someone typing at a terminal, and a program embedding the library has its own ideas about
//! where its files live and when it is allowed to touch the network.

use std::env;
use std::fmt;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::process;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use hf_hub::progress::{DownloadEvent, Progress as Reported, ProgressEvent, ProgressHandler};
use hf_hub::HFClientSync;

use crate::Manifest;

type Error = Box<dyn std::error::Error>;

/// The suffixes a model's files carry, which is what tells a path from a mistyped name.
///
/// A manifest is what `-m` is given; the weights are named here as well because typing one of
/// those is an easy enough mistake to make, and it should be answered as the file it is rather
/// than as a name nobody published.
const MANIFEST_SUFFIX: &str = Manifest::SUFFIX;
const WEIGHTS_SUFFIX: &str = Manifest::WEIGHTS_SUFFIX;

/// Where the cache goes when the environment does not say.
const CACHE_ENV: &str = "WAIFU_CACHE";

/// How much is read from the network at a time.
const DOWNLOAD_BUFFER: usize = 1 << 20;

/// How often a fetch says where it has got to. Short enough that the screen looks alive, long
/// enough that it is not redrawing for a bar that has not moved.
const PROGRESS_INTERVAL: Duration = Duration::from_millis(100);

/// What the directory a Hugging Face fetch is given to itself is called, inside the cache
/// directory of the model it belongs to. On the same filesystem as the package it becomes, so the
/// rename that puts it there is a rename and not a copy.
///
/// A prefix rather than the whole name: every attempt gets a directory of its own -- see
/// [`staging`] -- and what they have in common is this. Deleting the model takes them with it,
/// since they are inside the directory that is deleted.
const INCOMING: &str = ".incoming";

/// Which host a model is fetched from, when the environment names one rather than leaving it to
/// be worked out.
const MIRROR_ENV: &str = "WAIFU_MIRROR";

/// The Hugging Face host, when something other than Hugging Face itself is standing in for it.
///
/// Read here for one reason only: hf-hub reads it, so a fetch goes wherever it points, and a
/// screen that named huggingface.co regardless would be naming a host the fetch never touches.
/// Someone behind a Hugging Face mirror sets this and expects to be told about the mirror.
const HF_ENDPOINT_ENV: &str = "HF_ENDPOINT";

/// Where a Hugging Face fetch goes when [`HF_ENDPOINT_ENV`] does not say. hf-hub's own default,
/// written out again here because the crate does not offer it to be asked for.
const HUGGING_FACE: &str = "https://huggingface.co";

/// What is asked for to find out whether Hugging Face is reachable.
///
/// `generate_204` is the address Android asks to find out whether it is behind a captive portal:
/// it answers with a status and no body at all, so this costs one round trip rather than a page.
/// It is google.com either way, which is the point -- the question is not really about Google but
/// about whether this machine can reach the part of the internet Hugging Face is on.
const REACHABILITY_PROBE: &str = "https://www.google.com/generate_204";

/// How long the probe waits before deciding the answer is no.
///
/// A blocked address does not refuse, it hangs, so this is a guess at how long is worth spending
/// to find that out. Long enough that a slow link is not mistaken for a blocked one, short enough
/// that someone who really is offline is not left staring at nothing before the download they
/// asked for fails anyway.
const PROBE_TIMEOUT: Duration = Duration::from_secs(3);

/// Which package of a model a fetch is on. Carried as one thing because it travels as one, down
/// through the download and into every word it says.
#[derive(Clone, Copy)]
struct Part {
    /// Counting from one.
    at: usize,
    /// How many there are in all, or zero while that is still unknown.
    of: usize,
}

/// Whoever is watching a fetch: where it says how it is getting on, and where it asks whether
/// to carry on.
///
/// One thing rather than two arguments, for the reason [`Part`] above is one thing: they travel
/// together down through the fetch and into every file it brings down. And they belong together
/// -- somebody watching a download of several gigabytes is the same somebody who decides it has
/// gone on long enough.
struct Watching<'a> {
    /// Told how far along the fetch is, on a clock rather than at every byte.
    report: &'a mut dyn FnMut(Progress),
    /// Asked, between one piece of the work and the next, whether to give up.
    stop: &'a dyn Fn() -> bool,
}

impl Watching<'_> {
    fn say(&mut self, progress: Progress) {
        (self.report)(progress);
    }

    /// Whether to stop where this is. Asked at the places a fetch can stop and nowhere else: what
    /// it costs to answer is not this side's to guess at, and a question asked per byte is a
    /// question somebody has to make cheap.
    fn stopping(&self) -> bool {
        (self.stop)()
    }
}

/// What a name resolves to, once fetched: something `-m` draws pictures with, or something
/// `-voice` reads sentences with. The two are never offered in the same picker -- [`listed`] is
/// pictures and [`listed_voices`] is voices -- but they are fetched, cached and named through the
/// one table and the one set of functions, since none of that differs by kind.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Kind {
    Picture,
    Voice,
}

/// A model that has a name, and where it is published.
struct Published {
    /// The name `-m` or `-voice` takes, version and all.
    name: &'static str,
    /// Human-readable name shown in the picker alongside the short name.
    full_name: &'static str,
    /// The Hugging Face repository it lives in.
    repo: &'static str,
    /// The manifest, which is fetched first: it says what the model is and names its packages.
    manifest: &'static str,
    /// Whether what it was trained on means it draws explicit pictures readily, prompted for them
    /// or not. A property of the weights rather than of any one run, which is why it is written
    /// here beside the repository and not worked out from a prompt: the list can say what a model
    /// is before it is chosen, and a screen can leave it out until somebody asks to see it. Only
    /// meaningful for [`Kind::Picture`] -- always `false` on a voice, which [`listed`] never
    /// offers a screen anyway.
    explicit: bool,
    /// Which picker this belongs in.
    kind: Kind,
}

/// Every model this build knows by name.
const CATALOG: &[Published] = &[
    Published {
        name: "sdxl:base:v1.0",
        full_name: "Stable Diffusion XL Base 1.0",
        repo: "ling0322/libwaifu-sdxl-base-1.0",
        manifest: "sdxl-base-1.0.yaml",
        explicit: false,
        kind: Kind::Picture,
    },
    Published {
        name: "sdxl:wai:v17",
        full_name: "WAI Illustrious v17",
        repo: "ling0322/libwaifu-wai-illustrious-v17",
        manifest: "wai-illustrious-v17.yaml",
        explicit: true,
        kind: Kind::Picture,
    },
    Published {
        name: "sdxl:noob:v1.1",
        full_name: "NoobAI XL v1.1",
        repo: "ling0322/libwaifu-noobai-xl-v1.1",
        manifest: "noobai-xl-v1.1.yaml",
        explicit: true,
        kind: Kind::Picture,
    },
    Published {
        name: "sdxl:illust:v2.0",
        full_name: "Illustrious XL v2.0-STABLE",
        repo: "ling0322/libwaifu-illustrious-xl-v2.0",
        manifest: "illustrious-xl-v2.0.yaml",
        explicit: true,
        kind: Kind::Picture,
    },
    Published {
        name: "sdxl:obsession:v24",
        full_name: "One Obsession v24",
        repo: "ling0322/libwaifu-one-obsession-v24",
        manifest: "one-obsession-v24.yaml",
        explicit: true,
        kind: Kind::Picture,
    },
    Published {
        name: "anima:turbo:v1.1",
        full_name: "Anima Turbo v1.1",
        repo: "ling0322/libwaifu-anima-turbo-v1.1",
        manifest: "anima-turbo-v1.1.yaml",
        explicit: true,
        kind: Kind::Picture,
    },
    Published {
        name: "anima:miaomiao:v1.6",
        full_name: "MiaoMiao Harem v1.6",
        repo: "ling0322/libwaifu-miaomiao-harem-v1.6",
        manifest: "miaomiao-harem-v1.6.yaml",
        explicit: true,
        kind: Kind::Picture,
    },
    Published {
        name: "krea2:turbo:v1.0",
        full_name: "Krea 2 Turbo",
        repo: "ling0322/libwaifu-krea2-turbo",
        manifest: "krea2-turbo.yaml",
        explicit: false,
        kind: Kind::Picture,
    },
    // The same weights with the matrices quantized, out of the same repository: half the package
    // and half the card, for about four times the error in the text encoder. A name of its own
    // rather than a flag, because which one is on the disk is what a run has to be told.
    Published {
        name: "krea2:turbo-fp8:v1.0",
        full_name: "Krea 2 Turbo (fp8)",
        repo: "ling0322/libwaifu-krea2-turbo",
        manifest: "krea2-turbo-fp8.yaml",
        explicit: false,
        kind: Kind::Picture,
    },
    // The only voice here so far, and named `<family>:<version>` rather than
    // `<family>:<model>:<version>`: there is no second IndexTTS variant to tell it apart from, so
    // a `model` slot would name nothing.
    Published {
        name: "indextts:v2.5",
        full_name: "IndexTTS 2.5",
        repo: "ling0322/libwaifu-indextts-2.5",
        manifest: "indextts25.yaml",
        explicit: false,
        kind: Kind::Voice,
    },
];

/// Where a package is fetched from.
///
/// Every model is published to both, under the same repository name and byte for byte the same
/// files, so which one is used changes only how long it takes to arrive.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Mirror {
    HuggingFace,
    ModelScope,
}

impl Mirror {
    /// What to call it on screen. Written the way each hub writes its own name.
    fn name(self) -> &'static str {
        match self {
            Mirror::HuggingFace => "Hugging Face",
            Mirror::ModelScope => "ModelScope",
        }
    }

    /// What the environment asks for, if it asks for anything.
    fn named(name: &str) -> Option<Mirror> {
        match name.trim().to_ascii_lowercase().as_str() {
            "huggingface" | "hf" => Some(Mirror::HuggingFace),
            "modelscope" | "ms" => Some(Mirror::ModelScope),
            _ => None,
        }
    }

    /// Where one file of `repo` lives.
    ///
    /// The two differ by host and by what the main branch is called -- Hugging Face's `main`
    /// against ModelScope's `master` -- and in nothing else, because the repositories were named
    /// the same on both.
    ///
    /// The Hugging Face address is built to the shape hf-hub asks for and off the same host it
    /// will use, since hf-hub is what does the asking there now. What it is not is the address
    /// the bytes arrive from: the hub answers a `resolve` with a redirect into storage, so the
    /// last hop is a CDN name nobody typed. This is where the file lives, which is the question.
    fn url(self, repo: &str, file: &str) -> String {
        match self {
            Mirror::HuggingFace => {
                let host = hugging_face_endpoint();
                format!("{host}/{repo}/resolve/main/{file}")
            }
            Mirror::ModelScope => {
                format!("https://modelscope.cn/models/{repo}/resolve/master/{file}")
            }
        }
    }
}

/// The Hugging Face host a fetch will actually be made against.
///
/// A trailing slash is taken off so that joining a path to it does not produce a doubled one: the
/// value is typed by hand into a shell profile, where `https://hf-mirror.com/` is as likely as
/// the same thing without one.
fn hugging_face_endpoint() -> String {
    match env::var(HF_ENDPOINT_ENV) {
        Ok(host) if !host.trim().is_empty() => host.trim().trim_end_matches('/').to_string(),
        _ => HUGGING_FACE.to_string(),
    }
}

/// Which mirror to fetch from, worked out once and then remembered.
///
/// Asked for by name in the environment, or decided by whether google.com answers: where it does
/// not, Hugging Face almost certainly will not either, and ModelScope carries the same files.
/// Deciding it once matters -- a model is several packages, and probing before each would spend
/// the timeout again every time the answer is no.
fn mirror() -> Mirror {
    static CHOSEN: std::sync::OnceLock<Mirror> = std::sync::OnceLock::new();
    *CHOSEN.get_or_init(|| {
        if let Ok(asked) = env::var(MIRROR_ENV) {
            if let Some(mirror) = Mirror::named(&asked) {
                return mirror;
            }
            // A name nobody knows is worth saying something about rather than quietly ignoring,
            // since the whole point of setting it was to be sure which one is used.
            eprintln!(
                "{MIRROR_ENV}={asked:?} names no mirror -- expected \"huggingface\" or \
                 \"modelscope\". Working it out instead."
            );
        }

        if reaches_the_wider_internet() {
            Mirror::HuggingFace
        } else {
            Mirror::ModelScope
        }
    })
}

/// Whether google.com answers, which is what stands in for "Hugging Face is reachable from here".
///
/// It is a guess and not a lookup of where anyone is. Somewhere google.com is blocked and Hugging
/// Face is not, this sends the fetch to ModelScope, which holds the same files and is no worse
/// than being right. Somewhere there is no network at all, it also says no -- and the fetch then
/// fails against ModelScope rather than against Hugging Face, which is the same failure either
/// way. [`MIRROR_ENV`] is there for both.
fn reaches_the_wider_internet() -> bool {
    let agent = ureq::Agent::config_builder()
        .timeout_global(Some(PROBE_TIMEOUT))
        .build()
        .new_agent();

    // Any answer at all is the answer: a redirect or an error page still means the packets got
    // there and came back, which is the whole question. Only nothing coming back is a no.
    agent.get(REACHABILITY_PROBE).call().is_ok()
}

/// Names that follow whatever is current rather than naming a version.
///
/// `sdxl:base` is what someone types when they want the base model and do not care which release
/// of it; it keeps working when a v2 arrives, and `sdxl:base:v1` keeps meaning what it says.
const ALIASES: &[(&str, &str)] = &[
    ("sdxl:base", "sdxl:base:v1.0"),
    ("sdxl:wai", "sdxl:wai:v17"),
    ("sdxl:noob", "sdxl:noob:v1.1"),
    ("sdxl:illust", "sdxl:illust:v2.0"),
    ("sdxl:obsession", "sdxl:obsession:v24"),
    ("anima:turbo", "anima:turbo:v1.1"),
    ("anima:miaomiao", "anima:miaomiao:v1.6"),
    ("krea2:turbo", "krea2:turbo:v1.0"),
    ("krea2:turbo-fp8", "krea2:turbo-fp8:v1.0"),
    ("indextts", "indextts:v2.5"),
];

/// The spellings these names had before a version carried its dot.
///
/// A version is written the way its publisher writes it -- `v1.1` is NoobAI-XL v1.1, where `v11`
/// read as eleven -- but `sdxl:noob:v11` is the name the README handed out, so it is in scripts
/// and in shell history by now. These resolve like any other alias and are left out of `names()`,
/// so what is offered is one spelling of each model rather than two.
const SUPERSEDED: &[(&str, &str)] = &[
    ("sdxl:base:v1", "sdxl:base:v1.0"),
    ("sdxl:noob:v11", "sdxl:noob:v1.1"),
];

/// What a fetch has to say for itself while it runs.
///
/// A download is minutes long and the caller decides how to show it: the command line prints a
/// line that rewrites itself, and the screen draws a bar. Neither belongs in here.
pub enum Progress<'a> {
    /// Which hub the packages are coming from, said once, before the first byte of the first one.
    ///
    /// It is said here and not asked for from outside because this is where it is settled: which
    /// hub is reachable is worked out on the first fetch and not before, so anyone asking earlier
    /// -- a screen drawing a list, say -- would be the one paying for the probe, and would be
    /// asking a question whose answer is not yet true.
    From { hub: &'static str },
    /// Bytes of `file` fetched so far, and how many there are when the server said. `part` says
    /// which package of the model this is, counting from one, and `parts` how many there are in
    /// all -- zero while that is still unknown, which it is until the first package has been read
    /// and named its neighbours.
    Fetching {
        file: &'a str,
        done: u64,
        total: Option<u64>,
        part: usize,
        parts: usize,
    },
    /// `file` is whole and in the cache.
    Fetched {
        file: &'a str,
        bytes: u64,
        part: usize,
        parts: usize,
    },
}

/// What a fetch comes back with when the `stop` it was given said to give up.
///
/// An error rather than a third kind of success. A stop is taken at a dozen places inside a fetch
/// -- before the first byte, between packages, in the middle of one -- and every one of them
/// already knows how to hand a failure back up through `?`; a success that had to be carried
/// through each of them by hand would be the same journey written out a second time. What tells
/// it apart from a failure is [`stopped`], which is the one question a caller asks before it
/// prints the words below as a complaint rather than as an answer.
///
/// The words say what is left on the disk, because that is what somebody who has just stopped a
/// download of several gigabytes wants to know, and it is not the same in both of the places a
/// stop can be taken.
#[derive(Debug)]
pub struct Stopped(&'static str);

impl Stopped {
    /// A stop taken where nothing was left running: between packages, or inside one coming over
    /// plain HTTP, where what has arrived is a `.part` the next fetch carries on from.
    fn kept() -> Self {
        Self("stopped -- what has come down is kept, and fetching it again carries on from there")
    }

    /// A stop taken inside a package coming from Hugging Face, which cannot be called back.
    ///
    /// See [`abandon`]: that package goes on arriving and is kept when it lands. Said out loud
    /// because it is visible from outside the program -- the light on the router does not go out
    /// when the bar does -- and a stop that quietly went on downloading would read as a lie.
    fn kept_and_finishing() -> Self {
        Self(
            "stopped -- what has come down is kept, and the package it was on goes on arriving in \
             the background rather than being thrown away",
        )
    }
}

impl fmt::Display for Stopped {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.0)
    }
}

impl std::error::Error for Stopped {}

/// Whether a fetch ended because it was asked to, rather than because something went wrong.
///
/// The difference is the whole of what a caller does with it: a failure is a complaint on the
/// screen, and a stop is the button working.
pub fn stopped(error: &Error) -> bool {
    error.downcast_ref::<Stopped>().is_some()
}

/// Whether every package of a published model is already in the cache.
///
/// A model is several packages and the first names the rest, so a model that was interrupted
/// partway has its first part and not its others. That reads as not cached, which is what makes
/// the answer worth asking for rather than guessing from one file.
pub fn is_cached(name: &str) -> bool {
    match (published(name), cache_directory()) {
        (Some(published), Ok(cache)) => is_cached_in(published, &cache),
        _ => false,
    }
}

/// The manifest of a model that is already in the cache, or None where it is not all there.
///
/// Nothing is fetched and nothing is opened: this is where the file would be, having checked that
/// it and everything it names are on the disk. For a screen that wants to say what a model is
/// before anybody has asked for the model itself.
pub fn cached_manifest(name: &str) -> Option<PathBuf> {
    let (published, cache) = (published(name)?, cache_directory().ok()?);
    match is_cached_in(published, &cache) {
        true => Some(
            cache
                .join(published.repo.replace('/', "--"))
                .join(published.manifest),
        ),
        false => None,
    }
}

/// How much of a cached model is on disk, for a screen that offers to fetch one.
pub fn cached_bytes(name: &str) -> u64 {
    match (published(name), cache_directory()) {
        (Some(published), Ok(cache)) => cached_bytes_in(published, &cache),
        _ => 0,
    }
}

fn is_cached_in(published: &Published, cache: &Path) -> bool {
    let directory = cache.join(published.repo.replace('/', "--"));
    let manifest = directory.join(published.manifest);
    if !manifest.exists() {
        return false;
    }

    // A manifest that cannot be read for what it names is not a model anyone can draw with,
    // whatever is on disk beside it, so it counts as not there.
    match files_named_by(&manifest) {
        Ok(files) => files.iter().all(|name| directory.join(name).exists()),
        Err(_) => false,
    }
}

fn cached_bytes_in(published: &Published, cache: &Path) -> u64 {
    let directory = cache.join(published.repo.replace('/', "--"));
    let Ok(entries) = fs::read_dir(&directory) else {
        return 0;
    };

    // Everything in the model's own directory, `.part` files included: what this answers is "how
    // much of this is already here", and a resumed fetch does start from what is in the `.part`.
    entries
        .flatten()
        // Followed rather than read off the entry, so that a cache someone has pointed at models
        // with symbolic links measures the models rather than the links.
        .filter_map(|entry| fs::metadata(entry.path()).ok())
        .filter(|meta| meta.is_file())
        .map(|meta| meta.len())
        .sum()
}

/// Throw away what has been fetched of a named model, packages and half-packages alike.
///
/// The whole directory the model was fetched into, which is its own and holds nothing else: the
/// packages, any `.part` a stopped fetch left, and the staging directory hf-hub writes through.
/// A model nothing has been fetched of is not an error to remove -- that is the state being asked
/// for, and it is already the state.
///
/// What is lost is a download and nothing more. Nothing outside the cache is touched, and a
/// package someone exported and keeps elsewhere on the disk is not something this can reach.
pub fn remove(name: &str) -> Result<(), Error> {
    let Some(published) = published(name) else {
        return Err(format!("there is no model called \"{name}\"").into());
    };

    remove_in(published, &cache_directory()?)
}

fn remove_in(published: &Published, cache: &Path) -> Result<(), Error> {
    let directory = cache.join(published.repo.replace('/', "--"));
    match fs::remove_dir_all(&directory) {
        Ok(()) => Ok(()),
        Err(failed) if failed.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(failed) => Err(format!("could not delete {}: {failed}", directory.display()).into()),
    }
}

/// The published model a name refers to, following an alias if it is one.
fn published(name: &str) -> Option<&'static Published> {
    let name = canonical(name);
    CATALOG.iter().find(|model| model.name == name)
}

/// The catalogue name something typed refers to, following an alias if it is one.
fn canonical(name: &str) -> &str {
    ALIASES
        .iter()
        .chain(SUPERSEDED)
        .find(|(alias, _)| *alias == name)
        .map_or(name, |(_, target)| *target)
}

/// Every name that can be asked for, aliases included, for the usage text and for error messages.
pub fn full_name(name: &str) -> Option<&'static str> {
    let versioned = canonical(name);
    CATALOG
        .iter()
        .find(|m| m.name == versioned)
        .map(|m| m.full_name)
}

/// One model a screen can offer, and what is on the disk for it.
///
/// Pictures and voices alike, in two lists rather than one: [`listed`] feeds the picture picker
/// and [`listed_voices`] the voice picker, and a voice in the first would be a name someone could
/// click that fails the moment it is chosen.
pub struct Listed {
    pub name: &'static str,
    pub full_name: &'static str,
    /// Whether every package of it is already in the cache.
    pub cached: bool,
    /// What is on disk for it, which is most of a model for one that was interrupted.
    pub bytes: u64,
    /// Whether it draws explicit pictures readily. Carried out to the screen rather than acted on
    /// here: what is offered is every model, and which of them are shown is the screen's to say.
    pub explicit: bool,
}

/// The picture models to offer, in the order a list should show them.
///
/// Versioned names are left out. Someone who wants `sdxl:base:v1.0` in particular can ask for it
/// by name, and a list is for someone who does not yet know what to ask for.
pub fn listed() -> Vec<Listed> {
    listed_of(Kind::Picture)
}

/// The voices to offer, the same way: `indextts` rather than `indextts:v2.5`.
pub fn listed_voices() -> Vec<Listed> {
    listed_of(Kind::Voice)
}

/// Every unversioned name of one kind, which is every alias: each published model has one, and
/// counting colons would not do -- a voice's versioned name has as many as a picture's alias.
fn listed_of(kind: Kind) -> Vec<Listed> {
    names()
        .into_iter()
        .filter(|name| ALIASES.iter().any(|(alias, _)| alias == name))
        .filter(|name| published(name).is_some_and(|model| model.kind == kind))
        .map(|name| Listed {
            name,
            full_name: full_name(name).unwrap_or(""),
            cached: is_cached(name),
            bytes: cached_bytes(name),
            explicit: published(name).is_some_and(|model| model.explicit),
        })
        .collect()
}

pub fn names() -> Vec<&'static str> {
    let mut names: Vec<&'static str> = ALIASES
        .iter()
        .map(|(alias, _)| *alias)
        .chain(CATALOG.iter().map(|model| model.name))
        .collect();
    names.sort_unstable();
    names
}

/// Whether what was typed reads as a name rather than as a path.
///
/// Only used to choose the error message when neither a name nor a file matches: a Windows path
/// has a colon in it too, so this cannot be what decides that something *is* a name -- matching
/// the table does that.
fn reads_as_a_name(model: &str) -> bool {
    model.contains(':')
        && !model.contains('/')
        && !model.contains('\\')
        && !model.ends_with(MANIFEST_SUFFIX)
        && !model.ends_with(WEIGHTS_SUFFIX)
}

/// Turn what `-m` was given into a manifest on disk, fetching the model if it is a name,
/// telling `report` how it is getting on as it goes.
///
/// A path is taken as it is written. A known name is fetched into the cache, and what comes back
/// is the model's manifest -- the file that says what it is and names its packages, which is what
/// [`Manifest::open`] expects to be handed.
///
/// There is no variant that reports nowhere. A fetch is minutes long, and the one caller there
/// is has a bar to draw it on; somewhere for it to say so is not optional.
///
/// `stop` is asked, between one piece of the work and the next, whether to give up; a fetch that
/// gives up comes back as [`Stopped`]. Asked rather than told, because a stop arrives from
/// whichever thread the button was pressed on and there is nothing here to interrupt: this side
/// is a stack of blocking calls, and where it can look up from them is where it asks.
pub fn resolve_reporting(
    model: &str,
    report: &mut dyn FnMut(Progress),
    stop: &dyn Fn() -> bool,
) -> Result<PathBuf, Error> {
    if let Some(published) = published(model) {
        return fetch(published, &mut Watching { report, stop });
    }

    let path = PathBuf::from(model);
    if path.exists() {
        return Ok(path);
    }

    if reads_as_a_name(model) {
        return Err(format!(
            "there is no model called \"{model}\". The names this build knows are: {}",
            names().join(", ")
        )
        .into());
    }
    Err(format!("model file \"{}\" does not exist", path.display()).into())
}

/// Make sure the manifest of `published` and every file it names is in the cache, and say where
/// the manifest is.
fn fetch(published: &Published, watching: &mut Watching) -> Result<PathBuf, Error> {
    // Before anything is made or asked. Working out which hub is reachable is three seconds of
    // waiting on a machine that can reach neither, and somebody who changed their mind before any
    // of that began is owed a fetch that leaves nothing behind.
    if watching.stopping() {
        return Err(Stopped::kept().into());
    }

    // One directory per repository, so that a model's files sit beside each other: the manifest
    // names them by file name and they are read from its own directory.
    let directory = cache_directory()?.join(published.repo.replace('/', "--"));
    fs::create_dir_all(&directory)?;

    // Before anything is asked for, and only here: settling this is what the probe is for, and
    // doing it now means whoever is watching is told the answer rather than a guess at it.
    watching.say(Progress::From {
        hub: mirror().name(),
    });

    // The manifest first, and without a part number: it is a couple of kilobytes rather than one
    // of the things being counted, and until it has been read there is no count to give.
    let manifest = directory.join(published.manifest);
    download(
        published.repo,
        published.manifest,
        &manifest,
        Part { at: 0, of: 0 },
        watching,
    )?;

    let files = files_named_by(&manifest)?;
    for (index, file) in files.iter().enumerate() {
        let at = directory.join(file);
        let which = Part {
            at: index + 1,
            of: files.len(),
        };
        download(published.repo, file, &at, which, watching)?;
    }
    Ok(manifest)
}

/// Every file a model is made of, as its manifest names them: the weights and the vocabularies.
///
/// The names come out of a file fetched over the network and are about to be joined to a
/// directory and written to, so what a manifest may name is checked before any of it is used: a
/// neighbour of its own and nothing else. [`Manifest::file`] is where that rule lives, and asking
/// for the paths is what applies it -- which is why this goes through the manifest rather than
/// reading the lists itself.
fn files_named_by(manifest: &Path) -> Result<Vec<String>, Error> {
    let manifest = Manifest::open(manifest)?;
    let files = manifest.files();

    for name in &files {
        manifest.file(name)?;
    }
    Ok(files)
}

/// Where fetched models are kept.
///
/// `WAIFU_CACHE` overrides it. Otherwise this is the ordinary cache directory for the platform,
/// which is where something re-downloadable belongs: losing it costs a download and nothing else.
fn cache_directory() -> Result<PathBuf, Error> {
    if let Some(directory) = env::var_os(CACHE_ENV) {
        return Ok(PathBuf::from(directory));
    }

    let base = if cfg!(windows) {
        env::var_os("LOCALAPPDATA").map(PathBuf::from)
    } else {
        env::var_os("XDG_CACHE_HOME")
            .map(PathBuf::from)
            .or_else(|| env::var_os("HOME").map(|home| PathBuf::from(home).join(".cache")))
    };

    let base = base.ok_or_else(|| {
        format!(
            "cannot tell where to keep downloaded models: no home directory in the environment. \
             Set {CACHE_ENV} to a directory to use."
        )
    })?;
    Ok(base.join("libwaifu").join("models"))
}

/// Fetch one file of a repository to `destination`, unless it is already there.
///
/// Which of the two below does the work is which mirror the machine settled on. Neither writes to
/// `destination` until it holds a whole file, so an interrupted fetch is never mistaken for a
/// model: a half written package would otherwise be opened on the next run and fail as a corrupt
/// one, somewhere that cannot point back at the download that truncated it.
fn download(
    repo: &str,
    file: &str,
    destination: &Path,
    part: Part,
    watching: &mut Watching,
) -> Result<(), Error> {
    // Asked once per package, which is what makes a stop between them prompt: the two below are
    // what a stop inside one costs, and neither of them has to be paid to stop before the next
    // one starts.
    if watching.stopping() {
        return Err(Stopped::kept().into());
    }

    // Already fetched, and nothing to say about it: a caller that draws a bar would rather see
    // the bar start at the first file that is actually being fetched.
    if destination.exists() {
        return Ok(());
    }

    match mirror() {
        Mirror::HuggingFace => download_from_hugging_face(repo, file, destination, part, watching),
        Mirror::ModelScope => download_over_http(repo, file, destination, part, watching),
    }
}

/// A staging directory of this attempt's own, inside `directory`.
///
/// One name for all of them would be two downloads writing one file, because a stopped fetch
/// leaves its download running -- see [`abandon`] -- and the next fetch of the same model can
/// begin while it is still going. The process id is in the name for the same reason one attempt's
/// number is: two copies of this program sharing a cache are two sets of attempts.
///
/// What a directory here outlives is the program being killed outright, which is the one ending
/// that runs nothing afterwards. It is inside the model's own directory, so deleting the model --
/// which the screen offers -- takes it too.
fn staging(directory: &Path) -> PathBuf {
    static ATTEMPT: AtomicU64 = AtomicU64::new(0);

    let attempt = ATTEMPT.fetch_add(1, Ordering::Relaxed);
    directory.join(format!("{INCOMING}.{}.{attempt}", process::id()))
}

/// Fetch one file from Hugging Face, through the client Hugging Face publishes.
///
/// Not ureq against a URL built here, which is what this used to be: how that hub answers is a
/// moving target -- a redirect to a CDN, a file stored in xet rather than behind a plain GET, the
/// header that says a file has moved -- and hf-hub is where that knowledge is kept up to date.
///
/// It downloads to a directory of its own choosing within the cache, not to a name of ours, and
/// it writes straight to the file it names rather than to anything like a `.part`. So it is given
/// a directory to itself and what lands there is renamed into place: what is under `destination`
/// is then either a whole package or nothing, which is the promise the rest of this file is
/// written against. What is lost against the ModelScope path below is resuming -- an interrupted
/// fetch starts the package again.
///
/// A stop is taken on the same clock the progress is read on, and costs the package in flight:
/// see [`abandon`] for where that one goes.
fn download_from_hugging_face(
    repo: &str,
    file: &str,
    destination: &Path,
    part: Part,
    watching: &mut Watching,
) -> Result<(), Error> {
    let (owner, name) = repo
        .split_once('/')
        .ok_or_else(|| format!("{repo} is not an owner and a repository name"))?;
    let directory = destination
        .parent()
        .ok_or_else(|| format!("{} is not in a directory", destination.display()))?;
    let staging = staging(directory);

    // The download runs on a thread of its own and this one watches the counters it keeps, rather
    // than the other way about: hf-hub hands progress to a handler that has to be shared across
    // threads, and `report` is a closure belonging to whoever called in here and is not.
    let counted = Arc::new(Counted::default());
    let fetching = {
        let (owner, name, file) = (owner.to_string(), name.to_string(), file.to_string());
        let (staging, counted) = (staging.clone(), Arc::clone(&counted));
        thread::spawn(move || -> Result<(), String> {
            let client = HFClientSync::new().map_err(|failed| failed.to_string())?;
            client
                .model(owner, name)
                .download_file()
                .filename(file)
                .local_dir(staging)
                .progress(Reported::new(Counting(counted)))
                .send()
                .map(|_| ())
                .map_err(|failed| failed.to_string())
        })
    };

    // On the same clock the other path draws on, and for the same reasons: see copy_reporting.
    while !fetching.is_finished() {
        thread::sleep(PROGRESS_INTERVAL);

        // Asked before the line below rather than after it, so that the last thing the bar was
        // told is where the fetch got to rather than where it was a tenth of a second earlier.
        // Nothing is said about the stop here: that is the caller's to say, and it is handed the
        // words for it below.
        if watching.stopping() {
            abandon(
                fetching,
                staging,
                destination.to_path_buf(),
                file.to_string(),
            );
            return Err(Stopped::kept_and_finishing().into());
        }

        let total = counted.total.load(Ordering::Relaxed);
        watching.say(Progress::Fetching {
            file,
            done: counted.done.load(Ordering::Relaxed),
            total: (total > 0).then_some(total),
            part: part.at,
            parts: part.of,
        });
    }
    fetching
        .join()
        .map_err(|_| format!("the fetch of {file} ended without saying why"))??;

    let arrived = staging.join(file);
    fs::rename(&arrived, destination)?;
    // The directory is this attempt's alone, so what is left in it is this attempt's leavings and
    // there is nothing in there to be careful of.
    let _ = fs::remove_dir_all(&staging);

    watching.say(Progress::Fetched {
        file,
        bytes: fs::metadata(destination)
            .map(|meta| meta.len())
            .unwrap_or(0),
        part: part.at,
        parts: part.of,
    });
    Ok(())
}

/// Hands a download that is still running to a thread that waits for it, so that whoever asked
/// for the stop is not the one waiting.
///
/// There is no way to call hf-hub back. The download is a blocking call on a thread of its own,
/// inside an executor of its own, and what it offers to be told along the way is a progress
/// handler that returns nothing. So a stop taken in the middle of a package cannot be a stop to
/// the package: it can only be a stop to waiting for it, which is the part anybody is watching.
///
/// What arrives is then put where it belongs rather than deleted. It is a whole package, the line
/// has already been paid for it, and the alternative is throwing away gigabytes that the next
/// fetch of this model would bring down again. A package that does not arrive -- a failure, or
/// the program ending first -- leaves the staging directory, which goes here with it.
fn abandon(
    fetching: JoinHandle<Result<(), String>>,
    staging: PathBuf,
    destination: PathBuf,
    file: String,
) {
    thread::spawn(move || {
        // A rename only where the download said it finished. Anything else and what is under
        // there is a part of a package, and a part of a package renamed into place is what the
        // whole staging dance exists to prevent.
        if matches!(fetching.join(), Ok(Ok(()))) {
            let _ = fs::rename(staging.join(&file), &destination);
        }
        let _ = fs::remove_dir_all(&staging);
    });
}

/// How many bytes of the file being fetched have arrived, as the thread fetching it counts them.
#[derive(Default)]
struct Counted {
    done: AtomicU64,
    total: AtomicU64,
}

/// Keeps [`Counted`] up to date from what hf-hub says, and does nothing else: drawing anything
/// from a thread that is not the one drawing the screen is how two writers end up in one terminal.
struct Counting(Arc<Counted>);

impl ProgressHandler for Counting {
    fn on_progress(&self, event: &ProgressEvent) {
        let ProgressEvent::Download(event) = event else {
            return;
        };
        match event {
            DownloadEvent::Start { total_bytes, .. } => {
                self.0.total.store(*total_bytes, Ordering::Relaxed);
            }
            // One file is asked for at a time, so the last of these is about that file. A total
            // of zero is hf-hub saying it does not know one, which is not a total.
            DownloadEvent::Progress { files } => {
                if let Some(file) = files.last() {
                    self.0.done.store(file.bytes_completed, Ordering::Relaxed);
                    if file.total_bytes > 0 {
                        self.0.total.store(file.total_bytes, Ordering::Relaxed);
                    }
                }
            }
            // What a file stored in xet reports instead, counted over the batch rather than per
            // file -- which for one file is the same number.
            DownloadEvent::AggregateProgress {
                bytes_completed,
                total_bytes,
                ..
            } => {
                self.0.done.store(*bytes_completed, Ordering::Relaxed);
                self.0.total.store(*total_bytes, Ordering::Relaxed);
            }
            DownloadEvent::Complete => {}
        }
    }
}

/// Fetch one file over plain HTTP, which is how ModelScope is asked.
///
/// The download goes to a `.part` beside the destination and is renamed once it is whole, and a
/// `.part` left behind is resumed rather than restarted -- which is also what a stop leaves
/// behind here, so stopping this one costs nothing but the buffer it was in the middle of.
fn download_over_http(
    repo: &str,
    file: &str,
    destination: &Path,
    part: Part,
    watching: &mut Watching,
) -> Result<(), Error> {
    let url = mirror().url(repo, file);
    let partial = PathBuf::from(format!("{}.part", destination.display()));
    let have = fs::metadata(&partial).map(|meta| meta.len()).unwrap_or(0);

    let mut request = ureq::get(&url);
    if have > 0 {
        request = request.header("Range", &format!("bytes={have}-"));
    }
    let response = request.call()?;

    // Whether what is coming back starts where the last attempt stopped, rather than at the
    // beginning of the file.
    //
    // 206 is how a server is meant to say it took the range. ModelScope takes it and answers 200
    // anyway, with a Content-Range and a body that really does start partway in, so the header is
    // the thing to believe and the status only the fallback. Reading it from the status alone is
    // not a slow download but a corrupt one: the tail arrives, is written from the front of a
    // truncated file, matches the length that came with it, and is renamed into place as whole.
    let content_range = response
        .headers()
        .get("content-range")
        .and_then(|value| value.to_str().ok())
        .map(str::to_string);
    let began_at = content_range.as_deref().and_then(range_begins_at);

    // Only ever asked for `bytes={have}-`, so anything else is a server doing something this does
    // not understand, and guessing at it is how a package gets quietly written wrong.
    if let Some(began_at) = began_at {
        if began_at != have {
            return Err(format!(
                "{file} came back starting at byte {began_at}, but {have} bytes are already here \
                 and that is where it was asked to carry on from. The part that arrived before is \
                 kept: delete it to fetch the file from the beginning."
            )
            .into());
        }
    }

    let resuming = have > 0 && (began_at == Some(have) || response.status().as_u16() == 206);
    let remaining = response
        .headers()
        .get("content-length")
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.parse::<u64>().ok());
    let total = remaining.map(|length| length + if resuming { have } else { 0 });

    let mut writer = if resuming {
        let mut file = OpenOptions::new().write(true).open(&partial)?;
        file.seek(SeekFrom::End(0))?;
        file
    } else {
        File::create(&partial)?
    };

    let start = if resuming { have } else { 0 };
    let mut reader = response.into_body().into_reader();
    let written = copy_reporting(&mut reader, &mut writer, file, start, total, part, watching)?;
    writer.sync_all()?;
    drop(writer);

    // A truncated transfer that still ended cleanly would otherwise be renamed into place and
    // read as a corrupt package later, where nothing points back at the download.
    if let Some(total) = total {
        if written != total {
            return Err(format!(
                "{file} arrived incomplete: {written} bytes of {total}. The part that did arrive \
                 is kept, so running this again resumes it."
            )
            .into());
        }
    }

    fs::rename(&partial, destination)?;
    Ok(())
}

/// Which byte of the whole file a `Content-Range` header says its body begins at.
///
/// The header reads `bytes 1000000-1062187849/1062187850`. Only the first number is wanted: the
/// last says where this piece ends and the one past the slash how long the file is, and both are
/// already known from elsewhere. `bytes */1234`, which is what a server sends when it is refusing
/// a range rather than answering one, has no beginning and reads as none.
fn range_begins_at(header: &str) -> Option<u64> {
    header
        .trim()
        .strip_prefix("bytes")?
        .trim_start()
        .split('-')
        .next()?
        .trim()
        .parse()
        .ok()
}

/// Copy the body across, saying how it is going on the way.
///
/// The line is rewritten in place rather than added to, because this runs on the terminal the
/// drawing screen is about to take over and a screenful of progress lines is not worth keeping.
///
/// A stop here comes back as [`Stopped`] rather than as a short count, which is the difference
/// between the caller keeping the `.part` and the caller complaining that the file arrived
/// incomplete. What has been written stays written: `writer` is the file itself and not a buffer
/// over it, so the bytes that were read are on the disk and the next fetch carries on from them.
fn copy_reporting(
    reader: &mut impl Read,
    writer: &mut impl Write,
    name: &str,
    start: u64,
    total: Option<u64>,
    part: Part,
    watching: &mut Watching,
) -> Result<u64, Error> {
    let mut buffer = vec![0u8; DOWNLOAD_BUFFER];
    let mut done = start;

    // Said before the first byte, because the gap between one package finishing and the next
    // showing a number is a stretch where the screen would otherwise hold a full bar and look
    // hung. This is what moves it to the new file at nothing.
    watching.say(Progress::Fetching {
        file: name,
        done,
        total,
        part: part.at,
        parts: part.of,
    });
    let mut said = Instant::now();

    loop {
        let read = reader.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        writer.write_all(&buffer[..read])?;
        done += read as u64;

        // After the write rather than before the read, so that a stop is taken with the buffer
        // that was in flight on the disk rather than dropped. A megabyte at a time, which is as
        // long as a stop here ever waits.
        if watching.stopping() {
            return Err(Stopped::kept().into());
        }

        // On a clock rather than on a fraction of the file. A percent of two gigabytes is twenty
        // megabytes, which is seconds of silence on an ordinary line and reads as a hang; and on
        // a fast one it is several redraws a second for a bar that moved a pixel. Time is what
        // the watcher actually measures the wait in.
        if said.elapsed() >= PROGRESS_INTERVAL {
            said = Instant::now();
            watching.say(Progress::Fetching {
                file: name,
                done,
                total,
                part: part.at,
                parts: part.of,
            });
        }
    }

    watching.say(Progress::Fetched {
        file: name,
        bytes: done,
        part: part.at,
        parts: part.of,
    });
    Ok(done)
}

#[cfg(test)]
mod tests {
    use hf_hub::progress::{FileProgress, FileStatus};

    use super::*;

    /// [`resolve_reporting`] with nothing listening and nothing stopping it, for the tests that
    /// are about what it refuses rather than about what it fetches.
    fn quietly(model: &str) -> Result<PathBuf, Error> {
        resolve_reporting(model, &mut |_| (), &|| false)
    }

    #[test]
    fn a_name_finds_what_it_names() {
        let model = published("sdxl:base:v1.0").expect("the catalog has it");
        assert_eq!(model.repo, "ling0322/libwaifu-sdxl-base-1.0");
        assert!(model.manifest.ends_with(MANIFEST_SUFFIX));
    }

    #[test]
    fn an_alias_follows_what_it_points_at() {
        let alias = published("sdxl:base").expect("an alias resolves");
        let target = published("sdxl:base:v1.0").expect("and so does what it points at");
        assert_eq!(alias.name, target.name);
        assert_eq!(alias.repo, target.repo);
    }

    #[test]
    fn every_alias_points_at_something_real() {
        for (alias, target) in ALIASES {
            assert!(
                CATALOG.iter().any(|model| model.name == *target),
                "{alias} points at {target}, which is not in the catalog"
            );
        }
    }

    #[test]
    fn each_mirror_names_the_same_file_its_own_way() {
        let model = published("sdxl:base:v1.0").expect("the catalog has it");
        let hf = Mirror::HuggingFace.url(model.repo, model.manifest);
        let ms = Mirror::ModelScope.url(model.repo, model.manifest);

        // Written out against the default host, so somewhere HF_ENDPOINT points at a mirror this
        // is a different string for a good reason; the test below is the one for that case.
        if env::var_os(HF_ENDPOINT_ENV).is_none() {
            assert_eq!(
                hf,
                "https://huggingface.co/ling0322/libwaifu-sdxl-base-1.0/resolve/main/\
                 sdxl-base-1.0.yaml"
            );
        }
        assert_eq!(
            ms,
            "https://modelscope.cn/models/ling0322/libwaifu-sdxl-base-1.0/resolve/master/\
             sdxl-base-1.0.yaml"
        );

        // The branch names differ, which is the easy thing to get wrong when copying one line to
        // make the other: Hugging Face's main against ModelScope's master.
        assert!(hf.contains("/resolve/main/"));
        assert!(ms.contains("/resolve/master/"));
    }

    #[test]
    fn the_hugging_face_address_follows_the_hub_the_fetch_will_use() {
        // hf-hub fetches from HF_ENDPOINT where it is set, so a screen naming huggingface.co
        // regardless would be naming a host nothing is asked of. Set for this test only.
        let previous = env::var_os(HF_ENDPOINT_ENV);
        let model = published("sdxl:base:v1.0").expect("the catalog has it");

        env::set_var(HF_ENDPOINT_ENV, "https://hf-mirror.com");
        assert_eq!(
            Mirror::HuggingFace.url(model.repo, model.manifest),
            "https://hf-mirror.com/ling0322/libwaifu-sdxl-base-1.0/resolve/main/\
             sdxl-base-1.0.yaml"
        );

        // Typed with the slash a host name invites, and it makes no difference.
        env::set_var(HF_ENDPOINT_ENV, "https://hf-mirror.com/");
        assert!(Mirror::HuggingFace
            .url(model.repo, model.manifest)
            .starts_with("https://hf-mirror.com/ling0322/"));

        // Set to nothing at all is the same as not set: an emptied variable is someone turning
        // the mirror off, not asking for a host with no name.
        env::set_var(HF_ENDPOINT_ENV, "");
        assert_eq!(hugging_face_endpoint(), HUGGING_FACE);

        match previous {
            Some(value) => env::set_var(HF_ENDPOINT_ENV, value),
            None => env::remove_var(HF_ENDPOINT_ENV),
        }
    }

    #[test]
    fn every_model_can_be_asked_for_from_either_mirror() {
        // Both mirrors carry every model under the same repository name, so a catalog entry that
        // only exists on one side would be a fetch that works for some people and not others.
        for model in CATALOG {
            for mirror in [Mirror::HuggingFace, Mirror::ModelScope] {
                let url = mirror.url(model.repo, model.manifest);
                assert!(url.starts_with("https://"), "{url}");
                assert!(url.contains(model.repo), "{url}");
                assert!(url.ends_with(model.manifest), "{url}");
            }
        }
    }

    #[test]
    fn what_the_hub_says_about_a_fetch_becomes_a_count_of_bytes() {
        let counted = Arc::new(Counted::default());
        let counting = Counting(Arc::clone(&counted));

        // The size comes with the start of the fetch, before a byte of the file has.
        counting.on_progress(&ProgressEvent::Download(DownloadEvent::Start {
            total_files: 1,
            total_bytes: 2_020_000_000,
        }));
        assert_eq!(counted.done.load(Ordering::Relaxed), 0);
        assert_eq!(counted.total.load(Ordering::Relaxed), 2_020_000_000);

        counting.on_progress(&ProgressEvent::Download(DownloadEvent::Progress {
            files: vec![FileProgress {
                filename: "sdxl-base-1.0-00001-of-00004.waifupkg".to_string(),
                bytes_completed: 1_010_000_000,
                total_bytes: 2_020_000_000,
                status: FileStatus::InProgress,
            }],
        }));
        assert_eq!(counted.done.load(Ordering::Relaxed), 1_010_000_000);

        // A size of zero is hf-hub saying it does not know one, and must not be read as a file
        // that is zero bytes long: the bar would jump to full and stay there.
        counting.on_progress(&ProgressEvent::Download(DownloadEvent::Progress {
            files: vec![FileProgress {
                filename: "sdxl-base-1.0-00001-of-00004.waifupkg".to_string(),
                bytes_completed: 1_500_000_000,
                total_bytes: 0,
                status: FileStatus::InProgress,
            }],
        }));
        assert_eq!(counted.done.load(Ordering::Relaxed), 1_500_000_000);
        assert_eq!(counted.total.load(Ordering::Relaxed), 2_020_000_000);

        // What a file kept in xet reports instead, which is counted over the whole batch.
        counting.on_progress(&ProgressEvent::Download(DownloadEvent::AggregateProgress {
            bytes_completed: 1_900_000_000,
            total_bytes: 2_020_000_000,
            bytes_per_sec: Some(15e6),
        }));
        assert_eq!(counted.done.load(Ordering::Relaxed), 1_900_000_000);
        assert_eq!(counted.total.load(Ordering::Relaxed), 2_020_000_000);
    }

    /// Ignored because it asks the real Hugging Face for a real file. Run it with
    /// `cargo test --features cli -- --ignored` after touching the fetch.
    #[test]
    #[ignore]
    fn a_file_fetched_from_hugging_face_lands_where_it_was_asked_for() {
        let directory = env::temp_dir().join(format!("libwaifu-fetch-{}", std::process::id()));
        fs::create_dir_all(&directory).expect("somewhere to fetch into");
        let destination = directory.join("config.json");

        // Something small and public that is not going anywhere, since what is under test is the
        // plumbing and not the file.
        let mut finished = false;
        download_from_hugging_face(
            "hf-internal-testing/tiny-random-gpt2",
            "config.json",
            &destination,
            Part { at: 1, of: 1 },
            &mut Watching {
                report: &mut |progress| finished |= matches!(progress, Progress::Fetched { .. }),
                stop: &|| false,
            },
        )
        .expect("the file is there and so is the network");

        // Under the name it was asked for, whole, and with nothing left in the way.
        assert!(destination.exists(), "{}", destination.display());
        assert!(fs::metadata(&destination).expect("it is a file").len() > 0);
        assert!(!staged_in(&directory), "the staging is cleared");

        // And it said so when it finished, which is what stops the bar.
        assert!(finished, "the fetch said nothing about finishing");

        let _ = fs::remove_dir_all(&directory);
    }

    /// Whether any staging directory is left in `directory`, under whichever attempt named it.
    fn staged_in(directory: &Path) -> bool {
        fs::read_dir(directory)
            .expect("a directory to look in")
            .filter_map(Result::ok)
            .any(|entry| entry.file_name().to_string_lossy().starts_with(INCOMING))
    }

    #[test]
    fn a_stopped_download_keeps_what_arrived_and_says_it_was_stopped() {
        // What the body of a download looks like from in here: bytes, arriving a read at a time.
        let body = vec![7u8; DOWNLOAD_BUFFER * 3];
        let mut written = Vec::new();

        // Stopped as the second buffer lands, which is what a click in the middle of a package of
        // several gigabytes is.
        let asked = AtomicU64::new(0);
        let failed = copy_reporting(
            &mut body.as_slice(),
            &mut written,
            "second.safetensors",
            0,
            Some(body.len() as u64),
            Part { at: 2, of: 3 },
            &mut Watching {
                report: &mut |_| (),
                stop: &|| asked.fetch_add(1, Ordering::Relaxed) > 0,
            },
        )
        .expect_err("it was asked to stop");

        // A stop, and not the kind of ending anybody needs to be shown a complaint about. What it
        // says is what is left behind, which is the question somebody who just stopped a download
        // is about to ask.
        assert!(stopped(&failed), "{failed}");
        assert!(failed.to_string().contains("kept"), "{failed}");

        // And what had arrived by then is written rather than dropped: this is the `.part`, and
        // the fetch after it asks the server to carry on from where this one got to.
        assert_eq!(written.len(), DOWNLOAD_BUFFER * 2);
    }

    #[test]
    fn a_fetch_stopped_before_it_starts_asks_the_network_for_nothing() {
        // Nothing is asked of the network here, and nothing is made on the disk: the stop is
        // taken before the probe that settles which hub to ask, which is the point. That probe is
        // three seconds of waiting, and it is not worth spending on a fetch already called off.
        let failed =
            resolve_reporting("sdxl:base", &mut |_| (), &|| true).expect_err("it was stopped");
        assert!(stopped(&failed), "{failed}");

        // Whatever else an ending means, a stop does not mean the name was wrong.
        let refused = quietly("sdxl:nope").expect_err("no such model");
        assert!(!stopped(&refused), "{refused}");
    }

    #[test]
    fn two_attempts_at_one_model_stage_through_different_directories() {
        // Why there is a directory per attempt: a stopped fetch leaves its download running, and
        // the fetch that starts after it must not be writing the same file.
        let directory = env::temp_dir().join("waifu-staging");
        assert_ne!(staging(&directory), staging(&directory));

        // Both of them recognisable as the leavings of a fetch, which is what the model's own
        // directory being deleted relies on.
        assert!(staging(&directory)
            .file_name()
            .expect("a name")
            .to_string_lossy()
            .starts_with(INCOMING));
    }

    #[test]
    fn a_content_range_says_where_its_body_starts() {
        // What ModelScope sends, alongside a 200 rather than the 206 it ought to be.
        assert_eq!(
            range_begins_at("bytes 1000000-1062187849/1062187850"),
            Some(1000000)
        );
        assert_eq!(range_begins_at("bytes 0-15/16"), Some(0));

        // A server refusing the range rather than answering it has no beginning to give.
        assert_eq!(range_begins_at("bytes */1062187850"), None);

        // And anything that is not a byte range at all.
        assert_eq!(range_begins_at("items 1-2/3"), None);
        assert_eq!(range_begins_at(""), None);
    }

    #[test]
    fn a_mirror_can_be_asked_for_by_name() {
        assert_eq!(Mirror::named("modelscope"), Some(Mirror::ModelScope));
        assert_eq!(Mirror::named("ModelScope"), Some(Mirror::ModelScope));
        assert_eq!(Mirror::named("  ms "), Some(Mirror::ModelScope));
        assert_eq!(Mirror::named("huggingface"), Some(Mirror::HuggingFace));
        assert_eq!(Mirror::named("HF"), Some(Mirror::HuggingFace));

        // Anything else is not quietly read as one of them.
        assert_eq!(Mirror::named("hugging face"), None);
        assert_eq!(Mirror::named(""), None);
        assert_eq!(Mirror::named("mirror"), None);
    }

    #[test]
    fn each_hub_is_called_what_it_calls_itself() {
        // What a fetch says it is talking to. The screen shows this and nothing else about where
        // the packages come from, so it is the whole of the answer and worth spelling right.
        assert_eq!(Mirror::HuggingFace.name(), "Hugging Face");
        assert_eq!(Mirror::ModelScope.name(), "ModelScope");
    }

    #[test]
    fn a_name_nobody_published_is_not_a_name() {
        assert!(published("sdxl:refiner").is_none());
        assert!(published("sdxl").is_none());
    }

    #[test]
    fn names_are_listed_for_the_usage_text() {
        let names = names();
        assert!(names.contains(&"sdxl:base"));
        assert!(names.contains(&"sdxl:base:v1.0"));
        assert!(names.contains(&"sdxl:wai"));
        assert!(names.contains(&"sdxl:wai:v17"));
        assert!(names.contains(&"sdxl:noob"));
        assert!(names.contains(&"sdxl:noob:v1.1"));
        assert!(names.contains(&"sdxl:illust"));
        assert!(names.contains(&"sdxl:illust:v2.0"));
        assert!(names.contains(&"sdxl:obsession"));
        assert!(names.contains(&"sdxl:obsession:v24"));
        assert!(names.contains(&"anima:turbo"));
        assert!(names.contains(&"anima:turbo:v1.1"));
        assert!(names.contains(&"anima:miaomiao"));
        assert!(names.contains(&"anima:miaomiao:v1.6"));

        // The spellings these replaced are answered but not offered: one name each.
        assert!(!names.contains(&"sdxl:base:v1"));
        assert!(!names.contains(&"sdxl:noob:v11"));
    }

    #[test]
    fn what_a_model_draws_travels_out_with_it() {
        // The list offers the unversioned names, and the label is written beside the versioned
        // one. A lookup that stopped at the alias would report every model as drawing nothing
        // explicit, which is a screen that quietly stops hiding anything.
        let listed = listed();
        let said = |name: &str| {
            listed
                .iter()
                .find(|model| model.name == name)
                .unwrap_or_else(|| panic!("{name} is offered"))
                .explicit
        };

        assert!(said("sdxl:noob"));
        assert!(said("sdxl:wai"));
        assert!(said("sdxl:illust"));
        assert!(said("sdxl:obsession"));
        assert!(said("anima:turbo"));
        assert!(!said("sdxl:base"));
        assert!(!said("krea2:turbo"));

        // And the two names for one set of weights say the same thing about them.
        assert_eq!(said("krea2:turbo"), said("krea2:turbo-fp8"));
    }

    #[test]
    fn a_voice_is_offered_in_the_voice_list_and_not_as_a_picture() {
        // `-voice` resolves a voice exactly the way `-m` resolves a picture -- same table, same
        // functions -- but `listed()` is what feeds the picture picker, and a voice put there
        // would be a name someone could click that fails the moment it is chosen.
        assert!(published("indextts").is_some());
        assert_eq!(full_name("indextts"), Some("IndexTTS 2.5"));
        assert!(names().contains(&"indextts"));

        assert!(
            !listed().iter().any(|model| model.name == "indextts"),
            "a voice is in the picture picker"
        );

        // And the voice list is voices only, under the name the page starts with rather than the
        // versioned one -- which is the name the page compares against to mark it chosen.
        let voices = listed_voices();
        assert_eq!(
            voices.iter().map(|voice| voice.name).collect::<Vec<_>>(),
            ["indextts"]
        );
        assert_eq!(voices[0].full_name, "IndexTTS 2.5");
    }

    #[test]
    fn the_name_a_model_used_to_have_still_finds_it() {
        // `sdxl:noob:v11` went out in the README and is in scripts by now. It has to keep meaning
        // NoobAI-XL v1.1, whatever the catalogue calls that today.
        for (superseded, current) in SUPERSEDED {
            let old = published(superseded).expect("the name it used to have");
            let new = published(current).expect("the name it has now");
            assert_eq!(old.name, new.name, "{superseded} no longer finds {current}");
            assert_eq!(full_name(superseded), full_name(current));
        }
    }

    #[test]
    fn no_two_models_are_the_same_model() {
        // A table entry is written by copying the one above it, so the thing to check is that the
        // copy was finished: no two models share a name, and none shares a first package.
        //
        // The repository is not on that list, because two entries sharing one is a real thing
        // rather than a slip: Krea 2 publishes its float package and its quantized one together,
        // and which of the two a run gets is the manifest it asks for. What must differ is
        // therefore the manifest, and that is checked whether the repository repeats or not.
        for (index, model) in CATALOG.iter().enumerate() {
            for other in &CATALOG[index + 1..] {
                assert_ne!(model.name, other.name);
                assert_ne!(
                    model.manifest, other.manifest,
                    "{} and {}",
                    model.name, other.name
                );
            }
        }
    }

    #[test]
    fn every_model_is_named_the_way_the_others_are() {
        // A picture is `<family>:<model>:<version>` -- there is more than one SDXL, so which one
        // needs its own word. A voice is `<family>:<version>`: only one IndexTTS exists to name,
        // and a `model` slot naming nothing is worse than one field fewer. Cheap to check, and it
        // is the sort of thing a copied table entry gets wrong.
        //
        // The family is one of the kinds this build can fetch a model of rather than anything at
        // all: a name is what someone types before they have the model, so it should say what
        // they are about to fetch. Add to this list when the runtime learns another -- of a
        // picture model or, as `indextts` did, of a voice.
        const FAMILIES: [&str; 4] = ["sdxl", "anima", "krea2", "indextts"];

        for model in CATALOG {
            let fields: Vec<&str> = model.name.split(':').collect();
            let expected = match model.kind {
                Kind::Picture => 3,
                Kind::Voice => 2,
            };
            assert_eq!(
                fields.len(),
                expected,
                "{} is not the shape a {:?} is named",
                model.name,
                model.kind
            );
            assert!(
                FAMILIES.contains(&fields[0]),
                "{} is in no family this build knows: {FAMILIES:?}",
                model.name
            );
            assert!(model.repo.contains('/'), "{} has no namespace", model.repo);
            assert!(
                model.manifest.ends_with(MANIFEST_SUFFIX),
                "{} is not a manifest",
                model.manifest
            );
        }
    }

    #[test]
    fn a_path_is_not_read_as_a_name() {
        // The suffix, a separator, or no colon at all: each is enough to say "this is a file".
        // Either suffix, since typing the weights rather than the manifest is an easy slip and
        // should be answered as the file it is.
        assert!(!reads_as_a_name("sdxl.yaml"));
        assert!(!reads_as_a_name("sdxl.safetensors"));
        assert!(!reads_as_a_name("models/sdxl-base.yaml"));
        assert!(!reads_as_a_name(r"C:\models\sdxl-base.yaml"));
        assert!(!reads_as_a_name("./sdxl"));
        assert!(reads_as_a_name("sdxl:base"));
        assert!(reads_as_a_name("sdxl:typo"));
    }

    #[test]
    fn an_unknown_name_says_what_the_known_ones_are() {
        let error = quietly("sdxl:nope").unwrap_err().to_string();
        assert!(error.contains("no model called"), "{error}");
        assert!(error.contains("sdxl:base"), "{error}");
    }

    #[test]
    fn deleting_a_model_takes_the_whole_of_it() {
        let cache = std::env::temp_dir().join(format!("waifu-delete-{}", std::process::id()));
        let _ = fs::remove_dir_all(&cache);
        let model = published("sdxl:base").expect("the base model");
        let directory = cache.join(model.repo.replace('/', "--"));

        // A model that was fetched and then stopped partway: whole files, a `.part` of the one it
        // was on, and the directory hf-hub stages through. All of it is the model.
        let staging = directory.join(format!("{INCOMING}.4171.0"));
        fs::create_dir_all(&staging).expect("somewhere to put it");
        fs::write(directory.join(model.manifest), b"a manifest").expect("the manifest");
        fs::write(directory.join("second.safetensors.part"), b"half of one")
            .expect("a part file");
        fs::write(staging.join("third"), b"staged").expect("something staged");

        remove_in(model, &cache).expect("it goes");
        assert!(!directory.exists());
        assert_eq!(cached_bytes_in(model, &cache), 0);

        // The cache itself stays, and the models beside this one with it: what was asked for was
        // one model, and the directory above it is not this one's to take.
        assert!(cache.exists());

        // Asked for again, with nothing left to delete. That is the state being asked for and it
        // is already true, so it is not an error -- a second press of the key says nothing new.
        remove_in(model, &cache).expect("nothing to do is not a failure");

        // A name nobody published names no directory, and is refused rather than reaching for one.
        assert!(remove("sdxl:nope").is_err());

        let _ = fs::remove_dir_all(&cache);
    }

    #[test]
    fn a_model_that_is_not_on_disk_is_not_cached() {
        let cache = std::env::temp_dir().join(format!("waifu-cache-{}", std::process::id()));
        let _ = fs::remove_dir_all(&cache);
        let model = published("sdxl:base").expect("the base model");

        // Nothing there at all.
        assert!(!is_cached_in(model, &cache));
        assert_eq!(cached_bytes_in(model, &cache), 0);

        // A manifest that is not one it can read. Half a download looks like this, and it must
        // not read as a model that is ready to draw with -- though the bytes still count, because
        // a resumed fetch starts from them.
        let directory = cache.join(model.repo.replace('/', "--"));
        fs::create_dir_all(&directory).expect("a directory to put it in");
        fs::write(directory.join(model.manifest), b"not yaml").expect("the manifest");

        assert!(!is_cached_in(model, &cache));
        assert_eq!(cached_bytes_in(model, &cache), 8);

        // And a name nobody published is not cached either, rather than a panic.
        assert!(!is_cached("sdxl:nope"));
        assert_eq!(cached_bytes("sdxl:nope"), 0);

        let _ = fs::remove_dir_all(&cache);
    }

    #[test]
    fn a_missing_file_is_reported_as_a_file() {
        let error = quietly("no-such-model.waifupkg").unwrap_err().to_string();
        assert!(error.contains("does not exist"), "{error}");
    }

    #[test]
    fn the_cache_follows_the_environment() {
        // Set for this test only; the point is that the variable wins over the platform default.
        let previous = env::var_os(CACHE_ENV);
        env::set_var(CACHE_ENV, "/tmp/waifu-cache-test");
        assert_eq!(
            cache_directory().unwrap(),
            PathBuf::from("/tmp/waifu-cache-test")
        );
        match previous {
            Some(value) => env::set_var(CACHE_ENV, value),
            None => env::remove_var(CACHE_ENV),
        }
    }
}
