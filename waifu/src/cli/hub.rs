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
//! `-m` takes either a path to a package or a name like `sdxl:base`. A name is looked up in the
//! table below, fetched if it is not in the cache already, and what comes back is a path -- so
//! everything past [`resolve`] works on a file the same way it always did.
//!
//! Every model is published twice, to Hugging Face and to ModelScope, under the same repository
//! name and byte for byte the same files. Which one a fetch uses is [`mirror`]'s to decide, and
//! `WAIFU_MIRROR` overrides it.
//!
//! Only the first package of a model is named in the table. A model split over several packages
//! already says so in its own configuration, so the rest of the file names are read out of the
//! first one rather than written down twice in a table that could come to disagree with it.
//!
//! This lives in the command line tool rather than in the library: a name is a convenience for
//! someone typing at a terminal, and a program embedding the library has its own ideas about
//! where its files live and when it is allowed to touch the network.

use std::env;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use hf_hub::progress::{DownloadEvent, Progress as Reported, ProgressEvent, ProgressHandler};
use hf_hub::HFClientSync;

use crate::{IniConfig, Sdxl, ZipFile};

type Error = Box<dyn std::error::Error>;

/// The suffix a package file carries, which is what tells a path from a mistyped name.
const PACKAGE_SUFFIX: &str = ".waifupkg";

/// Where the cache goes when the environment does not say.
const CACHE_ENV: &str = "WAIFU_CACHE";

/// How much is read from the network at a time.
const DOWNLOAD_BUFFER: usize = 1 << 20;

/// How often a fetch says where it has got to. Short enough that the screen looks alive, long
/// enough that it is not redrawing for a bar that has not moved.
const PROGRESS_INTERVAL: Duration = Duration::from_millis(100);

/// The directory a Hugging Face fetch is given to itself, inside the cache directory of the model
/// it belongs to. On the same filesystem as the package it becomes, so the rename that puts it
/// there is a rename and not a copy.
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

/// A model that has a name, and where it is published.
struct Published {
    /// The name `-m` takes, version and all.
    name: &'static str,
    /// The Hugging Face repository it lives in.
    repo: &'static str,
    /// The package to fetch first: it holds the configuration, and names the other parts.
    first_part: &'static str,
}

/// Every model this build knows by name.
const CATALOG: &[Published] = &[
    Published {
        name: "sdxl:base:v1",
        repo: "ling0322/libwaifu-sdxl-base-1.0",
        first_part: "sdxl-base-1.0-00001-of-00004.waifupkg",
    },
    Published {
        name: "sdxl:wai:v17",
        repo: "ling0322/libwaifu-wai-illustrious-v17",
        first_part: "wai-illustrious-v17-00001-of-00004.waifupkg",
    },
    Published {
        name: "sdxl:noob:v11",
        repo: "ling0322/libwaifu-noobai-xl-v11",
        first_part: "noobai-xl-v11-00001-of-00004.waifupkg",
    },
];

/// Where a package is fetched from.
///
/// The same three models are published to both, under the same repository names and byte for
/// byte the same files, so which one is used changes only how long it takes to arrive.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Mirror {
    HuggingFace,
    ModelScope,
}

impl Mirror {
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
    ("sdxl:base", "sdxl:base:v1"),
    ("sdxl:wai", "sdxl:wai:v17"),
    ("sdxl:noob", "sdxl:noob:v11"),
];

/// What a fetch has to say for itself while it runs.
///
/// A download is minutes long and the caller decides how to show it: the command line prints a
/// line that rewrites itself, and the screen draws a bar. Neither belongs in here.
pub enum Progress<'a> {
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

/// How much of a cached model is on disk, for a screen that offers to fetch one.
pub fn cached_bytes(name: &str) -> u64 {
    match (published(name), cache_directory()) {
        (Some(published), Ok(cache)) => cached_bytes_in(published, &cache),
        _ => 0,
    }
}

fn is_cached_in(published: &Published, cache: &Path) -> bool {
    let directory = cache.join(published.repo.replace('/', "--"));
    let first = directory.join(published.first_part);
    if !first.exists() {
        return false;
    }

    // A first part that cannot be read for what it names is not a model anyone can draw with,
    // whatever is on disk beside it, so it counts as not there.
    match parts_named_by(&first) {
        Ok(parts) => parts.iter().all(|part| directory.join(part).exists()),
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

/// The address a named model is fetched from, for a screen that says where it comes from.
///
/// Through the same mirror the fetch will use, and not a second spelling of the address that
/// could drift from it: on a machine that has settled on ModelScope, showing a huggingface.co
/// address would be a guess about what enter is going to do.
///
/// This is the first package of the model; the rest come from beside it. `None` for anything that
/// is not a published name, which is a path and comes from no one.
pub fn source_url(name: &str) -> Option<String> {
    published(name).map(|model| mirror().url(model.repo, model.first_part))
}

/// The published model a name refers to, following an alias if it is one.
fn published(name: &str) -> Option<&'static Published> {
    let name = ALIASES
        .iter()
        .find(|(alias, _)| *alias == name)
        .map_or(name, |(_, target)| *target);
    CATALOG.iter().find(|model| model.name == name)
}

/// Every name that can be asked for, aliases included, for the usage text and for error messages.
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
        && !model.ends_with(PACKAGE_SUFFIX)
}

/// Turn what `-m` was given into a package on disk, fetching it if it is a name.
///
/// A path is taken as it is written. A known name is fetched into the cache, and what comes back
/// is the first package of the model -- the one that holds the configuration and names the rest,
/// which is what [`crate::Sdxl::from_package`] expects to be handed.
pub fn resolve(model: &str) -> Result<PathBuf, Error> {
    resolve_reporting(model, &mut print_progress)
}

/// The same, telling `report` how it is getting on rather than printing.
pub fn resolve_reporting(
    model: &str,
    report: &mut dyn FnMut(Progress),
) -> Result<PathBuf, Error> {
    if let Some(published) = published(model) {
        return fetch(published, report);
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

/// Make sure every package of `published` is in the cache, and say where the first one is.
fn fetch(published: &Published, report: &mut dyn FnMut(Progress)) -> Result<PathBuf, Error> {
    // One directory per repository, so that the parts of a model sit beside each other: the first
    // package names its neighbours by file name and reads them from its own directory.
    let directory = cache_directory()?.join(published.repo.replace('/', "--"));
    fs::create_dir_all(&directory)?;

    // How many packages there are is not known until the first one has been read, so the first is
    // fetched without a count. Saying "part 1" of an unsaid number beats saying nothing: it is
    // what tells someone watching that more will follow.
    let first = directory.join(published.first_part);
    download(published.repo, published.first_part, &first, Part { at: 1, of: 0 }, report)?;

    let rest = parts_named_by(&first)?;
    let parts = rest.len() + 1;
    for (index, part) in rest.iter().enumerate() {
        let at = directory.join(part);
        let which = Part {
            at: index + 2,
            of: parts,
        };
        download(published.repo, part, &at, which, report)?;
    }
    Ok(first)
}

/// The other packages a model is split over, as the first package names them.
///
/// A model small enough to have been written as one file names none, which is not an error: it is
/// what an unsplit package looks like.
fn parts_named_by(first: &Path) -> Result<Vec<String>, Error> {
    let package = ZipFile::open(first)?;
    let ini = IniConfig::parse(&package.read_to_string(crate::MODEL_CONFIG)?)?;
    let Ok(section) = ini.section(Sdxl::MODEL_SECTION) else {
        return Ok(Vec::new());
    };
    let Ok(listed) = section.get_str(Sdxl::SHARDS_KEY) else {
        return Ok(Vec::new());
    };

    let mut parts = Vec::new();
    for part in listed.split(',').map(str::trim).filter(|p| !p.is_empty()) {
        // The list comes out of a file fetched over the network, and it is about to be joined to
        // a directory and written to. A package may name a neighbour of its own and nothing else,
        // which is the rule ZipFile::sibling reads by; it is enforced here too, because here is
        // where the name decides what gets written.
        if part == "."
            || part == ".."
            || part.contains('/')
            || part.contains('\\')
            || Path::new(part).components().count() != 1
        {
            return Err(format!(
                "the model names {part:?} as one of its parts, and a package may only name a \
                 neighbour of its own"
            )
            .into());
        }
        // The first package lists itself along with the others; it is already here.
        if Some(part) == first.file_name().and_then(|name| name.to_str()) {
            continue;
        }
        parts.push(part.to_string());
    }
    Ok(parts)
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
    report: &mut dyn FnMut(Progress),
) -> Result<(), Error> {
    // Already fetched, and nothing to say about it: a caller that draws a bar would rather see
    // the bar start at the first file that is actually being fetched.
    if destination.exists() {
        return Ok(());
    }

    match mirror() {
        Mirror::HuggingFace => download_from_hugging_face(repo, file, destination, part, report),
        Mirror::ModelScope => download_over_http(repo, file, destination, part, report),
    }
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
fn download_from_hugging_face(
    repo: &str,
    file: &str,
    destination: &Path,
    part: Part,
    report: &mut dyn FnMut(Progress),
) -> Result<(), Error> {
    let (owner, name) = repo
        .split_once('/')
        .ok_or_else(|| format!("{repo} is not an owner and a repository name"))?;
    let directory = destination
        .parent()
        .ok_or_else(|| format!("{} is not in a directory", destination.display()))?;
    let staging = directory.join(INCOMING);

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
        let total = counted.total.load(Ordering::Relaxed);
        report(Progress::Fetching {
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
    // Only ever holds the one file, and only while it is arriving. Left behind if something else
    // is in there, which is not this function's to throw away.
    let _ = fs::remove_dir(&staging);

    report(Progress::Fetched {
        file,
        bytes: fs::metadata(destination).map(|meta| meta.len()).unwrap_or(0),
        part: part.at,
        parts: part.of,
    });
    Ok(())
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
/// `.part` left behind is resumed rather than restarted.
fn download_over_http(
    repo: &str,
    file: &str,
    destination: &Path,
    part: Part,
    report: &mut dyn FnMut(Progress),
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
    let written = copy_reporting(&mut reader, &mut writer, file, start, total, part, report)?;
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
fn copy_reporting(
    reader: &mut impl Read,
    writer: &mut impl Write,
    name: &str,
    start: u64,
    total: Option<u64>,
    part: Part,
    report: &mut dyn FnMut(Progress),
) -> io::Result<u64> {
    let mut buffer = vec![0u8; DOWNLOAD_BUFFER];
    let mut done = start;

    // Said before the first byte, because the gap between one package finishing and the next
    // showing a number is a stretch where the screen would otherwise hold a full bar and look
    // hung. This is what moves it to the new file at nothing.
    report(Progress::Fetching {
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

        // On a clock rather than on a fraction of the file. A percent of two gigabytes is twenty
        // megabytes, which is seconds of silence on an ordinary line and reads as a hang; and on
        // a fast one it is several redraws a second for a bar that moved a pixel. Time is what
        // the watcher actually measures the wait in.
        if said.elapsed() >= PROGRESS_INTERVAL {
            said = Instant::now();
            report(Progress::Fetching {
                file: name,
                done,
                total,
                part: part.at,
                parts: part.of,
            });
        }
    }

    report(Progress::Fetched {
        file: name,
        bytes: done,
        part: part.at,
        parts: part.of,
    });
    Ok(done)
}

/// What the command line does with a fetch's progress: one line that rewrites itself.
fn print_progress(progress: Progress) {
    match progress {
        Progress::Fetching {
            file,
            done,
            total,
            part,
            parts,
        } => {
            let of = if parts > 0 {
                format!(" (part {part} of {parts})")
            } else {
                String::new()
            };
            match total {
                Some(total) if total > 0 => eprint!(
                    "\rfetching {file}{of}: {}% of {}",
                    done * 100 / total,
                    megabytes(total)
                ),
                _ => eprint!("\rfetching {file}{of}: {}", megabytes(done)),
            }
            let _ = io::stderr().flush();
        }
        Progress::Fetched { file, bytes, .. } => {
            eprintln!("\rfetching {file}: done ({})    ", megabytes(bytes));
        }
    }
}

fn megabytes(bytes: u64) -> String {
    format!("{:.1} MB", bytes as f64 / 1_000_000.0)
}

#[cfg(test)]
mod tests {
    use hf_hub::progress::{FileProgress, FileStatus};

    use super::*;

    #[test]
    fn a_name_finds_what_it_names() {
        let model = published("sdxl:base:v1").expect("the catalog has it");
        assert_eq!(model.repo, "ling0322/libwaifu-sdxl-base-1.0");
        assert!(model.first_part.ends_with(PACKAGE_SUFFIX));
    }

    #[test]
    fn an_alias_follows_what_it_points_at() {
        let alias = published("sdxl:base").expect("an alias resolves");
        let target = published("sdxl:base:v1").expect("and so does what it points at");
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
        let model = published("sdxl:base:v1").expect("the catalog has it");
        let hf = Mirror::HuggingFace.url(model.repo, model.first_part);
        let ms = Mirror::ModelScope.url(model.repo, model.first_part);

        // Written out against the default host, so somewhere HF_ENDPOINT points at a mirror this
        // is a different string for a good reason; the test below is the one for that case.
        if env::var_os(HF_ENDPOINT_ENV).is_none() {
            assert_eq!(
                hf,
                "https://huggingface.co/ling0322/libwaifu-sdxl-base-1.0/resolve/main/\
                 sdxl-base-1.0-00001-of-00004.waifupkg"
            );
        }
        assert_eq!(
            ms,
            "https://modelscope.cn/models/ling0322/libwaifu-sdxl-base-1.0/resolve/master/\
             sdxl-base-1.0-00001-of-00004.waifupkg"
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
        let model = published("sdxl:base:v1").expect("the catalog has it");

        env::set_var(HF_ENDPOINT_ENV, "https://hf-mirror.com");
        assert_eq!(
            Mirror::HuggingFace.url(model.repo, model.first_part),
            "https://hf-mirror.com/ling0322/libwaifu-sdxl-base-1.0/resolve/main/\
             sdxl-base-1.0-00001-of-00004.waifupkg"
        );

        // Typed with the slash a host name invites, and it makes no difference.
        env::set_var(HF_ENDPOINT_ENV, "https://hf-mirror.com/");
        assert!(Mirror::HuggingFace
            .url(model.repo, model.first_part)
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
                let url = mirror.url(model.repo, model.first_part);
                assert!(url.starts_with("https://"), "{url}");
                assert!(url.contains(model.repo), "{url}");
                assert!(url.ends_with(model.first_part), "{url}");
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
            &mut |progress| finished |= matches!(progress, Progress::Fetched { .. }),
        )
        .expect("the file is there and so is the network");

        // Under the name it was asked for, whole, and with nothing left in the way.
        assert!(destination.exists(), "{}", destination.display());
        assert!(fs::metadata(&destination).expect("it is a file").len() > 0);
        assert!(!directory.join(INCOMING).exists(), "the staging is cleared");

        // And it said so when it finished, which is what stops the bar.
        assert!(finished, "the fetch said nothing about finishing");

        let _ = fs::remove_dir_all(&directory);
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
    fn a_name_says_where_it_would_be_fetched_from() {
        let url = source_url("sdxl:wai").expect("a published name has an address");
        assert!(url.starts_with("https://"), "{url}");
        assert!(url.contains("libwaifu-wai-illustrious-v17"), "{url}");
        assert!(url.ends_with(PACKAGE_SUFFIX), "{url}");

        // The address shown and the address fetched are the same string, worked out the same way
        // -- a screen that named the other mirror would be a guess about what enter will do.
        let model = published("sdxl:wai").expect("the catalog has it");
        assert_eq!(url, mirror().url(model.repo, model.first_part));

        // A path comes from nobody.
        assert!(source_url("some-model.waifupkg").is_none());
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
        assert!(names.contains(&"sdxl:base:v1"));
        assert!(names.contains(&"sdxl:wai"));
        assert!(names.contains(&"sdxl:wai:v17"));
        assert!(names.contains(&"sdxl:noob"));
        assert!(names.contains(&"sdxl:noob:v11"));
    }

    #[test]
    fn no_two_models_are_the_same_model() {
        // A table entry is written by copying the one above it, so the thing to check is that
        // the copy was finished: no two models share a name, a repository, or a first package.
        for (index, model) in CATALOG.iter().enumerate() {
            for other in &CATALOG[index + 1..] {
                assert_ne!(model.name, other.name);
                assert_ne!(model.repo, other.repo, "{} and {}", model.name, other.name);
                assert_ne!(
                    model.first_part, other.first_part,
                    "{} and {}",
                    model.name, other.name
                );
            }
        }
    }

    #[test]
    fn every_model_is_named_the_way_the_others_are() {
        // A name is `sdxl:<model>:<version>`, and the first package of every one of them is a
        // package. Cheap to check, and it is the sort of thing a copied table entry gets wrong.
        for model in CATALOG {
            let fields: Vec<&str> = model.name.split(':').collect();
            assert_eq!(fields.len(), 3, "{} is not sdxl:model:version", model.name);
            assert_eq!(fields[0], "sdxl", "{} is not an sdxl model", model.name);
            assert!(model.repo.contains('/'), "{} has no namespace", model.repo);
            assert!(
                model.first_part.ends_with(PACKAGE_SUFFIX),
                "{} is not a package",
                model.first_part
            );
        }
    }

    #[test]
    fn a_path_is_not_read_as_a_name() {
        // The suffix, a separator, or no colon at all: each is enough to say "this is a file".
        assert!(!reads_as_a_name("sdxl.waifupkg"));
        assert!(!reads_as_a_name("models/sdxl-base.waifupkg"));
        assert!(!reads_as_a_name(r"C:\models\sdxl-base.waifupkg"));
        assert!(!reads_as_a_name("./sdxl"));
        assert!(reads_as_a_name("sdxl:base"));
        assert!(reads_as_a_name("sdxl:typo"));
    }

    #[test]
    fn an_unknown_name_says_what_the_known_ones_are() {
        let error = resolve("sdxl:nope").unwrap_err().to_string();
        assert!(error.contains("no model called"), "{error}");
        assert!(error.contains("sdxl:base"), "{error}");
    }

    #[test]
    fn a_model_that_is_not_on_disk_is_not_cached() {
        let cache = std::env::temp_dir().join(format!("waifu-cache-{}", std::process::id()));
        let _ = fs::remove_dir_all(&cache);
        let model = published("sdxl:base").expect("the base model");

        // Nothing there at all.
        assert!(!is_cached_in(model, &cache));
        assert_eq!(cached_bytes_in(model, &cache), 0);

        // A first part that is not a package it can read. Half a download looks like this, and it
        // must not read as a model that is ready to draw with -- though the bytes still count,
        // because a resumed fetch starts from them.
        let directory = cache.join(model.repo.replace('/', "--"));
        fs::create_dir_all(&directory).expect("a directory to put it in");
        fs::write(directory.join(model.first_part), b"not a zip").expect("a first part");

        assert!(!is_cached_in(model, &cache));
        assert_eq!(cached_bytes_in(model, &cache), 9);

        // And a name nobody published is not cached either, rather than a panic.
        assert!(!is_cached("sdxl:nope"));
        assert_eq!(cached_bytes("sdxl:nope"), 0);

        let _ = fs::remove_dir_all(&cache);
    }

    #[test]
    fn a_missing_file_is_reported_as_a_file() {
        let error = resolve("no-such-model.waifupkg").unwrap_err().to_string();
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
