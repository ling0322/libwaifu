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
//! table below, fetched from Hugging Face if it is not in the cache already, and what comes back
//! is a path -- so everything past [`resolve`] works on a file the same way it always did.
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

use crate::{IniConfig, Sdxl, ZipFile};

type Error = Box<dyn std::error::Error>;

/// The suffix a package file carries, which is what tells a path from a mistyped name.
const PACKAGE_SUFFIX: &str = ".waifupkg";

/// Where the cache goes when the environment does not say.
const CACHE_ENV: &str = "WAIFU_CACHE";

/// How much is read from the network at a time.
const DOWNLOAD_BUFFER: usize = 1 << 20;

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
];

/// Names that follow whatever is current rather than naming a version.
///
/// `sdxl:base` is what someone types when they want the base model and do not care which release
/// of it; it keeps working when a v2 arrives, and `sdxl:base:v1` keeps meaning what it says.
const ALIASES: &[(&str, &str)] = &[("sdxl:base", "sdxl:base:v1"), ("sdxl:wai", "sdxl:wai:v17")];

/// What a fetch has to say for itself while it runs.
///
/// A download is minutes long and the caller decides how to show it: the command line prints a
/// line that rewrites itself, and the screen draws a bar. Neither belongs in here.
pub enum Progress<'a> {
    /// Bytes of `file` fetched so far, and how many there are when the server said.
    Fetching {
        file: &'a str,
        done: u64,
        total: Option<u64>,
    },
    /// `file` is whole and in the cache.
    Fetched { file: &'a str, bytes: u64 },
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

    let first = directory.join(published.first_part);
    download(published.repo, published.first_part, &first, report)?;

    for part in parts_named_by(&first)? {
        download(published.repo, &part, &directory.join(&part), report)?;
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
/// The download goes to a `.part` beside it and is renamed once it is whole, so an interrupted
/// fetch is never mistaken for a model: a half written package would otherwise be opened on the
/// next run and fail as a corrupt one. A `.part` left behind is resumed rather than restarted.
fn download(
    repo: &str,
    file: &str,
    destination: &Path,
    report: &mut dyn FnMut(Progress),
) -> Result<(), Error> {
    // Already fetched, and nothing to say about it: a caller that draws a bar would rather see
    // the bar start at the first file that is actually being fetched.
    if destination.exists() {
        return Ok(());
    }

    let url = format!("https://huggingface.co/{repo}/resolve/main/{file}");
    let partial = PathBuf::from(format!("{}.part", destination.display()));
    let have = fs::metadata(&partial).map(|meta| meta.len()).unwrap_or(0);

    let mut request = ureq::get(&url);
    if have > 0 {
        request = request.header("Range", &format!("bytes={have}-"));
    }
    let response = request.call()?;

    // 206 means the server took the range and is sending the rest; anything else is the whole
    // file, and what was already fetched is written over rather than appended to.
    let resuming = response.status().as_u16() == 206 && have > 0;
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
    let written = copy_reporting(&mut reader, &mut writer, file, start, total, report)?;
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
    report: &mut dyn FnMut(Progress),
) -> io::Result<u64> {
    let mut buffer = vec![0u8; DOWNLOAD_BUFFER];
    let mut done = start;
    let mut reported = u64::MAX;

    loop {
        let read = reader.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        writer.write_all(&buffer[..read])?;
        done += read as u64;

        // Once a percent, or once every ten megabytes when the length is unknown. A caller that
        // draws would redraw for nothing at every buffer, and one that prints would scroll.
        let mark = match total {
            Some(total) if total > 0 => done * 100 / total,
            _ => done / (10 * DOWNLOAD_BUFFER as u64),
        };
        if mark != reported {
            reported = mark;
            report(Progress::Fetching {
                file: name,
                done,
                total,
            });
        }
    }

    report(Progress::Fetched {
        file: name,
        bytes: done,
    });
    Ok(done)
}

/// What the command line does with a fetch's progress: one line that rewrites itself.
fn print_progress(progress: Progress) {
    match progress {
        Progress::Fetching { file, done, total } => {
            match total {
                Some(total) if total > 0 => {
                    eprint!("\rfetching {file}: {}% of {}", done * 100 / total, megabytes(total))
                }
                _ => eprint!("\rfetching {file}: {}", megabytes(done)),
            }
            let _ = io::stderr().flush();
        }
        Progress::Fetched { file, bytes } => {
            eprintln!("\rfetching {file}: done ({})    ", megabytes(bytes));
        }
    }
}

fn megabytes(bytes: u64) -> String {
    format!("{:.1} MB", bytes as f64 / 1_000_000.0)
}

#[cfg(test)]
mod tests {
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
    }

    #[test]
    fn two_models_are_told_apart() {
        let base = published("sdxl:base").expect("the base model");
        let wai = published("sdxl:wai").expect("the fine tune");
        assert_ne!(base.repo, wai.repo);
        assert_ne!(base.first_part, wai.first_part);
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
