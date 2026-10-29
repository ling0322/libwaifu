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

//! libwaifu_modelmanager, as the screens ask it: which models are published, which are here,
//! where they are kept, and fetching one with news of how it is going down a channel.

use std::ffi::c_void;
use std::path::{Path, PathBuf};
use std::ptr;
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::Mutex;

use crate::c::{
    self, WaifuModelFetch, WaifuModelManagerCompletionEvent as WaifuCompletionEvent,
    WaifuModelManagerProgressEvent as WaifuProgressEvent,
    WaifuModelManagerStatusCode as WaifuStatusCode,
};

/// The settings file the model directory is written into, as the screens name it.
pub const CONFIG_FILE_NAME: &str = "config.toml";

/// The environment variable that decides the model directory over everything else.
pub const CACHE_ENV: &str = "WAIFU_CACHE";

/// One published model, as the catalogue says it.
#[derive(Clone, Debug, PartialEq)]
pub struct Listed {
    pub name: &'static str,
    pub full_name: &'static str,
    /// "image", "speech" or "conversion"; a model that does two is listed twice.
    pub kind: String,
    /// Whether every package of it is here.
    pub cached: bool,
    /// What is on the disk of it, which is most of a model for one that was interrupted.
    pub bytes: u64,
    pub explicit: bool,
    /// The directory it is in, where any of it is on the disk.
    pub directory: Option<PathBuf>,
}

/// A catalogue name for the rest of the run. The names are a few dozen at most and the same each
/// time the catalogue is read, so each is kept once, and the screens can hold them as plainly as
/// they could when the catalogue was a table compiled in.
pub fn kept(name: &str) -> &'static str {
    static NAMES: Mutex<Vec<&'static str>> = Mutex::new(Vec::new());
    let mut names = NAMES.lock().unwrap_or_else(|held| held.into_inner());
    if let Some(name) = names.iter().find(|kept| **kept == name) {
        return name;
    }
    let name: &'static str = Box::leak(name.to_string().into_boxed_str());
    names.push(name);
    name
}

/// Every published model, in the catalogue's order.
pub fn catalog() -> Vec<Listed> {
    let Some(said) = c::manager_json(c::waifu_modelmanager_catalog_json()) else {
        return Vec::new();
    };
    said["models"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|model| {
            Some(Listed {
                name: kept(model["name"].as_str()?),
                full_name: kept(model["full_name"].as_str()?),
                kind: model["kind"].as_str()?.to_string(),
                cached: model["cached"].as_bool().unwrap_or(false),
                bytes: model["bytes_on_disk"].as_u64().unwrap_or(0),
                explicit: model["explicit"].as_bool().unwrap_or(false),
                directory: model["directory"].as_str().map(PathBuf::from),
            })
        })
        .collect()
}

/// The published models that draw.
pub fn pictures() -> Vec<Listed> {
    catalog()
        .into_iter()
        .filter(|model| model.kind == "image")
        .collect()
}

/// Whether `name` is a published model that speaks, converts or takes a picture apart rather than
/// draws.
pub fn is_a_voice(name: &str) -> bool {
    catalog()
        .iter()
        .any(|model| model.name == name && model.kind != "image")
}

/// Which of the three places that can name the model directory named it.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Said {
    /// `WAIFU_CACHE`, which wins over everything: it is what a script sets for one run.
    Environment,
    /// `model_dir` in `config.toml`, which is what the model screen writes.
    Config,
    /// Nobody: the platform's cache directory.
    Default,
}

impl Said {
    /// How the model screen says where the directory came from.
    pub fn describe(self) -> &'static str {
        match self {
            Said::Environment => CACHE_ENV,
            Said::Config => CONFIG_FILE_NAME,
            Said::Default => "default",
        }
    }
}

/// Where models are fetched into, who said so, and what else there is to know about it.
#[derive(Clone, Debug, PartialEq)]
pub struct Directory {
    pub now: PathBuf,
    pub said: Said,
    /// Where they would go if nobody said.
    pub default: Option<PathBuf>,
    /// The settings file a directory chosen here is written into.
    pub config_file: Option<PathBuf>,
}

/// Where models are fetched into, or why that cannot be told -- a settings file that does not
/// read, most likely.
pub fn directory() -> Result<Directory, String> {
    let said =
        c::manager_json(c::waifu_modelmanager_directory_json()).ok_or_else(c::manager_error)?;
    Ok(Directory {
        now: PathBuf::from(said["directory"].as_str().unwrap_or_default()),
        said: match said["said"].as_str() {
            Some("environment") => Said::Environment,
            Some("config") => Said::Config,
            _ => Said::Default,
        },
        default: said["default"].as_str().map(PathBuf::from),
        config_file: said["config_file"].as_str().map(PathBuf::from),
    })
}

/// Models are fetched into `directory` from now on, or the default for None.
pub fn set_directory(directory: Option<&Path>) -> Result<(), String> {
    let path = directory.map(|directory| c::c_string(&directory.to_string_lossy()));
    let code = unsafe {
        c::waifu_modelmanager_set_directory(path.as_ref().map_or(ptr::null(), |path| path.as_ptr()))
    };
    match code {
        WaifuStatusCode::WAIFU_MODELMANAGER_OK => Ok(()),
        _ => Err(c::manager_error()),
    }
}

/// Throws away what has been fetched of `name`.
pub fn remove(name: &str) -> Result<(), String> {
    let name = c::c_string(name);
    match unsafe { c::waifu_modelmanager_remove(name.as_ptr()) } {
        WaifuStatusCode::WAIFU_MODELMANAGER_OK => Ok(()),
        _ => Err(c::manager_error()),
    }
}

/// The manifest of `model` -- a published name, or a manifest's own path -- for the engine to be
/// handed: the fetched one, or a bundled one where nothing has been. An error where there is none
/// here.
pub fn manifest_path(model: &str) -> Result<PathBuf, String> {
    let model = c::c_string(model);
    c::manager_string(unsafe { c::waifu_modelmanager_manifest_path(model.as_ptr()) })
        .map(PathBuf::from)
        .ok_or_else(c::manager_error)
}

/// What a fetch has to say.
#[derive(Clone, Debug, PartialEq)]
pub enum Word {
    /// A file, how far into it, how long it is when that is known, which package of how many it
    /// is -- `parts` being zero until the first package has named the others -- and which hub it
    /// comes from once that is settled.
    Fetching {
        hub: Option<String>,
        file: String,
        done: u64,
        total: Option<u64>,
        part: usize,
        parts: usize,
    },
    /// Every package is here.
    Done,
    /// It was asked to stop, and did; what is said is what was kept.
    Stopped(String),
    Failed(String),
}

/// What a stop leaves behind, as the screens say it.
const STOPPED: &str =
    "stopped -- what has come down is kept, and fetching it again carries on from there";

/// A fetch running in libwaifu_modelmanager, and the channel its news comes down.
pub struct Fetch {
    handle: *mut WaifuModelFetch,
    /// What the callbacks are handed, owned here and let go of once they cannot be called again.
    sender: *mut Sender<Word>,
    pub news: Receiver<Word>,
}

// The handle is only cancelled and freed, which the library allows from any thread.
unsafe impl Send for Fetch {}

impl Fetch {
    /// Starts fetching `name`.
    pub fn start(name: &str) -> Result<Fetch, String> {
        let (sender, news) = channel();
        let sender = Box::into_raw(Box::new(sender));
        let model = c::c_string(name);
        let handle = unsafe {
            c::waifu_modelmanager_fetch_async(
                model.as_ptr(),
                sender as *mut c_void,
                Some(on_progress),
                Some(on_complete),
            )
        };
        if handle.is_null() {
            // Refused at the call: no callback will ever run, so the sender is this side's again.
            drop(unsafe { Box::from_raw(sender) });
            return Err(c::manager_error());
        }
        Ok(Fetch {
            handle,
            sender,
            news,
        })
    }

    /// Asks it to stop. What has come down is kept.
    pub fn stop(&self) {
        unsafe { c::waifu_modelmanager_fetch_cancel(self.handle) };
    }
}

impl Drop for Fetch {
    fn drop(&mut self) {
        // Stops it if it is still going, and waits for its thread: no callback runs after this.
        unsafe { c::waifu_modelmanager_fetch_free(self.handle) };
        drop(unsafe { Box::from_raw(self.sender) });
    }
}

unsafe extern "C" fn on_progress(user_data: *mut c_void, event: *const WaifuProgressEvent) {
    let (sender, event) = (&*(user_data as *const Sender<Word>), &*event);
    let _ = sender.send(Word::Fetching {
        hub: c::text(event.hub),
        file: c::text(event.file).unwrap_or_default(),
        done: event.done,
        total: (event.total > 0).then_some(event.total),
        part: event.part as usize,
        parts: event.parts as usize,
    });
}

unsafe extern "C" fn on_complete(user_data: *mut c_void, event: *const WaifuCompletionEvent) {
    let (sender, status) = (&*(user_data as *const Sender<Word>), &(*event).status);
    let _ = sender.send(match status.code {
        WaifuStatusCode::WAIFU_MODELMANAGER_OK => Word::Done,
        WaifuStatusCode::WAIFU_MODELMANAGER_CANCELLED => Word::Stopped(STOPPED.to_string()),
        _ => {
            Word::Failed(c::text(status.message).unwrap_or_else(|| "the fetch failed".to_string()))
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_catalogue_lists_pictures_and_voices() {
        let catalog = catalog();
        assert!(catalog
            .iter()
            .any(|model| model.name == "sdxl:base" && model.kind == "image"));
        assert!(is_a_voice("indextts"));
        assert!(!is_a_voice("sdxl:base"));
        assert!(pictures().iter().all(|model| model.kind == "image"));
    }

    #[test]
    fn a_name_is_kept_once() {
        // Two strings with the same letters at two addresses, which come back as one.
        let built = String::from("sdxl:") + "base";
        assert!(std::ptr::eq(kept("sdxl:base"), kept(&built)));
    }

    #[test]
    fn a_fetch_of_a_model_nobody_published_fails_at_the_end() {
        let fetch = Fetch::start("no-such-model:v9").unwrap();
        let end = loop {
            match fetch
                .news
                .recv_timeout(std::time::Duration::from_secs(20))
                .unwrap()
            {
                Word::Fetching { .. } => continue,
                end => break end,
            }
        };
        assert!(
            matches!(&end, Word::Failed(message) if message.contains("no-such-model:v9")),
            "{end:?}"
        );
    }
}
