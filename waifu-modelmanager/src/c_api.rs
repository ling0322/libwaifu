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

//! The C API: the catalogue, fetching, where models are kept, and the log, as `extern "C"`.
//! Everything outside this crate goes through these.
//!
//! Nothing with any logic in it lives here. This converts -- C strings to `&str`, what comes back
//! to borrowed C views, panics to status codes -- and keeps the promises docs/ffi.md makes about
//! when a callback can arrive.

#![allow(clippy::missing_safety_doc)]

use std::ffi::{c_char, c_void};
use std::panic::{self, AssertUnwindSafe};
use std::ptr;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::Instant;

mod support;

use support::{
    c_string, caught, required, set_last_error, status, string_in, string_out, Gate, UserData,
};
pub use support::{
    WaifuModelManagerCompleteCallback, WaifuModelManagerCompletionEvent,
    WaifuModelManagerLogCallback, WaifuModelManagerLogEvent, WaifuModelManagerLogLevel,
    WaifuModelManagerProgressCallback, WaifuModelManagerProgressEvent, WaifuModelManagerStatus,
    WaifuModelManagerStatusCode,
};

use crate::fetch::{self, Failed, Fetch};
use crate::{config, hub, log};

use WaifuModelManagerStatusCode::*;

/// What a caller compiled against this header expects. Bumped when anything in it changes in a
/// way an older caller would misread.
pub const WAIFU_MODELMANAGER_ABI_VERSION: u32 = 1;

/// The ABI this library was built with: WAIFU_MODELMANAGER_ABI_VERSION.
#[no_mangle]
pub extern "C" fn waifu_modelmanager_abi_version() -> u32 {
    WAIFU_MODELMANAGER_ABI_VERSION
}

/// This thread's last message from this library; valid until this thread's next call into it.
#[no_mangle]
pub extern "C" fn waifu_modelmanager_last_error() -> *const c_char {
    support::last_error()
}

/// Frees a string a call into this library returned. NULL is fine.
#[no_mangle]
pub unsafe extern "C" fn waifu_modelmanager_string_free(s: *mut c_char) {
    support::free_string(s)
}

// -- logging ---------------------------------------------------------------------------------

/// Every line this library would print goes to the callback instead, from whichever thread wrote
/// it. NULL puts them back on the console. libwaifu's lines are waifu_set_log_callback's.
#[no_mangle]
pub extern "C" fn waifu_modelmanager_set_log_callback(
    user_data: *mut c_void,
    callback: WaifuModelManagerLogCallback,
) {
    caught((), || log::set_sink(support::log_sink(user_data, callback)))
}

/// Lines below the level are not made at all. INFO unless set.
#[no_mangle]
pub extern "C" fn waifu_modelmanager_set_log_level(level: WaifuModelManagerLogLevel) {
    caught((), || log::set_level(support::log_level(level)))
}

// -- the model manager -----------------------------------------------------------------------

/// The published models, aliases only, as JSON. Free with waifu_modelmanager_string_free. NULL on
/// failure.
#[no_mangle]
pub extern "C" fn waifu_modelmanager_catalog_json() -> *mut c_char {
    caught(ptr::null_mut(), || {
        string_out(&fetch::catalog_json().to_string())
    })
}

/// Where downloads go. Free with waifu_modelmanager_string_free. NULL on failure.
#[no_mangle]
pub extern "C" fn waifu_modelmanager_directory() -> *mut c_char {
    caught(ptr::null_mut(), || match hub::model_directory() {
        Ok((directory, _)) => string_out(&directory.to_string_lossy()),
        Err(error) => {
            set_last_error(&error.to_string());
            ptr::null_mut()
        }
    })
}

/// Where downloads go, who said so and what else could, as JSON:
/// `{"directory", "said": "environment"|"config"|"default", "default", "config_file"}`, the last
/// two null where they cannot be told. Free with waifu_modelmanager_string_free. NULL on failure.
#[no_mangle]
pub extern "C" fn waifu_modelmanager_directory_json() -> *mut c_char {
    caught(ptr::null_mut(), || {
        match hub::model_directory() {
        Ok((directory, said)) => string_out(
            &serde_json::json!({
                "directory": directory.to_string_lossy(),
                "said": match said {
                    hub::Said::Environment => "environment",
                    hub::Said::Config => "config",
                    hub::Said::Default => "default",
                },
                "default": hub::default_directory().ok().map(|default| default.to_string_lossy().into_owned()),
                "config_file": config::path().map(|file| file.to_string_lossy().into_owned()),
            })
            .to_string(),
        ),
        Err(error) => {
            set_last_error(&error.to_string());
            ptr::null_mut()
        }
    }
    })
}

/// The manifest of a published model, as a path for libwaifu's load and describe: the one in the
/// download directory where it has been fetched, and otherwise one of the bundled manifests. Only
/// a model whose catalogue entry says `cached` has its weights beside it. A manifest's own path is
/// handed back as it is. Free with waifu_modelmanager_string_free. NULL, with
/// waifu_modelmanager_last_error, where there is no manifest here: fetch the model, or every
/// manifest, first.
#[no_mangle]
pub unsafe extern "C" fn waifu_modelmanager_manifest_path(name: *const c_char) -> *mut c_char {
    caught(ptr::null_mut(), || {
        let name = match required(name, "name") {
            Ok(name) => name,
            Err(error) => {
                set_last_error(&error);
                return ptr::null_mut();
            }
        };
        if hub::full_name(&name).is_none() {
            if std::path::Path::new(&name).is_file() {
                return string_out(&name);
            }
            set_last_error(&format!("there is no model called \"{name}\": neither a published name nor a manifest that exists"));
            return ptr::null_mut();
        }
        match hub::local_manifest(&name) {
            Some(path) => string_out(&path.to_string_lossy()),
            None => {
                set_last_error(&format!(
                    "the manifest of \"{name}\" is not here: fetch it first"
                ));
                ptr::null_mut()
            }
        }
    })
}

/// What a screen should start a model's boxes at, as JSON: the kind of model's defaults with what
/// its manifest's `suggested:` block says laid over them. `model` is the path of a manifest -- see
/// waifu_modelmanager_manifest_path -- or "tones". A picture model's are `width`, `height`, `steps`,
/// `guidance`, `takes_guidance`, `sizes`, `prompt` and `avoid`; a voice's are `speed` and
/// `temperature`, and `conversion.steps` for one that also converts. None of it is needed to run a
/// model, whose request says every number itself. Free with waifu_modelmanager_string_free. NULL,
/// with waifu_modelmanager_last_error, for anything that is not a manifest on the disk.
#[no_mangle]
pub unsafe extern "C" fn waifu_modelmanager_defaults_json(model: *const c_char) -> *mut c_char {
    caught(ptr::null_mut(), || {
        let model = match required(model, "model") {
            Ok(model) => model,
            Err(error) => {
                set_last_error(&error);
                return ptr::null_mut();
            }
        };
        match crate::defaults::defaults_json(&model) {
            Some(defaults) => string_out(&defaults.to_string()),
            None => {
                set_last_error(&format!("there is no manifest at \"{model}\""));
                ptr::null_mut()
            }
        }
    })
}

/// Downloads go to `path` from now on, saved to config.toml; NULL or "" for the default.
#[no_mangle]
pub unsafe extern "C" fn waifu_modelmanager_set_directory(
    path: *const c_char,
) -> WaifuModelManagerStatusCode {
    caught(WAIFU_MODELMANAGER_ERR_INTERNAL, || {
        let path = match string_in(path, "path") {
            Ok(path) => path.filter(|path| !path.is_empty()),
            Err(error) => {
                set_last_error(&error);
                return WAIFU_MODELMANAGER_ERR_INVALID_ARGUMENT;
            }
        };
        match config::set_model_dir(path.as_deref().map(std::path::Path::new)) {
            Ok(_) => WAIFU_MODELMANAGER_OK,
            Err(error) => {
                set_last_error(&error.to_string());
                WAIFU_MODELMANAGER_ERR_INTERNAL
            }
        }
    })
}

/// Reads the manifests of published models out of `directory` where the download directory has
/// none: an app built with every model's manifest (waifu-tools' `fetch_manifests` fetches them)
/// can describe a model before anything is downloaded: waifu_modelmanager_manifest_path hands
/// back the bundled one where nothing has been fetched. Laid out as the download directory is,
/// and only read. NULL or "" for none. Not saved.
#[no_mangle]
pub unsafe extern "C" fn waifu_modelmanager_set_bundled_manifests(
    directory: *const c_char,
) -> WaifuModelManagerStatusCode {
    caught(WAIFU_MODELMANAGER_ERR_INTERNAL, || {
        let directory = match string_in(directory, "directory") {
            Ok(directory) => directory.filter(|directory| !directory.is_empty()),
            Err(error) => {
                set_last_error(&error);
                return WAIFU_MODELMANAGER_ERR_INVALID_ARGUMENT;
            }
        };
        hub::set_bundled_manifests(directory.map(std::path::PathBuf::from));
        WAIFU_MODELMANAGER_OK
    })
}

/// Deletes a downloaded package.
#[no_mangle]
pub unsafe extern "C" fn waifu_modelmanager_remove(
    name: *const c_char,
) -> WaifuModelManagerStatusCode {
    caught(WAIFU_MODELMANAGER_ERR_INTERNAL, || {
        let name = match required(name, "name") {
            Ok(name) => name,
            Err(error) => {
                set_last_error(&error);
                return WAIFU_MODELMANAGER_ERR_INVALID_ARGUMENT;
            }
        };
        if hub::full_name(&name).is_none() {
            set_last_error(&format!("there is no published model called \"{name}\""));
            return WAIFU_MODELMANAGER_ERR_UNKNOWN_MODEL;
        }
        match hub::remove(&name) {
            Ok(()) => WAIFU_MODELMANAGER_OK,
            Err(error) => {
                set_last_error(&error.to_string());
                WAIFU_MODELMANAGER_ERR_INTERNAL
            }
        }
    })
}

/// A download running on a thread of its own.
pub struct WaifuModelFetch {
    stop: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
}

/// Starts fetching a published model on a thread of its own and returns at once. Fetching one
/// already here finishes at once. Where it went is waifu_modelmanager_manifest_path's to say.
/// NULL, with waifu_modelmanager_last_error and no callback ever, where it cannot start.
#[no_mangle]
pub unsafe extern "C" fn waifu_modelmanager_fetch_async(
    model: *const c_char,
    user_data: *mut c_void,
    on_progress: WaifuModelManagerProgressCallback,
    on_complete: WaifuModelManagerCompleteCallback,
) -> *mut WaifuModelFetch {
    caught(ptr::null_mut(), || {
        let model = match required(model, "model") {
            Ok(model) => model,
            Err(error) => {
                set_last_error(&error);
                return ptr::null_mut();
            }
        };
        spawn_fetch(
            user_data,
            on_progress,
            on_complete,
            move |on_progress, stop| fetch::fetch(&model, on_progress, stop).map(|_| ()),
        )
    })
}

/// Starts fetching the manifest of every published model that is not here yet -- a couple of
/// kilobytes each, and none of their packages -- on a thread of its own, and returns at once.
/// What a model suggests is in its manifest, so a screen that has called this can fill in a
/// model's settings the moment it is chosen. Each manifest is a part of the whole in the progress;
/// with every one already here it finishes at once. Cancelled and freed as a model's fetch is.
/// NULL, with waifu_modelmanager_last_error and no callback ever, where it cannot start.
#[no_mangle]
pub unsafe extern "C" fn waifu_modelmanager_fetch_manifests_async(
    user_data: *mut c_void,
    on_progress: WaifuModelManagerProgressCallback,
    on_complete: WaifuModelManagerCompleteCallback,
) -> *mut WaifuModelFetch {
    caught(ptr::null_mut(), || {
        spawn_fetch(user_data, on_progress, on_complete, fetch::fetch_manifests)
    })
}

/// Runs `work` on a thread of its own, with the callbacks it reports to and a stop it can be
/// cancelled by, and hands back the handle that stop is reached through.
fn spawn_fetch(
    user_data: *mut c_void,
    on_progress: WaifuModelManagerProgressCallback,
    on_complete: WaifuModelManagerCompleteCallback,
    work: impl FnOnce(&mut dyn FnMut(&Fetch), &dyn Fn() -> bool) -> Result<(), Failed> + Send + 'static,
) -> *mut WaifuModelFetch {
    let gate = Gate::default();
    let user_data = UserData(user_data);
    let stop = Arc::new(AtomicBool::new(false));

    let thread = thread::Builder::new()
        .name("waifu-fetch".to_string())
        .spawn({
            let stop = Arc::clone(&stop);
            let gate = gate.clone();
            move || {
                let started = Instant::now();
                let mut tell = |fetch: &Fetch| {
                    if on_progress.is_none() {
                        return;
                    }
                    gate.wait();
                    let words = fetch.words();
                    support::tell_progress(
                        on_progress,
                        user_data,
                        &support::Progress {
                            fraction: fetch.fraction(),
                            done: fetch.done,
                            total: fetch.total.unwrap_or(0),
                            part: fetch.part as u32,
                            parts: fetch.parts as u32,
                            file: Some(fetch.file.as_str()).filter(|file| !file.is_empty()),
                            words: &words,
                            seconds: started.elapsed().as_secs_f64(),
                            hub: fetch.hub,
                        },
                    );
                };
                let ended = panic::catch_unwind(AssertUnwindSafe(|| {
                    work(&mut tell, &|| stop.load(Ordering::Relaxed))
                }))
                .unwrap_or_else(|_| {
                    Err(Failed::Fetch(
                        "a bug in the library: the fetch panicked".to_string(),
                    ))
                });

                if on_complete.is_none() {
                    return;
                }
                gate.wait();
                let (code, message) = match &ended {
                    Ok(()) => (WAIFU_MODELMANAGER_OK, None),
                    Err(Failed::Stopped) => (WAIFU_MODELMANAGER_CANCELLED, None),
                    Err(Failed::UnknownModel(message)) => {
                        (WAIFU_MODELMANAGER_ERR_UNKNOWN_MODEL, Some(message.as_str()))
                    }
                    Err(Failed::Fetch(message)) => {
                        (WAIFU_MODELMANAGER_ERR_FETCH, Some(message.as_str()))
                    }
                };
                tell_completion(on_complete, user_data, code, message);
            }
        });
    match thread {
        Ok(thread) => {
            let fetch = Box::into_raw(Box::new(WaifuModelFetch {
                stop,
                thread: Some(thread),
            }));
            gate.open();
            fetch
        }
        Err(error) => {
            set_last_error(&format!("could not start the download's thread: {error}"));
            ptr::null_mut()
        }
    }
}

/// Calls `callback`, if there is one, with how a job that makes nothing ended.
pub fn tell_completion(
    callback: WaifuModelManagerCompleteCallback,
    user_data: UserData,
    code: WaifuModelManagerStatusCode,
    message: Option<&str>,
) {
    let Some(callback) = callback else {
        return;
    };
    let message = message.map(c_string);
    let event = WaifuModelManagerCompletionEvent {
        status: status(code, &message),
    };
    unsafe { callback(user_data.0, &event) };
}

#[no_mangle]
pub unsafe extern "C" fn waifu_modelmanager_fetch_cancel(fetch: *mut WaifuModelFetch) {
    if let Some(fetch) = fetch.as_ref() {
        fetch.stop.store(true, Ordering::Relaxed);
    }
}

/// After on_complete; cancels first if it has not had it, and waits for the thread to end.
#[no_mangle]
pub unsafe extern "C" fn waifu_modelmanager_fetch_free(fetch: *mut WaifuModelFetch) {
    if fetch.is_null() {
        return;
    }
    let mut fetch = Box::from_raw(fetch);
    fetch.stop.store(true, Ordering::Relaxed);
    if let Some(thread) = fetch.thread.take() {
        let _ = thread.join();
    }
}

#[cfg(test)]
mod tests;
