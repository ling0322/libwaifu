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

//! The C API, called the way C calls it: through the `extern "C"` functions, with callbacks that
//! are `extern "C"` functions and a `user_data` pointer -- here, to a channel the test waits on.
//! Nothing here touches the network.

use super::*;

use std::ffi::CStr;
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::Mutex;
use std::time::Duration;


const WAIT: Duration = Duration::from_secs(20);

/// What a test hears of one fetch, sent from the callbacks.
#[derive(Debug, PartialEq)]
enum Heard {
    Progress(String),
    Complete(WaifuModelManagerStatusCode, Option<String>),
}

unsafe fn text(pointer: *const c_char) -> Option<String> {
    (!pointer.is_null()).then(|| CStr::from_ptr(pointer).to_string_lossy().into_owned())
}

unsafe fn sender<'a>(user_data: *mut c_void) -> &'a Sender<Heard> {
    &*(user_data as *const Sender<Heard>)
}

unsafe extern "C" fn on_progress(user_data: *mut c_void, event: *const WaifuModelManagerProgressEvent) {
    let event = &*event;
    let _ = sender(user_data).send(Heard::Progress(text(event.words).unwrap_or_default()));
}

unsafe extern "C" fn on_complete(user_data: *mut c_void, event: *const WaifuModelManagerCompletionEvent) {
    let status = &(*event).status;
    let _ = sender(user_data).send(Heard::Complete(status.code, text(status.message)));
}

fn last_error() -> String {
    unsafe { text(waifu_modelmanager_last_error()) }.unwrap_or_default()
}

#[test]
fn the_abi_is_the_header_s() {
    assert_eq!(waifu_modelmanager_abi_version(), WAIFU_MODELMANAGER_ABI_VERSION);
}

#[test]
fn what_is_published_is_json_and_freed_by_the_caller() {
    unsafe {
        let catalog = waifu_modelmanager_catalog_json();
        assert!(!catalog.is_null());
        assert!(text(catalog).unwrap().contains("\"sdxl:base\""));
        waifu_modelmanager_string_free(catalog);
    }
}

#[test]
fn where_models_go_is_json_and_says_who_said_so() {
    unsafe {
        let said = waifu_modelmanager_directory_json();
        assert!(!said.is_null(), "{}", last_error());
        let text = text(said).unwrap();
        waifu_modelmanager_string_free(said);
        for key in ["\"directory\"", "\"said\"", "\"default\"", "\"config_file\""] {
            assert!(text.contains(key), "{text}");
        }
    }
}

#[test]
fn a_manifest_path_is_a_path_or_none() {
    unsafe {
        // A path that is a manifest is handed back as it is.
        let file = std::env::temp_dir().join(format!("waifu-c-api-{}.yaml", std::process::id()));
        std::fs::write(&file, "weights:\n  - a.safetensors\n").unwrap();
        let asked = std::ffi::CString::new(file.to_str().unwrap()).unwrap();
        let path = waifu_modelmanager_manifest_path(asked.as_ptr());
        assert_eq!(text(path).as_deref(), file.to_str());
        waifu_modelmanager_string_free(path);
        std::fs::remove_file(&file).unwrap();

        assert!(waifu_modelmanager_manifest_path(c"no-such-model:v9".as_ptr()).is_null());
        assert!(last_error().contains("no-such-model:v9"), "{}", last_error());
        assert!(waifu_modelmanager_manifest_path(ptr::null()).is_null());
    }
}

#[test]
fn bundled_manifests_are_a_directory_or_none() {
    unsafe {
        // Where they are is the caller's; what is read from them is hub's to test.
        let directory = std::env::temp_dir();
        let directory = std::ffi::CString::new(directory.to_str().expect("a UTF-8 path")).unwrap();
        assert_eq!(waifu_modelmanager_set_bundled_manifests(directory.as_ptr()), WAIFU_MODELMANAGER_OK);
        assert_eq!(waifu_modelmanager_set_bundled_manifests(c"".as_ptr()), WAIFU_MODELMANAGER_OK);
        assert_eq!(waifu_modelmanager_set_bundled_manifests(ptr::null()), WAIFU_MODELMANAGER_OK);
    }
}

#[test]
fn removing_a_model_nobody_published_is_refused_by_name() {
    unsafe {
        assert_eq!(waifu_modelmanager_remove(c"no-such-model:v9".as_ptr()), WAIFU_MODELMANAGER_ERR_UNKNOWN_MODEL);
        assert!(last_error().contains("no-such-model:v9"), "{}", last_error());
        assert_eq!(waifu_modelmanager_remove(ptr::null()), WAIFU_MODELMANAGER_ERR_INVALID_ARGUMENT);
    }
}

#[test]
fn a_fetch_refused_at_the_door_is_null_and_calls_nothing() {
    unsafe {
        let (send, heard) = mpsc::channel();
        let user_data = &send as *const Sender<Heard> as *mut c_void;
        let fetch = waifu_modelmanager_fetch_async(ptr::null(), user_data, Some(on_progress), Some(on_complete));
        assert!(fetch.is_null());
        assert!(last_error().contains("model"), "{}", last_error());
        assert!(heard.recv_timeout(Duration::from_millis(100)).is_err());
    }
}

#[test]
fn a_model_nobody_published_is_unknown_at_the_end() {
    unsafe {
        let (send, heard): (Sender<Heard>, Receiver<Heard>) = mpsc::channel();
        let user_data = &send as *const Sender<Heard> as *mut c_void;
        let fetch = waifu_modelmanager_fetch_async(
            c"no-such-model:v9".as_ptr(),
            user_data,
            Some(on_progress),
            Some(on_complete),
        );
        assert!(!fetch.is_null(), "{}", last_error());
        match heard.recv_timeout(WAIT).expect("an end") {
            Heard::Complete(WAIFU_MODELMANAGER_ERR_UNKNOWN_MODEL, Some(message)) => {
                assert!(message.contains("no-such-model:v9"), "{message}")
            }
            other => panic!("{other:?}"),
        }
        waifu_modelmanager_fetch_free(fetch);
    }
}

unsafe extern "C" fn on_log(user_data: *mut c_void, event: *const WaifuModelManagerLogEvent) {
    let lines = &*(user_data as *const Mutex<Vec<(WaifuModelManagerLogLevel, String)>>);
    let event = &*event;
    lines
        .lock()
        .unwrap()
        .push((event.level, text(event.message).unwrap_or_default()));
}

#[test]
fn a_log_callback_hears_the_library_s_lines() {
    let lines: Box<Mutex<Vec<(WaifuModelManagerLogLevel, String)>>> = Box::default();
    waifu_modelmanager_set_log_callback(&*lines as *const _ as *mut c_void, Some(on_log));
    log::write(log::Level::Warning, "hub", "a line for the app");
    waifu_modelmanager_set_log_callback(ptr::null_mut(), None);

    let lines = lines.lock().unwrap();
    assert!(
        lines
            .iter()
            .any(|(level, line)| *level == WaifuModelManagerLogLevel::WAIFU_MODELMANAGER_LOG_WARNING && line == "a line for the app"),
        "{lines:?}"
    );
}
