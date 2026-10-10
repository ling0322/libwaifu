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

use super::*;

use std::sync::mpsc::{self, Receiver, Sender};
use std::time::Duration;

const WAIT: Duration = Duration::from_secs(20);

/// What a test hears of one job, sent from the callbacks.
#[derive(Debug, PartialEq)]
enum Heard {
    Progress(WaifuStage, String),
    Complete(WaifuStatusCode, Option<String>),
    Audio(WaifuStatusCode, usize, u32),
    Image(WaifuStatusCode, Option<(u32, u32, usize)>),
}

unsafe fn text(pointer: *const c_char) -> Option<String> {
    (!pointer.is_null()).then(|| CStr::from_ptr(pointer).to_string_lossy().into_owned())
}

unsafe fn sender<'a>(user_data: *mut c_void) -> &'a Sender<Heard> {
    &*(user_data as *const Sender<Heard>)
}

unsafe extern "C" fn on_progress(user_data: *mut c_void, event: *const WaifuProgressEvent) {
    let event = &*event;
    let _ = sender(user_data).send(Heard::Progress(event.stage, text(event.words).unwrap_or_default()));
}

unsafe extern "C" fn on_complete(user_data: *mut c_void, event: *const WaifuCompletionEvent) {
    let status = &(*event).status;
    let _ = sender(user_data).send(Heard::Complete(status.code, text(status.message)));
}

unsafe extern "C" fn on_audio(user_data: *mut c_void, event: *const WaifuAudioCompletionEvent) {
    let event = &*event;
    let (count, rate) = event.audio.as_ref().map_or((0, 0), |audio| (audio.count, audio.rate));
    let _ = sender(user_data).send(Heard::Audio(event.status.code, count, rate));
}

unsafe extern "C" fn on_image(user_data: *mut c_void, event: *const WaifuImageCompletionEvent) {
    let event = &*event;
    let image = event.image.as_ref().map(|image| (image.width, image.height, image.len));
    let _ = sender(user_data).send(Heard::Image(event.status.code, image));
}

/// A channel whose sending end lives as long as the test, behind the pointer the callbacks get.
fn a_channel() -> (Box<Sender<Heard>>, Receiver<Heard>) {
    let (send, receive) = mpsc::channel();
    (Box::new(send), receive)
}

fn user_data(sender: &Sender<Heard>) -> *mut c_void {
    sender as *const Sender<Heard> as *mut c_void
}

/// The first end a job comes to, skipping its progress.
fn the_end(heard: &Receiver<Heard>) -> Heard {
    loop {
        match heard.recv_timeout(WAIT).expect("an end") {
            Heard::Progress(..) => continue,
            end => return end,
        }
    }
}

fn last_error() -> String {
    unsafe { text(waifu_last_error()) }.unwrap_or_default()
}

fn a_speak_request(text: &CStr) -> WaifuSpeakRequest {
    WaifuSpeakRequest {
        struct_size: std::mem::size_of::<WaifuSpeakRequest>() as u32,
        text: text.as_ptr(),
        like: ptr::null(),
        speed: 1.0,
        temperature: 0.8,
        style: ptr::null(),
        seed: 7,
    }
}

#[test]
fn the_abi_is_the_header_s() {
    assert_eq!(waifu_abi_version(), WAIFU_ABI_VERSION);
}

#[test]
fn what_is_published_and_what_a_model_is_are_json_and_freed_by_the_caller() {
    unsafe {
        let catalog = waifu_modelmanager_catalog_json();
        assert!(!catalog.is_null());
        assert!(text(catalog).unwrap().contains("\"sdxl:base\""));
        waifu_string_free(catalog);

        let described = waifu_modelmanager_describe_json(c"sdxl:base".as_ptr());
        assert!(text(described).unwrap().contains("\"image_keys\""));
        waifu_string_free(described);

        assert!(waifu_modelmanager_describe_json(c"no-such-model:v9".as_ptr()).is_null());
        assert!(last_error().contains("no-such-model:v9"), "{}", last_error());

        let machine = waifu_machine_json();
        assert!(text(machine).unwrap().contains("\"accelerators\""));
        waifu_string_free(machine);
    }
}

#[test]
fn a_call_refused_at_the_door_returns_0_and_calls_nothing() {
    unsafe {
        let engine = waifu_engine_new(WaifuDevice::WAIFU_DEVICE_CPU);
        assert!(!engine.is_null(), "{}", last_error());
        let (send, heard) = a_channel();

        // A placeholder no image answers.
        let request = WaifuDrawRequest {
            struct_size: std::mem::size_of::<WaifuDrawRequest>() as u32,
            prompt: c"the girl in <|girl|>".as_ptr(),
            negative: ptr::null(),
            images: ptr::null(),
            image_count: 0,
            width: 64,
            height: 64,
            steps: 1,
            guidance: 5.0,
            seed: 7,
            strength: 0.8,
            control_scale: 1.0,
        };
        let job = waifu_engine_draw_async(engine, &request, user_data(&send), Some(on_progress), Some(on_image));
        assert_eq!(job, 0);
        assert!(last_error().contains("<|girl|>"), "{}", last_error());

        // A struct the caller says is smaller than this library's: compiled against another header.
        let short = WaifuDrawRequest {
            struct_size: 8,
            prompt: c"a cat".as_ptr(),
            ..request
        };
        assert_eq!(waifu_engine_draw_async(engine, &short, user_data(&send), None, Some(on_image)), 0);
        assert!(last_error().contains("struct_size"), "{}", last_error());

        // A NULL where a string has to be.
        assert_eq!(waifu_engine_load_async(engine, ptr::null(), user_data(&send), None, Some(on_complete)), 0);

        waifu_engine_free(engine);
        // Nothing was ever called for any of them.
        assert!(heard.recv_timeout(Duration::from_millis(100)).is_err());
    }
}

#[test]
fn a_voice_is_loaded_and_speaks_through_the_c_api() {
    unsafe {
        let engine = waifu_engine_new(WaifuDevice::WAIFU_DEVICE_CPU);
        assert!(!engine.is_null(), "{}", last_error());

        let (send, heard) = a_channel();
        let job = waifu_engine_load_async(engine, c"tones".as_ptr(), user_data(&send), Some(on_progress), Some(on_complete));
        assert_ne!(job, 0);
        assert_eq!(the_end(&heard), Heard::Complete(WaifuStatusCode::WAIFU_OK, None));

        let (send, heard) = a_channel();
        let request = a_speak_request(c"hello there");
        let job = waifu_engine_speak_async(engine, &request, user_data(&send), Some(on_progress), Some(on_audio));
        assert_ne!(job, 0);

        // Progress before the end, with words for a status line, and the end with the samples.
        let first = heard.recv_timeout(WAIT).unwrap();
        assert!(matches!(first, Heard::Progress(WaifuStage::WAIFU_STAGE_ENCODING, ref words) if !words.is_empty()), "{first:?}");
        match the_end(&heard) {
            Heard::Audio(WaifuStatusCode::WAIFU_OK, count, rate) => {
                assert!(count > 0);
                assert!(rate > 0);
            }
            other => panic!("{other:?}"),
        }

        // An image asked of a voice is the wrong kind, said at the end and not at the call.
        let (send, heard) = a_channel();
        let request = WaifuDrawRequest {
            struct_size: std::mem::size_of::<WaifuDrawRequest>() as u32,
            prompt: c"a cat".as_ptr(),
            negative: c"".as_ptr(),
            images: ptr::null(),
            image_count: 0,
            width: 64,
            height: 64,
            steps: 1,
            guidance: 5.0,
            seed: 7,
            strength: 0.8,
            control_scale: 1.0,
        };
        assert_ne!(waifu_engine_draw_async(engine, &request, user_data(&send), None, Some(on_image)), 0);
        assert_eq!(the_end(&heard), Heard::Image(WaifuStatusCode::WAIFU_ERR_NOT_LOADED, None));

        waifu_engine_free(engine);
    }
}

#[test]
fn freeing_the_engine_ends_every_job_it_had() {
    unsafe {
        let engine = waifu_engine_new(WaifuDevice::WAIFU_DEVICE_CPU);
        let (send, heard) = a_channel();
        waifu_engine_load_async(engine, c"tones".as_ptr(), user_data(&send), None, Some(on_complete));
        let request = a_speak_request(c"hello there");
        for _ in 0..3 {
            waifu_engine_speak_async(engine, &request, user_data(&send), None, Some(on_audio));
        }
        waifu_engine_free(engine);

        // Four jobs, four ends, whatever each came to.
        let ends = heard.try_iter().filter(|heard| !matches!(heard, Heard::Progress(..))).count();
        assert_eq!(ends, 4);
    }
}

#[test]
fn a_model_nobody_published_is_unknown_at_the_end() {
    unsafe {
        let engine = waifu_engine_new(WaifuDevice::WAIFU_DEVICE_CPU);
        let (send, heard) = a_channel();
        assert_ne!(waifu_engine_load_async(engine, c"no-such-model:v9".as_ptr(), user_data(&send), None, Some(on_complete)), 0);
        match the_end(&heard) {
            Heard::Complete(WaifuStatusCode::WAIFU_ERR_UNKNOWN_MODEL, Some(message)) => {
                assert!(message.contains("no-such-model:v9"))
            }
            other => panic!("{other:?}"),
        }
        waifu_engine_free(engine);
    }
}

unsafe extern "C" fn on_log(user_data: *mut c_void, event: *const WaifuLogEvent) {
    let lines = &*(user_data as *const Mutex<Vec<(WaifuLogLevel, String)>>);
    let event = &*event;
    lines
        .lock()
        .unwrap()
        .push((event.level, text(event.message).unwrap_or_default()));
}

#[test]
fn a_log_callback_hears_the_library_s_lines() {
    let lines: Box<Mutex<Vec<(WaifuLogLevel, String)>>> = Box::default();
    waifu_set_log_callback(&*lines as *const _ as *mut c_void, Some(on_log));
    waifu::log::write(waifu::log::Level::Warning, "test", "a line for the app");
    waifu_set_log_callback(ptr::null_mut(), None);

    let lines = lines.lock().unwrap();
    assert!(
        lines
            .iter()
            .any(|(level, line)| *level == WaifuLogLevel::WAIFU_LOG_WARNING && line == "a line for the app"),
        "{lines:?}"
    );
}
