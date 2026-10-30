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

use std::ffi::CStr;
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::Mutex;
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
    let _ = sender(user_data).send(Heard::Progress(
        event.stage,
        text(event.words).unwrap_or_default(),
    ));
}

unsafe extern "C" fn on_complete(user_data: *mut c_void, event: *const WaifuCompletionEvent) {
    let status = &(*event).status;
    let _ = sender(user_data).send(Heard::Complete(status.code, text(status.message)));
}

unsafe extern "C" fn on_audio(user_data: *mut c_void, event: *const WaifuAudioCompletionEvent) {
    let event = &*event;
    let (count, rate) = event
        .audio
        .as_ref()
        .map_or((0, 0), |audio| (audio.count, audio.rate));
    let _ = sender(user_data).send(Heard::Audio(event.status.code, count, rate));
}

unsafe extern "C" fn on_image(user_data: *mut c_void, event: *const WaifuImageCompletionEvent) {
    let event = &*event;
    let image = event
        .image
        .as_ref()
        .map(|image| (image.width, image.height, image.len));
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
fn what_a_model_is_and_the_machine_are_json_and_freed_by_the_caller() {
    unsafe {
        let manifest =
            std::ffi::CString::new(crate::engine::tests::a_manifest("c-api-sdxl", "sdxl")).unwrap();
        let described = waifu_describe_json(manifest.as_ptr());
        assert!(text(described).unwrap().contains("\"image_keys\""));
        waifu_string_free(described);

        assert!(waifu_describe_json(c"no-such-model:v9".as_ptr()).is_null());
        assert!(
            last_error().contains("no-such-model:v9"),
            "{}",
            last_error()
        );

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
        let job = waifu_engine_draw_async(
            engine,
            &request,
            user_data(&send),
            Some(on_progress),
            Some(on_image),
        );
        assert_eq!(job, 0);
        assert!(last_error().contains("<|girl|>"), "{}", last_error());

        // A struct the caller says is smaller than this library's: compiled against another header.
        let short = WaifuDrawRequest {
            struct_size: 8,
            prompt: c"a cat".as_ptr(),
            ..request
        };
        assert_eq!(
            waifu_engine_draw_async(engine, &short, user_data(&send), None, Some(on_image)),
            0
        );
        assert!(last_error().contains("struct_size"), "{}", last_error());

        // A NULL where a string has to be.
        assert_eq!(
            waifu_engine_load_async(
                engine,
                ptr::null(),
                user_data(&send),
                None,
                Some(on_complete)
            ),
            0
        );

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
        let job = waifu_engine_load_async(
            engine,
            c"tones".as_ptr(),
            user_data(&send),
            Some(on_progress),
            Some(on_complete),
        );
        assert_ne!(job, 0);
        assert_eq!(
            the_end(&heard),
            Heard::Complete(WaifuStatusCode::WAIFU_OK, None)
        );

        let (send, heard) = a_channel();
        let request = a_speak_request(c"hello there");
        let job = waifu_engine_speak_async(
            engine,
            &request,
            user_data(&send),
            Some(on_progress),
            Some(on_audio),
        );
        assert_ne!(job, 0);

        // Progress before the end, with words for a status line, and the end with the samples.
        let first = heard.recv_timeout(WAIT).unwrap();
        assert!(
            matches!(first, Heard::Progress(WaifuStage::WAIFU_STAGE_ENCODING, ref words) if !words.is_empty()),
            "{first:?}"
        );
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
        assert_ne!(
            waifu_engine_draw_async(engine, &request, user_data(&send), None, Some(on_image)),
            0
        );
        assert_eq!(
            the_end(&heard),
            Heard::Image(WaifuStatusCode::WAIFU_ERR_NOT_LOADED, None)
        );

        waifu_engine_free(engine);
    }
}

#[test]
fn freeing_the_engine_ends_every_job_it_had() {
    unsafe {
        let engine = waifu_engine_new(WaifuDevice::WAIFU_DEVICE_CPU);
        let (send, heard) = a_channel();
        waifu_engine_load_async(
            engine,
            c"tones".as_ptr(),
            user_data(&send),
            None,
            Some(on_complete),
        );
        let request = a_speak_request(c"hello there");
        for _ in 0..3 {
            waifu_engine_speak_async(engine, &request, user_data(&send), None, Some(on_audio));
        }
        waifu_engine_free(engine);

        // Four jobs, four ends, whatever each came to.
        let ends = heard
            .try_iter()
            .filter(|heard| !matches!(heard, Heard::Progress(..)))
            .count();
        assert_eq!(ends, 4);
    }
}

#[test]
fn a_name_that_is_no_manifest_is_unknown_at_the_end() {
    unsafe {
        let engine = waifu_engine_new(WaifuDevice::WAIFU_DEVICE_CPU);
        let (send, heard) = a_channel();
        assert_ne!(
            waifu_engine_load_async(
                engine,
                c"no-such-model:v9".as_ptr(),
                user_data(&send),
                None,
                Some(on_complete)
            ),
            0
        );
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
    crate::log::write(crate::log::Level::Warning, "a line for the app");
    waifu_set_log_callback(ptr::null_mut(), None);

    let lines = lines.lock().unwrap();
    assert!(
        lines
            .iter()
            .any(|(level, line)| *level == WaifuLogLevel::WAIFU_LOG_WARNING
                && line == "a line for the app"),
        "{lines:?}"
    );
}

#[test]
fn engine_options_compiled_against_another_header_are_refused() {
    unsafe {
        let engine = waifu_engine_new_with_options(ptr::null());
        assert!(engine.is_null());
        assert!(last_error().contains("NULL"), "{}", last_error());

        let short = WaifuEngineOptions {
            struct_size: 4,
            device: WaifuDevice::WAIFU_DEVICE_CPU,
        };
        let engine = waifu_engine_new_with_options(&short);
        assert!(engine.is_null());
        assert!(last_error().contains("struct_size"), "{}", last_error());
    }
}

#[test]
fn a_memory_limit_is_for_a_device_that_has_one() {
    // The processor's allocator holds nothing back, so there is nothing for a limit to limit.
    assert!(!waifu_set_memory_limit(
        WaifuDevice::WAIFU_DEVICE_CPU,
        1 << 30
    ));
    assert!(last_error().contains("cpu"), "{}", last_error());

    // Every device with memory of its own has one, where the device is here: set, and lifted again
    // so that no other test runs under it.
    for (device, waifu_device) in [
        (crate::flint::Device::Metal, WaifuDevice::WAIFU_DEVICE_METAL),
        (crate::flint::Device::Cuda, WaifuDevice::WAIFU_DEVICE_CUDA),
        (
            crate::flint::Device::Vulkan,
            WaifuDevice::WAIFU_DEVICE_VULKAN,
        ),
    ] {
        if device.is_available() {
            assert!(waifu_set_memory_limit(waifu_device, 1 << 40));
            assert!(waifu_set_memory_limit(waifu_device, 0));
        }
    }
}

#[test]
fn a_document_is_written_from_layers_and_freed_by_the_caller() {
    let name = CString::new("front hair").unwrap();
    let pixels = [10u8, 20, 30, 255, 40, 50, 60, 128];
    let layer = WaifuPsdLayer {
        name: name.as_ptr(),
        left: 3,
        top: 4,
        image: WaifuRgbaImage {
            width: 2,
            height: 1,
            rgba: pixels.as_ptr(),
            len: pixels.len(),
        },
    };
    for mode in [
        WaifuPsdMode::WAIFU_PSD_RGB,
        WaifuPsdMode::WAIFU_PSD_GRAYSCALE,
    ] {
        let mut len = 0;
        let written = unsafe { waifu_psd_write(&layer, 1, ptr::null(), 0, 8, 8, mode, &mut len) };
        assert!(!written.is_null(), "{}", last_error());
        let bytes = unsafe { std::slice::from_raw_parts(written, len) };
        assert_eq!(&bytes[..4], b"8BPS");
        // The colour mode the header names: RGB is 3, grayscale 1.
        let colour = u16::from_be_bytes([bytes[24], bytes[25]]);
        assert_eq!(
            colour,
            if mode == WaifuPsdMode::WAIFU_PSD_RGB {
                3
            } else {
                1
            }
        );
        unsafe { waifu_bytes_free(written, len) };
    }

    // Pixels that are not the layer's size are refused, and say so.
    let short = WaifuPsdLayer {
        image: WaifuRgbaImage {
            width: 3,
            ..layer.image
        },
        ..layer
    };
    let mut len = 0;
    let refused = unsafe {
        waifu_psd_write(
            &short,
            1,
            ptr::null(),
            0,
            8,
            8,
            WaifuPsdMode::WAIFU_PSD_RGB,
            &mut len,
        )
    };
    assert!(refused.is_null());
    assert!(last_error().contains("front hair"), "{}", last_error());
    unsafe { waifu_bytes_free(ptr::null_mut(), 0) };

    // With its depth, kept after it in the hidden group, and read back in that order.
    let depth_name = CString::new("depth/front hair").unwrap();
    let whole = |name: &CString| WaifuPsdLayer {
        name: name.as_ptr(),
        left: 3,
        top: 4,
        image: WaifuRgbaImage {
            width: 2,
            height: 1,
            rgba: pixels.as_ptr(),
            len: pixels.len(),
        },
    };
    let depth = whole(&depth_name);
    let written = unsafe {
        waifu_psd_write(
            &whole(&name),
            1,
            &depth,
            1,
            8,
            8,
            WaifuPsdMode::WAIFU_PSD_RGB,
            &mut len,
        )
    };
    assert!(!written.is_null(), "{}", last_error());
    let document =
        crate::see_through::read_psd(unsafe { std::slice::from_raw_parts(written, len) }).unwrap();
    let names: Vec<&str> = document.layers.iter().map(|l| l.name.as_str()).collect();
    assert_eq!(names, ["front hair", "depth/front hair"]);
    unsafe { waifu_bytes_free(written, len) };
}

#[test]
fn a_top_of_two_depths_is_stratified_and_freed_by_the_caller() {
    // A top 20 by 10, its left half near and its right half far.
    let (w, h) = (20usize, 10usize);
    let pixels: Vec<u8> = (0..w * h).flat_map(|_| [90, 90, 90, 255]).collect();
    let depth: Vec<u8> = (0..w * h)
        .flat_map(|p| {
            let d = if p % w < 10 { 40 } else { 200 };
            [d, d, d, 255]
        })
        .collect();
    let name = CString::new("topwear").unwrap();
    let image = |data: &Vec<u8>| WaifuRgbaImage {
        width: w as u32,
        height: h as u32,
        rgba: data.as_ptr(),
        len: data.len(),
    };
    let layer = WaifuDepthLayer {
        name: name.as_ptr(),
        left: 5,
        top: 6,
        image: image(&pixels),
        depth: image(&depth),
    };
    unsafe {
        let out = waifu_stratify(&layer, 1);
        assert!(!out.is_null(), "{}", last_error());
        let layers = std::slice::from_raw_parts((*out).layers, (*out).count);
        let names: Vec<String> = layers
            .iter()
            .map(|l| CStr::from_ptr(l.name).to_string_lossy().into_owned())
            .collect();
        // The back under the front, both where the top was.
        assert_eq!(names, ["topwear-1", "topwear-0"]);
        assert_eq!((layers[1].left, layers[1].top), (5, 6));
        let back = &layers[0];
        assert_eq!((back.image.width, back.depth.width), (20, 20));
        assert_eq!(*back.depth.rgba.add(4 * 19), 200, "its depth, red");
        waifu_depth_layers_free(out);

        // A depth that is not the layer's size is refused.
        let short = WaifuDepthLayer {
            depth: WaifuRgbaImage {
                width: 3,
                ..image(&depth)
            },
            ..layer
        };
        assert!(waifu_stratify(&short, 1).is_null());
        assert!(last_error().contains("topwear"), "{}", last_error());
        waifu_depth_layers_free(ptr::null_mut());
    }
}
