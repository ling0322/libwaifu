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

//! The C API: the engine, what a model is, the machine and the log, as `extern "C"`. Everything
//! outside this crate goes through these. A model is the path of its manifest; listing, fetching
//! and removing published models is libwaifu_modelmanager's, which shares nothing with this.
//!
//! Nothing with any logic in it lives here. This converts -- C strings to `&str`, structs to the
//! engine's requests, what comes back to borrowed C views, panics to status codes -- and keeps
//! the promises docs/ffi.md makes about when a callback can arrive.

#![allow(non_camel_case_types)]
#![allow(clippy::missing_safety_doc)]

use std::ffi::{c_char, c_void, CString};
use std::ptr;

mod support;

use support::{caught, required, set_last_error, status, string_in, string_out, Gate, UserData};
pub use support::{
    WaifuCompleteCallback, WaifuCompletionEvent, WaifuLogCallback, WaifuLogEvent, WaifuLogLevel,
    WaifuProgressCallback, WaifuProgressEvent, WaifuStage, WaifuStatus, WaifuStatusCode,
};

use crate::engine::{
    self, DrawRequest, Ended, Engine, EngineOptions, Failure, SpeakRequest, Stage,
    VoiceConversionRequest,
};
use crate::runtime::{self, DeviceOption};
use crate::wav::Sound;

use WaifuStatusCode::*;

/// What a caller compiled against this header expects. Bumped when anything in it changes in a
/// way an older caller would misread.
pub const WAIFU_ABI_VERSION: u32 = 2;

// -- status ----------------------------------------------------------------------------------

/// The ABI this library was built with: WAIFU_ABI_VERSION.
#[no_mangle]
pub extern "C" fn waifu_abi_version() -> u32 {
    WAIFU_ABI_VERSION
}

/// This thread's last message from this library; valid until this thread's next call into it.
#[no_mangle]
pub extern "C" fn waifu_last_error() -> *const c_char {
    support::last_error()
}

/// Frees a string a synchronous call returned. NULL is fine.
#[no_mangle]
pub unsafe extern "C" fn waifu_string_free(s: *mut c_char) {
    support::free_string(s)
}

// -- logging ---------------------------------------------------------------------------------

/// Every line the library would print -- flint's and the Rust side's -- goes to the callback
/// instead, from whichever thread wrote it. NULL puts them back on the console. Set it before
/// anything else: flint writes what hardware it found the first time a device is asked about.
#[no_mangle]
pub extern "C" fn waifu_set_log_callback(user_data: *mut c_void, callback: WaifuLogCallback) {
    caught((), || {
        crate::log::set_sink(support::log_sink(user_data, callback))
    })
}

/// Lines below the level are not made at all. INFO unless set.
#[no_mangle]
pub extern "C" fn waifu_set_log_level(level: WaifuLogLevel) {
    caught((), || crate::log::set_level(support::log_level(level)))
}

// -- the machine -----------------------------------------------------------------------------

/// The processor, the memory, the card and the accelerators this build can use, as JSON. Free
/// with waifu_string_free. NULL on failure.
#[no_mangle]
pub extern "C" fn waifu_machine_json() -> *mut c_char {
    caught(ptr::null_mut(), || {
        string_out(&engine::machine_json().to_string())
    })
}

/// Every device, as JSON: `{"devices": [{"name", "available", "free"}]}`, named as the -device flag
/// names them -- cpu, cuda, metal, vulkan -- with `free` the bytes the device's
/// allocator may still take where it can be used -- its memory limit, or the device's memory, less
/// what this process holds there -- and null for the processor or a device that cannot be used. Not
/// the card as other programs see it: that is waifu_machine_json's `vram`. Asking CUDA builds a
/// context on the card, a few hundred megabytes of it. Free with waifu_string_free. NULL on
/// failure.
#[no_mangle]
pub extern "C" fn waifu_devices_json() -> *mut c_char {
    caught(ptr::null_mut(), || {
        string_out(&engine::devices_json().to_string())
    })
}

/// This build, as JSON: `{"version", "revision"}`, the crate's version and the full commit it was
/// built from, or null for a build made without git. Free with waifu_string_free. NULL on
/// failure.
#[no_mangle]
pub extern "C" fn waifu_version_json() -> *mut c_char {
    caught(ptr::null_mut(), || {
        let revision = Some(env!("WAIFU_REVISION")).filter(|revision| !revision.is_empty());
        string_out(
            &serde_json::json!({
                "version": env!("CARGO_PKG_VERSION"),
                "revision": revision,
            })
            .to_string(),
        )
    })
}

// -- progress --------------------------------------------------------------------------------

fn stage_out(stage: Stage) -> WaifuStage {
    match stage {
        Stage::Reading => WaifuStage::WAIFU_STAGE_READING,
        Stage::Encoding => WaifuStage::WAIFU_STAGE_ENCODING,
        Stage::Drawing => WaifuStage::WAIFU_STAGE_DRAWING,
        Stage::Decoding => WaifuStage::WAIFU_STAGE_DECODING,
        Stage::Listening => WaifuStage::WAIFU_STAGE_LISTENING,
        Stage::Saying => WaifuStage::WAIFU_STAGE_SAYING,
        Stage::Sounding => WaifuStage::WAIFU_STAGE_SOUNDING,
    }
}

/// The engine's progress callback, calling the caller's.
fn progress_to(
    gate: &Gate,
    user_data: UserData,
    callback: WaifuProgressCallback,
) -> engine::OnProgress {
    let gate = gate.clone();
    Box::new(move |progress| {
        if callback.is_none() {
            return;
        }
        gate.wait();
        support::tell_progress(
            callback,
            user_data,
            &support::Progress {
                stage: stage_out(progress.stage),
                fraction: progress.fraction,
                done: progress.done,
                total: progress.total,
                part: 0,
                parts: 0,
                file: None,
                words: &progress.words,
                seconds: progress.seconds,
            },
        );
    })
}

/// What an ending is called, and its message, as a WaifuStatus needs them -- with the CString kept
/// alive by the caller for as long as the status is.
fn status_of<T>(ended: &Ended<T>) -> (WaifuStatusCode, Option<CString>) {
    match ended {
        Ended::Done(_) => (WAIFU_OK, None),
        Ended::Cancelled => (WAIFU_CANCELLED, None),
        Ended::Failed(failure, message) => (
            match failure {
                Failure::UnknownModel => WAIFU_ERR_UNKNOWN_MODEL,
                Failure::Model => WAIFU_ERR_MODEL,
                Failure::NotLoaded => WAIFU_ERR_NOT_LOADED,
                Failure::Internal => WAIFU_ERR_INTERNAL,
            },
            Some(support::c_string(message)),
        ),
    }
}

fn completion_to<T>(
    gate: &Gate,
    user_data: UserData,
    callback: WaifuCompleteCallback,
) -> engine::OnComplete<T> {
    let gate = gate.clone();
    Box::new(move |ended: Ended<T>| {
        let Some(callback) = callback else {
            return;
        };
        gate.wait();
        let (code, message) = status_of(&ended);
        let event = WaifuCompletionEvent {
            status: status(code, &message),
        };
        let user_data = user_data;
        unsafe { callback(user_data.0, &event) };
    })
}

// -- what a model is -------------------------------------------------------------------------

/// What a model is, without reading it, as JSON: the path of its manifest, whose weights need not
/// be beside it, or "tones". Free with waifu_string_free. NULL, with waifu_last_error, for
/// anything else.
#[no_mangle]
pub unsafe extern "C" fn waifu_describe_json(model: *const c_char) -> *mut c_char {
    caught(ptr::null_mut(), || {
        let model = match required(model, "model") {
            Ok(model) => model,
            Err(error) => {
                set_last_error(&error);
                return ptr::null_mut();
            }
        };
        match engine::describe_json(&model) {
            Some(described) => string_out(&described.to_string()),
            None => {
                set_last_error(&crate::describe::no_manifest_at(&model));
                ptr::null_mut()
            }
        }
    })
}

// -- the engine ------------------------------------------------------------------------------

/// One thread that holds a model and runs jobs on it, one at a time, in the order asked.
pub struct WaifuEngine {
    engine: Engine,
}

/// 0 is never a job: an _async call returns it when it refuses at the call, with the reason in
/// waifu_last_error(), and then calls none of its callbacks. Any other id gets exactly one
/// on_complete.
pub type WaifuJob = u64;

#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WaifuDevice {
    WAIFU_DEVICE_AUTO = 0,
    WAIFU_DEVICE_CPU = 1,
    WAIFU_DEVICE_METAL = 2,
    WAIFU_DEVICE_CUDA = 3,
    /// The same as WAIFU_DEVICE_CUDA, which now keeps on the card what fits of a model and copies
    /// the rest over as each weight is used. Kept for callers built against an older header.
    WAIFU_DEVICE_CUDA_CPU_OFFLOAD = 4,
    WAIFU_DEVICE_VULKAN = 5,
    /// The same as WAIFU_DEVICE_METAL, which now keeps in the model's files whatever does not fit
    /// on the device. Kept for callers built against an older header, and last, so the numbers
    /// above keep their meaning.
    WAIFU_DEVICE_METAL_DISK_OFFLOAD = 6,
}

/// A picture as rows of RGB.
#[repr(C)]
pub struct WaifuImage {
    pub width: u32,
    pub height: u32,
    /// width * height * 3 bytes, rows top to bottom.
    pub rgb: *const u8,
    pub len: usize,
}

/// An image a run is given, by name. A name the prompt writes as <|name|> is read as part of
/// what it is told; "start_from" (img2img: see strength) and "control" (a ControlNet's condition:
/// see control_scale) are kept for images that are drawn with, and the prompt cannot name them.
#[repr(C)]
pub struct WaifuImageInput {
    /// Letters, digits and _.
    pub key: *const c_char,
    /// Any size, scaled by the engine.
    pub image: *const WaifuImage,
}

/// Refused at the call, as WAIFU_ERR_INVALID_ARGUMENT: two images with one key; a <|name|> in the
/// prompt with no image by that name, or naming a kept one; an image the prompt does not name and
/// whose key is not a kept one; and any other <|...|>, which the model would read as one of its
/// own markers.
#[repr(C)]
pub struct WaifuDrawRequest {
    /// sizeof(WaifuDrawRequest), as the caller was compiled with.
    pub struct_size: u32,
    pub prompt: *const c_char,
    /// Words only; "" or NULL for none; ignored by a model without guidance.
    pub negative: *const c_char,
    /// NULL, 0 for none.
    pub images: *const WaifuImageInput,
    pub image_count: usize,
    /// 64..2048, multiples of the model's `alignment` in its description (16; 32 for SDXL).
    pub width: u32,
    pub height: u32,
    pub steps: u32,
    pub guidance: f32,
    /// The caller's: the same seed and settings draw the same image.
    pub seed: u64,
    /// With "start_from": how far it walks away from it, 0..1.
    pub strength: f32,
    /// With "control": how hard it holds to it.
    pub control_scale: f32,
}

#[repr(C)]
pub struct WaifuImageCompletionEvent {
    pub status: WaifuStatus,
    /// NULL unless status.code is WAIFU_OK.
    pub image: *const WaifuImage,
}

pub type WaifuImageCompleteCallback =
    Option<unsafe extern "C" fn(user_data: *mut c_void, event: *const WaifuImageCompletionEvent)>;

/// The samples, not a file: what goes in -- a recording to sound like, or to convert -- and what
/// comes out. Decoding a file to this and writing this to a file are the caller's.
#[repr(C)]
pub struct WaifuAudio {
    /// Mono, -1..1.
    pub samples: *const f32,
    pub count: usize,
    pub rate: u32,
}

#[repr(C)]
pub struct WaifuSpeakRequest {
    /// sizeof(WaifuSpeakRequest), as the caller was compiled with.
    pub struct_size: u32,
    pub text: *const c_char,
    /// A recording to sound like, or NULL.
    pub like: *const WaifuAudio,
    pub speed: f32,
    pub temperature: f32,
    /// NULL or "" for the voice's own way.
    pub style: *const c_char,
    pub seed: u64,
}

#[repr(C)]
pub struct WaifuVoiceConversionRequest {
    /// sizeof(WaifuVoiceConversionRequest), as the caller was compiled with.
    pub struct_size: u32,
    /// What is said.
    pub source: *const WaifuAudio,
    /// Whose voice to say it in.
    pub reference: *const WaifuAudio,
    pub steps: u32,
    pub convert_style: bool,
    pub seed: u64,
}

/// Shared by speak and voice_conversion: both make audio.
#[repr(C)]
pub struct WaifuAudioCompletionEvent {
    pub status: WaifuStatus,
    /// NULL unless status.code is WAIFU_OK.
    pub audio: *const WaifuAudio,
}

pub type WaifuAudioCompleteCallback =
    Option<unsafe extern "C" fn(user_data: *mut c_void, event: *const WaifuAudioCompletionEvent)>;

/// How an engine is started. Zeroed, apart from struct_size, it is waifu_engine_new on
/// WAIFU_DEVICE_AUTO.
#[repr(C)]
pub struct WaifuEngineOptions {
    /// sizeof(WaifuEngineOptions), as the caller was compiled with.
    pub struct_size: u32,
    pub device: WaifuDevice,
}

/// Starts the engine's thread. NULL, with waifu_last_error, if the device is not available.
#[no_mangle]
pub extern "C" fn waifu_engine_new(device: WaifuDevice) -> *mut WaifuEngine {
    let options = WaifuEngineOptions {
        struct_size: std::mem::size_of::<WaifuEngineOptions>() as u32,
        device,
    };
    unsafe { waifu_engine_new_with_options(&options) }
}

/// As waifu_engine_new, the way `options` says. NULL, with waifu_last_error, if the device is not
/// available or the options are refused.
#[no_mangle]
pub unsafe extern "C" fn waifu_engine_new_with_options(
    options: *const WaifuEngineOptions,
) -> *mut WaifuEngine {
    caught(ptr::null_mut(), || {
        let size = options.as_ref().map_or(0, |options| options.struct_size);
        let options = match request_in(options, size, "WaifuEngineOptions") {
            Ok(options) => options,
            Err(error) => {
                set_last_error(&error);
                return ptr::null_mut();
            }
        };
        let options = EngineOptions {
            device: device_option(options.device),
        };
        match Engine::new(options) {
            Ok(engine) => Box::into_raw(Box::new(WaifuEngine { engine })),
            Err(error) => {
                set_last_error(&error);
                ptr::null_mut()
            }
        }
    })
}

/// Limits the memory of `device` -- the tensors of whatever runs on it, the weights kept there
/// among them -- to `bytes`; 0 for no limit but the device's. For the whole process rather than an
/// engine: set it before an engine reads a model. The model's smallest weights then fill most of
/// what it allows, and a run that runs short lets weights on the device go rather than failing.
/// Metal, CUDA and Vulkan have a limit: false, with waifu_last_error, for the processor.
#[no_mangle]
pub extern "C" fn waifu_set_memory_limit(device: WaifuDevice, bytes: u64) -> bool {
    caught(false, || {
        match runtime::set_memory_limit(device_option(device), bytes) {
            Ok(()) => true,
            Err(error) => {
                set_last_error(&error);
                false
            }
        }
    })
}

/// The device a caller asked for, as the engine names it.
fn device_option(device: WaifuDevice) -> DeviceOption {
    match device {
        WaifuDevice::WAIFU_DEVICE_AUTO => DeviceOption::Auto,
        WaifuDevice::WAIFU_DEVICE_CPU => DeviceOption::Cpu,
        WaifuDevice::WAIFU_DEVICE_METAL => DeviceOption::Metal,
        WaifuDevice::WAIFU_DEVICE_CUDA => DeviceOption::Cuda,
        WaifuDevice::WAIFU_DEVICE_CUDA_CPU_OFFLOAD => DeviceOption::Cuda,
        WaifuDevice::WAIFU_DEVICE_VULKAN => DeviceOption::Vulkan,
        WaifuDevice::WAIFU_DEVICE_METAL_DISK_OFFLOAD => DeviceOption::Metal,
    }
}

/// Stops what is running, cancels what waits (each gets on_complete with WAIFU_CANCELLED), drops
/// the model and joins the thread -- which can take until the end of the step that is running.
#[no_mangle]
pub unsafe extern "C" fn waifu_engine_free(engine: *mut WaifuEngine) {
    if !engine.is_null() {
        caught((), || drop(Box::from_raw(engine)));
    }
}

/// Stops a job: after the step it is on where it runs, before it starts where it waits.
#[no_mangle]
pub unsafe extern "C" fn waifu_engine_cancel(engine: *mut WaifuEngine, job: WaifuJob) {
    if let Some(engine) = engine.as_ref() {
        caught((), || engine.engine.cancel(job));
    }
}

#[no_mangle]
pub unsafe extern "C" fn waifu_engine_cancel_all(engine: *mut WaifuEngine) {
    if let Some(engine) = engine.as_ref() {
        caught((), || engine.engine.cancel_all());
    }
}

/// What the engine runs on, named as waifu_devices_json names it -- never "auto", which is settled
/// when it starts. Free with waifu_string_free. NULL for a NULL engine.
#[no_mangle]
pub unsafe extern "C" fn waifu_engine_device(engine: *const WaifuEngine) -> *mut c_char {
    match engine.as_ref() {
        Some(engine) => caught(ptr::null_mut(), || {
            string_out(engine.engine.runtime().name())
        }),
        None => ptr::null_mut(),
    }
}

/// Runs an `_async` entry point: its id, or 0 with the reason where it refuses at the call. The
/// gate opens once the id is ready to be handed back, which is when the job's callbacks may run.
fn submitted(gate: &Gate, submit: impl FnOnce() -> Result<WaifuJob, String>) -> WaifuJob {
    let job = caught(Err(String::new()), submit);
    match job {
        Ok(job) => {
            gate.open();
            job
        }
        Err(error) => {
            if !error.is_empty() {
                set_last_error(&error);
            }
            0
        }
    }
}

/// Reads the model whose manifest is at `model` onto the device, dropping the one before; "tones"
/// is the stand-in voice. A published model is fetched, and its manifest found, with
/// libwaifu_modelmanager first. What was loaded is waifu_describe_json's to say.
#[no_mangle]
pub unsafe extern "C" fn waifu_engine_load_async(
    engine: *mut WaifuEngine,
    model: *const c_char,
    user_data: *mut c_void,
    on_progress: WaifuProgressCallback,
    on_complete: WaifuCompleteCallback,
) -> WaifuJob {
    let gate = Gate::default();
    submitted(&gate, || {
        let engine = engine.as_ref().ok_or("the engine is NULL")?;
        let model = required(model, "model")?;
        let user_data = UserData(user_data);
        Ok(engine.engine.load(
            &model,
            progress_to(&gate, user_data, on_progress),
            completion_to(&gate, user_data, on_complete),
        ))
    })
}

#[no_mangle]
pub unsafe extern "C" fn waifu_engine_unload_async(
    engine: *mut WaifuEngine,
    user_data: *mut c_void,
    on_complete: WaifuCompleteCallback,
) -> WaifuJob {
    let gate = Gate::default();
    submitted(&gate, || {
        let engine = engine.as_ref().ok_or("the engine is NULL")?;
        Ok(engine
            .engine
            .unload(completion_to(&gate, UserData(user_data), on_complete)))
    })
}

// -- images ----------------------------------------------------------------------------------

/// A request struct, read only if it is at least as large as the one this library knows.
unsafe fn request_in<'a, T>(request: *const T, size: u32, what: &str) -> Result<&'a T, String> {
    let request = request
        .as_ref()
        .ok_or_else(|| format!("the {what} is NULL"))?;
    if (size as usize) < std::mem::size_of::<T>() {
        return Err(format!(
            "the {what} says it is {size} bytes, and this library reads {}: set struct_size to sizeof",
            std::mem::size_of::<T>()
        ));
    }
    Ok(request)
}

unsafe fn image_in(image: *const WaifuImage, what: &str) -> Result<engine::Image, String> {
    let image = image.as_ref().ok_or_else(|| format!("{what} is NULL"))?;
    if image.rgb.is_null() && image.len > 0 {
        return Err(format!("{what} has no pixels"));
    }
    let rgb = match image.len {
        0 => Vec::new(),
        len => std::slice::from_raw_parts(image.rgb, len).to_vec(),
    };
    Ok(engine::Image {
        width: image.width,
        height: image.height,
        rgb,
    })
}

/// Draws an image. 0, with the reason, for a request no model could run.
#[no_mangle]
pub unsafe extern "C" fn waifu_engine_draw_async(
    engine: *mut WaifuEngine,
    request: *const WaifuDrawRequest,
    user_data: *mut c_void,
    on_progress: WaifuProgressCallback,
    on_complete: WaifuImageCompleteCallback,
) -> WaifuJob {
    let gate = Gate::default();
    submitted(&gate, || {
        let engine = engine.as_ref().ok_or("the engine is NULL")?;
        let size = request.as_ref().map_or(0, |request| request.struct_size);
        let request = request_in(request, size, "WaifuDrawRequest")?;

        let mut images = Vec::with_capacity(request.image_count);
        if request.image_count > 0 {
            if request.images.is_null() {
                return Err("image_count is not 0 and images is NULL".to_string());
            }
            for input in std::slice::from_raw_parts(request.images, request.image_count) {
                let key = required(input.key, "an image's key")?;
                let image = image_in(input.image, &format!("the image under \"{key}\""))?;
                images.push((key, image));
            }
        }

        let request = DrawRequest {
            prompt: required(request.prompt, "prompt")?,
            negative: string_in(request.negative, "negative")?.unwrap_or_default(),
            images,
            width: request.width,
            height: request.height,
            steps: request.steps,
            guidance: request.guidance,
            seed: request.seed,
            strength: request.strength,
            control_scale: request.control_scale,
        };

        let user_data = UserData(user_data);
        let on_complete = {
            let gate = gate.clone();
            Box::new(move |ended: Ended<engine::Image>| {
                let Some(callback) = on_complete else {
                    return;
                };
                gate.wait();
                let (code, message) = status_of(&ended);
                let image = match &ended {
                    Ended::Done(image) => Some(WaifuImage {
                        width: image.width,
                        height: image.height,
                        rgb: image.rgb.as_ptr(),
                        len: image.rgb.len(),
                    }),
                    _ => None,
                };
                let event = WaifuImageCompletionEvent {
                    status: status(code, &message),
                    image: image
                        .as_ref()
                        .map_or(ptr::null(), |image| image as *const WaifuImage),
                };
                let user_data = user_data;
                unsafe { callback(user_data.0, &event) };
            })
        };
        engine.engine.draw(
            request,
            progress_to(&gate, user_data, on_progress),
            on_complete,
        )
    })
}

// -- audio -----------------------------------------------------------------------------------

unsafe fn audio_in(audio: *const WaifuAudio, what: &str) -> Result<Sound, String> {
    let audio = audio.as_ref().ok_or_else(|| format!("{what} is NULL"))?;
    if audio.samples.is_null() && audio.count > 0 {
        return Err(format!("{what} has no samples"));
    }
    if audio.rate == 0 {
        return Err(format!("{what} has a rate of 0"));
    }
    let samples = match audio.count {
        0 => Vec::new(),
        count => std::slice::from_raw_parts(audio.samples, count).to_vec(),
    };
    Ok(Sound::new(samples, audio.rate))
}

fn audio_completion_to(
    gate: &Gate,
    user_data: UserData,
    callback: WaifuAudioCompleteCallback,
) -> engine::OnComplete<Sound> {
    let gate = gate.clone();
    Box::new(move |ended: Ended<Sound>| {
        let Some(callback) = callback else {
            return;
        };
        gate.wait();
        let (code, message) = status_of(&ended);
        let audio = match &ended {
            Ended::Done(sound) => Some(WaifuAudio {
                samples: sound.samples.as_ptr(),
                count: sound.samples.len(),
                rate: sound.rate,
            }),
            _ => None,
        };
        let event = WaifuAudioCompletionEvent {
            status: status(code, &message),
            audio: audio
                .as_ref()
                .map_or(ptr::null(), |audio| audio as *const WaifuAudio),
        };
        let user_data = user_data;
        unsafe { callback(user_data.0, &event) };
    })
}

/// Reads text out in the loaded voice.
#[no_mangle]
pub unsafe extern "C" fn waifu_engine_speak_async(
    engine: *mut WaifuEngine,
    request: *const WaifuSpeakRequest,
    user_data: *mut c_void,
    on_progress: WaifuProgressCallback,
    on_complete: WaifuAudioCompleteCallback,
) -> WaifuJob {
    let gate = Gate::default();
    submitted(&gate, || {
        let engine = engine.as_ref().ok_or("the engine is NULL")?;
        let size = request.as_ref().map_or(0, |request| request.struct_size);
        let request = request_in(request, size, "WaifuSpeakRequest")?;
        let like = match request.like.is_null() {
            true => None,
            false => Some(audio_in(request.like, "like")?),
        };
        let request = SpeakRequest {
            text: required(request.text, "text")?,
            like,
            speed: request.speed,
            temperature: request.temperature,
            style: string_in(request.style, "style")?,
            seed: request.seed,
        };
        let user_data = UserData(user_data);
        engine.engine.speak(
            request,
            progress_to(&gate, user_data, on_progress),
            audio_completion_to(&gate, user_data, on_complete),
        )
    })
}

/// Says a recording again in another voice: what is said, and how, is the source's; whose voice
/// it is, the reference's.
#[no_mangle]
pub unsafe extern "C" fn waifu_engine_voice_conversion_async(
    engine: *mut WaifuEngine,
    request: *const WaifuVoiceConversionRequest,
    user_data: *mut c_void,
    on_progress: WaifuProgressCallback,
    on_complete: WaifuAudioCompleteCallback,
) -> WaifuJob {
    let gate = Gate::default();
    submitted(&gate, || {
        let engine = engine.as_ref().ok_or("the engine is NULL")?;
        let size = request.as_ref().map_or(0, |request| request.struct_size);
        let request = request_in(request, size, "WaifuVoiceConversionRequest")?;
        let request = VoiceConversionRequest {
            source: audio_in(request.source, "source")?,
            reference: audio_in(request.reference, "reference")?,
            steps: request.steps,
            convert_style: request.convert_style,
            seed: request.seed,
        };
        let user_data = UserData(user_data);
        engine.engine.voice_conversion(
            request,
            progress_to(&gate, user_data, on_progress),
            audio_completion_to(&gate, user_data, on_complete),
        )
    })
}

#[cfg(test)]
mod tests;
