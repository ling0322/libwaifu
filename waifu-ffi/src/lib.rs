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

//! The C API over libwaifu: the engine, the model manager and the log, as `extern "C"`.
//!
//! Nothing with any logic in it lives here. This converts -- C strings to `&str`, structs to the
//! engine's requests, what comes back to borrowed C views, panics to status codes -- and keeps
//! the promises docs/ffi.md makes about when a callback can arrive.

#![allow(non_camel_case_types)]
#![allow(clippy::missing_safety_doc)]

use std::cell::RefCell;
use std::ffi::{c_char, c_void, CStr, CString};
use std::panic::{self, AssertUnwindSafe};
use std::ptr;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::{self, JoinHandle};

use waifu::engine::{self, DrawRequest, Ended, Engine, Failure, SpeakRequest, Stage, VoiceConversionRequest};
use waifu::runtime::DeviceOption;
use waifu::wav::Sound;

/// What a caller compiled against this header expects. Bumped when anything in it changes in a
/// way an older caller would misread.
pub const WAIFU_ABI_VERSION: u32 = 1;

// -- status ----------------------------------------------------------------------------------

/// What a call or a job came to.
#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WaifuStatusCode {
    WAIFU_OK = 0,
    WAIFU_CANCELLED = 1,
    /// A NULL, a string that is not UTF-8, a size out of range, a request that names an image
    /// it does not give.
    WAIFU_ERR_INVALID_ARGUMENT = 2,
    /// Neither a published name nor a manifest that exists.
    WAIFU_ERR_UNKNOWN_MODEL = 3,
    /// The download failed.
    WAIFU_ERR_FETCH = 4,
    /// The model could not be read, or cannot do what was asked.
    WAIFU_ERR_MODEL = 5,
    /// A job with no model loaded, or a model of the wrong kind.
    WAIFU_ERR_NOT_LOADED = 6,
    /// A panic in the library: a bug.
    WAIFU_ERR_INTERNAL = 7,
}

use WaifuStatusCode::*;

/// How a job ended, as on_complete is told it.
#[repr(C)]
pub struct WaifuStatus {
    pub code: WaifuStatusCode,
    /// Why, when code is not WAIFU_OK; NULL when it is.
    pub message: *const c_char,
}

thread_local! {
    static LAST_ERROR: RefCell<CString> = RefCell::new(CString::default());
}

fn set_last_error(message: &str) {
    let message = CString::new(message.replace('\0', " ")).unwrap_or_default();
    LAST_ERROR.with(|last| *last.borrow_mut() = message);
}

/// The ABI this library was built with: WAIFU_ABI_VERSION.
#[no_mangle]
pub extern "C" fn waifu_abi_version() -> u32 {
    WAIFU_ABI_VERSION
}

/// This thread's last message; valid until this thread's next call.
#[no_mangle]
pub extern "C" fn waifu_last_error() -> *const c_char {
    LAST_ERROR.with(|last| last.borrow().as_ptr())
}

/// Frees a string a synchronous call returned. NULL is fine.
#[no_mangle]
pub unsafe extern "C" fn waifu_string_free(s: *mut c_char) {
    if !s.is_null() {
        drop(CString::from_raw(s));
    }
}

/// Runs a synchronous entry point, turning a panic into WAIFU_ERR_INTERNAL with its message.
fn caught<T>(otherwise: T, body: impl FnOnce() -> T) -> T {
    match panic::catch_unwind(AssertUnwindSafe(body)) {
        Ok(value) => value,
        Err(cause) => {
            let said = cause
                .downcast_ref::<&str>()
                .map(|said| said.to_string())
                .or_else(|| cause.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "it panicked".to_string());
            set_last_error(&format!("a bug in the library: {said}"));
            otherwise
        }
    }
}

/// A string handed in, or the reason it cannot be read. NULL is `allow_null`'s to decide.
unsafe fn string_in(pointer: *const c_char, what: &str) -> Result<Option<String>, String> {
    if pointer.is_null() {
        return Ok(None);
    }
    CStr::from_ptr(pointer)
        .to_str()
        .map(|text| Some(text.to_string()))
        .map_err(|_| format!("{what} is not UTF-8"))
}

unsafe fn required(pointer: *const c_char, what: &str) -> Result<String, String> {
    string_in(pointer, what)?.ok_or_else(|| format!("{what} is NULL"))
}

fn string_out(text: &str) -> *mut c_char {
    CString::new(text.replace('\0', " ")).unwrap_or_default().into_raw()
}

// -- logging ---------------------------------------------------------------------------------

#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WaifuLogLevel {
    WAIFU_LOG_DEBUG = 0,
    WAIFU_LOG_INFO = 1,
    WAIFU_LOG_WARNING = 2,
    WAIFU_LOG_ERROR = 3,
    /// The last line: the process aborts when the callback returns.
    WAIFU_LOG_FATAL = 4,
}

#[repr(C)]
pub struct WaifuLogEvent {
    pub level: WaifuLogLevel,
    /// Where it was written: "interface.cc:84" for flint, a module's name for Rust.
    pub source: *const c_char,
    /// The line alone, with no level or time in front of it.
    pub message: *const c_char,
}

/// Called on whichever thread wrote the line, which can be the caller's own. It must be safe to
/// call from any thread, and must not call into the library.
pub type WaifuLogCallback = Option<unsafe extern "C" fn(user_data: *mut c_void, event: *const WaifuLogEvent)>;

/// A caller's pointer, handed back as it was given, on whichever thread a callback runs.
#[derive(Clone, Copy)]
struct UserData(*mut c_void);
unsafe impl Send for UserData {}
unsafe impl Sync for UserData {}

/// Every line the library would print -- flint's and the Rust side's -- goes to the callback
/// instead, from whichever thread wrote it. NULL puts them back on the console. Set it before
/// anything else: flint writes what hardware it found the first time a device is asked about.
#[no_mangle]
pub extern "C" fn waifu_set_log_callback(user_data: *mut c_void, callback: WaifuLogCallback) {
    caught((), || {
        let Some(callback) = callback else {
            waifu::log::set_sink(None);
            return;
        };
        let user_data = UserData(user_data);
        waifu::log::set_sink(Some(Box::new(move |level, source, message| {
            let source = CString::new(source.replace('\0', " ")).unwrap_or_default();
            let message = CString::new(message.replace('\0', " ")).unwrap_or_default();
            let event = WaifuLogEvent {
                level: match level {
                    waifu::log::Level::Debug => WaifuLogLevel::WAIFU_LOG_DEBUG,
                    waifu::log::Level::Info => WaifuLogLevel::WAIFU_LOG_INFO,
                    waifu::log::Level::Warning => WaifuLogLevel::WAIFU_LOG_WARNING,
                    waifu::log::Level::Error => WaifuLogLevel::WAIFU_LOG_ERROR,
                    waifu::log::Level::Fatal => WaifuLogLevel::WAIFU_LOG_FATAL,
                },
                source: source.as_ptr(),
                message: message.as_ptr(),
            };
            let user_data = user_data;
            unsafe { callback(user_data.0, &event) };
        })));
    })
}

/// Lines below the level are not made at all. INFO unless set.
#[no_mangle]
pub extern "C" fn waifu_set_log_level(level: WaifuLogLevel) {
    caught((), || {
        waifu::log::set_level(match level {
            WaifuLogLevel::WAIFU_LOG_DEBUG => waifu::log::Level::Debug,
            WaifuLogLevel::WAIFU_LOG_INFO => waifu::log::Level::Info,
            WaifuLogLevel::WAIFU_LOG_WARNING => waifu::log::Level::Warning,
            WaifuLogLevel::WAIFU_LOG_ERROR => waifu::log::Level::Error,
            WaifuLogLevel::WAIFU_LOG_FATAL => waifu::log::Level::Fatal,
        })
    })
}

// -- the machine -----------------------------------------------------------------------------

/// The processor, the memory, the card and the accelerators this build can use, as JSON. Free
/// with waifu_string_free. NULL on failure.
#[no_mangle]
pub extern "C" fn waifu_machine_json() -> *mut c_char {
    caught(ptr::null_mut(), || string_out(&engine::machine_json().to_string()))
}

// -- progress --------------------------------------------------------------------------------

#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WaifuStage {
    /// file, done/total in bytes, part/parts.
    WAIFU_STAGE_FETCHING = 0,
    /// Reading weights onto the device: no fraction.
    WAIFU_STAGE_READING = 1,
    /// The prompt, or the picture img2img starts from.
    WAIFU_STAGE_ENCODING = 2,
    /// done/total in steps.
    WAIFU_STAGE_DRAWING = 3,
    WAIFU_STAGE_DECODING = 4,
    /// A conversion reading its recordings.
    WAIFU_STAGE_LISTENING = 5,
    /// done/total in tokens; total is an estimate and can be passed.
    WAIFU_STAGE_SAYING = 6,
    /// The vocoder.
    WAIFU_STAGE_SOUNDING = 7,
}

#[repr(C)]
pub struct WaifuProgressEvent {
    pub stage: WaifuStage,
    /// 0..1 of the whole job, or -1 where nothing can say.
    pub fraction: f64,
    /// What the stage counts; 0, 0 where it counts nothing.
    pub done: u64,
    pub total: u64,
    /// FETCHING only: which file of how many.
    pub part: u32,
    pub parts: u32,
    /// FETCHING only, else NULL.
    pub file: *const c_char,
    /// "step 3 of 8", for a status line.
    pub words: *const c_char,
    /// Since the job started.
    pub seconds: f64,
}

pub type WaifuProgressCallback = Option<unsafe extern "C" fn(user_data: *mut c_void, event: *const WaifuProgressEvent)>;

/// How a job that makes nothing ended: a fetch, a load, an unload.
#[repr(C)]
pub struct WaifuCompletionEvent {
    pub status: WaifuStatus,
}

pub type WaifuCompleteCallback = Option<unsafe extern "C" fn(user_data: *mut c_void, event: *const WaifuCompletionEvent)>;

// -- keeping the promises about when a callback can arrive ---------------------------------------

/// Shut until the `_async` call that made it has returned: a callback that arrives before then
/// waits. The engine's thread can finish a job before the call that queued it is done handing
/// back its id, and a caller whose continuation is set up from that id would miss the end.
#[derive(Clone, Default)]
struct Gate(Arc<(Mutex<bool>, Condvar)>);

impl Gate {
    fn open(&self) {
        let (open, opened) = &*self.0;
        *open.lock().unwrap_or_else(|held| held.into_inner()) = true;
        opened.notify_all();
    }

    fn wait(&self) {
        let (open, opened) = &*self.0;
        let mut is_open = open.lock().unwrap_or_else(|held| held.into_inner());
        while !*is_open {
            is_open = opened.wait(is_open).unwrap_or_else(|held| held.into_inner());
        }
    }
}

fn stage_out(stage: Stage) -> WaifuStage {
    match stage {
        Stage::Fetching => WaifuStage::WAIFU_STAGE_FETCHING,
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
fn progress_to(gate: &Gate, user_data: UserData, callback: WaifuProgressCallback) -> engine::OnProgress {
    let gate = gate.clone();
    Box::new(move |progress| {
        let Some(callback) = callback else {
            return;
        };
        gate.wait();
        let file = progress.file.as_deref().map(|file| CString::new(file.replace('\0', " ")).unwrap_or_default());
        let words = CString::new(progress.words.replace('\0', " ")).unwrap_or_default();
        let event = WaifuProgressEvent {
            stage: stage_out(progress.stage),
            fraction: progress.fraction.unwrap_or(-1.0),
            done: progress.done,
            total: progress.total,
            part: progress.part,
            parts: progress.parts,
            file: file.as_ref().map_or(ptr::null(), |file| file.as_ptr()),
            words: words.as_ptr(),
            seconds: progress.seconds,
        };
        let user_data = user_data;
        unsafe { callback(user_data.0, &event) };
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
                Failure::Fetch => WAIFU_ERR_FETCH,
                Failure::Model => WAIFU_ERR_MODEL,
                Failure::NotLoaded => WAIFU_ERR_NOT_LOADED,
                Failure::Internal => WAIFU_ERR_INTERNAL,
            },
            Some(CString::new(message.replace('\0', " ")).unwrap_or_default()),
        ),
    }
}

fn status(code: WaifuStatusCode, message: &Option<CString>) -> WaifuStatus {
    WaifuStatus {
        code,
        message: message.as_ref().map_or(ptr::null(), |message| message.as_ptr()),
    }
}

fn completion_to<T>(gate: &Gate, user_data: UserData, callback: WaifuCompleteCallback) -> engine::OnComplete<T> {
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

// -- the model manager -----------------------------------------------------------------------

/// The published models, aliases only, as JSON. Free with waifu_string_free. NULL on failure.
#[no_mangle]
pub extern "C" fn waifu_modelmanager_catalog_json() -> *mut c_char {
    caught(ptr::null_mut(), || string_out(&engine::catalog_json().to_string()))
}

/// What a model is, without reading it, as JSON: a published name or a manifest path. Free with
/// waifu_string_free. NULL, with waifu_last_error, for a name that is neither.
#[no_mangle]
pub unsafe extern "C" fn waifu_modelmanager_describe_json(model: *const c_char) -> *mut c_char {
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
                set_last_error(&format!("there is no model called \"{model}\""));
                ptr::null_mut()
            }
        }
    })
}

/// Where downloads go. Free with waifu_string_free. NULL on failure.
#[no_mangle]
pub extern "C" fn waifu_modelmanager_directory() -> *mut c_char {
    caught(ptr::null_mut(), || match waifu::hub::model_directory() {
        Ok((directory, _)) => string_out(&directory.to_string_lossy()),
        Err(error) => {
            set_last_error(&error.to_string());
            ptr::null_mut()
        }
    })
}

/// Downloads go to `path` from now on, saved to config.toml; NULL or "" for the default.
#[no_mangle]
pub unsafe extern "C" fn waifu_modelmanager_set_directory(path: *const c_char) -> WaifuStatusCode {
    caught(WAIFU_ERR_INTERNAL, || {
        let path = match string_in(path, "path") {
            Ok(path) => path.filter(|path| !path.is_empty()),
            Err(error) => {
                set_last_error(&error);
                return WAIFU_ERR_INVALID_ARGUMENT;
            }
        };
        match waifu::config::set_model_dir(path.as_deref().map(std::path::Path::new)) {
            Ok(_) => WAIFU_OK,
            Err(error) => {
                set_last_error(&error.to_string());
                WAIFU_ERR_INTERNAL
            }
        }
    })
}

/// Deletes a downloaded package.
#[no_mangle]
pub unsafe extern "C" fn waifu_modelmanager_remove(name: *const c_char) -> WaifuStatusCode {
    caught(WAIFU_ERR_INTERNAL, || {
        let name = match required(name, "name") {
            Ok(name) => name,
            Err(error) => {
                set_last_error(&error);
                return WAIFU_ERR_INVALID_ARGUMENT;
            }
        };
        if waifu::hub::full_name(&name).is_none() {
            set_last_error(&format!("there is no published model called \"{name}\""));
            return WAIFU_ERR_UNKNOWN_MODEL;
        }
        match waifu::hub::remove(&name) {
            Ok(()) => WAIFU_OK,
            Err(error) => {
                set_last_error(&error.to_string());
                WAIFU_ERR_INTERNAL
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
/// already here finishes at once. Where it went is waifu_modelmanager_describe_json's to say.
/// NULL, with waifu_last_error and no callback ever, where it cannot start.
#[no_mangle]
pub unsafe extern "C" fn waifu_modelmanager_fetch_async(
    model: *const c_char,
    user_data: *mut c_void,
    on_progress: WaifuProgressCallback,
    on_complete: WaifuCompleteCallback,
) -> *mut WaifuModelFetch {
    caught(ptr::null_mut(), || {
        let model = match required(model, "model") {
            Ok(model) => model,
            Err(error) => {
                set_last_error(&error);
                return ptr::null_mut();
            }
        };
        let gate = Gate::default();
        let user_data = UserData(user_data);
        let mut on_progress = progress_to(&gate, user_data, on_progress);
        let on_complete = completion_to::<std::path::PathBuf>(&gate, user_data, on_complete);
        let stop = Arc::new(AtomicBool::new(false));

        let thread = thread::Builder::new().name("waifu-fetch".to_string()).spawn({
            let stop = Arc::clone(&stop);
            move || {
                let ended = panic::catch_unwind(AssertUnwindSafe(|| {
                    engine::fetch(&model, &mut on_progress, &|| stop.load(Ordering::Relaxed))
                }))
                .unwrap_or_else(|_| Ended::Failed(Failure::Internal, "a bug in the library: the fetch panicked".to_string()));
                on_complete(ended);
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
    })
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

// -- the engine ------------------------------------------------------------------------------

#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WaifuDevice {
    WAIFU_DEVICE_AUTO = 0,
    WAIFU_DEVICE_CPU = 1,
    WAIFU_DEVICE_METAL = 2,
    WAIFU_DEVICE_CUDA = 3,
    WAIFU_DEVICE_CUDA_CPU_OFFLOAD = 4,
    WAIFU_DEVICE_VULKAN = 5,
}

/// One thread that holds a model and runs jobs on it, one at a time, in the order asked.
pub struct WaifuEngine {
    engine: Engine,
}

/// 0 is never a job: an _async call returns it when it refuses at the call, with the reason in
/// waifu_last_error(), and then calls none of its callbacks. Any other id gets exactly one
/// on_complete.
pub type WaifuJob = u64;

/// Starts the engine's thread. NULL, with waifu_last_error, if the device is not available.
#[no_mangle]
pub extern "C" fn waifu_engine_new(device: WaifuDevice) -> *mut WaifuEngine {
    caught(ptr::null_mut(), || {
        let option = match device {
            WaifuDevice::WAIFU_DEVICE_AUTO => DeviceOption::Auto,
            WaifuDevice::WAIFU_DEVICE_CPU => DeviceOption::Cpu,
            WaifuDevice::WAIFU_DEVICE_METAL => DeviceOption::Metal,
            WaifuDevice::WAIFU_DEVICE_CUDA => DeviceOption::Cuda,
            WaifuDevice::WAIFU_DEVICE_CUDA_CPU_OFFLOAD => DeviceOption::CudaCpuOffload,
            WaifuDevice::WAIFU_DEVICE_VULKAN => DeviceOption::Vulkan,
        };
        match Engine::new(option) {
            Ok(engine) => Box::into_raw(Box::new(WaifuEngine { engine })),
            Err(error) => {
                set_last_error(&error);
                ptr::null_mut()
            }
        }
    })
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

/// Fetches if needed, then reads onto the device, dropping the model before. What was loaded is
/// waifu_modelmanager_describe_json's to say.
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
        Ok(engine.engine.unload(completion_to(&gate, UserData(user_data), on_complete)))
    })
}

// -- images ----------------------------------------------------------------------------------

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

/// A request struct, read only if it is at least as large as the one this library knows.
unsafe fn request_in<'a, T>(request: *const T, size: u32, what: &str) -> Result<&'a T, String> {
    let request = request.as_ref().ok_or_else(|| format!("the {what} is NULL"))?;
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
                    image: image.as_ref().map_or(ptr::null(), |image| image as *const WaifuImage),
                };
                let user_data = user_data;
                unsafe { callback(user_data.0, &event) };
            })
        };
        engine.engine.draw(request, progress_to(&gate, user_data, on_progress), on_complete)
    })
}

// -- audio -----------------------------------------------------------------------------------

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

fn audio_completion_to(gate: &Gate, user_data: UserData, callback: WaifuAudioCompleteCallback) -> engine::OnComplete<Sound> {
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
            audio: audio.as_ref().map_or(ptr::null(), |audio| audio as *const WaifuAudio),
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
