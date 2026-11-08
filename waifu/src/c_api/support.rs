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

//! What the C API is made of besides its functions: the types every part of it takes and hands
//! back -- status, the log, progress and results -- and the helpers that convert to and from
//! them. Written into waifu.h with the functions.
//!
//! Every enum of the header is an `int32_t` and a `#define` for each value, not a C enum: a number
//! a caller hands in that this library does not know is an integer to refuse, not a value Rust
//! must never see.

#![allow(non_camel_case_types)]
#![allow(clippy::missing_safety_doc)]

use std::cell::RefCell;
use std::ffi::{c_char, c_void, CStr, CString};
use std::panic::{self, AssertUnwindSafe};
use std::ptr;
use std::sync::{Arc, Condvar, Mutex};

use crate::log::{Level, Sink};

// -- status ----------------------------------------------------------------------------------

/// What a refused call or a job came to.
pub type WaifuStatusCode = i32;
pub const WAIFU_OK: WaifuStatusCode = 0;
pub const WAIFU_CANCELLED: WaifuStatusCode = 1;
/// What a request cannot be run with. Most of it is refused at the call instead -- a NULL, a
/// string that is not UTF-8, a size out of range, a model the pipeline's task is not supported by,
/// another task's run -- which a job never sees.
pub const WAIFU_ERR_INVALID_ARGUMENT: WaifuStatusCode = 2;
/// No manifest at the path.
pub const WAIFU_ERR_UNKNOWN_MODEL: WaifuStatusCode = 3;
// 4 was FETCH, which libwaifu never said once downloads were libwaifu_modelmanager's: not reused.
/// The model could not be read, or cannot do what was asked.
pub const WAIFU_ERR_MODEL: WaifuStatusCode = 5;
/// A run on a pipeline with no model loaded.
pub const WAIFU_ERR_NOT_LOADED: WaifuStatusCode = 6;
/// A panic in the library: a bug.
pub const WAIFU_ERR_INTERNAL: WaifuStatusCode = 7;
/// A pipeline asked for on a device this build or machine cannot use.
pub const WAIFU_ERR_DEVICE_UNAVAILABLE: WaifuStatusCode = 8;

#[repr(C)]
pub struct WaifuStatus {
    pub code: WaifuStatusCode,
    /// Why, or NULL for WAIFU_OK.
    pub message: *const c_char,
}

/// Why a call was refused: what `waifu_last_error` and `waifu_last_error_code` are set to.
#[derive(Debug)]
pub struct Refused {
    pub code: WaifuStatusCode,
    pub message: String,
}

impl Refused {
    pub fn new(code: WaifuStatusCode, message: impl Into<String>) -> Refused {
        Refused {
            code,
            message: message.into(),
        }
    }
}

/// A sentence alone is a request that cannot be run: most of what is refused.
impl From<String> for Refused {
    fn from(message: String) -> Refused {
        Refused::new(WAIFU_ERR_INVALID_ARGUMENT, message)
    }
}

impl From<&str> for Refused {
    fn from(message: &str) -> Refused {
        Refused::new(WAIFU_ERR_INVALID_ARGUMENT, message)
    }
}

thread_local! {
    static LAST_ERROR: RefCell<(CString, WaifuStatusCode)> =
        RefCell::new((CString::default(), WAIFU_OK));
}

/// Says why this thread's last refused call was refused. Only a refused call sets it.
pub fn refuse(refused: &Refused) {
    let message = c_string(&refused.message);
    LAST_ERROR.with(|last| *last.borrow_mut() = (message, refused.code));
}

/// This thread's last refusal; valid until its next one.
pub fn last_error() -> *const c_char {
    LAST_ERROR.with(|last| last.borrow().0.as_ptr())
}

pub fn last_error_code() -> WaifuStatusCode {
    LAST_ERROR.with(|last| last.borrow().1)
}

/// What a panic said.
fn said(cause: Box<dyn std::any::Any + Send>) -> String {
    cause
        .downcast_ref::<&str>()
        .map(|said| said.to_string())
        .or_else(|| cause.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "it panicked".to_string())
}

/// Runs an entry point that is never refused, with `otherwise` as what it returns where it panics.
pub fn caught<T>(otherwise: T, body: impl FnOnce() -> T) -> T {
    panic::catch_unwind(AssertUnwindSafe(body)).unwrap_or(otherwise)
}

/// Runs an entry point that can be refused: its value, or `otherwise` with the refusal -- a panic
/// as WAIFU_ERR_INTERNAL -- set as this thread's last error.
pub fn refusable<T>(otherwise: T, body: impl FnOnce() -> Result<T, Refused>) -> T {
    let refused = match panic::catch_unwind(AssertUnwindSafe(body)) {
        Ok(Ok(value)) => return value,
        Ok(Err(refused)) => refused,
        Err(cause) => Refused::new(
            WAIFU_ERR_INTERNAL,
            format!("a bug in the library: {}", said(cause)),
        ),
    };
    refuse(&refused);
    otherwise
}

/// A string handed in, or the reason it cannot be read. NULL is the caller's to decide.
pub unsafe fn string_in(pointer: *const c_char, what: &str) -> Result<Option<String>, String> {
    if pointer.is_null() {
        return Ok(None);
    }
    CStr::from_ptr(pointer)
        .to_str()
        .map(|text| Some(text.to_string()))
        .map_err(|_| format!("{what} is not UTF-8"))
}

/// A string handed in that may not be NULL.
pub unsafe fn required(pointer: *const c_char, what: &str) -> Result<String, String> {
    string_in(pointer, what)?.ok_or_else(|| format!("{what} is NULL"))
}

/// `text` as C can read it: a NUL inside it would end it early, so it is a space instead.
pub fn c_string(text: &str) -> CString {
    CString::new(text.replace('\0', " ")).unwrap_or_default()
}

// -- logging ---------------------------------------------------------------------------------

pub type WaifuLogLevel = i32;
pub const WAIFU_LOG_DEBUG: WaifuLogLevel = 0;
pub const WAIFU_LOG_INFO: WaifuLogLevel = 1;
pub const WAIFU_LOG_WARNING: WaifuLogLevel = 2;
pub const WAIFU_LOG_ERROR: WaifuLogLevel = 3;
/// The process aborts when the callback returns.
pub const WAIFU_LOG_FATAL: WaifuLogLevel = 4;

#[repr(C)]
pub struct WaifuLog {
    pub level: WaifuLogLevel,
    /// Where it was written, as a file and a line: "interface.cc:84", "engine.rs:712".
    pub source: *const c_char,
    /// The line alone, with no level or time in front of it.
    pub message: *const c_char,
    /// When it was written, in UTC: "2026-10-23T18:50:41Z".
    pub time: *const c_char,
}

/// Called on whichever thread wrote the line, which can be the caller's own, possibly from several
/// threads at once. Must not call into the library.
pub type WaifuLogCallback =
    Option<unsafe extern "C" fn(user_data: *mut c_void, log: *const WaifuLog)>;

/// A caller's pointer, handed back as it was given, on whichever thread a callback runs.
#[derive(Clone, Copy)]
pub struct UserData(pub *mut c_void);
unsafe impl Send for UserData {}
unsafe impl Sync for UserData {}

/// The sink that calls `callback`, or None -- the console -- where there is no callback.
pub fn log_sink(user_data: *mut c_void, callback: WaifuLogCallback) -> Option<Sink> {
    let callback = callback?;
    let user_data = UserData(user_data);
    Some(Box::new(move |level, time, source, message| {
        let time = c_string(time);
        let source = c_string(source);
        let message = c_string(message);
        let log = WaifuLog {
            level: match level {
                Level::Debug => WAIFU_LOG_DEBUG,
                Level::Info => WAIFU_LOG_INFO,
                Level::Warning => WAIFU_LOG_WARNING,
                Level::Error => WAIFU_LOG_ERROR,
                Level::Fatal => WAIFU_LOG_FATAL,
            },
            source: source.as_ptr(),
            message: message.as_ptr(),
            time: time.as_ptr(),
        };
        let user_data = user_data;
        unsafe { callback(user_data.0, &log) };
    }))
}

/// The level a caller named, or None for a number this library does not know.
pub fn log_level(level: WaifuLogLevel) -> Option<Level> {
    match level {
        WAIFU_LOG_DEBUG => Some(Level::Debug),
        WAIFU_LOG_INFO => Some(Level::Info),
        WAIFU_LOG_WARNING => Some(Level::Warning),
        WAIFU_LOG_ERROR => Some(Level::Error),
        WAIFU_LOG_FATAL => Some(Level::Fatal),
        _ => None,
    }
}

// -- progress --------------------------------------------------------------------------------

/// What a job is doing. Ranges by task; done/total count what the comment says, 0, 0 otherwise.
pub type WaifuStage = i32;
pub const WAIFU_STAGE_LOADING_WEIGHTS: WaifuStage = 1;

pub const WAIFU_STAGE_IMAGE_GENERATION_ENCODING_PROMPT: WaifuStage = 100;
/// Steps.
pub const WAIFU_STAGE_IMAGE_GENERATION_DENOISING: WaifuStage = 101;
pub const WAIFU_STAGE_IMAGE_GENERATION_DECODING_IMAGE: WaifuStage = 102;

pub const WAIFU_STAGE_TEXT_TO_SPEECH_READING_TEXT: WaifuStage = 200;
/// Tokens; total is an estimate, and can be passed.
pub const WAIFU_STAGE_TEXT_TO_SPEECH_GENERATING_TOKENS: WaifuStage = 201;
pub const WAIFU_STAGE_TEXT_TO_SPEECH_SYNTHESIZING_AUDIO: WaifuStage = 202;

pub const WAIFU_STAGE_VOICE_CONVERSION_READING_RECORDINGS: WaifuStage = 300;
/// Tokens; total is an estimate.
pub const WAIFU_STAGE_VOICE_CONVERSION_GENERATING_TOKENS: WaifuStage = 301;
/// Steps.
pub const WAIFU_STAGE_VOICE_CONVERSION_DENOISING: WaifuStage = 302;
/// Vocoder windows.
pub const WAIFU_STAGE_VOICE_CONVERSION_SYNTHESIZING_AUDIO: WaifuStage = 303;

/// The whole picture, then the head, then the depth model that orders the layers.
pub const WAIFU_STAGE_LAYER_DECOMPOSITION_ENCODING_IMAGE: WaifuStage = 400;
/// Steps.
pub const WAIFU_STAGE_LAYER_DECOMPOSITION_DENOISING_LAYERS: WaifuStage = 401;
/// Layers.
pub const WAIFU_STAGE_LAYER_DECOMPOSITION_DECODING_LAYERS: WaifuStage = 402;
pub const WAIFU_STAGE_LAYER_DECOMPOSITION_ENCODING_HEAD: WaifuStage = 403;
/// Steps.
pub const WAIFU_STAGE_LAYER_DECOMPOSITION_DENOISING_HEAD_LAYERS: WaifuStage = 404;
/// Layers.
pub const WAIFU_STAGE_LAYER_DECOMPOSITION_DECODING_HEAD_LAYERS: WaifuStage = 405;
/// Layers.
pub const WAIFU_STAGE_LAYER_DECOMPOSITION_ENCODING_FOR_DEPTH: WaifuStage = 406;
/// Steps.
pub const WAIFU_STAGE_LAYER_DECOMPOSITION_DENOISING_DEPTH: WaifuStage = 407;
/// Layers.
pub const WAIFU_STAGE_LAYER_DECOMPOSITION_DECODING_DEPTH: WaifuStage = 408;
pub const WAIFU_STAGE_LAYER_DECOMPOSITION_CUTTING_INTO_PARTS: WaifuStage = 409;

/// Pieces the recording is cut into.
pub const WAIFU_STAGE_SPEECH_TO_TEXT_READING_RECORDING: WaifuStage = 500;
/// Tokens; total is an estimate, and can be passed.
pub const WAIFU_STAGE_SPEECH_TO_TEXT_GENERATING_TOKENS: WaifuStage = 501;

/// Saying and sounding take turns, a hop of tokens at a time.
pub const WAIFU_STAGE_STREAMING_TEXT_TO_SPEECH_READING_TEXT: WaifuStage = 600;
/// Tokens; total is an estimate.
pub const WAIFU_STAGE_STREAMING_TEXT_TO_SPEECH_GENERATING_TOKENS: WaifuStage = 601;
/// Parts handed over.
pub const WAIFU_STAGE_STREAMING_TEXT_TO_SPEECH_SYNTHESIZING_AUDIO: WaifuStage = 602;

#[repr(C)]
pub struct WaifuProgress {
    pub stage: WaifuStage,
    /// 0..1 of the whole job, or -1 where nothing can say.
    pub fraction: f64,
    /// Of the stage, done already.
    pub done: u64,
    pub total: u64,
    /// Since the job started.
    pub seconds: f64,
}

pub type WaifuProgressCallback =
    Option<unsafe extern "C" fn(user_data: *mut c_void, progress: *const WaifuProgress)>;

/// Every pointer in a result is valid until the callback returns.
#[repr(C)]
pub struct WaifuResult {
    pub status: WaifuStatus,
}

pub type WaifuCallback =
    Option<unsafe extern "C" fn(user_data: *mut c_void, result: *const WaifuResult)>;

/// A status whose message lives as long as `message` does.
pub fn status(code: WaifuStatusCode, message: &Option<CString>) -> WaifuStatus {
    WaifuStatus {
        code,
        message: message
            .as_ref()
            .map_or(ptr::null(), |message| message.as_ptr()),
    }
}

// -- keeping the promises about when a callback can arrive ---------------------------------------

/// Shut until the `_async` call that made it has its job's id, which is when it opens: a callback
/// that arrives before then waits, so that no callback runs for a call that may yet be refused.
/// It does not hold callbacks back until the call has returned -- nothing can know when the caller
/// has the id -- which is why a job's state is its `user_data` (docs/ffi-abi4.md).
#[derive(Clone, Default)]
pub struct Gate(Arc<(Mutex<bool>, Condvar)>);

impl Gate {
    pub fn open(&self) {
        let (open, opened) = &*self.0;
        *open.lock().unwrap_or_else(|held| held.into_inner()) = true;
        opened.notify_all();
    }

    pub fn wait(&self) {
        let (open, opened) = &*self.0;
        let mut is_open = open.lock().unwrap_or_else(|held| held.into_inner());
        while !*is_open {
            is_open = opened
                .wait(is_open)
                .unwrap_or_else(|held| held.into_inner());
        }
    }
}
