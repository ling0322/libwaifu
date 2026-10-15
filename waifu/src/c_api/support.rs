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
//! back -- status, the log, progress and completion -- and the helpers that convert to and from
//! them. Written into waifu.h with the functions.

#![allow(non_camel_case_types)]
#![allow(clippy::missing_safety_doc)]

use std::cell::RefCell;
use std::ffi::{c_char, c_void, CStr, CString};
use std::panic::{self, AssertUnwindSafe};
use std::ptr;
use std::sync::{Arc, Condvar, Mutex};

use crate::log::{Level, Sink};

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

/// Says why this thread's last call failed, for its `_last_error` to hand back.
pub fn set_last_error(message: &str) {
    let message = c_string(message);
    LAST_ERROR.with(|last| *last.borrow_mut() = message);
}

/// This thread's last message; valid until this thread's next call.
pub fn last_error() -> *const c_char {
    LAST_ERROR.with(|last| last.borrow().as_ptr())
}

/// Frees a string [`string_out`] made. NULL is fine.
pub unsafe fn free_string(s: *mut c_char) {
    if !s.is_null() {
        drop(CString::from_raw(s));
    }
}

/// Runs a synchronous entry point, turning a panic into its message in the last error and
/// `otherwise` as what the call returns.
pub fn caught<T>(otherwise: T, body: impl FnOnce() -> T) -> T {
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

/// A string for the caller to free with the library's `_string_free`.
pub fn string_out(text: &str) -> *mut c_char {
    c_string(text).into_raw()
}

/// `text` as C can read it: a NUL inside it would end it early, so it is a space instead.
pub fn c_string(text: &str) -> CString {
    CString::new(text.replace('\0', " ")).unwrap_or_default()
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
pub struct UserData(pub *mut c_void);
unsafe impl Send for UserData {}
unsafe impl Sync for UserData {}

/// The sink that calls `callback`, or None -- the console -- where there is no callback.
pub fn log_sink(user_data: *mut c_void, callback: WaifuLogCallback) -> Option<Sink> {
    let callback = callback?;
    let user_data = UserData(user_data);
    Some(Box::new(move |level, source, message| {
        let source = c_string(source);
        let message = c_string(message);
        let event = WaifuLogEvent {
            level: match level {
                Level::Debug => WaifuLogLevel::WAIFU_LOG_DEBUG,
                Level::Info => WaifuLogLevel::WAIFU_LOG_INFO,
                Level::Warning => WaifuLogLevel::WAIFU_LOG_WARNING,
                Level::Error => WaifuLogLevel::WAIFU_LOG_ERROR,
                Level::Fatal => WaifuLogLevel::WAIFU_LOG_FATAL,
            },
            source: source.as_ptr(),
            message: message.as_ptr(),
        };
        let user_data = user_data;
        unsafe { callback(user_data.0, &event) };
    }))
}

pub fn log_level(level: WaifuLogLevel) -> Level {
    match level {
        WaifuLogLevel::WAIFU_LOG_DEBUG => Level::Debug,
        WaifuLogLevel::WAIFU_LOG_INFO => Level::Info,
        WaifuLogLevel::WAIFU_LOG_WARNING => Level::Warning,
        WaifuLogLevel::WAIFU_LOG_ERROR => Level::Error,
        WaifuLogLevel::WAIFU_LOG_FATAL => Level::Fatal,
    }
}

// -- progress --------------------------------------------------------------------------------

#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WaifuStage {
    /// Reading weights onto the device: no fraction. 0 was fetching, which is
    /// libwaifu_modelmanager's now; the others keep their numbers.
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
    /// Reserved: 0, 0 and NULL. They said which file of a download, which is
    /// libwaifu_modelmanager's now, and are kept so the event keeps its layout.
    pub part: u32,
    pub parts: u32,
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

/// A progress report, before it is made into C.
pub struct Progress<'a> {
    pub stage: WaifuStage,
    pub fraction: Option<f64>,
    pub done: u64,
    pub total: u64,
    pub part: u32,
    pub parts: u32,
    pub file: Option<&'a str>,
    pub words: &'a str,
    pub seconds: f64,
}

/// Calls `callback`, if there is one, with `progress`.
pub fn tell_progress(callback: WaifuProgressCallback, user_data: UserData, progress: &Progress) {
    let Some(callback) = callback else {
        return;
    };
    let file = progress.file.map(c_string);
    let words = c_string(progress.words);
    let event = WaifuProgressEvent {
        stage: progress.stage,
        fraction: progress.fraction.unwrap_or(-1.0),
        done: progress.done,
        total: progress.total,
        part: progress.part,
        parts: progress.parts,
        file: file.as_ref().map_or(ptr::null(), |file| file.as_ptr()),
        words: words.as_ptr(),
        seconds: progress.seconds,
    };
    unsafe { callback(user_data.0, &event) };
}

/// A status whose message lives as long as `message` does.
pub fn status(code: WaifuStatusCode, message: &Option<CString>) -> WaifuStatus {
    WaifuStatus {
        code,
        message: message.as_ref().map_or(ptr::null(), |message| message.as_ptr()),
    }
}

// -- keeping the promises about when a callback can arrive ---------------------------------------

/// Shut until the `_async` call that made it has returned: a callback that arrives before then
/// waits. A job's thread can finish before the call that started it is done handing back its
/// handle, and a caller whose continuation is set up from that handle would miss the end.
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
            is_open = opened.wait(is_open).unwrap_or_else(|held| held.into_inner());
        }
    }
}
