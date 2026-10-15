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
//! them. Written into waifu_modelmanager.h with the functions, under this library's own names:
//! libwaifu has types of its own, and the two headers share nothing.

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
pub enum WaifuModelManagerStatusCode {
    WAIFU_MODELMANAGER_OK = 0,
    WAIFU_MODELMANAGER_CANCELLED = 1,
    /// A NULL, or a string that is not UTF-8.
    WAIFU_MODELMANAGER_ERR_INVALID_ARGUMENT = 2,
    /// No published model by that name, or no manifest at that path.
    WAIFU_MODELMANAGER_ERR_UNKNOWN_MODEL = 3,
    /// The download failed.
    WAIFU_MODELMANAGER_ERR_FETCH = 4,
    /// A panic in the library, or a file that could not be read or written.
    WAIFU_MODELMANAGER_ERR_INTERNAL = 5,
}

/// How a job ended, as on_complete is told it.
#[repr(C)]
pub struct WaifuModelManagerStatus {
    pub code: WaifuModelManagerStatusCode,
    /// Why, when code is not WAIFU_MODELMANAGER_OK; NULL when it is.
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
pub enum WaifuModelManagerLogLevel {
    WAIFU_MODELMANAGER_LOG_DEBUG = 0,
    WAIFU_MODELMANAGER_LOG_INFO = 1,
    WAIFU_MODELMANAGER_LOG_WARNING = 2,
    WAIFU_MODELMANAGER_LOG_ERROR = 3,
    /// The last line: the process aborts when the callback returns.
    WAIFU_MODELMANAGER_LOG_FATAL = 4,
}

#[repr(C)]
pub struct WaifuModelManagerLogEvent {
    pub level: WaifuModelManagerLogLevel,
    /// Where it was written: a module's name.
    pub source: *const c_char,
    /// The line alone, with no level or time in front of it.
    pub message: *const c_char,
}

/// Called on whichever thread wrote the line, which can be the caller's own. It must be safe to
/// call from any thread, and must not call into the library.
pub type WaifuModelManagerLogCallback = Option<unsafe extern "C" fn(user_data: *mut c_void, event: *const WaifuModelManagerLogEvent)>;

/// A caller's pointer, handed back as it was given, on whichever thread a callback runs.
#[derive(Clone, Copy)]
pub struct UserData(pub *mut c_void);
unsafe impl Send for UserData {}
unsafe impl Sync for UserData {}

/// The sink that calls `callback`, or None -- the console -- where there is no callback.
pub fn log_sink(user_data: *mut c_void, callback: WaifuModelManagerLogCallback) -> Option<Sink> {
    let callback = callback?;
    let user_data = UserData(user_data);
    Some(Box::new(move |level, source, message| {
        let source = c_string(source);
        let message = c_string(message);
        let event = WaifuModelManagerLogEvent {
            level: match level {
                Level::Debug => WaifuModelManagerLogLevel::WAIFU_MODELMANAGER_LOG_DEBUG,
                Level::Info => WaifuModelManagerLogLevel::WAIFU_MODELMANAGER_LOG_INFO,
                Level::Warning => WaifuModelManagerLogLevel::WAIFU_MODELMANAGER_LOG_WARNING,
                Level::Error => WaifuModelManagerLogLevel::WAIFU_MODELMANAGER_LOG_ERROR,
                Level::Fatal => WaifuModelManagerLogLevel::WAIFU_MODELMANAGER_LOG_FATAL,
            },
            source: source.as_ptr(),
            message: message.as_ptr(),
        };
        let user_data = user_data;
        unsafe { callback(user_data.0, &event) };
    }))
}

pub fn log_level(level: WaifuModelManagerLogLevel) -> Level {
    match level {
        WaifuModelManagerLogLevel::WAIFU_MODELMANAGER_LOG_DEBUG => Level::Debug,
        WaifuModelManagerLogLevel::WAIFU_MODELMANAGER_LOG_INFO => Level::Info,
        WaifuModelManagerLogLevel::WAIFU_MODELMANAGER_LOG_WARNING => Level::Warning,
        WaifuModelManagerLogLevel::WAIFU_MODELMANAGER_LOG_ERROR => Level::Error,
        WaifuModelManagerLogLevel::WAIFU_MODELMANAGER_LOG_FATAL => Level::Fatal,
    }
}

// -- progress --------------------------------------------------------------------------------

#[repr(C)]
pub struct WaifuModelManagerProgressEvent {
    /// 0..1 of the file, or -1 where the server did not say how long it is.
    pub fraction: f64,
    /// Bytes of the file; total is 0 where it is not known.
    pub done: u64,
    pub total: u64,
    /// Which file of how many; parts is 0 until the first has named the rest.
    pub part: u32,
    pub parts: u32,
    /// The file, or NULL before the first.
    pub file: *const c_char,
    /// "fetching sdxl:base from Hugging Face -- ... (1 of 3)", for a status line.
    pub words: *const c_char,
    /// Since the job started.
    pub seconds: f64,
    /// Which hub the files come from, once that is settled; NULL before.
    pub hub: *const c_char,
}

pub type WaifuModelManagerProgressCallback = Option<unsafe extern "C" fn(user_data: *mut c_void, event: *const WaifuModelManagerProgressEvent)>;

/// How a fetch ended.
#[repr(C)]
pub struct WaifuModelManagerCompletionEvent {
    pub status: WaifuModelManagerStatus,
}

pub type WaifuModelManagerCompleteCallback = Option<unsafe extern "C" fn(user_data: *mut c_void, event: *const WaifuModelManagerCompletionEvent)>;

/// A progress report, before it is made into C.
pub struct Progress<'a> {
    pub fraction: Option<f64>,
    pub done: u64,
    pub total: u64,
    pub part: u32,
    pub parts: u32,
    pub file: Option<&'a str>,
    pub words: &'a str,
    pub seconds: f64,
    pub hub: Option<&'a str>,
}

/// Calls `callback`, if there is one, with `progress`.
pub fn tell_progress(callback: WaifuModelManagerProgressCallback, user_data: UserData, progress: &Progress) {
    let Some(callback) = callback else {
        return;
    };
    let file = progress.file.map(c_string);
    let words = c_string(progress.words);
    let hub = progress.hub.map(c_string);
    let event = WaifuModelManagerProgressEvent {
        fraction: progress.fraction.unwrap_or(-1.0),
        done: progress.done,
        total: progress.total,
        part: progress.part,
        parts: progress.parts,
        file: file.as_ref().map_or(ptr::null(), |file| file.as_ptr()),
        words: words.as_ptr(),
        seconds: progress.seconds,
        hub: hub.as_ref().map_or(ptr::null(), |hub| hub.as_ptr()),
    };
    unsafe { callback(user_data.0, &event) };
}

/// A status whose message lives as long as `message` does.
pub fn status(code: WaifuModelManagerStatusCode, message: &Option<CString>) -> WaifuModelManagerStatus {
    WaifuModelManagerStatus {
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
