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

//! The library's log: every line it would print, to a closure of the caller's instead.

use std::ffi::{c_char, c_void, CStr};
use std::sync::Mutex;

use crate::{sys, Result};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum LogLevel {
    Debug,
    Info,
    Warning,
    Error,
    /// The process aborts when the callback returns.
    Fatal,
    /// A level a newer library has and this crate does not know.
    Other(i32),
}

impl LogLevel {
    fn from_raw(level: sys::WaifuLogLevel) -> LogLevel {
        match level {
            sys::WAIFU_LOG_DEBUG => LogLevel::Debug,
            sys::WAIFU_LOG_INFO => LogLevel::Info,
            sys::WAIFU_LOG_WARNING => LogLevel::Warning,
            sys::WAIFU_LOG_ERROR => LogLevel::Error,
            sys::WAIFU_LOG_FATAL => LogLevel::Fatal,
            other => LogLevel::Other(other),
        }
    }

    fn raw(self) -> sys::WaifuLogLevel {
        match self {
            LogLevel::Debug => sys::WAIFU_LOG_DEBUG,
            LogLevel::Info => sys::WAIFU_LOG_INFO,
            LogLevel::Warning => sys::WAIFU_LOG_WARNING,
            LogLevel::Error => sys::WAIFU_LOG_ERROR,
            LogLevel::Fatal => sys::WAIFU_LOG_FATAL,
            LogLevel::Other(other) => other,
        }
    }
}

/// One line, its strings borrowed for the length of the callback.
#[derive(Clone, Copy, Debug)]
pub struct LogEvent<'a> {
    pub level: LogLevel,
    /// Where it was written: "engine.rs:712".
    pub source: &'a str,
    pub message: &'a str,
    /// When, in UTC: "2026-10-23T18:50:41Z".
    pub time: &'a str,
}

type Callback = Box<dyn Fn(&LogEvent) + Send + Sync>;

/// The callback the library holds now, kept alive here until another replaces it.
static CURRENT: Mutex<Option<Box<Callback>>> = Mutex::new(None);

unsafe fn borrowed<'a>(pointer: *const c_char) -> &'a str {
    match pointer.is_null() {
        true => "",
        false => CStr::from_ptr(pointer).to_str().unwrap_or(""),
    }
}

unsafe extern "C" fn trampoline(user_data: *mut c_void, log: *const sys::WaifuLog) {
    let (callback, log) = (&*(user_data as *const Callback), &*log);
    let event = LogEvent {
        level: LogLevel::from_raw(log.level),
        source: borrowed(log.source),
        message: borrowed(log.message),
        time: borrowed(log.time),
    };
    // A panic must not unwind into the library.
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| callback(&event)));
}

/// Hands every line to `f`, on whichever thread wrote it, possibly several at once. `f` must not
/// call into this crate.
pub fn set_callback(f: impl Fn(&LogEvent) + Send + Sync + 'static) -> Result<()> {
    let functions = sys::functions()?;
    let mut current = CURRENT.lock().unwrap_or_else(|held| held.into_inner());
    let callback: Box<Callback> = Box::new(Box::new(f));
    let user_data = &*callback as *const Callback as *mut c_void;
    // Once this returns the library calls the one before never again, and it can go.
    unsafe { (functions.waifu_set_log_callback)(user_data, Some(trampoline)) };
    *current = Some(callback);
    Ok(())
}

/// Lines go to the console again.
pub fn clear_callback() -> Result<()> {
    let functions = sys::functions()?;
    let mut current = CURRENT.lock().unwrap_or_else(|held| held.into_inner());
    unsafe { (functions.waifu_set_log_callback)(std::ptr::null_mut(), None) };
    *current = None;
    Ok(())
}

/// Lines below `level` are not made at all. Info unless set.
pub fn set_level(level: LogLevel) -> Result<()> {
    let functions = sys::functions()?;
    unsafe { (functions.waifu_set_log_level)(level.raw()) };
    Ok(())
}
