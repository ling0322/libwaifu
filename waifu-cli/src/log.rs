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

//! Where the libraries' log lines go: one sink for both of them, in this process.
//!
//! Each library writes its lines to a callback once it is handed one, and this hands each of them
//! one that puts every line through [`hand_over`] -- the console until a screen sets a sink of its
//! own, as the drawing screen does. A fatal line is the last thing that happens before the process
//! aborts, so the terminal is given back before it is written, wherever it goes.

use std::ffi::c_void;
use std::ptr;
use std::sync::RwLock;

use crate::c::{self, WaifuLogEvent, WaifuLogLevel, WaifuModelManagerLogEvent, WaifuModelManagerLogLevel};

/// How much a line matters.
#[derive(Clone, Copy, Debug, Eq, PartialEq, PartialOrd, Ord)]
pub enum Level {
    Debug,
    Info,
    Warning,
    Error,
    /// The last line: the process aborts once the sink returns.
    Fatal,
}

impl Level {
    pub fn name(self) -> &'static str {
        match self {
            Level::Debug => "DEBUG",
            Level::Info => "INFO",
            Level::Warning => "WARNING",
            Level::Error => "ERROR",
            Level::Fatal => "FATAL",
        }
    }
}

/// What receives a line: its level, where it was written, and the line alone. Called on
/// whichever thread wrote it.
pub type Sink = Box<dyn Fn(Level, &str, &str) + Send + Sync>;

static SINK: RwLock<Option<Sink>> = RwLock::new(None);

/// Sends every line to `sink` from now on, or back to the console for `None`.
pub fn set_sink(sink: Option<Sink>) {
    *SINK.write().unwrap_or_else(|held| held.into_inner()) = sink;
}

/// Sends both libraries' lines here, from now on. Called once, first: the tensor library says
/// what hardware it found the first time a device is asked about.
pub fn listen() {
    c::waifu_set_log_callback(ptr::null_mut(), Some(from_waifu));
    c::waifu_modelmanager_set_log_callback(ptr::null_mut(), Some(from_the_model_manager));
}

/// A line to the sink where there is one, and the console where there is not.
fn hand_over(level: Level, source: &str, message: &str) {
    if level == Level::Fatal {
        crate::tui::give_the_screen_back();
    }
    let sink = SINK.read().unwrap_or_else(|held| held.into_inner());
    match sink.as_ref() {
        Some(sink) => sink(level, source, message),
        None => eprintln!("{} {source}] {message}", level.name()),
    }
}

unsafe extern "C" fn from_waifu(_: *mut c_void, event: *const WaifuLogEvent) {
    let event = &*event;
    let level = match event.level {
        WaifuLogLevel::WAIFU_LOG_DEBUG => Level::Debug,
        WaifuLogLevel::WAIFU_LOG_INFO => Level::Info,
        WaifuLogLevel::WAIFU_LOG_WARNING => Level::Warning,
        WaifuLogLevel::WAIFU_LOG_ERROR => Level::Error,
        WaifuLogLevel::WAIFU_LOG_FATAL => Level::Fatal,
    };
    let source = c::text(event.source).unwrap_or_default();
    hand_over(level, &source, &c::text(event.message).unwrap_or_default());
}

unsafe extern "C" fn from_the_model_manager(_: *mut c_void, event: *const WaifuModelManagerLogEvent) {
    let event = &*event;
    let level = match event.level {
        WaifuModelManagerLogLevel::WAIFU_MODELMANAGER_LOG_DEBUG => Level::Debug,
        WaifuModelManagerLogLevel::WAIFU_MODELMANAGER_LOG_INFO => Level::Info,
        WaifuModelManagerLogLevel::WAIFU_MODELMANAGER_LOG_WARNING => Level::Warning,
        WaifuModelManagerLogLevel::WAIFU_MODELMANAGER_LOG_ERROR => Level::Error,
        WaifuModelManagerLogLevel::WAIFU_MODELMANAGER_LOG_FATAL => Level::Fatal,
    };
    let source = c::text(event.source).unwrap_or_default();
    hand_over(level, &source, &c::text(event.message).unwrap_or_default());
}
