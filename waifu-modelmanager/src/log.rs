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

//! Where the log lines go: the console, unless something says otherwise.
//!
//! An app has no console its user would ever read, so it hands over a sink, and from then on every
//! line -- a download that is taking another route, say -- reaches it instead. `waifu::log` is
//! built over this one and sends the tensor library's lines here too, so that in one image there
//! is one sink and one level.

use std::sync::atomic::{AtomicI32, Ordering};
use std::sync::RwLock;

/// How much a line matters. In the tensor library's order, and with its numbers, so that one
/// level set here is the same level there.
#[derive(Clone, Copy, Debug, Eq, PartialEq, PartialOrd, Ord)]
#[repr(i32)]
pub enum Level {
    Debug = 0,
    Info = 1,
    Warning = 2,
    Error = 3,
    /// The last line: the process aborts once the sink returns.
    Fatal = 4,
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

/// What receives a line: its level, where it was written -- a module's name, or a file and line
/// of the tensor library's -- and the line alone.
///
/// Called on whichever thread wrote the line, so it has to be safe to call from any of them, and
/// it must not set another sink from inside itself, which would wait on the lock it is called
/// under.
pub type Sink = Box<dyn Fn(Level, &str, &str) + Send + Sync>;

static SINK: RwLock<Option<Sink>> = RwLock::new(None);
static LEVEL: AtomicI32 = AtomicI32::new(Level::Info as i32);

/// Sends every line to `sink` from now on, or back to the console for `None`.
pub fn set_sink(sink: Option<Sink>) {
    *SINK.write().unwrap_or_else(|held| held.into_inner()) = sink;
}

/// Writes no line below `level`. Info unless set.
pub fn set_level(level: Level) {
    LEVEL.store(level as i32, Ordering::Relaxed);
}

/// Writes one line.
pub fn write(level: Level, source: &str, message: &str) {
    if (level as i32) < LEVEL.load(Ordering::Relaxed) {
        return;
    }
    hand_over(level, source, message);
}

/// The line to the sink where there is one, and the console where there is not, without asking
/// the level: for a line whose level was checked already, where it was written.
pub fn hand_over(level: Level, source: &str, message: &str) {
    let sink = SINK.read().unwrap_or_else(|held| held.into_inner());
    match sink.as_ref() {
        Some(sink) => sink(level, source, message),
        None => eprintln!("{} {source}] {message}", level.name()),
    }
}
