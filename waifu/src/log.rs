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

//! Where the library's log lines go: one place for the Rust side's and the tensor library's.
//!
//! On the console unless something says otherwise, which is what the command line wants. An app
//! has no console its user would ever read, so it hands over a sink, and from then on every line
//! -- what hardware flint found, a download that is taking another route, the line a fatal error
//! ends the process with -- reaches it instead.

use std::ffi::CStr;
use std::os::raw::c_char;
use std::sync::atomic::{AtomicI32, Ordering};
use std::sync::RwLock;

use crate::flint;

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
    fn from_flint(level: i32) -> Level {
        match level {
            i32::MIN..=0 => Level::Debug,
            1 => Level::Info,
            2 => Level::Warning,
            3 => Level::Error,
            _ => Level::Fatal,
        }
    }

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

/// What receives a line: its level, where it was written -- "interface.cc:84" for the tensor
/// library, a module's name for this side -- and the line alone.
///
/// Called on whichever thread wrote the line, so it has to be safe to call from any of them, and
/// it must not set another sink from inside itself, which would wait on the lock it is called
/// under.
pub type Sink = Box<dyn Fn(Level, &str, &str) + Send + Sync>;

static SINK: RwLock<Option<Sink>> = RwLock::new(None);
static LEVEL: AtomicI32 = AtomicI32::new(Level::Info as i32);

/// Sends every line to `sink` from now on, or back to the console for `None`.
///
/// Best set before anything else is asked of the library: the tensor library writes what
/// hardware it found the first time a device is asked about, and a sink set after that has
/// already missed it.
pub fn set_sink(sink: Option<Sink>) {
    let has_one = sink.is_some();
    *SINK.write().unwrap_or_else(|held| held.into_inner()) = sink;
    flint::set_log_sink(match has_one {
        true => Some(from_flint),
        false => None,
    });
}

/// Writes no line below `level`, on either side. Info unless set.
pub fn set_level(level: Level) {
    LEVEL.store(level as i32, Ordering::Relaxed);
    flint::set_log_level(level as i32);
}

/// Writes one line from this side of the library.
pub fn write(level: Level, source: &str, message: &str) {
    if (level as i32) < LEVEL.load(Ordering::Relaxed) {
        return;
    }
    hand_over(level, source, message);
}

/// The line to the sink where there is one, and the console where there is not.
fn hand_over(level: Level, source: &str, message: &str) {
    let sink = SINK.read().unwrap_or_else(|held| held.into_inner());
    match sink.as_ref() {
        Some(sink) => sink(level, source, message),
        None => eprintln!("{} {source}] {message}", level.name()),
    }
}

/// What the tensor library calls with a line, once a sink is set. Its own level has already been
/// checked by the time this is reached.
extern "C" fn from_flint(level: i32, source: *const c_char, message: *const c_char) {
    let text = |pointer: *const c_char| match pointer.is_null() {
        true => String::new(),
        false => unsafe { CStr::from_ptr(pointer) }.to_string_lossy().into_owned(),
    };
    hand_over(Level::from_flint(level), &text(source), &text(message));
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::sync::{Arc, Mutex};

    /// One test rather than several: the sink is the process's, and two tests setting it at once
    /// would each see the other's lines.
    #[test]
    fn every_line_from_either_side_reaches_the_sink_and_none_below_the_level() {
        let heard: Arc<Mutex<Vec<(Level, String, String)>>> = Arc::default();
        let keep = Arc::clone(&heard);
        set_sink(Some(Box::new(move |level, source, message| {
            keep.lock()
                .unwrap()
                .push((level, source.to_string(), message.to_string()))
        })));
        set_level(Level::Info);

        write(Level::Warning, "hub", "taking ModelScope instead");
        write(Level::Debug, "hub", "not this");
        // A line as the tensor library hands one over, through its C API.
        from_flint(1, c"interface.cc:84".as_ptr(), c"Use asimdfhm backend.".as_ptr());

        set_sink(None);
        let heard = heard.lock().unwrap();
        assert_eq!(heard.len(), 2, "{heard:?}");
        assert_eq!(
            heard[0],
            (Level::Warning, "hub".to_string(), "taking ModelScope instead".to_string())
        );
        assert_eq!(heard[1].0, Level::Info);
        assert_eq!(heard[1].1, "interface.cc:84");
        assert_eq!(heard[1].2, "Use asimdfhm backend.");
    }
}
