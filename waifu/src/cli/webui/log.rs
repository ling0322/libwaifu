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

//! What the server says in the terminal while it runs: the requests it answers, and what the
//! worker does with them.
//!
//! One line each, to stderr, stamped the way the tensor library stamps its own lines -- UTC, to
//! the second -- so that the two read as one log when they are interleaved, which they are
//! whenever a model is being read.

use std::fmt::Display;
use std::io::{self, IsTerminal};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// The routes the page asks several times a second. A line for each would bury everything else,
/// so they are logged only when they fail.
const POLLED: [&str; 3] = ["/api/state", "/api/progress", "/api/machine"];

/// Writes one line.
///
/// On a terminal the line it lands on is cleared first. While a model is being read the terminal
/// has a progress line on it that rewrites itself with `\r`, and a log line that started in the
/// middle of it would be glued onto the end of it.
pub fn line(what: impl Display) {
    let clear = match io::stderr().is_terminal() {
        true => "\r\x1b[2K",
        false => "",
    };
    eprintln!("{clear}{} {what}", stamp(SystemTime::now()));
}

/// Logs a request that was answered, unless it is one of the page's polls and went well.
pub fn request(method: &str, path: &str, status: u16, took: Duration) {
    if worth_logging(method, path, status) {
        line(format_args!(
            "{method} {path} {status} {}ms",
            took.as_millis()
        ));
    }
}

/// Whether a request is worth a line: anything that failed, and anything that is not a poll.
fn worth_logging(method: &str, path: &str, status: u16) -> bool {
    status >= 400 || method != "GET" || !POLLED.contains(&path)
}

/// Text cut to `most` characters, with an ellipsis where it was cut: a prompt is a paragraph and
/// a log line is a line.
pub fn clipped(text: &str, most: usize) -> String {
    let flat = text.split_whitespace().collect::<Vec<_>>().join(" ");
    match flat.chars().count() > most {
        true => format!("{}...", flat.chars().take(most).collect::<String>()),
        false => flat,
    }
}

/// A moment as `2026-09-25T23:21:14Z`.
fn stamp(at: SystemTime) -> String {
    let seconds = at
        .duration_since(UNIX_EPOCH)
        .map(|since| since.as_secs())
        .unwrap_or(0);
    let (days, rest) = (seconds / 86_400, seconds % 86_400);
    let (year, month, day) = civil(days as i64);

    format!(
        "{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}Z",
        rest / 3600,
        rest % 3600 / 60,
        rest % 60
    )
}

/// The calendar date `days` after 1970-01-01, by Howard Hinnant's `civil_from_days`. Written out
/// rather than pulled in: it is the one date this program ever formats.
fn civil(days: i64) -> (i64, u32, u32) {
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let month = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    let year = yoe + era * 400 + i64::from(month <= 2);

    (year, month, day)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_moment_is_stamped_the_way_the_tensor_library_stamps_one() {
        let at = UNIX_EPOCH + Duration::from_secs(1_790_378_474);
        assert_eq!(stamp(at), "2026-09-25T23:21:14Z");
        assert_eq!(stamp(UNIX_EPOCH), "1970-01-01T00:00:00Z");
        // The day after a leap day, which is where a hand-written calendar goes wrong first.
        let at = UNIX_EPOCH + Duration::from_secs(1_709_251_200);
        assert_eq!(stamp(at), "2024-03-01T00:00:00Z");
    }

    #[test]
    fn the_pages_polls_are_quiet_unless_they_fail() {
        for polled in POLLED {
            assert!(!worth_logging("GET", polled, 200), "{polled}");
            assert!(worth_logging("GET", polled, 500), "{polled}");
        }
        assert!(worth_logging("POST", "/api/generate", 200));
        assert!(worth_logging("GET", "/", 200));
        assert!(worth_logging("GET", "/picture/waifu-0001.png", 200));
    }

    #[test]
    fn a_long_prompt_is_cut_to_a_line() {
        assert_eq!(clipped("a cat", 10), "a cat");
        assert_eq!(clipped("a cat\n  on a mat", 20), "a cat on a mat");
        assert_eq!(clipped("abcdefghij", 4), "abcd...");
    }
}
