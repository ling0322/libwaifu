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

//! The libwaifu command line tool: draw a picture with a model.
//!
//! Behind the `cli` feature, because it is the only thing in this crate that needs a terminal
//! library. The `waifu` binary is a shim over [`run`]; everything else here is its
//! implementation.

mod args;
mod ask;
mod draw;
mod field;
mod files;
mod hub;
mod picker;

use std::process::ExitCode;

use ratatui::layout::Rect;
use ratatui::style::{Color, Style, Stylize};
use ratatui::text::{Line, Span};
use ratatui::Frame;

/// The commit this was built from, stamped in by build.rs. "unknown" when it was built from a
/// copy of the source with no git around it, which is a thing that has to keep working.
const REVISION: &str = env!("WAIFU_REVISION");

/// What built this, as every screen says it.
///
/// On both of them rather than one: whichever is on screen when something goes wrong is the one
/// that ends up in a screenshot, and a screenshot that cannot say which code it came from is
/// worth much less than one that can.
fn built_from() -> Line<'static> {
    Line::from(vec![
        Span::raw(" libwaifu").bold(),
        Span::raw("  "),
        Span::styled(REVISION, Style::new().fg(Color::DarkGray)),
    ])
}

/// A rectangle of at most `width` by `height`, in the middle of `area`.
///
/// Where every box that opens on top of a screen puts itself. Shared rather than written twice
/// because two of them already want it and the third will.
fn centred(area: Rect, width: u16, height: u16) -> Rect {
    let width = width.min(area.width);
    let height = height.min(area.height);

    Rect {
        x: area.x + (area.width - width) / 2,
        y: area.y + (area.height - height) / 2,
        width,
        height,
    }
}

/// A progress bar: a run of coloured cells, with its label centred on it.
///
/// Written out rather than left to ratatui's `Gauge`, which paints the filled part with `\u{2588}`
/// glyphs and the label's own patch with a background colour instead. The two agree on the grid
/// and not on the screen: in macOS Terminal the block glyph does not quite fill its cell, so the
/// bar comes out ribbed and steps up where the label sits on it. Colouring the background of
/// every cell the same way leaves a font nothing to disagree about.
///
/// `area` is the inside of whatever box the bar goes in; the caller draws the box.
fn bar(frame: &mut Frame, area: Rect, ratio: f64, label: &str, colour: Color) {
    if area.is_empty() {
        return;
    }

    let filled = (ratio.clamp(0.0, 1.0) * f64::from(area.width)).round() as u16;
    let text: Vec<char> = label.chars().take(area.width as usize).collect();
    let start = area.x + (area.width - text.len() as u16) / 2;
    let row = area.y + area.height / 2;

    let buffer = frame.buffer_mut();
    for y in area.top()..area.bottom() {
        for x in area.left()..area.right() {
            buffer[(x, y)].set_char(' ').set_bg(if x - area.x < filled {
                colour
            } else {
                Color::Black
            });
        }
    }

    // The label reads against whatever it lands on: black where the bar has reached it, and the
    // bar's own colour on the empty part, which is the only pair that stays legible on both.
    for (at, letter) in text.into_iter().enumerate() {
        let x = start + at as u16;
        buffer[(x, row)]
            .set_char(letter)
            .set_fg(if x - area.x < filled {
                Color::Black
            } else {
                colour
            });
    }
}

fn print_usage() {
    eprintln!("Usage: waifu COMMAND");
    eprintln!();
    eprintln!("Commands:");
    eprintln!("    draw           Draw a picture with your waifu");
    eprintln!();
    eprintln!("Run 'waifu COMMAND -h' for more information on a command.");
}

/// Runs the command line tool: reads the process arguments and dispatches to a subcommand.
pub fn run() -> ExitCode {
    let arguments: Vec<String> = std::env::args().skip(1).collect();
    let Some(command) = arguments.first() else {
        print_usage();
        return ExitCode::FAILURE;
    };

    let rest = &arguments[1..];
    let result = match command.as_str() {
        "draw" => draw::main(rest),
        other => {
            eprintln!("Invalid command \"{other}\"\n");
            print_usage();
            return ExitCode::FAILURE;
        }
    };

    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("Error: {error}");
            ExitCode::FAILURE
        }
    }
}
