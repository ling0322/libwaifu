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
mod tasks;

use std::io::{self, IsTerminal};
use std::process::ExitCode;

use ratatui::layout::Rect;
use ratatui::style::{Color, Style, Stylize};
use ratatui::text::{Line, Span};
use ratatui::{DefaultTerminal, Frame};

type Error = Box<dyn std::error::Error>;

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

/// Takes the terminal and paints over whatever was on it.
///
/// The clear is the whole reason this is a function. ratatui writes only what changed since the
/// frame before, and the first frame is compared against a blank one, so a cell that is blank in
/// the drawing is a cell it never writes -- it is relying on the alternate screen it just entered
/// being empty. Which it is, unless the terminal was already in one: entering it a second time
/// changes nothing, and then every blank cell shows whatever the last program left there.
///
/// That is not a rare shape. It is what a program that took the screen and died without giving it
/// back leaves behind, which until a moment ago was what a failed check inside the tensor library
/// did, and one of those can leave every run after it looking corrupted.
fn take_the_screen() -> Result<DefaultTerminal, Error> {
    let mut terminal = ratatui::init();
    terminal.clear()?;

    Ok(terminal)
}

fn print_usage() {
    eprintln!("Usage: waifu COMMAND");
    eprintln!();
    eprintln!("Commands:");
    for task in tasks::ALL {
        eprintln!("    {:<15}{}", task.name, task.about);
    }
    eprintln!();
    eprintln!("Run 'waifu COMMAND -h' for more information on a command.");

    // Said only where it is true. This same usage is what a run with no terminal to draw on gets
    // instead of the box, and telling that run to do the thing it just could not do would be
    // worse than saying nothing.
    if can_take_the_screen() {
        eprintln!("Run 'waifu' on its own to pick one on screen.");
    }
}

/// Whether there is a terminal to put a screen on.
///
/// Piped or redirected -- a script, a CI job, a shell completion -- there is nobody at a keyboard
/// to pick anything with, and `ratatui::init` on something that is not a terminal panics rather
/// than reporting it.
fn can_take_the_screen() -> bool {
    io::stdin().is_terminal() && io::stdout().is_terminal()
}

/// Whether what was typed is a request for this usage rather than for a command.
///
/// Asked only of arguments with no command in front of them: with one, `-h` belongs to that
/// command and is its own usage to print. Without one there is nothing else it could mean, and
/// answering it by opening a box someone has to press escape out of would be a poor answer to
/// somebody asking what the commands are.
fn wants_usage(arguments: &[String]) -> bool {
    arguments
        .iter()
        .any(|argument| matches!(argument.as_str(), "-h" | "--h" | "-help" | "--help"))
}

/// Runs the command line tool: reads the process arguments and dispatches to a subcommand.
pub fn run() -> ExitCode {
    let arguments: Vec<String> = std::env::args().skip(1).collect();

    // The first argument is the command, unless it starts with a dash, in which case no command
    // was named at all: `waifu -m sdxl:base` says nothing about which task and plenty about how to
    // run it, so the flags are kept for whichever task is picked and handed on as they were typed.
    let (task, rest) = match arguments.first().filter(|first| !first.starts_with('-')) {
        Some(command) => match tasks::named(command) {
            Some(task) => (task, &arguments[1..]),
            None => {
                eprintln!("Invalid command \"{command}\"\n");
                print_usage();
                return ExitCode::FAILURE;
            }
        },
        None => {
            if wants_usage(&arguments) {
                print_usage();
                return ExitCode::SUCCESS;
            }

            // Nothing to draw on, so the usage is the answer, and a failure as it always was:
            // whatever asked for this was not asking for a box.
            if !can_take_the_screen() {
                print_usage();
                return ExitCode::FAILURE;
            }

            match ask_what_to_do() {
                Ok(Some(task)) => (task, &arguments[..]),
                // Nothing picked. Nothing went wrong either, so nothing is printed and the shell
                // is given back a zero.
                Ok(None) => return ExitCode::SUCCESS,
                Err(error) => {
                    eprintln!("Error: {error}");
                    return ExitCode::FAILURE;
                }
            }
        }
    };

    match task.run(rest) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("Error: {error}");
            ExitCode::FAILURE
        }
    }
}

/// Puts the task list on screen and gives the terminal back before whatever was picked runs.
///
/// Given back rather than passed along: a command takes the screen itself -- `draw` prints what
/// hardware the tensor library found before its own screen goes up -- and a command that did not
/// would have been handed a terminal it has no use for.
fn ask_what_to_do() -> Result<Option<&'static tasks::Task>, Error> {
    let mut terminal = take_the_screen()?;
    let chosen = tasks::choose(&mut terminal);
    ratatui::restore();

    chosen
}
