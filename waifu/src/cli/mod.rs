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
mod draw;
mod field;
mod hub;
mod picker;
mod png;

use std::process::ExitCode;

use ratatui::style::{Color, Style, Stylize};
use ratatui::text::{Line, Span};

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
