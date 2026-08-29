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
mod png;

use std::process::ExitCode;

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
