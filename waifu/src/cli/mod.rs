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
//! Behind the `cli` feature, because it is the only thing in this crate that needs a server and
//! an image encoder. The `waifu` binary is a shim over [`run`]; everything else here is its
//! implementation.
//!
//! There is one thing the tool does, and the command line exists to say how to do it rather than
//! which to do: `waifu` on its own is the same as `waifu webui`, and both open a page in a browser
//! where everything else is asked.

mod args;
mod hub;
mod webui;

use std::process::ExitCode;

type Error = Box<dyn std::error::Error>;

/// The commit this was built from, stamped in by build.rs. "unknown" when it was built from a
/// copy of the source with no git around it, which is a thing that has to keep working.
pub(crate) const REVISION: &str = env!("WAIFU_REVISION");

/// What the tool can be asked to do.
///
/// One row, and a table anyway: the usage prints from it and the command line dispatches through
/// it, so a command cannot be in one and missing from the other -- and the day a second one lands
/// is not the day to go looking for the three places that named the first.
struct Task {
    /// What it is called, which is what is typed.
    name: &'static str,
    /// What it does, in the one line the usage has room for.
    about: &'static str,
    /// The command itself, handed everything that was not its name.
    main: fn(&[String]) -> Result<(), Error>,
}

static ALL: &[Task] = &[Task {
    name: "webui",
    about: "Open the web UI: draw a picture, or say something, with your waifu",
    main: webui::main,
}];

/// The task a word names, if it names one.
fn named(word: &str) -> Option<&'static Task> {
    ALL.iter().find(|task| task.name == word)
}

fn print_usage() {
    eprintln!("Usage: waifu COMMAND");
    eprintln!();
    eprintln!("Commands:");
    for task in ALL {
        eprintln!("    {:<15}{}", task.name, task.about);
    }
    eprintln!();
    eprintln!("Run 'waifu COMMAND -h' for more information on a command.");
}

/// Whether what was typed is a request for this usage rather than for a command.
///
/// Asked only of arguments with no command in front of them: with one, `-h` belongs to that
/// command and is its own usage to print.
fn wants_usage(arguments: &[String]) -> bool {
    arguments
        .iter()
        .any(|argument| matches!(argument.as_str(), "-h" | "--h" | "-help" | "--help"))
}

/// Runs the command line tool: reads the process arguments and dispatches to a subcommand.
pub fn run() -> ExitCode {
    let arguments: Vec<String> = std::env::args().skip(1).collect();

    // The first argument is the command, unless it starts with a dash, in which case no command
    // was named at all: `waifu -m sdxl:base` says nothing about which task and plenty about how
    // to run it, so the flags are kept for the task that is about to run and handed on as they
    // were typed.
    let (task, rest) = match arguments.first().filter(|first| !first.starts_with('-')) {
        Some(command) => match named(command) {
            Some(task) => (task, &arguments[1..]),
            None => {
                eprintln!("Invalid command \"{command}\"\n");
                print_usage();
                return ExitCode::FAILURE;
            }
        },
        // Nothing but flags, or nothing at all. There is one thing this tool does, so doing it is
        // a better answer than a usage telling someone to type the only word there is.
        None if wants_usage(&arguments) => {
            print_usage();
            return ExitCode::SUCCESS;
        }
        None => (&ALL[0], &arguments[..]),
    };

    match task.run(rest) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("Error: {error}");
            ExitCode::FAILURE
        }
    }
}

impl Task {
    fn run(&self, arguments: &[String]) -> Result<(), Error> {
        (self.main)(arguments)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_task_is_a_word_the_command_line_takes() {
        for task in ALL {
            assert!(
                std::ptr::eq(named(task.name).unwrap(), task),
                "{}",
                task.name
            );
        }
        assert!(named("sing").is_none());
    }

    #[test]
    fn a_request_for_help_is_not_a_command() {
        for asked in ["-h", "--help"] {
            assert!(wants_usage(&[asked.to_string()]));
        }
        assert!(!wants_usage(&["-m".to_string(), "sdxl:base".to_string()]));
    }
}
