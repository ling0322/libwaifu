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

//! What a session is for, and what was settled before the drawing screen opened.
//!
//! The terminal settles the model -- the task is txt2img, the one there is -- and the drawing
//! screen is opened for exactly that, on the device `-device` or `auto` starts it on. It does not
//! change the model: a different one is a fetch, and it is asked on the screen before it, where
//! the wait can be watched and stopped. The device it does change, from its device row: that is a
//! read of the model onto another card, which happens behind the screen.

use crate::args::Runtime;

/// One kind of session, and the boxes the drawing screen has. There is one, txt2img: it is kept
/// a name of its own because `-task` still takes it and the drawing screen's heading says it.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Task {
    Txt2Img,
}

impl Task {
    /// Every task, in the order the usage names them.
    pub const ALL: &'static [Task] = &[Task::Txt2Img];

    /// What it is called: on the command line and in the drawing screen's heading.
    pub fn name(self) -> &'static str {
        match self {
            Task::Txt2Img => "txt2img",
        }
    }

    /// The names of every task this build runs, as a usage lists them.
    pub fn names() -> String {
        Task::ALL
            .iter()
            .map(|task| task.name())
            .collect::<Vec<_>>()
            .join(", ")
    }

    /// The task a word names, which is [`Task::name`] read backwards.
    pub fn named(word: &str) -> Option<Task> {
        Task::ALL
            .iter()
            .copied()
            .find(|task| task.name() == word.trim().to_lowercase())
    }
}

/// What the terminal hands to the drawing screen: a task, the model it runs, and where.
///
/// The model is on the disk by the time one of these exists. The terminal fetched it under its own
/// bar, or the command line named a file that is already there.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Launch {
    pub task: Task,
    /// The path of the model's manifest: what libwaifu is handed to read.
    pub model: String,
    /// What the screens call it: the catalogue name it was picked by, or the manifest's file name.
    pub label: String,
    pub runtime: Runtime,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_task_is_a_word_the_command_line_takes() {
        for task in Task::ALL {
            assert_eq!(Task::named(task.name()), Some(*task), "{}", task.name());
        }
        assert_eq!(Task::named("TXT2IMG"), Some(Task::Txt2Img));
        assert_eq!(Task::named("sing"), None);

        // Drawing from a picture is the engine's, for the programs built over it: the terminal
        // refuses the name the way it refuses one it never had.
        assert_eq!(Task::named("img2img"), None);
    }

    #[test]
    fn the_terminal_draws_and_does_not_speak() {
        // Speech is the engine's, for the programs built over it: the terminal's task is the one
        // the drawing screen has boxes for.
        assert_eq!(Task::named("text2speech"), None);
        assert_eq!(Task::named("speech2speech"), None);
    }
}
