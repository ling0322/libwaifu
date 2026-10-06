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

//! What a session is for, and what was settled before the page opened.
//!
//! The terminal settles three things -- the task, the model, the device -- and the page is served
//! for exactly those. The page does not change any of them: it has no model list, no download
//! button and no device box, because all three are questions with a long wait behind a wrong
//! answer, and the terminal is where the wait can be watched and stopped.

use crate::cli::args::Runtime;
/// One kind of session: a tab on the page, and a row in the terminal's first list.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Task {
    Txt2Img,
    Img2Img,
    Text2Speech,
    /// Voice conversion: CosyVoice3's in every build, and Seed-VC's -- which is GPL-3.0 -- in a
    /// build with the `gpl` feature.
    Speech2Speech,
}

/// Why an MIT build will not run Seed-VC, said wherever it is asked for.
pub const NO_GPL: &str = "Seed-VC is GPL-3.0 and this build is MIT: build with `--features gpl` \
     (CMake: -DENABLE_GPL=ON) to have it, or convert with cosyvoice, which every build has";

impl Task {
    /// Every task, in the order the terminal lists them and the usage names them.
    pub const ALL: &'static [Task] = &[
        Task::Txt2Img,
        Task::Img2Img,
        Task::Text2Speech,
        Task::Speech2Speech,
    ];

    /// What it is called: on the command line, in the list, and on the page's tab.
    pub fn name(self) -> &'static str {
        match self {
            Task::Txt2Img => "txt2img",
            Task::Img2Img => "img2img",
            Task::Text2Speech => "text2speech",
            Task::Speech2Speech => "speech2speech",
        }
    }

    /// What it does, in the one line the list has room for.
    pub fn about(self) -> &'static str {
        match self {
            Task::Txt2Img => "Draw a picture from a prompt",
            Task::Img2Img => "Draw a picture from a prompt and a picture to start from",
            Task::Text2Speech => "Read some text out loud, in a voice",
            Task::Speech2Speech => "Say a recording again in the voice of another one",
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

/// What the terminal hands to the page: a task, the model or voice it runs, and where.
///
/// The model is on the disk by the time one of these exists. The terminal fetched it under its own
/// bar, or the command line named a file that is already there.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Launch {
    pub task: Task,
    /// A catalogue name or a manifest's path: what the worker hands to the hub to read.
    pub model: String,
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
    }

    #[test]
    fn voice_conversion_is_a_task_in_every_build() {
        // CosyVoice3 converts in an MIT build; only Seed-VC needs the `gpl` feature, and says so.
        assert_eq!(Task::named("speech2speech"), Some(Task::Speech2Speech));
        assert!(Task::ALL.contains(&Task::Speech2Speech));
        assert!(NO_GPL.contains("GPL-3.0") && NO_GPL.contains("--features gpl"));
    }
}
