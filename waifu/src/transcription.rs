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

//! A recording in, the words said in it out: the speech2text task.
//!
//! What [`Voice`](crate::Voice) is for text2speech and [`Converter`](crate::Converter) for
//! speech2speech, and for the same reason: the engine, the C API and the programs over them are
//! written against [`Transcriber`], and a second model implements it without anything above this
//! file changing. [Qwen3-ASR](crate::qwen3_asr) is the first.

use std::ops::ControlFlow;

use crate::wav::Sound;
use crate::Result;

/// What a transcription is asked for, beyond the recording.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TranscriptionOptions {
    /// The language it is in, by one of the names [`Transcriber::languages`] lists, or `None` for
    /// the model to tell. Named, a model that would otherwise say which it heard writes the words
    /// only.
    pub language: Option<String>,
    /// Words that help it hear the recording -- names, terms, what it is about -- or empty for
    /// none. Not a prompt it answers: it is read before the recording, as context.
    pub context: String,
}

/// What came out of a recording.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Transcript {
    /// What was said.
    pub text: String,
    /// The language it was said in, as the model names it -- "Chinese", "English" -- or the one it
    /// was told to hear; empty where it heard no speech. A recording in more than one, cut into
    /// pieces, names each in turn: "Chinese,English".
    pub language: String,
}

/// How far along a transcription is, as the reporter given to [`Transcriber::transcribe`] is told.
///
/// Counted over the whole recording, so that a bar drawn from it only moves forward: a long one is
/// cut into pieces, each heard and then written out.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TranscriptionProgress {
    /// Hearing piece `done` of `total`, which is the encoder: a pass over the whole piece.
    Listening { done: i32, total: i32 },
    /// Writing out what was heard, `done` tokens so far of about `expected` -- an estimate from
    /// the length of the recording, as [`SpeechProgress::Saying`](crate::SpeechProgress)'s is: a
    /// model that writes one token at a time does not know how many until it stops.
    Writing { done: i32, expected: i32 },
}

/// Something that writes down what is said in a recording.
///
/// Not `Send`, for the reason a voice is not: it is made of tensors, and a tensor stays on the
/// thread that made it.
pub trait Transcriber {
    /// The words of `recording`, at any rate and of any length -- or `None`, where `report` asked
    /// it to stop.
    fn transcribe(
        &self,
        recording: &Sound,
        options: &TranscriptionOptions,
        report: &mut dyn FnMut(TranscriptionProgress) -> ControlFlow<()>,
    ) -> Result<Option<Transcript>>;

    /// What it is called on screen.
    fn name(&self) -> &'static str;

    /// The languages it can be told a recording is in, by the names
    /// [`TranscriptionOptions::language`] takes. Empty where it cannot be told.
    fn languages(&self) -> Vec<String>;
}
