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

//! A reading for the mouth to say: its sound, and the shapes its voice said it in, if it said
//! them. **An experiment, with the mouth shapes.**
//!
//! The voice's visemes say which shape the mouth is in -- A, I, U -- but not how far open: a
//! consonant other than b, p or m is labelled as the vowels either side of it, so the mouth never
//! shuts between syllables, while the sound rises and falls five times a second. So the sound says
//! how open: loud, the viseme's shape whole; quiet, the drawn mouth, shut. MBP and REST shut it
//! whatever the sound. With no visemes -- a voice that says none, a microphone -- an open mouth is
//! A's.
//!
//! The loudness is libwaifu's where it is handed over (`set_loudness`, a text to speech reading's
//! LOUDNESS); where it is not, it is heard from the samples (`push`) the same way: frame i the
//! milliseconds [i, i + 1) * 10, the dB of the RMS of the 20 ms about its middle.

use crate::mouths::Viseme;

/// How many milliseconds a frame of the loudness heard from the samples is, and how many of them
/// each is heard over, about its middle: libwaifu's.
const HOP_MS: u64 = 10;
const WINDOW_MS: u64 = 20;
/// How far ahead of the sound the mouth reads it: it is eased into its shapes over about 50 ms,
/// and a mouth seen to move a little before it is heard reads as in time, one after as late.
pub const LEAD: f64 = 0.040;
/// The loudness that is shut and the one that is open whole, in dB under the loudest of the
/// reading so far.
const SHUT_DB: f32 = 30.0;
const OPEN_DB: f32 = 8.0;

#[derive(Clone, Debug, Default)]
pub struct Speech {
    rate: u32,
    /// How many samples it has been handed, all told.
    heard: usize,
    /// The samples from `kept_from` on: what the frames still to hear reach back to, no more --
    /// and none once the loudness is given.
    kept: Vec<f32>,
    kept_from: usize,
    /// The loudness, a frame of `step` seconds at a time, in dB: every frame whose window has been
    /// heard.
    loudness: Vec<f32>,
    step: f64,
    /// Whether `loudness` is what was handed over, not heard from the samples.
    given: bool,
    loudest: f32,
    visemes: Vec<Viseme>,
    frame: f64,
}

impl Speech {
    /// A reading at `rate` samples a second, with nothing of it yet.
    pub fn new(rate: u32) -> Speech {
        Speech {
            rate: rate.max(1),
            loudest: f32::MIN,
            step: HOP_MS as f64 / 1000.0,
            frame: 0.04,
            ..Speech::default()
        }
    }

    pub fn rate(&self) -> u32 {
        self.rate
    }

    /// How many samples it has been handed.
    pub fn len(&self) -> usize {
        self.heard
    }

    pub fn is_empty(&self) -> bool {
        self.heard == 0
    }

    /// How many of its samples it is keeping: no more than the frames still to hear reach back to.
    pub fn kept(&self) -> usize {
        self.kept.len()
    }

    /// The samples of frame i's window, [from, to): from its middle, (i + 1/2) * HOP_MS, half a
    /// window either way, each the sample its time falls in -- worked out for each frame, so that a
    /// rate a frame is not whole samples of keeps to the grid.
    fn window(&self, i: usize) -> (usize, usize) {
        let rate = self.rate as u64;
        // Twice the milliseconds, to keep the middle whole.
        let middle = (2 * i as u64 + 1) * HOP_MS;
        let from = middle.saturating_sub(WINDOW_MS) * rate / 2000;
        let to = (middle + WINDOW_MS) * rate / 2000;
        (from as usize, to as usize)
    }

    /// The reading's next samples, mono, -1 to 1: a streaming reading's parts as they come. The
    /// loudness is heard from them unless it has been handed over.
    pub fn push(&mut self, samples: &[f32]) {
        self.heard += samples.len();
        if self.given {
            return;
        }
        self.kept.extend_from_slice(samples);
        // Frame i is heard over the window about its middle, once all of that has come.
        loop {
            let (from, to) = self.window(self.loudness.len());
            if to > self.heard {
                break;
            }
            let s = &self.kept[from.max(self.kept_from) - self.kept_from..to - self.kept_from];
            let power = s.iter().map(|v| v * v).sum::<f32>() / s.len().max(1) as f32;
            let db = 10.0 * (power + 1e-10).log10();
            self.loudest = self.loudest.max(db);
            self.loudness.push(db);
        }
        // What no frame still to hear reaches back to goes.
        let need = self.window(self.loudness.len()).0;
        if need > self.kept_from {
            self.kept
                .drain(..(need - self.kept_from).min(self.kept.len()));
            self.kept_from = need;
        }
    }

    /// How loud the reading is, in dB, a frame of `frame_ms` each, frame i the milliseconds
    /// [i, i + 1) * frame_ms -- libwaifu's LOUDNESS: all of it so far, again with each part of a
    /// streaming reading. From then on the samples are not listened to.
    pub fn set_loudness(&mut self, db: &[f32], frame_ms: u32) {
        self.loudness = db.to_vec();
        self.step = frame_ms.max(1) as f64 / 1000.0;
        self.given = true;
        self.loudest = db.iter().copied().fold(f32::MIN, f32::max);
        self.kept = Vec::new();
    }

    /// The shapes the voice said, every one from the reading's start, a frame of `frame_ms` each:
    /// all of them so far, again with each part of a streaming reading. A shape held for one frame
    /// between two others is taken for a stumble and given the one before -- but MBP, whose shut
    /// lips are often no longer.
    pub fn set_visemes(&mut self, visemes: &[i32], frame_ms: u32) {
        let mut v: Vec<Viseme> = visemes
            .iter()
            .map(|&i| Viseme::from_index(i).unwrap_or(Viseme::Rest))
            .collect();
        for i in 1..v.len().saturating_sub(1) {
            if v[i] != v[i - 1] && v[i] != v[i + 1] && v[i] != Viseme::Mbp {
                v[i] = v[i - 1];
            }
        }
        self.visemes = v;
        self.frame = frame_ms.max(1) as f64 / 1000.0;
    }

    /// How open the mouth is `seconds` into the reading, 0 shut to 1 as open as its shape:
    /// `None` past what has been heard of it.
    pub fn openness(&self, seconds: f64) -> Option<f32> {
        if seconds < 0.0 || self.loudness.is_empty() {
            return None;
        }
        // Between the middles of the frames either side.
        let at = (seconds / self.step - 0.5).max(0.0);
        let i = at.floor() as usize;
        if i + 1 >= self.loudness.len() {
            return None;
        }
        let f = (at - at.floor()) as f32;
        let db = self.loudness[i] * (1.0 - f) + self.loudness[i + 1] * f;
        // Under the loudest so far -- but never so quiet a reading that its hiss opens the mouth.
        let top = self.loudest.max(-40.0);
        let t = ((db - (top - SHUT_DB)) / (SHUT_DB - OPEN_DB)).clamp(0.0, 1.0);
        Some(t * t * (3.0 - 2.0 * t))
    }

    /// The shape the voice said at `seconds`, or `None` where it said none there.
    pub fn viseme(&self, seconds: f64) -> Option<Viseme> {
        if seconds < 0.0 {
            return None;
        }
        self.visemes.get((seconds / self.frame) as usize).copied()
    }

    /// The mouth `seconds` into the reading, read `LEAD` ahead, as each shape's weight by
    /// `Viseme as usize`: the shape as open as the sound, the drawn mouth the rest. `None` past what
    /// has been heard: the mouth at rest.
    pub fn mouth(&self, seconds: f64) -> Option<[f32; 8]> {
        let t = seconds + LEAD;
        let open = self.openness(t)?;
        let shape = match self.viseme(t) {
            Some(v) => v,
            // A voice that says no shapes: open is A's.
            None if self.visemes.is_empty() => Viseme::A,
            // Past the shapes settled so far: at rest until they are.
            None => return None,
        };
        let mut w = [0.0; 8];
        match shape {
            Viseme::Rest | Viseme::Mbp => w[shape as usize] = 1.0,
            v => {
                w[v as usize] = open;
                w[Viseme::Rest as usize] = 1.0 - open;
            }
        }
        Some(w)
    }
}
