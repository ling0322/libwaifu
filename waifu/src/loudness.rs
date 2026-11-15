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

//! How loud a reading is from moment to moment: the C API's LOUDNESS, for a face to open its mouth
//! as loud as the sound is, or anything else that follows it.
//!
//! A frame a `FRAME_MS`, frame i the milliseconds [i, i + 1) * FRAME_MS of the audio, as the
//! visemes are framed; each the RMS of the samples in the `WINDOW_MS` about the frame's middle --
//! cut at the audio's start and end -- in dBFS, a full-scale sine's being -3, and never under
//! `FLOOR`. A streaming reading's frames are settled as its parts come, a frame once its window has
//! all been heard, so that none is ever past the audio handed over; `finish` settles the rest.

/// How long a frame is: short enough to see the dips between syllables, which last 30 to 60 ms.
pub const FRAME_MS: u32 = 10;
/// How much of the audio a frame is the loudness of, about its middle.
pub const WINDOW_MS: u32 = 20;
/// The quietest a frame is said to be: silence, and what is as good as.
pub const FLOOR: f32 = -100.0;

/// A reading's loudness, as its samples come.
#[derive(Clone, Debug)]
pub struct Loudness {
    frame: usize,
    half: usize,
    /// The samples from `kept_from` on: what the frames still to settle reach back to.
    kept: Vec<f32>,
    kept_from: usize,
    /// How many samples have come, all told.
    heard: usize,
    frames: Vec<f32>,
}

impl Loudness {
    /// The loudness of a reading at `rate` samples a second.
    pub fn new(rate: u32) -> Loudness {
        let frame = (rate as usize * FRAME_MS as usize / 1000).max(1);
        let half = (rate as usize * WINDOW_MS as usize / 2000).max(1);
        Loudness {
            frame,
            half,
            kept: Vec::new(),
            kept_from: 0,
            heard: 0,
            frames: Vec::new(),
        }
    }

    /// The loudness of all of `samples`, at `rate`.
    pub fn of(samples: &[f32], rate: u32) -> Vec<f32> {
        let mut loudness = Loudness::new(rate);
        loudness.push(samples);
        loudness.finish();
        loudness.frames
    }

    /// The frames settled so far, in dBFS.
    pub fn frames(&self) -> &[f32] {
        &self.frames
    }

    /// The reading's next samples; every frame whose window they complete is settled.
    pub fn push(&mut self, samples: &[f32]) {
        self.kept.extend_from_slice(samples);
        self.heard += samples.len();
        // Frame i's window ends at its middle and half a window: settled once that has come.
        while self.middle(self.frames.len()) + self.half <= self.heard {
            self.settle();
        }
        // What no frame still to settle reaches back to goes.
        let need = self.middle(self.frames.len()).saturating_sub(self.half);
        if need > self.kept_from {
            self.kept.drain(..need - self.kept_from);
            self.kept_from = need;
        }
    }

    /// No more samples: the frames to the end, ceil(milliseconds / FRAME_MS) of them in all, their
    /// windows cut at the end of the audio.
    pub fn finish(&mut self) {
        let total = self.heard.div_ceil(self.frame);
        while self.frames.len() < total {
            self.settle();
        }
    }

    fn middle(&self, i: usize) -> usize {
        i * self.frame + self.frame / 2
    }

    /// The next frame, from what is kept of its window.
    fn settle(&mut self) {
        let middle = self.middle(self.frames.len());
        let from = middle.saturating_sub(self.half).max(self.kept_from);
        let to = (middle + self.half).min(self.heard);
        let window = &self.kept[from - self.kept_from..to.max(from) - self.kept_from];
        let power = if window.is_empty() {
            0.0
        } else {
            window.iter().map(|&s| (s as f64) * (s as f64)).sum::<f64>() / window.len() as f64
        };
        let db = if power > 0.0 {
            (10.0 * power.log10()) as f32
        } else {
            FLOOR
        };
        self.frames.push(db.max(FLOOR));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A sine of 600 Hz: twelve whole periods in a window, so that its RMS is a sine's exactly.
    fn sine(seconds: f64, amplitude: f32, rate: u32) -> Vec<f32> {
        (0..(seconds * rate as f64) as usize)
            .map(|i| {
                amplitude * (i as f32 * 2.0 * std::f32::consts::PI * 600.0 / rate as f32).sin()
            })
            .collect()
    }

    #[test]
    fn a_full_scale_sine_is_minus_three_and_silence_the_floor() {
        let mut samples = sine(0.1, 1.0, 24000);
        samples.extend(vec![0.0; 2400]);
        let frames = Loudness::of(&samples, 24000);
        // 200 ms, a frame each 10.
        assert_eq!(frames.len(), 20);
        assert!((frames[4] + 3.01).abs() < 0.05, "{}", frames[4]);
        assert_eq!(frames[16], FLOOR);
        // Half as loud is 6 dB under.
        let half = Loudness::of(&sine(0.1, 0.5, 24000), 24000);
        assert!((half[4] - frames[4] + 6.02).abs() < 0.05);
    }

    #[test]
    fn frames_are_ceil_of_the_milliseconds() {
        assert_eq!(Loudness::of(&vec![0.1; 241], 24000).len(), 2);
        assert_eq!(Loudness::of(&vec![0.1; 240], 24000).len(), 1);
        assert!(Loudness::of(&[], 24000).is_empty());
    }

    #[test]
    fn streamed_in_parts_it_is_the_same_and_never_ahead() {
        let samples = sine(0.37, 0.3, 24000);
        let whole = Loudness::of(&samples, 24000);
        let mut parts = Loudness::new(24000);
        let mut heard = 0;
        for part in samples.chunks(1234) {
            parts.push(part);
            heard += part.len();
            // Never a frame whose window is past what has been heard: the last settled one's
            // middle, a frame of 240 in, and half its window of 480 after that, have all come.
            let settled = parts.frames().len();
            assert!(
                settled == 0 || (settled - 1) * 240 + 120 + 240 <= heard,
                "{settled} frames of {heard} samples"
            );
        }
        parts.finish();
        assert_eq!(parts.frames(), whole.as_slice());
    }
}
