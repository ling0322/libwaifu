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

//! The mouth shape of every 40 ms of a reading: a [`Viseme`] for each speech token the flow is
//! handed, laid on the audio the token became.
//!
//! Three steps, each its own type:
//!
//! - [`VisemeModel`]: a token and the token either side of it -- the flow's `input_embedding` of
//!   each, 240 numbers -- to a distribution over the eight visemes, through one hidden layer of
//!   256. `tools/cosyvoice3_visemes.py` trained it on readings in Chinese, Japanese and English,
//!   labelled from their text, and writes it as one more weights file of the package.
//! - [`VisemeTrack`]: those distributions laid on the reading's samples -- token `j` of a sentence
//!   is the `j`-th equal share of the sentence's audio -- and read back a frame of [`FRAME_MS`] at
//!   a time. At speed one a frame is a token; at another speed a frame is the tokens under it.
//! - [`VisemeSmoother`]: each frame's distribution to one viseme.
//!
//! A package that has it lists the file under `weights:` and names the classes, in order, under
//! `config: cosyvoice3: visemes:` -- `[A, E, I, O, U, "@", MBP, REST]` -- which is how a manifest
//! says it can without its weights being read.

use crate::flint::{DType, Device, Weights};
use crate::speech::Viseme;
use crate::{Error, Result};

/// How many milliseconds a frame holds: a speech token's, at 25 a second.
pub const FRAME_MS: u32 = 40;

/// What the package's `config: cosyvoice3:` names the classes under.
pub const MANIFEST_KEY: &str = "visemes";

/// Where its weights are.
pub const NAMESPACE: &str = "cosyvoice3.visemes";

/// What it reads each token as: the flow's own embedding of it.
const EMBEDDING: &str = "cosyvoice3.flow.input_embedding.weight";

const CLASSES: usize = Viseme::ALL.len();
const TOKENS: usize = 6561;
const WIDTH: usize = 80;
/// A token and one either side.
const WINDOW: usize = 3;
const HIDDEN: usize = 256;

/// A distribution over [`Viseme::ALL`], in that order.
pub type Shape = [f32; CLASSES];

/// Whether `listed` -- the manifest's `visemes`, its items joined by commas -- names the classes
/// this reads, in the order it reads them.
pub fn check_classes(listed: &str) -> Result<()> {
    let listed: Vec<&str> = listed.split(',').map(str::trim).collect();
    let known: Vec<&str> = Viseme::ALL.iter().map(|v| v.spelling()).collect();
    match listed == known {
        true => Ok(()),
        false => Err(Error::model(format!(
            "the package's visemes are {listed:?}, and this reads {known:?}, in that order"
        ))),
    }
}

/// A token, with its neighbours, to a distribution over the visemes:
/// `softmax(w2 gelu(w1 [E[t-1], E[t], E[t+1]] + b1) + b2)`, a neighbour past either end of the
/// sentence being zeros. Run on the host, where it is a few hundred thousand multiplications a
/// second of speech.
pub struct VisemeModel {
    /// `(6561, 80)`.
    embedding: Vec<f32>,
    /// `(256, 240)`, the input's normalization folded in.
    w1: Vec<f32>,
    b1: Vec<f32>,
    /// `(8, 256)`.
    w2: Vec<f32>,
    b2: Vec<f32>,
}

impl VisemeModel {
    /// Out of a package's weights, onto the host.
    pub fn build(weights: &Weights) -> Result<VisemeModel> {
        let host = |name: &str, shape: &[i32]| -> Result<Vec<f32>> {
            let tensor = weights.load(name, shape)?.to_device(Device::Cpu)?;
            Ok(match tensor.dtype() {
                DType::Float => tensor.to_vec_f32()?,
                _ => tensor.cast(DType::Float)?.to_vec_f32()?,
            })
        };
        let at = |part: &str| format!("{NAMESPACE}.{part}");
        let (hidden, input) = (HIDDEN as i32, (WINDOW * WIDTH) as i32);
        VisemeModel::new(
            host(EMBEDDING, &[TOKENS as i32, WIDTH as i32])?,
            host(&at("w1"), &[hidden, input])?,
            host(&at("b1"), &[hidden])?,
            host(&at("w2"), &[CLASSES as i32, hidden])?,
            host(&at("b2"), &[CLASSES as i32])?,
        )
    }

    /// Of its parts, each row-major in the shape [`VisemeModel`]'s fields say.
    pub fn new(
        embedding: Vec<f32>,
        w1: Vec<f32>,
        b1: Vec<f32>,
        w2: Vec<f32>,
        b2: Vec<f32>,
    ) -> Result<VisemeModel> {
        let sizes = [
            (embedding.len(), TOKENS * WIDTH),
            (w1.len(), HIDDEN * WINDOW * WIDTH),
            (b1.len(), HIDDEN),
            (w2.len(), CLASSES * HIDDEN),
            (b2.len(), CLASSES),
        ];
        if sizes.iter().any(|(got, want)| got != want) {
            return Err(Error::model(format!(
                "the viseme model's parts are {:?} long, not {:?}",
                sizes.map(|(got, _)| got),
                sizes.map(|(_, want)| want)
            )));
        }
        Ok(VisemeModel {
            embedding,
            w1,
            b1,
            w2,
            b2,
        })
    }

    /// Token `at` of `tokens`, a sentence: its distribution, read with the tokens either side.
    pub fn shape_at(&self, tokens: &[i32], at: usize) -> Shape {
        let row = |token: Option<&i32>| -> Option<&[f32]> {
            let token = usize::try_from(*token?).ok().filter(|&t| t < TOKENS)?;
            Some(&self.embedding[token * WIDTH..(token + 1) * WIDTH])
        };
        let window = [
            at.checked_sub(1).and_then(|before| row(tokens.get(before))),
            row(tokens.get(at)),
            row(tokens.get(at + 1)),
        ];

        let mut hidden = self.b1.clone();
        for (h, out) in hidden.iter_mut().enumerate() {
            let weights = &self.w1[h * WINDOW * WIDTH..(h + 1) * WINDOW * WIDTH];
            for (k, row) in window.iter().enumerate() {
                if let Some(row) = row {
                    let weights = &weights[k * WIDTH..(k + 1) * WIDTH];
                    *out += weights.iter().zip(*row).map(|(w, x)| w * x).sum::<f32>();
                }
            }
            *out = gelu(*out);
        }

        let mut scores = [0f32; CLASSES];
        for (c, score) in scores.iter_mut().enumerate() {
            let weights = &self.w2[c * HIDDEN..(c + 1) * HIDDEN];
            *score = self.b2[c] + weights.iter().zip(&hidden).map(|(w, h)| w * h).sum::<f32>();
        }
        softmax(scores)
    }

    /// Every token of `tokens`, a sentence.
    pub fn shapes(&self, tokens: &[i32]) -> Vec<Shape> {
        (0..tokens.len())
            .map(|at| self.shape_at(tokens, at))
            .collect()
    }
}

/// The exact GELU, `x Φ(x)`, as torch's default is.
fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + erf(x / std::f32::consts::SQRT_2))
}

/// Abramowitz and Stegun's 7.1.26, within 1.5e-7 of erf everywhere: well inside what a float32
/// hidden layer carries.
fn erf(x: f32) -> f32 {
    let (a1, a2, a3, a4, a5, p) = (
        0.254_829_6_f64,
        -0.284_496_74,
        1.421_413_7,
        -1.453_152_1,
        1.061_405_4,
        0.327_591_1,
    );
    let sign = x.signum() as f64;
    let x = (x as f64).abs();
    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - ((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t * (-x * x).exp();
    (sign * y) as f32
}

fn softmax(scores: Shape) -> Shape {
    let most = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut out = scores.map(|s| (s - most).exp());
    let total: f32 = out.iter().sum();
    out.iter_mut().for_each(|p| *p /= total);
    out
}

/// The likeliest viseme of a frame; the first of a tie.
fn likeliest(shape: &Shape) -> Viseme {
    let mut best = 0;
    for (k, p) in shape.iter().enumerate() {
        if *p > shape[best] {
            best = k;
        }
    }
    Viseme::ALL[best]
}

/// What the mouth does at rest, for a frame nothing was said in.
fn at_rest() -> Shape {
    let mut shape = [0.0; CLASSES];
    shape[Viseme::Rest as usize] = 1.0;
    shape
}

/// Each frame's distribution to one viseme, without flickering between two that are close.
///
/// TODO: Viterbi over a transition matrix learnt from the soft labels `tools/cosyvoice3_visemes.py`
/// writes, the scores divided by each class's share of them, with a fixed lag of a few frames for
/// a streaming reading -- which costs nothing there, the language model being ahead of the audio.
/// For now each frame's likeliest viseme, which flickers where two are close.
#[derive(Clone, Debug, Default)]
pub struct VisemeSmoother {}

impl VisemeSmoother {
    pub fn new() -> VisemeSmoother {
        VisemeSmoother::default()
    }

    /// The whole of a reading at once.
    pub fn smooth(&self, frames: &[Shape]) -> Vec<Viseme> {
        let mut smoother = self.clone();
        let mut out: Vec<Viseme> = frames.iter().filter_map(|f| smoother.push(*f)).collect();
        out.extend(smoother.finish());
        out
    }

    /// The next frame; the viseme of the earliest frame not yet decided, where one now is. Every
    /// frame pushed comes out once, in order, here or from [`finish`](VisemeSmoother::finish).
    pub fn push(&mut self, frame: Shape) -> Option<Viseme> {
        Some(likeliest(&frame))
    }

    /// The end of the reading: every frame not yet decided.
    pub fn finish(&mut self) -> Vec<Viseme> {
        Vec::new()
    }
}

/// A stretch of a reading's samples one token became, and its distribution.
#[derive(Clone, Copy, Debug)]
struct Segment {
    start: u64,
    end: u64,
    shape: Shape,
}

/// The tokens of a reading laid on its samples, read back a frame of [`FRAME_MS`] at a time.
///
/// Frame `i` is samples `[i, i + 1) * frame`: its distribution is that of the tokens under it,
/// each by how much of the frame it covers, and a frame no token covers is at rest. Tokens are
/// laid in order, and frames come out in order, each once.
#[derive(Debug)]
pub struct VisemeTrack {
    /// Samples a frame.
    frame: u64,
    segments: Vec<Segment>,
    /// The first segment the next frame can lie over.
    first: usize,
    /// Frames handed to the smoother so far.
    framed: u64,
    smoother: VisemeSmoother,
}

impl VisemeTrack {
    /// For audio at `rate`.
    pub fn new(rate: u32) -> VisemeTrack {
        VisemeTrack {
            frame: (rate as u64 * FRAME_MS as u64 / 1000).max(1),
            segments: Vec::new(),
            first: 0,
            framed: 0,
            smoother: VisemeSmoother::new(),
        }
    }

    /// Samples `[start, end)` are one token, of `shape`.
    pub fn lay(&mut self, start: u64, end: u64, shape: Shape) {
        if end > start {
            self.segments.push(Segment { start, end, shape });
        }
    }

    /// Samples `[start, end)` are `shapes`, a sentence's tokens, each an equal share of them.
    pub fn lay_sentence(&mut self, start: u64, end: u64, shapes: &[Shape]) {
        let (length, count) = (end.saturating_sub(start), shapes.len() as u64);
        for (j, shape) in shapes.iter().enumerate() {
            let j = j as u64;
            self.lay(
                start + length * j / count,
                start + length * (j + 1) / count,
                *shape,
            );
        }
    }

    /// The visemes decided of every frame that lies wholly before sample `upto` and under the
    /// tokens laid so far, that had not been handed over before.
    pub fn settle(&mut self, upto: u64) -> Vec<Viseme> {
        let laid = self.segments.last().map_or(0, |s| s.end);
        let ready = upto.min(laid);
        let mut out = Vec::new();
        while (self.framed + 1) * self.frame <= ready {
            let start = self.framed * self.frame;
            let shape = self.shape_of(start, start + self.frame);
            self.framed += 1;
            out.extend(self.smoother.push(shape));
        }
        out
    }

    /// The end of a reading of `total` samples: the visemes of every frame not yet handed over,
    /// the last of them as much of a frame as is left.
    pub fn finish(&mut self, total: u64) -> Vec<Viseme> {
        let mut out = self.settle(total);
        while self.framed * self.frame < total {
            let start = self.framed * self.frame;
            let shape = self.shape_of(start, (start + self.frame).min(total));
            self.framed += 1;
            out.extend(self.smoother.push(shape));
        }
        out.extend(self.smoother.finish());
        out
    }

    /// The distribution of samples `[start, end)`.
    fn shape_of(&mut self, start: u64, end: u64) -> Shape {
        while self.first < self.segments.len() && self.segments[self.first].end <= start {
            self.first += 1;
        }
        let mut sum = [0f64; CLASSES];
        let mut covered = 0u64;
        for segment in &self.segments[self.first..] {
            if segment.start >= end {
                break;
            }
            let overlap = segment.end.min(end) - segment.start.max(start);
            covered += overlap;
            for (s, p) in sum.iter_mut().zip(segment.shape) {
                *s += overlap as f64 * p as f64;
            }
        }
        match covered {
            0 => at_rest(),
            _ => sum.map(|s| (s / covered as f64) as f32),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn one_hot(viseme: Viseme) -> Shape {
        let mut shape = [0.0; CLASSES];
        shape[viseme as usize] = 1.0;
        shape
    }

    #[test]
    fn erf_is_erf() {
        for (x, want) in [
            (0.0, 0.0),
            (0.5, 0.520_499_9),
            (1.0, 0.842_700_8),
            (-2.0, -0.995_322_3),
        ] {
            assert!((erf(x) - want).abs() < 2e-7, "erf({x}) = {}", erf(x));
        }
    }

    #[test]
    fn the_package_spells_the_classes_in_order() {
        check_classes("A,E,I,O,U,@,MBP,REST").unwrap();
        check_classes("A, E, I, O, U, @, MBP, REST").unwrap();
        assert!(check_classes("A,E,I,O,U,MBP,REST,@").is_err());
        assert!(check_classes("A,E,I,O,U").is_err());
    }

    /// A model whose hidden layer copies the middle token's first eight numbers, and whose output
    /// is them: so a token whose embedding is one-hot in the first eight is that viseme.
    fn copying() -> VisemeModel {
        let mut embedding = vec![0.0; TOKENS * WIDTH];
        for token in 0..TOKENS {
            embedding[token * WIDTH + token % CLASSES] = 4.0;
        }
        let mut w1 = vec![0.0; HIDDEN * WINDOW * WIDTH];
        for c in 0..CLASSES {
            w1[c * WINDOW * WIDTH + WIDTH + c] = 1.0;
        }
        let mut w2 = vec![0.0; CLASSES * HIDDEN];
        for c in 0..CLASSES {
            w2[c * HIDDEN + c] = 4.0;
        }
        VisemeModel::new(embedding, w1, vec![0.0; HIDDEN], w2, vec![0.0; CLASSES]).unwrap()
    }

    #[test]
    fn a_token_reads_its_own_embedding() {
        let model = copying();
        let shapes = model.shapes(&[0, 6, 7, 3]);
        let said: Vec<Viseme> = shapes.iter().map(likeliest).collect();
        assert_eq!(said, [Viseme::A, Viseme::Mbp, Viseme::Rest, Viseme::O]);
        for shape in shapes {
            assert!((shape.iter().sum::<f32>() - 1.0).abs() < 1e-5);
        }
        assert!(VisemeModel::new(vec![0.0; 3], vec![], vec![], vec![], vec![]).is_err());
    }

    #[test]
    fn at_speed_one_a_frame_is_a_token() {
        // 24 kHz: 960 samples a frame and a token.
        let mut track = VisemeTrack::new(24000);
        let said = [Viseme::A, Viseme::Mbp, Viseme::I];
        track.lay_sentence(0, 3 * 960, &said.map(one_hot));
        assert_eq!(track.finish(3 * 960), said);
    }

    #[test]
    fn a_frame_is_the_tokens_under_it() {
        // Two tokens over one and a half frames, 720 samples each: the first frame is three
        // quarters A and a quarter O, the half frame left all O.
        let mut track = VisemeTrack::new(24000);
        track.lay_sentence(0, 1440, &[one_hot(Viseme::A), one_hot(Viseme::O)]);
        let first = track.shape_of(0, 960);
        assert!((first[Viseme::A as usize] - 0.75).abs() < 1e-6);
        assert!((first[Viseme::O as usize] - 0.25).abs() < 1e-6);
        let mut track = VisemeTrack::new(24000);
        track.lay_sentence(0, 1440, &[one_hot(Viseme::A), one_hot(Viseme::O)]);
        assert_eq!(track.finish(1440), [Viseme::A, Viseme::O]);
    }

    #[test]
    fn as_many_frames_as_the_audio_has_milliseconds_over_forty() {
        for total in [1, 959, 960, 961, 10_000] {
            let mut track = VisemeTrack::new(24000);
            track.lay_sentence(0, total, &[one_hot(Viseme::E); 3]);
            assert_eq!(
                track.finish(total).len() as u64,
                total.div_ceil(960),
                "{total}"
            );
        }
        // Nothing laid at all: every frame at rest.
        let mut track = VisemeTrack::new(24000);
        assert_eq!(track.finish(2000), [Viseme::Rest; 3]);
    }

    #[test]
    fn streaming_hands_over_no_frame_past_the_audio() {
        let mut track = VisemeTrack::new(24000);
        for j in 0..5u64 {
            track.lay(j * 960, (j + 1) * 960, one_hot(Viseme::U));
        }
        // Two and a half tokens' audio handed over: two frames.
        assert_eq!(track.settle(2400).len(), 2);
        // Nothing new until a whole frame more is.
        assert!(track.settle(2500).is_empty());
        assert_eq!(track.settle(4800).len(), 3);
        // The tokens laid, not the audio, bound it as well.
        assert!(track.settle(10_000).is_empty());
        assert!(track.finish(4800).is_empty());
    }

    #[test]
    fn the_smoother_hands_every_frame_over_once() {
        let frames = [
            one_hot(Viseme::A),
            one_hot(Viseme::Schwa),
            one_hot(Viseme::Rest),
        ];
        assert_eq!(
            VisemeSmoother::new().smooth(&frames),
            [Viseme::A, Viseme::Schwa, Viseme::Rest]
        );
    }
}
