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

//! What a recording becomes before any of IndexTTS-2.5's models sees it.
//!
//! Two analyses, and they are the same analysis twice. [`w2v_bert`] produces the 160-wide
//! features [`crate::indextts::w2v_bert`] reads; [`crate::indextts::campplus`] produces the 80-wide ones
//! [`crate::indextts::campplus`] reads. Both are Kaldi's filterbank -- the same frames, the same window,
//! the same triangles -- and they differ only in what is done to the result afterwards.
//!
//! ```text
//! let spectrograms = Spectrograms::new(device)?;
//! let features = spectrograms.w2v_bert(&wave_16k)?;       // (1, frames / 2, 160), on the device
//! let energies = spectrograms.campplus(&wave_16k)?;       // (1, frames, 80)
//! let mel = spectrograms.reference_mel(&wave_22k)?;       // (1, 80, frames)
//! ```
//!
//! [`Spectrograms`] is what the pipeline runs. The host functions -- [`w2v_bert`], [`campplus`],
//! [`reference_mel`] -- walk a frame sample by sample, are what the tests hold against torch, and
//! are what the device graphs are held against in turn.
//!
//! # Kaldi's frame is one matrix
//!
//! Kaldi does three things to a frame before its transform: removes its mean, runs a pre-emphasis
//! filter along it -- one that reads the sample before the one it writes -- and multiplies by
//! Povey's window. None of those is a transform on its own, but every one of them is linear in the
//! frame's 400 samples, and so is the 512-point DFT after them. Together they are one `(512, 400)`
//! matrix -- the real and imaginary part of each of 256 bins, as sums over the frame -- which
//! [`kaldi_basis`] works out once, and a graph convolves the recording by at Kaldi's hop. The whole
//! of the frame processing is one operator. The 257th bin, Nyquist's, is left out: Kaldi's
//! filterbank gives it no weight.
//!
//! # The two differences, and why neither is the obvious one
//!
//! **The scaling is not one of them.** `SeamlessM4TFeatureExtractor` multiplies the waveform by
//! `2^15` before the transform, because Kaldi's conventions are 16-bit integers; `infer_v2_5.py`
//! hands CAMPPlus a float waveform in `[-1, 1]` and does not. That looks like it matters and
//! almost never does: a scale factor is a constant offset once the mel energies are logged, and
//! *both* paths remove a constant offset afterwards -- one by subtracting each band's mean over
//! time, the other by standardizing each band. The one place it does matter is silence, where
//! `mel_floor` clamps before the log and a quiet enough signal lands on the floor rather than
//! being scaled. So [`Analysis::scale`] is spelled out for each rather than assumed away.
//!
//! **What actually differs is the last step.** w2v-bert's features are standardized per band --
//! mean and deviation over time, `ddof=1` -- and then **stacked in pairs**, so 80 bands at 100
//! frames a second become 160 numbers at 50. CAMPPlus's have their per-band mean subtracted and
//! nothing else, and stay 80 wide.
//!
//! # The window is Povey's, which is a Hann raised to 0.85
//!
//! Not a Hann, not a Hamming. Kaldi's default for a filterbank is `hann^0.85`, and it is
//! non-periodic -- the denominator is `n - 1` and not `n`. Both of those are the kind of thing
//! that is off by a hair everywhere and wrong nowhere in particular.

use std::collections::HashMap;
use std::f64::consts::PI;

use crate::audio::{
    apply_filterbank, conv1d, hann_window, magnitude, mel_filterbank, pad1d, stft, stft_basis,
    MelScale, Padding,
};
use crate::flint::{DType, Device, Extent, Graph, Ir, RunContext, Tensor, Value};
use crate::{Error, Result};

/// How a waveform is cut into frames and turned into log mel energies.
///
/// The defaults are Kaldi's, which is what both callers want; what they disagree about is
/// [`Analysis::scale`], and what they do afterwards.
#[derive(Clone, Copy, Debug)]
pub struct Analysis {
    pub rate: f64,
    /// 25 milliseconds at 16 kHz.
    pub frame_length: usize,
    /// 10 milliseconds at 16 kHz.
    pub hop: usize,
    /// The next power of two at or above [`Analysis::frame_length`], which is what
    /// `round_to_power_of_two` means. 512 for a 400-sample frame.
    pub fft_length: usize,
    pub bins: usize,
    pub low: f64,
    /// Zero means the Nyquist frequency, which is Kaldi's convention for "as high as it goes".
    pub high: f64,
    pub preemphasis: f64,
    /// What a mel energy is clamped to before the logarithm, so that silence is a large negative
    /// number rather than an infinite one.
    pub mel_floor: f64,
    /// What the waveform is multiplied by first. See the module note: this matters at the floor
    /// and nowhere else.
    pub scale: f64,
}

impl Analysis {
    /// What `SeamlessM4TFeatureExtractor` does, which is what [`w2v_bert`] is built on.
    pub fn seamless() -> Analysis {
        Analysis {
            rate: 16000.0,
            frame_length: 400,
            hop: 160,
            fft_length: 512,
            bins: 80,
            low: 20.0,
            high: 0.0,
            preemphasis: 0.97,
            mel_floor: 1.192_092_955_078_125e-07,
            scale: 32768.0,
        }
    }

    /// What `torchaudio.compliance.kaldi.fbank(dither=0)` does, which is what [`campplus`] is
    /// built on. The same analysis, handed a waveform nobody scaled.
    pub fn kaldi() -> Analysis {
        Analysis {
            scale: 1.0,
            ..Analysis::seamless()
        }
    }

    /// How many frames a waveform of this length yields, `snip_edges` being true: a frame that
    /// would run off the end is not taken, and nothing is padded.
    pub fn frames(&self, samples: usize) -> usize {
        if samples < self.frame_length {
            return 0;
        }

        1 + (samples - self.frame_length) / self.hop
    }

    /// How many frequency bins the filterbank is built over. One short of the transform's own
    /// count, because `SeamlessM4TFeatureExtractor` builds `(256, 80)` and pads a zero row --
    /// the Nyquist bin contributes to nothing.
    fn frequency_bins(&self) -> usize {
        self.fft_length / 2
    }

    fn nyquist(&self) -> f64 {
        match self.high {
            high if high > 0.0 => high,
            _ => self.rate / 2.0,
        }
    }
}

/// Kaldi's mel scale. Not HTK's (which uses 2595 and a base-ten logarithm) and not Slaney's.
pub fn hertz_to_mel(hertz: f64) -> f64 {
    1127.0 * (1.0 + hertz / 700.0).ln()
}

/// The inverse of [`hertz_to_mel`].
pub fn mel_to_hertz(mel: f64) -> f64 {
    700.0 * ((mel / 1127.0).exp() - 1.0)
}

/// Povey's window: a non-periodic Hann raised to the power 0.85.
pub fn povey_window(length: usize) -> Vec<f64> {
    (0..length)
        .map(|index| {
            let phase = 2.0 * std::f64::consts::PI * index as f64 / (length as f64 - 1.0);

            (0.5 - 0.5 * phase.cos()).powf(0.85)
        })
        .collect()
}

/// The triangular filterbank, `(frequency_bins, bins)` row-major.
///
/// **The triangles are built in mel space, not in hertz.** That is what `torchaudio` does and
/// what `triangularize_in_mel_space` asks `transformers` for, and it is not the same filterbank
/// as the usual one: the corners are evenly spaced in mel either way, but here the *slopes* are
/// linear in mel as well, so a filter is symmetric on the mel axis rather than on the hertz axis.
pub fn filterbank(analysis: &Analysis) -> Vec<f64> {
    let bins = analysis.bins;
    let frequency_bins = analysis.frequency_bins();

    // The corners: `bins + 2` points evenly spaced in mel, so every filter has a left foot, a
    // peak and a right foot, and neighbours share them.
    let low = hertz_to_mel(analysis.low);
    let high = hertz_to_mel(analysis.nyquist());
    let corners: Vec<f64> = (0..bins + 2)
        .map(|index| low + (high - low) * index as f64 / (bins + 1) as f64)
        .collect();

    // Where each transform bin sits, on the same axis.
    let width = analysis.rate / (frequency_bins as f64 * 2.0);
    let centres: Vec<f64> = (0..frequency_bins)
        .map(|index| hertz_to_mel(width * index as f64))
        .collect();

    let mut weights = vec![0.0f64; frequency_bins * bins];
    for (row, centre) in centres.iter().enumerate() {
        for filter in 0..bins {
            let (left, peak, right) = (corners[filter], corners[filter + 1], corners[filter + 2]);

            let rising = (centre - left) / (peak - left);
            let falling = (right - centre) / (right - peak);

            weights[row * bins + filter] = rising.min(falling).max(0.0);
        }
    }

    weights
}

/// An in-place radix-2 Cooley-Tukey transform of `real` and `imaginary`, whose length is a power
/// of two.
///
/// Written here rather than taken from [`crate::audio`] because that one is a matrix multiply
/// meant for a graph, and this is a few hundred frames on the host where `n log n` is free.
fn transform(real: &mut [f64], imaginary: &mut [f64]) {
    let n = real.len();
    debug_assert!(n.is_power_of_two());
    debug_assert_eq!(imaginary.len(), n);

    // Bit-reversal permutation.
    let mut target = 0;
    for source in 1..n {
        let mut bit = n >> 1;
        while target & bit != 0 {
            target ^= bit;
            bit >>= 1;
        }
        target |= bit;

        if source < target {
            real.swap(source, target);
            imaginary.swap(source, target);
        }
    }

    let mut span = 2;
    while span <= n {
        let angle = -2.0 * std::f64::consts::PI / span as f64;
        let (step_cos, step_sin) = (angle.cos(), angle.sin());

        for start in (0..n).step_by(span) {
            let (mut twiddle_real, mut twiddle_imaginary) = (1.0f64, 0.0f64);

            for offset in 0..span / 2 {
                let (here, there) = (start + offset, start + offset + span / 2);

                let product_real =
                    real[there] * twiddle_real - imaginary[there] * twiddle_imaginary;
                let product_imaginary =
                    real[there] * twiddle_imaginary + imaginary[there] * twiddle_real;

                real[there] = real[here] - product_real;
                imaginary[there] = imaginary[here] - product_imaginary;
                real[here] += product_real;
                imaginary[here] += product_imaginary;

                let next_real = twiddle_real * step_cos - twiddle_imaginary * step_sin;
                twiddle_imaginary = twiddle_real * step_sin + twiddle_imaginary * step_cos;
                twiddle_real = next_real;
            }
        }

        span <<= 1;
    }
}

/// The shared core: a waveform in, `(frames, bins)` log mel energies out.
///
/// Every frame has its mean removed, then a pre-emphasis filter run along it, then Povey's
/// window applied, and then the power spectrum through the filterbank and a logarithm. The order
/// is Kaldi's and is load-bearing -- removing the mean *after* pre-emphasis, for instance, would
/// remove a different number.
pub fn log_mel(wave: &[f32], analysis: &Analysis) -> (Vec<f32>, usize) {
    let frames = analysis.frames(wave.len());
    let bins = analysis.bins;
    let frequency_bins = analysis.frequency_bins();
    let window = povey_window(analysis.frame_length);
    let weights = filterbank(analysis);

    let mut out = vec![0.0f32; frames * bins];
    let mut real = vec![0.0f64; analysis.fft_length];
    let mut imaginary = vec![0.0f64; analysis.fft_length];

    for frame in 0..frames {
        let start = frame * analysis.hop;

        real[..analysis.frame_length]
            .iter_mut()
            .zip(&wave[start..start + analysis.frame_length])
            .for_each(|(slot, sample)| *slot = f64::from(*sample) * analysis.scale);
        real[analysis.frame_length..].fill(0.0);
        imaginary.fill(0.0);

        // The mean of the frame, gone.
        let mean = real[..analysis.frame_length].iter().sum::<f64>() / analysis.frame_length as f64;
        real[..analysis.frame_length]
            .iter_mut()
            .for_each(|sample| *sample -= mean);

        // Pre-emphasis, walked backwards so that each sample reads the one before it as it was.
        for index in (1..analysis.frame_length).rev() {
            real[index] -= analysis.preemphasis * real[index - 1];
        }
        real[0] *= 1.0 - analysis.preemphasis;

        real[..analysis.frame_length]
            .iter_mut()
            .zip(&window)
            .for_each(|(sample, weight)| *sample *= weight);

        transform(&mut real, &mut imaginary);

        // The power spectrum, through the triangles, floored, logged.
        for filter in 0..bins {
            let mut total = 0.0f64;
            for bin in 0..frequency_bins {
                let weight = weights[bin * bins + filter];
                if weight != 0.0 {
                    total += weight * (real[bin] * real[bin] + imaginary[bin] * imaginary[bin]);
                }
            }

            out[frame * bins + filter] = total.max(analysis.mel_floor).ln() as f32;
        }
    }

    (out, frames)
}

/// The features [`crate::indextts::w2v_bert`] reads: `(frames / 2, 2 * bins)`.
///
/// Standardized per band over time and then stacked in pairs, which is how 80 bands at a hundred
/// frames a second become 160 numbers at fifty. An odd frame at the end is dropped, because a
/// pair needs two -- where the extractor pads it with a zero frame and masks that pair out of
/// w2v-bert's attention. Twenty milliseconds at the edge, and masked either way.
///
/// The deviation is the sample one, dividing by `n - 1`. That is `torch.var`'s default and not
/// `numpy.var`'s, and `transformers` has a comment about it for the same reason this does.
pub fn w2v_bert(wave: &[f32]) -> (Vec<f32>, usize) {
    let analysis = Analysis::seamless();
    let (energies, frames) = log_mel(wave, &analysis);
    let bins = analysis.bins;

    let standardized = standardize(&energies, frames, bins, 1e-7);
    let pairs = frames / 2;

    let mut out = vec![0.0f32; pairs * 2 * bins];
    for pair in 0..pairs {
        let (first, second) = (2 * pair, 2 * pair + 1);

        out[pair * 2 * bins..pair * 2 * bins + bins]
            .copy_from_slice(&standardized[first * bins..first * bins + bins]);
        out[pair * 2 * bins + bins..(pair + 1) * 2 * bins]
            .copy_from_slice(&standardized[second * bins..second * bins + bins]);
    }

    (out, pairs)
}

/// The features [`crate::indextts::campplus`] reads: `(frames, bins)`, each band's mean over time removed.
///
/// No deviation, and no stacking. `infer_v2_5.py` writes this as `feat - feat.mean(dim=0)`, and
/// the comment beside it says it is a second filterbank energy feature -- which it is not, it is
/// the same one normalized differently.
pub fn campplus(wave: &[f32]) -> (Vec<f32>, usize) {
    let analysis = Analysis::kaldi();
    let (mut energies, frames) = log_mel(wave, &analysis);
    let bins = analysis.bins;

    for band in 0..bins {
        let mean = (0..frames)
            .map(|frame| f64::from(energies[frame * bins + band]))
            .sum::<f64>()
            / frames.max(1) as f64;

        for frame in 0..frames {
            energies[frame * bins + band] -= mean as f32;
        }
    }

    (energies, frames)
}

/// Each band to zero mean and unit deviation over time, the deviation being the sample one.
fn standardize(energies: &[f32], frames: usize, bins: usize, epsilon: f64) -> Vec<f32> {
    let mut out = energies.to_vec();

    for band in 0..bins {
        let mean = (0..frames)
            .map(|frame| f64::from(energies[frame * bins + band]))
            .sum::<f64>()
            / frames.max(1) as f64;

        // `n - 1`, which is what torch divides by and numpy does not.
        let spread = (0..frames)
            .map(|frame| {
                let centred = f64::from(energies[frame * bins + band]) - mean;

                centred * centred
            })
            .sum::<f64>()
            / (frames.max(2) - 1) as f64;

        let scale = (spread + epsilon).sqrt();
        for frame in 0..frames {
            out[frame * bins + band] =
                ((f64::from(energies[frame * bins + band]) - mean) / scale) as f32;
        }
    }

    out
}

/// What S2Mel conditions on and BigVGAN reads: the 22.05 kHz mel spectrogram, `(bins, frames)`.
///
/// This one is not Kaldi's. It is BigVGAN's `mel_spectrogram`, which `infer_v2_5.py` builds with
/// S2Mel's `spect_params` and calls `ref_mel`: the signal reflected by `(n_fft - hop) / 2` at each
/// end, a periodic Hann, the *magnitude* rather than the power -- `sqrt(re² + im² + 1e-9)` -- then
/// Slaney's filterbank as `librosa` builds it, and `ln(max(x, 1e-5))`.
///
/// The layout is band-major, `(80, frames)`, because that is what both of its readers want: S2Mel
/// holds it as the prompt, `(1, 80, T)`, and BigVGAN reads the same shape.
pub fn reference_mel(wave: &[f32]) -> Result<(Vec<f32>, usize)> {
    const RATE: u32 = 22050;
    const FFT: usize = 1024;
    const HOP: usize = 256;
    const BANDS: usize = 80;

    let pad = (FFT - HOP) / 2;
    let padded = reflect(wave, pad);
    let frames = match padded.len() >= FFT {
        true => 1 + (padded.len() - FFT) / HOP,
        false => 0,
    };

    let bins = FFT / 2 + 1;
    let weights = mel_filterbank(
        RATE,
        FFT,
        BANDS,
        0.0,
        f64::from(RATE) / 2.0,
        MelScale::Slaney,
        true,
    )?
    .to_vec_f32()?;

    // Periodic: `torch.hann_window`'s default, and the opposite of Povey's window above.
    let window: Vec<f64> = (0..FFT)
        .map(|n| 0.5 - 0.5 * (2.0 * std::f64::consts::PI * n as f64 / FFT as f64).cos())
        .collect();

    let mut out = vec![0.0f32; BANDS * frames];
    let mut magnitude = vec![0.0f64; bins];
    let mut real = vec![0.0f64; FFT];
    let mut imaginary = vec![0.0f64; FFT];

    for frame in 0..frames {
        let start = frame * HOP;
        for (index, slot) in real.iter_mut().enumerate() {
            *slot = f64::from(padded[start + index]) * window[index];
        }
        imaginary.fill(0.0);

        transform(&mut real, &mut imaginary);

        for (bin, slot) in magnitude.iter_mut().enumerate() {
            *slot = (real[bin] * real[bin] + imaginary[bin] * imaginary[bin] + 1e-9).sqrt();
        }

        for band in 0..BANDS {
            let row = &weights[band * bins..(band + 1) * bins];
            let total: f64 = row
                .iter()
                .zip(&magnitude)
                .map(|(weight, value)| f64::from(*weight) * value)
                .sum();

            out[band * frames + frame] = total.max(1e-5).ln() as f32;
        }
    }

    Ok((out, frames))
}

/// `wave` with `pad` samples mirrored onto each end, the edge sample itself not repeated -- what
/// `torch.nn.functional.pad(mode="reflect")` does.
fn reflect(wave: &[f32], pad: usize) -> Vec<f32> {
    let length = wave.len();
    let mut out = Vec::with_capacity(length + 2 * pad);

    // A signal shorter than the reflection is reflected as far as it goes and then repeated from
    // its edge, rather than read out of bounds. Only a recording under 17 ms reaches this.
    let at = |index: isize| -> f32 {
        let last = length as isize - 1;
        let mut index = index;
        if index < 0 {
            index = -index;
        }
        if index > last {
            index = 2 * last - index;
        }

        wave[index.clamp(0, last.max(0)) as usize]
    };

    for index in (1..=pad as isize).rev() {
        out.push(at(index));
    }
    out.extend_from_slice(wave);
    for index in 0..pad as isize {
        out.push(at(length as isize - 2 - index));
    }

    out
}

/// A waveform at `from` hertz, at `to` hertz: `torchaudio.functional.resample`'s windowed sinc.
///
/// The reference recording arrives at whatever rate it was recorded at and is needed at two
/// others -- 16 kHz for w2v-bert and CAMPPlus, 22.05 kHz for the mel -- and `infer_v2_5.py` gets
/// both with `torchaudio.transforms.Resample`, whose defaults are a Hann-windowed sinc six zero
/// crossings wide, rolled off to 99% of the lower Nyquist.
///
/// # Why not [`crate::audio::resample`]
///
/// That one runs in a graph and is the textbook shape: stuff zeros up to the common multiple,
/// low-pass, keep every `down`th sample. For 44.1 kHz to 16 kHz the common multiple is 160 times
/// the input, so fifteen seconds of speech becomes a hundred million samples through a filter
/// fourteen thousand taps long -- almost all of it computing outputs that are then thrown away.
/// This is the polyphase form instead, which computes only the samples that are kept: one short
/// dot product per output sample, against one of `to / gcd` precomputed kernels.
pub fn resample(wave: &[f32], from: u32, to: u32) -> Vec<f32> {
    if from == to || wave.is_empty() {
        return wave.to_vec();
    }

    const ZERO_CROSSINGS: f64 = 6.0;
    const ROLLOFF: f64 = 0.99;

    let divisor = greatest_common_divisor(from, to);
    let (orig, new) = ((from / divisor) as usize, (to / divisor) as usize);

    let base = orig.min(new) as f64 * ROLLOFF;
    let width = (ZERO_CROSSINGS * orig as f64 / base).ceil() as usize;
    let taps = 2 * width + orig;

    // One kernel per output phase, exactly as torchaudio lays them out: the phase's offset, then
    // the tap's position against the input rate, both in units of the cutoff.
    let mut kernels = vec![0.0f64; new * taps];
    for phase in 0..new {
        for tap in 0..taps {
            let position = -(phase as f64) / new as f64 + (tap as f64 - width as f64) / orig as f64;
            let t = (position * base).clamp(-ZERO_CROSSINGS, ZERO_CROSSINGS);

            let window = (t * std::f64::consts::PI / ZERO_CROSSINGS / 2.0)
                .cos()
                .powi(2);
            let t = t * std::f64::consts::PI;
            let sinc = match t == 0.0 {
                true => 1.0,
                false => t.sin() / t,
            };

            kernels[phase * taps + tap] = sinc * window * base / orig as f64;
        }
    }

    // Zeros ahead by `width` and behind by `width + orig`, which is what lets every output read a
    // whole kernel's worth of input.
    let mut padded = vec![0.0f64; width];
    padded.extend(wave.iter().map(|sample| f64::from(*sample)));
    padded.extend(std::iter::repeat_n(0.0, width + orig));

    let length = (new as u64 * wave.len() as u64).div_ceil(orig as u64) as usize;

    (0..length)
        .map(|index| {
            let (block, phase) = (index / new, index % new);
            let start = block * orig;
            let kernel = &kernels[phase * taps..(phase + 1) * taps];

            kernel
                .iter()
                .zip(&padded[start..start + taps])
                .map(|(weight, sample)| weight * sample)
                .sum::<f64>() as f32
        })
        .collect()
}

fn greatest_common_divisor(a: u32, b: u32) -> u32 {
    match b {
        0 => a,
        _ => greatest_common_divisor(b, a % b),
    }
}

// -------------------------------------------------------------------------------------------
// On the device
// -------------------------------------------------------------------------------------------

/// Kaldi's frame processing and transform as one `(512, 1, 400)` bank, the waveform unscaled. See
/// the module note.
pub fn kaldi_basis(analysis: &Analysis) -> Result<Tensor> {
    let length = analysis.frame_length;
    let bins = analysis.fft_length / 2;
    let window = povey_window(length);

    // `frame[m]` to `windowed[n]`: the mean removed, pre-emphasis walked backwards so each sample
    // reads its neighbour as it was, then the window.
    let centred = |n: usize, m: usize| -> f64 {
        let identity = if n == m { 1.0 } else { 0.0 };
        (identity - 1.0 / length as f64) * analysis.scale
    };
    let mut frame = vec![0.0f64; length * length];
    for n in 0..length {
        for m in 0..length {
            let emphasised = match n {
                0 => (1.0 - analysis.preemphasis) * centred(0, m),
                _ => centred(n, m) - analysis.preemphasis * centred(n - 1, m),
            };
            frame[n * length + m] = window[n] * emphasised;
        }
    }

    // Row `bin` of the transform times `frame`, a row of the frame matrix at a time so that each
    // angle is worked out once.
    let mut values = vec![0.0f32; 2 * bins * length];
    let mut real = vec![0.0f64; length];
    let mut imaginary = vec![0.0f64; length];
    for bin in 0..bins {
        real.fill(0.0);
        imaginary.fill(0.0);
        for n in 0..length {
            let angle = 2.0 * PI * (bin * n) as f64 / analysis.fft_length as f64;
            let (cos, sin) = (angle.cos(), angle.sin());
            let row = &frame[n * length..(n + 1) * length];
            for m in 0..length {
                real[m] += cos * row[m];
                imaginary[m] -= sin * row[m];
            }
        }
        for m in 0..length {
            values[bin * length + m] = real[m] as f32;
            values[(bins + bin) * length + m] = imaginary[m] as f32;
        }
    }

    Ok(Tensor::from_f32(
        &[2 * bins as i32, 1, length as i32],
        &values,
    )?)
}

/// What [`reference_mel`] is configured with: BigVGAN's `mel_spectrogram` at 22.05 kHz.
const REFERENCE_RATE: u32 = 22050;
const REFERENCE_FFT: usize = 1024;
const REFERENCE_HOP: i32 = 256;
const REFERENCE_BANDS: usize = 80;

/// `max(x, floor)`, as `floor + relu(x - floor)`; `floor` is a one-element constant.
fn at_least(g: &Graph, x: Value, floor: Value) -> Value {
    g.add(g.relu(g.sub(x, floor)), floor)
}

fn scalar(g: &Graph, value: f32, device: Device) -> Result<Value> {
    Ok(g.constant(Tensor::from_f32(&[1], &[value])?.to_device(device)?))
}

/// The three analyses as graphs on the device, their banks and filterbanks built once.
///
/// Each takes a waveform as the host has it -- the recording is read and resampled there -- and
/// gives back a tensor on the device, ready for the model that reads it: nothing but the samples
/// crosses the bus.
pub struct Spectrograms {
    device: Device,
    kaldi_basis: Tensor,
    kaldi_filterbank: Tensor,
    mel_basis: Tensor,
    mel_filterbank: Tensor,
}

impl Spectrograms {
    /// Every bank and filterbank, worked out on the host and moved to `device`.
    pub fn new(device: Device) -> Result<Spectrograms> {
        let on = |tensor: Tensor| -> Result<Tensor> { Ok(tensor.to_device(device)?) };

        let kaldi = Analysis::kaldi();
        let bins = kaldi.fft_length / 2;
        // `filterbank` is `(bins, bands)` with the Nyquist bin; the graph wants `(bands, bins)`
        // without it, which it never weighs.
        let weights = filterbank(&kaldi);
        let transposed: Vec<f32> = (0..kaldi.bins)
            .flat_map(|band| (0..bins).map(move |bin| (band, bin)))
            .map(|(band, bin)| weights[bin * kaldi.bins + band] as f32)
            .collect();

        // Periodic: `torch.hann_window`'s default, and the opposite of Povey's.
        let window = hann_window(REFERENCE_FFT);

        Ok(Spectrograms {
            device,
            kaldi_basis: on(kaldi_basis(&kaldi)?)?,
            kaldi_filterbank: on(Tensor::from_f32(
                &[kaldi.bins as i32, bins as i32],
                &transposed,
            )?)?,
            mel_basis: on(stft_basis(REFERENCE_FFT, &window)?)?,
            mel_filterbank: on(mel_filterbank(
                REFERENCE_RATE,
                REFERENCE_FFT,
                REFERENCE_BANDS,
                0.0,
                f64::from(REFERENCE_RATE) / 2.0,
                MelScale::Slaney,
                true,
            )?)?,
        })
    }

    fn upload(&self, wave: &[f32]) -> Result<Tensor> {
        Ok(Tensor::from_f32(&[1, 1, wave.len() as i32], wave)?.to_device(self.device)?)
    }

    /// Run a graph that reads no weights, only the wave it is handed.
    fn run(&self, g: &Graph, wave: &[f32]) -> Result<Tensor> {
        let nothing: HashMap<String, Tensor> = HashMap::new();
        let wave = self.upload(wave)?;
        let context = RunContext::new(&nothing).input("wave", &wave);
        Ir::compile(g)
            .run(&context)?
            .into_iter()
            .next()
            .map(|(_, tensor)| tensor)
            .ok_or_else(|| Error::model("a feature graph produced nothing"))
    }

    /// [`log_mel`] as nodes: the graph's `"wave"` `(1, 1, samples)` to Kaldi's log mel energies,
    /// frame-major, `(1, frames, 80)`. The scale is applied to the samples, as Kaldi applies it.
    fn kaldi_log_mel(&self, g: &Graph, analysis: &Analysis) -> Result<Value> {
        let bins = (analysis.fft_length / 2) as i32;
        let (dtype, device) = (DType::Float, self.device);

        let wave = match analysis.scale == 1.0 {
            true => g.input("wave"),
            false => g.mul_scalar(g.input("wave"), analysis.scale as f32),
        };
        let spectrum = conv1d(
            g,
            wave,
            g.constant(self.kaldi_basis.clone()),
            None,
            analysis.hop as i32,
            0,
            1,
            1,
            dtype,
            device,
        )?;
        let real = g.slice(spectrum, 1, 0, bins);
        let imaginary = g.slice(spectrum, 1, bins, 2 * bins);
        let power = g.add(g.square(real), g.square(imaginary));
        let mel = apply_filterbank(g, power, g.constant(self.kaldi_filterbank.clone()));
        let logged = g.log(at_least(
            g,
            mel,
            scalar(g, analysis.mel_floor as f32, device)?,
        ));

        Ok(g.contiguous(g.transpose(logged, 1, 2)))
    }

    fn frames(analysis: &Analysis, wave: &[f32]) -> Result<usize> {
        match analysis.frames(wave.len()) {
            0 => Err(Error::model(
                "the recording is too short to hear a voice in",
            )),
            frames => Ok(frames),
        }
    }

    /// [`campplus`] on the device: Kaldi's log mel energies of a 16 kHz recording, unscaled, with
    /// each band's mean over time removed, `(1, frames, 80)`.
    pub fn campplus(&self, wave: &[f32]) -> Result<Tensor> {
        let analysis = Analysis::kaldi();
        let frames = Self::frames(&analysis, wave)?;

        let g = Graph::new();
        let framewise = self.kaldi_log_mel(&g, &analysis)?;
        let mean = g.mul_scalar(g.sum(framewise, 1), 1.0 / frames as f32);
        let mean = g.view(mean, [analysis.bins as i32]);
        g.output("fbank", g.sub(framewise, mean));

        self.run(&g, wave)
    }

    /// [`w2v_bert`] on the device: Kaldi's log mel energies of a 16 kHz recording scaled by
    /// `2^15`, each band standardized over time with the sample deviation, then frames stacked in
    /// pairs, `(1, frames / 2, 160)`. An odd last frame counts towards the statistics and is then
    /// dropped, as the host's is.
    pub fn w2v_bert(&self, wave: &[f32]) -> Result<Tensor> {
        let analysis = Analysis::seamless();
        let frames = Self::frames(&analysis, wave)?;
        let pairs = (frames / 2) as i32;
        if pairs == 0 {
            return Err(Error::model(
                "the recording is too short to hear a voice in",
            ));
        }
        let bands = analysis.bins as i32;

        let g = Graph::new();
        let framewise = self.kaldi_log_mel(&g, &analysis)?;
        let mean = g.mul_scalar(g.sum(framewise, 1), 1.0 / frames as f32);
        let centred = g.sub(framewise, g.view(mean, [bands]));

        // `n - 1`, which is what torch divides by and numpy does not.
        let spread = g.mul_scalar(
            g.sum(g.square(centred), 1),
            1.0 / (frames.max(2) - 1) as f32,
        );
        let scale = g.sqrt(g.add(spread, scalar(&g, 1e-7, self.device)?));
        let standardized = g.div(centred, g.view(scale, [bands]));

        let kept = g.contiguous(g.slice(standardized, 1, 0, 2 * pairs));
        g.output(
            "features",
            g.view(
                kept,
                [Extent::At(1), Extent::At(pairs), Extent::At(2 * bands)],
            ),
        );

        self.run(&g, wave)
    }

    /// [`reference_mel`] on the device: BigVGAN's mel of a 22.05 kHz recording, `(1, 80, frames)`.
    ///
    /// Reflected by `(1024 - 256) / 2` at each end and not centred, a periodic Hann of 1024, the
    /// magnitude `sqrt(re^2 + im^2 + 1e-9)`, Slaney's filterbank to 11.025 kHz, `ln(max(x, 1e-5))`.
    pub fn reference_mel(&self, wave: &[f32]) -> Result<Tensor> {
        let pad = (REFERENCE_FFT as i32 - REFERENCE_HOP) / 2;
        if wave.len() <= pad as usize {
            return Err(Error::model(
                "the recording is too short to take a spectrogram of",
            ));
        }
        let (dtype, device) = (DType::Float, self.device);

        let g = Graph::new();
        let x = pad1d(
            &g,
            g.input("wave"),
            pad,
            pad,
            Padding::Reflect,
            dtype,
            device,
        )?;
        let (real, imaginary) = stft(
            &g,
            x,
            g.constant(self.mel_basis.clone()),
            REFERENCE_FFT as i32,
            REFERENCE_HOP,
            None,
            dtype,
            device,
        )?;
        let magnitude = magnitude(&g, real, imaginary, 1e-9, dtype, device)?;
        let mel = apply_filterbank(&g, magnitude, g.constant(self.mel_filterbank.clone()));
        g.output("mel", g.log(at_least(&g, mel, scalar(&g, 1e-5, device)?)));

        self.run(&g, wave)
    }
}
