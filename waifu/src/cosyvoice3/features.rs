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

//! The two spectrograms CosyVoice3 reads a recording as, both computed on the host.
//!
//! - [`whisper_mel`] is what the speech tokenizer reads: Whisper's `log_mel_spectrogram` with 128
//!   bands, of the 16 kHz recording.
//! - [`speech_mel`] is what the flow continues and the vocoder reads: Matcha's `mel_spectrogram`
//!   at 24 kHz, 1920 wide with a hop of 480 -- fifty frames a second, two per speech token.
//!
//! Neither window is a power of two, so the transform here is a direct DFT against a table of the
//! window's own sinusoids rather than [`crate::indextts::features`]'s radix-2 one. For a recording
//! of the few seconds a prompt is, that is a fraction of a second on the host.

use crate::audio::{hann_window, mel_filterbank, MelScale};
use crate::Result;

/// The magnitudes-or-powers of every frame's DFT, `frames` rows of `n / 2 + 1`.
struct Dft {
    n: usize,
    cos: Vec<f64>,
    sin: Vec<f64>,
    window: Vec<f64>,
}

impl Dft {
    fn new(n: usize) -> Dft {
        let angle = |k: usize| 2.0 * std::f64::consts::PI * k as f64 / n as f64;
        Dft {
            n,
            cos: (0..n).map(|k| angle(k).cos()).collect(),
            sin: (0..n).map(|k| angle(k).sin()).collect(),
            window: hann_window(n).into_iter().map(f64::from).collect(),
        }
    }

    /// `|X[k]|^2` of the windowed frame starting at `signal[start]`.
    fn power(&self, signal: &[f64], start: usize, out: &mut [f64]) {
        let n = self.n;
        let frame: Vec<f64> = (0..n).map(|i| signal[start + i] * self.window[i]).collect();
        for (k, slot) in out.iter_mut().enumerate() {
            let (mut re, mut im) = (0.0, 0.0);
            let mut at = 0usize;
            for value in &frame {
                re += value * self.cos[at];
                im -= value * self.sin[at];
                at += k;
                if at >= n {
                    at -= n;
                }
            }
            *slot = re * re + im * im;
        }
    }
}

/// `signal` with `pad` samples mirrored onto each end, the edge not repeated: `mode="reflect"`.
fn reflect(signal: &[f32], pad: usize) -> Vec<f64> {
    let length = signal.len() as isize;
    let at = |mut index: isize| -> f64 {
        let last = length - 1;
        if index < 0 {
            index = -index;
        }
        if index > last {
            index = 2 * last - index;
        }
        f64::from(signal[index.clamp(0, last.max(0)) as usize])
    };

    (-(pad as isize)..length + pad as isize).map(at).collect()
}

/// Whisper's `log_mel_spectrogram(audio, n_mels=128)` of a 16 kHz recording: `(128, frames)`,
/// band-major, and how many frames that is -- one per 160 samples.
///
/// `torch.stft` centred with reflection, a periodic Hann of 400, the power spectrum with its last
/// frame dropped, Slaney's filterbank, `log10` floored at 1e-10, then everything more than eight
/// below the loudest raised to it and the lot mapped by `(x + 4) / 4`.
pub fn whisper_mel(wave: &[f32]) -> Result<(Vec<f32>, usize)> {
    const FFT: usize = 400;
    const HOP: usize = 160;
    const BANDS: usize = 128;

    let bins = FFT / 2 + 1;
    let padded = reflect(wave, FFT / 2);
    let frames = wave.len() / HOP;
    let filters =
        mel_filterbank(16000, FFT, BANDS, 0.0, 8000.0, MelScale::Slaney, true)?.to_vec_f32()?;

    let dft = Dft::new(FFT);
    let mut power = vec![0.0f64; bins];
    let mut out = vec![0.0f64; BANDS * frames];
    for frame in 0..frames {
        dft.power(&padded, frame * HOP, &mut power);
        for band in 0..BANDS {
            let row = &filters[band * bins..(band + 1) * bins];
            let total: f64 = row.iter().zip(&power).map(|(w, p)| f64::from(*w) * p).sum();
            out[band * frames + frame] = total.max(1e-10).log10();
        }
    }

    let loudest = out.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    Ok((
        out.into_iter()
            .map(|x| ((x.max(loudest - 8.0) + 4.0) / 4.0) as f32)
            .collect(),
        frames,
    ))
}

/// Matcha's `mel_spectrogram` as `cosyvoice3.yaml` configures it, of a 24 kHz recording:
/// `(80, frames)`, band-major, fifty frames a second.
///
/// Reflected by `(1920 - 480) / 2` at each end and not centred, a periodic Hann of 1920, the
/// magnitude `sqrt(re^2 + im^2 + 1e-9)`, Slaney's filterbank to 12 kHz, and `ln(max(x, 1e-5))`.
pub fn speech_mel(wave: &[f32]) -> Result<(Vec<f32>, usize)> {
    const FFT: usize = 1920;
    const HOP: usize = 480;
    const BANDS: usize = 80;

    let bins = FFT / 2 + 1;
    let padded = reflect(wave, (FFT - HOP) / 2);
    let frames = match padded.len() >= FFT {
        true => 1 + (padded.len() - FFT) / HOP,
        false => 0,
    };
    let filters =
        mel_filterbank(24000, FFT, BANDS, 0.0, 12000.0, MelScale::Slaney, true)?.to_vec_f32()?;

    let dft = Dft::new(FFT);
    let mut power = vec![0.0f64; bins];
    let mut out = vec![0.0f32; BANDS * frames];
    for frame in 0..frames {
        dft.power(&padded, frame * HOP, &mut power);
        let magnitude: Vec<f64> = power.iter().map(|p| (p + 1e-9).sqrt()).collect();
        for band in 0..BANDS {
            let row = &filters[band * bins..(band + 1) * bins];
            let total: f64 = row
                .iter()
                .zip(&magnitude)
                .map(|(w, m)| f64::from(*w) * m)
                .sum();
            out[band * frames + frame] = total.max(1e-5).ln() as f32;
        }
    }

    Ok((out, frames))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_counts_follow_the_hops() {
        let second_16k = vec![0.01f32; 16000];
        assert_eq!(whisper_mel(&second_16k).unwrap().1, 100);

        let second_24k = vec![0.01f32; 24000];
        assert_eq!(speech_mel(&second_24k).unwrap().1, 50);
    }

    #[test]
    fn a_tone_lands_in_the_band_that_holds_it() {
        // 1 kHz at 24 kHz: the loudest band is the one whose filter peaks nearest 1 kHz.
        let wave: Vec<f32> = (0..24000)
            .map(|i| (2.0 * std::f32::consts::PI * 1000.0 * i as f32 / 24000.0).sin() * 0.5)
            .collect();
        let (mel, frames) = speech_mel(&wave).unwrap();
        let frame = frames / 2;
        let loudest = (0..80)
            .max_by(|a, b| {
                mel[a * frames + frame]
                    .partial_cmp(&mel[b * frames + frame])
                    .unwrap()
            })
            .unwrap();
        // 1 kHz is mel 15 on Slaney's scale, and 12 kHz is mel 51.1; 82 edges across that are
        // 0.631 apart, so band `b`, centred on edge `b + 1`, peaks at 1 kHz for b = 15 / 0.631 - 1.
        assert_eq!(loudest, 23);
    }
}
