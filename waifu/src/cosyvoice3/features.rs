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

//! The three spectrograms CosyVoice3 reads a recording as, all computed on the device.
//!
//! - [`Features::whisper_mel`] is what the speech tokenizer reads: Whisper's
//!   `log_mel_spectrogram` with 128 bands, of the 16 kHz recording.
//! - [`Features::speech_mel`] is what the flow continues: Matcha's `mel_spectrogram` at 24 kHz,
//!   1920 wide with a hop of 480 -- fifty frames a second, two per speech token.
//! - [`Features::kaldi_fbank`] is what CAMPPlus reads: `torchaudio.compliance.kaldi.fbank` of the
//!   16 kHz recording, each band's mean over time removed. It is the very feature IndexTTS-2.5
//!   hands the same CAMPPlus, so it is IndexTTS's [`Spectrograms::campplus`], whose module note
//!   says how Kaldi's frame processing becomes one convolution.
//!
//! Each is a strided convolution by a bank of windowed sinusoids -- [`crate::audio::stft`] -- and
//! then a filterbank and a logarithm, as a graph. The banks and filterbanks depend on nothing but
//! the transform, so they are built once, when the model is read, and kept on the device.

use std::collections::HashMap;

use crate::audio::{
    apply_filterbank, hann_window, magnitude, mel_filterbank, pad1d, stft, stft_basis, MelScale,
    Padding,
};
use crate::flint::{DType, Device, Extent, Graph, Ir, RunContext, Tensor, Value};
use crate::indextts::features::Spectrograms;
use crate::{Error, Result};

/// What Matcha's `mel_spectrogram` is configured with in `cosyvoice3.yaml`.
const SPEECH_FFT: usize = 1920;
const SPEECH_HOP: i32 = 480;
const SPEECH_BANDS: usize = 80;

/// Whisper's.
const WHISPER_FFT: usize = 400;
const WHISPER_HOP: i32 = 160;
const WHISPER_BANDS: usize = 128;

/// The three transforms' banks and filterbanks, on the device.
pub struct Features {
    device: Device,
    speech_basis: Tensor,
    speech_filterbank: Tensor,
    whisper_basis: Tensor,
    whisper_filterbank: Tensor,
    kaldi: Spectrograms,
}

/// `max(x, floor)`, as `floor + relu(x - floor)`; `floor` is a one-element constant.
fn at_least(g: &Graph, x: Value, floor: Value) -> Value {
    g.add(g.relu(g.sub(x, floor)), floor)
}

fn scalar(g: &Graph, value: f32, device: Device) -> Result<Value> {
    Ok(g.constant(Tensor::from_f32(&[1], &[value])?.to_device(device)?))
}

impl Features {
    /// Every bank and filterbank, worked out on the host once and moved to `device`.
    pub fn new(device: Device) -> Result<Features> {
        let on = |tensor: Tensor| -> Result<Tensor> { Ok(tensor.to_device(device)?) };

        let speech_window = hann_window(SPEECH_FFT);
        let whisper_window = hann_window(WHISPER_FFT);

        Ok(Features {
            device,
            speech_basis: on(stft_basis(SPEECH_FFT, &speech_window)?)?,
            speech_filterbank: on(mel_filterbank(
                24000,
                SPEECH_FFT,
                SPEECH_BANDS,
                0.0,
                12000.0,
                MelScale::Slaney,
                true,
            )?)?,
            whisper_basis: on(stft_basis(WHISPER_FFT, &whisper_window)?)?,
            whisper_filterbank: on(mel_filterbank(
                16000,
                WHISPER_FFT,
                WHISPER_BANDS,
                0.0,
                8000.0,
                MelScale::Slaney,
                true,
            )?)?,
            kaldi: Spectrograms::new(device)?,
        })
    }

    fn upload(&self, wave: &[f32]) -> Result<Tensor> {
        Ok(Tensor::from_f32(&[1, 1, wave.len() as i32], wave)?.to_device(self.device)?)
    }

    /// Run a graph that reads no weights, only the wave it is handed.
    fn run(&self, g: &Graph, wave: &Tensor) -> Result<Tensor> {
        let nothing: HashMap<String, Tensor> = HashMap::new();
        let context = RunContext::new(&nothing).input("wave", wave);
        Ir::compile(g)
            .run(&context)?
            .into_iter()
            .next()
            .map(|(_, tensor)| tensor)
            .ok_or_else(|| Error::model("a feature graph produced nothing"))
    }

    /// Matcha's `mel_spectrogram` of a 24 kHz recording: `(1, 80, frames)`, fifty frames a second.
    ///
    /// Reflected by `(1920 - 480) / 2` at each end and not centred, a periodic Hann of 1920, the
    /// magnitude `sqrt(re^2 + im^2 + 1e-9)`, Slaney's filterbank to 12 kHz, and `ln(max(x, 1e-5))`.
    pub fn speech_mel(&self, wave: &[f32]) -> Result<Tensor> {
        let pad = ((SPEECH_FFT as i32) - SPEECH_HOP) / 2;
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
            g.constant(self.speech_basis.clone()),
            SPEECH_FFT as i32,
            SPEECH_HOP,
            None,
            dtype,
            device,
        )?;
        let magnitude = magnitude(&g, real, imaginary, 1e-9, dtype, device)?;
        let mel = apply_filterbank(&g, magnitude, g.constant(self.speech_filterbank.clone()));
        g.output("mel", g.log(at_least(&g, mel, scalar(&g, 1e-5, device)?)));

        self.run(&g, &self.upload(wave)?)
    }

    /// Whisper's `log_mel_spectrogram(audio, n_mels=128)` of a 16 kHz recording: `(1, 128,
    /// frames)`, one frame per 160 samples.
    ///
    /// `torch.stft` centred with reflection, a periodic Hann of 400, the power spectrum with its
    /// last frame dropped, Slaney's filterbank, `log10` floored at 1e-10, then everything more than
    /// eight below the loudest raised to it and the lot mapped by `(x + 4) / 4`.
    pub fn whisper_mel(&self, wave: &[f32]) -> Result<Tensor> {
        if wave.len() < 2 * WHISPER_HOP as usize || wave.len() <= WHISPER_FFT / 2 {
            return Err(Error::model(
                "the recording is too short to hear a voice in",
            ));
        }
        let (dtype, device) = (DType::Float, self.device);

        let g = Graph::new();
        let (real, imaginary) = stft(
            &g,
            g.input("wave"),
            g.constant(self.whisper_basis.clone()),
            WHISPER_FFT as i32,
            WHISPER_HOP,
            Some(Padding::Reflect),
            dtype,
            device,
        )?;
        let power = g.add(g.square(real), g.square(imaginary));
        let power = g.slice(power, 2, 0, -1);
        let mel = apply_filterbank(&g, power, g.constant(self.whisper_filterbank.clone()));
        let logged = g.mul_scalar(
            g.log(at_least(&g, mel, scalar(&g, 1e-10, device)?)),
            1.0 / std::f32::consts::LN_10,
        );

        // The loudest value in the whole spectrogram, as a one-element value, and everything
        // brought up to eight below it.
        let flat = g.view(logged, [Extent::At(1), Extent::prod(logged, 0, 3)]);
        let floor = g.sub(g.max(flat, -1), scalar(&g, 8.0, device)?);
        let raised = at_least(&g, logged, floor);
        g.output(
            "mel",
            g.mul_scalar(g.add(raised, scalar(&g, 4.0, device)?), 0.25),
        );

        self.run(&g, &self.upload(wave)?)
    }

    /// `kaldi.fbank(num_mel_bins=80, dither=0)` of a 16 kHz recording with each band's mean over
    /// time removed, as CosyVoice's frontend hands it to CAMPPlus: `(1, frames, 80)`. IndexTTS's
    /// frontend hands its CAMPPlus the same thing -- see [`Spectrograms::campplus`].
    pub fn kaldi_fbank(&self, wave: &[f32]) -> Result<Tensor> {
        self.kaldi.campplus(wave)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn features() -> Features {
        Features::new(Device::Cpu).unwrap()
    }

    #[test]
    fn frame_counts_follow_the_hops() {
        let features = features();
        let second_16k = vec![0.01f32; 16000];
        assert_eq!(
            features.whisper_mel(&second_16k).unwrap().shape(),
            vec![1, 128, 100]
        );
        assert_eq!(
            features.kaldi_fbank(&second_16k).unwrap().shape(),
            vec![1, 98, 80]
        );

        let second_24k = vec![0.01f32; 24000];
        assert_eq!(
            features.speech_mel(&second_24k).unwrap().shape(),
            vec![1, 80, 50]
        );
    }

    #[test]
    fn a_tone_lands_in_the_band_that_holds_it() {
        // 1 kHz at 24 kHz. 1 kHz is mel 15 on Slaney's scale, and 12 kHz is mel 51.1; 82 edges
        // across that are 0.631 apart, so band `b`, centred on edge `b + 1`, peaks at 1 kHz for
        // b = 15 / 0.631 - 1.
        let wave: Vec<f32> = (0..24000)
            .map(|i| (2.0 * std::f32::consts::PI * 1000.0 * i as f32 / 24000.0).sin() * 0.5)
            .collect();
        let mel = features().speech_mel(&wave).unwrap();
        let frames = mel.shape_at(2).unwrap() as usize;
        let values = mel.to_vec_f32().unwrap();
        let frame = frames / 2;
        let loudest = (0..80)
            .max_by(|a, b| {
                values[a * frames + frame]
                    .partial_cmp(&values[b * frames + frame])
                    .unwrap()
            })
            .unwrap();
        assert_eq!(loudest, 23);
    }

    #[test]
    fn the_kaldi_bank_is_the_host_filterbank() {
        // The one-matrix frame processing against IndexTTS's host fbank, which walks a frame
        // sample by sample and was checked against torchaudio's.
        let wave: Vec<f32> = (0..16000)
            .map(|i| ((i as f32 * 0.013).sin() + 0.3 * (i as f32 * 0.37).cos()) * 0.2)
            .collect();
        let got = features().kaldi_fbank(&wave).unwrap().to_vec_f32().unwrap();
        let (want, frames) = crate::indextts::features::campplus(&wave);
        assert_eq!(got.len(), frames * 80);
        let worst = got
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(worst < 1e-3, "worst difference {worst}");
    }
}
