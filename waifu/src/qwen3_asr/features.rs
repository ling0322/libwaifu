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

//! What Qwen3-ASR hears a recording as: transformers' `WhisperFeatureExtractor` with 128 bands,
//! on the device.
//!
//! The same spectrogram CosyVoice3's speech tokenizer reads, and computed the same way -- a
//! similar implementation lives in `cosyvoice3::features::Features::whisper_mel`, and a fix to
//! one likely belongs in the other too. It is copied rather than shared because the two models
//! are not the same model.

use std::collections::HashMap;

use crate::audio::{apply_filterbank, halves, hann_window, mel_filterbank, MelScale};
use crate::flint::{DType, Device, Extent, Graph, Ir, RunContext, Tensor, Value, Weights};
use crate::{Error, Result};

const FFT: usize = 400;
const HOP: i32 = 160;
const RATE: u32 = 16000;

/// The window and the filterbank, on the device.
pub struct Features {
    device: Device,
    bands: usize,
    window: Tensor,
    filterbank: Tensor,
}

/// `max(x, floor)`, as `floor + relu(x - floor)`, for a `floor` that is a value of the graph's.
fn at_least(g: &Graph, x: Value, floor: Value) -> Value {
    g.add(g.relu(g.sub(x, floor)), floor)
}

fn scalar(g: &Graph, value: f32, device: Device) -> Result<Value> {
    Ok(g.constant(Tensor::from_f32(&[1], &[value])?.to_device(device)?))
}

impl Features {
    pub fn new(bands: usize, device: Device) -> Result<Features> {
        Ok(Features {
            device,
            bands,
            window: Tensor::from_f32(&[FFT as i32], &hann_window(FFT))?.to_device(device)?,
            filterbank: mel_filterbank(RATE, FFT, bands, 0.0, 8000.0, MelScale::Slaney, true)?
                .to_device(device)?,
        })
    }

    /// How many frames `samples` samples at 16 kHz come to: one per hop, the last dropped.
    pub fn frames_of(samples: usize) -> usize {
        samples / HOP as usize
    }

    /// The log mel of a 16 kHz recording, `(1, bands, frames)`.
    ///
    /// `torch.stft` centred with reflection, a periodic Hann of 400, the power spectrum with its
    /// last frame dropped, Slaney's filterbank, `log10` floored at 1e-10, then everything more than
    /// eight below the loudest raised to it and the lot mapped by `(x + 4) / 4`.
    pub fn mel(&self, wave: &[f32]) -> Result<Tensor> {
        if wave.len() < 2 * HOP as usize || wave.len() <= FFT / 2 {
            return Err(Error::model(
                "the recording is too short to hear anything in",
            ));
        }
        let device = self.device;

        let g = Graph::new();
        let spectrum = g.stft(
            g.input("wave"),
            g.constant(self.window.clone()),
            FFT as i32,
            HOP,
            true,
        );
        let (real, imaginary) = halves(&g, spectrum, FFT as i32);
        let power = g.add(g.square(real), g.square(imaginary));
        let power = g.slice(power, 2, 0, -1);
        let mel = apply_filterbank(&g, power, g.constant(self.filterbank.clone()));
        let logged = g.mul_scalar(
            g.log(g.clamp(mel, 1e-10, f32::INFINITY)),
            1.0 / std::f32::consts::LN_10,
        );

        let flat = g.view(logged, [Extent::At(1), Extent::prod(logged, 0, 3)]);
        let floor = g.sub(g.max(flat, -1), scalar(&g, 8.0, device)?);
        let raised = at_least(&g, logged, floor);
        g.output(
            "mel",
            g.mul_scalar(g.add(raised, scalar(&g, 4.0, device)?), 0.25),
        );

        let wave = Tensor::from_f32(&[1, 1, wave.len() as i32], wave)?.to_device(device)?;
        let nothing = Weights::from_tensors(HashMap::new(), device)?;
        let context = RunContext::new(&nothing).input("wave", &wave);
        let mel = Ir::compile(&g)
            .run(&context)?
            .into_iter()
            .next()
            .map(|(_, tensor)| tensor)
            .ok_or_else(|| Error::model("the feature graph produced nothing"))?;
        debug_assert_eq!(mel.shape_at(1).ok(), Some(self.bands as i32));
        Ok(mel.cast(DType::Float)?)
    }
}
