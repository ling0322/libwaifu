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

//! HiFT, CosyVoice3's vocoder: the 24 kHz mel in, the waveform out.
//!
//! `CausalHiFTGenerator` -- a neural source-filter network in front of an ISTFTNet -- run once over
//! a whole sentence, the way `token2wav` runs it with `finalize=True`. Three parts, and only the
//! two that are convolution stacks are graphs:
//!
//! 1. **The pitch.** [`Hift::pitch`]: five causal convolutions with ELUs and a linear read-out,
//!    one fundamental frequency per mel frame. Upstream runs this in float64 because a streaming
//!    reading re-derives it chunk by chunk; a whole sentence at once does not need that, and this
//!    runs it at the model's precision.
//! 2. **The source**, on the host: [`source`] turns the pitch into nine harmonics of a sine at
//!    24 kHz, noise where the frame is unvoiced, merged into one excitation by a linear layer and
//!    a `tanh`. Then [`stft`] of that, sixteen points with a hop of four.
//! 3. **The filter.** [`Hift::decode`]: the mel, upsampled 8, 5 and 3 times by causal
//!    convolutions, the source's spectrum mixed in at each rate, residual blocks of snakes, and a
//!    last convolution to eighteen channels -- a log magnitude and a phase for each of nine bins
//!    -- which [`istft`] turns into samples.
//!
//! # The noise is the caller's
//!
//! Upstream's `SineGen2` draws, at construction, a starting phase for each harmonic and a buffer
//! of uniform noise seven million samples long, from whatever state torch's generator is in by
//! then. Neither is a weight. Here they are arguments, [`Noise`], so that a test can hand in what
//! the reference used and the pipeline can draw its own from the reading's seed.
//!
//! One consequence worth knowing: the starting phase is added to the first *sample* of each
//! harmonic, and the linear resampling that follows reads samples 239 and 240 of every 480. So it
//! never reaches the output, in upstream or here -- it is carried only so that the arithmetic is
//! upstream's rather than a simplification of it.

use std::fmt;
use std::rc::Rc;

use crate::audio::{pad1d, snake, Padding};
use crate::flint::{
    check_parameters, DType, Device, Graph, Ir, ParamSource, RunContext, Tensor, Value,
};
use crate::layers::Linear;
use crate::{Error, Result};

pub const RATE: u32 = 24000;
/// Samples per mel frame: 8 * 5 * 3 upsampling, then the inverse transform's hop of four.
pub const HOP: usize = 480;
const HARMONICS: usize = 9;
const FFT: usize = 16;
const FFT_HOP: usize = 4;
const BINS: usize = FFT / 2 + 1;
const SINE_AMP: f32 = 0.1;
const NOISE_STD: f32 = 0.003;
const VOICED_THRESHOLD: f32 = 10.0;
const AUDIO_LIMIT: f32 = 0.99;

const BASE: i32 = 512;
const UPSAMPLES: [(i32, i32); 3] = [(8, 16), (5, 11), (3, 7)];
const KERNELS: [i32; 3] = [3, 7, 11];
const SOURCE_KERNELS: [i32; 3] = [7, 7, 11];
const DILATIONS: [i32; 3] = [1, 3, 5];

/// The random numbers the source is built from; see the module note.
pub struct Noise {
    /// One starting phase per harmonic, the first zero: `(9)`.
    pub start: Vec<f32>,
    /// Uniform on `[0, 1)`, nine per sample: `(samples, 9)`.
    pub sine: Vec<f32>,
}

impl Noise {
    /// Drawn from `uniform`, for `samples` samples, in the order upstream's buffers are laid out.
    pub fn draw(samples: usize, uniform: &mut dyn FnMut() -> f32) -> Noise {
        let mut start: Vec<f32> = (0..HARMONICS).map(|_| uniform()).collect();
        start[0] = 0.0;
        Noise {
            start,
            sine: (0..samples * HARMONICS).map(|_| uniform()).collect(),
        }
    }
}

fn constant(g: &Graph, value: f32, dtype: DType, device: Device) -> Result<Value> {
    Ok(g.constant(
        Tensor::from_f32(&[1], &[value])?
            .to_device(device)?
            .cast(dtype)?,
    ))
}

fn leaky_relu(g: &Graph, x: Value, slope: f32) -> Value {
    g.add(g.mul_scalar(g.relu(x), 1.0 - slope), g.mul_scalar(x, slope))
}

/// `x` for `x > 0`, `e^x - 1` otherwise.
fn elu(g: &Graph, x: Value, one: Value) -> Value {
    let negative = g.neg(g.relu(g.neg(x)));
    g.add(g.relu(x), g.sub(g.exp(negative), one))
}

/// A convolution padded `left` and `right` with zeros first, which is how every causal one here is
/// written: `CausalConv1d` pads one side by its `causal_padding` and convolves with none.
#[allow(clippy::too_many_arguments)]
fn conv(
    g: &Graph,
    x: Value,
    in_channels: i32,
    out_channels: i32,
    kernel: i32,
    stride: i32,
    dilation: i32,
    (left, right): (i32, i32),
    dtype: DType,
    device: Device,
) -> Result<Value> {
    let padded = pad1d(g, x, left, right, Padding::Zero, dtype, device)?;
    Ok(g.conv1d(
        padded,
        g.load("weight", &[out_channels, in_channels, kernel]),
        Some(g.load("bias", &[out_channels])),
        stride,
        0,
        dilation,
        1,
    ))
}

/// A causal residual block: for each dilation, snake, convolve, snake, convolve, add.
fn resblock(
    g: &Graph,
    x: Value,
    channels: i32,
    kernel: i32,
    dtype: DType,
    device: Device,
) -> Result<Value> {
    let mut x = x;
    for (index, dilation) in DILATIONS.iter().enumerate() {
        let name = index.to_string();
        let alpha = |group: &str| g.subgraph(group).subgraph(&name).load("alpha", &[channels]);

        let xt = snake(g, x, alpha("activations1"), None, 1e-9, dtype, device)?;
        let xt = conv(
            &g.subgraph("convs1").subgraph(&name),
            xt,
            channels,
            channels,
            kernel,
            1,
            *dilation,
            ((kernel - 1) * dilation, 0),
            dtype,
            device,
        )?;
        let xt = snake(g, xt, alpha("activations2"), None, 1e-9, dtype, device)?;
        let xt = conv(
            &g.subgraph("convs2").subgraph(&name),
            xt,
            channels,
            channels,
            kernel,
            1,
            1,
            (kernel - 1, 0),
            dtype,
            device,
        )?;
        x = g.add(xt, x);
    }
    Ok(x)
}

/// The pitch predictor: `(1, 80, T)` mel in, `(1, T)` hertz out.
fn write_pitch(g: &Graph, dtype: DType, device: Device) -> Result<()> {
    let one = constant(g, 1.0, dtype, device)?;
    let condnet = g.subgraph("f0_predictor").subgraph("condnet");
    let mel = g.cast(g.input("mel"), dtype);

    // The first reads three frames ahead; the other four only behind.
    let mut x = conv(
        &condnet.subgraph("0"),
        mel,
        80,
        BASE,
        4,
        1,
        1,
        (0, 3),
        dtype,
        device,
    )?;
    x = elu(g, x, one);
    for layer in [2, 4, 6, 8] {
        x = conv(
            &condnet.subgraph(&layer.to_string()),
            x,
            BASE,
            BASE,
            3,
            1,
            1,
            (2, 0),
            dtype,
            device,
        )?;
        x = elu(g, x, one);
    }

    let framewise = g.contiguous(g.transpose(x, 1, 2));
    let f0 = Linear::graph(
        &g.subgraph("f0_predictor").subgraph("classifier"),
        framewise,
        BASE,
        1,
        true,
    );
    g.output("f0", g.cast(g.abs(f0), DType::Float));
    Ok(())
}

/// The filter: `(1, 80, T)` mel and `(1, 18, F)` source spectrum in, `(1, 18, F)` out.
fn write_decode(g: &Graph, dtype: DType, device: Device) -> Result<()> {
    let mel = g.cast(g.input("mel"), dtype);
    let source = g.cast(g.input("source"), dtype);
    let spectrum = (2 * BINS) as i32;

    let mut x = conv(
        &g.subgraph("conv_pre"),
        mel,
        80,
        BASE,
        5,
        1,
        1,
        (0, 4),
        dtype,
        device,
    )?;

    // How far each stage's source is taken down to meet it: 15, 3, then 1.
    let downs = [15, 3, 1];
    let mut channels = BASE;
    for (index, (rate, kernel)) in UPSAMPLES.iter().enumerate() {
        let out_channels = channels / 2;
        let name = index.to_string();

        x = leaky_relu(g, x, 0.1);
        // `CausalConv1dUpsample`: nearest-neighbour by `rate`, then a convolution padded behind.
        let upsampled = {
            let stretched = g.unsqueeze(x, 3);
            let mut repeated = stretched;
            for _ in 1..*rate {
                repeated = g.cat(repeated, stretched, 3);
            }
            g.view(
                repeated,
                [
                    crate::flint::Extent::At(1),
                    crate::flint::Extent::At(channels),
                    crate::flint::Extent::prod(repeated, 2, 4),
                ],
            )
        };
        x = conv(
            &g.subgraph("ups").subgraph(&name),
            upsampled,
            channels,
            out_channels,
            *kernel,
            1,
            1,
            (kernel - 1, 0),
            dtype,
            device,
        )?;

        if index == UPSAMPLES.len() - 1 {
            // `ReflectionPad1d((1, 0))`: the second sample, copied in front of the first.
            x = g.cat(g.contiguous(g.slice(x, 2, 1, 2)), x, 2);
        }

        let down = downs[index];
        let si = match down {
            1 => conv(
                &g.subgraph("source_downs").subgraph(&name),
                source,
                spectrum,
                out_channels,
                1,
                1,
                1,
                (0, 0),
                dtype,
                device,
            )?,
            _ => conv(
                &g.subgraph("source_downs").subgraph(&name),
                source,
                spectrum,
                out_channels,
                2 * down,
                down,
                1,
                (down - 1, 0),
                dtype,
                device,
            )?,
        };
        let si = resblock(
            &g.subgraph("source_resblocks").subgraph(&name),
            si,
            out_channels,
            SOURCE_KERNELS[index],
            dtype,
            device,
        )?;
        x = g.add(x, si);

        let mut total: Option<Value> = None;
        for (offset, kernel) in KERNELS.iter().enumerate() {
            let block = resblock(
                &g.subgraph("resblocks")
                    .subgraph(&(index * KERNELS.len() + offset).to_string()),
                x,
                out_channels,
                *kernel,
                dtype,
                device,
            )?;
            total = Some(match total {
                None => block,
                Some(sum) => g.add(sum, block),
            });
        }
        x = g.mul_scalar(total.expect("three kernels"), 1.0 / KERNELS.len() as f32);
        channels = out_channels;
    }

    x = leaky_relu(g, x, 0.01);
    let x = conv(
        &g.subgraph("conv_post"),
        x,
        channels,
        spectrum,
        7,
        1,
        1,
        (6, 0),
        dtype,
        device,
    )?;
    g.output("spectrum", g.cast(x, DType::Float));
    Ok(())
}

/// A periodic Hann window, `scipy.signal.get_window("hann", n, fftbins=True)`.
fn window() -> Vec<f64> {
    (0..FFT)
        .map(|n| 0.5 - 0.5 * (2.0 * std::f64::consts::PI * n as f64 / FFT as f64).cos())
        .collect()
}

/// `torch.stft(x, 16, 4, 16, hann, center=True)` -- reflected by eight each side -- as `(18, F)`:
/// nine real rows then nine imaginary ones, `F = len / 4 + 1`.
pub fn stft(signal: &[f32]) -> (Vec<f32>, usize) {
    let pad = FFT / 2;
    let length = signal.len() as isize;
    let at = |mut index: isize| -> f64 {
        if index < 0 {
            index = -index;
        }
        if index >= length {
            index = 2 * (length - 1) - index;
        }
        f64::from(signal[index as usize])
    };
    let frames = signal.len() / FFT_HOP + 1;
    let window = window();

    let mut out = vec![0.0f32; 2 * BINS * frames];
    for frame in 0..frames {
        let start = (frame * FFT_HOP) as isize - pad as isize;
        for bin in 0..BINS {
            let (mut re, mut im) = (0.0, 0.0);
            for (n, weight) in window.iter().enumerate() {
                let value = at(start + n as isize) * weight;
                let angle = 2.0 * std::f64::consts::PI * (bin * n) as f64 / FFT as f64;
                re += value * angle.cos();
                im -= value * angle.sin();
            }
            out[bin * frames + frame] = re as f32;
            out[(BINS + bin) * frames + frame] = im as f32;
        }
    }
    (out, frames)
}

/// `torch.istft` of `(magnitude, phase)` as the decoder hands them over -- `exp` of the first nine
/// rows, clipped at a hundred, and `sin` of the other nine -- with `center=True`: `(F - 1) * 4`
/// samples, clamped to 0.99.
pub fn istft(spectrum: &[f32], frames: usize) -> Vec<f32> {
    let window = window();
    let length = (frames - 1) * FFT_HOP + FFT;
    let mut sum = vec![0.0f64; length];
    let mut envelope = vec![0.0f64; length];

    let mut frame_samples = [0.0f64; FFT];
    for frame in 0..frames {
        // The full spectrum from the half, conjugate symmetry filling in the rest; the imaginary
        // parts of the bins that are their own conjugate are ignored, as `irfft` ignores them.
        for (n, sample) in frame_samples.iter_mut().enumerate() {
            let mut total = 0.0;
            for bin in 0..BINS {
                let magnitude = f64::from(spectrum[bin * frames + frame]).exp().min(100.0);
                let phase = f64::from(spectrum[(BINS + bin) * frames + frame]).sin();
                let (re, im) = (magnitude * phase.cos(), magnitude * phase.sin());
                let angle = 2.0 * std::f64::consts::PI * (bin * n) as f64 / FFT as f64;
                let term = re * angle.cos() - im * angle.sin();
                total += match bin == 0 || bin == FFT / 2 {
                    true => re * angle.cos(),
                    false => 2.0 * term,
                };
            }
            *sample = total / FFT as f64;
        }

        for n in 0..FFT {
            sum[frame * FFT_HOP + n] += frame_samples[n] * window[n];
            envelope[frame * FFT_HOP + n] += window[n] * window[n];
        }
    }

    let pad = FFT / 2;
    (pad..length - pad)
        .map(|index| {
            let value = sum[index] / envelope[index].max(1e-11);
            (value as f32).clamp(-AUDIO_LIMIT, AUDIO_LIMIT)
        })
        .collect()
}

/// `SourceModuleHnNSF` over the pitch: one excitation sample per output sample, `T * 480` of them.
///
/// `linear` is `l_linear`'s nine weights and its bias.
pub fn source(f0: &[f32], noise: &Noise, linear: &[f32], bias: f32) -> Result<Vec<f32>> {
    let frames = f0.len();
    let samples = frames * HOP;
    if noise.sine.len() < samples * HARMONICS {
        return Err(Error::model(
            "the vocoder was handed less noise than it has samples",
        ));
    }

    // The phase advance per sample of each harmonic, in turns, before and after the linear
    // resampling to one value per frame. The pitch is constant across a frame's 480 samples, so
    // only the first sample -- which alone has the starting phase added -- differs from the rest.
    let advance = |frame: usize, harmonic: usize, first: bool| -> f32 {
        let hertz = f0[frame] * (harmonic + 1) as f32;
        let mut turns = (hertz / RATE as f32).rem_euclid(1.0);
        if first && frame == 0 {
            turns += noise.start[harmonic];
        }
        turns
    };
    let sample = |index: usize, harmonic: usize| -> f32 {
        advance(index / HOP, harmonic, index.is_multiple_of(HOP))
    };

    // `interpolate(scale_factor=1/480, mode="linear")`: frame `i` reads halfway between samples
    // `480 i + 239` and `480 i + 240`.
    let mut phase = vec![0.0f32; frames * HARMONICS];
    let mut running = [0.0f32; HARMONICS];
    for frame in 0..frames {
        for harmonic in 0..HARMONICS {
            let left = sample(frame * HOP + HOP / 2 - 1, harmonic);
            let right = sample(frame * HOP + HOP / 2, harmonic);
            running[harmonic] += 0.5 * left + 0.5 * right;
            phase[frame * HARMONICS + harmonic] =
                running[harmonic] * 2.0 * std::f32::consts::PI * HOP as f32;
        }
    }

    let mut out = Vec::with_capacity(samples);
    for index in 0..samples {
        let frame = index / HOP;
        let voiced = f0[frame] > VOICED_THRESHOLD;
        let amplitude = match voiced {
            true => NOISE_STD,
            false => SINE_AMP / 3.0,
        };

        let mut merged = bias;
        for harmonic in 0..HARMONICS {
            let sine = phase[frame * HARMONICS + harmonic].sin() * SINE_AMP;
            let noise = amplitude * noise.sine[index * HARMONICS + harmonic];
            let wave = match voiced {
                true => sine + noise,
                false => noise,
            };
            merged += linear[harmonic] * wave;
        }
        out.push(merged.tanh());
    }

    Ok(out)
}

/// HiFT with its weights behind it.
pub struct Hift {
    pitch: Ir,
    decode: Ir,
    linear: Vec<f32>,
    bias: f32,
    weights: Rc<dyn ParamSource>,
    device: Device,
}

impl fmt::Debug for Hift {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Hift")
            .field("device", &self.device)
            .finish_non_exhaustive()
    }
}

impl Hift {
    pub fn build(
        name: &str,
        weights: &Rc<dyn ParamSource>,
        dtype: DType,
        device: Device,
    ) -> Result<Hift> {
        let pitch = Graph::new();
        write_pitch(&pitch.subgraph(name), dtype, device)?;
        check_parameters(&pitch, weights.as_ref())?;

        let decode = Graph::new();
        write_decode(&decode.subgraph(name), dtype, device)?;
        check_parameters(&decode, weights.as_ref())?;

        // `l_linear`, read once onto the host where the source is built.
        let read = Graph::new();
        let sub = read
            .subgraph(name)
            .subgraph("m_source")
            .subgraph("l_linear");
        read.output("weight", sub.load("weight", &[1, HARMONICS as i32]));
        read.output("bias", sub.load("bias", &[1]));
        let outputs = Ir::compile(&read).run(&RunContext::new(weights.as_ref()))?;
        let host = |index: usize| -> Result<Vec<f32>> {
            Ok(outputs[index]
                .1
                .to_device(Device::Cpu)?
                .cast(DType::Float)?
                .to_vec_f32()?)
        };
        let linear = host(0)?;
        let bias = host(1)?[0];

        Ok(Hift {
            pitch: Ir::compile(&pitch),
            decode: Ir::compile(&decode),
            linear,
            bias,
            weights: Rc::clone(weights),
            device,
        })
    }

    fn upload_mel(&self, mel: &[f32], frames: usize) -> Result<Tensor> {
        Ok(Tensor::from_f32(&[1, 80, frames as i32], mel)?.to_device(self.device)?)
    }

    /// One fundamental frequency per frame of `mel` `(80, frames)`, band-major.
    pub fn pitch(&self, mel: &[f32], frames: usize) -> Result<Vec<f32>> {
        let mel = self.upload_mel(mel, frames)?;
        let outputs = self
            .pitch
            .run(&RunContext::new(&*self.weights).input("mel", &mel))?;
        Ok(outputs[0].1.to_device(Device::Cpu)?.to_vec_f32()?)
    }

    /// The excitation for `f0`, `f0.len() * 480` samples.
    pub fn source(&self, f0: &[f32], noise: &Noise) -> Result<Vec<f32>> {
        source(f0, noise, &self.linear, self.bias)
    }

    /// The filter over `mel` `(80, frames)` and the excitation, to samples.
    pub fn decode(&self, mel: &[f32], frames: usize, excitation: &[f32]) -> Result<Vec<f32>> {
        let (spectrum, spectrum_frames) = stft(excitation);
        let mel = self.upload_mel(mel, frames)?;
        let source = Tensor::from_f32(&[1, 2 * BINS as i32, spectrum_frames as i32], &spectrum)?
            .to_device(self.device)?;
        let outputs = self.decode.run(
            &RunContext::new(&*self.weights)
                .input("mel", &mel)
                .input("source", &source),
        )?;
        let out = outputs[0].1.to_device(Device::Cpu)?.to_vec_f32()?;
        let frames_out = out.len() / (2 * BINS);
        Ok(istft(&out, frames_out))
    }

    /// The whole vocoder: `mel` `(80, frames)` band-major to `frames * 480` samples.
    pub fn forward(&self, mel: &[f32], frames: usize, noise: &Noise) -> Result<Vec<f32>> {
        let f0 = self.pitch(mel, frames)?;
        let excitation = self.source(&f0, noise)?;
        self.decode(mel, frames, &excitation)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_unvoiced_source_is_noise_alone() {
        let f0 = vec![0.0f32; 2];
        let noise = Noise {
            start: vec![0.0; HARMONICS],
            sine: vec![0.5; 2 * HOP * HARMONICS],
        };
        let linear = vec![1.0f32; HARMONICS];
        let out = source(&f0, &noise, &linear, 0.0).unwrap();
        let want = (9.0 * SINE_AMP / 3.0 * 0.5).tanh();
        assert!(out.iter().all(|x| (x - want).abs() < 1e-6));
    }
}
