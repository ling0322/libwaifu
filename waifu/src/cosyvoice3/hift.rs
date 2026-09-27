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
//! a whole sentence, the way `token2wav` runs it with `finalize=True`. Three graphs, run one after
//! the other on the device, with nothing coming back to the host until the samples do:
//!
//! 1. **The pitch.** [`Hift::pitch`]: five causal convolutions with ELUs and a linear read-out,
//!    one fundamental frequency per mel frame. Upstream runs this in float64 because a streaming
//!    reading re-derives it chunk by chunk; a whole sentence at once does not need that, and this
//!    runs it at the model's precision.
//! 2. **The source.** [`Hift::source`]: `SourceModuleHnNSF` over the pitch -- nine harmonics of a
//!    sine at 24 kHz, noise where the frame is unvoiced, merged into one excitation by a linear
//!    layer and a `tanh`.
//! 3. **The filter.** [`Hift::decode`]: a sixteen-point transform of the excitation; the mel,
//!    upsampled 8, 5 and 3 times by causal convolutions, the excitation's spectrum mixed in at each
//!    rate, residual blocks of snakes, and a last convolution to eighteen channels -- a log
//!    magnitude and a phase for each of nine bins -- which the inverse transform turns into samples.
//!
//! # The source, as operators
//!
//! `SineGen2` works at the audio rate, but everything in it is constant across a frame's 480
//! samples, so this works per frame and repeats each frame's values 480 times at the end:
//!
//! - the phase advance of harmonic `h` in a frame is `f0 * (h + 1) / 24000` turns -- the pitch
//!   times a row of nine constants, which is a matrix product;
//! - upstream downsamples that to one value a frame and `cumsum`s it, then scales by `2 pi * 480`,
//!   and so does this;
//! - `f0 > 10` is a step, written `1 - relu(1 - relu(f0 - 10) * 1e6)`: exact except within a
//!   millionth of a hertz above ten;
//! - `% 1` is left out. Adding whole turns to a frame's advance adds whole multiples of
//!   `2 pi * 480` to every later phase, which a sine does not see, and for speech the advance is
//!   under one turn anyway.
//!
//! # The noise is the caller's
//!
//! Upstream's `SineGen2` draws, at construction, a starting phase for each harmonic and a buffer of
//! uniform noise seven million samples long, from whatever state torch's generator is in by then.
//! Neither is a weight. Here the noise is an argument -- a `(1, samples, 9)` tensor on the device --
//! so that a test can hand in what the reference used and the pipeline can draw its own.
//!
//! The starting phase is not taken at all. Upstream adds it to the first *sample* of each harmonic,
//! and the linear resampling that follows reads samples 239 and 240 of every 480, so it never
//! reaches the output there either.

use std::fmt;
use std::rc::Rc;

use crate::audio::{
    conv_transpose1d, hann_window, istft, istft_basis, pad1d, snake, stft, stft_basis, Padding,
};
use crate::flint::{
    check_parameters, DType, Device, Extent, Graph, Ir, ParamSource, RunContext, Tensor, Value,
};
use crate::layers::Linear;
use crate::{Error, Result};

pub const RATE: u32 = 24000;
/// Samples per mel frame: 8 * 5 * 3 upsampling, then the inverse transform's hop of four.
pub const HOP: usize = 480;
/// The fundamental and its eight overtones.
pub const HARMONICS: usize = 9;
const FFT: usize = 16;
const FFT_HOP: i32 = 4;
const BINS: usize = FFT / 2 + 1;
const SINE_AMP: f32 = 0.1;
const NOISE_STD: f32 = 0.003;
const VOICED_THRESHOLD: f32 = 10.0;
const AUDIO_LIMIT: f32 = 0.99;
/// The largest magnitude the inverse transform is handed, as upstream clips it.
const MAGNITUDE_LIMIT: f32 = 100.0;

const BASE: i32 = 512;
const UPSAMPLES: [(i32, i32); 3] = [(8, 16), (5, 11), (3, 7)];
const KERNELS: [i32; 3] = [3, 7, 11];
const SOURCE_KERNELS: [i32; 3] = [7, 7, 11];
const DILATIONS: [i32; 3] = [1, 3, 5];

fn tensor_on(shape: &[i32], values: &[f32], dtype: DType, device: Device) -> Result<Tensor> {
    Ok(Tensor::from_f32(shape, values)?
        .to_device(device)?
        .cast(dtype)?)
}

fn constant(g: &Graph, value: f32, dtype: DType, device: Device) -> Result<Value> {
    Ok(g.constant(tensor_on(&[1], &[value], dtype, device)?))
}

fn leaky_relu(g: &Graph, x: Value, slope: f32) -> Value {
    g.add(g.mul_scalar(g.relu(x), 1.0 - slope), g.mul_scalar(x, slope))
}

/// `x` for `x > 0`, `e^x - 1` otherwise.
fn elu(g: &Graph, x: Value, one: Value) -> Value {
    let negative = g.neg(g.relu(g.neg(x)));
    g.add(g.relu(x), g.sub(g.exp(negative), one))
}

/// `min(x, limit)`, as `limit - relu(limit - x)`. `limit` is a one-element constant, and a binary
/// operation broadcasts only its right operand, so the subtraction is written as a negation added.
fn at_most(g: &Graph, x: Value, limit: Value) -> Value {
    g.add(g.neg(g.relu(g.add(g.neg(x), limit))), limit)
}

/// `max(x, floor)`, as `floor + relu(x - floor)`.
fn at_least(g: &Graph, x: Value, floor: Value) -> Value {
    g.add(g.relu(g.sub(x, floor)), floor)
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
    g.output(
        "f0",
        g.cast(
            g.view(g.abs(f0), [Extent::At(1), Extent::of(f0, 1)]),
            DType::Float,
        ),
    );
    Ok(())
}

/// `(frames, width)`, each row repeated [`HOP`] times: `(frames * 480, width)`. A zero tensor
/// `(480, frames, width)` plus the rows -- which broadcast over its leading axis -- is every row 480
/// times over, and turning that to `(frames, 480, width)` puts each frame's copies together.
fn per_sample(g: &Graph, x: Value, width: i32, dtype: DType, device: Device) -> Value {
    let zeros = g.zeros(
        [Extent::At(HOP as i32), Extent::of(x, 0), Extent::At(width)],
        dtype,
        device,
    );
    let repeated = g.contiguous(g.transpose(g.add(zeros, x), 0, 1));
    g.view(repeated, [Extent::prod(repeated, 0, 2), Extent::At(width)])
}

/// The source: `f0` `(1, T)` and uniform `noise` `(1, T * 480, 9)` in, the excitation
/// `(1, 1, T * 480)` out. See the module note for how each of upstream's steps is written.
fn write_source(g: &Graph, dtype: DType, device: Device) -> Result<()> {
    let one = constant(g, 1.0, dtype, device)?;
    let width = HARMONICS as i32;

    let f0 = g.cast(g.input("f0"), dtype);
    let f0 = g.view(f0, [Extent::of(f0, 1), Extent::At(1)]);

    // Turns per sample of each harmonic: the pitch times (h + 1) / 24000, as a `(T, 1)` by
    // `(1, 9)` product.
    let per_harmonic: Vec<f32> = (0..HARMONICS)
        .map(|h| (h + 1) as f32 / RATE as f32)
        .collect();
    let harmonics = g.constant(tensor_on(&[1, width], &per_harmonic, dtype, device)?);
    let advance = g.matmul(f0, harmonics);

    // `cumsum(rad) * 2 pi`, then `* 480` as the nearest-neighbour upsampling scales it.
    let phase = g.mul_scalar(
        g.mul_scalar(g.cumsum(advance, 0), 2.0 * std::f32::consts::PI),
        HOP as f32,
    );
    let sines = g.mul_scalar(g.sin(phase), SINE_AMP);

    // Voiced where the pitch is above ten hertz, as a step, then one per harmonic.
    let above = g.relu(g.sub(f0, constant(g, VOICED_THRESHOLD, dtype, device)?));
    let voiced = at_most(g, g.mul_scalar(above, 1e6), one);
    let ones = g.constant(tensor_on(&[1, width], &[1.0; HARMONICS], dtype, device)?);
    let voiced = g.matmul(voiced, ones);

    // The sine where voiced, and noise everywhere: `NOISE_STD` of it where voiced and a third of
    // `SINE_AMP` where not.
    let sines = g.mul(sines, voiced);
    let amplitude = g.add(
        g.mul_scalar(voiced, NOISE_STD - SINE_AMP / 3.0),
        constant(g, SINE_AMP / 3.0, dtype, device)?,
    );

    let noise = g.cast(g.input("noise"), dtype);
    let noise = g.view(noise, [Extent::of(noise, 1), Extent::At(width)]);
    let waves = g.add(
        per_sample(g, sines, width, dtype, device),
        g.mul(per_sample(g, amplitude, width, dtype, device), noise),
    );

    let merged = g.tanh(Linear::graph(
        &g.subgraph("m_source").subgraph("l_linear"),
        waves,
        width,
        1,
        true,
    ));
    g.output(
        "excitation",
        g.cast(
            g.view(
                merged,
                [Extent::At(1), Extent::At(1), Extent::of(merged, 0)],
            ),
            DType::Float,
        ),
    );
    Ok(())
}

/// The filter: `(1, 80, T)` mel and the `(1, 1, T * 480)` excitation in, `(T * 480)` samples out.
fn write_decode(g: &Graph, dtype: DType, device: Device) -> Result<()> {
    let mel = g.cast(g.input("mel"), dtype);
    let excitation = g.cast(g.input("excitation"), dtype);
    let spectrum = (2 * BINS) as i32;
    let window = hann_window(FFT);

    // `torch.stft(s, 16, 4, 16, hann)`: centred, reflected, nine bins real then nine imaginary.
    let basis = g.constant(stft_basis(FFT, &window)?.to_device(device)?.cast(dtype)?);
    let (real, imaginary) = stft(
        g,
        excitation,
        basis,
        FFT as i32,
        FFT_HOP,
        Some(Padding::Reflect),
        dtype,
        device,
    )?;
    let source = g.cat(real, imaginary, 1);

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
                    Extent::At(1),
                    Extent::At(channels),
                    Extent::prod(repeated, 2, 4),
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

    // `_istft`: `exp` of the first nine channels, clipped at a hundred, is the magnitude, and `sin`
    // of the other nine is the phase.
    let magnitude = at_most(
        g,
        g.exp(g.slice(x, 1, 0, BINS as i32)),
        constant(g, MAGNITUDE_LIMIT, dtype, device)?,
    );
    let phase = g.sin(g.slice(x, 1, BINS as i32, spectrum));
    let real = g.mul(magnitude, g.cos(phase));
    let imaginary = g.mul(magnitude, g.sin(phase));

    // The overlap of the squared window, for however many frames there are: a row of ones as long
    // as the frames, transposed-convolved by the window's square at the hop.
    let one = constant(g, 1.0, dtype, device)?;
    let ones = g.add(g.mul_scalar(g.slice(real, 1, 0, 1), 0.0), one);
    let squared: Vec<f32> = window.iter().map(|w| w * w).collect();
    let squared = g.constant(tensor_on(&[1, 1, FFT as i32], &squared, dtype, device)?);
    let envelope = conv_transpose1d(
        g, ones, squared, None, 1, 1, FFT as i32, FFT_HOP, 0, dtype, device,
    )?;
    let envelope = at_least(g, envelope, constant(g, 1e-11, dtype, device)?);

    let inverse = g.constant(istft_basis(FFT, &window)?.to_device(device)?.cast(dtype)?);
    let wave = istft(
        g, real, imaginary, inverse, envelope, FFT as i32, FFT_HOP, true, dtype, device,
    )?;

    // `torch.clamp(x, -0.99, 0.99)`.
    let limit = constant(g, AUDIO_LIMIT, dtype, device)?;
    let wave = at_most(g, wave, limit);
    let wave = at_least(g, wave, constant(g, -AUDIO_LIMIT, dtype, device)?);
    g.output(
        "wave",
        g.cast(g.view(wave, [Extent::prod(wave, 0, 3)]), DType::Float),
    );
    Ok(())
}

/// HiFT with its weights behind it.
pub struct Hift {
    pitch: Ir,
    source: Ir,
    decode: Ir,
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
        let compile = |write: fn(&Graph, DType, Device) -> Result<()>| -> Result<Ir> {
            let g = Graph::new();
            write(&g.subgraph(name), dtype, device)?;
            check_parameters(&g, weights.as_ref())?;
            Ok(Ir::compile(&g))
        };

        Ok(Hift {
            pitch: compile(write_pitch)?,
            source: compile(write_source)?,
            decode: compile(write_decode)?,
            weights: Rc::clone(weights),
            device,
        })
    }

    pub fn device(&self) -> Device {
        self.device
    }

    fn one(&self, ir: &Ir, run: RunContext<'_>) -> Result<Tensor> {
        ir.run(&run)?
            .into_iter()
            .next()
            .map(|(_, tensor)| tensor)
            .ok_or_else(|| Error::model("a vocoder graph produced nothing"))
    }

    /// One fundamental frequency per frame of `mel` `(1, 80, T)`: `(1, T)` hertz.
    pub fn pitch(&self, mel: &Tensor) -> Result<Tensor> {
        self.one(
            &self.pitch,
            RunContext::new(&*self.weights).input("mel", mel),
        )
    }

    /// The excitation for `f0` `(1, T)`, `(1, 1, T * 480)`, from `noise` `(1, T * 480, 9)` uniform
    /// on `[0, 1)`.
    pub fn source(&self, f0: &Tensor, noise: &Tensor) -> Result<Tensor> {
        let samples = f0.shape_at(1)? as i64 * HOP as i64;
        if noise.shape_at(1)? as i64 != samples || noise.shape_at(2)? != HARMONICS as i32 {
            return Err(Error::model(format!(
                "the vocoder wants ({samples}, {HARMONICS}) of noise and was handed {:?}",
                noise.shape()
            )));
        }
        self.one(
            &self.source,
            RunContext::new(&*self.weights)
                .input("f0", f0)
                .input("noise", noise),
        )
    }

    /// The filter over `mel` `(1, 80, T)` and the excitation `(1, 1, T * 480)`: `(T * 480)` samples.
    pub fn decode(&self, mel: &Tensor, excitation: &Tensor) -> Result<Tensor> {
        self.one(
            &self.decode,
            RunContext::new(&*self.weights)
                .input("mel", mel)
                .input("excitation", excitation),
        )
    }

    /// The whole vocoder: `mel` `(1, 80, T)` to `(T * 480)` samples, on the device.
    pub fn forward(&self, mel: &Tensor, noise: &Tensor) -> Result<Tensor> {
        let f0 = self.pitch(mel)?;
        let excitation = self.source(&f0, noise)?;
        self.decode(mel, &excitation)
    }
}
