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

//! CAMPPlus: filterbank energies of someone speaking in, one vector of who they are out.
//!
//! This is the speaker encoder IndexTTS-2.5 conditions on. It takes 80-band Kaldi filterbank
//! energies, mean-normalized over time, and produces a 192-dimensional embedding -- the thing
//! that tells the rest of the pipeline whose voice to use. It is 3D-Speaker's model under Apache
//! 2.0, published as `funasr/campplus` and shared by several speech systems.
//!
//! Two halves. A small 2-D convolutional front end walks down the frequency axis, halving it
//! three times, and then folds what is left of frequency into the channels -- so a spectrogram
//! becomes a sequence. Then a densely connected time-delay network reads that sequence, and a
//! statistics pooling turns however many frames there were into a fixed pair of moments.
//!
//! # What this needs that was not here
//!
//! Nothing, in the end, but two of the pieces are worth pointing at because neither is obvious.
//!
//! **Batch normalization is not an operator.** In `eval` a `BatchNorm` is a fixed per-channel
//! affine and nothing more, so `tools/campplus_exporter.py` folds each one to the `scale` and
//! `shift` that describe it, and [`batch_norm`] is a multiply and an add. The two vectors are
//! stored shaped `(C, 1)` -- or `(C, 1, 1)` for the 2-D ones -- because that is what broadcasts
//! against `(N, C, L)` without a transpose.
//!
//! **A stride down one axis only.** Every downsampling in the front end is `stride=(2, 1)`:
//! halve frequency, leave time alone. `conv2d` here takes one stride and applies it to both, so
//! [`halve_frequency`] convolves at stride one and then keeps every second row, which is the same
//! arithmetic at twice the cost on an axis that is at most eighty long. The rows it keeps are the
//! ones a strided convolution would have centred on, which is what makes the two equal rather
//! than merely similar.

use crate::audio::conv1d;
use crate::flint::{DType, Device, Extent, Graph, Value};
use crate::Result;

/// The widths of one CAMPPlus. The counts -- three blocks, twelve then twenty-four then sixteen
/// layers deep -- are the architecture's and are not configurable, because they are hard-coded in
/// the reference too.
#[derive(Clone, Copy, Debug)]
pub struct Config {
    /// Filterbank bands in. Divisible by eight, since the front end halves frequency three times.
    pub feat_dim: i32,
    /// The embedding out. IndexTTS-2.5 asks for 192, not the reference class's default of 512.
    pub embedding_size: i32,
    /// Channels each dense layer adds to the running width.
    pub growth_rate: i32,
    /// The bottleneck inside a dense layer, as a multiple of `growth_rate`.
    pub bn_size: i32,
    /// What the first time-delay layer widens to.
    pub init_channels: i32,
}

impl Config {
    /// What IndexTTS-2.5 builds: `CAMPPlus(feat_dim=80, embedding_size=192)`.
    pub fn indextts() -> Config {
        Config {
            feat_dim: 80,
            embedding_size: 192,
            growth_rate: 32,
            bn_size: 4,
            init_channels: 128,
        }
    }

    fn bn_channels(&self) -> i32 {
        self.bn_size * self.growth_rate
    }
}

/// Channels the front end uses throughout, which the reference does not make configurable.
const FRONT_CHANNELS: i32 = 32;

/// The three dense blocks: how deep, what kernel, what dilation.
const BLOCKS: [(i32, i32, i32); 3] = [(12, 3, 1), (24, 3, 2), (16, 3, 2)];

/// How many frames one segment of [`CAMLayer`](segment_pooling)'s attention covers.
const SEGMENT: i32 = 100;

/// A batch normalization that has stopped normalizing: `x * scale + shift`, per channel.
///
/// `spatial` is how many axes follow the channel, which is what the two vectors were shaped for
/// -- one for `(N, C, L)`, two for `(N, C, H, W)`. See the module note.
#[track_caller]
pub fn batch_norm(g: &Graph, x: Value, channels: i32, spatial: usize) -> Value {
    let mut shape = vec![channels];
    shape.extend(std::iter::repeat_n(1, spatial));

    let scale = g.load("scale", &shape);
    let shift = g.load("shift", &shape);

    g.add(g.mul(x, scale), shift)
}

/// Keep every second row of the frequency axis of `x` `(N, C, F, T)`, `F` even.
///
/// This is the second half of a `stride=(2, 1)` convolution, the first half being an ordinary
/// stride-one one. Viewing `F` as `(F / 2, 2)` puts rows `0, 2, 4, ...` at index zero of the new
/// axis, and those are exactly the rows a strided convolution would have produced.
#[track_caller]
fn halve_frequency(g: &Graph, x: Value, channels: i32, freq: i32, time: Extent) -> Value {
    assert!(freq % 2 == 0, "the frequency axis has to halve evenly");

    let split = g.view(
        x,
        [
            Extent::of(x, 0),
            Extent::At(channels),
            Extent::At(freq / 2),
            Extent::At(2),
            time,
        ],
    );
    let kept = g.slice(split, 3, 0, 1);

    g.view(
        g.contiguous(kept),
        [
            Extent::of(x, 0),
            Extent::At(channels),
            Extent::At(freq / 2),
            time,
        ],
    )
}

/// One 2-D convolution of the front end, with the frequency stride done afterwards.
///
/// `stride` is the frequency stride; time is never strided here. Every one of these is bias-free
/// in the reference, the bias having been folded into the normalization that follows.
#[track_caller]
#[allow(clippy::too_many_arguments)]
fn front_conv(
    g: &Graph,
    x: Value,
    in_channels: i32,
    out_channels: i32,
    kernel: i32,
    padding: i32,
    stride: i32,
    freq: i32,
    time: Extent,
) -> Value {
    let weight = g.load("weight", &[out_channels, in_channels, kernel, kernel]);
    let y = g.conv2d(x, weight, None, 1, padding, 1, 1);

    match stride {
        1 => y,
        _ => halve_frequency(g, y, out_channels, freq, time),
    }
}

/// A residual block of the front end: two convolutions, and a shortcut that has to be projected
/// when the block changes shape.
#[track_caller]
fn res_block(g: &Graph, x: Value, channels: i32, stride: i32, freq: i32, time: Extent) -> Value {
    let out_freq = freq / stride;

    let y = front_conv(
        &g.subgraph("conv1"),
        x,
        channels,
        channels,
        3,
        1,
        stride,
        freq,
        time,
    );
    let y = g.relu(batch_norm(&g.subgraph("bn1"), y, channels, 2));

    let y = front_conv(
        &g.subgraph("conv2"),
        y,
        channels,
        channels,
        3,
        1,
        1,
        out_freq,
        time,
    );
    let y = batch_norm(&g.subgraph("bn2"), y, channels, 2);

    // The reference builds a shortcut only when the shape changes, and here the channels never
    // do, so a stride of one means the identity and no weights to ask for.
    let shortcut = match stride {
        1 => x,
        _ => {
            let sub = g.subgraph("shortcut");
            let projected = front_conv(
                &sub.subgraph("0"),
                x,
                channels,
                channels,
                1,
                0,
                stride,
                freq,
                time,
            );
            batch_norm(&sub.subgraph("1"), projected, channels, 2)
        }
    };

    g.relu(g.add(y, shortcut))
}

/// The front end: `(N, F, T)` in, `(N, 32 * F / 8, T)` out.
///
/// Frequency is halved three times -- twice by the residual layers and once by the convolution
/// after them -- and what is left of it is folded into the channels, which is what turns a
/// spectrogram into the sequence the time-delay network reads.
#[track_caller]
pub fn front_end(g: &Graph, x: Value, config: &Config, time: Extent) -> Value {
    let channels = FRONT_CHANNELS;
    let freq = config.feat_dim;

    // (N, F, T) -> (N, 1, F, T): the front end reads a spectrogram as a one-channel picture.
    let x = g.unsqueeze(x, 1);

    let x = front_conv(&g.subgraph("conv1"), x, 1, channels, 3, 1, 1, freq, time);
    let mut x = g.relu(batch_norm(&g.subgraph("bn1"), x, channels, 2));

    let mut freq = freq;
    for layer in ["layer1", "layer2"] {
        let sub = g.subgraph(layer);
        // Two blocks, of which only the first changes the shape.
        x = res_block(&sub.subgraph("0"), x, channels, 2, freq, time);
        freq /= 2;
        x = res_block(&sub.subgraph("1"), x, channels, 1, freq, time);
    }

    let x = front_conv(
        &g.subgraph("conv2"),
        x,
        channels,
        channels,
        3,
        1,
        2,
        freq,
        time,
    );
    freq /= 2;
    let x = g.relu(batch_norm(&g.subgraph("bn2"), x, channels, 2));

    // (N, C, F, T) -> (N, C * F, T).
    g.view(
        g.contiguous(x),
        [Extent::of(x, 0), Extent::At(channels * freq), time],
    )
}

/// The mean of each segment of `SEGMENT` frames, held for the length of the segment.
///
/// This is `CAMLayer.seg_pooling`: an average over non-overlapping windows, then each window's
/// value stretched back across the frames it covered. The last window is short -- `ceil_mode` --
/// and is averaged over the frames it actually has rather than over a window padded with zeros,
/// which is why the divisor is a vector rather than a number.
///
/// Stretching is a matrix multiply by a row of ones: there is no operator that repeats an
/// element, and this needs none.
#[track_caller]
fn segment_pooling(
    g: &Graph,
    x: Value,
    channels: i32,
    frames: i32,
    dtype: DType,
    device: Device,
) -> Result<Value> {
    let segments = (frames + SEGMENT - 1) / SEGMENT;
    let padded = segments * SEGMENT;

    // Sum over each whole window, with the tail zero-padded so the windows divide evenly.
    let wide = crate::audio::pad1d(
        g,
        x,
        0,
        padded - frames,
        crate::audio::Padding::Zero,
        dtype,
        device,
    )?;
    let split = g.view(
        wide,
        [
            Extent::of(x, 0),
            Extent::At(channels),
            Extent::At(segments),
            Extent::At(SEGMENT),
        ],
    );
    let totals = g.sum(split, 3);

    // How many frames each window really had. Only the last can be short.
    let mut counts = vec![SEGMENT as f32; segments as usize];
    counts[segments as usize - 1] = (frames - (segments - 1) * SEGMENT) as f32;
    let counts = g.constant(
        crate::flint::Tensor::from_f32(&[segments], &counts)?
            .to_device(device)?
            .cast(dtype)?,
    );
    let means = g.div(totals, counts);

    // (N, C, segments) -> (N, C, segments, SEGMENT) -> (N, C, padded) -> the frames that existed.
    let ones = g.constant(
        crate::flint::Tensor::from_f32(&[1, SEGMENT], &vec![1.0; SEGMENT as usize])?
            .to_device(device)?
            .cast(dtype)?,
    );
    let stretched = g.matmul(g.unsqueeze(means, 3), ones);
    let stretched = g.view(
        g.contiguous(stretched),
        [Extent::of(x, 0), Extent::At(channels), Extent::At(padded)],
    );

    Ok(g.slice(stretched, 2, 0, frames))
}

/// The context-aware masking of one dense layer: a local convolution, gated by what the whole
/// utterance and each segment of it look like.
#[track_caller]
#[allow(clippy::too_many_arguments)]
fn cam_layer(
    g: &Graph,
    x: Value,
    config: &Config,
    kernel: i32,
    dilation: i32,
    frames: i32,
    dtype: DType,
    device: Device,
) -> Result<Value> {
    let bn_channels = config.bn_channels();
    let reduced = bn_channels / 2;
    let padding = (kernel - 1) / 2 * dilation;

    let local = {
        let sub = g.subgraph("linear_local");
        let weight = sub.load("weight", &[config.growth_rate, bn_channels, kernel]);
        conv1d(
            &sub, x, weight, None, 1, padding, dilation, 1, dtype, device,
        )?
    };

    // The context is the whole utterance plus the segment: a mean over all of time, broadcast
    // back over it, added to the segment means.
    let overall = g.unsqueeze(g.div_scalar(g.sum(x, 2), frames as f32), 2);
    let segments = segment_pooling(g, x, bn_channels, frames, dtype, device)?;
    let context = g.add(segments, overall);

    let context = {
        let sub = g.subgraph("linear1");
        let weight = sub.load("weight", &[reduced, bn_channels, 1]);
        let bias = sub.load("bias", &[reduced]);
        g.relu(conv1d(
            &sub,
            context,
            weight,
            Some(bias),
            1,
            0,
            1,
            1,
            dtype,
            device,
        )?)
    };

    let mask = {
        let sub = g.subgraph("linear2");
        let weight = sub.load("weight", &[config.growth_rate, reduced, 1]);
        let bias = sub.load("bias", &[config.growth_rate]);
        g.sigmoid(conv1d(
            &sub,
            context,
            weight,
            Some(bias),
            1,
            0,
            1,
            1,
            dtype,
            device,
        )?)
    };

    Ok(g.mul(local, mask))
}

/// One layer of a dense block: normalize, bottleneck, normalize, and the masked convolution.
#[track_caller]
#[allow(clippy::too_many_arguments)]
fn dense_layer(
    g: &Graph,
    x: Value,
    config: &Config,
    in_channels: i32,
    kernel: i32,
    dilation: i32,
    frames: i32,
    dtype: DType,
    device: Device,
) -> Result<Value> {
    let bn_channels = config.bn_channels();

    let y = g.relu(batch_norm(
        &g.subgraph("nonlinear1").subgraph("batchnorm"),
        x,
        in_channels,
        1,
    ));

    let y = {
        let sub = g.subgraph("linear1");
        let weight = sub.load("weight", &[bn_channels, in_channels, 1]);
        conv1d(&sub, y, weight, None, 1, 0, 1, 1, dtype, device)?
    };

    let y = g.relu(batch_norm(
        &g.subgraph("nonlinear2").subgraph("batchnorm"),
        y,
        bn_channels,
        1,
    ));

    cam_layer(
        &g.subgraph("cam_layer"),
        y,
        config,
        kernel,
        dilation,
        frames,
        dtype,
        device,
    )
}

/// A dense block: every layer reads everything before it, and adds its own channels to the pile.
#[track_caller]
#[allow(clippy::too_many_arguments)]
fn dense_block(
    g: &Graph,
    x: Value,
    config: &Config,
    in_channels: i32,
    layers: i32,
    kernel: i32,
    dilation: i32,
    frames: i32,
    dtype: DType,
    device: Device,
) -> Result<(Value, i32)> {
    let mut running = x;
    let mut channels = in_channels;

    for index in 0..layers {
        let produced = dense_layer(
            &g.subgraph(&format!("tdnnd{}", index + 1)),
            running,
            config,
            channels,
            kernel,
            dilation,
            frames,
            dtype,
            device,
        )?;

        running = g.cat(running, produced, 1);
        channels += config.growth_rate;
    }

    Ok((running, channels))
}

/// Mean and standard deviation over time, laid end to end: `(N, C, T)` in, `(N, 2C)` out.
///
/// The deviation is the unbiased one, over `T - 1`, because that is what `torch.std` gives by
/// default and therefore what the reference pooled with.
#[track_caller]
fn statistics(g: &Graph, x: Value, frames: i32) -> Value {
    let mean = g.div_scalar(g.sum(x, 2), frames as f32);

    let centred = g.sub(x, g.unsqueeze(mean, 2));
    let variance = g.div_scalar(g.sum(g.square(centred), 2), (frames - 1) as f32);

    g.cat(mean, g.sqrt(variance), 1)
}

/// The whole model: `(N, T, F)` filterbank energies in, `(N, embedding_size)` out.
///
/// `frames` is how many frames the input has, which the graph needs as a number rather than as a
/// shape: the segment pooling has to know how many segments that comes to, and a graph cannot ask
/// a tensor how long it is at the time it is written.
#[track_caller]
pub fn graph(
    g: &Graph,
    x: Value,
    config: &Config,
    frames: i32,
    dtype: DType,
    device: Device,
) -> Result<Value> {
    // (N, T, F) -> (N, F, T), which is what the front end reads.
    let x = g.contiguous(g.transpose(x, 1, 2));

    let head = front_end(&g.subgraph("head"), x, config, Extent::At(frames));

    let xvector = g.subgraph("xvector");
    let channels = FRONT_CHANNELS * (config.feat_dim / 8);

    // The one strided layer: five frames wide, and it halves time.
    let frames = (frames + 2 * 2 - 5) / 2 + 1;
    let x = {
        let sub = xvector.subgraph("tdnn");
        let weight = sub
            .subgraph("linear")
            .load("weight", &[config.init_channels, channels, 5]);
        let y = conv1d(
            &sub.subgraph("linear"),
            head,
            weight,
            None,
            2,
            2,
            1,
            1,
            dtype,
            device,
        )?;
        g.relu(batch_norm(
            &sub.subgraph("nonlinear").subgraph("batchnorm"),
            y,
            config.init_channels,
            1,
        ))
    };

    let mut running = x;
    let mut channels = config.init_channels;

    for (index, &(layers, kernel, dilation)) in BLOCKS.iter().enumerate() {
        let (out, widened) = dense_block(
            &xvector.subgraph(&format!("block{}", index + 1)),
            running,
            config,
            channels,
            layers,
            kernel,
            dilation,
            frames,
            dtype,
            device,
        )?;

        // A transition normalizes, rectifies and halves the width it was handed.
        let sub = xvector.subgraph(&format!("transit{}", index + 1));
        let narrowed = g.relu(batch_norm(
            &sub.subgraph("nonlinear").subgraph("batchnorm"),
            out,
            widened,
            1,
        ));
        let weight = sub
            .subgraph("linear")
            .load("weight", &[widened / 2, widened, 1]);
        running = conv1d(
            &sub.subgraph("linear"),
            narrowed,
            weight,
            None,
            1,
            0,
            1,
            1,
            dtype,
            device,
        )?;
        channels = widened / 2;
    }

    let running = g.relu(batch_norm(
        &xvector.subgraph("out_nonlinear").subgraph("batchnorm"),
        running,
        channels,
        1,
    ));

    let pooled = statistics(g, running, frames);

    // The last projection reads a vector rather than a sequence, so it is a one-tap convolution
    // over a length of one; the normalization after it has no affine and is scale and shift all
    // the same.
    let sub = xvector.subgraph("dense");
    let weight = sub
        .subgraph("linear")
        .load("weight", &[config.embedding_size, channels * 2, 1]);
    let projected = conv1d(
        &sub.subgraph("linear"),
        g.unsqueeze(pooled, 2),
        weight,
        None,
        1,
        0,
        1,
        1,
        dtype,
        device,
    )?;

    let normalized = batch_norm(
        &sub.subgraph("nonlinear").subgraph("batchnorm"),
        projected,
        config.embedding_size,
        1,
    );

    Ok(g.squeeze(normalized, 2))
}
