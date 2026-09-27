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

//! HuBERT-large: 16 kHz audio in, 1024 numbers every twenty milliseconds out.
//!
//! `facebook/hubert-large-ll60k`, cut after its eighteenth layer the way ASTRAL cuts it -- the
//! layer list truncated and the final `layer_norm` replaced by an identity -- so what comes out is
//! the eighteenth layer's residual stream, not normalized.
//!
//! Two halves:
//!
//! - **A convolutional front end.** Seven convolutions, 512 wide, strides `5 2 2 2 2 2 2`: 320
//!   samples a frame. Each is followed by a layer norm over its channels and a GELU -- the
//!   `feat_extract_norm: layer` variety, where every convolution is normalized, not only the first.
//! - **A pre-norm transformer.** A projection to 1024, a convolutional position embedding added
//!   once, then eighteen layers that normalize before attention and before the feed-forward
//!   (`do_stable_layer_norm`).
//!
//! # The position embedding is a grouped convolution, run as sixteen
//!
//! It is a `Conv1d(1024, 1024, 128, padding=64, groups=16)`, weight-normed (the package holds it
//! folded). `flint`'s convolution on a card takes one group only, so it is written as what a
//! grouped convolution is: sixteen 64-channel convolutions over sixteen slices of the input, their
//! outputs laid side by side. An even kernel with symmetric padding makes one frame too many, and
//! upstream's `HubertSamePadLayer` drops the last; so does this.
//!
//! # The waveform is standardized first
//!
//! `Wav2Vec2FeatureExtractor(do_normalize=True)` takes the mean off and divides by
//! `sqrt(var + 1e-7)`, the variance a population one. [`normalize`] does that on the host.

use crate::audio::conv1d;
use crate::flint::{DType, Device, Extent, Graph, Value};
use crate::layers::{LayerNorm, Linear};
use crate::Result;

/// HuBERT-large's widths, from `config.json`, and how many of its layers are read.
#[derive(Clone, Debug)]
pub struct Config {
    /// `(kernel, stride)` of each front-end convolution.
    pub convolutions: Vec<(i32, i32)>,
    pub conv_dim: i32,
    pub hidden_size: i32,
    pub heads: i32,
    pub intermediate_size: i32,
    pub position_kernel: i32,
    pub position_groups: i32,
    /// How many transformer layers run: ASTRAL's `ssl_output_layer`.
    pub layers: i32,
    pub eps: f32,
}

impl Config {
    pub fn hubert_large() -> Config {
        Config {
            convolutions: vec![(10, 5), (3, 2), (3, 2), (3, 2), (3, 2), (2, 2), (2, 2)],
            conv_dim: 512,
            hidden_size: 1024,
            heads: 16,
            intermediate_size: 4096,
            position_kernel: 128,
            position_groups: 16,
            layers: 18,
            eps: 1e-5,
        }
    }

    /// How many frames `samples` samples make: each convolution, unpadded, in turn.
    pub fn frames(&self, samples: usize) -> i32 {
        let mut length = samples as i64;
        for (kernel, stride) in &self.convolutions {
            if length < i64::from(*kernel) {
                return 0;
            }
            length = (length - i64::from(*kernel)) / i64::from(*stride) + 1;
        }
        length as i32
    }
}

/// The waveform to zero mean and unit variance, as `Wav2Vec2FeatureExtractor` hands it over.
pub fn normalize(wave: &[f32]) -> Vec<f32> {
    let count = wave.len().max(1) as f64;
    let mean = wave.iter().map(|x| f64::from(*x)).sum::<f64>() / count;
    let variance = wave
        .iter()
        .map(|x| (f64::from(*x) - mean).powi(2))
        .sum::<f64>()
        / count;
    let scale = 1.0 / (variance + 1e-7).sqrt();

    wave.iter()
        .map(|x| ((f64::from(*x) - mean) * scale) as f32)
        .collect()
}

/// `(N, C, T)` normalized over `C` by the layer norm in `g`, as `(N, T, C)`.
fn channel_norm(g: &Graph, x: Value, channels: i32, eps: f32) -> Value {
    LayerNorm::graph(g, g.contiguous(g.transpose(x, 1, 2)), channels, eps)
}

/// The front end: `(1, 1, samples)` in, `(1, frames, conv_dim)` out.
fn front_end(
    g: &Graph,
    wave: Value,
    config: &Config,
    dtype: DType,
    device: Device,
) -> Result<Value> {
    let layers = g.subgraph("conv_layers");
    let mut running = wave;
    let mut channels = 1;

    for (index, (kernel, stride)) in config.convolutions.iter().enumerate() {
        let sub = layers.subgraph(&index.to_string());
        let conv = sub.subgraph("conv");
        let weight = conv.load("weight", &[config.conv_dim, channels, *kernel]);
        let bias = conv.load("bias", &[config.conv_dim]);

        let out = conv1d(
            &conv,
            running,
            weight,
            Some(bias),
            *stride,
            0,
            1,
            1,
            dtype,
            device,
        )?;
        let normed = g.gelu(channel_norm(
            &sub.subgraph("layer_norm"),
            out,
            config.conv_dim,
            config.eps,
        ));

        running = g.contiguous(g.transpose(normed, 1, 2));
        channels = config.conv_dim;
    }

    Ok(g.contiguous(g.transpose(running, 1, 2)))
}

/// The convolutional position embedding of `x` `(1, T, D)`, `(1, T, D)`.
fn position_embedding(
    g: &Graph,
    x: Value,
    config: &Config,
    frames: i32,
    dtype: DType,
    device: Device,
) -> Result<Value> {
    let d = config.hidden_size;
    let width = d / config.position_groups;
    let kernel = config.position_kernel;

    let conv = g.subgraph("conv");
    let weight = conv.load("weight", &[d, width, kernel]);
    let bias = conv.load("bias", &[d]);

    let channels_first = g.contiguous(g.transpose(x, 1, 2));
    let mut out: Option<Value> = None;
    for group in 0..config.position_groups {
        let (from, to) = (group * width, (group + 1) * width);
        let part = conv1d(
            &conv,
            g.contiguous(g.slice(channels_first, 1, from, to)),
            g.contiguous(g.slice(weight, 0, from, to)),
            Some(g.contiguous(g.slice(bias, 0, from, to))),
            1,
            kernel / 2,
            1,
            1,
            dtype,
            device,
        )?;
        out = Some(match out {
            None => part,
            Some(sofar) => g.cat(sofar, part, 1),
        });
    }

    let out = out.expect("at least one group");
    // One frame too many from an even kernel: `HubertSamePadLayer` drops the last.
    let trimmed = g.contiguous(g.slice(out, 2, 0, frames));

    Ok(g.contiguous(g.transpose(g.gelu(trimmed), 1, 2)))
}

/// One pre-norm encoder layer over `x` `(1, T, D)`.
fn layer(g: &Graph, x: Value, config: &Config, frames: i32) -> Value {
    let d = config.hidden_size;
    let heads = config.heads;
    let head_dim = d / heads;

    let normed = LayerNorm::graph(&g.subgraph("layer_norm"), x, d, config.eps);

    let attention = g.subgraph("attention");
    let split = |name: &str| {
        let projected = Linear::graph(&attention.subgraph(name), normed, d, d, true);
        let shaped = g.view(
            projected,
            [
                Extent::At(1),
                Extent::At(frames),
                Extent::At(heads),
                Extent::At(head_dim),
            ],
        );
        g.contiguous(g.transpose(shaped, 1, 2))
    };

    let out = g.attention(split("q_proj"), split("k_proj"), split("v_proj"), false);
    let merged = g.view(
        g.contiguous(g.transpose(out, 1, 2)),
        [Extent::At(1), Extent::At(frames), Extent::At(d)],
    );
    let x = g.add(
        x,
        Linear::graph(&attention.subgraph("out_proj"), merged, d, d, true),
    );

    let normed = LayerNorm::graph(&g.subgraph("final_layer_norm"), x, d, config.eps);
    let ff = g.subgraph("feed_forward");
    let inner = g.gelu(Linear::graph(
        &ff.subgraph("intermediate_dense"),
        normed,
        d,
        config.intermediate_size,
        true,
    ));

    g.add(
        x,
        Linear::graph(
            &ff.subgraph("output_dense"),
            inner,
            config.intermediate_size,
            d,
            true,
        ),
    )
}

/// HuBERT over `wave`, `(1, 1, samples)` already [`normalize`]d: `(1, frames, hidden_size)`, the
/// residual stream after `config.layers` layers.
///
/// `frames` is [`Config::frames`] of the sample count.
pub fn graph(
    g: &Graph,
    wave: Value,
    config: &Config,
    frames: i32,
    dtype: DType,
    device: Device,
) -> Result<Value> {
    let features = front_end(
        &g.subgraph("feature_extractor"),
        wave,
        config,
        dtype,
        device,
    )?;

    let projection = g.subgraph("feature_projection");
    let normed = LayerNorm::graph(
        &projection.subgraph("layer_norm"),
        features,
        config.conv_dim,
        config.eps,
    );
    let mut x = Linear::graph(
        &projection.subgraph("projection"),
        normed,
        config.conv_dim,
        config.hidden_size,
        true,
    );

    let encoder = g.subgraph("encoder");
    let position = position_embedding(
        &encoder.subgraph("pos_conv_embed"),
        x,
        config,
        frames,
        dtype,
        device,
    )?;
    x = g.add(x, position);

    let layers = encoder.subgraph("layers");
    for index in 0..config.layers {
        x = layer(&layers.subgraph(&index.to_string()), x, config, frames);
    }

    Ok(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_frame_is_320_samples() {
        let config = Config::hubert_large();
        // Six seconds at 16 kHz: upstream's HuBERT gives 299 frames.
        assert_eq!(config.frames(96_000), 299);
        assert_eq!(config.frames(400), 1);
        assert_eq!(config.frames(399), 0);
    }

    #[test]
    fn normalizing_leaves_zero_mean_and_unit_variance() {
        let wave: Vec<f32> = (0..1000)
            .map(|i| (i as f32 * 0.1).sin() * 0.3 + 0.05)
            .collect();
        let normed = normalize(&wave);
        let mean = normed.iter().map(|x| f64::from(*x)).sum::<f64>() / 1000.0;
        let variance = normed
            .iter()
            .map(|x| (f64::from(*x) - mean).powi(2))
            .sum::<f64>()
            / 1000.0;

        assert!(mean.abs() < 1e-6, "mean {mean}");
        assert!((variance - 1.0).abs() < 1e-4, "variance {variance}");
    }
}
