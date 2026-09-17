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

//! The Qwen-Image decoder, which turns a sixteen-channel latent into a picture.
//!
//! It is published as a *video* autoencoder: five-dimensional weights, causal in time. flint has
//! no conv3d and does not need one. For a single frame the causal padding in front is zeros and
//! nothing in the network grows the time axis, so every convolution is exactly a 2-D one over the
//! last temporal slice -- proved in `tools/causal_conv3d_folding_test.py`, and done by the
//! exporter, so what arrives here is already flat. The `time_conv` weights are not in the package
//! at all: they sit behind a cache branch that one frame never takes.
//!
//! Two things differ from [`sdxl::VaeDecoder`](crate::VaeDecoder), beyond the channel count:
//!
//! *The normalization is RMSNorm over the channels*, written in the checkpoint as
//! `F.normalize(x, dim=1) * sqrt(C) * gamma`. flint normalizes over the last dimension, so the
//! image is turned inside out for each of them and back again -- there are about thirty-five, and
//! it is still cheaper than a kernel that does not exist.
//!
//! *The latent is normalized per channel*, sixteen means and sixteen deviations, where SDXL has a
//! single scaling factor. Undoing that is the first thing the decoder does.

use std::fmt;
use std::rc::Rc;

use super::config::VaeConfig;
use crate::error::{Error, Result};
use crate::flint::{
    check_parameters, DType, Device, Extent, Graph, Ir, ParamSource, RunContext, Tensor, Value,
};
use crate::layers::{Conv2d, Linear};

/// What the decoder is made of, in the order the package names it.
///
/// The checkpoint keeps the original names rather than the ones diffusers converts them to, and
/// the upsample list is flat: a run of residual blocks, then a resample, then more, with the
/// channel count stepping down as it goes. Which is which is read from the weights rather than
/// written here, so a differently shaped VAE would need no new table.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Stage {
    /// Two convolutions and a skip, with a shortcut when the width changes.
    Residual { input: i32, output: i32 },
    /// Nearest-neighbour doubling and a convolution that halves the channels.
    Resample { input: i32, output: i32 },
}

/// Normalize over the channels of an `(N, C, H, W)`.
///
/// `F.normalize(x, dim=1) * sqrt(C) * gamma` is `x / sqrt(mean(x^2)) * gamma` along C, which is
/// RMSNorm -- but along the channel rather than the last dimension, so the image is transposed
/// into `(N, H, W, C)` and back.
fn channel_norm(g: &Graph, input: Value, channels: i32, name: &str) -> Value {
    let batch = Extent::of(input, 0);
    let height = Extent::of(input, 2);
    let width = Extent::of(input, 3);

    // (N, C, H, W) -> (N, H, W, C)
    let x = g.contiguous(g.transpose(g.transpose(input, 1, 2), 2, 3));
    let weight = g.subgraph(name).load(Linear::WEIGHT, &[channels]);
    let x = g.rms_norm(x, weight, super::VAE_NORM_EPS);

    // and back
    let x = g.contiguous(g.transpose(g.transpose(x, 2, 3), 1, 2));
    g.view(x, [batch, Extent::At(channels), height, width])
}

/// One residual block: norm, activate, convolve, twice, plus what came in.
fn residual(
    g: &Graph,
    input: Value,
    in_channels: i32,
    out_channels: i32,
    has_shortcut: bool,
) -> Value {
    let x = channel_norm(g, input, in_channels, "residual.0");
    let x = g.silu(x);
    let x = Conv2d::graph(
        &g.subgraph("residual.2"),
        x,
        in_channels,
        out_channels,
        3,
        1,
        1,
    );

    let x = channel_norm(g, x, out_channels, "residual.3");
    let x = g.silu(x);
    let x = Conv2d::graph(
        &g.subgraph("residual.6"),
        x,
        out_channels,
        out_channels,
        3,
        1,
        1,
    );

    // Only when the width changes; a one by one convolution rather than a reshape.
    let skip = match has_shortcut {
        true => Conv2d::graph(
            &g.subgraph("shortcut"),
            input,
            in_channels,
            out_channels,
            1,
            1,
            0,
        ),
        false => input,
    };
    g.add(skip, x)
}

/// The one attention in the decoder, at its smallest resolution, every pixel over every other.
fn attention(g: &Graph, input: Value, channels: i32) -> Value {
    let batch = Extent::of(input, 0);
    let height = Extent::of(input, 2);
    let width = Extent::of(input, 3);
    let positions = Extent::prod(input, 2, 4);

    let x = channel_norm(g, input, channels, "norm");

    // The projections are one by one convolutions in the checkpoint, which over an image is the
    // same arithmetic as a linear over the sequence it is about to become.
    let qkv = Conv2d::graph(&g.subgraph("to_qkv"), x, channels, 3 * channels, 1, 1, 0);
    let qkv = g.contiguous(g.transpose(
        g.view(qkv, [batch, Extent::At(3 * channels), positions]),
        1,
        2,
    ));

    let mut parts = Vec::with_capacity(3);
    for index in 0..3 {
        let part = g.contiguous(g.slice(qkv, 2, index * channels, (index + 1) * channels));
        // One head, as wide as the channels.
        parts.push(g.view(
            part,
            [batch, Extent::At(1), positions, Extent::At(channels)],
        ));
    }

    let x = g.attention(parts[0], parts[1], parts[2], false);
    let x = g.view(x, [batch, positions, Extent::At(channels)]);
    let x = g.contiguous(g.transpose(x, 1, 2));
    let x = g.view(x, [batch, Extent::At(channels), height, width]);

    let x = Conv2d::graph(&g.subgraph("proj"), x, channels, channels, 1, 1, 0);
    g.add(input, x)
}

/// The whole decoder, written into `g`.
fn write(
    config: &VaeConfig,
    stages: &[Stage],
    deepest: i32,
    dtype: DType,
    device: Device,
    g: &Graph,
) {
    let latent = g.input("latent");
    let mean = g.input("latents_mean");
    let std = g.input("latents_std");

    // The autoencoder runs wide whatever the sampler above it runs in, the way SDXL's does.
    let x = g.cast(g.to_device(latent, device), dtype);

    // Out of the normalization the model works in: sixteen channels, each with its own pair.
    let x = g.add(g.mul(x, g.cast(std, dtype)), g.cast(mean, dtype));

    let x = Conv2d::graph(
        &g.subgraph("conv2"),
        x,
        config.latent_channels,
        config.latent_channels,
        1,
        1,
        0,
    );
    let mut x = Conv2d::graph(
        &g.subgraph("decoder.conv1"),
        x,
        config.latent_channels,
        deepest,
        3,
        1,
        1,
    );

    let middle = g.subgraph("decoder.middle");
    x = residual(&middle.subgraph("0"), x, deepest, deepest, false);
    x = attention(&middle.subgraph("1"), x, deepest);
    x = residual(&middle.subgraph("2"), x, deepest, deepest, false);

    let ups = g.subgraph("decoder.upsamples");
    for (index, stage) in stages.iter().enumerate() {
        let sub = ups.subgraph(&index.to_string());
        x = match *stage {
            Stage::Residual { input, output } => residual(&sub, x, input, output, input != output),
            Stage::Resample { input, output } => {
                let up = g.upsample_nearest2d(x, 2);
                Conv2d::graph(&sub.subgraph("resample.1"), up, input, output, 3, 1, 1)
            }
        };
    }

    let last = stages
        .last()
        .map(|stage| match *stage {
            Stage::Residual { output, .. } | Stage::Resample { output, .. } => output,
        })
        .unwrap_or(deepest);

    let head = g.subgraph("decoder.head");
    let x = channel_norm(&head, x, last, "0");
    let x = g.silu(x);
    g.output(
        "image",
        Conv2d::graph(&head.subgraph("2"), x, last, 3, 3, 1, 1),
    );
}

/// Work out the decoder's shape from the weights it holds.
///
/// The upsample list is flat and its widths step down as it goes, so rather than a table that
/// would have to agree with the package, each entry is read: a `resample.1` makes it a resample,
/// a `residual.2` a residual block, and their own weights say how wide.
fn stages(weights: &dyn ParamSource, name: &str) -> Result<Vec<Stage>> {
    let mut found = Vec::new();

    // The shapes rather than the weights: this is reading the architecture, and a weight moved
    // across the bus to be measured and dropped would be a whole autoencoder's worth of copying
    // to learn what the package already says in its header.
    let at = |shape: &[i32], axis: usize, what: &str| -> Result<i32> {
        shape
            .get(axis)
            .copied()
            .ok_or_else(|| Error::model(format!("{what} has shape {shape:?}, which has no {axis}")))
    };

    for index in 0.. {
        let prefix = format!("{name}.decoder.upsamples.{index}");
        let resample = format!("{prefix}.resample.1.weight");
        let residual = format!("{prefix}.residual.2.weight");

        if let Some(shape) = weights.shape_of(&resample) {
            found.push(Stage::Resample {
                input: at(&shape, 1, &resample)?,
                output: at(&shape, 0, &resample)?,
            });
        } else if let Some(shape) = weights.shape_of(&residual) {
            let second = format!("{prefix}.residual.6.weight");
            let out = weights
                .shape_of(&second)
                .ok_or_else(|| Error::model(format!("{prefix} has no second convolution")))?;

            found.push(Stage::Residual {
                input: at(&shape, 1, &residual)?,
                output: at(&out, 0, &second)?,
            });
        } else {
            break;
        }
    }

    if found.is_empty() {
        return Err(Error::model(format!("{name} holds no decoder stages")));
    }
    Ok(found)
}

pub struct VaeDecoder {
    config: VaeConfig,
    ir: Ir,
    weights: Rc<dyn ParamSource>,
    mean: Tensor,
    std: Tensor,
}

impl fmt::Debug for VaeDecoder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "VaeDecoder({} instructions)", self.ir.len())
    }
}

impl VaeDecoder {
    pub fn build(
        config: VaeConfig,
        name: &str,
        weights: &Rc<dyn ParamSource>,
        device: Device,
        float_type: DType,
    ) -> Result<VaeDecoder> {
        let stages = stages(weights.as_ref(), name)?;

        // How wide the decoder is at its deepest, which is what the first convolution opens it
        // out to. Read rather than written down, so that a differently shaped VAE needs no edit.
        let opens_to = format!("{name}.decoder.conv1.weight");
        let deepest = weights
            .shape_of(&opens_to)
            .and_then(|shape| shape.first().copied())
            .ok_or_else(|| Error::model(format!("{name} has no decoder")))?;

        // Not `with_weights`: an autoencoder is convolutions, and its few projections are not
        // what a quantized weight is for. See the note in `sdxl::vae`.
        let graph = Graph::new();

        write(
            &config,
            &stages,
            deepest,
            float_type,
            device,
            &graph.subgraph(name),
        );
        check_parameters(&graph, weights.as_ref())?;

        let channels = config.latent_channels;
        let shape = [1, channels, 1, 1];
        let mean = Tensor::from_f32(&shape, &config.latents_mean)?.to_device(device)?;
        let std = Tensor::from_f32(&shape, &config.latents_std)?.to_device(device)?;

        Ok(VaeDecoder {
            weights: Rc::clone(weights),
            ir: Ir::compile(&graph),
            config,
            mean,
            std,
        })
    }

    pub fn config(&self) -> &VaeConfig {
        &self.config
    }

    pub fn ir(&self) -> &Ir {
        &self.ir
    }

    /// `latent` is `(1, 16, H, W)`; what comes back is `(1, 3, H * 8, W * 8)` in `-1..=1`.
    pub fn forward(&self, latent: &Tensor) -> Result<Tensor> {
        let dim = latent.dim()?;
        if dim != 4 {
            return Err(Error::model(format!(
                "the decoder takes a latent as (1, C, H, W), got a {dim}-D tensor"
            )));
        }

        let channels = latent.shape_at(1)?;
        if channels != self.config.latent_channels {
            return Err(Error::model(format!(
                "a {channels}-channel latent is not the {} this decoder reads",
                self.config.latent_channels
            )));
        }

        let run = RunContext::new(&*self.weights)
            .input("latent", latent)
            .input("latents_mean", &self.mean)
            .input("latents_std", &self.std);

        let outputs = self.ir.run(&run)?;

        outputs
            .iter()
            .find(|(name, _)| name == "image")
            .map(|(_, tensor)| tensor.clone())
            .ok_or_else(|| Error::model("the decoder produced no image"))
    }
}
