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

//! The decoder that gives a layer its alpha.
//!
//! An SDXL autoencoder decodes three channels, and a layer of an illustration needs four: what is
//! there and how much of it is. LayerDiffuse's answer is a second, small U-Net that reads the
//! colours the autoencoder decoded and the latent they came from, and says both the colour and the
//! alpha again -- `[alpha, R, G, B]`, in that order, each in `[0, 1]`.
//!
//! It is seven resolutions deep, from the picture's own down to a sixty-fourth of it, and the latent
//! joins at an eighth, which is the size it already is. Its blocks are the autoencoder's -- residual
//! blocks with no timestep, attention with a group norm in front -- but narrow, normalized over
//! four groups, and attended over with heads only eight wide. Eight is not a head the compiled
//! FlashAttention kernels take, so the attention runs on the fallback, which takes the queries a
//! block at a time.

use std::fmt;
use std::rc::Rc;

use crate::error::{Error, Result};
use crate::flint::{
    check_parameters, DType, Extent, Graph, Ir, RunContext, Tensor, Value, Weights,
};
use crate::layers::{Conv2d, GroupNorm, Linear};

/// The epsilon every block normalizes with, which is a constructor default upstream.
const NORM_EPS: f32 = 1e-5;

/// What the package records about the alpha decoder.
#[derive(Clone, Debug)]
pub struct AlphaConfig {
    /// The width at each resolution, the picture's own first.
    pub block_out_channels: Vec<i32>,
    /// Whether each resolution attends, on the way down; the way up mirrors it.
    pub attention: Vec<bool>,
    pub layers_per_block: i32,
    pub norm_num_groups: i32,
    pub head_dim: i32,
    /// The resolution the latent is added at, before that resolution's first block.
    pub latent_level: usize,
    /// How deep the latent is.
    pub latent_channels: i32,
}

impl AlphaConfig {
    fn levels(&self) -> usize {
        self.block_out_channels.len()
    }
}

fn resnet_block(
    g: &Graph,
    config: &AlphaConfig,
    input: Value,
    in_channels: i32,
    out_channels: i32,
) -> Value {
    let groups = config.norm_num_groups;

    let x = GroupNorm::graph(&g.subgraph("norm1"), input, in_channels, groups, NORM_EPS);
    let x = Conv2d::graph(
        &g.subgraph("conv1"),
        g.silu(x),
        in_channels,
        out_channels,
        3,
        1,
        1,
    );
    let x = GroupNorm::graph(&g.subgraph("norm2"), x, out_channels, groups, NORM_EPS);
    let x = Conv2d::graph(
        &g.subgraph("conv2"),
        g.silu(x),
        out_channels,
        out_channels,
        3,
        1,
        1,
    );

    let residual = match in_channels == out_channels {
        true => input,
        false => Conv2d::graph(
            &g.subgraph("shortcut"),
            input,
            in_channels,
            out_channels,
            1,
            1,
            0,
        ),
    };
    g.add(residual, x)
}

/// Self attention over the pixels, in heads `head_dim` wide, around a group norm and a residual.
fn attention_block(g: &Graph, config: &AlphaConfig, input: Value, channels: i32) -> Value {
    let batch = Extent::of(input, 0);
    let positions = Extent::prod(input, 2, 4);
    let heads = channels / config.head_dim;

    let x = GroupNorm::graph(
        &g.subgraph("norm"),
        input,
        channels,
        config.norm_num_groups,
        NORM_EPS,
    );
    let x = g.view(x, [batch, Extent::At(channels), positions]);
    let x = g.contiguous(g.transpose(x, 1, 2));

    let qkv = Linear::graph(&g.subgraph("qkv_proj"), x, channels, 3 * channels, true);
    let mut parts = Vec::with_capacity(3);
    for index in 0..3 {
        let part = g.contiguous(g.slice(qkv, 2, index * channels, (index + 1) * channels));
        let part = g.view(
            part,
            [
                batch,
                positions,
                Extent::At(heads),
                Extent::At(config.head_dim),
            ],
        );
        parts.push(g.contiguous(g.transpose(part, 1, 2)));
    }

    let x = g.attention(parts[0], parts[1], parts[2], false);
    let x = g.view(
        g.contiguous(g.transpose(x, 1, 2)),
        [batch, positions, Extent::At(channels)],
    );
    let x = Linear::graph(&g.subgraph("out_proj"), x, channels, channels, true);

    let x = g.view(
        g.contiguous(g.transpose(x, 1, 2)),
        [
            batch,
            Extent::At(channels),
            Extent::of(input, 2),
            Extent::of(input, 3),
        ],
    );
    g.add(input, x)
}

fn write(config: &AlphaConfig, g: &Graph) {
    let levels = config.levels();
    let width = |level: usize| config.block_out_channels[level];
    // The latent joins what is about to enter that resolution, which is still as wide as the
    // resolution before it.
    let latent_width = width(config.latent_level.saturating_sub(1));

    let latent = Conv2d::graph(
        &g.subgraph("latent_conv_in"),
        g.input("latent"),
        config.latent_channels,
        latent_width,
        1,
        1,
        0,
    );
    let mut x = Conv2d::graph(
        &g.subgraph("conv_in"),
        g.input("pixel"),
        3,
        width(0),
        3,
        1,
        1,
    );

    let mut residuals = vec![x];
    for level in 0..levels {
        if level == config.latent_level {
            x = g.add(x, latent);
        }
        let block = g.subgraph(&format!("down{level}"));
        for layer in 0..config.layers_per_block {
            let from = match layer {
                0 if level == 0 => width(0),
                0 => width(level - 1),
                _ => width(level),
            };
            x = resnet_block(
                &block.subgraph(&format!("resnet{layer}")),
                config,
                x,
                from,
                width(level),
            );
            if config.attention[level] {
                x = attention_block(
                    &block.subgraph(&format!("attn{layer}")),
                    config,
                    x,
                    width(level),
                );
            }
            residuals.push(x);
        }
        if level + 1 < levels {
            x = Conv2d::graph(
                &block.subgraph("downsample0"),
                x,
                width(level),
                width(level),
                3,
                2,
                1,
            );
            residuals.push(x);
        }
    }

    let widest = width(levels - 1);
    x = resnet_block(&g.subgraph("mid.resnet0"), config, x, widest, widest);
    x = attention_block(&g.subgraph("mid.attn0"), config, x, widest);
    x = resnet_block(&g.subgraph("mid.resnet1"), config, x, widest, widest);

    for index in 0..levels {
        let level = levels - 1 - index;
        let block = g.subgraph(&format!("up{index}"));
        let out = width(level);
        let prev = match index {
            0 => widest,
            _ => width(level + 1),
        };
        let skip_last = width(level.saturating_sub(1));
        let layers = config.layers_per_block + 1;
        for layer in 0..layers {
            let skip = match layer == layers - 1 {
                true => skip_last,
                false => out,
            };
            let from = match layer {
                0 => prev,
                _ => out,
            };
            let residual = residuals
                .pop()
                .expect("the way down produces exactly what the way up reads");
            x = resnet_block(
                &block.subgraph(&format!("resnet{layer}")),
                config,
                g.cat(x, residual, 1),
                from + skip,
                out,
            );
            if config.attention[level] {
                x = attention_block(&block.subgraph(&format!("attn{layer}")), config, x, out);
            }
        }
        if index + 1 < levels {
            x = Conv2d::graph(
                &block.subgraph("upsample0"),
                g.upsample_nearest2d(x, 2),
                out,
                out,
                3,
                1,
                1,
            );
        }
    }
    assert!(residuals.is_empty(), "the way up left residuals unread");

    let x = GroupNorm::graph(
        &g.subgraph("conv_norm_out"),
        x,
        width(0),
        config.norm_num_groups,
        NORM_EPS,
    );
    let x = Conv2d::graph(&g.subgraph("conv_out"), g.silu(x), width(0), 4, 3, 1, 1);
    g.output("alpha_rgb", x);
}

fn check(config: &AlphaConfig) -> Result<()> {
    let levels = config.levels();
    if levels < 2
        || config.attention.len() != levels
        || config.latent_level == 0
        || config.latent_level >= levels
    {
        return Err(Error::model(
            "the alpha decoder's config disagrees with itself about how many resolutions it has",
        ));
    }
    for (level, &channels) in config.block_out_channels.iter().enumerate() {
        if channels % config.norm_num_groups != 0 {
            return Err(Error::model(format!(
                "{channels} channels at level {level} do not divide into {} groups",
                config.norm_num_groups
            )));
        }
        if config.attention[level] && (config.head_dim <= 0 || channels % config.head_dim != 0) {
            return Err(Error::model(format!(
                "{channels} channels at level {level} do not divide into heads {} wide",
                config.head_dim
            )));
        }
    }
    Ok(())
}

pub struct AlphaDecoder {
    config: AlphaConfig,
    ir: Ir,
    weights: Rc<Weights>,
    dtype: DType,
}

impl fmt::Debug for AlphaDecoder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "AlphaDecoder({} instructions)", self.ir.len())
    }
}

impl AlphaDecoder {
    /// `dtype` is what it computes in; its weights are in the package at that precision.
    pub fn build(
        config: AlphaConfig,
        name: &str,
        weights: &Rc<Weights>,
        dtype: DType,
    ) -> Result<AlphaDecoder> {
        check(&config)?;
        let graph = Graph::new();
        write(&config, &graph.subgraph(name));
        check_parameters(&graph, weights.as_ref())?;
        Ok(AlphaDecoder {
            ir: Ir::compile(&graph),
            weights: Rc::clone(weights),
            config,
            dtype,
        })
    }

    /// `[alpha, R, G, B]` for one layer, `(1, 4, H, W)` and clipped to `[0, 1]`.
    ///
    /// `pixel` is what the autoencoder decoded, `(1, 3, H, W)` brought from `[-1, 1]` to
    /// `[0, 1]`, and `latent` is the latent it decoded, `(1, C, H / 8, W / 8)` with the
    /// autoencoder's scaling divided back out.
    pub fn forward(&self, pixel: &Tensor, latent: &Tensor) -> Result<Tensor> {
        let scale = 1 << self.config.latent_level;
        let (height, width) = (pixel.shape_at(2)?, pixel.shape_at(3)?);
        let divisor = 1 << (self.config.levels() - 1);
        if height % divisor != 0 || width % divisor != 0 {
            return Err(Error::model(format!(
                "a {height} by {width} picture does not survive being halved {} times",
                self.config.levels() - 1
            )));
        }
        if latent.shape_at(2)? * scale != height || latent.shape_at(3)? * scale != width {
            return Err(Error::model(format!(
                "a {} by {} latent is not an eighth of a {height} by {width} picture",
                latent.shape_at(2)?,
                latent.shape_at(3)?
            )));
        }

        // Brought to the weights and their precision, wherever they were made: a picture read off
        // the disk is on the host whatever the model is on, as the autoencoder's halves allow too.
        let device = self.weights.device();
        let pixel = pixel.to_device(device)?.cast(self.dtype)?;
        let latent = latent.to_device(device)?.cast(self.dtype)?;
        let context = RunContext::new(&*self.weights)
            .input("pixel", &pixel)
            .input("latent", &latent);
        let outputs = self.ir.run(&context)?;
        let out = outputs
            .into_iter()
            .find(|(name, _)| name == "alpha_rgb")
            .map(|(_, tensor)| tensor)
            .ok_or_else(|| Error::model("the alpha decoder produced nothing"))?;
        Ok(crate::flint::functional::clamp(&out, 0.0, 1.0)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> AlphaConfig {
        AlphaConfig {
            block_out_channels: vec![32, 32, 64, 128, 256, 512, 512],
            attention: vec![false, false, false, false, true, true, true],
            layers_per_block: 2,
            norm_num_groups: 4,
            head_dim: 8,
            latent_level: 3,
            latent_channels: 4,
        }
    }

    #[test]
    fn the_decoder_is_written_without_any_weights() {
        let config = config();
        check(&config).unwrap();
        let graph = Graph::new();
        write(&config, &graph.subgraph("alpha"));
        let listing = graph.to_string();

        assert!(listing.contains("alpha.latent_conv_in.weight"));
        assert!(listing.contains("alpha.down6.attn1.qkv_proj.weight"));
        assert!(!listing.contains("alpha.down3.attn0."));
        assert!(!listing.contains("alpha.down6.downsample0."));
        assert!(listing.contains("alpha.up0.attn2.out_proj.weight"));
        assert!(listing.contains("alpha.up6.resnet2.conv2.weight"));
        assert!(!listing.contains("alpha.up6.upsample0."));
        assert!(listing.contains("alpha.mid.attn0.norm.weight"));
    }

    #[test]
    fn every_weight_is_asked_for_once() {
        let graph = Graph::new();
        write(&config(), &graph.subgraph("alpha"));
        let mut names = graph.parameters();
        let total = names.len();
        names.sort();
        names.dedup();
        assert_eq!(names.len(), total);
    }
}
