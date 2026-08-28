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

//! The decoder half of SDXL's autoencoder, which turns a latent into an image.
//!
//! A latent is eight times smaller than the image on each axis and four channels deep instead of
//! three, so this is where most of the pixels are made: a 128 by 128 latent becomes 1024 by 1024.
//! Only the decoder is here. Going the other way is what an encoder is for, and text to image
//! never asks for it.

use crate::error::{Error, Result};
use crate::flint::{functional as F, DType, Tensor};
use crate::layers::{Conv2d, GroupNorm};
use crate::var_builder::VarBuilder;

/// Two convolutions around a normalization and an activation, added back to what came in. Where
/// the channel count changes, the shortcut is a 1x1 convolution rather than the input itself.
#[derive(Debug)]
struct ResnetBlock {
    norm1: GroupNorm,
    conv1: Conv2d,
    norm2: GroupNorm,
    conv2: Conv2d,
    shortcut: Option<Conv2d>,
}

impl ResnetBlock {
    fn build(
        in_channels: i32,
        out_channels: i32,
        config: &VaeConfig,
        vb: &VarBuilder,
    ) -> Result<ResnetBlock> {
        let shortcut = if in_channels == out_channels {
            None
        } else {
            Some(Conv2d::build(
                in_channels,
                out_channels,
                1,
                1,
                0,
                &vb.with_name("shortcut"),
            )?)
        };

        Ok(ResnetBlock {
            norm1: GroupNorm::build(
                in_channels,
                config.norm_num_groups,
                config.norm_eps,
                &vb.with_name("norm1"),
            )?,
            conv1: Conv2d::build(in_channels, out_channels, 3, 1, 1, &vb.with_name("conv1"))?,
            norm2: GroupNorm::build(
                out_channels,
                config.norm_num_groups,
                config.norm_eps,
                &vb.with_name("norm2"),
            )?,
            conv2: Conv2d::build(out_channels, out_channels, 3, 1, 1, &vb.with_name("conv2"))?,
            shortcut,
        })
    }

    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let x = self.norm1.forward(input)?;
        let x = F::silu(&x)?;
        let x = self.conv1.forward(&x)?;

        let x = self.norm2.forward(&x)?;
        let x = F::silu(&x)?;
        let x = self.conv2.forward(&x)?;

        let residual = match &self.shortcut {
            Some(shortcut) => shortcut.forward(input)?,
            None => input.clone(),
        };

        Ok(F::add(&residual, &x)?)
    }
}

/// The one attention in the decoder, at the smallest resolution it works at.
///
/// It is a single head as wide as the whole channel count, which is not a shape the compiled
/// FlashAttention kernels take, so it runs on the fallback. That fallback takes the queries a
/// block at a time, which is what keeps a 1024 by 1024 image from needing half a gigabyte for the
/// score matrix alone.
#[derive(Debug)]
struct AttentionBlock {
    norm: GroupNorm,
    qkv_weight: Tensor,
    qkv_bias: Tensor,
    out_proj_weight: Tensor,
    out_proj_bias: Tensor,
    channels: i32,
}

impl AttentionBlock {
    fn build(channels: i32, config: &VaeConfig, vb: &VarBuilder) -> Result<AttentionBlock> {
        Ok(AttentionBlock {
            norm: GroupNorm::build(
                channels,
                config.norm_num_groups,
                config.norm_eps,
                &vb.with_name("norm"),
            )?,
            qkv_weight: vb.get("qkv_proj.weight", &[3 * channels, channels])?,
            qkv_bias: vb.get("qkv_proj.bias", &[3 * channels])?,
            out_proj_weight: vb.get("out_proj.weight", &[channels, channels])?,
            out_proj_bias: vb.get("out_proj.bias", &[channels])?,
            channels,
        })
    }

    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let shape = input.shape();
        let (height, width) = (shape[2], shape[3]);
        let positions = height * width;

        let x = self.norm.forward(input)?;

        // Every pixel is a position that attends to every other, so the image is read as a
        // sequence: (1, C, H, W) becomes (1, H * W, C).
        let x = x
            .view(&[1, self.channels, positions])?
            .transpose(1, 2)?
            .contiguous()?;

        let qkv = F::add(
            &F::matmul(&x, &self.qkv_weight.transpose(0, 1)?)?,
            &self.qkv_bias,
        )?;

        let mut parts = Vec::with_capacity(3);
        for index in 0..3 {
            parts.push(
                qkv.slice(2, index * self.channels, (index + 1) * self.channels)?
                    .contiguous()?
                    // One head, as wide as the channels: (1, 1, H * W, C).
                    .view(&[1, 1, positions, self.channels])?,
            );
        }

        let attended = F::attention(&parts[0], &parts[1], &parts[2], false)?;
        let attended = attended.view(&[1, positions, self.channels])?;

        let projected = F::add(
            &F::matmul(&attended, &self.out_proj_weight.transpose(0, 1)?)?,
            &self.out_proj_bias,
        )?;

        let projected =
            projected
                .transpose(1, 2)?
                .contiguous()?
                .view(&[1, self.channels, height, width])?;

        Ok(F::add(input, &projected)?)
    }
}

/// One rung of the decoder: some residual blocks, and the doubling that follows them.
#[derive(Debug)]
struct UpBlock {
    resnets: Vec<ResnetBlock>,
    upsample: Option<Conv2d>,
}

impl UpBlock {
    fn build(
        in_channels: i32,
        out_channels: i32,
        upsample: bool,
        config: &VaeConfig,
        vb: &VarBuilder,
    ) -> Result<UpBlock> {
        // A decoder runs one more block per rung than the encoder does.
        let mut resnets = Vec::new();
        for index in 0..config.layers_per_block + 1 {
            let from = if index == 0 {
                in_channels
            } else {
                out_channels
            };
            resnets.push(ResnetBlock::build(
                from,
                out_channels,
                config,
                &vb.with_name(&format!("resnet{index}")),
            )?);
        }

        let upsample = if upsample {
            Some(Conv2d::build(
                out_channels,
                out_channels,
                3,
                1,
                1,
                &vb.with_name("upsample0"),
            )?)
        } else {
            None
        };

        Ok(UpBlock { resnets, upsample })
    }

    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let mut x = input.clone();
        for resnet in &self.resnets {
            x = resnet.forward(&x)?;
        }

        // Nearest, then a convolution to smooth what repeating pixels left behind.
        if let Some(upsample) = &self.upsample {
            x = upsample.forward(&F::upsample_nearest2d(&x, 2)?)?;
        }

        Ok(x)
    }
}

/// What the package records about the autoencoder.
#[derive(Clone, Debug)]
pub struct VaeConfig {
    pub latent_channels: i32,
    /// Widest first, as the encoder counts them; the decoder walks them backwards.
    pub block_out_channels: Vec<i32>,
    pub layers_per_block: i32,
    pub norm_num_groups: i32,
    pub norm_eps: f32,
    /// What a latent is divided by before the model sees it, which is how the two were trained.
    pub scaling_factor: f32,
}

#[derive(Debug)]
pub struct VaeDecoder {
    config: VaeConfig,
    /// What its weights are in, which for SDXL is float32 whatever the rest of the model is in.
    dtype: DType,
    post_quant_conv: Conv2d,
    conv_in: Conv2d,
    mid_resnet0: ResnetBlock,
    mid_attn: AttentionBlock,
    mid_resnet1: ResnetBlock,
    up_blocks: Vec<UpBlock>,
    conv_norm_out: GroupNorm,
    conv_out: Conv2d,
}

impl VaeDecoder {
    pub fn build(config: VaeConfig, vb: &VarBuilder) -> Result<VaeDecoder> {
        if config.block_out_channels.len() < 2 {
            return Err(Error::model("an autoencoder needs at least two rungs"));
        }

        // The decoder starts where the encoder ended, at the deepest channel count.
        let widest = *config.block_out_channels.last().unwrap();

        let mut up_blocks = Vec::new();
        let reversed: Vec<i32> = config.block_out_channels.iter().rev().copied().collect();
        for (index, out_channels) in reversed.iter().enumerate() {
            let in_channels = if index == 0 {
                widest
            } else {
                reversed[index - 1]
            };
            up_blocks.push(UpBlock::build(
                in_channels,
                *out_channels,
                // The last rung is already at full size and has nothing left to double.
                index + 1 < reversed.len(),
                &config,
                &vb.with_name(&format!("up{index}")),
            )?);
        }

        let narrowest = reversed[reversed.len() - 1];
        Ok(VaeDecoder {
            post_quant_conv: Conv2d::build(
                config.latent_channels,
                config.latent_channels,
                1,
                1,
                0,
                &vb.with_name("post_quant_conv"),
            )?,
            conv_in: Conv2d::build(
                config.latent_channels,
                widest,
                3,
                1,
                1,
                &vb.with_name("conv_in"),
            )?,
            mid_resnet0: ResnetBlock::build(widest, widest, &config, &vb.with_name("mid.resnet0"))?,
            mid_attn: AttentionBlock::build(widest, &config, &vb.with_name("mid.attn0"))?,
            mid_resnet1: ResnetBlock::build(widest, widest, &config, &vb.with_name("mid.resnet1"))?,
            up_blocks,
            conv_norm_out: GroupNorm::build(
                narrowest,
                config.norm_num_groups,
                config.norm_eps,
                &vb.with_name("conv_norm_out"),
            )?,
            conv_out: Conv2d::build(narrowest, 3, 3, 1, 1, &vb.with_name("conv_out"))?,
            dtype: vb.float_type(),
            config,
        })
    }

    pub fn config(&self) -> &VaeConfig {
        &self.config
    }

    /// What this decoder computes in. A latent of any float type is cast to it on the way in.
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// The image a latent stands for, as `(1, 3, H * 8, W * 8)` in roughly `[-1, 1]`.
    ///
    /// `latent` is `(1, C, H, W)` as the sampler leaves it, still scaled the way the model was
    /// trained; dividing that back out is the first thing done here. It is cast to whatever this
    /// decoder was built in, so a half latent from the sampler is read by a float32 decoder
    /// without the caller arranging it.
    ///
    /// SDXL's autoencoder has to be built in float32. Its own config says so with `force_upcast`,
    /// and it is not a matter of precision but of range: the activations grow through the up
    /// blocks -- about 84 at the mid block, 570, then 4046 -- and one convolution of the last one
    /// passes 65504, which is as far as half goes. Everything after that would be a NaN.
    pub fn forward(&self, latent: &Tensor) -> Result<Tensor> {
        let dim = latent.dim()?;
        if dim != 4 {
            return Err(Error::model(format!(
                "a latent is (N, C, H, W), got a {dim}-D tensor"
            )));
        }

        let channels = latent.shape_at(1)?;
        if channels != self.config.latent_channels {
            return Err(Error::model(format!(
                "a latent of {channels} channels is not the {} this decoder reads",
                self.config.latent_channels
            )));
        }

        let latent = latent.cast(self.dtype)?;
        let x = F::div_scalar(&latent, self.config.scaling_factor)?;
        let x = self.post_quant_conv.forward(&x)?;
        let x = self.conv_in.forward(&x)?;

        let x = self.mid_resnet0.forward(&x)?;
        let x = self.mid_attn.forward(&x)?;
        let mut x = self.mid_resnet1.forward(&x)?;

        for block in &self.up_blocks {
            x = block.forward(&x)?;
        }

        let x = self.conv_norm_out.forward(&x)?;
        let x = F::silu(&x)?;

        self.conv_out.forward(&x)
    }
}
