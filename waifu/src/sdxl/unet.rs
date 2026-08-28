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

//! The U-Net, which is where all of the work of generating an image happens.
//!
//! It is asked the same question at every step: given this noisy latent, at this much noise, and
//! this prompt, what is the noise in it? Its shape is the name -- the latent is halved twice on
//! the way down, carried across at every resolution, and doubled twice on the way back up -- and
//! the conditioning enters through cross attention in the transformer blocks that sit between the
//! residual blocks at the two smaller resolutions.
//!
//! SDXL is the widest of these: 2.6 billion parameters, ten transformer layers deep at its
//! smallest resolution, and conditioned on the two text encoders side by side.

use crate::error::{Error, Result};
use crate::flint::{functional as F, Tensor};
use crate::layers::{Conv2d, GroupNorm, LayerNorm, Linear};
use crate::var_builder::VarBuilder;

/// The largest period the sinusoidal timestep embedding uses, which is what everything derived
/// from the original DDPM code has used since.
const MAX_PERIOD: f64 = 10000.0;

/// What the package records about the U-Net.
#[derive(Clone, Debug)]
pub struct UnetConfig {
    pub latent_channels: i32,
    /// The width at each resolution, narrowest first. The first entry is also the width of the
    /// timestep embedding before it is projected.
    pub block_out_channels: Vec<i32>,
    /// Residual blocks per resolution on the way down. The way up runs one more, to consume the
    /// residual the downsampling left behind.
    pub layers_per_block: i32,
    /// Transformer layers inside each attention, per resolution. Zero means that resolution has
    /// no attention at all, which is where SDXL saves the most: at full size it is convolutions
    /// only.
    pub transformer_layers_per_block: Vec<i32>,
    /// Attention heads per resolution. The head is what is left over: 64 wide at every one.
    pub num_heads: Vec<i32>,
    pub norm_num_groups: i32,
    /// How wide the conditioning is, which for SDXL is the two text encoders concatenated.
    pub cross_attention_dim: i32,
    /// How wide each of the six numbers describing the image size is embedded.
    pub addition_time_embed_dim: i32,
    /// What the added embedding reads: the pooled text vector and those six numbers embedded.
    pub projection_class_embeddings_input_dim: i32,
}

impl UnetConfig {
    /// The width of the timestep embedding once it has been projected, which is what every
    /// residual block is handed.
    fn time_embed_dim(&self) -> i32 {
        self.block_out_channels[0] * 4
    }

    fn levels(&self) -> usize {
        self.block_out_channels.len()
    }
}

/// The sinusoidal embedding of one timestep, as `(1, dim)` on the host.
///
/// Computed here rather than read from the checkpoint, since it is a formula. In double precision
/// on the CPU, which costs nothing at this size and keeps the frequencies exact -- the highest of
/// them is a ten-thousandth, and rounding it early moves the embedding by more than the model's
/// own precision does.
fn timestep_embedding(timestep: f64, dim: i32) -> Result<Tensor> {
    if dim <= 0 || dim % 2 != 0 {
        return Err(Error::model(format!(
            "a sinusoidal embedding needs an even width, got {dim}"
        )));
    }

    let half = (dim / 2) as usize;
    let mut values = vec![0.0f32; dim as usize];
    for index in 0..half {
        let frequency = (-MAX_PERIOD.ln() * index as f64 / half as f64).exp();
        let angle = timestep * frequency;

        // Cosine first: diffusers calls this flip_sin_to_cos, and every SDXL checkpoint was
        // trained with it on.
        values[index] = angle.cos() as f32;
        values[half + index] = angle.sin() as f32;
    }

    Ok(Tensor::from_f32(&[1, dim], &values)?)
}

/// The other embedding SDXL adds: the size the image was asked for, and where it was cropped
/// from, six numbers in all, each embedded the way a timestep is and laid end to end.
fn time_ids_embedding(time_ids: &[f32], dim: i32) -> Result<Tensor> {
    let mut embedded = Vec::new();
    for id in time_ids {
        embedded.push(timestep_embedding(*id as f64, dim)?);
    }

    let mut out = embedded[0].clone();
    for part in &embedded[1..] {
        out = F::cat(&out, part, -1)?;
    }

    Ok(out)
}

/// Two convolutions, a normalization before each, and the timestep added in between.
///
/// The timestep is how the block is told how much noise it is looking at. It arrives as one
/// vector for the whole image and is added to every pixel, which is the cheapest way a
/// convolution can be conditioned on something that has no position.
#[derive(Debug)]
struct ResnetBlock {
    norm1: GroupNorm,
    conv1: Conv2d,
    time_proj: Linear,
    norm2: GroupNorm,
    conv2: Conv2d,
    shortcut: Option<Conv2d>,
    out_channels: i32,
}

impl ResnetBlock {
    fn build(
        in_channels: i32,
        out_channels: i32,
        config: &UnetConfig,
        vb: &VarBuilder,
    ) -> Result<ResnetBlock> {
        let groups = config.norm_num_groups;
        Ok(ResnetBlock {
            norm1: GroupNorm::build(in_channels, groups, 1e-5, &vb.with_name("norm1"))?,
            conv1: Conv2d::build(in_channels, out_channels, 3, 1, 1, &vb.with_name("conv1"))?,
            time_proj: Linear::build(
                config.time_embed_dim(),
                out_channels,
                true,
                &vb.with_name("time_proj"),
            )?,
            norm2: GroupNorm::build(out_channels, groups, 1e-5, &vb.with_name("norm2"))?,
            conv2: Conv2d::build(out_channels, out_channels, 3, 1, 1, &vb.with_name("conv2"))?,
            shortcut: if in_channels == out_channels {
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
            },
            out_channels,
        })
    }

    /// `input` is `(1, C, H, W)` and `temb` is `(1, time_embed_dim)`.
    fn forward(&self, input: &Tensor, temb: &Tensor) -> Result<Tensor> {
        let x = self.norm1.forward(input)?;
        let x = F::silu(&x)?;
        let x = self.conv1.forward(&x)?;

        // The activation comes before the projection here, not after, which is what the reference
        // does and what the weights were trained for.
        let t = self.time_proj.forward(&F::silu(temb)?)?;
        let x = F::add(&x, &t.view(&[1, self.out_channels, 1, 1])?)?;

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

/// One attention of a transformer block, either over the image itself or over the prompt.
///
/// Self attention takes all three projections from one tensor and so fuses them into one weight.
/// Cross attention reads its keys and values from the conditioning, which is a different width,
/// so only those two fuse and the query stays on its own.
#[derive(Debug)]
struct Attention {
    qkv_proj: Option<Tensor>,
    q_proj: Option<Tensor>,
    kv_proj: Option<Tensor>,
    out_proj: Linear,
    num_heads: i32,
    head_dim: i32,
}

impl Attention {
    fn build(
        channels: i32,
        num_heads: i32,
        cross: Option<i32>,
        vb: &VarBuilder,
    ) -> Result<Attention> {
        if num_heads <= 0 || channels % num_heads != 0 {
            return Err(Error::model(format!(
                "module {:?}: {channels} channels do not divide into {num_heads} heads",
                vb.name()
            )));
        }

        // None of these projections carries a bias, which is what attention_bias=False means.
        let (qkv_proj, q_proj, kv_proj) = match cross {
            None => (
                Some(vb.get("qkv_proj.weight", &[3 * channels, channels])?),
                None,
                None,
            ),
            Some(context_dim) => (
                None,
                Some(vb.get("q_proj.weight", &[channels, channels])?),
                Some(vb.get("kv_proj.weight", &[2 * channels, context_dim])?),
            ),
        };

        Ok(Attention {
            qkv_proj,
            q_proj,
            kv_proj,
            out_proj: Linear::build(channels, channels, true, &vb.with_name("out_proj"))?,
            num_heads,
            head_dim: channels / num_heads,
        })
    }

    /// `(1, L, D)` to `(1, H, L, Dh)`, which is the shape the attention takes.
    fn split_heads(&self, input: &Tensor) -> Result<Tensor> {
        let length = input.shape_at(1)?;
        Ok(input
            .contiguous()?
            .view(&[1, length, self.num_heads, self.head_dim])?
            .transpose(1, 2)?
            .contiguous()?)
    }

    /// `input` is `(1, L, D)`. `context` is what the keys and values come from, which is `input`
    /// itself when this is self attention.
    fn forward(&self, input: &Tensor, context: Option<&Tensor>) -> Result<Tensor> {
        let length = input.shape_at(1)?;
        let channels = self.num_heads * self.head_dim;

        let (q, k, v) = match (&self.qkv_proj, &self.q_proj, &self.kv_proj) {
            (Some(qkv_proj), _, _) => {
                let qkv = F::matmul(input, &qkv_proj.transpose(0, 1)?)?;
                (
                    qkv.slice(2, 0, channels)?,
                    qkv.slice(2, channels, 2 * channels)?,
                    qkv.slice(2, 2 * channels, 3 * channels)?,
                )
            }
            (None, Some(q_proj), Some(kv_proj)) => {
                let context = context.ok_or_else(|| {
                    Error::model("cross attention was called without any conditioning")
                })?;
                let q = F::matmul(input, &q_proj.transpose(0, 1)?)?;
                let kv = F::matmul(context, &kv_proj.transpose(0, 1)?)?;
                (
                    q.slice(2, 0, channels)?,
                    kv.slice(2, 0, channels)?,
                    kv.slice(2, channels, 2 * channels)?,
                )
            }
            _ => {
                return Err(Error::model(
                    "attention has neither a fused nor a split projection",
                ))
            }
        };

        let x = F::attention(
            &self.split_heads(&q)?,
            &self.split_heads(&k)?,
            &self.split_heads(&v)?,
            false,
        )?;
        let x = x
            .transpose(1, 2)?
            .contiguous()?
            .view(&[1, length, channels])?;

        self.out_proj.forward(&x)
    }
}

/// Self attention, then attention over the prompt, then a feed forward, each around a
/// normalization and a residual.
#[derive(Debug)]
struct TransformerBlock {
    norm1: LayerNorm,
    attn1: Attention,
    norm2: LayerNorm,
    attn2: Attention,
    norm3: LayerNorm,
    gate: Linear,
    ff_out: Linear,
}

impl TransformerBlock {
    fn build(
        channels: i32,
        num_heads: i32,
        config: &UnetConfig,
        vb: &VarBuilder,
    ) -> Result<TransformerBlock> {
        Ok(TransformerBlock {
            norm1: LayerNorm::build(channels, 1e-5, &vb.with_name("norm1"))?,
            attn1: Attention::build(channels, num_heads, None, &vb.with_name("attn1"))?,
            norm2: LayerNorm::build(channels, 1e-5, &vb.with_name("norm2"))?,
            attn2: Attention::build(
                channels,
                num_heads,
                Some(config.cross_attention_dim),
                &vb.with_name("attn2"),
            )?,
            norm3: LayerNorm::build(channels, 1e-5, &vb.with_name("norm3"))?,
            // The gated feed forward projects to twice its inner width, half of which gates the
            // other half.
            gate: Linear::build(channels, 8 * channels, true, &vb.with_name("ff.gate.proj"))?,
            ff_out: Linear::build(4 * channels, channels, true, &vb.with_name("ff.out_proj"))?,
        })
    }

    fn forward(&self, input: &Tensor, context: &Tensor) -> Result<Tensor> {
        let x = F::add(
            input,
            &self.attn1.forward(&self.norm1.forward(input)?, None)?,
        )?;
        let x = F::add(
            &x,
            &self
                .attn2
                .forward(&self.norm2.forward(&x)?, Some(context))?,
        )?;

        let feed_forward = self.gate.forward(&self.norm3.forward(&x)?)?;
        let feed_forward = self.ff_out.forward(&F::geglu(&feed_forward)?)?;

        Ok(F::add(&x, &feed_forward)?)
    }
}

/// A stack of transformer blocks that reads an image as a sequence of pixels.
///
/// The image goes in as `(1, C, H, W)` and comes back the same. In between every pixel is one
/// position of a sequence, which is what makes cross attention over the prompt possible at all
/// and what makes this the expensive part of the model.
#[derive(Debug)]
struct Transformer {
    norm: GroupNorm,
    in_proj: Linear,
    blocks: Vec<TransformerBlock>,
    out_proj: Linear,
    channels: i32,
}

impl Transformer {
    fn build(
        channels: i32,
        num_heads: i32,
        depth: i32,
        config: &UnetConfig,
        vb: &VarBuilder,
    ) -> Result<Transformer> {
        let mut blocks = Vec::new();
        for index in 0..depth {
            blocks.push(TransformerBlock::build(
                channels,
                num_heads,
                config,
                &vb.with_name(&format!("block{index}")),
            )?);
        }

        Ok(Transformer {
            norm: GroupNorm::build(
                channels,
                config.norm_num_groups,
                1e-6,
                &vb.with_name("norm"),
            )?,
            in_proj: Linear::build(channels, channels, true, &vb.with_name("in_proj"))?,
            blocks,
            out_proj: Linear::build(channels, channels, true, &vb.with_name("out_proj"))?,
            channels,
        })
    }

    fn forward(&self, input: &Tensor, context: &Tensor) -> Result<Tensor> {
        let shape = input.shape();
        let (height, width) = (shape[2], shape[3]);
        let positions = height * width;

        let x = self.norm.forward(input)?;

        // (1, C, H, W) to (1, H * W, C): a pixel is a position, and its channels are its vector.
        let x = x
            .view(&[1, self.channels, positions])?
            .transpose(1, 2)?
            .contiguous()?;

        let mut x = self.in_proj.forward(&x)?;
        for block in &self.blocks {
            x = block.forward(&x, context)?;
        }
        let x = self.out_proj.forward(&x)?;

        let x = x
            .transpose(1, 2)?
            .contiguous()?
            .view(&[1, self.channels, height, width])?;

        Ok(F::add(input, &x)?)
    }
}

/// One resolution on the way down: residual blocks, each optionally followed by a transformer,
/// and the halving that ends the block.
#[derive(Debug)]
struct DownBlock {
    resnets: Vec<ResnetBlock>,
    attentions: Vec<Transformer>,
    downsample: Option<Conv2d>,
}

impl DownBlock {
    fn build(
        in_channels: i32,
        out_channels: i32,
        num_heads: i32,
        depth: i32,
        downsample: bool,
        config: &UnetConfig,
        vb: &VarBuilder,
    ) -> Result<DownBlock> {
        let mut resnets = Vec::new();
        let mut attentions = Vec::new();
        for index in 0..config.layers_per_block {
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
            if depth > 0 {
                attentions.push(Transformer::build(
                    out_channels,
                    num_heads,
                    depth,
                    config,
                    &vb.with_name(&format!("attn{index}")),
                )?);
            }
        }

        Ok(DownBlock {
            resnets,
            attentions,
            downsample: if downsample {
                // Stride two, which is the halving itself; there is no pooling anywhere here.
                Some(Conv2d::build(
                    out_channels,
                    out_channels,
                    3,
                    2,
                    1,
                    &vb.with_name("downsample0"),
                )?)
            } else {
                None
            },
        })
    }

    /// Runs the block and pushes everything the way up will want onto `residuals`.
    fn forward(
        &self,
        input: &Tensor,
        temb: &Tensor,
        context: &Tensor,
        residuals: &mut Vec<Tensor>,
    ) -> Result<Tensor> {
        let mut x = input.clone();
        for (index, resnet) in self.resnets.iter().enumerate() {
            x = resnet.forward(&x, temb)?;
            if let Some(attention) = self.attentions.get(index) {
                x = attention.forward(&x, context)?;
            }
            residuals.push(x.clone());
        }

        if let Some(downsample) = &self.downsample {
            x = downsample.forward(&x)?;
            residuals.push(x.clone());
        }

        Ok(x)
    }
}

/// One resolution on the way up. Each residual block is handed what the way down left at this
/// resolution, concatenated onto its input, which is the connection the shape is named for.
#[derive(Debug)]
struct UpBlock {
    resnets: Vec<ResnetBlock>,
    attentions: Vec<Transformer>,
    upsample: Option<Conv2d>,
}

impl UpBlock {
    fn build(
        in_channels: i32,
        prev_channels: i32,
        out_channels: i32,
        num_heads: i32,
        depth: i32,
        upsample: bool,
        config: &UnetConfig,
        vb: &VarBuilder,
    ) -> Result<UpBlock> {
        let layers = config.layers_per_block + 1;

        let mut resnets = Vec::new();
        let mut attentions = Vec::new();
        for index in 0..layers {
            // The residuals arrive widest first, and the last one at this resolution is the one
            // the previous level's downsampling produced, which is narrower than the rest.
            let skip = if index == layers - 1 {
                in_channels
            } else {
                out_channels
            };
            let from = if index == 0 {
                prev_channels
            } else {
                out_channels
            };

            resnets.push(ResnetBlock::build(
                from + skip,
                out_channels,
                config,
                &vb.with_name(&format!("resnet{index}")),
            )?);
            if depth > 0 {
                attentions.push(Transformer::build(
                    out_channels,
                    num_heads,
                    depth,
                    config,
                    &vb.with_name(&format!("attn{index}")),
                )?);
            }
        }

        Ok(UpBlock {
            resnets,
            attentions,
            upsample: if upsample {
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
            },
        })
    }

    fn forward(
        &self,
        input: &Tensor,
        temb: &Tensor,
        context: &Tensor,
        residuals: &mut Vec<Tensor>,
    ) -> Result<Tensor> {
        let mut x = input.clone();
        for (index, resnet) in self.resnets.iter().enumerate() {
            let residual = residuals.pop().ok_or_else(|| {
                Error::model("the way up asked for a residual the way down never produced")
            })?;

            x = resnet.forward(&F::cat(&x, &residual, 1)?, temb)?;
            if let Some(attention) = self.attentions.get(index) {
                x = attention.forward(&x, context)?;
            }
        }

        if let Some(upsample) = &self.upsample {
            x = upsample.forward(&F::upsample_nearest2d(&x, 2)?)?;
        }

        Ok(x)
    }
}

/// What one step of sampling is conditioned on, beside the latent itself.
pub struct UnetCondition<'a> {
    /// `(1, L, cross_attention_dim)`: the two text encoders side by side.
    pub context: &'a Tensor,
    /// `(1, 1280)`: the pooled vector of the second encoder.
    pub pooled: &'a Tensor,
    /// The size SDXL is told about: the original height and width, the top and left it was
    /// cropped at, and the height and width being asked for.
    pub time_ids: [f32; 6],
}

#[derive(Debug)]
pub struct Unet {
    config: UnetConfig,
    conv_in: Conv2d,
    time_linear1: Linear,
    time_linear2: Linear,
    add_linear1: Linear,
    add_linear2: Linear,
    down_blocks: Vec<DownBlock>,
    mid_resnet0: ResnetBlock,
    mid_attn: Transformer,
    mid_resnet1: ResnetBlock,
    up_blocks: Vec<UpBlock>,
    conv_norm_out: GroupNorm,
    conv_out: Conv2d,
}

impl Unet {
    pub fn build(config: UnetConfig, vb: &VarBuilder) -> Result<Unet> {
        let levels = config.levels();
        if levels < 2
            || config.transformer_layers_per_block.len() != levels
            || config.num_heads.len() != levels
        {
            return Err(Error::model(
                "the U-Net config disagrees with itself about how many resolutions it has",
            ));
        }

        let widest = config.block_out_channels[levels - 1];
        let time_embed_dim = config.time_embed_dim();

        let mut down_blocks = Vec::new();
        for index in 0..levels {
            let out_channels = config.block_out_channels[index];
            let in_channels = if index == 0 {
                config.block_out_channels[0]
            } else {
                config.block_out_channels[index - 1]
            };
            down_blocks.push(DownBlock::build(
                in_channels,
                out_channels,
                config.num_heads[index],
                config.transformer_layers_per_block[index],
                // The smallest resolution is where the way down ends; there is nothing to halve.
                index + 1 < levels,
                &config,
                &vb.with_name(&format!("down{index}")),
            )?);
        }

        // The way up walks the same resolutions backwards.
        let mut up_blocks = Vec::new();
        let mut prev_channels = widest;
        for index in 0..levels {
            let level = levels - 1 - index;
            let out_channels = config.block_out_channels[level];
            // Which residual the last block at this resolution will be handed: the one the level
            // below left behind, or this level's own where there is no level below.
            let in_channels = config.block_out_channels[level.saturating_sub(1)];

            up_blocks.push(UpBlock::build(
                in_channels,
                prev_channels,
                out_channels,
                config.num_heads[level],
                config.transformer_layers_per_block[level],
                index + 1 < levels,
                &config,
                &vb.with_name(&format!("up{index}")),
            )?);
            prev_channels = out_channels;
        }

        Ok(Unet {
            conv_in: Conv2d::build(
                config.latent_channels,
                config.block_out_channels[0],
                3,
                1,
                1,
                &vb.with_name("conv_in"),
            )?,
            time_linear1: Linear::build(
                config.block_out_channels[0],
                time_embed_dim,
                true,
                &vb.with_name("time_embd.linear1"),
            )?,
            time_linear2: Linear::build(
                time_embed_dim,
                time_embed_dim,
                true,
                &vb.with_name("time_embd.linear2"),
            )?,
            add_linear1: Linear::build(
                config.projection_class_embeddings_input_dim,
                time_embed_dim,
                true,
                &vb.with_name("add_embd.linear1"),
            )?,
            add_linear2: Linear::build(
                time_embed_dim,
                time_embed_dim,
                true,
                &vb.with_name("add_embd.linear2"),
            )?,
            down_blocks,
            mid_resnet0: ResnetBlock::build(widest, widest, &config, &vb.with_name("mid.resnet0"))?,
            mid_attn: Transformer::build(
                widest,
                config.num_heads[levels - 1],
                config.transformer_layers_per_block[levels - 1],
                &config,
                &vb.with_name("mid.attn0"),
            )?,
            mid_resnet1: ResnetBlock::build(widest, widest, &config, &vb.with_name("mid.resnet1"))?,
            up_blocks,
            conv_norm_out: GroupNorm::build(
                config.block_out_channels[0],
                config.norm_num_groups,
                1e-5,
                &vb.with_name("conv_norm_out"),
            )?,
            conv_out: Conv2d::build(
                config.block_out_channels[0],
                config.latent_channels,
                3,
                1,
                1,
                &vb.with_name("conv_out"),
            )?,
            config,
        })
    }

    pub fn config(&self) -> &UnetConfig {
        &self.config
    }

    /// The two embeddings the whole model is conditioned on, added together as one vector.
    fn conditioning(&self, timestep: f32, condition: &UnetCondition<'_>) -> Result<Tensor> {
        let device = condition.pooled.device();
        let dtype = condition.pooled.dtype();

        let sinusoid = timestep_embedding(timestep as f64, self.config.block_out_channels[0])?
            .to_device(device)?
            .cast(dtype)?;
        let temb = self.time_linear1.forward(&sinusoid)?;
        let temb = self.time_linear2.forward(&F::silu(&temb)?)?;

        // The pooled prompt and the six numbers about the image, side by side.
        let sizes = time_ids_embedding(&condition.time_ids, self.config.addition_time_embed_dim)?
            .to_device(device)?
            .cast(dtype)?;
        let added = F::cat(
            &condition
                .pooled
                .view(&[1, condition.pooled.numel() as i32])?,
            &sizes,
            -1,
        )?;

        let added = self.add_linear1.forward(&added)?;
        let added = self.add_linear2.forward(&F::silu(&added)?)?;

        Ok(F::add(&temb, &added)?)
    }

    /// The noise this U-Net believes is in `latent`, which is the same shape as the latent.
    ///
    /// `timestep` says how much noise the latent is supposed to hold, on the 0 to 999 scale the
    /// model was trained on. It is a float rather than an integer because some samplers ask about
    /// points between two training steps.
    pub fn forward(
        &self,
        latent: &Tensor,
        timestep: f32,
        condition: &UnetCondition<'_>,
    ) -> Result<Tensor> {
        let dim = latent.dim()?;
        if dim != 4 {
            return Err(Error::model(format!(
                "a latent is <float16>(N, C, H, W), got a {dim}-D tensor"
            )));
        }

        let channels = latent.shape_at(1)?;
        if channels != self.config.latent_channels {
            return Err(Error::model(format!(
                "a latent of {channels} channels is not the {} this U-Net reads",
                self.config.latent_channels
            )));
        }

        // Every resolution halves the one before it, so an odd size would lose a row on the way
        // down and never get it back.
        let divisor = 1 << (self.config.levels() - 1);
        let (height, width) = (latent.shape_at(2)?, latent.shape_at(3)?);
        if height % divisor != 0 || width % divisor != 0 {
            return Err(Error::model(format!(
                "a {height} by {width} latent does not survive being halved {} times",
                self.config.levels() - 1
            )));
        }

        let temb = self.conditioning(timestep, condition)?;
        let context = condition.context;

        let mut x = self.conv_in.forward(latent)?;

        // What the way down produces at every resolution, for the way up to read back.
        let mut residuals = vec![x.clone()];
        for block in &self.down_blocks {
            x = block.forward(&x, &temb, context, &mut residuals)?;
        }

        x = self.mid_resnet0.forward(&x, &temb)?;
        x = self.mid_attn.forward(&x, context)?;
        x = self.mid_resnet1.forward(&x, &temb)?;

        for block in &self.up_blocks {
            x = block.forward(&x, &temb, context, &mut residuals)?;
        }

        if !residuals.is_empty() {
            return Err(Error::model(format!(
                "the way up left {} residuals unread",
                residuals.len()
            )));
        }

        let x = self.conv_norm_out.forward(&x)?;
        let x = F::silu(&x)?;

        self.conv_out.forward(&x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The sinusoidal embedding is a formula rather than something read from a checkpoint, so it
    /// can be checked against what it is supposed to be without any weights at hand.
    #[test]
    fn the_timestep_embedding_is_cosines_then_sines() {
        let embedding = timestep_embedding(981.0, 8).unwrap();
        assert_eq!(embedding.shape(), vec![1, 8]);

        let values = embedding.to_vec_f32().unwrap();
        for index in 0..4 {
            let frequency = (-MAX_PERIOD.ln() * index as f64 / 4.0).exp();
            let angle = 981.0 * frequency;
            assert!((values[index] as f64 - angle.cos()).abs() < 1e-6);
            assert!((values[4 + index] as f64 - angle.sin()).abs() < 1e-6);
        }
    }

    #[test]
    fn the_timestep_embedding_starts_at_the_lowest_frequency() {
        // The first pair is always the same, whatever the timestep: an angle of one radian per
        // step, so the cosine of the step and its sine.
        let values = timestep_embedding(1.0, 4).unwrap().to_vec_f32().unwrap();
        assert!((values[0] as f64 - 1.0f64.cos()).abs() < 1e-6);
        assert!((values[2] as f64 - 1.0f64.sin()).abs() < 1e-6);
    }

    #[test]
    fn a_timestep_embedding_needs_an_even_width() {
        assert!(timestep_embedding(0.0, 7).is_err());
        assert!(timestep_embedding(0.0, 0).is_err());
    }

    #[test]
    fn the_size_embedding_is_one_timestep_embedding_for_each_number() {
        let ids = [1024.0, 1024.0, 0.0, 0.0, 512.0, 512.0];
        let embedding = time_ids_embedding(&ids, 8).unwrap();
        assert_eq!(embedding.shape(), vec![1, 48]);

        let values = embedding.to_vec_f32().unwrap();
        for (index, id) in ids.iter().enumerate() {
            let one = timestep_embedding(*id as f64, 8)
                .unwrap()
                .to_vec_f32()
                .unwrap();
            assert_eq!(&values[index * 8..(index + 1) * 8], &one[..]);
        }
    }
}
