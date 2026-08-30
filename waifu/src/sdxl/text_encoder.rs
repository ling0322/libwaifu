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

//! CLIP's text encoder, which is what turns a prompt into what the U-Net is conditioned on.
//!
//! SDXL runs two of these side by side and concatenates what they produce: a 768 wide one whose
//! activation is the sigmoid approximation OpenAI's CLIP was trained with, and a 1280 wide one
//! that uses the ordinary GELU and also carries the projection SDXL takes its pooled conditioning
//! from. Everything else about them is the same, so one type covers both.

use crate::error::{Error, Result};
use crate::flint::{functional as F, Tensor};
use crate::layers::{Embedding, LayerNorm, Linear};
use crate::var_builder::VarBuilder;

/// What separates the two encoders, and what the package records for each.
#[derive(Clone, Copy, Debug)]
pub struct ClipTextConfig {
    pub hidden_size: i32,
    pub intermediate_size: i32,
    pub num_layers: i32,
    pub num_heads: i32,
    /// How many positions the embedding table holds, which is the hard limit on a prompt: there
    /// simply is no vector for position 78.
    pub context_length: i32,
    pub vocab_size: i32,
    /// True for the sigmoid approximation, false for the ordinary GELU.
    pub quick_gelu: bool,
    pub norm_eps: f32,
    /// The id whose position the pooled output is read from.
    pub eot_token_id: i32,
}

/// One transformer layer: attention over what came before, then a feed forward, each around a
/// normalization and a residual.
#[derive(Debug)]
struct ClipLayer {
    input_norm: LayerNorm,
    qkv_proj: Linear,
    out_proj: Linear,
    post_attn_norm: LayerNorm,
    fc1: Linear,
    fc2: Linear,
}

impl ClipLayer {
    fn build(config: &ClipTextConfig, vb: &VarBuilder) -> Result<ClipLayer> {
        let d = config.hidden_size;
        Ok(ClipLayer {
            input_norm: LayerNorm::build(d, config.norm_eps, &vb.with_name("input_norm"))?,
            qkv_proj: Linear::build(d, 3 * d, true, &vb.with_name("attn.qkv_proj"))?,
            out_proj: Linear::build(d, d, true, &vb.with_name("attn.out_proj"))?,
            post_attn_norm: LayerNorm::build(d, config.norm_eps, &vb.with_name("post_attn_norm"))?,
            fc1: Linear::build(d, config.intermediate_size, true, &vb.with_name("mlp.fc1"))?,
            fc2: Linear::build(config.intermediate_size, d, true, &vb.with_name("mlp.fc2"))?,
        })
    }

    /// `input` is `(1, L, D)`, and so is what comes back.
    fn forward(&self, config: &ClipTextConfig, input: &Tensor) -> Result<Tensor> {
        let x = self.input_norm.forward(input)?;
        let x = self.attention(config, &x)?;
        let x = F::add(input, &x)?;

        let residual = x;
        let x = self.post_attn_norm.forward(&residual)?;
        let x = self.fc1.forward(&x)?;
        let x = if config.quick_gelu {
            F::quick_gelu(&x)?
        } else {
            F::gelu(&x)?
        };
        let x = self.fc2.forward(&x)?;

        Ok(F::add(&residual, &x)?)
    }

    fn attention(&self, config: &ClipTextConfig, input: &Tensor) -> Result<Tensor> {
        let length = input.shape_at(1)?;
        let d = config.hidden_size;
        let head_dim = d / config.num_heads;

        // One projection produces all three; they sit end to end along the width.
        let qkv = self.qkv_proj.forward(input)?;
        let mut parts = Vec::with_capacity(3);
        for index in 0..3 {
            let part = qkv.slice(2, index * d, (index + 1) * d)?;
            // (1, L, D) to the (1, H, L, Dh) the attention wants.
            parts.push(
                part.contiguous()?
                    .view(&[1, length, config.num_heads, head_dim])?
                    .transpose(1, 2)?
                    .contiguous()?,
            );
        }

        // A text encoder reads left to right, so a position may not see what follows it.
        let x = F::attention(&parts[0], &parts[1], &parts[2], true)?;
        let x = x.transpose(1, 2)?.contiguous()?.view(&[1, length, d])?;

        self.out_proj.forward(&x)
    }
}

/// What one encoder produces for a prompt.
pub struct ClipTextOutput {
    /// `(1, L, D)`, taken from the layer before the last one: SDXL conditions on that rather than
    /// on the final output, which is what a clip skip of two means.
    pub hidden: Tensor,
    /// `(1, projection)`, read at the end-of-text position and put through the projection. Only
    /// the second encoder has one; the first hands back the hidden state at that position.
    pub pooled: Tensor,
}

#[derive(Debug)]
pub struct ClipTextEncoder {
    config: ClipTextConfig,
    token_embedding: Embedding,
    position_embedding: Tensor,
    layers: Vec<ClipLayer>,
    final_norm: LayerNorm,
    text_projection: Option<Tensor>,
}

impl ClipTextEncoder {
    pub fn build(config: ClipTextConfig, vb: &VarBuilder) -> Result<ClipTextEncoder> {
        let mut layers = Vec::with_capacity(config.num_layers as usize);
        for index in 0..config.num_layers {
            layers.push(ClipLayer::build(
                &config,
                &vb.with_name(&format!("block{index}")),
            )?);
        }

        let text_projection = if vb.has("text_proj.weight") {
            Some(vb.get(
                "text_proj.weight",
                &[config.hidden_size, config.hidden_size],
            )?)
        } else {
            None
        };

        Ok(ClipTextEncoder {
            token_embedding: Embedding::build(
                config.hidden_size,
                config.vocab_size,
                &vb.with_name("token_embd"),
            )?,
            position_embedding: vb.get_widened(
                "position_embd.weight",
                &[config.context_length, config.hidden_size],
            )?,
            layers,
            final_norm: LayerNorm::build(
                config.hidden_size,
                config.norm_eps,
                &vb.with_name("final_norm"),
            )?,
            text_projection,
            config,
        })
    }

    pub fn config(&self) -> &ClipTextConfig {
        &self.config
    }

    /// `input_ids` is `<long>(L)`, already wrapped in its markers and padded out to the context
    /// length by whoever built it.
    pub fn forward(&self, input_ids: &Tensor) -> Result<ClipTextOutput> {
        let dim = input_ids.dim()?;
        if dim != 1 {
            return Err(Error::model(format!(
                "a text encoder takes token ids as <long>(L), got a {dim}-D tensor"
            )));
        }

        let length = input_ids.shape_at(0)?;
        if length > self.config.context_length {
            return Err(Error::model(format!(
                "{length} tokens is past the {} positions this encoder holds",
                self.config.context_length
            )));
        }

        // Where a token sits is as much of its meaning as which token it is, and the two are
        // simply added.
        let embedded = self.token_embedding.forward(input_ids)?;
        let positions = self.position_embedding.slice(0, 0, length)?;
        let mut x = F::add(&embedded, &positions)?.unsqueeze(0)?;

        // The layer before the last is what SDXL conditions on, so it is kept as it goes past.
        let mut penultimate = None;
        for (index, layer) in self.layers.iter().enumerate() {
            x = layer.forward(&self.config, &x)?;
            if index + 2 == self.layers.len() {
                penultimate = Some(x.clone());
            }
        }

        let hidden = penultimate.ok_or_else(|| {
            Error::model("a text encoder needs at least two layers to have a penultimate one")
        })?;

        // The pooled vector is read at the end-of-text marker, which is the last thing the prompt
        // said and therefore the only position that has seen all of it.
        let last = self.final_norm.forward(&x)?;
        let eot = self.eot_position(input_ids)?;
        let pooled = last.subtensor(0)?.slice(0, eot, eot + 1)?.contiguous()?;

        let pooled = match &self.text_projection {
            Some(projection) => F::matmul(&pooled, &projection.transpose(0, 1)?)?,
            None => pooled,
        };

        Ok(ClipTextOutput { hidden, pooled })
    }

    /// The first end-of-text marker, which is where the prompt stopped and the padding began.
    fn eot_position(&self, input_ids: &Tensor) -> Result<i32> {
        let ids = input_ids.to_vec_i64()?;
        ids.iter()
            .position(|id| *id as i32 == self.config.eot_token_id)
            .map(|index| index as i32)
            .ok_or_else(|| Error::model("the token ids hold no end-of-text marker"))
    }
}
