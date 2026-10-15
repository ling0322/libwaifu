// Copyright 2025 The NVIDIA Team and The HuggingFace Team. All rights reserved.
// Copyright 2026 Xiaoyang Chen
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Ported from diffusers 0.41.0, src/diffusers/models/transformers/transformer_cosmos.py
// (https://github.com/huggingface/diffusers), the `CosmosTransformer3DModel` configuration
// Anima uses. Changes from the original: written in Rust as a flint graph for one still image
// (one frame, batch one, the all-zero padding mask, no learnable position embedding, no
// controlnet or image context); the self attention's q, k and v and the cross attention's k and
// v are each one fused projection, as the package stores them; the rotary table holds half a
// head rather than the duplicated whole; and the residual stream is kept in float32 whatever
// the sublayers run in.

//! `CosmosTransformer3DModel`: the Cosmos-Predict2 denoiser over patches of a sixteen-channel
//! latent.
//!
//! Each block is a self attention over the picture, a cross attention into the prompt and a
//! feed forward, all three modulated by the timestep. It returns a velocity; turning that into a
//! picture is [`FlowSampler`](super::FlowSampler)'s business.

use std::fmt;
use std::rc::Rc;

use super::config::DitConfig;
use crate::error::{Error, Result};
use crate::flint::{
    check_parameters, functional as F, DType, Device, Extent, Graph, Ir, RunContext, Tensor, Value,
    Weights,
};
use crate::layers::Linear;

/// `CosmosRotaryPosEmbed` for a still image of `rows` by `columns` patches, as
/// `(rows * columns, heads, head_dim / 2)` tables.
///
/// The head is split between time, height and width, in that order, each axis with its own base
/// (see [`DitConfig::rope_bases`]). A still image sits at frame zero, so the temporal angles are
/// all zero: present, with a cosine of one.
fn rotary_pos_embed(
    config: &DitConfig,
    rows: i32,
    columns: i32,
    device: Device,
) -> Result<(Tensor, Tensor)> {
    let (dim_t, dim_h, dim_w) = config.rope_axes();
    let (theta_t, theta_h, theta_w) = config.rope_bases();
    let freqs = |dim: i32, theta: f64| -> Vec<f64> {
        (0..dim / 2)
            .map(|index| 1.0 / theta.powf(2.0 * index as f64 / dim as f64))
            .collect()
    };
    let (freqs_t, freqs_h, freqs_w) = (
        freqs(dim_t, theta_t),
        freqs(dim_h, theta_h),
        freqs(dim_w, theta_w),
    );

    let half = (config.head_dim / 2) as usize;
    let count = (rows * columns * config.num_heads) as usize * half;
    let mut cos = Vec::with_capacity(count);
    let mut sin = Vec::with_capacity(count);
    for row in 0..rows {
        for column in 0..columns {
            let angles: Vec<f64> = freqs_t
                .iter()
                .map(|_| 0.0)
                .chain(freqs_h.iter().map(|f| row as f64 * f))
                .chain(freqs_w.iter().map(|f| column as f64 * f))
                .collect();
            for _ in 0..config.num_heads {
                cos.extend(angles.iter().map(|angle| angle.cos() as f32));
                sin.extend(angles.iter().map(|angle| angle.sin() as f32));
            }
        }
    }

    let shape = [rows * columns, config.num_heads, config.head_dim / 2];
    Ok((
        Tensor::from_f32(&shape, &cos)?.to_device(device)?,
        Tensor::from_f32(&shape, &sin)?.to_device(device)?,
    ))
}

/// `Timesteps(hidden_size, flip_sin_to_cos=True, downscale_freq_shift=0)`: the sinusoid a
/// timestep is projected to, cosines first, as `(1, 1, channels)`.
fn time_proj(timestep: f32, channels: i32, device: Device) -> Result<Tensor> {
    let half = (channels / 2) as usize;
    let mut values = vec![0.0f32; channels as usize];
    for index in 0..half {
        let exponent = -(10000.0f64).ln() * index as f64 / half as f64;
        let angle = timestep as f64 * exponent.exp();
        values[index] = angle.cos() as f32;
        values[half + index] = angle.sin() as f32;
    }

    Ok(Tensor::from_f32(&[1, 1, channels], &values)?.to_device(device)?)
}

/// `CosmosAdaLayerNormZero` (`parts` 3) and `CosmosAdaLayerNorm` (`parts` 2) without the norm:
/// `linear_2(linear_1(silu(embedded_timestep))) + temb`, cut into shift, scale and, with three
/// parts, gate. Each comes back as `(D)`, which broadcasts over every token.
fn ada_modulation(
    config: &DitConfig,
    g: &Graph,
    embedded_timestep: Value,
    temb: Value,
    parts: i32,
) -> Vec<Value> {
    let d = config.hidden_size;
    let lora = config.adaln_lora_dim;

    let hidden = Linear::graph(&g.subgraph("1"), g.silu(embedded_timestep), d, lora, false);
    let out = Linear::graph(&g.subgraph("2"), hidden, lora, parts * d, false);

    // `CosmosAdaLayerNorm` adds only the first two thirds of `temb`.
    let out = g.add(out, g.slice(temb, 2, 0, parts * d));
    let out = g.view(out, [Extent::At(parts * d)]);
    (0..parts)
        .map(|index| g.slice(out, 0, index * d, (index + 1) * d))
        .collect()
}

/// `norm(x) * (1 + scale) + shift`, with `norm` a LayerNorm without weights.
///
/// `x` is the float32 stream and the result is cast to `narrow` only once modulated: the shift a
/// deep block applies is large enough that adding it in half precision overflows.
fn ada_norm(g: &Graph, x: Value, shift: Value, scale: Value, narrow: DType) -> Value {
    let normed = g.layer_norm(x, None, None, super::NORM_EPS);
    let scaled = g.add(g.mul(normed, g.cast(scale, DType::Float)), normed);
    g.cast(g.add(scaled, g.cast(shift, DType::Float)), narrow)
}

/// `hidden_states + gate * out`, multiplied in float32 for the same reason as [`ada_norm`].
fn gated_residual(g: &Graph, x: Value, out: Value, gate: Value) -> Value {
    g.add(
        x,
        g.mul(g.cast(out, DType::Float), g.cast(gate, DType::Float)),
    )
}

/// A `(1, L, heads * head_dim)` projection viewed as heads and, where `norm` names one,
/// put through that per-head RMSNorm (`norm_q`, `norm_k`).
fn heads(g: &Graph, x: Value, norm: Option<&str>, config: &DitConfig) -> Value {
    let x = g.view(
        x,
        [
            Extent::At(1),
            Extent::of(x, 1),
            config.num_heads.into(),
            config.head_dim.into(),
        ],
    );
    match norm {
        Some(norm) => {
            let weight = g.subgraph(norm).load(Linear::WEIGHT, &[config.head_dim]);
            g.rms_norm(x, weight, super::NORM_EPS)
        }
        None => x,
    }
}

/// `apply_rotary_emb(..., use_real=True, use_real_unbind_dim=-2)`: the head's first half is the
/// real part and its second the imaginary.
fn apply_rotary(g: &Graph, x: Value, cos: Value, sin: Value, half: i32) -> Value {
    let real = g.slice(x, 3, 0, half);
    let imag = g.slice(x, 3, half, 2 * half);
    let first = g.sub(g.mul(real, cos), g.mul(imag, sin));
    let second = g.add(g.mul(imag, cos), g.mul(real, sin));
    g.cat(first, second, 3)
}

/// `CosmosAttnProcessor2_0` from the normalized and rotated heads on: attend, merge the heads and
/// project out through `to_out`.
fn attention(g: &Graph, query: Value, key: Value, value: Value, config: &DitConfig) -> Value {
    let length = Extent::of(query, 1);
    let out = g.attention(
        g.contiguous(g.transpose(query, 1, 2)),
        g.contiguous(g.transpose(key, 1, 2)),
        g.contiguous(g.transpose(value, 1, 2)),
        false,
    );
    let out = g.contiguous(g.transpose(out, 1, 2));
    let out = g.view(out, [Extent::At(1), length, config.hidden_size.into()]);

    let d = config.hidden_size;
    Linear::graph(&g.subgraph("out_proj"), out, d, d, false)
}

/// `CosmosTransformerBlock`, without the controlnet and image context branches.
#[allow(clippy::too_many_arguments)]
fn block(
    config: &DitConfig,
    g: &Graph,
    x: Value,
    context: Value,
    embedded_timestep: Value,
    temb: Value,
    rope: (Value, Value),
    narrow: DType,
) -> Value {
    let d = config.hidden_size;
    let half = config.head_dim / 2;

    // 1. Self attention, rotated by each patch's position.
    let norm1 = ada_modulation(
        config,
        &g.subgraph("adaln_self_attn"),
        embedded_timestep,
        temb,
        3,
    );
    let normed = ada_norm(g, x, norm1[0], norm1[1], narrow);
    let attn1 = g.subgraph("self_attn");
    let qkv = Linear::graph(&attn1.subgraph("qkv_proj"), normed, d, 3 * d, false);
    let query = heads(
        &attn1,
        g.contiguous(g.slice(qkv, 2, 0, d)),
        Some("q_norm"),
        config,
    );
    let key = heads(
        &attn1,
        g.contiguous(g.slice(qkv, 2, d, 2 * d)),
        Some("k_norm"),
        config,
    );
    let value = heads(
        &attn1,
        g.contiguous(g.slice(qkv, 2, 2 * d, 3 * d)),
        None,
        config,
    );
    let query = apply_rotary(g, query, rope.0, rope.1, half);
    let key = apply_rotary(g, key, rope.0, rope.1, half);
    let out = attention(&attn1, query, key, value, config);
    let x = gated_residual(g, x, out, norm1[2]);

    // 2. Cross attention into the prompt, which is not rotated. Its keys and values read the
    // adapter's width, `context_dim`, rather than this model's.
    let norm2 = ada_modulation(
        config,
        &g.subgraph("adaln_cross_attn"),
        embedded_timestep,
        temb,
        3,
    );
    let normed = ada_norm(g, x, norm2[0], norm2[1], narrow);
    let attn2 = g.subgraph("cross_attn");
    let query = Linear::graph(&attn2.subgraph("q_proj"), normed, d, d, false);
    let kv = Linear::graph(
        &attn2.subgraph("kv_proj"),
        context,
        config.context_dim,
        2 * d,
        false,
    );
    let query = heads(&attn2, query, Some("q_norm"), config);
    let key = heads(
        &attn2,
        g.contiguous(g.slice(kv, 2, 0, d)),
        Some("k_norm"),
        config,
    );
    let value = heads(&attn2, g.contiguous(g.slice(kv, 2, d, 2 * d)), None, config);
    let out = attention(&attn2, query, key, value, config);
    let x = gated_residual(g, x, out, norm2[2]);

    // 3. Feed forward: `FeedForward(activation_fn="gelu", bias=False)`.
    let norm3 = ada_modulation(config, &g.subgraph("adaln_mlp"), embedded_timestep, temb, 3);
    let normed = ada_norm(g, x, norm3[0], norm3[1], narrow);
    let up = Linear::graph(&g.subgraph("mlp.layer1"), normed, d, config.mlp_size, false);
    let down = Linear::graph(
        &g.subgraph("mlp.layer2"),
        g.gelu(up),
        config.mlp_size,
        d,
        false,
    );
    gated_residual(g, x, down, norm3[2])
}

/// `CosmosTransformer3DModel.forward` from the patch embedding to `proj_out`, written into `g`.
fn write(config: &DitConfig, float_type: DType, g: &Graph) {
    let d = config.hidden_size;
    let tokens = g.input("tokens");
    let context = g.input("context");
    let timesteps_proj = g.cast(g.input("timesteps_proj"), float_type);
    let rope = (
        g.cast(g.input("rope_cos"), float_type),
        g.cast(g.input("rope_sin"), float_type),
    );

    // `patch_embed.proj`, onto the float32 residual stream.
    let embedder = g.subgraph("x_embedder");
    let patches = Linear::graph(&embedder, tokens, config.patchify_channels, d, false);
    let mut x = g.cast(patches, DType::Float);

    // `CosmosEmbedding`: `temb` is the timestep's projection through `t_embedder`, and what the
    // blocks modulate on is `embedded_timestep`, the RMSNorm of the sinusoid itself.
    let hidden = Linear::graph(&g.subgraph("t_embedder.1"), timesteps_proj, d, d, false);
    let temb = Linear::graph(&g.subgraph("t_embedder.2"), g.silu(hidden), d, 3 * d, false);
    let embedded_timestep = {
        let weight = g.subgraph("t_norm").load(Linear::WEIGHT, &[d]);
        g.rms_norm(timesteps_proj, weight, super::NORM_EPS)
    };

    for index in 0..config.num_blocks {
        x = block(
            config,
            &g.subgraph(&format!("block{index}")),
            x,
            context,
            embedded_timestep,
            temb,
            rope,
            float_type,
        );
    }

    // `norm_out` and `proj_out`.
    let final_g = g.subgraph("final");
    let norm_out = ada_modulation(
        config,
        &final_g.subgraph("adaln"),
        embedded_timestep,
        temb,
        2,
    );
    let x = ada_norm(g, x, norm_out[0], norm_out[1], float_type);
    let out = config.latent_channels * config.patch_size * config.patch_size;
    g.output(
        "velocity",
        Linear::graph(&final_g.subgraph("linear"), x, d, out, false),
    );
}

pub struct Dit {
    config: DitConfig,
    ir: Ir,
    weights: Rc<Weights>,
    float_type: DType,
}

impl fmt::Debug for Dit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Dit({} instructions)", self.ir.len())
    }
}

impl Dit {
    pub fn build(
        config: DitConfig,
        name: &str,
        weights: &Rc<Weights>,
        float_type: DType,
    ) -> Result<Dit> {
        let graph = Graph::with_weights(config.weight_format);
        write(&config, float_type, &graph.subgraph(name));
        check_parameters(&graph, weights.as_ref())?;

        Ok(Dit {
            ir: Ir::compile(&graph),
            weights: Rc::clone(weights),
            config,
            float_type,
        })
    }

    pub fn config(&self) -> &DitConfig {
        &self.config
    }

    pub fn ir(&self) -> &Ir {
        &self.ir
    }

    /// `CosmosPatchEmbed` before its projection, with the padding mask concatenated first.
    ///
    /// `(1, C, H, W)` becomes `(1, H / p * W / p, (C + 1) * p * p)`, the channel leading within
    /// each patch. The mask is all zeros for every image drawn here. The batch is left out of
    /// the shuffle: it is always one, and `contiguous` copies at most five dimensions.
    fn patchify(&self, latent: &Tensor) -> Result<Tensor> {
        let patch = self.config.patch_size;
        let (height, width) = (latent.shape_at(2)?, latent.shape_at(3)?);
        let (rows, columns) = (height / patch, width / patch);

        let mask = Tensor::zeros(&[1, 1, height, width], latent.dtype(), latent.device())?;
        let x = F::cat(latent, &mask, 1)?;
        let x = x.view(&[self.config.latent_channels + 1, rows, patch, columns, patch])?;

        // (c, rows, ph, columns, pw) -> (rows, columns, c, ph, pw)
        let x = x.transpose(0, 1)?.transpose(1, 3)?.transpose(2, 3)?;
        Ok(x.contiguous()?
            .view(&[1, rows * columns, self.config.patchify_channels])?)
    }

    /// The unpatchify at the end of `CosmosTransformer3DModel.forward`.
    ///
    /// `proj_out` writes each patch as `(ph, pw, c)`, with the channel trailing, so this is not
    /// the inverse of [`Dit::patchify`].
    fn unpatchify(&self, tokens: &Tensor, height: i32, width: i32) -> Result<Tensor> {
        let patch = self.config.patch_size;
        let channels = self.config.latent_channels;
        let (rows, columns) = (height / patch, width / patch);

        // (rows, columns, ph, pw, c) -> (c, rows, ph, columns, pw)
        let x = tokens.view(&[rows, columns, patch, patch, channels])?;
        let x = x.transpose(0, 4)?.transpose(1, 4)?.transpose(3, 4)?;
        Ok(x.contiguous()?.view(&[1, channels, height, width])?)
    }

    /// One step's velocity for `latent` at `timestep`, conditioned on `context`.
    ///
    /// `latent` is `(1, C, H, W)` with both sides a multiple of the patch size; `timestep` is the
    /// sigma itself, in `0..=1`, the scheduler's timestep over its thousand training steps as
    /// diffusers' Anima pipeline passes it. `context` is `(1, 512, D)` from the adapter.
    pub fn forward(&self, latent: &Tensor, timestep: f32, context: &Tensor) -> Result<Tensor> {
        let dim = latent.dim()?;
        if dim != 4 {
            return Err(Error::model(format!(
                "the denoiser takes a latent as (1, C, H, W), got a {dim}-D tensor"
            )));
        }

        let (height, width) = (latent.shape_at(2)?, latent.shape_at(3)?);
        let patch = self.config.patch_size;
        if height % patch != 0 || width % patch != 0 {
            return Err(Error::model(format!(
                "a {height} by {width} latent does not divide into {patch} by {patch} patches"
            )));
        }

        let device = latent.device();
        let tokens = self.patchify(latent)?;
        let timesteps_proj = time_proj(timestep, self.config.hidden_size, device)?;
        let (cos, sin) = rotary_pos_embed(&self.config, height / patch, width / patch, device)?;

        let run = RunContext::new(&*self.weights)
            .input("tokens", &tokens)
            .input("context", context)
            .input("timesteps_proj", &timesteps_proj)
            .input("rope_cos", &cos)
            .input("rope_sin", &sin);

        let velocity = self
            .ir
            .run(&run)?
            .into_iter()
            .find(|(name, _)| name == "velocity")
            .map(|(_, tensor)| tensor)
            .ok_or_else(|| Error::model("the denoiser produced no velocity"))?;

        self.unpatchify(&velocity, height, width)
    }

    pub fn float_type(&self) -> DType {
        self.float_type
    }
}
