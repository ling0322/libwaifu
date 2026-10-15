// Copyright 2026 The HuggingFace Team. All rights reserved.
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
// Ported from diffusers 0.41.0,
// src/diffusers/models/condition_embedders/condition_embedder_anima.py
// (https://github.com/huggingface/diffusers). Changes from the original: written in Rust as a
// flint graph; the self attention's q, k and v and the cross attention's k and v are each one
// fused projection, as the package stores them; the rotary tables hold half a head rather than
// the duplicated whole; masks, the zero padding to the context length and gradient
// checkpointing are left out (the pipeline pads, and a single prompt needs no mask).

//! `AnimaTextConditioner`: the bridge from Qwen3 onto the cross attention the denoiser was
//! pretrained for.
//!
//! Cosmos-Predict2 was trained against T5, and Anima conditions on Qwen3. This stands between
//! them: six blocks whose queries are T5 token ids embedded here -- which is why its table is
//! 32128 wide and not the encoder's 151936 -- and whose context is what Qwen3 made of the same
//! prompt. The denoiser never sees the encoder's states directly.
//!
//! Unlike the denoiser's blocks there is no modulation, the norms are RMSNorm, the feed forward
//! carries biases, and the cross attention rotates its keys by the context's own positions.

use std::fmt;
use std::rc::Rc;

use super::config::AdapterConfig;
use crate::error::{Error, Result};
use crate::flint::{
    check_parameters, DType, Device, Extent, Graph, Ir, RunContext, Tensor, Value, Weights,
};
use crate::layers::{Embedding, Linear};

/// `AnimaRotaryEmbedding`'s base, which the conditioner does not configure.
const ROPE_THETA: f64 = 10000.0;

/// `AnimaRotaryEmbedding` for positions `0..length`, as `(length, heads, head_dim / 2)` tables.
///
/// diffusers concatenates the frequencies with themselves to cover the whole head; half is enough
/// here because [`apply_rotary`] multiplies each half by the same angles.
fn rotary_embedding(
    length: i32,
    heads: i32,
    head_dim: i32,
    device: Device,
) -> Result<(Tensor, Tensor)> {
    let half = head_dim / 2;
    let inv_freq: Vec<f64> = (0..half)
        .map(|index| 1.0 / ROPE_THETA.powf(2.0 * index as f64 / head_dim as f64))
        .collect();

    let count = (length * heads * half) as usize;
    let mut cos = Vec::with_capacity(count);
    let mut sin = Vec::with_capacity(count);
    for position in 0..length {
        for _ in 0..heads {
            for freq in &inv_freq {
                let angle = position as f64 * freq;
                cos.push(angle.cos() as f32);
                sin.push(angle.sin() as f32);
            }
        }
    }

    let shape = [length, heads, half];
    Ok((
        Tensor::from_f32(&shape, &cos)?.to_device(device)?,
        Tensor::from_f32(&shape, &sin)?.to_device(device)?,
    ))
}

/// `_apply_rotary_pos_emb`: `x * cos + rotate_half(x) * sin` on `(1, L, heads, head_dim)`.
fn apply_rotary(g: &Graph, x: Value, cos: Value, sin: Value, half: i32) -> Value {
    let x1 = g.slice(x, 3, 0, half);
    let x2 = g.slice(x, 3, half, 2 * half);

    // rotate_half(x) is (-x2, x1).
    let first = g.sub(g.mul(x1, cos), g.mul(x2, sin));
    let second = g.add(g.mul(x2, cos), g.mul(x1, sin));
    g.cat(first, second, 3)
}

/// A `(1, L, heads * head_dim)` projection viewed as heads and put through `norm` (`q_norm` or
/// `k_norm`), which normalizes each head on its own.
fn normed_heads(g: &Graph, x: Value, norm: &str, config: &AdapterConfig) -> Value {
    let x = g.view(
        x,
        [
            Extent::At(1),
            Extent::of(x, 1),
            config.num_heads.into(),
            config.head_dim.into(),
        ],
    );
    let weight = g.subgraph(norm).load(Linear::WEIGHT, &[config.head_dim]);
    g.rms_norm(x, weight, super::NORM_EPS)
}

/// `AnimaTextConditionerAttnProcessor` from the projections on: normalize and rotate the query
/// and key, attend without a causal mask, and project back out through `o_proj`.
///
/// `query` reads the positions `rope` holds and `key` the ones `key_rope` holds, which are the
/// same table in the self attention and the context's in the cross attention.
#[allow(clippy::too_many_arguments)]
fn attention(
    config: &AdapterConfig,
    g: &Graph,
    query: Value,
    key: Value,
    value: Value,
    rope: (Value, Value),
    key_rope: (Value, Value),
) -> Value {
    let half = config.head_dim / 2;
    let length = Extent::of(query, 1);

    let query = normed_heads(g, query, "q_norm", config);
    let query = apply_rotary(g, query, rope.0, rope.1, half);
    let key = normed_heads(g, key, "k_norm", config);
    let key = apply_rotary(g, key, key_rope.0, key_rope.1, half);
    let value = g.view(
        value,
        [
            Extent::At(1),
            Extent::of(value, 1),
            config.num_heads.into(),
            config.head_dim.into(),
        ],
    );

    let out = g.attention(
        g.transpose(query, 1, 2),
        g.transpose(key, 1, 2),
        g.transpose(value, 1, 2),
        false,
    );
    let out = g.contiguous(g.transpose(out, 1, 2));
    let out = g.view(out, [Extent::At(1), length, config.hidden_size.into()]);

    let d = config.hidden_size;
    Linear::graph(&g.subgraph("out_proj"), out, d, d, false)
}

/// The `RMSNorm` in front of each sublayer.
fn norm(g: &Graph, name: &str, x: Value, width: i32) -> Value {
    let weight = g.subgraph(name).load(Linear::WEIGHT, &[width]);
    g.rms_norm(x, weight, super::NORM_EPS)
}

/// `AnimaTextConditionerBlock`: self attention, cross attention into `context`, then the MLP,
/// each on an RMSNorm of the stream and added back onto it.
fn block(
    config: &AdapterConfig,
    g: &Graph,
    hidden: Value,
    context: Value,
    rope: (Value, Value),
    context_rope: (Value, Value),
) -> Value {
    let d = config.hidden_size;

    let normed = norm(g, "norm_self_attn", hidden, d);
    let qkv = Linear::graph(&g.subgraph("self_attn.qkv_proj"), normed, d, 3 * d, false);
    let attn = attention(
        config,
        &g.subgraph("self_attn"),
        g.contiguous(g.slice(qkv, 2, 0, d)),
        g.contiguous(g.slice(qkv, 2, d, 2 * d)),
        g.contiguous(g.slice(qkv, 2, 2 * d, 3 * d)),
        rope,
        rope,
    );
    let hidden = g.add(hidden, attn);

    let normed = norm(g, "norm_cross_attn", hidden, d);
    let query = Linear::graph(&g.subgraph("cross_attn.q_proj"), normed, d, d, false);
    let kv = Linear::graph(&g.subgraph("cross_attn.kv_proj"), context, d, 2 * d, false);
    let attn = attention(
        config,
        &g.subgraph("cross_attn"),
        query,
        g.contiguous(g.slice(kv, 2, 0, d)),
        g.contiguous(g.slice(kv, 2, d, 2 * d)),
        rope,
        context_rope,
    );
    let hidden = g.add(hidden, attn);

    let normed = norm(g, "norm_mlp", hidden, d);
    let up = Linear::graph(&g.subgraph("mlp.fc1"), normed, d, config.mlp_size, true);
    let down = Linear::graph(&g.subgraph("mlp.fc2"), g.gelu(up), config.mlp_size, d, true);
    g.add(hidden, down)
}

/// `AnimaTextConditioner.forward`, written into `g`.
fn write(config: &AdapterConfig, float_type: DType, g: &Graph) {
    let d = config.hidden_size;
    let ids = g.input("input_ids");
    let context = g.input("context");
    let rope = (
        g.cast(g.input("rope_cos"), float_type),
        g.cast(g.input("rope_sin"), float_type),
    );
    let context_rope = (
        g.cast(g.input("context_rope_cos"), float_type),
        g.cast(g.input("context_rope_sin"), float_type),
    );

    // `in_proj` is the identity when `model_dim` equals `target_dim`, as it does in every Anima
    // release, so the package holds none.
    let embedded = Embedding::graph(&g.subgraph("embed"), ids, d, config.vocab_size, float_type);
    let mut hidden = g.unsqueeze(embedded, 0);

    for index in 0..config.num_blocks {
        hidden = block(
            config,
            &g.subgraph(&format!("block{index}")),
            hidden,
            context,
            rope,
            context_rope,
        );
    }

    let projected = Linear::graph(&g.subgraph("out_proj"), hidden, d, d, true);
    g.output("context", norm(g, "norm", projected, d));
}

pub struct Adapter {
    config: AdapterConfig,
    ir: Ir,
    weights: Rc<Weights>,
}

impl fmt::Debug for Adapter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Adapter({} instructions)", self.ir.len())
    }
}

impl Adapter {
    pub fn build(
        config: AdapterConfig,
        name: &str,
        weights: &Rc<Weights>,
        float_type: DType,
    ) -> Result<Adapter> {
        let graph = Graph::with_weights(config.weight_format);
        write(&config, float_type, &graph.subgraph(name));
        check_parameters(&graph, weights.as_ref())?;

        Ok(Adapter {
            ir: Ir::compile(&graph),
            weights: Rc::clone(weights),
            config,
        })
    }

    pub fn config(&self) -> &AdapterConfig {
        &self.config
    }

    pub fn ir(&self) -> &Ir {
        &self.ir
    }

    /// `input_ids` is the prompt under the T5 tokenizer, `<long>(L)`; `hidden` is what the text
    /// encoder made of the same prompt under its own, `(1, C, D)`. The two lengths are unrelated.
    ///
    /// What comes back is `(1, L, D)`. Padding it to the denoiser's context length is the
    /// pipeline's business.
    pub fn forward(&self, input_ids: &Tensor, hidden: &Tensor) -> Result<Tensor> {
        let dim = input_ids.dim()?;
        if dim != 1 {
            return Err(Error::model(format!(
                "the adapter takes token ids as <long>(L), got a {dim}-D tensor"
            )));
        }

        let device = input_ids.device();
        let (heads, head_dim) = (self.config.num_heads, self.config.head_dim);
        let (cos, sin) = rotary_embedding(input_ids.shape_at(0)?, heads, head_dim, device)?;
        let (context_cos, context_sin) =
            rotary_embedding(hidden.shape_at(1)?, heads, head_dim, device)?;

        let run = RunContext::new(&*self.weights)
            .input("input_ids", input_ids)
            .input("context", hidden)
            .input("rope_cos", &cos)
            .input("rope_sin", &sin)
            .input("context_rope_cos", &context_cos)
            .input("context_rope_sin", &context_sin);

        self.ir
            .run(&run)?
            .into_iter()
            .find(|(name, _)| name == "context")
            .map(|(_, tensor)| tensor)
            .ok_or_else(|| Error::model("the adapter produced no context"))
    }
}
