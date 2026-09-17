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

//! Qwen3-0.6B, read for its hidden states rather than for another token.
//!
//! Anima conditions on a language model instead of on CLIP, and this is that model: twenty-eight
//! decoder layers, grouped query attention with sixteen query heads against eight for the keys,
//! an RMSNorm on each head's queries and keys before they are rotated, and a SwiGLU feed forward.
//! It runs once over the whole prompt and what comes back is every position's final hidden state,
//! which [`Adapter`](super::Adapter) then cross-attends into.
//!
//! It is still a causal model and is run as one. Nothing here generates -- there is no cache and
//! no lm_head, and the token after the prompt is of no interest -- but the mask is the mask it was
//! trained with, and taking it off would change every state it produces.
//!
//! Written as a [`Graph`] in the same way as [`ClipTextEncoder`](crate::ClipTextEncoder), for the
//! same reasons. The one thing it needs that a graph cannot hold is the rotary table: it depends
//! on how long the prompt is, and a graph is compiled once, so the cosines and sines are computed
//! per call and handed in as inputs.

use std::fmt;
use std::rc::Rc;

use super::config::TextConfig;
use crate::error::{Error, Result};
use crate::flint::{
    check_parameters, DType, Extent, Graph, Ir, ParamSource, RunContext, Tensor, Value,
};
use crate::layers::{Embedding, Linear};

/// The cosines and sines of one prompt's positions, as the graph takes them.
///
/// Laid out `(length, heads, head_dim / 2)` rather than `(length, head_dim / 2)` because a binary
/// operation here broadcasts the right operand over the *leading* dimensions of the left: the
/// trailing ones have to match, and the left is `(1, length, heads, head_dim / 2)`.
struct Rotary {
    cos: Tensor,
    sin: Tensor,
}

impl Rotary {
    /// Neither the base nor the pairing is Anima's own: this is the table Qwen3 was trained with,
    /// and the denoiser's is built differently, out of three axes and a different base. They both
    /// rotate `x[i]` against `x[i + half]`, and that is all they share.
    fn build(config: &TextConfig, length: i32, device: crate::flint::Device) -> Result<Rotary> {
        let half = (config.head_dim / 2) as usize;
        let heads = config.num_heads as usize;
        let count = length as usize * heads * half;

        let mut cos = Vec::with_capacity(count);
        let mut sin = Vec::with_capacity(count);
        for position in 0..length as usize {
            for _ in 0..heads {
                for index in 0..half {
                    let exponent = 2.0 * index as f64 / config.head_dim as f64;
                    let angle = position as f64 / (config.rope_theta as f64).powf(exponent);
                    cos.push(angle.cos() as f32);
                    sin.push(angle.sin() as f32);
                }
            }
        }

        let shape = [length, config.num_heads, config.head_dim / 2];
        Ok(Rotary {
            cos: Tensor::from_f32(&shape, &cos)?.to_device(device)?,
            sin: Tensor::from_f32(&shape, &sin)?.to_device(device)?,
        })
    }
}

/// Rotate the last dimension of `x`, pairing its first half against its second.
fn rotate(g: &Graph, x: Value, cos: Value, sin: Value, half: i32) -> Value {
    let first = g.slice(x, 3, 0, half);
    let second = g.slice(x, 3, half, 2 * half);

    let left = g.sub(g.mul(first, cos), g.mul(second, sin));
    let right = g.add(g.mul(first, sin), g.mul(second, cos));
    g.cat(left, right, 3)
}

/// One decoder layer, written into `g`. `input` is `(1, L, D)` and so is what comes back.
fn layer(config: &TextConfig, g: &Graph, input: Value, cos: Value, sin: Value) -> Value {
    let d = config.hidden_size;
    let head_dim = config.head_dim;
    let half = head_dim / 2;
    let queries = config.num_heads * head_dim;
    let keys = config.num_kv_heads * head_dim;
    let length = Extent::of(input, 1);

    let x = {
        let weight = g.subgraph("input_norm").load(Linear::WEIGHT, &[d]);
        g.rms_norm(input, weight, config.norm_eps)
    };

    // One projection for all three. They sit end to end along the width, and the two halves of
    // the key and value are narrower than the query's because the attention is grouped.
    let qkv = Linear::graph(
        &g.subgraph("attn.qkv_proj"),
        x,
        d,
        queries + 2 * keys,
        false,
    );
    // Made whole before they are reshaped: a slice of the fused projection is a view with a gap
    // in it, and a norm over the head dimension wants its rows where it can reach them.
    let q = g.contiguous(g.slice(qkv, 2, 0, queries));
    let k = g.contiguous(g.slice(qkv, 2, queries, queries + keys));
    let v = g.contiguous(g.slice(qkv, 2, queries + keys, queries + 2 * keys));

    let q = g.view(
        q,
        [
            Extent::At(1),
            length,
            config.num_heads.into(),
            head_dim.into(),
        ],
    );
    let k = g.view(
        k,
        [
            Extent::At(1),
            length,
            config.num_kv_heads.into(),
            head_dim.into(),
        ],
    );
    let v = g.view(
        v,
        [
            Extent::At(1),
            length,
            config.num_kv_heads.into(),
            head_dim.into(),
        ],
    );

    // The norm is over one head rather than the whole width, which is what makes it a norm of the
    // head dimension and why the weights are 128 long and not 1024.
    let q = {
        let weight = g.subgraph("attn.q_norm").load(Linear::WEIGHT, &[head_dim]);
        g.rms_norm(q, weight, config.norm_eps)
    };
    let k = {
        let weight = g.subgraph("attn.k_norm").load(Linear::WEIGHT, &[head_dim]);
        g.rms_norm(k, weight, config.norm_eps)
    };

    // The key's table is the query's, narrowed to the heads it has.
    let k_cos = g.slice(cos, 1, 0, config.num_kv_heads);
    let k_sin = g.slice(sin, 1, 0, config.num_kv_heads);
    let q = rotate(g, q, cos, sin, half);
    let k = rotate(g, k, k_cos, k_sin, half);

    // What `attention` wants is (batch, heads, length, head_dim), and it takes the keys and
    // values already narrow: grouped query attention needs no expansion here.
    let q = g.transpose(q, 1, 2);
    let k = g.transpose(k, 1, 2);
    let v = g.transpose(v, 1, 2);
    let out = g.attention(q, k, v, true);
    // Transposing back leaves the heads interleaved rather than laid end to end, and a view
    // cannot see past that, so it is made whole first.
    let out = g.contiguous(g.transpose(out, 1, 2));
    let out = g.view(out, [Extent::At(1), length, queries.into()]);

    let x = g.add(
        input,
        Linear::graph(&g.subgraph("attn.out_proj"), out, queries, d, false),
    );

    let residual = x;
    let normed = {
        let weight = g.subgraph("post_attn_norm").load(Linear::WEIGHT, &[d]);
        g.rms_norm(residual, weight, config.norm_eps)
    };

    // The gate and the projection are one weight, gate first, because that is the half `swiglu`
    // puts the swish on.
    let gated = Linear::graph(
        &g.subgraph("mlp.gate_up_proj"),
        normed,
        d,
        2 * config.mlp_size,
        false,
    );
    let hidden = g.swiglu(gated);
    let down = Linear::graph(
        &g.subgraph("mlp.down_proj"),
        hidden,
        config.mlp_size,
        d,
        false,
    );

    g.add(residual, down)
}

/// The whole encoder, written into `g`.
fn write(config: &TextConfig, float_type: DType, g: &Graph) {
    let ids = g.input("input_ids");

    // Built in float32, because the angles are worked out on the host and a cosine deserves the
    // width; cast once here rather than per layer, since what they multiply is whatever the pass
    // runs in and a binary operation will not mix the two.
    let cos = g.cast(g.input("rope_cos"), float_type);
    let sin = g.cast(g.input("rope_sin"), float_type);

    let embedded = Embedding::graph(
        &g.subgraph("embed"),
        ids,
        config.hidden_size,
        config.vocab_size,
        float_type,
    );
    let mut x = g.unsqueeze(embedded, 0);

    for index in 0..config.num_layers {
        x = layer(config, &g.subgraph(&format!("block{index}")), x, cos, sin);
    }

    let weight = g
        .subgraph("final_norm")
        .load(Linear::WEIGHT, &[config.hidden_size]);
    g.output("hidden", g.rms_norm(x, weight, config.norm_eps));
}

pub struct TextEncoder {
    config: TextConfig,
    ir: Ir,
    weights: Rc<dyn ParamSource>,
}

impl fmt::Debug for TextEncoder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "TextEncoder({} instructions)", self.ir.len())
    }
}

impl TextEncoder {
    /// `name` is the namespace the package holds this encoder under.
    pub fn build(
        config: TextConfig,
        name: &str,
        weights: &Rc<dyn ParamSource>,
        float_type: DType,
    ) -> Result<TextEncoder> {
        if config.num_heads % config.num_kv_heads != 0 {
            return Err(Error::model(format!(
                "{} query heads do not group evenly into {} key heads",
                config.num_heads, config.num_kv_heads
            )));
        }

        let graph = Graph::with_weights(config.weight_format);
        write(&config, float_type, &graph.subgraph(name));
        check_parameters(&graph, weights.as_ref())?;

        Ok(TextEncoder {
            weights: Rc::clone(weights),
            ir: Ir::compile(&graph),
            config,
        })
    }

    pub fn config(&self) -> &TextConfig {
        &self.config
    }

    /// The pass this encoder runs, for printing.
    pub fn ir(&self) -> &Ir {
        &self.ir
    }

    /// `input_ids` is `<long>(L)`. What comes back is `(1, L, D)`: every position's final state,
    /// because the adapter attends into all of them rather than reading one.
    pub fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        let dim = input_ids.dim()?;
        if dim != 1 {
            return Err(Error::model(format!(
                "a text encoder takes token ids as <long>(L), got a {dim}-D tensor"
            )));
        }

        let length = input_ids.shape_at(0)?;
        let rotary = Rotary::build(&self.config, length, input_ids.device())?;

        let run = RunContext::new(&*self.weights)
            .input("input_ids", input_ids)
            .input("rope_cos", &rotary.cos)
            .input("rope_sin", &rotary.sin);

        let outputs = self.ir.run(&run)?;

        outputs
            .iter()
            .find(|(name, _)| name == "hidden")
            .map(|(_, tensor)| tensor.clone())
            .ok_or_else(|| Error::model("the encoder produced no hidden states"))
    }
}
