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

//! The bridge from Qwen3 onto the cross-attention the denoiser was pretrained for.
//!
//! Cosmos-Predict2 was trained against T5, and Anima conditions on Qwen3. This is what stands
//! between them: six blocks whose *queries* are a T5 vocabulary embedded here -- which is why its
//! table is 32128 wide and not the encoder's 151936 -- and whose context is what Qwen3 made of the
//! same prompt. The denoiser never sees the encoder's states directly. It sees only what comes out
//! of this.
//!
//! It is not the denoiser's kind of block. The norms are RMSNorm rather than LayerNorm, there is
//! no modulation at all, the feed forward carries biases, and the attention is bidirectional --
//! a prompt is not a sequence being continued. It rotates the cross-attention's keys too, against
//! the context's own positions, where the denoiser rotates nothing on its cross attention.

use std::fmt;
use std::rc::Rc;

use super::config::AdapterConfig;
use crate::error::{Error, Result};
use crate::flint::{
    check_parameters, DType, Extent, Graph, Held, Ir, ParamSource, RunContext, Tensor, Value,
};
use crate::layers::{Embedding, Linear};

/// The base every rotation in the adapter is built on. Not the encoder's million, and not
/// reachable from the package: the reference fixes it at ten thousand.
const ROPE_THETA: f64 = 10000.0;

/// Cosines and sines for one run's two sequences: the query stream and the context.
struct Rotary {
    cos: Tensor,
    sin: Tensor,
    context_cos: Tensor,
    context_sin: Tensor,
}

/// One table, `(length, heads, head_dim / 2)`, laid out so that its trailing dimensions match the
/// `(1, length, heads, head_dim / 2)` it multiplies.
fn table(
    length: i32,
    heads: i32,
    head_dim: i32,
    device: crate::flint::Device,
) -> Result<(Tensor, Tensor)> {
    let half = (head_dim / 2) as usize;
    let count = length as usize * heads as usize * half;
    let mut cos = Vec::with_capacity(count);
    let mut sin = Vec::with_capacity(count);

    for position in 0..length as usize {
        for _ in 0..heads {
            for index in 0..half {
                let exponent = 2.0 * index as f64 / head_dim as f64;
                let angle = position as f64 / ROPE_THETA.powf(exponent);
                cos.push(angle.cos() as f32);
                sin.push(angle.sin() as f32);
            }
        }
    }

    let shape = [length, heads, head_dim / 2];
    Ok((
        Tensor::from_f32(&shape, &cos)?.to_device(device)?,
        Tensor::from_f32(&shape, &sin)?.to_device(device)?,
    ))
}

/// Rotate the last dimension of `x`, pairing its first half against its second.
fn rotate(g: &Graph, x: Value, cos: Value, sin: Value, half: i32) -> Value {
    let first = g.slice(x, 3, 0, half);
    let second = g.slice(x, 3, half, 2 * half);

    let left = g.sub(g.mul(first, cos), g.mul(second, sin));
    let right = g.add(g.mul(first, sin), g.mul(second, cos));
    g.cat(left, right, 3)
}

/// Split a `(1, L, heads * head_dim)` projection into heads and normalize each one.
fn heads(g: &Graph, x: Value, norm: &str, config: &AdapterConfig, eps: f32) -> Value {
    let length = Extent::of(x, 1);
    let viewed = g.view(
        x,
        [
            Extent::At(1),
            length,
            config.num_heads.into(),
            config.head_dim.into(),
        ],
    );
    let weight = g.subgraph(norm).load(Linear::WEIGHT, &[config.head_dim]);
    g.rms_norm(viewed, weight, eps)
}

/// Attention laid out the way flint wants it, and put back the way the stream wants it.
fn attend(g: &Graph, q: Value, k: Value, v: Value, width: i32) -> Value {
    let length = Extent::of(q, 1);
    let q = g.transpose(q, 1, 2);
    let k = g.transpose(k, 1, 2);
    let v = g.transpose(v, 1, 2);

    // Bidirectional: this is a prompt being read, not continued.
    let out = g.attention(q, k, v, false);
    let out = g.contiguous(g.transpose(out, 1, 2));
    g.view(out, [Extent::At(1), length, width.into()])
}

/// One block, written into `g`.
fn block(
    config: &AdapterConfig,
    g: &Graph,
    input: Value,
    context: Value,
    rope: [Value; 4],
) -> Value {
    let d = config.hidden_size;
    let half = config.head_dim / 2;
    let eps = super::NORM_EPS;
    let [cos, sin, context_cos, context_sin] = rope;

    // -- self attention, over the query stream alone -------------------------------------
    let normed = {
        let weight = g.subgraph("norm_self_attn").load(Linear::WEIGHT, &[d]);
        g.rms_norm(input, weight, eps)
    };
    let qkv = Linear::graph(&g.subgraph("self_attn.qkv_proj"), normed, d, 3 * d, false);
    let q = g.contiguous(g.slice(qkv, 2, 0, d));
    let k = g.contiguous(g.slice(qkv, 2, d, 2 * d));
    let v = g.contiguous(g.slice(qkv, 2, 2 * d, 3 * d));

    let sub = g.subgraph("self_attn");
    let q = rotate(g, heads(&sub, q, "q_norm", config, eps), cos, sin, half);
    let k = rotate(g, heads(&sub, k, "k_norm", config, eps), cos, sin, half);
    let v = g.view(
        v,
        [
            Extent::At(1),
            Extent::of(input, 1),
            config.num_heads.into(),
            config.head_dim.into(),
        ],
    );

    let out = attend(g, q, k, v, d);
    let x = g.add(
        input,
        Linear::graph(&g.subgraph("self_attn.out_proj"), out, d, d, false),
    );

    // -- cross attention, into what the encoder made of the prompt ------------------------
    let normed = {
        let weight = g.subgraph("norm_cross_attn").load(Linear::WEIGHT, &[d]);
        g.rms_norm(x, weight, eps)
    };
    let sub = g.subgraph("cross_attn");
    let q = Linear::graph(&g.subgraph("cross_attn.q_proj"), normed, d, d, false);
    let q = rotate(g, heads(&sub, q, "q_norm", config, eps), cos, sin, half);

    // The key and value are the only things here that read the context, which is why they are one
    // projection and the query is not.
    let kv = Linear::graph(&g.subgraph("cross_attn.kv_proj"), context, d, 2 * d, false);
    let k = g.contiguous(g.slice(kv, 2, 0, d));
    let v = g.contiguous(g.slice(kv, 2, d, 2 * d));
    let k = rotate(
        g,
        heads(&sub, k, "k_norm", config, eps),
        context_cos,
        context_sin,
        half,
    );
    let v = g.view(
        v,
        [
            Extent::At(1),
            Extent::of(context, 1),
            config.num_heads.into(),
            config.head_dim.into(),
        ],
    );

    let out = attend(g, q, k, v, d);
    let x = g.add(
        x,
        Linear::graph(&g.subgraph("cross_attn.out_proj"), out, d, d, false),
    );

    // -- feed forward, which unlike the denoiser's carries biases -------------------------
    let residual = x;
    let normed = {
        let weight = g.subgraph("norm_mlp").load(Linear::WEIGHT, &[d]);
        g.rms_norm(residual, weight, eps)
    };
    let hidden = Linear::graph(&g.subgraph("mlp.fc1"), normed, d, config.mlp_size, true);
    let hidden = g.gelu(hidden);
    let down = Linear::graph(&g.subgraph("mlp.fc2"), hidden, config.mlp_size, d, true);

    g.add(residual, down)
}

/// The whole adapter, written into `g`.
fn write(config: &AdapterConfig, float_type: DType, g: &Graph) {
    let ids = g.input("input_ids");
    let context = g.input("context");
    let rope = [
        g.cast(g.input("rope_cos"), float_type),
        g.cast(g.input("rope_sin"), float_type),
        g.cast(g.input("context_rope_cos"), float_type),
        g.cast(g.input("context_rope_sin"), float_type),
    ];

    // No projection after the embedding: the reference has one only when the widths differ, and
    // here they do not, which is why the package holds no `in_proj`.
    let embedded = Embedding::graph(
        &g.subgraph("embed"),
        ids,
        config.hidden_size,
        config.vocab_size,
        float_type,
    );
    let mut x = g.unsqueeze(embedded, 0);

    for index in 0..config.num_blocks {
        x = block(
            config,
            &g.subgraph(&format!("block{index}")),
            x,
            context,
            rope,
        );
    }

    // The projection comes before the norm, not after it.
    let projected = Linear::graph(
        &g.subgraph("out_proj"),
        x,
        config.hidden_size,
        config.hidden_size,
        true,
    );
    let weight = g
        .subgraph("norm")
        .load(Linear::WEIGHT, &[config.hidden_size]);
    g.output("context", g.rms_norm(projected, weight, super::NORM_EPS));
}

pub struct Adapter {
    config: AdapterConfig,
    ir: Ir,
    held: Held,
    weights: Rc<dyn ParamSource>,
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
        weights: &Rc<dyn ParamSource>,
        float_type: DType,
    ) -> Result<Adapter> {
        let graph = Graph::with_weights(config.weight_format);
        write(&config, float_type, &graph.subgraph(name));
        check_parameters(&graph, weights.as_ref())?;

        let ir = Ir::compile(&graph, weights.residency());
        let held = ir.load(weights.as_ref())?;

        Ok(Adapter {
            weights: Rc::clone(weights),
            held,
            ir,
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
    /// encoder made of the same prompt under its own, `(1, C, D)`. The two lengths are unrelated
    /// and neither has to match the other.
    ///
    /// What comes back is `(1, L, D)`, still the length of the T5 ids. Padding it out to the
    /// denoiser's context length is the pipeline's business, not this one's.
    pub fn forward(&self, input_ids: &Tensor, hidden: &Tensor) -> Result<Tensor> {
        let dim = input_ids.dim()?;
        if dim != 1 {
            return Err(Error::model(format!(
                "the adapter takes token ids as <long>(L), got a {dim}-D tensor"
            )));
        }

        let length = input_ids.shape_at(0)?;
        let context_length = hidden.shape_at(1)?;
        let device = input_ids.device();

        let (cos, sin) = table(length, self.config.num_heads, self.config.head_dim, device)?;
        let (context_cos, context_sin) = table(
            context_length,
            self.config.num_heads,
            self.config.head_dim,
            device,
        )?;
        let rotary = Rotary {
            cos,
            sin,
            context_cos,
            context_sin,
        };

        let run = RunContext::new(&*self.weights)
            .held(&self.held)
            .input("input_ids", input_ids)
            .input("context", hidden)
            .input("rope_cos", &rotary.cos)
            .input("rope_sin", &rotary.sin)
            .input("context_rope_cos", &rotary.context_cos)
            .input("context_rope_sin", &rotary.context_sin);

        let outputs = self.ir.run(&run)?;

        outputs
            .iter()
            .find(|(name, _)| name == "context")
            .map(|(_, tensor)| tensor.clone())
            .ok_or_else(|| Error::model("the adapter produced no context"))
    }
}
