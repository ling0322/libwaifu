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

//! The Qwen3-VL text tower, read for twelve of its hidden states rather than for another token.
//!
//! Thirty-six decoder layers, grouped query attention with thirty-two query heads against eight
//! for the keys, an RMSNorm on each head's queries and keys before they are rotated, and a SwiGLU
//! feed forward. The same shape as Anima's Qwen3 encoder, six times the size, and read
//! differently: what comes back is not the last layer's states but *twelve* layers' states
//! stacked, because that stack is what Krea 2 conditions on.
//!
//! # What is easy to get wrong here
//!
//! *The states are tapped before the final norm.* `output_hidden_states` hands back each layer's
//! output as it leaves the residual stream, and only the entry past the last layer has
//! `model.norm` applied to it. Krea 2 taps layers 2 through 35 of 36, so none of them is normed
//! and the final norm is not in the package at all.
//!
//! *The encoder is a vision-language model and its vision half is not here.* For a prompt with no
//! picture in it the interleaved mRoPE it was trained with is exactly the ordinary rotary
//! embedding -- the three position axes all carry the token's index, so the angle is the same
//! whichever section of the head a frequency belongs to. A prompt with pictures in it is read by
//! [`TextEncoder::forward_with_images`], which is handed what a vision tower made of them -- Krea 2
//! has none, Qwen-Image 2.1 does -- and gives each picture's slots the positions of a grid.
//!
//! *It is still a causal model and is run as one.* Nothing here generates -- there is no cache
//! and no lm_head -- but the mask is the mask it was trained with.
//!
//! *There is no padding.* The reference pads the prompt out to a fixed 512 and masks the padding
//! away; this runs the prompt at its own length, which produces the same states for every token
//! that is not padding. `docs/krea2.md` is where that is proved rather than asserted.

use std::cell::RefCell;
use std::fmt;
use std::rc::Rc;

use super::config::EncoderConfig;
use crate::error::{Error, Result};
use crate::flint::{
    check_parameters, functional as F, DType, Extent, Graph, Ir, RunContext, Tensor, Value, Weights,
};
use crate::layers::{Embedding, Linear};

/// How Qwen3-VL shares a head's frequencies out among its three position axes: twenty-four to
/// the frame, twenty each to the row and the column, *interleaved* -- frequency `i` reads the row
/// where `i % 3 == 1` and the column where `i % 3 == 2`, up to three times the axis's share, and
/// the frame everywhere else. `rope_parameters.mrope_section` in the published configuration; the
/// same for the 4B and the 8B, so it is a constant rather than a line in every package.
const MROPE_SECTION: [usize; 3] = [24, 20, 20];

/// Which of `(frame, row, column)` frequency `index` of a head reads.
fn mrope_axis(index: usize) -> usize {
    match index % 3 {
        1 if index < 3 * MROPE_SECTION[1] => 1,
        2 if index < 3 * MROPE_SECTION[2] => 2,
        _ => 0,
    }
}

/// Where each token sits, as `(frame, row, column)`: `(n, n, n)` for the `n`th token of a prompt
/// with no picture in it.
pub fn text_positions(length: i32) -> Vec<[i32; 3]> {
    (0..length).map(|n| [n, n, n]).collect()
}

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
    /// Qwen3's table, at this encoder's base. Pairs `x[i]` against `x[i + half]`, which is the
    /// language model's convention and *not* the denoiser's -- [`super::Dit`] pairs adjacent
    /// elements instead, and the two are not interchangeable.
    ///
    /// `positions` is one `(frame, row, column)` per token. Each frequency reads the axis
    /// [`mrope_axis`] gives it, so a token whose three are equal gets the ordinary table.
    fn build(
        config: &EncoderConfig,
        positions: &[[i32; 3]],
        device: crate::flint::Device,
    ) -> Result<Rotary> {
        let half = (config.head_dim / 2) as usize;
        let heads = config.num_heads as usize;
        let length = positions.len();
        let count = length * heads * half;

        let mut cos = Vec::with_capacity(count);
        let mut sin = Vec::with_capacity(count);
        for position in positions {
            for _ in 0..heads {
                for index in 0..half {
                    let exponent = 2.0 * index as f64 / config.head_dim as f64;
                    let at = position[mrope_axis(index)] as f64;
                    let angle = at / (config.rope_theta as f64).powf(exponent);
                    cos.push(angle.cos() as f32);
                    sin.push(angle.sin() as f32);
                }
            }
        }

        let shape = [length as i32, config.num_heads, config.head_dim / 2];
        Ok(Rotary {
            cos: Tensor::from_f32(&shape, &cos)?.to_device(device)?,
            sin: Tensor::from_f32(&shape, &sin)?.to_device(device)?,
        })
    }
}

/// Rotate the last dimension of `x`, pairing its first half against its second.
pub(crate) fn rotate(g: &Graph, x: Value, cos: Value, sin: Value, half: i32) -> Value {
    let first = g.slice(x, 3, 0, half);
    let second = g.slice(x, 3, half, 2 * half);

    let left = g.sub(g.mul(first, cos), g.mul(second, sin));
    let right = g.add(g.mul(first, sin), g.mul(second, cos));
    g.cat(left, right, 3)
}

/// One decoder layer, written into `g`. `input` is `(1, L, D)` and so is what comes back.
fn layer(config: &EncoderConfig, g: &Graph, input: Value, cos: Value, sin: Value) -> Value {
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

    // The norm is over one head rather than the whole width, which is what makes its weights 128
    // long and not 2560.
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
///
/// What it outputs is one tensor rather than twelve: the tapped states stacked on a new third
/// dimension, `(1, L, 12, D)`, which is the shape the denoiser's text fusion reads.
///
/// With `deepstack` the graph reads pictures too. `visual` is what the vision tower made of them,
/// one row per image slot, and `visual_at` picks each token's input row out of the embeddings
/// with `visual` under them -- a word's own index, or `L + j` at slot `j`: the reference's
/// `masked_scatter`, as a gather. The vision tower's earlier layers (DeepStack) are added at the
/// slots after the first `deepstack` decoder layers the same way: `deepstack{k}` has a row of
/// zeros on top, and `deepstack_at` is 0 at a word and `1 + j` at slot `j`.
fn write(config: &EncoderConfig, float_type: DType, deepstack: Option<usize>, g: &Graph) {
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
    let embedded = match deepstack {
        Some(_) => {
            let visual = g.cast(g.input("visual"), float_type);
            g.lookup(g.cat(embedded, visual, 0), g.input("visual_at"))
        }
        None => embedded,
    };

    // Index zero is the embedding output, which is what `output_hidden_states` calls layer zero.
    // Nothing tapped here asks for it, and a package is free to.
    let mut states = vec![g.unsqueeze(embedded, 0)];
    for index in 0..config.num_layers {
        let last = *states.last().expect("the embedding is always there");
        let mut state = layer(
            config,
            &g.subgraph(&format!("block{index}")),
            last,
            cos,
            sin,
        );
        // Added inside the text model's loop, before the next layer reads the state and before
        // `output_hidden_states` records it.
        if deepstack.is_some_and(|count| (index as usize) < count) {
            let rows = g.cast(g.input(&format!("deepstack{index}")), float_type);
            let added = g.lookup(rows, g.input("deepstack_at"));
            state = g.add(state, g.unsqueeze(added, 0));
        }
        states.push(state);
    }

    // The stack the denoiser reads, in the order the package names the layers. Each state is
    // `(1, L, D)` and becomes `(1, L, 1, D)` so that the concatenation has somewhere to go.
    let stacked = config
        .select_layers
        .iter()
        .map(|&layer| g.unsqueeze(states[layer as usize], 2))
        .reduce(|left, right| g.cat(left, right, 2))
        .expect("a package with no tapped layers does not parse");

    g.output("hidden", stacked);
}

/// Pictures in a prompt, as [`TextEncoder::forward_with_images`] reads them: what a vision tower
/// made of each, and where every token sits.
pub struct EncoderImages {
    /// One `(frame, row, column)` per token of the prompt: a word's three are equal, and a
    /// picture's slots lay its grid out at the frame it starts at.
    pub positions: Vec<[i32; 3]>,
    /// Whether each token is one of a picture's slots, an `<|image_pad|>` whose embedding is
    /// replaced by the picture's row.
    pub slots: Vec<bool>,
    /// `(V, D)`: one row per slot, in the order the slots come.
    pub embeds: Tensor,
    /// `(V, D)` each: the vision tower's DeepStack rows, added at the slots after the first
    /// `deepstack.len()` decoder layers.
    pub deepstack: Vec<Tensor>,
}

pub struct TextEncoder {
    config: EncoderConfig,
    float_type: DType,
    name: String,
    ir: Ir,
    /// The same encoder reading pictures, compiled the first time a prompt has some, for the
    /// number of DeepStack layers it was handed.
    with_images: RefCell<Option<(usize, Rc<Ir>)>>,
    weights: Rc<Weights>,
}

impl fmt::Debug for TextEncoder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "TextEncoder({} instructions)", self.ir.len())
    }
}

fn compile(
    config: &EncoderConfig,
    float_type: DType,
    deepstack: Option<usize>,
    name: &str,
    weights: &Weights,
) -> Result<Ir> {
    let graph = Graph::with_weights(config.weight_format);
    write(config, float_type, deepstack, &graph.subgraph(name));
    check_parameters(&graph, weights)?;
    Ok(Ir::compile(&graph))
}

impl TextEncoder {
    /// `name` is the namespace the package holds this encoder under.
    pub fn build(
        config: EncoderConfig,
        name: &str,
        weights: &Rc<Weights>,
        float_type: DType,
    ) -> Result<TextEncoder> {
        if config.num_heads % config.num_kv_heads != 0 {
            return Err(Error::model(format!(
                "{} query heads do not group evenly into {} key heads",
                config.num_heads, config.num_kv_heads
            )));
        }

        let ir = compile(&config, float_type, None, name, weights.as_ref())?;

        Ok(TextEncoder {
            weights: Rc::clone(weights),
            ir,
            with_images: RefCell::new(None),
            name: name.to_string(),
            float_type,
            config,
        })
    }

    pub fn config(&self) -> &EncoderConfig {
        &self.config
    }

    /// The pass this encoder runs, for printing.
    pub fn ir(&self) -> &Ir {
        &self.ir
    }

    /// `input_ids` is `<long>(L)`. What comes back is `(1, L, 12, D)`: the twelve tapped layers'
    /// state at every position, which is the whole of what the denoiser is told about the prompt.
    pub fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        let length = token_count(input_ids)?;
        let rotary = Rotary::build(&self.config, &text_positions(length), input_ids.device())?;

        let run = RunContext::new(&*self.weights)
            .input("input_ids", input_ids)
            .input("rope_cos", &rotary.cos)
            .input("rope_sin", &rotary.sin);

        hidden(self.ir.run(&run)?)
    }

    /// [`TextEncoder::forward`] for a prompt with pictures in it: each slot reads `images`' row
    /// rather than its own embedding, the DeepStack rows are added at the slots after the first
    /// layers, and every token is rotated by the position `images` gives it.
    pub fn forward_with_images(
        &self,
        input_ids: &Tensor,
        images: &EncoderImages,
    ) -> Result<Tensor> {
        let length = token_count(input_ids)?;
        if images.positions.len() != length as usize || images.slots.len() != length as usize {
            return Err(Error::model(format!(
                "{length} tokens, {} positions and {} slot flags: one of each per token",
                images.positions.len(),
                images.slots.len()
            )));
        }
        let slots = images.slots.iter().filter(|&&slot| slot).count() as i32;
        for rows in std::iter::once(&images.embeds).chain(&images.deepstack) {
            if rows.dim()? != 2 || rows.shape_at(0)? != slots {
                return Err(Error::model(format!(
                    "the prompt has {slots} image slots, and the vision rows are {:?}",
                    rows.shape()
                )));
            }
        }
        if images.deepstack.len() >= self.config.num_layers as usize {
            return Err(Error::model(format!(
                "{} DeepStack rows for a {}-layer encoder",
                images.deepstack.len(),
                self.config.num_layers
            )));
        }

        let ir = self.compiled_with_images(images.deepstack.len())?;
        let device = input_ids.device();

        // Where each token's input row and DeepStack row come from: see `write`.
        let mut visual_at = Vec::with_capacity(length as usize);
        let mut deepstack_at = Vec::with_capacity(length as usize);
        let mut slot = 0i64;
        for (index, &is_slot) in images.slots.iter().enumerate() {
            if is_slot {
                visual_at.push(length as i64 + slot);
                deepstack_at.push(1 + slot);
                slot += 1;
            } else {
                visual_at.push(index as i64);
                deepstack_at.push(0);
            }
        }
        let visual_at = Tensor::from_i64(&[length], &visual_at)?.to_device(device)?;
        let deepstack_at = Tensor::from_i64(&[length], &deepstack_at)?.to_device(device)?;

        let width = images.embeds.shape_at(1)?;
        let zero = Tensor::zeros(&[1, width], self.float_type, device)?;
        let deepstack = images
            .deepstack
            .iter()
            .map(|rows| {
                Ok(F::cat(
                    &zero,
                    &rows.to_device(device)?.cast(self.float_type)?,
                    0,
                )?)
            })
            .collect::<Result<Vec<_>>>()?;
        let names = (0..deepstack.len())
            .map(|index| format!("deepstack{index}"))
            .collect::<Vec<_>>();

        let rotary = Rotary::build(&self.config, &images.positions, device)?;
        let embeds = images.embeds.to_device(device)?;

        let mut run = RunContext::new(&*self.weights)
            .input("input_ids", input_ids)
            .input("rope_cos", &rotary.cos)
            .input("rope_sin", &rotary.sin)
            .input("visual", &embeds)
            .input("visual_at", &visual_at)
            .input("deepstack_at", &deepstack_at);
        for (name, rows) in names.iter().zip(&deepstack) {
            run = run.input(name, rows);
        }

        hidden(ir.run(&run)?)
    }

    fn compiled_with_images(&self, deepstack: usize) -> Result<Rc<Ir>> {
        let mut cached = self.with_images.borrow_mut();
        if let Some((count, ir)) = cached.as_ref() {
            if *count == deepstack {
                return Ok(Rc::clone(ir));
            }
        }
        let ir = Rc::new(compile(
            &self.config,
            self.float_type,
            Some(deepstack),
            &self.name,
            self.weights.as_ref(),
        )?);
        *cached = Some((deepstack, Rc::clone(&ir)));
        Ok(ir)
    }
}

fn token_count(input_ids: &Tensor) -> Result<i32> {
    let dim = input_ids.dim()?;
    if dim != 1 {
        return Err(Error::model(format!(
            "a text encoder takes token ids as <long>(L), got a {dim}-D tensor"
        )));
    }
    Ok(input_ids.shape_at(0)?)
}

fn hidden(outputs: Vec<(String, Tensor)>) -> Result<Tensor> {
    outputs
        .into_iter()
        .find(|(name, _)| name == "hidden")
        .map(|(_, tensor)| tensor)
        .ok_or_else(|| Error::model("the encoder produced no hidden states"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shares_the_frequencies_out_the_way_qwen3_vl_does() {
        // `Qwen3VLTextRotaryEmbedding.recomposition_frequencies` over a 128-wide head: 64
        // frequencies, of which 1, 4, .., 58 are the row's and 2, 5, .., 59 the column's.
        let axes = (0..64).map(mrope_axis).collect::<Vec<_>>();
        assert_eq!(&axes[..7], &[0, 1, 2, 0, 1, 2, 0]);
        assert_eq!(&axes[57..], &[0, 1, 2, 0, 0, 0, 0]);
        assert_eq!(axes.iter().filter(|&&axis| axis == 0).count(), 24);
        assert_eq!(axes.iter().filter(|&&axis| axis == 1).count(), 20);
        assert_eq!(axes.iter().filter(|&&axis| axis == 2).count(), 20);
    }
}
