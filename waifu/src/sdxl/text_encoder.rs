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
//!
//! # Written down rather than run
//!
//! This is the first model in the library that is a [`Graph`] rather than a tree of layers each
//! holding its own weights. [`ClipTextEncoder::build`] writes the whole pass down, compiles it
//! into an [`Ir`], and reads the weights that graph turns out to ask for; `forward` hands the
//! token ids to the [`Ir`] and takes back what it named.
//!
//! What that buys is in [`Ir`](crate::flint::Ir): a weight is let go of at the instruction that
//! read it last, and the whole pass can be printed and read before it is run. What it costs is
//! the two things a graph cannot say, and both of them show up here:
//!
//! *A graph holds no shapes,* so nothing here can ask a tensor how long it is. Where the eager
//! pass read `input.shape_at(1)`, this says [`Extent::of`] -- however long that value turns out
//! to be -- and the size is resolved against the real tensor when the instruction runs. The one
//! length written down is `context_length`, and only as the shape the position table really has
//! in the package.
//!
//! *A graph holds no branches,* so where the end-of-text marker sits cannot be worked out inside
//! it. It is handed over as an input instead, and the pooled vector is a [`lookup`](Graph::lookup)
//! at that position rather than a slice at a number the code knew.

use std::fmt;
use std::rc::Rc;

use crate::error::{Error, Result};
use crate::flint::{
    check_parameters, DType, Extent, Graph, Ir, ParamSource, RunContext, Tensor, Value,
    WeightFormat,
};
use crate::layers::{Embedding, LayerNorm, Linear};

/// What separates the two encoders, and what the package records for each.
#[derive(Clone, Copy, Debug)]
pub struct ClipTextConfig {
    pub hidden_size: i32,
    pub intermediate_size: i32,
    pub num_layers: i32,
    pub num_heads: i32,
    /// How many positions the embedding table holds, which is the hard limit on a prompt: there
    /// simply is no vector for position 78. It is also the one length the encoder is written for.
    pub context_length: i32,
    pub vocab_size: i32,
    /// True for the sigmoid approximation, false for the ordinary GELU.
    pub quick_gelu: bool,
    pub norm_eps: f32,
    /// The id whose position the pooled output is read from.
    pub eot_token_id: i32,
    /// How the package stored the matrices this multiplies by, which decides what its projections
    /// are built out of. See [`WeightFormat`].
    pub weight_format: WeightFormat,
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

/// One transformer layer, written into `g`: attention over what came before, then a feed forward,
/// each around a normalization and a residual.
///
/// `input` is `(1, L, D)`, and so is what comes back.
fn clip_layer(config: &ClipTextConfig, g: &Graph, input: Value) -> Value {
    let d = config.hidden_size;

    let x = LayerNorm::graph(&g.subgraph("input_norm"), input, d, config.norm_eps);
    let x = attention(config, g, x);
    let x = g.add(input, x);

    let residual = x;
    let x = LayerNorm::graph(&g.subgraph("post_attn_norm"), residual, d, config.norm_eps);
    let x = Linear::graph(&g.subgraph("mlp.fc1"), x, d, config.intermediate_size, true);
    let x = match config.quick_gelu {
        true => g.quick_gelu(x),
        false => g.gelu(x),
    };
    let x = Linear::graph(&g.subgraph("mlp.fc2"), x, config.intermediate_size, d, true);

    g.add(residual, x)
}

/// The attention of one layer, written into `g`.
fn attention(config: &ClipTextConfig, g: &Graph, input: Value) -> Value {
    let d = config.hidden_size;
    let head_dim = d / config.num_heads;

    // One projection produces all three; they sit end to end along the width.
    // However many positions the layer was handed, which is what the eager pass read off the
    // tensor and what a graph has to name instead.
    let length = Extent::of(input, 1);

    let qkv = Linear::graph(&g.subgraph("attn.qkv_proj"), input, d, 3 * d, true);
    let mut parts = Vec::with_capacity(3);
    for index in 0..3 {
        let part = g.slice(qkv, 2, index * d, (index + 1) * d);
        // (1, L, D) to the (1, H, L, Dh) the attention wants.
        let shape = [
            Extent::At(1),
            length,
            Extent::At(config.num_heads),
            Extent::At(head_dim),
        ];
        parts.push(g.contiguous(g.transpose(g.view(g.contiguous(part), shape), 1, 2)));
    }

    // A text encoder reads left to right, so a position may not see what follows it.
    let x = g.attention(parts[0], parts[1], parts[2], true);
    let x = g.contiguous(g.transpose(x, 1, 2));
    let x = g.view(x, [Extent::At(1), length, Extent::At(d)]);

    Linear::graph(&g.subgraph("attn.out_proj"), x, d, d, true)
}

/// The whole encoder, written into `g`, which is already narrowed to the namespace the package
/// holds it under.
///
/// `float_type` is what the pass runs in, which the two casts here need and nothing else does:
/// every other weight arrives in the precision it is used in, because [`resident`] put it there
/// as it read it.
///
/// `has_projection` is not a thing the config says. Only the second of SDXL's two encoders holds
/// a `text_proj`, and which one this is is settled by looking in the package -- so the graph is
/// written against a config *and* a file, and [`ClipTextEncoder::build`] is where the two meet.
fn write(config: &ClipTextConfig, float_type: DType, has_projection: bool, g: &Graph) {
    let d = config.hidden_size;

    let ids = g.input("input_ids");
    let eot = g.input("eot");

    // Where a token sits is as much of its meaning as which token it is, and the two are simply
    // added. The position table is widened to what the pass runs in for that reason: an addition
    // would rather have two of the same thing.
    let embedded = Embedding::graph(
        &g.subgraph("token_embd"),
        ids,
        d,
        config.vocab_size,
        float_type,
    );
    let table = g
        .subgraph("position_embd")
        .load(Embedding::WEIGHT, &[config.context_length, d]);
    let positions = g.slice(table, 0, 0, Extent::of(ids, 0));
    let mut x = g.unsqueeze(g.add(embedded, g.cast(positions, float_type)), 0);

    // The layer before the last is what SDXL conditions on, so it is named as it goes past.
    for index in 0..config.num_layers {
        x = clip_layer(config, &g.subgraph(&format!("block{index}")), x);
        if index + 2 == config.num_layers {
            g.output("hidden", x);
        }
    }

    let last = LayerNorm::graph(&g.subgraph("final_norm"), x, d, config.norm_eps);

    // The pooled vector is read at the end-of-text marker, which is the last thing the prompt
    // said and therefore the only position that has seen all of it. Which position that is comes
    // in as `eot`, so this is a lookup and not a slice.
    let pooled = g.lookup(g.subtensor(last, 0), eot);
    let pooled = match has_projection {
        true => {
            let projection = g.subgraph("text_proj").load(Linear::WEIGHT, &[d, d]);
            g.matmul(pooled, g.transpose(projection, 0, 1))
        }
        false => pooled,
    };
    g.output("pooled", pooled);
}

/// One of what a run handed back, by the name the graph gave it.
fn output(outputs: &[(String, Tensor)], name: &str) -> Result<Tensor> {
    outputs
        .iter()
        .find(|(other, _)| other == name)
        .map(|(_, tensor)| tensor.clone())
        .ok_or_else(|| Error::model(format!("a text encoder produces no {name:?}")))
}

pub struct ClipTextEncoder {
    config: ClipTextConfig,
    ir: Ir,
    /// Every weight the package holds, which the four halves of a model share. See
    /// [`resident`](crate::flint::resident).
    weights: Rc<dyn ParamSource>,
}

impl fmt::Debug for ClipTextEncoder {
    /// How big it is rather than what is in it. Both halves of this are large and both have a
    /// better way of being read: [`ClipTextEncoder::ir`] prints the pass, and the weights are
    /// named in it.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ClipTextEncoder({} instructions)", self.ir.len())
    }
}

impl ClipTextEncoder {
    /// `name` is the namespace the package holds this encoder under, which is what its weights
    /// are named from.
    pub fn build(
        config: ClipTextConfig,
        name: &str,
        weights: &Rc<dyn ParamSource>,
        float_type: DType,
    ) -> Result<ClipTextEncoder> {
        if config.num_layers < 2 {
            return Err(Error::model(
                "a text encoder needs at least two layers to have a penultimate one",
            ));
        }

        let graph = Graph::with_weights(config.weight_format);
        let has_projection = weights.has(&format!("{name}.text_proj.weight"));
        write(&config, float_type, has_projection, &graph.subgraph(name));
        check_parameters(&graph, weights.as_ref())?;

        Ok(ClipTextEncoder {
            weights: Rc::clone(weights),
            ir: Ir::compile(&graph),
            config,
        })
    }

    pub fn config(&self) -> &ClipTextConfig {
        &self.config
    }

    /// The pass this encoder runs, for printing.
    ///
    /// What an eager model could not be asked: every instruction, in the order it runs, with the
    /// weight each one reads named as the package names it.
    pub fn ir(&self) -> &Ir {
        &self.ir
    }

    /// `input_ids` is `<long>(L)`, wrapped in its markers by whoever built it. Any `L` up to the
    /// context length: the pass reads how long it is rather than having been written for it.
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

        // Which position the pooled vector is read at depends on the prompt, and a graph holds no
        // branches, so it is worked out here and handed over as a value.
        let eot = self.eot_position(input_ids)?;
        let eot = Tensor::from_i64(&[1], &[eot as i64])?.to_device(input_ids.device())?;

        let context = RunContext::new(&*self.weights)
            .input("input_ids", input_ids)
            .input("eot", &eot);

        let outputs = self.ir.run(&context)?;

        Ok(ClipTextOutput {
            hidden: output(&outputs, "hidden")?,
            pooled: output(&outputs, "pooled")?,
        })
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
