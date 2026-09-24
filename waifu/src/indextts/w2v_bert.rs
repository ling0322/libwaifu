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

//! w2v-bert-2.0: the conformer that reads speech into the features everything else conditions on.
//!
//! Meta's model, shared by several speech systems. IndexTTS-2.5 runs it over the reference audio
//! and hands what comes out to [`crate::indextts::semantic_codec`], which turns it into tokens.
//!
//! # Sixteen layers of twenty-four
//!
//! The release has twenty-four encoder layers and IndexTTS reads `hidden_states[17]` -- the
//! output of the seventeenth entry, which is the embedding plus sixteen layers. The last eight
//! are never used and are not run here. That is a third of the model that does not have to exist
//! at inference, which is worth knowing before anyone exports 580 M parameters.
//!
//! # A conformer layer is four things, and two of them are halved
//!
//! Feed forward, attention, convolution, feed forward. The two feed forwards are added back at
//! half weight -- `x * 0.5 + residual` -- which is what "macaron" means and is not a detail: at
//! full weight the residual stream doubles through every layer.
//!
//! # Two things worth pointing at
//!
//! **The depthwise convolution is causal.** Every other depthwise convolution in this repository
//! pads symmetrically; this one pads `kernel - 1` on the left and nothing on the right, so a
//! frame never sees the future. [`depthwise_conv1d`] takes one padding for both sides, so the
//! padding is done in front of it and the convolution is asked for none.
//!
//! **The positions are relative keys, not rotary.** The score between two frames gets a term that
//! depends only on how far apart they are, clamped to sixty-four back and eight forward. Which
//! embedding that is for every pair is a table of indices this builds on the host, and `lookup`
//! turns it into the `(T, T, head_dim)` the scores need -- so the gather is an operator that
//! exists rather than one that does not.

use crate::audio::{conv1d, depthwise_conv1d, pad1d, Padding};
use crate::flint::{DType, Device, Extent, Graph, Tensor, Value};
use crate::layers::{LayerNorm, Linear};
use crate::Result;

/// The widths of one w2v-bert-2.0, as `facebook/w2v-bert-2.0`'s `config.json` states them.
#[derive(Clone, Copy, Debug)]
pub struct Config {
    /// What the feature extractor hands over: 80 mel bands, stacked in pairs.
    pub feature_dim: i32,
    pub hidden_size: i32,
    pub num_heads: i32,
    pub intermediate_size: i32,
    /// How wide the depthwise convolution is. Causal, so it reaches this far back and no further.
    pub conv_kernel: i32,
    /// How far the relative position embedding distinguishes, backwards and forwards.
    pub left_positions: i32,
    pub right_positions: i32,
    pub layer_norm_eps: f32,
}

impl Config {
    /// What IndexTTS-2.5 fetches, which is the released configuration unchanged.
    pub fn w2v_bert_2() -> Config {
        Config {
            feature_dim: 160,
            hidden_size: 1024,
            num_heads: 16,
            intermediate_size: 4096,
            conv_kernel: 31,
            left_positions: 64,
            right_positions: 8,
            layer_norm_eps: 1e-5,
        }
    }

    /// How many layers IndexTTS-2.5 actually needs: it reads `hidden_states[17]`, which is the
    /// embedding and then sixteen of the twenty-four.
    pub const USED_LAYERS: i32 = 16;

    fn head_dim(&self) -> i32 {
        self.hidden_size / self.num_heads
    }
}

/// Which relative-position embedding every pair of frames uses: `(frames * frames)` indices.
///
/// The distance is `key - query`, clamped to `[-left, right]` and shifted so the nearest
/// backwards position is zero. Built here rather than in the graph because it depends on nothing
/// but the length, and because `lookup` wants indices.
pub fn distance_indices(frames: i32, config: &Config, device: Device) -> Result<Tensor> {
    let mut values = Vec::with_capacity((frames * frames) as usize);

    for query in 0..frames {
        for key in 0..frames {
            let distance = (key - query).clamp(-config.left_positions, config.right_positions);
            values.push(i64::from(distance + config.left_positions));
        }
    }

    Ok(Tensor::from_i64(&[frames * frames], &values)?.to_device(device)?)
}

/// The macaron feed forward: widen, swish, narrow.
#[track_caller]
fn feed_forward(g: &Graph, x: Value, config: &Config) -> Value {
    let wide = Linear::graph(
        &g.subgraph("intermediate_dense"),
        x,
        config.hidden_size,
        config.intermediate_size,
        true,
    );

    Linear::graph(
        &g.subgraph("output_dense"),
        g.silu(wide),
        config.intermediate_size,
        config.hidden_size,
        true,
    )
}

/// Attention whose scores carry a term for how far apart two frames are.
#[track_caller]
fn attention(g: &Graph, x: Value, config: &Config, frames: i32, distances: Value) -> Value {
    let (heads, head_dim) = (config.num_heads, config.head_dim());
    let hidden = config.hidden_size;

    let project = |name: &str| {
        let flat = Linear::graph(&g.subgraph(name), x, hidden, hidden, true);
        let split = g.view(
            flat,
            [
                Extent::of(x, 0),
                Extent::At(frames),
                Extent::At(heads),
                Extent::At(head_dim),
            ],
        );

        g.contiguous(g.transpose(split, 1, 2))
    };

    let (query, key, value) = (
        project("linear_q"),
        project("linear_k"),
        project("linear_v"),
    );

    // The ordinary scores.
    let scores = g.div_scalar(
        g.matmul(query, g.contiguous(g.transpose(key, 2, 3))),
        (head_dim as f32).sqrt(),
    );

    // And the term that depends only on the distance. `positions` is (T, T, head_dim): for every
    // query frame, the embedding of its distance to every key frame.
    let table = g.subgraph("distance_embedding").load(
        "weight",
        &[config.left_positions + config.right_positions + 1, head_dim],
    );
    let positions = g.view(
        g.lookup(table, distances),
        [Extent::At(frames), Extent::At(frames), Extent::At(head_dim)],
    );

    // One matrix multiply per query frame, with the frame as the batch axis: `(T, B * H, D)`
    // against `(T, D, T)` gives `(T, B * H, T)`, which is the term for every pair.
    let by_frame = g.contiguous(g.transpose(g.transpose(query, 1, 2), 0, 1));
    let by_frame = g.view(
        by_frame,
        [
            Extent::At(frames),
            Extent::prod(query, 0, 2),
            Extent::At(head_dim),
        ],
    );
    let relative = g.matmul(by_frame, g.contiguous(g.transpose(positions, 1, 2)));

    let relative = g.view(
        g.contiguous(relative),
        [
            Extent::At(frames),
            Extent::of(x, 0),
            Extent::At(heads),
            Extent::At(frames),
        ],
    );
    let relative = g.transpose(g.transpose(relative, 0, 1), 1, 2);

    let scores = g.add(
        scores,
        g.div_scalar(g.contiguous(relative), (head_dim as f32).sqrt()),
    );

    let out = g.matmul(g.softmax(scores), value);
    let merged = g.view(
        g.contiguous(g.transpose(out, 1, 2)),
        [
            Extent::of(x, 0),
            Extent::At(frames),
            Extent::At(heads * head_dim),
        ],
    );

    Linear::graph(&g.subgraph("linear_out"), merged, hidden, hidden, true)
}

/// The convolution module: a gated pointwise pair around a causal depthwise convolution.
#[track_caller]
fn conv_module(
    g: &Graph,
    x: Value,
    config: &Config,
    dtype: DType,
    device: Device,
) -> Result<Value> {
    let hidden = config.hidden_size;

    let x = LayerNorm::graph(&g.subgraph("layer_norm"), x, hidden, config.layer_norm_eps);

    // (N, T, C) -> (N, C, T): the convolutions read time.
    let x = g.contiguous(g.transpose(x, 1, 2));

    let first = g.subgraph("pointwise_conv1");
    let weight = first.load("weight", &[2 * hidden, hidden, 1]);
    let gated = conv1d(&first, x, weight, None, 1, 0, 1, 1, dtype, device)?;

    // A gated linear unit over the channel axis, which is halved by it.
    let left = g.contiguous(g.slice(gated, 1, 0, hidden));
    let right = g.contiguous(g.slice(gated, 1, hidden, 2 * hidden));
    let x = g.mul(left, g.sigmoid(right));

    // Causal: the whole padding goes on the left, so a frame never reads the future.
    let padded = pad1d(
        g,
        x,
        config.conv_kernel - 1,
        0,
        Padding::Zero,
        dtype,
        device,
    )?;

    let depthwise = g.subgraph("depthwise_conv");
    let weight = depthwise.load("weight", &[hidden, 1, config.conv_kernel]);
    let x = depthwise_conv1d(
        &depthwise,
        padded,
        weight,
        None,
        hidden,
        config.conv_kernel,
        0,
        1,
        dtype,
        device,
    )?;

    // The normalization between the two is over channels, so time goes last and comes back.
    let x = LayerNorm::graph(
        &g.subgraph("depthwise_layer_norm"),
        g.contiguous(g.transpose(x, 1, 2)),
        hidden,
        config.layer_norm_eps,
    );
    let x = g.silu(g.contiguous(g.transpose(x, 1, 2)));

    let second = g.subgraph("pointwise_conv2");
    let weight = second.load("weight", &[hidden, hidden, 1]);
    let x = conv1d(&second, x, weight, None, 1, 0, 1, 1, dtype, device)?;

    Ok(g.contiguous(g.transpose(x, 1, 2)))
}

/// One conformer layer: feed forward, attention, convolution, feed forward, normalize.
#[track_caller]
#[allow(clippy::too_many_arguments)]
fn encoder_layer(
    g: &Graph,
    x: Value,
    config: &Config,
    frames: i32,
    distances: Value,
    dtype: DType,
    device: Device,
) -> Result<Value> {
    let hidden = config.hidden_size;
    let eps = config.layer_norm_eps;

    // The two feed forwards are added back at half weight. Not a detail: at full weight the
    // residual stream doubles through every layer.
    let normed = LayerNorm::graph(&g.subgraph("ffn1_layer_norm"), x, hidden, eps);
    let x = g.add(
        g.mul_scalar(feed_forward(&g.subgraph("ffn1"), normed, config), 0.5),
        x,
    );

    let normed = LayerNorm::graph(&g.subgraph("self_attn_layer_norm"), x, hidden, eps);
    let x = g.add(
        attention(&g.subgraph("self_attn"), normed, config, frames, distances),
        x,
    );

    let x = g.add(
        conv_module(&g.subgraph("conv_module"), x, config, dtype, device)?,
        x,
    );

    let normed = LayerNorm::graph(&g.subgraph("ffn2_layer_norm"), x, hidden, eps);
    let x = g.add(
        g.mul_scalar(feed_forward(&g.subgraph("ffn2"), normed, config), 0.5),
        x,
    );

    Ok(LayerNorm::graph(
        &g.subgraph("final_layer_norm"),
        x,
        hidden,
        eps,
    ))
}

/// The encoder as IndexTTS-2.5 reads it: `(N, T, feature_dim)` in, `(N, T, hidden_size)` out.
///
/// `layers` is how many of the twenty-four to run. IndexTTS wants [`Config::USED_LAYERS`], which
/// is sixteen; running all of them would be a third more work for a tensor nothing reads.
#[track_caller]
#[allow(clippy::too_many_arguments)]
pub fn graph(
    g: &Graph,
    x: Value,
    config: &Config,
    frames: i32,
    layers: i32,
    distances: Value,
    dtype: DType,
    device: Device,
) -> Result<Value> {
    let projection = g.subgraph("feature_projection");
    let normed = LayerNorm::graph(
        &projection.subgraph("layer_norm"),
        x,
        config.feature_dim,
        config.layer_norm_eps,
    );
    let mut running = Linear::graph(
        &projection.subgraph("projection"),
        normed,
        config.feature_dim,
        config.hidden_size,
        true,
    );

    let encoder = g.subgraph("encoder").subgraph("layers");
    for index in 0..layers {
        running = encoder_layer(
            &encoder.subgraph(&index.to_string()),
            running,
            config,
            frames,
            distances,
            dtype,
            device,
        )?;
    }

    Ok(running)
}
