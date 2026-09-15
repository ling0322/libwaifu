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

//! The layers every model here is built from.
//!
//! Each one is a name and a shape rather than a thing that holds weights. What a layer knows is
//! what its parameters are called, what shape each of them has, and what to do with them; `graph`
//! writes that into a [`Graph`] and hands back the [`Value`] it produces, and the weights
//! themselves are read later by whoever runs the graph.
//!
//! So there is nothing to build and nothing to hold: these are namespaces, not objects. A layer
//! appears in a model as one call --
//!
//! ```
//! use waifu::flint::Graph;
//! use waifu::Linear;
//!
//! let g = Graph::new();
//! let x = g.input("hidden");
//! let projected = Linear::graph(&g.subgraph("proj"), x, 4, 8, true);
//!
//! // The names are the package's, made by the namespace the layer was written into.
//! assert_eq!(g.parameters(), vec!["proj.weight", "proj.bias"]);
//! ```
//!
//! -- and the reason the names are constants here rather than strings at the call site is that a
//! model asks for a weight in two places, the graph that reads it and the exporter that wrote it,
//! and those two must not be free to disagree.

use crate::flint::{DType, Graph, Value};

/// Normalization over the last dimension: subtract the mean, divide by the standard deviation,
/// scale and shift.
pub struct LayerNorm;

impl LayerNorm {
    pub const WEIGHT: &'static str = "weight";
    pub const BIAS: &'static str = "bias";

    /// Written into `g`, which reads its weights out of `g`'s namespace.
    #[track_caller]
    pub fn graph(g: &Graph, input: Value, d_model: i32, eps: f32) -> Value {
        let weight = g.load(Self::WEIGHT, &[d_model]);
        let bias = g.load(Self::BIAS, &[d_model]);

        g.layer_norm(input, Some(weight), Some(bias), eps)
    }
}

/// A table of word embeddings, read by token id.
pub struct Embedding;

impl Embedding {
    pub const WEIGHT: &'static str = "weight";

    /// The embeddings of `input` `<long>(L)`, as `<float>(L, D)`, written into `g`.
    ///
    /// `dtype` is what the rows come out as. A lookup copies rows out of the table, so what comes
    /// back is whatever the table is held as -- half, on a machine that stores its weights that
    /// way -- and the cast is here rather than in the operator because the table is the large
    /// thing and stays where it is, while this is a handful of rows.
    #[track_caller]
    pub fn graph(g: &Graph, input: Value, d_model: i32, vocab_size: i32, dtype: DType) -> Value {
        let weight = g.load(Self::WEIGHT, &[vocab_size, d_model]);

        g.cast(g.lookup(weight, input), dtype)
    }
}

/// A two dimensional convolution, with a bias, as every convolution in a diffusion model has one.
///
/// Only what those models ask for: a square kernel, one group, and no dilation.
pub struct Conv2d;

impl Conv2d {
    pub const WEIGHT: &'static str = "weight";
    pub const BIAS: &'static str = "bias";

    /// `input` is `(N, C, H, W)`, and so is what comes back.
    #[track_caller]
    pub fn graph(
        g: &Graph,
        input: Value,
        in_channels: i32,
        out_channels: i32,
        kernel: i32,
        stride: i32,
        padding: i32,
    ) -> Value {
        let weight = g.load(Self::WEIGHT, &[out_channels, in_channels, kernel, kernel]);
        let bias = g.load(Self::BIAS, &[out_channels]);

        g.conv2d(input, weight, Some(bias), stride, padding, 1, 1)
    }
}

/// Normalization over a group of channels and all of the space they cover.
///
/// This is what a diffusion model normalizes with. A batch of one image says nothing about its
/// own statistics, which is why the mean and variance are taken this way rather than over the
/// batch.
pub struct GroupNorm;

impl GroupNorm {
    pub const WEIGHT: &'static str = "weight";
    pub const BIAS: &'static str = "bias";

    /// `input` is `(N, C, H, W)`, and so is what comes back.
    ///
    /// This does not check that the channels divide into the groups: a graph holds no shapes, so
    /// there is nothing here to check it against. Whoever writes the graph is the one that knows
    /// both numbers, and it is theirs to check.
    #[track_caller]
    pub fn graph(g: &Graph, input: Value, channels: i32, groups: i32, eps: f32) -> Value {
        let weight = g.load(Self::WEIGHT, &[channels]);
        let bias = g.load(Self::BIAS, &[channels]);

        g.group_norm(input, Some(weight), Some(bias), groups, eps)
    }
}

/// A fully connected layer, with an optional bias.
pub struct Linear;

impl Linear {
    pub const WEIGHT: &'static str = "weight";
    pub const BIAS: &'static str = "bias";

    /// Written into `g`, which reads its weights out of `g`'s namespace.
    ///
    /// The weight is stored the way the package holds it, `(out_dim, in_dim)`, and transposed
    /// here rather than when it is read: a transpose is a node and costs nothing to run, and the
    /// alternative is a graph that asks for a weight the file does not hold.
    #[track_caller]
    pub fn graph(g: &Graph, input: Value, in_dim: i32, out_dim: i32, has_bias: bool) -> Value {
        let weight = g.load(Self::WEIGHT, &[out_dim, in_dim]);
        let x = g.matmul(input, g.transpose(weight, 0, 1));

        match has_bias {
            true => g.add(x, g.load(Self::BIAS, &[out_dim])),
            false => x,
        }
    }
}
