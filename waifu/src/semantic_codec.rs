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

//! The semantic codec: the alphabet the GPT reads and writes, and the way back out of it.
//!
//! MaskGCT's codec, as IndexTTS-2.5 vendors it. Both directions are here, and **the one the
//! pipeline actually runs is [`decode`]**, which is worth saying first because it is the opposite
//! of what it looks like:
//!
//! - [`decode`] turns the GPT's tokens back into `hidden_size` features, and that is what
//!   `infer_v2_5.py` calls -- `S_infer = self.semantic_codec.decode(codes)` -- before S2Mel's
//!   length regulator reads them.
//! - [`similarity`] and [`nearest`] are the encoder, `quantize`. IndexTTS-2.5 **does not call
//!   it.** The line that would, `_, S_ref = self.semantic_codec.quantize(spk_cond_emb)`, is
//!   commented out in the released source and replaced by `S_ref = self.get_emb(...)`: the
//!   reference audio reaches the length regulator as w2v-bert's continuous features, never
//!   having been through a codebook at all.
//!
//! So the encoder is kept for what it is -- the definition of the alphabet, and the thing that
//! says what a token *means* -- and not because anything here needs it to say a sentence.
//!
//! # Which way round each half goes
//!
//! Encoding: a convolution halves the frame rate, twelve ConvNeXt blocks read what is left, every
//! frame is projected down to eight numbers and replaced by the nearest of 8192 codebook entries.
//! The token is that entry's index.
//!
//! Decoding is the mirror and the same shapes: the entry is looked up, projected back up to
//! `hidden_size`, read by twelve more ConvNeXt blocks -- a second backbone with its own weights,
//! structurally identical to the encoder's -- and then the frame rate is doubled back. One token
//! becomes two frames, which is the ratio it stood for.
//!
//! # The nearest entry is found on the host
//!
//! `flint` can take a maximum but cannot say *where* it was: there is no argmax. So the graph
//! computes the similarity -- `(T, 8) @ (8, 8192)` -- and [`nearest`] walks the result here.
//!
//! That is less of a compromise than it sounds. Both sides of the comparison are L2 normalized,
//! which makes the Euclidean distance `2 - 2 * cosine`, so the nearest entry is the largest dot
//! product and the whole search is one matrix multiply the graph is good at. What comes back is
//! `T * 8192` floats, once per utterance; the walk over them is memory-bound and short beside the
//! twelve ConvNeXt blocks in front of it.

use crate::audio::{conv1d, depthwise_conv1d};
use crate::flint::{DType, Device, Extent, Graph, Value};
use crate::layers::Linear;
use crate::Result;

/// The widths of the codec, as `config.yaml` states them.
#[derive(Clone, Copy, Debug)]
pub struct Config {
    /// What w2v-bert hands over, and what the codec works in.
    pub hidden_size: i32,
    /// How many entries the codebook has. The GPT's alphabet.
    pub codebook_size: i32,
    /// How wide one entry is -- much narrower than the features, which is what "factorized" means.
    pub codebook_dim: i32,
    /// The ConvNeXt trunk's width.
    pub vocos_dim: i32,
    pub vocos_intermediate_dim: i32,
    pub vocos_num_layers: i32,
}

impl Config {
    /// What IndexTTS-2.5's `config.yaml` describes.
    pub fn indextts() -> Config {
        Config {
            hidden_size: 1024,
            codebook_size: 8192,
            codebook_dim: 8,
            vocos_dim: 384,
            vocos_intermediate_dim: 2048,
            vocos_num_layers: 12,
        }
    }
}

/// One ConvNeXt block: a depthwise convolution over time, then a pointwise pair over channels.
///
/// The depthwise convolution is seven frames wide and mixes no channels; the two linears mix
/// channels and no frames. That separation is the whole idea, and it is why this needs
/// [`depthwise_conv1d`] rather than a grouped [`conv1d`] -- see that function for why the two are
/// not the same thing on a card.
#[track_caller]
fn convnext_block(
    g: &Graph,
    x: Value,
    dim: i32,
    intermediate: i32,
    dtype: DType,
    device: Device,
) -> Result<Value> {
    let residual = x;

    let weight = g.subgraph("dwconv").load("weight", &[dim, 1, 7]);
    let bias = g.subgraph("dwconv").load("bias", &[dim]);
    let y = depthwise_conv1d(
        &g.subgraph("dwconv"),
        x,
        weight,
        Some(bias),
        dim,
        7,
        3,
        1,
        dtype,
        device,
    )?;

    // (N, C, T) -> (N, T, C): the two linears read channels, and so does the normalization.
    let y = g.contiguous(g.transpose(y, 1, 2));

    let norm = g.subgraph("norm");
    let y = g.layer_norm(
        y,
        Some(norm.load("weight", &[dim])),
        Some(norm.load("bias", &[dim])),
        1e-6,
    );

    let y = Linear::graph(&g.subgraph("pwconv1"), y, dim, intermediate, true);
    let y = Linear::graph(&g.subgraph("pwconv2"), g.gelu(y), intermediate, dim, true);

    // One learned scale per channel, which is what keeps a deep stack of these stable.
    let y = g.mul(y, g.load("gamma", &[dim]));

    Ok(g.add(residual, g.contiguous(g.transpose(y, 1, 2))))
}

/// The ConvNeXt trunk: `(N, in_channels, T)` in, `(N, T, dim)` out.
#[track_caller]
#[allow(clippy::too_many_arguments)]
pub fn vocos_backbone(
    g: &Graph,
    x: Value,
    in_channels: i32,
    config: &Config,
    dtype: DType,
    device: Device,
) -> Result<Value> {
    let dim = config.vocos_dim;

    let embed = g.subgraph("embed");
    let weight = embed.load("weight", &[dim, in_channels, 7]);
    let bias = embed.load("bias", &[dim]);
    let x = conv1d(&embed, x, weight, Some(bias), 1, 3, 1, 1, dtype, device)?;

    // The first normalization is over channels, so it happens with time last and is put back.
    let norm = g.subgraph("norm");
    let x = g.layer_norm(
        g.contiguous(g.transpose(x, 1, 2)),
        Some(norm.load("weight", &[dim])),
        Some(norm.load("bias", &[dim])),
        1e-6,
    );

    let mut running = g.contiguous(g.transpose(x, 1, 2));
    for index in 0..config.vocos_num_layers {
        running = convnext_block(
            &g.subgraph("convnext").subgraph(&index.to_string()),
            running,
            dim,
            config.vocos_intermediate_dim,
            dtype,
            device,
        )?;
    }

    let last = g.subgraph("final_layer_norm");

    Ok(g.layer_norm(
        g.contiguous(g.transpose(running, 1, 2)),
        Some(last.load("weight", &[dim])),
        Some(last.load("bias", &[dim])),
        1e-6,
    ))
}

/// `x / ||x||` over the last axis, which is what makes a Euclidean nearest neighbour a cosine one.
///
/// `rank` is how many axes `x` has. The sum over the last one drops it, and putting it back has
/// to name the position rather than `-1`, which inserts before the last axis rather than after
/// it -- and a `(1, T, 1)` that came out `(1, 1, T)` broadcasts against nothing.
#[track_caller]
fn normalize(
    g: &Graph,
    x: Value,
    rank: i32,
    eps: f32,
    dtype: DType,
    device: Device,
) -> Result<Value> {
    let energy = g.unsqueeze(g.sum(g.square(x), -1), rank - 1);
    let floor = g.constant(
        crate::flint::Tensor::from_f32(&[1], &[eps])?
            .to_device(device)?
            .cast(dtype)?,
    );

    Ok(g.div(x, g.sqrt(g.add(energy, floor))))
}

/// How much each frame looks like each codebook entry: `(N, T, codebook_size)`.
///
/// This is everything the graph can do towards finding the nearest entry. Which entry that is --
/// the argmax over the last axis -- is [`nearest`], on the host, because there is no operator
/// here that reports where a maximum was.
///
/// `x` is `(N, T, hidden_size)`, the features as w2v-bert hands them over.
#[track_caller]
pub fn similarity(
    g: &Graph,
    x: Value,
    config: &Config,
    frames: i32,
    dtype: DType,
    device: Device,
) -> Result<Value> {
    // Halve the frame rate. One token covers two of w2v-bert's frames.
    let down = g.subgraph("down");
    let weight = down.load("weight", &[config.hidden_size, config.hidden_size, 3]);
    let bias = down.load("bias", &[config.hidden_size]);
    let halved = conv1d(
        &down,
        g.contiguous(g.transpose(x, 1, 2)),
        weight,
        Some(bias),
        2,
        1,
        1,
        1,
        dtype,
        device,
    )?;

    let encoder = g.subgraph("encoder");
    let trunk = vocos_backbone(
        &encoder.subgraph("0"),
        g.gelu(halved),
        config.hidden_size,
        config,
        dtype,
        device,
    )?;

    // The trunk works in `vocos_dim` and the quantizer in `hidden_size`, so one projection back.
    let features = Linear::graph(
        &encoder.subgraph("1"),
        trunk,
        config.vocos_dim,
        config.hidden_size,
        true,
    );

    // Down to the codebook's own width -- eight numbers, from a thousand.
    let quantizer = g.subgraph("quantizer").subgraph("quantizers").subgraph("0");
    let project = quantizer.subgraph("in_project");
    let weight = project.load("weight", &[config.codebook_dim, config.hidden_size, 1]);
    let bias = project.load("bias", &[config.codebook_dim]);
    let latents = conv1d(
        &project,
        g.contiguous(g.transpose(features, 1, 2)),
        weight,
        Some(bias),
        1,
        0,
        1,
        1,
        dtype,
        device,
    )?;

    // Both sides normalized, so the dot product ranks the same way the distance does.
    let latents = normalize(
        g,
        g.contiguous(g.transpose(latents, 1, 2)),
        3,
        1e-12,
        dtype,
        device,
    )?;
    let codebook = normalize(
        g,
        quantizer
            .subgraph("codebook")
            .load("weight", &[config.codebook_size, config.codebook_dim]),
        2,
        1e-12,
        dtype,
        device,
    )?;

    let _ = frames;
    let _ = Extent::At(0);

    Ok(g.matmul(latents, g.transpose(codebook, 0, 1)))
}

/// Which codebook entry each frame is nearest to, from the [`similarity`] the graph produced.
///
/// One index per frame, which is one semantic token. The walk is here rather than in the graph
/// because `flint` has no argmax; see the module note for why that costs less than it sounds.
pub fn nearest(similarity: &[f32], codebook_size: usize) -> Vec<i32> {
    assert!(
        similarity.len().is_multiple_of(codebook_size),
        "the similarity is one row per frame"
    );

    similarity
        .chunks(codebook_size)
        .map(|row| {
            let mut best = 0;
            for (index, value) in row.iter().enumerate() {
                if *value > row[best] {
                    best = index;
                }
            }

            best as i32
        })
        .collect()
}

/// Tokens in, the features S2Mel's length regulator reads out: `(N, tokens)` to `(N, 2 * tokens,
/// hidden_size)`.
///
/// This is the half the pipeline runs. `codes` are the GPT's semantic tokens as indices into the
/// codebook, flattened to one dimension -- `lookup` takes a flat index tensor, the way
/// [`crate::w2v_bert`]'s distance table does -- and `tokens` is how many there are per sequence.
///
/// # It is not `similarity` run backwards
///
/// The decoder has its own twelve ConvNeXt blocks with their own weights, under `decoder` rather
/// than `encoder`. The two are the same shape and nothing else: a codec is not an invertible
/// function, and the trunk that reads a token is not the transpose of the one that wrote it.
///
/// # Where the frames come back
///
/// `down` halved the frame rate when the alphabet was built, so `decode` doubles it again at the
/// end -- each frame repeated once, then a width-three convolution over the pair. Upstream is
/// `F.interpolate(scale_factor=2, mode="nearest")` followed by `up`, and a repeat *is* nearest
/// interpolation at exactly two: concatenating the sequence with itself along a new last axis and
/// then flattening that axis puts every frame beside its own copy, which is what a view of
/// `(N, C, T, 2)` as `(N, C, 2T)` does with the last axis fastest.
///
/// There is no activation in front of this backbone. `quantize` puts a GELU after `down` and
/// `decode` puts none after `out_project`, which is asymmetric and is what the release does.
#[track_caller]
pub fn decode(
    g: &Graph,
    codes: Value,
    config: &Config,
    tokens: i32,
    dtype: DType,
    device: Device,
) -> Result<Value> {
    let quantizer = g.subgraph("quantizer").subgraph("quantizers").subgraph("0");

    // The entry each token stands for: (N * T) indices -> (N, T, codebook_dim).
    let codebook = quantizer
        .subgraph("codebook")
        .load("weight", &[config.codebook_size, config.codebook_dim]);
    let entries = g.view(
        g.lookup(codebook, codes),
        [
            Extent::At(1),
            Extent::At(tokens),
            Extent::At(config.codebook_dim),
        ],
    );

    // Back up to the width the trunk works in: eight numbers to a thousand.
    let project = quantizer.subgraph("out_project");
    let weight = project.load("weight", &[config.hidden_size, config.codebook_dim, 1]);
    let bias = project.load("bias", &[config.hidden_size]);
    let widened = conv1d(
        &project,
        g.contiguous(g.transpose(entries, 1, 2)),
        weight,
        Some(bias),
        1,
        0,
        1,
        1,
        dtype,
        device,
    )?;

    let decoder = g.subgraph("decoder");
    let trunk = vocos_backbone(
        &decoder.subgraph("0"),
        widened,
        config.hidden_size,
        config,
        dtype,
        device,
    )?;
    let features = Linear::graph(
        &decoder.subgraph("1"),
        trunk,
        config.vocos_dim,
        config.hidden_size,
        true,
    );

    // Double the frame rate back. (N, C, T) -> (N, C, T, 2) -> (N, C, 2T), last axis fastest, so
    // every frame is followed by its own copy.
    let by_channel = g.contiguous(g.transpose(features, 1, 2));
    let paired = g.cat(g.unsqueeze(by_channel, 3), g.unsqueeze(by_channel, 3), 3);
    let doubled = g.view(
        g.contiguous(paired),
        [
            Extent::of(features, 0),
            Extent::At(config.hidden_size),
            Extent::At(2 * tokens),
        ],
    );

    let up = g.subgraph("up");
    let weight = up.load("weight", &[config.hidden_size, config.hidden_size, 3]);
    let bias = up.load("bias", &[config.hidden_size]);
    let smoothed = conv1d(&up, doubled, weight, Some(bias), 1, 1, 1, 1, dtype, device)?;

    Ok(g.contiguous(g.transpose(smoothed, 1, 2)))
}
