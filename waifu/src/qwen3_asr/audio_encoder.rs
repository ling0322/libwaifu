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

//! Qwen3-ASR's audio encoder: a recording's mel in, one row for the decoder per 80 ms out.
//!
//! Upstream's `Qwen3ASRAudioEncoder`, the encoder of Qwen3-Omni's audio tower. Three things make
//! it more than a Whisper encoder, and each is reproduced as upstream does it rather than as a
//! cleaner version of it would:
//!
//! - **The convolutions read hundred-frame chunks, not the recording.** The mel is cut every
//!   second, the last piece padded with zeros to the length of the others, and the three
//!   stride-two convolutions run over the pieces as a batch. Every whole chunk becomes thirteen
//!   tokens. The padding is part of the arithmetic -- the bias of the second convolution, run over
//!   the first's output on zeros, is not zero -- so the last chunk is padded and not cut short.
//! - **The position table starts again in every chunk.** Rows `0..13` of a sinusoidal table are
//!   added to each chunk's thirteen tokens; position is a place within a second, not in the
//!   recording.
//! - **Attention stays inside eight-second windows.** The tokens of eight chunks -- 104 -- attend
//!   to each other and to nothing else, and the last window is whatever is left. Here the whole
//!   windows are attended to under [`MaskType::BlockDiagonal`] where the device has a kernel for
//!   it ([`Windows::Masked`], CUDA's), and as one batch of attention for the whole windows and a
//!   second for the last everywhere else ([`Windows::Split`]): the same mask, without a mask.
//!
//! Then pre-norm layers with GELU, a last norm, and two projections out to the decoder's width.

use crate::flint::{DType, Extent, Graph, MaskType, PadMode, Tensor, Value};
use crate::layers::{Conv2d, LayerNorm, Linear};
use crate::Result;

use super::config::{tokens_of, AudioConfig};

/// `nn.LayerNorm`'s.
const NORM_EPS: f32 = 1e-5;

/// How attention is kept inside its windows: the two give the same answer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Windows {
    /// One attention under a block diagonal mask, which needs a kernel that takes one: CUDA's
    /// fused kernel, in half, on Ampere and after.
    Masked,
    /// The whole windows as one batch of attention with no mask and the last as another, which
    /// every device runs.
    Split,
}

/// How one recording's mel is cut up, worked out on the host.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Geometry {
    /// Mel frames.
    pub frames: i32,
    /// Chunks the convolutions read.
    pub chunks: i32,
    /// How long every chunk is once the last is padded: a whole chunk, or the recording where it
    /// is shorter than one.
    pub padded: i32,
    /// Tokens the convolutions make of a padded chunk.
    pub per_chunk: i32,
    /// Tokens kept, which is what the decoder is handed.
    pub tokens: i32,
    /// Tokens that attend to one another.
    pub window: i32,
}

impl Geometry {
    pub fn of(frames: i32, config: &AudioConfig) -> Geometry {
        let chunk = config.chunk();
        let chunks = (frames + chunk - 1) / chunk;
        let padded = frames.min(chunk);
        let per_chunk = (0..3).fold(padded, |length, _| (length - 1) / 2 + 1);
        Geometry {
            frames,
            chunks,
            padded,
            per_chunk,
            tokens: tokens_of(frames, chunk),
            window: per_chunk * (config.n_window_infer / chunk),
        }
    }
}

/// `SinusoidsPositionEmbedding(length, channels)`: sines in the first half, cosines in the second.
pub fn positions(length: i32, channels: i32) -> Result<Tensor> {
    let half = (channels / 2) as usize;
    let increment = (10000f64).ln() / (half as f64 - 1.0);
    let mut table = Vec::with_capacity(length as usize * channels as usize);
    for position in 0..length {
        let angles: Vec<f64> = (0..half)
            .map(|index| position as f64 * (-increment * index as f64).exp())
            .collect();
        table.extend(angles.iter().map(|angle| angle.sin() as f32));
        table.extend(angles.iter().map(|angle| angle.cos() as f32));
    }
    Ok(Tensor::from_f32(&[length, channels], &table)?)
}

/// Attention within windows of `window` rows of `(tokens, d)` queries, keys and values.
fn windowed(
    g: &Graph,
    (q, k, v): (Value, Value, Value),
    tokens: i32,
    window: i32,
    heads: i32,
    head_dim: i32,
) -> Value {
    let width = heads * head_dim;
    let attend = |from: i32, count: i32, windows: i32| {
        let length = count / windows;
        let split = |x: Value| {
            let x = g.slice(x, 0, from, from + count);
            let x = g.view(
                x,
                [
                    Extent::At(windows),
                    Extent::At(length),
                    Extent::At(heads),
                    Extent::At(head_dim),
                ],
            );
            g.contiguous(g.transpose(x, 1, 2))
        };
        let out = g.attention(split(q), split(k), split(v), MaskType::None);
        g.view(g.contiguous(g.transpose(out, 1, 2)), [count, width])
    };

    let whole = tokens / window;
    let rest = tokens % window;
    match (whole, rest) {
        (0, rest) => attend(0, rest, 1),
        (whole, 0) => attend(0, whole * window, whole),
        (whole, rest) => g.cat(
            attend(0, whole * window, whole),
            attend(whole * window, rest, 1),
            0,
        ),
    }
}

/// Attention within windows of `window` rows, as one call under a block diagonal mask.
fn masked(
    g: &Graph,
    (q, k, v): (Value, Value, Value),
    tokens: i32,
    window: i32,
    heads: i32,
    head_dim: i32,
) -> Value {
    let split = |x: Value| {
        let x = g.view(x, [1, tokens, heads, head_dim]);
        g.contiguous(g.transpose(x, 1, 2))
    };
    let out = g.attention(
        split(q),
        split(k),
        split(v),
        MaskType::BlockDiagonal(window),
    );
    g.view(
        g.contiguous(g.transpose(out, 1, 2)),
        [tokens, heads * head_dim],
    )
}

fn layer(
    g: &Graph,
    x: Value,
    geometry: &Geometry,
    config: &AudioConfig,
    windows: Windows,
) -> Value {
    let d = config.d_model;

    let attn = g.subgraph("self_attn");
    let normed = LayerNorm::graph(&g.subgraph("self_attn_layer_norm"), x, d, NORM_EPS);
    let project = |name: &str, input: Value| Linear::graph(&attn.subgraph(name), input, d, d, true);
    let qkv = (
        project("q_proj", normed),
        project("k_proj", normed),
        project("v_proj", normed),
    );
    let (tokens, window, heads) = (geometry.tokens, geometry.window, config.heads);
    let attended = match windows {
        Windows::Masked => masked(g, qkv, tokens, window, heads, d / heads),
        Windows::Split => windowed(g, qkv, tokens, window, heads, d / heads),
    };
    let x = g.add(x, project("out_proj", attended));

    let normed = LayerNorm::graph(&g.subgraph("final_layer_norm"), x, d, NORM_EPS);
    let hidden = g.gelu(Linear::graph(
        &g.subgraph("fc1"),
        normed,
        d,
        config.ffn_dim,
        true,
    ));
    g.add(
        x,
        Linear::graph(&g.subgraph("fc2"), hidden, config.ffn_dim, d, true),
    )
}

/// The encoder over `mel` `(1, mel_bins, frames)`, with `positions` the first `per_chunk` rows of
/// the table in `dtype`: `(tokens, output_dim)`.
pub fn graph(
    g: &Graph,
    mel: Value,
    positions: Value,
    geometry: &Geometry,
    config: &AudioConfig,
    dtype: DType,
    windows: Windows,
) -> Value {
    let (chunks, padded) = (geometry.chunks, geometry.padded);
    let channels = config.downsample_hidden_size;

    // `(1, bands, frames)` padded out to whole chunks, then `(chunks, 1, bands, padded)`.
    let x = g.cast(mel, dtype);
    let x = g.pad1d(x, 0, chunks * padded - geometry.frames, PadMode::Zero);
    let x = g.view(
        x,
        [
            Extent::At(config.mel_bins),
            Extent::At(chunks),
            Extent::At(padded),
        ],
    );
    let x = g.unsqueeze(g.contiguous(g.transpose(x, 0, 1)), 1);

    let mut x = x;
    for (index, name) in ["conv2d1", "conv2d2", "conv2d3"].iter().enumerate() {
        let inputs = if index == 0 { 1 } else { channels };
        x = g.gelu(Conv2d::graph(
            &g.subgraph(name),
            x,
            inputs,
            channels,
            3,
            2,
            1,
        ));
    }

    // `(chunks, channels, bands, time)` to `(chunks, time, channels * bands)`, channel-major as
    // upstream's `permute(0, 3, 1, 2)` flattens it.
    let bands = config.bands_after();
    let x = g.contiguous(g.transpose(g.transpose(x, 2, 3), 1, 2));
    let x = g.view(
        x,
        [
            Extent::At(chunks),
            Extent::At(geometry.per_chunk),
            Extent::At(channels * bands),
        ],
    );
    let x = Linear::graph(
        &g.subgraph("conv_out"),
        x,
        channels * bands,
        config.d_model,
        false,
    );
    let x = g.add(x, positions);

    // The chunks one after the other, and the padding's tokens off the end.
    let x = g.view(
        x,
        [
            Extent::At(chunks * geometry.per_chunk),
            Extent::At(config.d_model),
        ],
    );
    let mut x = g.slice(x, 0, 0, geometry.tokens);

    for index in 0..config.layers {
        x = layer(
            &g.subgraph("layers").subgraph(&index.to_string()),
            x,
            geometry,
            config,
            windows,
        );
    }

    let x = LayerNorm::graph(&g.subgraph("ln_post"), x, config.d_model, NORM_EPS);
    let x = g.gelu(Linear::graph(
        &g.subgraph("proj1"),
        x,
        config.d_model,
        config.d_model,
        true,
    ));
    Linear::graph(
        &g.subgraph("proj2"),
        x,
        config.d_model,
        config.output_dim,
        true,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> AudioConfig {
        AudioConfig {
            mel_bins: 128,
            d_model: 896,
            layers: 18,
            heads: 14,
            ffn_dim: 3584,
            downsample_hidden_size: 480,
            output_dim: 1024,
            n_window: 50,
            n_window_infer: 800,
            max_source_positions: 1500,
        }
    }

    #[test]
    fn a_recording_is_cut_into_seconds_and_eight_second_windows() {
        // 3.48 s: three whole chunks and 48 frames, one window.
        let short = Geometry::of(348, &config());
        assert_eq!(
            (
                short.chunks,
                short.padded,
                short.per_chunk,
                short.tokens,
                short.window
            ),
            (4, 100, 13, 45, 104)
        );
        // 13.74 s: two windows, the second short.
        let long = Geometry::of(1374, &config());
        assert_eq!((long.chunks, long.tokens, long.window), (14, 179, 104));
        // Under a second is one chunk as long as the recording, and one window.
        let tiny = Geometry::of(60, &config());
        assert_eq!(
            (tiny.chunks, tiny.padded, tiny.per_chunk, tiny.tokens),
            (1, 60, 8, 8)
        );
        assert!(tiny.window >= tiny.tokens);
    }

    #[test]
    fn the_position_table_is_sines_then_cosines() {
        let table = positions(3, 8).unwrap().to_vec_f32().unwrap();
        // Position 0: sin 0 = 0, cos 0 = 1.
        assert_eq!(&table[..8], &[0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]);
        // Position 1, frequency 0: sin 1.
        assert!((table[8] - 1f32.sin()).abs() < 1e-6);
        // The last frequency is 1 / 10000.
        assert!((table[8 + 3] - (1e-4f32).sin()).abs() < 1e-7);
    }
}
