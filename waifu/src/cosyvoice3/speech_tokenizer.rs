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

//! S3Tokenizer v3: a recording's Whisper mel in, twenty-five speech tokens a second out.
//!
//! `speech_tokenizer_v3.onnx`, as S3Tokenizer's PyTorch port reads it: two strided convolutions,
//! twelve pre-norm blocks 1280 wide, and a finite scalar quantizer. Each block's attention has two
//! things a Whisper encoder's does not -- a rotary embedding, rotating half against half with base
//! 10 000, and an FSMN memory: the values run through a depthwise convolution 31 wide and added
//! back onto the attention's output.
//!
//! The quantizer is eight numbers per frame, each squashed by `tanh`, rounded to -1, 0 or 1 and
//! read as a digit in base three. The eight projections come out of the graph; the rest is on the
//! host, where a rounding that ties to even is one call rather than an operator.

use crate::flint::{DType, Device, Extent, Graph, Value};
use crate::layers::Linear;
use crate::Result;

pub const MELS: i32 = 128;
pub const WIDTH: i32 = 1280;
pub const HEADS: i32 = 20;
pub const LAYERS: i32 = 12;
pub const FSMN_KERNEL: i32 = 31;
/// Eight ternary digits, 3^8 = 6561 tokens.
pub const DIGITS: i32 = 8;

/// How many frames the two stride-two convolutions leave of `frames`.
pub fn frames_out(frames: i32) -> i32 {
    let once = |t: i32| (t + 2 - 3) / 2 + 1;
    once(once(frames))
}

fn layer_norm(g: &Graph, x: Value) -> Value {
    g.layer_norm(
        x,
        Some(g.load("weight", &[WIDTH])),
        Some(g.load("bias", &[WIDTH])),
        1e-5,
    )
}

/// Rotate `(1, T, H, D)` half against half, by `(T, H, D / 2)` tables.
fn rotate(g: &Graph, x: Value, cos: Value, sin: Value, half: i32) -> Value {
    let first = g.slice(x, 3, 0, half);
    let second = g.slice(x, 3, half, 2 * half);
    let left = g.sub(g.mul(first, cos), g.mul(second, sin));
    let right = g.add(g.mul(first, sin), g.mul(second, cos));
    g.cat(left, right, 3)
}

fn block(g: &Graph, x: Value, cos: Value, sin: Value) -> Value {
    let head_dim = WIDTH / HEADS;
    let length = Extent::of(x, 1);
    let attn = g.subgraph("attn");

    let normed = layer_norm(&g.subgraph("attn_ln"), x);
    let q = Linear::graph(&attn.subgraph("query"), normed, WIDTH, WIDTH, true);
    let k = Linear::graph(&attn.subgraph("key"), normed, WIDTH, WIDTH, false);
    let v = Linear::graph(&attn.subgraph("value"), normed, WIDTH, WIDTH, true);

    let heads = |value: Value| {
        g.view(
            value,
            [
                Extent::At(1),
                length,
                Extent::At(HEADS),
                Extent::At(head_dim),
            ],
        )
    };
    let q = rotate(g, heads(q), cos, sin, head_dim / 2);
    let k = rotate(g, heads(k), cos, sin, head_dim / 2);

    // The FSMN memory: the values, before any rotation, convolved along time channel by channel
    // and added back onto themselves.
    let memory = {
        let across = g.contiguous(g.transpose(v, 1, 2));
        let weight = attn
            .subgraph("fsmn_block")
            .load("weight", &[WIDTH, 1, FSMN_KERNEL]);
        let convolved = g.conv1d(across, weight, None, 1, FSMN_KERNEL / 2, 1, WIDTH);
        g.add(g.contiguous(g.transpose(convolved, 1, 2)), v)
    };

    let attended = g.attention(
        g.contiguous(g.transpose(q, 1, 2)),
        g.contiguous(g.transpose(k, 1, 2)),
        g.contiguous(g.transpose(heads(v), 1, 2)),
        false,
    );
    let attended = g.view(
        g.contiguous(g.transpose(attended, 1, 2)),
        [Extent::At(1), length, Extent::At(WIDTH)],
    );
    let out = g.add(
        Linear::graph(&attn.subgraph("out"), attended, WIDTH, WIDTH, true),
        memory,
    );
    let x = g.add(x, out);

    let mlp = g.subgraph("mlp");
    let normed = layer_norm(&g.subgraph("mlp_ln"), x);
    let wide = g.gelu(Linear::graph(
        &mlp.subgraph("0"),
        normed,
        WIDTH,
        4 * WIDTH,
        true,
    ));
    g.add(
        x,
        Linear::graph(&mlp.subgraph("2"), wide, 4 * WIDTH, WIDTH, true),
    )
}

/// The encoder and the quantizer's projection: `(1, 128, T)` in, `(1, T', 8)` out, before
/// `tanh`. `cos` and `sin` are [`rotary`]'s tables for `T'` frames.
pub fn graph(
    g: &Graph,
    mel: Value,
    cos: Value,
    sin: Value,
    dtype: DType,
    _device: Device,
) -> Result<Value> {
    let encoder = g.subgraph("encoder");
    let conv = |name: &str, x: Value, in_channels: i32| {
        let sub = encoder.subgraph(name);
        let weight = sub.load("weight", &[WIDTH, in_channels, 3]);
        let bias = sub.load("bias", &[WIDTH]);
        g.gelu(g.conv1d(x, weight, Some(bias), 2, 1, 1, 1))
    };

    let x = conv("conv1", g.cast(mel, dtype), MELS);
    let x = conv("conv2", x, WIDTH);
    let mut x = g.contiguous(g.transpose(x, 1, 2));

    for index in 0..LAYERS {
        x = block(
            &encoder.subgraph("blocks").subgraph(&index.to_string()),
            x,
            g.cast(cos, dtype),
            g.cast(sin, dtype),
        );
    }

    Ok(Linear::graph(
        &g.subgraph("quantizer")
            .subgraph("_codebook")
            .subgraph("project_down"),
        x,
        WIDTH,
        DIGITS,
        true,
    ))
}

/// The rotary tables for `frames` positions, `(frames, heads, head_dim / 2)`, as float32 values.
pub fn rotary(frames: i32) -> (Vec<f32>, Vec<f32>, [i32; 3]) {
    let half = (WIDTH / HEADS / 2) as usize;
    let head_dim = (WIDTH / HEADS) as f64;
    let mut cos = Vec::new();
    let mut sin = Vec::new();
    for position in 0..frames {
        for _ in 0..HEADS {
            for index in 0..half {
                let frequency = 1.0 / 10000f64.powf(2.0 * index as f64 / head_dim);
                // `precompute_freqs_cis` works in float32, and so does this.
                let angle = (position as f32) * (frequency as f32);
                cos.push(angle.cos());
                sin.push(angle.sin());
            }
        }
    }
    (cos, sin, [frames, HEADS, half as i32])
}

/// The quantizer on the host: `(T', 8)` projections to `T'` tokens.
pub fn quantize(projections: &[f32]) -> Vec<i32> {
    projections
        .chunks(DIGITS as usize)
        .map(|digits| {
            digits
                .iter()
                .enumerate()
                .map(|(place, value)| {
                    // `h.tanh() * 0.9990000128746033`, `round()` -- ties to even -- then `+ 1`.
                    // Upstream's literal is 0.9990000128746033, which is 0.999 as a float32.
                    let digit = (value.tanh() * 0.999_f32).round_ties_even() + 1.0;
                    digit as i32 * 3i32.pow(place as u32)
                })
                .sum()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn three_seconds_is_seventy_five_tokens() {
        assert_eq!(frames_out(300), 75);
        assert_eq!(frames_out(348), 87);
    }

    #[test]
    fn digits_are_little_endian_base_three() {
        let mut projections = vec![0.0f32; 8];
        assert_eq!(quantize(&projections), vec![3280]); // every digit 1
        projections[0] = 5.0; // tanh -> ~1, digit 2
        projections[1] = -5.0; // digit 0
        assert_eq!(quantize(&projections), vec![3280 + 1 - 3]);
    }
}
