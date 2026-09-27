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

//! ASTRAL: HuBERT's features in, one token every twenty milliseconds out.
//!
//! `Plachta/ASTRAL-quantization`'s encoder and quantizer, and nothing else of it. Seed-VC v2 runs
//! two of them over the same HuBERT features:
//!
//! - **wide**, 2048 codes (eleven bits), which keeps how something was said as well as what;
//!   the CFM draws the mel from these;
//! - **narrow**, 32 codes (five bits), which keeps little more than what was said; the AR reads
//!   these when it is asked to re-say the source in the reference's manner.
//!
//! Each is a 1x1 convolution down to 512, twelve ConvNeXt V2 blocks, and binary spherical
//! quantization: a projection to one number per bit, whose signs are the code. The quantizer
//! normalizes onto the sphere first, which moves no sign, so the code is the signs of the
//! projection and nothing more is computed. The ASR head and decoder ASTRAL was trained with are
//! not in the released `_light` checkpoints and are not needed.
//!
//! # GRN normalizes over time, not over channels
//!
//! ConvNeXt V2's global response normalization is `x * N(x)` where `N` is each channel's L2 norm
//! over the *spatial* axes divided by that norm's mean over channels. Ported to one dimension,
//! upstream takes `torch.norm(x, dim=1)` of a `(B, T, C)` tensor: the norm runs over time. So a
//! frame's output depends on the whole utterance -- and on where a long one is cut into pieces --
//! and this does the same.

use crate::audio::depthwise_conv1d;
use crate::flint::{DType, Device, Graph, Tensor, Value};
use crate::layers::{LayerNorm, Linear};
use crate::Result;

/// The width everything inside the encoder works at.
pub const DIM: i32 = 512;
/// What HuBERT hands it.
pub const INPUT_DIM: i32 = 1024;
const INTERMEDIATE: i32 = 1536;
const BLOCKS: i32 = 12;
const KERNEL: i32 = 7;

/// Bits in a wide code: 2048 codes.
pub const WIDE_BITS: i32 = 11;
/// Bits in a narrow code: 32 codes.
pub const NARROW_BITS: i32 = 5;

/// ConvNeXt V2's global response normalization over `x` `(1, T, C)`. See the module note.
fn grn(g: &Graph, x: Value, channels: i32, dtype: DType, device: Device) -> Result<Value> {
    let epsilon = g.constant(
        Tensor::from_f32(&[1], &[1e-6])?
            .to_device(device)?
            .cast(dtype)?,
    );

    // (1, T, C) -> (1, C): each channel's norm over time.
    let norms = g.sqrt(g.sum(g.square(x), 1));
    // (1, C) -> (1, 1): their mean over channels.
    let mean = g.unsqueeze(g.div_scalar(g.sum(norms, -1), channels as f32), -1);
    let scale = g.unsqueeze(g.div(norms, g.add(mean, epsilon)), 1);

    let gamma = g.load("gamma", &[1, 1, channels]);
    let beta = g.load("beta", &[1, 1, channels]);

    // The full-size operand first: `flint` broadcasts the right-hand side only.
    Ok(g.add(g.add(g.mul(g.mul(x, scale), gamma), beta), x))
}

/// One ConvNeXt V2 block over `x` `(1, C, T)`.
fn block(g: &Graph, x: Value, dtype: DType, device: Device) -> Result<Value> {
    let dw = g.subgraph("dwconv");
    let mixed = depthwise_conv1d(
        &dw,
        x,
        dw.load("weight", &[DIM, 1, KERNEL]),
        Some(dw.load("bias", &[DIM])),
        DIM,
        KERNEL,
        KERNEL / 2,
        1,
        dtype,
        device,
    )?;

    // `channels_first` layer norm, which is a layer norm over the channels of `(1, T, C)`.
    let normed = LayerNorm::graph(
        &g.subgraph("norm"),
        g.contiguous(g.transpose(mixed, 1, 2)),
        DIM,
        1e-6,
    );
    let wide = g.gelu(Linear::graph(
        &g.subgraph("pwconv1"),
        normed,
        DIM,
        INTERMEDIATE,
        true,
    ));
    let wide = grn(&g.subgraph("grn"), wide, INTERMEDIATE, dtype, device)?;
    let narrow = Linear::graph(&g.subgraph("pwconv2"), wide, INTERMEDIATE, DIM, true);

    Ok(g.add(x, g.contiguous(g.transpose(narrow, 1, 2))))
}

/// The encoder and the quantizer's projection over HuBERT's `hidden` `(1, T, 1024)`:
/// `(1, T, bits)`, whose signs are the code. See [`tokens`].
pub fn graph(g: &Graph, hidden: Value, bits: i32, dtype: DType, device: Device) -> Result<Value> {
    let encoder = g.subgraph("encoder");

    // The 1x1 input convolution, as the matrix it is.
    let projection = encoder.subgraph("input_projection");
    let weight = g.view(
        projection.load("weight", &[DIM, INPUT_DIM, 1]),
        [DIM, INPUT_DIM],
    );
    let projected = g.add(
        g.matmul(hidden, g.transpose(weight, 0, 1)),
        projection.load("bias", &[DIM]),
    );

    let mut running = g.contiguous(g.transpose(projected, 1, 2));
    let blocks = encoder.subgraph("blocks");
    for index in 0..BLOCKS {
        running = block(&blocks.subgraph(&index.to_string()), running, dtype, device)?;
    }

    Ok(Linear::graph(
        &g.subgraph("quantizer").subgraph("project_in"),
        g.contiguous(g.transpose(running, 1, 2)),
        DIM,
        bits,
        true,
    ))
}

/// The codes `projected` `(T * bits)` stands for, one per frame.
///
/// Bit `j` of a frame is whether its `j`-th number is above zero -- exactly zero is a zero bit,
/// as `torch.where(x > 0, ...)` has it -- and the first number is the most significant bit.
pub fn tokens(projected: &[f32], bits: i32) -> Vec<i32> {
    projected
        .chunks(bits as usize)
        .map(|frame| {
            frame
                .iter()
                .fold(0, |code, value| (code << 1) | i32::from(*value > 0.0))
        })
        .collect()
}

/// `tokens` with every run of one token said once: upstream's `duration_reduction_func` with an
/// n-gram of one, which is how it is always called.
pub fn reduce_durations(tokens: &[i32]) -> Vec<i32> {
    let mut out: Vec<i32> = Vec::with_capacity(tokens.len());
    for token in tokens {
        if out.last() != Some(token) {
            out.push(*token);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_first_number_is_the_most_significant_bit() {
        let projected = [0.3, -0.1, 0.0, 2.0, -5.0, -1.0, -1.0, -1.0, -1.0, 0.1];
        assert_eq!(tokens(&projected, 5), vec![0b10010, 0b00001]);
    }

    #[test]
    fn a_run_is_said_once() {
        assert_eq!(
            reduce_durations(&[3, 3, 3, 1, 1, 3, 2, 2]),
            vec![3, 1, 3, 2]
        );
        assert_eq!(reduce_durations(&[7]), vec![7]);
        assert!(reduce_durations(&[]).is_empty());
    }
}
