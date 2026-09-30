// Copyright (c) 2026 Xiaoyang Chen
//
// Part of libwaifu's port of Seed-VC (https://github.com/Plachtaa/seed-vc), which is licensed
// under the GNU General Public License version 3 -- and so is this file, unlike the rest of
// libwaifu, which is MIT. It is compiled only with the `gpl` feature (CMake: -DENABLE_GPL=ON), and a
// build with it on is covered by the GPL as a whole. See LICENSE-GPL-3.0 at the top of the repository.
//
// This program is free software: you can redistribute it and/or modify it under the terms of the
// GNU General Public License, version 3, as published by the Free Software Foundation.
//
// This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
// without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See
// the GNU General Public License for more details.
//
// SPDX-License-Identifier: GPL-3.0-only

//! Seed-VC v2's CFM: wide tokens and a voice in, a mel spectrogram out.
//!
//! Three pieces: the length regulator that lays the tokens onto mel frames, the DiT that gives one
//! step's direction, and [`solve_euler`], the loop that walks from noise to a mel with it.
//!
//! # How the DiT differs from IndexTTS's S2Mel
//!
//! Both come from the same author and look alike; [`crate::indextts::s2mel`] is the v1 lineage.
//! This one has no WaveNet, no U-net skips and no long skip; the timestep and the speaker are not
//! spread over every frame but put in front of the sequence as **two extra tokens**, time first,
//! and cut off again after the last layer. Its adaptive normalizations are DiT's six-way ones --
//! a shift, a scale and a gate for attention and for the feed-forward, all projected from the
//! timestep -- applied as `norm(x) * (1 + scale) + shift`, and the gate multiplies what each
//! sub-layer adds back.
//!
//! # The rotary table is the one the weights were trained under
//!
//! Interleaved pairs, as in S2Mel ([`crate::indextts::s2mel::rotate`] does the turning), but the
//! table is upstream's own, stored in the checkpoint at bfloat16 and read out of the package --
//! see `tools/seed_vc_exporter.py`. Rebuilding it here in full precision would move every cosine
//! by up to a part in five hundred.
//!
//! # Guidance has two rates
//!
//! `inference_v2.py` guides twice over, as MegaTTS3 does: once toward the content
//! (intelligibility) and once toward the voice (similarity). With both rates above zero the
//! network runs on three versions of its input -- everything; the content without the voice; and
//! nothing -- and they combine as
//!
//! ```text
//! (1 + a + b) * everything - a * nothing - b * content only
//! ```
//!
//! with `a` the intelligibility rate and `b` the similarity one. The three go through as one batch.

use crate::flint::{DType, Device, Extent, Graph, Value};
use crate::indextts::s2mel::{mish, rotate};
use crate::layers::Linear;
use crate::Result;

/// The DiT's widths, from `configs/v2/vc_wrapper.yaml`.
#[derive(Clone, Copy, Debug)]
pub struct Config {
    pub in_channels: i32,
    pub hidden_dim: i32,
    pub depth: i32,
    pub heads: i32,
    pub content_dim: i32,
    pub style_dim: i32,
    pub frequency_embedding_size: i32,
    pub eps: f32,
}

impl Config {
    pub fn seed_vc() -> Config {
        Config {
            in_channels: 80,
            hidden_dim: 512,
            depth: 13,
            heads: 8,
            content_dim: 512,
            style_dim: 192,
            frequency_embedding_size: 256,
            eps: 1e-5,
        }
    }

    pub fn head_dim(&self) -> i32 {
        self.hidden_dim / self.heads
    }

    /// Two thirds of four times the width, rounded up to a multiple of 256: `gpt_fast`'s rule.
    fn intermediate(&self) -> i32 {
        let two_thirds = 2 * (4 * self.hidden_dim) / 3;
        (two_thirds + 255) / 256 * 256
    }
}

/// The width of a wide token's embedding, and of the CFM's condition.
pub const CONDITION_DIM: i32 = 512;
/// How many wide codes there are.
pub const CODEBOOK: i32 = 2048;
const REGULATOR_STAGES: i32 = 4;

/// The CFM's length regulator: wide `tokens` `(tokens)` onto `frames` mel frames,
/// `(1, frames, 512)`.
///
/// An embedding, upstream's `F.interpolate(size=frames, mode="nearest")` -- which
/// [`Graph::upsample_nearest1d`] reproduces index for index, float32 rounding included -- and four
/// convolution-group norm-Mish stages. The fifth module of upstream's `Sequential` is an identity,
/// since it is 512 in and out.
pub fn regulator(
    g: &Graph,
    tokens: Value,
    frames: i32,
    dtype: DType,
    device: Device,
) -> Result<Value> {
    let channels = CONDITION_DIM;
    let table = g
        .subgraph("embedding")
        .load("weight", &[CODEBOOK, channels]);
    let embedded = g.cast(g.unsqueeze(g.lookup(table, tokens), 0), dtype);

    let over_time = g.contiguous(g.transpose(embedded, 1, 2));
    let mut running = g.upsample_nearest1d(over_time, frames);

    let model = g.subgraph("model");
    for stage in 0..REGULATOR_STAGES {
        let sub = model.subgraph(&(stage * 3).to_string());
        let weight = sub.load("weight", &[channels, channels, 3]);
        let bias = sub.load("bias", &[channels]);
        running =
            crate::audio::conv1d(&sub, running, weight, Some(bias), 1, 1, 1, 1, dtype, device)?;

        // One group, so `(N, C, 1, T)` normalizes over the same numbers `(N, C, T)` does.
        let norm = model.subgraph(&(stage * 3 + 1).to_string());
        running = g.squeeze(
            g.group_norm(
                g.unsqueeze(running, 2),
                Some(norm.load("weight", &[channels])),
                Some(norm.load("bias", &[channels])),
                1,
                1e-5,
            ),
            2,
        );
        running = mish(g, running, dtype, device)?;
    }

    Ok(g.contiguous(g.transpose(running, 1, 2)))
}

/// The sinusoid a timestep is embedded as before its projection, `(1, size)`: cosines then
/// sines, at a thousand times the time -- computed in float32 as upstream computes it.
pub fn timestep_embedding(t: f32, size: i32) -> Vec<f32> {
    let half = (size / 2) as usize;
    let log_period = (10000.0f64).ln() as f32;
    let mut values = vec![0.0f32; size as usize];

    for index in 0..half {
        let frequency = (-log_period * index as f32 / half as f32).exp();
        let angle = 1000.0f32 * t * frequency;
        values[index] = angle.cos();
        values[half + index] = angle.sin();
    }
    values
}

/// `norm(x) * (1 + scale) + shift`, the RMS norm's own weight inside.
fn modulate(
    g: &Graph,
    x: Value,
    norm_weight: Value,
    shift: Value,
    scale: Value,
    eps: f32,
) -> Value {
    let normed = g.rms_norm(x, norm_weight, eps);
    g.add(g.add(g.mul(normed, scale), normed), shift)
}

/// One transformer block over `x` `(N, L, D)`, modulated by `c` `(N, 1, D)`.
#[allow(clippy::too_many_arguments)]
fn block(
    g: &Graph,
    x: Value,
    c: Value,
    config: &Config,
    length: i32,
    cos: Value,
    sin: Value,
) -> Value {
    let d = config.hidden_dim;
    let (heads, head_dim) = (config.heads, config.head_dim());

    let norm = g.subgraph("attention_norm");
    let emb = Linear::graph(&norm.subgraph("linear"), g.silu(c), d, 6 * d, true);
    let chunk = |index: i32| g.slice(emb, -1, index * d, (index + 1) * d);
    let (shift_msa, scale_msa, gate_msa) = (chunk(0), chunk(1), chunk(2));
    let (shift_mlp, scale_mlp, gate_mlp) = (chunk(3), chunk(4), chunk(5));

    let normed = modulate(
        g,
        x,
        norm.subgraph("norm").load("weight", &[d]),
        shift_msa,
        scale_msa,
        config.eps,
    );

    // Attention: one projection to query, key and value, interleaved rotary, not causal.
    let attention = g.subgraph("attention");
    let qkv = Linear::graph(&attention.subgraph("wqkv"), normed, d, 3 * d, false);
    let part = |index: i32| {
        let taken = g.contiguous(g.slice(qkv, -1, index * d, (index + 1) * d));
        let shaped = g.view(
            taken,
            [
                Extent::of(x, 0),
                Extent::At(length),
                Extent::At(heads),
                Extent::At(head_dim),
            ],
        );
        g.contiguous(g.transpose(shaped, 1, 2))
    };
    let query = rotate(g, part(0), heads, head_dim, length, cos, sin);
    let key = rotate(g, part(1), heads, head_dim, length, cos, sin);
    let out = g.attention(query, key, part(2), false);
    let merged = g.view(
        g.contiguous(g.transpose(out, 1, 2)),
        [Extent::of(x, 0), Extent::At(length), Extent::At(d)],
    );
    let attended = Linear::graph(&attention.subgraph("wo"), merged, d, d, false);
    // The full-size operand first: `flint` broadcasts the right-hand side only.
    let x = g.add(x, g.mul(attended, gate_msa));

    let normed = modulate(
        g,
        x,
        g.subgraph("ffn_norm").load("weight", &[d]),
        shift_mlp,
        scale_mlp,
        config.eps,
    );
    let ff = g.subgraph("feed_forward");
    let inner = config.intermediate();
    let gate = Linear::graph(&ff.subgraph("w1"), normed, d, inner, false);
    let up = Linear::graph(&ff.subgraph("w3"), normed, d, inner, false);
    let down = Linear::graph(&ff.subgraph("w2"), g.mul(g.silu(gate), up), inner, d, false);

    g.add(x, g.mul(down, gate_mlp))
}

/// One step's direction: `(N, in_channels, frames)`.
///
/// `x` and `prompt_x` are `(N, in_channels, frames)`, `cond` the regulated condition
/// `(N, frames, content_dim)`, `style` `(N, style_dim)`, `time` [`timestep_embedding`]s
/// `(N, frequency_embedding_size)`, and `cos`, `sin` the first `frames + 2` rows of the rotary
/// table -- two more than the frames, for the time and style tokens in front.
#[allow(clippy::too_many_arguments)]
pub fn graph(
    g: &Graph,
    x: Value,
    prompt_x: Value,
    cond: Value,
    style: Value,
    time: Value,
    config: &Config,
    frames: i32,
    cos: Value,
    sin: Value,
) -> Value {
    let d = config.hidden_dim;
    let length = frames + 2;

    let t1 = {
        let sub = g.subgraph("t_embedder").subgraph("mlp");
        let first = Linear::graph(
            &sub.subgraph("0"),
            time,
            config.frequency_embedding_size,
            d,
            true,
        );
        Linear::graph(&sub.subgraph("2"), g.silu(first), d, d, true)
    };

    let cond = Linear::graph(
        &g.subgraph("cond_projection"),
        cond,
        config.content_dim,
        d,
        true,
    );
    let mel = g.contiguous(g.transpose(x, 1, 2));
    let prompt = g.contiguous(g.transpose(prompt_x, 1, 2));
    let joined = g.cat(g.cat(mel, prompt, -1), cond, -1);
    let merged = Linear::graph(
        &g.subgraph("cond_x_merge_linear"),
        joined,
        2 * config.in_channels + d,
        d,
        true,
    );

    let style = Linear::graph(&g.subgraph("style_in"), style, config.style_dim, d, true);
    let sequence = g.cat(
        g.cat(g.unsqueeze(t1, 1), g.unsqueeze(style, 1), 1),
        merged,
        1,
    );

    let c = g.unsqueeze(t1, 1);
    let transformer = g.subgraph("transformer");
    let layers = transformer.subgraph("layers");
    let mut running = sequence;
    for index in 0..config.depth {
        running = block(
            &layers.subgraph(&index.to_string()),
            running,
            c,
            config,
            length,
            cos,
            sin,
        );
    }

    // The final norm: `scale` comes first in its projection, then `shift`.
    let norm = transformer.subgraph("norm");
    let emb = Linear::graph(&norm.subgraph("linear"), g.silu(c), d, 2 * d, true);
    let scale = g.slice(emb, -1, 0, d);
    let shift = g.slice(emb, -1, d, 2 * d);
    let normed = modulate(
        g,
        running,
        norm.subgraph("norm").load("weight", &[d]),
        shift,
        scale,
        config.eps,
    );

    // The time and style tokens off the front.
    let frames_only = g.contiguous(g.slice(normed, 1, 2, length));
    let mlp = g.subgraph("final_mlp");
    let hidden = g.silu(Linear::graph(&mlp.subgraph("0"), frames_only, d, d, true));
    let out = Linear::graph(&mlp.subgraph("2"), hidden, d, config.in_channels, true);

    g.contiguous(g.transpose(out, 1, 2))
}

/// The times the trajectory is evaluated at: `steps + 1` points from 0 to 1, bunched toward the
/// start as `1 - cos(pi / 2 * t)` bunches them, in float32 as upstream builds them.
pub fn time_span(steps: i32) -> Vec<f32> {
    let points = steps + 1;
    let step = 1.0f32 / steps as f32;
    (0..points)
        .map(|index| {
            // `torch.linspace` in float32: the first half counted up from the start, the second
            // half counted down from the end, so that both ends are exact.
            let t = match index < points / 2 {
                true => step * index as f32,
                false => 1.0f32 - step * (points - index - 1) as f32,
            };
            t + -((std::f32::consts::PI / 2.0 * t).cos() - 1.0 + t)
        })
        .collect()
}

/// The two guidance rates, as `inference_v2.py` names them.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Guidance {
    /// Toward the content: how far the full pass is pushed from the pass with nothing.
    pub intelligibility: f32,
    /// Toward the voice: how far it is pushed from the pass with the content but no voice.
    pub similarity: f32,
}

/// Which versions of the input one step runs the network on, in the order they are batched.
///
/// `(true, true)` is everything; `(false, true)` the content without the prompt and the style;
/// `(false, false)` nothing at all.
pub fn passes(guidance: Guidance) -> Vec<(bool, bool)> {
    let (a, b) = (guidance.intelligibility, guidance.similarity);
    match (a == 0.0, b == 0.0) {
        (true, true) => vec![(true, true)],
        (true, false) => vec![(true, true), (false, true)],
        (false, true) => vec![(true, true), (false, false)],
        (false, false) => vec![(true, true), (false, true), (false, false)],
    }
}

/// The passes' directions, `passes.len()` rows of `row` numbers each, combined into one.
pub fn combine(guidance: Guidance, directions: &[f32], row: usize) -> Vec<f32> {
    let (a, b) = (guidance.intelligibility, guidance.similarity);
    let at = |pass: usize| &directions[pass * row..(pass + 1) * row];

    match (a == 0.0, b == 0.0) {
        (true, true) => at(0).to_vec(),
        (true, false) => at(0)
            .iter()
            .zip(at(1))
            .map(|(full, content)| (1.0 + b) * full - b * content)
            .collect(),
        (false, true) => at(0)
            .iter()
            .zip(at(1))
            .map(|(full, nothing)| (1.0 + a) * full - a * nothing)
            .collect(),
        (false, false) => at(0)
            .iter()
            .zip(at(1))
            .zip(at(2))
            .map(|((full, content), nothing)| (1.0 + a + b) * full - a * nothing - b * content)
            .collect(),
    }
}

/// Walk from `noise` `(channels * frames)` to a mel, `evaluate(x, t)` giving the guided direction
/// at `x` and time `t`.
///
/// Upstream's `solve_euler`. The first `prompt_frames` frames of `x` -- the part the prompt covers
/// -- are zeroed before the first step and after every one, so the model is never asked to draw
/// what it was given. `t` is carried by adding `dt`, and `dt` re-derived from the span each step,
/// as upstream carries them.
pub fn solve_euler<F>(
    noise: &[f32],
    channels: usize,
    frames: usize,
    prompt_frames: usize,
    steps: i32,
    mut evaluate: F,
) -> Result<Vec<f32>>
where
    F: FnMut(&[f32], f32) -> Result<Vec<f32>>,
{
    let span = time_span(steps);
    let prompt_frames = prompt_frames.min(frames);
    let zero = |x: &mut [f32]| {
        for channel in 0..channels {
            x[channel * frames..channel * frames + prompt_frames].fill(0.0);
        }
    };

    let mut x = noise.to_vec();
    zero(&mut x);

    let mut t = span[0];
    let mut dt = span[1] - span[0];
    for step in 1..span.len() {
        let direction = evaluate(&x, t)?;
        for (value, velocity) in x.iter_mut().zip(&direction) {
            *value += dt * velocity;
        }
        t += dt;
        if step < span.len() - 1 {
            dt = span[step + 1] - t;
        }
        zero(&mut x);
    }

    Ok(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_span_runs_from_zero_to_one_bunched_at_the_start() {
        let span = time_span(4);
        assert_eq!(span.len(), 5);
        assert_eq!(span[0], 0.0);
        assert!((span[4] - 1.0).abs() < 1e-6);
        // 1 - cos(pi / 8) for the first quarter: far less than a quarter.
        assert!((span[1] - (1.0 - (std::f32::consts::PI / 8.0).cos())).abs() < 1e-6);
    }

    #[test]
    fn a_constant_direction_integrates_to_one_unit_of_time() {
        let out = solve_euler(&[0.0; 6], 2, 3, 1, 7, |_, _| Ok(vec![2.0; 6])).unwrap();
        // The prompt's frame held at zero, every other frame moved by twice one unit of time.
        assert_eq!(out[0], 0.0);
        assert_eq!(out[3], 0.0);
        for index in [1, 2, 4, 5] {
            assert!((out[index] - 2.0).abs() < 1e-5, "{out:?}");
        }
    }

    #[test]
    fn guidance_runs_only_the_passes_its_rates_need() {
        let both = Guidance {
            intelligibility: 0.7,
            similarity: 0.5,
        };
        assert_eq!(passes(both).len(), 3);
        let combined = combine(both, &[1.0, 2.0, 4.0], 1);
        assert!((combined[0] - ((1.0 + 0.7 + 0.5) * 1.0 - 0.7 * 4.0 - 0.5 * 2.0)).abs() < 1e-6);

        let none = Guidance {
            intelligibility: 0.0,
            similarity: 0.0,
        };
        assert_eq!(passes(none), vec![(true, true)]);
    }

    #[test]
    fn the_timestep_is_cosines_then_sines() {
        let embedding = timestep_embedding(0.0, 8);
        assert_eq!(embedding, vec![1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]);
    }
}
