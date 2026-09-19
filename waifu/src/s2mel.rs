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

//! S2Mel's denoiser: semantic tokens and a voice in, a mel spectrogram out.
//!
//! This is the middle of IndexTTS-2.5. The GPT in front of it produces semantic tokens; this
//! turns them into the mel spectrogram a BigVGAN makes audio of. It is a flow-matching model, so
//! what the network itself computes is one step of a trajectory -- given a partly denoised mel
//! and a time, the direction to move -- and a sampler calls it a handful of times.
//!
//! What it is conditioned on is worth naming, because there are four different things:
//!
//! - `cond`, the semantic tokens, one 512-wide vector per output frame;
//! - `prompt_x`, the reference speaker's own mel for the frames that have one and zeros after,
//!   which is what makes this continue a voice rather than invent one;
//! - `style`, 192 numbers from [`crate::campplus`], which is who is speaking;
//! - `t`, where along the trajectory this step is.
//!
//! # The shape of it
//!
//! A thirteen-layer transformer over the frame axis, and then a WaveNet over the same axis. The
//! transformer is `gpt_fast`'s rather than a diffusion model's: RMS normalization, rotary
//! positions, a SwiGLU feed forward. What makes it a *diffusion* transformer is that every
//! normalization is adaptive -- the timestep projects to a weight and a bias that scale and shift
//! it -- and that half the layers hand a skip connection forward to the other half, U-net
//! fashion, which is what `uvit_skip_connection` means.
//!
//! # Two things here are not what a reader would guess
//!
//! **The rotary embedding is interleaved, not split.** `flint`'s `rotary_embedding` rotates the
//! NeoX way, pairing element `i` with element `i + D/2`. This model pairs `2i` with `2i + 1`,
//! which is the other convention entirely and gives different numbers for the same weights. So
//! [`rotate`] does it explicitly, out of the cosine and sine tables the caller passes in -- the
//! same thing `anima`'s transformer does, and for the same reason.
//!
//! **The content is read as continuous, whatever the configuration says.** `config.yaml` sets
//! `content_type: 'discrete'` and the model has a `cond_embedder` for it, but the reference's
//! `forward` assigns `cond_in_module = self.cond_projection` unconditionally, with the line that
//! would have chosen between them commented out just above. So the embedding is never reached and
//! the projection always is. This follows the code rather than the configuration.

use crate::audio::conv1d;
use crate::flint::{DType, Device, Extent, Graph, Tensor, Value};
use crate::layers::Linear;
use crate::Result;

/// The widths of one S2Mel denoiser, as `config.yaml` states them.
#[derive(Clone, Copy, Debug)]
pub struct Config {
    /// Mel bands, in and out.
    pub in_channels: i32,
    /// The transformer's width.
    pub hidden_dim: i32,
    /// How many transformer layers.
    pub depth: i32,
    pub num_heads: i32,
    /// The width of one semantic token.
    pub content_dim: i32,
    /// What [`crate::campplus`] produces.
    pub style_dim: i32,
    /// The WaveNet's width, which is also the transformer's here.
    pub wavenet_dim: i32,
    pub wavenet_layers: i32,
    pub wavenet_kernel: i32,
    /// How wide the sinusoid a timestep is embedded as is, before its own projection.
    pub frequency_embedding_size: i32,
}

impl Config {
    /// What IndexTTS-2.5's `config.yaml` describes.
    pub fn indextts() -> Config {
        Config {
            in_channels: 80,
            hidden_dim: 512,
            depth: 13,
            num_heads: 8,
            content_dim: 512,
            style_dim: 192,
            wavenet_dim: 512,
            wavenet_layers: 8,
            wavenet_kernel: 5,
            frequency_embedding_size: 256,
        }
    }

    fn head_dim(&self) -> i32 {
        self.hidden_dim / self.num_heads
    }

    /// `gpt_fast` works this out rather than being told: two thirds of four times the width,
    /// rounded up to a multiple of 256.
    fn intermediate(&self) -> i32 {
        let hidden = 4 * self.hidden_dim;
        let two_thirds = 2 * hidden / 3;

        two_thirds.div_euclid(256) * 256 + if two_thirds % 256 == 0 { 0 } else { 256 }
    }

    /// Which layers hand a skip connection forward, and which receive one.
    ///
    /// The reference takes `i < depth / 2` and `i > depth / 2`, so with an odd depth the middle
    /// layer does neither, and the receiving layers take them off the end of the list -- the
    /// first to emit is the last to be received by.
    fn skips(&self) -> (Vec<i32>, Vec<i32>) {
        let half = self.depth / 2;

        (
            (0..self.depth).filter(|index| *index < half).collect(),
            (0..self.depth).filter(|index| *index > half).collect(),
        )
    }
}

/// The sinusoid a timestep is turned into, before the projection that follows it.
///
/// Computed on the host because a timestep is one number and this is a table of its cosines: the
/// graph would have to be told the number anyway, and telling it the embedding instead is one
/// input rather than several nodes.
///
/// The scale of a thousand is the reference's, and so is the order -- every cosine, then every
/// sine, rather than the two interleaved.
pub fn timestep_embedding(t: f32, size: i32, device: Device) -> Result<Tensor> {
    let half = (size / 2) as usize;
    let mut values = vec![0.0f32; size as usize];

    for index in 0..half {
        let frequency = (-(10000.0f64).ln() * index as f64 / half as f64).exp();
        let angle = 1000.0 * t as f64 * frequency;

        values[index] = angle.cos() as f32;
        values[half + index] = angle.sin() as f32;
    }

    Ok(Tensor::from_f32(&[1, size], &values)?.to_device(device)?)
}

/// The rotary tables this model turns by: cosine and sine, `(frames, head_dim / 2)` each.
///
/// Half as wide as the head, because a rotation turns pairs and there are half as many pairs as
/// there are numbers.
pub fn rotary_tables(
    frames: i32,
    head_dim: i32,
    base: f64,
    device: Device,
) -> Result<(Tensor, Tensor)> {
    let half = (head_dim / 2) as usize;
    let mut cos = vec![0.0f32; frames as usize * half];
    let mut sin = vec![0.0f32; frames as usize * half];

    for position in 0..frames as usize {
        for index in 0..half {
            let frequency = 1.0 / base.powf(2.0 * index as f64 / head_dim as f64);
            let angle = position as f64 * frequency;

            cos[position * half + index] = angle.cos() as f32;
            sin[position * half + index] = angle.sin() as f32;
        }
    }

    let shape = [frames, half as i32];

    Ok((
        Tensor::from_f32(&shape, &cos)?.to_device(device)?,
        Tensor::from_f32(&shape, &sin)?.to_device(device)?,
    ))
}

/// Turn `x` `(N, heads, T, head_dim)` by the angles in `cos` and `sin`, `(T, head_dim / 2)`.
///
/// Pairs are `(2i, 2i + 1)`, which is the interleaved convention -- see the module note on why
/// this is written out rather than handed to `flint`'s rotary operator.
///
/// The head axis sits before time here so that the tables, which are `(T, head_dim / 2)`, line up
/// against the last two axes and broadcast over the rest without being moved.
#[track_caller]
pub fn rotate(
    g: &Graph,
    x: Value,
    heads: i32,
    head_dim: i32,
    frames: i32,
    cos: Value,
    sin: Value,
) -> Value {
    let pairs = head_dim / 2;
    let shape = [
        Extent::of(x, 0),
        Extent::At(heads),
        Extent::At(frames),
        Extent::At(pairs),
        Extent::At(2),
    ];

    let split = g.view(x, shape);
    let even = g.squeeze(g.slice(split, 4, 0, 1), 4);
    let odd = g.squeeze(g.slice(split, 4, 1, 2), 4);

    let (even, odd) = (g.contiguous(even), g.contiguous(odd));

    let turned_even = g.sub(g.mul(even, cos), g.mul(odd, sin));
    let turned_odd = g.add(g.mul(odd, cos), g.mul(even, sin));

    let stacked = g.cat(g.unsqueeze(turned_even, 4), g.unsqueeze(turned_odd, 4), 4);

    g.view(
        g.contiguous(stacked),
        [
            Extent::of(x, 0),
            Extent::At(heads),
            Extent::At(frames),
            Extent::At(head_dim),
        ],
    )
}

/// `weight * rmsnorm(x) + bias`, where the weight and the bias are projected from `c`.
///
/// The normalization has a weight of its own as well, inside it, which is the `RMSNorm` the
/// reference wraps; what the timestep projects is a second scale and a shift on top of that.
#[track_caller]
fn adaptive_norm(g: &Graph, x: Value, c: Value, dim: i32, eps: f32) -> Value {
    let projected = Linear::graph(&g.subgraph("project_layer"), c, dim, 2 * dim, true);

    let weight = g.slice(projected, -1, 0, dim);
    let bias = g.slice(projected, -1, dim, 2 * dim);

    let normed = g.rms_norm(x, g.subgraph("norm").load("weight", &[dim]), eps);

    g.add(g.mul(normed, weight), bias)
}

/// Attention over the frame axis: one projection to all three of query, key and value, rotary
/// positions, and one back.
#[track_caller]
#[allow(clippy::too_many_arguments)]
fn attention(g: &Graph, x: Value, config: &Config, frames: i32, cos: Value, sin: Value) -> Value {
    let (heads, head_dim) = (config.num_heads, config.head_dim());
    let width = heads * head_dim;

    // One weight for all three, which is why it is three times as wide.
    let qkv = Linear::graph(&g.subgraph("wqkv"), x, config.hidden_dim, 3 * width, false);

    let part = |index: i32| {
        let taken = g.slice(qkv, -1, index * width, (index + 1) * width);
        let heads_first = g.transpose(
            g.view(
                g.contiguous(taken),
                [
                    Extent::of(x, 0),
                    Extent::At(frames),
                    Extent::At(heads),
                    Extent::At(head_dim),
                ],
            ),
            1,
            2,
        );

        g.contiguous(heads_first)
    };

    let query = rotate(g, part(0), heads, head_dim, frames, cos, sin);
    let key = rotate(g, part(1), heads, head_dim, frames, cos, sin);
    let value = part(2);

    // Not causal: this reads a whole utterance at once, and the reference's mask is all true for
    // a batch of one at full length. See the module note.
    let out = g.attention(query, key, value, false);

    let merged = g.view(
        g.contiguous(g.transpose(out, 1, 2)),
        [Extent::of(x, 0), Extent::At(frames), Extent::At(width)],
    );

    Linear::graph(&g.subgraph("wo"), merged, width, config.hidden_dim, false)
}

/// The SwiGLU the transformer feeds forward with.
#[track_caller]
fn feed_forward(g: &Graph, x: Value, config: &Config) -> Value {
    let inner = config.intermediate();

    let gate = Linear::graph(&g.subgraph("w1"), x, config.hidden_dim, inner, false);
    let up = Linear::graph(&g.subgraph("w3"), x, config.hidden_dim, inner, false);
    let joined = g.mul(g.silu(gate), up);

    Linear::graph(&g.subgraph("w2"), joined, inner, config.hidden_dim, false)
}

/// One transformer layer, with the skip connection a U-net layer receives folded in first.
#[track_caller]
#[allow(clippy::too_many_arguments)]
fn block(
    g: &Graph,
    x: Value,
    c: Value,
    config: &Config,
    frames: i32,
    cos: Value,
    sin: Value,
    skip: Option<Value>,
    eps: f32,
) -> Value {
    let x = match skip {
        None => x,
        Some(skip) => Linear::graph(
            &g.subgraph("skip_in_linear"),
            g.cat(x, skip, -1),
            2 * config.hidden_dim,
            config.hidden_dim,
            true,
        ),
    };

    let normed = adaptive_norm(&g.subgraph("attention_norm"), x, c, config.hidden_dim, eps);
    let x = g.add(
        x,
        attention(&g.subgraph("attention"), normed, config, frames, cos, sin),
    );

    let normed = adaptive_norm(&g.subgraph("ffn_norm"), x, c, config.hidden_dim, eps);

    g.add(x, feed_forward(&g.subgraph("feed_forward"), normed, config))
}

/// The WaveNet that reads the transformer's output back over the frame axis.
///
/// Eight layers of a gated convolution, each conditioned on the timestep: one projection turns
/// the timestep into a pair of biases per layer, and every layer adds its own pair before the
/// tanh and the sigmoid that gate it. Every layer's skip output is summed, which is what comes
/// back; the residual is what carries on.
#[track_caller]
fn wavenet(
    g: &Graph,
    x: Value,
    conditioning: Value,
    config: &Config,
    frames: i32,
    dtype: DType,
    device: Device,
) -> Result<Value> {
    let channels = config.wavenet_dim;
    let gated = 2 * channels;
    let padding = (config.wavenet_kernel - 1) / 2;

    // One projection for every layer's pair of biases, sliced apart below.
    let condition = {
        // `SConv1d` wraps a `conv` that wraps a `conv`, so the weight sits two names deep.
        let sub = g.subgraph("cond_layer").subgraph("conv").subgraph("conv");
        let weight = sub.load("weight", &[gated * config.wavenet_layers, channels, 1]);
        let bias = sub.load("bias", &[gated * config.wavenet_layers]);
        conv1d(
            &sub,
            conditioning,
            weight,
            Some(bias),
            1,
            0,
            1,
            1,
            dtype,
            device,
        )?
    };

    let mut running = x;
    let mut total: Option<Value> = None;

    for index in 0..config.wavenet_layers {
        let last = index == config.wavenet_layers - 1;

        let inner = {
            let sub = g
                .subgraph("in_layers")
                .subgraph(&index.to_string())
                .subgraph("conv")
                .subgraph("conv");
            let weight = sub.load("weight", &[gated, channels, config.wavenet_kernel]);
            let bias = sub.load("bias", &[gated]);
            conv1d(
                &sub,
                running,
                weight,
                Some(bias),
                1,
                padding,
                1,
                1,
                dtype,
                device,
            )?
        };

        // This layer's slice of the timestep projection, one frame wide and broadcast over time.
        let offset = index * gated;
        let mine = g.slice(condition, 1, offset, offset + gated);
        let acts = g.add(inner, mine);

        let tanh = g.tanh(g.slice(acts, 1, 0, channels));
        let sigmoid = g.sigmoid(g.slice(acts, 1, channels, gated));
        let gate = g.mul(g.contiguous(tanh), g.contiguous(sigmoid));

        let out_channels = if last { channels } else { gated };
        let res_skip = {
            let sub = g
                .subgraph("res_skip_layers")
                .subgraph(&index.to_string())
                .subgraph("conv")
                .subgraph("conv");
            let weight = sub.load("weight", &[out_channels, channels, 1]);
            let bias = sub.load("bias", &[out_channels]);
            conv1d(&sub, gate, weight, Some(bias), 1, 0, 1, 1, dtype, device)?
        };

        let skip = match last {
            true => res_skip,
            false => {
                let residual = g.contiguous(g.slice(res_skip, 1, 0, channels));
                running = g.add(running, residual);

                g.contiguous(g.slice(res_skip, 1, channels, gated))
            }
        };

        total = Some(match total {
            None => skip,
            Some(sofar) => g.add(sofar, skip),
        });
    }

    let _ = frames;

    Ok(total.expect("a wavenet of at least one layer sums at least one skip"))
}

/// One step of the trajectory: `(N, 80, T)` out, the direction to move the mel in.
///
/// `x` is the mel as it stands, `prompt_x` the reference speaker's own mel padded with zeros,
/// `cond` the semantic tokens `(N, T, content_dim)`, `style` the speaker `(N, style_dim)`, and
/// the two timestep embeddings are [`timestep_embedding`] at this step's time -- two of them
/// because the transformer and the WaveNet each project their own.
/// The transformer trunk: everything up to and including its own final normalization.
///
/// `(N, T, hidden_dim)` out. Public because it is the half worth testing on its own -- the skip
/// connections and the rotary convention both live here, and both are the kind of thing that is
/// wrong while still producing a tensor of the right shape.
#[track_caller]
#[allow(clippy::too_many_arguments)]
pub fn trunk(
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
    dtype: DType,
    device: Device,
) -> Result<Value> {
    Ok(trunk_parts(
        g, x, prompt_x, cond, style, time, config, frames, cos, sin, dtype, device,
    )?
    .0)
}

/// The trunk, and the two things the head after it needs back: the mel it started from, for the
/// long skip, and the projected timestep, for the final modulation.
#[track_caller]
#[allow(clippy::too_many_arguments)]
fn trunk_parts(
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
    dtype: DType,
    device: Device,
) -> Result<(Value, Value, Value)> {
    const EPS: f32 = 1e-5;

    let hidden = config.hidden_dim;

    // The timestep, as the transformer's layers will read it: one frame wide, so that every
    // adaptive normalization broadcasts it over the whole utterance.
    let t1 = {
        let sub = g.subgraph("t_embedder").subgraph("mlp");
        let first = Linear::graph(
            &sub.subgraph("0"),
            time,
            config.frequency_embedding_size,
            hidden,
            true,
        );
        Linear::graph(&sub.subgraph("2"), g.silu(first), hidden, hidden, true)
    };

    let cond = Linear::graph(
        &g.subgraph("cond_projection"),
        cond,
        config.content_dim,
        hidden,
        true,
    );

    // (N, 80, T) -> (N, T, 80) for both mels, which is the axis order everything above works in.
    let mel = g.contiguous(g.transpose(x, 1, 2));
    let prompt = g.contiguous(g.transpose(prompt_x, 1, 2));

    // The style is one vector for the whole utterance, so it is held across every frame.
    let held = {
        let ones = g.constant(
            Tensor::from_f32(&[1, frames], &vec![1.0; frames as usize])?
                .to_device(device)?
                .cast(dtype)?,
        );
        let spread = g.matmul(g.unsqueeze(style, 2), ones);

        g.contiguous(g.transpose(spread, 1, 2))
    };

    let joined = g.cat(g.cat(g.cat(mel, prompt, -1), cond, -1), held, -1);
    let width = 2 * config.in_channels + hidden + config.style_dim;

    let mut running = Linear::graph(
        &g.subgraph("cond_x_merge_linear"),
        joined,
        width,
        hidden,
        true,
    );

    let c = g.unsqueeze(t1, 1);
    let transformer = g.subgraph("transformer");
    let (emit, receive) = config.skips();

    let mut held_skips: Vec<Value> = Vec::new();
    for index in 0..config.depth {
        // The receiving layers take them off the end: the first emitted is the last received.
        let skip = match receive.contains(&index) {
            true => held_skips.pop(),
            false => None,
        };

        running = block(
            &transformer.subgraph("layers").subgraph(&index.to_string()),
            running,
            c,
            config,
            frames,
            cos,
            sin,
            skip,
            EPS,
        );

        if emit.contains(&index) {
            held_skips.push(running);
        }
    }

    let x_res = adaptive_norm(&transformer.subgraph("norm"), running, c, hidden, EPS);

    Ok((x_res, mel, t1))
}

/// One step of the trajectory: `(N, in_channels, T)` out, the direction to move the mel in.
///
/// `x` is the mel as it stands, `prompt_x` the reference speaker's own mel padded with zeros,
/// `cond` the semantic tokens `(N, T, content_dim)`, `style` the speaker `(N, style_dim)`, and
/// the two timestep embeddings are [`timestep_embedding`] at this step's time -- two of them
/// because the transformer and the WaveNet each project their own.
#[track_caller]
#[allow(clippy::too_many_arguments)]
pub fn graph(
    g: &Graph,
    x: Value,
    prompt_x: Value,
    cond: Value,
    style: Value,
    time: Value,
    time2: Value,
    config: &Config,
    frames: i32,
    cos: Value,
    sin: Value,
    dtype: DType,
    device: Device,
) -> Result<Value> {
    let hidden = config.hidden_dim;

    let (x_res, mel, t1) = trunk_parts(
        g, x, prompt_x, cond, style, time, config, frames, cos, sin, dtype, device,
    )?;

    let t2 = {
        let sub = g.subgraph("t_embedder2").subgraph("mlp");
        let first = Linear::graph(
            &sub.subgraph("0"),
            time2,
            config.frequency_embedding_size,
            config.wavenet_dim,
            true,
        );
        Linear::graph(
            &sub.subgraph("2"),
            g.silu(first),
            config.wavenet_dim,
            config.wavenet_dim,
            true,
        )
    };

    // The long skip: the mel this step started from, read again at the far end.
    let x_res = Linear::graph(
        &g.subgraph("skip_linear"),
        g.cat(x_res, mel, -1),
        hidden + config.in_channels,
        hidden,
        true,
    );

    let narrowed = Linear::graph(
        &g.subgraph("conv1"),
        x_res,
        hidden,
        config.wavenet_dim,
        true,
    );

    let waved = wavenet(
        &g.subgraph("wavenet"),
        g.contiguous(g.transpose(narrowed, 1, 2)),
        g.unsqueeze(t2, 2),
        config,
        frames,
        dtype,
        device,
    )?;

    let residual = Linear::graph(
        &g.subgraph("res_projection"),
        x_res,
        hidden,
        config.wavenet_dim,
        true,
    );
    let joined = g.add(g.contiguous(g.transpose(waved, 1, 2)), residual);

    // The final layer's normalization has no weight of its own -- `elementwise_affine=False` --
    // and what modulates it is `1 + scale`, unlike the adaptive normalizations above, which use
    // the projected weight as it comes.
    let out = {
        let sub = g.subgraph("final_layer");
        let modulation = Linear::graph(
            &sub.subgraph("adaLN_modulation").subgraph("1"),
            g.silu(t1),
            hidden,
            2 * config.wavenet_dim,
            true,
        );
        let shift = g.unsqueeze(g.slice(modulation, -1, 0, config.wavenet_dim), 1);
        let scale = g.unsqueeze(
            g.slice(modulation, -1, config.wavenet_dim, 2 * config.wavenet_dim),
            1,
        );

        let normed = g.layer_norm(joined, None, None, 1e-6);
        let modulated = g.add(g.add(g.mul(normed, scale), normed), shift);

        Linear::graph(
            &sub.subgraph("linear"),
            modulated,
            config.wavenet_dim,
            config.wavenet_dim,
            true,
        )
    };

    let sub = g.subgraph("conv2");
    let weight = sub.load("weight", &[config.in_channels, config.wavenet_dim, 1]);
    let bias = sub.load("bias", &[config.in_channels]);

    conv1d(
        &sub,
        g.contiguous(g.transpose(out, 1, 2)),
        weight,
        Some(bias),
        1,
        0,
        1,
        1,
        dtype,
        device,
    )
}
