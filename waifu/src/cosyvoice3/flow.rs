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

//! CosyVoice3's flow: speech tokens in, the 24 kHz mel the vocoder reads out.
//!
//! `CausalMaskedDiffWithDiT`, run the way `token2wav` runs it for a whole sentence at once. Two
//! graphs:
//!
//! 1. **The condition.** The prompt's tokens and the sentence's, end to end, embedded 80 wide and
//!    through the look-ahead layer -- a convolution four wide that reads three tokens ahead, a
//!    leaky ReLU, a causal one three wide, and the embedding added back -- then every row twice,
//!    since there are two mel frames to a token. That is `mu`. Beside it the speaker, normalized
//!    and projected to 80, and the solver's fixed starting noise.
//! 2. **The estimator**, a DiT twenty-two blocks deep, 1024 wide. Its input is the noisy mel, the
//!    prompt's own mel in front of zeros, `mu` and the speaker, 320 wide together, projected, and
//!    given a causal grouped convolution's worth of position; each block is attention and a
//!    feed-forward modulated by the time, adaLN-Zero style.
//!
//! [`Flow::draw`] runs the second ten times: Euler steps along a cosine schedule, each one a
//! batch of two -- the guided pass and one with no `mu`, no speaker and no prompt -- mixed as
//! `1.7 * guided - 0.7 * unguided`.
//!
//! # The rotary embedding reaches one head
//!
//! `x_transformers`' `apply_rotary_pos_emb` rotates the first `rot_dim` channels of whatever it is
//! handed and leaves the rest. F5-TTS's `AttnProcessor`, which this DiT is, hands it the query and
//! the key *before* they are split into heads, and `rot_dim` is one head's 64. So only the first
//! of sixteen heads is rotated -- by pairs that are adjacent, `(2i, 2i + 1)`, not half against
//! half -- and the other fifteen attend with no position at all but the convolution's. It is what
//! the model was trained with, and what [`rotate_first_head`] does.

use std::fmt;
use std::rc::Rc;

use crate::audio::{pad1d, Padding};
use crate::flint::{
    check_parameters, functional as F, DType, Device, Extent, Graph, Ir, ParamSource, RunContext,
    Tensor, Value,
};
use crate::indextts::gpt::gelu_new;
use crate::layers::Linear;
use crate::{Error, Result};

#[derive(Clone, Copy, Debug)]
pub struct Config {
    pub mel: i32,
    pub vocab: i32,
    pub speaker_dim: i32,
    pub lookahead_channels: i32,
    pub lookahead: i32,
    pub token_mel_ratio: i32,
    pub dim: i32,
    pub depth: i32,
    pub heads: i32,
    pub head_dim: i32,
    pub ff_mult: i32,
    pub position_kernel: i32,
    pub position_groups: i32,
    pub time_dim: i32,
    /// How many frames of fixed noise the package holds: three hundred seconds.
    pub noise_frames: i32,
}

impl Config {
    pub fn cosyvoice3() -> Config {
        Config {
            mel: 80,
            vocab: 6561,
            speaker_dim: 192,
            lookahead_channels: 1024,
            lookahead: 3,
            token_mel_ratio: 2,
            dim: 1024,
            depth: 22,
            heads: 16,
            head_dim: 64,
            ff_mult: 2,
            position_kernel: 31,
            position_groups: 16,
            time_dim: 256,
            noise_frames: 15000,
        }
    }
}

fn constant(g: &Graph, value: f32, dtype: DType, device: Device) -> Result<Value> {
    Ok(g.constant(
        Tensor::from_f32(&[1], &[value])?
            .to_device(device)?
            .cast(dtype)?,
    ))
}

fn leaky_relu(g: &Graph, x: Value) -> Value {
    g.add(g.mul_scalar(g.relu(x), 0.99), g.mul_scalar(x, 0.01))
}

/// `x * tanh(softplus(x))`, as `x * (1 - 2 / ((1 + e^x)^2 + 1))`. There is no logarithm to take,
/// and where `e^x` overflows the fraction is zero and the answer `x`, which is right.
///
/// The division is `rsqrt` squared because a binary operation here broadcasts only its right
/// operand, so a constant cannot be the numerator; the reciprocal of infinity is still zero.
fn mish(g: &Graph, x: Value, one: Value, two: Value) -> Value {
    let grown = g.add(g.square(g.add(g.exp(x), one)), one);
    let fraction = g.mul(g.square(g.rsqrt(grown)), two);
    g.mul(x, g.neg(g.sub(fraction, one)))
}

/// The condition graph: `mu` `(1, 2 T, 80)`, the projected speaker `(1, 80)`, and the noise for
/// `frames` mel frames, `(1, frames, 80)`.
fn write_condition(g: &Graph, config: &Config, dtype: DType, device: Device) -> Result<()> {
    let tokens = g.input("tokens");
    let table = g
        .subgraph("input_embedding")
        .load("weight", &[config.vocab, config.mel]);
    let embedded = g.unsqueeze(g.lookup(table, tokens), 0);

    let lookahead = g.subgraph("pre_lookahead_layer");
    let across = g.contiguous(g.transpose(embedded, 1, 2));
    let x = pad1d(g, across, 0, config.lookahead, Padding::Zero, dtype, device)?;
    let conv1 = lookahead.subgraph("conv1");
    let x = g.conv1d(
        x,
        conv1.load(
            "weight",
            &[config.lookahead_channels, config.mel, config.lookahead + 1],
        ),
        Some(conv1.load("bias", &[config.lookahead_channels])),
        1,
        0,
        1,
        1,
    );
    let x = leaky_relu(g, x);
    let x = pad1d(g, x, 2, 0, Padding::Zero, dtype, device)?;
    let conv2 = lookahead.subgraph("conv2");
    let x = g.conv1d(
        x,
        conv2.load("weight", &[config.mel, config.lookahead_channels, 3]),
        Some(conv2.load("bias", &[config.mel])),
        1,
        0,
        1,
        1,
    );
    let h = g.add(g.contiguous(g.transpose(x, 1, 2)), embedded);

    // Every row twice, in place: `repeat_interleave(2, dim=1)`.
    let paired = g.cat(g.unsqueeze(h, 2), g.unsqueeze(h, 2), 2);
    let mu = g.view(
        paired,
        [
            Extent::At(1),
            Extent::prod(paired, 1, 3),
            Extent::At(config.mel),
        ],
    );
    g.output("mu", mu);

    // `F.normalize`: the embedding over its length, the length floored at 1e-12.
    let raw = g.cast(g.input("speaker"), dtype);
    let length = g.sqrt(g.sum(g.square(raw), -1));
    let floor = constant(g, 1e-12, dtype, device)?;
    let length = g.add(g.relu(g.sub(length, floor)), floor);
    let speaker = Linear::graph(
        &g.subgraph("spk_embed_affine_layer"),
        g.div(raw, length),
        config.speaker_dim,
        config.mel,
        true,
    );
    g.output("speaker", speaker);

    let noise = g.load("rand_noise", &[1, config.mel, config.noise_frames]);
    let noise = g.slice(noise, 2, 0, Extent::of(mu, 1));
    g.output("noise", g.contiguous(g.transpose(noise, 1, 2)));

    Ok(())
}

/// Rotate the first `head_dim` channels of `x` `(B, M, dim)` by adjacent pairs, and leave the
/// rest. `cos` and `sin` are `(M, head_dim / 2, 1)`. See the module note.
fn rotate_first_head(g: &Graph, x: Value, cos: Value, sin: Value, config: &Config) -> Value {
    let length = Extent::of(x, 1);
    let rotated = g.contiguous(g.slice(x, 2, 0, config.head_dim));
    let rest = g.slice(x, 2, config.head_dim, config.dim);

    let pairs = g.view(
        rotated,
        [
            Extent::of(x, 0),
            length,
            Extent::At(config.head_dim / 2),
            Extent::At(2),
        ],
    );
    let even = g.slice(pairs, 3, 0, 1);
    let odd = g.slice(pairs, 3, 1, 2);
    let new_even = g.sub(g.mul(even, cos), g.mul(odd, sin));
    let new_odd = g.add(g.mul(odd, cos), g.mul(even, sin));
    let joined = g.view(
        g.cat(new_even, new_odd, 3),
        [Extent::of(x, 0), length, Extent::At(config.head_dim)],
    );

    g.cat(joined, g.contiguous(rest), 2)
}

/// The layer norm with no weights of its own, then `* (1 + scale) + shift`.
fn modulate(g: &Graph, x: Value, shift: Value, scale: Value, one: Value) -> Value {
    let normed = g.layer_norm(x, None, None, 1e-6);
    g.add(g.mul(normed, g.add(scale, one)), shift)
}

/// One of `count` equal pieces of a `(dim * count)` vector.
fn chunk(g: &Graph, modulation: Value, index: i32, dim: i32) -> Value {
    g.contiguous(g.slice(modulation, 0, index * dim, (index + 1) * dim))
}

#[allow(clippy::too_many_arguments)]
fn block(
    g: &Graph,
    x: Value,
    time: Value,
    cos: Value,
    sin: Value,
    one: Value,
    config: &Config,
    dtype: DType,
    device: Device,
) -> Result<Value> {
    let dim = config.dim;
    let (batch, length) = (Extent::of(x, 0), Extent::of(x, 1));

    let modulation = g.view(
        Linear::graph(
            &g.subgraph("attn_norm").subgraph("linear"),
            g.silu(time),
            dim,
            6 * dim,
            true,
        ),
        [6 * dim],
    );
    let piece = |index| chunk(g, modulation, index, dim);
    let (shift_msa, scale_msa, gate_msa) = (piece(0), piece(1), piece(2));
    let (shift_mlp, scale_mlp, gate_mlp) = (piece(3), piece(4), piece(5));

    let normed = modulate(g, x, shift_msa, scale_msa, one);
    let attn = g.subgraph("attn");
    let q = Linear::graph(&attn.subgraph("to_q"), normed, dim, dim, true);
    let k = Linear::graph(&attn.subgraph("to_k"), normed, dim, dim, true);
    let v = Linear::graph(&attn.subgraph("to_v"), normed, dim, dim, true);
    let q = rotate_first_head(g, q, cos, sin, config);
    let k = rotate_first_head(g, k, cos, sin, config);

    let heads = |value: Value| {
        let split = g.view(
            value,
            [
                batch,
                length,
                Extent::At(config.heads),
                Extent::At(config.head_dim),
            ],
        );
        g.contiguous(g.transpose(split, 1, 2))
    };
    let attended = g.attention(heads(q), heads(k), heads(v), false);
    let attended = g.view(
        g.contiguous(g.transpose(attended, 1, 2)),
        [batch, length, Extent::At(dim)],
    );
    let attended = Linear::graph(
        &attn.subgraph("to_out").subgraph("0"),
        attended,
        dim,
        dim,
        true,
    );
    let x = g.add(x, g.mul(attended, gate_msa));

    let normed = modulate(g, x, shift_mlp, scale_mlp, one);
    let ff = g.subgraph("ff").subgraph("ff");
    let inner = dim * config.ff_mult;
    let wide = Linear::graph(&ff.subgraph("0").subgraph("0"), normed, dim, inner, true);
    let wide = gelu_new(g, wide, dtype, device)?;
    let narrow = Linear::graph(&ff.subgraph("2"), wide, inner, dim, true);

    Ok(g.add(x, g.mul(narrow, gate_mlp)))
}

/// The estimator: `x` `(B, M, 80)`, `context` `(B, M, 240)` -- the prompt's mel, `mu` and the
/// speaker, in that order -- and the time's sinusoid `(1, 256)`, in; the velocity `(B, M, 80)` out.
fn write_estimator(g: &Graph, config: &Config, dtype: DType, device: Device) -> Result<()> {
    let dim = config.dim;
    let one = constant(g, 1.0, dtype, device)?;
    let two = constant(g, 2.0, dtype, device)?;

    let time = {
        let mlp = g.subgraph("time_embed").subgraph("time_mlp");
        let hidden = Linear::graph(
            &mlp.subgraph("0"),
            g.cast(g.input("time"), dtype),
            config.time_dim,
            dim,
            true,
        );
        Linear::graph(&mlp.subgraph("2"), g.silu(hidden), dim, dim, true)
    };

    let x = g.cast(g.input("x"), dtype);
    let context = g.cast(g.input("context"), dtype);
    let cos = g.cast(g.input("cos"), dtype);
    let sin = g.cast(g.input("sin"), dtype);

    let embed = g.subgraph("input_embed");
    let x = Linear::graph(
        &embed.subgraph("proj"),
        g.cat(x, context, 2),
        4 * config.mel,
        dim,
        true,
    );

    // The causal convolutional position: two grouped convolutions, each padded on the left only.
    let position = {
        let sub = embed.subgraph("conv_pos_embed");
        let mut running = x;
        for name in ["conv1", "conv2"] {
            let conv = sub.subgraph(name).subgraph("0");
            let across = g.contiguous(g.transpose(running, 1, 2));
            let padded = pad1d(
                g,
                across,
                config.position_kernel - 1,
                0,
                Padding::Zero,
                dtype,
                device,
            )?;
            let convolved = g.conv1d(
                padded,
                conv.load(
                    "weight",
                    &[dim, dim / config.position_groups, config.position_kernel],
                ),
                Some(conv.load("bias", &[dim])),
                1,
                0,
                1,
                config.position_groups,
            );
            running = mish(g, g.contiguous(g.transpose(convolved, 1, 2)), one, two);
        }
        running
    };
    let mut x = g.add(position, x);

    for index in 0..config.depth {
        x = block(
            &g.subgraph("transformer_blocks")
                .subgraph(&index.to_string()),
            x,
            time,
            cos,
            sin,
            one,
            config,
            dtype,
            device,
        )?;
    }

    let modulation = g.view(
        Linear::graph(
            &g.subgraph("norm_out").subgraph("linear"),
            g.silu(time),
            dim,
            2 * dim,
            true,
        ),
        [2 * dim],
    );
    // `scale, shift = chunk(2)`: the other way round from the blocks'.
    let scale = chunk(g, modulation, 0, dim);
    let shift = chunk(g, modulation, 1, dim);
    let x = modulate(g, x, shift, scale, one);

    let velocity = Linear::graph(&g.subgraph("proj_out"), x, dim, config.mel, true);
    g.output("velocity", g.cast(velocity, DType::Float));

    Ok(())
}

/// `SinusPositionEmbedding(256)` of `t`, scaled by a thousand, in float32 as upstream builds it.
pub fn time_sinusoid(t: f32, width: i32) -> Vec<f32> {
    let half = (width / 2) as usize;
    let step = (10000f32).ln() / (half as f32 - 1.0);
    let angles: Vec<f32> = (0..half)
        .map(|i| 1000.0 * t * (i as f32 * -step).exp())
        .collect();
    angles
        .iter()
        .map(|a| a.sin())
        .chain(angles.iter().map(|a| a.cos()))
        .collect()
}

/// The times the estimator is evaluated at over `steps` Euler steps: `t` as [`Flow::draw`] advances
/// it, `t += dt` with `dt` the gap to the next point of the schedule, in float32 -- which is not
/// quite the schedule's own points, and has to be the same bits the loop reaches.
pub fn step_times(steps: usize) -> Vec<f32> {
    let span = schedule(steps);
    let mut times = Vec::with_capacity(steps);
    let mut t = span[0];
    let mut dt = span[1] - span[0];
    for step in 1..span.len() {
        times.push(t);
        t += dt;
        if step < span.len() - 1 {
            dt = span[step + 1] - t;
        }
    }
    times
}

/// The rotary cosines and sines of positions `0..frames`, `(frames, head_dim / 2, 1)` each, on
/// `device`: `position * 10000^(-2i / head_dim)` in float32, as `x_transformers` works it out.
fn rotary_table(config: &Config, frames: i32, device: Device) -> Result<(Tensor, Tensor)> {
    let half = config.head_dim / 2;
    let mut cos = Vec::with_capacity((frames * half) as usize);
    let mut sin = Vec::with_capacity((frames * half) as usize);
    for position in 0..frames {
        for index in 0..half {
            let frequency = 1.0 / 10000f32.powf(2.0 * index as f32 / config.head_dim as f32);
            let angle = position as f32 * frequency;
            cos.push(angle.cos());
            sin.push(angle.sin());
        }
    }
    let shape = [frames, half, 1];
    Ok((
        Tensor::from_f32(&shape, &cos)?.to_device(device)?,
        Tensor::from_f32(&shape, &sin)?.to_device(device)?,
    ))
}

/// `1 - cos(pi / 2 * i / steps)` for `i` in `0..=steps`, in float32.
pub fn schedule(steps: usize) -> Vec<f32> {
    (0..=steps)
        .map(|i| {
            let t = i as f32 / steps as f32;
            1.0 - (t * 0.5 * std::f32::consts::PI).cos()
        })
        .collect()
}

/// What the flow is given for a whole sentence, on the device: `mu`, the projected speaker, and the
/// noise it starts from.
pub struct Condition {
    /// `(1, M, 80)`, frame-major.
    mu: Tensor,
    /// `(1, 80)`.
    speaker: Tensor,
    /// `(1, M, 80)`, the solver's starting point.
    noise: Tensor,
    frames: i32,
}

impl Condition {
    /// `mu`, `(1, frames, 80)` frame-major.
    pub fn mu(&self) -> &Tensor {
        &self.mu
    }

    /// How many mel frames the prompt and the sentence make together.
    pub fn frames(&self) -> i32 {
        self.frames
    }
}

/// The flow with its weights behind it.
pub struct Flow {
    config: Config,
    condition: Ir,
    estimator: Ir,
    weights: Rc<dyn ParamSource>,
    device: Device,
    /// The rotary cosines and sines of every frame the flow can draw, `(noise_frames, head_dim / 2,
    /// 1)` each, on the device: 3.8 MB, worked out once and sliced to a sentence's length.
    rotary: (Tensor, Tensor),
}

impl fmt::Debug for Flow {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Flow")
            .field("depth", &self.config.depth)
            .field("device", &self.device)
            .finish_non_exhaustive()
    }
}

impl Flow {
    pub fn build(
        config: Config,
        name: &str,
        weights: &Rc<dyn ParamSource>,
        dtype: DType,
        device: Device,
    ) -> Result<Flow> {
        let condition = Graph::new();
        write_condition(&condition.subgraph(name), &config, DType::Float, device)?;
        check_parameters(&condition, weights.as_ref())?;

        let estimator = Graph::new();
        write_estimator(
            &estimator
                .subgraph(name)
                .subgraph("decoder")
                .subgraph("estimator"),
            &config,
            dtype,
            device,
        )?;
        check_parameters(&estimator, weights.as_ref())?;

        Ok(Flow {
            rotary: rotary_table(&config, config.noise_frames, device)?,
            config,
            condition: Ir::compile(&condition),
            estimator: Ir::compile(&estimator),
            weights: Rc::clone(weights),
            device,
        })
    }

    pub fn config(&self) -> &Config {
        &self.config
    }

    fn upload(&self, shape: &[i32], values: &[f32]) -> Result<Tensor> {
        Ok(Tensor::from_f32(shape, values)?.to_device(self.device)?)
    }

    /// `mu`, the projected speaker and the noise, for the prompt's tokens followed by `tokens`.
    /// `speaker` is CAMPPlus's `(1, 192)` as it came out, on the device; it is normalized in the
    /// graph, as upstream normalizes it.
    pub fn condition(&self, prompt: &[i32], tokens: &[i32], speaker: &Tensor) -> Result<Condition> {
        let all: Vec<i64> = prompt.iter().chain(tokens).map(|t| i64::from(*t)).collect();
        if let Some(token) = all
            .iter()
            .find(|t| **t < 0 || **t >= i64::from(self.config.vocab))
        {
            return Err(Error::model(format!(
                "speech token {token} is outside the flow's table"
            )));
        }
        let frames = all.len() as i32 * self.config.token_mel_ratio;
        if frames > self.config.noise_frames {
            return Err(Error::model(format!(
                "{frames} mel frames is more than the {} the flow has noise for -- five minutes",
                self.config.noise_frames
            )));
        }

        let ids = Tensor::from_i64(&[all.len() as i32], &all)?.to_device(self.device)?;
        let run = RunContext::new(&*self.weights)
            .input("tokens", &ids)
            .input("speaker", speaker);
        let outputs = self.condition.run(&run)?;
        let find = |wanted: &str| -> Result<Tensor> {
            outputs
                .iter()
                .find(|(name, _)| name == wanted)
                .map(|(_, tensor)| tensor.clone())
                .ok_or_else(|| Error::model(format!("the flow produced no {wanted}")))
        };

        Ok(Condition {
            mu: find("mu")?,
            speaker: find("speaker")?,
            noise: find("noise")?,
            frames,
        })
    }

    /// The rotary rows of the first `frames` positions, `(frames, head_dim / 2, 1)`: views of the
    /// table built when the flow was read.
    fn rotary(&self, frames: i32) -> Result<(Tensor, Tensor)> {
        if frames > self.config.noise_frames {
            return Err(Error::model(format!(
                "{frames} mel frames is more than the {} the flow has positions for",
                self.config.noise_frames
            )));
        }
        Ok((
            self.rotary.0.slice(0, 0, frames)?,
            self.rotary.1.slice(0, 0, frames)?,
        ))
    }

    /// The guided and unguided context of `condition`, `(2, M, 240)`: the prompt's mel `(1, P, 80)`
    /// in front of zeros, `mu`, and the speaker on every row; then all zeros, which is what the
    /// unguided half of classifier-free guidance is handed.
    ///
    /// Put together from device operations rather than in the condition graph, because a graph's
    /// lengths are the lengths of its values and `M - P` -- how many zeros follow the prompt -- is
    /// a difference of two, which an extent cannot say.
    pub fn context_tensor(&self, condition: &Condition, prompt_mel: &Tensor) -> Result<Tensor> {
        let mel = self.config.mel;
        let frames = condition.frames;
        let prompt_frames = prompt_mel.shape_at(1)?;
        if prompt_frames > frames {
            return Err(Error::model(
                "the prompt is longer than the whole of what is drawn",
            ));
        }

        let zeros =
            |length: i32, width: i32| Tensor::zeros(&[1, length, width], DType::Float, self.device);
        let cond = match prompt_frames < frames {
            true => F::cat(prompt_mel, &zeros(frames - prompt_frames, mel)?, 1)?,
            false => prompt_mel.clone(),
        };
        let speaker = F::add(&zeros(frames, mel)?, &condition.speaker.view(&[mel])?)?;

        let guided = F::cat(&F::cat(&cond, &condition.mu, 2)?, &speaker, 2)?;
        Ok(F::cat(&guided, &zeros(frames, 3 * mel)?, 0)?)
    }

    /// The rotary table of `frames` positions on the device, for [`Flow::velocity`].
    pub fn rotary_tables(&self, frames: i32) -> Result<(Tensor, Tensor)> {
        self.rotary(frames)
    }

    /// The estimator once, on both halves of the guided batch: `x` `(2, M, 80)` frame-major, the
    /// context from [`Flow::context_tensor`], the time as [`time_sinusoid`] on the device `(1, 256)`,
    /// and the rotary rows from [`Flow::rotary_tables`]. The velocity comes back `(2, M, 80)`, on
    /// the device.
    pub fn velocity(
        &self,
        x: &Tensor,
        context: &Tensor,
        time: &Tensor,
        rotary: &(Tensor, Tensor),
    ) -> Result<Tensor> {
        let run = RunContext::new(&*self.weights)
            .input("x", x)
            .input("context", context)
            .input("time", time)
            .input("cos", &rotary.0)
            .input("sin", &rotary.1);
        Ok(self.estimator.run(&run)?.remove(0).1)
    }

    /// Ten Euler steps from the fixed noise to the mel of the sentence: `(1, frames, 80)`
    /// frame-major on the device, the prompt's frames already cut off the front. `prompt_mel` is
    /// `(1, P, 80)` on the device.
    ///
    /// Every step stays on the device: the two halves of the guided batch are the same `x`, the
    /// guidance mixes the estimator's two answers, and the step adds `dt` of that to `x`.
    pub fn draw(
        &self,
        condition: &Condition,
        prompt_mel: &Tensor,
        steps: usize,
        cfg_rate: f32,
    ) -> Result<Tensor> {
        let frames = condition.frames;
        let prompt_frames = prompt_mel.shape_at(1)?;
        let context = self.context_tensor(condition, prompt_mel)?;
        let rotary = self.rotary(frames)?;
        let span = schedule(steps);
        // Every step's time as the estimator reads it, up front: once a sentence rather than once a
        // step, and not kept past it -- ten small vectors are nothing beside ten passes of the DiT.
        let times = step_times(steps)
            .into_iter()
            .map(|t| {
                self.upload(
                    &[1, self.config.time_dim],
                    &time_sinusoid(t, self.config.time_dim),
                )
            })
            .collect::<Result<Vec<Tensor>>>()?;

        let mut x = condition.noise.clone();
        let mut t = span[0];
        let mut dt = span[1] - span[0];
        for step in 1..span.len() {
            let doubled = F::cat(&x, &x, 0)?;
            let velocity = self.velocity(&doubled, &context, &times[step - 1], &rotary)?;
            let guided = velocity.slice(0, 0, 1)?;
            let unguided = velocity.slice(0, 1, 2)?;
            let mixed = F::sub(
                &F::mul_scalar(&guided, 1.0 + cfg_rate)?,
                &F::mul_scalar(&unguided, cfg_rate)?,
            )?;
            x = F::add(&x, &F::mul_scalar(&mixed, dt)?)?;
            t += dt;
            if step < span.len() - 1 {
                dt = span[step + 1] - t;
            }
        }

        Ok(x.slice(1, prompt_frames, frames)?.contiguous()?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_schedule_runs_from_zero_to_one_slowly_at_first() {
        let span = schedule(10);
        assert_eq!(span.len(), 11);
        assert!(span[0].abs() < 1e-7 && (span[10] - 1.0).abs() < 1e-6);
        assert!(span[1] - span[0] < span[10] - span[9]);
    }

    #[test]
    fn the_sinusoid_is_sines_then_cosines() {
        let at_zero = time_sinusoid(0.0, 256);
        assert!(at_zero[..128].iter().all(|x| *x == 0.0));
        assert!(at_zero[128..].iter().all(|x| *x == 1.0));
    }
}
