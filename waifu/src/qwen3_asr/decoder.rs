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

//! Qwen3-ASR's decoder: the prompt with the recording in the middle of it, then the words, one
//! token at a time.
//!
//! Qwen3-0.6B: twenty-eight layers, 1024 wide, sixteen query heads against eight for the keys, an
//! RMSNorm on each head's queries and keys before they are rotated, SwiGLU, and a head that is the
//! embedding table read again. A prefill graph reads the prompt and a step graph one more token
//! against what the layers kept, each compiled once.
//!
//! Upstream rotates with interleaved MRoPE -- three position axes whose frequencies take turns --
//! but a sequence of text and audio puts every axis at the same position (`get_rope_index` is a
//! running count, repeated three times), and three equal axes interleaved are one axis. So this is
//! the plain rotary table, pairing `x[i]` with `x[i + half]`.
//!
//! A similar implementation lives in `cosyvoice3::lm` (a Qwen2 with speech tables) and another in
//! `anima::text_encoder` (Qwen3 with no cache); a fix to the shared parts of one likely belongs in
//! the others too. They are copies because they are not the same model.

use std::cell::RefCell;
use std::fmt;
use std::rc::Rc;

use crate::error::Error;
use crate::flint::{
    check_parameters, DType, Device, Extent, Graph, Ir, MaskType, RunContext, Tensor, Value,
    Weights,
};
use crate::layers::Linear;
use crate::Result;

use super::config::TextConfig;

/// The keys and values one layer kept, `(1, kv_heads, L, head_dim)` each.
#[derive(Clone, Copy, Debug)]
struct Kept {
    keys: Value,
    values: Value,
}

/// Rotate `x` `(1, L, H, D)` by pairing its first half against its second, Qwen's way.
fn rotate(g: &Graph, x: Value, cos: Value, sin: Value, half: i32) -> Value {
    let first = g.slice(x, 3, 0, half);
    let second = g.slice(x, 3, half, 2 * half);

    let left = g.sub(g.mul(first, cos), g.mul(second, sin));
    let right = g.add(g.mul(first, sin), g.mul(second, cos));
    g.cat(left, right, 3)
}

/// One decoder layer. `x` is `(1, L, D)`; `cos` and `sin` are `(L, heads, head_dim / 2)`.
fn layer(
    g: &Graph,
    x: Value,
    past: Option<Kept>,
    cos: Value,
    sin: Value,
    config: &TextConfig,
) -> (Value, Kept) {
    let d = config.hidden_size;
    let (heads, kv_heads, head_dim) = (config.heads, config.kv_heads, config.head_dim);
    let length = Extent::of(x, 1);

    let normed = g.rms_norm(
        x,
        g.subgraph("input_layernorm").load(Linear::WEIGHT, &[d]),
        config.norm_eps,
    );

    let attn = g.subgraph("self_attn");
    let project = |name: &str, out: i32| Linear::graph(&attn.subgraph(name), normed, d, out, false);
    let split = |value: Value, count: i32| {
        g.view(
            value,
            [
                Extent::At(1),
                length,
                Extent::At(count),
                Extent::At(head_dim),
            ],
        )
    };
    let norm = |value: Value, name: &str| {
        g.rms_norm(
            value,
            attn.subgraph(name).load(Linear::WEIGHT, &[head_dim]),
            config.norm_eps,
        )
    };

    let q = norm(split(project("q_proj", heads * head_dim), heads), "q_norm");
    let k = norm(
        split(project("k_proj", kv_heads * head_dim), kv_heads),
        "k_norm",
    );
    let v = split(project("v_proj", kv_heads * head_dim), kv_heads);

    let half = head_dim / 2;
    let q = rotate(g, q, cos, sin, half);
    let k = rotate(
        g,
        k,
        g.slice(cos, 1, 0, kv_heads),
        g.slice(sin, 1, 0, kv_heads),
        half,
    );

    let q = g.contiguous(g.transpose(q, 1, 2));
    let k = g.contiguous(g.transpose(k, 1, 2));
    let v = g.contiguous(g.transpose(v, 1, 2));

    let kept = match past {
        None => Kept { keys: k, values: v },
        Some(past) => Kept {
            keys: g.cat(past.keys, k, 2),
            values: g.cat(past.values, v, 2),
        },
    };

    // Causal, aligned to the bottom right: one query against the whole history sees all of it.
    let out = g.attention(q, kept.keys, kept.values, MaskType::Causal);
    let out = g.view(
        g.contiguous(g.transpose(out, 1, 2)),
        [Extent::At(1), length, Extent::At(heads * head_dim)],
    );
    let x = g.add(
        x,
        Linear::graph(&attn.subgraph("o_proj"), out, heads * head_dim, d, false),
    );

    let normed = g.rms_norm(
        x,
        g.subgraph("post_attention_layernorm")
            .load(Linear::WEIGHT, &[d]),
        config.norm_eps,
    );
    let mlp = g.subgraph("mlp");
    let gate = Linear::graph(
        &mlp.subgraph("gate_proj"),
        normed,
        d,
        config.mlp_size,
        false,
    );
    let up = Linear::graph(&mlp.subgraph("up_proj"), normed, d, config.mlp_size, false);
    let down = Linear::graph(
        &mlp.subgraph("down_proj"),
        g.mul(g.silu(gate), up),
        config.mlp_size,
        d,
        false,
    );

    (g.add(x, down), kept)
}

/// The stack over `(1, L, D)` embeddings, and the scores of the token after the last of them,
/// `(1, vocab)`. `g` is the package's namespace: `model.*` and `lm_head`.
fn stack(
    g: &Graph,
    x: Value,
    past: Option<&[Kept]>,
    cos: Value,
    sin: Value,
    config: &TextConfig,
    tied: bool,
) -> (Value, Vec<Kept>) {
    let model = g.subgraph("model");
    let mut running = x;
    let mut kept = Vec::with_capacity(config.layers as usize);
    for index in 0..config.layers {
        let (out, one) = layer(
            &model.subgraph("layers").subgraph(&index.to_string()),
            running,
            past.map(|past| past[index as usize]),
            cos,
            sin,
            config,
        );
        running = out;
        kept.push(one);
    }

    let last = g.slice(running, 1, -1, Extent::End);
    let normed = g.rms_norm(
        last,
        model
            .subgraph("norm")
            .load(Linear::WEIGHT, &[config.hidden_size]),
        config.norm_eps,
    );
    let head = match tied {
        true => model.subgraph("embed_tokens"),
        false => g.subgraph("lm_head"),
    };
    let scores = Linear::graph(
        &head,
        g.view(normed, [1, config.hidden_size]),
        config.hidden_size,
        config.vocab_size,
        false,
    );

    (scores, kept)
}

fn rows(g: &Graph, ids: Value, config: &TextConfig, dtype: DType) -> Value {
    let table = g
        .subgraph("model")
        .subgraph("embed_tokens")
        .load(Linear::WEIGHT, &[config.vocab_size, config.hidden_size]);
    g.unsqueeze(g.cast(g.lookup(table, ids), dtype), 0)
}

/// The cosines and sines of positions `0..positions`, `(positions, heads, head_dim / 2)` each, on
/// the device: built once and sliced per pass, as `cosyvoice3::lm`'s table is. Every head's row is
/// the same numbers, stored per head because a binary operation broadcasts its right operand over
/// leading dimensions only.
struct Rotary {
    cos: Tensor,
    sin: Tensor,
    positions: i32,
}

impl Rotary {
    /// A minute of speech and what is written about it, before the table has to grow.
    const INITIAL_POSITIONS: i32 = 2048;

    fn build(config: &TextConfig, positions: i32, device: Device, dtype: DType) -> Result<Rotary> {
        let half = (config.head_dim / 2) as usize;
        let heads = config.heads as usize;
        let count = positions as usize * heads * half;
        let mut cos = Vec::with_capacity(count);
        let mut sin = Vec::with_capacity(count);
        for position in 0..positions {
            let angles: Vec<(f32, f32)> = (0..half)
                .map(|index| {
                    let exponent = 2.0 * index as f64 / config.head_dim as f64;
                    let angle = position as f64 / config.rope_theta.powf(exponent);
                    (angle.cos() as f32, angle.sin() as f32)
                })
                .collect();
            for _ in 0..heads {
                for (c, s) in &angles {
                    cos.push(*c);
                    sin.push(*s);
                }
            }
        }

        let shape = [positions, config.heads, half as i32];
        Ok(Rotary {
            cos: Tensor::from_f32(&shape, &cos)?
                .to_device(device)?
                .cast(dtype)?,
            sin: Tensor::from_f32(&shape, &sin)?
                .to_device(device)?
                .cast(dtype)?,
            positions,
        })
    }
}

/// What a pass left behind: the scores of the next token, and every layer's cache.
pub struct Reading {
    logits: Tensor,
    kept: Vec<(Tensor, Tensor)>,
    /// How many positions the cache holds, which is also the position of the next token.
    length: i32,
}

impl Reading {
    /// The scores of the next token, `vocab` of them, on the host.
    pub fn scores(&self) -> Result<Vec<f32>> {
        Ok(self
            .logits
            .to_device(Device::Cpu)?
            .cast(DType::Float)?
            .to_vec_f32()?)
    }

    pub fn len(&self) -> i32 {
        self.length
    }

    pub fn is_empty(&self) -> bool {
        self.length == 0
    }
}

/// The decoder with its weights behind it.
pub struct Decoder {
    config: TextConfig,
    prefill: Ir,
    step: Ir,
    past_names: Vec<(String, String)>,
    kept_names: Vec<(String, String)>,
    weights: Rc<Weights>,
    device: Device,
    dtype: DType,
    rotary: RefCell<Rotary>,
}

impl fmt::Debug for Decoder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Decoder")
            .field("layers", &self.config.layers)
            .field("device", &self.device)
            .finish_non_exhaustive()
    }
}

impl Decoder {
    pub fn build(
        config: TextConfig,
        name: &str,
        tied: bool,
        weights: &Rc<Weights>,
        dtype: DType,
        device: Device,
    ) -> Result<Decoder> {
        let past_names: Vec<(String, String)> = (0..config.layers)
            .map(|index| (format!("past.{index}.k"), format!("past.{index}.v")))
            .collect();
        let kept_names: Vec<(String, String)> = (0..config.layers)
            .map(|index| (format!("kept.{index}.k"), format!("kept.{index}.v")))
            .collect();

        // The prompt: text, the recording's rows, text.
        let prefill = Graph::new();
        {
            let g = prefill.subgraph(name);
            let head = rows(&g, g.input("head"), &config, dtype);
            let audio = g.unsqueeze(g.cast(g.input("audio"), dtype), 0);
            let tail = rows(&g, g.input("tail"), &config, dtype);
            let x = g.cat(g.cat(head, audio, 1), tail, 1);
            let (scores, kept) = stack(&g, x, None, g.input("cos"), g.input("sin"), &config, tied);
            g.output("logits", scores);
            for (one, (k, v)) in kept.iter().zip(&kept_names) {
                g.output(k, one.keys);
                g.output(v, one.values);
            }
        }
        check_parameters(&prefill, weights.as_ref())?;

        // One token against what was kept.
        let step = Graph::new();
        {
            let g = step.subgraph(name);
            let x = rows(&g, g.input("token"), &config, dtype);
            let past: Vec<Kept> = past_names
                .iter()
                .map(|(k, v)| Kept {
                    keys: g.input(k),
                    values: g.input(v),
                })
                .collect();
            let (scores, kept) = stack(
                &g,
                x,
                Some(&past),
                g.input("cos"),
                g.input("sin"),
                &config,
                tied,
            );
            g.output("logits", scores);
            for (one, (k, v)) in kept.iter().zip(&kept_names) {
                g.output(k, one.keys);
                g.output(v, one.values);
            }
        }

        Ok(Decoder {
            rotary: RefCell::new(Rotary::build(
                &config,
                Rotary::INITIAL_POSITIONS,
                device,
                dtype,
            )?),
            config,
            prefill: Ir::compile(&prefill),
            step: Ir::compile(&step),
            past_names,
            kept_names,
            weights: Rc::clone(weights),
            device,
            dtype,
        })
    }

    pub fn config(&self) -> &TextConfig {
        &self.config
    }

    /// The rotary rows of positions `from..from + count`: views of the table, which is rebuilt
    /// longer first if these run past its end.
    fn rotary(&self, from: i32, count: i32) -> Result<(Tensor, Tensor)> {
        let end = from + count;
        if end > self.rotary.borrow().positions {
            let positions = end.max(2 * self.rotary.borrow().positions);
            *self.rotary.borrow_mut() =
                Rotary::build(&self.config, positions, self.device, self.dtype)?;
        }

        let table = self.rotary.borrow();
        Ok((
            table.cos.slice(0, from, end)?,
            table.sin.slice(0, from, end)?,
        ))
    }

    fn ids(&self, ids: &[i32]) -> Result<Tensor> {
        if let Some(id) = ids
            .iter()
            .find(|id| **id < 0 || **id >= self.config.vocab_size)
        {
            return Err(Error::model(format!("id {id} is outside the vocabulary")));
        }
        let values: Vec<i64> = ids.iter().map(|id| i64::from(*id)).collect();
        Ok(Tensor::from_i64(&[ids.len() as i32], &values)?.to_device(self.device)?)
    }

    /// Read `head`, the recording's `(N, hidden)` rows, and `tail`, and score the first token.
    pub fn prefill(&self, head: &[i32], audio: &Tensor, tail: &[i32]) -> Result<Reading> {
        if head.is_empty() || tail.is_empty() {
            return Err(Error::model(
                "the decoder was handed a prompt with nothing around the recording",
            ));
        }
        let audio_rows = audio.shape_at(0)?;
        let length = head.len() as i32 + audio_rows + tail.len() as i32;
        let (cos, sin) = self.rotary(0, length)?;
        let head = self.ids(head)?;
        let tail = self.ids(tail)?;

        let run = RunContext::new(self.weights.as_ref())
            .input("head", &head)
            .input("audio", audio)
            .input("tail", &tail)
            .input("cos", &cos)
            .input("sin", &sin);

        self.take(self.prefill.run(&run)?, length)
    }

    /// Read one more token and score the one after it.
    pub fn step(&self, reading: &Reading, token: i32) -> Result<Reading> {
        let (cos, sin) = self.rotary(reading.length, 1)?;
        let ids = self.ids(&[token])?;

        let mut run = RunContext::new(self.weights.as_ref())
            .input("token", &ids)
            .input("cos", &cos)
            .input("sin", &sin);
        for (index, (k, v)) in self.past_names.iter().enumerate() {
            run = run
                .input(k, &reading.kept[index].0)
                .input(v, &reading.kept[index].1);
        }

        self.take(self.step.run(&run)?, reading.length + 1)
    }

    fn take(&self, outputs: Vec<(String, Tensor)>, length: i32) -> Result<Reading> {
        let find = |wanted: &str| -> Result<Tensor> {
            outputs
                .iter()
                .find(|(name, _)| name == wanted)
                .map(|(_, tensor)| tensor.clone())
                .ok_or_else(|| Error::model(format!("the decoder produced no {wanted}")))
        };

        let kept = self
            .kept_names
            .iter()
            .map(|(k, v)| Ok((find(k)?, find(v)?)))
            .collect::<Result<Vec<_>>>()?;

        Ok(Reading {
            logits: find("logits")?,
            kept,
            length,
        })
    }
}

/// The index of the largest score: greedy decoding, which is what upstream's generation config
/// asks for (`do_sample: false`).
pub fn argmax(scores: &[f32]) -> usize {
    scores
        .iter()
        .enumerate()
        .fold((0, f32::NEG_INFINITY), |best, (index, score)| {
            if *score > best.1 {
                (index, *score)
            } else {
                best
            }
        })
        .0
}
