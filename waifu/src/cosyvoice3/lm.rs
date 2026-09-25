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

//! CosyVoice3's language model: text in, the speech tokens the flow reads out.
//!
//! A stock Qwen2-0.5B -- twenty-four layers, 896 wide, fourteen query heads against two for the
//! keys -- read with two extra tables. `speech_embedding` embeds the speech tokens, and also the
//! two markers the sequence is built with; `llm_decoder` scores the next speech token. Qwen's own
//! text head is never used, and is not in the package.
//!
//! # What the sequence is made of
//!
//! ```text
//! [ sos ][ text ids ... ][ task ][ prompt speech tokens ... ] -> speech tokens, one at a time
//! ```
//!
//! `sos` and `task` are rows 6561 and 6563 of `speech_embedding`, not text tokens. The text is the
//! system prompt -- `You are a helpful assistant.<|endofprompt|>` -- then, for a zero-shot reading,
//! the transcript of the recording, and then the sentence; the prompt's speech tokens are there
//! only in that zero-shot case. Nothing of the speaker's embedding goes in: `CosyVoice3LM` builds
//! its input without it, unlike CosyVoice 1.
//!
//! # Drawing a token
//!
//! [`Sampler`] is upstream's `ras_sampling`, on the host: nucleus sampling at `top_p` 0.8 over at
//! most `top_k` 25 tokens, and -- if the token drawn already appears in the last ten said -- that
//! token struck out and one drawn from the whole distribution instead. It is on the host because
//! it is a sort of 6761 scores and a loop, and because its random numbers are then this crate's
//! own and repeat on every device.
//!
//! One faithful oddity: while fewer than `min_len` tokens have been said, upstream masks index
//! 6561 -- `speech_token_size`, which in CosyVoice3 is `sos` -- and not the end token 6562. Every
//! index from 6561 up stops a reading. So the minimum length does not prevent the end token, and
//! it does not here either.

use std::fmt;
use std::rc::Rc;

use crate::error::Error;
use crate::flint::{
    check_parameters, DType, Device, Extent, Graph, Ir, ParamSource, RunContext, Tensor, Value,
};
use crate::layers::Linear;
use crate::Result;

/// Qwen2-0.5B's widths, from `CosyVoice-BlankEN/config.json`, and the speech tables beside it.
#[derive(Clone, Copy, Debug)]
pub struct Config {
    pub hidden_size: i32,
    pub layers: i32,
    pub heads: i32,
    pub kv_heads: i32,
    pub head_dim: i32,
    pub mlp_size: i32,
    pub vocab_size: i32,
    pub rope_theta: f64,
    pub norm_eps: f32,
    /// How many speech tokens there are: 3^8, what S3Tokenizer's FSQ can say.
    pub speech_token_size: i32,
    /// Rows of `speech_embedding` and `llm_decoder`: the tokens and two hundred more, of which
    /// three are named and every one stops a reading.
    pub speech_rows: i32,
}

impl Config {
    pub fn cosyvoice3() -> Config {
        Config {
            hidden_size: 896,
            layers: 24,
            heads: 14,
            kv_heads: 2,
            head_dim: 64,
            mlp_size: 4864,
            vocab_size: 151936,
            rope_theta: 1_000_000.0,
            norm_eps: 1e-6,
            speech_token_size: 6561,
            speech_rows: 6761,
        }
    }

    pub fn sos(&self) -> i32 {
        self.speech_token_size
    }

    pub fn task_id(&self) -> i32 {
        self.speech_token_size + 2
    }

    /// Whether `token` ends a reading, which every index from `speech_token_size` up does.
    pub fn stops(&self, token: i32) -> bool {
        token >= self.speech_token_size
    }
}

/// Upstream's `ras_sampling` settings, and a temperature it does not have.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sampling {
    pub top_p: f32,
    pub top_k: usize,
    /// How many of the last tokens said the drawn one is looked for among.
    pub win_size: usize,
    /// The share of that window it has to fill to be struck out: 0.1 of ten is once.
    pub tau_r: f32,
    /// Divides the scores before the softmax. Upstream has none, which is one; zero is greedy.
    pub temperature: f32,
}

impl Sampling {
    pub fn cosyvoice3() -> Sampling {
        Sampling {
            top_p: 0.8,
            top_k: 25,
            win_size: 10,
            tau_r: 0.1,
            temperature: 1.0,
        }
    }
}

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
    config: &Config,
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
    let project = |name: &str, out: i32, bias: bool| {
        Linear::graph(&attn.subgraph(name), normed, d, out, bias)
    };
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

    let q = split(project("q_proj", heads * head_dim, true), heads);
    let k = split(project("k_proj", kv_heads * head_dim, true), kv_heads);
    let v = split(project("v_proj", kv_heads * head_dim, true), kv_heads);

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
    let out = g.attention(q, kept.keys, kept.values, true);
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

/// The stack over `(1, L, D)` embeddings, and the log probability of the next speech token after
/// the last of them, `(1, speech_rows)`.
fn stack(
    g: &Graph,
    x: Value,
    past: Option<&[Kept]>,
    cos: Value,
    sin: Value,
    config: &Config,
) -> (Value, Vec<Kept>) {
    let mut running = x;
    let mut kept = Vec::with_capacity(config.layers as usize);
    for index in 0..config.layers {
        let (out, one) = layer(
            &g.subgraph("layers").subgraph(&index.to_string()),
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
        g.subgraph("norm")
            .load(Linear::WEIGHT, &[config.hidden_size]),
        config.norm_eps,
    );
    let scores = Linear::graph(
        &g.subgraph("llm_decoder"),
        g.view(normed, [1, config.hidden_size]),
        config.hidden_size,
        config.speech_rows,
        false,
    );

    (scores, kept)
}

fn speech_rows(g: &Graph, ids: Value, config: &Config) -> Value {
    let table = g
        .subgraph("speech_embedding")
        .load(Linear::WEIGHT, &[config.speech_rows, config.hidden_size]);
    g.unsqueeze(g.lookup(table, ids), 0)
}

fn text_rows(g: &Graph, ids: Value, config: &Config) -> Value {
    let table = g
        .subgraph("embed_tokens")
        .load(Linear::WEIGHT, &[config.vocab_size, config.hidden_size]);
    g.unsqueeze(g.lookup(table, ids), 0)
}

/// CosyVoice3's language model with its weights behind it: a prefill graph and a step graph,
/// each compiled once.
pub struct Lm {
    config: Config,
    prefill: Ir,
    step: Ir,
    past_names: Vec<(String, String)>,
    kept_names: Vec<(String, String)>,
    weights: Rc<dyn ParamSource>,
    device: Device,
    dtype: DType,
}

impl fmt::Debug for Lm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Lm")
            .field("layers", &self.config.layers)
            .field("device", &self.device)
            .finish_non_exhaustive()
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
    /// The log probabilities of the next speech token, `speech_rows` of them, on the host.
    pub fn log_probabilities(&self) -> Result<Vec<f32>> {
        let scores = self
            .logits
            .to_device(Device::Cpu)?
            .cast(DType::Float)?
            .to_vec_f32()?;
        Ok(log_softmax(&scores))
    }

    pub fn len(&self) -> i32 {
        self.length
    }

    pub fn is_empty(&self) -> bool {
        self.length == 0
    }
}

impl Lm {
    pub fn build(
        config: Config,
        name: &str,
        weights: &Rc<dyn ParamSource>,
        dtype: DType,
        device: Device,
    ) -> Result<Lm> {
        let past_names: Vec<(String, String)> = (0..config.layers)
            .map(|index| (format!("past.{index}.k"), format!("past.{index}.v")))
            .collect();
        let kept_names: Vec<(String, String)> = (0..config.layers)
            .map(|index| (format!("kept.{index}.k"), format!("kept.{index}.v")))
            .collect();

        let cast = |g: &Graph, input: &str| g.cast(g.input(input), dtype);

        // The whole prefix: `sos`, the text, and `task` followed by any prompt tokens.
        let prefill = Graph::new();
        {
            let g = prefill.subgraph(name);
            let head = speech_rows(&g, g.input("head"), &config);
            let text = text_rows(&g, g.input("text"), &config);
            let tail = speech_rows(&g, g.input("tail"), &config);
            let x = g.cat(g.cat(head, text, 1), tail, 1);
            let (scores, kept) = stack(&g, x, None, cast(&g, "cos"), cast(&g, "sin"), &config);
            g.output("logits", scores);
            for (one, (k, v)) in kept.iter().zip(&kept_names) {
                g.output(k, one.keys);
                g.output(v, one.values);
            }
        }
        check_parameters(&prefill, weights.as_ref())?;

        // One speech token against what was kept.
        let step = Graph::new();
        {
            let g = step.subgraph(name);
            let x = speech_rows(&g, g.input("token"), &config);
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
                cast(&g, "cos"),
                cast(&g, "sin"),
                &config,
            );
            g.output("logits", scores);
            for (one, (k, v)) in kept.iter().zip(&kept_names) {
                g.output(k, one.keys);
                g.output(v, one.values);
            }
        }

        Ok(Lm {
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

    pub fn config(&self) -> &Config {
        &self.config
    }

    /// The rotary table for positions `from..from + count`, `(count, heads, head_dim / 2)`.
    fn rotary(&self, from: i32, count: i32) -> Result<(Tensor, Tensor)> {
        let config = &self.config;
        let half = (config.head_dim / 2) as usize;
        let heads = config.heads as usize;

        let mut cos = Vec::with_capacity(count as usize * heads * half);
        let mut sin = Vec::with_capacity(count as usize * heads * half);
        for position in from..from + count {
            for _ in 0..heads {
                for index in 0..half {
                    let exponent = 2.0 * index as f64 / config.head_dim as f64;
                    let angle = position as f64 / config.rope_theta.powf(exponent);
                    cos.push(angle.cos() as f32);
                    sin.push(angle.sin() as f32);
                }
            }
        }

        let shape = [count, config.heads, half as i32];
        Ok((
            Tensor::from_f32(&shape, &cos)?.to_device(self.device)?,
            Tensor::from_f32(&shape, &sin)?.to_device(self.device)?,
        ))
    }

    fn ids(&self, ids: &[i32]) -> Result<Tensor> {
        let values: Vec<i64> = ids.iter().map(|id| i64::from(*id)).collect();
        Ok(Tensor::from_i64(&[ids.len() as i32], &values)?.to_device(self.device)?)
    }

    /// Read `[sos][text][task][prompt]` and score the first speech token.
    pub fn prefill(&self, text: &[i32], prompt: &[i32]) -> Result<Reading> {
        if text.is_empty() {
            return Err(Error::model("the language model was handed no text"));
        }
        if let Some(id) = text
            .iter()
            .find(|id| **id < 0 || **id >= self.config.vocab_size)
        {
            return Err(Error::model(format!(
                "text id {id} is outside the vocabulary"
            )));
        }

        let length = (text.len() + prompt.len() + 2) as i32;
        let (cos, sin) = self.rotary(0, length)?;
        let head = self.ids(&[self.config.sos()])?;
        let body = self.ids(text)?;
        let mut tail_ids = vec![self.config.task_id()];
        tail_ids.extend_from_slice(prompt);
        let tail = self.ids(&tail_ids)?;

        let run = RunContext::new(&*self.weights)
            .input("head", &head)
            .input("text", &body)
            .input("tail", &tail)
            .input("cos", &cos)
            .input("sin", &sin);

        self.take(self.prefill.run(&run)?, length)
    }

    /// Read one more speech token and score the one after it.
    pub fn step(&self, reading: &Reading, token: i32) -> Result<Reading> {
        let (cos, sin) = self.rotary(reading.length, 1)?;
        let ids = self.ids(&[token])?;

        let mut run = RunContext::new(&*self.weights)
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
                .ok_or_else(|| Error::model(format!("the language model produced no {wanted}")))
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

    /// Say `text`, continuing `prompt`, and hand back the speech tokens -- or `None` where
    /// `report`, handed the count so far, asked to stop. `text_len` is what the length limits are
    /// worked out from: upstream's `text_len - prompt_text_len`.
    pub fn generate(
        &self,
        text: &[i32],
        prompt: &[i32],
        text_len: usize,
        sampling: &Sampling,
        sampler: &mut Sampler,
        report: &mut dyn FnMut(i32) -> std::ops::ControlFlow<()>,
    ) -> Result<Option<Vec<i32>>> {
        let min_len = text_len * 2;
        let max_len = text_len * 20;

        let mut reading = self.prefill(text, prompt)?;
        let mut said: Vec<i32> = Vec::new();

        for index in 0..max_len {
            let mut scores = reading.log_probabilities()?;
            if index < min_len {
                scores[self.config.speech_token_size as usize] = f32::NEG_INFINITY;
            }

            let token = sampler.draw(&scores, &said, sampling) as i32;
            if self.config.stops(token) {
                break;
            }

            said.push(token);
            if report(said.len() as i32).is_break() {
                return Ok(None);
            }
            reading = self.step(&reading, token)?;
        }

        Ok(Some(said))
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }
}

/// `log_softmax` over a row of scores, in double precision.
pub fn log_softmax(scores: &[f32]) -> Vec<f32> {
    let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let total: f64 = scores.iter().map(|s| (*s as f64 - max).exp()).sum();
    let log_total = total.ln() + max;
    scores
        .iter()
        .map(|s| (*s as f64 - log_total) as f32)
        .collect()
}

/// Upstream's `ras_sampling`, with a generator of this crate's own.
///
/// The generator is SplitMix64: small, seedable, and the same sequence on every machine, which is
/// all a reading asks of it. It is not torch's, so the same seed does not draw what upstream does.
#[derive(Clone, Debug)]
pub struct Sampler {
    state: u64,
}

impl Sampler {
    pub fn new(seed: u64) -> Sampler {
        Sampler { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// Uniform on `[0, 1)`.
    pub fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// One index of `weights`, with probability in proportion to each.
    fn categorical(&mut self, weights: &[(usize, f64)]) -> usize {
        let total: f64 = weights.iter().map(|(_, w)| w).sum();
        let mut at = self.uniform() * total;
        for (index, weight) in weights {
            if at < *weight {
                return *index;
            }
            at -= weight;
        }
        weights.last().map(|(index, _)| *index).unwrap_or(0)
    }

    /// The probabilities of `scores` at `temperature`; greedy is left to the caller.
    fn softmax(scores: &[f32], temperature: f32) -> Vec<f64> {
        let t = f64::from(temperature.max(1e-6));
        let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
        let exps: Vec<f64> = scores
            .iter()
            .map(|s| match s.is_finite() {
                true => ((*s as f64 - max) / t).exp(),
                false => 0.0,
            })
            .collect();
        let total: f64 = exps.iter().sum();
        exps.into_iter().map(|e| e / total).collect()
    }

    /// `ras_sampling(scores, said)`: nucleus sampling, and a redraw from everything if the token
    /// drawn is one said recently.
    pub fn draw(&mut self, scores: &[f32], said: &[i32], sampling: &Sampling) -> usize {
        if sampling.temperature <= 0.0 {
            return argmax(scores);
        }

        let token = self.nucleus(scores, sampling);

        let window = &said[said.len().saturating_sub(sampling.win_size)..];
        let repeats = window.iter().filter(|id| **id as usize == token).count();
        if repeats as f32 >= sampling.win_size as f32 * sampling.tau_r {
            let mut scores = scores.to_vec();
            scores[token] = f32::NEG_INFINITY;
            let probabilities = Self::softmax(&scores, sampling.temperature);
            let all: Vec<(usize, f64)> = probabilities.into_iter().enumerate().collect();
            return self.categorical(&all);
        }

        token
    }

    /// `nucleus_sampling`: the likeliest tokens, while their total is under `top_p` and there are
    /// fewer than `top_k` of them, then one of those in proportion.
    fn nucleus(&mut self, scores: &[f32], sampling: &Sampling) -> usize {
        let probabilities = Self::softmax(scores, sampling.temperature);
        let mut order: Vec<usize> = (0..probabilities.len()).collect();
        // Stable and descending, as upstream's `sort(descending=True, stable=True)`.
        order.sort_by(|a, b| {
            probabilities[*b]
                .partial_cmp(&probabilities[*a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut kept = Vec::new();
        let mut total = 0.0;
        for index in order {
            if total < f64::from(sampling.top_p) && kept.len() < sampling.top_k {
                total += probabilities[index];
                kept.push((index, probabilities[index]));
            } else {
                break;
            }
        }

        self.categorical(&kept)
    }
}

fn argmax(scores: &[f32]) -> usize {
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

#[cfg(test)]
mod tests {
    use super::*;

    fn sampling() -> Sampling {
        Sampling::cosyvoice3()
    }

    #[test]
    fn nucleus_keeps_only_the_head_of_the_distribution() {
        // Two tokens hold 0.9 between them; the rest are far below. top_p stops after the second.
        let mut scores = vec![-20.0f32; 100];
        scores[7] = 0.0;
        scores[3] = -0.5;
        let mut sampler = Sampler::new(1);
        for _ in 0..200 {
            let token = sampler.draw(&scores, &[], &sampling());
            assert!(token == 7 || token == 3, "drew {token}");
        }
    }

    #[test]
    fn a_recent_token_is_struck_out_and_redrawn() {
        // Token 5 is certain under nucleus sampling; said once in the window, it is struck out.
        let mut scores = vec![-30.0f32; 10];
        scores[5] = 10.0;
        scores[2] = 0.0;
        let mut sampler = Sampler::new(3);
        for _ in 0..50 {
            assert_eq!(sampler.draw(&scores, &[1, 5], &sampling()), 2);
        }
        // Outside the ten-token window it is not.
        let long: Vec<i32> = std::iter::once(5)
            .chain(std::iter::repeat_n(1, 10))
            .collect();
        assert_eq!(sampler.draw(&scores, &long, &sampling()), 5);
    }

    #[test]
    fn the_same_seed_draws_the_same_tokens() {
        let scores: Vec<f32> = (0..50).map(|i| -(i as f32) * 0.1).collect();
        let draw = |seed| {
            let mut sampler = Sampler::new(seed);
            (0..20)
                .map(|_| sampler.draw(&scores, &[], &sampling()))
                .collect::<Vec<_>>()
        };
        assert_eq!(draw(9), draw(9));
        assert_ne!(draw(9), draw(10));
    }

    #[test]
    fn log_softmax_normalizes() {
        let out = log_softmax(&[1.0, 2.0, 3.0]);
        let total: f64 = out.iter().map(|x| (*x as f64).exp()).sum();
        assert!((total - 1.0).abs() < 1e-6);
    }
}
