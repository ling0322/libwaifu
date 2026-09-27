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

//! Seed-VC v2's AR: narrow tokens in, wide tokens in the reference's manner out.
//!
//! What makes the style path re-say the source with the reference's accent and pacing rather than
//! only in its voice. A twelve-layer decoder, 768 wide, twelve query heads against two for the
//! keys, reads
//!
//! ```text
//! [ sep ][ narrow: reference's, then source's ][ sep ][ reference's wide ] -> wide tokens
//! ```
//!
//! and continues the reference's wide tokens with the source's content. The narrow tokens arrive
//! with their runs collapsed ([`crate::seed_vc::astral::reduce_durations`]), so the durations of
//! what comes out are the AR's own.
//!
//! # Positions start again after the second `sep`
//!
//! The first `sep` and the narrow tokens are positions `0..=L`; the second `sep` is position 0
//! again, and the wide tokens `1, 2, ...` after it. The cache is laid out in the order things
//! were read, so the attention is causal over that order while the rotary positions restart.
//!
//! # The head is not the embedding
//!
//! The model's configuration ties its head to its embedding, and training scores through the
//! embedding; generation calls `forward_generate`, which scores through `output`, a separate
//! weight that differs from the embedding by up to 0.57. What upstream runs is what is run here.
//!
//! # Drawing a token
//!
//! [`Sampler::draw`] is upstream's `logits_to_probs` on the host: a repetition penalty, the end
//! token struck out while fewer than ten have been said, nucleus sampling at `top_p` over the raw
//! scores, and only then the temperature. One faithful oddity: `decode_one_token_ar` hands the
//! penalty `previous_tokens[0]` -- the first token said, not all of them -- so the penalty only
//! ever lands on that one. At the default penalty of 1.0 it is nothing either way.

use std::rc::Rc;

use crate::error::Error;
use crate::flint::{DType, Device, Extent, Graph, Ir, ParamSource, RunContext, Tensor, Value};
use crate::indextts::s2mel::rotate;
use crate::layers::Linear;
use crate::Result;

/// The AR's widths, from `configs/v2/vc_wrapper.yaml` and `BaseModelArgs`' defaults.
#[derive(Clone, Copy, Debug)]
pub struct Config {
    pub dim: i32,
    pub layers: i32,
    pub heads: i32,
    pub kv_heads: i32,
    pub head_dim: i32,
    pub intermediate: i32,
    /// The wide codes and the end token after them.
    pub vocab_size: i32,
    /// How many narrow codes there are.
    pub narrow_codes: i32,
    /// The rotary table's rows, and the cache's length upstream allocates.
    pub max_positions: i32,
    pub eps: f32,
}

impl Config {
    pub fn seed_vc() -> Config {
        Config {
            dim: 768,
            layers: 12,
            heads: 12,
            kv_heads: 2,
            head_dim: 64,
            intermediate: 2304,
            vocab_size: 2049,
            narrow_codes: 32,
            max_positions: 4096,
            eps: 1e-5,
        }
    }

    /// The token that ends a reading: the last row.
    pub fn eos(&self) -> i32 {
        self.vocab_size - 1
    }
}

/// How the next token is drawn.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sampling {
    pub top_p: f32,
    pub temperature: f32,
    pub repetition_penalty: f32,
}

impl Sampling {
    /// `inference_v2.py`'s defaults.
    pub fn seed_vc() -> Sampling {
        Sampling {
            top_p: 0.9,
            temperature: 1.0,
            repetition_penalty: 1.0,
        }
    }
}

/// How many tokens are said before the end token may be.
const MIN_TOKENS: usize = 10;
/// Upstream's loop: one token, then at most four thousand more.
const MAX_TOKENS: usize = 4001;

#[derive(Clone, Copy, Debug)]
struct Kept {
    keys: Value,
    values: Value,
}

fn layer(
    g: &Graph,
    x: Value,
    past: Option<Kept>,
    cos: Value,
    sin: Value,
    config: &Config,
) -> (Value, Kept) {
    let d = config.dim;
    let (heads, kv_heads, head_dim) = (config.heads, config.kv_heads, config.head_dim);
    let kv = kv_heads * head_dim;
    let length = Extent::of(x, 1);

    let normed = g.rms_norm(
        x,
        g.subgraph("attention_norm").load("weight", &[d]),
        config.eps,
    );

    let attention = g.subgraph("attention");
    let qkv = Linear::graph(&attention.subgraph("wqkv"), normed, d, d + 2 * kv, false);
    let split = |from: i32, to: i32, count: i32| {
        let taken = g.contiguous(g.slice(qkv, -1, from, to));
        let shaped = g.view(
            taken,
            [
                Extent::At(1),
                length,
                Extent::At(count),
                Extent::At(head_dim),
            ],
        );
        g.contiguous(g.transpose(shaped, 1, 2))
    };

    // `rotate` wants the length as a number; one graph per prefill length is compiled anyway.
    let turn = |value: Value, count: i32| -> Value {
        let frames = Extent::of(value, 2);
        rotate_any(g, value, count, head_dim, frames, cos, sin)
    };

    let q = turn(split(0, d, heads), heads);
    let k = turn(split(d, d + kv, kv_heads), kv_heads);
    let v = split(d + kv, d + 2 * kv, kv_heads);

    let kept = match past {
        None => Kept { keys: k, values: v },
        Some(past) => Kept {
            keys: g.cat(past.keys, k, 2),
            values: g.cat(past.values, v, 2),
        },
    };

    let out = g.attention(q, kept.keys, kept.values, true);
    let merged = g.view(
        g.contiguous(g.transpose(out, 1, 2)),
        [Extent::At(1), length, Extent::At(heads * head_dim)],
    );
    let h = g.add(
        x,
        Linear::graph(
            &attention.subgraph("wo"),
            merged,
            heads * head_dim,
            d,
            false,
        ),
    );

    let normed = g.rms_norm(h, g.subgraph("ffn_norm").load("weight", &[d]), config.eps);
    let ff = g.subgraph("feed_forward");
    let gate = Linear::graph(&ff.subgraph("w1"), normed, d, config.intermediate, false);
    let up = Linear::graph(&ff.subgraph("w3"), normed, d, config.intermediate, false);
    let down = Linear::graph(
        &ff.subgraph("w2"),
        g.mul(g.silu(gate), up),
        config.intermediate,
        d,
        false,
    );

    (g.add(h, down), kept)
}

/// [`rotate`] for a length only the graph knows: the same interleaved pairs.
fn rotate_any(
    g: &Graph,
    x: Value,
    heads: i32,
    head_dim: i32,
    length: Extent,
    cos: Value,
    sin: Value,
) -> Value {
    match length {
        Extent::At(frames) => rotate(g, x, heads, head_dim, frames, cos, sin),
        _ => {
            let pairs = head_dim / 2;
            let split = g.view(
                x,
                [
                    Extent::At(1),
                    Extent::At(heads),
                    length,
                    Extent::At(pairs),
                    Extent::At(2),
                ],
            );
            let even = g.contiguous(g.squeeze(g.slice(split, 4, 0, 1), 4));
            let odd = g.contiguous(g.squeeze(g.slice(split, 4, 1, 2), 4));
            let turned_even = g.sub(g.mul(even, cos), g.mul(odd, sin));
            let turned_odd = g.add(g.mul(odd, cos), g.mul(even, sin));
            let stacked = g.cat(g.unsqueeze(turned_even, 4), g.unsqueeze(turned_odd, 4), 4);
            g.view(
                g.contiguous(stacked),
                [
                    Extent::At(1),
                    Extent::At(heads),
                    length,
                    Extent::At(head_dim),
                ],
            )
        }
    }
}

/// The stack over `x` `(1, L, D)`, and the scores of the token after the last of it, `(1, V)`.
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
    let layers = g.subgraph("model").subgraph("layers");
    for index in 0..config.layers {
        let (out, one) = layer(
            &layers.subgraph(&index.to_string()),
            running,
            past.map(|past| past[index as usize]),
            cos,
            sin,
            config,
        );
        running = out;
        kept.push(one);
    }

    let model = g.subgraph("model");
    let last = g.slice(running, 1, -1, Extent::End);
    let normed = g.rms_norm(
        last,
        model.subgraph("norm").load("weight", &[config.dim]),
        config.eps,
    );
    let scores = Linear::graph(
        &model.subgraph("output"),
        g.view(normed, [1, config.dim]),
        config.dim,
        config.vocab_size,
        false,
    );

    (scores, kept)
}

fn wide_rows(g: &Graph, ids: Value, config: &Config) -> Value {
    let table = g
        .subgraph("model")
        .subgraph("embeddings")
        .load("weight", &[config.vocab_size, config.dim]);
    g.unsqueeze(g.lookup(table, ids), 0)
}

/// What a pass left behind: the next token's scores, every layer's cache, and where the rotary
/// positions and the cache stand.
pub struct Reading {
    logits: Tensor,
    kept: Vec<(Tensor, Tensor)>,
    /// The rotary position of the last token read.
    position: i32,
    /// How many tokens the cache holds.
    length: i32,
}

impl Reading {
    /// The raw scores of the next token, `vocab_size` of them, on the host.
    pub fn logits(&self) -> Result<Vec<f32>> {
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

/// The AR with its weights behind it.
pub struct Ar {
    config: Config,
    name: String,
    regulator: String,
    step: Ir,
    past_names: Vec<(String, String)>,
    kept_names: Vec<(String, String)>,
    cos: Vec<f32>,
    sin: Vec<f32>,
    weights: Rc<dyn ParamSource>,
    device: Device,
    dtype: DType,
}

impl Ar {
    /// `name` is the AR's namespace in the package and `regulator` the narrow tokens' embedding's.
    pub fn build(
        config: Config,
        name: &str,
        regulator: &str,
        weights: &Rc<dyn ParamSource>,
        dtype: DType,
        device: Device,
    ) -> Result<Ar> {
        let past_names: Vec<(String, String)> = (0..config.layers)
            .map(|index| (format!("past.{index}.k"), format!("past.{index}.v")))
            .collect();
        let kept_names: Vec<(String, String)> = (0..config.layers)
            .map(|index| (format!("kept.{index}.k"), format!("kept.{index}.v")))
            .collect();

        let step = Graph::new();
        {
            let g = step.subgraph(name);
            let x = g.cast(wide_rows(&g, g.input("token"), &config), dtype);
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
                g.cast(g.input("cos"), dtype),
                g.cast(g.input("sin"), dtype),
                &config,
            );
            g.output("logits", scores);
            for (one, (k, v)) in kept.iter().zip(&kept_names) {
                g.output(k, one.keys);
                g.output(v, one.values);
            }
        }

        let half = config.head_dim / 2;
        let table = |which: &str| -> Result<Vec<f32>> {
            Ok(weights
                .load(&format!("{name}.{which}"), &[config.max_positions, half])?
                .to_device(Device::Cpu)?
                .cast(DType::Float)?
                .to_vec_f32()?)
        };

        Ok(Ar {
            config,
            name: name.to_string(),
            regulator: regulator.to_string(),
            step: Ir::compile(&step),
            past_names,
            kept_names,
            cos: table("rotary_cos")?,
            sin: table("rotary_sin")?,
            weights: Rc::clone(weights),
            device,
            dtype,
        })
    }

    pub fn config(&self) -> &Config {
        &self.config
    }

    /// The rotary rows for `positions`, `(len, head_dim / 2)` each, out of the package's table.
    fn rotary(&self, positions: &[i32]) -> Result<(Tensor, Tensor)> {
        let half = (self.config.head_dim / 2) as usize;
        let mut cos = Vec::with_capacity(positions.len() * half);
        let mut sin = Vec::with_capacity(positions.len() * half);
        for position in positions {
            let row = *position as usize * half;
            cos.extend_from_slice(&self.cos[row..row + half]);
            sin.extend_from_slice(&self.sin[row..row + half]);
        }

        let shape = [positions.len() as i32, half as i32];
        Ok((
            Tensor::from_f32(&shape, &cos)?.to_device(self.device)?,
            Tensor::from_f32(&shape, &sin)?.to_device(self.device)?,
        ))
    }

    fn ids(&self, ids: &[i32]) -> Result<Tensor> {
        let values: Vec<i64> = ids.iter().map(|id| i64::from(*id)).collect();
        Ok(Tensor::from_i64(&[ids.len() as i32], &values)?.to_device(self.device)?)
    }

    /// Read `[sep][narrow][sep][prompt]` and score the first new wide token.
    pub fn prefill(&self, narrow: &[i32], prompt: &[i32]) -> Result<Reading> {
        let config = &self.config;
        if narrow.is_empty() || prompt.is_empty() {
            return Err(Error::model(
                "the AR needs narrow tokens to read and a reference's wide tokens to continue",
            ));
        }
        if let Some(code) = narrow
            .iter()
            .find(|c| **c < 0 || **c >= config.narrow_codes)
        {
            return Err(Error::model(format!("narrow code {code} is out of range")));
        }
        if let Some(code) = prompt.iter().find(|c| **c < 0 || **c >= config.eos()) {
            return Err(Error::model(format!("wide code {code} is out of range")));
        }

        let length = (narrow.len() + prompt.len() + 2) as i32;
        let longest = (narrow.len() + 1).max(prompt.len() + 1) as i32;
        if length >= config.max_positions || longest >= config.max_positions {
            return Err(Error::model(format!(
                "{length} tokens is more than the AR's {} positions",
                config.max_positions
            )));
        }

        let mut positions: Vec<i32> = (0..=narrow.len() as i32).collect();
        positions.push(0);
        positions.extend(1..=prompt.len() as i32);
        let (cos, sin) = self.rotary(&positions)?;

        let graph = Graph::new();
        {
            let g = graph.subgraph(&self.name);
            let sep = g.view(g.load("sep_token_emb", &[config.dim]), [1, 1, config.dim]);
            let sep = g.cast(sep, self.dtype);
            let narrow_table = graph
                .subgraph(&self.regulator)
                .subgraph("embedding")
                .load("weight", &[config.narrow_codes, config.dim]);
            let condition = g.cast(
                g.unsqueeze(g.lookup(narrow_table, g.input("narrow")), 0),
                self.dtype,
            );
            let target = g.cast(wide_rows(&g, g.input("prompt"), config), self.dtype);
            let x = g.cat(g.cat(g.cat(sep, condition, 1), sep, 1), target, 1);

            let (scores, kept) = stack(
                &g,
                x,
                None,
                g.cast(g.input("cos"), self.dtype),
                g.cast(g.input("sin"), self.dtype),
                config,
            );
            g.output("logits", scores);
            for (one, (k, v)) in kept.iter().zip(&self.kept_names) {
                g.output(k, one.keys);
                g.output(v, one.values);
            }
        }

        let narrow_ids = self.ids(narrow)?;
        let prompt_ids = self.ids(prompt)?;
        let run = RunContext::new(&*self.weights)
            .input("narrow", &narrow_ids)
            .input("prompt", &prompt_ids)
            .input("cos", &cos)
            .input("sin", &sin);

        let outputs = Ir::compile(&graph).run(&run)?;
        self.take(outputs, prompt.len() as i32, length)
    }

    /// Read one more wide token and score the one after it.
    pub fn step(&self, reading: &Reading, token: i32) -> Result<Reading> {
        let position = reading.position + 1;
        if position >= self.config.max_positions || reading.length >= self.config.max_positions {
            return Err(Error::model("the AR has run out of positions"));
        }

        let (cos, sin) = self.rotary(&[position])?;
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

        self.take(self.step.run(&run)?, position, reading.length + 1)
    }

    fn take(&self, outputs: Vec<(String, Tensor)>, position: i32, length: i32) -> Result<Reading> {
        let find = |wanted: &str| -> Result<Tensor> {
            outputs
                .iter()
                .find(|(name, _)| name == wanted)
                .map(|(_, tensor)| tensor.clone())
                .ok_or_else(|| Error::model(format!("the AR produced no {wanted}")))
        };

        let kept = self
            .kept_names
            .iter()
            .map(|(k, v)| Ok((find(k)?, find(v)?)))
            .collect::<Result<Vec<_>>>()?;

        Ok(Reading {
            logits: find("logits")?,
            kept,
            position,
            length,
        })
    }

    /// Continue `prompt` -- the reference's wide tokens -- with what `narrow` says: the wide tokens
    /// said, without the end token. `None` where `report`, handed the count so far, asked to stop.
    pub fn generate(
        &self,
        narrow: &[i32],
        prompt: &[i32],
        sampling: &Sampling,
        sampler: &mut Sampler,
        report: &mut dyn FnMut(i32) -> std::ops::ControlFlow<()>,
    ) -> Result<Option<Vec<i32>>> {
        let mut reading = self.prefill(narrow, prompt)?;
        let room = (self.config.max_positions - reading.length).max(0) as usize;
        let limit = MAX_TOKENS.min(room);

        let mut said: Vec<i32> = Vec::new();
        while said.len() < limit {
            let mut scores = reading.logits()?;
            let token = sampler.draw(
                &mut scores,
                &said,
                sampling,
                said.len() < MIN_TOKENS,
                self.config.eos() as usize,
            ) as i32;
            if token == self.config.eos() {
                break;
            }

            said.push(token);
            if report(said.len() as i32).is_break() {
                return Ok(None);
            }
            if said.len() == limit {
                break;
            }
            reading = self.step(&reading, token)?;
        }

        Ok(Some(said))
    }
}

/// Upstream's sampling, on the host, with this crate's own random numbers.
pub struct Sampler {
    state: u64,
}

impl Sampler {
    pub fn new(seed: u64) -> Sampler {
        Sampler {
            state: seed ^ 0x9E37_79B9_7F4A_7C15,
        }
    }

    /// splitmix64.
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// A number in `[0, 1)`.
    pub fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// The next token out of `scores`, which are changed in the drawing.
    ///
    /// Upstream's `logits_to_probs`, in its order: the penalty on the first token said, the end
    /// token struck while `suppress_eos`, the nucleus kept by the raw scores' softmax -- every
    /// token whose running total is within `top_p`, and the most likely always -- and the
    /// temperature applied to what is left.
    pub fn draw(
        &mut self,
        scores: &mut [f32],
        said: &[i32],
        sampling: &Sampling,
        suppress_eos: bool,
        eos: usize,
    ) -> usize {
        if let Some(first) = said.first() {
            let score = &mut scores[*first as usize];
            *score = match *score < 0.0 {
                true => *score * sampling.repetition_penalty,
                false => *score / sampling.repetition_penalty,
            };
        }
        if suppress_eos {
            scores[eos] = f32::NEG_INFINITY;
        }

        let mut order: Vec<usize> = (0..scores.len()).collect();
        order.sort_by(|a, b| scores[*b].total_cmp(&scores[*a]));

        let top = f64::from(scores[order[0]]);
        let weights: Vec<f64> = order
            .iter()
            .map(|index| (f64::from(scores[*index]) - top).exp())
            .collect();
        let total: f64 = weights.iter().sum();

        let temperature = f64::from(sampling.temperature.max(1e-5));
        let mut kept: Vec<(usize, f64)> = Vec::new();
        let mut cumulative = 0.0;
        for (rank, (index, weight)) in order.iter().zip(&weights).enumerate() {
            // Upstream removes every token whose running total, itself included, is past `top_p`
            // -- all but the first, which is always kept.
            cumulative += weight / total;
            if rank > 0 && cumulative > f64::from(sampling.top_p) {
                break;
            }
            kept.push((*index, (f64::from(scores[*index]) - top) / temperature));
        }

        // Softmax over what was kept, at the temperature, and one draw from it.
        let best = kept
            .iter()
            .map(|(_, logit)| *logit)
            .fold(f64::NEG_INFINITY, f64::max);
        let probabilities: Vec<(usize, f64)> = kept
            .iter()
            .map(|(index, logit)| (*index, (logit - best).exp()))
            .collect();
        let total: f64 = probabilities.iter().map(|(_, p)| p).sum();

        let mut target = self.uniform() * total;
        for (index, probability) in &probabilities {
            target -= probability;
            if target < 0.0 {
                return *index;
            }
        }
        probabilities
            .last()
            .map(|(index, _)| *index)
            .unwrap_or(order[0])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sampling() -> Sampling {
        Sampling {
            top_p: 0.5,
            temperature: 1.0,
            repetition_penalty: 1.0,
        }
    }

    #[test]
    fn the_nucleus_keeps_only_what_fits_under_top_p() {
        // Softmax of these is about [0.64, 0.24, 0.09, 0.03]: at 0.5 only the first is kept,
        // and it is kept although it alone is already past 0.5.
        let mut sampler = Sampler::new(7);
        for _ in 0..200 {
            let mut scores = vec![3.0, 2.0, 1.0, 0.0];
            assert_eq!(sampler.draw(&mut scores, &[], &sampling(), false, 3), 0);
        }
    }

    #[test]
    fn the_end_token_is_struck_while_suppressed() {
        let mut sampler = Sampler::new(3);
        for _ in 0..200 {
            let mut scores = vec![0.0, 0.0, 10.0];
            let drawn = sampler.draw(
                &mut scores,
                &[],
                &Sampling {
                    top_p: 1.0,
                    ..sampling()
                },
                true,
                2,
            );
            assert_ne!(drawn, 2);
        }
    }

    #[test]
    fn only_the_first_token_said_is_penalized() {
        let mut sampler = Sampler::new(1);
        let mut scores = vec![2.0, -2.0, 4.0, 1.0];
        let penalized = Sampling {
            repetition_penalty: 2.0,
            ..sampling()
        };
        sampler.draw(&mut scores, &[1, 2], &penalized, false, 3);
        assert_eq!(scores, vec![2.0, -4.0, 4.0, 1.0]);
    }

    #[test]
    fn the_same_seed_draws_the_same_tokens() {
        let draw = |seed| {
            let mut sampler = Sampler::new(seed);
            (0..32)
                .map(|_| {
                    let mut scores = vec![0.5, 0.4, 0.3, 0.2, 0.1];
                    sampler.draw(
                        &mut scores,
                        &[],
                        &Sampling {
                            top_p: 1.0,
                            ..sampling()
                        },
                        false,
                        4,
                    )
                })
                .collect::<Vec<_>>()
        };
        assert_eq!(draw(11), draw(11));
        assert_ne!(draw(11), draw(12));
    }
}
