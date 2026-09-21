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

//! IndexTTS-2.5's GPT: text and a voice in, the semantic tokens S2Mel reads out.
//!
//! The largest of the models here, and the one the rest of the pipeline is arranged around. What
//! it generates is not audio and not mel -- it is the codec's alphabet, one token per two frames
//! of the speech to come, produced one at a time until a stop token.
//!
//! # It is an ordinary GPT-2, fed unusually
//!
//! Twenty-four layers, 1280 wide, twenty heads: a stock `GPT2Model`. Two things are taken out of
//! it. Its token embedding is unused, because what goes in is embeddings this module assembles
//! rather than ids. And its *position* embedding is replaced by zeros -- `null_position_embeddings`
//! in the reference -- because the positions are added earlier, by a learned table applied to the
//! text and to the mel separately. A reader who assumes GPT-2's own `wpe` is in play gets a model
//! that is subtly wrong everywhere.
//!
//! # What the prefix is made of
//!
//! ```text
//! [ zero padding ][ speaker + emotion ][ 0 ][ 0 ][ text tokens ][ start mel ]
//! ```
//!
//! Three conditioning rows -- one carrying the speaker vector from [`crate::campplus`] plus an
//! emotion vector, then two zeros -- and then the text, each token embedded and given both a
//! position and a language. The padding is on the *left*, so that generation always begins at the
//! same offset from the end.
//!
//! [`Gpt`] leaves the padding out, and the diagram keeps it because the reference has it. Padding
//! is what makes sentences of different lengths line up in one batch; this reads one sentence at
//! a time, and a padded row is then only a position for the attention to read that carries
//! nothing. What it buys back is the whole of the difference: nothing downstream has to be told
//! where the real prefix started.
//!
//! # Two shapes that are not what they look like
//!
//! **`Conv1D` is not `Linear`.** HuggingFace's GPT-2 uses its own `Conv1D`, whose weight is
//! `(in, out)` where `nn.Linear` stores `(out, in)`. [`conv1d_linear`] reads it as stored and
//! multiplies without transposing; using [`crate::layers::Linear`] here would load the right
//! numbers in the wrong order and produce plausible nonsense.
//!
//! **The activation is the tanh approximation.** `gelu_new`, not the exact GELU `flint` has. The
//! two differ by about a thousandth, which is small enough to look like noise in one layer and
//! not small enough to survive twenty-four. [`gelu_new`] writes it out.
//!
//! # Saying it, one token at a time
//!
//! [`Gpt`] is this model with weights behind it and the loop on top. Two graphs, compiled once
//! each and run many times: the **prefill**, which reads the whole prefix and keeps what every
//! layer computed, and the **step**, which reads one token against what was kept.
//!
//! What a step keeps is a [`Kept`] per layer -- the keys and the values of every position so far.
//! The step concatenates its own onto them and attends over the lot, and because `flint`'s causal
//! mask is aligned to the bottom right of the score matrix, a single query against a longer
//! history needs no mask of its own and no special case. `causal` stays true and means the same
//! thing in both graphs.
//!
//! Attention is the only part of a step that grows with the history. Everything else -- twenty-
//! four blocks of projections, and a head eight thousand wide -- is one position's worth of work
//! rather than the whole prefix's, which is the difference between a reading that finishes and
//! one that does not.

use std::fmt;
use std::ops::ControlFlow;
use std::rc::Rc;

use crate::error::Error;
use crate::flint::{
    check_parameters, functional as F, Bound, DType, Device, Extent, Graph, Ir, ParamSource,
    Preloaded, RunContext, Tensor, Value,
};
use crate::Result;

/// The widths of IndexTTS-2.5's GPT, as `config.yaml` states them.
#[derive(Clone, Copy, Debug)]
pub struct Config {
    pub model_dim: i32,
    pub layers: i32,
    pub heads: i32,
    /// How many tokens the codec's alphabet has, plus the start and stop tokens.
    pub number_mel_codes: i32,
    pub number_text_tokens: i32,
    pub max_mel_tokens: i32,
    pub max_text_tokens: i32,
    pub start_mel_token: i32,
    pub stop_mel_token: i32,
    pub start_text_token: i32,
    pub stop_text_token: i32,
    /// How many rows the language table has: `len(LANGUAGE_DICT) + 1`, which is 107.
    ///
    /// Not the nine or so a reader expects of a model that advertises five languages. The table
    /// is Whisper's list of 106 -- every language *that* vocabulary can name, most of which this
    /// model was never trained to say -- and one more. This said 9 until the real checkpoint was
    /// read, which is the one kind of mistake a test against synthetic weights cannot make: a
    /// parameter source that answers every shape agrees with whatever it is asked for.
    pub languages: i32,
    /// How many rows the *mel* position table has, which is not [`Config::max_mel_tokens`].
    ///
    /// `max_mel_tokens + 2 + max_conditioning_inputs`, which the reference works out in
    /// `build_hf_gpt_transformer` and never writes into `config.yaml` -- so 1815 + 2 + 1 = 1818,
    /// and a reader who assumes the table is as long as the token limit is three rows short. It
    /// is a field rather than that expression because `max_conditioning_inputs` is a constructor
    /// default rather than anything the config states, and [`check_parameters`] is what says so
    /// if a checkpoint ever disagrees.
    pub mel_positions: i32,
    /// How wide the speaker vector is: 192, which is what [`crate::campplus`] produces.
    pub speaker_dim: i32,
    pub layer_norm_eps: f32,
}

impl Config {
    /// What IndexTTS-2.5's `config.yaml` describes.
    pub fn indextts() -> Config {
        Config {
            model_dim: 1280,
            layers: 24,
            heads: 20,
            number_mel_codes: 8194,
            number_text_tokens: 60509,
            max_mel_tokens: 1815,
            max_text_tokens: 600,
            start_mel_token: 8192,
            stop_mel_token: 8193,
            start_text_token: 0,
            stop_text_token: 1,
            languages: 107,
            mel_positions: 1815 + 2 + 1,
            speaker_dim: 192,
            layer_norm_eps: 1e-5,
        }
    }

    fn head_dim(&self) -> i32 {
        self.model_dim / self.heads
    }

    /// How many rows of conditioning sit in front of the text: one for the speaker and the
    /// emotion together, then two zeros.
    pub const CONDITIONING_ROWS: i32 = 3;
}

/// One of HuggingFace's `Conv1D` layers, which is a linear layer with its weight stored the other
/// way round: `(in, out)` rather than `(out, in)`.
///
/// So there is no transpose here, and that absence is the whole point. See the module note.
#[track_caller]
pub fn conv1d_linear(g: &Graph, x: Value, in_dim: i32, out_dim: i32) -> Value {
    let weight = g.load("weight", &[in_dim, out_dim]);
    let bias = g.load("bias", &[out_dim]);

    g.add(g.matmul(x, weight), bias)
}

/// `gelu_new`: the tanh approximation GPT-2 was trained with, not the exact one.
///
/// `0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))`. The difference from the exact GELU
/// is about a thousandth of the value -- invisible in one layer, and not invisible in
/// twenty-four.
#[track_caller]
pub fn gelu_new(g: &Graph, x: Value, dtype: DType, device: Device) -> Result<Value> {
    let constant = |value: f32| -> Result<Value> {
        Ok(g.constant(
            Tensor::from_f32(&[1], &[value])?
                .to_device(device)?
                .cast(dtype)?,
        ))
    };

    let cubed = g.mul(g.square(x), x);
    let inner = g.mul(
        g.add(x, g.mul(cubed, constant(0.044715)?)),
        constant((2.0f32 / std::f32::consts::PI).sqrt())?,
    );

    let gate = g.add(g.tanh(inner), constant(1.0)?);

    Ok(g.mul(g.mul(x, gate), constant(0.5)?))
}

/// The keys and the values one layer has computed so far: `(N, heads, L, head_dim)` each.
///
/// `L` is every position the layer has seen, not the one it was just handed. A [`Gpt::step`]
/// takes the previous step's and hands back one position longer.
#[derive(Clone, Copy, Debug)]
pub struct Kept {
    pub keys: Value,
    pub values: Value,
}

/// One GPT-2 block: normalize, attend, normalize, widen and narrow.
///
/// `past` is what this layer kept from everything before `x`, or `None` for a pass that starts
/// from nothing. What comes back beside the output is the two of them joined -- which is what the
/// next token reads, and is also exactly what the attention here just ran over.
///
/// The sequence length is read off `x` rather than passed in. It used to be a parameter, and a
/// parameter is a length baked into the compiled graph: it would have meant one compilation per
/// sentence, since no two are the same length, and two more for every step of every reading.
#[track_caller]
fn block(
    g: &Graph,
    x: Value,
    past: Option<Kept>,
    config: &Config,
    dtype: DType,
    device: Device,
) -> Result<(Value, Kept)> {
    let (dim, heads, head_dim) = (config.model_dim, config.heads, config.head_dim());
    let eps = config.layer_norm_eps;

    let normed = {
        let sub = g.subgraph("ln_1");
        g.layer_norm(
            x,
            Some(sub.load("weight", &[dim])),
            Some(sub.load("bias", &[dim])),
            eps,
        )
    };

    // One projection for query, key and value together, which is why it is three times as wide.
    let qkv = conv1d_linear(&g.subgraph("attn").subgraph("c_attn"), normed, dim, 3 * dim);

    let part = |index: i32| {
        let taken = g.contiguous(g.slice(qkv, -1, index * dim, (index + 1) * dim));
        let split = g.view(
            taken,
            [
                Extent::of(x, 0),
                Extent::of(x, 1),
                Extent::At(heads),
                Extent::At(head_dim),
            ],
        );

        g.contiguous(g.transpose(split, 1, 2))
    };

    // Whatever this pass computed, behind whatever the passes before it did. Dimension two is the
    // one positions are counted along once the heads have been moved in front of them.
    let kept = match past {
        None => Kept {
            keys: part(1),
            values: part(2),
        },
        Some(past) => Kept {
            keys: g.cat(past.keys, part(1), 2),
            values: g.cat(past.values, part(2), 2),
        },
    };

    // Causal: a token reads what came before it and nothing after. The mask is aligned to the
    // bottom right, so one query against a long history sees all of it -- which is what a step
    // wants, and is the reason a step needs no second attention written for it.
    let attended = g.attention(part(0), kept.keys, kept.values, true);
    let merged = g.view(
        g.contiguous(g.transpose(attended, 1, 2)),
        [Extent::of(x, 0), Extent::of(x, 1), Extent::At(dim)],
    );

    let x = g.add(
        x,
        conv1d_linear(&g.subgraph("attn").subgraph("c_proj"), merged, dim, dim),
    );

    let normed = {
        let sub = g.subgraph("ln_2");
        g.layer_norm(
            x,
            Some(sub.load("weight", &[dim])),
            Some(sub.load("bias", &[dim])),
            eps,
        )
    };

    let mlp = g.subgraph("mlp");
    let wide = conv1d_linear(&mlp.subgraph("c_fc"), normed, dim, 4 * dim);
    let narrow = conv1d_linear(
        &mlp.subgraph("c_proj"),
        gelu_new(g, wide, dtype, device)?,
        4 * dim,
        dim,
    );

    Ok((g.add(x, narrow), kept))
}

/// The transformer stack over a sequence of embeddings: `(N, L, model_dim)` in and out, and what
/// every layer kept along the way.
///
/// No position embedding is added here. GPT-2's own is replaced by zeros in the reference, and
/// the positions this model uses were added to the text and the mel before they arrived.
///
/// `past` is one [`Kept`] per layer, in layer order, or `None` to start from nothing. What comes
/// back is the same list one position longer, which is what the next call passes back in.
#[track_caller]
pub fn backbone_keeping(
    g: &Graph,
    x: Value,
    past: Option<&[Kept]>,
    config: &Config,
    dtype: DType,
    device: Device,
) -> Result<(Value, Vec<Kept>)> {
    if let Some(past) = past {
        if past.len() != config.layers as usize {
            return Err(Error::model(format!(
                "the stack has {} layers and was handed {} layers of cache",
                config.layers,
                past.len()
            )));
        }
    }

    let gpt = g.subgraph("gpt");
    let mut running = x;
    let mut kept = Vec::with_capacity(config.layers as usize);

    for index in 0..config.layers {
        let (out, one) = block(
            &gpt.subgraph("h").subgraph(&index.to_string()),
            running,
            past.map(|past| past[index as usize]),
            config,
            dtype,
            device,
        )?;

        running = out;
        kept.push(one);
    }

    let last = gpt.subgraph("ln_f");

    Ok((
        g.layer_norm(
            running,
            Some(last.load("weight", &[config.model_dim])),
            Some(last.load("bias", &[config.model_dim])),
            config.layer_norm_eps,
        ),
        kept,
    ))
}

/// The transformer stack with nothing before it and nothing kept after it.
///
/// What a caller wants when it is reading a whole sequence once and has no next token to feed --
/// which is every use of this but [`Gpt`]'s own.
#[track_caller]
pub fn backbone(
    g: &Graph,
    x: Value,
    config: &Config,
    dtype: DType,
    device: Device,
) -> Result<Value> {
    Ok(backbone_keeping(g, x, None, config, dtype, device)?.0)
}

/// The embeddings of one text sequence: the token, where it sits, and what language it is in.
///
/// `text` is `(N, L)` of ids already wrapped in the start and stop tokens, `positions` is
/// `(L)` counting from zero, and `language` is `(N)`.
#[track_caller]
pub fn text_embeddings(
    g: &Graph,
    text: Value,
    positions: Value,
    language: Value,
    config: &Config,
) -> Value {
    let token = g.lookup(
        g.subgraph("text_embedding")
            .load("weight", &[config.number_text_tokens + 1, config.model_dim]),
        text,
    );

    let place = g.lookup(
        g.subgraph("text_pos_embedding")
            .subgraph("emb")
            .load("weight", &[config.max_text_tokens + 2, config.model_dim]),
        positions,
    );

    let tongue = g.lookup(
        g.subgraph("lang_embedding")
            .load("weight", &[config.languages, config.model_dim]),
        language,
    );

    g.add(g.add(token, place), tongue)
}

/// The embeddings of a stretch of mel tokens: the token, and where it sits.
///
/// `mel` is `(N, L)` of codec tokens and `positions` is `(L)` counting from zero. Two terms
/// rather than the text's three -- what language it is in was said in front of the text and is
/// not said again.
///
/// The prefix's last row goes through here too: the start token is a mel token at position zero,
/// so the first token generated is at position one, and a loop that starts its own count at zero
/// is off by one for the whole reading.
#[track_caller]
pub fn mel_embeddings(g: &Graph, mel: Value, positions: Value, config: &Config) -> Value {
    let token = g.lookup(
        g.subgraph("mel_embedding")
            .load("weight", &[config.number_mel_codes, config.model_dim]),
        mel,
    );

    let place = g.lookup(
        g.subgraph("mel_pos_embedding")
            .subgraph("emb")
            .load("weight", &[config.mel_positions, config.model_dim]),
        positions,
    );

    g.add(token, place)
}

/// What the generated tokens are scored by: normalize, then project onto the codec's alphabet.
#[track_caller]
pub fn head(g: &Graph, x: Value, config: &Config) -> Value {
    let norm = g.subgraph("final_norm");
    let normed = g.layer_norm(
        x,
        Some(norm.load("weight", &[config.model_dim])),
        Some(norm.load("bias", &[config.model_dim])),
        config.layer_norm_eps,
    );

    // `mel_head` is an `nn.Linear`, so its weight is stored `(out, in)` and does transpose --
    // unlike everything inside the GPT-2 stack. See the module note.
    let sub = g.subgraph("mel_head");
    let weight = sub.load("weight", &[config.number_mel_codes, config.model_dim]);
    let bias = sub.load("bias", &[config.number_mel_codes]);

    g.add(g.matmul(normed, g.transpose(weight, 0, 1)), bias)
}

/// The speaker vector and the emotion vector, as the three rows that sit in front of the text.
///
/// `speaker` is what [`crate::campplus`] produced, `(N, 192)`, and `emotion` is `(N, model_dim)`.
/// The reference adds them and then pads with two rows of zeros; those two are not learned and
/// carry nothing, which is worth knowing before anyone goes looking for their weights.
#[track_caller]
pub fn conditioning(
    g: &Graph,
    speaker: Value,
    emotion: Value,
    config: &Config,
    speaker_dim: i32,
    dtype: DType,
    device: Device,
) -> Result<Value> {
    let sub = g.subgraph("spk_emb_proj");
    let weight = sub.load("weight", &[config.model_dim, speaker_dim]);
    let bias = sub.load("bias", &[config.model_dim]);

    let projected = g.add(g.matmul(speaker, g.transpose(weight, 0, 1)), bias);

    // (N, model_dim) -> (N, 1, model_dim), plus the emotion, then two rows of nothing.
    let first = g.add(g.unsqueeze(projected, 1), g.unsqueeze(emotion, 1));
    let empty = g.zeros(
        [
            Extent::of(speaker, 0),
            Extent::At(Config::CONDITIONING_ROWS - 1),
            Extent::At(config.model_dim),
        ],
        dtype,
        device,
    );

    Ok(g.cat(first, empty, 1))
}

/// One forward pass over a whole sequence: embeddings in, one score per codec token out.
///
/// `(N, L, model_dim)` in, `(N, L, number_mel_codes)` out -- every position scored at once, which
/// is what a check against the reference wants and what [`Gpt`] deliberately does not do (it
/// scores the last position only, because the others are not going to be sampled).
#[track_caller]
pub fn graph(
    g: &Graph,
    embeddings: Value,
    config: &Config,
    dtype: DType,
    device: Device,
) -> Result<Value> {
    let hidden = backbone(g, embeddings, config, dtype, device)?;

    Ok(head(g, hidden, config))
}

/// What the step graph calls layer `index`'s keys and values going in, and coming out.
///
/// Two families of name for what is the same tensor one step apart: a step is handed `past_k3`
/// and hands back `k3`, and the loop moves one to the other. They are `Vec<String>` on [`Gpt`]
/// rather than made here per step, because a [`RunContext`] borrows the names it is given.
fn past_name(kind: &str, index: i32) -> String {
    format!("past_{kind}{index}")
}

fn kept_name(kind: &str, index: i32) -> String {
    format!("{kind}{index}")
}

/// How the last position is scored, in both graphs.
///
/// The head is a matmul onto eight thousand codec tokens, and only the last row of it is ever
/// sampled. Taking that row *before* the head rather than after is what keeps a prefill over a
/// six-hundred-token sentence from computing -- and throwing away -- six hundred of them.
#[track_caller]
fn score_last(g: &Graph, hidden: Value, config: &Config) -> Value {
    let last = g.contiguous(g.slice(hidden, 1, -1, Bound::End));

    g.squeeze(head(g, last, config), 1)
}

/// How much freedom a reading is given, and when it is made to stop.
///
/// These are the knobs `flint` binds rather than every knob the reference offers: it also has
/// typical sampling, which has no operator here and is not approximated with one that is nearby.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sampling {
    /// Zero takes the likeliest token every time, which is the only setting that repeats.
    pub temperature: f32,
    /// Keep this many of the likeliest tokens; zero or less keeps all of them.
    pub top_k: i32,
    /// Keep the likeliest tokens up to this much probability between them. One keeps all of them.
    pub top_p: f32,
    /// Divide the score of everything already said by this, so a reading does not get stuck. One
    /// leaves the scores alone and skips the pass entirely.
    pub repetition_penalty: f32,
    /// The most tokens to generate, over and above the stop token. Clamped to the model's own
    /// [`Config::max_mel_tokens`], since the position table runs out there.
    pub max_tokens: i32,
}

impl Sampling {
    /// The same reading every time: no temperature, no filtering, no penalty.
    ///
    /// What a test wants, and what a comparison against another implementation has to use --
    /// two samplers agreeing on a distribution is not something a draw can show.
    pub fn greedy(max_tokens: i32) -> Sampling {
        Sampling {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            repetition_penalty: 1.0,
            max_tokens,
        }
    }

    /// What the sampler will accept, said here rather than found out down there.
    ///
    /// The operator behind the draw states its preconditions as `CHECK`s, and a `CHECK` that
    /// fails ends the process rather than returning. A temperature below zero or a `top_p`
    /// outside `(0, 1]` is a caller's mistake and should read like one, so it is caught where
    /// there is still somebody to tell.
    ///
    /// `top_k` is not here because it needs no refusing: anything at or below zero means keep
    /// every token, and anything above the alphabet is trimmed to it.
    fn check(&self) -> Result<()> {
        if !self.temperature.is_finite() || self.temperature < 0.0 {
            return Err(Error::model(format!(
                "a temperature of {} is not one to sample with: it has to be finite and no less \
                 than zero, where zero is greedy",
                self.temperature
            )));
        }

        if !self.top_p.is_finite() || self.top_p <= 0.0 || self.top_p > 1.0 {
            return Err(Error::model(format!(
                "a top_p of {} is outside (0, 1], so there is no set of tokens it describes",
                self.top_p
            )));
        }

        if !self.repetition_penalty.is_finite() || self.repetition_penalty <= 0.0 {
            return Err(Error::model(format!(
                "a repetition penalty of {} is not one to divide by; one leaves the scores alone",
                self.repetition_penalty
            )));
        }

        Ok(())
    }
}

/// IndexTTS-2.5's GPT, with its weights behind it and the loop on top.
///
/// Two graphs compiled at construction and run many times after it -- see the module note for
/// what each one is for. Both read the same weights, and the prefill is the one checked against
/// the package, because it is the one that names every parameter the model has.
pub struct Gpt {
    config: Config,
    /// The whole prefix at once: conditioning, text, start token. Keeps every layer's keys.
    prefill: Ir,
    /// One token against what was kept.
    step: Ir,
    /// The weights each graph keeps across runs, read once at construction. Two of them because
    /// the two graphs are compiled apart, and a `Preloaded` is a table of handles rather than a
    /// second copy of the bytes -- the step's entries are the same tensors the prefill's are.
    prefill_preloaded: Preloaded,
    step_preloaded: Preloaded,
    /// The names the step graph knows its cache by, held so that a [`RunContext`] can borrow
    /// them. Indexed by layer, keys then values.
    past_names: Vec<(String, String)>,
    kept_names: Vec<(String, String)>,
    weights: Rc<dyn ParamSource>,
    float_type: DType,
    device: Device,
}

impl fmt::Debug for Gpt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Gpt")
            .field("layers", &self.config.layers)
            .field("prefill", &self.prefill.len())
            .field("step", &self.step.len())
            .field("device", &self.device)
            .finish_non_exhaustive()
    }
}

impl Gpt {
    /// Read the model out of `weights`, under the name the package stores it by.
    pub fn build(
        config: Config,
        name: &str,
        weights: &Rc<dyn ParamSource>,
        float_type: DType,
        device: Device,
    ) -> Result<Gpt> {
        let past_names: Vec<(String, String)> = (0..config.layers)
            .map(|index| (past_name("k", index), past_name("v", index)))
            .collect();
        let kept_names: Vec<(String, String)> = (0..config.layers)
            .map(|index| (kept_name("k", index), kept_name("v", index)))
            .collect();

        let prefill = Graph::new();
        Self::write_prefill(
            &prefill.subgraph(name),
            &config,
            &kept_names,
            float_type,
            device,
        )?;
        check_parameters(&prefill, weights.as_ref())?;

        let step = Graph::new();
        Self::write_step(
            &step.subgraph(name),
            &config,
            &past_names,
            &kept_names,
            float_type,
            device,
        )?;

        let prefill = Ir::compile(&prefill, weights.residency());
        let prefill_preloaded = prefill.load(weights.as_ref())?;

        let step = Ir::compile(&step, weights.residency());
        let step_preloaded = step.load(weights.as_ref())?;

        Ok(Gpt {
            prefill,
            step,
            prefill_preloaded,
            step_preloaded,
            past_names,
            kept_names,
            weights: Rc::clone(weights),
            config,
            float_type,
            device,
        })
    }

    pub fn config(&self) -> &Config {
        &self.config
    }

    pub fn float_type(&self) -> DType {
        self.float_type
    }

    /// The prefix, scored, with every layer's keys and values kept.
    ///
    /// There is no left padding here, and the diagram in the module note has some. Padding is
    /// what makes several sentences of different lengths line up in one batch; this reads one
    /// sentence at a time, and a padded row would only be a position for the attention to read
    /// that carries nothing.
    fn write_prefill(
        g: &Graph,
        config: &Config,
        kept_names: &[(String, String)],
        dtype: DType,
        device: Device,
    ) -> Result<()> {
        let conditioned = conditioning(
            g,
            g.input("speaker"),
            g.input("emotion"),
            config,
            config.speaker_dim,
            dtype,
            device,
        )?;

        let text = text_embeddings(
            g,
            g.input("text"),
            g.input("text_positions"),
            g.input("language"),
            config,
        );

        // The start token is the last row of the prefix and the first row of the mel sequence,
        // which is why it is embedded by the mel table and not by anything of the prefix's.
        let start = mel_embeddings(g, g.input("start_mel"), g.input("mel_positions"), config);

        let prefix = g.cat(g.cat(conditioned, text, 1), start, 1);
        let (hidden, kept) = backbone_keeping(g, prefix, None, config, dtype, device)?;

        g.output("logits", score_last(g, hidden, config));
        for (one, (keys, values)) in kept.iter().zip(kept_names) {
            g.output(keys, one.keys);
            g.output(values, one.values);
        }

        Ok(())
    }

    /// One token, scored against what the passes before it kept.
    fn write_step(
        g: &Graph,
        config: &Config,
        past_names: &[(String, String)],
        kept_names: &[(String, String)],
        dtype: DType,
        device: Device,
    ) -> Result<()> {
        let x = mel_embeddings(g, g.input("mel"), g.input("mel_positions"), config);

        let past: Vec<Kept> = past_names
            .iter()
            .map(|(keys, values)| Kept {
                keys: g.input(keys),
                values: g.input(values),
            })
            .collect();

        let (hidden, kept) = backbone_keeping(g, x, Some(&past), config, dtype, device)?;

        g.output("logits", score_last(g, hidden, config));
        for (one, (keys, values)) in kept.iter().zip(kept_names) {
            g.output(keys, one.keys);
            g.output(values, one.values);
        }

        Ok(())
    }

    /// Read the whole prefix -- conditioning, text, start token -- and score what comes next.
    ///
    /// `speaker` is `(1, speaker_dim)` from [`crate::campplus`] and `emotion` is `(1, model_dim)`;
    /// the reference has a conformer and a perceiver that produce the second one from a recording,
    /// and neither is written here, so it is passed in. `text` is the ids of the sentence
    /// *without* its start and stop tokens -- those belong to the model and are put on here.
    pub fn prefill(
        &self,
        speaker: &Tensor,
        emotion: &Tensor,
        text: &[i32],
        language: i32,
    ) -> Result<Reading> {
        let config = &self.config;

        if text.len() > config.max_text_tokens as usize {
            return Err(Error::model(format!(
                "the text is {} tokens and the model reads at most {}",
                text.len(),
                config.max_text_tokens
            )));
        }

        let wrapped: Vec<i64> = std::iter::once(config.start_text_token as i64)
            .chain(text.iter().map(|id| i64::from(*id)))
            .chain(std::iter::once(config.stop_text_token as i64))
            .collect();
        let length = wrapped.len() as i32;

        let text = Tensor::from_i64(&[1, length], &wrapped)?.to_device(self.device)?;
        let positions = Tensor::from_i64(&[length], &(0..i64::from(length)).collect::<Vec<i64>>())?
            .to_device(self.device)?;
        let language = Tensor::from_i64(&[1], &[i64::from(language)])?.to_device(self.device)?;
        let start = Tensor::from_i64(&[1, 1], &[i64::from(config.start_mel_token)])?
            .to_device(self.device)?;
        let at_zero = Tensor::from_i64(&[1], &[0])?.to_device(self.device)?;

        let run = RunContext::new(&*self.weights)
            .preloaded(&self.prefill_preloaded)
            .input("speaker", speaker)
            .input("emotion", emotion)
            .input("text", &text)
            .input("text_positions", &positions)
            .input("language", &language)
            .input("start_mel", &start)
            .input("mel_positions", &at_zero);

        self.take(self.prefill.run(&run)?)
    }

    /// Read one more token against what `reading` kept, and score what comes after it.
    ///
    /// `at` is where `token` sits in the mel sequence, counting the start token as zero -- so the
    /// first token generated is at one. It is a parameter and not a running total held here
    /// because a [`Reading`] is what carries the state of a reading, and two of them can be
    /// stepped independently.
    pub fn step(&self, reading: &Reading, token: i32, at: i32) -> Result<Reading> {
        let position = Tensor::from_i64(&[1], &[i64::from(at)])?.to_device(self.device)?;
        let mel = Tensor::from_i64(&[1, 1], &[i64::from(token)])?.to_device(self.device)?;

        let mut run = RunContext::new(&*self.weights)
            .preloaded(&self.step_preloaded)
            .input("mel", &mel)
            .input("mel_positions", &position);
        for (index, (keys, values)) in self.past_names.iter().enumerate() {
            run = run
                .input(keys, &reading.kept[index].0)
                .input(values, &reading.kept[index].1);
        }

        self.take(self.step.run(&run)?)
    }

    /// Say `text` as the voice `speaker` describes, and hand back the codec tokens to say it
    /// with.
    ///
    /// The loop over [`Gpt::prefill`] and [`Gpt::step`], with [`Sampling`] deciding what to draw
    /// and when to stop. A caller that wants a sampler this one does not offer can run those two
    /// itself; this is the common case rather than the only way through.
    ///
    /// `report` is handed the number of tokens said so far and can stop the reading, which is
    /// [`None`] rather than an error: half a sentence is not a reading. The stop token itself is
    /// not in what comes back.
    pub fn generate(
        &self,
        speaker: &Tensor,
        emotion: &Tensor,
        text: &[i32],
        language: i32,
        sampling: &Sampling,
        report: &mut dyn FnMut(i32) -> ControlFlow<()>,
    ) -> Result<Option<Vec<i32>>> {
        sampling.check()?;

        let mut reading = self.prefill(speaker, emotion, text, language)?;
        let mut said: Vec<i32> = Vec::new();

        // The position table runs out at the model's own limit, so a reading is stopped by it
        // whatever it was asked for.
        let limit = sampling.max_tokens.min(self.config.max_mel_tokens);

        while (said.len() as i32) < limit {
            let token = self.draw(&reading.logits, &said, sampling)?;
            if token == self.config.stop_mel_token {
                break;
            }

            said.push(token);
            if report(said.len() as i32).is_break() {
                return Ok(None);
            }

            // The start token was position zero, so what was just said is position `len`.
            reading = self.step(&reading, token, said.len() as i32)?;
        }

        Ok(Some(said))
    }

    /// Pull the scores and the cache out of what a graph returned, by name.
    fn take(&self, outputs: Vec<(String, Tensor)>) -> Result<Reading> {
        let find = |wanted: &str| -> Result<Tensor> {
            outputs
                .iter()
                .find(|(name, _)| name == wanted)
                .map(|(_, tensor)| tensor.clone())
                .ok_or_else(|| Error::model(format!("the GPT produced no {wanted}")))
        };

        let logits = find("logits")?;
        let kept = self
            .kept_names
            .iter()
            .map(|(keys, values)| Ok((find(keys)?, find(values)?)))
            .collect::<Result<Vec<(Tensor, Tensor)>>>()?;

        Ok(Reading { logits, kept })
    }

    /// One token out of one row of scores.
    ///
    /// The scores are cast to float first. Both the penalty and the draw are written for float
    /// logits, and a model running in half would otherwise get a cast inside the sampler on one
    /// device and an error on the other.
    fn draw(&self, logits: &Tensor, said: &[i32], sampling: &Sampling) -> Result<i32> {
        let mut logits = logits.cast(DType::Float)?.contiguous()?;

        if sampling.repetition_penalty != 1.0 && !said.is_empty() {
            let history: Vec<i64> = said.iter().map(|id| i64::from(*id)).collect();
            let history =
                Tensor::from_i64(&[1, said.len() as i32], &history)?.to_device(self.device)?;

            F::repetition_penalty(&mut logits, &history, sampling.repetition_penalty)?;
        }

        let one = |value: f32| -> Result<Tensor> {
            Ok(Tensor::from_f32(&[1], &[value])?.to_device(self.device)?)
        };

        // `top_k` is trimmed to the alphabet rather than passed through: the operator refuses one
        // larger than the vocabulary, and "keep thirty of eight thousand" is a perfectly ordinary
        // thing to ask of a model that has forty in a test. Below zero is keep-everything, which
        // is what zero already says.
        let top_k = sampling.top_k.clamp(0, self.config.number_mel_codes);
        let top_k = Tensor::from_i32(&[1], &[top_k])?.to_device(self.device)?;

        let drawn = F::sample_with_params(
            &logits,
            &one(sampling.temperature)?,
            &top_k,
            &one(sampling.top_p)?,
        )?;

        let drawn = drawn.to_device(Device::Cpu)?.to_vec_i64()?;

        drawn
            .first()
            .map(|id| *id as i32)
            .ok_or_else(|| Error::model("the sampler drew nothing"))
    }
}

/// What one pass of either graph left behind: the scores to sample, and the cache to pass on.
///
/// Held rather than returned loose because the two travel together -- a caller that samples a
/// token from the scores has to hand back the cache that produced them, and one reading's cache
/// in another reading's step is a mistake with no shape error to announce it.
pub struct Reading {
    logits: Tensor,
    kept: Vec<(Tensor, Tensor)>,
}

impl Reading {
    /// One score per token of the codec's alphabet, `(1, number_mel_codes)`, for the position
    /// after everything read so far.
    pub fn scores(&self) -> &Tensor {
        &self.logits
    }

    /// How many positions the cache holds, which is everything this reading has read.
    pub fn len(&self) -> Result<i32> {
        match self.kept.first() {
            None => Ok(0),
            Some((keys, _)) => Ok(keys.shape_at(2)?),
        }
    }

    pub fn is_empty(&self) -> Result<bool> {
        Ok(self.len()? == 0)
    }
}

impl fmt::Debug for Reading {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Reading")
            .field("read", &self.len().unwrap_or(-1))
            .field("layers", &self.kept.len())
            .finish()
    }
}
