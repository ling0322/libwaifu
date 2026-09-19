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

use crate::flint::{DType, Device, Extent, Graph, Tensor, Value};
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
    /// How many languages the language embedding distinguishes, plus one.
    pub languages: i32,
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
            languages: 9,
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

/// One GPT-2 block: normalize, attend, normalize, widen and narrow.
#[track_caller]
fn block(
    g: &Graph,
    x: Value,
    config: &Config,
    length: i32,
    dtype: DType,
    device: Device,
) -> Result<Value> {
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
                Extent::At(length),
                Extent::At(heads),
                Extent::At(head_dim),
            ],
        );

        g.contiguous(g.transpose(split, 1, 2))
    };

    // Causal: a token reads what came before it and nothing after.
    let attended = g.attention(part(0), part(1), part(2), true);
    let merged = g.view(
        g.contiguous(g.transpose(attended, 1, 2)),
        [Extent::of(x, 0), Extent::At(length), Extent::At(dim)],
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

    Ok(g.add(x, narrow))
}

/// The transformer stack over a sequence of embeddings: `(N, L, model_dim)` in and out.
///
/// No position embedding is added here. GPT-2's own is replaced by zeros in the reference, and
/// the positions this model uses were added to the text and the mel before they arrived.
#[track_caller]
pub fn backbone(
    g: &Graph,
    x: Value,
    config: &Config,
    length: i32,
    dtype: DType,
    device: Device,
) -> Result<Value> {
    let gpt = g.subgraph("gpt");
    let mut running = x;

    for index in 0..config.layers {
        running = block(
            &gpt.subgraph("h").subgraph(&index.to_string()),
            running,
            config,
            length,
            dtype,
            device,
        )?;
    }

    let last = gpt.subgraph("ln_f");

    Ok(g.layer_norm(
        running,
        Some(last.load("weight", &[config.model_dim])),
        Some(last.load("bias", &[config.model_dim])),
        config.layer_norm_eps,
    ))
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

/// One forward pass over the whole prefix: embeddings in, one score per codec token out.
///
/// `(N, L, model_dim)` in, `(N, L, number_mel_codes)` out. This is the prefill -- every position
/// at once. Generating from it one token at a time is the next thing; see `docs/indextts_gpt.md`.
#[track_caller]
pub fn graph(
    g: &Graph,
    embeddings: Value,
    config: &Config,
    length: i32,
    dtype: DType,
    device: Device,
) -> Result<Value> {
    let hidden = backbone(g, embeddings, config, length, dtype, device)?;

    Ok(head(g, hidden, config))
}
