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

//! The emotion vector IndexTTS-2.5 conditions on when it is not handed one.
//!
//! [`crate::indextts::gpt`]'s prefix is a speaker row, an emotion row and two empty ones, and
//! `Gpt::prefill` takes that emotion row ready-made -- which is the path `inference_speech` takes
//! when a caller supplies one. This is where it comes from otherwise: the same w2v-bert features
//! [`crate::indextts::semantic_codec`] reads, run through a conformer, boiled down to a single vector by a
//! perceiver, and widened twice into the GPT's 1280.
//!
//! ```text
//! let config = Config::indextts();
//! let positions = emotion::positional_encoding(
//!     Config::subsampled(frames), config.encoder_dim, device,
//! )?;
//! let emotion = emotion::graph(
//!     &g, features, &config, frames, g.input("positions"), dtype, device,
//! )?;
//! ```
//!
//! # Four stages, under the checkpoint's own names
//!
//! `emo_conditioning_encoder` is a WeNet conformer: one convolution that halves the length, then
//! four blocks of attention, convolution and feed forward. `emo_perceiver_encoder` is
//! lucidrains' `PerceiverResampler` as NaturalSpeech 2 uses it, with **one** latent rather than
//! the thirty-two the speaker path asks for -- which is why a vector comes out of it and not a
//! sequence. `emovec_layer` widens that 1024 to 1280, and `emo_layer` maps 1280 to 1280.
//!
//! 153 tensors between them, and every name here is the checkpoint's, unrenamed and unfolded, for
//! the reason the rest of this model gives: a tensor that turns out to be wrong can be compared
//! against the one it came from.
//!
//! # The speaker path is not this, and the release says so
//!
//! `UnifiedVoice.__init__` also builds a `conditioning_encoder` and a `perceiver_encoder` -- near
//! twins of these two, differing mostly in that the perceiver there resamples to thirty-two
//! latents rather than one. They are not implemented here, and the checkpoint is what settles it:
//! **the 2.5 release ships no weights for either of them.** They would turn up in
//! `load_checkpoint`'s missing keys and sit at their random initialization, which does no harm
//! because 2.5 never calls them. It builds its speaker row with `spk_cond_mode` `"campplus"`
//! instead -- [`crate::indextts::campplus`] produces 192 numbers and `spk_emb_proj` widens them, which is
//! what [`crate::indextts::gpt::conditioning`] already does.
//!
//! # Three things upstream does that are easy to get wrong
//!
//! **The relative attention has no shift.** The score is Transformer-XL's -- `(q + u) . k` plus
//! `(q + v) . p`, where `p` is the sinusoid table run through `linear_pos` -- but WeNet's
//! `rel_shift`, which is what turns the second term's absolute positions into relative ones, is
//! commented out in the released source with a note that it is useless for speech. So the
//! position term is an ordinary matrix multiply against the first `T` rows of the table, and a
//! reimplementation that helpfully puts the shift back is wrong.
//!
//! **The gated feed forward gates the *second* half.** [`crate::flint::Graph::geglu`] computes
//! `gelu(first) * second`; `GEGLU` here is `gelu(second) * first`. The halves are the other way
//! round, so [`gated_feed_forward`] writes the two slices out rather than calling the operator,
//! and the weight stays laid out the way the checkpoint holds it. Both use the exact GELU, which
//! is the one thing about the two that does agree.
//!
//! **There is no mask.** Upstream pads a batch of recordings out to the longest and carries a
//! length mask through both stages; `emo_cond_mask_pad` is the `ConstantPad1d` that prepends a
//! `True` for the one latent. One recording at a time -- which is what a reading is -- makes
//! every position valid, so the masked softmax is an ordinary one and there is nothing to pad.
//!
//! # What this costs
//!
//! 163 M parameters, and 134 M of them are in one matrix. `Conv2dSubsampling2` flattens a
//! `(512, 511)` image per frame and `embed.out.0` reads all 261,632 of those numbers at once.
//! That is upstream's arithmetic and not a transcription error -- the released tensor really is
//! `(512, 261632)` -- but it is worth knowing before anyone wonders why a four-block conformer
//! weighs as much as the twenty-four-layer stack it feeds.

use crate::audio::{conv1d, depthwise_conv1d};
use crate::flint::{DType, Device, Extent, Graph, Tensor, Value};
use crate::layers::{Conv2d, LayerNorm, Linear};
use crate::Result;

/// The widths of the emotion path, as `config.yaml`'s `gpt.emo_condition_module` states them.
#[derive(Clone, Copy, Debug)]
pub struct Config {
    /// What w2v-bert hands over, and what the perceiver's latents are as wide as. Both 1024,
    /// which is this release's coincidence rather than a constraint: the conformer narrows to
    /// [`Config::encoder_dim`] in between and `proj_context` widens it back.
    pub input_dim: i32,
    /// How wide the conformer is: `output_size`.
    pub encoder_dim: i32,
    pub encoder_heads: i32,
    /// The conformer feed forward's hidden width: `linear_units`.
    pub encoder_units: i32,
    /// `num_blocks`. Four, against the speaker path's six.
    pub encoder_blocks: i32,
    /// How wide the depthwise convolution is. Odd, and padded symmetrically -- unlike
    /// [`crate::indextts::w2v_bert`]'s, which is causal.
    pub cnn_kernel: i32,
    /// How wide the perceiver works, which is also how wide its output is.
    pub latent_dim: i32,
    /// How many latents read the recording. **One**, which is why a vector comes out.
    pub latents: i32,
    pub perceiver_depth: i32,
    pub perceiver_heads: i32,
    /// Fixed at 64 by `PerceiverResampler`'s own default rather than by the configuration file,
    /// so the perceiver's inner width is 256 and not [`Config::latent_dim`].
    pub perceiver_head_dim: i32,
    /// `perceiver_mult`, which sets the gated feed forward's width -- see
    /// [`Config::perceiver_units`].
    pub perceiver_mult: f32,
    /// The GPT's width, which is what `emo_layer` finally produces.
    pub model_dim: i32,
    pub layer_norm_eps: f32,
}

impl Config {
    /// IndexTTS-2.5's, which is `config.yaml` unchanged.
    pub fn indextts() -> Config {
        Config {
            input_dim: 1024,
            encoder_dim: 512,
            encoder_heads: 4,
            encoder_units: 1024,
            encoder_blocks: 4,
            cnn_kernel: 15,
            latent_dim: 1024,
            latents: 1,
            perceiver_depth: 2,
            perceiver_heads: 4,
            perceiver_head_dim: 64,
            perceiver_mult: 2.0,
            model_dim: 1280,
            layer_norm_eps: 1e-5,
        }
    }

    /// How long the conformer's output is, given how long its input is.
    ///
    /// `Conv2dSubsampling2` is one 3 by 3 convolution at stride two with no padding, so this is
    /// the ordinary arithmetic for one -- not `frames / 2`, which is off by one for every length
    /// and is the kind of mistake a view notices only once it is already wrong.
    pub fn subsampled(frames: i32) -> i32 {
        (frames - 3) / 2 + 1
    }

    /// How wide that same convolution leaves the feature axis, which is the other factor in the
    /// 261,632 columns `embed.out.0` reads.
    fn feature_columns(&self) -> i32 {
        (self.input_dim - 3) / 2 + 1
    }

    fn encoder_head_dim(&self) -> i32 {
        self.encoder_dim / self.encoder_heads
    }

    /// The gated feed forward's hidden width: `int(dim * mult * 2 / 3)`, truncated.
    ///
    /// 1365 for this release, out of 1024 and a multiplier of two. The truncation is load-bearing
    /// -- `1024 * 2 * 2 / 3` is 1365.33 -- and the projection in front of it produces twice this,
    /// because half of what it produces is the gate.
    fn perceiver_units(&self) -> i32 {
        (f64::from(self.latent_dim) * f64::from(self.perceiver_mult) * 2.0 / 3.0) as i32
    }
}

/// What [`perceiver`]'s final normalization is given for an epsilon.
///
/// Upstream's `RMSNorm` is `F.normalize(x, dim=-1) * sqrt(dim) * gamma`, which is an RMS
/// normalization written the long way round: dividing by the L2 norm and multiplying by
/// `sqrt(dim)` is dividing by the root mean square. What differs is where the guard against a
/// zero vector sits -- `F.normalize` clamps the norm at 1e-12, `rms_norm` adds its epsilon to the
/// mean of squares -- and at 1e-12 neither is reachable by a vector with anything in it.
const RMS_EPS: f32 = 1e-12;

/// The sinusoid table `RelPositionalEncoding` carries, for as many positions as are wanted.
///
/// `(1, positions, dim)`, built on the host because it depends on nothing but its own shape.
///
/// # This is not quite the table the release runs with
///
/// The checkpoint stores one too -- `emo_conditioning_encoder.embed.pos_enc.pe`, five thousand
/// rows -- and it is not what this builds. It is **exactly this table rounded to `bfloat16`**,
/// widened back to `float32` and saved: every stored value sits on the `bfloat16` grid, and
/// `sinusoid(5000, 512).to(torch.bfloat16).float()` reproduces the released tensor with a
/// difference of zero. Somebody saved the model through `bfloat16` once and the buffer kept the
/// scar. It costs 7.5e-4 on average and 2.0e-3 at worst, which is one `bfloat16` step either side
/// of one.
///
/// Upstream runs with the stored one: `pe` is a persistent buffer, `load_checkpoint` calls
/// `load_state_dict(strict=False)` -- which only forgives *absent* keys, and this one is present
/// -- so the exact table computed in `PositionalEncoding.__init__` is overwritten the moment the
/// checkpoint loads. The exporter therefore writes it out like any other tensor, and a caller
/// with a package should hand [`graph`] the rows of *that* rather than the rows of this.
///
/// Exporting it rather than rebuilding it is a choice and could go the other way now that the
/// rounding is pinned down: ten megabytes against never having to argue that this function's
/// arithmetic rounds to the same `bfloat16` grid `torch` did. The ten megabytes are 0.3% of the
/// package, and the argument is not free, so the bytes won.
///
/// What this function is for is the tests, and running without a package. The difference it makes
/// is 0.13% of the position term after `linear_pos` -- below the noise floor of the half
/// precision this model runs in on a card, and worth stating rather than worth worrying about.
pub fn positional_encoding(positions: i32, dim: i32, device: Device) -> Result<Tensor> {
    let mut values = vec![0.0f32; (positions * dim) as usize];
    let decay = -(10000.0f64.ln()) / f64::from(dim);

    for position in 0..positions {
        let row = (position * dim) as usize;

        for pair in 0..dim / 2 {
            let angle = f64::from(position) * (f64::from(2 * pair) * decay).exp();

            values[row + (2 * pair) as usize] = angle.sin() as f32;
            values[row + (2 * pair + 1) as usize] = angle.cos() as f32;
        }
    }

    Ok(Tensor::from_f32(&[1, positions, dim], &values)?.to_device(device)?)
}

/// `self.embed`: `(N, frames, input_dim)` in, `(N, T', encoder_dim)` out, `T'` about half.
///
/// `Conv2dSubsampling2` is one convolution over the features read as an image one channel deep,
/// and then the whole of what it produced at each output frame -- every channel times every
/// surviving feature column -- flattened and projected. That flattening is why `embed.out.0` is
/// the size it is.
///
/// The scale at the end is `RelPositionalEncoding`'s `xscale`, which is the only thing that
/// encoding does to the signal: it multiplies by `sqrt(encoder_dim)` and hands the position table
/// back *beside* the embedding rather than added to it. Which is the whole of what makes this
/// encoding relative -- the table reaches the model only through [`attention`]'s second term.
#[track_caller]
pub fn embed(g: &Graph, x: Value, config: &Config, frames: i32) -> Value {
    let columns = config.feature_columns();
    let flattened = config.encoder_dim * columns;

    // (N, T, F) -> (N, 1, T, F): one image, one channel deep.
    let image = g.unsqueeze(x, 1);
    let convolved = Conv2d::graph(
        &g.subgraph("conv").subgraph("0"),
        image,
        1,
        config.encoder_dim,
        3,
        2,
        0,
    );

    // (N, C, T', F') -> (N, T', C * F'): a frame's worth of everything the convolution said.
    let rows = g.contiguous(g.transpose(g.relu(convolved), 1, 2));
    let flat = g.view(
        rows,
        [
            Extent::of(x, 0),
            Extent::At(Config::subsampled(frames)),
            Extent::At(flattened),
        ],
    );

    let projected = Linear::graph(
        &g.subgraph("out").subgraph("0"),
        flat,
        flattened,
        config.encoder_dim,
        true,
    );

    g.mul_scalar(projected, (config.encoder_dim as f32).sqrt())
}

/// The conformer's feed forward: widen, swish, narrow. `PositionwiseFeedForward`, whose
/// activation this configuration sets to SiLU.
#[track_caller]
fn feed_forward(g: &Graph, x: Value, config: &Config) -> Value {
    let wide = Linear::graph(
        &g.subgraph("w_1"),
        x,
        config.encoder_dim,
        config.encoder_units,
        true,
    );

    Linear::graph(
        &g.subgraph("w_2"),
        g.silu(wide),
        config.encoder_units,
        config.encoder_dim,
        true,
    )
}

/// `RelPositionMultiHeadedAttention`: the ordinary scores, plus a term built from the position
/// table and a pair of learned biases.
///
/// `positions` is [`positional_encoding`] for `frames` rows, `(1, frames, encoder_dim)`. It is a
/// graph input rather than a constant so that one table serves every block and every length up to
/// its own -- and so that a caller who already has one does not pay for a second.
#[track_caller]
fn attention(g: &Graph, x: Value, config: &Config, frames: i32, positions: Value) -> Value {
    let (heads, head_dim) = (config.encoder_heads, config.encoder_head_dim());
    let dim = config.encoder_dim;

    // (N, T, H, D), which is the layout the two biases are added in.
    let project = |name: &str| {
        let flat = Linear::graph(&g.subgraph(name), x, dim, dim, true);

        g.view(
            flat,
            [
                Extent::of(x, 0),
                Extent::At(frames),
                Extent::At(heads),
                Extent::At(head_dim),
            ],
        )
    };

    let query = project("linear_q");
    let key = g.contiguous(g.transpose(project("linear_k"), 1, 2));
    let value = g.contiguous(g.transpose(project("linear_v"), 1, 2));

    // One bias per head, added to every query before its own half of the score.
    let bias_u = g.load("pos_bias_u", &[heads, head_dim]);
    let bias_v = g.load("pos_bias_v", &[heads, head_dim]);
    let with_u = g.contiguous(g.transpose(g.add(query, bias_u), 1, 2));
    let with_v = g.contiguous(g.transpose(g.add(query, bias_v), 1, 2));

    // The table, projected and split into heads: (1, T, dim) -> (H, D, T). The batch axis goes
    // away rather than being broadcast, because a matmul whose right side has fewer dimensions
    // broadcasts it over the left side's batch -- which is exactly what one table for every
    // recording means.
    let projected = Linear::graph(&g.subgraph("linear_pos"), positions, dim, dim, false);
    let by_head = g.contiguous(g.transpose(
        g.view(
            projected,
            [Extent::At(frames), Extent::At(heads), Extent::At(head_dim)],
        ),
        0,
        1,
    ));
    let by_head = g.contiguous(g.transpose(by_head, 1, 2));

    // Matrices a and c, then b and d. No `rel_shift` between them: the released source has it
    // commented out, so these positions stay absolute.
    let content = g.matmul(with_u, g.contiguous(g.transpose(key, 2, 3)));
    let position = g.matmul(with_v, by_head);
    let scores = g.div_scalar(g.add(content, position), (head_dim as f32).sqrt());

    let out = g.matmul(g.softmax(scores), value);
    let merged = g.view(
        g.contiguous(g.transpose(out, 1, 2)),
        [
            Extent::of(x, 0),
            Extent::At(frames),
            Extent::At(heads * head_dim),
        ],
    );

    Linear::graph(&g.subgraph("linear_out"), merged, dim, dim, true)
}

/// `ConvolutionModule`: a gated pointwise pair around a symmetric depthwise convolution.
///
/// The normalization inside it is over channels, so time goes last for it and comes back. Note
/// that the module has no normalization of its own at the front -- the block applies `norm_conv`
/// before calling this, which is where [`crate::indextts::w2v_bert`]'s equivalent differs.
#[track_caller]
fn conv_module(
    g: &Graph,
    x: Value,
    config: &Config,
    dtype: DType,
    device: Device,
) -> Result<Value> {
    let channels = config.encoder_dim;

    // (N, T, C) -> (N, C, T): the convolutions read time.
    let x = g.contiguous(g.transpose(x, 1, 2));

    let first = g.subgraph("pointwise_conv1");
    let weight = first.load("weight", &[2 * channels, channels, 1]);
    let bias = first.load("bias", &[2 * channels]);
    let gated = conv1d(&first, x, weight, Some(bias), 1, 0, 1, 1, dtype, device)?;

    // A gated linear unit over the channel axis, which is halved by it.
    let left = g.contiguous(g.slice(gated, 1, 0, channels));
    let right = g.contiguous(g.slice(gated, 1, channels, 2 * channels));
    let x = g.mul(left, g.sigmoid(right));

    // Symmetric, so a frame reads as far forward as it reads back.
    let depthwise = g.subgraph("depthwise_conv");
    let weight = depthwise.load("weight", &[channels, 1, config.cnn_kernel]);
    let bias = depthwise.load("bias", &[channels]);
    let x = depthwise_conv1d(
        &depthwise,
        x,
        weight,
        Some(bias),
        channels,
        config.cnn_kernel,
        (config.cnn_kernel - 1) / 2,
        1,
        dtype,
        device,
    )?;

    let normed = LayerNorm::graph(
        &g.subgraph("norm"),
        g.contiguous(g.transpose(x, 1, 2)),
        channels,
        config.layer_norm_eps,
    );
    let x = g.contiguous(g.transpose(g.silu(normed), 1, 2));

    let second = g.subgraph("pointwise_conv2");
    let weight = second.load("weight", &[channels, channels, 1]);
    let bias = second.load("bias", &[channels]);
    let x = conv1d(&second, x, weight, Some(bias), 1, 0, 1, 1, dtype, device)?;

    Ok(g.contiguous(g.transpose(x, 1, 2)))
}

/// One `ConformerEncoderLayer`: attention, convolution, feed forward, each around a residual, and
/// a normalization over the lot.
///
/// **Not macaron.** `macaron_style` is false here, so there is one feed forward rather than two
/// halves of one, and `ff_scale` is one rather than a half. That is the opposite of
/// [`crate::indextts::w2v_bert`], whose two feed forwards *are* halved -- the two conformers in this
/// pipeline differ on exactly that, and a scale copied from the wrong one changes every layer.
#[track_caller]
#[allow(clippy::too_many_arguments)]
fn encoder_layer(
    g: &Graph,
    x: Value,
    config: &Config,
    frames: i32,
    positions: Value,
    dtype: DType,
    device: Device,
) -> Result<Value> {
    let dim = config.encoder_dim;
    let eps = config.layer_norm_eps;

    let normed = LayerNorm::graph(&g.subgraph("norm_mha"), x, dim, eps);
    let x = g.add(
        attention(&g.subgraph("self_attn"), normed, config, frames, positions),
        x,
    );

    let normed = LayerNorm::graph(&g.subgraph("norm_conv"), x, dim, eps);
    let x = g.add(
        conv_module(&g.subgraph("conv_module"), normed, config, dtype, device)?,
        x,
    );

    let normed = LayerNorm::graph(&g.subgraph("norm_ff"), x, dim, eps);
    let x = g.add(feed_forward(&g.subgraph("feed_forward"), normed, config), x);

    Ok(LayerNorm::graph(&g.subgraph("norm_final"), x, dim, eps))
}

/// `emo_conditioning_encoder`: `(N, frames, input_dim)` in, `(N, T', encoder_dim)` out.
///
/// [`embed`] halves the length and narrows the features, then [`Config::encoder_blocks`] blocks
/// read what is left, and `after_norm` closes the stack -- which a stack that normalizes before
/// each sub-block needs, or its last residual leaves unnormalized.
#[track_caller]
#[allow(clippy::too_many_arguments)]
pub fn encoder(
    g: &Graph,
    x: Value,
    config: &Config,
    frames: i32,
    positions: Value,
    dtype: DType,
    device: Device,
) -> Result<Value> {
    let mut running = embed(&g.subgraph("embed"), x, config, frames);

    let blocks = g.subgraph("encoders");
    for index in 0..config.encoder_blocks {
        running = encoder_layer(
            &blocks.subgraph(&index.to_string()),
            running,
            config,
            Config::subsampled(frames),
            positions,
            dtype,
            device,
        )?;
    }

    Ok(LayerNorm::graph(
        &g.subgraph("after_norm"),
        running,
        config.encoder_dim,
        config.layer_norm_eps,
    ))
}

/// The perceiver's cross attention, whose queries are also part of what they read.
///
/// `cross_attn_include_queries` is what does that: the context is the latents concatenated in
/// front of the recording, so a latent can carry its own state forward as well as gather from the
/// frames. `context` is `(N, frames, latent_dim)` and already projected.
#[track_caller]
fn perceiver_attention(
    g: &Graph,
    latents: Value,
    context: Value,
    config: &Config,
    frames: i32,
) -> Value {
    let (heads, head_dim) = (config.perceiver_heads, config.perceiver_head_dim);
    let inner = heads * head_dim;
    let dim = config.latent_dim;
    let read = config.latents + frames;

    let whole = g.cat(latents, context, 1);

    // One projection produces the keys and the values, halved along the last axis.
    let queries = Linear::graph(&g.subgraph("to_q"), latents, dim, inner, false);
    let pairs = Linear::graph(&g.subgraph("to_kv"), whole, dim, 2 * inner, false);
    let keys = g.contiguous(g.slice(pairs, 2, 0, inner));
    let values = g.contiguous(g.slice(pairs, 2, inner, 2 * inner));

    let split = |flat: Value, length: i32| {
        let shaped = g.view(
            flat,
            [
                Extent::of(latents, 0),
                Extent::At(length),
                Extent::At(heads),
                Extent::At(head_dim),
            ],
        );

        g.contiguous(g.transpose(shaped, 1, 2))
    };

    let query = split(queries, config.latents);
    let key = split(keys, read);
    let value = split(values, read);

    // Scaled on the query side, the way `Attend` does it, which is the same arithmetic.
    let scores = g.mul_scalar(
        g.matmul(query, g.contiguous(g.transpose(key, 2, 3))),
        1.0 / (head_dim as f32).sqrt(),
    );
    let out = g.matmul(g.softmax(scores), value);
    let merged = g.view(
        g.contiguous(g.transpose(out, 1, 2)),
        [
            Extent::of(latents, 0),
            Extent::At(config.latents),
            Extent::At(inner),
        ],
    );

    Linear::graph(&g.subgraph("to_out"), merged, inner, dim, false)
}

/// The perceiver's `FeedForward`, which is a GEGLU between two projections.
///
/// The subgraphs are named `0` and `2` because upstream builds this as an `nn.Sequential` and the
/// activation in the middle is position one. Keeping the numbers is what lets the exporter copy
/// `layers.0.1.0.weight` across without renaming it.
///
/// **The gate is the second half**, which is the other way round from
/// [`crate::flint::Graph::geglu`]. Hence the two slices rather than the operator.
#[track_caller]
fn gated_feed_forward(g: &Graph, x: Value, config: &Config) -> Value {
    let dim = config.latent_dim;
    let units = config.perceiver_units();

    let wide = Linear::graph(&g.subgraph("0"), x, dim, 2 * units, true);
    let value = g.contiguous(g.slice(wide, 2, 0, units));
    let gate = g.contiguous(g.slice(wide, 2, units, 2 * units));

    Linear::graph(
        &g.subgraph("2"),
        g.mul(g.gelu(gate), value),
        units,
        dim,
        true,
    )
}

/// `emo_perceiver_encoder`: `(N, frames, encoder_dim)` in, `(N, latents, latent_dim)` out.
///
/// `frames` is how long the conformer's output is, not how long its input was -- so
/// [`Config::subsampled`], which is what [`graph`] hands it.
#[track_caller]
pub fn perceiver(
    g: &Graph,
    x: Value,
    config: &Config,
    frames: i32,
    dtype: DType,
    device: Device,
) -> Value {
    let dim = config.latent_dim;
    let context = Linear::graph(
        &g.subgraph("proj_context"),
        x,
        config.encoder_dim,
        dim,
        true,
    );

    // The latents are one parameter shared by every recording, and this graph does not know how
    // many recordings there are. Adding a zero of the right shape is how it gets a copy each --
    // one node, against a `view` that would have to be told the batch size up front.
    let table = g.load("latents", &[config.latents, dim]);
    let empty = g.zeros(
        [
            Extent::of(x, 0),
            Extent::At(config.latents),
            Extent::At(dim),
        ],
        dtype,
        device,
    );
    let mut latents = g.add(g.unsqueeze(table, 0), empty);

    let layers = g.subgraph("layers");
    for index in 0..config.perceiver_depth {
        let layer = layers.subgraph(&index.to_string());

        latents = g.add(
            perceiver_attention(&layer.subgraph("0"), latents, context, config, frames),
            latents,
        );
        latents = g.add(
            gated_feed_forward(&layer.subgraph("1"), latents, config),
            latents,
        );
    }

    let gamma = g.subgraph("norm").load("gamma", &[dim]);

    g.rms_norm(latents, gamma, RMS_EPS)
}

/// The whole of it: w2v-bert features in, the `(N, model_dim)` emotion row out.
///
/// `x` is `(N, frames, input_dim)` -- `hidden_states[17]` of [`crate::indextts::w2v_bert`], standardized by
/// the release's own mean and deviation, which is the same tensor the semantic codec is handed.
/// `positions` is [`positional_encoding`] for [`Config::subsampled`] rows.
///
/// What comes back goes straight into [`crate::indextts::gpt::Gpt::prefill`]'s `emotion`.
#[track_caller]
#[allow(clippy::too_many_arguments)]
pub fn graph(
    g: &Graph,
    x: Value,
    config: &Config,
    frames: i32,
    positions: Value,
    dtype: DType,
    device: Device,
) -> Result<Value> {
    let encoded = encoder(
        &g.subgraph("emo_conditioning_encoder"),
        x,
        config,
        frames,
        positions,
        dtype,
        device,
    )?;

    let resampled = perceiver(
        &g.subgraph("emo_perceiver_encoder"),
        encoded,
        config,
        Config::subsampled(frames),
        dtype,
        device,
    );

    // One latent, so the axis it sits on carries nothing and goes away -- upstream's `squeeze(1)`.
    let vector = g.squeeze(resampled, 1);

    let widened = Linear::graph(
        &g.subgraph("emovec_layer"),
        vector,
        config.latent_dim,
        config.model_dim,
        true,
    );

    Ok(Linear::graph(
        &g.subgraph("emo_layer"),
        widened,
        config.model_dim,
        config.model_dim,
        true,
    ))
}
