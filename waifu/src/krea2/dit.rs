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


//! The denoiser: one stream of tokens, the prompt in front of the picture, twenty-eight blocks
//! deep and six thousand wide.
//!
//! A block is a self attention and a SwiGLU, both modulated by the timestep -- and that is all.
//! There is no cross attention here, because there is nothing to cross into: the prompt was
//! concatenated onto the front of the sequence before the first block and every block reads it
//! along with everything else.
//!
//! What happens before that is [`Dit`]'s other half, the *text fusion*. The prompt arrives as
//! twelve layers of the encoder's hidden states stacked per token, `(L, 12, 2560)`. Two blocks
//! attend **across the twelve**, one token at a time, with the sentence as the batch; a matrix of
//! twelve numbers collapses the stack; two more blocks attend **along the sentence**; and a
//! projection widens the result to the model.
//!
//! # What is easy to get wrong here
//!
//! Five things, none of which fails loudly. `docs/krea2.md` has them at more length.
//!
//! *The layerwise blocks attend across layers, not tokens.* The reshape is
//! `(B, L, 12, D) -> (B * L, 12, D)`, so twelve is the sequence and the sentence is the batch.
//! Reading it the other way round type checks all the way to a picture.
//!
//! *The rotary pairing is adjacent.* `x[2i]` is rotated against `x[2i + 1]`, where every language
//! model here -- including the Qwen3-VL encoder that feeds this one -- pairs `x[i]` against
//! `x[i + half]`. The two are the same arithmetic over a different permutation of the head.
//!
//! *The norm scales are stored zero centred.* Every RMSNorm in the checkpoint multiplies by
//! `1 + weight`. The exporter folds the one in, so what the package holds is the multiplier
//! itself; nothing here adds it again.
//!
//! *The modulation is one vector for the whole model.* `time_mod_proj` produces a single `6 * D`
//! and each block adds its own learned table to it. There is no per-block projection of the
//! timestep, which is what the Cosmos transformer next door has.
//!
//! *The GELUs are the tanh approximation.* flint's `gelu` is the exact one, which is right for
//! every other model here and wrong for this one, so it is written out -- see [`gelu_tanh`].

use std::cell::RefCell;
use std::fmt;
use std::rc::Rc;

use super::config::DitConfig;
use crate::error::{Error, Result};
use crate::flint::{
    check_parameters, DType, Device, Extent, Graph, Held, Ir, ParamSource, RunContext, Tensor,
    Value,
};
use crate::layers::Linear;

/// What every attention in this model is called in the package, and what it is made of.
///
/// One projection for the query, the key, the value *and* the gate: four matrices that all read
/// the same normalized stream, so they are one matrix and one multiply. The runtime splits the
/// result back on widths it knows.
const QKVG: &str = "attn.qkvg_proj";

/// The rotary table for one sequence, built on the host because a graph is compiled once and this
/// depends on how large the picture is.
struct Placed {
    cos: Tensor,
    sin: Tensor,
    /// What it was built for, so that a second step at the same size reuses it.
    key: (i32, i32, i32),
}

/// Where each token sits, across three axes, with the head divided between them.
///
/// The text tokens all sit at the origin -- a prompt has no geometry -- and the image tokens at
/// `(0, row, column)` in the order patchify laid them out, which is row major. A still picture is
/// at time zero, so the first axis contributes a cosine of one everywhere: present, and not the
/// same as absent.
fn rope_table(
    config: &DitConfig,
    text_length: i32,
    rows: i32,
    columns: i32,
    device: Device,
) -> Result<Placed> {
    let (dim_t, dim_h, dim_w) = config.rope_axes();
    let heads = config.num_heads as usize;
    let theta = config.rope_theta as f64;

    // One base for all three axes, unlike Anima's, which extrapolates the spatial ones.
    let frequencies = |dim: i32| -> Vec<f64> {
        (0..dim / 2)
            .map(|index| 1.0 / theta.powf(2.0 * index as f64 / dim as f64))
            .collect()
    };
    let (freq_t, freq_h, freq_w) = (
        frequencies(dim_t),
        frequencies(dim_h),
        frequencies(dim_w),
    );

    let pairs = config.rope_pairs() as usize;
    let tokens = (text_length + rows * columns) as usize;
    let mut cos = Vec::with_capacity(tokens * heads * pairs);
    let mut sin = Vec::with_capacity(tokens * heads * pairs);

    let mut place = |row: i32, column: i32| {
        let mut angles = Vec::with_capacity(pairs);
        angles.extend(freq_t.iter().map(|_| 0.0));
        angles.extend(freq_h.iter().map(|f| row as f64 * f));
        angles.extend(freq_w.iter().map(|f| column as f64 * f));

        for _ in 0..heads {
            for angle in &angles {
                cos.push(angle.cos() as f32);
                sin.push(angle.sin() as f32);
            }
        }
    };

    for _ in 0..text_length {
        place(0, 0);
    }
    for row in 0..rows {
        for column in 0..columns {
            place(row, column);
        }
    }

    // `(tokens, heads, pairs, 1)` rather than `(tokens, heads, pairs)`: the head is read as
    // `(.., pairs, 2)` where it is rotated, and a binary operation matches trailing dimensions.
    // Shaped here rather than in the graph because the key's table is a slice of the query's,
    // and a slice is a view with a gap in it that a reshape cannot see past.
    let shape = [tokens as i32, config.num_heads, config.rope_pairs(), 1];
    Ok(Placed {
        cos: Tensor::from_f32(&shape, &cos)?.to_device(device)?,
        sin: Tensor::from_f32(&shape, &sin)?.to_device(device)?,
        key: (text_length, rows, columns),
    })
}

/// The sinusoid a timestep becomes before anything is done to it.
///
/// Cosine first and then sine, the exponent divided by the half width rather than by one less
/// than it, and the timestep multiplied by a thousand on the way in -- the model is handed a
/// sigma in `0..=1` and embeds a number in `0..=1000`. All three are the opposite of the commoner
/// convention and all three are what this was trained with.
fn timestep_sinusoid(timestep: f32, channels: i32, device: Device) -> Result<Tensor> {
    let half = (channels / 2) as usize;
    let mut values = vec![0.0f32; channels as usize];

    for index in 0..half {
        let exponent = -(10000.0f64).ln() * index as f64 / half as f64;
        let angle = timestep as f64 * 1000.0 * exponent.exp();
        values[index] = angle.cos() as f32;
        values[half + index] = angle.sin() as f32;
    }

    Ok(Tensor::from_f32(&[1, 1, channels], &values)?.to_device(device)?)
}

/// `0.5x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 x^3)))`, which is `F.gelu(approximate="tanh")`.
///
/// flint's [`Graph::gelu`] is the exact one, `x * Phi(x)`, which is what every other model here
/// asks for. The two differ by about 1e-3 around the shoulder, and all three of this model's
/// GELUs -- two in the timestep path, one in the text projection -- are the approximation, so
/// using the exact one puts that error into the modulation of every block at every step.
///
/// Written as `h + h * tanh(...)` rather than `h * (1 + tanh(...))` because flint adds tensors to
/// tensors and not scalars to them.
fn gelu_tanh(g: &Graph, x: Value) -> Value {
    const ROOT_TWO_OVER_PI: f32 = 0.797_884_56;
    const COEFFICIENT: f32 = 0.044_715;

    let cube = g.mul(x, g.square(x));
    let inner = g.mul_scalar(g.add(x, g.mul_scalar(cube, COEFFICIENT)), ROOT_TWO_OVER_PI);

    let half = g.mul_scalar(x, 0.5);
    g.add(half, g.mul(half, g.tanh(inner)))
}

/// `(1 + scale) * norm(x) + shift`, written as `norm(x) * scale + norm(x) + shift` so that no
/// scalar has to be added to a tensor.
///
/// `x` arrives in float32 and what comes back is in `narrow`: the residual stream is wide and the
/// sublayers are not. The modulation is done wide and only its result narrowed, which is where the
/// widths have to fall -- the shift a deep block applies is large enough that folding it in half
/// precision can reach infinity, and an infinity minus an infinity is the NaN that comes out the
/// far end.
fn modulate(
    g: &Graph,
    x: Value,
    norm: &str,
    width: i32,
    eps: f32,
    scale: Value,
    shift: Value,
    narrow: DType,
) -> Value {
    let weight = g.cast(g.subgraph(norm).load(Linear::WEIGHT, &[width]), DType::Float);
    let normed = g.rms_norm(x, weight, eps);

    let scaled = g.add(g.mul(normed, g.cast(scale, DType::Float)), normed);
    g.cast(g.add(scaled, g.cast(shift, DType::Float)), narrow)
}

/// Add a sublayer's gated output back onto the wide residual stream.
///
/// Both halves are widened before they are multiplied, not after: a gate of a hundred against an
/// output of a thousand is an infinity in half and an ordinary number in whole.
fn accumulate(g: &Graph, x: Value, out: Value, gate: Value) -> Value {
    let wide = g.mul(g.cast(out, DType::Float), g.cast(gate, DType::Float));
    g.add(x, wide)
}

/// A block's modulation: the one vector the whole model shares, plus this block's own table, cut
/// into `parts`.
///
/// Each piece comes back flat so that it broadcasts over the `(1, L, D)` it modulates -- what is
/// carried is one timestep, not one per token.
fn modulation(g: &Graph, shared: Value, width: i32, parts: i32) -> Vec<Value> {
    let table = g.load("scale_shift_table", &[parts, width]);
    let whole = g.add(shared, g.view(table, [Extent::At(parts * width)]));

    (0..parts)
        .map(|index| g.slice(whole, 0, index * width, (index + 1) * width))
        .collect()
}

/// Split a projection into heads and normalize each one.
fn heads(g: &Graph, x: Value, norm: &str, count: i32, head_dim: i32, eps: f32) -> Value {
    let batch = Extent::of(x, 0);
    let length = Extent::of(x, 1);
    let viewed = g.view(x, [batch, length, count.into(), head_dim.into()]);

    let weight = g.subgraph(norm).load(Linear::WEIGHT, &[head_dim]);
    g.rms_norm(viewed, weight, eps)
}

/// Rotate the head's pairs, pairing each even element against the odd one beside it.
///
/// Not the half-against-half rotation [`TextEncoder`](super::TextEncoder) does. The tables hold
/// one angle per pair, so the head is read as `(.., pairs, 2)` and put back the same way.
///
/// The batch is dropped on the way in and put back on the way out. It is one -- this model draws
/// one picture at a time -- and what that buys is a rank of four rather than five, which is as
/// deep as flint's binary kernels go over a tensor with a gap in it. The two slices below are
/// exactly that.
fn rotate(g: &Graph, x: Value, cos: Value, sin: Value, pairs: i32) -> Value {
    let length = Extent::of(x, 1);
    let count = Extent::of(x, 2);

    let split = g.view(x, [length, count, pairs.into(), Extent::At(2)]);
    let even = g.slice(split, 3, 0, 1);
    let odd = g.slice(split, 3, 1, 2);

    let left = g.sub(g.mul(even, cos), g.mul(odd, sin));
    let right = g.add(g.mul(even, sin), g.mul(odd, cos));

    g.view(
        g.cat(left, right, 3),
        [Extent::At(1), length, count, (2 * pairs).into()],
    )
}

/// The attention every block in this model has: grouped, gated, and normalized per head.
///
/// `rotary` is the pair of tables for the sequence, or None where there are no positions to place
/// -- which is the text fusion, whose tokens have none.
#[allow(clippy::too_many_arguments)]
fn attend(
    g: &Graph,
    x: Value,
    width: i32,
    num_heads: i32,
    num_kv_heads: i32,
    head_dim: i32,
    eps: f32,
    rotary: Option<(Value, Value)>,
) -> Value {
    let batch = Extent::of(x, 0);
    let length = Extent::of(x, 1);
    let queries = num_heads * head_dim;
    let keys = num_kv_heads * head_dim;

    // Query, key, value and gate in one multiply. The gate is as wide as the model, not as the
    // keys: it multiplies the attention's output rather than anything inside it.
    let fused = Linear::graph(
        &g.subgraph(QKVG),
        x,
        width,
        queries + 2 * keys + width,
        false,
    );
    let q = g.contiguous(g.slice(fused, 2, 0, queries));
    let k = g.contiguous(g.slice(fused, 2, queries, queries + keys));
    let v = g.contiguous(g.slice(fused, 2, queries + keys, queries + 2 * keys));
    let gate = g.contiguous(g.slice(
        fused,
        2,
        queries + 2 * keys,
        queries + 2 * keys + width,
    ));

    let q = heads(g, q, "attn.q_norm", num_heads, head_dim, eps);
    let k = heads(g, k, "attn.k_norm", num_kv_heads, head_dim, eps);
    let v = g.view(v, [batch, length, num_kv_heads.into(), head_dim.into()]);

    let pairs = head_dim / 2;
    let (q, k) = match rotary {
        None => (q, k),
        Some((cos, sin)) => (
            rotate(g, q, cos, sin, pairs),
            // The key's table is the query's, narrowed to the heads it has.
            rotate(
                g,
                k,
                g.slice(cos, 1, 0, num_kv_heads),
                g.slice(sin, 1, 0, num_kv_heads),
                pairs,
            ),
        ),
    };

    let q = g.contiguous(g.transpose(q, 1, 2));
    let k = g.contiguous(g.transpose(k, 1, 2));
    let v = g.contiguous(g.transpose(v, 1, 2));

    // A picture is looked at whole, and so is the prompt in front of it: nothing here is causal.
    let out = g.attention(q, k, v, false);
    let out = g.contiguous(g.transpose(out, 1, 2));
    let out = g.view(out, [batch, length, queries.into()]);

    // The gate is the last thing before the output projection, and it is a sigmoid of a
    // projection of the *input* rather than of anything the attention produced.
    let gated = g.mul(out, g.sigmoid(gate));
    Linear::graph(&g.subgraph("attn.out_proj"), gated, width, width, false)
}

/// The SwiGLU every block in this model has. The gate and the up projection are one weight, gate
/// first, because that is the half `swiglu` puts the swish on.
fn feed_forward(g: &Graph, x: Value, width: i32, hidden: i32) -> Value {
    let gated = Linear::graph(&g.subgraph("ff.gate_up_proj"), x, width, 2 * hidden, false);
    Linear::graph(&g.subgraph("ff.down"), g.swiglu(gated), hidden, width, false)
}

/// One text fusion block: an attention and a feed forward, both pre-normed, neither modulated.
///
/// The same block whether it is reading across the twelve tapped layers or along the sentence.
/// What differs is the shape it is handed, and it does not need to know which.
fn fusion_block(config: &DitConfig, g: &Graph, x: Value, narrow: DType) -> Value {
    let width = config.text_hidden_size;
    let head_dim = width / config.text_num_heads;

    let normed = modulate_free(g, x, "norm1", width, config.norm_eps, narrow);
    let out = attend(
        g,
        normed,
        width,
        config.text_num_heads,
        config.text_num_kv_heads,
        head_dim,
        config.norm_eps,
        None,
    );
    let x = g.add(x, g.cast(out, DType::Float));

    let normed = modulate_free(g, x, "norm2", width, config.norm_eps, narrow);
    let out = feed_forward(g, normed, width, config.text_mlp_size);
    g.add(x, g.cast(out, DType::Float))
}

/// An RMSNorm off the wide stream and onto the narrow one, where there is no modulation to apply.
fn modulate_free(
    g: &Graph,
    x: Value,
    norm: &str,
    width: i32,
    eps: f32,
    narrow: DType,
) -> Value {
    let weight = g.cast(g.subgraph(norm).load(Linear::WEIGHT, &[width]), DType::Float);
    g.cast(g.rms_norm(x, weight, eps), narrow)
}

/// The text fusion: twelve layers of encoder states in, one sequence of tokens out.
fn text_fusion(config: &DitConfig, g: &Graph, context: Value, narrow: DType) -> Value {
    let width = config.text_hidden_size;
    let length = Extent::of(context, 1);
    let layers = config.num_text_layers;

    // The sentence becomes the batch and the twelve tapped layers become the sequence: these
    // blocks attend across what the encoder was thinking at different depths about *one* token,
    // and never between one token and the next.
    let mut x = g.cast(
        g.view(context, [length, layers.into(), width.into()]),
        DType::Float,
    );
    for index in 0..config.text_layerwise_blocks {
        x = fusion_block(config, &g.subgraph(&format!("layerwise{index}")), x, narrow);
    }

    // Twelve numbers, one per tapped layer, collapsing the stack. Written as a bare multiply
    // rather than through `Linear` because a matrix twelve wide is not a candidate for the
    // quantized path -- an FP8 multiply wants its inner dimension in multiples of sixteen -- and
    // the package stores it unquantized whatever the rest of the model is stored as.
    let x = g.contiguous(g.transpose(x, 1, 2));
    let weight = g.subgraph("projector").load(Linear::WEIGHT, &[1, layers]);
    let x = g.matmul(x, g.transpose(g.cast(weight, DType::Float), 0, 1));

    // And back to one sequence of tokens, which is what the rest of the model reads.
    let mut x = g.view(x, [Extent::At(1), length, width.into()]);
    for index in 0..config.text_refiner_blocks {
        x = fusion_block(config, &g.subgraph(&format!("refiner{index}")), x, narrow);
    }

    x
}

/// One block of the stream, written into `g`.
fn block(
    config: &DitConfig,
    g: &Graph,
    x: Value,
    shared: Value,
    cos: Value,
    sin: Value,
    narrow: DType,
) -> Value {
    let d = config.hidden_size;
    let parts = modulation(g, shared, d, 6);

    let normed = modulate(g, x, "norm1", d, config.norm_eps, parts[0], parts[1], narrow);
    let out = attend(
        g,
        normed,
        d,
        config.num_heads,
        config.num_kv_heads,
        config.head_dim,
        config.norm_eps,
        Some((cos, sin)),
    );
    let x = accumulate(g, x, out, parts[2]);

    let normed = modulate(g, x, "norm2", d, config.norm_eps, parts[3], parts[4], narrow);
    let out = feed_forward(g, normed, d, config.mlp_size);
    accumulate(g, x, out, parts[5])
}

/// The whole denoiser, written into `g`. Patches and a prompt in, patches out.
fn write(config: &DitConfig, float_type: DType, g: &Graph) {
    // Cast rather than taken as they arrive: a caller with a float32 latent in hand is not
    // wrong, and a stream half of which is half precision is a matmul that refuses at run time.
    let tokens = g.cast(g.input("tokens"), float_type);
    let context = g.input("context");
    let sinusoid = g.cast(g.input("sinusoid"), float_type);
    let cos = g.cast(g.input("rope_cos"), float_type);
    let sin = g.cast(g.input("rope_sin"), float_type);

    let d = config.hidden_size;

    // The timestep, twice over: `emb` is what the last layer modulates on, and `shared` is the
    // six-part vector every block adds its own table to. The second is a projection of the first
    // and not the other way round.
    let emb = {
        let sub = g.subgraph("time_embed");
        let hidden = Linear::graph(
            &sub.subgraph("linear_1"),
            sinusoid,
            config.timestep_embed_dim,
            d,
            true,
        );
        Linear::graph(&sub.subgraph("linear_2"), gelu_tanh(g, hidden), d, d, true)
    };
    let shared = g.view(
        Linear::graph(
            &g.subgraph("time_mod_proj"),
            gelu_tanh(g, emb),
            d,
            6 * d,
            true,
        ),
        [Extent::At(6 * d)],
    );
    let emb = g.view(emb, [Extent::At(d)]);

    // The prompt: twelve layers fused into one sequence, then widened to the model.
    let prompt = text_fusion(config, &g.subgraph("text_fusion"), context, float_type);
    let prompt = {
        let sub = g.subgraph("txt_in");
        let normed = modulate_free(
            &sub,
            prompt,
            "norm",
            config.text_hidden_size,
            config.norm_eps,
            float_type,
        );
        let hidden = Linear::graph(
            &sub.subgraph("linear_1"),
            normed,
            config.text_hidden_size,
            d,
            true,
        );
        Linear::graph(&sub.subgraph("linear_2"), gelu_tanh(g, hidden), d, d, true)
    };

    let picture = Linear::graph(&g.subgraph("img_in"), tokens, config.in_channels, d, true);

    // One stream. The prompt goes in front, which is where the position table put its zeros and
    // where the output slice expects to find it.
    //
    // The residual stream runs in float32 whatever the sublayers run in: twelve billion
    // parameters of accumulated residual is not a thing to keep in half, and the reference runs
    // it in bfloat16, which has float32's range where half does not.
    let mut x = g.cast(g.cat(prompt, picture, 1), DType::Float);

    for index in 0..config.num_blocks {
        x = block(
            config,
            &g.subgraph(&format!("block{index}")),
            x,
            shared,
            cos,
            sin,
            float_type,
        );
    }

    // The last layer modulates on two parts rather than six, and on `emb` rather than on the
    // vector the blocks share.
    let final_g = g.subgraph("final");
    let table = final_g.load("scale_shift_table", &[2, d]);
    let part = |index: i32| {
        final_g.add(
            emb,
            final_g.view(final_g.slice(table, 0, index, index + 1), [Extent::At(d)]),
        )
    };
    let x = modulate(
        &final_g,
        x,
        "norm",
        d,
        config.norm_eps,
        part(0),
        part(1),
        float_type,
    );

    g.output(
        "velocity",
        Linear::graph(&final_g.subgraph("linear"), x, d, config.in_channels, true),
    );
}

pub struct Dit {
    config: DitConfig,
    ir: Ir,
    /// The weights that are kept on the card between passes, which under
    /// [`Residency::LowVram`](crate::Residency::LowVram) is none of them.
    held: Held,
    weights: Rc<dyn ParamSource>,
    float_type: DType,
    /// The rotary table for whatever was drawn last. A step is one call and a run is many at the
    /// same size, so building it once is twenty-five million sines the second step does not have
    /// to spend -- and the table for a 1024 by 1024 picture is fifty megabytes, which is not a
    /// thing to send across the bus eight times over.
    placed: RefCell<Option<Placed>>,
}

impl fmt::Debug for Dit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Dit({} instructions)", self.ir.len())
    }
}

impl Dit {
    pub fn build(
        config: DitConfig,
        name: &str,
        weights: &Rc<dyn ParamSource>,
        float_type: DType,
    ) -> Result<Dit> {
        let graph = Graph::with_weights(config.weight_format);
        write(&config, float_type, &graph.subgraph(name));
        check_parameters(&graph, weights.as_ref())?;

        let ir = Ir::compile(&graph, weights.residency());
        let held = ir.load(weights.as_ref())?;

        Ok(Dit {
            weights: Rc::clone(weights),
            held,
            ir,
            config,
            float_type,
            placed: RefCell::new(None),
        })
    }

    pub fn config(&self) -> &DitConfig {
        &self.config
    }

    pub fn ir(&self) -> &Ir {
        &self.ir
    }

    pub fn float_type(&self) -> DType {
        self.float_type
    }

    /// Cut a latent into the tokens the graph reads.
    ///
    /// `(1, C, H, W)` becomes `(1, H / p * W / p, C * p * p)`, channel first within a patch. No
    /// mask channel is concatenated: that is the Cosmos transformer next door, and this one reads
    /// exactly the latent.
    ///
    /// The batch is left out of the shuffle throughout: it is always one here, and the copy
    /// behind `contiguous` is written for ranks up to five.
    fn patchify(&self, latent: &Tensor) -> Result<Tensor> {
        let patch = self.config.patch_size;
        let channels = self.config.latent_channels;
        let (height, width) = (latent.shape_at(2)?, latent.shape_at(3)?);
        let (rows, columns) = (height / patch, width / patch);

        let x = latent.view(&[channels, rows, patch, columns, patch])?;

        // (c, rows, ph, columns, pw) -> (rows, columns, c, ph, pw)
        let x = x.transpose(0, 1)?.transpose(1, 3)?.transpose(2, 3)?;
        Ok(x.contiguous()?
            .view(&[1, rows * columns, self.config.in_channels])?)
    }

    /// Put the tokens back into a latent. Exactly [`Dit::patchify`] backwards, which is worth
    /// saying because Anima's pair are not inverses and this one's are.
    fn unpatchify(&self, tokens: &Tensor, height: i32, width: i32) -> Result<Tensor> {
        let patch = self.config.patch_size;
        let channels = self.config.latent_channels;
        let (rows, columns) = (height / patch, width / patch);

        let x = tokens.view(&[rows, columns, channels, patch, patch])?;

        // (rows, columns, c, ph, pw) -> (c, rows, ph, columns, pw)
        let x = x.transpose(2, 3)?.transpose(1, 3)?.transpose(0, 1)?;
        Ok(x.contiguous()?.view(&[1, channels, height, width])?)
    }

    /// One step's velocity for `latent` at `timestep`, conditioned on `context`.
    ///
    /// `latent` is `(1, C, H, W)` with both sides a multiple of the patch size; `timestep` is the
    /// sigma itself, in `0..=1`, which the embedding multiplies by a thousand on its own.
    /// `context` is `(1, L, 12, D)` from the encoder -- every tapped layer of every prompt token,
    /// and no padding.
    pub fn forward(&self, latent: &Tensor, timestep: f32, context: &Tensor) -> Result<Tensor> {
        let dim = latent.dim()?;
        if dim != 4 {
            return Err(Error::model(format!(
                "the denoiser takes a latent as (1, C, H, W), got a {dim}-D tensor"
            )));
        }
        let dim = context.dim()?;
        if dim != 4 {
            return Err(Error::model(format!(
                "the denoiser takes a prompt as (1, L, layers, D), got a {dim}-D tensor"
            )));
        }

        let (height, width) = (latent.shape_at(2)?, latent.shape_at(3)?);
        let patch = self.config.patch_size;
        if height % patch != 0 || width % patch != 0 {
            return Err(Error::model(format!(
                "a {height} by {width} latent does not divide into {patch} by {patch} patches"
            )));
        }

        let layers = context.shape_at(2)?;
        if layers != self.config.num_text_layers {
            return Err(Error::model(format!(
                "the prompt carries {layers} tapped layers where this model reads {}",
                self.config.num_text_layers
            )));
        }

        let device = latent.device();
        let text_length = context.shape_at(1)?;
        let (rows, columns) = (height / patch, width / patch);

        let tokens = self.patchify(latent)?;
        let sinusoid = timestep_sinusoid(timestep, self.config.timestep_embed_dim, device)?;

        let mut placed = self.placed.borrow_mut();
        if placed.as_ref().map(|table| table.key) != Some((text_length, rows, columns)) {
            *placed = Some(rope_table(&self.config, text_length, rows, columns, device)?);
        }
        let table = placed.as_ref().expect("it was just built");

        let run = RunContext::new(&*self.weights)
            .input("tokens", &tokens)
            .input("context", context)
            .input("sinusoid", &sinusoid)
            .input("rope_cos", &table.cos)
            .input("rope_sin", &table.sin);

        let outputs = self.ir.run(&self.held, &run)?;

        let velocity = outputs
            .iter()
            .find(|(name, _)| name == "velocity")
            .map(|(_, tensor)| tensor.clone())
            .ok_or_else(|| Error::model("the denoiser produced no velocity"))?;

        // The prompt rides through every block with the picture and is dropped here. The last
        // layer runs over it too, which is a few thousand rows of wasted arithmetic against four
        // thousand of wanted -- and the alternative is a slice the graph would have to make on a
        // length it does not know until it runs.
        let picture = velocity
            .slice(1, text_length, text_length + rows * columns)?
            .contiguous()?;
        self.unpatchify(&picture, height, width)
    }
}
