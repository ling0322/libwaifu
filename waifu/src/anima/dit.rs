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

//! The denoiser: a Cosmos-Predict2 transformer over patches of a sixteen-channel latent.
//!
//! Twenty-eight blocks -- forty in the community expansion, which comes from the package rather
//! than from here -- each a self attention over the picture, a cross attention into the prompt,
//! and a feed forward, with all three modulated by the timestep.
//!
//! What it returns is a velocity, not a denoised latent: this is a flow model, and turning that
//! into a picture is [`FlowSampler`](super::FlowSampler)'s business.
//!
//! # What is easy to get wrong here
//!
//! Four things, none of which fails loudly. `docs/anima.md` has them at more length.
//!
//! *The timestep embedder's linears do not feed the modulation.* Under AdaLN-LoRA they produce
//! only the 6144-wide LoRA term added to every block's modulation; what the blocks modulate on is
//! the RMSNorm of the raw sinusoid those linears were handed.
//!
//! *The norms are LayerNorm and carry no weights.* Mean subtracting,
//! `elementwise_affine=False` -- which is why no block in the package holds a norm tensor -- and
//! not the RMSNorm this model uses everywhere else.
//!
//! *Rotation is for the self attention only.* The cross attention reads padded text, which has no
//! geometry to place.
//!
//! *Patchify and unpatchify are not inverses.* Going in the channel leads, `(c r m n)`; coming out
//! it trails, `(p1 p2 t c)`. Both are done outside the graph, in [`Dit::forward`], because a graph
//! cannot divide a dimension it will not know until it runs.

use std::fmt;
use std::rc::Rc;

use super::config::DitConfig;
use crate::error::{Error, Result};
use crate::flint::{
    check_parameters, functional as F, DType, Device, Extent, Graph, Ir, ParamSource, RunContext,
    Tensor, Value,
};
use crate::layers::Linear;

/// The rotary table for one resolution, and the sinusoid for one timestep.
///
/// Both depend on what is being drawn rather than on the weights, so both are computed per call
/// and handed to the compiled graph as inputs.
struct Placed {
    cos: Tensor,
    sin: Tensor,
}

/// Where each token sits, across three axes, with the head divided between them.
fn rope_table(config: &DitConfig, rows: i32, columns: i32, device: Device) -> Result<Placed> {
    let (dim_t, dim_h, dim_w) = config.rope_axes();
    let (base_t, base_h, base_w) = config.rope_bases();
    let heads = config.num_heads as usize;

    // A base per axis, not one shared between them: the two spatial axes are extrapolated four
    // fold and time is not, so height and width rotate on 42870.9 where time rotates on 10000.
    let frequencies = |dim: i32, base: f64| -> Vec<f64> {
        (0..dim / 2)
            .map(|index| {
                let exponent = 2.0 * index as f64 / dim as f64;
                1.0 / base.powf(exponent)
            })
            .collect()
    };
    let (freq_t, freq_h, freq_w) = (
        frequencies(dim_t, base_t),
        frequencies(dim_h, base_h),
        frequencies(dim_w, base_w),
    );

    let half = (config.head_dim / 2) as usize;
    let tokens = (rows * columns) as usize;
    let mut cos = Vec::with_capacity(tokens * heads * half);
    let mut sin = Vec::with_capacity(tokens * heads * half);

    // Tokens are in the order patchify laid them: row major. The axes are concatenated t, h, w,
    // and a still image sits at time zero, so the temporal angles are all zero -- present, and
    // contributing a cosine of one, which is not the same as absent.
    for row in 0..rows {
        for column in 0..columns {
            let mut angles = Vec::with_capacity(half);
            angles.extend(freq_t.iter().map(|_| 0.0));
            angles.extend(freq_h.iter().map(|f| row as f64 * f));
            angles.extend(freq_w.iter().map(|f| column as f64 * f));

            for _ in 0..heads {
                for angle in &angles {
                    cos.push(angle.cos() as f32);
                    sin.push(angle.sin() as f32);
                }
            }
        }
    }

    let shape = [rows * columns, config.num_heads, config.head_dim / 2];
    Ok(Placed {
        cos: Tensor::from_f32(&shape, &cos)?.to_device(device)?,
        sin: Tensor::from_f32(&shape, &sin)?.to_device(device)?,
    })
}

/// The sinusoid a timestep becomes before anything is done to it.
///
/// Cosine first and then sine, and the exponent divided by the half width rather than by one less
/// than it. Both are the opposite of the commoner convention, and both are what this was trained
/// with.
fn timestep_sinusoid(timestep: f32, channels: i32, device: Device) -> Result<Tensor> {
    let half = (channels / 2) as usize;
    let mut values = vec![0.0f32; channels as usize];

    for index in 0..half {
        let exponent = -(10000.0f64).ln() * index as f64 / half as f64;
        let angle = timestep as f64 * exponent.exp();
        values[index] = angle.cos() as f32;
        values[half + index] = angle.sin() as f32;
    }

    Ok(Tensor::from_f32(&[1, 1, channels], &values)?.to_device(device)?)
}

/// `norm(x) * (1 + scale) + shift`, written as `norm(x) * scale + norm(x) + shift` so that no
/// scalar has to be added to a tensor.
///
/// `x` arrives in float32 and what comes back is in `narrow`: the residual stream is wide and the
/// sublayers are not. See [`write`].
/// The modulation is done wide and only its result narrowed, which is where the widths have to
/// fall: the shift a deep block applies is large enough that folding it in half precision reaches
/// infinity, and an infinity minus an infinity is the NaN that comes out the far end.
fn modulate(g: &Graph, x: Value, shift: Value, scale: Value, narrow: DType) -> Value {
    let normed = g.layer_norm(x, None, None, super::NORM_EPS);
    let scaled = g.add(g.mul(normed, g.cast(scale, DType::Float)), normed);
    g.cast(g.add(scaled, g.cast(shift, DType::Float)), narrow)
}

/// Add a sublayer's gated output back onto the wide residual stream.
///
/// Both halves are widened before they are multiplied, not after. The reference is an `addcmul`
/// against a float32 stream with the gate cast to match, and the order is not a detail: a gate of
/// a hundred against an output of a thousand is an infinity in half and a perfectly ordinary
/// number in whole.
fn accumulate(g: &Graph, x: Value, out: Value, gate: Value) -> Value {
    let wide = g.mul(g.cast(out, DType::Float), g.cast(gate, DType::Float));
    g.add(x, wide)
}

/// A block's modulation: the small two-layer projection of the timestep plus the shared LoRA,
/// cut into `parts`.
///
/// Each piece comes back flat so that it broadcasts over the `(1, L, D)` it modulates -- what is
/// carried is one timestep, not one per token.
fn modulation(
    config: &DitConfig,
    g: &Graph,
    name: &str,
    emb: Value,
    lora: Value,
    parts: i32,
) -> Vec<Value> {
    let d = config.hidden_size;
    let sub = g.subgraph(name);

    let hidden = Linear::graph(
        &sub.subgraph("1"),
        g.silu(emb),
        d,
        config.adaln_lora_dim,
        false,
    );
    let out = Linear::graph(
        &sub.subgraph("2"),
        hidden,
        config.adaln_lora_dim,
        parts * d,
        false,
    );

    // The final layer takes only as much of the LoRA as it has parts: shift and scale, no gate.
    let carried = g.slice(lora, 2, 0, parts * d);
    let whole = g.view(g.add(out, carried), [Extent::At(parts * d)]);

    (0..parts)
        .map(|index| g.slice(whole, 0, index * d, (index + 1) * d))
        .collect()
}

/// Split a projection into heads and normalize each one.
fn heads(g: &Graph, x: Value, norm: &str, config: &DitConfig, length: Extent) -> Value {
    let viewed = g.view(
        x,
        [
            Extent::At(1),
            length,
            config.num_heads.into(),
            config.head_dim.into(),
        ],
    );
    let weight = g.subgraph(norm).load(Linear::WEIGHT, &[config.head_dim]);
    g.rms_norm(viewed, weight, super::NORM_EPS)
}

/// Rotate the last dimension, pairing its first half against its second.
fn rotate(g: &Graph, x: Value, cos: Value, sin: Value, half: i32) -> Value {
    let first = g.slice(x, 3, 0, half);
    let second = g.slice(x, 3, half, 2 * half);

    let left = g.sub(g.mul(first, cos), g.mul(second, sin));
    let right = g.add(g.mul(first, sin), g.mul(second, cos));
    g.cat(left, right, 3)
}

/// Attention laid out the way flint wants it, and put back the way the stream wants it.
fn attend(g: &Graph, q: Value, k: Value, v: Value, length: Extent, width: i32) -> Value {
    let q = g.contiguous(g.transpose(q, 1, 2));
    let k = g.contiguous(g.transpose(k, 1, 2));
    let v = g.contiguous(g.transpose(v, 1, 2));

    // An image is looked at whole; nothing here is being continued.
    let out = g.attention(q, k, v, false);
    let out = g.contiguous(g.transpose(out, 1, 2));
    g.view(out, [Extent::At(1), length, width.into()])
}

/// One block, written into `g`.
#[allow(clippy::too_many_arguments)]
fn block(
    config: &DitConfig,
    g: &Graph,
    input: Value,
    context: Value,
    emb: Value,
    lora: Value,
    cos: Value,
    sin: Value,
    narrow: DType,
) -> Value {
    let d = config.hidden_size;
    let half = config.head_dim / 2;
    let length = Extent::of(input, 1);
    let context_length = Extent::of(context, 1);

    // -- self attention, over the picture --------------------------------------------------
    let parts = modulation(config, g, "adaln_self_attn", emb, lora, 3);
    let normed = modulate(g, input, parts[0], parts[1], narrow);

    let qkv = Linear::graph(&g.subgraph("self_attn.qkv_proj"), normed, d, 3 * d, false);
    let q = g.contiguous(g.slice(qkv, 2, 0, d));
    let k = g.contiguous(g.slice(qkv, 2, d, 2 * d));
    let v = g.contiguous(g.slice(qkv, 2, 2 * d, 3 * d));

    let sub = g.subgraph("self_attn");
    let q = rotate(g, heads(&sub, q, "q_norm", config, length), cos, sin, half);
    let k = rotate(g, heads(&sub, k, "k_norm", config, length), cos, sin, half);
    let v = g.view(
        v,
        [
            Extent::At(1),
            length,
            config.num_heads.into(),
            config.head_dim.into(),
        ],
    );

    let out = attend(g, q, k, v, length, d);
    let out = Linear::graph(&g.subgraph("self_attn.out_proj"), out, d, d, false);
    let x = accumulate(g, input, out, parts[2]);

    // -- cross attention, into the prompt --------------------------------------------------
    let parts = modulation(config, g, "adaln_cross_attn", emb, lora, 3);
    let normed = modulate(g, x, parts[0], parts[1], narrow);

    let sub = g.subgraph("cross_attn");
    let q = Linear::graph(&g.subgraph("cross_attn.q_proj"), normed, d, d, false);
    let q = heads(&sub, q, "q_norm", config, length);

    // The context is the adapter's width, not this model's, so the key and value read 1024 and
    // produce 2048 apiece.
    let kv = Linear::graph(
        &g.subgraph("cross_attn.kv_proj"),
        context,
        config.context_dim,
        2 * d,
        false,
    );
    let k = g.contiguous(g.slice(kv, 2, 0, d));
    let v = g.contiguous(g.slice(kv, 2, d, 2 * d));
    let k = heads(&sub, k, "k_norm", config, context_length);
    let v = g.view(
        v,
        [
            Extent::At(1),
            context_length,
            config.num_heads.into(),
            config.head_dim.into(),
        ],
    );

    // Not rotated, unlike everything else here.
    let out = attend(g, q, k, v, length, d);
    let out = Linear::graph(&g.subgraph("cross_attn.out_proj"), out, d, d, false);
    let x = accumulate(g, x, out, parts[2]);

    // -- feed forward, ungated and without biases -------------------------------------------
    let parts = modulation(config, g, "adaln_mlp", emb, lora, 3);
    let normed = modulate(g, x, parts[0], parts[1], narrow);

    let hidden = Linear::graph(&g.subgraph("mlp.layer1"), normed, d, config.mlp_size, false);
    let hidden = g.gelu(hidden);
    let down = Linear::graph(&g.subgraph("mlp.layer2"), hidden, config.mlp_size, d, false);

    accumulate(g, x, down, parts[2])
}

/// The whole denoiser, written into `g`. Patches in, patches out.
fn write(config: &DitConfig, float_type: DType, g: &Graph) {
    let tokens = g.input("tokens");
    let context = g.input("context");
    let sinusoid = g.cast(g.input("sinusoid"), float_type);
    let cos = g.cast(g.input("rope_cos"), float_type);
    let sin = g.cast(g.input("rope_sin"), float_type);

    let d = config.hidden_size;
    // The residual stream runs in float32 whatever the sublayers run in. This model's residuals
    // are large enough that in half they reach infinity part way through the first block and every
    // step after it is NaN -- which is what ComfyUI means by keeping the stream wide and running
    // the attention and the feed forward narrow.
    let mut x = g.cast(
        Linear::graph(
            &g.subgraph("x_embedder"),
            tokens,
            config.patchify_channels,
            d,
            false,
        ),
        DType::Float,
    );

    // The two linears here produce the LoRA and nothing else; what the blocks modulate on is the
    // sinusoid that went into them, normalized. Reading `emb` as the projection is the obvious
    // mistake and it is the wrong one.
    let projected = Linear::graph(&g.subgraph("t_embedder.1"), sinusoid, d, d, false);
    let lora = Linear::graph(
        &g.subgraph("t_embedder.2"),
        g.silu(projected),
        d,
        3 * d,
        false,
    );
    let emb = {
        let weight = g.subgraph("t_norm").load(Linear::WEIGHT, &[d]);
        g.rms_norm(sinusoid, weight, super::NORM_EPS)
    };

    for index in 0..config.num_blocks {
        x = block(
            config,
            &g.subgraph(&format!("block{index}")),
            x,
            context,
            emb,
            lora,
            cos,
            sin,
            float_type,
        );
    }

    let final_g = g.subgraph("final");
    let parts = modulation(config, &final_g, "adaln", emb, lora, 2);
    let x = modulate(g, x, parts[0], parts[1], float_type);
    let out = config.latent_channels * config.patch_size * config.patch_size;

    g.output(
        "velocity",
        Linear::graph(&final_g.subgraph("linear"), x, d, out, false),
    );
}

pub struct Dit {
    config: DitConfig,
    ir: Ir,
    weights: Rc<dyn ParamSource>,
    float_type: DType,
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

        Ok(Dit {
            weights: Rc::clone(weights),
            ir,
            config,
            float_type,
        })
    }

    pub fn config(&self) -> &DitConfig {
        &self.config
    }

    pub fn ir(&self) -> &Ir {
        &self.ir
    }

    /// Cut a latent into the tokens the graph reads.
    ///
    /// `(1, C, H, W)` becomes `(1, H / p * W / p, (C + 1) * p * p)`, channel first within a patch.
    /// The extra channel is the padding mask Cosmos concatenates: zeros for everything drawn here,
    /// and read all the same.
    /// The batch is left out of the shuffle throughout: it is always one here, and the copy
    /// behind `contiguous` is written for ranks up to five. With the batch it would be six.
    fn patchify(&self, latent: &Tensor) -> Result<Tensor> {
        let patch = self.config.patch_size;
        let (height, width) = (latent.shape_at(2)?, latent.shape_at(3)?);
        let (rows, columns) = (height / patch, width / patch);

        let mask = Tensor::zeros(&[1, 1, height, width], latent.dtype(), latent.device())?;
        let x = F::cat(latent, &mask, 1)?;

        let channels = self.config.latent_channels + 1;
        let x = x.view(&[channels, rows, patch, columns, patch])?;

        // (c, rows, ph, columns, pw) -> (rows, columns, c, ph, pw)
        let x = x.transpose(0, 1)?.transpose(1, 3)?.transpose(2, 3)?;
        Ok(x.contiguous()?
            .view(&[1, rows * columns, self.config.patchify_channels])?)
    }

    /// Put the tokens back into a latent, in the order the final layer wrote them.
    ///
    /// Not the inverse of [`Dit::patchify`]: the channel trails here where it led there.
    fn unpatchify(&self, tokens: &Tensor, height: i32, width: i32) -> Result<Tensor> {
        let patch = self.config.patch_size;
        let channels = self.config.latent_channels;
        let (rows, columns) = (height / patch, width / patch);

        let x = tokens.view(&[rows, columns, patch, patch, channels])?;

        // (rows, columns, p1, p2, c) -> (c, rows, p1, columns, p2). The channel trails going in
        // and leads coming out, which is the whole of the difference between this and patchify.
        let x = x.transpose(0, 4)?.transpose(1, 4)?.transpose(3, 4)?;
        Ok(x.contiguous()?.view(&[1, channels, height, width])?)
    }

    /// One step's velocity for `latent` at `timestep`, conditioned on `context`.
    ///
    /// `latent` is `(1, C, H, W)` with both sides a multiple of the patch size; `timestep` is the
    /// sigma itself, in `0..=1`, because Anima's sampling multiplier is one rather than the
    /// thousand a flow model usually carries. `context` is `(1, 512, D)` from the adapter.
    pub fn forward(&self, latent: &Tensor, timestep: f32, context: &Tensor) -> Result<Tensor> {
        let dim = latent.dim()?;
        if dim != 4 {
            return Err(Error::model(format!(
                "the denoiser takes a latent as (1, C, H, W), got a {dim}-D tensor"
            )));
        }

        let (height, width) = (latent.shape_at(2)?, latent.shape_at(3)?);
        let patch = self.config.patch_size;
        if height % patch != 0 || width % patch != 0 {
            return Err(Error::model(format!(
                "a {height} by {width} latent does not divide into {patch} by {patch} patches"
            )));
        }

        let device = latent.device();
        let tokens = self.patchify(latent)?;
        let sinusoid = timestep_sinusoid(timestep, self.config.hidden_size, device)?;
        let placed = rope_table(&self.config, height / patch, width / patch, device)?;

        let run = RunContext::new(&*self.weights)
            .input("tokens", &tokens)
            .input("context", context)
            .input("sinusoid", &sinusoid)
            .input("rope_cos", &placed.cos)
            .input("rope_sin", &placed.sin);

        let outputs = self.ir.run(&run)?;

        let velocity = outputs
            .iter()
            .find(|(name, _)| name == "velocity")
            .map(|(_, tensor)| tensor.clone())
            .ok_or_else(|| Error::model("the denoiser produced no velocity"))?;

        self.unpatchify(&velocity, height, width)
    }

    pub fn float_type(&self) -> DType {
        self.float_type
    }
}
