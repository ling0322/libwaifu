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

//! The denoiser: one sequence, the prompt in front of the picture, thirty-two blocks deep and
//! four thousand wide.
//!
//! A block is a self attention and a SwiGLU, each behind an affine-free LayerNorm scaled by
//! `1 + scale` and gated by `tanh(gate)`. The scales and gates are not the block's: one projection
//! of the timestep makes all four for the whole model.
//!
//! # What is easy to get wrong here
//!
//! *The attention is block causal.* The prompt attends to itself causally and never to the
//! picture; the picture attends to all of the prompt and all of itself. So each block runs two
//! attentions rather than one masked one -- the prompt's queries over the prompt's keys with a
//! causal mask, and the picture's over everything with none -- which is the arithmetic
//! diffusers' own mask-free processor does.
//!
//! *The prompt is modulated at t = 0.* `causal_condition` gives every prompt token the modulation
//! of timestep zero, and only the picture's tokens that of the step. So the two are kept as two
//! residual streams, each modulated by its own row, and put side by side only for the
//! projections. A consequence is that the prompt's half of every block is the same at every step
//! -- the reference caches its keys and values after the first step. This recomputes them, which
//! is a few percent of the work at 1024 by 1024.
//!
//! *There is no shift.* The modulation is a scale and a gate, twice: `(1 + scale) * norm(x)`, and
//! nothing added. The last layer's is a scale alone.
//!
//! *The rotary pairing is adjacent* and the positions are three dimensional: the prompt's tokens
//! advance along all three axes together, and the picture sits at the frame after the prompt with
//! its rows and columns centred on zero.
//!
//! *The GELU in the prompt's projection is the tanh approximation*, and the timestep's two
//! linears have a swish between them, not a GELU.

use std::cell::RefCell;
use std::fmt;
use std::rc::Rc;

use super::config::{ControlConfig, DitConfig};
use crate::error::{Error, Result};
use crate::flint::{
    check_parameters, DType, Device, Extent, Graph, Ir, ParamSource, RunContext, Tensor, Value,
};
use crate::krea2::dit::{gelu_tanh, rotate, timestep_sinusoid};
use crate::layers::Linear;

/// The rotary table for one sequence, built on the host because it depends on the prompt's
/// length and the picture's size.
struct Placed {
    cos: Tensor,
    sin: Tensor,
    key: (i32, i32, i32),
}

/// Where each token sits, as `QwenImage21Rope` places it.
///
/// The prompt's `n`th token is at `(n, n, n)`. The picture is one block at frame `L` -- the
/// position the prompt reached -- with its rows at `-(H - H / 2) .. H / 2` and its columns the
/// same, so that where a picture's tokens are does not depend on how long the prompt was.
fn rope_table(
    config: &DitConfig,
    text_length: i32,
    rows: i32,
    columns: i32,
    device: Device,
) -> Result<Placed> {
    let (dim_t, dim_h, dim_w) = config.rope_axes;
    let heads = config.num_heads as usize;
    let theta = config.rope_theta as f64;

    let frequencies = |dim: i32| -> Vec<f64> {
        (0..dim / 2)
            .map(|index| 1.0 / theta.powf(2.0 * index as f64 / dim as f64))
            .collect()
    };
    let (freq_t, freq_h, freq_w) = (frequencies(dim_t), frequencies(dim_h), frequencies(dim_w));

    let pairs = config.rope_pairs() as usize;
    let tokens = (text_length + rows * columns) as usize;
    let mut cos = Vec::with_capacity(tokens * heads * pairs);
    let mut sin = Vec::with_capacity(tokens * heads * pairs);

    let mut place = |frame: i32, row: i32, column: i32| {
        let angles = freq_t
            .iter()
            .map(|f| frame as f64 * f)
            .chain(freq_h.iter().map(|f| row as f64 * f))
            .chain(freq_w.iter().map(|f| column as f64 * f))
            .collect::<Vec<_>>();

        for _ in 0..heads {
            for angle in &angles {
                cos.push(angle.cos() as f32);
                sin.push(angle.sin() as f32);
            }
        }
    };

    for position in 0..text_length {
        place(position, position, position);
    }
    for row in -(rows - rows / 2)..rows / 2 {
        for column in -(columns - columns / 2)..columns / 2 {
            place(text_length, row, column);
        }
    }

    let shape = [tokens as i32, config.num_heads, config.rope_pairs(), 1];
    Ok(Placed {
        cos: Tensor::from_f32(&shape, &cos)?.to_device(device)?,
        sin: Tensor::from_f32(&shape, &sin)?.to_device(device)?,
        key: (text_length, rows, columns),
    })
}

/// `(1 + scale) * layer_norm(x)`, off the wide stream and onto the narrow one.
fn modulate(g: &Graph, x: Value, eps: f32, scale: Value, narrow: DType) -> Value {
    let normed = g.layer_norm(x, None, None, eps);
    let scaled = g.add(g.mul(normed, g.cast(scale, DType::Float)), normed);
    g.cast(scaled, narrow)
}

/// Add a sublayer's output back onto the wide stream, through `tanh(gate)`.
fn accumulate(g: &Graph, x: Value, out: Value, gate: Value) -> Value {
    let gate = g.tanh(g.cast(gate, DType::Float));
    g.add(x, g.mul(g.cast(out, DType::Float), gate))
}

/// The four pieces of one timestep's modulation: attention scale and gate, feed forward scale
/// and gate, each flat so that it broadcasts over a `(1, L, D)`.
fn pieces(g: &Graph, modulation: Value, width: i32) -> [Value; 4] {
    std::array::from_fn(|index| {
        g.slice(
            modulation,
            0,
            index as i32 * width,
            (index as i32 + 1) * width,
        )
    })
}

/// Split a projection into heads and normalize each one.
fn heads(g: &Graph, x: Value, norm: &str, count: i32, head_dim: i32, eps: f32) -> Value {
    let length = Extent::of(x, 1);
    let viewed = g.view(x, [Extent::At(1), length, count.into(), head_dim.into()]);
    let weight = g.subgraph(norm).load(Linear::WEIGHT, &[head_dim]);
    g.rms_norm(viewed, weight, eps)
}

/// One block over both streams. `text` is `(1, L, D)` and `image` `(1, N, D)`, both float32;
/// `now` and `zero` are the step's modulation and timestep zero's.
#[allow(clippy::too_many_arguments)]
fn block(
    config: &DitConfig,
    g: &Graph,
    text: Value,
    image: Value,
    now: Value,
    zero: Value,
    cos: Value,
    sin: Value,
    narrow: DType,
) -> (Value, Value) {
    let d = config.hidden_size;
    let eps = config.norm_eps;
    let [scale1, gate1, scale2, gate2] = pieces(g, now, d);
    let [text_scale1, text_gate1, text_scale2, text_gate2] = pieces(g, zero, d);
    let split = Extent::of(text, 1);

    // The attention. Both streams go through one projection, side by side.
    let joined = g.cat(
        modulate(g, text, eps, text_scale1, narrow),
        modulate(g, image, eps, scale1, narrow),
        1,
    );
    let qkv = Linear::graph(&g.subgraph("attn.qkv_proj"), joined, d, 3 * d, false);
    let q = g.contiguous(g.slice(qkv, 2, 0, d));
    let k = g.contiguous(g.slice(qkv, 2, d, 2 * d));
    let v = g.contiguous(g.slice(qkv, 2, 2 * d, 3 * d));

    let (count, head_dim) = (config.num_heads, config.head_dim);
    let q = heads(g, q, "attn.q_norm", count, head_dim, eps);
    let k = heads(g, k, "attn.k_norm", count, head_dim, eps);
    let length = Extent::of(v, 1);
    let v = g.view(v, [Extent::At(1), length, count.into(), head_dim.into()]);

    let pairs = config.rope_pairs();
    let q = g.contiguous(g.transpose(rotate(g, q, cos, sin, pairs), 1, 2));
    let k = g.contiguous(g.transpose(rotate(g, k, cos, sin, pairs), 1, 2));
    let v = g.contiguous(g.transpose(v, 1, 2));

    // The prompt reads the prompt, causally; the picture reads everything.
    let prompt = g.attention(
        g.contiguous(g.slice(q, 2, 0, split)),
        g.contiguous(g.slice(k, 2, 0, split)),
        g.contiguous(g.slice(v, 2, 0, split)),
        true,
    );
    let picture = g.attention(g.contiguous(g.slice(q, 2, split, Extent::End)), k, v, false);
    let out = g.contiguous(g.transpose(g.cat(prompt, picture, 2), 1, 2));
    let out = g.view(out, [Extent::At(1), length, d.into()]);
    let out = Linear::graph(&g.subgraph("attn.out_proj"), out, d, d, false);

    let text = accumulate(g, text, g.contiguous(g.slice(out, 1, 0, split)), text_gate1);
    let image = accumulate(
        g,
        image,
        g.contiguous(g.slice(out, 1, split, Extent::End)),
        gate1,
    );

    // The feed forward, the same way.
    let joined = g.cat(
        modulate(g, text, eps, text_scale2, narrow),
        modulate(g, image, eps, scale2, narrow),
        1,
    );
    let gated = Linear::graph(
        &g.subgraph("ff.gate_up_proj"),
        joined,
        d,
        2 * config.mlp_size,
        false,
    );
    let out = Linear::graph(
        &g.subgraph("ff.down"),
        g.swiglu(gated),
        config.mlp_size,
        d,
        false,
    );

    let text = accumulate(g, text, g.contiguous(g.slice(out, 1, 0, split)), text_gate2);
    let image = accumulate(
        g,
        image,
        g.contiguous(g.slice(out, 1, split, Extent::End)),
        gate2,
    );
    (text, image)
}

/// The ControlNet's chain, which runs to its end before the denoiser's first block does.
///
/// `text` and `image` are the denoiser's two streams where they enter its first block; the chain
/// reads them once, there, and never again. What comes back is one skip per block of the chain,
/// `(1, L + N, D)` in the narrow type, prompt first, for [`write`] to add after the blocks
/// `control.layers` names.
///
/// The chain is a stream of its own shaped like the denoiser's: the conditioning is projected
/// into the picture's positions, the prompt's positions are left at zero, and the first block's
/// `before_proj` adds the denoiser's input on top. A zero through `before_proj` is its bias, so
/// the prompt's half of the chain starts as the prompt plus that bias.
#[allow(clippy::too_many_arguments)]
fn write_control(
    config: &DitConfig,
    control: &ControlConfig,
    g: &Graph,
    text: Value,
    image: Value,
    now: Value,
    zero: Value,
    cos: Value,
    sin: Value,
    narrow: DType,
) -> Vec<Value> {
    let d = config.hidden_size;
    let conditioning = g.cast(g.input("control"), narrow);

    // Not a `Linear`: it reads 129 channels, which an FP8 multiply cannot, and it is small
    // enough that every package keeps it at full width.
    let projected = {
        let sub = g.subgraph("control.img_in");
        let weight = sub.load(Linear::WEIGHT, &[d, control.in_channels]);
        let bias = sub.load(Linear::BIAS, &[d]);
        g.add(g.matmul(conditioning, g.transpose(weight, 0, 1)), bias)
    };

    let before = g.subgraph("control.block0.before_proj");
    let mut image = g.add(
        g.cast(Linear::graph(&before, projected, d, d, true), DType::Float),
        image,
    );
    let mut text = g.add(
        text,
        g.cast(before.load(Linear::BIAS, &[d]), DType::Float),
    );

    let mut skips = Vec::with_capacity(control.layers.len());
    for index in 0..control.layers.len() {
        let sub = g.subgraph(&format!("control.block{index}"));
        (text, image) = block(config, &sub, text, image, now, zero, cos, sin, narrow);

        let joined = g.cat(g.cast(text, narrow), g.cast(image, narrow), 1);
        skips.push(Linear::graph(&sub.subgraph("after_proj"), joined, d, d, true));
    }
    skips
}

/// Add one of the control chain's skips onto both streams, `scale` times over.
fn add_skip(g: &Graph, text: Value, image: Value, skip: Value, scale: Value) -> (Value, Value) {
    let split = Extent::of(text, 1);
    let part = |begin: Extent, end: Extent| {
        g.mul(
            g.cast(g.contiguous(g.slice(skip, 1, begin, end)), DType::Float),
            scale,
        )
    };
    (
        g.add(text, part(Extent::At(0), split)),
        g.add(image, part(split, Extent::End)),
    )
}

/// The whole denoiser, written into `g`; with `control`, the control chain beside it.
fn write(config: &DitConfig, control: Option<&ControlConfig>, float_type: DType, g: &Graph) {
    let tokens = g.cast(g.input("tokens"), float_type);
    let context = g.input("context");
    let sinusoid = g.cast(g.input("sinusoid"), float_type);
    let cos = g.cast(g.input("rope_cos"), float_type);
    let sin = g.cast(g.input("rope_sin"), float_type);

    let d = config.hidden_size;

    // Two timesteps at once: row zero is the step's and row one is zero's, for the prompt.
    let emb = {
        let sub = g.subgraph("time_embed");
        let hidden = Linear::graph(
            &sub.subgraph("linear_1"),
            sinusoid,
            config.timestep_embed_dim,
            d,
            false,
        );
        Linear::graph(&sub.subgraph("linear_2"), g.silu(hidden), d, d, false)
    };
    let modulation = Linear::graph(&g.subgraph("modulation"), g.silu(emb), d, 4 * d, false);
    let row = |value: Value, index: i32, width: i32| {
        g.view(
            g.contiguous(g.slice(value, 1, index, index + 1)),
            [Extent::At(width)],
        )
    };
    let now = row(modulation, 0, 4 * d);
    let zero = row(modulation, 1, 4 * d);

    // The prompt's way in: a zero-centred RMSNorm, folded by the exporter, computed wide.
    let text = {
        let sub = g.subgraph("txt_in");
        let weight = g.cast(
            sub.subgraph("norm")
                .load(Linear::WEIGHT, &[config.context_size]),
            DType::Float,
        );
        let normed = g.cast(
            g.rms_norm(g.cast(context, DType::Float), weight, config.norm_eps),
            float_type,
        );
        let hidden = Linear::graph(
            &sub.subgraph("linear_1"),
            normed,
            config.context_size,
            d,
            false,
        );
        Linear::graph(&sub.subgraph("linear_2"), gelu_tanh(g, hidden), d, d, false)
    };
    let image = Linear::graph(
        &g.subgraph("img_in"),
        tokens,
        config.latent_channels,
        d,
        false,
    );

    // Two residual streams in float32, whatever the sublayers run in.
    let mut text = g.cast(text, DType::Float);
    let mut image = g.cast(image, DType::Float);

    // The skips and the scale they are added at: a `(D,)` of one value, since an element-wise
    // operand broadcasts over leading dimensions and not from a single element.
    let skips = control.map(|control| {
        let skips = write_control(
            config, control, g, text, image, now, zero, cos, sin, float_type,
        );
        (control, skips, g.input("control_scale"))
    });

    for index in 0..config.num_blocks {
        (text, image) = block(
            config,
            &g.subgraph(&format!("block{index}")),
            text,
            image,
            now,
            zero,
            cos,
            sin,
            float_type,
        );

        if let Some((control, skips, scale)) = &skips {
            if let Some(at) = control.layers.iter().position(|&layer| layer == index) {
                (text, image) = add_skip(g, text, image, skips[at], *scale);
            }
        }
    }

    // The last layer reads only the picture: the prompt's rows would be thrown away. A scale and
    // no shift, off the step's own embedding.
    let final_g = g.subgraph("final");
    let step = g.contiguous(g.slice(emb, 1, 0, 1));
    let scale = Linear::graph(&final_g.subgraph("linear"), g.silu(step), d, d, false);
    let scale = g.view(scale, [Extent::At(d)]);
    let x = modulate(&final_g, image, config.norm_eps, scale, float_type);
    g.output(
        "velocity",
        Linear::graph(
            &final_g.subgraph("proj"),
            x,
            d,
            config.latent_channels,
            false,
        ),
    );
}

pub struct Dit {
    config: DitConfig,
    ir: Ir,
    /// The same denoiser with the control chain beside it, when a control package was read.
    controlled: Option<(ControlConfig, Ir)>,
    weights: Rc<dyn ParamSource>,
    float_type: DType,
    placed: RefCell<Option<Placed>>,
}

impl fmt::Debug for Dit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Dit({} instructions", self.ir.len())?;
        if let Some((_, controlled)) = &self.controlled {
            write!(f, ", {} with control", controlled.len())?;
        }
        write!(f, ")")
    }
}

fn compile(
    config: &DitConfig,
    control: Option<&ControlConfig>,
    name: &str,
    weights: &dyn ParamSource,
    float_type: DType,
) -> Result<Ir> {
    let graph = Graph::with_weights(config.weight_format);
    write(config, control, float_type, &graph.subgraph(name));
    check_parameters(&graph, weights)?;
    Ok(Ir::compile(&graph))
}

impl Dit {
    pub fn build(
        config: DitConfig,
        name: &str,
        weights: &Rc<dyn ParamSource>,
        float_type: DType,
    ) -> Result<Dit> {
        Self::build_with_control(config, None, name, weights, float_type)
    }

    /// The denoiser, and with `control` a second graph of it that runs the control chain too,
    /// whose weights `weights` has to hold under `{name}.control`.
    pub fn build_with_control(
        config: DitConfig,
        control: Option<ControlConfig>,
        name: &str,
        weights: &Rc<dyn ParamSource>,
        float_type: DType,
    ) -> Result<Dit> {
        let ir = compile(&config, None, name, weights.as_ref(), float_type)?;
        let controlled = match control {
            Some(control) => {
                let ir = compile(&config, Some(&control), name, weights.as_ref(), float_type)?;
                Some((control, ir))
            }
            None => None,
        };

        Ok(Dit {
            weights: Rc::clone(weights),
            ir,
            controlled,
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

    /// The control chain's configuration, when one was read.
    pub fn control(&self) -> Option<&ControlConfig> {
        self.controlled.as_ref().map(|(control, _)| control)
    }

    pub fn float_type(&self) -> DType {
        self.float_type
    }

    /// One step's velocity for `latent` at `timestep`, conditioned on `context`.
    ///
    /// `latent` is `(1, C, H, W)`; `timestep` is what the model is handed, in `0..=1`, which the
    /// embedding multiplies by a thousand on its own; `context` is `(1, L, D)` from the encoder,
    /// the system turn already dropped.
    pub fn forward(&self, latent: &Tensor, timestep: f32, context: &Tensor) -> Result<Tensor> {
        self.run(&self.ir, latent, timestep, context, None)
    }

    /// [`Dit::forward`] steered by the control chain.
    ///
    /// `conditioning` is `(1, 129, H, W)` on the latent's grid: the control picture's latent, the
    /// inpainting mask (one where the picture is kept) and the kept picture's latent, the latents
    /// normalized the way the sampler's are. `scale` multiplies every skip; at zero the answer is
    /// the plain denoiser's.
    pub fn forward_controlled(
        &self,
        latent: &Tensor,
        timestep: f32,
        context: &Tensor,
        conditioning: &Tensor,
        scale: f32,
    ) -> Result<Tensor> {
        let Some((control, ir)) = &self.controlled else {
            return Err(Error::model(
                "this denoiser was built without a control package, so it has no control chain",
            ));
        };

        let (rows, columns) = (latent.shape_at(2)?, latent.shape_at(3)?);
        let expected = [1, control.in_channels, rows, columns];
        let found = (0..4)
            .map(|dim| conditioning.shape_at(dim))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        if conditioning.dim()? != 4 || found != expected {
            return Err(Error::model(format!(
                "the control conditioning is {found:?}, and this latent needs {expected:?}"
            )));
        }

        let device = latent.device();
        let tokens = conditioning
            .view(&[1, control.in_channels, rows * columns])?
            .transpose(1, 2)?
            .contiguous()?
            .to_device(device)?;

        let width = self.config.hidden_size;
        let scale = Tensor::from_f32(&[width], &vec![scale; width as usize])?.to_device(device)?;

        self.run(ir, latent, timestep, context, Some((&tokens, &scale)))
    }

    fn run(
        &self,
        ir: &Ir,
        latent: &Tensor,
        timestep: f32,
        context: &Tensor,
        control: Option<(&Tensor, &Tensor)>,
    ) -> Result<Tensor> {
        if latent.dim()? != 4 {
            return Err(Error::model(format!(
                "the denoiser takes a latent as (1, C, H, W), got a {}-D tensor",
                latent.dim()?
            )));
        }
        if context.dim()? != 3 {
            return Err(Error::model(format!(
                "the denoiser takes a prompt as (1, L, D), got a {}-D tensor",
                context.dim()?
            )));
        }

        let channels = self.config.latent_channels;
        let (rows, columns) = (latent.shape_at(2)?, latent.shape_at(3)?);
        let device = latent.device();
        let text_length = context.shape_at(1)?;

        // Unpatched: one latent pixel is one token, its channels the features.
        let tokens = latent
            .view(&[1, channels, rows * columns])?
            .transpose(1, 2)?
            .contiguous()?;

        let step = timestep_sinusoid(timestep, self.config.timestep_embed_dim, device)?;
        let zero = timestep_sinusoid(0.0, self.config.timestep_embed_dim, device)?;
        let sinusoid = crate::flint::functional::cat(&step, &zero, 1)?;

        let mut placed = self.placed.borrow_mut();
        if placed.as_ref().map(|table| table.key) != Some((text_length, rows, columns)) {
            *placed = Some(rope_table(
                &self.config,
                text_length,
                rows,
                columns,
                device,
            )?);
        }
        let table = placed.as_ref().expect("it was just built");

        let mut run = RunContext::new(&*self.weights)
            .input("tokens", &tokens)
            .input("context", context)
            .input("sinusoid", &sinusoid)
            .input("rope_cos", &table.cos)
            .input("rope_sin", &table.sin);
        if let Some((conditioning, scale)) = control {
            run = run
                .input("control", conditioning)
                .input("control_scale", scale);
        }

        let outputs = ir.run(&run)?;
        let velocity = outputs
            .iter()
            .find(|(name, _)| name == "velocity")
            .map(|(_, tensor)| tensor.clone())
            .ok_or_else(|| Error::model("the denoiser produced no velocity"))?;

        Ok(velocity
            .transpose(1, 2)?
            .contiguous()?
            .view(&[1, channels, rows, columns])?)
    }
}
