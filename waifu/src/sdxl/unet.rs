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

//! The U-Net, which is where all of the work of generating an image happens.
//!
//! It is asked the same question at every step: given this noisy latent, at this much noise, and
//! this prompt, what is the noise in it? Its shape is the name -- the latent is halved twice on
//! the way down, carried across at every resolution, and doubled twice on the way back up -- and
//! the conditioning enters through cross attention in the transformer blocks that sit between the
//! residual blocks at the two smaller resolutions.
//!
//! SDXL is the widest of these: 2.6 billion parameters, ten transformer layers deep at its
//! smallest resolution, and conditioned on the two text encoders side by side.
//!
//! # Written down rather than run
//!
//! Like [`text_encoder`](super::text_encoder), this is a [`Graph`]: [`Unet::build`] writes the
//! whole pass down, compiles it into an [`Ir`] and reads the weights that graph turns out to ask
//! for, and [`Unet::forward`] hands the latent to the `Ir` and takes back what it named. It is
//! the model this buys the most for -- 2.6 billion parameters is most of the package, and a
//! sampler walks the whole of it twenty times or more for one picture, so the pass is written
//! once and the walking is what repeats.
//!
//! The two things a graph cannot say both show up here, and differently from how they did in the
//! text encoder:
//!
//! *A graph holds no shapes.* The latent's height and width are read off the value with
//! [`Extent`], never written down, which is what lets one compiled pass answer at any size. Where
//! a transformer reads an image as a sequence it needs `H * W` rather than either on its own, and
//! that is [`Extent::prod`] -- the one piece of arithmetic an extent does.
//!
//! *A graph holds no branches, and no formulas it has no operator for.* The sinusoidal embeddings
//! of the timestep and of the six numbers about the image are sines and cosines, which nothing
//! here computes. They stay on the host, where [`timestep_embedding`] has always computed them in
//! double precision, and arrive as inputs already widened to a row per latent.

use std::fmt;
use std::rc::Rc;

use crate::error::{Error, Result};
use crate::flint::{
    check_parameters, functional as F, Extent, Graph, Ir, ParamSource, RunContext, Tensor, Value,
    WeightFormat,
};
use crate::layers::{Conv2d, GroupNorm, LayerNorm, Linear};

/// The largest period the sinusoidal timestep embedding uses, which is what everything derived
/// from the original DDPM code has used since.
const MAX_PERIOD: f64 = 10000.0;

/// What the package records about the U-Net.
#[derive(Clone, Debug)]
pub struct UnetConfig {
    pub latent_channels: i32,
    /// The width at each resolution, narrowest first. The first entry is also the width of the
    /// timestep embedding before it is projected.
    pub block_out_channels: Vec<i32>,
    /// Residual blocks per resolution on the way down. The way up runs one more, to consume the
    /// residual the downsampling left behind.
    pub layers_per_block: i32,
    /// Transformer layers inside each attention, per resolution. Zero means that resolution has
    /// no attention at all, which is where SDXL saves the most: at full size it is convolutions
    /// only.
    pub transformer_layers_per_block: Vec<i32>,
    /// Attention heads per resolution. The head is what is left over: 64 wide at every one.
    pub num_heads: Vec<i32>,
    pub norm_num_groups: i32,
    /// How wide the conditioning is, which for SDXL is the two text encoders concatenated.
    pub cross_attention_dim: i32,
    /// How wide each of the six numbers describing the image size is embedded.
    pub addition_time_embed_dim: i32,
    /// What the added embedding reads: the pooled text vector and those six numbers embedded.
    pub projection_class_embeddings_input_dim: i32,
    /// How the package stored the matrices this multiplies by, which decides what its projections
    /// are built out of. See [`WeightFormat`].
    pub weight_format: WeightFormat,
}

impl UnetConfig {
    /// The width of the timestep embedding once it has been projected, which is what every
    /// residual block is handed.
    fn time_embed_dim(&self) -> i32 {
        self.block_out_channels[0] * 4
    }

    fn levels(&self) -> usize {
        self.block_out_channels.len()
    }

    /// How wide the pooled text vector is, which is what is left of the added embedding's input
    /// once the six numbers about the image have taken their share.
    ///
    /// The eager pass never had to know this -- it reshaped the pooled vector to whatever it
    /// turned out to be -- and the graph does, since a shape it writes down is a shape it checks.
    fn pooled_dim(&self) -> i32 {
        self.projection_class_embeddings_input_dim - 6 * self.addition_time_embed_dim
    }
}

/// The sinusoidal embedding of one timestep, as `(1, dim)` on the host.
///
/// Computed here rather than read from the checkpoint, since it is a formula. In double precision
/// on the CPU, which costs nothing at this size and keeps the frequencies exact -- the highest of
/// them is a ten-thousandth, and rounding it early moves the embedding by more than the model's
/// own precision does.
fn timestep_embedding(timestep: f64, dim: i32) -> Result<Tensor> {
    if dim <= 0 || dim % 2 != 0 {
        return Err(Error::model(format!(
            "a sinusoidal embedding needs an even width, got {dim}"
        )));
    }

    let half = (dim / 2) as usize;
    let mut values = vec![0.0f32; dim as usize];
    for index in 0..half {
        let frequency = (-MAX_PERIOD.ln() * index as f64 / half as f64).exp();
        let angle = timestep * frequency;

        // Cosine first: diffusers calls this flip_sin_to_cos, and every SDXL checkpoint was
        // trained with it on.
        values[index] = angle.cos() as f32;
        values[half + index] = angle.sin() as f32;
    }

    Ok(Tensor::from_f32(&[1, dim], &values)?)
}

/// The other embedding SDXL adds: the size the image was asked for, and where it was cropped
/// from, six numbers in all, each embedded the way a timestep is and laid end to end.
fn time_ids_embedding(time_ids: &[f32], dim: i32) -> Result<Tensor> {
    let mut embedded = Vec::new();
    for id in time_ids {
        embedded.push(timestep_embedding(*id as f64, dim)?);
    }

    let mut out = embedded[0].clone();
    for part in &embedded[1..] {
        out = F::cat(&out, part, -1)?;
    }

    Ok(out)
}

/// `rows` copies of a one-row tensor, stacked.
///
/// The timestep and the numbers about the image are the same for every latent in a batch -- only
/// the prompt differs -- so they are built once and handed a row to each. The eager pass repeated
/// them after projecting them; this repeats before, which is the same answer a row at a time and
/// is what lets the graph hold no batch size of its own.
fn repeat_rows(row: &Tensor, rows: i32) -> Result<Tensor> {
    let mut out = row.clone();
    for _ in 1..rows {
        out = F::cat(&out, row, 0)?;
    }
    Ok(out)
}

/// What every block of the U-Net is handed beside its own input.
///
/// Two values that reach every level unchanged: how noisy the latent is, which is added, and what
/// the prompt said, which is attended over. Carried together because nothing that takes one
/// without the other exists here.
#[derive(Clone, Copy)]
struct Conditioning {
    temb: Value,
    context: Value,
}

/// Two convolutions, a normalization before each, and the timestep added in between.
///
/// The timestep is how the block is told how much noise it is looking at. It arrives as one
/// vector for the whole image and is added to every pixel, which is the cheapest way a
/// convolution can be conditioned on something that has no position.
fn resnet_block(
    g: &Graph,
    config: &UnetConfig,
    input: Value,
    temb: Value,
    in_channels: i32,
    out_channels: i32,
) -> Value {
    let groups = config.norm_num_groups;

    let x = GroupNorm::graph(&g.subgraph("norm1"), input, in_channels, groups, 1e-5);
    let x = g.silu(x);
    let x = Conv2d::graph(&g.subgraph("conv1"), x, in_channels, out_channels, 3, 1, 1);

    // The activation comes before the projection here, not after, which is what the reference
    // does and what the weights were trained for.
    let t = Linear::graph(
        &g.subgraph("time_proj"),
        g.silu(temb),
        config.time_embed_dim(),
        out_channels,
        true,
    );
    let t = g.view(
        t,
        [
            Extent::of(t, 0),
            Extent::At(out_channels),
            Extent::At(1),
            Extent::At(1),
        ],
    );
    let x = g.add(x, t);

    let x = GroupNorm::graph(&g.subgraph("norm2"), x, out_channels, groups, 1e-5);
    let x = g.silu(x);
    let x = Conv2d::graph(&g.subgraph("conv2"), x, out_channels, out_channels, 3, 1, 1);

    let residual = match in_channels == out_channels {
        true => input,
        false => Conv2d::graph(
            &g.subgraph("shortcut"),
            input,
            in_channels,
            out_channels,
            1,
            1,
            0,
        ),
    };

    g.add(residual, x)
}

/// `(N, L, D)` to the `(N, H, L, Dh)` an attention takes.
///
/// Both lengths are read off the value rather than passed in: in cross attention the keys and
/// values are as long as the prompt and the queries are as long as the image, and this is handed
/// all three.
fn split_heads(g: &Graph, input: Value, num_heads: i32, head_dim: i32) -> Value {
    let x = g.contiguous(input);
    let shape = [
        Extent::of(x, 0),
        Extent::of(x, 1),
        Extent::At(num_heads),
        Extent::At(head_dim),
    ];

    g.contiguous(g.transpose(g.view(x, shape), 1, 2))
}

/// One attention of a transformer block, either over the image itself or over the prompt.
///
/// `context` is what the keys and values are read from and how wide it is, and `None` is self
/// attention: it takes all three projections from `input` and so fuses them into one weight.
/// Cross attention's context is a different width, so only the keys and values fuse and the query
/// stays on its own.
fn attention(
    g: &Graph,
    input: Value,
    context: Option<(Value, i32)>,
    channels: i32,
    num_heads: i32,
) -> Value {
    let head_dim = channels / num_heads;
    let batch = Extent::of(input, 0);
    let length = Extent::of(input, 1);

    // None of these projections carries a bias, which is what attention_bias=False means.
    let (query, key, value) = match context {
        None => {
            let qkv = Linear::graph(
                &g.subgraph("qkv_proj"),
                input,
                channels,
                3 * channels,
                false,
            );
            (
                g.slice(qkv, 2, 0, channels),
                g.slice(qkv, 2, channels, 2 * channels),
                g.slice(qkv, 2, 2 * channels, 3 * channels),
            )
        }
        Some((context, context_dim)) => {
            let query = Linear::graph(&g.subgraph("q_proj"), input, channels, channels, false);
            let kv = Linear::graph(
                &g.subgraph("kv_proj"),
                context,
                context_dim,
                2 * channels,
                false,
            );
            (
                query,
                g.slice(kv, 2, 0, channels),
                g.slice(kv, 2, channels, 2 * channels),
            )
        }
    };

    let x = g.attention(
        split_heads(g, query, num_heads, head_dim),
        split_heads(g, key, num_heads, head_dim),
        split_heads(g, value, num_heads, head_dim),
        false,
    );
    let x = g.view(
        g.contiguous(g.transpose(x, 1, 2)),
        [batch, length, Extent::At(channels)],
    );

    Linear::graph(&g.subgraph("out_proj"), x, channels, channels, true)
}

/// Self attention, then attention over the prompt, then a feed forward, each around a
/// normalization and a residual.
fn transformer_block(
    g: &Graph,
    config: &UnetConfig,
    input: Value,
    context: Value,
    channels: i32,
    num_heads: i32,
) -> Value {
    let normed = LayerNorm::graph(&g.subgraph("norm1"), input, channels, 1e-5);
    let x = g.add(
        input,
        attention(&g.subgraph("attn1"), normed, None, channels, num_heads),
    );

    let normed = LayerNorm::graph(&g.subgraph("norm2"), x, channels, 1e-5);
    let x = g.add(
        x,
        attention(
            &g.subgraph("attn2"),
            normed,
            Some((context, config.cross_attention_dim)),
            channels,
            num_heads,
        ),
    );

    // The gated feed forward projects to twice its inner width, half of which gates the other
    // half, so what comes out of the gate is four times as wide as what goes back in.
    let normed = LayerNorm::graph(&g.subgraph("norm3"), x, channels, 1e-5);
    let feed_forward = Linear::graph(
        &g.subgraph("ff.gate.proj"),
        normed,
        channels,
        8 * channels,
        true,
    );
    let feed_forward = Linear::graph(
        &g.subgraph("ff.out_proj"),
        g.geglu(feed_forward),
        4 * channels,
        channels,
        true,
    );

    g.add(x, feed_forward)
}

/// A stack of transformer blocks that reads an image as a sequence of pixels.
///
/// The image goes in as `(N, C, H, W)` and comes back the same. In between every pixel is one
/// position of a sequence, which is what makes cross attention over the prompt possible at all
/// and what makes this the expensive part of the model.
fn transformer(
    g: &Graph,
    config: &UnetConfig,
    input: Value,
    context: Value,
    channels: i32,
    num_heads: i32,
    depth: i32,
) -> Value {
    let x = GroupNorm::graph(
        &g.subgraph("norm"),
        input,
        channels,
        config.norm_num_groups,
        1e-6,
    );

    // (N, C, H, W) to (N, H * W, C): a pixel is a position, and its channels are its vector. How
    // many positions that is is the one place a size here is two of the input's multiplied
    // together rather than one of them read off.
    let x = g.view(
        x,
        [
            Extent::of(input, 0),
            Extent::At(channels),
            Extent::prod(input, 2, 4),
        ],
    );
    let x = g.contiguous(g.transpose(x, 1, 2));

    let mut x = Linear::graph(&g.subgraph("in_proj"), x, channels, channels, true);
    for index in 0..depth {
        x = transformer_block(
            &g.subgraph(&format!("block{index}")),
            config,
            x,
            context,
            channels,
            num_heads,
        );
    }
    let x = Linear::graph(&g.subgraph("out_proj"), x, channels, channels, true);

    // And back to an image, at the height and width the one that came in still knows.
    let x = g.view(
        g.contiguous(g.transpose(x, 1, 2)),
        [
            Extent::of(input, 0),
            Extent::At(channels),
            Extent::of(input, 2),
            Extent::of(input, 3),
        ],
    );

    g.add(input, x)
}

/// One resolution on the way down: residual blocks, each optionally followed by a transformer,
/// and the halving that ends the block.
///
/// Everything the way up will want is pushed onto `residuals` as it is produced.
fn down_block(
    g: &Graph,
    config: &UnetConfig,
    index: usize,
    input: Value,
    cond: Conditioning,
    residuals: &mut Vec<Value>,
) -> Value {
    let out_channels = config.block_out_channels[index];
    let depth = config.transformer_layers_per_block[index];
    let num_heads = config.num_heads[index];

    let mut x = input;
    for layer in 0..config.layers_per_block {
        let in_channels = match layer {
            0 if index == 0 => config.block_out_channels[0],
            0 => config.block_out_channels[index - 1],
            _ => out_channels,
        };

        x = resnet_block(
            &g.subgraph(&format!("resnet{layer}")),
            config,
            x,
            cond.temb,
            in_channels,
            out_channels,
        );
        if depth > 0 {
            x = transformer(
                &g.subgraph(&format!("attn{layer}")),
                config,
                x,
                cond.context,
                out_channels,
                num_heads,
                depth,
            );
        }
        residuals.push(x);
    }

    // The smallest resolution is where the way down ends; there is nothing to halve.
    if index + 1 < config.levels() {
        // Stride two, which is the halving itself; there is no pooling anywhere here.
        x = Conv2d::graph(
            &g.subgraph("downsample0"),
            x,
            out_channels,
            out_channels,
            3,
            2,
            1,
        );
        residuals.push(x);
    }

    x
}

/// One resolution on the way up. Each residual block is handed what the way down left at this
/// resolution, concatenated onto its input, which is the connection the shape is named for.
///
/// `index` counts from the widest, which is the order the way up runs in and the reverse of the
/// order the config lists.
fn up_block(
    g: &Graph,
    config: &UnetConfig,
    index: usize,
    input: Value,
    cond: Conditioning,
    residuals: &mut Vec<Value>,
) -> Value {
    let levels = config.levels();
    let level = levels - 1 - index;

    let out_channels = config.block_out_channels[level];
    let depth = config.transformer_layers_per_block[level];
    let num_heads = config.num_heads[level];

    // What the block before this one handed over: the narrowest level for the first of them,
    // since that is where the way down ended, and the previous level's width after that.
    let prev_channels = match index {
        0 => config.block_out_channels[levels - 1],
        _ => config.block_out_channels[levels - index],
    };
    // The residuals arrive widest first, and the last one at this resolution is the one the
    // previous level's downsampling produced, which is narrower than the rest.
    let skip_channels = config.block_out_channels[level.saturating_sub(1)];

    let layers = config.layers_per_block + 1;
    let mut x = input;
    for layer in 0..layers {
        let skip = match layer == layers - 1 {
            true => skip_channels,
            false => out_channels,
        };
        let from = match layer {
            0 => prev_channels,
            _ => out_channels,
        };

        let residual = residuals
            .pop()
            .expect("the way down produces exactly what the way up reads");

        x = resnet_block(
            &g.subgraph(&format!("resnet{layer}")),
            config,
            g.cat(x, residual, 1),
            cond.temb,
            from + skip,
            out_channels,
        );
        if depth > 0 {
            x = transformer(
                &g.subgraph(&format!("attn{layer}")),
                config,
                x,
                cond.context,
                out_channels,
                num_heads,
                depth,
            );
        }
    }

    if index + 1 < levels {
        x = Conv2d::graph(
            &g.subgraph("upsample0"),
            g.upsample_nearest2d(x, 2),
            out_channels,
            out_channels,
            3,
            1,
            1,
        );
    }

    x
}

/// The two embeddings the whole model is conditioned on, added together as one vector.
///
/// Both arrive already sinusoidal and already a row per latent -- see the module documentation
/// for why the sines are not here -- and what the graph does with them is the projections.
fn conditioning(config: &UnetConfig, g: &Graph) -> Value {
    let time_embed_dim = config.time_embed_dim();

    let sinusoid = g.input("timestep_embedding");
    let temb = Linear::graph(
        &g.subgraph("time_embd.linear1"),
        sinusoid,
        config.block_out_channels[0],
        time_embed_dim,
        true,
    );
    let temb = Linear::graph(
        &g.subgraph("time_embd.linear2"),
        g.silu(temb),
        time_embed_dim,
        time_embed_dim,
        true,
    );

    // The pooled prompt and the six numbers about the image, side by side.
    let pooled = g.input("pooled");
    let pooled = g.view(
        pooled,
        [Extent::of(pooled, 0), Extent::At(config.pooled_dim())],
    );
    let added = g.cat(pooled, g.input("time_ids_embedding"), -1);

    let added = Linear::graph(
        &g.subgraph("add_embd.linear1"),
        added,
        config.projection_class_embeddings_input_dim,
        time_embed_dim,
        true,
    );
    let added = Linear::graph(
        &g.subgraph("add_embd.linear2"),
        g.silu(added),
        time_embed_dim,
        time_embed_dim,
        true,
    );

    g.add(temb, added)
}

/// The whole U-Net, written into `g`, which is already narrowed to the namespace the package
/// holds it under.
///
/// Infallible: everything it could disagree with itself about is a thing about the config alone,
/// and [`Unet::build`] has already said no to it.
fn write(config: &UnetConfig, g: &Graph) {
    let levels = config.levels();
    let widest = config.block_out_channels[levels - 1];

    let latent = g.input("latent");
    let cond = Conditioning {
        temb: conditioning(config, g),
        context: g.input("context"),
    };

    let mut x = Conv2d::graph(
        &g.subgraph("conv_in"),
        latent,
        config.latent_channels,
        config.block_out_channels[0],
        3,
        1,
        1,
    );

    // What the way down produces at every resolution, for the way up to read back.
    let mut residuals = vec![x];
    for index in 0..levels {
        x = down_block(
            &g.subgraph(&format!("down{index}")),
            config,
            index,
            x,
            cond,
            &mut residuals,
        );
    }

    x = resnet_block(
        &g.subgraph("mid.resnet0"),
        config,
        x,
        cond.temb,
        widest,
        widest,
    );
    x = transformer(
        &g.subgraph("mid.attn0"),
        config,
        x,
        cond.context,
        widest,
        config.num_heads[levels - 1],
        config.transformer_layers_per_block[levels - 1],
    );
    x = resnet_block(
        &g.subgraph("mid.resnet1"),
        config,
        x,
        cond.temb,
        widest,
        widest,
    );

    for index in 0..levels {
        x = up_block(
            &g.subgraph(&format!("up{index}")),
            config,
            index,
            x,
            cond,
            &mut residuals,
        );
    }

    // An identity rather than a check: the way down pushes one for the first convolution, one per
    // residual block and one per halving, and the way up reads one per residual block of its own,
    // which is one more per level. Those are equal for every config there is.
    assert!(
        residuals.is_empty(),
        "the way up left {} residuals unread",
        residuals.len()
    );

    let x = GroupNorm::graph(
        &g.subgraph("conv_norm_out"),
        x,
        config.block_out_channels[0],
        config.norm_num_groups,
        1e-5,
    );
    let x = Conv2d::graph(
        &g.subgraph("conv_out"),
        g.silu(x),
        config.block_out_channels[0],
        config.latent_channels,
        3,
        1,
        1,
    );

    g.output("noise", x);
}

/// What one step of sampling is conditioned on, beside the latent itself.
pub struct UnetCondition<'a> {
    /// `(N, L, cross_attention_dim)`: the two text encoders side by side.
    ///
    /// One row per latent in the batch. Guidance sends the prompt and its absence through
    /// together, so N is two there and the rows must be in the same order as the latents.
    pub context: &'a Tensor,
    /// `(N, 1280)`: the pooled vector of the second encoder, a row per latent.
    pub pooled: &'a Tensor,
    /// The size SDXL is told about: the original height and width, the top and left it was
    /// cropped at, and the height and width being asked for.
    pub time_ids: [f32; 6],
}

pub struct Unet {
    config: UnetConfig,
    ir: Ir,
    /// Every weight the package holds, which the four halves of a model share. See
    /// [`resident`](crate::flint::resident).
    weights: Rc<dyn ParamSource>,
}

impl fmt::Debug for Unet {
    /// How big it is rather than what is in it. Both halves of this are large and both have a
    /// better way of being read: [`Unet::ir`] prints the pass, and the weights are named in it.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Unet({} instructions)", self.ir.len())
    }
}

impl Unet {
    /// `name` is the namespace the package holds this U-Net under, which is what its weights are
    /// named from; `weights` is the whole package, read once and shared.
    pub fn build(config: UnetConfig, name: &str, weights: &Rc<dyn ParamSource>) -> Result<Unet> {
        check(&config)?;

        let graph = Graph::with_weights(config.weight_format);
        write(&config, &graph.subgraph(name));
        check_parameters(&graph, weights.as_ref())?;

        let ir = Ir::compile(&graph);

        Ok(Unet {
            weights: Rc::clone(weights),
            ir,
            config,
        })
    }

    pub fn config(&self) -> &UnetConfig {
        &self.config
    }

    /// The pass this U-Net runs, for printing.
    ///
    /// What an eager model could not be asked: every instruction, in the order it runs, with the
    /// weight each one reads named as the package names it.
    pub fn ir(&self) -> &Ir {
        &self.ir
    }

    /// The noise this U-Net believes is in `latent`, which is the same shape as the latent.
    ///
    /// `timestep` says how much noise the latent is supposed to hold, on the 0 to 999 scale the
    /// model was trained on. It is a float rather than an integer because some samplers ask about
    /// points between two training steps.
    pub fn forward(
        &self,
        latent: &Tensor,
        timestep: f32,
        condition: &UnetCondition<'_>,
    ) -> Result<Tensor> {
        let dim = latent.dim()?;
        if dim != 4 {
            return Err(Error::model(format!(
                "a latent is <float16>(N, C, H, W), got a {dim}-D tensor"
            )));
        }

        let channels = latent.shape_at(1)?;
        if channels != self.config.latent_channels {
            return Err(Error::model(format!(
                "a latent of {channels} channels is not the {} this U-Net reads",
                self.config.latent_channels
            )));
        }

        // Every resolution halves the one before it, so an odd size would lose a row on the way
        // down and never get it back. The graph cannot say this -- it holds no shapes at all --
        // so it is said here, where the tensor is, rather than being found out as a shape a
        // concatenation on the way up will not take.
        let divisor = 1 << (self.config.levels() - 1);
        let (height, width) = (latent.shape_at(2)?, latent.shape_at(3)?);
        if height % divisor != 0 || width % divisor != 0 {
            return Err(Error::model(format!(
                "a {height} by {width} latent does not survive being halved {} times",
                self.config.levels() - 1
            )));
        }

        let (device, dtype) = (condition.pooled.device(), condition.pooled.dtype());
        let batch = condition.pooled.shape_at(0)?;

        let time = timestep_embedding(timestep as f64, self.config.block_out_channels[0])?
            .to_device(device)?
            .cast(dtype)?;
        let sizes = time_ids_embedding(&condition.time_ids, self.config.addition_time_embed_dim)?
            .to_device(device)?
            .cast(dtype)?;

        let (time, sizes) = (repeat_rows(&time, batch)?, repeat_rows(&sizes, batch)?);
        let context = RunContext::new(&*self.weights)
            .input("latent", latent)
            .input("context", condition.context)
            .input("pooled", condition.pooled)
            .input("timestep_embedding", &time)
            .input("time_ids_embedding", &sizes);

        let outputs = self.ir.run(&context)?;

        outputs
            .into_iter()
            .find(|(name, _)| name == "noise")
            .map(|(_, tensor)| tensor)
            .ok_or_else(|| Error::model("the U-Net produced no noise"))
    }
}

/// Everything about a config that the graph would otherwise find out as a kernel refusing a
/// shape, said once before anything is written down.
///
/// A graph holds no shapes, so it cannot check itself the way an eagerly built layer did as it
/// read each weight. What is left to check is the config against itself, which is all of these:
/// every one of them was a check inside some layer's `build` before.
fn check(config: &UnetConfig) -> Result<()> {
    let levels = config.levels();
    if levels < 2
        || config.transformer_layers_per_block.len() != levels
        || config.num_heads.len() != levels
    {
        return Err(Error::model(
            "the U-Net config disagrees with itself about how many resolutions it has",
        ));
    }

    if config.norm_num_groups <= 0 {
        return Err(Error::model(format!(
            "{} is not a number of groups to normalize over",
            config.norm_num_groups
        )));
    }

    for level in 0..levels {
        let channels = config.block_out_channels[level];

        // Every group normalization in the model normalizes either one level's width or two of
        // them concatenated, and a sum of multiples of the groups is one too, so checking the
        // widths is checking all of them.
        if channels % config.norm_num_groups != 0 {
            return Err(Error::model(format!(
                "{channels} channels at level {level} do not divide into {} groups",
                config.norm_num_groups
            )));
        }

        // Where there is no transformer the head count is never read, and a config that says
        // something unusable about a level it does not attend over is not wrong about anything.
        let heads = config.num_heads[level];
        if config.transformer_layers_per_block[level] > 0 && (heads <= 0 || channels % heads != 0) {
            return Err(Error::model(format!(
                "{channels} channels at level {level} do not divide into {heads} heads"
            )));
        }
    }

    if config.pooled_dim() <= 0 {
        return Err(Error::model(format!(
            "the added embedding reads {} numbers, which is not more than the {} the six sizes \
             take",
            config.projection_class_embeddings_input_dim,
            6 * config.addition_time_embed_dim
        )));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> UnetConfig {
        UnetConfig {
            latent_channels: 4,
            block_out_channels: vec![320, 640, 1280],
            layers_per_block: 2,
            transformer_layers_per_block: vec![0, 2, 10],
            num_heads: vec![5, 10, 20],
            norm_num_groups: 32,
            cross_attention_dim: 2048,
            addition_time_embed_dim: 256,
            projection_class_embeddings_input_dim: 2816,
            weight_format: WeightFormat::Float,
        }
    }

    /// The sinusoidal embedding is a formula rather than something read from a checkpoint, so it
    /// can be checked against what it is supposed to be without any weights at hand.
    #[test]
    fn the_timestep_embedding_is_cosines_then_sines() {
        let embedding = timestep_embedding(981.0, 8).unwrap();
        assert_eq!(embedding.shape(), vec![1, 8]);

        let values = embedding.to_vec_f32().unwrap();
        for index in 0..4 {
            let frequency = (-MAX_PERIOD.ln() * index as f64 / 4.0).exp();
            let angle = 981.0 * frequency;
            assert!((values[index] as f64 - angle.cos()).abs() < 1e-6);
            assert!((values[4 + index] as f64 - angle.sin()).abs() < 1e-6);
        }
    }

    #[test]
    fn the_timestep_embedding_starts_at_the_lowest_frequency() {
        // The first pair is always the same, whatever the timestep: an angle of one radian per
        // step, so the cosine of the step and its sine.
        let values = timestep_embedding(1.0, 4).unwrap().to_vec_f32().unwrap();
        assert!((values[0] as f64 - 1.0f64.cos()).abs() < 1e-6);
        assert!((values[2] as f64 - 1.0f64.sin()).abs() < 1e-6);
    }

    #[test]
    fn a_timestep_embedding_needs_an_even_width() {
        assert!(timestep_embedding(0.0, 7).is_err());
        assert!(timestep_embedding(0.0, 0).is_err());
    }

    #[test]
    fn the_size_embedding_is_one_timestep_embedding_for_each_number() {
        let ids = [1024.0, 1024.0, 0.0, 0.0, 512.0, 512.0];
        let embedding = time_ids_embedding(&ids, 8).unwrap();
        assert_eq!(embedding.shape(), vec![1, 48]);

        let values = embedding.to_vec_f32().unwrap();
        for (index, id) in ids.iter().enumerate() {
            let one = timestep_embedding(*id as f64, 8)
                .unwrap()
                .to_vec_f32()
                .unwrap();
            assert_eq!(&values[index * 8..(index + 1) * 8], &one[..]);
        }
    }

    /// The pass can be written and read without a package to read weights out of, which is the
    /// half of a graph that costs nothing to test.
    #[test]
    fn the_whole_pass_is_written_without_any_weights() {
        let config = config();
        let graph = Graph::new();
        write(&config, &graph.subgraph("sdxl.unet"));

        let listing = graph.to_string();

        // The way down and the way up both reach every level, and the names are the package's.
        for level in 0..3 {
            assert!(listing.contains(&format!("sdxl.unet.down{level}.resnet0.conv1.weight")));
            assert!(listing.contains(&format!("sdxl.unet.up{level}.resnet0.conv1.weight")));
        }
        assert!(listing.contains("sdxl.unet.mid.attn0.block9.attn2.kv_proj.weight"));

        // The full-size level has no attention at all, which is where SDXL saves the most.
        assert!(!listing.contains("sdxl.unet.down0.attn0."));

        // Nothing in it was written for one latent size: the only sizes are the widths, and the
        // height and width are read off whatever comes in.
        assert!(
            listing.contains("dims(%"),
            "no folded extent in the listing"
        );
        assert!(listing.contains("dim(%"), "no read extent in the listing");
    }

    #[test]
    fn every_weight_the_graph_asks_for_is_asked_for_once() {
        let graph = Graph::new();
        write(&config(), &graph.subgraph("sdxl.unet"));

        // A load per weight and no more: the graph reads each of them exactly once, so a package
        // is walked once per build however many nodes go on to read the value.
        let mut names = graph.parameters();
        let total = names.len();
        names.sort();
        names.dedup();

        assert_eq!(names.len(), total);
    }

    #[test]
    fn a_config_that_disagrees_with_itself_is_refused() {
        let mut narrow = config();
        narrow.block_out_channels = vec![320];
        assert!(check(&narrow).is_err());

        let mut unheaded = config();
        unheaded.num_heads = vec![5, 10, 21];
        assert!(check(&unheaded).is_err());

        // The same head count at the level that has no transformer to read it, which is fine.
        let mut unread = config();
        unread.num_heads = vec![7, 10, 20];
        assert!(check(&unread).is_ok());

        let mut ungrouped = config();
        ungrouped.norm_num_groups = 48;
        assert!(check(&ungrouped).is_err());

        let mut unpooled = config();
        unpooled.projection_class_embeddings_input_dim = 1536;
        assert!(check(&unpooled).is_err());
    }
}
