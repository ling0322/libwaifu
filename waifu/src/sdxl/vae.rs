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

//! SDXL's autoencoder, which stands between a latent and the image it means.
//!
//! A latent is eight times smaller than the image on each axis and four channels deep instead of
//! three, so the decoder is where most of the pixels are made: a 128 by 128 latent becomes 1024 by
//! 1024. The encoder goes the other way, which is where image to image begins -- text to image
//! starts from noise and never asks for it.
//!
//! # Written down rather than run
//!
//! Both halves are a [`Graph`], the way the [`unet`](super::unet) and the
//! [`text_encoder`](super::text_encoder) are: `build` writes the pass down, compiles it into an
//! [`Ir`] and reads the weights that graph turns out to ask for, and `forward` hands over the
//! tensor and takes back what the graph named.
//!
//! Three things about this model are worth saying, because none of them looked the same in the
//! two models written down before it:
//!
//! *The one attention reads the image as a sequence*, every pixel attending to every other, so it
//! needs `H * W` where the graph has only `H` and `W`. That is [`Extent::prod`], the same as in
//! the U-Net's transformers -- but here the sequence is the whole picture at full resolution,
//! which is why it sits at the smallest one.
//!
//! *Where a latent is and what it is in* are part of the pass rather than something `forward`
//! arranges first. SDXL's autoencoder has to run in float32 while the sampler above it runs in
//! half, so a cast was always going to happen; writing it down as a [`to_device`](Graph::to_device)
//! and a [`cast`](Graph::cast) at the head of the graph means the listing says so.
//!
//! *The encoder produces two things*, a mean and a log variance, and a graph names its outputs, so
//! [`VaeEncoder::moments`] is one run that hands back both rather than a pass that computes a pair
//! and a caller that pulls it apart.

use std::fmt;
use std::rc::Rc;

use crate::error::{Error, Result};
use crate::flint::{
    check_parameters, functional as F, DType, Device, Extent, Graph, Held, Ir, ParamSource, RunContext,
    Tensor, Value,
};
use crate::layers::{Conv2d, GroupNorm, Linear};

/// What the package records about the autoencoder.
#[derive(Clone, Debug)]
pub struct VaeConfig {
    pub latent_channels: i32,
    /// Widest first, as the encoder counts them; the decoder walks them backwards.
    pub block_out_channels: Vec<i32>,
    pub layers_per_block: i32,
    pub norm_num_groups: i32,
    pub norm_eps: f32,
    /// What a latent is divided by before the model sees it, which is how the two were trained.
    pub scaling_factor: f32,
}

impl VaeConfig {
    /// How many rungs the two halves climb, which is one less doubling than that.
    fn rungs(&self) -> usize {
        self.block_out_channels.len()
    }

    /// The deepest channel count, which is where the encoder ends and the decoder starts.
    fn widest(&self) -> i32 {
        self.block_out_channels[self.rungs() - 1]
    }
}

/// Two convolutions around a normalization and an activation, added back to what came in. Where
/// the channel count changes, the shortcut is a 1x1 convolution rather than the input itself.
fn resnet_block(
    g: &Graph,
    config: &VaeConfig,
    input: Value,
    in_channels: i32,
    out_channels: i32,
) -> Value {
    let (groups, eps) = (config.norm_num_groups, config.norm_eps);

    let x = GroupNorm::graph(&g.subgraph("norm1"), input, in_channels, groups, eps);
    let x = g.silu(x);
    let x = Conv2d::graph(&g.subgraph("conv1"), x, in_channels, out_channels, 3, 1, 1);

    let x = GroupNorm::graph(&g.subgraph("norm2"), x, out_channels, groups, eps);
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

/// The one attention either half has, at the smallest resolution it works at.
///
/// It is a single head as wide as the whole channel count, which is not a shape the compiled
/// FlashAttention kernels take, so it runs on the fallback. That fallback takes the queries a
/// block at a time, which is what keeps a 1024 by 1024 image from needing half a gigabyte for the
/// score matrix alone.
fn attention_block(g: &Graph, config: &VaeConfig, input: Value, channels: i32) -> Value {
    let batch = Extent::of(input, 0);
    // Every pixel is a position that attends to every other, and how many that is is the product
    // of the two the graph can see rather than a number it was told.
    let positions = Extent::prod(input, 2, 4);

    let x = GroupNorm::graph(
        &g.subgraph("norm"),
        input,
        channels,
        config.norm_num_groups,
        config.norm_eps,
    );

    // (N, C, H, W) to (N, H * W, C): the image as the sequence it is about to be read as.
    let x = g.view(x, [batch, Extent::At(channels), positions]);
    let x = g.contiguous(g.transpose(x, 1, 2));

    let qkv = Linear::graph(&g.subgraph("qkv_proj"), x, channels, 3 * channels, true);
    let mut parts = Vec::with_capacity(3);
    for index in 0..3 {
        let part = g.contiguous(g.slice(qkv, 2, index * channels, (index + 1) * channels));
        // One head, as wide as the channels: (N, 1, H * W, C).
        parts.push(g.view(
            part,
            [batch, Extent::At(1), positions, Extent::At(channels)],
        ));
    }

    let x = g.attention(parts[0], parts[1], parts[2], false);
    let x = g.view(x, [batch, positions, Extent::At(channels)]);
    let x = Linear::graph(&g.subgraph("out_proj"), x, channels, channels, true);

    // And back to an image, at the height and width the one that came in still knows.
    let x = g.view(
        g.contiguous(g.transpose(x, 1, 2)),
        [
            batch,
            Extent::At(channels),
            Extent::of(input, 2),
            Extent::of(input, 3),
        ],
    );

    g.add(input, x)
}

/// The mid block, which both halves run at their deepest channel count and which is the only
/// place either of them attends over anything.
fn mid(g: &Graph, config: &VaeConfig, input: Value, channels: i32) -> Value {
    let x = resnet_block(
        &g.subgraph("mid.resnet0"),
        config,
        input,
        channels,
        channels,
    );
    let x = attention_block(&g.subgraph("mid.attn0"), config, x, channels);

    resnet_block(&g.subgraph("mid.resnet1"), config, x, channels, channels)
}

/// One rung of the decoder: some residual blocks, and the doubling that follows them.
fn up_block(
    g: &Graph,
    config: &VaeConfig,
    input: Value,
    in_channels: i32,
    out_channels: i32,
    upsample: bool,
) -> Value {
    // A decoder runs one more block per rung than the encoder does.
    let mut x = input;
    for index in 0..config.layers_per_block + 1 {
        let from = match index {
            0 => in_channels,
            _ => out_channels,
        };
        x = resnet_block(
            &g.subgraph(&format!("resnet{index}")),
            config,
            x,
            from,
            out_channels,
        );
    }

    // Nearest, then a convolution to smooth what repeating pixels left behind.
    match upsample {
        true => Conv2d::graph(
            &g.subgraph("upsample0"),
            g.upsample_nearest2d(x, 2),
            out_channels,
            out_channels,
            3,
            1,
            1,
        ),
        false => x,
    }
}

/// `input` `(N, C, H, W)` with a column of zeros down its right side and a row along its bottom.
///
/// A stride of two over a 3x3 kernel needs one row and one column of padding to keep the size
/// exactly halved, and a convolution with `padding = 1` would take it from both sides. The
/// encoder's halving takes it from the far side only, which is what the reference does: padding
/// both sides would start the first window one pixel earlier and move every output pixel with it.
fn pad_right_bottom(g: &Graph, input: Value) -> Value {
    // The zeros are a slice of the input multiplied by nothing rather than a `zeros` node, so that
    // the shape, the type and the device all come from the value being padded without any of the
    // three being written down here -- which in a graph is not merely tidier, since it is a graph
    // that does not know any of them.
    let column = g.mul_scalar(g.contiguous(g.slice(input, 3, 0, 1)), 0.0);
    let widened = g.cat(input, column, 3);

    let row = g.mul_scalar(g.contiguous(g.slice(widened, 2, 0, 1)), 0.0);

    g.cat(widened, row, 2)
}

/// One rung of the encoder: some residual blocks, and the halving that follows them.
fn down_block(
    g: &Graph,
    config: &VaeConfig,
    input: Value,
    in_channels: i32,
    out_channels: i32,
    downsample: bool,
) -> Value {
    // One block fewer per rung than the decoder runs, which is the asymmetry the two halves were
    // trained with rather than an oversight here.
    let mut x = input;
    for index in 0..config.layers_per_block {
        let from = match index {
            0 => in_channels,
            _ => out_channels,
        };
        x = resnet_block(
            &g.subgraph(&format!("resnet{index}")),
            config,
            x,
            from,
            out_channels,
        );
    }

    match downsample {
        true => Conv2d::graph(
            &g.subgraph("downsample0"),
            pad_right_bottom(g, x),
            out_channels,
            out_channels,
            3,
            2,
            0,
        ),
        false => x,
    }
}

/// The decoder, written into `g`, which is already narrowed to the namespace the package holds it
/// under.
///
/// `dtype` and `device` are where the weights were placed, and so where the latent has to be
/// brought to meet them. See the module documentation for why that is part of the pass.
fn write_decoder(config: &VaeConfig, dtype: DType, device: Device, g: &Graph) {
    // The decoder starts where the encoder ended, at the deepest channel count, and walks the
    // rungs backwards.
    let widest = config.widest();
    let reversed: Vec<i32> = config.block_out_channels.iter().rev().copied().collect();
    let narrowest = reversed[reversed.len() - 1];

    let latent = g.input("latent");
    let x = g.cast(g.to_device(latent, device), dtype);
    let x = g.div_scalar(x, config.scaling_factor);

    let x = Conv2d::graph(
        &g.subgraph("post_quant_conv"),
        x,
        config.latent_channels,
        config.latent_channels,
        1,
        1,
        0,
    );
    let x = Conv2d::graph(
        &g.subgraph("conv_in"),
        x,
        config.latent_channels,
        widest,
        3,
        1,
        1,
    );

    let mut x = mid(g, config, x, widest);
    for (index, out_channels) in reversed.iter().enumerate() {
        let in_channels = match index {
            0 => widest,
            _ => reversed[index - 1],
        };
        x = up_block(
            &g.subgraph(&format!("up{index}")),
            config,
            x,
            in_channels,
            *out_channels,
            // The last rung is already at full size and has nothing left to double.
            index + 1 < reversed.len(),
        );
    }

    let x = GroupNorm::graph(
        &g.subgraph("conv_norm_out"),
        x,
        narrowest,
        config.norm_num_groups,
        config.norm_eps,
    );
    let x = Conv2d::graph(&g.subgraph("conv_out"), g.silu(x), narrowest, 3, 3, 1, 1);

    g.output("image", x);
}

/// The encoder, written into `g`, which names two outputs rather than one: see
/// [`VaeEncoder::moments`].
fn write_encoder(config: &VaeConfig, dtype: DType, device: Device, g: &Graph) {
    // The encoder walks the channel counts as they are written, narrowest first, which is the
    // order the decoder reverses.
    let narrowest = config.block_out_channels[0];
    let widest = config.widest();

    let image = g.input("image");
    // Onto the device before the type, so that a narrowing is done where the tensor ends up rather
    // than a widened copy being carried across the bus.
    let x = g.cast(g.to_device(image, device), dtype);

    let mut x = Conv2d::graph(&g.subgraph("conv_in"), x, 3, narrowest, 3, 1, 1);
    for (index, out_channels) in config.block_out_channels.iter().enumerate() {
        let in_channels = match index {
            0 => narrowest,
            _ => config.block_out_channels[index - 1],
        };
        x = down_block(
            &g.subgraph(&format!("down{index}")),
            config,
            x,
            in_channels,
            *out_channels,
            // The last rung is already at the latent's size and has nothing left to halve.
            index + 1 < config.rungs(),
        );
    }

    let x = mid(g, config, x, widest);

    let x = GroupNorm::graph(
        &g.subgraph("conv_norm_out"),
        x,
        widest,
        config.norm_num_groups,
        config.norm_eps,
    );

    // The encoder does not produce a latent but a distribution over one, as a mean and a log
    // variance side by side, so everything from here is twice as deep as a latent.
    let channels = config.latent_channels;
    let moments = 2 * channels;
    let x = Conv2d::graph(&g.subgraph("conv_out"), g.silu(x), widest, moments, 3, 1, 1);
    let x = Conv2d::graph(&g.subgraph("quant_conv"), x, moments, moments, 1, 1, 0);

    g.output("mean", g.contiguous(g.slice(x, 1, 0, channels)));
    g.output("logvar", g.contiguous(g.slice(x, 1, channels, moments)));
}

/// Everything about a config that the graph would otherwise find out as a kernel refusing a shape.
///
/// A graph holds no shapes, so it cannot check itself the way an eagerly built layer did as it
/// read each weight. What is left is the config against itself, and both halves are checked here
/// because both are written from the same numbers.
fn check(config: &VaeConfig) -> Result<()> {
    if config.rungs() < 2 {
        return Err(Error::model("an autoencoder needs at least two rungs"));
    }

    if config.norm_num_groups <= 0 {
        return Err(Error::model(format!(
            "{} is not a number of groups to normalize over",
            config.norm_num_groups
        )));
    }

    // Every group normalization in either half normalizes one rung's width, so checking the
    // widths is checking all of them.
    for (rung, channels) in config.block_out_channels.iter().enumerate() {
        if channels % config.norm_num_groups != 0 {
            return Err(Error::model(format!(
                "{channels} channels at rung {rung} do not divide into {} groups",
                config.norm_num_groups
            )));
        }
    }

    Ok(())
}

/// One of what a run handed back, by the name the graph gave it.
fn output(outputs: &[(String, Tensor)], name: &str) -> Result<Tensor> {
    outputs
        .iter()
        .find(|(other, _)| other == name)
        .map(|(_, tensor)| tensor.clone())
        .ok_or_else(|| Error::model(format!("an autoencoder produces no {name:?}")))
}

pub struct VaeDecoder {
    config: VaeConfig,
    /// What its weights are in, which for SDXL is float32 whatever the rest of the model is in.
    dtype: DType,
    /// Where its weights are, which is where a latent has to be brought to meet them.
    device: Device,
    ir: Ir,
    held: Held,
    /// Every weight the package holds, which the four halves of a model share. See
    /// [`resident`](crate::flint::resident).
    weights: Rc<dyn ParamSource>,
}

impl fmt::Debug for VaeDecoder {
    /// How big it is rather than what is in it. Both halves of this are large and both have a
    /// better way of being read: [`VaeDecoder::ir`] prints the pass, and the weights are named
    /// in it.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "VaeDecoder({} instructions)", self.ir.len())
    }
}

impl VaeDecoder {
    /// `name` is the namespace the package holds this half under, which is what its weights are
    /// named from.
    pub fn build(
        config: VaeConfig,
        name: &str,
        weights: &Rc<dyn ParamSource>,
        device: Device,
        float_type: DType,
    ) -> Result<VaeDecoder> {
        check(&config)?;

        // Not `with_weights`: an autoencoder is convolutions, its few projections are small, and
        // SDXL runs this half in float32 where the FP8 multiply on CUDA takes float16. So the
        // format a package names is for the halves that have the matrices in them.
        let graph = Graph::new();
        write_decoder(&config, float_type, device, &graph.subgraph(name));
        check_parameters(&graph, weights.as_ref())?;

        let ir = Ir::compile(&graph, weights.residency());
        let held = ir.load(weights.as_ref())?;

        Ok(VaeDecoder {
            weights: Rc::clone(weights),
            held,
            ir,
            dtype: float_type,
            device,
            config,
        })
    }

    pub fn config(&self) -> &VaeConfig {
        &self.config
    }

    /// The pass this decoder runs, for printing.
    pub fn ir(&self) -> &Ir {
        &self.ir
    }

    /// What this decoder computes in. A latent of any float type is cast to it on the way in.
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Where this decoder is. A latent anywhere else is brought here on the way in.
    pub fn device(&self) -> Device {
        self.device
    }

    /// The image a latent stands for, as `(1, 3, H * 8, W * 8)` in roughly `[-1, 1]`.
    ///
    /// `latent` is `(1, C, H, W)` as the sampler leaves it, still scaled the way the model was
    /// trained; dividing that back out is the first thing the pass does. It is cast to whatever
    /// this decoder was built in, so a half latent from the sampler is read by a float32 decoder
    /// without the caller arranging it.
    ///
    /// SDXL's autoencoder has to be built in float32. Its own config says so with `force_upcast`,
    /// and it is not a matter of precision but of range: the activations grow through the up
    /// blocks -- about 84 at the mid block, 570, then 4046 -- and one convolution of the last one
    /// passes 65504, which is as far as half goes. Everything after that would be a NaN.
    pub fn forward(&self, latent: &Tensor) -> Result<Tensor> {
        let dim = latent.dim()?;
        if dim != 4 {
            return Err(Error::model(format!(
                "a latent is (N, C, H, W), got a {dim}-D tensor"
            )));
        }

        let channels = latent.shape_at(1)?;
        if channels != self.config.latent_channels {
            return Err(Error::model(format!(
                "a latent of {channels} channels is not the {} this decoder reads",
                self.config.latent_channels
            )));
        }

        let context = RunContext::new(&*self.weights).input("latent", latent);

        output(&self.ir.run(&self.held, &context)?, "image")
    }
}

/// The encoder half, which turns an image into the latent that stands for it.
///
/// Image to image is what this is for: an image goes in, the sampler picks up the latent partway
/// along its schedule instead of at pure noise, and the decoder turns what comes out back into
/// pixels. Text to image never builds one.
pub struct VaeEncoder {
    config: VaeConfig,
    /// What its weights are in, which for SDXL is float32 whatever the rest of the model is in.
    dtype: DType,
    /// Where its weights are, which is where a picture has to be brought to meet them.
    device: Device,
    ir: Ir,
    held: Held,
    /// Every weight the package holds, which the four halves of a model share. See
    /// [`resident`](crate::flint::resident).
    weights: Rc<dyn ParamSource>,
}

impl fmt::Debug for VaeEncoder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "VaeEncoder({} instructions)", self.ir.len())
    }
}

impl VaeEncoder {
    /// `name` is the namespace the package holds this half under, which is what its weights are
    /// named from.
    pub fn build(
        config: VaeConfig,
        name: &str,
        weights: &Rc<dyn ParamSource>,
        device: Device,
        float_type: DType,
    ) -> Result<VaeEncoder> {
        check(&config)?;

        // Not `with_weights`: an autoencoder is convolutions, its few projections are small, and
        // SDXL runs this half in float32 where the FP8 multiply on CUDA takes float16. So the
        // format a package names is for the halves that have the matrices in them.
        let graph = Graph::new();
        write_encoder(&config, float_type, device, &graph.subgraph(name));
        check_parameters(&graph, weights.as_ref())?;

        let ir = Ir::compile(&graph, weights.residency());
        let held = ir.load(weights.as_ref())?;

        Ok(VaeEncoder {
            weights: Rc::clone(weights),
            held,
            ir,
            dtype: float_type,
            device,
            config,
        })
    }

    pub fn config(&self) -> &VaeConfig {
        &self.config
    }

    /// The pass this encoder runs, for printing.
    pub fn ir(&self) -> &Ir {
        &self.ir
    }

    /// What this encoder computes in. An image of any float type is cast to it on the way in.
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Where this encoder is. An image anywhere else is brought here on the way in.
    pub fn device(&self) -> Device {
        self.device
    }

    /// How much smaller than its image a latent is on each axis, which is 8 for SDXL.
    pub fn scale(&self) -> i32 {
        1 << (self.config.rungs() - 1)
    }

    /// The latent an image stands for, as `(1, C, H / 8, W / 8)` scaled the way the sampler wants
    /// it -- the same scaling [`VaeDecoder::forward`] divides back out.
    ///
    /// `image` is `(1, 3, H, W)` in roughly `[-1, 1]`, which is the range the decoder produces and
    /// the one `to_rgb8` reverses, so an image read from a file has to be brought into it first.
    /// It may be on any device and in any float type: a picture read off the disk is on the host
    /// whatever the model is on, and the pass is what crosses.
    ///
    /// This takes the mode of the distribution rather than a draw from it. Image to image adds
    /// noise of its own on top of this latent, and enough of it that the posterior's own spread
    /// disappears underneath; drawing here as well would only make the same call with the same
    /// seed give a different image. [`VaeEncoder::moments`] is there for a caller who wants to
    /// sample anyway.
    pub fn forward(&self, image: &Tensor) -> Result<Tensor> {
        let (mean, _) = self.moments(image)?;
        Ok(F::mul_scalar(&mean, self.config.scaling_factor)?)
    }

    /// The distribution the encoder really produces: a mean and a log variance, both
    /// `(1, C, H / 8, W / 8)` and neither scaled.
    ///
    /// A latent drawn from it is `mean + exp(logvar / 2) * randn`, and is scaled by
    /// [`VaeConfig::scaling_factor`] before the sampler sees it. The graph names both, so this is
    /// one run and not two.
    pub fn moments(&self, image: &Tensor) -> Result<(Tensor, Tensor)> {
        let dim = image.dim()?;
        if dim != 4 {
            return Err(Error::model(format!(
                "an image is (N, C, H, W), got a {dim}-D tensor"
            )));
        }

        let channels = image.shape_at(1)?;
        if channels != 3 {
            return Err(Error::model(format!(
                "an image of {channels} channels is not the 3 this encoder reads"
            )));
        }

        // Each rung halves, and a size that does not divide would be silently rounded down by the
        // convolutions into a latent that decodes to a different image than the one handed in.
        // The graph holds no shapes, so this is said here or not at all.
        let scale = self.scale();
        let (height, width) = (image.shape_at(2)?, image.shape_at(3)?);
        if height % scale != 0 || width % scale != 0 {
            return Err(Error::model(format!(
                "a {height} by {width} image does not divide by the {scale} this encoder shrinks by"
            )));
        }

        let context = RunContext::new(&*self.weights).input("image", image);
        let outputs = self.ir.run(&self.held, &context)?;

        Ok((output(&outputs, "mean")?, output(&outputs, "logvar")?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> VaeConfig {
        VaeConfig {
            latent_channels: 4,
            block_out_channels: vec![128, 256, 512, 512],
            layers_per_block: 2,
            norm_num_groups: 32,
            norm_eps: 1e-6,
            scaling_factor: 0.13025,
        }
    }

    /// Both passes can be written and read without a package to read weights out of, which is the
    /// half of a graph that costs nothing to test.
    #[test]
    fn the_decoder_is_written_without_any_weights() {
        let g = Graph::new();
        write_decoder(
            &config(),
            DType::Float,
            Device::Cpu,
            &g.subgraph("sdxl.vae.decoder"),
        );
        let listing = g.to_string();

        // Every rung, walked backwards: the widest first and the narrowest last.
        assert!(listing.contains("sdxl.vae.decoder.up0.resnet0.conv1.weight"));
        assert!(listing.contains("sdxl.vae.decoder.up3.resnet2.conv2.weight"));
        assert!(listing.contains("sdxl.vae.decoder.mid.attn0.qkv_proj.weight"));

        // A decoder runs one more residual block per rung than the encoder does.
        assert!(listing.contains("sdxl.vae.decoder.up0.resnet2."));
        // And the last rung has nothing left to double.
        assert!(listing.contains("sdxl.vae.decoder.up2.upsample0."));
        assert!(!listing.contains("sdxl.vae.decoder.up3.upsample0."));

        // Nothing in it was written for one latent size: the attention folds the height and width
        // together rather than being told how many pixels there are.
        assert!(listing.contains("dims(%"), "{listing}");
    }

    #[test]
    fn the_encoder_is_written_without_any_weights() {
        let g = Graph::new();
        write_encoder(
            &config(),
            DType::Float,
            Device::Cpu,
            &g.subgraph("sdxl.vae.encoder"),
        );
        let listing = g.to_string();

        assert!(listing.contains("sdxl.vae.encoder.down0.resnet0.conv1.weight"));
        // One block fewer per rung than the decoder, so there is no third one.
        assert!(!listing.contains("sdxl.vae.encoder.down0.resnet2."));
        assert!(listing.contains("sdxl.vae.encoder.down2.downsample0."));
        assert!(!listing.contains("sdxl.vae.encoder.down3.downsample0."));

        // Two outputs, which is what a distribution over a latent is.
        assert!(listing.contains("-> mean = "), "{listing}");
        assert!(listing.contains("-> logvar = "), "{listing}");
    }

    #[test]
    fn a_config_that_disagrees_with_itself_is_refused() {
        let mut one_rung = config();
        one_rung.block_out_channels = vec![128];
        assert!(check(&one_rung).is_err());

        let mut ungrouped = config();
        ungrouped.norm_num_groups = 48;
        assert!(check(&ungrouped).is_err());

        assert!(check(&config()).is_ok());
    }
}
