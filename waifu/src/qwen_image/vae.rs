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

//! The Qwen-Image 2.1 decoder: sixty-four latent channels in, four image channels out, sixteen
//! times larger on each side.
//!
//! It is Wan 2.2's autoencoder specialized to a still picture, and not the Qwen-Image one in
//! [`crate::qwen_vae`] -- which is Wan 2.1's, sixteen channels and eight times. What the two share
//! is written once: the RMSNorm over channels and the one attention at the bottom.
//!
//! What this one has that the other does not is a shortcut around each upsampling stage. The
//! stage is three residual blocks and a nearest-neighbour doubling with a convolution; beside it,
//! `DupUp3D` repeats the stage's input channels and folds them out into space and time. For a
//! single frame only the *last* of the time copies is kept (`first_chunk`), so what it comes to is
//! a fixed choice of input channel for each output channel and each of the four sub-pixels -- a
//! gather and a pixel shuffle, done here as a row lookup and two transposes. See [`dup_up`].
//!
//! Every convolution is already two dimensional in the checkpoint, and the `time_conv` of each
//! temporal upsample is not in the package: the first frame of a decode never reaches it.

use std::fmt;
use std::rc::Rc;

use super::config::VaeConfig;
use crate::error::{Error, Result};
use crate::flint::{
    check_parameters, DType, Device, Extent, Graph, Ir, ParamSource, RunContext, Tensor, Value,
};
use crate::layers::Conv2d;
use crate::sdxl::vae::pad_right_bottom;
use crate::qwen_vae::{attention, channel_norm};

/// One stage of the decoder or the encoder, as its weights and the package describe it.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Stage {
    input: i32,
    output: i32,
    residuals: i32,
    /// Whether the stage doubles the picture (in the decoder) or halves it (in the encoder), and
    /// so has a resample and a shortcut around it.
    resamples: bool,
    /// Whether the resample was a temporal one too, which only the shortcut remembers.
    temporal: bool,
}

/// Norm, swish, convolve, twice, plus what came in.
fn residual(g: &Graph, input: Value, in_channels: i32, out_channels: i32) -> Value {
    let x = g.silu(channel_norm(g, input, in_channels, "norm1"));
    let x = Conv2d::graph(&g.subgraph("conv1"), x, in_channels, out_channels, 3, 1, 1);
    let x = g.silu(channel_norm(g, x, out_channels, "norm2"));
    let x = Conv2d::graph(&g.subgraph("conv2"), x, out_channels, out_channels, 3, 1, 1);

    let skip = match in_channels == out_channels {
        true => input,
        false => Conv2d::graph(
            &g.subgraph("conv_shortcut"),
            input,
            in_channels,
            out_channels,
            1,
            1,
            0,
        ),
    };
    g.add(skip, x)
}

/// Which input channel each `(output channel, sub-row, sub-column)` of `DupUp3D` reads, for the
/// first frame of a decode.
///
/// `DupUp3D` repeats each of `C_in` channels `r` times, views the result as `(C_out, t, 2, 2)`
/// per pixel and moves the last three into time and space; `first_chunk` keeps time slice `t - 1`.
/// So the channel read at `(o, i, j)` is `((o * t + t - 1) * 2 + i) * 2 + j`, divided by `r`.
fn dup_up_indices(input: i32, output: i32, temporal: bool) -> Vec<i64> {
    let frames = if temporal { 2 } else { 1 };
    let repeats = output * frames * 4 / input;

    let mut indices = Vec::with_capacity(output as usize * 4);
    for o in 0..output {
        for i in 0..2 {
            for j in 0..2 {
                let repeated = ((o * frames + frames - 1) * 2 + i) * 2 + j;
                indices.push((repeated / repeats) as i64);
            }
        }
    }
    indices
}

/// The shortcut around an upsampling stage: `(1, C_in, H, W)` to `(1, C_out, 2H, 2W)`.
fn dup_up(g: &Graph, input: Value, stage: &Stage, device: Device) -> Result<Value> {
    let indices = dup_up_indices(stage.input, stage.output, stage.temporal);
    let indices = Tensor::from_i64(&[indices.len() as i32], &indices)?.to_device(device)?;

    let height = Extent::of(input, 2);
    let width = Extent::of(input, 3);
    let table = g.view(input, [Extent::At(stage.input), Extent::prod(input, 2, 4)]);
    let picked = g.lookup(table, g.constant(indices));

    // (o, i, j, H, W) -> (o, H, i, W, j)
    let x = g.view(
        picked,
        [
            stage.output.into(),
            Extent::At(2),
            Extent::At(2),
            height,
            width,
        ],
    );
    let x = g.transpose(g.transpose(x, 2, 3), 1, 2);
    let x = g.contiguous(g.transpose(x, 3, 4));
    // (o, H, 2, W, 2) is (1, o, 2H, 2W) in memory.
    Ok(g.view(
        x,
        [
            Extent::At(1),
            stage.output.into(),
            Extent::prod(x, 1, 3),
            Extent::prod(x, 3, 5),
        ],
    ))
}

/// Which row of the space-to-depth table each `(output channel, member)` of `AvgDown3D` reads,
/// for a single frame, and how many members each output channel averages.
///
/// `AvgDown3D` pads time in *front* to a multiple of its temporal factor, folds each channel's
/// `(t, 2, 2)` block into `C_in * t * 4` channels, groups them `g` at a time into `C_out` and takes
/// each group's mean. A single frame padded to two puts a frame of zeros first, so a group whose
/// members all lie in that frame averages zeros: it reads row `4 * C_in`, the zero row after the
/// table. The table's row `c * 4 + 2 * i + j` is channel `c` at sub-row `i` and sub-column `j`.
fn avg_down_indices(input: i32, output: i32, temporal: bool) -> (Vec<i64>, i32) {
    let frames = if temporal { 2 } else { 1 };
    let folded = input * frames * 4;
    let group = folded / output;

    let mut indices = Vec::with_capacity(folded as usize);
    for flat in 0..folded {
        let (channel, within) = (flat / (frames * 4), flat % (frames * 4));
        let (frame, pixel) = (within / 4, within % 4);
        let padded = temporal && frame == 0;
        indices.push(match padded {
            true => 4 * input as i64,
            false => (channel * 4 + pixel) as i64,
        });
    }
    (indices, group)
}

/// The shortcut around a downsampling stage: `(1, C_in, 2H, 2W)` to `(1, C_out, H, W)`. `halved`
/// is any `(.., .., H, W)` value, which is where the halved size is read from.
fn avg_down(g: &Graph, input: Value, halved: Value, stage: &Stage, device: Device) -> Result<Value> {
    let (indices, group) = avg_down_indices(stage.input, stage.output, stage.temporal);
    let indices = Tensor::from_i64(&[indices.len() as i32], &indices)?.to_device(device)?;

    let height = Extent::of(halved, 2);
    let width = Extent::of(halved, 3);

    // (c, H, i, W, j) -> (c, i, j, H, W), then one row per (c, i, j) and a row of zeros after.
    let x = g.view(
        input,
        [
            stage.input.into(),
            height,
            Extent::At(2),
            width,
            Extent::At(2),
        ],
    );
    let x = g.transpose(g.transpose(g.transpose(x, 1, 2), 3, 4), 2, 3);
    let table = g.view(
        g.contiguous(x),
        [Extent::At(4 * stage.input), Extent::prod(halved, 2, 4)],
    );
    let zeros = g.mul_scalar(g.contiguous(g.slice(table, 0, 0, 1)), 0.0);
    let table = g.cat(table, zeros, 0);

    let picked = g.lookup(table, g.constant(indices));
    let grouped = g.view(
        picked,
        [
            stage.output.into(),
            Extent::At(group),
            Extent::prod(halved, 2, 4),
        ],
    );
    let mean = g.div_scalar(g.sum(grouped, 1), group as f32);
    Ok(g.view(
        mean,
        [Extent::At(1), stage.output.into(), height, width],
    ))
}

/// The encoder, written into `g`: a picture in `-1..=1` to the normalized latent the denoiser
/// reads, the mode of the posterior and not a draw from it.
fn write_encoder(
    config: &VaeConfig,
    stages: &[Stage],
    dtype: DType,
    device: Device,
    g: &Graph,
) -> Result<()> {
    let image = g.input("image");
    let mean = g.input("latents_mean");
    let std = g.input("latents_std");
    let channels = config.latent_channels;

    let encoder = g.subgraph("encoder");
    let first = stages[0].input;
    let x = g.cast(g.to_device(image, device), dtype);
    let mut x = Conv2d::graph(
        &encoder.subgraph("conv_in"),
        x,
        config.image_channels,
        first,
        3,
        1,
        1,
    );

    for (index, stage) in stages.iter().enumerate() {
        let sub = encoder.subgraph(&format!("down_blocks.{index}"));
        let entering = x;

        let mut width = stage.input;
        for resnet in 0..stage.residuals {
            x = residual(
                &sub.subgraph(&format!("resnets.{resnet}")),
                x,
                width,
                stage.output,
            );
            width = stage.output;
        }

        // A stage that does not halve has a shortcut that does not either: a group of one, which
        // is the stage's input as it came in.
        x = match stage.resamples {
            true => {
                let halved = Conv2d::graph(
                    &sub.subgraph("downsampler.resample.1"),
                    pad_right_bottom(g, x),
                    stage.output,
                    stage.output,
                    3,
                    2,
                    0,
                );
                g.add(halved, avg_down(g, entering, halved, stage, device)?)
            }
            false => g.add(x, entering),
        };
    }

    let deepest = stages.last().map(|stage| stage.output).unwrap_or(first);
    let middle = encoder.subgraph("mid_block");
    x = residual(&middle.subgraph("resnets.0"), x, deepest, deepest);
    x = attention(&middle.subgraph("attentions.0"), x, deepest);
    x = residual(&middle.subgraph("resnets.1"), x, deepest, deepest);

    let x = g.silu(channel_norm(&encoder, x, deepest, "norm_out"));
    let x = Conv2d::graph(
        &encoder.subgraph("conv_out"),
        x,
        deepest,
        2 * channels,
        3,
        1,
        1,
    );
    let moments = Conv2d::graph(
        &g.subgraph("quant_conv"),
        x,
        2 * channels,
        2 * channels,
        1,
        1,
        0,
    );

    // The first half is the mean, and the mean is the mode.
    let latent = g.contiguous(g.slice(moments, 1, 0, channels));
    let latent = g.div(g.sub(latent, g.cast(mean, dtype)), g.cast(std, dtype));
    g.output("latent", g.cast(latent, DType::Float));
    Ok(())
}

fn write(
    config: &VaeConfig,
    stages: &[Stage],
    deepest: i32,
    dtype: DType,
    device: Device,
    g: &Graph,
) -> Result<()> {
    let latent = g.input("latent");
    let mean = g.input("latents_mean");
    let std = g.input("latents_std");
    let channels = config.latent_channels;

    let x = g.cast(g.to_device(latent, device), dtype);
    let x = g.add(g.mul(x, g.cast(std, dtype)), g.cast(mean, dtype));
    let x = Conv2d::graph(
        &g.subgraph("post_quant_conv"),
        x,
        channels,
        channels,
        1,
        1,
        0,
    );

    let decoder = g.subgraph("decoder");
    let mut x = Conv2d::graph(&decoder.subgraph("conv_in"), x, channels, deepest, 3, 1, 1);

    let middle = decoder.subgraph("mid_block");
    x = residual(&middle.subgraph("resnets.0"), x, deepest, deepest);
    x = attention(&middle.subgraph("attentions.0"), x, deepest);
    x = residual(&middle.subgraph("resnets.1"), x, deepest, deepest);

    for (index, stage) in stages.iter().enumerate() {
        let sub = decoder.subgraph(&format!("up_blocks.{index}"));
        let entering = x;

        let mut width = stage.input;
        for resnet in 0..stage.residuals {
            x = residual(
                &sub.subgraph(&format!("resnets.{resnet}")),
                x,
                width,
                stage.output,
            );
            width = stage.output;
        }

        if stage.resamples {
            let up = g.upsample_nearest2d(x, 2);
            x = Conv2d::graph(
                &sub.subgraph("upsampler.resample.1"),
                up,
                stage.output,
                stage.output,
                3,
                1,
                1,
            );
            x = g.add(x, dup_up(g, entering, stage, device)?);
        }
    }

    let last = stages.last().map(|stage| stage.output).unwrap_or(deepest);
    let x = g.silu(channel_norm(&decoder, x, last, "norm_out"));
    g.output(
        "image",
        Conv2d::graph(
            &decoder.subgraph("conv_out"),
            x,
            last,
            config.image_channels,
            3,
            1,
            1,
        ),
    );
    Ok(())
}

/// The decoder's stages, read off the weights, with the temporal flags the package states.
fn stages(weights: &dyn ParamSource, name: &str, temporal: &[bool]) -> Result<Vec<Stage>> {
    let mut found = Vec::new();

    for index in 0.. {
        let prefix = format!("{name}.decoder.up_blocks.{index}");
        let first = format!("{prefix}.resnets.0.conv1.weight");
        let Some(shape) = weights.shape_of(&first) else {
            break;
        };

        let mut residuals = 0;
        while weights
            .shape_of(&format!("{prefix}.resnets.{residuals}.conv1.weight"))
            .is_some()
        {
            residuals += 1;
        }

        let resamples = weights
            .shape_of(&format!("{prefix}.upsampler.resample.1.weight"))
            .is_some();
        found.push(Stage {
            input: shape[1],
            output: shape[0],
            residuals,
            resamples,
            temporal: resamples && temporal.get(index).copied().unwrap_or(false),
        });
    }

    if found.is_empty() {
        return Err(Error::model(format!("{name} holds no decoder stages")));
    }
    for stage in &found {
        let frames = if stage.temporal { 2 } else { 1 };
        if stage.resamples && (stage.output * frames * 4) % stage.input != 0 {
            return Err(Error::model(format!(
                "a shortcut from {} channels to {} does not repeat evenly",
                stage.input, stage.output
            )));
        }
    }
    Ok(found)
}

/// The encoder's stages, read off the weights. `temporal` is the decoder's flags, as the package
/// states them: the encoder's are the same stages walked the other way.
fn encoder_stages(weights: &dyn ParamSource, name: &str, temporal: &[bool]) -> Result<Vec<Stage>> {
    let mut found = Vec::new();

    for index in 0.. {
        let prefix = format!("{name}.encoder.down_blocks.{index}");
        let Some(shape) = weights.shape_of(&format!("{prefix}.resnets.0.conv1.weight")) else {
            break;
        };

        let mut residuals = 0;
        while weights
            .shape_of(&format!("{prefix}.resnets.{residuals}.conv1.weight"))
            .is_some()
        {
            residuals += 1;
        }

        let resamples = weights
            .shape_of(&format!("{prefix}.downsampler.resample.1.weight"))
            .is_some();
        found.push(Stage {
            input: shape[1],
            output: shape[0],
            residuals,
            resamples,
            temporal: false,
        });
    }

    if found.is_empty() {
        return Err(Error::model(format!("{name} holds no encoder")));
    }

    let halvings = found.iter().filter(|stage| stage.resamples).count();
    if halvings != temporal.len() {
        return Err(Error::model(format!(
            "the encoder halves {halvings} times and the package states {} temporal flags",
            temporal.len()
        )));
    }
    let mut flags = temporal.iter().rev();
    for stage in found.iter_mut().filter(|stage| stage.resamples) {
        stage.temporal = *flags.next().expect("counted above");
    }

    for stage in &found {
        let frames = if stage.temporal { 2 } else { 1 };
        let evenly = match stage.resamples {
            true => (stage.input * frames * 4) % stage.output == 0,
            false => stage.input == stage.output,
        };
        if !evenly {
            return Err(Error::model(format!(
                "a shortcut from {} channels to {} does not average evenly",
                stage.input, stage.output
            )));
        }
    }
    Ok(found)
}

pub struct VaeDecoder {
    config: VaeConfig,
    ir: Ir,
    weights: Rc<dyn ParamSource>,
    mean: Tensor,
    std: Tensor,
}

impl fmt::Debug for VaeDecoder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "VaeDecoder({} instructions)", self.ir.len())
    }
}

impl VaeDecoder {
    pub fn build(
        config: VaeConfig,
        name: &str,
        weights: &Rc<dyn ParamSource>,
        device: Device,
        float_type: DType,
    ) -> Result<VaeDecoder> {
        let stages = stages(weights.as_ref(), name, &config.temporal_upsample)?;
        let opens_to = format!("{name}.decoder.conv_in.weight");
        let deepest = weights
            .shape_of(&opens_to)
            .and_then(|shape| shape.first().copied())
            .ok_or_else(|| Error::model(format!("{name} has no decoder")))?;

        let graph = Graph::new();
        write(
            &config,
            &stages,
            deepest,
            float_type,
            device,
            &graph.subgraph(name),
        )?;
        check_parameters(&graph, weights.as_ref())?;

        let shape = [1, config.latent_channels, 1, 1];
        let mean = Tensor::from_f32(&shape, &config.latents_mean)?.to_device(device)?;
        let std = Tensor::from_f32(&shape, &config.latents_std)?.to_device(device)?;

        let ir = Ir::compile(&graph);

        Ok(VaeDecoder {
            weights: Rc::clone(weights),
            ir,
            config,
            mean,
            std,
        })
    }

    pub fn config(&self) -> &VaeConfig {
        &self.config
    }

    /// `latent` is `(1, 64, H, W)`; what comes back is `(1, 4, H * 16, W * 16)` in roughly
    /// `-1..=1`, unclamped, the fourth channel alpha.
    pub fn forward(&self, latent: &Tensor) -> Result<Tensor> {
        if latent.dim()? != 4 || latent.shape_at(1)? != self.config.latent_channels {
            return Err(Error::model(format!(
                "the decoder takes a latent as (1, {}, H, W)",
                self.config.latent_channels
            )));
        }

        let run = RunContext::new(&*self.weights)
            .input("latent", latent)
            .input("latents_mean", &self.mean)
            .input("latents_std", &self.std);

        let outputs = self.ir.run(&run)?;
        outputs
            .iter()
            .find(|(name, _)| name == "image")
            .map(|(_, tensor)| tensor.clone())
            .ok_or_else(|| Error::model("the decoder produced no image"))
    }
}

/// The other half: a picture to the latent the denoiser reads, which is what a control picture
/// or an inpainting source has to become before the control chain can use it.
pub struct VaeEncoder {
    config: VaeConfig,
    ir: Ir,
    weights: Rc<dyn ParamSource>,
    mean: Tensor,
    std: Tensor,
}

impl fmt::Debug for VaeEncoder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "VaeEncoder({} instructions)", self.ir.len())
    }
}

impl VaeEncoder {
    pub fn build(
        config: VaeConfig,
        name: &str,
        weights: &Rc<dyn ParamSource>,
        device: Device,
        float_type: DType,
    ) -> Result<VaeEncoder> {
        let stages = encoder_stages(weights.as_ref(), name, &config.temporal_upsample)?;

        let graph = Graph::new();
        write_encoder(&config, &stages, float_type, device, &graph.subgraph(name))?;
        check_parameters(&graph, weights.as_ref())?;

        let shape = [1, config.latent_channels, 1, 1];
        let mean = Tensor::from_f32(&shape, &config.latents_mean)?.to_device(device)?;
        let std = Tensor::from_f32(&shape, &config.latents_std)?.to_device(device)?;

        Ok(VaeEncoder {
            weights: Rc::clone(weights),
            ir: Ir::compile(&graph),
            config,
            mean,
            std,
        })
    }

    pub fn config(&self) -> &VaeConfig {
        &self.config
    }

    /// `image` is `(1, 4, H, W)` in `-1..=1`, the fourth channel alpha, with `H` and `W`
    /// multiples of sixteen; what comes back is `(1, 64, H / 16, W / 16)` in float32, normalized
    /// by the package's mean and deviation the way a latent the sampler walks is.
    pub fn forward(&self, image: &Tensor) -> Result<Tensor> {
        let scale = self.config.scale;
        let dims = image.dim()?;
        if dims != 4 || image.shape_at(1)? != self.config.image_channels {
            return Err(Error::model(format!(
                "the encoder takes a picture as (1, {}, H, W)",
                self.config.image_channels
            )));
        }
        let (height, width) = (image.shape_at(2)?, image.shape_at(3)?);
        if height % scale != 0 || width % scale != 0 {
            return Err(Error::model(format!(
                "the encoder takes a picture whose sides are multiples of {scale}, got \
                 {width} x {height}"
            )));
        }

        let run = RunContext::new(&*self.weights)
            .input("image", image)
            .input("latents_mean", &self.mean)
            .input("latents_std", &self.std);

        let outputs = self.ir.run(&run)?;
        outputs
            .iter()
            .find(|(name, _)| name == "latent")
            .map(|(_, tensor)| tensor.clone())
            .ok_or_else(|| Error::model("the encoder produced no latent"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The released encoder's shortcuts, worked out by hand from `AvgDown3D`.
    #[test]
    fn the_shortcut_averages_what_avg_down_does() {
        // 96 to 96, spatial only: each channel is the mean of its own four sub-pixels.
        let (same, group) = avg_down_indices(96, 96, false);
        assert_eq!(group, 4);
        assert_eq!(&same[..8], &[0, 1, 2, 3, 4, 5, 6, 7]);

        // 96 to 192, temporal: the frame of zeros padded in front fills the even channels, and
        // the odd ones are the real frame's four sub-pixels.
        let (doubled, group) = avg_down_indices(96, 192, true);
        assert_eq!(group, 4);
        assert_eq!(&doubled[..12], &[384, 384, 384, 384, 0, 1, 2, 3, 384, 384, 384, 384]);
        assert_eq!(&doubled[12..16], &[4, 5, 6, 7]);
    }

    /// The four shortcuts of the released decoder, worked out by hand from `DupUp3D`.
    #[test]
    fn the_shortcut_reads_the_channels_dup_up_keeps() {
        // 1152 to 1152, temporal: eight copies of each channel, of which the second time slice's
        // four sub-pixels are all the same channel. A nearest-neighbour doubling.
        let same = dup_up_indices(1152, 1152, true);
        assert_eq!(&same[..8], &[0, 0, 0, 0, 1, 1, 1, 1]);

        // 1152 to 576, temporal: four copies, and the second time slice is the odd channel.
        let halved = dup_up_indices(1152, 576, true);
        assert_eq!(&halved[..8], &[1, 1, 1, 1, 3, 3, 3, 3]);

        // 576 to 288, spatial only: two copies, and the sub-row picks the channel.
        let shuffled = dup_up_indices(576, 288, false);
        assert_eq!(&shuffled[..8], &[0, 0, 1, 1, 2, 2, 3, 3]);
    }
}
