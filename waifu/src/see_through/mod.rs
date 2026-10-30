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

//! See-through: an anime illustration taken apart into the layers it was drawn in.
//!
//! [See-through](https://github.com/shitagaki-lab/see-through) (SIGGRAPH 2026) is two diffusion
//! models and some picture handling around them, and this is its `inference_psd.py`:
//!
//! 1. **Layers.** The picture is centred in a square and scaled to 1280. An SDXL U-Net that
//!    attends across the layers it is drawing as well as across the picture draws thirteen of them
//!    at once -- hair, head, clothing, what is held -- from the picture's latent, and LayerDiffuse's
//!    decoder gives each its alpha. The head it drew is then cut out of the original picture,
//!    enlarged to 1280, and drawn again as eleven layers of its own -- face, eyes, ears, mouth --
//!    which are scaled back and put where the head was.
//! 2. **Depth.** A Marigold model with the same attention across layers says, for each layer, how
//!    far away each of its pixels is.
//! 3. **Parts.** Each layer is cut to where it is drawn and given the median of its depth, and the
//!    parts are stacked farthest first: a document whose layers are in drawing order.
//!
//! Every layer is an RGBA picture with whatever was hidden behind other layers drawn in, which is
//! what makes it a starting point for a Live2D artist rather than a segmentation.
//!
//! Neither model reads a prompt anyone writes, so neither has a text encoder here: the package
//! carries what the encoders made of the layer names and of the empty prompt. See
//! `tools/see_through_exporter.py` and `docs/see_through.md`.

mod alpha;
mod compose;
pub mod image;
mod psd;
mod sampler;
mod stratify;
mod unet;

use std::collections::HashMap;
use std::ops::ControlFlow;
use std::rc::Rc;

use crate::error::{Error, Result};
use crate::flint::{functional as F, DType, Device, Tensor, WeightFormat, Weights};
use crate::manifest::Manifest;
use crate::mapping::Mapping;
use crate::sdxl::{VaeConfig, VaeDecoder, VaeEncoder};

pub use alpha::{AlphaConfig, AlphaDecoder};
pub use compose::Part;
pub use image::{Image, Rgba, Square};
pub use psd::{
    read as read_psd, write as write_psd, write_with_hidden as write_psd_with_hidden,
    Document as PsdDocument, HiddenGroup as PsdHiddenGroup, Layer as PsdLayer, Mode as PsdMode,
};
pub use sampler::{DdimVelocity, DpmSolverSde, ScheduleConfig};
pub use stratify::{stratify, DepthLayer, STRATIFIED};
pub use unet::{Unet, UnetCondition, UnetConfig};

use compose::{Drawn, LAYER_ALPHA};
use image::{center_square_pad_resize, mask_above, smart_resize_u8};

/// The epsilon every diffusers autoencoder normalizes with.
const VAE_NORM_EPS: f32 = 1e-6;

/// How far a latent is from its picture on each axis.
const VAE_SCALE: usize = 8;

/// Everything a package says about the two models.
#[derive(Clone, Debug)]
pub struct SeeThroughConfig {
    pub layers_unet: UnetConfig,
    pub layers_vae: VaeConfig,
    pub alpha: AlphaConfig,
    pub layers_schedule: ScheduleConfig,
    pub depth_unet: UnetConfig,
    pub depth_vae: VaeConfig,
    pub depth_schedule: ScheduleConfig,
    pub body_tags: Vec<String>,
    pub head_tags: Vec<String>,
    /// The slots the depth model is asked about, in order; the full page follows them.
    pub depth_tags: Vec<String>,
    /// The slots that are several layers laid over each other, and which, back first.
    pub composites: HashMap<String, Vec<String>>,
    pub resolution: usize,
    pub steps: i32,
    pub depth_resolution: usize,
    pub depth_steps: i32,
}

fn list(section: &Mapping, key: &str) -> Result<Vec<String>> {
    Ok(section
        .get_str(key)?
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect())
}

fn numbers(section: &Mapping, key: &str) -> Result<Vec<i32>> {
    section
        .get_str(key)?
        .split(',')
        .map(|part| {
            part.trim()
                .parse::<i32>()
                .map_err(|_| Error::model(format!("{key} is not a list of numbers")))
        })
        .collect()
}

fn vae_config(section: &Mapping, prefix: &str, latent_channels: i32) -> Result<VaeConfig> {
    Ok(VaeConfig {
        latent_channels,
        block_out_channels: numbers(section, &format!("{prefix}_block_out_channels"))?,
        layers_per_block: section.get(&format!("{prefix}_layers_per_block"))?,
        norm_num_groups: section.get(&format!("{prefix}_norm_num_groups"))?,
        norm_eps: VAE_NORM_EPS,
        scaling_factor: section.get(&format!("{prefix}_scaling_factor"))?,
    })
}

fn schedule(section: &Mapping, prefix: &str) -> Result<ScheduleConfig> {
    Ok(ScheduleConfig {
        num_train_timesteps: section.get(&format!("{prefix}_scheduler_num_train_timesteps"))?,
        beta_start: section.get(&format!("{prefix}_scheduler_beta_start"))?,
        beta_end: section.get(&format!("{prefix}_scheduler_beta_end"))?,
        steps_offset: section.get(&format!("{prefix}_scheduler_steps_offset"))?,
    })
}

impl SeeThroughConfig {
    pub fn from_section(section: &Mapping) -> Result<SeeThroughConfig> {
        let weight_format = section.get_weight_format_or("weight_format", WeightFormat::Float)?;
        let latent_channels: i32 = section.get("latent_channels")?;
        let depth_tags = list(section, "depth_tags")?;
        let mut composites = HashMap::new();
        for tag in &depth_tags {
            let key = format!("depth_composite_{}", tag.replace(' ', "_"));
            if section.has(&key) {
                composites.insert(tag.clone(), list(section, &key)?);
            }
        }
        let alpha_attention = numbers(section, "alpha_attention")?;

        Ok(SeeThroughConfig {
            layers_unet: UnetConfig::from_section(section, "layers_unet", weight_format)?,
            layers_vae: vae_config(section, "layers_vae", latent_channels)?,
            alpha: AlphaConfig {
                block_out_channels: numbers(section, "alpha_block_out_channels")?,
                attention: alpha_attention.iter().map(|&a| a != 0).collect(),
                layers_per_block: section.get("alpha_layers_per_block")?,
                norm_num_groups: section.get("alpha_norm_num_groups")?,
                head_dim: section.get("alpha_head_dim")?,
                latent_level: section.get("alpha_latent_level")?,
                latent_channels,
            },
            layers_schedule: schedule(section, "layers")?,
            depth_unet: UnetConfig::from_section(section, "depth_unet", weight_format)?,
            depth_vae: vae_config(section, "depth_vae", latent_channels)?,
            depth_schedule: schedule(section, "depth")?,
            body_tags: list(section, "body_tags")?,
            head_tags: list(section, "head_tags")?,
            depth_tags,
            composites,
            resolution: section.get("layers_resolution")?,
            steps: section.get("layers_steps")?,
            depth_resolution: section.get("depth_resolution")?,
            depth_steps: section.get("depth_steps")?,
        })
    }
}

/// How a picture is taken apart.
#[derive(Clone, Debug)]
pub struct SeeThroughOptions {
    pub seed: u64,
    /// Steps of the layers model; the package's own when None.
    pub steps: Option<i32>,
    /// Steps of the depth model; the package's own when None.
    pub depth_steps: Option<i32>,
    /// Split hands, eyes, eyebrows and ears that come in two pieces into a left and a right part.
    /// Upstream's `--tblr_split`, off unless asked for there too.
    pub split_pairs: bool,
    /// The side of the square the layers are drawn at, a multiple of 64; the package's own (1280,
    /// what the model was trained at) when None. Smaller is faster and needs less memory.
    pub resolution: Option<usize>,
    /// The side of the square depth is found at, a multiple of 64; the package's own when None.
    pub depth_resolution: Option<usize>,
}

impl Default for SeeThroughOptions {
    fn default() -> Self {
        SeeThroughOptions {
            seed: 0,
            steps: None,
            depth_steps: None,
            split_pairs: false,
            resolution: None,
            depth_resolution: None,
        }
    }
}

/// Where a run is, for a caller that wants to show it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SeeThroughProgress {
    /// The picture encoded for a pass of the layers model, the body's (0) or the head's (1): said
    /// before the encoder runs, which in float32 at 1280 takes a while.
    Encoding { pass: i32 },
    /// A step of the layers model on the body (pass 0) or the head (pass 1).
    Layers { pass: i32, done: i32, total: i32 },
    /// A pass's layers decoded, `done` of `total` already: said before each one.
    Decoding { pass: i32, done: i32, total: i32 },
    /// The depth model's slots encoded, `done` of `total` already: said before each one.
    DepthEncoding { done: i32, total: i32 },
    /// A step of the depth model.
    Depth { done: i32, total: i32 },
    /// The depth maps decoded, `done` of `total` already: said before each one.
    DepthDecoding { done: i32, total: i32 },
    /// Cutting the layers into parts.
    Assembling,
}

/// A picture taken apart.
#[derive(Clone, Debug)]
pub struct Decomposition {
    /// The side of the square canvas the parts are on: the picture centred and scaled to it.
    pub canvas: usize,
    /// Where the picture is on that canvas and how big, as `(x, y, width, height)`.
    pub picture: (usize, usize, usize, usize),
    /// The parts, bottom first.
    pub parts: Vec<Part>,
}

/// The group a document keeps its parts' depth in: hidden and closed, after every part, so that it
/// is at the top of the layer panel and out of the way, and the document is whole without the
/// depth document beside it. Each part's depth is a layer of it named [`depth_layer_name`].
pub const DEPTH_GROUP: &str = "depth";

/// What a part's depth is called in [`DEPTH_GROUP`]: `depth/topwear`.
pub fn depth_layer_name(tag: &str) -> String {
    format!("{DEPTH_GROUP}/{tag}")
}

/// `parts`, bottom first, as a document on a `width` by `height` canvas, with each one's depth --
/// gray, 0 near to 255 far, under the part's own alpha -- in [`DEPTH_GROUP`] after them.
pub fn document_with_depth(parts: &[Part], width: usize, height: usize) -> Vec<u8> {
    let layers: Vec<psd::Layer> = parts
        .iter()
        .map(|part| psd::Layer {
            name: part.tag.clone(),
            left: part.left as i32,
            top: part.top as i32,
            pixels: part.pixels.clone(),
        })
        .collect();
    let depths: Vec<psd::Layer> = parts
        .iter()
        .map(|part| {
            let n = part.depth.width * part.depth.height;
            let mut pixels = Image::new(part.depth.width, part.depth.height, 4);
            for p in 0..n {
                pixels.data[p * 4..p * 4 + 3].fill(part.depth.data[p]);
                pixels.data[p * 4 + 3] = part.pixels.data[p * 4 + 3];
            }
            psd::Layer {
                name: depth_layer_name(&part.tag),
                left: part.left as i32,
                top: part.top as i32,
                pixels,
            }
        })
        .collect();
    let group = psd::HiddenGroup {
        name: DEPTH_GROUP,
        layers: &depths,
    };
    psd::write_with_hidden(&layers, Some(group), width, height, psd::Mode::Rgb)
}

impl Decomposition {
    /// A Photoshop document of the parts, bottom first, on the square canvas, each one's depth in
    /// a hidden group after them ([`document_with_depth`]).
    pub fn to_psd(&self) -> Vec<u8> {
        document_with_depth(&self.parts, self.canvas, self.canvas)
    }

    /// The same document with each part's depth in place of its colour, in grayscale, under the
    /// part's own alpha.
    pub fn to_depth_psd(&self) -> Vec<u8> {
        let layers: Vec<psd::Layer> = self
            .parts
            .iter()
            .map(|part| {
                // Each part's depth under its own alpha, so a part's cut-out corners show what is
                // beneath them rather than a rectangle of the depth around it.
                let n = part.depth.width * part.depth.height;
                let mut pixels = Image::new(part.depth.width, part.depth.height, 2);
                for p in 0..n {
                    pixels.data[p * 2] = part.depth.data[p];
                    pixels.data[p * 2 + 1] = part.pixels.data[p * 4 + 3];
                }
                psd::Layer {
                    name: part.tag.clone(),
                    left: part.left as i32,
                    top: part.top as i32,
                    pixels,
                }
            })
            .collect();
        psd::write(&layers, self.canvas, self.canvas, psd::Mode::Grayscale)
    }
}

/// One pass of the layers model's conditioning: one row per layer.
struct Condition {
    context: Tensor,
    pooled: Tensor,
}

pub struct SeeThrough {
    config: SeeThroughConfig,
    layers_unet: Unet,
    layers_decoder: VaeDecoder,
    layers_encoder: VaeEncoder,
    alpha: AlphaDecoder,
    body: Condition,
    head: Condition,
    depth_unet: Unet,
    depth_decoder: VaeDecoder,
    depth_encoder: VaeEncoder,
    depth_context: Tensor,
    device: Device,
    dtype: DType,
}

impl SeeThrough {
    pub const MODEL_TYPE: &'static str = "see_through";

    pub fn from_manifest(
        device: Device,
        device_weight_bytes: u64,
        manifest: &Manifest,
    ) -> Result<SeeThrough> {
        let model_type = manifest.section("model")?.get_str("type")?.to_string();
        if model_type != Self::MODEL_TYPE {
            return Err(Error::model(format!(
                "this package holds a model of kind {model_type:?}, which SeeThrough does not \
                 read"
            )));
        }
        let config = SeeThroughConfig::from_section(manifest.section(Self::MODEL_TYPE)?)?;
        let dtype = F::default_float_type(device)?;
        let weights: Rc<Weights> = Rc::new(Weights::from_files(
            &manifest.weight_paths()?,
            device,
            device_weight_bytes,
        )?);

        let named = |name: &str| format!("{}.{name}", Self::MODEL_TYPE);
        let constant = |name: &str| -> Result<Tensor> {
            let name = named(name);
            let shape = weights
                .shape_of(&name)
                .ok_or_else(|| Error::model(format!("the package has no {name}")))?;
            Ok(weights
                .load(&name, &shape)?
                .to_device(device)?
                .cast(dtype)?)
        };
        let condition = |pass: &str| -> Result<Condition> {
            Ok(Condition {
                context: constant(&format!("layers.{pass}.context"))?,
                pooled: constant(&format!("layers.{pass}.pooled"))?,
            })
        };

        Ok(SeeThrough {
            layers_unet: Unet::build(config.layers_unet.clone(), &named("layers.unet"), &weights)?,
            layers_decoder: VaeDecoder::build(
                config.layers_vae.clone(),
                &named("layers.vae"),
                &weights,
                device,
                DType::Float,
            )?,
            layers_encoder: VaeEncoder::build(
                config.layers_vae.clone(),
                &named("layers.vae.encoder"),
                &weights,
                device,
                DType::Float,
            )?,
            alpha: AlphaDecoder::build(
                config.alpha.clone(),
                &named("layers.alpha"),
                &weights,
                dtype,
            )?,
            body: condition("body")?,
            head: condition("head")?,
            depth_unet: Unet::build(config.depth_unet.clone(), &named("depth.unet"), &weights)?,
            depth_decoder: VaeDecoder::build(
                config.depth_vae.clone(),
                &named("depth.vae"),
                &weights,
                device,
                DType::Float,
            )?,
            depth_encoder: VaeEncoder::build(
                config.depth_vae.clone(),
                &named("depth.vae.encoder"),
                &weights,
                device,
                DType::Float,
            )?,
            depth_context: constant("depth.context")?,
            config,
            device,
            dtype,
        })
    }

    pub fn config(&self) -> &SeeThroughConfig {
        &self.config
    }

    /// The layers model's U-Net, for a caller that wants to ask it something the pipeline does not.
    pub fn layers_unet(&self) -> &Unet {
        &self.layers_unet
    }

    pub fn layers_decoder(&self) -> &VaeDecoder {
        &self.layers_decoder
    }

    pub fn layers_encoder(&self) -> &VaeEncoder {
        &self.layers_encoder
    }

    pub fn alpha_decoder(&self) -> &AlphaDecoder {
        &self.alpha
    }

    /// What the layers model is conditioned on in the body pass (0) or the head pass (1): one row
    /// of `(context, pooled)` per layer.
    pub fn condition(&self, pass: usize) -> (&Tensor, &Tensor) {
        let condition = match pass {
            0 => &self.body,
            _ => &self.head,
        };
        (&condition.context, &condition.pooled)
    }

    pub fn depth_unet(&self) -> &Unet {
        &self.depth_unet
    }

    pub fn depth_decoder(&self) -> &VaeDecoder {
        &self.depth_decoder
    }

    pub fn depth_encoder(&self) -> &VaeEncoder {
        &self.depth_encoder
    }

    /// The empty prompt, as the depth model reads it: `(1, 2, 1024)`.
    pub fn depth_context(&self) -> &Tensor {
        &self.depth_context
    }

    /// One layer from the latent the layers model left, `(1, 4, h, w)` scaled as the sampler
    /// leaves it: the autoencoder's colour, the alpha decoder's alpha and colour, the alpha
    /// multiplied by the picture's own -- nothing is drawn where the picture was transparent.
    pub fn decode_layer(&self, latent: &Tensor, page_alpha: &[f32]) -> Result<Rgba> {
        let scaling = self.config.layers_vae.scaling_factor;
        let pixel = self.layers_decoder.forward(latent)?;
        let pixel = F::add(&F::mul_scalar(&pixel, 0.5)?, &half_like(&pixel)?)?;
        let unscaled = F::mul_scalar(&latent.cast(DType::Float)?, 1.0 / scaling)?;
        let out = self.alpha.forward(&pixel, &unscaled)?;
        let (height, width) = (out.shape_at(2)? as usize, out.shape_at(3)? as usize);
        if page_alpha.len() != width * height {
            return Err(Error::model("the picture's alpha is not the layer's size"));
        }
        let values = out
            .cast(DType::Float)?
            .to_device(Device::Cpu)?
            .to_vec_f32()?;
        let plane = width * height;
        let mut rgba = Image::new(width, height, 4);
        for p in 0..plane {
            rgba.data[p * 4 + 3] = (values[p] * page_alpha[p] * 255.0).clamp(0.0, 255.0) as u8;
            for c in 0..3 {
                rgba.data[p * 4 + c] =
                    (values[(c + 1) * plane + p] * 255.0).clamp(0.0, 255.0) as u8;
            }
        }
        Ok(rgba)
    }

    /// What the depth model is shown, one picture a slot and the full page last: each layer with
    /// its faint alpha cleared, a composite slot its layers laid over each other, and the page
    /// with its alpha the sum of all of theirs.
    pub fn depth_slots(&self, page: &Rgba, layers: &HashMap<String, Rgba>) -> Vec<Rgba> {
        let canvas = page.width;
        let mut slots: Vec<Rgba> = Vec::new();
        for tag in &self.config.depth_tags {
            match self.config.composites.get(tag) {
                Some(parts) => {
                    let parts: Vec<Rgba> =
                        parts.iter().map(|t| cleared(layers, t, canvas)).collect();
                    let refs: Vec<&Rgba> = parts.iter().collect();
                    slots.push(image::blend_over(&refs));
                }
                None => slots.push(cleared(layers, tag, canvas)),
            }
        }
        let mut full = page.clone();
        for p in 0..canvas * canvas {
            let sum: f32 = slots.iter().map(|s| s.data[p * 4 + 3] as f32 / 255.0).sum();
            full.data[p * 4 + 3] = (sum.min(1.0) * 255.0) as u8;
        }
        slots.push(full);
        slots
    }

    /// One slot's latent, `(1, 4, h, w)`: its transparent parts filled from around them, then
    /// encoded. `slot` is already at the depth model's size.
    pub fn encode_slot(&self, slot: &Rgba) -> Result<Tensor> {
        let side = slot.width;
        let mut unit = slot.to_f32();
        for v in unit.data.iter_mut() {
            *v /= 255.0;
        }
        let rgb = image::pad_rgb(&unit);
        let mut planar = Vec::with_capacity(side * side * 3);
        for c in 0..3 {
            for p in 0..side * side {
                planar.push(rgb.data[p * 3 + c] * 2.0 - 1.0);
            }
        }
        let tensor = Tensor::from_f32(&[1, 3, side as i32, side as i32], &planar)?;
        Ok(self.depth_encoder.forward(&tensor)?.cast(self.dtype)?)
    }

    /// One depth map from the latent the depth model left, `(1, 4, h, w)`: the mean of the three
    /// channels the autoencoder decodes, from `[-1, 1]` to `[0, 1]`.
    pub fn decode_depth(&self, latent: &Tensor) -> Result<Image<f32>> {
        let decoded = self.depth_decoder.forward(latent)?;
        let (height, width) = (decoded.shape_at(2)? as usize, decoded.shape_at(3)? as usize);
        let values = decoded.to_device(Device::Cpu)?.to_vec_f32()?;
        let plane = width * height;
        let mut map = Image::new(width, height, 1);
        for p in 0..plane {
            let mean = (values[p] + values[plane + p] + values[2 * plane + p]) / 3.0;
            map.data[p] = (mean.clamp(-1.0, 1.0) + 1.0) / 2.0;
        }
        Ok(map)
    }

    /// Take `picture` apart.
    pub fn decompose(
        &self,
        picture: &Rgba,
        options: &SeeThroughOptions,
        report: &mut dyn FnMut(SeeThroughProgress) -> ControlFlow<()>,
    ) -> Result<Option<Decomposition>> {
        if picture.channels != 4 || picture.width == 0 || picture.height == 0 {
            return Err(Error::model(
                "a picture to take apart is RGBA and not empty",
            ));
        }
        F::manual_seed(self.device, options.seed)?;

        let resolution = options.resolution.unwrap_or(self.config.resolution);
        let depth_resolution = options
            .depth_resolution
            .unwrap_or(self.config.depth_resolution);
        for side in [resolution, depth_resolution] {
            if side == 0 || side % 64 != 0 {
                return Err(Error::model(format!(
                    "a square {side} on a side is not a multiple of 64, which both models need"
                )));
            }
        }
        let (page, square) = center_square_pad_resize(picture, resolution);
        let Some(layers) = self.draw_all(picture, &page, square, resolution, options, report)?
        else {
            return Ok(None);
        };

        let Some(depths) = self.depth(&page, &layers, depth_resolution, options, report)? else {
            return Ok(None);
        };
        if report(SeeThroughProgress::Assembling).is_break() {
            return Ok(None);
        }

        let drawn: Vec<Drawn> = depths
            .iter()
            .map(|(tag, depth)| Drawn {
                tag,
                pixels: &layers[tag],
                depth,
            })
            .collect();
        let parts = compose::assemble(&drawn, &page, options.split_pairs);

        let scale = resolution as f64 / square.side as f64;
        Ok(Some(Decomposition {
            canvas: resolution,
            picture: (
                (square.x as f64 * scale) as usize,
                (square.y as f64 * scale) as usize,
                (picture.width as f64 * scale).round() as usize,
                (picture.height as f64 * scale).round() as usize,
            ),
            parts,
        }))
    }

    /// Every layer the model draws, on the square canvas: the body's, then the head's put back.
    fn draw_all(
        &self,
        picture: &Rgba,
        page: &Rgba,
        square: Square,
        resolution: usize,
        options: &SeeThroughOptions,
        report: &mut dyn FnMut(SeeThroughProgress) -> ControlFlow<()>,
    ) -> Result<Option<HashMap<String, Rgba>>> {
        let steps = options.steps.unwrap_or(self.config.steps);

        let Some(body) = self.draw(page, &self.body, 0, steps, report)? else {
            return Ok(None);
        };
        let mut layers: HashMap<String, Rgba> =
            self.config.body_tags.iter().cloned().zip(body).collect();

        // The head the body pass found, in the original picture's pixels.
        let head = layers
            .get("head")
            .and_then(|head| image::bounding_rect(&mask_above(head, 3, LAYER_ALPHA)));
        let empty = Image::new(resolution, resolution, 4);
        let Some((hx0, hy0, hw0, hh0)) = head else {
            for tag in &self.config.head_tags {
                layers.insert(tag.clone(), empty.clone());
            }
            return Ok(Some(layers));
        };

        let cut = head_crop(picture, square, (hx0, hy0, hw0, hh0), resolution);
        let (crop, at_x, at_y, head_page, head_square) =
            (cut.crop, cut.at_x, cut.at_y, cut.page, cut.square);
        let scale = square.side as f64 / resolution as f64;
        let Some(head_layers) = self.draw(&head_page, &self.head, 1, steps, report)? else {
            return Ok(None);
        };

        // Back to the scale of the canvas, and to where the head was on it.
        let side = (head_square.side as f64 / scale) as usize;
        let (px1, py1) = (
            (head_square.x as f64 / scale) as usize,
            (head_square.y as f64 / scale) as usize,
        );
        let (px2, py2) = (
            ((head_square.x as f64 + crop.width as f64) / scale) as usize,
            ((head_square.y as f64 + crop.height as f64) / scale) as usize,
        );
        for (tag, layer) in self.config.head_tags.iter().zip(head_layers) {
            let scaled = smart_resize_u8(&layer, side, side).crop(px1, py1, px2, py2);
            let mut full = empty.clone();
            full.paste(&scaled, at_x, at_y);
            layers.insert(tag.clone(), full);
        }
        Ok(Some(layers))
    }

    /// One pass of the layers model over `page`: as many layers as `condition` has rows.
    fn draw(
        &self,
        page: &Rgba,
        condition: &Condition,
        pass: i32,
        steps: i32,
        report: &mut dyn FnMut(SeeThroughProgress) -> ControlFlow<()>,
    ) -> Result<Option<Vec<Rgba>>> {
        let device = self.device;
        let mut noise =
            |shape: &[i32], _: Option<usize>| -> Result<Tensor> { Ok(F::randn(shape, device)?) };
        self.walk(
            page,
            condition,
            pass,
            steps,
            &mut noise,
            &mut |_, _, _| {},
            report,
        )
    }

    /// One pass of the layers model, with its randomness and its steps in the caller's hands.
    ///
    /// `noise(shape, None)` is the noise every layer starts from, `(1, 4, h, w)`, and
    /// `noise(shape, Some(step))` what the sampler adds after that step, `(N, 4, h, w)`; drawn
    /// from the device's generator in a run, and read from a record when a run is replayed against
    /// upstream's. `observe(step, latent, eps)` is shown what the U-Net was handed at each step and
    /// what it answered, and once more after the last, as `observe(steps, finished, finished)`.
    pub fn draw_pass(
        &self,
        page: &Rgba,
        pass: usize,
        steps: i32,
        noise: &mut dyn FnMut(&[i32], Option<usize>) -> Result<Tensor>,
        observe: &mut dyn FnMut(usize, &Tensor, &Tensor),
    ) -> Result<Vec<Rgba>> {
        let condition = match pass {
            0 => &self.body,
            _ => &self.head,
        };
        let walked = self.walk(
            page,
            condition,
            pass as i32,
            steps,
            noise,
            observe,
            &mut |_| ControlFlow::Continue(()),
        )?;
        Ok(walked.expect("nothing asked it to stop"))
    }

    #[allow(clippy::too_many_arguments)]
    fn walk(
        &self,
        page: &Rgba,
        condition: &Condition,
        pass: i32,
        steps: i32,
        noise: &mut dyn FnMut(&[i32], Option<usize>) -> Result<Tensor>,
        observe: &mut dyn FnMut(usize, &Tensor, &Tensor),
        report: &mut dyn FnMut(SeeThroughProgress) -> ControlFlow<()>,
    ) -> Result<Option<Vec<Rgba>>> {
        let frames = condition.context.shape_at(0)?;
        let side = page.width;
        let latent_side = (side / VAE_SCALE) as i32;

        // The picture's colour, its alpha dropped: what the U-Net reads beside every layer.
        let mut rgb = Vec::with_capacity(side * side * 3);
        for c in 0..3 {
            for p in 0..side * side {
                rgb.push(page.data[p * 4 + c] as f32 / 127.5 - 1.0);
            }
        }
        let rgb = Tensor::from_f32(&[1, 3, side as i32, side as i32], &rgb)?;
        if report(SeeThroughProgress::Encoding { pass }).is_break() {
            return Ok(None);
        }
        let encoded = self.layers_encoder.forward(&rgb)?.cast(self.dtype)?;
        let picture = repeat(&encoded, frames)?;

        let mut sampler = DpmSolverSde::new(&self.config.layers_schedule, steps)?;
        // Every layer starts from the same noise.
        let initial = noise(&[1, 4, latent_side, latent_side], None)?
            .to_device(self.device)?
            .cast(DType::Float)?;
        let mut latent = repeat(&initial, frames)?;
        let size = side as f32;
        let time_ids = [size, size, 0.0, 0.0, size, size];
        for index in 0..sampler.len() {
            let input = F::cat(&latent.cast(self.dtype)?, &picture, 1)?;
            let eps = self.layers_unet.forward(
                &input,
                sampler.timestep(index),
                &UnetCondition {
                    context: &condition.context,
                    pooled: Some(&condition.pooled),
                    time_ids,
                },
            )?;
            observe(index, &latent, &eps);
            let fresh = noise(&[frames, 4, latent_side, latent_side], Some(index))?
                .to_device(self.device)?
                .cast(DType::Float)?;
            latent = sampler.step(index, &eps, &latent, &fresh)?;
            let progress = SeeThroughProgress::Layers {
                pass,
                done: index as i32 + 1,
                total: sampler.len() as i32,
            };
            if report(progress).is_break() {
                return Ok(None);
            }
        }
        observe(sampler.len(), &latent, &latent);

        let page_alpha: Vec<f32> = (0..side * side)
            .map(|p| page.data[p * 4 + 3] as f32 / 255.0)
            .collect();
        let mut layers = Vec::with_capacity(frames as usize);
        for frame in 0..frames {
            let progress = SeeThroughProgress::Decoding {
                pass,
                done: frame,
                total: frames,
            };
            if report(progress).is_break() {
                return Ok(None);
            }
            let latent = latent.slice(0, frame, frame + 1)?.contiguous()?;
            layers.push(self.decode_layer(&latent, &page_alpha)?);
        }
        Ok(Some(layers))
    }

    /// The depth of every slot the depth model is asked about, at the canvas's size, as the parts
    /// the slot is made of: `(tag, depth)` in the order a document's ties are broken in.
    fn depth(
        &self,
        page: &Rgba,
        layers: &HashMap<String, Rgba>,
        side: usize,
        options: &SeeThroughOptions,
        report: &mut dyn FnMut(SeeThroughProgress) -> ControlFlow<()>,
    ) -> Result<Option<Vec<(String, Image<u8>)>>> {
        let canvas = page.width;
        let steps = options.depth_steps.unwrap_or(self.config.depth_steps);

        let cleared = |tag: &str| cleared(layers, tag, canvas);
        let slots = self.depth_slots(page, layers);
        let frames = slots.len() as i32;
        let mut encoded = Vec::with_capacity(slots.len());
        for (done, slot) in slots.iter().enumerate() {
            let progress = SeeThroughProgress::DepthEncoding {
                done: done as i32,
                total: frames,
            };
            if report(progress).is_break() {
                return Ok(None);
            }
            encoded.push(self.encode_slot(&smart_resize_u8(slot, side, side))?);
        }
        let mut stacked = encoded[0].clone();
        for e in &encoded[1..] {
            stacked = F::cat(&stacked, e, 0)?;
        }
        let page_latent = repeat(encoded.last().expect("the full page"), frames)?;
        let condition = F::cat(&page_latent, &stacked, 1)?;
        let context = repeat(&self.depth_context, frames)?;

        let sampler = DdimVelocity::new(&self.config.depth_schedule, steps)?;
        let latent_side = (side / VAE_SCALE) as i32;
        let mut depth = F::randn(&[frames, 4, latent_side, latent_side], self.device)?;
        for index in 0..sampler.len() {
            let input = F::cat(&condition, &depth.cast(self.dtype)?, 1)?;
            let velocity = self.depth_unet.forward(
                &input,
                sampler.timestep(index),
                &UnetCondition {
                    context: &context,
                    pooled: None,
                    time_ids: [0.0; 6],
                },
            )?;
            depth = sampler.step(index, &velocity, &depth)?;
            let progress = SeeThroughProgress::Depth {
                done: index as i32 + 1,
                total: sampler.len() as i32,
            };
            if report(progress).is_break() {
                return Ok(None);
            }
        }

        // Each slot's depth, in [0, 1] and at the canvas's size. The full page's is not wanted.
        let mut maps = Vec::with_capacity(self.config.depth_tags.len());
        for frame in 0..frames - 1 {
            let progress = SeeThroughProgress::DepthDecoding {
                done: frame,
                total: frames - 1,
            };
            if report(progress).is_break() {
                return Ok(None);
            }
            let latent = depth.slice(0, frame, frame + 1)?.contiguous()?;
            let map = image::resize_linear(&self.decode_depth(&latent)?, canvas, canvas);
            maps.push(Image::from_data(
                canvas,
                canvas,
                1,
                map.data
                    .iter()
                    .map(|&d| (d.clamp(0.0, 1.0) * 255.0) as u8)
                    .collect(),
            ));
        }

        // A composite slot's depth handed to its parts, front first; the rest as they are.
        let mut out = Vec::new();
        for (tag, map) in self.config.depth_tags.iter().zip(maps) {
            match self.config.composites.get(tag) {
                Some(parts) => {
                    let front_first: Vec<&String> = parts.iter().rev().collect();
                    let images: Vec<Rgba> = front_first.iter().map(|t| cleared(t)).collect();
                    let refs: Vec<&Rgba> = images.iter().collect();
                    for (t, d) in front_first
                        .into_iter()
                        .zip(compose::composite_depths(&map, &refs))
                    {
                        out.push((t.clone(), d));
                    }
                }
                None => out.push((tag.clone(), map)),
            }
        }
        Ok(Some(out))
    }
}

/// Where the head pass draws from: the head the body pass found, cut out of the original picture.
pub struct HeadCrop {
    /// The cut, in the original picture's pixels.
    pub crop: Rgba,
    /// Where its top left corner lands on the canvas.
    pub at_x: usize,
    pub at_y: usize,
    /// The cut centred in its square and scaled to `resolution`: what the head pass reads.
    pub page: Rgba,
    pub square: Square,
}

/// The head pass's picture, from the head layer's box `(x, y, w, h)` on the canvas -- upstream's
/// `_crop_head` and what `apply_layerdiff` does around it.
pub fn head_crop(
    picture: &Rgba,
    square: Square,
    head: (usize, usize, usize, usize),
    resolution: usize,
) -> HeadCrop {
    let (hx0, hy0, hw0, hh0) = head;
    let scale = square.side as f64 / resolution as f64;
    let (iw, ih) = (picture.width as i64, picture.height as i64);
    let hx = (hx0 as f64 * scale) as i64 - square.x as i64;
    let hy = (hy0 as f64 * scale) as i64 - square.y as i64;
    let hw = (hw0 as f64 * scale) as i64;
    let hh = (hh0 as f64 * scale) as i64;
    let (x1, x2) = widen(hx, hw, iw);
    let (y1, y2) = widen(hy, hh, ih);
    let crop = picture.crop(x1 as usize, y1 as usize, x2 as usize, y2 as usize);
    let at_x = (x1 as f64 / scale + square.x as f64 / scale) as usize;
    let at_y = (y1 as f64 / scale + square.y as f64 / scale) as usize;
    let (page, square) = center_square_pad_resize(&crop, resolution);
    HeadCrop {
        crop,
        at_x,
        at_y,
        page,
        square,
    }
}

/// The layer `tag` with its faint alpha cleared, or nothing where there is no such layer.
fn cleared(layers: &HashMap<String, Rgba>, tag: &str, canvas: usize) -> Rgba {
    let mut layer = layers
        .get(tag)
        .cloned()
        .unwrap_or_else(|| Image::new(canvas, canvas, 4));
    for p in 0..canvas * canvas {
        if layer.data[p * 4 + 3] < LAYER_ALPHA {
            layer.data[p * 4 + 3] = 0;
        }
    }
    layer
}

/// How far to widen the head's box on one axis: by a fifth of its length either side, unless it
/// already covers half the picture or would run off it. Upstream's `_crop_head`.
fn widen(at: i64, length: i64, limit: i64) -> (i64, i64) {
    let (mut lo, mut hi) = (at, at + length);
    if length < limit / 2 {
        let pad = (limit - at - length).min(at).min(length / 5);
        lo = (at - pad).clamp(0, limit);
        hi = (at + length + pad).clamp(0, limit);
    }
    (lo.clamp(0, limit), hi.clamp(0, limit))
}

/// `rows` copies of a one-row tensor, stacked.
fn repeat(row: &Tensor, rows: i32) -> Result<Tensor> {
    let mut out = row.clone();
    for _ in 1..rows {
        out = F::cat(&out, row, 0)?;
    }
    Ok(out)
}

/// A tensor of one half, the shape and kind of `like`.
fn half_like(like: &Tensor) -> Result<Tensor> {
    let mut out = Tensor::zeros(&like.shape(), like.dtype(), like.device())?;
    F::fill(&mut out, 0.5)?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_small_head_is_widened_by_a_fifth_either_side() {
        assert_eq!(widen(100, 50, 1000), (90, 160));
        // Not past the picture's edge.
        assert_eq!(widen(3, 50, 1000), (0, 56));
        // A head that is half the picture already is left alone.
        assert_eq!(widen(100, 600, 1000), (100, 700));
    }
}
