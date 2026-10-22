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

//! The parts joined up, with a prompt at one end and a picture at the other.
//!
//! 1. the prompt is wrapped in the chat template the model was sampled with: a system turn, the
//!    prompt as the user's, and the opening of the assistant's;
//! 2. Qwen3-VL reads the whole of that, and its last layer's states -- before its final norm --
//!    are what the denoiser reads, less the system turn's;
//! 3. the flow schedule walks a latent from noise, one denoiser pass per step, bent by a shift
//!    that depends on how large the picture is;
//! 4. the VAE turns the latent into four channels, the last of them alpha.
//!
//! A prompt may name pictures, which the user's turn opens with: `<image1>`, the picture's slots
//! between `<|vision_start|>` and `<|vision_end|>`, then a space and `<image2>`, and so on, and
//! then the prompt. The encoder reads each picture's vision rows in its slots; the denoiser reads
//! each picture's latent where its slots were, and draws a picture of its own after all of it.

use std::fmt;
use std::ops::ControlFlow;
use std::rc::Rc;

use super::dit::{Pictures, Segment};
use super::vision::{SeenPicture, VisionTower};
use super::{ControlConfig, Dit, QwenImageConfig, VaeDecoder, VaeEncoder};
use crate::error::{Error, Result};
use crate::flint::{functional as F, DType, Device, Residency, Tensor, Weights};
use crate::flow::{FlowSampler, SamplerConfig};
use crate::generation::{unwatched, GenerationDefaults, GenerationOptions, GenerationProgress};
use crate::krea2::{EncoderImages, TextEncoder};
use crate::manifest::Manifest;
use crate::tokenizer::Tokenizer;

/// What the denoiser reads for one prompt: its words, and the pictures it names.
#[derive(Clone)]
pub struct Prompted {
    /// `(1, L, D)`: the encoder's states for the prompt's words, the system turn dropped -- and,
    /// for a prompt that names pictures, the pictures' slots dropped too.
    pub context: Tensor,
    /// The pictures' latents and where they go among the words, for a prompt that names any.
    pub pictures: Option<Pictures>,
}

/// A picture the prompt names, read both ways.
struct Seen {
    vision: SeenPicture,
    /// `(1, h * w, C)`, the normalized latent as tokens.
    latent: Tensor,
    rows: i32,
    columns: i32,
}

/// `value` as bfloat16 holds it, rounded to the nearest even.
fn bfloat16(value: f32) -> f32 {
    let bits = value.to_bits();
    let rounded = bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) & 0xffff_0000;
    f32::from_bits(rounded)
}

/// What the denoiser is handed for a noise level `sigma`, which is not `sigma`.
///
/// The reference pipeline casts `sigma * 1000` to its latents' bfloat16, divides by a thousand in
/// bfloat16, and the model multiplies by a thousand again inside the sinusoid. bfloat16 has eight
/// bits of mantissa, so near a thousand it counts in fours: the model is handed 0.984375 for a
/// sigma of 0.9873, and at the sinusoid's fastest frequency that is two radians. The model was
/// sampled that way and is asked the same question here.
pub(crate) fn model_timestep(sigma: f32) -> f32 {
    bfloat16(bfloat16(sigma * 1000.0) / 1000.0)
}

/// Qwen-Image 2.1, with everything it needs to answer a prompt.
pub struct QwenImage {
    config: QwenImageConfig,
    tokenizer: Tokenizer,
    text_encoder: TextEncoder,
    dit: Dit,
    vae: VaeDecoder,
    /// The autoencoder's other half, built beside a control package or a vision tower: it is how
    /// a control picture reaches the control chain and a named picture the denoiser.
    encoder: Option<VaeEncoder>,
    /// Qwen3-VL's vision tower, in a package that has one: how a named picture reaches the
    /// encoder.
    vision: Option<VisionTower>,
    /// How many tokens the system turn comes to, which are read and then dropped. Counted from
    /// the package's vocabulary rather than written down.
    prefix_length: i32,
    device: Device,
    dtype: DType,
}

impl fmt::Debug for QwenImage {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("QwenImage")
            .field("blocks", &self.config.dit.num_blocks)
            .field("device", &self.device)
            .field("dtype", &self.dtype)
            .finish_non_exhaustive()
    }
}

impl QwenImage {
    /// What `model type` says for a model this can read.
    pub const MODEL_TYPE: &'static str = "qwen_image";

    pub const MODEL_SECTION: &'static str = "model";

    /// What closes the user's turn and opens the assistant's. The denoiser reads all of it.
    const PROMPT_SUFFIX: &'static str = "<|im_end|>\n<|im_start|>assistant\n";

    /// The model card's: forty steps and no guidance. The card draws at two megapixels; one is
    /// what fits a sixteen gigabyte card beside the weights streaming through it.
    pub const DEFAULTS: GenerationDefaults = GenerationDefaults {
        width: 1024,
        height: 1024,
        num_steps: 40,
        guidance_scale: 1.0,
    };

    /// What `model type` says for a control package this can read beside the model.
    pub const CONTROL_TYPE: &'static str = "qwen_image_control";

    /// How many pictures a prompt may name: the model card's "up to 10 reference images".
    pub const MAX_PICTURES: usize = 10;

    /// The area a named picture is read at, whatever size it arrives at or is drawn at: the
    /// reference's `output_resolution` squared.
    const PICTURE_AREA: f64 = 1024.0 * 1024.0;

    pub fn from_manifest(
        device: Device,
        residency: Residency,
        manifest: &Manifest,
    ) -> Result<QwenImage> {
        Self::from_manifests(device, residency, manifest, None)
    }

    /// The model, and with `control` the ControlNet package that steers it: one set of weights
    /// read out of both, since the control chain is part of the denoiser's graph.
    pub fn from_manifests(
        device: Device,
        residency: Residency,
        manifest: &Manifest,
        control: Option<&Manifest>,
    ) -> Result<QwenImage> {
        let model_section = manifest.section(Self::MODEL_SECTION)?;
        let model_type = model_section.get_str("type")?.to_string();
        if model_type != Self::MODEL_TYPE {
            return Err(Error::model(format!(
                "this manifest describes a model of kind {model_type:?}, which is not {:?}",
                Self::MODEL_TYPE
            )));
        }

        let config = QwenImageConfig::from_section(manifest.section(&model_type)?)?;

        let mut paths = manifest.weight_paths()?;
        let control_config = match control {
            Some(control) => {
                let kind = control
                    .section(Self::MODEL_SECTION)?
                    .get_str("type")?
                    .to_string();
                if kind != Self::CONTROL_TYPE {
                    return Err(Error::model(format!(
                        "the control package describes a model of kind {kind:?}, which is not \
                         {:?}",
                        Self::CONTROL_TYPE
                    )));
                }
                paths.extend(control.weight_paths()?);
                Some(ControlConfig::from_section(
                    control.section(ControlConfig::SECTION)?,
                    &config.dit,
                )?)
            }
            None => None,
        };

        let dtype = F::default_float_type(device)?;
        let weights: Rc<Weights> = Rc::new(Weights::from_files(&paths, device, residency)?);

        let named = |half: &str| format!("{model_type}.{half}");
        let tokenizer = Tokenizer::open(manifest)?;
        let prefix_length = tokenizer.encode(&Self::system_turn(&config))?.len() as i32;

        Ok(QwenImage {
            text_encoder: TextEncoder::build(
                config.encoder.clone(),
                &named("text"),
                &weights,
                dtype,
            )?,
            encoder: match control_config.is_some() || config.vision.is_some() {
                true => Some(VaeEncoder::build(
                    config.vae.clone(),
                    &named("vae"),
                    &weights,
                    device,
                    dtype,
                )?),
                false => None,
            },
            vision: match &config.vision {
                Some(vision) => Some(VisionTower::build(
                    vision.clone(),
                    &named("vision"),
                    &weights,
                    dtype,
                )?),
                None => None,
            },
            dit: Dit::build_with_control(
                config.dit.clone(),
                control_config,
                &named("dit"),
                &weights,
                dtype,
            )?,
            vae: VaeDecoder::build(config.vae.clone(), &named("vae"), &weights, device, dtype)?,
            prefix_length,
            tokenizer,
            config,
            device,
            dtype,
        })
    }

    /// The system turn, which is what the reference counts to know how many states to drop.
    fn system_turn(config: &QwenImageConfig) -> String {
        format!("<|im_start|>system\n{}<|im_end|>\n", config.system_prompt)
    }

    pub fn config(&self) -> &QwenImageConfig {
        &self.config
    }

    pub fn device(&self) -> Device {
        self.device
    }

    pub fn dit(&self) -> &Dit {
        &self.dit
    }

    pub fn text_encoder(&self) -> &TextEncoder {
        &self.text_encoder
    }

    pub fn vae(&self) -> &VaeDecoder {
        &self.vae
    }

    /// The template around `prompt`, as the `<long>(L)` the encoder reads.
    ///
    /// Tokenized as one string, the way the reference hands it to its processor. An empty prompt
    /// becomes a space, as the reference's does: Qwen has no start token, and an empty user turn
    /// is a different sentence from a blank one.
    pub fn ids(&self, prompt: &str) -> Result<Tensor> {
        let prompt = if prompt.is_empty() { " " } else { prompt };
        let text = format!(
            "{}<|im_start|>user\n{prompt}{}",
            Self::system_turn(&self.config),
            Self::PROMPT_SUFFIX
        );

        let ids: Vec<i64> = self
            .tokenizer
            .encode(&text)?
            .iter()
            .map(|&id| id as i64)
            .collect();
        Ok(Tensor::from_i64(&[ids.len() as i32], &ids)?.to_device(self.device)?)
    }

    /// What the denoiser reads for `prompt`: `(1, L, D)`, the system turn's states dropped.
    pub fn encode_prompt(&self, prompt: &str) -> Result<Tensor> {
        self.encode_ids(&self.ids(prompt)?)
    }

    /// [`QwenImage::encode_prompt`] for ids already in hand, system turn included.
    pub fn encode_ids(&self, ids: &Tensor) -> Result<Tensor> {
        let hidden = self.text_encoder.forward(ids)?;
        let (length, width) = (hidden.shape_at(1)?, hidden.shape_at(3)?);
        let kept = length - self.prefix_length;
        if kept <= 0 {
            return Err(Error::model(
                "the prompt came to nothing past the system turn",
            ));
        }

        Ok(hidden
            .slice(1, self.prefix_length, length)?
            .contiguous()?
            .view(&[1, kept, width])?)
    }

    pub fn vision(&self) -> Option<&VisionTower> {
        self.vision.as_ref()
    }

    /// Whether a prompt may name pictures: the package has the vision tower to read them with.
    /// Packages exported before the tower was read words only.
    pub fn reads_pictures(&self) -> bool {
        self.vision.is_some() && self.encoder.is_some()
    }

    /// The size a named picture `width` by `height` is read at: a megapixel at its own aspect
    /// ratio, each side a multiple of 32 -- the reference's `calculate_dimensions`, Python's
    /// rounding and all. Whatever size the picture is drawn at, the pictures it names are read at
    /// this one; resizing them to it is the caller's, since it is a resize of the picture and not
    /// of a tensor.
    pub fn picture_size(width: i32, height: i32) -> (i32, i32) {
        let ratio = width as f64 / height as f64;
        let wide = (Self::PICTURE_AREA * ratio).sqrt();
        let snap = |side: f64| ((side / 32.0).round_ties_even() as i32 * 32).max(32);
        (snap(wide), snap(wide / ratio))
    }

    /// The template around `prompt` with pictures in front of it, `grids` their slots as rows by
    /// columns: `<image1><|vision_start|>`, a slot per row and column, `<|vision_end|>`, and the
    /// next after a space -- the reference's `prompt_template_ti2i`. Tokenized as one string, as
    /// the processor does once it has written the slots out.
    pub fn ids_with_pictures(&self, prompt: &str, grids: &[(i32, i32)]) -> Result<Vec<i64>> {
        let vision = self.vision_config()?;
        let prompt = if prompt.is_empty() { " " } else { prompt };
        let mut named = String::new();
        for (index, &(rows, columns)) in grids.iter().enumerate() {
            if index > 0 {
                named.push(' ');
            }
            named += &format!("<image{}><|vision_start|>", index + 1);
            named += &"<|image_pad|>".repeat((rows * columns) as usize);
            named += "<|vision_end|>";
        }
        let text = format!(
            "{}<|im_start|>user\n{named}{prompt}{}",
            Self::system_turn(&self.config),
            Self::PROMPT_SUFFIX
        );

        let ids: Vec<i64> = self
            .tokenizer
            .encode(&text)?
            .iter()
            .map(|&id| id as i64)
            .collect();

        // The markers are the package's own tokens, and a vocabulary that split one would leave
        // every picture somewhere else than its rows.
        let count = |token: i64| ids.iter().filter(|&&id| id == token).count();
        let slots = grids
            .iter()
            .map(|(rows, columns)| rows * columns)
            .sum::<i32>();
        if count(vision.image_token_id) != slots as usize
            || count(vision.start_token_id) != grids.len()
            || count(vision.end_token_id) != grids.len()
        {
            return Err(Error::model(format!(
                "the prompt came to {} image slots and {} pictures, where {} pictures of {slots} \
                 slots were written: the prompt cannot name one of the model's markers",
                count(vision.image_token_id),
                count(vision.start_token_id),
                grids.len()
            )));
        }
        Ok(ids)
    }

    fn vision_config(&self) -> Result<&super::VisionConfig> {
        match (&self.config.vision, self.reads_pictures()) {
            (Some(vision), true) => Ok(vision),
            _ => Err(Error::model(
                "this copy of Qwen-Image 2.1 was packaged without the vision tower, so it reads \
                 words only; a package exported since the tower was reads pictures too",
            )),
        }
    }

    /// Read each picture both ways: through the vision tower for the encoder, and through the
    /// autoencoder for the denoiser.
    ///
    /// A picture is `(1, 3, H, W)` or `(1, 4, H, W)` in `-1..=1`, already the size
    /// [`QwenImage::picture_size`] gives it. The vision tower reads three channels -- an alpha is
    /// laid over white first, as the reference does for it -- and the autoencoder four, an opaque
    /// alpha added to a picture that has none.
    fn see(&self, pictures: &[Tensor]) -> Result<Vec<Seen>> {
        let alignment = self.vision_config()?.alignment();
        let (Some(vision), Some(encoder)) = (&self.vision, &self.encoder) else {
            unreachable!("vision_config checked both");
        };
        if pictures.is_empty() || pictures.len() > Self::MAX_PICTURES {
            return Err(Error::model(format!(
                "a prompt names one to {} pictures, and this one names {}",
                Self::MAX_PICTURES,
                pictures.len()
            )));
        }

        let channels = self.config.dit.latent_channels;
        pictures
            .iter()
            .map(|picture| {
                let picture = picture.to_device(self.device)?.cast(DType::Float)?;
                let shape = picture.shape();
                let (rgb, rgba) = match shape[..] {
                    [1, 3, _, _] => (picture.clone(), opaque(&picture)?),
                    [1, 4, _, _] => (over_white(&picture)?, picture.clone()),
                    _ => {
                        return Err(Error::model(format!(
                            "a picture is (1, 3, H, W) or (1, 4, H, W), got {shape:?}"
                        )))
                    }
                };
                if shape[2] % alignment != 0 || shape[3] % alignment != 0 {
                    return Err(Error::model(format!(
                        "a picture the prompt names is read with its sides multiples of \
                         {alignment}, and this one is {} by {}",
                        shape[3], shape[2]
                    )));
                }

                let seen = vision.forward(&rgb)?;
                let latent = encoder.forward(&rgba.cast(self.dtype)?)?;
                let (rows, columns) = (latent.shape_at(2)?, latent.shape_at(3)?);
                if rows != 2 * seen.rows || columns != 2 * seen.columns {
                    return Err(Error::model(format!(
                        "a {rows} x {columns} latent for {} x {} slots: each slot is to be 2 x 2 \
                         latent pixels",
                        seen.rows, seen.columns
                    )));
                }
                let latent = latent
                    .view(&[1, channels, rows * columns])?
                    .transpose(1, 2)?
                    .contiguous()?;
                Ok(Seen {
                    vision: seen,
                    latent,
                    rows,
                    columns,
                })
            })
            .collect()
    }

    /// What the denoiser reads for `prompt` naming `pictures`, in the order the user's turn
    /// names them: the picture at index `i` is `<image{i + 1}>`.
    pub fn encode_prompt_with_pictures(
        &self,
        prompt: &str,
        pictures: &[Tensor],
    ) -> Result<Prompted> {
        let seen = self.see(pictures)?;
        self.encode_seen(prompt, &seen)
    }

    fn encode_seen(&self, prompt: &str, seen: &[Seen]) -> Result<Prompted> {
        let vision = self.vision_config()?;
        let grids = seen
            .iter()
            .map(|picture| (picture.vision.rows, picture.vision.columns))
            .collect::<Vec<_>>();
        let ids = self.ids_with_pictures(prompt, &grids)?;
        let (positions, slots) = place(&ids, &grids, vision.image_token_id)?;

        let rows = |each: &dyn Fn(&Seen) -> &Tensor| -> Result<Tensor> {
            let mut parts = seen.iter().map(each);
            let first = parts
                .next()
                .expect("a prompt names at least one picture")
                .clone();
            parts.try_fold(first, |joined, part| Ok(F::cat(&joined, part, 0)?))
        };
        let images = EncoderImages {
            embeds: rows(&|picture| &picture.vision.embeds)?,
            deepstack: (0..vision.deepstack_layers.len())
                .map(|layer| rows(&|picture: &Seen| &picture.vision.deepstack[layer]))
                .collect::<Result<Vec<_>>>()?,
            positions,
            slots: slots.clone(),
        };

        let length = ids.len() as i32;
        let ids = Tensor::from_i64(&[length], &ids)?.to_device(self.device)?;
        let hidden = self.text_encoder.forward_with_images(&ids, &images)?;
        let width = hidden.shape_at(3)?;
        let hidden = hidden.contiguous()?.view(&[1, length, width])?;

        // The words the denoiser reads and the pictures between them, the system turn dropped.
        let mut layout = Vec::new();
        let mut words: Option<Tensor> = None;
        let mut picture = 0;
        let mut at = self.prefix_length as usize;
        while at < slots.len() {
            if slots[at] {
                let (rows, columns) = grids[picture];
                layout.push(Segment::Picture {
                    rows: seen[picture].rows,
                    columns: seen[picture].columns,
                });
                at += (rows * columns) as usize;
                picture += 1;
                continue;
            }
            let start = at;
            while at < slots.len() && !slots[at] {
                at += 1;
            }
            layout.push(Segment::Words((at - start) as i32));
            let part = hidden.slice(1, start as i32, at as i32)?;
            words = Some(match words {
                Some(joined) => F::cat(&joined, &part, 1)?,
                None => part.contiguous()?,
            });
        }

        let mut parts = seen.iter().map(|picture| &picture.latent);
        let first = parts
            .next()
            .expect("a prompt names at least one picture")
            .clone();
        let tokens = parts.try_fold(first, |joined, part| F::cat(&joined, part, 1))?;

        Ok(Prompted {
            context: words
                .ok_or_else(|| Error::model("the prompt came to no words past the system turn"))?
                .cast(self.dtype)?,
            pictures: Some(Pictures { layout, tokens }),
        })
    }

    /// The schedule for a picture `width` by `height`, which depends on its size -- unless the
    /// package fixes its own levels (Turbo's eight), when `num_steps` and the size are ignored.
    pub fn sampler(&self, width: i32, height: i32, num_steps: i32) -> Result<FlowSampler> {
        if let Some(sigmas) = &self.config.schedule.sigmas {
            let config = SamplerConfig {
                shift: 1.0,
                multiplier: 1.0,
            };
            return FlowSampler::from_sigmas(&config, sigmas);
        }
        let scale = self.config.vae.scale;
        let tokens = (width / scale) * (height / scale);
        let config = SamplerConfig {
            shift: self.config.schedule.shift_for(tokens),
            multiplier: 1.0,
        };
        FlowSampler::new(&config, num_steps)?.stretched(self.config.schedule.shift_terminal)
    }

    /// An image for `prompt`, as `<float>(1, 3, height, width)` in roughly `[-1, 1]`: the four
    /// channels the model draws, composited over white.
    pub fn generate(&self, prompt: &str, options: &GenerationOptions) -> Result<Tensor> {
        let image = self.generate_reporting(prompt, options, &mut unwatched)?;
        Ok(image.expect("a run nothing asked to stop runs to the end"))
    }

    /// [`QwenImage::generate`] for a caller who wants to watch it happen.
    pub fn generate_reporting(
        &self,
        prompt: &str,
        options: &GenerationOptions,
        report: &mut dyn FnMut(GenerationProgress) -> ControlFlow<()>,
    ) -> Result<Option<Tensor>> {
        let Some(latent) = self.generate_latent_reporting(prompt, options, None, report)? else {
            return Ok(None);
        };

        if report(GenerationProgress::Decoding).is_break() {
            return Ok(None);
        }
        over_white(&self.decode(&latent)?).map(Some)
    }

    /// The picture with its alpha, `<float>(1, 4, height, width)`, for a caller who wants the
    /// transparency the model can draw.
    pub fn generate_rgba(&self, prompt: &str, options: &GenerationOptions) -> Result<Tensor> {
        self.decode(&self.generate_latent(prompt, options)?)
    }

    /// The latent [`QwenImage::generate`] would decode.
    pub fn generate_latent(&self, prompt: &str, options: &GenerationOptions) -> Result<Tensor> {
        let latent = self.generate_latent_reporting(prompt, options, None, &mut unwatched)?;
        Ok(latent.expect("a run nothing asked to stop runs to the end"))
    }

    /// Whether a control package was read, so that [`QwenImage::generate_controlled`] can run.
    pub fn has_control(&self) -> bool {
        self.dit.control().is_some()
    }

    /// A control picture as the control chain reads it: `picture` is `(1, 3, H, W)` or
    /// `(1, 4, H, W)` in `-1..=1`, already the size the picture is to be drawn at -- a pose, a
    /// depth map, an edge map, whatever the preprocessor made -- and what comes back is the
    /// `(1, 129, H / 16, W / 16)` [`Dit::forward_controlled`] takes.
    ///
    /// The three channels are given an opaque alpha, encoded, and followed by a mask of zeros and
    /// a latent of zeros: nothing is kept, which is how the reference steers a picture drawn from
    /// noise rather than one being inpainted.
    pub fn control_conditioning(&self, picture: &Tensor) -> Result<Tensor> {
        let Some(encoder) = &self.encoder else {
            return Err(Error::model(
                "this model was read without a control package, so it has no control chain",
            ));
        };

        let picture = picture.to_device(self.device)?.cast(DType::Float)?;
        let shape = picture.shape();
        let rgba = match shape.as_slice() {
            [1, 3, _, _] => opaque(&picture)?,
            [1, 4, _, _] => picture,
            _ => {
                return Err(Error::model(format!(
                    "a control picture is (1, 3, H, W) or (1, 4, H, W), got {shape:?}"
                )))
            }
        };
        self.check_size(shape[3], shape[2])?;

        let latent = encoder.forward(&rgba.cast(self.dtype)?)?;
        let (rows, columns) = (latent.shape_at(2)?, latent.shape_at(3)?);
        let channels = self.config.dit.latent_channels;
        let kept = Tensor::zeros(&[1, channels + 1, rows, columns], DType::Float, self.device)?;
        Ok(F::cat(&latent.cast(DType::Float)?, &kept, 1)?)
    }

    /// [`QwenImage::generate`] steered by a control picture: `conditioning` is what
    /// [`QwenImage::control_conditioning`] made of it, and `scale` how hard it steers -- one is
    /// what the checkpoint was trained at, zero is no steering at all.
    pub fn generate_controlled(
        &self,
        prompt: &str,
        options: &GenerationOptions,
        conditioning: &Tensor,
        scale: f32,
    ) -> Result<Tensor> {
        let image = self.generate_controlled_reporting(
            prompt,
            options,
            conditioning,
            scale,
            &mut unwatched,
        )?;
        Ok(image.expect("a run nothing asked to stop runs to the end"))
    }

    /// [`QwenImage::generate_controlled`] for a caller who wants to watch it happen.
    pub fn generate_controlled_reporting(
        &self,
        prompt: &str,
        options: &GenerationOptions,
        conditioning: &Tensor,
        scale: f32,
        report: &mut dyn FnMut(GenerationProgress) -> ControlFlow<()>,
    ) -> Result<Option<Tensor>> {
        let control = Some((conditioning, scale));
        let Some(latent) = self.generate_latent_reporting(prompt, options, control, report)? else {
            return Ok(None);
        };

        if report(GenerationProgress::Decoding).is_break() {
            return Ok(None);
        }
        over_white(&self.decode(&latent)?).map(Some)
    }

    /// An image for `prompt` naming `pictures` -- to edit one, or to draw from several -- as
    /// [`QwenImage::generate`] returns one. The picture at index `i` is `<image{i + 1}>` to the
    /// model, which is how the prompt can tell them apart; each is `(1, 3, H, W)` or
    /// `(1, 4, H, W)` in `-1..=1`, already the size [`QwenImage::picture_size`] gives it.
    pub fn generate_with_pictures(
        &self,
        prompt: &str,
        pictures: &[Tensor],
        options: &GenerationOptions,
    ) -> Result<Tensor> {
        let image =
            self.generate_with_pictures_reporting(prompt, pictures, options, &mut unwatched)?;
        Ok(image.expect("a run nothing asked to stop runs to the end"))
    }

    /// [`QwenImage::generate_with_pictures`] for a caller who wants to watch it happen.
    pub fn generate_with_pictures_reporting(
        &self,
        prompt: &str,
        pictures: &[Tensor],
        options: &GenerationOptions,
        report: &mut dyn FnMut(GenerationProgress) -> ControlFlow<()>,
    ) -> Result<Option<Tensor>> {
        let Some(latent) =
            self.generate_latent_with_pictures_reporting(prompt, pictures, options, report)?
        else {
            return Ok(None);
        };

        if report(GenerationProgress::Decoding).is_break() {
            return Ok(None);
        }
        over_white(&self.decode(&latent)?).map(Some)
    }

    /// The latent [`QwenImage::generate_with_pictures`] would decode.
    pub fn generate_latent_with_pictures(
        &self,
        prompt: &str,
        pictures: &[Tensor],
        options: &GenerationOptions,
    ) -> Result<Tensor> {
        let latent = self.generate_latent_with_pictures_reporting(
            prompt,
            pictures,
            options,
            &mut unwatched,
        )?;
        Ok(latent.expect("a run nothing asked to stop runs to the end"))
    }

    fn generate_latent_with_pictures_reporting(
        &self,
        prompt: &str,
        pictures: &[Tensor],
        options: &GenerationOptions,
        report: &mut dyn FnMut(GenerationProgress) -> ControlFlow<()>,
    ) -> Result<Option<Tensor>> {
        self.vision_config()?;
        let latent = self.noise(options)?;
        let sampler = self.sampler(options.width, options.height, options.num_steps)?;

        if report(GenerationProgress::Encoding).is_break() {
            return Ok(None);
        }
        // Read once, and named by both prompts: the reference hands the negative prompt the
        // same pictures.
        let seen = self.see(pictures)?;
        let prompted = self.encode_seen(prompt, &seen)?;
        let unprompted = match options.guidance_scale != 1.0 {
            true => Some(self.encode_seen(&options.negative_prompt, &seen)?),
            false => None,
        };
        // The vision rows are the encoder's and are done with; only the latents walk on.
        drop(seen);

        self.denoise_with_pictures_reporting(
            &latent,
            &sampler,
            &prompted,
            unprompted.as_ref(),
            options,
            report,
        )
    }

    /// The latent [`QwenImage::generate_controlled`] would decode.
    pub fn generate_controlled_latent(
        &self,
        prompt: &str,
        options: &GenerationOptions,
        conditioning: &Tensor,
        scale: f32,
    ) -> Result<Tensor> {
        let control = Some((conditioning, scale));
        let latent = self.generate_latent_reporting(prompt, options, control, &mut unwatched)?;
        Ok(latent.expect("a run nothing asked to stop runs to the end"))
    }

    /// The latent a run starts from, seeded as `options` says.
    fn noise(&self, options: &GenerationOptions) -> Result<Tensor> {
        self.check_size(options.width, options.height)?;
        if let Some(seed) = options.seed {
            F::manual_seed(self.device, seed)?;
        }

        let scale = self.config.vae.scale;
        Ok(F::randn(
            &[
                1,
                self.config.dit.latent_channels,
                options.height / scale,
                options.width / scale,
            ],
            self.device,
        )?
        .cast(self.dtype)?)
    }

    fn generate_latent_reporting(
        &self,
        prompt: &str,
        options: &GenerationOptions,
        control: Option<(&Tensor, f32)>,
        report: &mut dyn FnMut(GenerationProgress) -> ControlFlow<()>,
    ) -> Result<Option<Tensor>> {
        let latent = self.noise(options)?;
        let sampler = self.sampler(options.width, options.height, options.num_steps)?;

        if report(GenerationProgress::Encoding).is_break() {
            return Ok(None);
        }
        let context = self.encode_prompt(prompt)?.cast(self.dtype)?;
        let unprompted = match options.guidance_scale != 1.0 {
            true => Some(
                self.encode_prompt(&options.negative_prompt)?
                    .cast(self.dtype)?,
            ),
            false => None,
        };

        self.denoise_reporting(
            &latent,
            &sampler,
            &context,
            unprompted.as_ref(),
            control,
            options,
            report,
        )
    }

    /// Walk `latent` down `sampler`'s schedule, steered by `control` -- a conditioning and its
    /// scale -- when there is one. None where `report` said to stop.
    #[allow(clippy::too_many_arguments)]
    pub fn denoise_reporting(
        &self,
        latent: &Tensor,
        sampler: &FlowSampler,
        context: &Tensor,
        unprompted: Option<&Tensor>,
        control: Option<(&Tensor, f32)>,
        options: &GenerationOptions,
        report: &mut dyn FnMut(GenerationProgress) -> ControlFlow<()>,
    ) -> Result<Option<Tensor>> {
        // Both passes of a guided step are steered, as the reference steers them.
        let answer = |latent: &Tensor, timestep: f32, plain: bool| {
            let context = match (plain, unprompted) {
                (true, Some(unprompted)) => unprompted,
                _ => context,
            };
            match control {
                Some((conditioning, scale)) => {
                    self.dit
                        .forward_controlled(latent, timestep, context, conditioning, scale)
                }
                None => self.dit.forward(latent, timestep, context),
            }
        };
        let guidance = unprompted.map(|_| options.guidance_scale);
        self.walk(latent, sampler, guidance, &answer, report)
    }

    /// [`QwenImage::denoise_reporting`] for a prompt that names pictures, and with `unprompted`
    /// the negative prompt naming the same ones.
    pub fn denoise_with_pictures_reporting(
        &self,
        latent: &Tensor,
        sampler: &FlowSampler,
        prompted: &Prompted,
        unprompted: Option<&Prompted>,
        options: &GenerationOptions,
        report: &mut dyn FnMut(GenerationProgress) -> ControlFlow<()>,
    ) -> Result<Option<Tensor>> {
        let answer = |latent: &Tensor, timestep: f32, plain: bool| {
            let prompted = match (plain, unprompted) {
                (true, Some(unprompted)) => unprompted,
                _ => prompted,
            };
            match &prompted.pictures {
                Some(pictures) => {
                    self.dit
                        .forward_with_pictures(latent, timestep, &prompted.context, pictures)
                }
                None => self.dit.forward(latent, timestep, &prompted.context),
            }
        };
        let guidance = unprompted.map(|_| options.guidance_scale);
        self.walk(latent, sampler, guidance, &answer, report)
    }

    /// The walk itself: `answer` is the velocity for a latent at a timestep, of the negative
    /// prompt where its last argument says so, and `guidance` the scale the two are mixed at when
    /// there is a negative prompt at all.
    fn walk(
        &self,
        latent: &Tensor,
        sampler: &FlowSampler,
        guidance: Option<f32>,
        answer: &dyn Fn(&Tensor, f32, bool) -> Result<Tensor>,
        report: &mut dyn FnMut(GenerationProgress) -> ControlFlow<()>,
    ) -> Result<Option<Tensor>> {
        let total = sampler.steps() as i32;
        let mut latent = latent.clone();

        for index in 0..sampler.steps() {
            let timestep = model_timestep(sampler.sigma(index)?);
            let velocity = answer(&latent, timestep, false)?;

            // `true_cfg_scale`: two passes, and a scale of one means none.
            let velocity = match guidance {
                None => velocity,
                Some(guidance) => {
                    let plain = answer(&latent, timestep, true)?;
                    let difference = F::sub(&velocity, &plain)?;
                    F::add(&plain, &F::mul_scalar(&difference, guidance)?)?
                }
            };

            latent = sampler.step(index, &latent, &velocity)?.cast(self.dtype)?;

            if report(GenerationProgress::Step {
                done: index as i32 + 1,
                total,
            })
            .is_break()
            {
                return Ok(None);
            }
        }

        Ok(Some(latent))
    }

    /// The four channels a latent stands for, `(1, 4, H * 16, W * 16)`, unclamped.
    pub fn decode(&self, latent: &Tensor) -> Result<Tensor> {
        self.vae.forward(&latent.cast(self.dtype)?)
    }

    /// Refuses a size the model cannot draw: each side a whole number of latent pixels, and the
    /// latent a whole number of the encoder's 2 by 2 image slots.
    fn check_size(&self, width: i32, height: i32) -> Result<()> {
        let alignment = 2 * self.config.vae.scale;
        if width <= 0 || height <= 0 || width % alignment != 0 || height % alignment != 0 {
            return Err(Error::model(format!(
                "{width} by {height} is not a multiple of {alignment}, which is what this model \
                 works in"
            )));
        }
        Ok(())
    }
}

/// `(1, 4, H, W)` in `-1..=1` to `(1, 3, H, W)`, the colour laid over white by the alpha.
///
/// What the rest of the runtime draws is three channels. A picture the model drew opaque comes
/// through unchanged; one it drew with a transparent background comes out on white, which is how
/// the reference flattens an RGBA picture before its own encoder reads it.
fn over_white(image: &Tensor) -> Result<Tensor> {
    let shape = image.shape();
    if shape.len() != 4 || shape[1] != 4 {
        return Err(Error::model(format!(
            "an RGBA image is (1, 4, H, W), got {shape:?}"
        )));
    }

    let plane = (shape[2] * shape[3]) as usize;
    let values = image
        .contiguous()?
        .to_device(Device::Cpu)?
        .cast(DType::Float)?
        .to_vec_f32()?;

    let unit = |value: f32| (value / 2.0 + 0.5).clamp(0.0, 1.0);
    let mut out = Vec::with_capacity(plane * 3);
    for channel in 0..3 {
        for pixel in 0..plane {
            let alpha = unit(values[3 * plane + pixel]);
            let colour = unit(values[channel * plane + pixel]);
            out.push((colour * alpha + (1.0 - alpha)) * 2.0 - 1.0);
        }
    }

    Ok(Tensor::from_f32(&[1, 3, shape[2], shape[3]], &out)?.to_device(image.device())?)
}

/// `(1, 3, H, W)` with an opaque alpha after it, which is what the autoencoder reads a picture
/// that has none as.
fn opaque(picture: &Tensor) -> Result<Tensor> {
    let (height, width) = (picture.shape_at(2)?, picture.shape_at(3)?);
    let ones = vec![1.0; (height * width) as usize];
    let alpha = Tensor::from_f32(&[1, 1, height, width], &ones)?.to_device(picture.device())?;
    Ok(F::cat(picture, &alpha, 1)?)
}

/// Where each of `ids` sits for the encoder, and which are pictures' slots: `get_rope_index`.
///
/// Words advance along all three axes together. A picture's slots, `rows` by `columns` of them,
/// are a grid at the frame the words reached -- `(s, s + row, s + column)` -- and the words after
/// it start `max(rows, columns)` further on.
fn place(ids: &[i64], grids: &[(i32, i32)], slot: i64) -> Result<(Vec<[i32; 3]>, Vec<bool>)> {
    let mut positions = Vec::with_capacity(ids.len());
    let mut slots = Vec::with_capacity(ids.len());
    let mut grids = grids.iter();
    let mut frame = 0;
    let mut at = 0;
    while at < ids.len() {
        if ids[at] != slot {
            positions.push([frame, frame, frame]);
            slots.push(false);
            frame += 1;
            at += 1;
            continue;
        }
        let Some(&(rows, columns)) = grids.next() else {
            return Err(Error::model(
                "the prompt has more pictures' slots than pictures",
            ));
        };
        let count = (rows * columns) as usize;
        if ids.len() < at + count || ids[at..at + count].iter().any(|&id| id != slot) {
            return Err(Error::model(format!(
                "a picture of {rows} x {columns} slots does not have that many in a row"
            )));
        }
        for row in 0..rows {
            for column in 0..columns {
                positions.push([frame, frame + row, frame + column]);
                slots.push(true);
            }
        }
        frame += rows.max(columns);
        at += count;
    }
    if grids.next().is_some() {
        return Err(Error::model(
            "the prompt has fewer pictures' slots than pictures",
        ));
    }
    Ok((positions, slots))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_a_named_picture_at_a_megapixel() {
        // `calculate_dimensions(1024 * 1024, ratio)`: a square stays 1024; a 4:3 photograph is
        // 1182 x 886.8 before rounding to 32, and Python rounds 36.94 and 27.71 to 37 and 28.
        assert_eq!(QwenImage::picture_size(512, 512), (1024, 1024));
        assert_eq!(QwenImage::picture_size(4000, 3000), (1184, 896));
        assert_eq!(QwenImage::picture_size(3000, 4000), (896, 1184));
        assert_eq!(QwenImage::picture_size(1920, 1080), (1376, 768));
    }

    #[test]
    fn places_slots_on_a_grid_the_way_get_rope_index_does() {
        const SLOT: i64 = 9;
        // Two words, a picture of 2 x 3 slots, one word.
        let ids = [1, 2, SLOT, SLOT, SLOT, SLOT, SLOT, SLOT, 3];
        let (positions, slots) = place(&ids, &[(2, 3)], SLOT).unwrap();
        assert_eq!(&positions[..2], &[[0, 0, 0], [1, 1, 1]]);
        assert_eq!(&positions[2..5], &[[2, 2, 2], [2, 2, 3], [2, 2, 4]]);
        assert_eq!(positions[5], [2, 3, 2]);
        // The word after it is max(2, 3) on from where the picture started.
        assert_eq!(positions[8], [5, 5, 5]);
        assert_eq!(slots.iter().filter(|&&slot| slot).count(), 6);

        assert!(place(&ids, &[(2, 2)], SLOT).is_err());
        assert!(place(&ids, &[(2, 3), (1, 1)], SLOT).is_err());
    }

    #[test]
    fn rounds_to_bfloat16_the_way_torch_does() {
        assert_eq!(bfloat16(1.0), 1.0);
        // 987.3 lies between 984 and 988, which are neighbours in bfloat16.
        assert_eq!(bfloat16(987.3), 988.0);
        assert_eq!(bfloat16(985.9), 984.0);
        // Ties go to the even mantissa: 986 is halfway and 984's mantissa is even.
        assert_eq!(bfloat16(986.0), 984.0);
    }

    #[test]
    fn hands_the_model_the_timestep_the_reference_does() {
        assert_eq!(model_timestep(1.0), 1.0);
        assert_eq!(model_timestep(0.0), 0.0);
        // bf16(987.3) is 988, and 0.988 in bfloat16 is 0.98828125.
        assert_eq!(model_timestep(0.9873), 0.988_281_25);
    }
}
