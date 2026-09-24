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

//! The four pieces put together: a prompt in, an image out.
//!
//! Everything here is assembly. The prompt is tokenized twice and read by two encoders, the
//! sampler walks a latent from noise to none by asking the U-Net what is in it, and the VAE turns
//! what is left into pixels. The one piece of arithmetic that lives nowhere else is classifier
//! free guidance: every step is run twice, once with the prompt and once without, and the answer
//! is pushed away from the one that ignored it.

use std::ops::ControlFlow;
use std::rc::Rc;

use crate::error::{Error, Result};
use crate::flint::{
    functional as F, DType, Device, ParamSource, Residency, Tensor, WeightFormat, Weights,
};
use crate::generation::{unwatched, GenerationOptions, GenerationProgress};
use crate::manifest::Manifest;
use crate::mapping::Mapping;
use crate::tokenizer::Tokenizer;

use super::sampler::{EulerSampler, SamplerConfig};
use super::text_encoder::{ClipTextConfig, ClipTextEncoder};
use super::unet::{Unet, UnetCondition, UnetConfig};
use super::vae::{VaeConfig, VaeDecoder, VaeEncoder};

/// The one clip skip these encoders implement: conditioning on the layer before the last.
const SUPPORTED_CLIP_SKIP: i32 = 2;

/// How much smaller a latent is than the image it stands for, on each axis.
pub const VAE_SCALE: i32 = 8;

/// The epsilon every diffusers autoencoder normalizes with. Not in the config there either, so it
/// is not in the package.
const VAE_NORM_EPS: f32 = 1e-6;

/// Everything a package says about an SDXL model.
#[derive(Clone, Debug)]
pub struct SdxlConfig {
    pub text: ClipTextConfig,
    pub text2: ClipTextConfig,
    pub unet: UnetConfig,
    pub vae: VaeConfig,
    pub sampler: SamplerConfig,
    /// How many tokens each encoder reads, which is the hard limit on a prompt.
    pub context_length: i32,
    pub bot_token_id: i32,
    pub eot_token_id: i32,
    /// What the first encoder pads with, which for CLIP is the end marker again.
    pub pad_token_id: i32,
    /// What the second pads with, which is not the same id.
    pub pad_token_id2: i32,
}

/// A comma separated list of numbers, as the package writes shapes that vary per resolution.
fn number_list(section: &Mapping, key: &str) -> Result<Vec<i32>> {
    section
        .get_str(key)?
        .split(',')
        .map(|part| {
            part.trim().parse::<i32>().map_err(|_| {
                Error::model(format!("{key} is not a list of numbers: {:?}", part.trim()))
            })
        })
        .collect()
}

impl SdxlConfig {
    pub fn from_section(section: &Mapping) -> Result<SdxlConfig> {
        let clip_skip: i32 = section.get_or("clip_skip", SUPPORTED_CLIP_SKIP)?;
        if clip_skip != SUPPORTED_CLIP_SKIP {
            return Err(Error::model(format!(
                "a clip skip of {clip_skip} is not the {SUPPORTED_CLIP_SKIP} these encoders read"
            )));
        }

        let context_length = section.get("context_length")?;

        // One key for the halves that have the matrices in them. The autoencoder does not read it:
        // it is convolutions, its few projections are small, and this half runs in float32 where
        // the CUDA FP8 multiply takes float16.
        let weight_format = section.get_weight_format_or("weight_format", WeightFormat::Float)?;
        let vocab_size = section.get("vocab_size")?;
        let eot_token_id = section.get("eot_token_id")?;

        let text_encoder = |prefix: &str, quick_gelu: bool| -> Result<ClipTextConfig> {
            Ok(ClipTextConfig {
                hidden_size: section.get(&format!("{prefix}_hidden_size"))?,
                intermediate_size: section.get(&format!("{prefix}_intermediate_size"))?,
                num_layers: section.get(&format!("{prefix}_num_layers"))?,
                num_heads: section.get(&format!("{prefix}_num_heads"))?,
                context_length,
                vocab_size,
                quick_gelu,
                norm_eps: section.get(&format!("{prefix}_norm_eps"))?,
                eot_token_id,
                weight_format,
            })
        };

        // The two differ in their activation, which the package names rather than assumes.
        let activation = |prefix: &str| -> Result<bool> {
            match section.get_str(&format!("{prefix}_hidden_act"))? {
                "quick_gelu" => Ok(true),
                "gelu" => Ok(false),
                other => Err(Error::model(format!(
                    "{prefix} activates with {other:?}, which is neither gelu nor quick_gelu"
                ))),
            }
        };

        Ok(SdxlConfig {
            text: text_encoder("text", activation("text")?)?,
            text2: text_encoder("text2", activation("text2")?)?,
            unet: UnetConfig {
                latent_channels: section.get("latent_channels")?,
                block_out_channels: number_list(section, "unet_block_out_channels")?,
                layers_per_block: section.get("unet_layers_per_block")?,
                transformer_layers_per_block: number_list(
                    section,
                    "unet_transformer_layers_per_block",
                )?,
                // diffusers calls this the head dimension and stores the head count in it, which
                // is a mistake old enough that every SDXL checkpoint now depends on it.
                num_heads: number_list(section, "unet_attention_head_dim")?,
                norm_num_groups: section.get("unet_norm_num_groups")?,
                cross_attention_dim: section.get("unet_cross_attention_dim")?,
                addition_time_embed_dim: section.get("unet_addition_time_embed_dim")?,
                projection_class_embeddings_input_dim: section
                    .get("unet_projection_class_embeddings_input_dim")?,
                weight_format,
            },
            vae: VaeConfig {
                latent_channels: section.get("latent_channels")?,
                block_out_channels: number_list(section, "vae_block_out_channels")?,
                layers_per_block: section.get("vae_layers_per_block")?,
                norm_num_groups: section.get("vae_norm_num_groups")?,
                norm_eps: VAE_NORM_EPS,
                scaling_factor: section.get("vae_scaling_factor")?,
            },
            sampler: SamplerConfig {
                num_train_timesteps: section.get("scheduler_num_train_timesteps")?,
                beta_start: section.get("scheduler_beta_start")?,
                beta_end: section.get("scheduler_beta_end")?,
                steps_offset: section.get("scheduler_steps_offset")?,
            },
            context_length,
            bot_token_id: section.get("bot_token_id")?,
            eot_token_id,
            pad_token_id: section.get("pad_token_id")?,
            pad_token_id2: section.get("pad_token_id2")?,
        })
    }
}

/// The context the VAE encoder is built from, or None where the package has no encoder in it.
///
/// One tensor is enough to tell the two apart: a package either has the whole half or none of it,
/// and a package with some of it is a broken file rather than an old one, which the modules below
/// will say better than a check here could.
fn has_encoder(weights: &dyn ParamSource, name: &str) -> bool {
    weights.has(&format!("{name}.conv_in.weight"))
}

/// What a prompt becomes once both encoders have read it.
pub struct PromptEmbedding {
    /// `(1, L, 2048)`: the two encoders side by side, which is what cross attention reads.
    pub context: Tensor,
    /// `(1, 1280)`: the pooled vector of the second, which the timestep embedding is added to.
    pub pooled: Tensor,
}

pub struct Sdxl {
    config: SdxlConfig,
    /// Every parameter of the whole model, which its four halves share. One store rather than
    /// four is what lets a question about the model -- how much it costs to hold, and one day
    /// what of it should be on the device right now -- be asked once.
    tokenizer: Tokenizer,
    text_encoder: ClipTextEncoder,
    text_encoder2: ClipTextEncoder,
    unet: Unet,
    vae: VaeDecoder,
    /// The other half of the autoencoder, which only a package exported since image to image was
    /// added carries. None is not a broken model: it draws from noise like any other.
    vae_encoder: Option<VaeEncoder>,
    device: Device,
    dtype: DType,
}

/// Refuse a package that holds something other than SDXL.
///
/// Asked before the configuration is read, because reading it is what used to do the asking. The
/// type doubles as the name of the section the configuration is in, so for as long as there was
/// one kind of model this was self-enforcing: a package that was not SDXL had no section to read.
/// An Anima package has one, and the keys in it are not these, so what came back was a complaint
/// about `vocab_size` -- the first key SDXL has and Anima does not -- rather than about the model.
///
/// Reaching this is now a caller that asked the wrong half of the crate: `waifu webui` looks at the
/// same key first and takes an Anima package to [`crate::Anima`], so what is left here is a
/// program that named this type itself.
fn check_model_type(model_type: &str) -> Result<()> {
    if model_type != Sdxl::MODEL_TYPE {
        return Err(Error::model(format!(
            "this package holds a model of kind {model_type:?}, which {:?} does not read. \
             {}::from_package is what reads a {model_type} one",
            Sdxl::MODEL_TYPE,
            model_type
        )));
    }
    Ok(())
}

impl Sdxl {
    /// The block of the manifest's `config:` that says what the model is.
    pub const MODEL_SECTION: &'static str = "model";

    /// What `model type` says for a model this can read.
    ///
    /// It doubles as the name of the section the configuration is in, which is why it went
    /// unchecked for as long as there was only one kind of model: reading the section named by
    /// the type and reading the section named `sdxl` were the same thing.
    pub const MODEL_TYPE: &'static str = "sdxl";

    /// Read the whole model `manifest` describes, onto `device`, with its weights waiting where
    /// `residency` says.
    ///
    /// The manifest says what the model is and which packages the weights are in. Which package a
    /// tensor was written to is not something the model has to know: they are read in the order
    /// they are named, into one namespace.
    ///
    /// [`Residency::LowVram`] is the one to reach for when the card has less memory on it than the
    /// weights have bytes. It draws the same picture, a good deal more slowly, and nothing below
    /// this line knows which of the two it is running under.
    pub fn from_manifest(
        device: Device,
        residency: Residency,
        manifest: &Manifest,
    ) -> Result<Sdxl> {
        let model_section = manifest.section(Self::MODEL_SECTION)?;
        let model_type = model_section.get_str("type")?.to_string();

        check_model_type(&model_type)?;

        let config = SdxlConfig::from_section(manifest.section(&model_type)?)?;

        // The weights, read in the order the manifest names them: each file is read through once,
        // in full, and they read into one namespace.
        let dtype = F::default_float_type(device)?;
        // The whole package, read once, and put where the residency says it waits. The four halves
        // are written against it and share it: a weight is found by the name the package holds it
        // under, and the four namespaces below are what keeps them apart.
        let weights: Rc<dyn ParamSource> = Rc::new(Weights::from_files(
            &manifest.weight_paths()?,
            device,
            residency,
        )?);

        let named = |half: &str| format!("{model_type}.{half}");
        let vae = named("vae");
        let vae_encoder = named("vae.encoder");

        Ok(Sdxl {
            text_encoder: ClipTextEncoder::build(
                config.text,
                &named("text_encoder"),
                &weights,
                dtype,
            )?,
            text_encoder2: ClipTextEncoder::build(
                config.text2,
                &named("text_encoder2"),
                &weights,
                dtype,
            )?,
            unet: Unet::build(config.unet.clone(), &named("unet"), &weights)?,
            // The autoencoder alone runs in float32. It is marked force_upcast and really does
            // need it -- see VaeDecoder::forward -- and the exporter writes its weights that way,
            // so this is the file's own precision rather than a widening of it.
            vae: VaeDecoder::build(config.vae.clone(), &vae, &weights, device, DType::Float)?,
            // Read only where it is. Packages published before image to image hold the decoder
            // alone, and refusing to load one of those would take text to image away from every
            // model already on disk to add a mode it was never going to be asked for.
            vae_encoder: match has_encoder(weights.as_ref(), &vae_encoder) {
                true => Some(VaeEncoder::build(
                    config.vae.clone(),
                    &vae_encoder,
                    &weights,
                    device,
                    DType::Float,
                )?),
                false => None,
            },
            tokenizer: Tokenizer::open(manifest)?,
            config,
            device,
            dtype,
        })
    }

    pub fn config(&self) -> &SdxlConfig {
        &self.config
    }

    pub fn device(&self) -> Device {
        self.device
    }

    /// The U-Net alone, for a caller that wants to ask it something the pipeline does not.
    pub fn unet(&self) -> &Unet {
        &self.unet
    }

    /// The ids one encoder reads: the prompt between its two markers, padded out to the context
    /// length. A prompt too long for that is cut rather than refused, which is what every other
    /// implementation does and what a prompt of a hundred tags will hit.
    fn token_ids(&self, text: &str, pad_token_id: i32) -> Result<Vec<i64>> {
        let length = self.config.context_length as usize;
        let mut ids = vec![pad_token_id as i64; length];

        ids[0] = self.config.bot_token_id as i64;
        let mut position = 1;
        for id in self.tokenizer.encode(text)? {
            if position + 1 >= length {
                break;
            }
            ids[position] = id as i64;
            position += 1;
        }
        ids[position] = self.config.eot_token_id as i64;

        Ok(ids)
    }

    /// What both encoders make of `text`.
    pub fn encode_prompt(&self, text: &str) -> Result<PromptEmbedding> {
        let length = self.config.context_length;

        let ids = Tensor::from_i64(&[length], &self.token_ids(text, self.config.pad_token_id)?)?
            .to_device(self.device)?;
        let ids2 = Tensor::from_i64(&[length], &self.token_ids(text, self.config.pad_token_id2)?)?
            .to_device(self.device)?;

        let out = self.text_encoder.forward(&ids)?;
        let out2 = self.text_encoder2.forward(&ids2)?;

        Ok(PromptEmbedding {
            context: F::cat(&out.hidden, &out2.hidden, -1)?,
            // Only the second encoder has a projection to pool through, which is why SDXL takes
            // this from it alone.
            pooled: out2.pooled,
        })
    }

    /// Walk `latent` from noise to none, and hand back what is left.
    ///
    /// `latent` is unit noise, `(1, C, H / 8, W / 8)`; the noise level the first step expects is
    /// applied here rather than by the caller. Split out from [`Sdxl::generate`] so that a run
    /// can be started from a known latent instead of a random one.
    pub fn denoise(
        &self,
        latent: &Tensor,
        prompt: &PromptEmbedding,
        negative: &PromptEmbedding,
        options: &GenerationOptions,
    ) -> Result<Tensor> {
        let sampler = EulerSampler::new(&self.config.sampler, options.num_steps)?;
        let latent = F::mul_scalar(latent, sampler.init_noise_sigma())?;
        let denoised =
            self.denoise_reporting(&latent, &sampler, prompt, negative, options, &mut unwatched)?;
        Ok(denoised.expect("a run nothing asked to stop runs to the end"))
    }

    /// [`Sdxl::denoise`], telling `report` after every step and giving up where it stands if
    /// `report` says to -- in which case there is no latent to hand back.
    ///
    /// `latent` is already at the noise level `sampler` begins at, rather than being unit noise:
    /// which schedule this is and where in it the walk starts are the two things that differ
    /// between a run from noise and a run from a picture, and both callers settle them first.
    fn denoise_reporting(
        &self,
        latent: &Tensor,
        sampler: &EulerSampler,
        prompt: &PromptEmbedding,
        negative: &PromptEmbedding,
        options: &GenerationOptions,
        report: &mut dyn FnMut(GenerationProgress) -> ControlFlow<()>,
    ) -> Result<Option<Tensor>> {
        // The size the image was asked for, which SDXL is told directly: it was trained on
        // pictures of many sizes and knows what it is being asked to imitate.
        let time_ids = [
            options.height as f32,
            options.width as f32,
            0.0,
            0.0,
            options.height as f32,
            options.width as f32,
        ];
        // Classifier free guidance asks the model two questions: what it makes of this latent
        // having read the prompt, and what it makes of it having read the negative one instead.
        // Neither answer depends on the other, so the two go through as one batch of two rather
        // than as two passes -- the same arithmetic over one pass of the weights instead of two,
        // which is most of what a step costs on a machine whose memory is the slow part.
        //
        // The unprompted row comes first, and the latents below are stacked in the same order.
        let guided = options.guidance_scale != 1.0;
        let (context, pooled);
        if guided {
            context = F::cat(&negative.context, &prompt.context, 0)?;
            pooled = F::cat(&negative.pooled, &prompt.pooled, 0)?;
        } else {
            context = prompt.context.clone();
            pooled = prompt.pooled.clone();
        }
        let condition = UnetCondition {
            context: &context,
            pooled: &pooled,
            time_ids,
        };

        let mut latent = latent.clone();
        for index in sampler.start()..sampler.len() {
            let timestep = sampler.timesteps()[index];
            let scaled = sampler.scale_model_input(&latent, index)?;

            // The same latent to both rows: what differs between them is the prompt alone.
            let batched = if guided {
                F::cat(&scaled, &scaled, 0)?
            } else {
                scaled
            };
            let answer = self.unet.forward(&batched, timestep, &condition)?;

            // What the model says about this latent without having read the prompt is what it
            // would say about anything, and the difference between the two answers is the part
            // the prompt is responsible for. Amplifying it is what makes a generated image look
            // like what was asked for.
            let noise = if guided {
                let unprompted = answer.slice(0, 0, 1)?;
                let prompted = answer.slice(0, 1, 2)?;
                let difference = F::sub(&prompted, &unprompted)?;
                F::add(
                    &unprompted,
                    &F::mul_scalar(&difference, options.guidance_scale)?,
                )?
            } else {
                answer
            };

            latent = sampler.step(&noise, &latent, index)?;

            // Counted over the steps this run walks rather than over the whole schedule, which
            // for a run from a picture is the tail of a longer one. Both are options.num_steps.
            let progress = GenerationProgress::Step {
                done: (index - sampler.start()) as i32 + 1,
                total: sampler.steps_to_run() as i32,
            };
            if report(progress).is_break() {
                return Ok(None);
            }
        }

        Ok(Some(latent))
    }

    /// The image a latent stands for, as `<float>(1, 3, H * 8, W * 8)` in roughly `[-1, 1]`.
    ///
    /// The autoencoder runs in float32 while everything before it runs in half, so the image is
    /// the one tensor a run hands back in a wider type than the model was loaded in. The latent
    /// is cast on the way in by the decoder itself.
    pub fn decode(&self, latent: &Tensor) -> Result<Tensor> {
        self.vae.forward(latent)
    }

    /// Whether this package carries the other half of the autoencoder.
    ///
    /// Asked before a run rather than during one: a screen that offers image to image for a
    /// package that has no encoder is offering a refusal that arrives after the wait.
    pub fn draws_from_a_picture(&self) -> bool {
        self.vae_encoder.is_some()
    }

    /// The latent an image stands for, as `(1, C, H / 8, W / 8)`, scaled the way the sampler
    /// wants it.
    ///
    /// `image` is `(1, 3, H, W)` in roughly `[-1, 1]`, which is what [`from_rgb8`] produces and
    /// what [`Sdxl::decode`] hands back. Fails on a package that holds no encoder.
    pub fn encode(&self, image: &Tensor) -> Result<Tensor> {
        let Some(encoder) = &self.vae_encoder else {
            return Err(Error::model(
                "this package has no VAE encoder, so it cannot start from a picture. It was \
                 written before image to image existed; exporting it again adds one",
            ));
        };

        encoder.forward(image)
    }

    /// An image for `prompt`, as `<float>(1, 3, height, width)` in roughly `[-1, 1]`.
    pub fn generate(&self, prompt: &str, options: &GenerationOptions) -> Result<Tensor> {
        let image = self.generate_reporting(prompt, options, &mut unwatched)?;
        Ok(image.expect("a run nothing asked to stop runs to the end"))
    }

    /// [`Sdxl::generate`] for a caller who wants to watch it happen.
    ///
    /// `report` hears about each part of the run as it is reached: the prompt before it is read,
    /// every step as it finishes, and the decode before it begins. It can end the run by
    /// returning [`ControlFlow::Break`], which is the only way to stop one -- a step, once
    /// started, runs to the end of itself. A run that stopped early hands back no image.
    pub fn generate_reporting(
        &self,
        prompt: &str,
        options: &GenerationOptions,
        report: &mut dyn FnMut(GenerationProgress) -> ControlFlow<()>,
    ) -> Result<Option<Tensor>> {
        let Some(latent) = self.generate_latent_reporting(prompt, options, report)? else {
            return Ok(None);
        };

        if report(GenerationProgress::Decoding).is_break() {
            return Ok(None);
        }
        self.decode(&latent).map(Some)
    }

    /// The latent [`Sdxl::generate`] would decode: everything but the last step.
    ///
    /// Split out because the latent is what a run is really about -- the decoder only makes it
    /// visible -- and because it is what to hold on to when generating several sizes of the same
    /// image or when the decoder is being worked on.
    pub fn generate_latent(&self, prompt: &str, options: &GenerationOptions) -> Result<Tensor> {
        let latent = self.generate_latent_reporting(prompt, options, &mut unwatched)?;
        Ok(latent.expect("a run nothing asked to stop runs to the end"))
    }

    /// [`Sdxl::generate_latent`], reporting to and interruptible by `report`.
    fn generate_latent_reporting(
        &self,
        prompt: &str,
        options: &GenerationOptions,
        report: &mut dyn FnMut(GenerationProgress) -> ControlFlow<()>,
    ) -> Result<Option<Tensor>> {
        self.check_size(options.width, options.height)?;
        self.seed(options)?;

        let latent = F::randn(
            &[
                1,
                self.config.unet.latent_channels,
                options.height / VAE_SCALE,
                options.width / VAE_SCALE,
            ],
            self.device,
        )?
        .cast(self.dtype)?;

        let Some((prompt, negative)) = self.read_prompts(prompt, options, report)? else {
            return Ok(None);
        };

        // Pure noise, lifted to the level the first step of the whole schedule expects.
        let sampler = EulerSampler::new(&self.config.sampler, options.num_steps)?;
        let latent = F::mul_scalar(&latent, sampler.init_noise_sigma())?;

        self.denoise_reporting(&latent, &sampler, &prompt, &negative, options, report)
    }

    /// An image for `prompt` that starts from `image` rather than from noise.
    ///
    /// `image` is `(1, 3, H, W)` in roughly `[-1, 1]`, which is what [`from_rgb8`] produces, and
    /// its own size is the size of what comes back -- `options.width` and `options.height` are
    /// not read here, since the picture already says.
    ///
    /// `options.num_steps` is the steps that run, the same as it is for a run from noise, and
    /// `options.strength` is how far to walk away from the picture: around 0.8 rewrites it while
    /// keeping its composition, and below about 0.3 there is little left for the prompt to do. It
    /// works by deciding how noisy the picture is when those steps begin, not by taking some of
    /// them away -- see [`EulerSampler::from_image`].
    pub fn generate_from_image(
        &self,
        image: &Tensor,
        prompt: &str,
        options: &GenerationOptions,
    ) -> Result<Tensor> {
        let image = self.generate_from_image_reporting(image, prompt, options, &mut unwatched)?;
        Ok(image.expect("a run nothing asked to stop runs to the end"))
    }

    /// [`Sdxl::generate_from_image`], reporting to and interruptible by `report`.
    pub fn generate_from_image_reporting(
        &self,
        image: &Tensor,
        prompt: &str,
        options: &GenerationOptions,
        report: &mut dyn FnMut(GenerationProgress) -> ControlFlow<()>,
    ) -> Result<Option<Tensor>> {
        let Some(latent) =
            self.generate_latent_from_image_reporting(image, prompt, options, report)?
        else {
            return Ok(None);
        };

        if report(GenerationProgress::Decoding).is_break() {
            return Ok(None);
        }
        self.decode(&latent).map(Some)
    }

    /// The latent [`Sdxl::generate_from_image`] would decode.
    fn generate_latent_from_image_reporting(
        &self,
        image: &Tensor,
        prompt: &str,
        options: &GenerationOptions,
        report: &mut dyn FnMut(GenerationProgress) -> ControlFlow<()>,
    ) -> Result<Option<Tensor>> {
        let shape = image.shape();
        if shape.len() != 4 || shape[0] != 1 || shape[1] != 3 {
            return Err(Error::model(format!(
                "a picture to start from is <float>(1, 3, H, W), got {shape:?}"
            )));
        }
        self.check_size(shape[3], shape[2])?;

        // First, because encoding the picture is a decode's worth of work and the screen would
        // otherwise sit on the last thing it said all the way through it.
        let Some((prompt, negative)) = self.read_prompts(prompt, options, report)? else {
            return Ok(None);
        };

        // Before the draw below rather than after: a caller who asked for a seed wants that draw
        // to be the one it names.
        self.seed(options)?;

        // Into the type the walk runs in. The autoencoder is float32 whatever the rest of the
        // model is -- it has to be -- so what it hands back is wider than the U-Net reads, and
        // the noise added to it has to match it either way.
        let encoded = self.encode(image)?.cast(self.dtype)?;

        // The tail of a longer schedule: options.num_steps of them run, and the strength decides
        // how long the schedule they are the tail of is, which is how noisy the picture is when
        // they start.
        let sampler =
            EulerSampler::from_image(&self.config.sampler, options.num_steps, options.strength)?;

        // The picture put back at the noise level that step expects to see. This is the whole of
        // what image to image is: the walk from here on is the ordinary one, and what keeps the
        // picture is that the walk is not long enough to forget it.
        let noise = F::randn(&encoded.shape(), self.device)?.cast(encoded.dtype())?;
        let latent = F::add(
            &encoded,
            &F::mul_scalar(&noise, sampler.sigmas()[sampler.start()])?,
        )?;

        self.denoise_reporting(&latent, &sampler, &prompt, &negative, options, report)
    }

    /// Refuses a size the U-Net cannot work at.
    ///
    /// Every resolution it works at halves the one before it, on top of the eight the VAE already
    /// stands for, so a size that does not divide would be rounded somewhere inside and come back
    /// as a shape that no longer lines up with the skip connection it has to be added to.
    fn check_size(&self, width: i32, height: i32) -> Result<()> {
        let alignment = VAE_SCALE * (1 << (self.config.unet.block_out_channels.len() - 1));
        if width <= 0 || height <= 0 || width % alignment != 0 || height % alignment != 0 {
            return Err(Error::model(format!(
                "{width} by {height} is not a multiple of {alignment}, which is what this model \
                 works in"
            )));
        }

        Ok(())
    }

    /// Fixes the draws this run makes, where the caller asked for a seed.
    fn seed(&self, options: &GenerationOptions) -> Result<()> {
        if let Some(seed) = options.seed {
            F::manual_seed(self.device, seed)?;
        }

        Ok(())
    }

    /// Both prompts, read once before the walk begins. None where `report` asked to stop.
    fn read_prompts(
        &self,
        prompt: &str,
        options: &GenerationOptions,
        report: &mut dyn FnMut(GenerationProgress) -> ControlFlow<()>,
    ) -> Result<Option<(PromptEmbedding, PromptEmbedding)>> {
        if report(GenerationProgress::Encoding).is_break() {
            return Ok(None);
        }

        let prompt = self.encode_prompt(prompt)?;
        let negative = self.encode_prompt(&options.negative_prompt)?;

        Ok(Some((prompt, negative)))
    }
}

/// An image as bytes, three per pixel, row by row.
///
/// A decoder ends in roughly `[-1, 1]`, which is what the halving and shifting here undoes.
/// Anything outside that range is clamped rather than wrapped, which is what makes an overexposed
/// image white instead of black.
pub fn to_rgb8(image: &Tensor) -> Result<Vec<u8>> {
    let shape = image.shape();
    if shape.len() != 4 || shape[0] != 1 || shape[1] != 3 {
        return Err(Error::model(format!(
            "an image is <float>(1, 3, H, W), got {shape:?}"
        )));
    }

    let (height, width) = (shape[2] as usize, shape[3] as usize);
    let values = image
        .to_device(Device::Cpu)?
        .cast(DType::Float)?
        .to_vec_f32()?;

    // The tensor holds one whole channel after another; a pixel wants its three together.
    let plane = height * width;
    let mut out = Vec::with_capacity(plane * 3);
    for pixel in 0..plane {
        for channel in 0..3 {
            let value = (values[channel * plane + pixel] / 2.0 + 0.5).clamp(0.0, 1.0);
            out.push((value * 255.0).round() as u8);
        }
    }

    Ok(out)
}

/// A picture read from a file as the tensor a model reads, `<float>(1, 3, height, width)`.
///
/// `pixels` is three bytes per pixel, row by row, which is the layout [`to_rgb8`] writes and the
/// one every image library hands back. This is that function backwards: the bytes are spread into
/// one plane per channel and mapped from `[0, 255]` onto the `[-1, 1]` the autoencoder was
/// trained on.
///
/// The tensor is on the CPU, in float32. Moving it to the device and narrowing it is the
/// encoder's business, and it does it on the way in.
pub fn from_rgb8(width: i32, height: i32, pixels: &[u8]) -> Result<Tensor> {
    if width <= 0 || height <= 0 {
        return Err(Error::model(format!(
            "{width} by {height} is not a size a picture can have"
        )));
    }

    let plane = width as usize * height as usize;
    if pixels.len() != plane * 3 {
        return Err(Error::model(format!(
            "{} bytes is not the {} a {width} by {height} picture of three channels has",
            pixels.len(),
            plane * 3
        )));
    }

    let mut values = vec![0.0f32; plane * 3];
    for pixel in 0..plane {
        for channel in 0..3 {
            values[channel * plane + pixel] = pixels[pixel * 3 + channel] as f32 / 127.5 - 1.0;
        }
    }

    Ok(Tensor::from_f32(&[1, 3, height, width], &values)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifest::Manifest;

    /// A package's configuration, as the exporter writes it.
    const CONFIG: &str = r#"
# What this model is, and which files it is made of.
#
# The safetensors beside this file hold the weights and nothing else; everything
# that says what they are is here.

weights:
  - sdxl-base.safetensors

config:
  sdxl:
    latent_channels: 4
    vae_scaling_factor: 0.13025
    vae_block_out_channels: 128,256,512,512
    vae_layers_per_block: 2
    vae_norm_num_groups: 32
    unet_block_out_channels: 320,640,1280
    unet_layers_per_block: 2
    unet_transformer_layers_per_block: 0,2,10
    unet_attention_head_dim: 5,10,20
    unet_norm_num_groups: 32
    unet_cross_attention_dim: 2048
    unet_addition_time_embed_dim: 256
    unet_projection_class_embeddings_input_dim: 2816
    scheduler_num_train_timesteps: 1000
    scheduler_beta_start: 0.00085
    scheduler_beta_end: 0.012
    scheduler_steps_offset: 1
    text_hidden_size: 768
    text_intermediate_size: 3072
    text_num_layers: 12
    text_num_heads: 12
    text_hidden_act: quick_gelu
    text_norm_eps: 1e-05
    text2_hidden_size: 1280
    text2_intermediate_size: 5120
    text2_num_layers: 32
    text2_num_heads: 20
    text2_hidden_act: gelu
    text2_norm_eps: 1e-05
    context_length: 77
    vocab_size: 49408
    bot_token_id: 49406
    eot_token_id: 49407
    pad_token_id: 49407
    pad_token_id2: 0
    clip_skip: 2
"#;

    fn parse(text: &str) -> Result<SdxlConfig> {
        let ini = Manifest::parse(text).unwrap();
        SdxlConfig::from_section(ini.section("sdxl").unwrap())
    }

    #[test]
    fn reads_what_the_exporter_writes() {
        let config = parse(CONFIG).unwrap();

        assert_eq!(config.unet.block_out_channels, vec![320, 640, 1280]);
        assert_eq!(config.unet.transformer_layers_per_block, vec![0, 2, 10]);
        assert_eq!(config.unet.num_heads, vec![5, 10, 20]);
        assert_eq!(config.vae.block_out_channels, vec![128, 256, 512, 512]);
        assert_eq!(config.vae.scaling_factor, 0.13025);
        assert_eq!(config.sampler.num_train_timesteps, 1000);
        assert_eq!(config.context_length, 77);
    }

    /// A package written before any of this says nothing, and nothing is float.
    #[test]
    fn a_package_that_does_not_say_holds_its_matrices_in_float() {
        let config = parse(CONFIG).unwrap();

        assert_eq!(config.unet.weight_format, WeightFormat::Float);
        assert_eq!(config.text.weight_format, WeightFormat::Float);
        assert_eq!(config.text2.weight_format, WeightFormat::Float);
    }

    /// And one key says it for every half that has matrices in it.
    #[test]
    fn a_package_may_say_it_stored_its_matrices_quantized() {
        let config = parse(&format!("{CONFIG}    weight_format: fp8\n")).unwrap();

        assert_eq!(config.unet.weight_format, WeightFormat::Fp8);
        assert_eq!(config.text.weight_format, WeightFormat::Fp8);
        assert_eq!(config.text2.weight_format, WeightFormat::Fp8);

        // A name nothing reads is refused where it is read. The alternative is a package that
        // meant to be quantized, is not, and says nothing about it.
        let error = parse(&format!("{CONFIG}    weight_format: e4m3\n")).unwrap_err();
        assert!(error.to_string().contains("\"e4m3\""), "{error}");
    }

    #[test]
    fn the_two_encoders_differ_in_the_ways_they_are_supposed_to() {
        let config = parse(CONFIG).unwrap();

        assert_eq!(config.text.hidden_size, 768);
        assert_eq!(config.text2.hidden_size, 1280);
        assert!(
            config.text.quick_gelu,
            "CLIP-L uses the sigmoid approximation"
        );
        assert!(
            !config.text2.quick_gelu,
            "OpenCLIP bigG uses the ordinary gelu"
        );

        // Both read the same prompt, so both hold the same number of positions and the same
        // vocabulary. They pad it differently, which is the one thing about the ids that is not
        // shared.
        assert_eq!(config.text.context_length, config.text2.context_length);
        assert_eq!(config.text.vocab_size, config.text2.vocab_size);
        assert_ne!(config.pad_token_id, config.pad_token_id2);
    }

    #[test]
    fn a_clip_skip_it_cannot_honour_is_refused() {
        // Conditioning on a different layer is a different model, not a different setting, and
        // silently using the penultimate one anyway would be wrong in a way nothing downstream
        // could notice.
        let text = CONFIG.replace("clip_skip: 2", "clip_skip: 1");
        assert!(parse(&text).is_err());
    }

    #[test]
    fn an_activation_it_does_not_have_is_refused() {
        let text = CONFIG.replace("text_hidden_act: quick_gelu", "text_hidden_act: relu");
        assert!(parse(&text).is_err());
    }

    #[test]
    fn a_missing_key_is_refused() {
        for key in [
            "unet_block_out_channels",
            "scheduler_beta_start",
            "text2_num_layers",
            "vae_scaling_factor",
        ] {
            let text: String = CONFIG
                .lines()
                .filter(|line| !line.trim_start().starts_with(&format!("{key}:")))
                .collect::<Vec<_>>()
                .join("\n");
            assert!(parse(&text).is_err(), "{key} was allowed to be missing");
        }
    }

    #[test]
    fn a_list_that_is_not_numbers_is_refused() {
        let text = CONFIG.replace(
            "unet_block_out_channels: 320,640,1280",
            "unet_block_out_channels: 320,wide,1280",
        );
        assert!(parse(&text).is_err());
    }

    #[test]
    fn an_image_becomes_bytes_in_the_order_a_picture_wants_them() {
        // Two pixels, and a channel each of black, grey and white: the tensor holds one whole
        // channel after another and the bytes hold one whole pixel after another.
        let image = Tensor::from_f32(&[1, 3, 1, 2], &[-1.0, -1.0, 0.0, 0.0, 1.0, 1.0]).unwrap();
        assert_eq!(to_rgb8(&image).unwrap(), vec![0, 128, 255, 0, 128, 255]);
    }

    #[test]
    fn an_image_outside_the_range_is_clamped() {
        let image = Tensor::from_f32(&[1, 3, 1, 1], &[-9.0, 0.0, 9.0]).unwrap();
        assert_eq!(to_rgb8(&image).unwrap(), vec![0, 128, 255]);
    }

    #[test]
    fn what_is_not_an_image_is_refused() {
        let four_channels = Tensor::zeros(&[1, 4, 2, 2], DType::Float, Device::Cpu).unwrap();
        assert!(to_rgb8(&four_channels).is_err());

        let three_d = Tensor::zeros(&[3, 2, 2], DType::Float, Device::Cpu).unwrap();
        assert!(to_rgb8(&three_d).is_err());
    }

    #[test]
    fn bytes_become_the_image_they_stood_for() {
        // The same two pixels the other way round: interleaved bytes spread into one plane per
        // channel, and [0, 255] mapped onto [-1, 1].
        let pixels = [0u8, 128, 255, 0, 128, 255];
        let image = from_rgb8(2, 1, &pixels).unwrap();

        assert_eq!(image.shape(), vec![1, 3, 1, 2]);
        let values = image.to_vec_f32().unwrap();
        assert_eq!(values[0], -1.0);
        assert_eq!(values[1], -1.0);
        assert!((values[2] - 128.0 / 127.5 + 1.0).abs() < 1e-6);
        assert_eq!(values[4], 1.0);
        assert_eq!(values[5], 1.0);
    }

    #[test]
    fn a_picture_survives_the_trip_out_and_back() {
        // Every byte a channel can hold, so that the two mappings are checked against each other
        // over the whole range rather than at the ends. They are not exact inverses -- 256 values
        // do not land on 256 of the floats between -1 and 1 -- but rounding is the only thing
        // allowed to differ, and it cancels.
        let pixels: Vec<u8> = (0..=255u8).flat_map(|v| [v, 255 - v, v / 2]).collect();
        let image = from_rgb8(16, 16, &pixels).unwrap();

        assert_eq!(to_rgb8(&image).unwrap(), pixels);
    }

    #[test]
    fn reads_only_the_kind_of_model_it_can_draw_with() {
        assert!(check_model_type("sdxl").is_ok());

        // Handing this half an Anima package. It got as far as parsing the `[anima]` section as
        // an SDXL one once, and complained that `vocab_size` was missing -- true, and it tells
        // nobody anything. What it has to say instead is what the package is and where to take it.
        let message = check_model_type("anima")
            .expect_err("anima is not sdxl")
            .to_string();
        assert!(message.contains("anima"), "{message}");
        assert!(message.contains("sdxl"), "{message}");
        assert!(
            !message.contains("vocab_size"),
            "it should say what the package is, not which key went missing: {message}"
        );
    }

    #[test]
    fn bytes_that_are_not_a_picture_are_refused() {
        // One byte short of the picture they claim to be, which is a caller that got its stride
        // wrong rather than something to read past the end of.
        assert!(from_rgb8(2, 2, &[0; 11]).is_err());
        assert!(from_rgb8(2, 2, &[0; 13]).is_err());
        assert!(from_rgb8(0, 2, &[]).is_err());
        assert!(from_rgb8(-2, 2, &[]).is_err());
    }
}
