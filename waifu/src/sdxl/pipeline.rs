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

use crate::error::{Error, Result};
use crate::flint::{functional as F, DType, Device, Tensor};
use crate::ini::IniSection;
use crate::tokenizer::Tokenizer;
use crate::var_builder::VarBuilder;
use crate::zip_file::ZipFile;

use super::sampler::{EulerSampler, SamplerConfig};
use super::text_encoder::{ClipTextConfig, ClipTextEncoder};
use super::unet::{Unet, UnetCondition, UnetConfig};
use super::vae::{VaeConfig, VaeDecoder};

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
fn number_list(section: &IniSection, key: &str) -> Result<Vec<i32>> {
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
    pub fn from_section(section: &IniSection) -> Result<SdxlConfig> {
        let clip_skip: i32 = section.get_or("clip_skip", SUPPORTED_CLIP_SKIP)?;
        if clip_skip != SUPPORTED_CLIP_SKIP {
            return Err(Error::model(format!(
                "a clip skip of {clip_skip} is not the {SUPPORTED_CLIP_SKIP} these encoders read"
            )));
        }

        let context_length = section.get("context_length")?;
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

/// What to ask for. The defaults are what SDXL was trained at.
#[derive(Clone, Debug)]
pub struct GenerationOptions {
    pub width: i32,
    pub height: i32,
    pub num_steps: i32,
    /// How hard to push away from the unprompted answer. One means not at all, which also skips
    /// the second U-Net pass and so runs twice as fast; five to eight is the usual range.
    pub guidance_scale: f32,
    /// What to steer away from. An empty string is the usual thing to steer away from and is not
    /// the same as no negative prompt at all -- the model has an opinion about the empty prompt.
    pub negative_prompt: String,
    pub seed: Option<u64>,
}

impl Default for GenerationOptions {
    fn default() -> GenerationOptions {
        GenerationOptions {
            width: 1024,
            height: 1024,
            num_steps: 30,
            guidance_scale: 5.0,
            negative_prompt: String::new(),
            seed: None,
        }
    }
}

/// How far along a run is, as the reporter given to [`Sdxl::generate_reporting`] is told.
///
/// The three are not the same size: encoding costs about as much as one step, a step is a step,
/// and the decode at the end is several. A bar drawn from the step count alone will sit at the
/// end for a while, which is the honest thing for it to do.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GenerationProgress {
    /// Reading the prompt, which happens once before the first step.
    Encoding,
    /// The denoising step that just finished, out of how many there are.
    Step { done: i32, total: i32 },
    /// Turning the finished latent into pixels.
    Decoding,
}

/// The reporter a run that nobody is watching gets.
fn unwatched(_: GenerationProgress) -> ControlFlow<()> {
    ControlFlow::Continue(())
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
    tokenizer: Tokenizer,
    text_encoder: ClipTextEncoder,
    text_encoder2: ClipTextEncoder,
    unet: Unet,
    vae: VaeDecoder,
    device: Device,
    dtype: DType,
}

impl Sdxl {
    /// The section of `model.ini` that says what the package holds.
    pub const MODEL_SECTION: &'static str = "model";

    /// What `model.ini` calls the list of packages a model too large for one file is split over.
    pub const SHARDS_KEY: &'static str = "model_parts";

    /// Read the whole model out of `package`, onto `device`.
    ///
    /// A model may be written as several packages beside each other, in which case this one holds
    /// the configuration and names the rest. Which package a tensor was written to is not
    /// something the model has to know: they are read in order into one namespace.
    pub fn from_package(device: Device, package: &ZipFile) -> Result<Sdxl> {
        let ini = crate::ini::IniConfig::parse(&package.read_to_string(crate::MODEL_CONFIG)?)?;
        let model_section = ini.section(Self::MODEL_SECTION)?;
        let model_type = model_section.get_str("type")?.to_string();
        let model_file = model_section.get_str("model_file")?.to_string();

        let config = SdxlConfig::from_section(ini.section(&model_type)?)?;

        // The parts are opened and then only walked: what a layer asks for is read when it asks,
        // so the packages stay open for as long as the builder does and the host never holds more
        // than the tensor being handed over.
        let mut parts = Vec::new();
        match model_section.get_str(Self::SHARDS_KEY) {
            Ok(names) => {
                for part in names.split(',').map(str::trim).filter(|p| !p.is_empty()) {
                    parts.push(package.sibling(part)?);
                }
                if parts.is_empty() {
                    return Err(Error::model(format!(
                        "{} is empty, so this model has no parameters to read",
                        Self::SHARDS_KEY
                    )));
                }
            }
            // No list means the model is the one file it was opened from, which is what a package
            // small enough not to need splitting looks like.
            Err(_) => parts.push(ZipFile::open(package.path())?),
        }

        let dtype = F::default_float_type(device)?;
        let vb = VarBuilder::from_packages(&parts, &model_file, device, dtype)?;
        let vb = vb.with_name(&model_type);

        Ok(Sdxl {
            text_encoder: ClipTextEncoder::build(config.text, &vb.with_name("text_encoder"))?,
            text_encoder2: ClipTextEncoder::build(config.text2, &vb.with_name("text_encoder2"))?,
            unet: Unet::build(config.unet.clone(), &vb.with_name("unet"))?,
            // The autoencoder alone is read in float32. It is marked force_upcast and really
            // does need it -- see VaeDecoder::forward -- and the exporter writes its weights
            // that way, so this is the file's own precision rather than a widening of it.
            vae: VaeDecoder::build(
                config.vae.clone(),
                &vb.with_name("vae").with_float_type(DType::Float),
            )?,
            tokenizer: Tokenizer::from_package(package)?,
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
    fn token_ids(&self, text: &str, pad_token_id: i32) -> Vec<i64> {
        let length = self.config.context_length as usize;
        let mut ids = vec![pad_token_id as i64; length];

        ids[0] = self.config.bot_token_id as i64;
        let mut position = 1;
        for id in self.tokenizer.encode(text) {
            if position + 1 >= length {
                break;
            }
            ids[position] = id as i64;
            position += 1;
        }
        ids[position] = self.config.eot_token_id as i64;

        ids
    }

    /// What both encoders make of `text`.
    pub fn encode_prompt(&self, text: &str) -> Result<PromptEmbedding> {
        let length = self.config.context_length;

        let ids = Tensor::from_i64(&[length], &self.token_ids(text, self.config.pad_token_id))?
            .to_device(self.device)?;
        let ids2 = Tensor::from_i64(&[length], &self.token_ids(text, self.config.pad_token_id2))?
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
        let denoised = self.denoise_reporting(latent, prompt, negative, options, &mut unwatched)?;
        Ok(denoised.expect("a run nothing asked to stop runs to the end"))
    }

    /// [`Sdxl::denoise`], telling `report` after every step and giving up where it stands if
    /// `report` says to -- in which case there is no latent to hand back.
    fn denoise_reporting(
        &self,
        latent: &Tensor,
        prompt: &PromptEmbedding,
        negative: &PromptEmbedding,
        options: &GenerationOptions,
        report: &mut dyn FnMut(GenerationProgress) -> ControlFlow<()>,
    ) -> Result<Option<Tensor>> {
        let sampler = EulerSampler::new(&self.config.sampler, options.num_steps)?;

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

        let mut latent = F::mul_scalar(latent, sampler.init_noise_sigma())?;
        for index in 0..sampler.len() {
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

            let progress = GenerationProgress::Step {
                done: index as i32 + 1,
                total: sampler.len() as i32,
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
        // Every resolution the U-Net works at halves the one before it, on top of the eight the
        // VAE already stands for.
        let alignment = VAE_SCALE * (1 << (self.config.unet.block_out_channels.len() - 1));
        if options.width <= 0
            || options.height <= 0
            || options.width % alignment != 0
            || options.height % alignment != 0
        {
            return Err(Error::model(format!(
                "{} by {} is not a multiple of {alignment}, which is what this model works in",
                options.width, options.height
            )));
        }

        if let Some(seed) = options.seed {
            F::manual_seed(self.device, seed)?;
        }

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

        if report(GenerationProgress::Encoding).is_break() {
            return Ok(None);
        }
        let prompt = self.encode_prompt(prompt)?;
        let negative = self.encode_prompt(&options.negative_prompt)?;

        self.denoise_reporting(&latent, &prompt, &negative, options, report)
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ini::IniConfig;

    /// A package's configuration, as the exporter writes it.
    const CONFIG: &str = "\
[sdxl]
latent_channels = 4
vae_scaling_factor = 0.13025
vae_block_out_channels = 128,256,512,512
vae_layers_per_block = 2
vae_norm_num_groups = 32
unet_block_out_channels = 320,640,1280
unet_layers_per_block = 2
unet_transformer_layers_per_block = 0,2,10
unet_attention_head_dim = 5,10,20
unet_norm_num_groups = 32
unet_cross_attention_dim = 2048
unet_addition_time_embed_dim = 256
unet_projection_class_embeddings_input_dim = 2816
scheduler_num_train_timesteps = 1000
scheduler_beta_start = 0.00085
scheduler_beta_end = 0.012
scheduler_steps_offset = 1
text_hidden_size = 768
text_intermediate_size = 3072
text_num_layers = 12
text_num_heads = 12
text_hidden_act = quick_gelu
text_norm_eps = 1e-05
text2_hidden_size = 1280
text2_intermediate_size = 5120
text2_num_layers = 32
text2_num_heads = 20
text2_hidden_act = gelu
text2_norm_eps = 1e-05
context_length = 77
vocab_size = 49408
bot_token_id = 49406
eot_token_id = 49407
pad_token_id = 49407
pad_token_id2 = 0
clip_skip = 2
";

    fn parse(text: &str) -> Result<SdxlConfig> {
        let ini = IniConfig::parse(text).unwrap();
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
        let text = CONFIG.replace("clip_skip = 2", "clip_skip = 1");
        assert!(parse(&text).is_err());
    }

    #[test]
    fn an_activation_it_does_not_have_is_refused() {
        let text = CONFIG.replace("text_hidden_act = quick_gelu", "text_hidden_act = relu");
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
                .filter(|line| !line.starts_with(key))
                .collect::<Vec<_>>()
                .join("\n");
            assert!(parse(&text).is_err(), "{key} was allowed to be missing");
        }
    }

    #[test]
    fn a_list_that_is_not_numbers_is_refused() {
        let text = CONFIG.replace(
            "unet_block_out_channels = 320,640,1280",
            "unet_block_out_channels = 320,wide,1280",
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
}
