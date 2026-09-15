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

//! The five parts joined up, with a prompt at one end and a picture at the other.
//!
//! What a prompt goes through here is longer than SDXL's and every stage of it is load bearing:
//!
//! 1. it is tokenized twice, by two unrelated vocabularies;
//! 2. Qwen3 reads the first set of ids into hidden states -- the *context*;
//! 3. the adapter embeds the second set into a *query* stream and cross-attends it into that
//!    context, which is what lets a Qwen3 encoder stand where Cosmos-Predict2 was pretrained
//!    against a T5 one;
//! 4. the result is padded out to 512, which is the only length the denoiser reads;
//! 5. the flow schedule walks a latent from noise, one denoiser pass per step;
//! 6. the VAE turns the latent into pixels.
//!
//! `docs/anima.md` is where the whole of it is written down, and none of steps 1 to 4 can be
//! skipped or reordered.

use std::fmt;
use std::ops::ControlFlow;

use crate::anima::{Adapter, AnimaConfig, Dit, FlowSampler, TextEncoder, VaeDecoder};
use crate::error::{Error, Result};
use crate::flint::{functional as F, DType, Device, Residency, Tensor};
use crate::generation::{unwatched, GenerationDefaults, GenerationOptions, GenerationProgress};
use crate::manifest::Manifest;
use crate::tokenizer::Tokenizer;

/// How much smaller a latent is than the picture it decodes to, on each side.
pub const VAE_SCALE: i32 = 8;

/// Anima, with everything it needs to answer a prompt.
pub struct Anima {
    config: AnimaConfig,
    /// The text encoder's vocabulary, and the adapter's. Two of them because the model reads the
    /// prompt twice -- see [`Anima::encode_prompt`], which is where the reason lives.
    qwen_tokenizer: Tokenizer,
    t5_tokenizer: Tokenizer,
    text_encoder: TextEncoder,
    adapter: Adapter,
    dit: Dit,
    vae: VaeDecoder,
    /// What the adapter's ids end with. Read out of the vocabulary once rather than written down,
    /// since it is the vocabulary's answer and not this file's.
    t5_eos: i32,
    device: Device,
    dtype: DType,
}

impl fmt::Debug for Anima {
    /// The weights are several gigabytes and the graphs are thousands of nodes, so neither is
    /// worth printing. What a caller wants from this is which model it has.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Anima")
            .field("blocks", &self.config.dit.num_blocks)
            .field("device", &self.device)
            .field("dtype", &self.dtype)
            .finish_non_exhaustive()
    }
}

impl Anima {
    /// What `model type` says for a model this can read.
    pub const MODEL_TYPE: &'static str = "anima";

    /// The block of the manifest's `config:` that says what the model is.
    pub const MODEL_SECTION: &'static str = "model";

    /// The two tokenizers, as the manifest names them. Neither is `tokenizer`: there is nothing
    /// to make one of the two the one a reader should assume.
    pub const QWEN_TOKENIZER_SECTION: &'static str = "qwen3";
    pub const T5_TOKENIZER_SECTION: &'static str = "t5";

    /// What the adapter's ids end with, as the T5 vocabulary spells it.
    const T5_EOS: &'static str = "</s>";

    /// What to ask a turbo release for, which is not what to ask SDXL for.
    ///
    /// Eight steps and no guidance. Anima's turbo v1.1 is distilled for exactly that, and the
    /// thirty steps at five that SDXL likes give a burnt, over-contrasted picture for four times
    /// the work. The aesthetic release wants 30 to 50 at 4 to 5 instead -- when one is packaged,
    /// this is what has to learn to read the difference out of the model rather than assume.
    pub const DEFAULTS: GenerationDefaults = GenerationDefaults {
        width: 1024,
        height: 1024,
        num_steps: 8,
        guidance_scale: 1.0,
    };

    /// Read the whole model `manifest` describes, onto `device`, with its weights waiting where
    /// `residency` says.
    ///
    /// The same shape as [`crate::Sdxl::from_manifest`] and for the same reasons: the packages a
    /// model was written as read into one namespace, and the four halves below share one store.
    pub fn from_manifest(
        device: Device,
        residency: Residency,
        manifest: &Manifest,
    ) -> Result<Anima> {
        let model_section = manifest.section(Self::MODEL_SECTION)?;
        let model_type = model_section.get_str("type")?.to_string();
        if model_type != Self::MODEL_TYPE {
            return Err(Error::model(format!(
                "this manifest describes a model of kind {model_type:?}, which is not {:?}",
                Self::MODEL_TYPE
            )));
        }

        let config = AnimaConfig::from_section(manifest.section(&model_type)?)?;

        let dtype = F::default_float_type(device)?;
        let file = manifest.params()?;
        let weights = residency.read(file, device)?;

        let named = |half: &str| format!("{model_type}.{half}");

        let t5_tokenizer = Tokenizer::open_section(manifest, Self::T5_TOKENIZER_SECTION)?;
        let t5_eos = t5_tokenizer.token_to_id(Self::T5_EOS)?;

        Ok(Anima {
            text_encoder: TextEncoder::build(config.text.clone(), &named("text"), &weights, dtype)?,
            adapter: Adapter::build(config.adapter.clone(), &named("adapter"), &weights, dtype)?,
            dit: Dit::build(config.dit.clone(), &named("dit"), &weights, dtype)?,
            vae: VaeDecoder::build(config.vae.clone(), &named("vae"), &weights, device, dtype)?,
            qwen_tokenizer: Tokenizer::open_section(manifest, Self::QWEN_TOKENIZER_SECTION)?,
            t5_tokenizer,
            t5_eos,
            config,
            device,
            dtype,
        })
    }

    pub fn config(&self) -> &AnimaConfig {
        &self.config
    }

    pub fn device(&self) -> Device {
        self.device
    }

    /// The denoiser alone, for a caller that wants to ask it something the pipeline does not.
    pub fn dit(&self) -> &Dit {
        &self.dit
    }

    /// What the denoiser reads, for `prompt`: `(1, 512, D)`.
    ///
    /// The prompt is tokenized twice, and the two answers are not two spellings of one thing. The
    /// Qwen3 ids are read into hidden states; the T5 ids are what the adapter *embeds*, and the
    /// adapter cross-attends its embedding of them into those hidden states. Which is why the T5
    /// vocabulary is 32128 wide: Cosmos-Predict2 was pretrained against T5, and the adapter is
    /// what lets a Qwen3 encoder stand in for it. Feeding the same ids to both, or skipping the
    /// adapter and handing the denoiser the hidden states directly, type checks and draws noise.
    ///
    /// The end marker goes on the T5 ids and not the Qwen3 ones, which is the reference's
    /// asymmetry rather than an oversight here.
    pub fn encode_prompt(&self, prompt: &str) -> Result<Tensor> {
        let hidden = self
            .text_encoder
            .forward(&self.ids(&self.qwen_tokenizer, prompt, None)?)?;

        let context = self.adapter.forward(
            &self.ids(&self.t5_tokenizer, prompt, Some(self.t5_eos))?,
            &hidden,
        )?;

        self.pad_context(&context)
    }

    /// One tokenizer's answer for `prompt`, as the `<long>(L)` both halves read.
    ///
    /// Cut at the context length rather than refused, which is what every other implementation
    /// does and what a prompt of a hundred tags will hit. On the T5 side that length is the
    /// denoiser's own and cutting is the only option; on the Qwen3 side it is a cap rather than a
    /// limit -- the adapter cross-attends into a context of any length -- but a prompt with more
    /// than 512 tags in it has already lost its tail on the other side, so the two are cut alike.
    ///
    /// The marker, where there is one, survives the cut by being put on afterwards.
    fn ids(&self, tokenizer: &Tokenizer, prompt: &str, ends_with: Option<i32>) -> Result<Tensor> {
        let room = self.config.context_length as usize - usize::from(ends_with.is_some());

        let mut ids: Vec<i64> = tokenizer
            .encode(prompt)?
            .iter()
            .take(room)
            .map(|&id| id as i64)
            .collect();
        ids.extend(ends_with.map(i64::from));

        // An empty prompt is a real request -- it is what a negative prompt usually is -- and the
        // encoder has an opinion about it. What it cannot take is a sequence of no tokens at all,
        // so the marker, or failing that a single space, stands in for one. What a space encodes
        // to is the vocabulary's business: T5 puts its own prefix on it, Qwen3 does not.
        if ids.is_empty() {
            ids.extend(tokenizer.encode(" ")?.iter().map(|&id| id as i64));
        }

        Ok(Tensor::from_i64(&[ids.len() as i32], &ids)?.to_device(self.device)?)
    }

    /// The adapter's output at the one length the denoiser reads, zero filled past the prompt.
    fn pad_context(&self, context: &Tensor) -> Result<Tensor> {
        let length = context.shape_at(1)?;
        let wanted = self.config.context_length;

        match length.cmp(&wanted) {
            std::cmp::Ordering::Equal => Ok(context.clone()),
            std::cmp::Ordering::Greater => Ok(context.slice(1, 0, wanted)?),
            std::cmp::Ordering::Less => {
                let width = context.shape_at(2)?;
                let padding =
                    Tensor::zeros(&[1, wanted - length, width], context.dtype(), self.device)?;
                Ok(F::cat(context, &padding, 1)?)
            }
        }
    }

    /// An image for `prompt`, as `<float>(1, 3, height, width)` in roughly `[-1, 1]`.
    pub fn generate(&self, prompt: &str, options: &GenerationOptions) -> Result<Tensor> {
        let image = self.generate_reporting(prompt, options, &mut unwatched)?;
        Ok(image.expect("a run nothing asked to stop runs to the end"))
    }

    /// [`Anima::generate`] for a caller who wants to watch it happen.
    ///
    /// `report` hears about each part of the run as it is reached, and can end it by returning
    /// [`ControlFlow::Break`] -- which is the only way to stop one, since a step runs to the end
    /// of itself once started. A run that stopped early hands back no image.
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

    /// The latent [`Anima::generate`] would decode: everything but the last step.
    pub fn generate_latent(&self, prompt: &str, options: &GenerationOptions) -> Result<Tensor> {
        let latent = self.generate_latent_reporting(prompt, options, &mut unwatched)?;
        Ok(latent.expect("a run nothing asked to stop runs to the end"))
    }

    fn generate_latent_reporting(
        &self,
        prompt: &str,
        options: &GenerationOptions,
        report: &mut dyn FnMut(GenerationProgress) -> ControlFlow<()>,
    ) -> Result<Option<Tensor>> {
        self.check_size(options.width, options.height)?;
        if let Some(seed) = options.seed {
            F::manual_seed(self.device, seed)?;
        }

        // Before the prompt is read, so that the picture a seed gives does not depend on how many
        // draws reading the prompt happened to make.
        let sampler = FlowSampler::new(&self.config.sampler, options.num_steps)?;
        let latent = F::randn(
            &[
                1,
                self.config.dit.latent_channels,
                options.height / VAE_SCALE,
                options.width / VAE_SCALE,
            ],
            self.device,
        )?
        .cast(self.dtype)?;
        let latent = F::mul_scalar(&latent, sampler.initial_noise_scale())?;

        if report(GenerationProgress::Encoding).is_break() {
            return Ok(None);
        }
        let context = self.encode_prompt(prompt)?.cast(self.dtype)?;
        let guided = options.guidance_scale != 1.0;
        let unprompted = match guided {
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
            options,
            report,
        )
    }

    /// Walk `latent` down the schedule, and hand back what is left. None where `report` said to
    /// stop.
    fn denoise_reporting(
        &self,
        latent: &Tensor,
        sampler: &FlowSampler,
        context: &Tensor,
        unprompted: Option<&Tensor>,
        options: &GenerationOptions,
        report: &mut dyn FnMut(GenerationProgress) -> ControlFlow<()>,
    ) -> Result<Option<Tensor>> {
        let total = sampler.steps() as i32;
        let mut latent = latent.clone();

        for index in 0..sampler.steps() {
            let timestep = sampler.timestep(index)?;
            let velocity = self.dit.forward(&latent, timestep, context)?;

            // Two passes rather than one batch of two: the denoiser draws one picture at a time,
            // so guidance costs a whole second pass here where SDXL gets it for rather less. It
            // is also why turbo, distilled to want none of it, is the release to reach for.
            let velocity = match unprompted {
                None => velocity,
                Some(unprompted) => {
                    let plain = self.dit.forward(&latent, timestep, unprompted)?;
                    let difference = F::sub(&velocity, &plain)?;
                    F::add(&plain, &F::mul_scalar(&difference, options.guidance_scale)?)?
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

    /// The picture a latent stands for, as `<float>(1, 3, H * 8, W * 8)` in roughly `[-1, 1]`.
    pub fn decode(&self, latent: &Tensor) -> Result<Tensor> {
        self.vae.forward(&latent.cast(self.dtype)?)
    }

    /// Refuses a size the denoiser cannot work at.
    ///
    /// Eight for the VAE and then the patch size on top of it, because a latent is cut into
    /// patches and a half patch has nowhere to go.
    fn check_size(&self, width: i32, height: i32) -> Result<()> {
        let alignment = VAE_SCALE * self.config.dit.patch_size;
        if width <= 0 || height <= 0 || width % alignment != 0 || height % alignment != 0 {
            return Err(Error::model(format!(
                "{width} by {height} is not a multiple of {alignment}, which is what this model \
                 works in"
            )));
        }

        Ok(())
    }
}
