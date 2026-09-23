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


//! The four parts joined up, with a prompt at one end and a picture at the other.
//!
//! 1. the prompt is wrapped in the chat template the model was sampled with -- a system line
//!    telling the encoder what kind of description this is, the prompt as the user's turn, and
//!    the five tokens that open the assistant's;
//! 2. Qwen3-VL reads the whole of that, and twelve of its thirty-six layers are tapped;
//! 3. the template's opening is dropped from the states, leaving the prompt and those five;
//! 4. the flow schedule walks a latent from noise, one denoiser pass per step;
//! 5. the VAE turns the latent into pixels.
//!
//! # The padding that is not here
//!
//! The reference pads every prompt out to 512 tokens, puts the five closing tokens *after* the
//! padding, and hands the denoiser all 512 with a mask saying which are real. This runtime pads
//! nothing, and the two produce the same picture rather than a similar one:
//!
//! * the encoder is causal and its padding is masked away as keys, so no real token ever reads
//!   one -- and the positions the reference gives its tokens count only real ones, so the five
//!   closing tokens sit at the same angles either way;
//! * the denoiser's first two text blocks attend across the twelve tapped layers of *one* token
//!   at a time, so one token's padding never reaches another's;
//! * every later block masks the padding out as keys, and the rows the padding produces are
//!   thrown away with the rest of the prompt at the output.
//!
//! What is left is a shorter sequence -- forty tokens rather than five hundred and twelve, for a
//! prompt of thirty-odd words -- which is the same answer and less of the attention that would
//! have produced it. `docs/krea2.md` writes this out in full.

use std::fmt;
use std::ops::ControlFlow;
use std::rc::Rc;

use crate::error::{Error, Result};
use crate::flint::{functional as F, DType, Device, ParamSource, Residency, Tensor, Weights};
use crate::flow::FlowSampler;
use crate::generation::{unwatched, GenerationDefaults, GenerationOptions, GenerationProgress};
use crate::krea2::{Dit, Krea2Config, TextEncoder};
use crate::manifest::Manifest;
use crate::qwen_vae::VaeDecoder;
use crate::tokenizer::Tokenizer;

/// How much smaller a latent is than the picture it decodes to, on each side.
pub const VAE_SCALE: i32 = 8;

/// Krea 2, with everything it needs to answer a prompt.
pub struct Krea2 {
    config: Krea2Config,
    tokenizer: Tokenizer,
    text_encoder: TextEncoder,
    dit: Dit,
    vae: VaeDecoder,
    /// How many tokens the template puts in front of the prompt and how many after it. Counted
    /// once, from the package's own vocabulary, rather than written down: they are 34 and 5 for
    /// the released tokenizer, and a vocabulary that spelled the template differently would make
    /// them something else without saying so.
    prefix_length: i32,
    suffix_length: i32,
    device: Device,
    dtype: DType,
}

impl fmt::Debug for Krea2 {
    /// The weights are a dozen gigabytes and the graphs are thousands of nodes, so neither is
    /// worth printing. What a caller wants from this is which model it has.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Krea2")
            .field("blocks", &self.config.dit.num_blocks)
            .field("device", &self.device)
            .field("dtype", &self.dtype)
            .finish_non_exhaustive()
    }
}

impl Krea2 {
    /// What `model type` says for a model this can read.
    pub const MODEL_TYPE: &'static str = "krea2";

    /// The block of the manifest's `config:` that says what the model is.
    pub const MODEL_SECTION: &'static str = "model";

    /// What goes in front of the prompt: a system turn saying what kind of description this is,
    /// and the opening of the user's.
    ///
    /// Not a decoration. The encoder is a chat model and these are the tokens it was conditioned
    /// with when the denoiser was trained; a prompt handed over without them is a different
    /// prompt, and the states that come back are states of a different sentence.
    const PROMPT_PREFIX: &'static str = "<|im_start|>system\nDescribe the image by detailing the \
        color, shape, size, texture, quantity, text, spatial relationships of the objects and \
        background:<|im_end|>\n<|im_start|>user\n";

    /// And what closes it: the end of the user's turn and the start of the assistant's. Five
    /// tokens, and the denoiser reads every one of them.
    const PROMPT_SUFFIX: &'static str = "<|im_end|>\n<|im_start|>assistant\n";

    /// What to ask a turbo release for, which is not what to ask SDXL for.
    ///
    /// Eight steps and no guidance. Krea 2 Turbo is distilled for exactly that; the undistilled
    /// release wants 28 steps at a guidance of 5.5 instead, and says so through its package's
    /// `suggested:` block rather than through this.
    pub const DEFAULTS: GenerationDefaults = GenerationDefaults {
        width: 1024,
        height: 1024,
        num_steps: 8,
        guidance_scale: 1.0,
    };

    /// Read the whole model `manifest` describes, onto `device`, with its weights waiting where
    /// `residency` says.
    pub fn from_manifest(
        device: Device,
        residency: Residency,
        manifest: &Manifest,
    ) -> Result<Krea2> {
        let model_section = manifest.section(Self::MODEL_SECTION)?;
        let model_type = model_section.get_str("type")?.to_string();
        if model_type != Self::MODEL_TYPE {
            return Err(Error::model(format!(
                "this manifest describes a model of kind {model_type:?}, which is not {:?}",
                Self::MODEL_TYPE
            )));
        }

        let config = Krea2Config::from_section(manifest.section(&model_type)?)?;

        let dtype = F::default_float_type(device)?;
        let weights: Rc<dyn ParamSource> = Rc::new(Weights::from_files(
            &manifest.weight_paths()?,
            device,
            residency,
        )?);

        let named = |half: &str| format!("{model_type}.{half}");
        let tokenizer = Tokenizer::open(manifest)?;

        Ok(Krea2 {
            text_encoder: TextEncoder::build(
                config.encoder.clone(),
                &named("text"),
                &weights,
                dtype,
            )?,
            dit: Dit::build(config.dit.clone(), &named("dit"), &weights, dtype)?,
            vae: VaeDecoder::build(config.vae.clone(), &named("vae"), &weights, device, dtype)?,
            prefix_length: tokenizer.encode(Self::PROMPT_PREFIX)?.len() as i32,
            suffix_length: tokenizer.encode(Self::PROMPT_SUFFIX)?.len() as i32,
            tokenizer,
            config,
            device,
            dtype,
        })
    }

    pub fn config(&self) -> &Krea2Config {
        &self.config
    }

    pub fn device(&self) -> Device {
        self.device
    }

    /// The denoiser alone, for a caller that wants to ask it something the pipeline does not.
    pub fn dit(&self) -> &Dit {
        &self.dit
    }

    /// The encoder alone, for the same reason.
    pub fn text_encoder(&self) -> &TextEncoder {
        &self.text_encoder
    }

    /// What the denoiser reads, for `prompt`: `(1, L, 12, D)`, where `L` is however many tokens
    /// the prompt came to plus the five the template closes with.
    ///
    /// The template's opening is encoded *with* the prompt and dropped afterwards, which is not
    /// the same as not encoding it: those thirty-four tokens are what every state after them was
    /// computed against. Dropping them from the ids instead would load, run, and draw a picture
    /// of a prompt the model was never conditioned on.
    pub fn encode_prompt(&self, prompt: &str) -> Result<Tensor> {
        let hidden = self.text_encoder.forward(&self.ids(prompt)?)?;

        let length = hidden.shape_at(1)?;
        Ok(hidden.slice(1, self.prefix_length, length)?.contiguous()?)
    }

    /// The template around `prompt`, as the `<long>(L)` the encoder reads.
    ///
    /// Cut at the length the model was sampled at rather than refused, which is what every other
    /// implementation does. The cut falls on the prompt, before the closing tokens are put on, so
    /// a prompt that runs over loses its tail and not the template.
    fn ids(&self, prompt: &str) -> Result<Tensor> {
        // The prefix and the prompt are tokenized as one string, because that is what the
        // reference hands its tokenizer and a byte-level vocabulary does not promise that the
        // pieces of a text tokenize like the text.
        let opening = format!("{}{prompt}", Self::PROMPT_PREFIX);
        let room = (self.config.context_length + self.prefix_length - self.suffix_length) as usize;

        let mut ids: Vec<i64> = self
            .tokenizer
            .encode(&opening)?
            .iter()
            .take(room)
            .map(|&id| id as i64)
            .collect();
        ids.extend(
            self.tokenizer
                .encode(Self::PROMPT_SUFFIX)?
                .iter()
                .map(|&id| id as i64),
        );

        Ok(Tensor::from_i64(&[ids.len() as i32], &ids)?.to_device(self.device)?)
    }

    /// An image for `prompt`, as `<float>(1, 3, height, width)` in roughly `[-1, 1]`.
    pub fn generate(&self, prompt: &str, options: &GenerationOptions) -> Result<Tensor> {
        let image = self.generate_reporting(prompt, options, &mut unwatched)?;
        Ok(image.expect("a run nothing asked to stop runs to the end"))
    }

    /// [`Krea2::generate`] for a caller who wants to watch it happen.
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

    /// The latent [`Krea2::generate`] would decode: everything but the last step.
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
            // so guidance costs a whole second pass through twelve billion parameters. It is also
            // why the turbo release, distilled to want none of it, is the one to reach for.
            //
            // Krea's own reference writes this as `cond + scale * (cond - uncond)` and calls a
            // scale of zero no guidance. That is this formula with the scale one lower, and this
            // is the spelling every other model here uses: one means none.
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
