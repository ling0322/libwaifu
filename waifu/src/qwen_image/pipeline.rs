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

use std::fmt;
use std::ops::ControlFlow;

use super::{Dit, QwenImageConfig, VaeDecoder};
use crate::error::{Error, Result};
use crate::flint::{functional as F, DType, Device, Residency, Tensor};
use crate::flow::{FlowSampler, SamplerConfig};
use crate::generation::{unwatched, GenerationDefaults, GenerationOptions, GenerationProgress};
use crate::krea2::TextEncoder;
use crate::manifest::Manifest;
use crate::tokenizer::Tokenizer;

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

    pub fn from_manifest(
        device: Device,
        residency: Residency,
        manifest: &Manifest,
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

        let dtype = F::default_float_type(device)?;
        let file = manifest.params()?;
        let weights = residency.read(file, device)?;

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
            dit: Dit::build(config.dit.clone(), &named("dit"), &weights, dtype)?,
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

    /// The schedule for a picture `width` by `height`, which depends on its size.
    pub fn sampler(&self, width: i32, height: i32, num_steps: i32) -> Result<FlowSampler> {
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
        let Some(latent) = self.generate_latent_reporting(prompt, options, report)? else {
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

        let sampler = self.sampler(options.width, options.height, options.num_steps)?;
        let scale = self.config.vae.scale;
        let latent = F::randn(
            &[
                1,
                self.config.dit.latent_channels,
                options.height / scale,
                options.width / scale,
            ],
            self.device,
        )?
        .cast(self.dtype)?;

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
            options,
            report,
        )
    }

    /// Walk `latent` down `sampler`'s schedule. None where `report` said to stop.
    pub fn denoise_reporting(
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
            let timestep = model_timestep(sampler.sigma(index)?);
            let velocity = self.dit.forward(&latent, timestep, context)?;

            // `true_cfg_scale`: two passes, and a scale of one means none.
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

#[cfg(test)]
mod tests {
    use super::*;

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
