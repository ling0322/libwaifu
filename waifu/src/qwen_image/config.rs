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

//! What a Qwen-Image 2.1 package says about itself.
//!
//! `tools/qwen_image_exporter.py` copies these out of the published `config.json` files and
//! checks them against the tensors. The few that are in no configuration -- the rotary base, the
//! system prompt, which of the encoder's layers is read -- it states, and they are parsed here like
//! the rest.

use crate::error::{Error, Result};
use crate::flint::WeightFormat;
use crate::krea2::EncoderConfig;
use crate::mapping::Mapping;

/// The single-stream DiT.
#[derive(Clone, Debug)]
pub struct DitConfig {
    /// 32 in the released model.
    pub num_blocks: i32,
    /// 4096, which is `num_heads * head_dim`.
    pub hidden_size: i32,
    pub num_heads: i32,
    pub head_dim: i32,
    /// The SwiGLU's width, three times the model's.
    pub mlp_size: i32,
    /// Sixty-four, read unpatched: one token is one latent pixel.
    pub latent_channels: i32,
    pub timestep_embed_dim: i32,
    pub rope_theta: f32,
    /// How the head is divided between frame, height and width: 16, 56 and 56.
    pub rope_axes: (i32, i32, i32),
    /// The epsilon of every norm in the denoiser.
    pub norm_eps: f32,
    /// How wide the prompt is where it arrives, which is the encoder's width.
    pub context_size: i32,
    pub weight_format: WeightFormat,
}

impl DitConfig {
    pub fn rope_pairs(&self) -> i32 {
        self.head_dim / 2
    }
}

/// The autoencoder this model draws through, which is its own and not the Qwen-Image one.
#[derive(Clone, Debug)]
pub struct VaeConfig {
    pub latent_channels: i32,
    /// Four: the picture comes out with an alpha channel.
    pub image_channels: i32,
    /// Sixteen: how much smaller a latent is than its picture on each side.
    pub scale: i32,
    /// Which of the decoder's upsampling stages double in time as well, first stage first. Only
    /// the shortcut around a stage depends on it, and no weight in the package records it.
    pub temporal_upsample: Vec<bool>,
    pub latents_mean: Vec<f32>,
    pub latents_std: Vec<f32>,
}

/// How the schedule is bent, which for this model depends on the size of the picture.
///
/// The shift is `exp(mu)` with `mu` interpolated along a line through `(base_seq_len,
/// base_shift)` and `(max_seq_len, max_shift)` by how many image tokens are drawn -- a larger
/// picture spends longer at high noise. And the schedule is stretched afterwards so that its last
/// step before zero lands on `shift_terminal` rather than wherever the bend put it.
#[derive(Clone, Copy, Debug)]
pub struct ScheduleConfig {
    pub base_shift: f32,
    pub max_shift: f32,
    pub base_seq_len: i32,
    pub max_seq_len: i32,
    pub shift_terminal: f32,
}

impl ScheduleConfig {
    /// `exp(mu)` for a picture of `tokens` latent pixels, which is the shift
    /// [`FlowSampler`](crate::flow::FlowSampler) bends by.
    ///
    /// The line is extrapolated rather than clamped past either end, as the reference does:
    /// a 2048 by 2048 picture is 16384 tokens, twice `max_seq_len`.
    pub fn shift_for(&self, tokens: i32) -> f32 {
        let slope = (self.max_shift - self.base_shift) as f64
            / (self.max_seq_len - self.base_seq_len) as f64;
        let intercept = self.base_shift as f64 - slope * self.base_seq_len as f64;
        (tokens as f64 * slope + intercept).exp() as f32
    }
}

/// What the denoiser returns. Only a velocity is sampled here.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Prediction {
    Flow,
}

/// The whole of it.
#[derive(Clone, Debug)]
pub struct QwenImageConfig {
    pub dit: DitConfig,
    pub encoder: EncoderConfig,
    pub vae: VaeConfig,
    pub schedule: ScheduleConfig,
    pub prediction: Prediction,
    /// The system turn every prompt is wrapped in, read by the encoder and then dropped.
    pub system_prompt: String,
}

fn number_list<T: std::str::FromStr>(section: &Mapping, key: &str) -> Result<Vec<T>> {
    section
        .get_str(key)?
        .split(',')
        .map(|part| {
            part.trim().parse::<T>().map_err(|_| {
                Error::model(format!("{key} is not a list of numbers: {:?}", part.trim()))
            })
        })
        .collect()
}

impl QwenImageConfig {
    /// The section a package's `[model] type` points at.
    pub const SECTION: &'static str = "qwen_image";

    pub fn from_section(section: &Mapping) -> Result<QwenImageConfig> {
        let weight_format = section.get_weight_format_or("weight_format", WeightFormat::Float)?;
        let latent_channels: i32 = section.get("latent_channels")?;

        let axes: Vec<i32> = number_list(section, "rope_axes")?;
        let [axis_t, axis_h, axis_w] = axes[..] else {
            return Err(Error::model(format!(
                "rope_axes holds {} numbers, and a position here has three axes",
                axes.len()
            )));
        };

        let encoder = EncoderConfig {
            num_layers: section.get("encoder_num_layers")?,
            hidden_size: section.get("encoder_hidden_size")?,
            vocab_size: section.get("encoder_vocab_size")?,
            num_heads: section.get("encoder_num_heads")?,
            num_kv_heads: section.get("encoder_num_kv_heads")?,
            head_dim: section.get("encoder_head_dim")?,
            mlp_size: section.get("encoder_mlp_size")?,
            rope_theta: section.get("encoder_rope_theta")?,
            norm_eps: section.get("encoder_rms_norm_eps")?,
            select_layers: number_list(section, "encoder_select_layers")?,
            weight_format,
        };

        let dit = DitConfig {
            num_blocks: section.get("num_blocks")?,
            hidden_size: section.get("hidden_size")?,
            num_heads: section.get("num_heads")?,
            head_dim: section.get("head_dim")?,
            mlp_size: section.get("mlp_size")?,
            latent_channels,
            timestep_embed_dim: section.get("timestep_embed_dim")?,
            rope_theta: section.get("rope_theta")?,
            rope_axes: (axis_t, axis_h, axis_w),
            norm_eps: section.get("norm_eps")?,
            context_size: encoder.hidden_size,
            weight_format,
        };

        // Each of these loads cleanly and draws noise when it is wrong.
        if dit.num_heads * dit.head_dim != dit.hidden_size {
            return Err(Error::model(format!(
                "{} heads of {} do not make a hidden size of {}",
                dit.num_heads, dit.head_dim, dit.hidden_size
            )));
        }
        if axis_t + axis_h + axis_w != dit.head_dim {
            return Err(Error::model(format!(
                "the rotary axes {axis_t}, {axis_h} and {axis_w} account for {} of a {}-wide head",
                axis_t + axis_h + axis_w,
                dit.head_dim
            )));
        }
        if [axis_t, axis_h, axis_w].iter().any(|axis| axis % 2 != 0) {
            return Err(Error::model(format!(
                "the rotary axes {axis_t}, {axis_h} and {axis_w} do not all hold a whole number \
                 of rotary pairs"
            )));
        }
        // The denoiser reads one layer, not a stack: a package that taps several is a Krea 2
        // package under the wrong name.
        match encoder.select_layers[..] {
            [layer] if layer >= 1 && layer <= encoder.num_layers => {}
            _ => {
                return Err(Error::model(format!(
                    "the encoder is tapped at {:?}, where this model reads one of its {} layers",
                    encoder.select_layers, encoder.num_layers
                )))
            }
        }

        let vae = VaeConfig {
            latent_channels,
            image_channels: section.get("image_channels")?,
            scale: section.get("vae_scale")?,
            temporal_upsample: number_list::<i32>(section, "vae_temporal_upsample")?
                .into_iter()
                .map(|flag| flag != 0)
                .collect(),
            latents_mean: number_list(section, "latents_mean")?,
            latents_std: number_list(section, "latents_std")?,
        };
        for (name, values) in [
            ("latents_mean", &vae.latents_mean),
            ("latents_std", &vae.latents_std),
        ] {
            if values.len() != latent_channels as usize {
                return Err(Error::model(format!(
                    "{name} has {} entries for {latent_channels} latent channels",
                    values.len()
                )));
            }
        }

        let schedule = ScheduleConfig {
            base_shift: section.get("sampler_base_shift")?,
            max_shift: section.get("sampler_max_shift")?,
            base_seq_len: section.get("sampler_base_seq_len")?,
            max_seq_len: section.get("sampler_max_seq_len")?,
            shift_terminal: section.get("sampler_shift_terminal")?,
        };
        if schedule.max_seq_len <= schedule.base_seq_len {
            return Err(Error::model(format!(
                "a shift interpolated from {} tokens to {} is not a line",
                schedule.base_seq_len, schedule.max_seq_len
            )));
        }
        if !(0.0..1.0).contains(&schedule.shift_terminal) {
            return Err(Error::model(format!(
                "a schedule cannot end at a noise level of {}",
                schedule.shift_terminal
            )));
        }

        let prediction = match section.get_str("prediction")? {
            "flow" => Prediction::Flow,
            other => {
                return Err(Error::model(format!(
                    "this package predicts {other:?}, which this runtime cannot sample"
                )))
            }
        };

        Ok(QwenImageConfig {
            dit,
            encoder,
            vae,
            schedule,
            prediction,
            system_prompt: section.get_str("system_prompt")?.to_string(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifest::Manifest;

    /// What `tools/qwen_image_exporter.py` writes for the release, kept here so that a change to
    /// one without the other is a test failure rather than a package nothing can read.
    const RELEASE: &str = r#"
weights:
  - qwen-image-2.1.safetensors

config:
  qwen_image:
    num_blocks: 32
    hidden_size: 4096
    num_heads: 32
    head_dim: 128
    mlp_size: 12288
    latent_channels: 64
    timestep_embed_dim: 256
    rope_theta: 10000
    rope_axes: [16, 56, 56]
    norm_eps: 1e-06
    encoder_num_layers: 36
    encoder_hidden_size: 4096
    encoder_vocab_size: 151936
    encoder_num_heads: 32
    encoder_num_kv_heads: 8
    encoder_head_dim: 128
    encoder_mlp_size: 12288
    encoder_rope_theta: 5e+06
    encoder_rms_norm_eps: 1e-06
    encoder_select_layers: [36]
    vae_scale: 16
    image_channels: 4
    vae_temporal_upsample: [1, 1, 1, 0]
    latents_mean: [0.5126, 0.7721, -0.0631, 1.3506, -0.7855, -2.1025, -0.3458, 1.3722, 1.8873, -1.7177, -0.651, 0.2732, 0.7562, -0.6163, -1.0277, 3.8363, 2.021, 0.0472, 0.932, 2.0087, 2.4954, -0.1391, -1.4249, 1.8464, -0.5236, 1.2826, 3.7046, -1.3035, 2.7286, -1.4518, -1.9036, -1.9955, -0.0342, -1.0265, -0.7636, 3.0555, 0.0746, -3.0751, -0.1076, 1.7376, -1.0914, -1.9435, -0.2784, -1.368, 0.4809, -0.4433, 0.3764, 0.5729, -2.0595, 1.096, -1.326, -2.0211, -5.0179, 0.5275, 4.0162, 1.8505, 0.3026, 1.9373, 1.4937, 0.2632, 0.5547, -1.7121, -0.1562, 0.0304]
    latents_std: [3.2001, 3.2936, 3.4321, 3.0091, 3.1061, 4.0379, 4.0705, 3.791, 3.0785, 3.65, 3.9308, 3.0904, 2.8778, 3.7675, 3.732, 5.0756, 3.2864, 4.0397, 3.1317, 4.0443, 2.9249, 3.9454, 3.0988, 4.2489, 3.4896, 3.8513, 3.9323, 3.4719, 3.7498, 4.283, 3.5694, 4.2467, 3.9037, 3.2947, 5.077, 3.5075, 3.27, 3.4767, 2.8063, 5.1125, 3.5327, 4.7833, 3.1286, 4.1819, 3.8527, 3.8312, 3.5605, 4.3875, 3.9624, 4.0168, 3.5643, 4.055, 5.5614, 4.2963, 4.408, 3.4959, 3.8747, 3.7608, 3.5735, 3.149, 3.7662, 3.6746, 3.4563, 3.8161]
    prediction: flow
    sampler_base_shift: 0.5
    sampler_max_shift: 0.9
    sampler_base_seq_len: 256
    sampler_max_seq_len: 8192
    sampler_shift_terminal: 0.02
    system_prompt: "Comprehend and analyze the provided prompt."
  model:
    type: qwen_image
"#;

    fn parse(text: &str) -> Result<QwenImageConfig> {
        let manifest = Manifest::parse(text)?;
        QwenImageConfig::from_section(manifest.section(QwenImageConfig::SECTION)?)
    }

    fn release() -> QwenImageConfig {
        parse(RELEASE).expect("the exporter's own output parses")
    }

    #[test]
    fn reads_what_the_exporter_writes() {
        let config = release();
        assert_eq!(config.dit.num_blocks, 32);
        assert_eq!(config.dit.hidden_size, 4096);
        assert_eq!(config.dit.rope_axes, (16, 56, 56));
        assert_eq!(config.dit.rope_pairs(), 64);
        assert_eq!(config.dit.context_size, 4096);
        assert_eq!(config.encoder.select_layers, vec![36]);
        assert_eq!(config.vae.image_channels, 4);
        assert_eq!(config.vae.scale, 16);
        assert_eq!(config.vae.temporal_upsample, vec![true, true, true, false]);
        assert_eq!(config.vae.latents_mean.len(), 64);
        assert_eq!(config.prediction, Prediction::Flow);
        assert_eq!(
            config.system_prompt,
            "Comprehend and analyze the provided prompt."
        );
        assert_eq!(config.dit.weight_format, WeightFormat::Float);
    }

    /// `calculate_shift`, at both ends of its line and past the far one.
    #[test]
    fn the_shift_follows_the_number_of_tokens() {
        let schedule = release().schedule;
        assert!((schedule.shift_for(256) - 0.5f32.exp()).abs() < 1e-5);
        assert!((schedule.shift_for(8192) - 0.9f32.exp()).abs() < 1e-5);

        // 1024 by 1024 is a 64 by 64 latent.
        let mu = 0.5 + (0.4 / 7936.0) * (4096.0 - 256.0);
        assert!((schedule.shift_for(4096) - (mu as f32).exp()).abs() < 1e-5);

        // 2048 by 2048 is past the line's end, and is extrapolated rather than clamped.
        assert!(schedule.shift_for(16384) > 0.9f32.exp());
    }

    #[test]
    fn refuses_a_stack_of_tapped_layers() {
        let wrong = RELEASE.replace(
            "encoder_select_layers: [36]",
            "encoder_select_layers: [2, 5]",
        );
        let message = parse(&wrong).expect_err("two taps").to_string();
        assert!(message.contains("tapped"), "{message}");

        let past = RELEASE.replace("encoder_select_layers: [36]", "encoder_select_layers: [37]");
        assert!(parse(&past).is_err(), "37 of 36");
    }

    #[test]
    fn refuses_rotary_axes_that_do_not_account_for_the_head() {
        let wrong = RELEASE.replace("rope_axes: [16, 56, 56]", "rope_axes: [16, 56, 54]");
        let message = parse(&wrong).expect_err("126 is not 128").to_string();
        assert!(message.contains("rotary axes"), "{message}");
    }

    #[test]
    fn refuses_heads_that_do_not_make_the_hidden_size() {
        let wrong = RELEASE.replace("num_heads: 32\n    head_dim", "num_heads: 31\n    head_dim");
        let message = parse(&wrong).expect_err("31 heads of 128").to_string();
        assert!(message.contains("hidden size"), "{message}");
    }

    #[test]
    fn refuses_latent_statistics_of_the_wrong_length() {
        let wrong = RELEASE.replace("3.4563, 3.8161]", "3.4563]");
        let message = parse(&wrong).expect_err("63 is not 64").to_string();
        assert!(message.contains("latents_std"), "{message}");
    }

    #[test]
    fn a_package_says_whether_it_stored_its_matrices_quantized() {
        let quantized = parse(&RELEASE.replace(
            "    prediction: flow",
            "    prediction: flow\n    weight_format: fp8",
        ))
        .unwrap();
        assert_eq!(quantized.dit.weight_format, WeightFormat::Fp8);
        assert_eq!(quantized.encoder.weight_format, WeightFormat::Fp8);
    }
}
