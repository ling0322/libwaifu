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

//! What an Anima package says about itself.
//!
//! `tools/anima_exporter.py` reads most of these off the weights rather than writing them down,
//! so a forty-block expansion and the twenty-eight block release produce the same keys with
//! different numbers. The few that are in no tensor -- the latent normalization, the rotary
//! bases, what the model predicts -- it states, and they are parsed here like the rest.

use super::sampler::SamplerConfig;
use crate::error::{Error, Result};
use crate::mapping::Mapping;

/// What the denoiser returns at each step, which decides what the sampler does with it.
///
/// SDXL checkpoints are all epsilon and `waifu::sdxl` reads nothing else. Anima is rectified
/// flow. The two are not interchangeable and nothing downstream can tell them apart by looking,
/// so the package says which it is and this refuses anything it does not know.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Prediction {
    /// The model returns a velocity along a straight path from noise to image.
    Flow,
}

impl Prediction {
    fn named(name: &str) -> Result<Prediction> {
        match name {
            "flow" => Ok(Prediction::Flow),
            other => Err(Error::model(format!(
                "this package predicts {other:?}, which this runtime cannot sample"
            ))),
        }
    }
}

/// The Cosmos-Predict2 transformer that does the denoising.
#[derive(Clone, Debug)]
pub struct DitConfig {
    /// 28 for the published releases, 40 for the community expansion.
    pub num_blocks: i32,
    pub hidden_size: i32,
    pub num_heads: i32,
    pub head_dim: i32,
    /// The ungated MLP's width, four times the hidden size.
    pub mlp_size: i32,
    /// The bottleneck the AdaLN modulation passes through: 2048 -> 256 -> 6144.
    pub adaln_lora_dim: i32,
    pub patch_size: i32,
    pub latent_channels: i32,
    /// What the patchify projection reads: `(latent_channels + 1) * patch_size * patch_size`.
    /// The extra channel is the padding mask Cosmos concatenates, zero for every image drawn
    /// here but read all the same.
    pub patchify_channels: i32,
    /// The base before extrapolation, which is 10000 on every axis. What each axis actually
    /// rotates on is `rope_bases`, which is this bent by the ratios below.
    pub rope_theta: f32,
    /// How far each position axis is extrapolated. Not bases: NTK factors, applied as
    /// `rope_theta * ratio ** (dim / (dim - 2))`.
    ///
    /// They are read from the package rather than assumed here because assuming them is how they
    /// came to be wrong. Anima asks for four times the spatial extent and leaves time alone, and
    /// a flat base on all three axes is 13% off on the velocity while still drawing a picture
    /// that looks perfectly reasonable. It was caught by running ComfyUI beside it.
    pub rope_h_ratio: f32,
    pub rope_w_ratio: f32,
    pub rope_t_ratio: f32,
    /// How wide the cross attention's context is, which is the adapter's width and not this
    /// model's: the denoiser is 2048 across and reads a 1024-wide prompt.
    pub context_dim: i32,
}

impl DitConfig {
    /// How the head dimension is divided between the three position axes.
    ///
    /// Cosmos places a token by time, height and width rather than by one index, and splits the
    /// head between them: a sixth each to height and width, rounded down to an even number of
    /// rotary pairs, and whatever is left to time. At the 128 Anima uses that is 42, 42 and 44.
    ///
    /// The remainder goes to time rather than being spread around, so the three do not generally
    /// come out equal and the order they are concatenated in -- t, then h, then w -- matters.
    pub fn rope_axes(&self) -> (i32, i32, i32) {
        let spatial = self.head_dim / 6 * 2;
        (self.head_dim - 2 * spatial, spatial, spatial)
    }

    /// What each axis rotates on, in the order `rope_axes` returns them: time, height, width.
    ///
    /// The extrapolation ratio is an NTK factor rather than a base, so it enters as
    /// `rope_theta * ratio ** (dim / (dim - 2))` and depends on how wide the axis is. At Anima's
    /// 4.0 over 42 dimensions that puts height and width at 42870.9, while time, which is not
    /// extrapolated, stays at the flat 10000.
    pub fn rope_bases(&self) -> (f64, f64, f64) {
        let (dim_t, dim_h, dim_w) = self.rope_axes();
        let base = |dim: i32, ratio: f32| {
            let exponent = dim as f64 / (dim - 2) as f64;
            self.rope_theta as f64 * (ratio as f64).powf(exponent)
        };
        (
            base(dim_t, self.rope_t_ratio),
            base(dim_h, self.rope_h_ratio),
            base(dim_w, self.rope_w_ratio),
        )
    }
}

/// The bridge from the text encoder to the cross-attention the denoiser was pretrained for.
#[derive(Clone, Debug)]
pub struct AdapterConfig {
    pub num_blocks: i32,
    pub hidden_size: i32,
    pub num_heads: i32,
    pub head_dim: i32,
    /// A T5 vocabulary, which is why it is not the encoder's. Its ids are the query stream.
    pub vocab_size: i32,
    /// Four times the hidden size. The reference fixes the ratio rather than storing it, and no
    /// tensor in the package contradicts it, so it is worked out rather than read.
    pub mlp_size: i32,
}

/// Qwen3-0.6B, read for its hidden states.
#[derive(Clone, Debug)]
pub struct TextConfig {
    pub num_layers: i32,
    pub hidden_size: i32,
    pub vocab_size: i32,
    pub num_heads: i32,
    /// Fewer than `num_heads`: the attention is grouped, and flint takes k and v already narrow.
    pub num_kv_heads: i32,
    pub head_dim: i32,
    pub mlp_size: i32,
    pub rope_theta: f32,
    pub norm_eps: f32,
}

/// The Qwen-Image VAE, as far as the runtime needs to know it.
#[derive(Clone, Debug)]
pub struct VaeConfig {
    pub latent_channels: i32,
    /// Per channel, not one factor for all of them the way SDXL has it.
    pub latents_mean: Vec<f32>,
    pub latents_std: Vec<f32>,
}

/// The whole of it.
#[derive(Clone, Debug)]
pub struct AnimaConfig {
    pub dit: DitConfig,
    pub adapter: AdapterConfig,
    pub text: TextConfig,
    pub vae: VaeConfig,
    pub prediction: Prediction,
    pub sampler: SamplerConfig,
    /// What the adapter pads to, and so how many tokens the cross attention reads.
    pub context_length: i32,
}

/// A comma separated list of numbers, as the package writes the latent statistics.
fn float_list(section: &Mapping, key: &str) -> Result<Vec<f32>> {
    section
        .get_str(key)?
        .split(',')
        .map(|part| {
            part.trim().parse::<f32>().map_err(|_| {
                Error::model(format!("{key} is not a list of numbers: {:?}", part.trim()))
            })
        })
        .collect()
}

impl AnimaConfig {
    /// The section a package's `[model] type` points at.
    pub const SECTION: &'static str = "anima";

    pub fn from_section(section: &Mapping) -> Result<AnimaConfig> {
        let latent_channels: i32 = section.get("latent_channels")?;
        let patch_size: i32 = section.get("patch_size")?;

        let dit = DitConfig {
            num_blocks: section.get("num_blocks")?,
            hidden_size: section.get("hidden_size")?,
            num_heads: section.get("num_heads")?,
            head_dim: section.get("head_dim")?,
            mlp_size: section.get("mlp_size")?,
            adaln_lora_dim: section.get("adaln_lora_dim")?,
            patch_size,
            latent_channels,
            patchify_channels: section.get("patchify_channels")?,
            rope_theta: section.get("rope_theta")?,
            rope_h_ratio: section.get("rope_h_extrapolation_ratio")?,
            rope_w_ratio: section.get("rope_w_extrapolation_ratio")?,
            rope_t_ratio: section.get("rope_t_extrapolation_ratio")?,
            context_dim: section.get("adapter_hidden_size")?,
        };

        // Every one of these is a silent failure if it is wrong: the tensors still load, the
        // shapes still multiply, and what comes out is noise. They are cheap to check here.
        if dit.num_heads * dit.head_dim != dit.hidden_size {
            return Err(Error::model(format!(
                "{} heads of {} do not make a hidden size of {}",
                dit.num_heads, dit.head_dim, dit.hidden_size
            )));
        }
        let expected = (latent_channels + 1) * patch_size * patch_size;
        if dit.patchify_channels != expected {
            return Err(Error::model(format!(
                "patchify reads {} channels where {latent_channels} latent channels and a \
                 padding mask over {patch_size}x{patch_size} patches make {expected}",
                dit.patchify_channels
            )));
        }

        let adapter_hidden: i32 = section.get("adapter_hidden_size")?;
        let adapter_heads: i32 = section.get("adapter_num_heads")?;
        if adapter_hidden % adapter_heads != 0 {
            return Err(Error::model(format!(
                "an adapter of {adapter_hidden} does not divide into {adapter_heads} heads"
            )));
        }

        let vae = VaeConfig {
            latent_channels,
            latents_mean: float_list(section, "latents_mean")?,
            latents_std: float_list(section, "latents_std")?,
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

        Ok(AnimaConfig {
            dit,
            adapter: AdapterConfig {
                num_blocks: section.get("adapter_blocks")?,
                hidden_size: adapter_hidden,
                num_heads: adapter_heads,
                head_dim: adapter_hidden / adapter_heads,
                vocab_size: section.get("adapter_vocab_size")?,
                mlp_size: 4 * adapter_hidden,
            },
            text: TextConfig {
                num_layers: section.get("text_num_layers")?,
                hidden_size: section.get("text_hidden_size")?,
                vocab_size: section.get("text_vocab_size")?,
                num_heads: section.get("text_num_heads")?,
                num_kv_heads: section.get("text_num_kv_heads")?,
                head_dim: section.get("text_head_dim")?,
                mlp_size: section.get("text_mlp_size")?,
                rope_theta: section.get("text_rope_theta")?,
                norm_eps: section.get("text_rms_norm_eps")?,
            },
            vae,
            prediction: Prediction::named(section.get_str("prediction")?)?,
            sampler: SamplerConfig {
                shift: section.get("sampler_shift")?,
                multiplier: section.get("sampler_multiplier")?,
            },
            context_length: section.get("context_length")?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifest::Manifest;

    /// What `tools/anima_exporter.py` writes for turbo v1.1, kept here so that a change to one
    /// without the other is a test failure rather than a package nothing can read.
    const TURBO_V11: &str = r#"
# What this model is, and which files it is made of.
#
# The safetensors beside this file hold the weights and nothing else; everything
# that says what they are is here.

weights:
  - anima-turbo-v11.safetensors

config:
  anima:
    num_blocks: 28
    hidden_size: 2048
    num_heads: 16
    head_dim: 128
    mlp_size: 8192
    adaln_lora_dim: 256
    patch_size: 2
    latent_channels: 16
    patchify_channels: 68
    rope_theta: 10000
    rope_h_extrapolation_ratio: 4
    rope_w_extrapolation_ratio: 4
    rope_t_extrapolation_ratio: 1
    adapter_blocks: 6
    adapter_hidden_size: 1024
    adapter_vocab_size: 32128
    adapter_num_heads: 16
    context_length: 512
    text_num_layers: 28
    text_hidden_size: 1024
    text_vocab_size: 151936
    text_num_heads: 16
    text_num_kv_heads: 8
    text_head_dim: 128
    text_mlp_size: 3072
    text_rope_theta: 1000000
    text_rms_norm_eps: 1e-6
    latents_mean: -0.7571,-0.7089,-0.9113,0.1075,-0.1745,0.9653,-0.1517,1.5508,0.4134,-0.0715,0.5517,-0.3632,-0.1922,-0.9497,0.2503,-0.2921
    latents_std: 2.8184,1.4541,2.3275,2.6558,1.2196,1.7708,2.6052,2.0743,3.2687,2.1526,2.8652,1.5579,1.6382,1.1253,2.8251,1.916
    prediction: flow
    sampler_shift: 3.0
    sampler_multiplier: 1.0
"#;

    fn parse(text: &str) -> Result<AnimaConfig> {
        let ini = Manifest::parse(text)?;
        AnimaConfig::from_section(ini.section(AnimaConfig::SECTION)?)
    }

    fn turbo() -> AnimaConfig {
        parse(TURBO_V11).expect("the exporter's own output parses")
    }

    #[test]
    fn reads_what_the_exporter_writes() {
        let config = turbo();
        assert_eq!(config.dit.num_blocks, 28);
        assert_eq!(config.dit.hidden_size, 2048);
        assert_eq!(config.dit.mlp_size, 8192);
        assert_eq!(config.dit.adaln_lora_dim, 256);
        assert_eq!(config.adapter.num_blocks, 6);
        // The adapter's heads are half the width of the denoiser's, which is why it carries its
        // own head dimension rather than borrowing one.
        assert_eq!(config.adapter.head_dim, 64);
        assert_eq!(config.adapter.vocab_size, 32128);
        assert_eq!(config.text.num_kv_heads, 8);
        assert_eq!(config.context_length, 512);
        assert_eq!(config.prediction, Prediction::Flow);
    }

    #[test]
    fn splits_the_head_between_the_three_axes() {
        let (t, h, w) = turbo().dit.rope_axes();
        assert_eq!((t, h, w), (44, 42, 42));
        assert_eq!(
            t + h + w,
            128,
            "the axes have to account for the whole head"
        );
    }

    #[test]
    fn extrapolates_the_spatial_axes_and_leaves_time_alone() {
        let (base_t, base_h, base_w) = turbo().dit.rope_bases();

        // 10000 * 4 ** (42 / 40). The ratio is an NTK factor and not the base, so it is bent by
        // how wide the axis is and does not come out a round number.
        assert!(
            (base_h - 42870.93).abs() < 0.01,
            "height should rotate on 42870.93, not {base_h}"
        );
        assert_eq!(base_h, base_w, "height and width are extrapolated alike");

        // The one that is not extrapolated. A flat 10000 on all three axes is what this runtime
        // shipped with: 13% off on the velocity, and it drew pictures that looked right.
        assert_eq!(base_t, 10000.0, "time should not be extrapolated");
    }

    #[test]
    fn the_axes_always_account_for_the_whole_head() {
        for head_dim in [32, 48, 64, 96, 128, 256] {
            let dit = DitConfig {
                head_dim,
                ..turbo().dit
            };
            let (t, h, w) = dit.rope_axes();
            assert_eq!(t + h + w, head_dim, "head_dim {head_dim} came apart");
            assert_eq!(h, w, "height and width should be given the same room");
            assert_eq!(
                h % 2,
                0,
                "an axis has to hold a whole number of rotary pairs"
            );
        }
    }

    #[test]
    fn keeps_the_latent_statistics_per_channel() {
        let config = turbo();
        assert_eq!(config.vae.latents_mean.len(), 16);
        assert_eq!(config.vae.latents_std.len(), 16);
        assert!((config.vae.latents_mean[0] - -0.7571).abs() < 1e-6);
        assert!((config.vae.latents_std[15] - 1.916).abs() < 1e-6);
    }

    #[test]
    fn refuses_a_prediction_it_cannot_sample() {
        let epsilon = TURBO_V11.replace("prediction: flow", "prediction: epsilon");
        let message = parse(&epsilon)
            .expect_err("epsilon is not flow")
            .to_string();
        assert!(message.contains("epsilon"), "{message}");
    }

    #[test]
    fn refuses_heads_that_do_not_make_the_hidden_size() {
        let wrong = TURBO_V11.replace("num_heads: 16", "num_heads: 12");
        let message = parse(&wrong)
            .expect_err("12 heads of 128 are not 2048")
            .to_string();
        assert!(message.contains("hidden size"), "{message}");
    }

    #[test]
    fn refuses_a_patchify_that_forgot_the_padding_mask() {
        // 64 rather than 68 is exactly what dropping the mask channel would look like, and it is
        // the one mistake here that would otherwise load and draw noise.
        let wrong = TURBO_V11.replace("patchify_channels: 68", "patchify_channels: 64");
        let message = parse(&wrong).expect_err("64 forgets the mask").to_string();
        assert!(message.contains("padding mask"), "{message}");
    }

    #[test]
    fn refuses_latent_statistics_of_the_wrong_length() {
        let wrong = TURBO_V11.replace("2.8251,1.916", "2.8251");
        let message = parse(&wrong)
            .expect_err("fifteen is not sixteen")
            .to_string();
        assert!(message.contains("latents_std"), "{message}");
    }

    #[test]
    fn reads_the_forty_block_expansion_too() {
        // The community 2.9B model differs from the release in one number, and the exporter reads
        // that number off the weights rather than being told.
        let expanded = TURBO_V11.replace("num_blocks: 28", "num_blocks: 40");
        assert_eq!(
            parse(&expanded)
                .expect("it is the same model")
                .dit
                .num_blocks,
            40
        );
    }
}
