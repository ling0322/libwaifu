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


//! What a Krea 2 package says about itself.
//!
//! `tools/krea2_exporter.py` reads most of these off the weights rather than writing them down,
//! so a release with a different depth or width produces the same keys with different numbers.
//! The few that are in no tensor -- the latent normalization, the rotary split, which of the
//! encoder's layers are tapped, what the model predicts -- it states, and they are parsed here
//! like the rest.

use crate::error::{Error, Result};
use crate::flint::WeightFormat;
use crate::flow::SamplerConfig;
use crate::mapping::Mapping;
use crate::qwen_vae::VaeConfig;

/// What the denoiser returns at each step, which decides what the sampler does with it.
///
/// The same question [`anima::Prediction`](crate::anima::Prediction) asks, asked again rather than
/// shared: a package says what it predicts, and a family that only ever answers one way still has
/// to be refused when it answers another.
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

/// The single-stream MMDiT that does the denoising, and the small transformer inside it that
/// reads the prompt.
#[derive(Clone, Debug)]
pub struct DitConfig {
    /// 28 in the released model.
    pub num_blocks: i32,
    /// 6144, which is `num_heads * head_dim` and not a number of its own.
    pub hidden_size: i32,
    pub num_heads: i32,
    /// Twelve against forty-eight: the attention is grouped four to one.
    pub num_kv_heads: i32,
    pub head_dim: i32,
    /// The SwiGLU's width, which is what the gate and the up projection each produce.
    pub mlp_size: i32,
    pub patch_size: i32,
    pub latent_channels: i32,
    /// What the patchify projection reads: `latent_channels * patch_size * patch_size`. Krea 2
    /// concatenates no mask channel, so unlike Anima's this is exactly the latent.
    pub in_channels: i32,
    /// How wide the timestep sinusoid is before the two linears widen it to `hidden_size`.
    pub timestep_embed_dim: i32,
    /// One base for all three position axes, and a small one: a thousand, where a language model
    /// uses ten thousand and up.
    pub rope_theta: f32,
    /// How the head is divided between time, height and width -- 32, 48 and 48 -- written down
    /// rather than worked out, because it is a choice and not a formula. See [`DitConfig::rope_axes`].
    pub rope_axes: (i32, i32, i32),
    /// The epsilon of every RMSNorm in the denoiser, which is not the encoder's.
    pub norm_eps: f32,
    /// How wide the prompt is where it arrives, which is the encoder's width and not this
    /// model's: the denoiser is 6144 across and the text fusion that feeds it runs at 2560.
    pub text_hidden_size: i32,
    pub text_num_heads: i32,
    /// Equal to `text_num_heads` in the release: the fusion blocks are not grouped.
    pub text_num_kv_heads: i32,
    pub text_mlp_size: i32,
    /// How many blocks attend across the stack of tapped layers, before the projector collapses
    /// it. Two.
    pub text_layerwise_blocks: i32,
    /// How many attend along the sentence afterwards. Two.
    pub text_refiner_blocks: i32,
    /// How many of the encoder's layers are stacked per token -- twelve -- which is what the
    /// projector reads and the one number the two halves of this model have to agree on.
    pub num_text_layers: i32,
    /// How the package stored the matrices this multiplies by, which decides what its projections
    /// are built out of. See [`WeightFormat`].
    pub weight_format: WeightFormat,
}

impl DitConfig {
    /// How the head dimension is divided between the three position axes, in the order they are
    /// concatenated: time, then height, then width.
    ///
    /// A still image sits at time zero, so the first 32 of every 128 are cosines of nothing --
    /// present, contributing a cosine of one, which is not the same as absent.
    pub fn rope_axes(&self) -> (i32, i32, i32) {
        self.rope_axes
    }

    /// How many rotary pairs the head holds, which is half of it.
    pub fn rope_pairs(&self) -> i32 {
        self.head_dim / 2
    }
}

/// The Qwen3-VL text tower, read for twelve of its hidden states.
///
/// Only the language half is in the package. The released encoder is a vision-language model and
/// its vision tower is two gigabytes of weights this never reaches: no picture is handed to it,
/// and for a text-only prompt the interleaved mRoPE it was trained with is exactly the ordinary
/// rotary embedding, because all three of its position axes carry the same index.
#[derive(Clone, Debug)]
pub struct EncoderConfig {
    pub num_layers: i32,
    pub hidden_size: i32,
    pub vocab_size: i32,
    pub num_heads: i32,
    /// Eight against thirty-two, and flint takes k and v already narrow.
    pub num_kv_heads: i32,
    pub head_dim: i32,
    pub mlp_size: i32,
    /// Five million. A language model trained for a long context rotates slowly.
    pub rope_theta: f32,
    pub norm_eps: f32,
    /// Which layers' outputs are stacked and handed to the denoiser, counting the embedding
    /// output as zero the way `output_hidden_states` does. Twelve of them, every third layer.
    ///
    /// In the package because they are a choice of the model's that no tensor records, and
    /// getting them wrong -- the last twelve, say, or every third counted from one -- loads
    /// cleanly and draws something plausible and wrong.
    pub select_layers: Vec<i32>,
    /// How the package stored the matrices this multiplies by. See [`WeightFormat`].
    pub weight_format: WeightFormat,
}

/// The whole of it.
#[derive(Clone, Debug)]
pub struct Krea2Config {
    pub dit: DitConfig,
    pub encoder: EncoderConfig,
    pub vae: VaeConfig,
    pub prediction: Prediction,
    pub sampler: SamplerConfig,
    /// The longest prompt the model was sampled with, in tokens, including the five the template
    /// puts after it. Prompts are cut to it.
    ///
    /// The reference pads every prompt out to this length and masks the padding away everywhere
    /// it could matter; this runtime does not pad at all, which is the same arithmetic and a
    /// shorter sequence. `docs/krea2.md` has the proof.
    pub context_length: i32,
}

/// A comma separated list of numbers, as the package writes the latent statistics and the rotary
/// split.
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

impl Krea2Config {
    /// The section a package's `[model] type` points at.
    pub const SECTION: &'static str = "krea2";

    pub fn from_section(section: &Mapping) -> Result<Krea2Config> {
        let latent_channels: i32 = section.get("latent_channels")?;
        let patch_size: i32 = section.get("patch_size")?;
        let num_text_layers: i32 = section.get("num_text_layers")?;

        // One key for the two halves that have matrices in them. The autoencoder does not read
        // it: it is convolutions, and its few projections are not what this is for.
        let weight_format = section.get_weight_format_or("weight_format", WeightFormat::Float)?;

        let axes: Vec<i32> = number_list(section, "rope_axes")?;
        let [axis_t, axis_h, axis_w] = axes[..] else {
            return Err(Error::model(format!(
                "rope_axes holds {} numbers, and a position here has three axes",
                axes.len()
            )));
        };

        let dit = DitConfig {
            num_blocks: section.get("num_blocks")?,
            hidden_size: section.get("hidden_size")?,
            num_heads: section.get("num_heads")?,
            num_kv_heads: section.get("num_kv_heads")?,
            head_dim: section.get("head_dim")?,
            mlp_size: section.get("mlp_size")?,
            patch_size,
            latent_channels,
            in_channels: section.get("in_channels")?,
            timestep_embed_dim: section.get("timestep_embed_dim")?,
            rope_theta: section.get("rope_theta")?,
            rope_axes: (axis_t, axis_h, axis_w),
            norm_eps: section.get("norm_eps")?,
            text_hidden_size: section.get("text_hidden_size")?,
            text_num_heads: section.get("text_num_heads")?,
            text_num_kv_heads: section.get("text_num_kv_heads")?,
            text_mlp_size: section.get("text_mlp_size")?,
            text_layerwise_blocks: section.get("text_layerwise_blocks")?,
            text_refiner_blocks: section.get("text_refiner_blocks")?,
            num_text_layers,
            weight_format,
        };

        // Every one of these is a silent failure if it is wrong: the tensors still load, the
        // shapes still multiply, and what comes out is noise. They are cheap to check here.
        if dit.num_heads * dit.head_dim != dit.hidden_size {
            return Err(Error::model(format!(
                "{} heads of {} do not make a hidden size of {}",
                dit.num_heads, dit.head_dim, dit.hidden_size
            )));
        }
        if dit.num_kv_heads <= 0 || dit.num_heads % dit.num_kv_heads != 0 {
            return Err(Error::model(format!(
                "{} query heads do not group evenly into {} key heads",
                dit.num_heads, dit.num_kv_heads
            )));
        }
        if axis_t + axis_h + axis_w != dit.head_dim {
            return Err(Error::model(format!(
                "the rotary axes {axis_t}, {axis_h} and {axis_w} account for \
                 {} of a {}-wide head",
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
        let expected = latent_channels * patch_size * patch_size;
        if dit.in_channels != expected {
            return Err(Error::model(format!(
                "patchify reads {} channels where {latent_channels} latent channels over \
                 {patch_size}x{patch_size} patches make {expected}",
                dit.in_channels
            )));
        }

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

        // The one number the two halves have to agree on: the denoiser's projector is a matrix
        // over the tapped layers, so a package that taps eleven and projects twelve is a package
        // whose weights do not multiply.
        if encoder.select_layers.len() as i32 != num_text_layers {
            return Err(Error::model(format!(
                "the encoder taps {} layers where the denoiser reads {num_text_layers}",
                encoder.select_layers.len()
            )));
        }
        if encoder.hidden_size != dit.text_hidden_size {
            return Err(Error::model(format!(
                "the encoder is {} wide where the text fusion that reads it is {}",
                encoder.hidden_size, dit.text_hidden_size
            )));
        }
        // Zero is the embedding output and `num_layers` is the last layer's, so both ends are
        // real answers; anything past that is a layer the encoder does not have.
        if let Some(past) = encoder
            .select_layers
            .iter()
            .find(|&&layer| layer < 0 || layer > encoder.num_layers)
        {
            return Err(Error::model(format!(
                "layer {past} is tapped from an encoder that has {}",
                encoder.num_layers
            )));
        }

        let vae = VaeConfig {
            latent_channels,
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

        Ok(Krea2Config {
            dit,
            encoder,
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

    /// What `tools/krea2_exporter.py` writes for the turbo release, kept here so that a change to
    /// one without the other is a test failure rather than a package nothing can read.
    const TURBO: &str = r#"
weights:
  - krea2-turbo.safetensors

config:
  krea2:
    num_blocks: 28
    hidden_size: 6144
    num_heads: 48
    num_kv_heads: 12
    head_dim: 128
    mlp_size: 16384
    patch_size: 2
    latent_channels: 16
    in_channels: 64
    timestep_embed_dim: 256
    rope_theta: 1000
    rope_axes: [32, 48, 48]
    norm_eps: 1e-05
    text_hidden_size: 2560
    text_num_heads: 20
    text_num_kv_heads: 20
    text_mlp_size: 6912
    text_layerwise_blocks: 2
    text_refiner_blocks: 2
    num_text_layers: 12
    context_length: 512
    encoder_num_layers: 36
    encoder_hidden_size: 2560
    encoder_vocab_size: 151936
    encoder_num_heads: 32
    encoder_num_kv_heads: 8
    encoder_head_dim: 128
    encoder_mlp_size: 9728
    encoder_rope_theta: 5e+06
    encoder_rms_norm_eps: 1e-06
    encoder_select_layers: [2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35]
    latents_mean: [-0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508, 0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921]
    latents_std: [2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743, 3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.916]
    prediction: flow
    sampler_shift: 3.15819
    sampler_multiplier: 1.0
"#;

    fn parse(text: &str) -> Result<Krea2Config> {
        let manifest = Manifest::parse(text)?;
        Krea2Config::from_section(manifest.section(Krea2Config::SECTION)?)
    }

    fn turbo() -> Krea2Config {
        parse(TURBO).expect("the exporter's own output parses")
    }

    #[test]
    fn reads_what_the_exporter_writes() {
        let config = turbo();
        assert_eq!(config.dit.num_blocks, 28);
        assert_eq!(config.dit.hidden_size, 6144);
        assert_eq!(config.dit.num_kv_heads, 12);
        assert_eq!(config.dit.mlp_size, 16384);
        assert_eq!(config.dit.in_channels, 64);
        assert_eq!(config.dit.rope_axes(), (32, 48, 48));
        assert_eq!(config.dit.rope_pairs(), 64);
        assert_eq!(config.dit.text_mlp_size, 6912);
        assert_eq!(config.encoder.num_layers, 36);
        assert_eq!(config.encoder.num_kv_heads, 8);
        assert_eq!(config.encoder.select_layers.len(), 12);
        assert_eq!(config.encoder.select_layers[0], 2);
        assert_eq!(config.context_length, 512);
        assert_eq!(config.prediction, Prediction::Flow);
    }

    /// `exp(1.15)`, which is the reference's fixed mu written the way this runtime bends a
    /// schedule. A package that wrote the mu itself would run a schedule that is barely bent and
    /// would still draw something.
    #[test]
    fn carries_the_distilled_shift_as_an_exponential() {
        let shift = turbo().sampler.shift;
        assert!(
            (shift - 1.15f32.exp()).abs() < 1e-4,
            "the shift is {shift}, not exp(1.15)"
        );
        assert_eq!(turbo().sampler.multiplier, 1.0);
    }

    #[test]
    fn refuses_heads_that_do_not_make_the_hidden_size() {
        let wrong = TURBO.replace("num_heads: 48", "num_heads: 47");
        let message = parse(&wrong)
            .expect_err("47 heads of 128 are not 6144")
            .to_string();
        assert!(message.contains("hidden size"), "{message}");
    }

    #[test]
    fn refuses_query_heads_that_do_not_group() {
        let wrong = TURBO.replace("num_kv_heads: 12", "num_kv_heads: 5");
        let message = parse(&wrong)
            .expect_err("48 does not group into 5")
            .to_string();
        assert!(message.contains("group"), "{message}");
    }

    #[test]
    fn refuses_rotary_axes_that_do_not_account_for_the_head() {
        let wrong = TURBO.replace("rope_axes: [32, 48, 48]", "rope_axes: [32, 48, 46]");
        let message = parse(&wrong).expect_err("126 is not 128").to_string();
        assert!(message.contains("rotary axes"), "{message}");

        // And one that adds up but cannot hold whole pairs.
        let odd = TURBO.replace("rope_axes: [32, 48, 48]", "rope_axes: [31, 48, 49]");
        let message = parse(&odd).expect_err("31 is not even").to_string();
        assert!(message.contains("rotary pairs"), "{message}");
    }

    #[test]
    fn refuses_a_patchify_that_does_not_match_the_patch() {
        // 16 rather than 64 is exactly what forgetting the patch would look like, and it is the
        // one mistake here that would otherwise load and draw noise.
        let wrong = TURBO.replace("in_channels: 64", "in_channels: 16");
        let message = parse(&wrong).expect_err("16 forgets the patch").to_string();
        assert!(message.contains("patchify"), "{message}");
    }

    /// The projector is a matrix over the tapped layers, so these two numbers are one number
    /// written twice and a package that disagrees with itself has weights that do not multiply.
    #[test]
    fn refuses_a_tapped_layer_count_the_denoiser_does_not_read() {
        let wrong = TURBO.replace("num_text_layers: 12", "num_text_layers: 11");
        let message = parse(&wrong).expect_err("twelve taps, eleven read").to_string();
        assert!(message.contains("taps"), "{message}");
    }

    #[test]
    fn refuses_a_tap_past_the_end_of_the_encoder() {
        let wrong = TURBO.replace("32, 35]", "32, 36]");
        assert!(
            parse(&wrong).is_ok(),
            "36 of 36 layers is the last layer's output, which is a real answer"
        );

        let past = TURBO.replace("32, 35]", "32, 37]");
        let message = parse(&past).expect_err("37 of 36").to_string();
        assert!(message.contains("tapped"), "{message}");
    }

    #[test]
    fn refuses_an_encoder_the_fusion_cannot_read() {
        let wrong = TURBO.replace("encoder_hidden_size: 2560", "encoder_hidden_size: 2048");
        let message = parse(&wrong).expect_err("2048 is not 2560").to_string();
        assert!(message.contains("wide"), "{message}");
    }

    #[test]
    fn refuses_a_prediction_it_cannot_sample() {
        let epsilon = TURBO.replace("prediction: flow", "prediction: epsilon");
        let message = parse(&epsilon)
            .expect_err("epsilon is not flow")
            .to_string();
        assert!(message.contains("epsilon"), "{message}");
    }

    #[test]
    fn refuses_latent_statistics_of_the_wrong_length() {
        let wrong = TURBO.replace("2.8251, 1.916]", "2.8251]");
        let message = parse(&wrong)
            .expect_err("fifteen is not sixteen")
            .to_string();
        assert!(message.contains("latents_std"), "{message}");
    }

    /// One key, read by the two halves that have matrices in them, and float when it is absent.
    #[test]
    fn a_package_says_whether_it_stored_its_matrices_quantized() {
        let config = turbo();
        assert_eq!(config.dit.weight_format, WeightFormat::Float);
        assert_eq!(config.encoder.weight_format, WeightFormat::Float);

        let quantized = parse(&format!("{TURBO}    weight_format: fp8\n")).unwrap();
        assert_eq!(quantized.dit.weight_format, WeightFormat::Fp8);
        assert_eq!(quantized.encoder.weight_format, WeightFormat::Fp8);

        let tensor_scale = parse(&format!("{TURBO}    weight_format: fp8_tensor_scale\n")).unwrap();
        assert_eq!(tensor_scale.dit.weight_format, WeightFormat::Fp8TensorScale);
        assert_eq!(tensor_scale.encoder.weight_format, WeightFormat::Fp8TensorScale);
    }
}
