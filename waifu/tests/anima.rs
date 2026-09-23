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

//! Anima's parts against what ComfyUI produces for the same inputs, by way of the test
//! tensors `tools/anima_comfy.py` writes into the package.
//!
//! The reference is ComfyUI run directly on the same weights, so agreeing with it here is
//! agreeing with the model rather than with a second opinion of our own. None of what passes
//! between these stages can be judged by
//! looking -- a hidden state, a context, a velocity -- and an error in any of them shows up only
//! as a picture that is subtly not the prompt, so they are compared as numbers.
//!
//! Each stage is measured against the reference's own inputs where it can be, so that what a
//! number says is where the error is rather than how far along it happened.

use std::cell::OnceCell;
use std::collections::HashMap;
use std::path::PathBuf;
use std::rc::Rc;

use waifu::anima::{
    Adapter, AdapterConfig, Dit, DitConfig, FlowSampler, SamplerConfig, TextConfig, TextEncoder,
    VaeConfig, VaeDecoder
};
use waifu::flint::{ParamSource, Tensor};
use waifu::{read_safetensors, DType, Device, Manifest, Residency, WeightFormat};

fn models_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../models")
}

fn device() -> Device {
    Device::Cuda
}

/// The whole package on the device, read once for this test binary rather than per test.
fn weights() -> Rc<dyn ParamSource> {
    thread_local! {
        static WEIGHTS: OnceCell<Rc<dyn ParamSource>> = const { OnceCell::new() };
    }

    WEIGHTS.with(|cell| {
        Rc::clone(cell.get_or_init(|| {
            let manifest = Manifest::open(models_dir().join("anima-turbo-v11.yaml")).unwrap();
            <dyn ParamSource>::from_files(
                &manifest.weight_paths().unwrap(),
                device(),
                Residency::Device,
            )
            .unwrap()
        }))
    })
}

fn cases() -> HashMap<String, Tensor> {
    read_safetensors(&[models_dir().join("anima-turbo-v11_test.safetensors")]).unwrap()
}

/// The root mean square of the difference over the root mean square of the reference, which says
/// whether two tensors are the same answer rather than whether any one element is.
fn relative_rmse(actual: &Tensor, reference: &Tensor) -> f32 {
    let a = actual
        .to_device(Device::Cpu)
        .unwrap()
        .cast(DType::Float)
        .unwrap()
        .to_vec_f32()
        .unwrap();
    let b = reference.to_vec_f32().unwrap();
    assert_eq!(a.len(), b.len(), "the shapes do not match");

    let error: f64 = a
        .iter()
        .zip(&b)
        .map(|(x, y)| (*x as f64 - *y as f64).powi(2))
        .sum();
    let scale: f64 = b.iter().map(|y| (*y as f64).powi(2)).sum();
    (error / scale).sqrt() as f32
}

/// What the package says Qwen3-0.6B is. Read off the weights by the exporter; written here so
/// that the test says what it is testing.
fn config() -> TextConfig {
    TextConfig {
        num_layers: 28,
        hidden_size: 1024,
        vocab_size: 151936,
        num_heads: 16,
        num_kv_heads: 8,
        head_dim: 128,
        mlp_size: 3072,
        rope_theta: 1e6,
        norm_eps: 1e-6,
        weight_format: WeightFormat::Float
    }
}

#[test]
#[ignore = "needs the anima package"]
fn encodes_a_prompt_the_way_the_reference_does() {
    let cases = cases();
    let ids = cases["test_case.qwen_ids"].clone();
    let length = ids.shape().last().copied().unwrap();
    let ids = ids.view(&[length]).unwrap().to_device(device()).unwrap();

    let encoder = TextEncoder::build(config(), "anima.text", &weights(), DType::Float16).unwrap();
    let hidden = encoder.forward(&ids).unwrap();
    assert_eq!(hidden.shape(), vec![1, length, 1024]);

    let reference = cases["test_case.hidden"].clone();
    let rmse = relative_rmse(&hidden, &reference);
    println!("qwen3 hidden rmse = {rmse}");

    // The reference runs in float32 and this in float16 over twenty-eight layers, so they will
    // not agree to the bit. They have to agree to a great deal better than the difference between
    // one prompt and another.
    assert!(rmse < 2e-2, "the hidden state drifted by {rmse}");
}

#[test]
#[ignore = "needs the anima package"]
fn refuses_heads_that_do_not_group() {
    let broken = TextConfig {
        num_kv_heads: 7,
        ..config()
    };
    let message = TextEncoder::build(broken, "anima.text", &weights(), DType::Float16)
        .expect_err("16 does not group into 7")
        .to_string();
    assert!(message.contains("group"), "{message}");
}

/// What the package says the adapter is.
fn adapter_config() -> AdapterConfig {
    AdapterConfig {
        num_blocks: 6,
        hidden_size: 1024,
        num_heads: 16,
        head_dim: 64,
        vocab_size: 32128,
        mlp_size: 4096,
        weight_format: WeightFormat::Float
    }
}

#[test]
#[ignore = "needs the anima package"]
fn the_adapter_matches_the_reference() {
    let cases = cases();

    let qwen_ids = cases["test_case.qwen_ids"].clone();
    let length = qwen_ids.shape().last().copied().unwrap();
    let qwen_ids = qwen_ids
        .view(&[length])
        .unwrap()
        .to_device(device())
        .unwrap();

    let t5_ids = cases["test_case.t5_ids"].clone();
    let t5_length = t5_ids.shape().last().copied().unwrap();
    let t5_ids = t5_ids
        .view(&[t5_length])
        .unwrap()
        .to_device(device())
        .unwrap();

    // Against this runtime's own hidden states rather than the reference's, so that what is
    // measured is the adapter and not the encoder in front of it a second time.
    let encoder = TextEncoder::build(config(), "anima.text", &weights(), DType::Float16).unwrap();
    let hidden = encoder.forward(&qwen_ids).unwrap();

    let adapter = Adapter::build(
        adapter_config(),
        "anima.adapter",
        &weights(),
        DType::Float16,
    )
    .unwrap();
    let context = adapter.forward(&t5_ids, &hidden).unwrap();
    assert_eq!(context.shape(), vec![1, t5_length, 1024]);

    let reference = cases["test_case.context"].clone();
    let rmse = relative_rmse(&context, &reference);
    println!("adapter context rmse = {rmse}");

    // Looser than the encoder's bound because the encoder's own drift arrives here first, and
    // the adapter's output is small enough that a relative measure magnifies it.
    assert!(rmse < 3e-2, "the context drifted by {rmse}");
}

/// What the package says the denoiser is.
fn dit_config() -> DitConfig {
    DitConfig {
        num_blocks: 28,
        hidden_size: 2048,
        num_heads: 16,
        head_dim: 128,
        mlp_size: 8192,
        adaln_lora_dim: 256,
        patch_size: 2,
        latent_channels: 16,
        patchify_channels: 68,
        rope_theta: 10000.0,
        // Four times the spatial extent, and time left alone. The base each axis rotates on is
        // these bent by how wide it is, which puts height and width at 42870.9.
        rope_h_ratio: 4.0,
        rope_w_ratio: 4.0,
        rope_t_ratio: 1.0,
        context_dim: 1024,
        weight_format: WeightFormat::Float
    }
}

#[test]
#[ignore = "needs the anima package"]
fn one_step_matches_the_reference() {
    let cases = cases();

    // Against the reference's own context rather than this runtime's, so that what is measured
    // is the denoiser alone and not the two stages in front of it a second time.
    let context = cases["test_case.context_padded"].to_device(device())
        .unwrap()
        .cast(DType::Float16)
        .unwrap();
    let latent = cases["test_case.latent"].to_device(device())
        .unwrap()
        .cast(DType::Float16)
        .unwrap();

    let dit = Dit::build(dit_config(), "anima.dit", &weights(), DType::Float16).unwrap();
    let velocity = dit.forward(&latent, 0.75, &context).unwrap();
    assert_eq!(velocity.shape(), latent.shape());

    let reference = cases["test_case.velocity"].clone();
    let rmse = relative_rmse(&velocity, &reference);
    println!("dit velocity rmse = {rmse}");

    // Twenty-eight blocks of fp16 against a float32 reference, and this model's residual stream
    // is large enough that ComfyUI keeps its own in float32 for exactly this reason.
    assert!(rmse < 5e-2, "the velocity drifted by {rmse}");
}

#[test]
#[ignore = "needs the anima package"]
fn refuses_a_latent_that_does_not_divide_into_patches() {
    let dit = Dit::build(dit_config(), "anima.dit", &weights(), DType::Float16).unwrap();
    let odd = waifu::flint::Tensor::zeros(&[1, 16, 15, 16], DType::Float16, device()).unwrap();
    let context = waifu::flint::Tensor::zeros(&[1, 512, 1024], DType::Float16, device()).unwrap();

    let message = dit
        .forward(&odd, 0.5, &context)
        .expect_err("15 is not a multiple of 2")
        .to_string();
    assert!(message.contains("patches"), "{message}");
}

/// The latent normalization the Qwen-Image VAE carries, one pair per channel.
fn vae_config() -> VaeConfig {
    VaeConfig {
        latent_channels: 16,
        latents_mean: vec![
            -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508, 0.4134, -0.0715,
            0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921,
        ],
        latents_std: vec![
            2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743, 3.2687, 2.1526, 2.8652,
            1.5579, 1.6382, 1.1253, 2.8251, 1.9160,
        ]
    }
}

#[test]
#[ignore = "needs the anima package"]
fn decodes_a_latent_the_way_the_reference_does() {
    let cases = cases();
    let latent = cases["test_case.latent"].to_device(device())
        .unwrap();

    let decoder = VaeDecoder::build(
        vae_config(),
        "anima.vae",
        &weights(),
        device(),
        DType::Float16,
    )
    .unwrap();
    let image = decoder.forward(&latent).unwrap();

    // An eighth of the side, three channels out of sixteen: a 16 by 16 latent is 128 by 128.
    assert_eq!(image.shape(), vec![1, 3, 128, 128]);

    let reference = cases["test_case.decoded"].clone();
    let rmse = relative_rmse(&image, &reference);
    println!("vae decoded rmse = {rmse}");

    // This one runs in float32 on both sides, so it should agree far more closely than the
    // stages above it, which are half against whole.
    assert!(rmse < 5e-3, "the picture drifted by {rmse}");
}

/// How much a picture differs from itself one pixel to the right, against how much it varies
/// overall.
///
/// This is the one thing that tells a drawing from static without looking at it. Neighbouring
/// pixels of a real image are nearly the same, so the ratio is small; in white noise they are
/// independent, and the mean absolute difference of two independent draws is about 1.13 standard
/// deviations. Anything near that has not drawn anything.
/// The root mean square of a tensor, for watching a sampling loop stay finite.
fn spread(x: &Tensor) -> f32 {
    let values = x
        .to_device(Device::Cpu)
        .unwrap()
        .cast(DType::Float)
        .unwrap()
        .to_vec_f32()
        .unwrap();
    let sum: f64 = values.iter().map(|v| (*v as f64).powi(2)).sum();
    (sum / values.len() as f64).sqrt() as f32
}

fn roughness(image: &Tensor) -> f32 {
    let width = image.shape_at(3).unwrap() as usize;
    let pixels = image
        .to_device(Device::Cpu)
        .unwrap()
        .cast(DType::Float)
        .unwrap()
        .to_vec_f32()
        .unwrap();

    let mut steps = 0.0f64;
    let mut count = 0usize;
    for (index, value) in pixels.iter().enumerate() {
        if index % width != width - 1 {
            steps += (pixels[index + 1] - value).abs() as f64;
            count += 1;
        }
    }

    let mean = pixels.iter().map(|v| *v as f64).sum::<f64>() / pixels.len() as f64;
    let variance = pixels
        .iter()
        .map(|v| (*v as f64 - mean).powi(2))
        .sum::<f64>()
        / pixels.len() as f64;

    ((steps / count as f64) / variance.sqrt()) as f32
}

#[test]
#[ignore = "needs the anima package"]
fn draws_a_picture_end_to_end() {
    let cases = cases();

    // From the reference's context, so that this is the sampler, the denoiser and the decoder
    // being tested together rather than the tokenizers, which are still to come.
    let context = cases["test_case.context_padded"].to_device(device())
        .unwrap()
        .cast(DType::Float16)
        .unwrap();

    let dit = Dit::build(dit_config(), "anima.dit", &weights(), DType::Float16).unwrap();
    let decoder = VaeDecoder::build(
        vae_config(),
        "anima.vae",
        &weights(),
        device(),
        DType::Float16,
    )
    .unwrap();

    // Turbo is distilled for this: eight steps at no guidance, one pass of the denoiser each.
    let sampler = FlowSampler::new(
        &SamplerConfig {
            shift: 3.0,
            multiplier: 1.0
        },
        8,
    )
    .unwrap();

    waifu::flint::functional::manual_seed(device(), 0x5eed).unwrap();
    let mut latent = waifu::flint::functional::randn(&[1, 16, 32, 32], device())
        .unwrap()
        .cast(DType::Float16)
        .unwrap();

    for index in 0..sampler.steps() {
        let velocity = dit
            .forward(&latent, sampler.timestep(index).unwrap(), &context)
            .unwrap();
        latent = sampler
            .step(index, &latent, &velocity)
            .unwrap()
            .cast(DType::Float16)
            .unwrap();
    }

    let image = decoder.forward(&latent).unwrap();
    assert_eq!(image.shape(), vec![1, 3, 256, 256]);

    let rough = roughness(&image);
    println!("end to end roughness = {rough}");

    // White noise sits near 1.13. A drawing is far below it; the eight-step turbo schedule on
    // this prompt comes out around a tenth of that, and anything under half is unambiguous.
    assert!(
        rough < 0.5,
        "what came out has the texture of noise, not of a picture: {rough}"
    );
}
