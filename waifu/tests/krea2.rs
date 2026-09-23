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


//! Krea 2's parts against what diffusers produces for the same inputs, by way of the test tensors
//! `tools/krea2_exporter.py` writes beside the package.
//!
//! The reference is diffusers' own `Krea2Pipeline` run on the same weights, so agreeing with it
//! here is agreeing with the model rather than with a second opinion of our own. None of what
//! passes between these stages can be judged by looking -- twelve stacked hidden states, a
//! velocity, a latent -- and an error in any of them shows up only as a picture that is subtly
//! not the prompt, so they are compared as numbers.
//!
//! # Every input here is one the model would really be handed
//!
//! The fixtures are taken off one real run of the reference: a prompt, the eight steps the
//! distilled release is for, and the picture at the end. So the denoiser is asked about the
//! latent that the fifth step of that walk actually reads -- part noise, part picture -- and the
//! autoencoder about the latent those eight steps actually left behind. Neither is `torch.randn`,
//! and the difference is not pedantry: a tenth of the pixels of a decoded random latent land
//! outside the range a picture lives in, where **none** of this one's do.
//!
//! The reference tensors were computed with the prompt padded out to 512 tokens and the padded
//! rows then dropped. This runtime never pads. That the two agree is the whole of the argument in
//! `docs/krea2.md`, and it is checked here rather than asserted there -- over a whole trajectory
//! as well as at one step.
//!
//! # Why this is one test and not six
//!
//! The package is thirty-four gigabytes and the harness gives every test its own thread, so a
//! thread-local fixture is read once per test rather than once per binary: six tests here would
//! read two hundred gigabytes off the disk to ask six questions. So the parts are checked in one
//! pass, in the order a picture travels them, and `--nocapture` is what shows the numbers.

use std::cell::OnceCell;
use std::collections::HashMap;
use std::path::PathBuf;
use std::rc::Rc;

use waifu::flint::{ParamSource, Tensor};
use waifu::krea2::{
    Dit, DitConfig, EncoderConfig, FlowSampler, SamplerConfig, TextEncoder, VaeConfig, VaeDecoder,
};
use waifu::{read_safetensors, DType, Device, Manifest, Residency, WeightFormat};

/// How many tokens of the encoder's answer the denoiser reads, which is all of them but the
/// thirty-four the template puts in front.
const PREFIX: i32 = 34;

/// The steps the reference walked, and the shift it walked them with: `exp(1.15)`, because the
/// release was distilled at a fixed `mu` of that. `krea2_sampler.rs` is where those nine sigmas
/// are checked against the scheduler's own.
const STEPS: i32 = 8;

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
            let manifest = Manifest::open(models_dir().join("krea2-turbo.yaml")).unwrap();
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
    read_safetensors(&[models_dir().join("krea2-turbo_test.safetensors")]).unwrap()
}

/// The root mean square of the difference over the root mean square of the reference, which says
/// whether two tensors are the same answer rather than whether any one element is.
fn relative_rmse(actual: &Tensor, reference: &Tensor) -> f32 {
    let a = read(actual);
    let b = read(reference);
    assert_eq!(a.len(), b.len(), "the shapes do not match");

    let error: f64 = a
        .iter()
        .zip(&b)
        .map(|(x, y)| (*x as f64 - *y as f64).powi(2))
        .sum();
    let scale: f64 = b.iter().map(|y| (*y as f64).powi(2)).sum();
    (error / scale).sqrt() as f32
}

/// A tensor as f32 on the host, whatever it was on the device.
fn read(tensor: &Tensor) -> Vec<f32> {
    tensor
        .contiguous()
        .unwrap()
        .to_device(Device::Cpu)
        .unwrap()
        .cast(DType::Float)
        .unwrap()
        .to_vec_f32()
        .unwrap()
}

/// One of the reference's tensors, on the device and in the type this runtime computes in.
fn fixture(cases: &HashMap<String, Tensor>, name: &str) -> Tensor {
    cases[name].clone()
        .to_device(device())
        .unwrap()
        .cast(DType::Float16)
        .unwrap()
}

/// The ids the reference ran, as the `<long>(L)` the encoder takes.
fn ids(cases: &HashMap<String, Tensor>) -> Tensor {
    let ids = cases["test_case.input_ids"].clone();
    let length = ids.shape().last().copied().unwrap();
    ids.view(&[length]).unwrap().to_device(device()).unwrap()
}

/// What the package says Qwen3-VL's language half is. Read off the published configuration by the
/// exporter; written here so that the test says what it is testing.
fn encoder_config() -> EncoderConfig {
    EncoderConfig {
        num_layers: 36,
        hidden_size: 2560,
        vocab_size: 151936,
        num_heads: 32,
        num_kv_heads: 8,
        head_dim: 128,
        mlp_size: 9728,
        rope_theta: 5e6,
        norm_eps: 1e-6,
        select_layers: vec![2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35],
        weight_format: WeightFormat::Float,
    }
}

/// And what it says the denoiser is.
fn dit_config() -> DitConfig {
    DitConfig {
        num_blocks: 28,
        hidden_size: 6144,
        num_heads: 48,
        num_kv_heads: 12,
        head_dim: 128,
        mlp_size: 16384,
        patch_size: 2,
        latent_channels: 16,
        in_channels: 64,
        timestep_embed_dim: 256,
        rope_theta: 1000.0,
        rope_axes: (32, 48, 48),
        norm_eps: 1e-5,
        text_hidden_size: 2560,
        text_num_heads: 20,
        text_num_kv_heads: 20,
        text_mlp_size: 6912,
        text_layerwise_blocks: 2,
        text_refiner_blocks: 2,
        num_text_layers: 12,
        weight_format: WeightFormat::Float,
    }
}

/// What the package says the autoencoder is: the Qwen-Image one, which Anima carries too.
fn vae_config() -> VaeConfig {
    VaeConfig {
        latent_channels: 16,
        latents_mean: vec![
            -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508, 0.4134, -0.0715,
            0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921,
        ],
        latents_std: vec![
            2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743, 3.2687, 2.1526,
            2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.916,
        ],
    }
}

#[test]
#[ignore = "needs the krea2 package"]
fn every_part_matches_the_reference() {
    let cases = cases();
    let weights = weights();

    // -- the encoder, read at twelve of its thirty-six layers ------------------------------
    let ids = ids(&cases);
    let length = ids.shape_at(0).unwrap();

    let encoder =
        TextEncoder::build(encoder_config(), "krea2.text", &weights, DType::Float16).unwrap();
    let hidden = encoder.forward(&ids).unwrap();
    assert_eq!(hidden.shape(), vec![1, length, 12, 2560]);

    // The template's opening is dropped from the states and not from the ids: those thirty-four
    // tokens are what every state after them was computed against.
    let kept = hidden.slice(1, PREFIX, length).unwrap();
    let reference = cases["test_case.hidden"].clone();
    assert_eq!(
        kept.shape(),
        reference.shape(),
        "the reference kept a different number of tokens, which means its padding did not fall \
         where this runtime thinks it does"
    );

    let rmse = relative_rmse(&kept, &reference);
    println!("qwen3-vl tapped states rmse = {rmse}");

    // The reference runs in bfloat16 and this in float16 over thirty-six layers, so they will not
    // agree to the bit. They have to agree to a great deal better than the difference between one
    // prompt and another.
    assert!(rmse < 3e-2, "the tapped states drifted by {rmse}");

    // Grouped query attention is the encoder's, and a package that said otherwise would build a
    // graph whose weights do not divide.
    let broken = EncoderConfig {
        num_kv_heads: 7,
        ..encoder_config()
    };
    let message = TextEncoder::build(broken, "krea2.text", &weights, DType::Float16)
        .expect_err("32 does not group into 7")
        .to_string();
    assert!(message.contains("group"), "{message}");

    // -- the denoiser, at the fifth step of a real walk ------------------------------------
    //
    // Against the reference's own states rather than the ones just computed, so that what is
    // measured is the denoiser and not the encoder in front of it a second time. The latent is
    // the one that step actually read: at sigma 0.76 it is most of the way from noise to picture,
    // which is where being wrong about the model and being wrong about the schedule look
    // different.
    let context = fixture(&cases, "test_case.hidden");
    let latent = fixture(&cases, "test_case.latent");
    let timestep = read(&cases["test_case.timestep"])[0];

    let dit = Dit::build(dit_config(), "krea2.dit", &weights, DType::Float16).unwrap();
    let velocity = dit.forward(&latent, timestep, &context).unwrap();
    assert_eq!(velocity.shape(), vec![1, 16, 16, 16]);

    let rmse = relative_rmse(&velocity, &cases["test_case.velocity"]);
    println!("velocity rmse = {rmse} (at sigma {timestep})");

    // Twenty-eight blocks of half precision against a bfloat16 reference, on inputs that are
    // *bit-identical*: the fixtures came out of the reference in bfloat16, and every bfloat16
    // value is exactly a float16 one, so the cast above costs nothing at all -- measured, and it
    // is 0.0. So what is left is the two float types, and that is very nearly the whole of it:
    // running diffusers' own denoiser in float16 against diffusers' own denoiser in bfloat16, on
    // these same inputs, moves the velocity by 3.51e-2, where this runtime is 3.60e-2 away.
    //
    // What the bound rules out is the order of magnitude above: the rotary pairing, the norm
    // folding and the shared modulation all fail there, and none of them fails loudly.
    assert!(rmse < 5e-2, "the velocity drifted by {rmse}");

    // Twelve tapped layers is what the projector is a matrix over, so eleven is not a shorter
    // prompt -- it is weights that do not multiply.
    let shallow = context.slice(2, 0, 11).unwrap().contiguous().unwrap();
    let message = dit
        .forward(&latent, timestep, &shallow)
        .expect_err("eleven is not twelve")
        .to_string();
    assert!(message.contains("tapped"), "{message}");

    // -- the autoencoder, on the latent the walk arrives at ---------------------------------
    let decoder = VaeDecoder::build(
        vae_config(),
        "krea2.vae",
        &weights,
        device(),
        DType::Float16,
    )
    .unwrap();
    let final_latent = cases["test_case.final_latent"].clone();
    let image = decoder
        .forward(&final_latent.to_device(device()).unwrap())
        .unwrap();
    assert_eq!(image.shape(), vec![1, 3, 128, 128]);

    let rmse = relative_rmse(&image, &cases["test_case.decoded"]);
    println!("decoded rmse = {rmse}");

    // The fixture is the decoder's own answer and not `vae.decode`'s, which clamps to -1..=1.
    // This runtime's decoder does not clamp -- `to_rgb8` does, where a picture becomes bytes --
    // and two pixels in a thousand of a real decode sit outside that range, which is 1.4e-3 of
    // difference belonging to neither implementation.
    //
    // What is left is half precision: this measures 1.7e-3 where diffusers' own float16 decode is
    // 1.6e-3 from the same float32 reference. Run at full width on the *processor*, where neither
    // side has a TF32 convolution under it, the two agree to 1.9e-6 -- so this graph is the
    // reference's arithmetic rather than a near copy of it, and everything above is the width.
    //
    // It is also the same decoder Anima's package carries, renamed on the way in: if this fails
    // and Anima's passes, the renaming is what is wrong.
    assert!(rmse < 3e-3, "the picture drifted by {rmse}");

    // -- and the whole walk, from the noise the reference started from ---------------------
    //
    // Eight steps of this runtime's denoiser and this runtime's schedule against the latent the
    // reference's eight steps left behind. One step agreeing is one step; a trajectory agreeing
    // is the sampler, the schedule and the denoiser agreeing together.
    //
    // It is also the loosest bound here, and deliberately so: a walk amplifies whatever a step
    // disagrees by, and at these widths the reference does not agree with *itself*. Running
    // diffusers' own denoiser in float16 against diffusers' own denoiser in bfloat16 -- same
    // noise, same schedule, same prompt, one dtype changed -- moves the final latent by 6.7e-2
    // and the picture it decodes to by 1.2e-1. This runtime is 9.9e-2 from the bfloat16 one,
    // which is that same effect and not a second one. So what this catches is a schedule walked
    // backwards or a step that does not step: those are half the latent away, not a tenth.
    let sampler = FlowSampler::new(
        &SamplerConfig {
            shift: 1.15f32.exp(),
            multiplier: 1.0,
        },
        STEPS,
    )
    .unwrap();

    let mut walked = fixture(&cases, "test_case.noise");
    for index in 0..sampler.steps() {
        let velocity = dit
            .forward(&walked, sampler.timestep(index).unwrap(), &context)
            .unwrap();
        walked = sampler
            .step(index, &walked, &velocity)
            .unwrap()
            .cast(DType::Float16)
            .unwrap();
    }

    let rmse = relative_rmse(&walked, &final_latent);
    println!("final latent rmse = {rmse} (over {STEPS} steps)");
    assert!(rmse < 1.5e-1, "the walk drifted by {rmse}");
}
