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

//! Qwen-Image 2.1's ControlNet-Union against VideoX-Fun, which trained it, by way of the tensors
//! `tools/qwen_image_exporter.py -controlnet .. -test_output` writes.
//!
//! The fixtures are one real controlled run, 256 by 256 for ten steps, steered by VideoX-Fun's
//! own pose picture: the encoder is asked about that picture, and the denoiser about the latent
//! the run's sixth step actually read.
//!
//! One test: the model and the control package together are forty-five gigabytes streamed
//! through the card, and every test would stream its own.

use std::collections::HashMap;
use std::path::PathBuf;

use waifu::flint::Tensor;
use waifu::qwen_image::QwenImage;
use waifu::{read_safetensors, DType, Device, Manifest, Residency};

fn models_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../models")
}

fn device() -> Device {
    Device::Cuda
}

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

fn fixture(cases: &HashMap<String, Tensor>, name: &str) -> Tensor {
    cases[name]
        .clone()
        .to_device(device())
        .unwrap()
        .cast(DType::Float16)
        .unwrap()
}

#[test]
#[ignore = "needs the qwen-image-2.1 and qwen-image-2.1-controlnet packages"]
fn the_control_chain_matches_videox_fun() {
    let manifest = Manifest::open(models_dir().join("qwen-image-2.1.yaml")).unwrap();
    let control = Manifest::open(models_dir().join("qwen-image-2.1-controlnet.yaml")).unwrap();
    let model =
        QwenImage::from_manifests(device(), Residency::LowVram, &manifest, Some(&control)).unwrap();
    assert!(model.has_control());
    let cases =
        read_safetensors(&[models_dir().join("qwen-image-2.1-controlnet_test.safetensors")])
            .unwrap();

    // -- the control picture, through the encoder -------------------------------------------
    let conditioning = model
        .control_conditioning(&cases["test_case.control_image"])
        .unwrap();
    let reference = &cases["test_case.control_context"];
    assert_eq!(conditioning.shape(), reference.shape());
    let rmse = relative_rmse(&conditioning, reference);
    println!("control conditioning rmse = {rmse}");
    assert!(rmse < 2e-2, "the encoder is {rmse} from the reference's");

    // -- the denoiser, steered and not, at the latent the reference's sixth step read --------
    let latent = fixture(&cases, "test_case.latent");
    let timestep = cases["test_case.timestep"].to_vec_f32().unwrap()[0];
    let hidden = fixture(&cases, "test_case.hidden");
    let context = fixture(&cases, "test_case.control_context");
    let dit = model.dit();

    for (name, scale) in [("velocity", 1.0), ("velocity_half", 0.5)] {
        let velocity = dit
            .forward_controlled(&latent, timestep, &hidden, &context, scale)
            .unwrap();
        let rmse = relative_rmse(&velocity, &cases[&format!("test_case.{name}")]);
        println!("controlled velocity at scale {scale}: rmse = {rmse}");
        assert!(rmse < 5e-2, "at scale {scale} the velocity is {rmse} off");
    }

    // VideoX-Fun carries its own copy of the denoiser; this is what says it is the one the
    // runtime was held to against diffusers.
    let plain = dit.forward(&latent, timestep, &hidden).unwrap();
    let rmse = relative_rmse(&plain, &cases["test_case.velocity_plain"]);
    println!("plain velocity rmse = {rmse}");
    assert!(rmse < 5e-2, "the uncontrolled velocity is {rmse} off");

    // And a scale of zero is no control at all.
    let unsteered = dit
        .forward_controlled(&latent, timestep, &hidden, &context, 0.0)
        .unwrap();
    let rmse = relative_rmse(&unsteered, &plain);
    println!("scale zero against no control: rmse = {rmse}");
    assert!(rmse < 1e-3, "a scale of zero still steers, by {rmse}");

    // -- the whole walk, from the noise the reference started from ----------------------------
    let options = waifu::GenerationOptions {
        width: 256,
        height: 256,
        num_steps: 10,
        ..QwenImage::DEFAULTS.options()
    };
    let sampler = model.sampler(256, 256, 10).unwrap();
    let sigmas = cases["test_case.sigmas"].to_vec_f32().unwrap();
    for (index, reference) in sigmas.iter().take(10).enumerate() {
        let sigma = sampler.sigma(index).unwrap();
        assert!(
            (sigma - reference).abs() < 1e-4,
            "step {index} is at sigma {sigma}, the reference's at {reference}"
        );
    }

    let walked = model
        .denoise_reporting(
            &fixture(&cases, "test_case.noise"),
            &sampler,
            &hidden,
            None,
            Some((&context, 1.0)),
            &options,
            &mut |_| std::ops::ControlFlow::Continue(()),
        )
        .unwrap()
        .unwrap();
    let rmse = relative_rmse(&walked, &cases["test_case.final_latent"]);
    println!("controlled final latent rmse = {rmse} (over 10 steps)");
    assert!(rmse < 1e-1, "the controlled walk drifted by {rmse}");
}
