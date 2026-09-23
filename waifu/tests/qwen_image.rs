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

//! Qwen-Image 2.1's parts against what diffusers' `QwenImage21Pipeline` produces for the same
//! inputs, by way of the tensors `tools/qwen_image_exporter.py -test_output` writes.
//!
//! The fixtures are taken off one real run of the reference, 256 by 256 for ten steps: the
//! denoiser is asked about the latent that run's sixth step actually read, and the autoencoder
//! about the latent the ten steps left behind. None of it is `torch.randn`.
//!
//! One test, not four: the package is thirty gigabytes, streamed through the card rather than held
//! on it, and the harness gives every test its own thread and so its own copy.

use std::path::PathBuf;

use waifu::flint::Tensor;
use waifu::qwen_image::QwenImage;
use waifu::{DType, Device, Manifest, ParamFile, Residency};

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

fn fixture(cases: &ParamFile, name: &str) -> Tensor {
    cases
        .get_unchecked(name)
        .unwrap()
        .to_device(device())
        .unwrap()
        .cast(DType::Float16)
        .unwrap()
}

#[test]
#[ignore = "needs the qwen-image-2.1 package"]
fn every_part_matches_the_reference() {
    let manifest = Manifest::open(models_dir().join("qwen-image-2.1.yaml")).unwrap();
    let model = QwenImage::from_manifest(device(), Residency::LowVram, &manifest).unwrap();
    let cases = ParamFile::open(&[models_dir().join("qwen-image-2.1_test.safetensors")]).unwrap();

    // -- the template and the encoder --------------------------------------------------------
    let prompt = "a red fox sitting in fresh snow, golden hour, photorealistic";
    let ids = model.ids(prompt).unwrap();
    let reference_ids = cases.get_unchecked("test_case.input_ids").unwrap();
    assert_eq!(
        ids.to_device(Device::Cpu).unwrap().to_vec_i64().unwrap(),
        reference_ids.to_vec_i64().unwrap(),
        "the template tokenizes differently from the reference's"
    );

    let hidden = model.encode_ids(&ids).unwrap();
    let reference = cases.get_unchecked("test_case.hidden").unwrap();
    assert_eq!(
        hidden.shape(),
        reference.shape(),
        "a different number of states was kept"
    );

    let exact = cases.get_unchecked("test_case.hidden_fp32").unwrap();
    let rmse = relative_rmse(&hidden, &exact);
    let theirs = relative_rmse(&reference, &exact);
    println!("qwen3-vl last-layer states rmse = {rmse} against float32 (the bfloat16 reference is {theirs})");
    // Against float32 and not against the bfloat16 reference, because the reference is the less
    // exact of the two: the last layer before the norm carries activations in the hundreds, and
    // bfloat16's eight bits put the reference 7.3e-2 from float32 -- most of it on the first
    // token the denoiser reads. This runtime in float16 measures 9.0e-3.
    assert!(rmse < 3e-2, "the encoder's states drifted by {rmse}");
    assert!(
        rmse < theirs,
        "the encoder is further from float32 than bfloat16 is"
    );

    // -- the denoiser, at the sixth step of the walk -----------------------------------------
    let context = fixture(&cases, "test_case.hidden");
    let latent = fixture(&cases, "test_case.latent");
    let timestep = read(&cases.get_unchecked("test_case.timestep").unwrap())[0];

    let velocity = model.dit().forward(&latent, timestep, &context).unwrap();
    assert_eq!(velocity.shape(), vec![1, 64, 16, 16]);
    let rmse = relative_rmse(
        &velocity,
        &cases.get_unchecked("test_case.velocity").unwrap(),
    );
    println!("velocity rmse = {rmse} (at timestep {timestep})");
    // Measured at 7.9e-3. The block-causal split, the t = 0 modulation of the prompt and the
    // centred positions each fail an order of magnitude above this and none of them loudly.
    assert!(rmse < 3e-2, "the velocity drifted by {rmse}");

    // -- the schedule ------------------------------------------------------------------------
    let sampler = model.sampler(256, 256, 10).unwrap();
    let sigmas = read(&cases.get_unchecked("test_case.sigmas").unwrap());
    assert_eq!(sampler.sigmas().len(), sigmas.len());
    for (ours, theirs) in sampler.sigmas().iter().zip(&sigmas) {
        assert!(
            (ours - theirs).abs() < 1e-5,
            "{:?} against {sigmas:?}",
            sampler.sigmas()
        );
    }

    // -- the autoencoder, on the latent the walk arrived at ---------------------------------
    let final_latent = fixture(&cases, "test_case.final_latent");
    let image = model.decode(&final_latent).unwrap();
    assert_eq!(image.shape(), vec![1, 4, 256, 256]);
    let rmse = relative_rmse(&image, &cases.get_unchecked("test_case.decoded").unwrap());
    println!("decoded rmse = {rmse}");
    // float16 against a float32 reference, unclamped on both sides: 7.5e-4. A shortcut that
    // read the wrong channels -- the first time slice of `DupUp3D` instead of the last -- is
    // visible at once.
    assert!(rmse < 3e-3, "the picture drifted by {rmse}");

    // -- and the whole walk, from the noise the reference started from ---------------------
    let options = waifu::GenerationOptions {
        width: 256,
        height: 256,
        num_steps: 10,
        ..QwenImage::DEFAULTS.options()
    };
    let walked = model
        .denoise_reporting(
            &fixture(&cases, "test_case.noise"),
            &sampler,
            &model
                .encode_prompt(prompt)
                .unwrap()
                .cast(DType::Float16)
                .unwrap(),
            None,
            &options,
            &mut |_| std::ops::ControlFlow::Continue(()),
        )
        .unwrap()
        .unwrap();
    let rmse = relative_rmse(&walked, &final_latent);
    println!("final latent rmse = {rmse} (over 10 steps)");
    // Ten steps of the runtime's own schedule, timestep rounding and denoiser: 1.35e-2.
    assert!(rmse < 5e-2, "the walk drifted by {rmse}");
}
