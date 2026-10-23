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
//! about the latent the ten steps left behind. None of it is `torch.randn`. A second run edits a
//! real photograph (`-test_picture`), and the same questions are asked of a prompt naming it.
//!
//! One test, not four: the package is thirty gigabytes, streamed through the card rather than held
//! on it, and the harness gives every test its own thread and so its own copy.

use std::collections::HashMap;
use std::path::PathBuf;

use waifu::flint::Tensor;
use waifu::qwen_image::QwenImage;
use waifu::{read_safetensors, DType, Device, Manifest};

fn models_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../models")
}

fn device() -> Device {
    Device::Cuda
}

fn relative_rmse(actual: &Tensor, reference: &Tensor) -> f32 {
    relative(&read(actual), &read(reference))
}

fn relative(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "the shapes do not match");
    let error: f64 = a
        .iter()
        .zip(b)
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

/// How far a package may be from the reference with a picture in the prompt: the encoder against
/// float32, the velocity, and the whole walk. The words-only bounds of each package, since the
/// picture changes what is read and not the precision it is read at.
struct Bounds {
    hidden: f32,
    velocity: f32,
    walk: f32,
}

/// A prompt that names a picture, against the `picture_*` cases `-test_picture` adds.
///
/// The vision tower is never quantized, so its bounds are the same for every package: float16
/// against float32, where the reference's own bfloat16 run is 1.9e-1 from float32 on the merged
/// rows and float16 torch 1.9e-2. DeepStack's rows are taken earlier and drift less.
fn check_pictures(model: &QwenImage, cases: &HashMap<String, Tensor>, bounds: Bounds) {
    assert!(
        cases.contains_key("test_case.picture_ids"),
        "the fixture has no picture cases: export it again with -test_picture"
    );
    let prompt = "Turn this photograph into a watercolor painting";
    let picture = cases["test_case.picture"].to_device(device()).unwrap();

    // -- the template, with the picture's slots written out -----------------------------------
    // The tower reads the picture laid over white, as the reference's processor is handed it.
    let shape = picture.shape();
    let rgba = read(&picture);
    let plane = (shape[2] * shape[3]) as usize;
    let rgb = (0..3 * plane)
        .map(|at| {
            let alpha = rgba[3 * plane + at % plane] / 2.0 + 0.5;
            (rgba[at] / 2.0 + 0.5) * alpha + (1.0 - alpha)
        })
        .map(|unit| unit * 2.0 - 1.0)
        .collect::<Vec<_>>();
    let rgb = Tensor::from_f32(&[1, 3, shape[2], shape[3]], &rgb)
        .unwrap()
        .to_device(device())
        .unwrap();
    let seen = model
        .vision()
        .expect("the package carries the vision tower")
        .forward(&rgb)
        .unwrap();
    let ids = model
        .ids_with_pictures(prompt, &[(seen.rows, seen.columns)])
        .unwrap();
    assert_eq!(
        ids,
        cases["test_case.picture_ids"].to_vec_i64().unwrap(),
        "the template with a picture in it tokenizes differently from the reference's"
    );

    // -- the vision tower ----------------------------------------------------------------------
    let rmse = relative_rmse(&seen.embeds, &cases["test_case.picture_embeds"]);
    println!("vision rows rmse = {rmse} against float32");
    // Measured at 3.8e-2 on Metal, the 608 x 416 cat of docs/qwen_image.md.
    assert!(rmse < 8e-2, "the vision tower's rows drifted by {rmse}");
    for (layer, rows) in seen.deepstack.iter().enumerate() {
        let rmse = relative_rmse(rows, &cases[&format!("test_case.picture_deepstack{layer}")]);
        println!("deepstack {layer} rmse = {rmse} against float32");
        assert!(rmse < 4e-2, "DeepStack's rows {layer} drifted by {rmse}");
    }

    // -- the encoder, reading the picture ----------------------------------------------------
    let prompted = model
        .encode_prompt_with_pictures(prompt, &[picture.clone()])
        .unwrap();
    let pictures = prompted.pictures.as_ref().unwrap();
    let slots = cases["test_case.picture_slots"].to_vec_i64().unwrap();
    let exact = read(&cases["test_case.picture_hidden_fp32"]);
    let width = exact.len() / slots.len();
    let words = |states: &[f32]| -> Vec<f32> {
        slots
            .iter()
            .enumerate()
            .filter(|(_, &slot)| slot == 0)
            .flat_map(|(row, _)| states[row * width..(row + 1) * width].to_vec())
            .collect()
    };
    let exact_words = words(&exact);
    let reference_words = words(&read(&cases["test_case.picture_hidden"]));
    let ours = read(&prompted.context);
    let rmse = relative(&ours, &exact_words);
    let theirs = relative(&reference_words, &exact_words);
    println!("encoder rmse = {rmse} against float32 (the bfloat16 reference is {theirs})");
    assert!(
        rmse < bounds.hidden,
        "the encoder's states drifted by {rmse}"
    );

    // -- the autoencoder's half, the picture's latent ----------------------------------------
    let latent = pictures
        .tokens
        .transpose(1, 2)
        .unwrap()
        .contiguous()
        .unwrap();
    let rmse = relative(&read(&latent), &read(&cases["test_case.picture_latent"]));
    println!("picture latent rmse = {rmse} against float32");
    assert!(rmse < 2e-2, "the picture's latent drifted by {rmse}");

    // -- the denoiser, on the reference's own states and step ---------------------------------
    let hidden = read(&cases["test_case.picture_hidden"]);
    let reference_words = words(&hidden);
    let context = Tensor::from_f32(
        &[1, (reference_words.len() / width) as i32, width as i32],
        &reference_words,
    )
    .unwrap()
    .to_device(device())
    .unwrap()
    .cast(DType::Float16)
    .unwrap();
    let step = fixture(cases, "test_case.picture_step");
    let timestep = read(&cases["test_case.picture_timestep"])[0];
    let velocity = model
        .dit()
        .forward_with_pictures(&step, timestep, &context, pictures)
        .unwrap();
    let rmse = relative_rmse(&velocity, &cases["test_case.picture_velocity"]);
    println!("velocity with a picture rmse = {rmse} (at timestep {timestep})");
    assert!(rmse < bounds.velocity, "the velocity drifted by {rmse}");

    // -- and the whole walk ---------------------------------------------------------------------
    let side = (cases["test_case.picture_noise"].shape()[3] * 16) as i32;
    let sampler = model.sampler(side, side, 10).unwrap();
    let sigmas = read(&cases["test_case.picture_sigmas"]);
    assert_eq!(sampler.sigmas().len(), sigmas.len());
    let options = waifu::GenerationOptions {
        width: side,
        height: side,
        ..QwenImage::DEFAULTS.options()
    };
    let walked = model
        .denoise_with_pictures_reporting(
            &fixture(cases, "test_case.picture_noise"),
            &sampler,
            &prompted,
            None,
            &options,
            &mut |_| std::ops::ControlFlow::Continue(()),
        )
        .unwrap()
        .unwrap();
    let rmse = relative_rmse(&walked, &cases["test_case.picture_final_latent"]);
    println!("final latent with a picture rmse = {rmse}");
    assert!(
        rmse < bounds.walk,
        "the walk with a picture drifted by {rmse}"
    );
}

#[test]
#[ignore = "needs the qwen-image-2.1 package"]
fn every_part_matches_the_reference() {
    let manifest = Manifest::open(models_dir().join("qwen-image-2.1.yaml")).unwrap();
    let model = QwenImage::from_manifest(device(), 0, &manifest).unwrap();
    let cases = read_safetensors(&[models_dir().join("qwen-image-2.1_test.safetensors")]).unwrap();

    // -- the template and the encoder --------------------------------------------------------
    let prompt = "a red fox sitting in fresh snow, golden hour, photorealistic";
    let ids = model.ids(prompt).unwrap();
    let reference_ids = &cases["test_case.input_ids"];
    assert_eq!(
        ids.to_device(Device::Cpu).unwrap().to_vec_i64().unwrap(),
        reference_ids.to_vec_i64().unwrap(),
        "the template tokenizes differently from the reference's"
    );

    let hidden = model.encode_ids(&ids).unwrap();
    let reference = &cases["test_case.hidden"];
    assert_eq!(
        hidden.shape(),
        reference.shape(),
        "a different number of states was kept"
    );

    let exact = &cases["test_case.hidden_fp32"];
    let rmse = relative_rmse(&hidden, exact);
    let theirs = relative_rmse(reference, exact);
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
    let timestep = read(&cases["test_case.timestep"])[0];

    let velocity = model.dit().forward(&latent, timestep, &context).unwrap();
    assert_eq!(velocity.shape(), vec![1, 64, 16, 16]);
    let rmse = relative_rmse(&velocity, &cases["test_case.velocity"]);
    println!("velocity rmse = {rmse} (at timestep {timestep})");
    // Measured at 7.9e-3. The block-causal split, the t = 0 modulation of the prompt and the
    // centred positions each fail an order of magnitude above this and none of them loudly.
    assert!(rmse < 3e-2, "the velocity drifted by {rmse}");

    // -- the schedule ------------------------------------------------------------------------
    let sampler = model.sampler(256, 256, 10).unwrap();
    let sigmas = read(&cases["test_case.sigmas"]);
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
    let rmse = relative_rmse(&image, &cases["test_case.decoded"]);
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

    // -- a prompt that names a picture -------------------------------------------------------
    check_pictures(
        &model,
        &cases,
        Bounds {
            hidden: 3e-2,
            velocity: 3e-2,
            walk: 5e-2,
        },
    );
}

/// Qwen-Image-2.1-Turbo, fp8: eight fixed steps rather than a bent schedule. The tolerances are
/// the fp8 package's (see "How close it is" in docs/qwen_image.md), not the float16 one's.
#[test]
#[ignore = "needs the qwen-image-2.1-turbo-fp8 package"]
fn turbo_fp8_matches_the_reference() {
    let manifest = Manifest::open(models_dir().join("qwen-image-2.1-turbo-fp8.yaml")).unwrap();
    let model = QwenImage::from_manifest(device(), 0, &manifest).unwrap();
    let cases =
        read_safetensors(&[models_dir().join("qwen-image-2.1-turbo_test.safetensors")]).unwrap();

    let prompt = "a red fox sitting in fresh snow, golden hour, photorealistic";
    let ids = model.ids(prompt).unwrap();
    assert_eq!(
        ids.to_device(Device::Cpu).unwrap().to_vec_i64().unwrap(),
        cases["test_case.input_ids"].to_vec_i64().unwrap(),
        "the template tokenizes differently from the reference's"
    );

    let hidden = model.encode_ids(&ids).unwrap();
    assert_eq!(hidden.shape(), cases["test_case.hidden"].shape());
    let rmse = relative_rmse(&hidden, &cases["test_case.hidden_fp32"]);
    println!("encoder rmse = {rmse} against float32");
    assert!(rmse < 2e-1, "the encoder's states drifted by {rmse}");

    // The schedule is the package's own eight levels and a zero, whatever steps are asked for.
    let sampler = model.sampler(256, 256, 40).unwrap();
    let sigmas = read(&cases["test_case.sigmas"]);
    assert_eq!(sampler.sigmas().len(), 9);
    assert_eq!(sampler.sigmas().len(), sigmas.len());
    for (ours, theirs) in sampler.sigmas().iter().zip(&sigmas) {
        assert!(
            (ours - theirs).abs() < 1e-5,
            "{:?} against {sigmas:?}",
            sampler.sigmas()
        );
    }

    let context = fixture(&cases, "test_case.hidden");
    let latent = fixture(&cases, "test_case.latent");
    let timestep = read(&cases["test_case.timestep"])[0];
    let velocity = model.dit().forward(&latent, timestep, &context).unwrap();
    let rmse = relative_rmse(&velocity, &cases["test_case.velocity"]);
    println!("velocity rmse = {rmse} (at timestep {timestep})");
    assert!(rmse < 6e-2, "the velocity drifted by {rmse}");

    let final_latent = fixture(&cases, "test_case.final_latent");
    let image = model.decode(&final_latent).unwrap();
    let rmse = relative_rmse(&image, &cases["test_case.decoded"]);
    println!("decoded rmse = {rmse}");
    assert!(rmse < 3e-3, "the picture drifted by {rmse}");

    let options = waifu::GenerationOptions {
        width: 256,
        height: 256,
        num_steps: 8,
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
            None,
            &options,
            &mut |_| std::ops::ControlFlow::Continue(()),
        )
        .unwrap()
        .unwrap();
    let rmse = relative_rmse(&walked, &final_latent);
    println!("final latent rmse = {rmse} (over 8 steps)");
    assert!(rmse < 2e-1, "the walk drifted by {rmse}");

    check_pictures(
        &model,
        &cases,
        Bounds {
            hidden: 2e-1,
            velocity: 6e-2,
            walk: 2e-1,
        },
    );
}
