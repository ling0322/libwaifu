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

//! The whole of SDXL: a package read from disk, a prompt in, an image out.
//!
//! Each piece has its own test against its own reference. This is the one that says they are put
//! together the way `StableDiffusionXLPipeline` puts them together -- the same latent denoised by
//! the same schedule with the same guidance lands in the same place.

use std::path::PathBuf;

use waifu::flint::Tensor;
use waifu::{DType, Device, GenerationOptions, Sdxl, VarBuilder, ZipFile};

/// What the reference denoising run in the exporter was written for.
const PROMPT: &str = "a photo of an astronaut riding a horse on mars";
const STEPS: i32 = 4;
const GUIDANCE: f32 = 5.0;
const SIZE: i32 = 256;

fn models_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../models")
}

fn model() -> Sdxl {
    let package = ZipFile::open(models_dir().join("sdxl-base.llmpkg")).unwrap();
    Sdxl::from_package(Device::Cuda, &package).unwrap()
}

fn cases() -> VarBuilder {
    let package = ZipFile::open(models_dir().join("sdxl-base_test.llmpkg")).unwrap();
    VarBuilder::from_reader(
        &mut package.open_entry("test_case.bin").unwrap(),
        Device::Cpu,
        DType::Float,
    )
    .unwrap()
}

fn options() -> GenerationOptions {
    GenerationOptions {
        width: SIZE,
        height: SIZE,
        num_steps: STEPS,
        guidance_scale: GUIDANCE,
        negative_prompt: String::new(),
        seed: None,
    }
}

fn to_cuda(tensor: &Tensor) -> Tensor {
    tensor
        .to_device(Device::Cuda)
        .unwrap()
        .cast(DType::Float16)
        .unwrap()
}

fn relative_rmse(actual: &Tensor, reference: &Tensor) -> f32 {
    let a = actual
        .to_device(Device::Cpu)
        .unwrap()
        .cast(DType::Float)
        .unwrap()
        .to_vec_f32()
        .unwrap();
    let b = reference
        .to_device(Device::Cpu)
        .unwrap()
        .cast(DType::Float)
        .unwrap()
        .to_vec_f32()
        .unwrap();
    assert_eq!(a.len(), b.len(), "the shapes do not match");

    let error: f64 = a
        .iter()
        .zip(&b)
        .map(|(x, y)| (*x as f64 - *y as f64).powi(2))
        .sum();
    let scale: f64 = b.iter().map(|y| (*y as f64).powi(2)).sum();
    (error / scale).sqrt() as f32
}

#[test]
#[ignore = "needs the sdxl package"]
fn reads_its_shape_out_of_the_package() {
    let model = model();
    let config = model.config();

    assert_eq!(config.text.hidden_size, 768);
    assert_eq!(config.text2.hidden_size, 1280);
    assert_eq!(config.unet.block_out_channels, vec![320, 640, 1280]);
    assert_eq!(config.vae.block_out_channels, vec![128, 256, 512, 512]);
    assert_eq!(config.sampler.num_train_timesteps, 1000);
    assert_eq!(model.device(), Device::Cuda);
}

#[test]
#[ignore = "needs the sdxl package"]
fn encodes_a_prompt_the_way_the_reference_does() {
    // The pieces of this are checked in sdxl_text_encoder.rs, which starts from the reference's
    // own token ids. This one starts from the text, so it also says the tokenizer, the markers
    // and the padding are what the encoders were given.
    let cases = cases();
    let model = model();

    let embedding = model.encode_prompt(PROMPT).unwrap();
    assert_eq!(embedding.context.shape(), vec![1, 77, 2048]);
    assert_eq!(embedding.pooled.shape(), vec![1, 1280]);

    let hidden = cases.get_unchecked("test_case.hidden").unwrap();
    let hidden2 = cases.get_unchecked("test_case.hidden2").unwrap();

    // Sliced where it lives: a half precision tensor on the host has no copy kernel behind it,
    // and making one contiguous is a copy.
    let context = &embedding.context;
    let first = context.slice(2, 0, 768).unwrap().contiguous().unwrap();
    let second = context.slice(2, 768, 2048).unwrap().contiguous().unwrap();

    assert!(relative_rmse(&first, &hidden) < 2e-2);
    assert!(relative_rmse(&second, &hidden2) < 2e-2);
    assert!(
        relative_rmse(
            &embedding.pooled,
            &cases.get_unchecked("test_case.pooled2").unwrap()
        ) < 2e-2
    );
}

#[test]
#[ignore = "needs the sdxl package"]
fn denoising_matches_the_reference_pipeline() {
    let cases = cases();
    let model = model();

    // The same unit noise the reference started from. Everything after it -- the schedule, the
    // guidance, the size the model is told about -- has to agree for four steps running.
    let latent = to_cuda(&cases.get_unchecked("test_case.latent").unwrap());

    let prompt = model.encode_prompt(PROMPT).unwrap();
    let negative = model.encode_prompt("").unwrap();

    let denoised = model
        .denoise(&latent, &prompt, &negative, &options())
        .unwrap();
    assert_eq!(denoised.shape(), latent.shape());

    let rmse = relative_rmse(
        &denoised,
        &cases.get_unchecked("test_case.denoised").unwrap(),
    );
    println!("denoise rmse = {rmse}");
    assert!(rmse < 3e-2, "four steps drifted by {rmse}");
}

#[test]
#[ignore = "needs the sdxl package"]
fn generates_a_latent_from_a_prompt() {
    let model = model();
    let mut options = options();
    options.seed = Some(7);

    let latent = model.generate_latent(PROMPT, &options).unwrap();
    assert_eq!(latent.shape(), vec![1, 4, SIZE / 8, SIZE / 8]);

    let values = latent
        .to_device(Device::Cpu)
        .unwrap()
        .cast(DType::Float)
        .unwrap()
        .to_vec_f32()
        .unwrap();
    assert!(values.iter().all(|x| x.is_finite()));

    // Four steps is not enough for a picture, but a latent that had gone wrong somewhere would be
    // flat or enormous rather than the handful of units a real one covers.
    let largest = values.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
    assert!(
        (0.5..50.0).contains(&largest),
        "the latent reaches {largest}"
    );
}

#[test]
#[ignore = "needs the sdxl package"]
fn decoding_a_denoised_latent_reports_the_overflow() {
    // SDXL's autoencoder is marked force_upcast and really does need float32: a denoised latent
    // overflows one convolution of its last up block in half precision. What is checked here is
    // that this is reported rather than handed back as a black picture. When the decoder can run
    // in float32 this is the test that should change.
    let cases = cases();
    let model = model();

    // The reference's own denoised latent, so this says something about the model rather than
    // about the runtime.
    let denoised = to_cuda(&cases.get_unchecked("test_case.denoised").unwrap());
    let error = model.decode(&denoised).unwrap_err();
    assert!(
        error.to_string().contains("float32"),
        "the overflow was reported as {error}"
    );

    // Pure noise still decodes, which is what sdxl_vae.rs compares against the reference.
    let latent = to_cuda(&cases.get_unchecked("test_case.latent").unwrap());
    assert!(model.decode(&latent).is_ok());
}

#[test]
#[ignore = "needs the sdxl package"]
fn the_seed_is_the_only_thing_that_makes_a_run_differ() {
    let model = model();
    let mut options = options();
    // Two steps rather than four: this is about which noise a run starts from, and running the
    // U-Net twice as many times says nothing more about that.
    options.num_steps = 2;

    options.seed = Some(11);
    let first = model.generate_latent(PROMPT, &options).unwrap();
    let again = model.generate_latent(PROMPT, &options).unwrap();
    assert_eq!(
        relative_rmse(&first, &again),
        0.0,
        "one seed gave two latents"
    );

    options.seed = Some(12);
    let other = model.generate_latent(PROMPT, &options).unwrap();
    assert!(
        relative_rmse(&first, &other) > 0.1,
        "two seeds gave one latent"
    );
}

#[test]
#[ignore = "needs the sdxl package"]
fn guidance_of_one_is_the_unprompted_answer() {
    // At a guidance of one the negative prompt is not consulted at all, and the second U-Net pass
    // is skipped for it. Anything else would be running the model twice for nothing.
    let cases = cases();
    let model = model();
    let latent = to_cuda(&cases.get_unchecked("test_case.latent").unwrap());

    let prompt = model.encode_prompt(PROMPT).unwrap();
    let negative = model.encode_prompt("a completely different thing").unwrap();
    let unrelated = model.encode_prompt("something else again").unwrap();

    let mut options = options();
    options.guidance_scale = 1.0;
    options.num_steps = 2;

    let first = model
        .denoise(&latent, &prompt, &negative, &options)
        .unwrap();
    let second = model
        .denoise(&latent, &prompt, &unrelated, &options)
        .unwrap();
    assert_eq!(relative_rmse(&first, &second), 0.0);
}

#[test]
#[ignore = "needs the sdxl package"]
fn refuses_a_size_it_cannot_work_in() {
    let model = model();

    // Eight for the autoencoder and four more for the U-Net's two halvings.
    for (width, height) in [(256, 100), (250, 256), (0, 256), (-256, 256)] {
        let mut options = options();
        options.width = width;
        options.height = height;
        assert!(
            model.generate_latent(PROMPT, &options).is_err(),
            "{width} by {height} was allowed"
        );
    }
}
