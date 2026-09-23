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

//! The whole of Anima, from a string to pixels.
//!
//! `anima.rs` checks each part against the reference with the ids and tensors handed to it. This
//! checks the joins: that a prompt reaches the denoiser as the same 512 by 1024 context the
//! reference computed, and that the schedule and the decoder around it produce a picture rather
//! than noise.

use std::collections::HashMap;
use std::ops::ControlFlow;
use std::path::PathBuf;

use waifu::flint::Tensor;
use waifu::{
    read_safetensors, Anima, DType, Device, GenerationOptions, GenerationProgress, Manifest,
    Residency,
};

/// The prompt the reference outputs were computed for.
const PROMPT: &str = "masterpiece, best quality, 1girl, solo, long hair, brown eyes, school \
                      uniform, smile";

fn models_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../models")
}

fn device() -> Device {
    Device::Cuda
}

fn model() -> Anima {
    let manifest = Manifest::open(models_dir().join("anima-turbo-v11.yaml")).unwrap();
    Anima::from_manifest(device(), Residency::Device, &manifest).unwrap()
}

fn cases() -> HashMap<String, Tensor> {
    read_safetensors(&[models_dir().join("anima-turbo-v11_test.safetensors")]).unwrap()
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

/// A tensor as f32 on the host, whatever it was on the device. Contiguous first: a slice of a
/// larger tensor cannot be copied off the card as it stands.
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

/// How much neighbouring pixels differ, over how much the picture varies overall.
///
/// Noise is near 1.4 -- every pixel independent of the one beside it -- and a drawn picture is
/// well under 1, because a drawn picture has areas in it. The same measure `anima.rs` ends with.
fn roughness(image: &Tensor) -> f32 {
    let shape = image.shape();
    let (height, width) = (shape[2] as usize, shape[3] as usize);
    let values = read(image);

    let mut steps = 0.0f64;
    let mut count = 0usize;
    let mut total = 0.0f64;
    let mut total_squared = 0.0f64;
    for row in 0..height {
        for column in 0..width {
            let value = values[row * width + column] as f64;
            total += value;
            total_squared += value * value;
            if column + 1 < width {
                let next = values[row * width + column + 1] as f64;
                steps += (next - value).abs();
                count += 1;
            }
        }
    }

    let mean = total / (height * width) as f64;
    let variance = total_squared / (height * width) as f64 - mean * mean;
    ((steps / count as f64) / variance.sqrt()) as f32
}

/// The join this whole change exists to make: a prompt, as a string, arriving at the denoiser as
/// what the reference computed for it.
///
/// Everything is in here -- both tokenizers, the encoder, the adapter, the end marker on one set
/// of ids and not the other, and the padding out to 512. Any one of them wrong moves this number.
#[test]
#[ignore = "needs the anima package"]
fn a_prompt_reaches_the_denoiser_as_the_reference_computed_it() {
    let context = model().encode_prompt(PROMPT).unwrap();
    assert_eq!(context.shape(), vec![1, 512, 1024]);

    let reference = cases()["test_case.context_padded"].clone();
    let rmse = relative_rmse(&context, &reference);
    println!("context rmse = {rmse}");

    // The adapter alone agrees to 0.0024 in `anima.rs`, and the encoder in front of it to 0.0029.
    // This is those two composed, so it lands a little above either.
    assert!(rmse < 0.01, "the context is not the reference's: {rmse}");
}

/// Past the prompt there is nothing but zeros, which is what the denoiser's cross attention reads
/// for a prompt shorter than 512 -- and which is the easiest part of the padding to get subtly
/// wrong, because a context of the right shape full of the wrong thing still draws something.
#[test]
#[ignore = "needs the anima package"]
fn the_context_is_zero_past_the_prompt() {
    let context = model().encode_prompt(PROMPT).unwrap();

    // 21 T5 ids: the prompt and its end marker.
    let padding = context.slice(1, 21, 512).unwrap();
    let largest = read(&padding)
        .into_iter()
        .fold(0.0f32, |worst, value| worst.max(value.abs()));

    assert_eq!(largest, 0.0, "the padding is not zeros");
}

#[test]
#[ignore = "needs the anima package"]
fn draws_a_picture_from_a_prompt() {
    let model = model();
    let options = GenerationOptions {
        width: 512,
        height: 512,
        seed: Some(0x5eed),
        ..Anima::DEFAULTS.options()
    };

    let image = model.generate(PROMPT, &options).unwrap();
    assert_eq!(image.shape(), vec![1, 3, 512, 512]);

    let rough = roughness(&image);
    println!("roughness = {rough}");
    assert!(rough < 0.8, "this is noise rather than a picture: {rough}");
}

/// A run says where it has got to, in order, and stops where it is told to.
#[test]
#[ignore = "needs the anima package"]
fn reports_every_step_and_can_be_stopped() {
    let model = model();
    let options = GenerationOptions {
        width: 256,
        height: 256,
        num_steps: 4,
        seed: Some(1),
        ..Anima::DEFAULTS.options()
    };

    let mut seen = Vec::new();
    let image = model
        .generate_reporting(PROMPT, &options, &mut |progress| {
            seen.push(progress);
            ControlFlow::Continue(())
        })
        .unwrap();
    assert!(image.is_some());
    assert_eq!(
        seen,
        vec![
            GenerationProgress::Encoding,
            GenerationProgress::Step { done: 1, total: 4 },
            GenerationProgress::Step { done: 2, total: 4 },
            GenerationProgress::Step { done: 3, total: 4 },
            GenerationProgress::Step { done: 4, total: 4 },
            GenerationProgress::Decoding,
        ]
    );

    // Stopping after the second step: no image, and the two steps after it never run.
    let mut steps = 0;
    let stopped = model
        .generate_reporting(PROMPT, &options, &mut |progress| {
            if let GenerationProgress::Step { .. } = progress {
                steps += 1;
                if steps == 2 {
                    return ControlFlow::Break(());
                }
            }
            ControlFlow::Continue(())
        })
        .unwrap();
    assert!(stopped.is_none(), "a stopped run hands back no image");
    assert_eq!(steps, 2);
}

/// The same seed draws the same picture, which is what makes a seed worth having.
#[test]
#[ignore = "needs the anima package"]
fn the_same_seed_draws_the_same_picture() {
    let model = model();
    let options = GenerationOptions {
        width: 256,
        height: 256,
        num_steps: 2,
        seed: Some(11),
        ..Anima::DEFAULTS.options()
    };

    let first = model.generate_latent(PROMPT, &options).unwrap();
    let second = model.generate_latent(PROMPT, &options).unwrap();
    assert_eq!(relative_rmse(&first, &second), 0.0);
}

#[test]
#[ignore = "needs the anima package"]
fn refuses_a_size_that_does_not_divide_into_patches() {
    let options = GenerationOptions {
        // A multiple of 8 for the VAE, but not of the 2 the denoiser patches by on top of it.
        width: 520,
        height: 512,
        ..Anima::DEFAULTS.options()
    };

    let message = model()
        .generate(PROMPT, &options)
        .expect_err("520 is not a multiple of 16")
        .to_string();
    assert!(message.contains("16"), "{message}");
}
