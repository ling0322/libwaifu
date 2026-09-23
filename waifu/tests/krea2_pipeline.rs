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


//! The whole of Krea 2, from a string to pixels.
//!
//! `krea2.rs` checks each part against the reference with the ids and tensors handed to it. This
//! checks the joins: that a prompt reaches the denoiser as the same stack of twelve layers the
//! reference computed -- padding and all, with this runtime carrying none of it -- and that the
//! schedule and the decoder around it produce a picture rather than noise.
//!
//! # Why this is one test
//!
//! The package is thirty-four gigabytes and the harness gives every test its own thread, so a
//! fixture is read once per test rather than once per binary. What follows is therefore a single
//! pass that asks several questions, with `--nocapture` to show the numbers.

use std::collections::HashMap;
use std::ops::ControlFlow;
use std::path::PathBuf;

use waifu::flint::Tensor;
use waifu::{
    read_safetensors, DType, Device, GenerationOptions, GenerationProgress, Krea2, Manifest,
    Residency,
};

/// The prompt the reference outputs were computed for.
const PROMPT: &str = "a red fox sitting in fresh snow, golden hour, photorealistic";

fn models_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../models")
}

fn device() -> Device {
    Device::Cuda
}

/// The model: twenty-four gigabytes of denoiser and eight of encoder, read once.
fn model() -> Krea2 {
    let manifest = Manifest::open(models_dir().join("krea2-turbo.yaml")).unwrap();
    Krea2::from_manifest(device(), Residency::Device, &manifest).unwrap()
}

fn cases() -> HashMap<String, Tensor> {
    read_safetensors(&[models_dir().join("krea2-turbo_test.safetensors")]).unwrap()
}

/// The root mean square of the difference over the root mean square of the reference.
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

/// How much neighbouring pixels differ, over how much the picture varies overall.
///
/// Noise is near 1.4 -- every pixel independent of the one beside it -- and a drawn picture is
/// well under 1, because a drawn picture has areas in it.
fn roughness(image: &Tensor) -> f32 {
    let shape = image.shape();
    let (height, width) = (shape[2] as usize, shape[3] as usize);
    let pixels = read(image);

    let mut step = 0.0f64;
    let mut count = 0usize;
    for channel in 0..shape[1] as usize {
        let plane = &pixels[channel * height * width..(channel + 1) * height * width];
        for row in 0..height {
            for column in 1..width {
                let difference = plane[row * width + column] - plane[row * width + column - 1];
                step += (difference as f64).powi(2);
                count += 1;
            }
        }
    }

    let mean: f64 = pixels.iter().map(|x| *x as f64).sum::<f64>() / pixels.len() as f64;
    let variance: f64 = pixels
        .iter()
        .map(|x| (*x as f64 - mean).powi(2))
        .sum::<f64>()
        / pixels.len() as f64;

    ((step / count as f64) / variance).sqrt() as f32
}

#[test]
#[ignore = "needs the krea2 package"]
fn draws_a_picture_the_reference_would_recognize() {
    let cases = cases();
    let model = model();

    // -- the prompt ------------------------------------------------------------------------
    let reference = cases["test_case.hidden"].clone();
    let context = model.encode_prompt(PROMPT).unwrap();

    // The shape first, because this is where the padding argument either holds or does not: the
    // reference ran 512 tokens and kept the ones its mask called real, and this ran exactly those
    // and nothing else. A runtime that tokenized the template differently would land here.
    assert_eq!(
        context.shape(),
        reference.shape(),
        "the prompt came to a different number of tokens than the reference kept"
    );

    let rmse = relative_rmse(&context, &reference);
    println!("prompt rmse = {rmse}");
    assert!(rmse < 3e-2, "the prompt drifted by {rmse}");

    // -- and the picture at the end of it ---------------------------------------------------
    let options = GenerationOptions {
        width: 128,
        height: 128,
        num_steps: 4,
        guidance_scale: 1.0,
        seed: Some(7),
        ..Default::default()
    };

    let image = model.generate(PROMPT, &options).unwrap();
    assert_eq!(image.shape(), vec![1, 3, 128, 128]);

    let rough = roughness(&image);
    println!("roughness = {rough}");
    assert!(rough < 1.0, "this is noise, not a picture: roughness {rough}");

    // The same seed twice. A run that differs from itself has something in it that is not the
    // model -- an uninitialized buffer, a table built from whatever was on the stack.
    let again = model.generate(PROMPT, &options).unwrap();
    assert_eq!(
        read(&image),
        read(&again),
        "the same seed drew two different pictures"
    );

    // -- what it refuses, and how it stops --------------------------------------------------
    //
    // A size the latent cannot be cut into patches at: eight for the autoencoder and two for the
    // patch, so a multiple of sixteen and nothing else.
    let awkward = GenerationOptions {
        width: 136,
        height: 128,
        ..options.clone()
    };
    let message = model
        .generate(PROMPT, &awkward)
        .expect_err("136 is not a multiple of 16")
        .to_string();
    assert!(message.contains("multiple of 16"), "{message}");

    // And a run that is told to stop, which stops between steps and hands back no picture.
    let mut seen = Vec::new();
    let stopped = model
        .generate_reporting(PROMPT, &options, &mut |progress| {
            seen.push(progress);
            match progress {
                GenerationProgress::Step { done, .. } if done == 2 => ControlFlow::Break(()),
                _ => ControlFlow::Continue(()),
            }
        })
        .unwrap();

    assert!(stopped.is_none(), "a run that was stopped drew a picture");
    assert_eq!(seen.first(), Some(&GenerationProgress::Encoding));
    assert_eq!(
        seen.last(),
        Some(&GenerationProgress::Step { done: 2, total: 4 })
    );
}
