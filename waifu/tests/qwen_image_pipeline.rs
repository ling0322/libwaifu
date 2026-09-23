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

//! The whole of Qwen-Image 2.1, from a string to pixels, through the calls a caller makes.
//!
//! `qwen_image.rs` checks each part against the reference. This checks that a prompt, a seed and
//! a size come out as an opaque picture rather than noise, at a size the reference's fixtures do
//! not cover, and that a run can be refused and stopped. The picture is also written out, as
//! `qwen_image_pipeline.ppm` under cargo's scratch directory, to be looked at.

use std::ops::ControlFlow;
use std::path::PathBuf;

use waifu::flint::Tensor;
use waifu::qwen_image::QwenImage;
use waifu::{DType, Device, GenerationOptions, GenerationProgress, Manifest, Residency};

const PROMPT: &str = "a red fox sitting in fresh snow, golden hour, photorealistic";

fn models_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../models")
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

/// How much neighbouring pixels differ, against how much pixels differ at all: near one for
/// noise, a good deal less for a picture.
fn roughness(pixels: &[f32], channels: usize, height: usize, width: usize) -> f32 {
    let mut step = 0.0f64;
    let mut count = 0usize;
    for channel in 0..channels {
        let plane = &pixels[channel * height * width..(channel + 1) * height * width];
        for row in 0..height {
            for column in 1..width {
                let difference = plane[row * width + column] - plane[row * width + column - 1];
                step += (difference as f64).powi(2);
                count += 1;
            }
        }
    }

    let colour = &pixels[..channels * height * width];
    let mean: f64 = colour.iter().map(|x| *x as f64).sum::<f64>() / colour.len() as f64;
    let variance: f64 = colour
        .iter()
        .map(|x| (*x as f64 - mean).powi(2))
        .sum::<f64>()
        / colour.len() as f64;
    ((step / count as f64) / variance).sqrt() as f32
}

/// The colour channels as a binary PPM, which needs no encoder.
fn write_ppm(pixels: &[f32], height: usize, width: usize) -> PathBuf {
    let plane = height * width;
    let mut bytes = format!("P6\n{width} {height}\n255\n").into_bytes();
    for pixel in 0..plane {
        for channel in 0..3 {
            let value = (pixels[channel * plane + pixel] / 2.0 + 0.5).clamp(0.0, 1.0);
            bytes.push((value * 255.0).round() as u8);
        }
    }

    let where_to = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("qwen_image_pipeline.ppm");
    std::fs::write(&where_to, bytes).unwrap();
    where_to
}

#[test]
#[ignore = "needs the qwen-image-2.1 package"]
fn draws_an_opaque_picture_and_can_be_stopped() {
    let manifest = Manifest::open(models_dir().join("qwen-image-2.1.yaml")).unwrap();
    let model = QwenImage::from_manifest(Device::Cuda, Residency::LowVram, &manifest).unwrap();

    let options = GenerationOptions {
        width: 512,
        height: 512,
        num_steps: 20,
        seed: Some(42),
        ..QwenImage::DEFAULTS.options()
    };

    let latent = model.generate_latent(PROMPT, &options).unwrap();
    assert_eq!(latent.shape(), vec![1, 64, 32, 32]);
    let image = model.decode(&latent).unwrap();
    assert_eq!(image.shape(), vec![1, 4, 512, 512]);

    let pixels = read(&image);
    assert!(
        pixels.iter().all(|x| x.is_finite()),
        "the picture holds a NaN or an infinity"
    );

    // A photograph of a fox is drawn opaque: alpha near one everywhere.
    let plane = 512 * 512;
    let alpha = pixels[3 * plane..]
        .iter()
        .map(|a| (a / 2.0 + 0.5) as f64)
        .sum::<f64>()
        / plane as f64;
    println!("mean alpha = {alpha}");
    assert!(alpha > 0.95, "an opaque prompt came out {alpha} opaque");

    let rough = roughness(&pixels, 3, 512, 512);
    println!("roughness = {rough}");
    assert!(
        rough < 0.5,
        "this is noise, not a picture: roughness {rough}"
    );

    println!("wrote {}", write_ppm(&pixels, 512, 512).display());

    // A size the model cannot draw at is refused before anything is read.
    let message = model
        .generate(
            PROMPT,
            &GenerationOptions {
                width: 500,
                ..options.clone()
            },
        )
        .expect_err("500 is not a multiple of 32")
        .to_string();
    assert!(message.contains("multiple of 32"), "{message}");

    // And a run told to stop at the first thing it reports stops there.
    let mut seen = Vec::new();
    let stopped = model
        .generate_reporting(PROMPT, &options, &mut |progress| {
            seen.push(progress);
            ControlFlow::Break(())
        })
        .unwrap();
    assert!(stopped.is_none(), "a run that was stopped drew a picture");
    assert_eq!(seen, vec![GenerationProgress::Encoding]);
}
