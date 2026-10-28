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

//! See-through against what upstream computes on one real illustration.
//!
//! `tools/see_through_reference.py` runs upstream's two pipelines on the illustration in its own
//! repository and records what goes in and out of each model; every input here is one of those.
//! The package is eleven gigabytes, so it is read once, by the one test that asks the models
//! anything, and that test asks all of its questions in turn.

use std::collections::HashMap;
use std::ops::ControlFlow;
use std::path::PathBuf;

use waifu::flint::{functional as F, Tensor};
use waifu::see_through::{
    image as picture, DdimVelocity, DpmSolverSde, Image, Rgba, SeeThrough, SeeThroughOptions,
};
use waifu::{read_safetensors, DType, Device, Manifest, UnetCondition};

fn models_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../models")
}

fn cases() -> HashMap<String, Tensor> {
    read_safetensors(&[models_dir().join("see-through_test.safetensors")]).unwrap()
}

fn host(tensor: &Tensor) -> Vec<f32> {
    tensor
        .to_device(Device::Cpu)
        .unwrap()
        .cast(DType::Float)
        .unwrap()
        .to_vec_f32()
        .unwrap()
}

fn relative_rmse(actual: &[f32], reference: &[f32]) -> f32 {
    assert_eq!(
        actual.len(),
        reference.len(),
        "two tensors of different sizes"
    );
    let (mut error, mut energy) = (0f64, 0f64);
    for (a, b) in actual.iter().zip(reference) {
        error += ((a - b) as f64).powi(2);
        energy += (*b as f64).powi(2);
    }
    (error / energy.max(1e-30)).sqrt() as f32
}

fn on_cuda(tensor: &Tensor, dtype: DType) -> Tensor {
    tensor.to_device(Device::Cuda).unwrap().cast(dtype).unwrap()
}

/// `(N, H, W, 4)` eight-bit pictures, as the reference stores them.
fn pictures(tensor: &Tensor) -> Vec<Rgba> {
    let shape = tensor.shape();
    let (n, h, w) = (shape[0] as usize, shape[1] as usize, shape[2] as usize);
    let bytes = tensor.to_device(Device::Cpu).unwrap().to_vec_u8().unwrap();
    (0..n)
        .map(|i| {
            let size = h * w * 4;
            Image::from_data(w, h, 4, bytes[i * size..(i + 1) * size].to_vec())
        })
        .collect()
}

/// How far apart two pictures are, as the mean difference of a channel out of 255.
fn mean_difference(a: &Rgba, b: &Rgba, channel: usize) -> f32 {
    let n = a.width * a.height;
    let sum: f32 = (0..n)
        .map(|p| (a.data[p * 4 + channel] as f32 - b.data[p * 4 + channel] as f32).abs())
        .sum();
    sum / n as f32
}

fn rgb_input(page: &Rgba) -> Tensor {
    let side = page.width;
    let mut rgb = Vec::with_capacity(side * side * 3);
    for c in 0..3 {
        for p in 0..side * side {
            rgb.push(page.data[p * 4 + c] as f32 / 127.5 - 1.0);
        }
    }
    Tensor::from_f32(&[1, 3, side as i32, side as i32], &rgb).unwrap()
}

#[test]
#[ignore = "needs the see-through package and its _test.safetensors"]
fn see_through_matches_the_reference() {
    let manifest = Manifest::open(models_dir().join("see-through.yaml")).unwrap();
    let model = SeeThrough::from_manifest(Device::Cuda, u64::MAX, &manifest).unwrap();
    let cases = cases();
    let dtype = DType::Float16;

    // -- the layers model ------------------------------------------------------------------
    let page = &pictures(&cases["layers.page"].unsqueeze(0).unwrap())[0];

    // The picture's latent, which the U-Net reads beside every layer.
    // Against the autoencoder in float32, as it runs here: upstream's bfloat16 is six percent
    // from that on its own.
    let latent = model.layers_encoder().forward(&rgb_input(page)).unwrap();
    let error = relative_rmse(&host(&latent), &host(&cases["layers.page_latent_f32"]));
    eprintln!("picture latent: {error}");
    assert!(error < 1e-3, "the picture's latent is {error} away");

    // One step of the U-Net partway along the real trajectory.
    let (context, pooled) = model.condition(0);
    let input = on_cuda(&cases["layers.unet_input"], dtype);
    let side = page.width as f32;
    let eps = model
        .layers_unet()
        .forward(
            &input,
            host(&cases["layers.unet_timestep"])[0],
            &UnetCondition {
                context,
                pooled: Some(pooled),
                time_ids: [side, side, 0.0, 0.0, side, side],
            },
        )
        .unwrap();
    let error = relative_rmse(&host(&eps), &host(&cases["layers.unet_output"]));
    eprintln!("layers U-Net: {error}");
    assert!(error < 3e-2, "the layers U-Net is {error} away");

    // The sampler's first two steps, which are its two orders, on what upstream handed it.
    let schedule = model.config().layers_schedule.clone();
    let mut sampler = DpmSolverSde::new(&schedule, 30).unwrap();
    let timesteps: Vec<f32> = sampler.timesteps().iter().map(|&t| t as f32).collect();
    assert_eq!(timesteps, host(&cases["layers.timesteps"]));
    for step in 0..2 {
        let at = |name: &str| cases[&format!("layers.step{step}.{name}")].clone();
        let out = sampler
            .step(step, &at("eps"), &at("sample"), &at("noise"))
            .unwrap();
        // diffusers steps in float32 and hands the result back in the model's bfloat16, whose
        // rounding alone is a tenth of a percent or two.
        let error = relative_rmse(&host(&out), &host(&at("out")));
        eprintln!("sampler step {step}: {error}");
        assert!(error < 5e-3, "sampler step {step} is {error} away");
    }

    // Every layer the model finished, decoded. The alpha decoder runs in float32 and upstream's
    // in bfloat16, so a few levels out of 255 is agreement.
    let page_alpha: Vec<f32> = (0..page.width * page.height)
        .map(|p| page.data[p * 4 + 3] as f32 / 255.0)
        .collect();
    let finals = &cases["layers.final"];
    let reference = pictures(&cases["layers.rgba"]);
    for (frame, expected) in reference.iter().enumerate() {
        let frame = frame as i32;
        let latent = on_cuda(&finals.slice(0, frame, frame + 1).unwrap(), DType::Float);
        let layer = model.decode_layer(&latent, &page_alpha).unwrap();
        let alpha = mean_difference(&layer, expected, 3);
        eprintln!(
            "layer {frame}: alpha {alpha}, red {}",
            mean_difference(&layer, expected, 0)
        );
        assert!(
            alpha < 3.0,
            "layer {frame}'s alpha is {alpha} out of 255 away"
        );
        let colour = mean_difference(&layer, expected, 0);
        assert!(
            colour < 4.0,
            "layer {frame}'s red is {colour} out of 255 away"
        );
    }

    // -- the depth model -------------------------------------------------------------------
    let body = &model.config().body_tags;
    let layers: HashMap<String, Rgba> = body.iter().cloned().zip(reference).collect();
    let slots = model.depth_slots(page, &layers);
    for (index, (slot, expected)) in slots
        .iter()
        .zip(pictures(&cases["depth.slots"]))
        .enumerate()
    {
        for channel in 0..4 {
            let difference = mean_difference(slot, &expected, channel);
            assert!(
                difference < 0.5,
                "slot {index} channel {channel}: {difference}"
            );
        }
    }

    // Each slot at the depth model's size, padded the way upstream pads it.
    let small = pictures(&cases["depth.slots_small"]);
    let side = small[0].width;
    for (index, slot) in slots.iter().enumerate() {
        let resized = picture::smart_resize_u8(slot, side, side);
        let difference = mean_difference(&resized, &small[index], 3);
        assert!(difference < 0.5, "slot {index} scaled is {difference} away");
    }
    let face = model
        .config()
        .depth_tags
        .iter()
        .position(|t| t == "face")
        .unwrap();
    for (name, index) in [("face", face), ("full", small.len() - 1)] {
        let mut unit = small[index].to_f32();
        unit.data.iter_mut().for_each(|v| *v /= 255.0);
        let padded = picture::pad_rgb(&unit);
        // The reference is planar, (3, H, W); this is interleaved.
        let reference = host(&cases[&format!("depth.pad_rgb.{name}")]);
        let plane = side * side;
        let mut planar = vec![0f32; plane * 3];
        for p in 0..plane {
            for c in 0..3 {
                planar[c * plane + p] = padded.data[p * 3 + c];
            }
        }
        let error = relative_rmse(&planar, &reference);
        assert!(error < 1e-3, "pad_rgb of the {name} is {error} away");
    }

    // The latents the depth model is conditioned on, from the reference's own small slots.
    let condition = host(&cases["depth.slot_latents_f32"]);
    let per_frame = condition.len() / small.len();
    for (index, slot) in small.iter().enumerate() {
        let latent = host(&model.encode_slot(slot).unwrap());
        let reference = &condition[index * per_frame..(index + 1) * per_frame];
        let error = relative_rmse(&latent, reference);
        assert!(error < 2e-3, "slot {index}'s latent is {error} away");
    }

    // One step of the depth U-Net on the real trajectory, and the schedule it was on.
    let sampler = DdimVelocity::new(&model.config().depth_schedule, 4).unwrap();
    let timesteps: Vec<f32> = sampler.timesteps().iter().map(|&t| t as f32).collect();
    assert_eq!(timesteps, host(&cases["depth.timesteps"]));
    let input = on_cuda(&cases["depth.unet_input"], dtype);
    let frames = input.shape_at(0).unwrap();
    let mut context = model.depth_context().clone();
    for _ in 1..frames {
        context = F::cat(&context, model.depth_context(), 0).unwrap();
    }
    let velocity = model
        .depth_unet()
        .forward(
            &input,
            host(&cases["depth.unet_timestep"])[0],
            &UnetCondition {
                context: &context,
                pooled: None,
                time_ids: [0.0; 6],
            },
        )
        .unwrap();
    let error = relative_rmse(&host(&velocity), &host(&cases["depth.unet_output"]));
    eprintln!("depth U-Net: {error}");
    assert!(error < 3e-2, "the depth U-Net is {error} away");

    // And every depth map decoded from the latents the depth model finished on.
    let finals = &cases["depth.final"];
    let maps = host(&cases["depth.maps"]);
    let plane = maps.len() / finals.shape_at(0).unwrap() as usize;
    for frame in 0..finals.shape_at(0).unwrap() {
        let latent = on_cuda(&finals.slice(0, frame, frame + 1).unwrap(), DType::Float);
        let map = model.decode_depth(&latent).unwrap();
        let at = frame as usize * plane;
        let error = relative_rmse(&map.data, &maps[at..at + plane]);
        assert!(error < 2e-2, "depth map {frame} is {error} away");
    }

    // -- the whole of it, at the reference's size ------------------------------------------
    let options = SeeThroughOptions {
        seed: 7,
        steps: Some(8),
        resolution: Some(page.width),
        depth_resolution: Some(side),
        split_pairs: true,
        ..SeeThroughOptions::default()
    };
    let mut reports = 0;
    let decomposition = model
        .decompose(page, &options, &mut |_| {
            reports += 1;
            ControlFlow::Continue(())
        })
        .unwrap()
        .expect("nothing asked it to stop");
    assert_eq!(reports, 8 + 8 + 4 + 1);
    assert!(
        decomposition.parts.len() >= 5,
        "{} parts",
        decomposition.parts.len()
    );
    // Farthest first.
    for pair in decomposition.parts.windows(2) {
        assert!(pair[0].depth_median >= pair[1].depth_median);
    }
    let psd = decomposition.to_psd();
    assert_eq!(&psd[..4], b"8BPS");
    std::fs::write(std::env::temp_dir().join("see-through-test.psd"), &psd).unwrap();
}
