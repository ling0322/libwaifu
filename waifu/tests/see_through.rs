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
//!
//! The bar is upstream's own. Each model is run again on those inputs in float32 and in bfloat16,
//! which is what upstream runs in, and the runtime passes where it is no further from float32 than
//! bfloat16 is: `<name>.f32` is the answer, `<name>.bf16_error` the distance it has to beat.
//! The package is eleven gigabytes, so it is read once, by the one test that asks the models
//! anything, and that test asks all of its questions in turn.

use std::collections::HashMap;
use std::ops::ControlFlow;
use std::path::PathBuf;

use waifu::flint::{functional as F, Tensor};
use waifu::see_through::{
    image as picture, DdimVelocity, DpmSolverSde, Image, Rgba, SeeThrough, SeeThroughOptions,
    UnetCondition,
};
use waifu::{read_safetensors, DType, Device, Manifest};

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

/// That `actual` is no further from upstream's float32 answer `name` than upstream's bfloat16 is.
fn within_bfloat16(cases: &HashMap<String, Tensor>, name: &str, actual: &[f32]) -> f32 {
    let error = relative_rmse(actual, &host(&cases[&format!("{name}.f32")]));
    let bound = host(&cases[&format!("{name}.bf16_error")])[0];
    eprintln!("{name}: {error:.3e} from float32; bfloat16 is {bound:.3e}");
    assert!(
        error <= bound,
        "{name} is {error:.3e} from float32, further than upstream's bfloat16 at {bound:.3e}"
    );
    error
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
    let latent = model.layers_encoder().forward(&rgb_input(page)).unwrap();
    within_bfloat16(&cases, "layers.page_latent", &host(&latent));

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
    within_bfloat16(&cases, "layers.unet", &host(&eps));

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
        within_bfloat16(&cases, &format!("layers.step{step}"), &host(&out));
    }

    // The first three layers the model finished, decoded: the autoencoder, then the alpha decoder
    // reading what it decoded and the latent it came from.
    let finals = &cases["layers.final"];
    let scale = model.config().layers_vae.scaling_factor;
    let (mut decoded, mut alphas) = (Vec::new(), Vec::new());
    for frame in 0..3 {
        let latent = on_cuda(&finals.slice(0, frame, frame + 1).unwrap(), DType::Float);
        let pixel = model.layers_decoder().forward(&latent).unwrap();
        decoded.extend(host(&pixel));
        let unit: Vec<f32> = host(&pixel).iter().map(|v| v * 0.5 + 0.5).collect();
        let unit = Tensor::from_f32(&pixel.shape(), &unit).unwrap();
        let unscaled = F::mul_scalar(&latent, 1.0 / scale).unwrap();
        alphas.extend(host(
            &model.alpha_decoder().forward(&unit, &unscaled).unwrap(),
        ));
    }
    within_bfloat16(&cases, "layers.decode", &decoded);
    within_bfloat16(&cases, "layers.alpha", &alphas);
    let reference = pictures(&cases["layers.rgba"]);

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
    let mut latents = Vec::new();
    for slot in &small {
        latents.extend(host(&model.encode_slot(slot).unwrap()));
    }
    within_bfloat16(&cases, "depth.slot_latents", &latents);

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
    within_bfloat16(&cases, "depth.unet", &host(&velocity));

    // And every depth map decoded from the latents the depth model finished on.
    let finals = &cases["depth.final"];
    let mut maps = Vec::new();
    for frame in 0..finals.shape_at(0).unwrap() {
        let latent = on_cuda(&finals.slice(0, frame, frame + 1).unwrap(), DType::Float);
        maps.extend(model.decode_depth(&latent).unwrap().data);
    }
    within_bfloat16(&cases, "depth.maps", &maps);

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
