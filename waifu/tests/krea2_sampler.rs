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


//! The schedule Krea 2 is sampled on, against the one diffusers' scheduler actually walked.
//!
//! Nine numbers and no weights, so this is the one test of this model that costs nothing to run.
//! It is also the one that catches the mistake that is easiest to make and hardest to see: the
//! two references write the same curve two ways --
//!
//! ```text
//! krea:  sigma = exp(mu) / (exp(mu) + (1 / t - 1))
//! anima: sigma = shift * t / (1 + (shift - 1) * t)
//! ```
//!
//! -- and a package that carried the distilled release's `mu = 1.15` where this runtime reads a
//! shift would walk a schedule that is barely bent at all, load without complaint, and draw a
//! soft, washed-out picture that looks like a bad prompt rather than a bad number.
//!
//! The sigmas here are `FlowMatchEulerDiscreteScheduler`'s own, read off it after the reference
//! ran the eight steps `tools/krea2_exporter.py` records.

use std::path::PathBuf;

use waifu::krea2::{FlowSampler, SamplerConfig};
use waifu::{DType, Device, ParamFile};

/// The number of steps the reference walked, which is what the distilled release is for.
const STEPS: i32 = 8;

fn models_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../models")
}

/// What the package says: `exp(1.15)`, because the release was distilled at a fixed `mu` of that.
fn config() -> SamplerConfig {
    SamplerConfig {
        shift: 1.15f32.exp(),
        multiplier: 1.0,
    }
}

#[test]
#[ignore = "needs the krea2 package"]
fn walks_the_schedule_the_reference_walked() {
    let cases = ParamFile::open(&[models_dir().join("krea2-turbo_test.safetensors")]).unwrap();
    let reference = cases
        .get_unchecked("test_case.sigmas")
        .unwrap()
        .to_device(Device::Cpu)
        .unwrap()
        .cast(DType::Float)
        .unwrap()
        .to_vec_f32()
        .unwrap();

    let sampler = FlowSampler::new(&config(), STEPS).unwrap();
    let sigmas = sampler.sigmas();

    // Nine for eight steps: the level each one starts at, and the zero the last one finishes at.
    assert_eq!(
        sigmas.len(),
        reference.len(),
        "the reference walked {} levels and this schedule holds {}",
        reference.len(),
        sigmas.len()
    );

    for (index, (ours, theirs)) in sigmas.iter().zip(&reference).enumerate() {
        println!("sigma {index}: {ours} against {theirs}");
        assert!(
            (ours - theirs).abs() < 1e-5,
            "sigma {index} is {ours} where the reference used {theirs}"
        );
    }

    // And what the denoiser is handed, which is the sigma itself rather than the thousandfold
    // timestep a flow model usually carries: the embedding inside the model multiplies by the
    // thousand. Passing it 750 instead of 0.75 loads fine and produces noise.
    for index in 0..STEPS as usize {
        assert_eq!(sampler.timestep(index).unwrap(), sampler.sigma(index).unwrap());
    }
}
