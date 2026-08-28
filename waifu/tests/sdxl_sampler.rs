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

//! The Euler schedule against the one `EulerDiscreteScheduler` builds for the same model.
//!
//! The sampler has no weights, so this compares numbers rather than tensors: the fifty noise
//! levels of a fifty step run, the timesteps that name them, and what one step does to a latent.
//! All of it in float32 on both sides, which is why the tolerances here are so much tighter than
//! anywhere else in these tests.

use std::path::PathBuf;

use waifu::flint::Tensor;
use waifu::{DType, Device, EulerSampler, SamplerConfig, VarBuilder, ZipFile};

/// What the reference package was written for.
const STEPS: i32 = 50;

fn cases() -> VarBuilder {
    let package = ZipFile::open(
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../models/sdxl-base_test.llmpkg"),
    )
    .unwrap();
    VarBuilder::from_reader(
        &mut package.open_entry("test_case.bin").unwrap(),
        Device::Cpu,
        DType::Float,
    )
    .unwrap()
}

fn reference(cases: &VarBuilder, name: &str) -> Vec<f32> {
    cases
        .get_unchecked(&format!("test_case.{name}"))
        .unwrap()
        .to_vec_f32()
        .unwrap()
}

fn sampler() -> EulerSampler {
    EulerSampler::new(&SamplerConfig::default(), STEPS).unwrap()
}

fn max_diff(actual: &[f32], expected: &[f32]) -> f32 {
    assert_eq!(actual.len(), expected.len(), "the lengths do not match");
    actual
        .iter()
        .zip(expected)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max)
}

#[test]
#[ignore = "needs the sdxl package"]
fn the_timesteps_match_the_reference() {
    let cases = cases();
    let sampler = sampler();

    // These are whole numbers on both sides, so anything but exact agreement is a real
    // disagreement about which noise level each step sits at.
    assert_eq!(sampler.timesteps(), &reference(&cases, "timesteps")[..]);
}

#[test]
#[ignore = "needs the sdxl package"]
fn the_noise_levels_match_the_reference() {
    let cases = cases();
    let sampler = sampler();
    let expected = reference(&cases, "sigmas");

    // The noisiest is around 14.6 and the quietest around 0.03, so this is tight across four
    // orders of magnitude. The schedule is a cumulative product over a thousand terms and gets
    // that close only if every one of them is right.
    let difference = max_diff(sampler.sigmas(), &expected);
    println!("largest sigma difference = {difference}");
    assert!(
        difference < 1e-4,
        "the noise levels drifted by {difference}"
    );
}

#[test]
#[ignore = "needs the sdxl package"]
fn the_starting_noise_level_matches() {
    let cases = cases();
    let expected = reference(&cases, "init_noise_sigma")[0];

    let difference = (sampler().init_noise_sigma() - expected).abs();
    assert!(
        difference < 1e-4,
        "the starting noise level drifted by {difference}"
    );
}

#[test]
#[ignore = "needs the sdxl package"]
fn scaling_the_input_matches_the_reference() {
    let cases = cases();
    let latent = cases.get_unchecked("test_case.latent").unwrap();

    let scaled = sampler().scale_model_input(&latent, 0).unwrap();
    let difference = max_diff(&scaled.to_vec_f32().unwrap(), &reference(&cases, "scaled"));
    assert!(
        difference < 1e-5,
        "the scaled latent drifted by {difference}"
    );
}

#[test]
#[ignore = "needs the sdxl package"]
fn one_step_matches_the_reference() {
    let cases = cases();
    let latent = cases.get_unchecked("test_case.latent").unwrap();
    let noise = cases.get_unchecked("test_case.noise").unwrap();

    let stepped = sampler().step(&noise, &latent, 0).unwrap();
    let difference = max_diff(
        &stepped.to_vec_f32().unwrap(),
        &reference(&cases, "stepped"),
    );
    println!("largest step difference = {difference}");
    assert!(difference < 1e-3, "one step drifted by {difference}");
}

#[test]
#[ignore = "needs the sdxl package"]
fn the_whole_walk_ends_where_it_should() {
    // Taken end to end with the model always reporting the sample itself as noise, the walk
    // multiplies by (1 + next - sigma) at every step, which lands on 1 + 0 - init exactly. It
    // says the steps really are consecutive: a schedule that skipped or repeated one would still
    // pass every comparison above, which only looks at the first.
    let sampler = sampler();
    let mut sample = Tensor::from_f32(&[1], &[1.0]).unwrap();

    for index in 0..sampler.len() {
        sample = sampler.step(&sample.clone(), &sample, index).unwrap();
    }

    let expected: f32 = sampler
        .sigmas()
        .windows(2)
        .fold(1.0, |value, pair| value * (1.0 + pair[1] - pair[0]));
    let difference = (sample.to_vec_f32().unwrap()[0] - expected).abs();
    assert!(difference < 1e-4, "the walk drifted by {difference}");
}
