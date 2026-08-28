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

//! The sampler, which is what turns one U-Net into an image.
//!
//! A U-Net only ever answers one question -- how much noise is in this? -- and the sampler is the
//! arithmetic that walks a latent from pure noise down to none by asking it repeatedly. This is
//! the Euler one every SDXL checkpoint ships with: at each step it takes the noise the model
//! reports as a direction and moves along it as far as the gap between two noise levels.
//!
//! It has no weights. Everything here is arithmetic on the betas the model was trained with, so
//! the only thing that can go wrong is getting that arithmetic wrong, which is what its tests are
//! about.

use crate::error::{Error, Result};
use crate::flint::{functional as F, Tensor};

/// The noise schedule a model was trained on. The default is what every SDXL checkpoint carries.
#[derive(Clone, Copy, Debug)]
pub struct SamplerConfig {
    /// How many noise levels the schedule is defined over, which is also the scale the U-Net's
    /// timestep is on.
    pub num_train_timesteps: i32,
    pub beta_start: f32,
    pub beta_end: f32,
    /// Added to every timestep. A quirk of the original code that the trained weights now expect,
    /// so it stays.
    pub steps_offset: i32,
}

impl Default for SamplerConfig {
    fn default() -> SamplerConfig {
        SamplerConfig {
            num_train_timesteps: 1000,
            beta_start: 0.00085,
            beta_end: 0.012,
            steps_offset: 1,
        }
    }
}

/// A schedule of noise levels and the timesteps that name them.
///
/// Built once for a given number of steps. `sigmas` holds one more entry than `timesteps`: the
/// last is zero, the noise level the image is supposed to end at.
#[derive(Debug)]
pub struct EulerSampler {
    timesteps: Vec<f32>,
    sigmas: Vec<f32>,
}

impl EulerSampler {
    pub fn new(config: &SamplerConfig, num_steps: i32) -> Result<EulerSampler> {
        if num_steps <= 0 || num_steps > config.num_train_timesteps {
            return Err(Error::model(format!(
                "{num_steps} steps is not between one and the {} the model was trained on",
                config.num_train_timesteps
            )));
        }

        // The betas are linear in their own square root, which is what "scaled linear" means and
        // what stable diffusion has used since the first one.
        let count = config.num_train_timesteps as usize;
        let start = (config.beta_start as f64).sqrt();
        let end = (config.beta_end as f64).sqrt();

        // How much of the original image survives up to each timestep, and beside it the noise
        // level that corresponds to: sigma is the ratio of what was added to what is left.
        let mut alphas_cumprod = 1.0f64;
        let mut all_sigmas = Vec::with_capacity(count);
        for index in 0..count {
            let root = start + (end - start) * index as f64 / (count - 1) as f64;
            alphas_cumprod *= 1.0 - root * root;
            all_sigmas.push(((1.0 - alphas_cumprod) / alphas_cumprod).sqrt());
        }

        // Evenly spaced over the training range, noisiest first, which is the "leading" spacing.
        let stride = config.num_train_timesteps / num_steps;
        let mut timesteps = Vec::with_capacity(num_steps as usize);
        let mut sigmas = Vec::with_capacity(num_steps as usize + 1);
        for index in (0..num_steps).rev() {
            let timestep = (index * stride + config.steps_offset) as f64;
            timesteps.push(timestep as f32);
            sigmas.push(interpolate(&all_sigmas, timestep) as f32);
        }

        // Where the walk ends, and the reason there is one more noise level than step.
        sigmas.push(0.0);

        Ok(EulerSampler { timesteps, sigmas })
    }

    /// The timesteps to run, noisiest first. One per step.
    pub fn timesteps(&self) -> &[f32] {
        &self.timesteps
    }

    /// The noise levels, one per timestep and a zero at the end.
    pub fn sigmas(&self) -> &[f32] {
        &self.sigmas
    }

    pub fn len(&self) -> usize {
        self.timesteps.len()
    }

    pub fn is_empty(&self) -> bool {
        self.timesteps.is_empty()
    }

    /// What to multiply the initial random noise by, so that it sits at the noise level the first
    /// step expects to see.
    pub fn init_noise_sigma(&self) -> f32 {
        let largest = self.sigmas.iter().copied().fold(0.0f32, f32::max);
        (largest * largest + 1.0).sqrt()
    }

    /// The latent as the U-Net wants to see it at step `index`.
    ///
    /// The model was trained on a latent of roughly unit variance, and a sample carrying sigma
    /// worth of noise has more than that, so it is divided back down before the model reads it.
    /// The sample itself keeps its own scale, which is what [`EulerSampler::step`] works on.
    pub fn scale_model_input(&self, sample: &Tensor, index: usize) -> Result<Tensor> {
        let sigma = self.sigma(index)?;
        Ok(F::div_scalar(sample, (sigma * sigma + 1.0).sqrt())?)
    }

    /// One step: the sample at noise level `sigmas[index]` moved to `sigmas[index + 1]`.
    ///
    /// `noise` is what the U-Net said is in `sample`. It is a direction rather than something to
    /// subtract outright, which is what makes this Euler's method: the step taken along it is the
    /// gap between the two noise levels, and the gap is what a smaller number of steps makes
    /// larger and cruder.
    pub fn step(&self, noise: &Tensor, sample: &Tensor, index: usize) -> Result<Tensor> {
        let sigma = self.sigma(index)?;
        let next = self.sigmas[index + 1];

        Ok(F::add(sample, &F::mul_scalar(noise, next - sigma)?)?)
    }

    fn sigma(&self, index: usize) -> Result<f32> {
        if index >= self.timesteps.len() {
            return Err(Error::model(format!(
                "step {index} is past the {} this schedule holds",
                self.timesteps.len()
            )));
        }

        Ok(self.sigmas[index])
    }
}

/// `values` read at a fractional index, the two nearest entries mixed in proportion.
///
/// Every timestep the leading spacing produces is a whole number, so this is a lookup in
/// practice. It is written out anyway because the schedule is not the only thing that decides
/// that -- a checkpoint with a step count that does not divide the training range evenly would
/// land between two entries, and rounding there would quietly shift the noise level.
fn interpolate(values: &[f64], at: f64) -> f64 {
    let at = at.clamp(0.0, (values.len() - 1) as f64);
    let below = at.floor() as usize;
    let above = at.ceil() as usize;

    values[below] + (values[above] - values[below]) * (at - below as f64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flint::{DType, Device};

    fn sampler(num_steps: i32) -> EulerSampler {
        EulerSampler::new(&SamplerConfig::default(), num_steps).unwrap()
    }

    #[test]
    fn walks_the_training_range_noisiest_first() {
        let sampler = sampler(50);
        assert_eq!(sampler.len(), 50);

        // 1000 training steps over 50, plus the offset every checkpoint expects.
        assert_eq!(sampler.timesteps()[0], 981.0);
        assert_eq!(sampler.timesteps()[49], 1.0);

        for pair in sampler.timesteps().windows(2) {
            assert!(pair[0] > pair[1], "the timesteps are not descending");
        }
    }

    #[test]
    fn the_noise_falls_to_nothing() {
        let sampler = sampler(20);

        // One more noise level than step, the last of which is zero: no noise left at all.
        assert_eq!(sampler.sigmas().len(), 21);
        assert_eq!(sampler.sigmas()[20], 0.0);

        for pair in sampler.sigmas().windows(2) {
            assert!(pair[0] > pair[1], "the noise levels are not descending");
        }
    }

    #[test]
    fn a_step_count_that_does_not_divide_evenly_still_works() {
        // 1000 over 7 is 142 with a remainder, so the walk stops short of the noisiest timestep
        // rather than stretching to reach it. This is what the reference does too.
        let sampler = sampler(7);
        assert_eq!(sampler.len(), 7);
        assert_eq!(sampler.timesteps()[0], 6.0 * 142.0 + 1.0);
    }

    #[test]
    fn refuses_a_step_count_it_cannot_walk() {
        let config = SamplerConfig::default();
        assert!(EulerSampler::new(&config, 0).is_err());
        assert!(EulerSampler::new(&config, -1).is_err());
        assert!(EulerSampler::new(&config, 1001).is_err());
    }

    #[test]
    fn interpolates_between_two_entries() {
        let values = [0.0, 1.0, 3.0];
        assert_eq!(interpolate(&values, 0.0), 0.0);
        assert_eq!(interpolate(&values, 1.0), 1.0);
        assert_eq!(interpolate(&values, 1.5), 2.0);
        assert_eq!(interpolate(&values, 2.0), 3.0);

        // Past either end it holds rather than running off the array.
        assert_eq!(interpolate(&values, -3.0), 0.0);
        assert_eq!(interpolate(&values, 9.0), 3.0);
    }

    #[test]
    fn a_step_past_the_end_is_refused() {
        let sampler = sampler(4);
        let sample = Tensor::zeros(&[1, 4, 8, 8], DType::Float, Device::Cpu).unwrap();
        assert!(sampler.step(&sample, &sample, 4).is_err());
        assert!(sampler.scale_model_input(&sample, 4).is_err());
    }
}
