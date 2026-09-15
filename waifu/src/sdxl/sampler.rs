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
use crate::flint::{functional as F, DType, Tensor};

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

/// A schedule of noise levels and the timesteps that name them, and where in it to begin.
///
/// Built once for a given number of steps. `sigmas` holds one more entry than `timesteps`: the
/// last is zero, the noise level the image is supposed to end at.
///
/// A run from noise walks the whole thing. A run from a picture walks the tail of a longer one --
/// see [`EulerSampler::from_image`] -- which is what `start` is for: every index below it is a
/// noise level this run never sees.
#[derive(Debug)]
pub struct EulerSampler {
    timesteps: Vec<f32>,
    sigmas: Vec<f32>,
    start: usize,
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

        Ok(EulerSampler {
            timesteps,
            sigmas,
            start: 0,
        })
    }

    /// The schedule for a run of `num_steps` that starts from a picture rather than from noise.
    ///
    /// `num_steps` is what it says on every other screen: the steps that will actually run. It is
    /// `strength` that decides how noisy the picture is when they start, by deciding how long the
    /// schedule they are the tail of is -- `num_steps / strength` of them, so that the run picks
    /// up that fraction of the way down.
    ///
    /// Which is the other way round from the way diffusers takes the same two numbers: there,
    /// `num_inference_steps` is the whole schedule and `strength` cuts it down, so asking for
    /// thirty steps at 0.8 runs twenty-four. The arithmetic is the same and the knob is the same;
    /// what differs is which of the two numbers is the one you can count on. ComfyUI counts on
    /// this one, and so does this, because a step count that quietly means something else as soon
    /// as another box is touched is a step count nobody can read.
    ///
    /// The latent to start from is the encoded picture plus unit noise times `sigmas()[start()]`.
    pub fn from_image(
        config: &SamplerConfig,
        num_steps: i32,
        strength: f32,
    ) -> Result<EulerSampler> {
        if !(0.0..=1.0).contains(&strength) {
            return Err(Error::model(format!(
                "a strength of {strength} is not between zero and one"
            )));
        }

        // No steps at all, which is a picture through the autoencoder and back. There is no
        // fraction of a schedule to take the tail of, so the schedule is the one that was asked
        // for and the walk begins past the end of it, where the noise level is zero.
        if strength == 0.0 {
            let mut sampler = EulerSampler::new(config, num_steps)?;
            sampler.start = sampler.timesteps.len();

            return Ok(sampler);
        }

        // The whole schedule this run is the tail of.
        //
        // The strength is read to six places first, which is more than any knob offers and less
        // than the error being undone: it is a number someone set to 0.8, and the nearest float
        // to 0.8 divides 20 into 24.9999996, which floors to a schedule one step short of the
        // twenty-five that was meant. Six places makes it 0.8 again and the division exact.
        //
        // Worked out in f64 and checked before it is narrowed: a strength near zero asks for a
        // schedule of millions of steps, and the number to say that about is the pair that was
        // asked for rather than the one it came to.
        let strength = (strength as f64 * 1e6).round() / 1e6;
        let whole = (num_steps as f64 / strength).floor();
        if !(1.0..=config.num_train_timesteps as f64).contains(&whole) {
            return Err(Error::model(format!(
                "{num_steps} steps at a strength of {strength} is the last {num_steps} of a \
                 {whole} step schedule, and this model was trained on {}",
                config.num_train_timesteps
            )));
        }

        let mut sampler = EulerSampler::new(config, whole as i32)?;
        sampler.start = sampler
            .timesteps
            .len()
            .saturating_sub(num_steps.max(0) as usize);

        Ok(sampler)
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

    /// The first step this run walks. Zero for a run from noise; further in for one from a
    /// picture, where everything below it is a noise level the picture stands in for.
    pub fn start(&self) -> usize {
        self.start
    }

    /// How many steps actually run, which is what a step count on a screen means.
    pub fn steps_to_run(&self) -> usize {
        self.timesteps.len() - self.start
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
    ///
    /// Taken in float32 whatever the model ran in, which is what the reference implementation
    /// does and for the same reason. A step adds a small correction to a large sample, and in
    /// half precision the correction loses its low bits to the addition -- the more steps, the
    /// more of them lost. The latent is four channels of a small grid, so widening it for one
    /// addition costs nothing worth measuring.
    pub fn step(&self, noise: &Tensor, sample: &Tensor, index: usize) -> Result<Tensor> {
        let sigma = self.sigma(index)?;
        let next = self.sigmas[index + 1];

        let stepped = F::add(
            &sample.cast(DType::Float)?,
            &F::mul_scalar(&noise.cast(DType::Float)?, next - sigma)?,
        )?;
        Ok(stepped.cast(noise.dtype())?)
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

    fn from_image(num_steps: i32, strength: f32) -> EulerSampler {
        EulerSampler::from_image(&SamplerConfig::default(), num_steps, strength).unwrap()
    }

    #[test]
    fn the_step_count_is_the_steps_that_run_whatever_the_strength() {
        // The whole point of the shape. Twenty steps is twenty steps at every strength, and what
        // the strength changes is the schedule they are the tail of.
        for strength in [1.0, 0.8, 0.5, 0.25] {
            let sampler = from_image(20, strength);
            assert_eq!(sampler.steps_to_run(), 20, "at a strength of {strength}");
            assert_eq!(sampler.len() - sampler.start(), 20);
        }
    }

    #[test]
    fn the_strength_is_how_far_down_the_schedule_the_tail_begins() {
        // Twenty of forty at a half, twenty of twenty-five at 0.8: the last num_steps of a
        // schedule num_steps/strength long, which is the arithmetic ComfyUI does.
        let half = from_image(20, 0.5);
        assert_eq!(half.len(), 40);
        assert_eq!(half.start(), 20);

        let most = from_image(20, 0.8);
        assert_eq!(most.len(), 25);
        assert_eq!(most.start(), 5);

        // Where the arithmetic is not exact the tail is what is left of the floor, which is what
        // ComfyUI's int(steps / denoise) gives: 20 / 0.75 is 26 and not 27.
        let rough = from_image(20, 0.75);
        assert_eq!(rough.len(), 26);
        assert_eq!(rough.start(), 6);

        // The whole of it at one, which is the same schedule text to image walks.
        let all = from_image(20, 1.0);
        assert_eq!(all.len(), 20);
        assert_eq!(all.start(), 0);
        assert_eq!(all.sigmas(), sampler(20).sigmas());
    }

    #[test]
    fn the_schedule_is_the_one_comfyui_would_have_built() {
        // Its KSampler::set_steps, which is where this shape comes from:
        //
        //     new_steps = int(steps / denoise)
        //     sigmas = calculate_sigmas(new_steps)[-(steps + 1):]
        //
        // The pairs below are that arithmetic run in python, so a change here that quietly moved
        // the tail would have to disagree with the tool people are coming from.
        for (steps, strength, whole) in [
            (20, 1.0, 20),
            (20, 0.9, 22),
            (20, 0.8, 25),
            (20, 0.75, 26),
            (20, 0.6, 33),
            (20, 0.5, 40),
            (20, 0.25, 80),
            (30, 0.8, 37),
        ] {
            let sampler = from_image(steps, strength);
            assert_eq!(sampler.len(), whole, "{steps} steps at {strength}");
            assert_eq!(sampler.start(), whole - steps as usize);
            assert_eq!(sampler.steps_to_run(), steps as usize);
        }
    }

    #[test]
    fn a_lower_strength_starts_from_less_noise() {
        // What the knob is for, said in the one number that decides it: the level the picture is
        // put back at. Every step count agrees, since the tail begins at the same fraction.
        for steps in [10, 20, 50] {
            let noise: Vec<f32> = [1.0, 0.8, 0.6, 0.4, 0.2]
                .iter()
                .map(|strength| {
                    let sampler = from_image(steps, *strength);
                    sampler.sigmas()[sampler.start()]
                })
                .collect();

            for pair in noise.windows(2) {
                assert!(pair[0] > pair[1], "at {steps} steps: {noise:?}");
            }
        }
    }

    #[test]
    fn no_strength_runs_no_steps_and_adds_no_noise() {
        // Allowed rather than refused: it is a picture that went through the autoencoder and came
        // back. The walk begins past the end, where the schedule's last noise level is zero.
        let sampler = from_image(20, 0.0);
        assert_eq!(sampler.steps_to_run(), 0);
        assert_eq!(sampler.start(), sampler.len());
        assert_eq!(sampler.sigmas()[sampler.start()], 0.0);
    }

    #[test]
    fn a_run_from_noise_starts_at_the_top() {
        let sampler = sampler(20);
        assert_eq!(sampler.start(), 0);
        assert_eq!(sampler.steps_to_run(), 20);
    }

    #[test]
    fn refuses_a_strength_outside_the_range() {
        let config = SamplerConfig::default();
        assert!(EulerSampler::from_image(&config, 20, -0.1).is_err());
        assert!(EulerSampler::from_image(&config, 20, 1.1).is_err());
        assert!(EulerSampler::from_image(&config, 20, f32::NAN).is_err());
    }

    #[test]
    fn refuses_more_steps_than_the_schedule_they_are_the_tail_of_can_hold() {
        // Thirty steps at a strength of 0.01 is the last thirty of three thousand, and the model
        // was trained on one thousand. Said as the pair that was asked for rather than as the
        // number it came to, which is not one anybody typed.
        let config = SamplerConfig::default();
        let error = EulerSampler::from_image(&config, 30, 0.01)
            .unwrap_err()
            .to_string();
        assert!(error.contains("30 steps"), "{error}");
        assert!(error.contains("0.01"), "{error}");

        // And the edge that does fit: a thousand steps exactly.
        assert!(EulerSampler::from_image(&config, 500, 0.5).is_ok());
        assert!(EulerSampler::from_image(&config, 501, 0.5).is_err());
    }

    #[test]
    fn a_step_past_the_end_is_refused() {
        let sampler = sampler(4);
        let sample = Tensor::zeros(&[1, 4, 8, 8], DType::Float, Device::Cpu).unwrap();
        assert!(sampler.step(&sample, &sample, 4).is_err());
        assert!(sampler.scale_model_input(&sample, 4).is_err());
    }
}
