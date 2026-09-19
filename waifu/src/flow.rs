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

//! Rectified flow, which is what [`Anima`](crate::Anima) and [`Krea2`](crate::Krea2) are sampled
//! on and what `waifu::sdxl` cannot do.
//!
//! The two schedules are not variants of each other. [`EulerSampler`](crate::EulerSampler) builds
//! sigmas out of `alphas_cumprod` and steps on the epsilon the model returns; this walks a
//! straight line. At sigma the latent is `sigma * noise + (1 - sigma) * image`, the model returns
//! the velocity along that line -- `noise - image`, the same at every point, which is what makes
//! the path straight -- and a step is simply how far along it to move.
//!
//! Two numbers come from the package rather than from here. The schedule is bent by a **shift**,
//! which spends more of the budget at high noise where the picture is decided; and the denoiser is
//! handed `sigma * multiplier`, where both families' multiplier is **one**. That last is worth
//! saying twice: a flow model is usually passed a thousandfold timestep, and passing this one 750
//! instead of 0.75 costs nothing at load time and produces noise.
//!
//! One schedule, two families, because it is one schedule: Krea 2's reference bends its sigmas
//! with `exp(mu) / (exp(mu) + (1 / t - 1))` and Anima's with `shift * t / (1 + (shift - 1) * t)`,
//! and those are the same curve written twice -- a package that wants Krea 2's `mu` writes
//! `exp(mu)` as its shift. See `docs/krea2.md`.

use crate::error::{Error, Result};
use crate::flint::{functional as F, Tensor};

/// What a package says about how to sample it.
#[derive(Clone, Copy, Debug)]
pub struct SamplerConfig {
    /// Bends the schedule towards high noise. One leaves it straight; Anima ships three.
    pub shift: f32,
    /// What the sigma is multiplied by on its way to the denoiser. One for Anima.
    pub multiplier: f32,
}

/// A schedule, built once for a given number of steps.
///
/// `sigmas` holds one more entry than there are steps: the level each step starts at, and the zero
/// the last one finishes at.
#[derive(Clone, Debug)]
pub struct FlowSampler {
    config: SamplerConfig,
    sigmas: Vec<f32>,
}

impl FlowSampler {
    pub fn new(config: &SamplerConfig, num_steps: i32) -> Result<FlowSampler> {
        if num_steps < 1 {
            return Err(Error::model(format!(
                "{num_steps} steps is not enough to draw anything"
            )));
        }
        if config.shift <= 0.0 {
            return Err(Error::model(format!(
                "a shift of {} does not describe a schedule",
                config.shift
            )));
        }

        // Evenly spaced in time, then bent: `shift * t / (1 + (shift - 1) * t)`, which fixes both
        // ends -- one stays one and zero stays zero -- and slows the walk down in between.
        let sigmas = (0..=num_steps)
            .map(|index| {
                let t = 1.0 - index as f32 / num_steps as f32;
                config.shift * t / (1.0 + (config.shift - 1.0) * t)
            })
            .collect();

        Ok(FlowSampler {
            config: *config,
            sigmas,
        })
    }

    /// The noise levels, one longer than [`FlowSampler::steps`].
    pub fn sigmas(&self) -> &[f32] {
        &self.sigmas
    }

    pub fn steps(&self) -> usize {
        self.sigmas.len() - 1
    }

    /// What the denoiser is handed at `index` -- the sigma, scaled by the package's multiplier.
    pub fn timestep(&self, index: usize) -> Result<f32> {
        self.sigma(index)
            .map(|sigma| sigma * self.config.multiplier)
    }

    pub fn sigma(&self, index: usize) -> Result<f32> {
        self.sigmas.get(index).copied().ok_or_else(|| {
            Error::model(format!(
                "step {index} is past the {} this schedule holds",
                self.steps()
            ))
        })
    }

    /// The latent a pass starts from: unit noise, since at sigma one there is nothing else in it.
    pub fn initial_noise_scale(&self) -> f32 {
        self.sigmas.first().copied().unwrap_or(1.0)
    }

    /// Move `latent` along `velocity` by one step.
    ///
    /// The whole of the sampler: `x + v * (next - now)`. There is no schedule arithmetic hiding
    /// in here because a straight path does not need any -- which is also why being wrong about
    /// the *timestep* handed to the model is so much easier than being wrong about this.
    pub fn step(&self, index: usize, latent: &Tensor, velocity: &Tensor) -> Result<Tensor> {
        let now = self.sigma(index)?;
        let next = self.sigma(index + 1)?;

        Ok(F::add(latent, &F::mul_scalar(velocity, next - now)?)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> SamplerConfig {
        SamplerConfig {
            shift: 3.0,
            multiplier: 1.0,
        }
    }

    #[test]
    fn runs_from_all_noise_down_to_none() {
        let sampler = FlowSampler::new(&config(), 10).unwrap();
        assert_eq!(sampler.steps(), 10);
        assert_eq!(sampler.sigmas().len(), 11);
        assert_eq!(sampler.sigmas()[0], 1.0, "it has to start at pure noise");
        assert_eq!(sampler.sigmas()[10], 0.0, "and finish with none left");
    }

    #[test]
    fn never_turns_back() {
        let sampler = FlowSampler::new(&config(), 12).unwrap();
        for pair in sampler.sigmas().windows(2) {
            assert!(pair[1] < pair[0], "{pair:?} goes the wrong way");
        }
    }

    #[test]
    fn the_shift_spends_the_budget_at_high_noise() {
        let straight = FlowSampler::new(
            &SamplerConfig {
                shift: 1.0,
                ..config()
            },
            8,
        )
        .unwrap();
        let bent = FlowSampler::new(&config(), 8).unwrap();

        // A shift of one is the identity, so the straight schedule is evenly spaced.
        assert!((straight.sigmas()[4] - 0.5).abs() < 1e-6);

        // Anima's three holds the sigmas higher for longer, which is the point of it: more of the
        // steps are spent where the picture is still being decided.
        for index in 1..8 {
            assert!(
                bent.sigmas()[index] > straight.sigmas()[index],
                "at {index}: {} is not above {}",
                bent.sigmas()[index],
                straight.sigmas()[index]
            );
        }
    }

    #[test]
    fn hands_the_denoiser_the_sigma_itself() {
        // The multiplier that is one, which is the thing about this model a flow sampler written
        // from memory would get wrong.
        let sampler = FlowSampler::new(&config(), 4).unwrap();
        for index in 0..4 {
            assert_eq!(
                sampler.timestep(index).unwrap(),
                sampler.sigma(index).unwrap()
            );
        }
    }

    #[test]
    fn scales_the_timestep_when_a_package_asks_for_it() {
        let thousandfold = SamplerConfig {
            multiplier: 1000.0,
            ..config()
        };
        let sampler = FlowSampler::new(&thousandfold, 4).unwrap();
        assert_eq!(sampler.timestep(0).unwrap(), 1000.0);
    }

    #[test]
    fn refuses_a_schedule_that_is_not_one() {
        assert!(FlowSampler::new(&config(), 0).is_err());
        assert!(FlowSampler::new(
            &SamplerConfig {
                shift: 0.0,
                ..config()
            },
            4
        )
        .is_err());
    }

    #[test]
    fn refuses_a_step_it_does_not_have() {
        let sampler = FlowSampler::new(&config(), 4).unwrap();
        assert!(sampler.sigma(4).is_ok(), "the last level is the zero");
        assert!(sampler.sigma(5).is_err());
    }
}
