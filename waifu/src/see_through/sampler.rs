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

//! The two samplers See-through's models were run with, neither of them the Euler one SDXL is.
//!
//! The layers model is walked by diffusers' `DPMSolverMultistepScheduler` as its scheduler config
//! sets it up: the SDE variant of DPM-Solver++, second order with the midpoint correction, over
//! the "leading" timesteps, ending at no noise at all. It asks the model for noise and steps on the
//! clean latent that implies, adding fresh noise of its own at every step.
//!
//! The depth model is walked by DDIM, four steps, over the "trailing" timesteps of a schedule
//! rescaled so that its last step is pure noise, and it is asked for velocity rather than noise.
//! That is Marigold's recipe, and four steps is enough because the depth it is asked for is an
//! image a good deal simpler than the picture it is read from.
//!
//! Both are arithmetic on the betas and have no weights. The schedules are computed in double
//! precision on the host and applied to the latent in float32, which is what diffusers upcasts to
//! for the step too.

use crate::error::{Error, Result};
use crate::flint::{functional as F, DType, Tensor};

/// The noise schedule a model was trained on.
#[derive(Clone, Debug)]
pub struct ScheduleConfig {
    pub num_train_timesteps: i32,
    pub beta_start: f64,
    pub beta_end: f64,
    pub steps_offset: i32,
}

/// How much of the signal is left at each training step, for diffusers' "scaled_linear" betas.
///
/// `zero_snr` rescales them so that the last step keeps none of it, which is what
/// `rescale_betas_zero_snr` does: the square roots of these are shifted so the last is zero and
/// scaled so the first is unchanged.
fn alphas_cumprod(config: &ScheduleConfig, zero_snr: bool) -> Vec<f64> {
    let n = config.num_train_timesteps as usize;
    let (start, end) = (config.beta_start.sqrt(), config.beta_end.sqrt());
    let mut product = 1.0;
    let mut out: Vec<f64> = (0..n)
        .map(|i| {
            let root = match n {
                1 => start,
                _ => start + (end - start) * i as f64 / (n - 1) as f64,
            };
            product *= 1.0 - root * root;
            product
        })
        .collect();

    if zero_snr {
        let roots: Vec<f64> = out.iter().map(|a| a.sqrt()).collect();
        let (first, last) = (roots[0], roots[n - 1]);
        out = roots
            .iter()
            .map(|r| {
                let r = (r - last) * first / (first - last);
                r * r
            })
            .collect();
    }
    out
}

fn as_float(tensor: &Tensor) -> Result<Tensor> {
    Ok(tensor.cast(DType::Float)?)
}

/// `a * x + b * y`, in float32.
fn axpby(a: f64, x: &Tensor, b: f64, y: &Tensor) -> Result<Tensor> {
    Ok(F::add(
        &F::mul_scalar(x, a as f32)?,
        &F::mul_scalar(y, b as f32)?,
    )?)
}

/// One noise level, as the VP parameterisation DPM-Solver works in: how much of the signal and of
/// the noise a latent there holds, and the log of their ratio.
#[derive(Clone, Copy, Debug)]
struct Level {
    alpha: f64,
    sigma: f64,
    lambda: f64,
}

impl Level {
    /// From diffusers' sigma, which is the noise relative to the signal.
    fn from_sigma(sigma: f64) -> Level {
        let alpha = 1.0 / (sigma * sigma + 1.0).sqrt();
        let sigma = sigma * alpha;
        Level {
            alpha,
            sigma,
            lambda: alpha.ln() - sigma.ln(),
        }
    }
}

/// DPM-Solver++, second order, with its SDE noise -- `algorithm_type="sde-dpmsolver++"`,
/// `solver_type="midpoint"`, `final_sigmas_type="zero"`, `lower_order_final=True`.
pub struct DpmSolverSde {
    timesteps: Vec<i32>,
    /// One more than there are timesteps: the last is no noise at all.
    levels: Vec<Level>,
    /// The clean latent the previous step implied, which the second order step corrects with.
    previous: Option<Tensor>,
}

impl DpmSolverSde {
    pub fn new(config: &ScheduleConfig, num_steps: i32) -> Result<DpmSolverSde> {
        if num_steps < 1 || num_steps >= config.num_train_timesteps {
            return Err(Error::model(format!(
                "{num_steps} steps of a {}-step schedule",
                config.num_train_timesteps
            )));
        }
        let alphas = alphas_cumprod(config, false);

        // "leading": evenly spaced from zero by `train // (steps + 1)`, the top one dropped, then
        // shifted up by the offset.
        let ratio = config.num_train_timesteps / (num_steps + 1);
        let timesteps: Vec<i32> = (1..=num_steps)
            .rev()
            .map(|i| (i * ratio + config.steps_offset).min(config.num_train_timesteps - 1))
            .collect();

        let mut levels: Vec<Level> = timesteps
            .iter()
            .map(|&t| {
                let a = alphas[t as usize];
                Level::from_sigma(((1.0 - a) / a).sqrt())
            })
            .collect();
        levels.push(Level {
            alpha: 1.0,
            sigma: 0.0,
            lambda: f64::INFINITY,
        });

        Ok(DpmSolverSde {
            timesteps,
            levels,
            previous: None,
        })
    }

    pub fn timesteps(&self) -> &[i32] {
        &self.timesteps
    }

    /// What the model is asked at step `index`: the training step, as the U-Net takes it.
    pub fn timestep(&self, index: usize) -> f32 {
        self.timesteps[index] as f32
    }

    pub fn len(&self) -> usize {
        self.timesteps.len()
    }

    pub fn is_empty(&self) -> bool {
        self.timesteps.is_empty()
    }

    /// The latent at step `index + 1`, from the one at `index`, the noise the model saw in it, and
    /// `noise`, fresh unit noise the shape of the latent.
    ///
    /// Steps have to be taken in order and each once: the second order correction reads what the
    /// step before implied.
    pub fn step(
        &mut self,
        index: usize,
        model_noise: &Tensor,
        sample: &Tensor,
        noise: &Tensor,
    ) -> Result<Tensor> {
        if index >= self.len() {
            return Err(Error::model(format!(
                "step {index} of a {}-step walk",
                self.len()
            )));
        }
        let (sample, model_noise) = (as_float(sample)?, as_float(model_noise)?);
        let here = self.levels[index];
        let next = self.levels[index + 1];

        // The clean latent the model's answer implies.
        let clean = axpby(
            1.0 / here.alpha,
            &sample,
            -here.sigma / here.alpha,
            &model_noise,
        )?;

        // The last step lands on no noise, which is the clean latent itself; diffusers takes it as
        // a first order step whose coefficients come to exactly that.
        let first_order = index == 0 || index + 1 == self.len() || self.previous.is_none();
        let out = if next.sigma == 0.0 {
            clean.clone()
        } else {
            let h = next.lambda - here.lambda;
            let decay = (-h).exp();
            let kept = next.sigma / here.sigma * decay;
            let gain = next.alpha * (1.0 - (-2.0 * h).exp());
            let spread = next.sigma * (1.0 - (-2.0 * h).exp()).sqrt();

            let mut x = axpby(kept, &sample, gain, &clean)?;
            if !first_order {
                let before = self.levels[index - 1];
                let r0 = (here.lambda - before.lambda) / h;
                let previous = self.previous.as_ref().expect("checked above");
                // D1 = (D0 - D_prev) / r0, added at half the gain.
                let difference = F::sub(&clean, previous)?;
                x = F::add(&x, &F::mul_scalar(&difference, (0.5 * gain / r0) as f32)?)?;
            }
            F::add(&x, &F::mul_scalar(&as_float(noise)?, spread as f32)?)?
        };

        self.previous = Some(clean);
        Ok(out)
    }
}

/// DDIM over a zero terminal SNR schedule, the model asked for velocity -- Marigold's sampler.
pub struct DdimVelocity {
    timesteps: Vec<i32>,
    alphas: Vec<f64>,
    stride: i32,
}

impl DdimVelocity {
    pub fn new(config: &ScheduleConfig, num_steps: i32) -> Result<DdimVelocity> {
        let train = config.num_train_timesteps;
        if num_steps < 1 || num_steps > train {
            return Err(Error::model(format!(
                "{num_steps} steps of a {train}-step schedule"
            )));
        }
        // "trailing": evenly spaced down from the top, one short of each mark.
        let ratio = train as f64 / num_steps as f64;
        let timesteps = (0..num_steps)
            .map(|i| (train as f64 - i as f64 * ratio).round() as i32 - 1)
            .collect();
        Ok(DdimVelocity {
            timesteps,
            alphas: alphas_cumprod(config, true),
            stride: train / num_steps,
        })
    }

    pub fn timesteps(&self) -> &[i32] {
        &self.timesteps
    }

    pub fn timestep(&self, index: usize) -> f32 {
        self.timesteps[index] as f32
    }

    pub fn len(&self) -> usize {
        self.timesteps.len()
    }

    pub fn is_empty(&self) -> bool {
        self.timesteps.is_empty()
    }

    /// The latent at step `index + 1` from the one at `index` and the velocity the model saw.
    pub fn step(&self, index: usize, velocity: &Tensor, sample: &Tensor) -> Result<Tensor> {
        let t = self.timesteps[index];
        let a = self.alphas[t as usize];
        // `set_alpha_to_one=False`: past the first training step is the first training step.
        let previous = t - self.stride;
        let a_prev = match previous >= 0 {
            true => self.alphas[previous as usize],
            false => self.alphas[0],
        };

        let (sample, velocity) = (as_float(sample)?, as_float(velocity)?);
        let clean = axpby(a.sqrt(), &sample, -(1.0 - a).sqrt(), &velocity)?;
        let noise = axpby(a.sqrt(), &velocity, (1.0 - a).sqrt(), &sample)?;
        axpby(a_prev.sqrt(), &clean, (1.0 - a_prev).sqrt(), &noise)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(steps_offset: i32) -> ScheduleConfig {
        ScheduleConfig {
            num_train_timesteps: 1000,
            beta_start: 0.00085,
            beta_end: 0.012,
            steps_offset,
        }
    }

    #[test]
    fn the_layers_schedule_leads_from_961() {
        let sampler = DpmSolverSde::new(&config(1), 30).unwrap();
        let t = sampler.timesteps();
        assert_eq!(t.len(), 30);
        assert_eq!(t[0], 961);
        assert_eq!(t[1], 929);
        assert_eq!(t[29], 33);
        assert_eq!(sampler.levels.last().unwrap().sigma, 0.0);
    }

    #[test]
    fn the_depth_schedule_trails_from_999_to_pure_noise() {
        let sampler = DdimVelocity::new(&config(0), 4).unwrap();
        assert_eq!(sampler.timesteps(), &[999, 749, 499, 249]);
        assert_eq!(sampler.alphas[999], 0.0);
        // The first step keeps what the unrescaled schedule keeps.
        let plain = alphas_cumprod(&config(0), false);
        assert!((sampler.alphas[0] - plain[0]).abs() < 1e-12);
    }

    #[test]
    fn the_unrescaled_schedule_is_what_diffusers_computes() {
        // diffusers' alphas_cumprod for these betas, at three training steps.
        let alphas = alphas_cumprod(&config(1), false);
        assert!((alphas[0] - 0.99915).abs() < 1e-5);
        assert!((alphas[999] - 0.0046600).abs() < 1e-5);
    }
}
