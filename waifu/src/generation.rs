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

//! What a caller asks a model to draw, and what it hears back while the drawing happens.
//!
//! Neither of these belongs to a particular model. They are the request and the running commentary
//! on it, and [`crate::Sdxl`] and [`crate::Anima`] answer the same two -- which is what lets the
//! screen hold either one without knowing which it has.

use std::ops::ControlFlow;

/// What to ask for. The defaults are what SDXL was trained at; a model that wants others says so
/// through [`GenerationDefaults`].
#[derive(Clone, Debug)]
pub struct GenerationOptions {
    pub width: i32,
    pub height: i32,
    pub num_steps: i32,
    /// How hard to push away from the unprompted answer. One means not at all, which also skips
    /// the second denoiser pass and so runs twice as fast; five to eight is the usual range.
    pub guidance_scale: f32,
    /// What to steer away from. An empty string is the usual thing to steer away from and is not
    /// the same as no negative prompt at all -- the model has an opinion about the empty prompt.
    pub negative_prompt: String,
    pub seed: Option<u64>,
    /// How far to walk away from a picture handed to a model that draws from one, between zero
    /// and one. Read by nothing else: drawing from noise always walks the whole schedule.
    ///
    /// It does not change how many steps run -- `num_steps` is that, either way. What it changes
    /// is how noisy the picture is when they start.
    pub strength: f32,
}

impl Default for GenerationOptions {
    fn default() -> GenerationOptions {
        GenerationOptions {
            width: 1024,
            height: 1024,
            num_steps: 30,
            guidance_scale: 5.0,
            negative_prompt: String::new(),
            seed: None,
            strength: 0.8,
        }
    }
}

/// What a model would rather be asked for, where that differs from [`GenerationOptions::default`].
///
/// A distilled model is the reason this exists. Anima's turbo release wants eight steps at no
/// guidance and produces a burnt, over-contrasted picture at the thirty and five SDXL likes, so
/// a screen that offers one set of defaults to both is offering the wrong ones to one of them.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GenerationDefaults {
    pub width: i32,
    pub height: i32,
    pub num_steps: i32,
    pub guidance_scale: f32,
}

impl Default for GenerationDefaults {
    fn default() -> GenerationDefaults {
        let options = GenerationOptions::default();
        GenerationDefaults {
            width: options.width,
            height: options.height,
            num_steps: options.num_steps,
            guidance_scale: options.guidance_scale,
        }
    }
}

impl GenerationDefaults {
    /// These defaults as a whole request, for a caller that has nothing else to say.
    pub fn options(&self) -> GenerationOptions {
        GenerationOptions {
            width: self.width,
            height: self.height,
            num_steps: self.num_steps,
            guidance_scale: self.guidance_scale,
            ..GenerationOptions::default()
        }
    }
}

/// How far along a run is, as the reporter given to a model's `generate_reporting` is told.
///
/// The three are not the same size: encoding costs about as much as one step, a step is a step,
/// and the decode at the end is several. A bar drawn from the step count alone will sit at the
/// end for a while, which is the honest thing for it to do.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GenerationProgress {
    /// Reading the prompt -- and the picture, on a run that starts from one -- which happens
    /// once before the first step.
    Encoding,
    /// The denoising step that just finished, out of how many there are.
    Step { done: i32, total: i32 },
    /// Turning the finished latent into pixels.
    Decoding,
}

/// The reporter a run that nobody is watching gets.
pub(crate) fn unwatched(_: GenerationProgress) -> ControlFlow<()> {
    ControlFlow::Continue(())
}
