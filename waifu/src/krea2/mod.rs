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


//! Krea 2, the third family here and the largest: twelve billion parameters of single-stream
//! MMDiT over the same sixteen-channel latents [`anima`](crate::anima) draws through.
//!
//! What makes it its own module rather than a variant of one:
//!
//! *There is one stream, not two.* The prompt and the picture are concatenated into a single
//! sequence and every block attends over the whole of it -- no cross attention, no separate text
//! branch after the first few blocks.
//!
//! *The prompt arrives as twelve layers rather than one.* The conditioning is not the encoder's
//! last hidden state; it is the states of twelve of its thirty-six decoder layers, stacked, and a
//! small transformer inside the denoiser attends *across that stack* before it ever looks along
//! the sentence. See [`Dit`].
//!
//! *Every block modulates on one vector.* There is no per-block projection of the timestep the
//! way Cosmos has: one `(1, 6D)` is computed once and each block adds its own learned table to it.
//!
//! What it shares is real and is not copied: the Qwen-Image autoencoder ([`crate::qwen_vae`]) and
//! the straight-line flow schedule ([`crate::flow`]), both of which Anima uses unchanged.
//!
//! `docs/krea2.md` is where the whole of it is written down, including the several things here
//! that are easy to get wrong and fail quietly.

mod config;
pub(crate) mod dit;
mod pipeline;
mod text_encoder;

pub use config::{DitConfig, EncoderConfig, Krea2Config, Prediction};
pub use dit::Dit;
pub use pipeline::{Krea2, VAE_SCALE};
pub use text_encoder::TextEncoder;

/// The two halves Krea 2 shares with [`Anima`](crate::Anima). The autoencoder is the same
/// Qwen-Image one, down to the sixteen means and deviations; the sampler is the same straight
/// line, bent by a shift this package writes as `exp(mu)`.
pub use crate::flow::{FlowSampler, SamplerConfig};
pub use crate::qwen_vae::{VaeConfig, VaeDecoder};
