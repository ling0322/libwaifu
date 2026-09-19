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

//! Anima, which draws the same kind of picture as [`sdxl`](crate::sdxl) and shares none of its
//! parts.
//!
//! Where SDXL is a U-Net over four-channel latents conditioned by two CLIP encoders and sampled
//! on epsilon, this is a transformer over sixteen-channel latents conditioned through a Qwen3
//! encoder and a bridge onto it, sampled on flow. The two have the package format in common and
//! little else. `docs/anima.md` is where the whole of it is written down.

mod adapter;
mod config;
mod dit;
mod pipeline;
mod text_encoder;

/// The epsilon every norm in this model that has one uses.
pub(crate) const NORM_EPS: f32 = 1e-6;

pub use adapter::Adapter;
pub use config::{AdapterConfig, AnimaConfig, DitConfig, Prediction, TextConfig};
pub use dit::Dit;
pub use pipeline::{Anima, VAE_SCALE};
pub use text_encoder::TextEncoder;

/// The two halves Anima shares with [`Krea2`](crate::Krea2), re-exported here so that a reader
/// who has this model in hand finds them where the rest of it is. The autoencoder is the same
/// Qwen-Image one; the sampler is the same straight line. Neither is Anima's own, and neither is
/// written twice -- see [`crate::flow`] and [`crate::qwen_vae`].
pub use crate::flow::{FlowSampler, SamplerConfig};
pub use crate::qwen_vae::{VaeConfig, VaeDecoder};
