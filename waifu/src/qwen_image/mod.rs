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

//! Qwen-Image 2.1: a seven billion parameter single-stream DiT, reading the last layer of a
//! Qwen3-VL-8B encoder and drawing through a sixty-four channel autoencoder of its own.
//!
//! What it shares is shared rather than copied: the Qwen3-VL text tower is
//! [`krea2::TextEncoder`](crate::krea2::TextEncoder), tapped at one layer instead of twelve; the
//! adjacent-pair rotation and the timestep sinusoid are Krea 2's; the autoencoder's channel norm
//! and attention are the Qwen-Image one's; the schedule is [`crate::flow`], bent by a shift that
//! depends on the picture's size and stretched to a terminal noise level.
//!
//! What is its own: a denoiser whose attention is block causal and whose prompt is modulated at
//! timestep zero ([`Dit`]), and a Wan 2.2-style decoder with shortcuts around its upsampling
//! stages and an alpha channel out ([`VaeDecoder`]). `docs/qwen_image.md` has the whole of it.

mod config;
mod dit;
mod pipeline;
mod vae;

pub use config::{
    ControlConfig, DitConfig, Prediction, QwenImageConfig, ScheduleConfig, VaeConfig,
};
pub use dit::Dit;
pub use pipeline::QwenImage;
pub use vae::{VaeDecoder, VaeEncoder};

pub use crate::flow::{FlowSampler, SamplerConfig};
pub use crate::krea2::{EncoderConfig, TextEncoder};
