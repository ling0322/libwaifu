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

//! Diffusion image generation on top of the flint tensor library.
//!
//! It reads a model package, builds the model it describes, and runs it; the tensor operations
//! themselves are the ones [`flint`] binds, and that module is the safe wrapper over the native
//! `libflint.a` this crate links.
//!
//! ```no_run
//! use waifu::Manifest;
//!
//! let manifest = Manifest::open("sdxl-base.yaml")?;
//! println!("{} tensors", manifest.params()?.len());
//! # Ok::<(), waifu::Error>(())
//! ```
//!
//! # Threading
//!
//! A [`flint::Tensor`] stays on the thread that made it, so everything built out of one does too.

/// Anima. Kept a module rather than flattened into the crate root the way [`sdxl`] is: the two
/// architectures name several of the same things -- a VAE configuration, a text configuration --
/// and they are not the same things.
pub mod anima;
/// Audio: convolution along one axis, a Fourier transform, a filterbank and a vocoder's
/// activation, all composed from the operators [`flint`] already has rather than written as
/// kernels of their own.
pub mod audio;
#[cfg(feature = "cli")]
pub mod cli;
mod error;
pub mod flint;
mod flow;
mod generation;
/// Krea 2. A module for the same reason [`anima`] is one, and the third family here: a
/// single-stream MMDiT over the same sixteen-channel latents, conditioned on twelve tapped layers
/// of a Qwen3-VL encoder. `docs/krea2.md` is where the whole of it is written down.
pub mod krea2;
mod layers;
mod manifest;
mod mapping;
mod param_file;
mod qwen_vae;
mod reader;
mod sdxl;
/// Turning text into a waveform. [`Voice`] is the shape of the question and [`Tones`] is the one
/// answer to it there is so far -- a stand-in that makes a noise where the syllables are, so that
/// the screen and everything under it can be written before there is a model to put behind it.
pub mod speech;
mod suggested;
mod tokenizer;
/// Reading and writing WAV, which is the one audio format this crate handles itself: a header, a
/// format chunk and the samples, and no codec to depend on.
pub mod wav;
mod yaml;

/// The tensor types a caller of this crate needs to name, re-exported so that the common case
/// does not have to reach into [`flint`]. [`Residency`](flint::Residency) is here for the same
/// reason: asking for a model is where it has to be said.
pub use flint::{DType, Device, Fp8Tensor, Nvfp4Tensor, Residency, WeightFormat};

pub use anima::Anima;
pub use error::{Error, Result};
pub use krea2::Krea2;
pub use generation::{GenerationDefaults, GenerationOptions, GenerationProgress};
pub use layers::{Conv2d, Embedding, GroupNorm, LayerNorm, Linear};
pub use manifest::Manifest;
pub use mapping::Mapping;
pub use param_file::ParamFile;
pub use reader::BinaryRead;
pub use speech::{SpeechDefaults, SpeechOptions, SpeechProgress, Tones, Voice};
pub use sdxl::{
    from_rgb8, to_rgb8, ClipTextConfig, ClipTextEncoder, ClipTextOutput, EulerSampler,
    PromptEmbedding, SamplerConfig, Sdxl, SdxlConfig, Unet, UnetCondition, UnetConfig, VaeConfig,
    VaeDecoder, VaeEncoder, VAE_SCALE,
};
pub use suggested::{Size, Suggestions};
pub use tokenizer::Tokenizer;
pub use wav::Sound;

/// The suffix a model's manifest carries, after the model's id: `sdxl-base.yaml`.
///
/// What a model is lives here, beside the files it names rather than inside one of them. See
/// [`Manifest`].
pub const MANIFEST_SUFFIX: &str = Manifest::SUFFIX;

/// The suffix a file of weights carries. See [`ParamFile`].
pub const WEIGHTS_SUFFIX: &str = Manifest::WEIGHTS_SUFFIX;
