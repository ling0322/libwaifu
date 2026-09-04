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
//! use waifu::{DType, Device, VarBuilder, ZipFile};
//!
//! let mut package = ZipFile::open("model.waifupkg")?;
//! let config = waifu::IniConfig::parse(&package.read_to_string(waifu::MODEL_CONFIG)?)?;
//! let mut params = package.open_entry(config.section("model")?.get_str("model_file")?)?;
//! let vb = VarBuilder::from_reader(&mut params, Device::Cpu, DType::Float)?;
//! println!("{} tensors", vb.len());
//! # Ok::<(), waifu::Error>(())
//! ```
//!
//! # Threading
//!
//! A [`flint::Tensor`] stays on the thread that made it, so everything built out of one does too.

mod bpe;
#[cfg(feature = "cli")]
pub mod cli;
mod error;
pub mod flint;
mod ini;
mod layers;
mod reader;
mod sdxl;
mod tokenizer;
mod var_builder;
mod zip_file;

pub use bpe::{BpeConfig, BpeEncoder, BpeModel, PreTokenizer, INVALID_TOKEN};
/// The tensor types a caller of this crate needs to name, re-exported so that the common case
/// does not have to reach into [`flint`].
pub use flint::{DType, Device, Nvfp4Tensor};

pub use error::{Error, Result};
pub use ini::{IniConfig, IniSection};
pub use layers::{Conv2d, Embedding, GroupNorm, LayerNorm, Linear};
pub use reader::BinaryRead;
pub use sdxl::{
    from_rgb8, to_rgb8, ClipTextConfig, ClipTextEncoder, ClipTextOutput, EulerSampler,
    GenerationOptions, GenerationProgress, PromptEmbedding, SamplerConfig, Sdxl, SdxlConfig, Unet,
    UnetCondition, UnetConfig, VaeConfig, VaeDecoder, VaeEncoder, VAE_SCALE,
};
pub use tokenizer::Tokenizer;
pub use var_builder::VarBuilder;
pub use zip_file::ZipFile;

/// The name of the configuration entry every model package holds.
pub const MODEL_CONFIG: &str = "model.ini";
