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

//! The model manager: which models are published, fetching them, and where they are kept.
//!
//! libwaifu_modelmanager. None of it runs a model, so none of it links the tensor library, and the
//! engine, libwaifu, does not link this: an app -- or the command line -- asks this for a model's
//! manifest and hands the engine its path. What anything outside this crate calls is the C API in
//! [`c_api`]; the Rust items are public for this crate's own tests and examples.

pub mod c_api;
pub mod config;
pub mod defaults;
pub mod fetch;
pub mod hub;
pub mod log;
pub mod suggested;
pub mod yaml;

use std::fmt;

/// A manifest, or the YAML it is written in, that does not hold what the format calls for.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FormatError(String);

impl FormatError {
    pub fn new(message: impl Into<String>) -> FormatError {
        FormatError(message.into())
    }
}

impl fmt::Display for FormatError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for FormatError {}
