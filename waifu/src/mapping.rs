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

//! A named block of settings, and asking it for one.
//!
//! This is what a model's configuration is read out of. It holds the values as they were written
//! and parses on the way out, because what a setting means is the model's to say and not the
//! file's: `320,640,1280` is three numbers to the U-Net and one string to anything else, and
//! `28` is an `i32` here and a `usize` there.
//!
//! Two readers produce these. A [`Manifest`](crate::Manifest) is where a model's own settings
//! come from, one mapping per block under its `config:` key; [`IniConfig`](crate::IniConfig) is
//! the `tokenizer.ini` still carried inside a package. They are different files in different
//! formats, and a block of named values is the whole of what they have in common -- which is why
//! that block is here rather than in either of them.
//!
//! Asking for a key that is missing, or for one that does not parse as what it is being read as,
//! is an error rather than a default: a model built on a guessed hyperparameter fails later and
//! less clearly than one that refused to be built.

use std::collections::BTreeMap;
use std::str::FromStr;

use crate::error::{Error, Result};
use crate::flint::WeightFormat;

/// One named block of `key: value` settings.
#[derive(Clone, Debug, Default)]
pub struct Mapping {
    name: String,
    entries: BTreeMap<String, String>,
}

impl Mapping {
    /// A block called `name`, holding `entries`.
    pub fn new(name: &str, entries: BTreeMap<String, String>) -> Mapping {
        Mapping {
            name: name.to_string(),
            entries,
        }
    }

    /// What the block is called, which is what its complaints name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Whether the block has a `key`.
    pub fn has(&self, key: &str) -> bool {
        self.entries.contains_key(key)
    }

    /// The value of `key` as written.
    pub fn get_str(&self, key: &str) -> Result<&str> {
        self.entries
            .get(key)
            .map(String::as_str)
            .ok_or_else(|| Error::format(format!("{}: key {key:?} not found", self.name)))
    }

    /// The value of `key`, parsed. Used for the numeric hyperparameters.
    pub fn get<T: FromStr>(&self, key: &str) -> Result<T> {
        let value = self.get_str(key)?;
        value.parse().map_err(|_| {
            Error::format(format!(
                "{}: key {key:?} holds {value:?}, which is not a {}",
                self.name,
                std::any::type_name::<T>()
            ))
        })
    }

    /// The value of `key`, parsed, or `default` when the key is absent. For the settings a model
    /// only writes down when it departs from the usual.
    pub fn get_or<T: FromStr>(&self, key: &str, default: T) -> Result<T> {
        if self.has(key) {
            self.get(key)
        } else {
            Ok(default)
        }
    }

    /// The value of `key` as a flag. Accepts the spellings the C++ reader accepts, which are also
    /// the ones YAML calls booleans.
    pub fn get_bool(&self, key: &str) -> Result<bool> {
        let value = self.get_str(key)?;
        match value.to_ascii_lowercase().as_str() {
            "1" | "true" | "yes" | "on" => Ok(true),
            "0" | "false" | "no" | "off" => Ok(false),
            other => Err(Error::format(format!(
                "{}: key {key:?} holds {other:?}, which is not a boolean",
                self.name
            ))),
        }
    }

    /// [`Mapping::get_bool`] with a default for an absent key.
    pub fn get_bool_or(&self, key: &str, default: bool) -> Result<bool> {
        if self.has(key) {
            self.get_bool(key)
        } else {
            Ok(default)
        }
    }

    /// How `key` says the package stored the matrices its model multiplies by, or `default` when
    /// it does not say -- which every package written so far does not.
    ///
    /// Spelled out here rather than left to `FromStr` so that a misspelling reads as one: the
    /// names are few and this can list them.
    pub fn get_weight_format_or(&self, key: &str, default: WeightFormat) -> Result<WeightFormat> {
        if !self.has(key) {
            return Ok(default);
        }

        let value = self.get_str(key)?;
        WeightFormat::from_name(value).ok_or_else(|| {
            Error::format(format!(
                "{}: key {key:?} holds {value:?}, which is neither \"float\" nor \"fp8\"",
                self.name
            ))
        })
    }
}
