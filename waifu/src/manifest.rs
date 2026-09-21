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

//! What a model is, as the one file that says so: `<model-id>.yaml`.
//!
//! A model used to be one archive of this project's own: a zip holding the weights, a `model.ini`
//! saying what they were, a `metadata.json`, and the tokenizer. Finding out what a model was cost
//! opening seven gigabytes, none of it could be read by anything but this, and a model written as
//! four files had one of them silently more important than the rest.
//!
//! A model is a small text file now, and the things it names are files anything can read:
//!
//! ```yaml
//! weights:
//!   - sdxl-base-00001-of-00002.safetensors
//!   - sdxl-base-00002-of-00002.safetensors
//!
//! tokenizers:
//!   tokenizer: sdxl-base.tokenizer.json
//!
//! config:
//!   model:
//!     type: sdxl
//!   sdxl:
//!     latent_channels: 4
//!
//! suggested:
//!   prompt: masterpiece, best quality
//!   sizes:
//!     - [1024, 1024]
//!     - [832, 1216]
//!   steps: 8
//!   guidance: 1.0
//! ```
//!
//! `weights` names the safetensors files, in the order they are to be read. `tokenizers` names a
//! `tokenizer.json` for each way this model turns text into ids -- one for most, two for Anima,
//! which reads its prompt through a Qwen3 vocabulary and a T5 one. `config` is what `model.ini`
//! said, block for block and key for key, with nothing renamed on the way across. [`Suggestions`]
//! is what `metadata.json` said, grown from one field to several.
//!
//! A tokenizer is one line because a tokenizer is one file: everything about how a model's text
//! becomes ids -- the vocabulary, the merge ranks, the normalizer, the pre-tokenizer -- is inside
//! the `tokenizer.json` its authors published, and the `tokenizers` crate reads it. There used to
//! be a block of flags here saying which algorithm and which whitespace rules to read a vocabulary
//! of our own back with, and every one of those flags was a claim that could be wrong.
//!
//! Every file is named, never globbed, and a name has to be a neighbour -- one path component, no
//! `..` -- because a manifest is a thing that gets downloaded and these names decide which files
//! are opened afterwards.
//!
//! Each block under `config:` becomes a [`Mapping`], which is where a model asks for its settings.
//! `latent_channels: 4` is the text `"4"` until the U-Net reads it as a number, because what a
//! setting means is the model's to say and not this file's -- the same `320,640,1280` is three
//! numbers to one reader and a string to the next. See [`crate::yaml`] for what of the language is
//! read and what is refused.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use crate::error::{Error, Result};
use crate::mapping::Mapping;
use crate::param_file::ParamFile;
use crate::suggested::Suggestions;
use crate::yaml::{self, Node};

/// What a model's manifest says.
#[derive(Clone, Debug, Default)]
pub struct Manifest {
    path: PathBuf,
    weights: Vec<String>,
    tokenizers: BTreeMap<String, String>,
    config: BTreeMap<String, Mapping>,
    suggested: Suggestions,
    explicit: bool,
}

impl Manifest {
    /// What a manifest file is called, after the model's id.
    pub const SUFFIX: &'static str = ".yaml";

    /// The blocks a manifest holds.
    pub const WEIGHTS: &'static str = "weights";
    pub const TOKENIZERS: &'static str = "tokenizers";
    pub const CONFIG: &'static str = "config";
    pub const SUGGESTED: &'static str = "suggested";
    /// Spelled after the tag the model cards carry, which is where the answer comes from -- and
    /// which is also the spelling the hubs will keep: ModelScope's moderation reverts a commit
    /// that says the same thing in plainer words, silently, reporting the upload as committed and
    /// rolling it back afterwards.
    pub const EXPLICIT: &'static str = "not_for_all_audiences";

    /// What a weights file is called.
    pub const WEIGHTS_SUFFIX: &'static str = ".safetensors";

    /// Read the manifest at `path`.
    ///
    /// Where it is matters as much as what is in it: the files it names are its neighbours, so the
    /// path is kept and [`Manifest::params`] reads them from beside it.
    pub fn open(path: impl AsRef<Path>) -> Result<Manifest> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path)
            .map_err(|error| Error::format(format!("{}: {error}", path.display())))?;

        let mut manifest = Manifest::parse(&text)
            .map_err(|error| Error::format(format!("{}: {error}", path.display())))?;
        manifest.path = path.to_path_buf();
        Ok(manifest)
    }

    /// The manifest a file holds, without yet saying which file that was.
    pub fn parse(text: &str) -> Result<Manifest> {
        let mut top = yaml::parse(text)?.into_map("the manifest")?;

        let Some(listed) = top.remove(Self::WEIGHTS) else {
            return Err(Error::format(format!(
                "there is no {:?} list, so this manifest names no weights",
                Self::WEIGHTS
            )));
        };
        let mut weights = Vec::new();
        for item in listed.into_seq(Self::WEIGHTS)? {
            weights.push(item.into_scalar(Self::WEIGHTS)?);
        }
        if weights.is_empty() {
            return Err(Error::format(format!(
                "{:?} is empty, so this model has no parameters to read",
                Self::WEIGHTS
            )));
        }

        // A model that tokenizes nothing is allowed: the block is what a model with a prompt
        // needs, and nothing here is in a position to say whether this one has one.
        let tokenizers = match top.remove(Self::TOKENIZERS) {
            Some(node) => {
                let mut named = BTreeMap::new();
                for (name, file) in node.into_map(Self::TOKENIZERS)? {
                    let file = file.into_scalar(&name)?;
                    named.insert(name, file);
                }
                named
            }
            None => BTreeMap::new(),
        };

        let Some(described) = top.remove(Self::CONFIG) else {
            return Err(Error::format(format!(
                "there is no {:?}, so this manifest does not say what the model is",
                Self::CONFIG
            )));
        };
        let mut config = BTreeMap::new();
        for (name, body) in described.into_map(Self::CONFIG)? {
            let block = Mapping::new(&name, settings(body, &name)?);
            config.insert(name, block);
        }

        // The one block that may be missing altogether. A model with no advice to give has none,
        // and that is not an absence worth complaining about: it is what every model had before
        // there was anywhere to say it.
        let suggested = match top.remove(Self::SUGGESTED) {
            Some(node) => Suggestions::from_fields(&node.into_map(Self::SUGGESTED)?),
            None => Suggestions::default(),
        };

        // What the model draws, which the catalogue also knows for a published one -- this is the
        // answer for a manifest handed over by path, which is in no catalogue at all. Missing
        // reads as false: every manifest written before there was anywhere to say this says
        // nothing, and a model nobody has labelled is not a model labelled yes.
        let explicit = top
            .remove(Self::EXPLICIT)
            .and_then(|node| node.as_str().map(|said| said.trim() == "true"))
            .unwrap_or(false);

        // A block this build has not heard of is stepped over rather than refused, the way an
        // unknown suggestion always was: a manifest from a newer writer still names the weights
        // and still says what the model is.
        Ok(Manifest {
            path: PathBuf::new(),
            weights,
            tokenizers,
            config,
            suggested,
            explicit,
        })
    }

    /// Where the manifest was read from, which is what the names in it are relative to.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// The model id, which is what the manifest is called with the suffix off.
    pub fn id(&self) -> &str {
        self.path
            .file_stem()
            .and_then(|stem| stem.to_str())
            .unwrap_or_default()
    }

    /// The weights files, in the order they are to be read.
    pub fn weights(&self) -> &[String] {
        &self.weights
    }

    /// The `tokenizer.json` the model calls `name`.
    ///
    /// A model with one calls it `"tokenizer"`; one that tokenizes its prompt more than once names
    /// each of them, and there is nothing to make one of those the default.
    pub fn tokenizer(&self, name: &str) -> Result<&str> {
        self.tokenizers.get(name).map(String::as_str).ok_or_else(|| {
            Error::format(match self.tokenizers.is_empty() {
                true => format!("this model names no tokenizers, and {name:?} would be one"),
                false => format!(
                    "this model has no tokenizer called {name:?}; it has {}",
                    self.tokenizers
                        .keys()
                        .map(String::as_str)
                        .collect::<Vec<_>>()
                        .join(", ")
                ),
            })
        })
    }

    /// The block of settings called `name`, which is where a model reads its own.
    ///
    /// `[model]` of the old `model.ini` is `config: model:` here and is still asked for as
    /// `"model"`, and so is `[tokenizer]` of the old `tokenizer.ini`: what moved is the file, not
    /// the names in it.
    pub fn section(&self, name: &str) -> Result<&Mapping> {
        self.config.get(name).ok_or_else(|| {
            Error::format(format!(
                "this model says nothing under {:?}, and that is where {name:?} would be",
                Self::CONFIG
            ))
        })
    }

    /// Whether the manifest says anything under `name`.
    pub fn has_section(&self, name: &str) -> bool {
        self.config.contains_key(name)
    }

    /// What the model suggests being asked for, which for an older model is nothing.
    pub fn suggested(&self) -> &Suggestions {
        &self.suggested
    }

    /// Whether what it was trained on means it draws explicit pictures readily, asked for them or
    /// not. False for a manifest that does not say, which is every manifest written before there
    /// was anywhere to say it.
    pub fn explicit(&self) -> bool {
        self.explicit
    }

    /// Where each weights file is, in the order the manifest names them.
    ///
    /// The names are checked by [`Manifest::file`], so this is also the answer to "may this model
    /// name these files at all" -- which is what a fetch wants to know before it has any of them
    /// on disk to open.
    pub fn weight_paths(&self) -> Result<Vec<PathBuf>> {
        self.weights.iter().map(|name| self.file(name)).collect()
    }

    /// Every tensor of the model, read onto the host.
    ///
    /// The files are read into one namespace, so which of them a tensor was written to is not
    /// something a model has to know.
    pub fn params(&self) -> Result<ParamFile> {
        ParamFile::open(&self.weight_paths()?)
    }

    /// Every file this model is made of, beside the manifest itself.
    ///
    /// The weights, and each tokenizer's `tokenizer.json`. This is what a fetch goes and gets,
    /// which is why it is worked out in one place: a list written down a second time in the
    /// downloader is a list that comes to disagree with the model.
    ///
    /// The manifest is not in it. Whoever is reading this already has that.
    pub fn files(&self) -> Vec<String> {
        let mut files = self.weights.clone();

        for named in self.tokenizers.values() {
            // Two tokenizers of one model may name one file, and a file is fetched once.
            if !files.iter().any(|file| file == named) {
                files.push(named.clone());
            }
        }
        files
    }

    /// Where a file the manifest names is, having checked that it may name it.
    ///
    /// A manifest is a file that gets downloaded, and the names in it decide what is opened on the
    /// far side of that. So a name is one component of a path and nothing else: not `..`, not a
    /// directory, not an absolute path to somewhere else on the machine. Used for the weights, and
    /// for the vocabulary a tokenizer block names.
    pub fn file(&self, name: &str) -> Result<PathBuf> {
        if name.is_empty()
            || name == "."
            || name == ".."
            || name.contains('/')
            || name.contains('\\')
            || Path::new(name).components().count() != 1
        {
            return Err(Error::format(format!(
                "this model names {name:?}, and a manifest may only name a neighbour of its own"
            )));
        }

        Ok(self.path.parent().unwrap_or(Path::new(".")).join(name))
    }
}

/// One block of `config:`, as the settings it holds.
///
/// A value written as a list -- `unet_block_out_channels: [320, 640, 1280]` -- comes out as its
/// items joined by commas, which is how every list in a model configuration has always been
/// written and how the model that reads one still splits it. A list here is a spelling of that
/// rather than a second kind of value, so there is still exactly one place that decides what a
/// setting means, and it is the model.
fn settings(node: Node, what: &str) -> Result<BTreeMap<String, String>> {
    let mut entries = BTreeMap::new();
    for (key, value) in node.into_map(what)? {
        let text = match value {
            Node::Seq(items) => {
                let mut parts = Vec::new();
                for item in items {
                    parts.push(item.into_scalar(&key)?);
                }
                parts.join(",")
            }
            other => other.into_scalar(&key)?,
        };
        entries.insert(key, text);
    }
    Ok(entries)
}
