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

//! The tokenizer a model carries, which turns text into the ids it reads.
//!
//! A model keeps the `tokenizer.json` its authors published beside its manifest, byte for byte,
//! and this reads it with the `tokenizers` crate -- the same implementation the reference runs. That is the
//! whole of it: the vocabulary, the merge ranks, the normalizer and the pre-tokenizer are all in
//! that file, and the algorithm to apply them is named there too, so nothing here has to decide
//! whether a vocabulary is byte level BPE or a sentencepiece unigram, or carry a flag saying which
//! spaces it eats.
//!
//! This replaced a BPE and a unigram encoder written here, over a vocabulary the exporter rebuilt
//! into a format of its own. Those agreed with the reference on the texts they were measured
//! against and disagreed on the ones they were not: the normalization each tokenizer does first --
//! NFC for CLIP and Qwen3, NFKC for T5 -- was never implemented, so a prompt with a full width
//! comma in it tokenized differently here than it did where the weights were trained. That is
//! closed by not having the question: the normalizer is in the file and this runs it.
//!
//! What it does not close is ftfy. The *slow* `CLIPTokenizer` runs its input through `ftfy.fix_text`
//! when ftfy is installed beside it, and no `tokenizer.json` describes that, so neither this nor
//! `CLIPTokenizerFast` does it. The two upstream tokenizers disagree with each other there, which
//! makes it upstream's difference rather than one of ours -- and the fast one is what diffusers
//! loads, so matching it is matching the pipeline these weights are run by. `docs/TODO.md` has the
//! measurement.

use std::rc::Rc;

use crate::error::{Error, Result};
use crate::manifest::Manifest;

/// A tokenizer, over the vocabulary a model carries.
///
/// Cloning shares it, which is worth having: a model and the requests running against it all
/// encode with the same one.
#[derive(Clone, Debug)]
pub struct Tokenizer {
    inner: Rc<tokenizers::Tokenizer>,
}

impl Tokenizer {
    /// What a model with one tokenizer calls it.
    pub const SECTION: &'static str = "tokenizer";

    /// Read the tokenizer `manifest` describes.
    pub fn open(manifest: &Manifest) -> Result<Tokenizer> {
        Self::open_section(manifest, Self::SECTION)
    }

    /// Read one named tokenizer of `manifest`.
    ///
    /// A model with one calls it [`Tokenizer::SECTION`] and is read by [`Tokenizer::open`]. A
    /// model that tokenizes its prompt more than once -- Anima runs it through a Qwen3 vocabulary
    /// and a T5 one, and the two answers do different jobs -- names each of them instead, and
    /// there is nothing to make one of the two the default.
    pub fn open_section(manifest: &Manifest, name: &str) -> Result<Tokenizer> {
        let named = manifest.tokenizer(name)?.to_string();

        // Beside the manifest, checked the way every name in a manifest is checked: one component
        // of a path and nothing else.
        let path = manifest.file(&named)?;
        let bytes = std::fs::read(&path)
            .map_err(|error| Error::format(format!("{}: {error}", path.display())))?;

        let inner = tokenizers::Tokenizer::from_bytes(bytes)
            .map_err(|err| Error::model(format!("cannot read tokenizer {named:?}: {err}")))?;

        Ok(Tokenizer {
            inner: Rc::new(inner),
        })
    }

    /// The ids of `text`, and nothing else.
    ///
    /// The markers a model wants around its prompt are the caller's: SDXL writes its own two in at
    /// fixed positions of a padded window, and Anima puts T5's end marker on one of its two id
    /// streams and nothing on the other. So the post-processor in `tokenizer.json`, which is what
    /// would add them, is asked to stay out of it.
    pub fn encode(&self, text: &str) -> Result<Vec<i32>> {
        let encoded = self
            .inner
            .encode(text, false)
            .map_err(|err| Error::model(format!("cannot encode {text:?}: {err}")))?;

        Ok(encoded.get_ids().iter().map(|&id| id as i32).collect())
    }

    /// The id of a token, by the text that names it.
    ///
    /// A missing one is an error rather than the unknown token: a pipeline naming a marker the
    /// vocabulary does not have is a mistake, and one that would otherwise be read as a prompt.
    pub fn token_to_id(&self, token: &str) -> Result<i32> {
        self.inner
            .token_to_id(token)
            .map(|id| id as i32)
            .ok_or_else(|| Error::model(format!("token {token:?} is not in the vocabulary")))
    }
}
