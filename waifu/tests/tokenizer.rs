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

//! Reading a model's tokenizer, on a vocabulary small enough to write down.
//!
//! What the model suites check is that the file a real model names is the one the exporter was
//! pointed at. What this checks is the part that has nothing to do with any particular model: that
//! the manifest's `tokenizers:` line is read, that the file it names is handed to the `tokenizers`
//! crate whole, and that the two things the runtime asks of that crate are the two it is
//! documented to ask -- the file's normalizer applied, the file's post-processor not.
//!
//! Those two are worth pinning rather than assuming. The normalizer is the thing the encoders
//! written here never had, and getting it for free is most of why the package carries a published
//! `tokenizer.json` at all; the post-processor is the thing they never had *either*, and here it
//! has to stay off, because every pipeline in this repository puts its own markers on.

use waifu::{Manifest, Tokenizer};

/// A whole tokenizer, in the format `tokenizers` reads.
///
/// NFKC normalizes, among much else, a full width letter to its ASCII one -- so `ｂ` has no
/// vocabulary entry of its own and still encodes, which is the normalizer being applied and not
/// merely stored. The post-processor would put `</s>` after the text, and must not.
const TOKENIZER_JSON: &str = r#"{
  "version": "1.0",
  "truncation": null,
  "padding": null,
  "added_tokens": [],
  "normalizer": {"type": "NFKC"},
  "pre_tokenizer": {"type": "Whitespace"},
  "post_processor": {
    "type": "TemplateProcessing",
    "single": [
      {"Sequence": {"id": "A", "type_id": 0}},
      {"SpecialToken": {"id": "</s>", "type_id": 0}}
    ],
    "pair": [{"Sequence": {"id": "A", "type_id": 0}}],
    "special_tokens": {
      "</s>": {"id": "</s>", "ids": [3], "tokens": ["</s>"]}
    }
  },
  "decoder": null,
  "model": {
    "type": "WordLevel",
    "vocab": {"a": 0, "b": 1, "[UNK]": 2, "</s>": 3},
    "unk_token": "[UNK]"
  }
}"#;

/// A model naming one tokenizer, under the name a single-tokenizer model uses.
///
/// The weights are named and never written: nothing here reads them, and a manifest that named
/// none would be refused before the tokenizer was reached.
fn one_tokenizer(dir: &std::path::Path) -> Manifest {
    std::fs::write(dir.join("one.tokenizer.json"), TOKENIZER_JSON).unwrap();

    let path = dir.join("one.yaml");
    std::fs::write(
        &path,
        "weights:\n  - one.safetensors\n\
         tokenizers:\n  tokenizer: one.tokenizer.json\n\
         config:\n  model:\n    type: sdxl\n",
    )
    .unwrap();

    Manifest::open(&path).unwrap()
}

fn temp_dir(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("waifu-tokenizer-{name}"));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

#[test]
fn reads_the_file_the_manifest_names() {
    let dir = temp_dir("reads");
    let tokenizer = Tokenizer::open(&one_tokenizer(&dir)).unwrap();

    assert_eq!(tokenizer.encode("a b").unwrap(), vec![0, 1]);
}

#[test]
fn applies_the_normalizer_in_the_file() {
    let dir = temp_dir("normalizer");
    let tokenizer = Tokenizer::open(&one_tokenizer(&dir)).unwrap();

    // The full width letter is not in the vocabulary. NFKC is what makes it the one that is, and
    // without the normalizer this would be the unknown token instead.
    assert_eq!(tokenizer.encode("ｂ").unwrap(), vec![1]);
    assert_eq!(tokenizer.encode("?").unwrap(), vec![2], "the unknown token");
}

#[test]
fn leaves_the_markers_to_the_caller() {
    let dir = temp_dir("markers");
    let tokenizer = Tokenizer::open(&one_tokenizer(&dir)).unwrap();

    // The post-processor in the file appends `</s>`. Encoding does not, because where a marker
    // goes is the pipeline's to say -- SDXL writes two into a padded window at fixed positions,
    // and Anima puts T5's on one of its two id streams and nothing on the other.
    assert_eq!(tokenizer.encode("a").unwrap(), vec![0]);

    // It is still reachable, which is how a pipeline puts it on.
    assert_eq!(tokenizer.token_to_id("</s>").unwrap(), 3);
}

#[test]
fn a_token_the_vocabulary_does_not_have_is_an_error() {
    let dir = temp_dir("missing-token");
    let tokenizer = Tokenizer::open(&one_tokenizer(&dir)).unwrap();

    // Not the unknown token: a pipeline naming a marker this vocabulary does not have is a
    // mistake, and one that would otherwise be read as part of the prompt.
    assert!(tokenizer.token_to_id("<|endoftext|>").is_err());
}

#[test]
fn a_model_can_carry_more_than_one() {
    let dir = temp_dir("two");
    std::fs::write(dir.join("two.qwen3.tokenizer.json"), TOKENIZER_JSON).unwrap();
    std::fs::write(dir.join("two.t5.tokenizer.json"), TOKENIZER_JSON).unwrap();

    let path = dir.join("two.yaml");
    std::fs::write(
        &path,
        "weights:\n  - two.safetensors\n\
         tokenizers:\n  \
         qwen3: two.qwen3.tokenizer.json\n  \
         t5: two.t5.tokenizer.json\n\
         config:\n  model:\n    type: anima\n",
    )
    .unwrap();
    let manifest = Manifest::open(&path).unwrap();

    // Anima's shape: two blocks, neither of them the default, because there is nothing to make
    // one of the two the one a reader should assume.
    assert!(Tokenizer::open_section(&manifest, "qwen3").is_ok());
    assert!(Tokenizer::open_section(&manifest, "t5").is_ok());
    assert!(Tokenizer::open(&manifest).is_err(), "no default");

    // And both of them are files the model is made of, so a fetch goes and gets them.
    assert_eq!(
        manifest.files(),
        [
            "two.safetensors",
            "two.qwen3.tokenizer.json",
            "two.t5.tokenizer.json"
        ]
    );
}

#[test]
fn a_file_that_is_not_a_tokenizer_is_an_error() {
    let dir = temp_dir("not-a-tokenizer");
    std::fs::write(dir.join("one.tokenizer.json"), b"this is not json").unwrap();

    let path = dir.join("bad.yaml");
    std::fs::write(
        &path,
        "weights:\n  - bad.safetensors\n\
         tokenizers:\n  tokenizer: one.tokenizer.json\n\
         config:\n  model:\n    type: sdxl\n",
    )
    .unwrap();

    let error = Tokenizer::open(&Manifest::open(&path).unwrap()).unwrap_err();
    assert!(
        error.to_string().contains("tokenizer.json"),
        "the error says which entry: {error}"
    );
}
