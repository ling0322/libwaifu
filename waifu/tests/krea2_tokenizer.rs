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


//! Krea 2's tokenizer against the real vocabulary, diffed token for token with what the published
//! tokenizer produced for the same texts.
//!
//! One vocabulary, unlike Anima's two: Qwen2's byte level BPE, which is the encoder's own. What
//! this asks is whether the file in the package is the one the exporter was pointed at and
//! whether the ids come back out of it unshifted -- a package built from the wrong revision fails
//! it.
//!
//! It also asks the one thing about this model's prompting that no reference tensor covers: that
//! the template around the prompt tokenizes here into the same ids the reference ran, special
//! tokens and all. `<|im_start|>` is a token and not five, and a vocabulary that read it as text
//! would produce a prompt the model was never conditioned on and would otherwise look fine.

use std::collections::HashMap;
use std::io::Read;
use std::path::PathBuf;

use waifu::flint::Tensor;
use waifu::{read_safetensors, Manifest, Tokenizer};

fn models_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../models")
}

fn manifest() -> Manifest {
    Manifest::open(models_dir().join("krea2-turbo.yaml")).unwrap()
}

fn test_cases() -> HashMap<String, Tensor> {
    read_safetensors(&[models_dir().join("krea2-turbo_test.safetensors")]).unwrap()
}

#[test]
#[ignore = "needs the krea2 package"]
fn the_tokenizer_matches_the_reference_token_for_token() {
    let tokenizer = Tokenizer::open(&manifest()).unwrap();

    let mut corpus = String::new();
    std::fs::File::open(models_dir().join("krea2-turbo_test_corpus.tsv"))
        .unwrap()
        .read_to_string(&mut corpus)
        .unwrap();

    let mut checked = 0;
    let mut mismatched = 0;
    for line in corpus.lines() {
        let (text, ids) = line
            .split_once('\t')
            .expect("a corpus line is text then its ids");
        let expected: Vec<i32> = ids
            .split_whitespace()
            .map(|id| id.parse().expect("an id is a number"))
            .collect();

        let actual = tokenizer.encode(text).unwrap();
        checked += 1;
        if actual != expected {
            mismatched += 1;
            if mismatched <= 8 {
                println!("mismatch on {text:?}\n  actual   {actual:?}\n  expected {expected:?}");
            }
        }
    }

    assert!(
        checked > 500,
        "the corpus is smaller than it should be: {checked}"
    );
    assert_eq!(mismatched, 0, "{mismatched} of {checked} texts disagree");
}

/// The ids the reference outputs were computed from, which is the join between what the tokenizer
/// does and what the weights do.
///
/// The prompt is not tokenized on its own here and is not on its own in the reference either: the
/// template's opening is part of the same string, because a byte level vocabulary does not promise
/// that the pieces of a text tokenize like the text.
#[test]
#[ignore = "needs the krea2 package"]
fn the_template_gives_the_ids_the_reference_ran_on() {
    let cases = test_cases();
    let expected: Vec<i32> = cases["test_case.input_ids"].to_vec_i64().unwrap()
        .iter()
        .map(|&id| id as i32)
        .collect();

    // The same two constants the pipeline wraps a prompt in, which are private to it -- so this
    // spells them out, and a change to either that nothing else noticed lands here.
    let prefix = "<|im_start|>system\nDescribe the image by detailing the color, shape, size, \
                  texture, quantity, text, spatial relationships of the objects and \
                  background:<|im_end|>\n<|im_start|>user\n";
    let suffix = "<|im_end|>\n<|im_start|>assistant\n";
    let prompt = "a red fox sitting in fresh snow, golden hour, photorealistic";

    let tokenizer = Tokenizer::open(&manifest()).unwrap();
    let opening = tokenizer.encode(prefix).unwrap();
    let closing = tokenizer.encode(suffix).unwrap();

    // Thirty-four and five. Neither is a decoration: the runtime drops exactly the first
    // thirty-four states, and the five at the end are read by the denoiser like any other token.
    // A vocabulary that read `<|im_start|>` as text rather than as one token would land here,
    // and would otherwise draw a picture of a prompt the model was never conditioned on.
    assert_eq!(opening.len(), 34, "the template's opening is not 34 tokens");
    assert_eq!(closing.len(), 5, "the template's closing is not 5 tokens");

    let mut actual = tokenizer.encode(&format!("{prefix}{prompt}")).unwrap();
    assert!(
        actual.starts_with(&opening),
        "the prompt changed how the opening in front of it tokenizes"
    );
    actual.extend(closing);

    assert_eq!(actual, expected);
}
