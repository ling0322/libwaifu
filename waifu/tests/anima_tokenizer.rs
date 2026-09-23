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

//! Anima's two tokenizers against the real vocabularies, diffed token for token with what the
//! published tokenizers produced for the same texts.
//!
//! Two of them, because Anima tokenizes the prompt twice: Qwen3's byte level BPE for the text
//! encoder, and T5's sentencepiece unigram for the query stream the adapter embeds. They are not
//! two spellings of one thing -- the unigram one segments by the highest scoring path where the
//! BPE one merges pairs -- and what says so here is that each package entry is the
//! `tokenizer.json` its authors published, which names its own algorithm.
//!
//! Which makes this a narrower test than it was, and still the one worth having. It no longer asks
//! whether an encoder written here walks a vocabulary the way the reference does; it asks whether
//! the two files in the package are the two the exporter was pointed at, and whether the ids come
//! back out of them unshifted. A package built from the wrong revision fails it.

use std::collections::HashMap;
use std::io::Read;
use std::path::PathBuf;

use waifu::flint::Tensor;
use waifu::{read_safetensors, Anima, Manifest, Tokenizer};

const PROMPT: &str = "masterpiece, best quality, 1girl, solo, long hair, brown eyes, school \
                      uniform, smile";

fn models_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../models")
}

fn manifest() -> Manifest {
    Manifest::open(models_dir().join("anima-turbo-v11.yaml")).unwrap()
}

fn test_cases() -> HashMap<String, Tensor> {
    read_safetensors(&[models_dir().join("anima-turbo-v11_test.safetensors")]).unwrap()
}

/// One tokenizer against the corpus the exporter wrote for it.
fn check_corpus(tokenizer: &Tokenizer, beside: &str) {
    let mut corpus = String::new();
    std::fs::File::open(models_dir().join(beside))
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

#[test]
#[ignore = "needs the anima package"]
fn the_text_encoder_tokenizer_matches_the_reference_token_for_token() {
    let tokenizer =
        Tokenizer::open_section(&manifest(), Anima::QWEN_TOKENIZER_SECTION).unwrap();
    check_corpus(&tokenizer, "anima-turbo-v11_test_qwen3_corpus.tsv");
}

#[test]
#[ignore = "needs the anima package"]
fn the_adapter_tokenizer_matches_the_reference_token_for_token() {
    let tokenizer =
        Tokenizer::open_section(&manifest(), Anima::T5_TOKENIZER_SECTION).unwrap();
    check_corpus(&tokenizer, "anima-turbo-v11_test_t5_corpus.tsv");
}

/// The ids the reference outputs were computed from, which is the join between what the tokenizers
/// do and what the weights do. The T5 side carries the end marker and the Qwen3 side carries
/// nothing, and that asymmetry is the model's rather than a slip: `t5_ids` is what the adapter
/// embeds, and Qwen3 is a base model that was never trained to see a marker on a bare prompt.
#[test]
#[ignore = "needs the anima package"]
fn both_tokenizers_give_the_ids_the_reference_ran_on() {
    let cases = test_cases();

    let reference = |name: &str| -> Vec<i32> {
        cases[name].clone()
            .to_vec_i64()
            .unwrap()
            .iter()
            .map(|id| *id as i32)
            .collect()
    };

    let qwen = Tokenizer::open_section(&manifest(), Anima::QWEN_TOKENIZER_SECTION).unwrap();
    assert_eq!(qwen.encode(PROMPT).unwrap(), reference("test_case.qwen_ids"));

    let t5 = Tokenizer::open_section(&manifest(), Anima::T5_TOKENIZER_SECTION).unwrap();
    let mut ids = t5.encode(PROMPT).unwrap();
    ids.push(t5.token_to_id("</s>").unwrap());
    assert_eq!(ids, reference("test_case.t5_ids"));
}
