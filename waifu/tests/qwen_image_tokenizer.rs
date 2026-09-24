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

//! Qwen-Image 2.1's tokenizer against the published one, token for token, and the template
//! around a prompt against the ids the reference ran on.
//!
//! Needs only the package's manifest and tokenizer and the test bundle, not its weights.

use std::collections::HashMap;
use std::io::Read;
use std::path::PathBuf;

use waifu::flint::Tensor;
use waifu::{read_safetensors, Manifest, Tokenizer};

fn models_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../models")
}

fn manifest() -> Manifest {
    Manifest::open(models_dir().join("qwen-image-2.1.yaml")).unwrap()
}

#[test]
#[ignore = "needs the qwen-image-2.1 package"]
fn the_tokenizer_matches_the_reference_token_for_token() {
    let tokenizer = Tokenizer::open(&manifest()).unwrap();

    let mut corpus = String::new();
    std::fs::File::open(models_dir().join("qwen-image-2.1_test_corpus.tsv"))
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

/// The template, spelled out here rather than borrowed from the pipeline, so that a change to it
/// that nothing else noticed lands here: the whole string is tokenized at once, and the system
/// turn it opens with is the number of states the pipeline drops.
#[test]
#[ignore = "needs the qwen-image-2.1 package"]
fn the_template_gives_the_ids_the_reference_ran_on() {
    let cases: HashMap<String, Tensor> =
        read_safetensors(&[models_dir().join("qwen-image-2.1_test.safetensors")]).unwrap();
    let expected: Vec<i32> = cases["test_case.input_ids"]
        .to_vec_i64()
        .unwrap()
        .iter()
        .map(|&id| id as i32)
        .collect();
    let dropped = cases["test_case.drop"].to_vec_i64().unwrap()[0] as usize;

    let system = "<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n";
    let prompt = "a red fox sitting in fresh snow, golden hour, photorealistic";
    let text = format!("{system}<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n");

    let tokenizer = Tokenizer::open(&manifest()).unwrap();
    assert_eq!(
        tokenizer.encode(system).unwrap().len(),
        dropped,
        "the system turn is not the number of states the reference drops"
    );
    assert_eq!(tokenizer.encode(&text).unwrap(), expected);
}
