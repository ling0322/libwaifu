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

//! The CLIP tokenizer against the real vocabulary, diffed token for token with what
//! `CLIPTokenizerFast` produced for the same texts.
//!
//! The reference is the fast tokenizer rather than the slow one, on both sides: the package
//! carries the `tokenizer.json` that `CLIPTokenizerFast` reads, and the exporter wrote this corpus
//! with that same tokenizer. The two upstream CLIP tokenizers do not agree with each other -- the
//! slow one runs ftfy over its input first, where no `tokenizer.json` says to -- so the choice is
//! which of them to be, and the fast one is the one diffusers loads.
//!
//! Which makes this a narrower test than it was, and still the one worth having: it says the
//! vocabulary in the package is the one the exporter was pointed at, and that ids come back out of
//! it unshifted. `docs/TODO.md` has the ftfy measurement.

use std::io::Read;
use std::path::PathBuf;

use waifu::{Manifest, ParamFile, Tokenizer};

const PROMPT: &str = "a photo of an astronaut riding a horse on mars";

fn models_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../models")
}

fn tokenizer() -> Tokenizer {
    let manifest = Manifest::open(models_dir().join("sdxl-base.yaml")).unwrap();
    Tokenizer::open(&manifest).unwrap()
}

fn test_cases() -> ParamFile {
    ParamFile::open(&[models_dir().join("sdxl-base_test.safetensors")]).unwrap()
}

#[test]
#[ignore = "needs the sdxl package"]
fn matches_the_reference_token_for_token() {
    let tokenizer = tokenizer();

    let mut corpus = String::new();
    std::fs::File::open(models_dir().join("sdxl-base_test_corpus.tsv"))
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
        checked > 1000,
        "the corpus is smaller than it should be: {checked}"
    );
    assert_eq!(mismatched, 0, "{mismatched} of {checked} texts disagree");
}

#[test]
#[ignore = "needs the sdxl package"]
fn wraps_a_prompt_the_way_the_text_encoder_is_fed() {
    // The ids the reference was computed from: the prompt between the two markers, padded out to
    // the context length. Only the middle of it is what encoding produces; the rest is what the
    // model layer has to add, and this says what that has to look like.
    let cases = test_cases();

    let reference: Vec<i32> = cases
        .get_unchecked("test_case.input_ids")
        .unwrap()
        .to_vec_i64()
        .unwrap()
        .iter()
        .map(|id| *id as i32)
        .collect();

    assert_eq!(reference.len(), 77, "the context length is fixed at 77");
    assert_eq!(reference[0], 49406, "it starts with <|startoftext|>");

    let body = tokenizer().encode(PROMPT).unwrap();
    assert_eq!(
        reference[1..=body.len()],
        body[..],
        "the prompt is the middle of it"
    );
    assert_eq!(reference[body.len() + 1], 49407, "then <|endoftext|>");

    // CLIP-L pads with the end marker, which is why the padding is not a token of its own. The
    // second encoder pads with 0 instead, which is why the package carries both.
    assert!(
        reference[body.len() + 2..].iter().all(|id| *id == 49407),
        "the rest is padding: {reference:?}"
    );
}
