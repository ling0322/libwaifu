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
//! `CLIPTokenizer` produced for the same texts.
//!
//! `bpe_clip.rs` checks the pre-tokenizer's mechanics on a vocabulary small enough to reason
//! about. This is the check that matters: 49408 real tokens and their merge ranks, which is the
//! only thing that can say whether the merge rule libwaifu uses -- the rank of the token a pair
//! joins into -- picks what CLIP's rule, the rank of the pair itself, would have picked. It does,
//! over every line of the corpus.
//!
//! The corpus is stored as ftfy left it, because CLIPTokenizer runs its input through ftfy before
//! matching its pattern and libwaifu does not. What is compared is therefore the merging, which is
//! the part libwaifu implements. `docs/TODO.md` has what that leaves out.

use std::io::Read;
use std::path::PathBuf;

use waifu::{DType, Device, Tokenizer, VarBuilder, ZipFile};

const PROMPT: &str = "a photo of an astronaut riding a horse on mars";

fn models_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../models")
}

fn tokenizer() -> Tokenizer {
    let package = ZipFile::open(models_dir().join("sdxl-base.llmpkg")).unwrap();
    Tokenizer::from_package(&package).unwrap()
}

fn test_package() -> ZipFile {
    ZipFile::open(models_dir().join("sdxl-base_test.llmpkg")).unwrap()
}

#[test]
#[ignore = "needs the sdxl package"]
fn matches_the_reference_token_for_token() {
    let tokenizer = tokenizer();

    let mut corpus = String::new();
    test_package()
        .open_entry("tokenizer_corpus.tsv")
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

        let actual = tokenizer.encode(text);
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
    let cases = VarBuilder::from_reader(
        &mut test_package().open_entry("test_case.bin").unwrap(),
        Device::Cpu,
        DType::Float,
    )
    .unwrap();

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

    let body = tokenizer().encode(PROMPT);
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
