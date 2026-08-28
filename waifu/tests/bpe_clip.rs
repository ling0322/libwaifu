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

//! The CLIP pre-tokenizer, which is what `simple_tokenizer.py` does before any merging: lowercase
//! the text, cut it into words, and give each word its own merge domain with a `</w>` marker on
//! its last byte.
//!
//! The vocabulary here is a stand-in, small enough to reason about. It says nothing about whether
//! a converted CLIP vocabulary agrees with the reference tokenizer, which is a separate question
//! and one only a diff against that tokenizer can answer.

use waifu::{BpeConfig, BpeEncoder, BpeModel, PreTokenizer};

/// One entry of the stand-in vocabulary: the bytes it stands for, and the rank of the merge that
/// produces it. A base symbol has no merge and no rank.
struct Entry {
    piece: &'static [u8],
    rank: Option<u32>,
}

fn base(piece: &'static [u8]) -> Entry {
    Entry { piece, rank: None }
}

fn merged(piece: &'static [u8], rank: u32) -> Entry {
    Entry {
        piece,
        rank: Some(rank),
    }
}

/// Writes the vocabulary in the format `BpeModel::from_reader` expects. Ranks become negated
/// weights, which is how `bpe_exporter.py` carries them, so the lowest rank is the best merge.
fn write_vocab(entries: &[Entry]) -> Vec<u8> {
    const MAGIC: i16 = 0x55aa;
    const FLAG_CONTROL: i8 = 2;

    let mut out = Vec::new();
    out.extend_from_slice(b"LLsp");
    out.extend_from_slice(&((entries.len() + 1) as i32).to_le_bytes());
    out.extend_from_slice(&MAGIC.to_le_bytes());

    for entry in entries {
        out.push(0u8);
        out.push(entry.piece.len() as u8);
        out.extend_from_slice(entry.piece);
        out.push(entry.piece.len() as u8);
        out.extend_from_slice(entry.piece);

        let weight = entry.rank.map(|rank| -(rank as f32)).unwrap_or(0.0);
        out.extend_from_slice(&weight.to_le_bytes());
    }

    // The end-of-text token, which is a control token and never merges with anything.
    let name = b"<|endoftext|>";
    out.push(FLAG_CONTROL as u8);
    out.push(0);
    out.push(name.len() as u8);
    out.extend_from_slice(name);
    out.extend_from_slice(&0.0f32.to_le_bytes());

    out.extend_from_slice(&MAGIC.to_le_bytes());
    out
}

/// Enough of a CLIP vocabulary to encode the handful of words these tests use. Every merge is one
/// its own rule would also find, so the vocabulary does not itself decide the answer.
fn vocab() -> Vec<Entry> {
    vec![
        base(b" "),
        base(b"a"),
        base(b"c"),
        base(b"s"),
        base(b"t"),
        base(b"2"),
        base(b"0"),
        base(b"!"),
        base(b"'"),
        base(b"a</w>"),
        base(b"c</w>"),
        base(b"s</w>"),
        base(b"t</w>"),
        base(b"2</w>"),
        base(b"0</w>"),
        base(b"!</w>"),
        base(b"'</w>"),
        merged(b"ca", 0),
        merged(b"at</w>", 1),
        merged(b"cat</w>", 2),
        merged(b"sat</w>", 3),
        merged(b"!!</w>", 4),
        merged(b"'s</w>", 5),
    ]
}

/// The tokens `text` encodes to, as the pieces they stand for, which reads better in a failure
/// than a row of ids does.
fn encode(text: &str) -> Vec<String> {
    let entries = vocab();
    let bytes = write_vocab(&entries);
    let model = BpeModel::from_reader(&mut &bytes[..]).unwrap();

    let config = BpeConfig {
        model_file: String::new(),
        add_prefix_space: false,
        split_by_unicode: false,
        pre_tokenizer: PreTokenizer::ClipWord,
    };

    let ids = BpeEncoder::new(&model, &config).encode(text);
    ids.iter()
        .map(|id| String::from_utf8_lossy(model.token_piece(*id).unwrap()).into_owned())
        .collect()
}

#[test]
fn merges_a_word_and_marks_where_it_ends() {
    // c, a and t</w> are the starting symbols; ("c","a") is the cheapest merge, then ("ca","t</w>").
    assert_eq!(encode("cat"), vec!["cat</w>"]);

    // The same word twice stays two words: nothing merges across the space, which is the whole
    // point of running each word on its own.
    assert_eq!(encode("cat cat"), vec!["cat</w>", "cat</w>"]);
}

#[test]
fn lowercases_and_ignores_how_much_whitespace_there_is() {
    assert_eq!(encode("CAT"), vec!["cat</w>"]);
    assert_eq!(encode("Cat"), vec!["cat</w>"]);
    assert_eq!(encode("  cat\t\ncat  "), vec!["cat</w>", "cat</w>"]);
}

#[test]
fn splits_the_way_the_pattern_does() {
    // A run of letters, a single digit at a time, and a run of punctuation.
    assert_eq!(encode("cat!!"), vec!["cat</w>", "!!</w>"]);
    assert_eq!(encode("20"), vec!["2</w>", "0</w>"]);
    assert_eq!(encode("cat 20"), vec!["cat</w>", "2</w>", "0</w>"]);

    // The contractions come before the character classes, so this is `cat` and `'s`, not `cat`
    // and `'` and `s`.
    assert_eq!(encode("cat's"), vec!["cat</w>", "'s</w>"]);
}

#[test]
fn a_word_that_does_not_merge_stays_as_its_bytes() {
    // `sat` has no merge for ("s","a"), so it takes ("a","t</w>") first and then ("s","at</w>").
    assert_eq!(encode("sat"), vec!["sat</w>"]);

    // A single letter is already a whole word.
    assert_eq!(encode("a"), vec!["a</w>"]);
}

#[test]
fn the_whole_text_mode_is_untouched() {
    // The sentencepiece behaviour still merges straight through a space, which is what a
    // vocabulary without a word-end marker needs.
    let entries = vocab();
    let bytes = write_vocab(&entries);
    let model = BpeModel::from_reader(&mut &bytes[..]).unwrap();

    let config = BpeConfig {
        model_file: String::new(),
        add_prefix_space: false,
        split_by_unicode: false,
        pre_tokenizer: PreTokenizer::Whole,
    };

    // No `</w>` anywhere, and the text is not lowercased on the way in.
    let ids = BpeEncoder::new(&model, &config).encode("ca");
    let pieces: Vec<String> = ids
        .iter()
        .map(|id| String::from_utf8_lossy(model.token_piece(*id).unwrap()).into_owned())
        .collect();
    assert_eq!(pieces, vec!["ca"]);
}
