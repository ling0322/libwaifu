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

//! CosyVoice3's text frontend: `CosyVoiceFrontEnd.text_normalize` and the functions in
//! `frontend_utils.py` it calls, in the order it calls them.
//!
//! Upstream normalizes numbers with WeTextProcessing when it is installed, and so does this --
//! through [`crate::indextts::normalize`], which is the same grammar's behaviour written as rules,
//! with the differences that module lists. Everything else is ported as written: a Chinese text
//! loses its blanks between Chinese characters, its superscripts, its full-width brackets and its
//! trailing commas, and has its `.` made `。`; either language is then cut at sentence punctuation
//! and packed into pieces of 60 to 80 -- characters for Chinese, tokens for English.
//!
//! A text holding `<|` and `|>` -- one that carries its own markers -- is not touched, as upstream
//! does not touch it.

use crate::indextts::normalize::{normalize, Language};
use crate::Result;

const TOKEN_MAX: usize = 80;
const TOKEN_MIN: usize = 60;
const MERGE: usize = 20;

pub fn contains_chinese(text: &str) -> bool {
    text.chars().any(|c| ('\u{4e00}'..='\u{9fff}').contains(&c))
}

/// `replace_blank`: a space survives only between two ASCII characters that are not spaces.
fn replace_blank(text: &str) -> String {
    let chars: Vec<char> = text.chars().collect();
    let mut out = String::with_capacity(text.len());
    for (index, c) in chars.iter().enumerate() {
        if *c == ' ' {
            // Upstream indexes `text[i + 1]` and `text[i - 1]`; at the ends that is an
            // IndexError or the last character, and a stripped text never has a blank there.
            let next = chars.get(index + 1);
            let previous = index.checked_sub(1).and_then(|i| chars.get(i));
            let ascii = |c: Option<&char>| c.is_some_and(|c| c.is_ascii() && *c != ' ');
            if ascii(next) && ascii(previous) {
                out.push(*c);
            }
        } else {
            out.push(*c);
        }
    }
    out
}

fn replace_corner_mark(text: &str) -> String {
    text.replace('²', "平方").replace('³', "立方")
}

fn remove_bracket(text: &str) -> String {
    text.replace(['（', '）', '【', '】', '`'], "")
        .replace("——", " ")
}

/// `re.sub(r'[，,、]+$', '。', text)`.
fn end_with_a_stop(text: &str) -> String {
    let trimmed = text.trim_end_matches(['，', ',', '、']);
    match trimmed.len() == text.len() {
        true => text.to_string(),
        false => format!("{trimmed}。"),
    }
}

/// Nothing but punctuation and symbols, or nothing at all: `^[\p{P}\p{S}]*$`.
fn is_only_punctuation(text: &str) -> bool {
    text.chars().all(|c| {
        c.is_ascii_punctuation() || (!c.is_alphanumeric() && !c.is_whitespace() && !c.is_control())
    })
}

/// `split_paragraph`: cut after sentence punctuation, then pack.
fn split_paragraph(
    text: &str,
    chinese: bool,
    length: &dyn Fn(&str) -> Result<usize>,
) -> Result<Vec<String>> {
    let stops: &[char] = match chinese {
        true => &['。', '？', '！', '；', '：', '、', '.', '?', '!', ';'],
        false => &['.', '?', '!', ';', ':'],
    };
    let measure = |piece: &str| -> Result<usize> {
        match chinese {
            true => Ok(piece.chars().count()),
            false => length(piece),
        }
    };

    let mut text: Vec<char> = text.chars().collect();
    if text.last().is_some_and(|c| !stops.contains(c)) {
        text.push(if chinese { '。' } else { '.' });
    }

    let mut utterances: Vec<String> = Vec::new();
    let mut start = 0;
    let mut index = 0;
    while index < text.len() {
        let c = text[index];
        if stops.contains(&c) {
            if index > start {
                utterances.push(text[start..=index].iter().collect());
            }
            match text.get(index + 1) {
                Some('"') | Some('”') => {
                    if let Some(last) = utterances.last_mut() {
                        last.push(text[index + 1]);
                    }
                    start = index + 2;
                }
                _ => start = index + 1,
            }
        }
        index += 1;
    }

    let mut out: Vec<String> = Vec::new();
    let mut current = String::new();
    for utterance in utterances {
        if measure(&format!("{current}{utterance}"))? > TOKEN_MAX && measure(&current)? > TOKEN_MIN
        {
            out.push(std::mem::take(&mut current));
        }
        current.push_str(&utterance);
    }
    if !current.is_empty() {
        if measure(&current)? < MERGE && !out.is_empty() {
            out.last_mut().expect("not empty").push_str(&current);
        } else {
            out.push(current);
        }
    }

    Ok(out)
}

/// `text_normalize(text, split=True)`: the sentences to read, one reading each.
///
/// `length` counts a piece in tokens, which is how an English text is packed.
pub fn sentences(text: &str, length: &dyn Fn(&str) -> Result<usize>) -> Result<Vec<String>> {
    if text.contains("<|") && text.contains("|>") {
        return Ok(vec![text.to_string()]);
    }
    let text = text.trim();
    if text.is_empty() {
        return Ok(Vec::new());
    }

    let pieces = match contains_chinese(text) {
        true => {
            let text = normalize(text, Language::Chinese).replace('\n', "");
            let text = replace_corner_mark(&replace_blank(&text));
            let text = text.replace('.', "。").replace(" - ", "，");
            let text = end_with_a_stop(&remove_bracket(&text));
            split_paragraph(&text, true, length)?
        }
        false => split_paragraph(&normalize(text, Language::English), false, length)?,
    };

    Ok(pieces
        .into_iter()
        .filter(|piece| !is_only_punctuation(piece))
        .collect())
}

/// `text_normalize(text, split=False)`: a transcript, normalized the same way and kept whole.
pub fn transcript(text: &str) -> String {
    if text.contains("<|") && text.contains("|>") {
        return text.to_string();
    }
    let text = text.trim();
    match contains_chinese(text) {
        true => {
            let text = normalize(text, Language::Chinese).replace('\n', "");
            let text = replace_corner_mark(&replace_blank(&text));
            let text = text.replace('.', "。").replace(" - ", "，");
            end_with_a_stop(&remove_bracket(&text))
        }
        false => normalize(text, Language::English),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn words(text: &str) -> Result<usize> {
        Ok(text.split_whitespace().count())
    }

    #[test]
    fn a_chinese_sentence_is_one_reading() {
        let got = sentences(
            "收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐。",
            &words,
        )
        .unwrap();
        assert_eq!(got.len(), 1);
    }

    #[test]
    fn chinese_gets_its_stops_and_loses_its_blanks() {
        assert_eq!(transcript("你好 世界，"), "你好世界。");
        assert_eq!(transcript("面积是5m²"), "面积是五m平方");
        assert_eq!(transcript("（括号）里."), "括号里。");
    }

    #[test]
    fn a_blank_between_latin_words_survives() {
        assert_eq!(replace_blank("我 用 Rust code 写"), "我用Rust code写");
    }

    #[test]
    fn long_chinese_is_packed_under_eighty_characters() {
        let sentence = "今天的天气非常好我们一起去公园散步吧。"; // 19 characters
        let text = sentence.repeat(10);
        let got = sentences(&text, &words).unwrap();
        assert!(got.len() > 1);
        assert!(got.iter().all(|piece| piece.chars().count() <= 80 + 19));
        assert_eq!(got.concat(), text);
    }

    #[test]
    fn english_numbers_are_written_out_and_a_stop_is_added() {
        let got = sentences("I have 3 apples", &words).unwrap();
        assert_eq!(got, vec!["I have three apples."]);
    }

    #[test]
    fn a_text_with_markers_is_left_alone() {
        let text = "You are a helpful assistant.<|endofprompt|>123";
        assert_eq!(sentences(text, &words).unwrap(), vec![text]);
    }

    #[test]
    fn punctuation_alone_is_not_read() {
        assert!(is_only_punctuation("。！"));
        assert!(!is_only_punctuation("好。"));
    }
}
