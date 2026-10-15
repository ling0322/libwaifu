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

//! Writing out what a number is read as, so a speech model never sees a digit: the text normalizer
//! IndexTTS-2.5 and CosyVoice3 both read their text through.
//!
//! A text-to-speech model is trained on words. `2026` is not a word, and what a model does with
//! one is whatever its tokenizer happens to do -- which is why every TTS system has a pass like
//! this in front of it, turning `$5.20` into "five point two dollars" before anything else looks
//! at the text.
//!
//! [`normalize`] is that pass, for Chinese, English and Spanish. Japanese is returned unchanged,
//! deliberately: see below.
//!
//! ```
//! use waifu::text_normalizer::{normalize, Language};
//!
//! assert_eq!(normalize("共1234人", Language::Chinese), "共一千二百三十四人");
//! assert_eq!(normalize("3.5% more", Language::English), "three point five percent more");
//! assert_eq!(normalize("Son 21 días", Language::Spanish), "Son veintiuno días");
//! ```
//!
//! # Why this is rules and not a grammar
//!
//! IndexTTS-2.5 uses two of them. `zh` and `en` go through **WeTextProcessing**, and `es` through
//! **NeMo**, both of which are weighted finite-state transducers compiled from pynini grammars.
//! Neither has a Rust implementation, and neither ships as anything a Rust program can read: they
//! are compiled FST archives plus the engine that walks them.
//!
//! So this is written as rules, and it does **not** try to be bug-compatible with either. That is
//! a decision worth writing down, because the reference disagrees with itself in places:
//!
//! | input | WeTextProcessing | here |
//! | --- | --- | --- |
//! | `1000` (en) | *ten hundred* | one thousand |
//! | `1234` (en) | *twelve thirty four* | one thousand two hundred and thirty four |
//! | `110` (zh) | *幺幺零* | 一百一十 |
//! | `-7` / `-3.5` (en) | *negative* seven / *minus* three point five | minus, both |
//!
//! The first two are a year-reading rule reaching numbers that are not years, and the third is a
//! phone-number heuristic reaching a number that is not a phone number. Reproducing them would
//! mean reproducing the guess that produced them, and a guess is not what a caller wants when it
//! hands this a quantity. Where the reference is unambiguous -- `2026` as 两千零二十六, `第3` as
//! 第三, `3.14` as 三点一四 -- this agrees with it, and the tests check that it does.
//!
//! # Japanese is skipped, and that is the whole of it
//!
//! Upstream's `nemo_tn.py` maps a language onto a NeMo grammar and leaves Japanese out, with a
//! comment saying why: NeMo has no Japanese text-normalization grammar. Its `normalize` returns
//! the text untouched for any language it has no grammar for, so `nemo_text_normalize(text, "ja")`
//! is the identity. This matches that exactly, by doing nothing.
//!
//! What Japanese does need is not normalization but *segmentation* -- upstream runs it through
//! MeCab and joins the pieces with spaces -- and that wants a dictionary rather than a rule. It is
//! not here. See `docs/speech.md`.
//!
//! # What is covered
//!
//! Cardinals, decimals, negatives, percentages, money, ordinals, Chinese dates written with
//! 年月日, and clock times written `H:MM`. What is deliberately not covered is listed at the
//! bottom of `docs/text_normalize.md`, so a caller can see the edge before walking off it.

/// Which language's rules to read a number by.
///
/// Not a general language tag: these are the four IndexTTS-2.5 distinguishes at this stage, and
/// the fourth is here so that "Japanese is skipped" is something the type can say rather than
/// something a caller has to remember.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Language {
    Chinese,
    English,
    Spanish,
    /// Returned unchanged. See the module note.
    Japanese,
}

impl Language {
    /// The tag IndexTTS-2.5 uses, which is what a manifest and a CLI flag will carry.
    pub fn from_tag(tag: &str) -> Option<Language> {
        match tag.to_ascii_lowercase().as_str() {
            "zh" | "zhen" | "cmn" | "zho" => Some(Language::Chinese),
            "en" | "eng" => Some(Language::English),
            "es" | "spa" => Some(Language::Spanish),
            "ja" | "jpn" => Some(Language::Japanese),
            _ => None,
        }
    }

    /// Whether this language has rules here at all.
    pub fn is_normalized(self) -> bool {
        self != Language::Japanese
    }
}

/// The rules for each language, one file each.
mod chinese;
mod english;
mod spanish;

// ---------------------------------------------------------------------------------------------
// What a currency symbol is called
// ---------------------------------------------------------------------------------------------

/// The symbol, and what it is read as in each of the three languages.
///
/// Cents are not split out: `$5.20` is "five point two dollars" rather than "five dollars twenty",
/// which is what the reference does too and is the reading that does not need to know whether a
/// currency has hundredths at all.
const CURRENCIES: [(char, [&str; 3]); 5] = [
    ('$', ["美元", "dollars", "dólares"]),
    ('¥', ["元", "yuan", "yuanes"]),
    ('€', ["欧元", "euros", "euros"]),
    ('£', ["英镑", "pounds", "libras"]),
    ('₩', ["韩元", "won", "wones"]),
];

fn currency_word(symbol: char, language: Language) -> Option<&'static str> {
    let slot = match language {
        Language::Chinese => 0,
        Language::English => 1,
        Language::Spanish => 2,
        Language::Japanese => return None,
    };

    CURRENCIES
        .iter()
        .find(|(candidate, _)| *candidate == symbol)
        .map(|(_, words)| words[slot])
}

// ---------------------------------------------------------------------------------------------
// The pass itself
// ---------------------------------------------------------------------------------------------

/// Read `text` the way `language` reads it: every number written out as words.
///
/// Japanese is returned unchanged. See the module note for why that is the faithful thing to do
/// rather than a gap.
pub fn normalize(text: &str, language: Language) -> String {
    if !language.is_normalized() {
        return text.to_string();
    }

    let chars: Vec<char> = text.chars().collect();
    let mut out = String::with_capacity(text.len());
    let mut at = 0;

    while at < chars.len() {
        if let Some((spoken, next)) = read_number(&chars, at, language) {
            out.push_str(&spoken);
            at = next;
            continue;
        }

        out.push(chars[at]);
        at += 1;
    }

    out
}

/// Everything a run of digits can turn out to be, tried in the order that matters.
///
/// The order is the whole of the logic here. A time has to be looked for before a plain number,
/// or `14:30` becomes two numbers with a colon between them; a Chinese date has to be looked for
/// before a plain number, or the year loses its digit-by-digit reading.
fn read_number(chars: &[char], at: usize, language: Language) -> Option<(String, usize)> {
    if let Some(found) = read_ordinal_prefix(chars, at, language) {
        return Some(found);
    }

    if let Some(found) = read_time(chars, at, language) {
        return Some(found);
    }

    if let Some(found) = read_chinese_date(chars, at, language) {
        return Some(found);
    }

    read_quantity(chars, at, language)
}

/// 第3 -- Chinese puts its ordinal marker in front, so it is read before the digits are.
fn read_ordinal_prefix(chars: &[char], at: usize, language: Language) -> Option<(String, usize)> {
    if language != Language::Chinese || chars[at] != '第' {
        return None;
    }

    let (value, next) = digits_at(chars, at + 1)?;
    let value: u64 = value.parse().ok()?;

    Some((
        format!("{}{}", chinese::ORDINAL_PREFIX, chinese::cardinal(value)),
        next,
    ))
}

/// `14:30`, and `14:30:05`.
fn read_time(chars: &[char], at: usize, language: Language) -> Option<(String, usize)> {
    let (hour, after_hour) = digits_at(chars, at)?;
    if hour.len() > 2 || after_hour >= chars.len() || chars[after_hour] != ':' {
        return None;
    }

    let (minute, after_minute) = digits_at(chars, after_hour + 1)?;
    if minute.len() != 2 {
        return None;
    }

    let hour: u64 = hour.parse().ok()?;
    let minute: u64 = minute.parse().ok()?;
    if hour > 23 || minute > 59 {
        return None;
    }

    let spoken = match language {
        Language::Chinese => format!(
            "{}点{}分",
            chinese::cardinal(hour),
            chinese::cardinal(minute)
        ),
        Language::English => format!("{} {}", english::cardinal(hour), english::cardinal(minute)),
        Language::Spanish => format!("{} {}", spanish::cardinal(hour), spanish::cardinal(minute)),
        Language::Japanese => return None,
    };

    Some((spoken, after_minute))
}

/// `2026年`, where the year is read digit by digit and the month and the day are not.
fn read_chinese_date(chars: &[char], at: usize, language: Language) -> Option<(String, usize)> {
    if language != Language::Chinese {
        return None;
    }

    let (value, next) = digits_at(chars, at)?;
    if next >= chars.len() || chars[next] != '年' {
        return None;
    }

    // 二零二六年 rather than 两千零二十六年: a year is a label, not a quantity.
    Some((format!("{}年", chinese::digits(&value)), next + 1))
}

/// A number, with whatever sign, currency, decimal part, percent sign or ordinal suffix it wears.
fn read_quantity(chars: &[char], at: usize, language: Language) -> Option<(String, usize)> {
    let mut cursor = at;

    // A currency symbol leads the amount in all three languages and is spoken after it.
    let currency = currency_word(chars[cursor], language).filter(|_| {
        chars
            .get(cursor + 1)
            .map(|c| c.is_ascii_digit())
            .unwrap_or(false)
    });
    if currency.is_some() {
        cursor += 1;
    }

    // A minus sign, but only where it is a sign rather than a hyphen between two numbers.
    let negative = matches!(chars[cursor], '-' | '−')
        && chars
            .get(cursor + 1)
            .map(|c| c.is_ascii_digit())
            .unwrap_or(false)
        && !at
            .checked_sub(1)
            .and_then(|before| chars.get(before))
            .map(|c| c.is_ascii_digit())
            .unwrap_or(false);
    if negative {
        cursor += 1;
    }

    if !chars.get(cursor)?.is_ascii_digit() {
        return None;
    }

    let (integer, after_integer) = grouped_digits(chars, cursor);
    cursor = after_integer;

    // A decimal point, which has to be followed by a digit to be one.
    let fraction = if chars.get(cursor) == Some(&'.')
        && chars
            .get(cursor + 1)
            .map(|c| c.is_ascii_digit())
            .unwrap_or(false)
    {
        let (digits, next) = digits_at(chars, cursor + 1)?;
        cursor = next;
        Some(digits)
    } else {
        None
    };

    let percent = chars.get(cursor) == Some(&'%');
    if percent {
        cursor += 1;
    }

    let ordinal = read_ordinal_suffix(chars, cursor, language, fraction.is_none());
    if let Some((_, next)) = ordinal {
        cursor = next;
    }

    let value: u64 = integer.parse().ok()?;
    let mut spoken = String::new();

    if negative {
        spoken.push_str(match language {
            Language::Chinese => chinese::NEGATIVE,
            Language::English => english::NEGATIVE,
            Language::Spanish => spanish::NEGATIVE,
            Language::Japanese => return None,
        });
        if language != Language::Chinese {
            spoken.push(' ');
        }
    }

    // Chinese puts 百分之 in front of the number; the other two put a word after it.
    if percent && language == Language::Chinese {
        spoken.push_str(chinese::PERCENT_PREFIX);
    }

    let body = if ordinal.is_some() {
        match language {
            Language::English => english::ordinal(value),
            Language::Spanish => spanish::ordinal(value),
            _ => return None,
        }
    } else {
        match language {
            Language::Chinese => chinese::cardinal(value),
            Language::English => english::cardinal(value),
            Language::Spanish => spanish::cardinal(value),
            Language::Japanese => return None,
        }
    };
    spoken.push_str(&body);

    if let Some(digits) = &fraction {
        let tail = match language {
            Language::Chinese => chinese::fraction(digits),
            Language::English => english::fraction(digits),
            Language::Spanish => spanish::fraction(digits),
            Language::Japanese => return None,
        };

        if language != Language::Chinese {
            spoken.push(' ');
        }
        spoken.push_str(&tail);
    }

    if percent && language != Language::Chinese {
        spoken.push(' ');
        spoken.push_str(match language {
            Language::English => english::PERCENT_SUFFIX,
            Language::Spanish => spanish::PERCENT_SUFFIX,
            _ => return None,
        });
    }

    if let Some(word) = currency {
        if language != Language::Chinese {
            spoken.push(' ');
        }
        spoken.push_str(word);
    }

    Some((spoken, cursor))
}

/// `1st`, `2nd`, `3rd`, `4th` -- and only where the suffix agrees with the number it follows, so
/// that the `st` in `1street` is left alone.
fn read_ordinal_suffix(
    chars: &[char],
    at: usize,
    language: Language,
    whole: bool,
) -> Option<((), usize)> {
    if language != Language::English || !whole {
        return None;
    }

    let suffix: String = chars
        .get(at..at + 2)?
        .iter()
        .collect::<String>()
        .to_ascii_lowercase();

    if !matches!(suffix.as_str(), "st" | "nd" | "rd" | "th") {
        return None;
    }

    // A letter after the suffix means it was the start of a word, not an ending.
    if chars
        .get(at + 2)
        .map(|c| c.is_alphabetic())
        .unwrap_or(false)
    {
        return None;
    }

    Some(((), at + 2))
}

/// A run of digits, with `,` accepted between groups of three and dropped.
fn grouped_digits(chars: &[char], at: usize) -> (String, usize) {
    let mut digits = String::new();
    let mut cursor = at;

    while cursor < chars.len() {
        if chars[cursor].is_ascii_digit() {
            digits.push(chars[cursor]);
            cursor += 1;
            continue;
        }

        // A comma is a separator only when exactly three digits follow it.
        let separated = chars[cursor] == ','
            && chars
                .get(cursor + 1..cursor + 4)
                .map(|next| next.iter().all(|c| c.is_ascii_digit()))
                .unwrap_or(false)
            && !chars
                .get(cursor + 4)
                .map(|c| c.is_ascii_digit())
                .unwrap_or(false);

        if separated {
            cursor += 1;
            continue;
        }

        break;
    }

    (digits, cursor)
}

/// A plain run of digits, or nothing.
fn digits_at(chars: &[char], at: usize) -> Option<(String, usize)> {
    let mut digits = String::new();
    let mut cursor = at;

    while cursor < chars.len() && chars[cursor].is_ascii_digit() {
        digits.push(chars[cursor]);
        cursor += 1;
    }

    if digits.is_empty() {
        None
    } else {
        Some((digits, cursor))
    }
}
