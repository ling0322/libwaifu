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

//! Writing out what a number is read as, so a speech model never sees a digit.
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
//! use waifu::indextts_normalize::{normalize, Language};
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
//! bottom of `docs/indextts_normalize.md`, so a caller can see the edge before walking off it.

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

// ---------------------------------------------------------------------------------------------
// Chinese
// ---------------------------------------------------------------------------------------------

mod chinese {
    const DIGITS: [&str; 10] = ["零", "一", "二", "三", "四", "五", "六", "七", "八", "九"];

    /// 个, 十, 百, 千 -- the places inside one group of four.
    const PLACES: [&str; 4] = ["", "十", "百", "千"];

    /// What every fourth place is worth. Chinese groups by ten thousand, not by a thousand, which
    /// is the single thing that makes this different from the other two.
    const GROUPS: [&str; 4] = ["", "万", "亿", "万亿"];

    /// One group of four digits, 1..=9999.
    ///
    /// `leading` says nothing has been spoken yet, anywhere in the number, which is what decides
    /// between 两 and 二. See [`cardinal`].
    ///
    /// Two other rules that are easy to state and easy to get wrong. A run of zeros inside a
    /// number becomes exactly one 零 however long it is, and a leading 一十 is said 十 -- ten is
    /// 十 and not 一十, while a hundred and ten is 一百一十 and not 一百十.
    fn group(value: u32, leading: bool) -> String {
        let mut out = String::new();
        let mut zero_pending = false;
        let mut started = false;

        for place in (0..4).rev() {
            let digit = (value / 10u32.pow(place)) % 10;

            if digit == 0 {
                if started {
                    zero_pending = true;
                }
                continue;
            }

            if zero_pending {
                out.push_str(DIGITS[0]);
                zero_pending = false;
            }

            if place == 1 && digit == 1 && !started {
                out.push_str(PLACES[1]);
            } else {
                let two = digit == 2 && place >= 2 && leading && !started;
                out.push_str(if two { "两" } else { DIGITS[digit as usize] });
                out.push_str(PLACES[place as usize]);
            }

            started = true;
        }

        out
    }

    /// A whole number, written out.
    ///
    /// # 两 and 二 are both "two"
    ///
    /// Which one a 2 takes is decided by where it sits, and the rule is narrower than it first
    /// looks: **两 only when the 2 is the first digit spoken in the whole number and a place word
    /// follows it.** Everywhere else it is 二.
    ///
    /// | | |
    /// | --- | --- |
    /// | 200, 2000, 20000 | 两百, 两千, 两万 -- leading, with a place |
    /// | 2, 20 | 二, 二十 -- leading, but the units and the tens take 二 |
    /// | 1234, 12000 | 一千**二**百三十四, 一万**二**千 -- not leading |
    /// | 2222 | **两**千**二**百**二**十二 -- all four rules in one number |
    ///
    /// An earlier version of this used the place alone, which reads 1234 as 一千两百三十四. That
    /// is wrong and it is wrong quietly: every test built from small numbers passes.
    pub fn cardinal(value: u64) -> String {
        if value == 0 {
            return DIGITS[0].to_string();
        }

        let mut groups = Vec::new();
        let mut rest = value;
        while rest > 0 {
            groups.push((rest % 10_000) as u32);
            rest /= 10_000;
        }

        let mut out = String::new();
        for index in (0..groups.len()).rev() {
            let value = groups[index];

            if value == 0 {
                // A whole empty group is still one 零, but only if something below it is spoken.
                if !out.is_empty()
                    && groups[..index].iter().any(|group| *group > 0)
                    && !out.ends_with(DIGITS[0])
                {
                    out.push_str(DIGITS[0]);
                }
                continue;
            }

            // 一亿零一: a group that does not fill its four places needs a 零 in front of it.
            if !out.is_empty() && value < 1000 && !out.ends_with(DIGITS[0]) {
                out.push_str(DIGITS[0]);
            }

            let leading = out.is_empty();

            // A group that is exactly 2 and carries 万 or 亿 takes 两 the same way 两百 does --
            // 20000 is 两万. `group` cannot see that, because inside it the 2 is a lone unit.
            if leading && value == 2 && index > 0 {
                out.push('两');
            } else {
                out.push_str(&group(value, leading));
            }

            out.push_str(GROUPS[index]);
        }

        out
    }

    /// Digit by digit, which is how a year is read: 2026 as 二零二六年.
    pub fn digits(text: &str) -> String {
        text.chars()
            .filter_map(|c| c.to_digit(10))
            .map(|d| DIGITS[d as usize])
            .collect()
    }

    /// The part after the point, which is always read digit by digit.
    pub fn fraction(text: &str) -> String {
        format!("点{}", digits(text))
    }

    pub const NEGATIVE: &str = "负";
    pub const PERCENT_PREFIX: &str = "百分之";
    pub const ORDINAL_PREFIX: &str = "第";
}

// ---------------------------------------------------------------------------------------------
// English
// ---------------------------------------------------------------------------------------------

mod english {
    const ONES: [&str; 20] = [
        "zero",
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
        "ten",
        "eleven",
        "twelve",
        "thirteen",
        "fourteen",
        "fifteen",
        "sixteen",
        "seventeen",
        "eighteen",
        "nineteen",
    ];

    const TENS: [&str; 10] = [
        "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
    ];

    const SCALES: [&str; 5] = ["", "thousand", "million", "billion", "trillion"];

    fn under_hundred(value: u32) -> String {
        if value < 20 {
            return ONES[value as usize].to_string();
        }

        let (tens, ones) = (value / 10, value % 10);
        if ones == 0 {
            TENS[tens as usize].to_string()
        } else {
            format!("{} {}", TENS[tens as usize], ONES[ones as usize])
        }
    }

    fn under_thousand(value: u32) -> String {
        if value < 100 {
            return under_hundred(value);
        }

        let (hundreds, rest) = (value / 100, value % 100);
        if rest == 0 {
            format!("{} hundred", ONES[hundreds as usize])
        } else {
            // "one hundred and one", which is the reading the reference uses too.
            format!(
                "{} hundred and {}",
                ONES[hundreds as usize],
                under_hundred(rest)
            )
        }
    }

    pub fn cardinal(value: u64) -> String {
        if value < 20 {
            return ONES[value as usize].to_string();
        }

        let mut groups = Vec::new();
        let mut rest = value;
        while rest > 0 {
            groups.push((rest % 1000) as u32);
            rest /= 1000;
        }

        let mut parts: Vec<String> = Vec::new();
        for index in (0..groups.len()).rev() {
            if groups[index] == 0 {
                continue;
            }

            let mut part = under_thousand(groups[index]);
            if index > 0 {
                part.push(' ');
                part.push_str(SCALES[index]);
            }
            parts.push(part);
        }

        // "two thousand and twenty six": the last group joins with "and" when it is small enough
        // to be a remainder rather than a magnitude of its own.
        if parts.len() > 1 && groups[0] > 0 && groups[0] < 100 {
            let tail = parts.pop().unwrap();
            format!("{} and {tail}", parts.join(" "))
        } else {
            parts.join(" ")
        }
    }

    /// first, second, twenty first -- the last word takes the suffix and the rest do not.
    pub fn ordinal(value: u64) -> String {
        let words = cardinal(value);
        let cut = words.rfind(' ').map(|at| at + 1).unwrap_or(0);
        let (head, last) = words.split_at(cut);

        let changed = match last {
            "one" => "first".to_string(),
            "two" => "second".to_string(),
            "three" => "third".to_string(),
            "five" => "fifth".to_string(),
            "eight" => "eighth".to_string(),
            "nine" => "ninth".to_string(),
            "twelve" => "twelfth".to_string(),
            other if other.ends_with('y') => format!("{}ieth", &other[..other.len() - 1]),
            other => format!("{other}th"),
        };

        format!("{head}{changed}")
    }

    pub fn fraction(text: &str) -> String {
        let spoken: Vec<&str> = text
            .chars()
            .filter_map(|c| c.to_digit(10))
            .map(|d| ONES[d as usize])
            .collect();

        format!("point {}", spoken.join(" "))
    }

    pub const NEGATIVE: &str = "minus";
    pub const PERCENT_SUFFIX: &str = "percent";
}

// ---------------------------------------------------------------------------------------------
// Spanish
// ---------------------------------------------------------------------------------------------

mod spanish {
    const ONES: [&str; 30] = [
        "cero",
        "uno",
        "dos",
        "tres",
        "cuatro",
        "cinco",
        "seis",
        "siete",
        "ocho",
        "nueve",
        "diez",
        "once",
        "doce",
        "trece",
        "catorce",
        "quince",
        "dieciséis",
        "diecisiete",
        "dieciocho",
        "diecinueve",
        "veinte",
        "veintiuno",
        "veintidós",
        "veintitrés",
        "veinticuatro",
        "veinticinco",
        "veintiséis",
        "veintisiete",
        "veintiocho",
        "veintinueve",
    ];

    const TENS: [&str; 10] = [
        "",
        "",
        "veinte",
        "treinta",
        "cuarenta",
        "cincuenta",
        "sesenta",
        "setenta",
        "ochenta",
        "noventa",
    ];

    /// The hundreds are their own words rather than "n hundred", and 500, 700 and 900 are not
    /// formed the way the other six are.
    const HUNDREDS: [&str; 10] = [
        "",
        "ciento",
        "doscientos",
        "trescientos",
        "cuatrocientos",
        "quinientos",
        "seiscientos",
        "setecientos",
        "ochocientos",
        "novecientos",
    ];

    fn under_hundred(value: u32) -> String {
        if value < 30 {
            return ONES[value as usize].to_string();
        }

        let (tens, ones) = (value / 10, value % 10);
        if ones == 0 {
            TENS[tens as usize].to_string()
        } else {
            // treinta y uno: from thirty up the two halves are joined by "y".
            format!("{} y {}", TENS[tens as usize], ONES[ones as usize])
        }
    }

    fn under_thousand(value: u32) -> String {
        if value < 100 {
            return under_hundred(value);
        }

        let (hundreds, rest) = (value / 100, value % 100);
        if rest == 0 {
            // Exactly one hundred is "cien"; a hundred and something is "ciento ...".
            return if hundreds == 1 {
                "cien".to_string()
            } else {
                HUNDREDS[hundreds as usize].to_string()
            };
        }

        format!("{} {}", HUNDREDS[hundreds as usize], under_hundred(rest))
    }

    pub fn cardinal(value: u64) -> String {
        if value < 30 {
            return ONES[value as usize].to_string();
        }

        let millions = value / 1_000_000;
        let rest = value % 1_000_000;

        let mut parts: Vec<String> = Vec::new();

        if millions > 0 {
            // "un millón", not "uno millón", and the plural changes the noun rather than a suffix.
            parts.push(if millions == 1 {
                "un millón".to_string()
            } else {
                format!("{} millones", cardinal(millions))
            });
        }

        let thousands = rest / 1000;
        if thousands > 0 {
            parts.push(if thousands == 1 {
                "mil".to_string()
            } else {
                format!("{} mil", under_thousand(thousands as u32))
            });
        }

        let units = rest % 1000;
        if units > 0 {
            parts.push(under_thousand(units as u32));
        }

        if parts.is_empty() {
            under_thousand(0)
        } else {
            parts.join(" ")
        }
    }

    /// primero through décimo, and the cardinal beyond -- which is what Spanish itself tends to
    /// do past ten in speech.
    pub fn ordinal(value: u64) -> String {
        match value {
            1 => "primero".to_string(),
            2 => "segundo".to_string(),
            3 => "tercero".to_string(),
            4 => "cuarto".to_string(),
            5 => "quinto".to_string(),
            6 => "sexto".to_string(),
            7 => "séptimo".to_string(),
            8 => "octavo".to_string(),
            9 => "noveno".to_string(),
            10 => "décimo".to_string(),
            other => cardinal(other),
        }
    }

    pub fn fraction(text: &str) -> String {
        let spoken: Vec<&str> = text
            .chars()
            .filter_map(|c| c.to_digit(10))
            .map(|d| ONES[d as usize])
            .collect();

        format!("coma {}", spoken.join(" "))
    }

    pub const NEGATIVE: &str = "menos";
    pub const PERCENT_SUFFIX: &str = "por ciento";
}

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
