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

//! Numbers read in English.

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
