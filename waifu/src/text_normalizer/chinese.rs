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

//! Numbers read in Chinese, which groups them by ten thousand rather than by a thousand.

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
