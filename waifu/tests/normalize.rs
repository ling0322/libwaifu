//! The text normalizer, against what each language actually says.
//!
//! There is no machine-written reference table here, and that is deliberate. The two systems
//! IndexTTS-2.5 uses -- WeTextProcessing for `zh`/`en`, NeMo for `es` -- are WFST grammars that
//! disagree with themselves in places (`1000` reads as *ten hundred*), so a table copied from
//! their output would pin this to their bugs as firmly as to their rules.
//!
//! So the expectations below are written out by hand, and `tools/indextts_normalize_reference.py` reports
//! where this and WeTextProcessing differ, so every divergence is one somebody chose. The cases
//! marked "agrees with the reference" are the ones that came out of running it.

use waifu::indextts::normalize::{normalize, Language};

const ZH: Language = Language::Chinese;
const EN: Language = Language::English;
const ES: Language = Language::Spanish;
const JA: Language = Language::Japanese;

#[track_caller]
fn says(input: &str, language: Language, want: &str) {
    let got = normalize(input, language);
    assert_eq!(got, want, "{input:?} in {language:?}");
}

// ---------------------------------------------------------------------------------------------
// Chinese
// ---------------------------------------------------------------------------------------------

/// The places, and the two rules that are easy to get wrong: one 零 for any run of zeros, and a
/// leading 一十 said as 十.
#[test]
fn chinese_counts() {
    says("0", ZH, "零");
    says("1", ZH, "一");
    says("2", ZH, "二");
    says("10", ZH, "十");
    says("11", ZH, "十一");
    says("12", ZH, "十二");
    says("20", ZH, "二十");
    says("21", ZH, "二十一");
    says("100", ZH, "一百");
    says("101", ZH, "一百零一");
    says("110", ZH, "一百一十");
    says("1000", ZH, "一千");
    says("1001", ZH, "一千零一");
    says("1010", ZH, "一千零一十");
    says("1100", ZH, "一千一百");
    says("1234", ZH, "一千二百三十四");
    says("1999", ZH, "一千九百九十九");
}

/// Chinese groups by ten thousand rather than by a thousand, which is the whole of why 万 and 亿
/// are places and "million" is not.
#[test]
fn chinese_groups_by_ten_thousand() {
    says("10000", ZH, "一万");
    says("100000", ZH, "十万");
    says("1000000", ZH, "一百万");
    says("100000000", ZH, "一亿");
    says("100000001", ZH, "一亿零一");
}

/// 两 only where the 2 leads the whole number and a place word follows it; 二 everywhere else.
///
/// Every line here agrees with WeTextProcessing, and the rule is narrower than "两 from the
/// hundreds up" -- which is what this did first, and which reads 1234 as 一千两百三十四. The
/// three internal cases are the ones that caught it.
#[test]
fn chinese_says_two_two_ways() {
    // Leading, with a place word.
    says("200", ZH, "两百");
    says("2000", ZH, "两千");
    says("20000", ZH, "两万");
    says("2026", ZH, "两千零二十六");
    says("20002", ZH, "两万零二");

    // Leading, but the units and the tens take 二 whatever else is true.
    says("2", ZH, "二");
    says("20", ZH, "二十");
    says("22", ZH, "二十二");

    // Not leading: something was spoken before it.
    says("1200", ZH, "一千二百");
    says("1234", ZH, "一千二百三十四");
    says("12000", ZH, "一万二千");
    says("1002", ZH, "一千零二");

    // All four rules in one number.
    says("2222", ZH, "两千二百二十二");
    says("222", ZH, "两百二十二");
}

#[test]
fn chinese_points_and_signs() {
    says("3.14", ZH, "三点一四");
    says("0.5", ZH, "零点五");
    says("-7", ZH, "负七");
    says("-3.5", ZH, "负三点五");
}

/// 百分之 goes in front, which is the one place the three languages put the same idea in
/// different halves of the phrase.
#[test]
fn chinese_percent_leads() {
    says("50%", ZH, "百分之五十");
    says("3.5%", ZH, "百分之三点五");
}

#[test]
fn chinese_ordinals_and_dates() {
    says("第1", ZH, "第一");
    says("第3", ZH, "第三");
    says("第10", ZH, "第十");
    says("第21", ZH, "第二十一");

    // A year is a label rather than a quantity, so its digits are read one at a time -- and the
    // month and the day beside it are not.
    says("2026年", ZH, "二零二六年");
    says("2026年9月20日", ZH, "二零二六年九月二十日");
}

#[test]
fn chinese_in_a_sentence() {
    says("共1234人", ZH, "共一千二百三十四人");
    says("他花了200元", ZH, "他花了两百元");
    says("$5.20", ZH, "五点二零美元");
    says("14:30", ZH, "十四点三十分");
}

// ---------------------------------------------------------------------------------------------
// English
// ---------------------------------------------------------------------------------------------

#[test]
fn english_counts() {
    says("0", EN, "zero");
    says("7", EN, "seven");
    says("10", EN, "ten");
    says("11", EN, "eleven");
    says("20", EN, "twenty");
    says("21", EN, "twenty one");
    says("100", EN, "one hundred");
    says("101", EN, "one hundred and one");
    says("110", EN, "one hundred and ten");
    says("1000000", EN, "one million");
}

/// Where this deliberately parts company with WeTextProcessing.
///
/// It reads `1000` as *ten hundred* and `1234` as *twelve thirty four*, which is a year rule
/// reaching numbers that are not years. A caller that hands this a quantity gets a quantity.
#[test]
fn english_reads_quantities_as_quantities() {
    says("1000", EN, "one thousand");
    says("1234", EN, "one thousand two hundred and thirty four");
    says("2026", EN, "two thousand and twenty six");
    says("1,234", EN, "one thousand two hundred and thirty four");
}

#[test]
fn english_points_and_signs() {
    says("3.14", EN, "three point one four");
    says("-7", EN, "minus seven");
    says("-3.5", EN, "minus three point five");
    says("50%", EN, "fifty percent");
    says("3.5%", EN, "three point five percent");
}

/// Only the last word takes the suffix, and six of them are irregular.
#[test]
fn english_ordinals() {
    says("1st", EN, "first");
    says("2nd", EN, "second");
    says("3rd", EN, "third");
    says("5th", EN, "fifth");
    says("8th", EN, "eighth");
    says("9th", EN, "ninth");
    says("10th", EN, "tenth");
    says("12th", EN, "twelfth");
    says("20th", EN, "twentieth");
    says("21st", EN, "twenty first");
}

/// A suffix is only a suffix when a word does not carry on through it.
#[test]
fn english_leaves_words_alone() {
    says("1street", EN, "onestreet");
}

/// The part after the point is read digit by digit, including a trailing zero.
///
/// Another deliberate divergence. WeTextProcessing says "five point two dollars" for `$5.20` in
/// English and 五点二零美元 in Chinese -- it drops the zero in one language and keeps it in the
/// other. Keeping it is the reading that does not throw away a digit somebody wrote, and it is
/// the same rule in all three languages, which is worth more here than matching either half of
/// an inconsistency.
#[test]
fn english_keeps_the_digits_after_the_point() {
    says(
        "It costs $5.20.",
        EN,
        "It costs five point two zero dollars.",
    );
    says("5.20", EN, "five point two zero");
    says("5.2", EN, "five point two");
}

// ---------------------------------------------------------------------------------------------
// Spanish
// ---------------------------------------------------------------------------------------------

/// The teens and the twenties are single words, and from thirty up the two halves take a "y".
#[test]
fn spanish_counts() {
    says("0", ES, "cero");
    says("7", ES, "siete");
    says("15", ES, "quince");
    says("16", ES, "dieciséis");
    says("20", ES, "veinte");
    says("21", ES, "veintiuno");
    says("29", ES, "veintinueve");
    says("30", ES, "treinta");
    says("31", ES, "treinta y uno");
    says("99", ES, "noventa y nueve");
}

/// The hundreds are their own words, and exactly one hundred is "cien" where a hundred and
/// something is "ciento".
#[test]
fn spanish_hundreds_are_words() {
    says("100", ES, "cien");
    says("101", ES, "ciento uno");
    says("200", ES, "doscientos");
    says("500", ES, "quinientos");
    says("700", ES, "setecientos");
    says("900", ES, "novecientos");
}

#[test]
fn spanish_thousands_and_millions() {
    says("1000", ES, "mil");
    says("2000", ES, "dos mil");
    says("1234", ES, "mil doscientos treinta y cuatro");
    says("1000000", ES, "un millón");
    says("2000000", ES, "dos millones");
}

#[test]
fn spanish_points_and_signs() {
    says("3.14", ES, "tres coma uno cuatro");
    says("-7", ES, "menos siete");
    says("50%", ES, "cincuenta por ciento");
}

#[test]
fn spanish_ordinals_stop_at_ten() {
    says("Son 21 días", ES, "Son veintiuno días");
}

// ---------------------------------------------------------------------------------------------
// Japanese
// ---------------------------------------------------------------------------------------------

/// Skipped, and skipped exactly.
///
/// Upstream's `nemo_tn.py` leaves Japanese out of its language table on purpose -- NeMo has no
/// Japanese grammar -- and its `normalize` returns the text untouched for anything it has no
/// grammar for. So the faithful implementation is the identity, and this is the test that says
/// somebody decided that rather than forgot.
#[test]
fn japanese_is_returned_untouched() {
    for text in [
        "2026年9月20日に1234人が来た。",
        "$5.20",
        "50%",
        "これは日本語のテストです。",
    ] {
        says(text, JA, text);
    }

    assert!(!Language::Japanese.is_normalized());
    assert!(Language::Chinese.is_normalized());
}

// ---------------------------------------------------------------------------------------------
// The scanner itself
// ---------------------------------------------------------------------------------------------

/// Text with no numbers in it comes back as it went in, byte for byte.
#[test]
fn text_without_numbers_is_untouched() {
    for (text, language) in [
        ("你好，世界。", ZH),
        ("Hello, world!", EN),
        ("¡Hola, mundo!", ES),
    ] {
        says(text, language, text);
    }
}

/// A hyphen between two numbers is a hyphen, not a sign.
#[test]
fn a_range_is_not_a_negative() {
    says("7-3", EN, "seven-three");
    says("7 - 3", EN, "seven - three");
    says("-3", EN, "minus three");
}

/// A comma is a thousands separator only where three digits follow it and no fourth.
#[test]
fn commas_separate_only_where_they_group() {
    says("1,234", EN, "one thousand two hundred and thirty four");
    says("1,23", EN, "one,twenty three");
    says(
        "1,2345",
        EN,
        "one,two thousand three hundred and forty five",
    );
}

#[test]
fn tags_map_to_languages() {
    assert_eq!(Language::from_tag("zh"), Some(ZH));
    assert_eq!(Language::from_tag("ZH"), Some(ZH));
    assert_eq!(Language::from_tag("zhen"), Some(ZH));
    assert_eq!(Language::from_tag("en"), Some(EN));
    assert_eq!(Language::from_tag("es"), Some(ES));
    assert_eq!(Language::from_tag("ja"), Some(JA));
    assert_eq!(Language::from_tag("de"), None);
}
