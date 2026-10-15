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

//! Numbers read in Spanish.

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
