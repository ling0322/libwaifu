#!/usr/bin/env python3
# The MIT License (MIT)
#
# Copyright (c) 2026 Xiaoyang Chen
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software
# and associated documentation files (the "Software"), to deal in the Software without
# restriction, including without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or
# substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""What WeTextProcessing says, beside what `waifu::indextts_normalize` says.

    pip install --target ~/.cache/libwaifu/pydeps wetext
    PYTHONPATH=~/.cache/libwaifu/pydeps .venv/bin/python tools/indextts_normalize_reference.py

This is **not** a generator for a test table, and the difference matters. The Rust tests state
what each language says, written out by hand; this prints where the two implementations disagree
so that every disagreement is one somebody looked at.

That is the opposite of how the other tools here work, and the reason is that the reference is
not always right. WeTextProcessing reads `1000` as "ten hundred" and `1234` as "twelve thirty
four" -- a year rule reaching numbers that are not years -- and `110` as 幺幺零, a phone-number
rule reaching a number that is not a phone number. A table generated from its output would pin
this crate to those readings.

So: agreement is the default and is checked; disagreement is allowed and is listed, with the
expected ones named below. Anything not on that list is a bug in one of the two, and this is what
says which cases to go and look at.

`es` is not compared. NeMo is what IndexTTS uses for Spanish, and `nemo_text_processing` does not
install here (`cdifflib` fails to build a wheel), so the Spanish rules are checked against the
language rather than against a second implementation -- see `waifu/tests/normalize.rs`.
"""

import argparse
import json
import subprocess
import sys

# Where the two are expected to differ, and why. Anything here is reported as "as designed";
# anything not here is reported as "look at this".
EXPECTED = {
    "1000": "year rule reaching a quantity: reference says ten hundred",
    "1234": "year rule reaching a quantity: reference says twelve thirty four",
    "2026": "year rule reaching a quantity: reference says twenty twenty six",
    "1,234": "year rule reaching a quantity",
    "110": "phone rule reaching a quantity: reference says 幺幺零",
    "111": "phone rule reaching a quantity: reference says 幺幺幺",
    "-7": "reference says negative here and minus elsewhere",
    "$5.20": "reference drops the trailing zero in en and keeps it in zh",
    "5.20": "reference drops the trailing zero in en and keeps it in zh",
    "5 km": "unit expansion is not implemented here",
}

CASES = {
    "zh": [
        "0", "1", "2", "7", "10", "11", "12", "20", "21", "22", "100", "101", "110", "111",
        "200", "222", "1000", "1001", "1002", "1010", "1100", "1200", "1234", "2000", "2026",
        "2222", "10000", "12000", "20000", "20002", "100000", "1000000", "100000000",
        "1999", "3.14", "0.5", "-7", "-3.5", "50%", "3.5%",
        "第1", "第3", "第10", "第21",
        "共1234人", "他花了200元", "$5.20", "5.20",
    ],
    "en": [
        "0", "1", "7", "10", "11", "12", "20", "21", "100", "101", "110", "1000", "1234",
        "1,234", "2026", "1000000", "3.14", "-7", "-3.5", "50%", "3.5%",
        "1st", "2nd", "3rd", "10th", "12th", "20th", "21st", "$5.20", "5.20", "5 km",
    ],
}

LANGUAGE = {"zh": "Chinese", "en": "English"}


def theirs(lang):
    from wetext import Normalizer

    if lang == "zh":
        return Normalizer(remove_erhua=False, lang="zh", operator="tn")

    return Normalizer(lang=lang, operator="tn")


def ours(cases):
    """Run the Rust side once, over every case, through a throwaway binary.

    A subprocess rather than a binding: this crate links `libflint.a`, and building a Python
    extension of it to normalize a string would be a great deal of machinery for a pure function.
    """
    program = """
use waifu::indextts_normalize::{normalize, Language};

fn main() {
    let mut input = String::new();
    std::io::Read::read_to_string(&mut std::io::stdin(), &mut input).unwrap();

    for line in input.lines() {
        let (tag, text) = line.split_once('\\t').unwrap();
        let language = Language::from_tag(tag).unwrap();
        println!("{}", normalize(text, language));
    }
}
"""

    import os
    import tempfile

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with tempfile.TemporaryDirectory() as scratch:
        source = os.path.join(root, "waifu", "examples")
        os.makedirs(source, exist_ok=True)
        path = os.path.join(source, "normalize_cases.rs")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(program)

        try:
            stdin = "\n".join(f"{lang}\t{text}" for lang, text in cases)
            done = subprocess.run(
                [
                    "cargo",
                    "run",
                    "--quiet",
                    "--manifest-path",
                    os.path.join(root, "waifu", "Cargo.toml"),
                    "--example",
                    "normalize_cases",
                ],
                input=stdin,
                capture_output=True,
                text=True,
                cwd=root,
            )
        finally:
            os.unlink(path)
            if not os.listdir(source):
                os.rmdir(source)

    if done.returncode != 0:
        sys.exit(f"the Rust side did not run:\n{done.stderr}")

    return done.stdout.splitlines()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-json", help="write the comparison here as well")
    arguments = parser.parse_args()

    flat = [(lang, text) for lang, texts in CASES.items() for text in texts]
    mine = ours(flat)

    if len(mine) != len(flat):
        sys.exit(f"asked for {len(flat)} cases and got {len(mine)} answers back")

    normalizers = {lang: theirs(lang) for lang in CASES}

    rows = []
    agreed = 0
    unexpected = 0

    for (lang, text), got in zip(flat, mine):
        want = normalizers[lang].normalize(text)
        same = got == want
        agreed += same

        note = None
        if not same:
            note = EXPECTED.get(text, "UNEXPECTED")
            unexpected += note == "UNEXPECTED"

        rows.append(
            {"lang": lang, "input": text, "reference": want, "ours": got, "agree": same, "note": note}
        )

    print(f"{agreed} of {len(rows)} agree with WeTextProcessing\n")

    for row in rows:
        if row["agree"]:
            continue
        mark = "!!" if row["note"] == "UNEXPECTED" else "  "
        print(f"{mark} [{row['lang']}] {row['input']!r}")
        print(f"     reference {row['reference']!r}")
        print(f"     ours      {row['ours']!r}")
        print(f"     {row['note']}")

    if arguments.json:
        with open(arguments.json, "w", encoding="utf-8") as handle:
            json.dump(rows, handle, ensure_ascii=False, indent=2)
        print("\nwrote", arguments.json)

    if unexpected:
        sys.exit(f"\n{unexpected} disagreements are not accounted for -- go and look at them")


if __name__ == "__main__":
    main()
