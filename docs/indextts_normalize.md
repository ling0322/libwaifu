# IndexTTS-2.5's text normalizer

Writing out what a number is read as, so the model never sees a digit. `$5.20` becomes "five
point two zero dollars", `共1234人` becomes `共一千二百三十四人`, and `Son 21 días` becomes
`Son veintiuno días`.

`waifu::indextts_normalize`, beside [the GPT](indextts_gpt.md) rather than at the crate root —
a diffusion model has no text to normalize, and this is the speech pipeline's.

## Three languages, and one that is skipped

| | upstream | here |
| --- | --- | --- |
| `zh`, `en` | WeTextProcessing, a WFST grammar | rules |
| `es` | NeMo, a different WFST grammar | rules |
| `ja` | **nothing** — deliberately | nothing |

Japanese is not an omission. Upstream's `nemo_tn.py` maps a language onto a NeMo grammar and
leaves `ja` out with a comment saying why — NeMo has no Japanese TN grammar — and its `normalize`
returns the text untouched for anything unmapped. `nemo_text_normalize(text, "ja")` is the
identity, so the faithful implementation is the identity, and there is a test asserting somebody
decided that rather than forgot.

What Japanese *does* need is segmentation, not normalization: upstream runs it through MeCab and
joins the pieces with spaces. That wants a dictionary rather than a rule and is not here.

## Why rules rather than a port

Both references are weighted finite-state transducers compiled from pynini grammars. Neither has
a Rust implementation and neither ships as anything a Rust program can read — they are compiled
FST archives plus an engine to walk them. Porting one is a subsystem the size of everything else
in this pipeline.

So these are rules, and they are **not bug-compatible**. That is a decision rather than a
shortfall, because the reference disagrees with itself:

```
$ PYTHONPATH=~/.cache/libwaifu/pydeps .venv/bin/python tools/indextts_normalize_reference.py
69 of 79 agree with WeTextProcessing
```

All ten disagreements are accounted for, and the script fails if one appears that is not:

| input | reference | here | why |
| --- | --- | --- | --- |
| `1000` (en) | ten hundred | one thousand | a year rule reaching a quantity |
| `1234` (en) | twelve thirty four | one thousand two hundred and thirty four | the same rule |
| `1,234` (en) | *thousand* two hundred and thirty four | one thousand … | the same rule, and it drops the "one" |
| `2026` (en) | twenty twenty six | two thousand and twenty six | the same rule |
| `110` (zh) | 幺幺零 | 一百一十 | a phone rule reaching a quantity |
| `111` (zh) | 幺幺幺 | 一百一十一 | the same rule |
| `-7` (en) | **negative** seven | minus seven | the reference says *minus* for `-3.5` |
| `5.20` (en) | five point two **oh** | five point two zero | and 五点二零 in Chinese, so it is inconsistent with itself |
| `5 km` | five kilometers | five km | unit expansion is not implemented |

Where the reference is unambiguous — `2026` as 两千零二十六, `第3` as 第三, `3.14` as 三点一四,
`50%` as 百分之五十 — this agrees with it, and the tests pin that.

## The one rule worth reading twice

**两 and 二 are both "two", and which one a 2 takes is narrower than it looks:** 两 only when the
2 is the first digit spoken in the whole number *and* a place word follows it.

| | |
| --- | --- |
| 200, 2000, 20000 | 两百, 两千, 两万 — leading, with a place |
| 2, 20 | 二, 二十 — leading, but the units and the tens take 二 |
| 1234, 12000 | 一千**二**百三十四, 一万**二**千 — not leading |
| 2222 | **两**千**二**百**二**十二 — every case in one number |

This was first written as "两 from the hundreds up", which reads 1234 as 一千两百三十四. It passed
every test built from small numbers.

## What is covered

Cardinals, decimals, negatives, percentages, money in five currencies, ordinals (`第3`, `3rd`),
Chinese dates written with 年月日, clock times written `H:MM`, and `,` as a thousands separator.

Chinese groups by ten thousand rather than by a thousand, so 万 and 亿 are places where "million"
is a word; 100000 is 十万 and not 一百千.

## What is not

- **Unit expansion.** `5 km` stays `five km`.
- **Cents split out.** `$5.20` is "five point two zero dollars", not "five dollars twenty".
- **Year reading by context.** `2026年` is read digit by digit because 年 says it is a year;
  `In 2026` is not, because nothing in the text says so.
- **Dates in `en` and `es`.** Only the Chinese 年月日 form is recognised.
- **Roman numerals, fractions written `1/2`, phone numbers, abbreviations, erhua.**
- **Spanish gender and apocope.** `veintiuno` does not become `veintiún` before a masculine noun.

## What is checked

```bash
cargo test --manifest-path waifu/Cargo.toml --test normalize

pip install --target ~/.cache/libwaifu/pydeps wetext
PYTHONPATH=~/.cache/libwaifu/pydeps .venv/bin/python tools/indextts_normalize_reference.py
```

The Rust tests state what each language says, written out by hand. The script prints where the
two implementations differ and exits non-zero on a difference that is not in its expected list —
so agreement is the default, and every divergence is one somebody chose.

Spanish is not compared against anything: `nemo_text_processing` does not install here (`cdifflib`
fails to build a wheel), so its rules are checked against the language instead.
