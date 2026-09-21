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

"""IndexTTS-2.5's vocabulary as a `tokenizer.json`: tiktoken in, the `tokenizers` crate out.

    .venv/bin/python tools/indextts_tokenizer_exporter.py -output models/indextts25-tokenizer.json

The release ships `multilingual_zh_ja_yue_char_del.tiktoken`, which is 58 836 lines of
`base64(token bytes) rank`. Nothing in Rust reads that. What this crate reads is a
`tokenizer.json`, because a package carries the tokenizer its authors published and the
`tokenizers` crate is what parses it -- so the vocabulary has to be rewritten into that shape
rather than a second BPE being written here.

# The three things that have to survive the rewrite

**The ids.** tiktoken numbers the merged tokens 0 .. 58 835 and then appends its specials in the
order `get_encoding` builds them: two markers, ninety-nine languages -- `num_languages` defaults
to 99 and the table holds 106, so the last seven are *not* special tokens -- eleven audio events,
four emotions, six more markers, thirty reserved, twenty TTS tokens, and 1 501 timestamps. That
comes to 1 673, and 58 836 + 1 673 = 60 509, which is exactly the `number_text_tokens` the GPT's
`text_embedding` is sized for. Get the order wrong anywhere and every id past that point shifts,
which is a model that speaks fluent nonsense rather than one that fails.

The lists are read out of upstream's own `tokenizer.py` rather than copied here, for the same
reason the other tools download their references: a list retyped is a list that drifts.

**The merges.** tiktoken stores only the rank of each token; BPE as the `tokenizers` crate wants
it needs the *pair* that produced it. That pair is recovered by replaying BPE over the token's own
bytes -- joining the lowest-ranked adjacent pair over and over, which is the only move the
algorithm makes -- and stopping when two pieces are left. Those two are what the last merge
joined, by construction rather than by inference. See `parents`, which also records the wrong way
of doing it and what that cost.

**The bytes.** tiktoken works on raw bytes; the `tokenizers` crate's byte-level BPE works on the
GPT-2 mapping of bytes onto printable code points. Every token is moved through that map, and the
pre-tokenizer keeps tiktoken's own splitting pattern so the two agree on where words end.

# It is checked, not assumed

A reconstruction can be wrong quietly, so nothing is written until the two encoders agree on
77 000-odd texts: **every token in the vocabulary on its own** -- each has to come back as the one
id it is -- twenty thousand random concatenations of them, where merges reach across a boundary
neither piece knew about, and ten readable probes in Chinese, Japanese, Cantonese, English and
emoji.

The first of those three is the one that matters, and it is worth saying why it is there. The ten
readable probes were the whole check to begin with, and they all passed while a sixth of the
vocabulary encoded wrongly.
"""

import argparse
import base64
import json
import os
import random
import re
import sys

TIKTOKEN = "multilingual_zh_ja_yue_char_del"

# tiktoken's own splitting pattern, which the pre-tokenizer has to keep.
PAT = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

# What `get_tokenizer(multilingual=True)` passes, and what decides how many languages are special.
NUM_LANGUAGES = 99

# What the round trip is checked on. Nothing here is decorative: each line is a script or a shape
# the vocabulary treats differently.
PROBES = (
    "hello world",
    "The quick brown fox jumps over 13 lazy dogs.",
    "今天天气很好，我们去散步吧。",
    "これは日本語のテストです。",
    "廣東話都要識講先得㗎。",
    "<|zh|> 一二三四五六七八九十",
    "<|en|> It costs $5.20, or 3.5% more.",
    "混合 English and 中文 in one line!",
    "   leading and trailing   ",
    "emoji 🙂 and symbols ±§¶",
)


def upstream_lists(source=None):
    """`LANGUAGES`, `AUDIO_EVENT`, `EMOTION` and `TTS_Vocal_Token`, out of upstream's own file.

    The dictionary literals are executed on their own rather than the module imported, because
    importing it would drag in torch, whisper and tiktoken for four lists of strings.
    """
    source = source or os.path.expanduser(
        "~/.cache/libwaifu/indextts-src/indextts/utils/tokenizer.py"
    )

    if not os.path.isfile(source):
        sys.exit(
            f"{source} is not there.\n"
            "It is indextts/utils/tokenizer.py from index-tts, which the other reference "
            "scripts fetch the same way."
        )

    text = open(source, encoding="utf-8").read()
    out = {}

    for name in ("LANGUAGES", "AUDIO_EVENT", "EMOTION", "TTS_Vocal_Token"):
        match = re.search(r"^" + name + r"\s*=\s*\{.*?^\}", text, re.S | re.M)
        if match is None:
            sys.exit(f"upstream's tokenizer.py no longer defines {name}")

        scope = {}
        exec(match.group(0), scope)
        out[name] = scope[name]

    return out


def specials(lists, num_languages=NUM_LANGUAGES):
    """The special tokens, in the order `get_encoding` appends them."""
    return [
        "<|endoftext|>",
        "<|startoftranscript|>",
        *[f"<|{lang}|>" for lang in list(lists["LANGUAGES"].keys())[:num_languages]],
        *[f"<|{event}|>" for event in lists["AUDIO_EVENT"]],
        *[f"<|{emotion}|>" for emotion in lists["EMOTION"]],
        "<|translate|>",
        "<|transcribe|>",
        "<|startoflm|>",
        "<|startofprev|>",
        "<|nospeech|>",
        "<|notimestamps|>",
        *[f"<|SPECIAL_TOKEN_{i}|>" for i in range(1, 31)],
        *[f"<|{tts}|>" for tts in lists["TTS_Vocal_Token"]],
        *[f"<|{i * 0.02:.2f}|>" for i in range(1501)],
    ]


def ranks(path):
    """`bytes -> rank`, as the released file stores it."""
    out = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            token, rank = line.split()
            out[base64.b64decode(token)] = int(rank)

    return out


def byte_encoder():
    """GPT-2's map of the 256 bytes onto printable code points."""
    printable = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("\xa1"), ord("\xac") + 1))
        + list(range(ord("\xae"), ord("\xff") + 1))
    )
    mapped = printable[:]

    spare = 0
    for byte in range(256):
        if byte not in printable:
            printable.append(byte)
            mapped.append(256 + spare)
            spare += 1

    return {byte: chr(code) for byte, code in zip(printable, mapped)}


def parents(token, rank_of):
    """The two pieces whose merge produced `token`, by running BPE on it.

    Not a guess at the split. BPE is replayed over the token's own bytes -- repeatedly joining the
    adjacent pair with the lowest rank, which is the only move the algorithm ever makes -- and
    stopped when two pieces are left. Those two are what the final merge joined, by construction.

    An earlier version of this took the split whose halves ranked earliest, which is a plausible
    reading of what a rank means and is wrong for one token in six: ` there` came apart into
    ` th` + `ere` where the model merges ` the` + `re`. The difference is invisible in a handful
    of probes and changes a sixth of every sentence, so the check below now encodes the entire
    vocabulary rather than ten lines of it.
    """
    parts = [bytes([byte]) for byte in token]

    while len(parts) > 2:
        best_rank, best_at = None, None
        for index in range(len(parts) - 1):
            rank = rank_of.get(parts[index] + parts[index + 1])
            if rank is not None and (best_rank is None or rank < best_rank):
                best_rank, best_at = rank, index

        if best_at is None:
            return None

        parts[best_at : best_at + 2] = [parts[best_at] + parts[best_at + 1]]

    return (parts[0], parts[1]) if len(parts) == 2 else None


def merges_of(rank_of):
    """The pair that produced each token, in the order the ranks put them.

    A merge's priority in `tokenizer.json` is its position in the list, and a token's rank is its
    priority in tiktoken, so walking the ranks in order is what makes the two agree.
    """
    merges = []

    for token, _ in sorted(rank_of.items(), key=lambda pair: pair[1]):
        if len(token) < 2:
            continue

        pair = parents(token, rank_of)
        if pair is not None:
            merges.append(pair)

    return merges


def build(rank_of, special_tokens):
    """The `tokenizer.json` as a dictionary."""
    encoder = byte_encoder()

    def spell(token: bytes) -> str:
        return "".join(encoder[byte] for byte in token)

    vocab = {spell(token): rank for token, rank in rank_of.items()}

    base = len(rank_of)
    added = []
    for offset, token in enumerate(special_tokens):
        vocab[token] = base + offset
        added.append(
            {
                "id": base + offset,
                "content": token,
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": True,
            }
        )

    merges = [f"{spell(left)} {spell(right)}" for left, right in merges_of(rank_of)]

    return {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": added,
        "normalizer": None,
        "pre_tokenizer": {
            "type": "Sequence",
            "pretokenizers": [
                {
                    "type": "Split",
                    "pattern": {"Regex": PAT},
                    "behavior": "Isolated",
                    "invert": False,
                },
                {
                    "type": "ByteLevel",
                    "add_prefix_space": False,
                    "trim_offsets": True,
                    "use_regex": False,
                },
            ],
        },
        "post_processor": None,
        "decoder": {
            "type": "ByteLevel",
            "add_prefix_space": False,
            "trim_offsets": True,
            "use_regex": False,
        },
        "model": {
            "type": "BPE",
            "dropout": None,
            "unk_token": None,
            "continuing_subword_prefix": None,
            "end_of_word_suffix": None,
            "fuse_unk": False,
            "byte_fallback": False,
            "ignore_merges": False,
            "vocab": vocab,
            "merges": merges,
        },
    }


def reference(rank_of, special_tokens):
    """tiktoken's own encoding, built the way `get_encoding` builds it."""
    import tiktoken

    return tiktoken.Encoding(
        name=TIKTOKEN,
        explicit_n_vocab=len(rank_of) + len(special_tokens),
        pat_str=PAT,
        mergeable_ranks=rank_of,
        special_tokens={token: len(rank_of) + i for i, token in enumerate(special_tokens)},
    )


def check(written, rank_of, special_tokens, joins=20000, seed=7):
    """Encode both ways and report what disagrees.

    Three sets, and the first two are the ones that matter. Ten readable probes catch nothing:
    the merge reconstruction was wrong for a sixth of the vocabulary while all ten of them passed.

    1. **Every token in the vocabulary, on its own.** A token has to encode to the single id it
       *is*. This is the whole table checked against itself, and it is what a wrong merge cannot
       survive.
    2. **Random concatenations of them**, where merges reach across a boundary that neither piece
       knew about -- which is the case the per-token check cannot reach.
    3. The readable probes, kept because a failure there says something a reader can act on.
    """
    from tokenizers import Tokenizer

    mine = Tokenizer.from_str(json.dumps(written))
    theirs = reference(rank_of, special_tokens)

    pieces = []
    for token in rank_of:
        try:
            pieces.append(token.decode("utf-8"))
        except UnicodeDecodeError:
            # A token that is half a character has no text of its own to be encoded from; it is
            # reached only as part of one, which the concatenations below do reach.
            pass

    random.seed(seed)
    texts = list(PROBES) + pieces
    texts.extend(
        "".join(random.choice(pieces) for _ in range(random.randint(2, 8)))
        for _ in range(joins)
    )

    wrong = []
    for text in texts:
        want = theirs.encode(text, allowed_special="all")
        got = mine.encode(text, add_special_tokens=True).ids
        if got != want:
            wrong.append((text, want, got))

    return wrong, len(texts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-output", help="where to write the tokenizer")
    parser.add_argument(
        "-tiktoken",
        default=os.path.expanduser(f"~/.cache/libwaifu/indextts25/{TIKTOKEN}.tiktoken"),
        help="the released vocabulary",
    )
    parser.add_argument("-source", help="upstream's indextts/utils/tokenizer.py")
    parser.add_argument("-skip_check", action="store_true", help="do not compare against tiktoken")
    arguments = parser.parse_args()

    rank_of = ranks(arguments.tiktoken)
    special_tokens = specials(upstream_lists(arguments.source))

    total = len(rank_of) + len(special_tokens)
    print(f"{len(rank_of)} merged tokens + {len(special_tokens)} special = {total}")

    written = build(rank_of, special_tokens)
    print(f"{len(written['model']['merges'])} merges reconstructed")

    if not arguments.skip_check:
        wrong, checked = check(written, rank_of, special_tokens)
        if wrong:
            for text, want, got in wrong[:5]:
                print(f"\ndisagree on {text!r}\n  tiktoken {want[:16]}\n  this     {got[:16]}")
            sys.exit(f"{len(wrong)} of {checked} texts disagree -- not writing the file")

        print(f"all {checked} texts agree with tiktoken")

    if arguments.output:
        os.makedirs(os.path.dirname(arguments.output) or ".", exist_ok=True)
        with open(arguments.output, "w", encoding="utf-8") as handle:
            json.dump(written, handle, ensure_ascii=False)
        print("wrote", arguments.output)


if __name__ == "__main__":
    main()
