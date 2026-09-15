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

"""Putting a model's own tokenizer into a package.

What a package carries is the `tokenizer.json` the model's authors published, byte for byte. The
runtime reads it with the `tokenizers` crate, which is the same implementation `transformers` runs
underneath its fast tokenizers, so there is one description of how this model's text becomes ids
and both sides read it.

This replaced `bpe_exporter.py`, which walked a vocabulary out of sentencepiece or a slow
tokenizer and rebuilt it into a format of its own, paired with an ini section of flags saying which
algorithm and which whitespace rules to read it back with. Every one of those flags was a claim
about the tokenizer that could be wrong, and the normalizers -- NFC, NFKC -- had no flag at all.
Carrying the file through has nothing to get wrong.

A tokenizer that has no `tokenizer.json` upstream is converted here, by `transformers`, which is
the same conversion it would do at load time on the machine that ran the reference.
"""

import configparser
import io

TOKENIZER_INI = "tokenizer.ini"

# The one key a section needs: everything else about a tokenizer is in the file it names.
FILE_KEY = "file"

# What a package with a single tokenizer calls it, matching `Tokenizer::SECTION`.
DEFAULT_SECTION = "tokenizer"


def read_tokenizer(source: str, subfolder: str = ""):
    """The fast tokenizer at `source`, which may be a hub name or a directory.

    Fast is not a preference, it is the requirement: a fast tokenizer is one backed by a
    `tokenizers` object, and that object is what can be written out as the `tokenizer.json` the
    runtime reads. `transformers` converts a slow one on the way -- T5 publishes a `spiece.model`
    and no json -- and that conversion is the one it does anyway.

    `subfolder` is `""` rather than `None` for no subfolder, because that is what transformers
    joins onto a path without checking.
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(source, subfolder=subfolder, use_fast=True)
    if not tokenizer.is_fast:
        raise SystemExit(
            f"{source} has no fast tokenizer, so there is no tokenizer.json to carry")

    return tokenizer


def tokenizer_json(tokenizer) -> bytes:
    """`tokenizer.json` for a fast tokenizer, as the bytes to store.

    Taken from the backing `tokenizers` object rather than by copying a file out of the repository,
    so that a tokenizer converted on the way here is written the same way as one that shipped with
    its json already.
    """
    return tokenizer.backend_tokenizer.to_str(pretty=True).encode("utf-8")


def encoder(tokenizer):
    """What this tokenizer makes of a text, as the bare ids, with no markers around them.

    Bare because that is what the runtime's `Tokenizer::encode` gives back: the markers a model
    wants are put on by the pipeline, which is the only place that knows where they go.
    """
    return lambda text: tokenizer(text, add_special_tokens=False).input_ids


def tokenizer_ini(sections, config=None) -> configparser.ConfigParser:
    """The `tokenizer.ini` naming where each tokenizer is stored.

    `sections` is `(section name, entry name)` pairs. A package with one tokenizer has a single
    pair under `DEFAULT_SECTION`; Anima tokenizes its prompt twice and so names both, because a
    reader that knows to look for the second has to be told where by the first.
    """
    if config is None:
        config = configparser.ConfigParser()

    for section, file in sections:
        config[section] = {FILE_KEY: file}

    return config


def write_tokenizers(package, tokenizers) -> None:
    """Write each tokenizer's json into `package`, and the one ini that names them.

    `tokenizers` is `(section name, entry name, tokenizer)` triples.
    """
    for _, file, tokenizer in tokenizers:
        with package.open(file, "w", force_zip64=True) as fp:
            fp.write(tokenizer_json(tokenizer))

    ini = tokenizer_ini((section, file) for section, file, _ in tokenizers)
    with package.open(TOKENIZER_INI, "w", force_zip64=True) as fp:
        ini.write(io.TextIOWrapper(fp))


def tokenizer_corpus(encode) -> str:
    """Texts and the ids `encode` gives them, as one `text<TAB>id id id` line each.

    What a corpus is for is the part of a tokenizer that no reference tensor can check: a package
    holds the vocabulary, and whether the runtime reads back the file the exporter wrote is a
    separate question from whether the weights are right. `waifu/tests/` reads these back and
    compares token for token.
    """
    import random

    random.seed(7)
    texts = []

    # The tags these models are actually prompted with.
    tags = [
        "1girl", "1boy", "solo", "long hair", "looking at viewer", "blush", "smile",
        "open mouth", "blue eyes", "simple background", "masterpiece", "best quality",
        "highly detailed", "absurdres", "hair ornament", "school uniform", "cherry blossoms",
        "cinematic lighting", "depth of field", "watercolour", "chibi", "from behind"]
    for _ in range(300):
        texts.append(", ".join(random.sample(tags, random.randint(1, 8))))

    # Prose, with the contractions and the digits that the byte level pattern treats specially.
    words = [
        "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog", "don't", "isn't",
        "we've", "I'll", "she'd", "astronaut", "riding", "horse", "mars", "photograph", "of",
        "2024", "8k", "ultra-realistic", "snake_case", "IT'S"]
    for _ in range(300):
        texts.append(" ".join(random.choices(words, k=random.randint(1, 16))))

    # The awkward whitespace, which is where the pre-tokenizers differ most, and the characters a
    # normalizer rewrites -- a ligature, a full width comma, a combining accent. Those last ones
    # were what the encoders written here got wrong, because they normalized nothing; they are in
    # the corpus now precisely because carrying the published file is what fixed them.
    texts += [
        "trailing spaces   ", "  leading spaces", "a  b", "line one\nline two", "tab\there",
        "hello!!! wow???", " , punctuation", "", "   ", "a", "1", ",",
        "ﬁne ligature", "ｆｕｌｌ　ｗｉｄｔｈ", "café combining",
        "a photo of an astronaut riding a horse on mars"]

    # A tab or a newline in the text would break the line it is written on. The pre-tokenizer's
    # handling of those is the published tokenizer's business now, not this repository's.
    return "".join(
        "{}\t{}\n".format(text, " ".join(str(token_id) for token_id in encode(text)))
        for text in texts
        if "\n" not in text and "\t" not in text)
