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

"""The two recordings `waifu/tests/indextts.rs` speaks in the voice of.

    pip install --target ~/.cache/libwaifu/pydeps pyarrow
    PYTHONPATH=~/.cache/libwaifu/pydeps .venv/bin/python tools/indextts_reference_voices.py

Writes `models/indextts25-reference.wav` and `models/indextts25-other.wav`: the first recording
between six and ten seconds long in LibriSpeech's test-clean, and the first after it by a
different speaker. Real speech, because a voice model conditioned on anything else is being
asked a question it was never trained to answer -- see the note in CLAUDE.md about reference
inputs.

Two speakers rather than one because what the test asks is comparative: that a sentence said in
the first voice sounds more like the first speaker than like the second. One recording alone
could only be held to a similarity threshold, and speaker embeddings of two unrelated people are
often within a threshold's width of each other.

The recordings come from `hf-audio/esb-datasets-test-only-sorted`, which is LibriSpeech under
CC BY 4.0, fetched once into the Hugging Face cache. `pyarrow` reads the parquet it ships as; it
goes in the side directory the other reference tools use rather than in the venv, whose pins hold
several models' reference tensors steady.
"""

import argparse
import io
import os

import numpy as np
import soundfile as sf

DATASET = "hf-audio/esb-datasets-test-only-sorted"
SPLIT = "librispeech/test.clean-00000-of-00001.parquet"


def recordings(path, lowest=6.0, highest=10.0):
    """Every recording of a usable length, with the speaker it is by."""
    import pyarrow.parquet as pq

    for row in pq.read_table(path).to_pylist():
        samples, rate = sf.read(io.BytesIO(row["audio"]["bytes"]))
        if lowest <= len(samples) / rate <= highest:
            yield samples, rate, row["id"].split("-")[0], row["text"]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-models", default="models", help="where to write the two recordings")
    arguments = parser.parse_args()

    from huggingface_hub import hf_hub_download

    path = hf_hub_download(repo_id=DATASET, filename=SPLIT, repo_type="dataset")

    chosen = []
    for samples, rate, speaker, text in recordings(path):
        if all(speaker != held for _, _, held, _ in chosen):
            chosen.append((samples, rate, speaker, text))
        if len(chosen) == 2:
            break

    for (samples, rate, speaker, text), name in zip(chosen, ("reference", "other")):
        out = os.path.join(arguments.models, f"indextts25-{name}.wav")
        sf.write(out, samples.astype(np.float32), rate, subtype="PCM_16")
        print(f"wrote {out}: speaker {speaker}, {len(samples) / rate:.2f} s -- {text!r}")


if __name__ == "__main__":
    main()
