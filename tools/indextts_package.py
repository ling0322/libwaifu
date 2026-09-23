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

"""IndexTTS-2.5 as one model: six exports and a vocabulary in, a manifest and its weights out.

    .venv/bin/python tools/indextts_package.py -output models/indextts25.safetensors

Every model IndexTTS-2.5 runs has an exporter of its own, and each of those writes one
safetensors file under the checkpoint's own names. This puts the six side by side, each under a
namespace of its own, and writes the manifest `waifu::IndexTts::from_manifest` reads:

| namespace | from | exporter |
| --- | --- | --- |
| `indextts2.gpt` | `gpt.pth` -- the GPT and the emotion path | `indextts_gpt_exporter.py` |
| `indextts2.w2v_bert` | `facebook/w2v-bert-2.0`, sixteen layers | `w2v_bert_exporter.py` |
| `indextts2.codec` | `codec.pth`, the decoder | `semantic_codec_exporter.py` |
| `indextts2.s2mel` | `s2mel.pth` | `s2mel_exporter.py` |
| `indextts2.campplus` | `funasr/campplus` | `campplus_exporter.py` |
| `indextts2.bigvgan` | `nvidia/bigvgan_v2_22khz_80band_256x` | `bigvgan_exporter.py` |

# Why a namespace each

A package is read as one table of names, and six models that were never meant to share one do not
keep out of each other's way: w2v-bert has an `encoder`, the codec had one too, and more than one
of them has a `conv_pre` or a `proj`. Krea 2 solves the same problem the same way, with
`krea2.text`, `krea2.dit` and `krea2.vae`.

# Nothing is narrowed

Every tensor is written at the float32 its exporter produced. The writer's habit is to narrow to
float16 on the way in, and for these models that is not a decision to take by default: upstream
runs S2Mel and BigVGAN with autocast off, in float32, and a vocoder is exactly where half a
precision's worth of error becomes audible. The package is therefore about the size of the
release itself, and making it smaller is a thing to measure before doing.

# The inputs are checked, not trusted

Each export is opened and its tensor count compared against what its exporter says it writes. A
stale file -- the codec's, say, from before it was the decoder half -- is a model that loads,
runs and says nothing, and this is the cheapest place to notice.
"""

import argparse
import os
from os import path

import torch
from safetensors import safe_open

from model_exporter import Context, open_weights, parse_size, stem_of

MODEL_TYPE = "indextts2"

# (namespace, file under -models, how many tensors that file has to hold, and one it must hold).
#
# The sentinel is what tells a stale file from a current one when the count happens to match: the
# codec's encoder and decoder halves are both 121 tensors, and only the decoder has `up.weight`.
PARTS = (
    ("gpt", "indextts25-gpt.safetensors", 454, "emo_conditioning_encoder.embed.pos_enc.pe"),
    ("w2v_bert", "w2v-bert-2.0.safetensors", 518, "semantic.std"),
    ("codec", "indextts25-codec.safetensors", 121, "up.weight"),
    ("s2mel", "indextts25-s2mel.safetensors", 236, "length_regulator.content_in_proj.weight"),
    ("campplus", "campplus.safetensors", 573, None),
    ("bigvgan", "bigvgan-22khz-80band.safetensors", 667, "conv_post.weight"),
)

TOKENIZER = "indextts25-tokenizer.json"

# What the pipeline is, beyond what the Rust configurations already say. Only the things a reading
# is actually steered by: the sampling the GPT does and the trajectory S2Mel follows, all of them
# upstream's defaults in `infer_v2_5.py`.
#
# There is no `suggested:` block. That block is a picture model's advice -- steps, guidance,
# sizes -- and a voice's starting values are what `waifu::Voice::defaults` says, read from here.
CONFIG = {
    "version": "2.5",
    "sample_rate": 22050,
    "reference_seconds": 15,
    "max_text_tokens_per_segment": 120,
    "temperature": 0.8,
    "top_k": 30,
    "top_p": 0.8,
    "repetition_penalty": 10.0,
    "max_mel_tokens": 1500,
    "diffusion_steps": 25,
    "cfg_rate": 0.7,
    "length_ratio": 1.72,
}

def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-output", required=True, help="the package, e.g. models/indextts25.safetensors")
    parser.add_argument("-models", default="models", help="where the six exports are")
    parser.add_argument("-part_size", default="2G", help="how large one file of weights may be")
    arguments = parser.parse_args()

    directory = path.dirname(path.abspath(arguments.output))
    stem = stem_of(arguments.output)

    # Checked first, all of them, so that a missing or stale input is found before anything is
    # written rather than after the first two gigabytes.
    for namespace, name, count, sentinel in PARTS:
        source = path.join(arguments.models, name)
        if not path.exists(source):
            raise SystemExit(f"{source} is missing -- run the {namespace} exporter first")

        with safe_open(source, "pt") as held:
            keys = set(held.keys())
        if len(keys) != count:
            raise SystemExit(f"{source} holds {len(keys)} tensors, and the {namespace} export "
                             f"writes {count} -- it is stale, or from another exporter")
        if sentinel is not None and sentinel not in keys:
            raise SystemExit(f"{source} has no {sentinel}, which the {namespace} export writes")

    tokenizer = path.join(arguments.models, TOKENIZER)
    if not path.exists(tokenizer):
        raise SystemExit(f"{tokenizer} is missing -- run tools/indextts_tokenizer_exporter.py")

    tokenizer_file = stem + ".tokenizer.json"
    with open(tokenizer, "rb") as source, open(path.join(directory, tokenizer_file), "wb") as out:
        out.write(source.read())
    print(f"wrote {tokenizer_file}")

    writer = open_weights(arguments.output, parse_size(arguments.part_size))

    total = 0
    for namespace, name, _, _ in PARTS:
        with safe_open(path.join(arguments.models, name), "pt") as held:
            for key in sorted(held.keys()):
                tensor = held.get_tensor(key)
                if tensor.dtype not in (torch.float32, torch.int64):
                    tensor = tensor.float()

                writer.write_tensor(
                    Context(f"{MODEL_TYPE}.{namespace}.{key}"),
                    tensor,
                    preserve_dtype=True,
                )
                total += tensor.numel()

        print(f"  {MODEL_TYPE}.{namespace}: {name}")

    config = {"model": {"type": MODEL_TYPE}, MODEL_TYPE: CONFIG}
    for name in writer.finish(config, None, {"tokenizer": tokenizer_file}):
        print(f"wrote {name}")
    print(f"{total / 1e6:.1f} M parameters in {len(PARTS)} models")


if __name__ == "__main__":
    main()
