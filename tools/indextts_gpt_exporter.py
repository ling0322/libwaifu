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

"""IndexTTS-2.5's GPT as safetensors: `gpt.pth` in, what `waifu::indextts_gpt` asks for out.

    .venv/bin/python tools/indextts_gpt_exporter.py -output models/indextts25-gpt.safetensors

The largest of the six models -- 3.3 GB of the release's 5.5 -- and the one the rest of the
pipeline is arranged around.

# Nothing is done to the weights

Unusually for an exporter here, this one renames nothing and folds nothing. `waifu::indextts_gpt`
was written against the checkpoint's own names, so `gpt.h.0.attn.c_attn.weight` in the file is
that in the graph, and a tensor that turns out to be wrong can be compared with the one it came
from without a mapping in between.

That includes the two shapes that look like mistakes and are not. `attn.c_attn.weight` is
`(1280, 3840)` -- HuggingFace's `Conv1D` stores `(in, out)` where `nn.Linear` stores `(out, in)`,
and the graph multiplies it untransposed. `mel_head.weight` is `(8194, 1280)`, because *that* one
really is an `nn.Linear`. The two conventions sit ten lines apart in the model and are preserved
here rather than reconciled, since reconciling them is what would have to be undone at load.

# What it does not export, and why the shapes are checked

The release's GPT checkpoint holds 456 tensors. This writes the 301 the graph reads and says what
it left, by prefix:

- `emo_conditioning_encoder`, `emo_perceiver_encoder`, `emovec_layer`, `emo_layer` -- the four
  stages that *make* an emotion vector out of a recording. `inference_speech` takes `emo_vec`
  directly when it is given one, and that is the path `waifu::indextts_gpt` implements: the
  vector it is handed is already the 1280-wide one `emo_layer` would have produced. So the model
  says a sentence without these; they are how the vector is made, not part of saying anything.
- `text_head` -- a training-time head with nothing downstream of it.

Leaving them out is a decision and not an oversight. The shapes that carry one are checked rather
than trusted, so a tensor upstream quietly changes size becomes an error here instead of a model
that loads and speaks nonsense -- `lang_embedding` is the cautionary one, since it is 107 rows
for a model that advertises five languages, and a plausible guess of 9 loads, runs and is wrong.

# Where the checkpoint comes from

`IndexTeam/IndexTTS-2.5` on Hugging Face, which is public and ungated. Pass `-checkpoint` to read
one already on disk.
"""

import argparse
import os

import torch

REPO = "IndexTeam/IndexTTS-2.5"
CHECKPOINT = "gpt.pth"

# What `config.yaml` says, and what `waifu::indextts_gpt::Config::indextts` repeats.
LAYERS = 24

# The two the config does not state. `mel_pos_embedding` is sized
# `max_mel_tokens + 2 + max_conditioning_inputs` inside `build_hf_gpt_transformer`, and the
# language table is `len(LANGUAGE_DICT) + 1` -- Whisper's list of 106, most of which this model
# was never trained to say.
MEL_POSITIONS = 1818
LANGUAGES = 107

# Every tensor the graph asks for that is not inside a block.
TOP_LEVEL = (
    "text_embedding.weight",
    "mel_embedding.weight",
    "text_pos_embedding.emb.weight",
    "mel_pos_embedding.emb.weight",
    "lang_embedding.weight",
    "spk_emb_proj.weight",
    "spk_emb_proj.bias",
    "final_norm.weight",
    "final_norm.bias",
    "mel_head.weight",
    "mel_head.bias",
    "gpt.ln_f.weight",
    "gpt.ln_f.bias",
)

# And every one inside a block, which is a stock GPT-2's.
PER_BLOCK = (
    "ln_1.weight",
    "ln_1.bias",
    "attn.c_attn.weight",
    "attn.c_attn.bias",
    "attn.c_proj.weight",
    "attn.c_proj.bias",
    "ln_2.weight",
    "ln_2.bias",
    "mlp.c_fc.weight",
    "mlp.c_fc.bias",
    "mlp.c_proj.weight",
    "mlp.c_proj.bias",
)

# The shapes that carry a decision, checked rather than trusted. Everything else is checked by
# being present at all.
EXPECTED_SHAPES = {
    "text_embedding.weight": (60510, 1280),
    "mel_embedding.weight": (8194, 1280),
    "text_pos_embedding.emb.weight": (602, 1280),
    "mel_pos_embedding.emb.weight": (MEL_POSITIONS, 1280),
    "lang_embedding.weight": (LANGUAGES, 1280),
    "spk_emb_proj.weight": (1280, 192),
    "mel_head.weight": (8194, 1280),
    "gpt.h.0.attn.c_attn.weight": (1280, 3840),
}


def wanted(layers=LAYERS):
    """Every name the graph loads, in the order it loads them."""
    names = list(TOP_LEVEL)
    for index in range(layers):
        names.extend(f"gpt.h.{index}.{leaf}" for leaf in PER_BLOCK)

    return names


def checkpoint(path=None):
    """The released GPT, from a file or from the hub."""
    if path is None:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(repo_id=REPO, filename=CHECKPOINT)

    return torch.load(path, map_location="cpu", weights_only=True)


def select(state, layers=LAYERS):
    """The tensors the graph reads, and what was left behind."""
    names = wanted(layers)

    missing = [name for name in names if name not in state]
    if missing:
        raise SystemExit(
            f"the checkpoint is missing {len(missing)} tensors the graph reads, "
            f"starting with {missing[0]} -- upstream has renamed something"
        )

    for name, shape in EXPECTED_SHAPES.items():
        got = tuple(state[name].shape)
        if got != shape:
            raise SystemExit(f"{name} is {got} in the checkpoint and {shape} in the graph")

    taken = {name: state[name].detach().contiguous() for name in names}
    left = sorted(set(state) - set(names))

    return taken, left


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-output", help="where to write the exported weights")
    parser.add_argument("-checkpoint", help="a gpt.pth already on disk")
    parser.add_argument("-layers", type=int, default=LAYERS)
    arguments = parser.parse_args()

    state = checkpoint(arguments.checkpoint)
    taken, left = select(state, arguments.layers)

    total = sum(value.numel() for value in taken.values())
    print(f"{len(taken)} tensors, {total / 1e6:.2f} M parameters")

    # What is left is the conditioning half, and it is listed by its prefix rather than tensor by
    # tensor -- 157 names would bury the one that matters if upstream ever adds a 158th.
    prefixes = sorted({name.split(".")[0] for name in left})
    print(f"{len(left)} tensors not exported, under: {', '.join(prefixes)}")

    if arguments.output:
        from safetensors.torch import save_file

        os.makedirs(os.path.dirname(arguments.output) or ".", exist_ok=True)
        save_file(taken, arguments.output)
        print("wrote", arguments.output)


if __name__ == "__main__":
    main()
