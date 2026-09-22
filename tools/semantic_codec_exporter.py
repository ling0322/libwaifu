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

"""The semantic codec as safetensors: `codec.pth` in, what `waifu::semantic_codec` reads out.

    .venv/bin/python tools/semantic_codec_exporter.py -output models/indextts25-codec.safetensors

MaskGCT's codec as IndexTTS-2.5 vendors it: the alphabet the GPT reads and writes, and the way
back out of it.

# It is the *decoder* that gets exported, which is the surprise

The checkpoint holds an encoder and a decoder, 116 tensors each, and the obvious guess -- that a
text-to-speech pipeline encodes -- is backwards. What `infer_v2_5.py` runs is

    S_infer = self.semantic_codec.decode(codes)

turning the GPT's tokens back into the `hidden_size` features S2Mel's length regulator reads. The
line that would run the encoder over the reference audio,

    # _, S_ref = self.semantic_codec.quantize(spk_cond_emb)

is **commented out in the released source** and replaced by `S_ref = self.get_emb(...)`: the
reference reaches the length regulator as w2v-bert's continuous features, never having been
through a codebook. So `down`, `encoder` and `in_project` are not written here, and `up`,
`decoder` and `out_project` are.

`waifu::semantic_codec` keeps both halves in code -- the encoder is what says what a token *means*
-- but a package carries what a reading runs.

That is 121 tensors of the 243, and rather less than half the bytes.

# One weight has to be folded

`quantizer.quantizers.0.out_project` is wrapped in `torch.nn.utils.weight_norm`, so the checkpoint
stores a direction `weight_v` and a magnitude `weight_g` rather than a weight. Inference wants the
product, and the product is what this writes:

    weight = weight_g * weight_v / ||weight_v||

normalizing over every axis but the first, which is what `weight_norm(dim=0)` means. `waifu` does
this in `tools/bigvgan_exporter.py` and `tools/s2mel_exporter.py` too, for the same reason -- a
magnitude and a direction are a thing training wants and inference does not.

Everything else keeps the checkpoint's own name, shape and value.

# What is checked

That the 121 names the graph loads are all present and all the right shape, and that folding
produced the shape the graph declares.

    cargo run --release --example parameters -- semantic_codec_decode

prints that list, and it is the authority; this file is written against it.

# Where the checkpoint comes from

`IndexTeam/IndexTTS-2.5` on Hugging Face, which is public and ungated. Pass `-checkpoint` to read
one already on disk.
"""

import argparse
import os

import torch

REPO = "IndexTeam/IndexTTS-2.5"
CHECKPOINT = "codec.pth"

# `config.yaml`'s `semantic_codec.vocos_num_layers`, which the graph repeats.
CONVNEXT_BLOCKS = 12

# The convolution that halves the frame rate, and the one projection after the backbone.
TOP_LEVEL = (
    "up.weight",
    "up.bias",
    "decoder.0.embed.weight",
    "decoder.0.embed.bias",
    "decoder.0.norm.weight",
    "decoder.0.norm.bias",
    "decoder.0.final_layer_norm.weight",
    "decoder.0.final_layer_norm.bias",
    "decoder.1.weight",
    "decoder.1.bias",
    "quantizer.quantizers.0.codebook.weight",
    "quantizer.quantizers.0.out_project.bias",
)

# One ConvNeXt block of the vocos backbone.
PER_BLOCK = (
    "dwconv.weight",
    "dwconv.bias",
    "norm.weight",
    "norm.bias",
    "pwconv1.weight",
    "pwconv1.bias",
    "pwconv2.weight",
    "pwconv2.bias",
    "gamma",
)

# The one weight stored as a direction and a magnitude, and what the graph calls the product.
FOLDED = "quantizer.quantizers.0.out_project.weight"

# The shapes that carry a decision. The codebook is the one to watch: 8192 entries of 8 numbers
# each, which is a very low-dimensional code and looks like a typo until you read the paper.
EXPECTED_SHAPES = {
    "quantizer.quantizers.0.codebook.weight": (8192, 8),
    "up.weight": (1024, 1024, 3),
    "decoder.0.embed.weight": (384, 1024, 7),
}


def wanted(blocks=CONVNEXT_BLOCKS):
    """Every name the graph loads, in the order it loads them."""
    names = list(TOP_LEVEL) + [FOLDED]
    for index in range(blocks):
        names.extend(f"decoder.0.convnext.{index}.{leaf}" for leaf in PER_BLOCK)

    return names


def checkpoint(path=None):
    """The released codec, from a file or from the hub."""
    if path is None:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(repo_id=REPO, filename=CHECKPOINT)

    state = torch.load(path, map_location="cpu", weights_only=True)

    return state["model"] if "model" in state else state


def fold_weight_norm(state, name):
    """`weight_g * weight_v / ||weight_v||`, normalizing over every axis but the first.

    `name` is what the folded weight is called, `<module>.weight`; `weight_norm` stores its two
    halves right beside it as `<module>.weight_g` and `<module>.weight_v`, so the names of those
    are this one with a suffix rather than a path of their own.
    """
    magnitude = state[f"{name}_g"].float()
    direction = state[f"{name}_v"].float()

    axes = tuple(range(1, direction.dim()))
    norm = direction.pow(2).sum(dim=axes, keepdim=True).sqrt()

    return magnitude * direction / norm


def select(state, blocks=CONVNEXT_BLOCKS):
    """The tensors the graph reads, and what was left behind."""
    names = wanted(blocks)

    # `FOLDED` is never in the checkpoint under that name -- it is the product of the two halves
    # beside it, so it is the halves that have to be there.
    missing = [name for name in names if name != FOLDED and name not in state]
    if f"{FOLDED}_g" not in state or f"{FOLDED}_v" not in state:
        missing.append(f"{FOLDED}_g and {FOLDED}_v")
    if missing:
        raise SystemExit(
            f"the checkpoint is missing {len(missing)} tensors the graph reads, "
            f"starting with {missing[0]} -- upstream has renamed something"
        )

    for name, shape in EXPECTED_SHAPES.items():
        got = tuple(state[name].shape)
        if got != shape:
            raise SystemExit(f"{name} is {got} in the checkpoint and {shape} in the graph")

    taken = {
        name: state[name].detach().float().contiguous()
        for name in names
        if name != FOLDED
    }
    taken[FOLDED] = fold_weight_norm(state, FOLDED).contiguous()

    # The two halves of the folded weight are spent, not left behind.
    consumed = set(taken) | {f"{FOLDED}_g", f"{FOLDED}_v"}
    left = sorted(set(state) - consumed)

    return taken, left


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-output", help="where to write the exported weights")
    parser.add_argument("-checkpoint", help="a codec.pth already on disk")
    arguments = parser.parse_args()

    state = checkpoint(arguments.checkpoint)
    taken, left = select(state)

    total = sum(value.numel() for value in taken.values())
    print(f"{len(taken)} tensors, {total / 1e6:.2f} M parameters")
    print(f"  {FOLDED} folded out of weight_g and weight_v, {tuple(taken[FOLDED].shape)}")

    dropped = sum(state[name].numel() for name in left)
    prefixes = sorted({name.split(".")[0] for name in left})
    print(
        f"{len(left)} tensors not exported, {dropped / 1e6:.2f} M parameters, "
        f"under: {', '.join(prefixes)} -- the encoder, which this release never calls"
    )

    if arguments.output:
        from safetensors.torch import save_file

        os.makedirs(os.path.dirname(arguments.output) or ".", exist_ok=True)
        save_file(taken, arguments.output)
        print("wrote", arguments.output)


if __name__ == "__main__":
    main()
