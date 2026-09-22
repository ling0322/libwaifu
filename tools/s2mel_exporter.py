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

"""S2Mel as safetensors: `s2mel.pth` in, what `waifu::s2mel` reads out.

    .venv/bin/python tools/s2mel_exporter.py -output models/indextts25-s2mel.safetensors

The middle of IndexTTS-2.5: semantic tokens in, the mel spectrogram a BigVGAN makes audio of.

# The checkpoint is nested, and the names are flattened here

`s2mel.pth` is not a state dict. It is `{"net": {"cfm": {...}, "length_regulator": {...},
"gpt_layer": {...}}}` -- three modules each holding a state dict of its own. This flattens the
first two into one namespace, which is the only renaming it does:

    net.cfm[estimator.transformer.layers.0.attention.wqkv.weight]
        -> transformer.layers.0.attention.wqkv.weight
    net.length_regulator[content_in_proj.weight]
        -> length_regulator.content_in_proj.weight

The `cfm.estimator.` prefix comes off because `waifu::s2mel::graph` *is* the estimator -- there is
no flow-matching wrapper on this side, `solve_euler` is the caller's loop -- and carrying a prefix
that names a Python class nobody here has would be worse than dropping it. `length_regulator`
keeps its prefix because `waifu::s2mel::length_regulator` is built into a subgraph of that name.

# `gpt_layer` is not exported

Three linear layers, 1280 wide down to 128, that `infer_v2_5.py` never calls: the two things it
asks of S2Mel are `models['length_regulator']` and `models['cfm'].inference`, and neither reaches
them. Dead at inference the way the GPT's `text_head` is, and left behind for the same reason.

# Eighteen weights are folded out of weight normalization

Every convolution in the WaveNet head, and the final linear, is wrapped in
`torch.nn.utils.weight_norm`, so the checkpoint stores a direction `weight_v` and a magnitude
`weight_g` rather than a weight. Inference wants the product:

    weight = weight_g * weight_v / ||weight_v||

normalizing over every axis but the first. Seventeen convolutions -- eight `in_layers`, eight
`res_skip_layers` and one `cond_layer` -- plus `final_layer.linear`, which is a `Linear` and folds
by the same rule with one fewer axis.

Nothing else changes. Names, shapes and values are otherwise the checkpoint's.

# What the skip connections cost, and why only six layers have one

`uvit_skip_connection` wires the back half of the transformer to the front half, so
`skip_in_linear` exists on layers 7 through 12 and not on 0 through 6. The checkpoint holds one
for every layer anyway; the seven that the graph never reads are left behind, which is why the
unexported list is longer than `gpt_layer` alone.

# Where the checkpoint comes from

`IndexTeam/IndexTTS-2.5` on Hugging Face, which is public and ungated. Pass `-checkpoint` to read
one already on disk.
"""

import argparse
import os

import torch

REPO = "IndexTeam/IndexTTS-2.5"
CHECKPOINT = "s2mel.pth"

# `config.yaml`'s `s2mel.DiT.depth`, and the layer after which a skip connection appears.
DEPTH = 13
FIRST_SKIP_LAYER = DEPTH // 2 + 1

# `config.yaml`'s `s2mel.wavenet.num_layers`.
WAVENET_LAYERS = 8

# Everything the estimator loads that is not inside a transformer layer or the WaveNet.
TOP_LEVEL = (
    "t_embedder.mlp.0.weight",
    "t_embedder.mlp.0.bias",
    "t_embedder.mlp.2.weight",
    "t_embedder.mlp.2.bias",
    "t_embedder2.mlp.0.weight",
    "t_embedder2.mlp.0.bias",
    "t_embedder2.mlp.2.weight",
    "t_embedder2.mlp.2.bias",
    "cond_projection.weight",
    "cond_projection.bias",
    "cond_x_merge_linear.weight",
    "cond_x_merge_linear.bias",
    "res_projection.weight",
    "res_projection.bias",
    "skip_linear.weight",
    "skip_linear.bias",
    "conv1.weight",
    "conv1.bias",
    "conv2.weight",
    "conv2.bias",
    "transformer.norm.norm.weight",
    "transformer.norm.project_layer.weight",
    "transformer.norm.project_layer.bias",
    "final_layer.adaLN_modulation.1.weight",
    "final_layer.adaLN_modulation.1.bias",
    "final_layer.linear.bias",
)

# One transformer layer. `wqkv` is fused and `wo`, `w1`, `w2`, `w3` carry no bias.
PER_LAYER = (
    "attention_norm.project_layer.weight",
    "attention_norm.project_layer.bias",
    "attention_norm.norm.weight",
    "attention.wqkv.weight",
    "attention.wo.weight",
    "ffn_norm.project_layer.weight",
    "ffn_norm.project_layer.bias",
    "ffn_norm.norm.weight",
    "feed_forward.w1.weight",
    "feed_forward.w3.weight",
    "feed_forward.w2.weight",
)

# The length regulator, which keeps its own prefix.
REGULATOR = (
    "length_regulator.content_in_proj.weight",
    "length_regulator.content_in_proj.bias",
) + tuple(
    f"length_regulator.model.{index}.{leaf}"
    for index in (0, 1, 3, 4, 6, 7, 9, 10, 12)
    for leaf in ("weight", "bias")
)

# The shapes that carry a decision. `cond_x_merge_linear` is the one to watch: 864 columns is
# 80 mel bands twice plus a 512-wide content vector plus... no, it is 80 + 80 + 512 + 192, the
# style vector included, and a reader who assumes 80 + 512 gets a model that loads.
EXPECTED_SHAPES = {
    "cond_x_merge_linear.weight": (512, 864),
    "transformer.layers.0.attention.wqkv.weight": (1536, 512),
    "length_regulator.content_in_proj.weight": (512, 1024),
}


def folded_names():
    """The eighteen weights the checkpoint stores as a direction and a magnitude."""
    names = ["final_layer.linear.weight", "wavenet.cond_layer.conv.conv.weight"]
    for index in range(WAVENET_LAYERS):
        names.append(f"wavenet.in_layers.{index}.conv.conv.weight")
        names.append(f"wavenet.res_skip_layers.{index}.conv.conv.weight")

    return names


def wanted():
    """Every name the graph loads, in the order it loads them."""
    names = list(TOP_LEVEL)

    for index in range(DEPTH):
        if index >= FIRST_SKIP_LAYER:
            names.append(f"transformer.layers.{index}.skip_in_linear.weight")
            names.append(f"transformer.layers.{index}.skip_in_linear.bias")
        names.extend(f"transformer.layers.{index}.{leaf}" for leaf in PER_LAYER)

    names.append("wavenet.cond_layer.conv.conv.bias")
    for index in range(WAVENET_LAYERS):
        names.append(f"wavenet.in_layers.{index}.conv.conv.bias")
        names.append(f"wavenet.res_skip_layers.{index}.conv.conv.bias")

    names.extend(folded_names())
    names.extend(REGULATOR)

    return names


def checkpoint(path=None):
    """The released S2Mel, flattened out of its three modules into one namespace."""
    if path is None:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(repo_id=REPO, filename=CHECKPOINT)

    state = torch.load(path, map_location="cpu", weights_only=True)
    net = state["net"] if "net" in state else state

    flat = {}
    for module, inner in net.items():
        for name, value in inner.items():
            if module == "cfm":
                # `cfm.estimator.x` is the estimator, which is what `waifu::s2mel::graph` is.
                flat[name[len("estimator.") :] if name.startswith("estimator.") else name] = value
            else:
                flat[f"{module}.{name}"] = value

    return flat


def fold_weight_norm(state, name):
    """`weight_g * weight_v / ||weight_v||`, normalizing over every axis but the first.

    `name` is what the folded weight is called, `<module>.weight`; `weight_norm` stores its two
    halves right beside it as `<module>.weight_g` and `<module>.weight_v`.
    """
    magnitude = state[f"{name}_g"].float()
    direction = state[f"{name}_v"].float()

    axes = tuple(range(1, direction.dim()))
    norm = direction.pow(2).sum(dim=axes, keepdim=True).sqrt()

    return magnitude * direction / norm


def select(state):
    """The tensors the graph reads, and what was left behind."""
    names = wanted()
    folded = set(folded_names())

    missing = [name for name in names if name not in folded and name not in state]
    missing += [
        name
        for name in folded
        if f"{name}_g" not in state or f"{name}_v" not in state
    ]
    if missing:
        raise SystemExit(
            f"the checkpoint is missing {len(missing)} tensors the graph reads, "
            f"starting with {sorted(missing)[0]} -- upstream has renamed something"
        )

    for name, shape in EXPECTED_SHAPES.items():
        got = tuple(state[name].shape)
        if got != shape:
            raise SystemExit(f"{name} is {got} in the checkpoint and {shape} in the graph")

    taken = {}
    for name in names:
        if name in folded:
            taken[name] = fold_weight_norm(state, name).contiguous()
        else:
            taken[name] = state[name].detach().float().contiguous()

    consumed = set(taken)
    for name in folded:
        consumed |= {f"{name}_g", f"{name}_v"}
    left = sorted(set(state) - consumed)

    return taken, left


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-output", help="where to write the exported weights")
    parser.add_argument("-checkpoint", help="an s2mel.pth already on disk")
    arguments = parser.parse_args()

    state = checkpoint(arguments.checkpoint)
    taken, left = select(state)

    total = sum(value.numel() for value in taken.values())
    print(f"{len(taken)} tensors, {total / 1e6:.2f} M parameters")
    print(f"  {len(folded_names())} folded out of weight_g and weight_v")

    dropped = sum(state[name].numel() for name in left)
    prefixes = sorted({name.split(".")[0] for name in left})
    print(
        f"{len(left)} tensors not exported, {dropped / 1e6:.2f} M parameters, "
        f"under: {', '.join(prefixes)}"
    )

    if arguments.output:
        from safetensors.torch import save_file

        os.makedirs(os.path.dirname(arguments.output) or ".", exist_ok=True)
        save_file(taken, arguments.output)
        print("wrote", arguments.output)


if __name__ == "__main__":
    main()
