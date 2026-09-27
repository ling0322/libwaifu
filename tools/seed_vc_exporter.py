#!/usr/bin/env python3
# Copyright (c) 2026 Xiaoyang Chen
#
# Part of libwaifu's port of Seed-VC (https://github.com/Plachtaa/seed-vc), which is licensed
# under the GNU General Public License version 3 -- and so is this file, unlike the rest of
# libwaifu, which is MIT. It exists only to build and check the package the `gpl` feature runs.
# See LICENSE-GPL-3.0 at the top of the repository.
#
# This program is free software: you can redistribute it and/or modify it under the terms of the
# GNU General Public License, version 3, as published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
# without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See
# the GNU General Public License for more details.
#
# SPDX-License-Identifier: GPL-3.0-only

"""Seed-VC v2 as one package: the release in, a manifest and its weights out.

    PYTHONPATH=~/.cache/libwaifu/torchaudio-stub \\
        .venv/bin/python tools/seed_vc_exporter.py -output models/seed_vc.safetensors

Reads the checkpoints `Plachtaa/seed-vc`'s `VoiceConversionWrapper.load_checkpoints` reads, from
the HuggingFace cache, and writes the eight models `waifu::seed_vc::SeedVc::from_manifest` runs,
each under a namespace of its own:

| namespace | from | what it is |
| --- | --- | --- |
| `seed_vc.hubert` | `facebook/hubert-large-ll60k` | the first 18 layers; weight norm folded |
| `seed_vc.wide` | `Plachta/ASTRAL-quantization` `bsq2048_light.pth` | ConvNeXt and BSQ's projection |
| `seed_vc.narrow` | the same, `bsq32_light.pth` | |
| `seed_vc.cfm` | `Plachta/Seed-VC` `v2/cfm_small.pth` | the DiT, and its rotary table |
| `seed_vc.cfm_regulator` | the same | token embedding, four conv stages |
| `seed_vc.ar` | `v2/ar_base.pth` | the 12-layer transformer, and its rotary table |
| `seed_vc.ar_regulator` | the same | the narrow tokens' embedding |
| `seed_vc.campplus` | `funasr/campplus` | by `campplus_exporter.py`'s own functions |
| `seed_vc.bigvgan` | `nvidia/bigvgan_v2_22khz_80band_256x` | by `bigvgan_exporter.py`'s own functions |

# What is left out

Everything inference never reads, which the checkpoints carry for training: HuBERT's layers 18
to 23 and its final norm (upstream cuts them off at load), ASTRAL's decoder and ASR head (not in
the `_light` files at all), the DiT's `x_embedder`, `skip_linear` and `content_mask_embedder`,
the AR's second codebook head and `style_in`, both regulators' `mask_token`, and the causal masks.

# The rotary tables are written, not rebuilt

Both transformers precompute `(cos, sin)` in float32 and store it as **bfloat16**; the DiT's is
a buffer in the checkpoint, the AR's is rebuilt at construction by the same function. The
rounding moves an angle's cosine by up to 2e-3, which is what the weights were trained under, so
the tables are written here as they were held -- the DiT's out of the checkpoint, the AR's by
upstream's own `precompute_freqs_cis` -- widened to float32, as `rotary_cos` and `rotary_sin`,
`(positions, head_dim / 2)` each.

# Nothing is narrowed

Every tensor is written at float32.
"""

import argparse
import os
import sys
from os import path

import torch

sys.path.insert(0, path.dirname(path.abspath(__file__)))

from model_exporter import Context, open_weights, parse_size, stem_of  # noqa: E402

MODEL_TYPE = "seed_vc"

# `inference_v2.py`'s defaults, and the wrapper's own limits. The Rust side has the same numbers
# as its defaults; these are here so a package says what it was exported as.
CONFIG = {
    "version": "2",
    "sample_rate": 22050,
    "diffusion_steps": 30,
    "intelligibility_cfg_rate": 0.7,
    "similarity_cfg_rate": 0.7,
    "top_p": 0.9,
    "temperature": 1.0,
    "repetition_penalty": 1.0,
    "reference_seconds": 25,
    "context_seconds": 30,
    "ar_max_content_len": 1500,
}

# How many tensors each namespace has to come out with; a count that moves is an export that
# changed shape.
COUNTS = {
    "hubert": 322,
    "wide": 124,
    "narrow": 124,
    "cfm": 136,
    "cfm_regulator": 17,
    "ar": 90,
    "ar_regulator": 1,
    "campplus": 573,
    "bigvgan": 667,
}

HUBERT_LAYERS = 18


def cached(repo, filename):
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo, filename)


def hubert_tensors():
    """`HubertModel` as transformers loads it, cut to the layers ASTRAL reads.

    The positional convolution is weight-normed; its `weight` is read off the module, which is
    `g * v / |v|` computed the way torch computes it, rather than refolded here.
    """
    from transformers import HubertModel

    model = HubertModel.from_pretrained("facebook/hubert-large-ll60k").eval()
    conv = model.encoder.pos_conv_embed.conv

    out = {}
    for key, tensor in model.state_dict().items():
        if key == "masked_spec_embed" or key.startswith("encoder.layer_norm."):
            continue
        if "pos_conv_embed.conv." in key and not key.endswith(".bias"):
            continue
        if key.startswith("encoder.layers."):
            if int(key.split(".")[2]) >= HUBERT_LAYERS:
                continue
        out[key] = tensor
    out["encoder.pos_conv_embed.conv.weight"] = conv.weight.detach().clone()
    return out


def astral_tensors(filename):
    state = torch.load(cached("Plachta/ASTRAL-quantization", filename), map_location="cpu",
                       weights_only=True)
    return {key: tensor for key, tensor in state.items()
            if key.startswith("encoder.") or key.startswith("quantizer.project_in.")}


def strip(state, prefix="module."):
    return {key.removeprefix(prefix): tensor for key, tensor in state.items()}


def rotary(table):
    """`(positions, pairs, 2)` bfloat16 as two float32 tables."""
    table = table.float()
    return {"rotary_cos": table[..., 0].contiguous(), "rotary_sin": table[..., 1].contiguous()}


def cfm_tensors():
    net = torch.load(cached("Plachta/Seed-VC", "v2/cfm_small.pth"), map_location="cpu",
                     weights_only=True)["net"]
    estimator = {key.removeprefix("estimator."): tensor
                 for key, tensor in strip(net["cfm"]).items()}

    unused = ("x_embedder.", "skip_linear.", "content_mask_embedder.", "transformer.causal_mask",
              "transformer.freqs_cis")
    out = {key: tensor for key, tensor in estimator.items() if not key.startswith(unused)}
    out.update(rotary(estimator["transformer.freqs_cis"]))

    regulator = {key: tensor for key, tensor in strip(net["length_regulator"]).items()
                 if key != "mask_token"}
    return out, regulator


def ar_tensors():
    from seed_vc_reference import importable

    importable()
    from modules.v2.ar import precompute_freqs_cis

    net = torch.load(cached("Plachta/Seed-VC", "v2/ar_base.pth"), map_location="cpu",
                     weights_only=True)["net"]
    unused = ("model.codebook_", "style_in.")
    out = {key: tensor for key, tensor in strip(net["ar"]).items() if not key.startswith(unused)}

    # `BaseModelArgs.max_seq_len` and `dim // n_head`, as the model builds its buffer.
    out.update(rotary(precompute_freqs_cis(4096, 768 // 12, 10000.0)))

    regulator = {key: tensor for key, tensor in strip(net["length_regulator"]).items()
                 if key != "mask_token"}
    return out, regulator


def campplus_tensors():
    import campplus_exporter

    return campplus_exporter.parameters(campplus_exporter.build(campplus_exporter.upstream()))


def bigvgan_tensors():
    import bigvgan_exporter

    model, _ = bigvgan_exporter.vocoder()
    return {name: value.detach() for name, value in model.state_dict().items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-output", required=True, help="the package, e.g. models/seed_vc.safetensors")
    parser.add_argument("-part_size", default="2G", help="how large one file of weights may be")
    arguments = parser.parse_args()

    torch.set_grad_enabled(False)
    os.makedirs(path.dirname(path.abspath(arguments.output)), exist_ok=True)
    stem_of(arguments.output)

    cfm, cfm_regulator = cfm_tensors()
    ar, ar_regulator = ar_tensors()
    parts = [
        ("hubert", hubert_tensors),
        ("wide", lambda: astral_tensors("bsq2048/bsq2048_light.pth")),
        ("narrow", lambda: astral_tensors("bsq32/bsq32_light.pth")),
        ("cfm", lambda: cfm),
        ("cfm_regulator", lambda: cfm_regulator),
        ("ar", lambda: ar),
        ("ar_regulator", lambda: ar_regulator),
        ("campplus", campplus_tensors),
        ("bigvgan", bigvgan_tensors),
    ]

    writer = open_weights(arguments.output, parse_size(arguments.part_size))
    total = 0
    for namespace, read in parts:
        tensors = read()
        if len(tensors) != COUNTS[namespace]:
            raise SystemExit(f"{namespace} came out as {len(tensors)} tensors, not "
                             f"{COUNTS[namespace]} -- the release or an exporter has changed")

        for key in sorted(tensors):
            tensor = tensors[key].detach()
            if tensor.dtype not in (torch.float32, torch.int64):
                tensor = tensor.float()
            writer.write_tensor(Context(f"{MODEL_TYPE}.{namespace}.{key}"), tensor.contiguous(),
                                preserve_dtype=True)
            total += tensor.numel()
        print(f"  {MODEL_TYPE}.{namespace}: {len(tensors)} tensors")

    config = {"model": {"type": MODEL_TYPE}, MODEL_TYPE: CONFIG}
    for name in writer.finish(config):
        print(f"wrote {name}")
    print(f"{total / 1e6:.1f} M parameters")


if __name__ == "__main__":
    main()
