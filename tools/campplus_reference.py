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

"""The numbers `waifu/tests/campplus.rs` checks the speaker encoder against, from CAMPPlus itself.

Nothing here is a reimplementation: it downloads 3D-Speaker's own `DTDNN.py` and the layers
beside it, builds a small one out of them, and prints what it produces as Rust constants. What
the Rust test then compares is this runtime's answer against the reference implementation's,
which is the only comparison worth making -- a second implementation of the same architecture,
written from the same paper, would agree with a bug as readily as with the model.

Run it with the project venv, from the repository root:

    .venv/bin/python tools/campplus_reference.py

It needs the network once, to fetch the reference implementation. It never fetches the 6.9 M
parameter checkpoint: the weights here are made up, and what is being checked is the graph they
run through. Exporting the real ones is `tools/campplus_exporter.py`.

# The model is small, but it is not simplified

Every width is cut -- 16 filterbank bands instead of 80, 16 initial channels instead of 128, a
growth of 4 instead of 32, an 8-dimensional embedding instead of 192 -- and every *count* is the
release's. The three dense blocks are still twelve, twenty-four and sixteen layers deep, because
those counts are hard-coded in `CAMPPlus.__init__` and because depth is exactly what a dense
block's bookkeeping gets wrong: each layer concatenates onto the channels of all the ones before
it, so an off-by-one in that running width is invisible at depth one and fatal at depth twelve.

The input is 420 frames for one reason: the attention inside `CAMLayer` pools over segments a
hundred frames long, and the tdnn in front of it halves the time axis, so anything shorter than
about two hundred frames leaves a single segment and the whole segment-pooling path untested. At
420 the model sees three, the last of them a ten-frame remainder, which is the case that
exercises `ceil_mode`.

# Why the batch normalizations are set up the way they are

`tools/campplus_exporter.py` folds each one to a scale and a shift, because that is all a
normalization in `eval` mode is. The test fills those two vectors from their own names, like
every other parameter -- so this script arranges for the reference's normalizations to *be* that
scale and shift rather than folding them afterwards:

    running_var  = 1 - eps,  running_mean = 0,  weight = scale,  bias = shift

which makes `(x - 0) / sqrt(1 - eps + eps) * scale + shift` exactly `x * scale + shift`. The
module still runs; it is the real `BatchNorm1d`, doing the real arithmetic, and no folding
happens twice in two languages where the two could disagree.

The normalization in `DenseLayer` is `affine=False`, so it has no weight to be the scale. There
the statistics carry it instead -- `var = 1 / scale^2 - eps` and `mean = -shift / scale` -- which
comes to the same affine.
"""

import math
import sys

import torch

from campplus_exporter import upstream, BATCH_NORM_EPS

# The model the test builds: the release's shape with every width cut down.
FEAT_DIM = 16
EMBEDDING_SIZE = 8
GROWTH_RATE = 4
BN_SIZE = 2
INIT_CHANNELS = 16

# Long enough that `CAMLayer.seg_pooling` sees three segments, the last a partial one.
FRAMES = 420

WEIGHT_SCALE = 0.25
INPUT_SCALE = 0.8

# How many numbers of each probe to print. The whole embedding, and enough of the intermediates
# to say *where* a graph went wrong rather than only that it did.
PROBE = 8


def hash32(name: str) -> int:
    """FNV-1a over the parameter's name. The Rust test computes the same thing the same way."""
    value = 0x811C9DC5
    for byte in name.encode():
        value = ((value ^ byte) * 0x01000193) & 0xFFFFFFFF

    return value


def fill(name: str, count: int, scale: float) -> torch.Tensor:
    """`count` numbers that are this parameter's and no other's.

    A parameter filled from its own name rather than from a generator means the two sides need no
    file between them and cannot disagree about the order they walk the model in -- which is the
    one thing a shared seed would leave free, and the one that goes wrong silently.

    The arithmetic is float64 in both languages and rounded once, at the end, so the two fills are
    equal bit for bit rather than close.
    """
    seed = (hash32(name) % 10007) * 0.001
    values = [math.sin(i * 0.7371 + seed) * scale for i in range(count)]

    return torch.tensor(values, dtype=torch.float32)


def encoder(DTDNN):
    """A small CAMPPlus with every parameter filled from its name."""
    model = DTDNN.CAMPPlus(
        feat_dim=FEAT_DIM,
        embedding_size=EMBEDDING_SIZE,
        growth_rate=GROWTH_RATE,
        bn_size=BN_SIZE,
        init_channels=INIT_CHANNELS,
    )
    model.eval()

    with torch.no_grad():
        for name, module in model.named_modules():
            kind = type(module).__name__

            if kind in ("Conv1d", "Conv2d"):
                weight = module.weight
                weight.copy_(
                    fill(f"{name}.weight", weight.numel(), WEIGHT_SCALE).reshape(
                        weight.shape
                    )
                )
                if module.bias is not None:
                    module.bias.copy_(
                        fill(f"{name}.bias", module.bias.numel(), WEIGHT_SCALE)
                    )

            elif kind in ("BatchNorm1d", "BatchNorm2d"):
                channels = module.running_mean.numel()

                # A scale that cannot be zero: the affine=False case divides by it below, and a
                # normalization that annihilates a channel would hide whatever feeds it.
                scale = fill(f"{name}.scale", channels, 0.5).abs() + 0.5
                shift = fill(f"{name}.shift", channels, 0.3)

                if module.weight is not None:
                    module.running_var.copy_(torch.full((channels,), 1.0 - BATCH_NORM_EPS))
                    module.running_mean.copy_(torch.zeros(channels))
                    module.weight.copy_(scale)
                    module.bias.copy_(shift)
                else:
                    # No affine to carry it, so the statistics do: the same x * scale + shift.
                    module.running_var.copy_(1.0 / (scale * scale) - BATCH_NORM_EPS)
                    module.running_mean.copy_(-shift / scale)

    return model


def probes(model, x):
    """The embedding, and three points on the way to it."""
    out = {}

    with torch.no_grad():
        # `CAMPPlus.forward` permutes and then runs `head`; the same two steps, kept apart so the
        # front end can be checked on its own.
        permuted = x.permute(0, 2, 1)
        head = model.head(permuted)
        out["head"] = head

        running = head
        for name, module in model.xvector.named_children():
            running = module(running)
            if name in ("tdnn", "block1"):
                out[name] = running

        out["embedding"] = running

    return out


def emit(name, tensor, limit):
    """One probe as a Rust constant: its shape, and `limit` numbers spread through it.

    Spread rather than the first few. These tensors are (1, C, T) behind a ReLU, so the first
    handful of numbers are one channel at the start of the signal and are quite often all zero --
    a probe that would pass against almost anything. Sampling at an odd stride walks channels and
    time together and lands somewhere with something in it.
    """
    flat = tensor.reshape(-1).tolist()

    if limit >= len(flat):
        shown, where = flat, "all"
    else:
        step = max(1, len(flat) // limit)
        indices = [(i * step + i * 7) % len(flat) for i in range(limit)]
        shown, where = [flat[i] for i in indices], f"at {indices}"

    print(f"/// `{name}`, {tuple(tensor.shape)} -- {len(shown)} of {len(flat)}, {where}.")
    print(f"const {name.upper()}: [f32; {len(shown)}] = [")
    for index in range(0, len(shown), 4):
        row = ", ".join(f"{value:.6e}" for value in shown[index : index + 4])
        print(f"    {row},")
    print("];")
    print()


def main():
    model = encoder(upstream())

    # The input a speaker encoder is handed is filterbank energies, which are mean-normalized over
    # time before they reach it -- `infer_v2_5.py` does `feat - feat.mean(dim=0, keepdim=True)`.
    # So the input here is centred the same way rather than left as raw fill.
    raw = fill("input", FRAMES * FEAT_DIM, INPUT_SCALE).reshape(1, FRAMES, FEAT_DIM)
    x = raw - raw.mean(dim=1, keepdim=True)

    out = probes(model, x)

    print(f"// Generated by tools/campplus_reference.py -- {FRAMES} frames, {FEAT_DIM} bands.")
    print()

    # The centring, as the numbers themselves rather than as an instruction to recompute it.
    # Both sides fill the input bit for bit from its name; a mean summed over 420 frames in two
    # languages would not agree to the last bit, and a fifty layer model is no place to find out
    # how much that matters.
    emit("input_mean", raw.mean(dim=1), FEAT_DIM)

    for name in ("head", "tdnn", "block1", "embedding"):
        tensor = out[name]
        emit(name, tensor, tensor.numel() if name == "embedding" else PROBE)

    shapes = {name: tuple(value.shape) for name, value in out.items()}
    print("// shapes:", shapes, file=sys.stderr)


if __name__ == "__main__":
    main()
