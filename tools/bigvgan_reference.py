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

"""The numbers `waifu/tests/bigvgan.rs` checks the vocoder against, from BigVGAN itself.

Nothing here is a reimplementation: it downloads NVIDIA's own `bigvgan.py` and the alias-free
activation beside it, builds a small one out of them, and prints what it produces as Rust
constants. What the Rust test then compares is this runtime's answer against the reference
implementation's, which is the only comparison worth making -- a second implementation of the
same architecture, written from the same paper, would agree with a bug as readily as with the
model.

The model is small (8 mel bands, two upsamplings by two, 24 samples out) rather than the 112 M
parameter release, because the point is the arithmetic and not the weights. Every parameter is
filled from its own name -- see `fill` -- so the Rust side can build the same model without a
file passing between them, and the test stays in the fast suite rather than needing a package.

Run it with the project venv, from the repository root:

    .venv/bin/python tools/bigvgan_reference.py

It needs the network once, to fetch the reference implementation into the HuggingFace cache.
"""

import contextlib
import math
import sys

import numpy

import torch

from bigvgan_exporter import upstream

# Only the reference's source is fetched here, never the 460 MB checkpoint: the weights this uses
# are made up, and what is being checked is the graph they run through. Exporting the real ones is
# `tools/bigvgan_exporter.py`, which is also where the download lives.

# The model the test builds, which is the released one's shape with every count cut down.
CONFIG = {
    "num_mels": 8,
    "upsample_rates": [2, 2],
    "upsample_kernel_sizes": [4, 4],
    "upsample_initial_channel": 16,
    "resblock": "1",
    "resblock_kernel_sizes": [3, 5],
    "resblock_dilation_sizes": [[1, 3], [1, 3]],
    "activation": "snakebeta",
    "snake_logscale": True,
    "use_tanh_at_final": False,
    "use_bias_at_final": False,
}

FRAMES = 6

# Small enough that the output stays inside the clamp the last layer applies: at 0.25 the loudest
# sample is 0.75 and none is against the bound. A weight the size a trained one has would saturate
# every sample to +-1, and the comparison would then pass on a model that was wrong everywhere.
WEIGHT_SCALE = 0.25

# And one that does not fit, for the two bounded endings. At 0.35 most of the output is outside
# [-1, 1] before the last layer, so this is what says the clamp and the tanh are there.
LOUD_SCALE = 0.35


def imported():
    """NVIDIA's BigVGAN, and the `AttrDict` its constructor takes a configuration as."""
    bigvgan = upstream()

    from env import AttrDict

    return bigvgan, AttrDict


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


def vocoder(bigvgan, AttrDict, scale: float, tanh: bool):
    """The model, with weight normalization folded away and every parameter filled from its name.

    `remove_weight_norm` is what `IndexTTS2.__init__` calls on the vocoder it loads, so a graph
    that reads a plain weight is reading what the reference runs -- see `tools/bigvgan_exporter.py`,
    which folds the same two tensors into one at export.
    """
    config = dict(CONFIG, use_tanh_at_final=tanh)

    with contextlib.redirect_stdout(sys.stderr):
        model = bigvgan.BigVGAN(AttrDict(config))
        model.remove_weight_norm()

    model.eval()

    with torch.no_grad():
        for name, parameter in model.named_parameters():
            # The snake's alpha and beta are exponentiated where they are read, so these are
            # logarithms: this scale leaves both within a few percent of one.
            parameter.copy_(fill(name, parameter.numel(), scale).view(parameter.shape))

    return model


def literal(value) -> str:
    """The shortest text that reads back as this float32 and no other.

    Shortest matters for more than tidiness: a longer decimal is a number float32 cannot hold, so
    `clippy::excessive_precision` warns on every one of them, and three hundred warnings in a test
    file is a thing people learn to scroll past.
    """
    text = numpy.format_float_scientific(numpy.float32(value), unique=True, trim="0")

    # numpy writes the exponent with a sign and the mantissa with a point, both of which Rust
    # takes; what it does not write is a mantissa without a point, which `trim="0"` prevents.
    return text


def rust(name: str, values) -> None:
    """One `const` for the test to read, in the precision it was computed at."""
    flat = [float(v) for v in values]
    body = ",\n    ".join(
        ", ".join(literal(v) for v in flat[i : i + 4]) for i in range(0, len(flat), 4)
    )

    print(f"const {name}: [f32; {len(flat)}] = [\n    {body},\n];\n")


def parameters(name: str, model) -> None:
    """Every weight the checkpoint holds, as the Rust test's list of what to make up.

    This is the half of the comparison that the numbers do not cover: a model that computed the
    right waveform out of weights it had invented names for would still be one no checkpoint can
    be loaded into. The test builds its weights from this list and from nothing else, so a graph
    asking for a name that is not here fails to build.
    """
    rows = [
        f'    ("{key}", &{list(value.shape)}),'.replace("[", "[").replace("]", "]")
        for key, value in model.named_parameters()
    ]

    print(f"const {name}: [(&str, &[i32]); {len(rows)}] = [")
    print("\n".join(rows))
    print("];\n")


def main() -> int:
    torch.set_grad_enabled(False)

    bigvgan, AttrDict = imported()
    from alias_free_activation.torch.act import Activation1d
    from alias_free_activation.torch.filter import kaiser_sinc_filter1d
    import activations

    # The filter every anti-aliased activation is built on, and the one thing in this model that
    # is computed rather than trained: 12 taps of a Kaiser-windowed sinc at half the rate.
    print("// Generated by tools/bigvgan_reference.py. Do not edit by hand.\n")
    rust("KAISER_12", kaiser_sinc_filter1d(0.5 / 2, 0.6 / 2, 12).flatten())

    # One channel of a signal short enough to print, through each half of an anti-aliased
    # activation on its own, so that a disagreement says which half.
    signal = fill("signal", 2 * 3 * 9, 1.0).view(2, 3, 9)
    rust("SIGNAL", signal.flatten())

    from alias_free_activation.torch.resample import DownSample1d, UpSample1d

    rust("UPSAMPLED", UpSample1d(2, 12)(signal).flatten())
    rust("DOWNSAMPLED", DownSample1d(2, 12)(signal).flatten())

    snake = activations.SnakeBeta(3, alpha_logscale=True)
    snake.alpha.copy_(fill("act.alpha", 3, WEIGHT_SCALE))
    snake.beta.copy_(fill("act.beta", 3, WEIGHT_SCALE))
    rust("ACTIVATED", Activation1d(activation=snake)(signal).flatten())

    # And the whole vocoder: quiet enough to stay off the bound, then loud enough to sit on it,
    # under each of the two endings a BigVGAN can be built with.
    mel = fill("mel", CONFIG["num_mels"] * FRAMES, 1.0).view(1, CONFIG["num_mels"], FRAMES)

    for name, scale, tanh in (
        ("WAVEFORM", WEIGHT_SCALE, False),
        ("WAVEFORM_CLAMPED", LOUD_SCALE, False),
        ("WAVEFORM_TANH", LOUD_SCALE, True),
    ):
        model = vocoder(bigvgan, AttrDict, scale, tanh)
        waveform = model(mel)

        if name == "WAVEFORM":
            parameters("PARAMETERS", model)

        rust(name, waveform.flatten())

        # A saturated output would compare equal against almost any model, so say it out loud.
        against = int((waveform.abs() >= 0.999).sum())
        print(f"// {name}: peak {waveform.abs().max().item():.4f}, {against} samples on the bound")
        print(f"// {sum(p.numel() for p in model.parameters())} parameters\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
