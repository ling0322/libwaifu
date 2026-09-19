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

"""The numbers `waifu/tests/s2mel.rs` checks the denoiser against, from S2Mel itself.

Nothing here is a reimplementation: it downloads IndexTTS's own `diffusion_transformer.py` and
the four files it leans on, builds a small one out of them, and prints what it produces as Rust
constants.

Run it with the project venv, from the repository root:

    .venv/bin/python tools/s2mel_reference.py

It needs the network once, to fetch the reference implementation, and it never fetches the 415 MB
checkpoint: the weights here are made up and what is being checked is the graph they run through.

# The model is small, but its counts are the release's

Thirteen transformer layers and eight WaveNet layers, as shipped; what is cut is every width --
16 mel bands rather than 80, a 64-wide transformer rather than 512, four heads rather than eight.

Depth is kept because of the skip connections. Half the transformer's layers hand a skip forward
to the other half, taken off the end of a list, so which layer receives which is a function of
the depth and of nothing else; at a depth of two there is nothing to get wrong, and at thirteen
there is an off-by-one in three different places.

# Weight normalization is folded before anything is filled

Several layers here are wrapped in `torch.nn.utils.weight_norm`, which stores a direction and a
magnitude and multiplies them on every call. That is a thing training wants; inference does not,
and the reference itself removes it when it loads a checkpoint. So this removes it first and then
fills the plain weight that is left, which is the same weight the runtime reads.
"""

import math
import os
import sys
import urllib.request

import torch

SOURCE_ROOT = "https://raw.githubusercontent.com/index-tts/index-tts/main/"
SOURCES = (
    "indextts/s2mel/modules/diffusion_transformer.py",
    "indextts/s2mel/modules/wavenet.py",
    "indextts/s2mel/modules/commons.py",
    "indextts/s2mel/modules/encodec.py",
    "indextts/s2mel/modules/gpt_fast/model.py",
    "indextts/s2mel/modules/length_regulator.py",
    "indextts/s2mel/dac/nn/quantize.py",
    "indextts/s2mel/dac/nn/layers.py",
)

# The release's counts, with every width cut.
IN_CHANNELS = 16
HIDDEN = 64
DEPTH = 13
HEADS = 4
CONTENT_DIM = 32
STYLE_DIM = 24
WAVENET_DIM = 64
WAVENET_LAYERS = 8
WAVENET_KERNEL = 5
FRAMES = 24

# The length regulator's widths, cut the same way. Four stages, as `config.yaml` has.
REGULATOR_IN = 40
REGULATOR_CHANNELS = 32
REGULATOR_STAGES = 4
TOKENS = 9

TIME = 0.3
WEIGHT_SCALE = 0.25
INPUT_SCALE = 0.6
PROBE = 8


def upstream(cache=None):
    """IndexTTS's DiT, downloaded and importable.

    The files import each other by absolute package name, so they are laid out as that package
    with an `__init__.py` at every level.
    """
    cache = cache or os.path.join(os.path.expanduser("~"), ".cache", "libwaifu", "s2mel-source")

    for name in SOURCES:
        target = os.path.join(cache, *name.split("/"))
        os.makedirs(os.path.dirname(target), exist_ok=True)
        if not os.path.isfile(target):
            urllib.request.urlretrieve(SOURCE_ROOT + name, target)

    for name in SOURCES:
        parts = name.split("/")[:-1]
        for depth in range(len(parts)):
            init = os.path.join(cache, *parts[: depth + 1], "__init__.py")
            if not os.path.isfile(init):
                open(init, "w").close()

    if cache not in sys.path:
        sys.path.insert(0, cache)

    from indextts.s2mel.modules import diffusion_transformer

    return diffusion_transformer


def hash32(name: str) -> int:
    """FNV-1a over the parameter's name. The Rust test computes the same thing the same way."""
    value = 0x811C9DC5
    for byte in name.encode():
        value = ((value ^ byte) * 0x01000193) & 0xFFFFFFFF

    return value


def fill(name: str, count: int, scale: float) -> torch.Tensor:
    """`count` numbers that are this parameter's and no other's."""
    seed = (hash32(name) % 10007) * 0.001
    values = [math.sin(i * 0.7371 + seed) * scale for i in range(count)]

    return torch.tensor(values, dtype=torch.float32)


def arguments():
    """The `args` the DiT reads its shape out of, in the shape `config.yaml` has."""
    from munch import Munch

    return Munch(
        DiT=Munch(
            hidden_dim=HIDDEN,
            num_heads=HEADS,
            depth=DEPTH,
            class_dropout_prob=0.1,
            block_size=8192,
            in_channels=IN_CHANNELS,
            style_condition=True,
            final_layer_type="wavenet",
            target="mel",
            content_dim=CONTENT_DIM,
            content_codebook_size=1024,
            content_type="discrete",
            f0_condition=False,
            n_f0_bins=512,
            content_codebooks=1,
            is_causal=False,
            long_skip_connection=True,
            zero_prompt_speech_token=False,
            time_as_token=False,
            style_as_token=False,
            uvit_skip_connection=True,
            add_resblock_in_transformer=False,
        ),
        wavenet=Munch(
            hidden_dim=WAVENET_DIM,
            num_layers=WAVENET_LAYERS,
            kernel_size=WAVENET_KERNEL,
            dilation_rate=1,
            p_dropout=0.2,
            style_condition=True,
        ),
        style_encoder=Munch(dim=STYLE_DIM),
    )


def denoiser(module):
    """A small DiT, weight normalization removed and every parameter filled from its name."""
    model = module.DiT(arguments())
    model.eval()

    # Off first, so that what gets filled is the weight the runtime will read.
    for child in model.modules():
        try:
            torch.nn.utils.remove_weight_norm(child)
        except (ValueError, RuntimeError, AttributeError):
            pass

    with torch.no_grad():
        for name, parameter in model.named_parameters():
            parameter.copy_(fill(name, parameter.numel(), WEIGHT_SCALE).reshape(parameter.shape))

    model.setup_caches(1, FRAMES)

    return model


def regulator():
    """A small InterpolateRegulator, every parameter filled from its name.

    `sampling_ratios` decides how many convolution stages there are and nothing else -- the resize
    is to whatever length is asked for, not to a ratio -- so four entries is four stages, which is
    what the release has.
    """
    from indextts.s2mel.modules.length_regulator import InterpolateRegulator

    model = InterpolateRegulator(
        channels=REGULATOR_CHANNELS,
        sampling_ratios=tuple([1] * REGULATOR_STAGES),
        is_discrete=False,
        in_channels=REGULATOR_IN,
        vector_quantize=False,
        codebook_size=1024,
        out_channels=REGULATOR_CHANNELS,
        groups=1,
        n_codebooks=1,
        quantizer_dropout=0.0,
        f0_condition=False,
    )
    model.eval()

    with torch.no_grad():
        for name, parameter in model.named_parameters():
            parameter.copy_(fill(name, parameter.numel(), WEIGHT_SCALE).reshape(parameter.shape))

    return model


def emit(name, tensor, limit):
    """One probe as a Rust constant: `limit` numbers spread through it."""
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
    model = denoiser(upstream())

    x = fill("x", IN_CHANNELS * FRAMES, INPUT_SCALE).reshape(1, IN_CHANNELS, FRAMES)
    prompt_x = fill("prompt_x", IN_CHANNELS * FRAMES, INPUT_SCALE).reshape(1, IN_CHANNELS, FRAMES)
    cond = fill("cond", FRAMES * CONTENT_DIM, INPUT_SCALE).reshape(1, FRAMES, CONTENT_DIM)
    style = fill("style", STYLE_DIM, INPUT_SCALE).reshape(1, STYLE_DIM)
    x_lens = torch.LongTensor([FRAMES])
    t = torch.tensor([TIME], dtype=torch.float32)

    # What the transformer handed on, so a failure can be placed on one side of it or the other.
    seen = {}
    model.transformer.register_forward_hook(
        lambda _module, _inputs, output: seen.__setitem__("transformer", output)
    )

    with torch.no_grad():
        out = model(x, prompt_x, x_lens, t, style, cond)

    # The length regulator, which is what produces the `cond` the denoiser reads. Its own model
    # and its own probe: it is a separate module in the reference too.
    stretch = regulator()
    tokens = fill("tokens", TOKENS * REGULATOR_IN, INPUT_SCALE).reshape(1, TOKENS, REGULATOR_IN)
    with torch.no_grad():
        stretched, _, _, _, _ = stretch(tokens, ylens=torch.LongTensor([FRAMES]))

    print(f"// Generated by tools/s2mel_reference.py -- {FRAMES} frames, {IN_CHANNELS} bands.")
    print()
    emit("regulated", stretched, PROBE)
    emit("transformer_out", seen["transformer"], PROBE)
    emit("denoised", out, PROBE)

    print("// shapes:", {"transformer": tuple(seen["transformer"].shape), "out": tuple(out.shape)},
          file=sys.stderr)


if __name__ == "__main__":
    main()
