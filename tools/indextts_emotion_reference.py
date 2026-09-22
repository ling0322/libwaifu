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

"""The numbers `waifu/tests/indextts_emotion.rs` checks the emotion path against.

    .venv/bin/python tools/indextts_emotion_reference.py

# The reference is upstream's own code, not a transcription of it

`ConformerEncoder` and `PerceiverResampler` are imported from the released source and run. This
matters more here than usual: a WeNet conformer has four or five places where a reimplementation
can be self-consistently wrong -- whether `rel_shift` is applied, which half a GLU gates, whether
the feed forward is halved for macaron style, whether the position table is added to the input or
only to the scores -- and every one of them still produces a tensor of the right shape. Comparing
against a second reading of the paper would find none of them. See the note in CLAUDE.md.

The source is the tree already cached under `~/.cache/libwaifu/indextts-src`, at the revision the
GPT was read from. `torchaudio` is stubbed because `indextts.utils.common` imports it at module
scope and nothing reached from here calls it -- the alternative is a 100 MB wheel for an import
statement.

# Every width is cut, and the weights scatter

Four blocks become two, 1024 features become 24, and the perceiver's latents narrow from 1024 to
20. A conformer block has no bookkeeping that turns on its width or on how many blocks precede
it, so what is left exercises the same arithmetic.

The weights are drawn by xorshift from each parameter's own name, in both languages, and *not* by
sweeping a sine the way the other reference scripts here do. The reason is written up in
`docs/indextts_gpt.md`: a swept parameter makes every key in a projection point nearly the same
way, the softmax over them comes out flat, and a flat softmax is an average -- which does not care
what order its keys arrived in. Attention is most of what this module is, so it is drawn scattered
or it is not being tested.

The name each draw is keyed on is the *whole* name, `emo_conditioning_encoder.` and all, because
that is what the Rust graph's parameter source is handed.
"""

import math
import os
import sys
import types

import torch

UPSTREAM = os.path.expanduser("~/.cache/libwaifu/indextts-src")

sys.path.insert(0, UPSTREAM)
sys.modules.setdefault("torchaudio", types.ModuleType("torchaudio"))

from indextts.gpt.conformer_encoder import ConformerEncoder  # noqa: E402
from indextts.gpt.perceiver import PerceiverResampler  # noqa: E402

# The cut widths. Every one of these appears in waifu/tests/indextts_emotion.rs as a constant.
INPUT_DIM = 24
ENCODER_DIM = 16
ENCODER_HEADS = 4
ENCODER_UNITS = 32
ENCODER_BLOCKS = 2
CNN_KERNEL = 7
LATENT_DIM = 20
LATENTS = 1
PERCEIVER_DEPTH = 2
PERCEIVER_HEADS = 4
PERCEIVER_HEAD_DIM = 8
PERCEIVER_MULT = 2.0
MODEL_DIM = 24
FRAMES = 21

WEIGHT_SCALE = 0.7
INPUT_SCALE = 1.0

ENCODER_PREFIX = "emo_conditioning_encoder."
PERCEIVER_PREFIX = "emo_perceiver_encoder."


def hash32(name):
    """FNV-1a, the same one waifu/tests/indextts_emotion.rs uses."""
    value = 0x811C9DC5
    for byte in name.encode():
        value = ((value ^ byte) * 0x01000193) & 0xFFFFFFFF
    return value


def scatter(name, count, scale):
    """`count` numbers in [-scale, scale], drawn by xorshift from `name`."""
    state = hash32(name) | 1
    out = []
    for _ in range(count):
        state = (state ^ (state << 13)) & 0xFFFFFFFF
        state = state ^ (state >> 17)
        state = (state ^ (state << 5)) & 0xFFFFFFFF
        out.append((state / 0xFFFFFFFF * 2.0 - 1.0) * scale)
    return out


def fill(module, prefix):
    """Give every parameter the numbers its full name draws."""
    with torch.no_grad():
        for name, parameter in module.named_parameters():
            values = scatter(prefix + name, parameter.numel(), WEIGHT_SCALE)
            parameter.copy_(torch.tensor(values, dtype=torch.float32).view(parameter.shape))


def probe_indices(length, limit=8):
    """The same handful of positions the Rust test samples."""
    step = max(length // limit, 1)
    return [(i * step + i * 7) % length for i in range(limit)]


def emit(name, tensor, comment):
    values = tensor.detach().flatten().tolist()
    indices = probe_indices(len(values))

    print(f"/// `{name}`, {tuple(tensor.shape)} -- {comment}")
    print(f"///   8 of {len(values)}, at {indices}.")
    print(f"const {name.upper()}: [f32; 8] = [")
    for index in indices:
        print(f"    {values[index]:.6e},")
    print("];")
    print()


def main():
    torch.manual_seed(0)

    encoder = ConformerEncoder(
        input_size=INPUT_DIM,
        output_size=ENCODER_DIM,
        attention_heads=ENCODER_HEADS,
        linear_units=ENCODER_UNITS,
        num_blocks=ENCODER_BLOCKS,
        dropout_rate=0.0,
        input_layer="conv2d2",
        cnn_module_kernel=CNN_KERNEL,
    ).eval()

    perceiver = PerceiverResampler(
        LATENT_DIM,
        dim_context=ENCODER_DIM,
        ff_mult=PERCEIVER_MULT,
        heads=PERCEIVER_HEADS,
        dim_head=PERCEIVER_HEAD_DIM,
        num_latents=LATENTS,
    ).eval()

    emovec_layer = torch.nn.Linear(LATENT_DIM, MODEL_DIM).eval()
    emo_layer = torch.nn.Linear(MODEL_DIM, MODEL_DIM).eval()

    fill(encoder, ENCODER_PREFIX)
    fill(perceiver, PERCEIVER_PREFIX)
    fill(emovec_layer, "emovec_layer.")
    fill(emo_layer, "emo_layer.")

    features = torch.tensor(
        scatter("features", FRAMES * INPUT_DIM, INPUT_SCALE), dtype=torch.float32
    ).view(1, FRAMES, INPUT_DIM)
    lengths = torch.tensor([FRAMES])

    with torch.no_grad():
        # The embedding on its own: the subsampling convolution, the flattening projection and
        # the scale by sqrt(output_size) that RelPositionalEncoding applies. The position table
        # comes back beside it rather than added to it, which is the whole of what makes this
        # encoding relative.
        masks = torch.ones(1, 1, FRAMES, dtype=torch.bool)
        embedded, positions, _ = encoder.embed(features, masks)

        encoded, mask = encoder(features, lengths)

        # The perceiver's own mask is the frame mask with one True prepended for the latent.
        # All-true here, since one recording is never padded.
        conds_mask = torch.nn.ConstantPad1d((LATENTS, 0), True)(mask.squeeze(1))
        resampled = perceiver(encoded, conds_mask)

        emotion = emo_layer(emovec_layer(resampled.squeeze(1)))

    print(f"// Generated by tools/indextts_emotion_reference.py -- {FRAMES} frames,", end="")
    print(f" {ENCODER_BLOCKS} blocks.")
    print()
    emit("positions", positions, "the sinusoid table, which nothing learned")
    emit("embedded", embedded, "subsampled, flattened, projected and scaled")
    emit("encoded", encoded, "all of the conformer")
    emit("resampled", resampled, "one latent, having read the whole recording")
    emit("emotion", emotion, "the row the GPT is handed")

    print(f"// subsampled length: {encoded.shape[1]} of {FRAMES}")


if __name__ == "__main__":
    main()
