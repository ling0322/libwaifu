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

"""The numbers `waifu/tests/w2v_bert.rs` checks the conformer against, from transformers itself.

No download at all, of source or of weights: `Wav2Vec2BertModel` is in the pinned transformers in
this venv, so the reference is imported rather than fetched. A small one is built, every
parameter is filled from its own name, and what it produces is printed as Rust constants.

    .venv/bin/python tools/w2v_bert_reference.py

# What is small and what is not

Every width is cut. The layer count is cut too -- four rather than twenty-four -- because a
conformer stack has no bookkeeping that depends on its depth: layer n reads layer n-1 and that is
all. What the test does keep is the *shape* of one layer, which is where the model is unusual:
two feed forwards added back at half weight, a causal depthwise convolution, and scores carrying
a term for how far apart two frames are.

Two hidden states are printed rather than one. IndexTTS reads `hidden_states[17]` out of a
twenty-four layer model -- the embedding plus sixteen layers -- so what matters is not only that
the stack is right but that stopping partway through lands where the reference says. The second
probe is that: the same model read after two layers rather than four.
"""

import math
import sys

import torch

FEATURE_DIM = 20
HIDDEN = 32
HEADS = 4
INTERMEDIATE = 64
CONV_KERNEL = 7
LEFT_POSITIONS = 6
RIGHT_POSITIONS = 2
LAYERS = 4
PARTIAL = 2
FRAMES = 14

WEIGHT_SCALE = 0.25
INPUT_SCALE = 0.6
PROBE = 8


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


def encoder():
    """A small w2v-bert, every parameter filled from its name and every dropout off."""
    from transformers import Wav2Vec2BertConfig, Wav2Vec2BertModel

    config = Wav2Vec2BertConfig(
        feature_projection_input_dim=FEATURE_DIM,
        hidden_size=HIDDEN,
        num_hidden_layers=LAYERS,
        num_attention_heads=HEADS,
        intermediate_size=INTERMEDIATE,
        conv_depthwise_kernel_size=CONV_KERNEL,
        position_embeddings_type="relative_key",
        left_max_position_embeddings=LEFT_POSITIONS,
        right_max_position_embeddings=RIGHT_POSITIONS,
        hidden_act="swish",
        add_adapter=False,
        use_intermediate_ffn_before_adapter=False,
        # Every dropout to zero as well as eval, so nothing depends on the mode being right.
        hidden_dropout=0.0,
        activation_dropout=0.0,
        attention_dropout=0.0,
        feat_proj_dropout=0.0,
        conformer_conv_dropout=0.0,
        layerdrop=0.0,
    )

    model = Wav2Vec2BertModel(config)
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
    model = encoder()

    features = fill("features", FRAMES * FEATURE_DIM, INPUT_SCALE).reshape(1, FRAMES, FEATURE_DIM)

    with torch.no_grad():
        out = model(input_features=features, output_hidden_states=True)

    # hidden_states[0] is the projection's output, so [n] is the output of n layers.
    print(f"// Generated by tools/w2v_bert_reference.py -- {FRAMES} frames, {LAYERS} layers.")
    print()
    emit("projected", out.hidden_states[0], PROBE)
    emit("partial", out.hidden_states[PARTIAL], PROBE)
    emit("encoded", out.hidden_states[LAYERS], PROBE)

    print("// shapes:", [tuple(h.shape) for h in out.hidden_states[:2]], file=sys.stderr)


if __name__ == "__main__":
    main()
