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

"""The numbers `waifu/tests/indextts_gpt.rs` checks the GPT against, from GPT-2 itself.

No download: `GPT2Model` is in the pinned transformers in this venv, and IndexTTS's GPT *is* a
GPT-2 -- twenty-four layers, 1280 wide, twenty heads -- fed embeddings rather than ids.

    .venv/bin/python tools/indextts_gpt_reference.py

# The position embedding is zeroed rather than monkeypatched

The reference deletes `gpt.wpe` and puts `null_position_embeddings` in its place, which returns
zeros. Zeroing the table is the same arithmetic and leaves a model that loads and runs like any
other GPT-2, so that is what this does. It is worth being loud about: a reader who assumes GPT-2's
own position embedding is in play gets a model that is wrong in every layer, and the positions
this model *does* use are added to the text and the mel before they ever reach the stack.

# What is small

Every width, and the layer count. A GPT-2 stack has no bookkeeping that turns on its depth, so
four layers exercise the same block twenty-four do. What the probes keep apart is the stack, the
head on top of it and the embeddings in front, so a failure says which.
"""

import math
import sys

import torch

MODEL_DIM = 64
LAYERS = 4
HEADS = 4
NUMBER_MEL_CODES = 40
NUMBER_TEXT_TOKENS = 50
MAX_TEXT_TOKENS = 30
LANGUAGES = 9
LENGTH = 12

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


def backbone():
    """A small GPT-2 with its position embedding zeroed and every parameter filled by name.

    The names are prefixed `gpt.` because that is what they are called inside `UnifiedVoice`,
    which holds the stack as `self.gpt`.
    """
    from transformers import GPT2Config, GPT2Model

    config = GPT2Config(
        vocab_size=NUMBER_MEL_CODES,
        n_positions=512,
        n_embd=MODEL_DIM,
        n_layer=LAYERS,
        n_head=HEADS,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
    )

    model = GPT2Model(config)
    model.eval()

    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name == "wpe.weight":
                parameter.zero_()
            elif name == "wte.weight":
                # Unused: what goes in is embeddings, not ids.
                parameter.zero_()
            else:
                parameter.copy_(
                    fill(f"gpt.{name}", parameter.numel(), WEIGHT_SCALE).reshape(parameter.shape)
                )

    return model


def head():
    """`final_norm` and `mel_head`, which are `UnifiedVoice`'s rather than GPT-2's."""
    norm = torch.nn.LayerNorm(MODEL_DIM)
    linear = torch.nn.Linear(MODEL_DIM, NUMBER_MEL_CODES)

    with torch.no_grad():
        norm.weight.copy_(fill("final_norm.weight", MODEL_DIM, WEIGHT_SCALE))
        norm.bias.copy_(fill("final_norm.bias", MODEL_DIM, WEIGHT_SCALE))
        linear.weight.copy_(
            fill("mel_head.weight", NUMBER_MEL_CODES * MODEL_DIM, WEIGHT_SCALE).reshape(
                NUMBER_MEL_CODES, MODEL_DIM
            )
        )
        linear.bias.copy_(fill("mel_head.bias", NUMBER_MEL_CODES, WEIGHT_SCALE))

    return norm.eval(), linear.eval()


def text_embeddings(text, positions, language):
    """The token, where it sits and what language it is in, added together."""
    token = torch.nn.Embedding(NUMBER_TEXT_TOKENS + 1, MODEL_DIM)
    place = torch.nn.Embedding(MAX_TEXT_TOKENS + 2, MODEL_DIM)
    tongue = torch.nn.Embedding(LANGUAGES, MODEL_DIM)

    with torch.no_grad():
        token.weight.copy_(
            fill("text_embedding.weight", (NUMBER_TEXT_TOKENS + 1) * MODEL_DIM, WEIGHT_SCALE)
            .reshape(NUMBER_TEXT_TOKENS + 1, MODEL_DIM)
        )
        place.weight.copy_(
            fill("text_pos_embedding.emb.weight", (MAX_TEXT_TOKENS + 2) * MODEL_DIM, WEIGHT_SCALE)
            .reshape(MAX_TEXT_TOKENS + 2, MODEL_DIM)
        )
        tongue.weight.copy_(
            fill("lang_embedding.weight", LANGUAGES * MODEL_DIM, WEIGHT_SCALE)
            .reshape(LANGUAGES, MODEL_DIM)
        )

        return token(text) + place(positions) + tongue(language)


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
    model = backbone()
    norm, linear = head()

    embeddings = fill("embeddings", LENGTH * MODEL_DIM, INPUT_SCALE).reshape(1, LENGTH, MODEL_DIM)

    with torch.no_grad():
        hidden = model(inputs_embeds=embeddings).last_hidden_state
        scores = linear(norm(hidden))

        text = torch.arange(LENGTH, dtype=torch.long).reshape(1, LENGTH) % NUMBER_TEXT_TOKENS
        positions = torch.arange(LENGTH, dtype=torch.long)
        language = torch.tensor([3], dtype=torch.long)
        embedded = text_embeddings(text, positions, language)

    print(f"// Generated by tools/indextts_gpt_reference.py -- {LENGTH} positions, {LAYERS} layers.")
    print()
    emit("hidden", hidden, PROBE)
    emit("scores", scores, PROBE)
    emit("embedded", embedded, PROBE)

    print("// shapes:", {"hidden": tuple(hidden.shape), "scores": tuple(scores.shape)},
          file=sys.stderr)


if __name__ == "__main__":
    main()
