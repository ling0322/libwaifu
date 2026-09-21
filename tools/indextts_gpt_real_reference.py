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

"""What the *released* GPT says, for the `#[ignore]`d test in `waifu/tests/indextts_gpt.rs`.

    .venv/bin/python tools/indextts_gpt_real_reference.py

The three fast probes check a four-layer model whose weights come from their own names. This
checks the real one: 574 M parameters, twenty-four layers, the actual `gpt.pth`. The two ask
different questions. The fast probes ask whether the arithmetic is GPT-2's; this asks whether the
weights are being read the way the checkpoint stores them -- which a model with every parameter
filled by the same rule cannot tell you, because such a model agrees with any transposition that
is self-consistent.

It also covers the whole prefix rather than the stack alone, so `lang_embedding` and
`mel_pos_embedding` are read at a real index out of a real table. `lang_embedding` is the one to
watch: it has 107 rows, and a graph that declared 9 would have loaded a table of the wrong shape
and never scored a row past the ninth.

# What is fixed, and why none of it is random

The speaker and emotion vectors are the ones this model would really be handed in shape but not
in provenance -- there is no recording here, so they are filled from their own names by the same
sine the fast probes use. That is a reference input of the kind the note in CLAUDE.md warns about,
and it is used anyway for one narrow reason: what is under test is that two implementations read
the same *weights* the same way, not that the model draws a good picture. Both sides see the
identical vector, and the comparison is exact arithmetic rather than perceptual.

The text is real: ids from the released vocabulary, wrapped in the start and stop tokens.

# Regenerate after changing anything the graph loads

    .venv/bin/python tools/indextts_gpt_exporter.py \\
        -checkpoint ~/.cache/libwaifu/indextts25/gpt.pth \\
        -output models/indextts25-gpt.safetensors
    .venv/bin/python tools/indextts_gpt_real_reference.py
"""

import math
import os
import sys

import torch

CHECKPOINT = os.path.expanduser("~/.cache/libwaifu/indextts25/gpt.pth")

MODEL_DIM = 1280
LAYERS = 24
HEADS = 20
NUMBER_MEL_CODES = 8194
SPEAKER_DIM = 192
START_MEL_TOKEN = 8192
START_TEXT_TOKEN = 0
STOP_TEXT_TOKEN = 1
CONDITIONING_ROWS = 3

# The sentence, as ids. Chosen to sit well inside the vocabulary and to be the same every run.
TEXT = [3402, 118, 7719, 45, 2210, 908, 31]
LANGUAGE = 3

# What the two conditioning vectors are filled with. The same rule as the fast probes.
INPUT_SCALE = 0.6

# How many tokens to step after the prefill, so the cache is exercised on real weights too.
STEPS = 3
PROBE = 8


def hash32(name: str) -> int:
    value = 0x811C9DC5
    for byte in name.encode("utf-8"):
        value = ((value ^ byte) * 0x01000193) & 0xFFFFFFFF

    return value


def fill(name: str, count: int, scale: float) -> torch.Tensor:
    seed = (hash32(name) % 10007) * 0.001

    return torch.tensor(
        [math.sin(i * 0.7371 + seed) * scale for i in range(count)], dtype=torch.float32
    )


def state():
    if not os.path.isfile(CHECKPOINT):
        sys.exit(
            f"{CHECKPOINT} is not there.\n"
            "Fetch the release first: it is public and ungated at IndexTeam/IndexTTS-2.5."
        )

    return torch.load(CHECKPOINT, map_location="cpu", weights_only=True)


def backbone(weights):
    """The released stack, in a stock `GPT2Model`, with the position embedding zeroed.

    `UnifiedVoice` holds it as `self.gpt` and replaces its position embedding with a function
    returning zeros. Zeroing the table is the same arithmetic and leaves a model that loads like
    any other GPT-2.
    """
    from transformers import GPT2Config, GPT2Model

    config = GPT2Config(
        vocab_size=NUMBER_MEL_CODES,
        n_positions=4096,
        n_embd=MODEL_DIM,
        n_layer=LAYERS,
        n_head=HEADS,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
    )

    model = GPT2Model(config).eval()

    taken = 0
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name in ("wpe.weight", "wte.weight"):
                parameter.zero_()
                continue

            parameter.copy_(weights[f"gpt.{name}"])
            taken += 1

    expected = LAYERS * 12 + 2
    if taken != expected:
        sys.exit(f"filled {taken} of the stack's parameters, expected {expected}")

    return model


def prefix(weights):
    """`[ speaker + emotion ][ 0 ][ 0 ][ text ][ start mel ]`, as embeddings."""
    speaker = fill("speaker", SPEAKER_DIM, INPUT_SCALE).reshape(1, SPEAKER_DIM)
    emotion = fill("emotion", MODEL_DIM, INPUT_SCALE).reshape(1, MODEL_DIM)

    projected = torch.nn.functional.linear(
        speaker, weights["spk_emb_proj.weight"], weights["spk_emb_proj.bias"]
    )
    first = projected.unsqueeze(1) + emotion.unsqueeze(1)
    conditioned = torch.cat([first, torch.zeros(1, CONDITIONING_ROWS - 1, MODEL_DIM)], dim=1)

    ids = torch.tensor([[START_TEXT_TOKEN] + TEXT + [STOP_TEXT_TOKEN]])
    positions = torch.arange(ids.shape[1])
    text = (
        weights["text_embedding.weight"][ids]
        + weights["text_pos_embedding.emb.weight"][positions]
        + weights["lang_embedding.weight"][torch.tensor([LANGUAGE])]
    )

    start = (
        weights["mel_embedding.weight"][torch.tensor([[START_MEL_TOKEN]])]
        + weights["mel_pos_embedding.emb.weight"][torch.tensor([0])]
    )

    return torch.cat([conditioned, text, start], dim=1)


def score(model, weights, embeddings):
    """Every position's hidden state, and the last one's scores over the codec's alphabet."""
    with torch.no_grad():
        hidden = model(inputs_embeds=embeddings).last_hidden_state

        normed = torch.nn.functional.layer_norm(
            hidden[:, -1:],
            (MODEL_DIM,),
            weights["final_norm.weight"],
            weights["final_norm.bias"],
            1e-5,
        )
        scores = torch.nn.functional.linear(
            normed, weights["mel_head.weight"], weights["mel_head.bias"]
        )

    return hidden, scores[:, 0]


def emit(name, tensor, limit):
    flat = tensor.reshape(-1)
    step = max(flat.numel() // limit, 1)
    indices = [(i * step + i * 7) % flat.numel() for i in range(limit)]

    print(f"/// `{name}`, {tuple(tensor.shape)} -- {limit} of {flat.numel()}, at {indices}.")
    print(f"const {name.upper()}: [f32; {limit}] = [")
    for index in indices:
        print(f"    {flat[index].item():.6e},")
    print("];")
    print()


def main():
    weights = state()
    model = backbone(weights)
    embeddings = prefix(weights)

    print(f"// Generated by tools/indextts_gpt_real_reference.py -- the released 574 M GPT.")
    print(f"// Prefix of {embeddings.shape[1]} positions, then {STEPS} tokens stepped.\n")

    _, scores = score(model, weights, embeddings)
    emit("prefill_scores", scores, PROBE)

    # Then a few tokens, fed back the way the loop feeds them: greedily, each at the mel position
    # after the start token's zero.
    said = []
    running = embeddings
    for index in range(STEPS):
        token = int(scores.argmax(dim=-1)[0])
        said.append(token)

        nxt = (
            weights["mel_embedding.weight"][torch.tensor([[token]])]
            + weights["mel_pos_embedding.emb.weight"][torch.tensor([index + 1])]
        )
        running = torch.cat([running, nxt], dim=1)
        _, scores = score(model, weights, running)

    print(f"/// The tokens a greedy reading says first, from the released weights.")
    print(f"const SAID: [i32; {STEPS}] = {said};".replace("[", "[").replace("]", "]"))
    print()
    emit("stepped_scores", scores, PROBE)


if __name__ == "__main__":
    main()
