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

"""IndexTTS-2.5's GPT as safetensors: `gpt.pth` in, what `waifu::indextts_gpt` asks for out.

    .venv/bin/python tools/indextts_gpt_exporter.py -output models/indextts25-gpt.safetensors

The largest of the six models -- 3.3 GB of the release's 5.5 -- and the one the rest of the
pipeline is arranged around.

# Nothing is done to the weights

Unusually for an exporter here, this one renames nothing and folds nothing. `waifu::indextts_gpt`
was written against the checkpoint's own names, so `gpt.h.0.attn.c_attn.weight` in the file is
that in the graph, and a tensor that turns out to be wrong can be compared with the one it came
from without a mapping in between.

That includes the two shapes that look like mistakes and are not. `attn.c_attn.weight` is
`(1280, 3840)` -- HuggingFace's `Conv1D` stores `(in, out)` where `nn.Linear` stores `(out, in)`,
and the graph multiplies it untransposed. `mel_head.weight` is `(8194, 1280)`, because *that* one
really is an `nn.Linear`. The two conventions sit ten lines apart in the model and are preserved
here rather than reconciled, since reconciling them is what would have to be undone at load.

# What it does not export, and why the shapes are checked

The release's GPT checkpoint holds 456 tensors. This writes 454 of them -- 301 for
`waifu::indextts_gpt` and 153 for `waifu::indextts_emotion` -- and leaves exactly one thing
behind: `text_head`, a training-time head with nothing downstream of it, and 77 M parameters of
it. That is a decision and not an oversight.

What is *not* in the checkpoint is worth as much as what is. `UnifiedVoice.__init__` builds a
`conditioning_encoder` and a `perceiver_encoder` -- the speaker path IndexTTS **2** used, and a
near twin of the emotion path this now exports -- and the 2.5 release ships **no weights for
either of them**. They would come up in `load_checkpoint`'s missing keys and stay at their random
initialization, which is harmless because 2.5 never calls them: it builds its speaker row with
`spk_cond_mode="campplus"`, which is 192 numbers from `waifu::campplus` through `spk_emb_proj`.
So the release itself settles a question the source alone leaves open, and `waifu` is right not
to implement them.

The shapes that carry a decision are checked rather than trusted, so a tensor upstream quietly
changes size becomes an error here instead of a model that loads and speaks nonsense --
`lang_embedding` is the cautionary one, since it is 107 rows for a model that advertises five
languages, and a plausible guess of 9 loads, runs and is wrong.

# The position table is exported, which was not the plan

`emo_conditioning_encoder.embed.pos_enc.pe` is a buffer and not a weight -- five thousand rows of
sinusoid that depend on nothing but their own shape -- so the obvious move is to rebuild it at
load and save ten megabytes. `report_position_buffer` is where that turned out to need an
argument: the released table is the closed form **rounded to `bfloat16`**, exactly and
reproducibly, and upstream runs with the rounded one rather than the exact one. It is written out
like any other tensor, and the check on it is a bit-for-bit equality. The function says why the
bytes won over rebuilding them.

# The emotion path, and the one tensor worth staring at

`emo_conditioning_encoder.embed.out.0.weight` is `(512, 261632)` -- 134 M parameters, more than
the conformer it belongs to and a fifth of everything written here. That is not a mistake. WeNet's
`Conv2dSubsampling2` runs one 3 by 3 convolution at stride two over the features read as an image,
and then flattens *all* of what it produced at each frame -- 512 channels by the 511 feature
columns that survive 1024 -- into one projection. The shape is checked for exactly that reason:
it is the one tensor here whose size a reader is most likely to assume is wrong.

# Where the checkpoint comes from

`IndexTeam/IndexTTS-2.5` on Hugging Face, which is public and ungated. Pass `-checkpoint` to read
one already on disk.
"""

import argparse
import math
import os

import torch

REPO = "IndexTeam/IndexTTS-2.5"
CHECKPOINT = "gpt.pth"

# What `config.yaml` says, and what `waifu::indextts_gpt::Config::indextts` repeats.
LAYERS = 24

# The two the config does not state. `mel_pos_embedding` is sized
# `max_mel_tokens + 2 + max_conditioning_inputs` inside `build_hf_gpt_transformer`, and the
# language table is `len(LANGUAGE_DICT) + 1` -- Whisper's list of 106, most of which this model
# was never trained to say.
MEL_POSITIONS = 1818
LANGUAGES = 107

# Every tensor the graph asks for that is not inside a block.
TOP_LEVEL = (
    "text_embedding.weight",
    "mel_embedding.weight",
    "text_pos_embedding.emb.weight",
    "mel_pos_embedding.emb.weight",
    "lang_embedding.weight",
    "spk_emb_proj.weight",
    "spk_emb_proj.bias",
    "final_norm.weight",
    "final_norm.bias",
    "mel_head.weight",
    "mel_head.bias",
    "gpt.ln_f.weight",
    "gpt.ln_f.bias",
)

# And every one inside a block, which is a stock GPT-2's.
PER_BLOCK = (
    "ln_1.weight",
    "ln_1.bias",
    "attn.c_attn.weight",
    "attn.c_attn.bias",
    "attn.c_proj.weight",
    "attn.c_proj.bias",
    "ln_2.weight",
    "ln_2.bias",
    "mlp.c_fc.weight",
    "mlp.c_fc.bias",
    "mlp.c_proj.weight",
    "mlp.c_proj.bias",
)

# The emotion path: `waifu::indextts_emotion`'s four stages, as `config.yaml`'s
# `gpt.emo_condition_module` sizes them.
EMOTION_BLOCKS = 4
PERCEIVER_DEPTH = 2

# The sinusoid buffer, which is exported rather than rebuilt -- see the note above.
POSITION_BUFFER = "emo_conditioning_encoder.embed.pos_enc.pe"

EMOTION_TOP_LEVEL = (
    "emo_conditioning_encoder.embed.conv.0.weight",
    "emo_conditioning_encoder.embed.conv.0.bias",
    "emo_conditioning_encoder.embed.out.0.weight",
    "emo_conditioning_encoder.embed.out.0.bias",
    "emo_conditioning_encoder.after_norm.weight",
    "emo_conditioning_encoder.after_norm.bias",
    "emo_perceiver_encoder.latents",
    "emo_perceiver_encoder.proj_context.weight",
    "emo_perceiver_encoder.proj_context.bias",
    "emo_perceiver_encoder.norm.gamma",
    POSITION_BUFFER,
    "emovec_layer.weight",
    "emovec_layer.bias",
    "emo_layer.weight",
    "emo_layer.bias",
)

# One conformer block. Thirty-one tensors: `linear_pos` has no bias and the two position biases
# are parameters rather than layers, which is why this is not a round number.
EMOTION_PER_BLOCK = (
    "self_attn.pos_bias_u",
    "self_attn.pos_bias_v",
    "self_attn.linear_q.weight",
    "self_attn.linear_q.bias",
    "self_attn.linear_k.weight",
    "self_attn.linear_k.bias",
    "self_attn.linear_v.weight",
    "self_attn.linear_v.bias",
    "self_attn.linear_out.weight",
    "self_attn.linear_out.bias",
    "self_attn.linear_pos.weight",
    "feed_forward.w_1.weight",
    "feed_forward.w_1.bias",
    "feed_forward.w_2.weight",
    "feed_forward.w_2.bias",
    "conv_module.pointwise_conv1.weight",
    "conv_module.pointwise_conv1.bias",
    "conv_module.depthwise_conv.weight",
    "conv_module.depthwise_conv.bias",
    "conv_module.norm.weight",
    "conv_module.norm.bias",
    "conv_module.pointwise_conv2.weight",
    "conv_module.pointwise_conv2.bias",
    "norm_ff.weight",
    "norm_ff.bias",
    "norm_mha.weight",
    "norm_mha.bias",
    "norm_conv.weight",
    "norm_conv.bias",
    "norm_final.weight",
    "norm_final.bias",
)

# One perceiver layer. The `0` and `1` are an `nn.ModuleList` of two, and the `1.0` and `1.2` are
# positions in the `nn.Sequential` the gated feed forward is -- position one is the activation,
# which has nothing to store.
EMOTION_PER_PERCEIVER_LAYER = (
    "0.to_q.weight",
    "0.to_kv.weight",
    "0.to_out.weight",
    "1.0.weight",
    "1.0.bias",
    "1.2.weight",
    "1.2.bias",
)

# The shapes that carry a decision, checked rather than trusted. Everything else is checked by
# being present at all.
EXPECTED_SHAPES = {
    "text_embedding.weight": (60510, 1280),
    "mel_embedding.weight": (8194, 1280),
    "text_pos_embedding.emb.weight": (602, 1280),
    "mel_pos_embedding.emb.weight": (MEL_POSITIONS, 1280),
    "lang_embedding.weight": (LANGUAGES, 1280),
    "spk_emb_proj.weight": (1280, 192),
    "mel_head.weight": (8194, 1280),
    "gpt.h.0.attn.c_attn.weight": (1280, 3840),
    # The emotion path. Every one of these encodes a decision the graph also makes: the
    # subsampling arithmetic, the head count, the single latent, and the truncated 1365.
    "emo_conditioning_encoder.embed.conv.0.weight": (512, 1, 3, 3),
    "emo_conditioning_encoder.embed.out.0.weight": (512, 261632),
    "emo_conditioning_encoder.encoders.0.self_attn.pos_bias_u": (4, 128),
    "emo_conditioning_encoder.encoders.0.conv_module.depthwise_conv.weight": (512, 1, 15),
    "emo_perceiver_encoder.latents": (1, 1024),
    "emo_perceiver_encoder.layers.0.0.to_q.weight": (256, 1024),
    "emo_perceiver_encoder.layers.0.1.0.weight": (2730, 1024),
    "emovec_layer.weight": (1280, 1024),
}


def wanted(layers=LAYERS):
    """Every name the two graphs load, in the order they load them."""
    names = list(TOP_LEVEL)
    for index in range(layers):
        names.extend(f"gpt.h.{index}.{leaf}" for leaf in PER_BLOCK)

    names.extend(EMOTION_TOP_LEVEL)
    for index in range(EMOTION_BLOCKS):
        names.extend(
            f"emo_conditioning_encoder.encoders.{index}.{leaf}"
            for leaf in EMOTION_PER_BLOCK
        )
    for index in range(PERCEIVER_DEPTH):
        names.extend(
            f"emo_perceiver_encoder.layers.{index}.{leaf}"
            for leaf in EMOTION_PER_PERCEIVER_LAYER
        )

    return names


def sinusoid(positions, dim):
    """`PositionalEncoding`'s table, which is upstream's own four lines of it."""
    position = torch.arange(0, positions).unsqueeze(1)
    divisor = torch.exp(torch.arange(0, dim, 2) * -(math.log(10000.0) / dim))

    table = torch.zeros(positions, dim)
    table[:, 0::2] = torch.sin(position * divisor)
    table[:, 1::2] = torch.cos(position * divisor)

    return table.unsqueeze(0)


def report_position_buffer(state):
    """Check the released position table against the sinusoid, and say what it is.

    `pe` is a registered buffer rather than a learned weight -- five thousand rows that depend on
    nothing but their own shape -- so the obvious thing is to rebuild it and save the ten
    megabytes. This is where that turned out to need an argument.

    **The released table is the closed form rounded to `bfloat16`.** Not `float16`: not every
    stored value is representable in `float16`, every one of them is representable in `bfloat16`,
    and `sinusoid(...).to(torch.bfloat16).float()` reproduces the released tensor to within one
    `bfloat16` step -- see below. Somebody saved this model through `bfloat16` once and the buffer
    kept the scar. Against the unrounded table it is 7.5e-4 out on average and 2.0e-3 at worst --
    and 2.0e-3 is 2^-9, which is exactly half a `bfloat16` step for a value just under one, i.e.
    the worst that round-to-nearest can do.

    Upstream runs with the stored table and not with the exact one. `pe` is persistent, so it is
    in the file; `load_checkpoint` calls `load_state_dict(strict=False)`, which only forgives
    *absent* keys and this one is present; so the table computed in `PositionalEncoding.__init__`
    is overwritten the moment the checkpoint loads.

    Which leaves a real choice, since the rounding is now pinned down exactly: write the tensor
    out, or rebuild it and round. This writes it out. Ten megabytes is 0.3% of the package, and
    the alternative is a standing claim that a Rust loop over `f64` sines rounds onto the same
    `bfloat16` grid `torch` reached through `float32` -- true as far as it was tested, and not
    worth having to keep true. `waifu::indextts_emotion::positional_encoding` builds the unrounded
    table for the tests and for running without a package, 2.0e-3 from what the release uses.

    ## The check tolerates one `bfloat16` step, not zero

    It was exact once: `torch.equal` against `sinusoid(...).to(torch.bfloat16).float()`, on the
    grounds that a mystery solved deserves a check that fails the moment it comes back. It came
    back on a machine that had done nothing wrong. `torch.sin`/`cos`/`exp` are vectorized, and
    which side of a `bfloat16` rounding boundary a value lands on depends on the last bit of the
    CPU's transcendental approximation -- which is not the same bit on every build. On one such
    machine, 1 435 of the buffer's 2 560 000 values (0.056%) disagreed with this file's own
    recomputation. The release hadn't moved: `IndexTeam/IndexTTS-2.5`'s commit history shows one
    upload of the weights, on 2026-08-10, and nothing since has touched them.

    The disagreement is bounded by `2**-8` in every one of those 1 435 cases -- one `bfloat16`
    step at magnitude 1, which is the *coarsest* step anywhere a sine or cosine lives, `[-1, 1]`.
    That bound holds even where the local step is far finer: `position * divisor` grows into the
    thousands of radians for a late position and an early dimension, and near a zero of `sin` or
    `cos` a float32-sized error in that argument becomes a large *relative* error in a tiny
    *output* -- one value came out 127 local steps off while still being under a millionth in
    absolute terms, dwarfed by `2**-8`. Counting local steps would have flagged it; counting
    absolute distance against the coarsest step in the table does not, and is what is checked.

    A changed position encoding does not hide in that bound: it moves most of the table by values
    that are `sin`/`cos` of a *different* argument, which agree with this one only by coincidence,
    not by one rounding step.
    """
    stored = state[POSITION_BUFFER].float()
    _, positions, dim = stored.shape
    exact = sinusoid(positions, dim)
    rounded = exact.to(torch.bfloat16).float()
    off = (stored - exact).abs()

    coarsest_bfloat16_step = 2**-8
    off_rounded = (stored - rounded).abs()
    if (off_rounded > coarsest_bfloat16_step).any():
        wrong = int((off_rounded > coarsest_bfloat16_step).sum())
        raise SystemExit(
            f"{POSITION_BUFFER} is no longer the sinusoid rounded to bfloat16 -- {wrong} of "
            f"{off_rounded.numel()} elements are more than one bfloat16 step ({coarsest_bfloat16_step:.3g}) "
            f"from the closed form, worst {off.max().item():.3g}. The release has changed its "
            f"position encoding, and docs/indextts_emotion.md is now telling a story about the old one"
        )

    return positions, dim, off.mean().item(), off.max().item()


def checkpoint(path=None):
    """The released GPT, from a file or from the hub."""
    if path is None:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(repo_id=REPO, filename=CHECKPOINT)

    return torch.load(path, map_location="cpu", weights_only=True)


def select(state, layers=LAYERS):
    """The tensors the graph reads, and what was left behind."""
    names = wanted(layers)

    missing = [name for name in names if name not in state]
    if missing:
        raise SystemExit(
            f"the checkpoint is missing {len(missing)} tensors the graph reads, "
            f"starting with {missing[0]} -- upstream has renamed something"
        )

    for name, shape in EXPECTED_SHAPES.items():
        got = tuple(state[name].shape)
        if got != shape:
            raise SystemExit(f"{name} is {got} in the checkpoint and {shape} in the graph")

    taken = {name: state[name].detach().contiguous() for name in names}
    left = sorted(set(state) - set(names))

    return taken, left


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-output", help="where to write the exported weights")
    parser.add_argument("-checkpoint", help="a gpt.pth already on disk")
    parser.add_argument("-layers", type=int, default=LAYERS)
    arguments = parser.parse_args()

    state = checkpoint(arguments.checkpoint)

    positions, dim, mean_off, max_off = report_position_buffer(state)
    print(
        f"{POSITION_BUFFER}: ({positions}, {dim}), exactly the sinusoid rounded to bfloat16 -- "
        f"{mean_off:.2g} mean and {max_off:.2g} worst off the unrounded one. "
        f"Exported, because upstream runs with it."
    )

    taken, left = select(state, arguments.layers)

    total = sum(value.numel() for value in taken.values())
    emotion = sum(
        value.numel()
        for name, value in taken.items()
        if name.startswith(("emo_", "emovec_"))
    )
    print(f"{len(taken)} tensors, {total / 1e6:.2f} M parameters")
    print(f"  of which the emotion path is {emotion / 1e6:.2f} M")

    # What is left is listed by its prefix rather than tensor by tensor -- a few dozen names would
    # bury the one that matters if upstream ever adds another.
    prefixes = sorted({name.split(".")[0] for name in left})
    print(f"{len(left)} tensors not exported, under: {', '.join(prefixes)}")

    if arguments.output:
        from safetensors.torch import save_file

        os.makedirs(os.path.dirname(arguments.output) or ".", exist_ok=True)
        save_file(taken, arguments.output)
        print("wrote", arguments.output)


if __name__ == "__main__":
    main()
