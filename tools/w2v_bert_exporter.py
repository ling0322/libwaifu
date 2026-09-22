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

"""w2v-bert-2.0 as safetensors: `facebook/w2v-bert-2.0` in, what `waifu::w2v_bert` reads out.

    .venv/bin/python tools/w2v_bert_exporter.py -output models/w2v-bert-2.0.safetensors

Meta's model under the MIT licence, published separately and shared by several speech systems.
Nothing about it is IndexTTS's -- except the two vectors at the end, which are (see below).

# A third of it is not exported

The release has twenty-four encoder layers. IndexTTS reads `hidden_states[17]`, which is the
feature projection plus **sixteen** of them, and never looks at the rest -- so layers 16 through
23 are not written here. That is 194 M of the 580 M parameters that do not have to be downloaded,
stored, read onto a card or run.

`waifu::w2v_bert::Config::USED_LAYERS` is that sixteen, and this file's `LAYERS` has to agree with
it. They are two constants in two languages saying the same thing, so the count is printed at the
end for a human to compare against `cargo run --example parameters -- w2v_bert`.

# Nothing is done to the weights

Names, shapes and values are the checkpoint's. `waifu::w2v_bert` was written against
`Wav2Vec2BertModel`'s own names, so `encoder.layers.3.self_attn.linear_q.weight` in the file is
that in the graph, and every one of the 516 tensors matched on the first try with no shape
disagreeing. There is no mapping here to get wrong.

# The two vectors that are IndexTTS's

`wav2vec2bert_stats.pt` in the IndexTTS release holds a `mean` and a `var` over w2v-bert's
features, and `infer_v2_5.py` standardizes with them before anything downstream sees a feature:

    feat = (feat - semantic_mean) / semantic_std

They are written here as `semantic.mean` and `semantic.std` because this is the file the features
come out of, and a standardization belongs with the thing it standardizes. **`std` is the square
root of the stored `var`**, taken here rather than at load, because `torch.sqrt(stat["var"])` is
what upstream does and a package should not ask its reader to remember that.

Pass `-stats` to read a `wav2vec2bert_stats.pt` already on disk; without it they are fetched from
the IndexTTS repository.
"""

import argparse
import os

import torch

REPO = "facebook/w2v-bert-2.0"
CHECKPOINT = "model.safetensors"

INDEXTTS_REPO = "IndexTeam/IndexTTS-2.5"
STATS = "wav2vec2bert_stats.pt"

# What `waifu::w2v_bert::Config::USED_LAYERS` says. The other eight are not written.
LAYERS = 16

# The feature projection, which is everything before the stack.
TOP_LEVEL = (
    "feature_projection.layer_norm.weight",
    "feature_projection.layer_norm.bias",
    "feature_projection.projection.weight",
    "feature_projection.projection.bias",
)

# One conformer layer: feed forward, attention, convolution, feed forward. Thirty-two tensors --
# the convolutions carry no bias and `distance_embedding` is a table, so it is not a round number.
PER_LAYER = (
    "ffn1_layer_norm.weight",
    "ffn1_layer_norm.bias",
    "ffn1.intermediate_dense.weight",
    "ffn1.intermediate_dense.bias",
    "ffn1.output_dense.weight",
    "ffn1.output_dense.bias",
    "self_attn_layer_norm.weight",
    "self_attn_layer_norm.bias",
    "self_attn.linear_q.weight",
    "self_attn.linear_q.bias",
    "self_attn.linear_k.weight",
    "self_attn.linear_k.bias",
    "self_attn.linear_v.weight",
    "self_attn.linear_v.bias",
    "self_attn.distance_embedding.weight",
    "self_attn.linear_out.weight",
    "self_attn.linear_out.bias",
    "conv_module.layer_norm.weight",
    "conv_module.layer_norm.bias",
    "conv_module.pointwise_conv1.weight",
    "conv_module.depthwise_conv.weight",
    "conv_module.depthwise_layer_norm.weight",
    "conv_module.depthwise_layer_norm.bias",
    "conv_module.pointwise_conv2.weight",
    "ffn2_layer_norm.weight",
    "ffn2_layer_norm.bias",
    "ffn2.intermediate_dense.weight",
    "ffn2.intermediate_dense.bias",
    "ffn2.output_dense.weight",
    "ffn2.output_dense.bias",
    "final_layer_norm.weight",
    "final_layer_norm.bias",
)

# The shapes that carry a decision. `distance_embedding` is the one to watch: it is
# `left + right + 1` rows, 73 of them, and a model that declared 64 would load and be wrong about
# every pair of frames more than sixty-four apart.
EXPECTED_SHAPES = {
    "feature_projection.projection.weight": (1024, 160),
    "encoder.layers.0.self_attn.distance_embedding.weight": (73, 64),
    "encoder.layers.0.conv_module.depthwise_conv.weight": (1024, 1, 31),
    "encoder.layers.0.ffn1.intermediate_dense.weight": (4096, 1024),
}


def wanted(layers=LAYERS):
    """Every name the graph loads, in the order it loads them."""
    names = list(TOP_LEVEL)
    for index in range(layers):
        names.extend(f"encoder.layers.{index}.{leaf}" for leaf in PER_LAYER)

    return names


def checkpoint(path=None):
    """The released model, from a file or from the hub."""
    from safetensors.torch import load_file

    if path is None:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(repo_id=REPO, filename=CHECKPOINT)

    return load_file(path)


def standardization(path=None):
    """IndexTTS's `mean` and `std` over these features, `std` being the root of the stored `var`."""
    if path is None:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(repo_id=INDEXTTS_REPO, filename=STATS)

    stats = torch.load(path, map_location="cpu", weights_only=True)

    return {
        "semantic.mean": stats["mean"].float().contiguous(),
        "semantic.std": torch.sqrt(stats["var"].float()).contiguous(),
    }


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

    taken = {name: state[name].detach().float().contiguous() for name in names}
    left = sorted(set(state) - set(names))

    return taken, left


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-output", help="where to write the exported weights")
    parser.add_argument("-checkpoint", help="a w2v-bert-2.0 model.safetensors already on disk")
    parser.add_argument("-stats", help="a wav2vec2bert_stats.pt already on disk")
    parser.add_argument("-layers", type=int, default=LAYERS)
    arguments = parser.parse_args()

    state = checkpoint(arguments.checkpoint)
    taken, left = select(state, arguments.layers)
    taken.update(standardization(arguments.stats))

    total = sum(value.numel() for value in taken.values())
    print(f"{len(taken)} tensors, {total / 1e6:.2f} M parameters ({arguments.layers} layers)")

    dropped = sum(state[name].numel() for name in left)
    print(f"{len(left)} tensors not exported, {dropped / 1e6:.2f} M parameters -- the last "
          f"{24 - arguments.layers} layers and the heads on top of them")

    if arguments.output:
        from safetensors.torch import save_file

        os.makedirs(os.path.dirname(arguments.output) or ".", exist_ok=True)
        save_file(taken, arguments.output)
        print("wrote", arguments.output)


if __name__ == "__main__":
    main()
