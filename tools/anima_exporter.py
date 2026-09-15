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

"""Export an Anima checkpoint to safetensors and the manifest that names them.

Anima is not an SDXL fine tune and this is not `sdxl_exporter.py` with different names. It is a
Cosmos-Predict2 diffusion transformer, a Qwen3-0.6B text encoder and the Qwen-Image VAE, published
as three separate safetensors files, and the whole of what it is is written down in `docs/anima.md`
-- read that first, because several of the decisions here only make sense against it.

Unlike the SDXL exporter this one does not walk a diffusers pipeline: there is no diffusers
implementation of Anima to walk. It reads the three state dictionaries directly and rewrites them,
which means the tensor names below are the contract with the runtime and nothing checks them for
us.

    .venv/bin/python tools/anima_exporter.py \
        -dit    models/anima/split_files/diffusion_models/anima-turbo-v1.1.safetensors \
        -text   models/anima/split_files/text_encoders/qwen_3_06b_base.safetensors \
        -vae    models/anima/split_files/vae/qwen_image_vae.safetensors \
        -output models/anima-turbo-v11.safetensors
"""

from __future__ import annotations

import argparse
import re
import sys
from os import path

import torch
from safetensors import safe_open

import anima_comfy as reference
from tokenizer_exporter import read_tokenizer, tokenizer_corpus, tokenizer_json
from model_exporter import Context, TensorBag, open_weights, parse_size, stem_of

# The two prefixes a published denoiser uses: the official release wraps everything in
# "model.diffusion_model.", the community 2.9B expansion in "net.". Neither is load bearing, so
# both are stripped and the rest of this file speaks in the bare names.
DIT_PREFIXES = ("model.diffusion_model.", "net.")

# fp16's largest finite value. bf16 carries fp32's exponent range and fp16 does not, so a weight
# that was comfortable in the checkpoint can become an infinity here. It never has yet, and the
# day it does the picture would come out as noise with nothing to say why, so it is worth a check.
FP16_MAX = 65504.0

# How the Qwen-Image VAE normalizes its sixteen latent channels. Unlike SDXL, which scales by one
# number, this is a mean and a standard deviation per channel, and they belong to the VAE rather
# than to any particular Anima release -- they are not in the weights, so they are written here.
LATENTS_MEAN = (-0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
                0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921)
LATENTS_STD = (2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
               3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160)


class Converter:
    """Reads the three checkpoints and writes one parameter file.

    Every tensor goes through `_write`, which is also where bfloat16 becomes something the format
    can hold. Tensors are fused where the runtime wants them fused -- a single qkv projection
    rather than three -- because that is a matrix multiply saved on every block of every step.
    """

    def __init__(self, writer) -> None:
        self._writer = writer
        self._widened = 0
        self._count = 0

    # ---- the plumbing -------------------------------------------------------------------

    def _write(self, ctx: Context, tensor: torch.Tensor) -> None:
        """Write one tensor, in a dtype the parameter format admits."""
        if tensor.dtype == torch.bfloat16:
            # Straight to fp32: the writer narrows to fp16 itself, and going through fp32 is what
            # makes the range check below meaningful rather than a check on an already-lost value.
            tensor = tensor.to(torch.float32)
            self._widened += 1

        if tensor.dtype == torch.float32:
            largest = tensor.abs().max().item()
            if largest > FP16_MAX:
                raise OverflowError(
                    f"{ctx.name} reaches {largest:g}, which fp16 cannot hold; "
                    "this model needs a wider parameter dtype than the format has")

        self._writer.write_tensor(ctx, tensor.contiguous())
        self._count += 1

    def _fuse(self, ctx: Context, *tensors: torch.Tensor) -> None:
        """Write several projections as one, concatenated on the output dimension."""
        self._write(ctx, torch.cat(tensors, dim=0))

    @property
    def written(self) -> int:
        return self._count

    @property
    def widened(self) -> int:
        return self._widened

    # ---- the denoiser -------------------------------------------------------------------

    def export_dit(self, ctx: Context, weights: dict, blocks: int) -> None:
        """The Cosmos-Predict2 transformer: patchify, timestep, N blocks, unpatchify."""
        # 68 = (16 latent + 1 padding mask) * 2 * 2. The mask channel is zeros for every image
        # this runtime draws, but the projection still reads it, so it is not dropped here.
        self._write(ctx.with_subname("x_embedder.weight"), weights["x_embedder.proj.1.weight"])

        self._write(ctx.with_subname("t_embedder.1.weight"), weights["t_embedder.1.linear_1.weight"])
        self._write(ctx.with_subname("t_embedder.2.weight"), weights["t_embedder.1.linear_2.weight"])
        self._write(ctx.with_subname("t_norm.weight"), weights["t_embedding_norm.weight"])

        for index in range(blocks):
            self._export_dit_block(ctx.with_subname(f"block{index}"), weights, f"blocks.{index}.")

        final = ctx.with_subname("final")
        self._write(final.with_subname("adaln.1.weight"),
                    weights["final_layer.adaln_modulation.1.weight"])
        self._write(final.with_subname("adaln.2.weight"),
                    weights["final_layer.adaln_modulation.2.weight"])
        self._write(final.with_subname("linear.weight"), weights["final_layer.linear.weight"])

    def _export_dit_block(self, ctx: Context, weights: dict, prefix: str) -> None:
        def at(name: str) -> torch.Tensor:
            return weights[prefix + name]

        # Self attention is square in every projection, so all three fuse. Cross attention reads
        # the 1024-wide adapter output, so only k and v can.
        self._fuse(ctx.with_subname("self_attn.qkv_proj.weight"),
                   at("self_attn.q_proj.weight"),
                   at("self_attn.k_proj.weight"),
                   at("self_attn.v_proj.weight"))
        self._write(ctx.with_subname("self_attn.q_norm.weight"), at("self_attn.q_norm.weight"))
        self._write(ctx.with_subname("self_attn.k_norm.weight"), at("self_attn.k_norm.weight"))
        self._write(ctx.with_subname("self_attn.out_proj.weight"), at("self_attn.output_proj.weight"))

        self._write(ctx.with_subname("cross_attn.q_proj.weight"), at("cross_attn.q_proj.weight"))
        self._fuse(ctx.with_subname("cross_attn.kv_proj.weight"),
                   at("cross_attn.k_proj.weight"),
                   at("cross_attn.v_proj.weight"))
        self._write(ctx.with_subname("cross_attn.q_norm.weight"), at("cross_attn.q_norm.weight"))
        self._write(ctx.with_subname("cross_attn.k_norm.weight"), at("cross_attn.k_norm.weight"))
        self._write(ctx.with_subname("cross_attn.out_proj.weight"),
                    at("cross_attn.output_proj.weight"))

        # Ungated, unlike the SwiGLU the Qwen3 encoder below uses. layer1 widens to 8192 and
        # layer2 reads all of it.
        self._write(ctx.with_subname("mlp.layer1.weight"), at("mlp.layer1.weight"))
        self._write(ctx.with_subname("mlp.layer2.weight"), at("mlp.layer2.weight"))

        # AdaLN-LoRA: 2048 -> 256 -> 6144, three of them, one per sub-block.
        for which in ("self_attn", "cross_attn", "mlp"):
            adaln = ctx.with_subname(f"adaln_{which}")
            self._write(adaln.with_subname("1.weight"), at(f"adaln_modulation_{which}.1.weight"))
            self._write(adaln.with_subname("2.weight"), at(f"adaln_modulation_{which}.2.weight"))

    # ---- the adapter --------------------------------------------------------------------

    def export_adapter(self, ctx: Context, weights: dict, blocks: int) -> None:
        """The bridge from Qwen3 to the cross-attention Cosmos was pretrained for.

        Its query stream is a T5 vocabulary -- hence 32128 -- embedded here and cross-attended
        into the Qwen3 hidden states. Only its output ever reaches the denoiser.
        """
        self._write(ctx.with_subname("embed.weight"), weights["llm_adapter.embed.weight"])

        for index in range(blocks):
            self._export_adapter_block(
                ctx.with_subname(f"block{index}"), weights, f"llm_adapter.blocks.{index}.")

        self._write(ctx.with_subname("norm.weight"), weights["llm_adapter.norm.weight"])
        self._write(ctx.with_subname("out_proj.weight"), weights["llm_adapter.out_proj.weight"])
        self._write(ctx.with_subname("out_proj.bias"), weights["llm_adapter.out_proj.bias"])

    def _export_adapter_block(self, ctx: Context, weights: dict, prefix: str) -> None:
        def at(name: str) -> torch.Tensor:
            return weights[prefix + name]

        # Every projection here is 1024 wide, but that is not enough to fuse them: what decides it
        # is what they read. Self attention takes all three from the same tensor and fuses whole;
        # cross attention's query comes from the stream and its key and value from the encoder's
        # states, so the query stays its own matrix and only the other two go together.
        attn = ctx.with_subname("self_attn")
        self._fuse(attn.with_subname("qkv_proj.weight"),
                   at("self_attn.q_proj.weight"),
                   at("self_attn.k_proj.weight"),
                   at("self_attn.v_proj.weight"))

        attn = ctx.with_subname("cross_attn")
        self._write(attn.with_subname("q_proj.weight"), at("cross_attn.q_proj.weight"))
        self._fuse(attn.with_subname("kv_proj.weight"),
                   at("cross_attn.k_proj.weight"),
                   at("cross_attn.v_proj.weight"))

        for which in ("self_attn", "cross_attn"):
            attn = ctx.with_subname(which)
            self._write(attn.with_subname("q_norm.weight"), at(f"{which}.q_norm.weight"))
            self._write(attn.with_subname("k_norm.weight"), at(f"{which}.k_norm.weight"))
            self._write(attn.with_subname("out_proj.weight"), at(f"{which}.o_proj.weight"))

        # The adapter carries biases where the denoiser carries none.
        self._write(ctx.with_subname("mlp.fc1.weight"), at("mlp.0.weight"))
        self._write(ctx.with_subname("mlp.fc1.bias"), at("mlp.0.bias"))
        self._write(ctx.with_subname("mlp.fc2.weight"), at("mlp.2.weight"))
        self._write(ctx.with_subname("mlp.fc2.bias"), at("mlp.2.bias"))

        self._write(ctx.with_subname("norm_self_attn.weight"), at("norm_self_attn.weight"))
        self._write(ctx.with_subname("norm_cross_attn.weight"), at("norm_cross_attn.weight"))
        self._write(ctx.with_subname("norm_mlp.weight"), at("norm_mlp.weight"))

    # ---- the text encoder ---------------------------------------------------------------

    def export_qwen3(self, ctx: Context, weights: dict, layers: int) -> None:
        """Qwen3-0.6B, read for its hidden states rather than for a next token.

        No lm_head is exported because nothing here generates: the encoder runs once over the
        prompt and the states are what the adapter cross-attends into.
        """
        self._write(ctx.with_subname("embed.weight"), weights["model.embed_tokens.weight"])

        for index in range(layers):
            self._export_qwen3_layer(
                ctx.with_subname(f"block{index}"), weights, f"model.layers.{index}.")

        self._write(ctx.with_subname("final_norm.weight"), weights["model.norm.weight"])

    def _export_qwen3_layer(self, ctx: Context, weights: dict, prefix: str) -> None:
        def at(name: str) -> torch.Tensor:
            return weights[prefix + name]

        # Grouped query attention: 16 query heads against 8 key/value heads, all of width 128.
        # They fuse anyway -- the runtime splits the result back on those known widths.
        self._fuse(ctx.with_subname("attn.qkv_proj.weight"),
                   at("self_attn.q_proj.weight"),
                   at("self_attn.k_proj.weight"),
                   at("self_attn.v_proj.weight"))
        self._write(ctx.with_subname("attn.q_norm.weight"), at("self_attn.q_norm.weight"))
        self._write(ctx.with_subname("attn.k_norm.weight"), at("self_attn.k_norm.weight"))
        self._write(ctx.with_subname("attn.out_proj.weight"), at("self_attn.o_proj.weight"))

        # flint's swiglu halves the last dimension as swish(x[..D / 2]) * x[D / 2..], so gate must
        # come first and up second. Swapping them is silent and wrong.
        self._fuse(ctx.with_subname("mlp.gate_up_proj.weight"),
                   at("mlp.gate_proj.weight"),
                   at("mlp.up_proj.weight"))
        self._write(ctx.with_subname("mlp.down_proj.weight"), at("mlp.down_proj.weight"))

        self._write(ctx.with_subname("input_norm.weight"), at("input_layernorm.weight"))
        self._write(ctx.with_subname("post_attn_norm.weight"), at("post_attention_layernorm.weight"))

    # ---- the VAE ------------------------------------------------------------------------

    def export_vae(self, ctx: Context, weights: dict) -> None:
        """The Qwen-Image VAE, folded from three dimensions to two.

        Its convolutions are causal in time and its weights are <out, in, kt, kh, kw>. For a
        single frame the causal padding in front is zeros and nothing in the network grows the
        time axis, so each of them is exactly a 2-D convolution over the last temporal slice --
        see tools/causal_conv3d_folding_test.py, which proves it, and docs/anima.md, which says
        why. The time_conv weights are not exported at all: they sit behind a cache branch an
        image never takes.
        """
        for name, tensor in sorted(weights.items()):
            if ".time_conv." in name:
                continue
            self._write(ctx.with_subname(self._vae_name(name)), self._fold(name, tensor))

    @staticmethod
    def _vae_name(name: str) -> str:
        """Keep the checkpoint's own names; only `gamma` is renamed to what it actually is."""
        return name[:-len("gamma")] + "weight" if name.endswith(".gamma") else name

    @staticmethod
    def _fold(name: str, tensor: torch.Tensor) -> torch.Tensor:
        """Narrow a 5-D convolution to 4-D, and a broadcast norm scale to a vector."""
        if tensor.dim() == 5:
            # <out, in, kt, kh, kw> -> <out, in, kh, kw>, keeping the last temporal tap: the two
            # in front of it are multiplied by the zeros the causal padding put there.
            return tensor[:, :, -1]
        if name.endswith(".gamma"):
            # <C, 1, 1, 1> or <C, 1, 1>, kept broadcastable in torch, wanted flat here.
            return tensor.reshape(-1)
        return tensor


def strip_prefix(weights: dict) -> dict:
    """Drop whichever wrapper prefix the denoiser was published under."""
    for prefix in DIT_PREFIXES:
        if any(key.startswith(prefix) for key in weights):
            return {key[len(prefix):]: value for key, value in weights.items()
                    if key.startswith(prefix)}
    return weights


def read(filename: str) -> dict:
    with safe_open(filename, framework="pt") as handle:
        return {key: handle.get_tensor(key) for key in handle.keys()}


def count_blocks(weights: dict, pattern: str) -> int:
    """How many blocks a state dictionary holds, by the highest index it names.

    Matched against the whole name rather than searched for inside it. `blocks.<n>.` occurs in the
    adapter as readily as in the denoiser, and a search rather than a match silently adds the one
    to the other.
    """
    found = {int(m.group(1)) for key in weights if (m := re.match(pattern, key))}
    if not found:
        raise LookupError(f"no blocks matched {pattern!r}")
    if found != set(range(len(found))):
        raise LookupError(f"{pattern!r} has gaps: {sorted(found)}")
    return len(found)


def generate_config(dit: dict, text: dict, blocks: int, layers: int) -> dict:
    """What a reader needs that the tensors do not say.

    A plain dict rather than a ConfigParser, for the same reason `sdxl_exporter.py` uses one: a
    value here is a number or a list of them, and a parser whose values are all strings flattens
    a list back into the text of it -- `"[-0.7571, ...]"`, brackets and all, which is neither a
    list the manifest writer can write nor something the reader can split on commas.
    """

    hidden = dit["x_embedder.proj.1.weight"].shape[0]
    latent = dit["final_layer.linear.weight"].shape[0] // 4

    # ComfyUI's model_detection.py reads the rotary extrapolation ratios off the latent width:
    # sixteen channels is the text-to-image branch, which asks for four times the spatial extent
    # and leaves time alone. Anything else has a different row in that table, and guessing it is
    # a silent error rather than a loud one.
    if latent != 16:
        raise SystemExit(
            f"{latent} latent channels: this exporter only knows the rotary extrapolation "
            f"ratios for the 16 channel text-to-image branch"
        )

    config = {"anima": {
        # the denoiser
        "num_blocks": str(blocks),
        "hidden_size": str(hidden),
        "num_heads": "16",
        "head_dim": str(hidden // 16),
        "mlp_size": str(dit["blocks.0.mlp.layer1.weight"].shape[0]),
        "adaln_lora_dim": str(dit["blocks.0.adaln_modulation_mlp.1.weight"].shape[0]),
        "patch_size": "2",
        "latent_channels": str(latent),
        # (16 + 1) * 2 * 2: Cosmos concatenates a padding-mask channel before patchifying
        "patchify_channels": str(dit["x_embedder.proj.1.weight"].shape[1]),
        # positions: dim_h = head_dim // 6 * 2, dim_w = dim_h, dim_t = the remainder.
        # Each axis has its own base, 10000 * ratio ** (dim / (dim - 2)). The ratio is an NTK
        # factor, not the base itself. They are written into the package rather than assumed by
        # the runtime because assuming them is exactly how they came to be wrong: a flat 10000
        # on all three axes draws a picture that looks right and is 13% off.
        "rope_theta": "10000",
        "rope_h_extrapolation_ratio": f"{reference.H_EXTRAPOLATION_RATIO:g}",
        "rope_w_extrapolation_ratio": f"{reference.W_EXTRAPOLATION_RATIO:g}",
        "rope_t_extrapolation_ratio": f"{reference.T_EXTRAPOLATION_RATIO:g}",
        # the adapter
        "adapter_blocks": str(count_blocks(dit, r"llm_adapter\.blocks\.(\d+)\.")),
        "adapter_hidden_size": str(dit["llm_adapter.out_proj.weight"].shape[0]),
        "adapter_vocab_size": str(dit["llm_adapter.embed.weight"].shape[0]),
        "adapter_num_heads": "16",
        "context_length": "512",
        # the text encoder
        "text_num_layers": str(layers),
        "text_hidden_size": str(text["model.embed_tokens.weight"].shape[1]),
        "text_vocab_size": str(text["model.embed_tokens.weight"].shape[0]),
        "text_num_heads": "16",
        "text_num_kv_heads": "8",
        "text_head_dim": "128",
        "text_mlp_size": str(text["model.layers.0.mlp.gate_proj.weight"].shape[0]),
        "text_rope_theta": "1000000",
        "text_rms_norm_eps": "1e-6",
        # the VAE: sixteen channels normalized one at a time, where SDXL has a single factor
        "latents_mean": [float(f"{value:g}") for value in LATENTS_MEAN],
        "latents_std": [float(f"{value:g}") for value in LATENTS_STD],
        # rectified flow, not the epsilon prediction every SDXL checkpoint carries
        "prediction": "flow",
        # How the schedule is bent and what the denoiser is handed. Neither is in any tensor:
        # they are Anima's entry in ComfyUI's model table. The multiplier of one is the unusual
        # half -- a flow model is normally passed a thousandfold timestep, and this one is passed
        # the sigma itself.
        "sampler_shift": "3.0",
        "sampler_multiplier": "1.0",
    }}
    return config


# What the reference outputs are computed for. Short, so that the tensors stay small, and fixed so
# that a change in them is a change in the model rather than in the prompt.
TEST_PROMPT = "masterpiece, best quality, 1girl, solo, long hair, brown eyes, school uniform, smile"

# Where the two tokenizers come from when no local copy is pointed at. Neither is published beside
# Anima's weights: the encoder is stock Qwen3-0.6B, and the T5 vocabulary the adapter's query
# stream is written in is the one Cosmos-Predict2 was pretrained against.
QWEN_TOKENIZER = "Qwen/Qwen3-0.6B-Base"
T5_TOKENIZER = "google/t5-v1_1-xxl"

# What the manifest calls them. Two of them, so neither can be the unnamed one a model with a
# single tokenizer carries. Each block names its vocabulary, which sits beside the manifest.
QWEN_TOKENIZER_SECTION = "qwen3"
T5_TOKENIZER_SECTION = "t5"
QWEN_TOKENIZER_SUFFIX = ".qwen3.tokenizer.json"
T5_TOKENIZER_SUFFIX = ".t5.tokenizer.json"
# Named after the test package and not after the model, so that the `_test` its name already
# carries is not written twice: `anima-turbo-v11_test.safetensors` puts its two corpora in
# `anima-turbo-v11_test_qwen3_corpus.tsv` and its neighbour, which is what
# `waifu/tests/anima_tokenizer.rs` opens.
QWEN_TOKENIZER_CORPUS = "_qwen3_corpus.tsv"
T5_TOKENIZER_CORPUS = "_t5_corpus.tsv"


def read_t5_tokenizer(source: str):
    """T5's tokenizer, from a `spiece.model`, a directory holding one, or a hub repository.

    A bare `spiece.model` is read by pointing transformers at the directory around it, which is
    what it wants; the conversion to a `tokenizer.json` is the same either way.
    """
    if path.isfile(source):
        source = path.dirname(source) or "."
    return read_tokenizer(source)


def reference_encoders(qwen, t5):
    """What the two tokenizers make of a text, as bare ids, and T5's end marker.

    These are the same two objects whose `tokenizer.json` goes into the package, which is the
    point: the corpus and the package cannot describe two different tokenizers, so what the corpus
    checks is that the runtime reads back the file the exporter wrote.

    Bare, because appending the end marker is the pipeline's business and not the tokenizer's, the
    same way the SDXL package's ids carry neither of CLIP's markers. `add_special_tokens=False` is
    what says so to both -- Qwen3 adds nothing of its own anyway, T5 would append `</s>`.
    """
    return encoder(qwen), encoder(t5), t5.eos_token_id


def export_test_cases(dit: dict, text: dict, vae: dict, qwen_ids, t5_ids) -> TensorBag:
    """Write what the reference makes of one prompt, so the runtime can be checked against it.

    The same idea as `sdxl_exporter.export_test_cases`, against `anima_comfy` -- which is ComfyUI -- rather than
    against diffusers, and covering the path in the order a picture travels it: the encoder's
    hidden states, the adapter's context, one denoising step, and one decode.

    All of it on a 16 by 16 latent -- a 128 by 128 image, 64 patches -- which is large enough to
    exercise every block and small enough to carry in the repository.

    `qwen_ids` and `t5_ids` are what the two tokenizers made of `TEST_PROMPT`, and they are written
    beside the tensors. They used to be written down in this file instead, so that the reference
    outputs did not depend on having either tokenizer to hand -- but four of the twenty were wrong,
    which nothing noticed, because the reference ran on the same wrong ids the runtime was checked
    against. The model carries the tokenizers, so there is nowhere left for a second copy of the
    answer to disagree.
    """
    ctx = Context("test_case")
    writer = TensorBag()
    def put(name: str, tensor: torch.Tensor) -> None:
        writer.write_tensor(ctx.with_subname(name), tensor, preserve_dtype=True)

    qwen_ids = torch.tensor([qwen_ids])
    t5_ids = torch.tensor([t5_ids])
    writer.write_tensor(ctx.with_subname("qwen_ids"), qwen_ids.to(torch.int64))
    writer.write_tensor(ctx.with_subname("t5_ids"), t5_ids.to(torch.int64))

    blocks = count_blocks(dit, r"blocks\.(\d+)\.")
    width = dit["x_embedder.proj.1.weight"].shape[0]

    with torch.no_grad():
        hidden = reference.TextEncoder(text)(qwen_ids)
        put("hidden", hidden)

        denoiser = reference.Denoiser(dit, blocks, width)

        # Before padding, which is the adapter's own output, and after, which is what the
        # denoiser actually reads. A runtime can go wrong in either.
        put("context", denoiser.adapt(hidden, t5_ids))
        padded = denoiser.adapt_padded(hidden, t5_ids)
        put("context_padded", padded)

        generator = torch.Generator().manual_seed(11)
        latent = torch.randn(1, 16, 16, 16, generator=generator)
        timestep = torch.tensor([0.75])
        put("latent", latent)
        put("timestep", timestep)

        # One step's velocity. Not a denoised latent: what the model returns is the flow, and
        # the sampler is checked against a schedule rather than against weights.
        put("velocity", denoiser(latent, timestep, padded))

        decoder = reference.VaeDecoder(vae, LATENTS_MEAN, LATENTS_STD)
        put("decoded", decoder(latent))

    return writer


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-dit", required=True, help="the denoiser safetensors.")
    parser.add_argument("-text", required=True, help="the Qwen3-0.6B text encoder safetensors.")
    parser.add_argument("-vae", required=True, help="the Qwen-Image VAE safetensors.")
    parser.add_argument("-output", default="anima.safetensors",
                        help="what to call the model. Names the weights and the manifest beside "
                             "them.")
    parser.add_argument(
        "-qwen_tokenizer", default=QWEN_TOKENIZER,
        help="where the text encoder's vocabulary comes from, as a directory or a hub repository. "
             "Anima publishes the encoder's weights without it, so this is stock Qwen3.")
    parser.add_argument(
        "-t5_tokenizer", default=T5_TOKENIZER,
        help="where the adapter's vocabulary comes from, as a spiece.model, a directory holding "
             "one, or a hub repository. The prompt is tokenized twice and this is the second of "
             "the two -- see docs/anima.md.")
    parser.add_argument(
        "-part-size", type=parse_size, default=None,
        help='split the package into parts of about this size, as in "2GB".')
    parser.add_argument(
        "-test_output", type=str, default=None,
        help="also write reference inputs and outputs here, for the runtime to be checked "
             "against. Runs the whole model on the CPU, which takes a few minutes, and needs "
             "-comfyui.")
    parser.add_argument(
        "-comfyui", type=str, default=None,
        help="a ComfyUI checkout, which is the reference the test outputs are computed with -- "
             "the same arrangement sdxl_exporter.py has with diffusers. Only -test_output needs "
             "it. See tools/anima_comfy.py for what to install and where.")
    args = parser.parse_args()

    # Before anything is read, so a missing checkout costs nothing but the message. The path goes
    # on sys.path here and the modules are imported stage by stage as they are needed.
    if args.test_output:
        if not args.comfyui:
            raise SystemExit("-test_output needs -comfyui: the reference outputs are what ComfyUI "
                             "computes, and there is no second implementation here to ask")
        reference.use(args.comfyui)

    dit = strip_prefix(read(args.dit))
    text = read(args.text)
    vae = read(args.vae)

    blocks = count_blocks(dit, r"blocks\.(\d+)\.")
    adapter_blocks = count_blocks(dit, r"llm_adapter\.blocks\.(\d+)\.")
    layers = count_blocks(text, r"model\.layers\.(\d+)\.")
    print(f"{blocks} denoiser blocks, {adapter_blocks} adapter blocks, {layers} encoder layers")

    qwen_tokenizer = read_tokenizer(args.qwen_tokenizer)
    t5_tokenizer = read_t5_tokenizer(args.t5_tokenizer)

    stem = stem_of(args.output)
    directory = path.dirname(path.abspath(args.output))

    # Both `tokenizer.json` files, beside the manifest, and a line each in it naming them. Two
    # rather than one, because it is one question -- how is this model's text turned into ids --
    # with two answers that do different jobs, and neither is the one to assume.
    qwen_file = stem + QWEN_TOKENIZER_SUFFIX
    t5_file = stem + T5_TOKENIZER_SUFFIX
    for name, tokenizer in ((qwen_file, qwen_tokenizer), (t5_file, t5_tokenizer)):
        with open(path.join(directory, name), "wb") as fp:
            fp.write(tokenizer_json(tokenizer))
        print(f"wrote {name}")

    writer = open_weights(args.output, args.part_size)
    converter = Converter(writer)
    converter.export_dit(Context("anima.dit"), dit, blocks)
    converter.export_adapter(Context("anima.adapter"), dit, adapter_blocks)
    converter.export_qwen3(Context("anima.text"), text, layers)
    converter.export_vae(Context("anima.vae"), vae)

    config = generate_config(dit, text, blocks, layers)
    config["model"] = {"type": "anima"}

    tokenizers = {QWEN_TOKENIZER_SECTION: qwen_file, T5_TOKENIZER_SECTION: t5_file}
    for name in writer.finish(config, None, tokenizers):
        print(f"wrote {name}")
    print(f"{converter.written} tensors, {converter.widened} of them narrowed from bfloat16")

    if args.test_output:
        qwen_encode, t5_encode, t5_eos = reference_encoders(qwen_tokenizer, t5_tokenizer)

        # The end marker is on the ids the reference ran, because it is on the ids the pipeline
        # hands the adapter. The corpora below hold the bare ones.
        export_test_cases(dit, text, vae, qwen_encode(TEST_PROMPT),
                          t5_encode(TEST_PROMPT) + [t5_eos]).save(args.test_output)

        test_stem = stem_of(args.test_output)
        test_dir = path.dirname(path.abspath(args.test_output))
        for suffix, encode in ((QWEN_TOKENIZER_CORPUS, qwen_encode),
                               (T5_TOKENIZER_CORPUS, t5_encode)):
            corpus = tokenizer_corpus(encode)
            name = test_stem + suffix
            with open(path.join(test_dir, name), "w", encoding="utf-8") as fp:
                fp.write(corpus)
            print(f"wrote {corpus.count(chr(10))} lines to {name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
