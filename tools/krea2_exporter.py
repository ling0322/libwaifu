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

"""Export a Krea 2 release to safetensors and the manifest that names them.

Krea 2 is neither an SDXL fine tune nor an Anima release: it is a twelve billion parameter
single-stream MMDiT, a Qwen3-VL-4B text encoder read at twelve of its layers, and the same
Qwen-Image VAE Anima draws through. The whole of what it is is written down in `docs/krea2.md`
-- read that first, because several of the decisions here only make sense against it.

This reads the published diffusers layout as it stands, which is what `hf download` leaves on the
disk:

    .venv/bin/python tools/krea2_exporter.py \\
        -model  models/krea2-turbo \\
        -output models/krea2-turbo.safetensors

`-fp8` writes the two halves that have matrices in them as E4M3 with a scale per output channel,
which halves the package -- twenty-four gigabytes to twelve -- for about 2.6e-2 of relative error
on each weight. See `docs/fp8.md`.

The released weights are gated and carry the Krea 2 Community License. Nothing here publishes
anything; what it writes is a package on your own disk, and where it goes afterwards is a decision
this tool does not make.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from os import path

import torch
from safetensors import safe_open

from tokenizer_exporter import encoder, read_tokenizer, tokenizer_corpus, tokenizer_json
from model_exporter import Context, TensorBag, open_weights, parse_size, stem_of

# fp16's largest finite value. bf16 carries fp32's exponent range and fp16 does not, so a weight
# that was comfortable in the checkpoint can become an infinity here. The check is cheap and the
# alternative is a picture of noise with nothing to say why.
FP16_MAX = 65504.0

# How the Qwen-Image VAE normalizes its sixteen latent channels: a mean and a deviation per
# channel rather than SDXL's single factor. They belong to the VAE and are read out of its
# `config.json` rather than written down here, since this model publishes one.
VAE_CONFIG = path.join("vae", "config.json")

# The timestep shift the distilled release was trained at. The reference writes the schedule as
# `exp(mu) / (exp(mu) + (1 / t - 1))` and this runtime writes it as `shift * t / (1 + (shift - 1)
# * t)`, which is the same curve with `shift = exp(mu)` -- so what goes in the package is that
# exponential rather than the mu itself.
DISTILLED_MU = 1.15

# What the undistilled release would want instead: a mu interpolated between these by how many
# image tokens are being drawn. Only the distilled one is exported here; a package for the other
# would have to carry the interpolation rather than one number, which the manifest has no room for
# yet and `docs/krea2.md` records as the reason it is turbo only.
BASE_SHIFT, MAX_SHIFT = 0.5, 1.15


def fold_norm(tensor: torch.Tensor) -> torch.Tensor:
    """A zero-centred RMSNorm scale, as the multiplier it stands for.

    Every RMSNorm in the denoiser multiplies by `1 + weight`; flint's multiplies by the weight. So
    the one is folded in here, once, rather than by an operator that would have to know which of
    the two kinds of norm it was running. The encoder's norms are *not* folded: Qwen3 stores the
    multiplier itself, and adding one to those is a model that draws grey soup.
    """
    return tensor.to(torch.float32) + 1.0


class Converter:
    """Reads the published checkpoints and writes one parameter file.

    Every tensor goes through `_write`, which is also where bfloat16 becomes something the format
    can hold. Tensors are fused where the runtime wants them fused -- one projection for the
    query, key, value and gate rather than four -- because that is three matrix multiplies saved
    on every attention of every block of every step.
    """

    def __init__(self, writer, fp8: bool) -> None:
        self._writer = writer
        self._fp8 = fp8
        self._widened = 0
        self._quantized = 0
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

    def _matrix(self, ctx: Context, tensor: torch.Tensor) -> None:
        """Write a weight the runtime multiplies by, quantized where the package is quantized.

        Only these. A norm scale, a bias, a modulation table and the twelve-wide projector are
        read as they are whatever the rest of the model is stored as -- the first three because
        nothing multiplies by them in the sense fp8 means, and the projector because an fp8
        multiply wants its inner dimension in multiples of sixteen and twelve is not one.
        """
        if not self._fp8:
            return self._write(ctx, tensor)

        self._writer.write_fp8_tensor(ctx, tensor.to(torch.float32))
        self._quantized += 1
        self._count += 1

    def _fuse(self, ctx: Context, *tensors: torch.Tensor) -> None:
        """Write several projections as one, concatenated on the output dimension."""
        self._matrix(ctx, torch.cat([t.to(torch.float32) for t in tensors], dim=0))

    @property
    def written(self) -> int:
        return self._count

    @property
    def widened(self) -> int:
        return self._widened

    @property
    def quantized(self) -> int:
        return self._quantized

    # ---- the denoiser -------------------------------------------------------------------

    def export_dit(self, ctx: Context, weights: dict, shape: dict) -> None:
        """The single-stream MMDiT: the prompt's fusion, the timestep, N blocks, the last layer."""
        self._write(ctx.with_subname("img_in.bias"), weights["img_in.bias"])
        self._matrix(ctx.with_subname("img_in.weight"), weights["img_in.weight"])

        for name in ("time_embed.linear_1", "time_embed.linear_2", "time_mod_proj"):
            self._matrix(ctx.with_subname(f"{name}.weight"), weights[f"{name}.weight"])
            self._write(ctx.with_subname(f"{name}.bias"), weights[f"{name}.bias"])

        self._export_text_fusion(ctx.with_subname("text_fusion"), weights, shape)

        # The projection that widens the fused prompt to the model. Its norm is one of the
        # zero-centred ones; its two linears carry biases.
        txt = ctx.with_subname("txt_in")
        self._write(txt.with_subname("norm.weight"), fold_norm(weights["txt_in.norm.weight"]))
        for name in ("linear_1", "linear_2"):
            self._matrix(txt.with_subname(f"{name}.weight"), weights[f"txt_in.{name}.weight"])
            self._write(txt.with_subname(f"{name}.bias"), weights[f"txt_in.{name}.bias"])

        for index in range(shape["num_blocks"]):
            self._export_block(
                ctx.with_subname(f"block{index}"), weights, f"transformer_blocks.{index}.",
                shape["num_heads"] * shape["head_dim"], shape["num_kv_heads"] * shape["head_dim"])

        # The last layer modulates on two parts rather than six, and on the timestep embedding
        # rather than on the vector the blocks share.
        final = ctx.with_subname("final")
        self._write(final.with_subname("scale_shift_table"),
                    weights["final_layer.scale_shift_table"])
        self._write(final.with_subname("norm.weight"), fold_norm(weights["final_layer.norm.weight"]))
        self._matrix(final.with_subname("linear.weight"), weights["final_layer.linear.weight"])
        self._write(final.with_subname("linear.bias"), weights["final_layer.linear.bias"])

    def _export_text_fusion(self, ctx: Context, weights: dict, shape: dict) -> None:
        """Two blocks across the twelve tapped layers, a matrix that collapses them, two along
        the sentence."""
        width = shape["text_hidden_size"]
        queries = keys = shape["text_num_heads"] * (width // shape["text_num_heads"])

        for kind, count in (("layerwise", shape["text_layerwise_blocks"]),
                            ("refiner", shape["text_refiner_blocks"])):
            for index in range(count):
                self._export_attention_block(
                    ctx.with_subname(f"{kind}{index}"), weights,
                    f"text_fusion.{kind}_blocks.{index}.", queries, keys)

        # Twelve numbers. Not `_matrix`: see `_matrix` for why twelve is not a width fp8 takes.
        self._write(ctx.with_subname("projector.weight"), weights["text_fusion.projector.weight"])

    def _export_block(self, ctx: Context, weights: dict, prefix: str, queries: int,
                      keys: int) -> None:
        """One block of the stream: an attention and a SwiGLU, and the table that modulates them."""
        self._write(ctx.with_subname("scale_shift_table"), weights[prefix + "scale_shift_table"])
        self._export_attention_block(ctx, weights, prefix, queries, keys)

    def _export_attention_block(self, ctx: Context, weights: dict, prefix: str, queries: int,
                                keys: int) -> None:
        """What every block here is made of, whether it is six thousand wide or two.

        The query, key, value and gate are one matrix in that order, which is the order the
        runtime slices it back on -- and the widths it slices at are the model's, so a checkpoint
        whose gate came before its value would load and draw noise.
        """
        def at(name: str) -> torch.Tensor:
            return weights[prefix + name]

        self._write(ctx.with_subname("norm1.weight"), fold_norm(at("norm1.weight")))
        self._write(ctx.with_subname("norm2.weight"), fold_norm(at("norm2.weight")))

        self._fuse(ctx.with_subname("attn.qkvg_proj.weight"),
                   at("attn.to_q.weight"),
                   at("attn.to_k.weight"),
                   at("attn.to_v.weight"),
                   at("attn.to_gate.weight"))
        self._write(ctx.with_subname("attn.q_norm.weight"), fold_norm(at("attn.norm_q.weight")))
        self._write(ctx.with_subname("attn.k_norm.weight"), fold_norm(at("attn.norm_k.weight")))
        self._matrix(ctx.with_subname("attn.out_proj.weight"), at("attn.to_out.0.weight"))

        # flint's swiglu halves the last dimension as swish(x[..D / 2]) * x[D / 2..], so the gate
        # comes first and the up projection second. Swapping them is silent and wrong.
        self._fuse(ctx.with_subname("ff.gate_up_proj.weight"),
                   at("ff.gate.weight"),
                   at("ff.up.weight"))
        self._matrix(ctx.with_subname("ff.down.weight"), at("ff.down.weight"))

    # ---- the text encoder ---------------------------------------------------------------

    def export_encoder(self, ctx: Context, weights: dict, layers: int) -> None:
        """Qwen3-VL's language half, read for hidden states rather than for a next token.

        Neither the vision tower nor the final norm is exported. The first is never reached -- no
        picture is handed to this encoder -- and the second is applied to nothing the denoiser
        reads: the twelve tapped layers all come off the residual stream before it.
        """
        self._write(ctx.with_subname("embed.weight"), weights["language_model.embed_tokens.weight"])

        for index in range(layers):
            self._export_encoder_layer(
                ctx.with_subname(f"block{index}"), weights, f"language_model.layers.{index}.")

    def _export_encoder_layer(self, ctx: Context, weights: dict, prefix: str) -> None:
        def at(name: str) -> torch.Tensor:
            return weights[prefix + name]

        # Grouped query attention: thirty-two query heads against eight for the keys, all 128
        # wide. They fuse anyway -- the runtime splits the result back on those known widths.
        self._fuse(ctx.with_subname("attn.qkv_proj.weight"),
                   at("self_attn.q_proj.weight"),
                   at("self_attn.k_proj.weight"),
                   at("self_attn.v_proj.weight"))
        # Qwen3 stores its norm scales as the multiplier, so these are not folded.
        self._write(ctx.with_subname("attn.q_norm.weight"), at("self_attn.q_norm.weight"))
        self._write(ctx.with_subname("attn.k_norm.weight"), at("self_attn.k_norm.weight"))
        self._matrix(ctx.with_subname("attn.out_proj.weight"), at("self_attn.o_proj.weight"))

        self._fuse(ctx.with_subname("mlp.gate_up_proj.weight"),
                   at("mlp.gate_proj.weight"),
                   at("mlp.up_proj.weight"))
        self._matrix(ctx.with_subname("mlp.down_proj.weight"), at("mlp.down_proj.weight"))

        self._write(ctx.with_subname("input_norm.weight"), at("input_layernorm.weight"))
        self._write(ctx.with_subname("post_attn_norm.weight"),
                    at("post_attention_layernorm.weight"))

    # ---- the VAE ------------------------------------------------------------------------

    def export_vae(self, ctx: Context, weights: dict) -> None:
        """The Qwen-Image VAE, both halves, folded from three dimensions to two and renamed.

        Its convolutions are causal in time and its weights are <out, in, kt, kh, kw>. For a
        single frame the causal padding in front is zeros and nothing in the network grows the
        time axis, so each of them is exactly a 2-D convolution over the last temporal slice --
        see tools/causal_conv3d_folding_test.py, which proves it, and docs/anima.md, which says
        why. The `time_conv` weights are not exported at all: they sit behind a cache branch an
        image never takes.

        This is the same autoencoder Anima carries and the same layout in the package, but it
        arrives under different names: Anima's is published as the original checkpoint and Krea 2
        publishes the diffusers conversion of it. `vae_name` is that difference and nothing else.
        """
        both = {name: tensor for name, tensor in weights.items()
                if ".time_conv." not in name}

        for name, tensor in sorted(both.items()):
            self._write(ctx.with_subname(vae_name(name)), fold_conv(name, tensor))


def fold_conv(name: str, tensor: torch.Tensor) -> torch.Tensor:
    """Narrow a 5-D convolution to 4-D, and a broadcast norm scale to a vector."""
    if tensor.dim() == 5:
        # <out, in, kt, kh, kw> -> <out, in, kh, kw>, keeping the last temporal tap: the two in
        # front of it are multiplied by the zeros the causal padding put there.
        return tensor[:, :, -1]
    if name.endswith(".gamma"):
        # <C, 1, 1, 1> or <C, 1, 1>, kept broadcastable in torch, wanted flat here.
        return tensor.reshape(-1)
    return tensor


# The decoder's stages are nested in the diffusers layout -- `up_blocks.N.resnets.M` and
# `up_blocks.N.upsamplers.0` -- where the original checkpoint keeps one flat `upsamples` list. The
# encoder's are flat in both. The runtime reads the flat shape, because that is what it works a
# decoder's widths out from.
_UP_BLOCK = re.compile(r"^decoder\.up_blocks\.(\d+)\.(resnets|upsamplers)\.(\d+)\.(.*)$")
_DOWN_BLOCK = re.compile(r"^encoder\.down_blocks\.(\d+)\.(.*)$")
_MID_BLOCK = re.compile(r"^(en|de)coder\.mid_block\.(resnets|attentions)\.(\d+)\.(.*)$")

#: What a residual block's parts are called in each layout. A name that is not one of these -- a
#: resample's own convolution, say -- passes through as it stands.
_RESNET = {"norm1": "residual.0", "conv1": "residual.2", "norm2": "residual.3",
           "conv2": "residual.6", "conv_shortcut": "shortcut"}


def _block_name(tail: str) -> str:
    """One part of a residual block, renamed, or the tail unchanged where it is not one."""
    part, _, rest = tail.partition(".")
    return f"{_RESNET[part]}.{rest}" if part in _RESNET else tail


def vae_name(name: str) -> str:
    """One diffusers name for the Qwen-Image VAE, as the original checkpoint spells it.

    `quant_conv` is the encoder's moment projection, which the original calls `conv1`, and
    `post_quant_conv` is the decoder's, which it calls `conv2`; everything else is a renaming of a
    block. A name this does not know is an error rather than a tensor quietly dropped: the
    runtime would then fail at load with a missing weight and nothing to say which.
    """
    name = name.replace(".gamma", ".weight")

    if name.startswith("post_quant_conv."):
        return name.replace("post_quant_conv.", "conv2.")
    if name.startswith("quant_conv."):
        return name.replace("quant_conv.", "conv1.")

    if match := _MID_BLOCK.match(name):
        half, kind, index, tail = match.groups()
        # middle.0 and middle.2 are the two residual blocks, middle.1 the attention between them.
        at = 2 * int(index) if kind == "resnets" else 1
        return f"{half}coder.middle.{at}.{_block_name(tail) if kind == 'resnets' else tail}"

    if match := _UP_BLOCK.match(name):
        block, kind, index, tail = match.groups()
        # The flat list: each stage's residual blocks, then its resample where it has one. How
        # many residual blocks a stage holds is the checkpoint's to say, so it is counted rather
        # than assumed -- `count_stages` does that once and this reads the answer.
        offset, resnets = _stage_offsets[int(block)]
        at = offset + int(index) + (0 if kind == "resnets" else resnets)
        return f"decoder.upsamples.{at}.{_block_name(tail) if kind == 'resnets' else tail}"

    if match := _DOWN_BLOCK.match(name):
        # Already flat, and in the same order, so only the parts of a residual block are renamed.
        index, tail = match.groups()
        return f"encoder.downsamples.{index}.{_block_name(tail)}"

    if name.endswith("conv_in.weight") or name.endswith("conv_in.bias"):
        return name.replace("conv_in.", "conv1.")
    if name.endswith("norm_out.weight"):
        return name.replace("norm_out.", "head.0.")
    if name.endswith("conv_out.weight") or name.endswith("conv_out.bias"):
        return name.replace("conv_out.", "head.2.")

    raise LookupError(f"{name} is not a name this knows how to rewrite")


#: Where each diffusers stage starts in the original's flat `upsamples` list, and how many
#: residual blocks it holds -- which is where its resample lands. Filled in by `count_stages`.
_stage_offsets: dict = {}


def count_stages(weights: dict) -> None:
    """Work out where each `up_blocks.N` begins in the flat `upsamples` list, from the weights.

    Read rather than written down: the decoder is three residual blocks and a resample per stage
    today, the last stage has no resample at all, and a VAE that was shaped differently would be
    renamed wrongly and silently by a table that assumed this one.
    """
    sizes: dict = {}
    for name in weights:
        if match := _UP_BLOCK.match(name):
            block, kind, index = int(match.group(1)), match.group(2), int(match.group(3))
            held = sizes.setdefault(block, {"resnets": 0, "other": 0})
            key = "resnets" if kind == "resnets" else "other"
            held[key] = max(held[key], index + 1)

    _stage_offsets.clear()
    running = 0
    for block in sorted(sizes):
        _stage_offsets[block] = (running, sizes[block]["resnets"])
        running += sizes[block]["resnets"] + sizes[block]["other"]


def read_json(directory: str, *parts: str) -> dict:
    with open(path.join(directory, *parts), encoding="utf-8") as fp:
        return json.load(fp)


def read_weights(*files: str) -> dict:
    """Every tensor of every file, by name."""
    held = {}
    for file in files:
        with safe_open(file, framework="pt") as handle:
            for key in handle.keys():
                held[key] = handle.get_tensor(key)
    return held


def transformer_files(directory: str) -> list:
    """The denoiser's safetensors, whether it was published as one file or as shards."""
    index = path.join(directory, "transformer", "diffusion_pytorch_model.safetensors.index.json")
    if path.exists(index):
        with open(index, encoding="utf-8") as fp:
            names = sorted(set(json.load(fp)["weight_map"].values()))
        return [path.join(directory, "transformer", name) for name in names]

    return [path.join(directory, "transformer", "diffusion_pytorch_model.safetensors")]


def count_blocks(weights: dict, pattern: str) -> int:
    """How many blocks a state dictionary holds, by the highest index it names."""
    found = {int(m.group(1)) for key in weights if (m := re.match(pattern, key))}
    if not found:
        raise LookupError(f"no blocks matched {pattern!r}")
    if found != set(range(len(found))):
        raise LookupError(f"{pattern!r} has gaps: {sorted(found)}")
    return len(found)


def shape_of(model: dict, text: dict) -> dict:
    """The numbers the exporter itself needs, checked against the weights that carry them.

    Every one of these is in the published `config.json` and every one of them is also implied by
    a tensor's shape. Read from the first and checked against the second, because a configuration
    that disagrees with its own weights is the one failure here that would otherwise produce a
    package that loads.
    """
    return {
        "num_blocks": model["num_layers"],
        "num_heads": model["num_attention_heads"],
        "num_kv_heads": model["num_key_value_heads"],
        "head_dim": model["attention_head_dim"],
        "text_hidden_size": model["text_hidden_dim"],
        "text_num_heads": model["text_num_attention_heads"],
        "text_layerwise_blocks": model["num_layerwise_text_blocks"],
        "text_refiner_blocks": model["num_refiner_text_blocks"],
        "encoder_layers": text["text_config"]["num_hidden_layers"],
    }


def check_shapes(shape: dict, dit: dict, text: dict) -> None:
    """What the configuration claims, against what the tensors are."""
    hidden = shape["num_heads"] * shape["head_dim"]
    checks = [
        ("img_in.weight", dit["img_in.weight"].shape[0], hidden),
        ("transformer_blocks.0.attn.to_q.weight",
         dit["transformer_blocks.0.attn.to_q.weight"].shape[0], hidden),
        ("transformer_blocks.0.attn.to_k.weight",
         dit["transformer_blocks.0.attn.to_k.weight"].shape[0],
         shape["num_kv_heads"] * shape["head_dim"]),
        ("text_fusion.projector.weight", dit["text_fusion.projector.weight"].shape[1],
         len(shape["select_layers"])),
        ("language_model.embed_tokens.weight",
         text["language_model.embed_tokens.weight"].shape[1], shape["encoder_hidden_size"]),
    ]
    for name, found, wanted in checks:
        if found != wanted:
            raise SystemExit(
                f"{name} is {found} where the configuration says {wanted}; this checkpoint and "
                f"its config.json do not describe the same model")

    counted = count_blocks(dit, r"transformer_blocks\.(\d+)\.")
    if counted != shape["num_blocks"]:
        raise SystemExit(f"{counted} blocks in the weights, {shape['num_blocks']} in the config")
    counted = count_blocks(text, r"language_model\.layers\.(\d+)\.")
    if counted != shape["encoder_layers"]:
        raise SystemExit(
            f"{counted} encoder layers in the weights, {shape['encoder_layers']} in the config")


def generate_config(model: dict, text: dict, index: dict, vae: dict, shape: dict,
                    fp8: bool) -> dict:
    """What a reader needs that the tensors do not say.

    A plain dict rather than a ConfigParser, for the same reason `anima_exporter.py` uses one: a
    value here is a number or a list of them, and a parser whose values are all strings flattens a
    list back into the text of it.
    """
    encoder = text["text_config"]
    patch = index.get("patch_size", 2)
    latent = vae["z_dim"]

    # Only the distilled release has one shift for every resolution. The undistilled one
    # interpolates between two by how many image tokens are being drawn, and this format has one
    # number, so it is refused here rather than exported at the wrong schedule.
    if not index.get("is_distilled", False):
        raise SystemExit(
            "this is not the distilled release: its timestep shift depends on the resolution, "
            "and a package says one shift. See docs/krea2.md.")

    config = {"krea2": {
        # the denoiser
        "num_blocks": str(shape["num_blocks"]),
        "hidden_size": str(shape["num_heads"] * shape["head_dim"]),
        "num_heads": str(shape["num_heads"]),
        "num_kv_heads": str(shape["num_kv_heads"]),
        "head_dim": str(shape["head_dim"]),
        "mlp_size": str(model["intermediate_size"]),
        "patch_size": str(patch),
        "latent_channels": str(latent),
        "in_channels": str(model["in_channels"]),
        "timestep_embed_dim": str(model["timestep_embed_dim"]),
        # One base for all three axes, and the split between them written out: it is a choice of
        # the model's rather than a formula, and 32 of every 128 go to a time axis a still picture
        # never leaves.
        "rope_theta": f"{model['rope_theta']:g}",
        "rope_axes": [int(axis) for axis in model["axes_dims_rope"]],
        "norm_eps": f"{model['norm_eps']:g}",
        # the text fusion inside it
        "text_hidden_size": str(model["text_hidden_dim"]),
        "text_num_heads": str(model["text_num_attention_heads"]),
        "text_num_kv_heads": str(model["text_num_key_value_heads"]),
        "text_mlp_size": str(model["text_intermediate_size"]),
        "text_layerwise_blocks": str(model["num_layerwise_text_blocks"]),
        "text_refiner_blocks": str(model["num_refiner_text_blocks"]),
        "num_text_layers": str(model["num_text_layers"]),
        "context_length": "512",
        # the text encoder
        "encoder_num_layers": str(encoder["num_hidden_layers"]),
        "encoder_hidden_size": str(encoder["hidden_size"]),
        "encoder_vocab_size": str(encoder["vocab_size"]),
        "encoder_num_heads": str(encoder["num_attention_heads"]),
        "encoder_num_kv_heads": str(encoder["num_key_value_heads"]),
        "encoder_head_dim": str(encoder["head_dim"]),
        "encoder_mlp_size": str(encoder["intermediate_size"]),
        "encoder_rope_theta": f"{encoder['rope_parameters']['rope_theta']:g}",
        "encoder_rms_norm_eps": f"{encoder['rms_norm_eps']:g}",
        # Which layers are tapped, counting the embedding output as zero. Not derivable from
        # anything: it is a choice the pipeline records and the weights do not.
        "encoder_select_layers": [int(layer) for layer in shape["select_layers"]],
        # the VAE: sixteen channels normalized one at a time, where SDXL has a single factor
        "latents_mean": [float(f"{value:g}") for value in vae["latents_mean"]],
        "latents_std": [float(f"{value:g}") for value in vae["latents_std"]],
        # rectified flow, not the epsilon prediction every SDXL checkpoint carries
        "prediction": "flow",
        # The schedule, as this runtime writes it. `exp(mu)` rather than `mu`, because the two
        # references write the same curve two ways -- see DISTILLED_MU above. The multiplier of
        # one is the unusual half: a flow model is normally handed a thousandfold timestep, and
        # this one embeds the thousand itself, so what it is passed is the sigma.
        "sampler_shift": f"{math.exp(DISTILLED_MU):.6g}",
        "sampler_multiplier": "1.0",
    }}

    if fp8:
        config["krea2"]["weight_format"] = "fp8"

    return config


# What the reference outputs are computed for. Short, so that the tensors stay small, and fixed so
# that a change in them is a change in the model rather than in the prompt.
TEST_PROMPT = "a red fox sitting in fresh snow, golden hour, photorealistic"

# The picture the fixtures are taken off: 128 by 128 is a 16 by 16 latent and 64 image tokens,
# which is large enough to exercise every block and small enough to carry in the repository, and
# eight steps is what the distilled release is for.
TEST_SIDE = 128
TEST_STEPS = 8
TEST_CHANNELS = 16

# Which step of those eight the denoiser is measured on. Halfway: the latent there is neither the
# noise it started from nor the picture it ends at, which is the only part of the walk where
# being wrong about the schedule and being wrong about the model look different.
STEP_AT = 4

# What the manifest calls the one tokenizer this model has.
TOKENIZER_SUFFIX = ".tokenizer.json"
TOKENIZER_CORPUS = "_corpus.tsv"


def export_test_cases(directory: str, prompt: str) -> TensorBag:
    """Write what the reference makes of one prompt, so the runtime can be checked against it.

    The reference is diffusers' own `Krea2Pipeline`, the way `sdxl_exporter.py`'s is diffusers'
    SDXL -- not a second implementation written here, which would only prove that two readings of
    the same paper agree.

    # Every input here is one the model would really be handed

    A part of a diffusion model is only worth comparing on the activations it was trained to see.
    `torch.randn` is not one of those: a denoiser reads a latent that is part noise and part
    picture, and an autoencoder reads a latent that came out of a denoiser -- hand either of them
    white noise and both implementations will produce *something*, agree or disagree by a number
    nobody can interpret, and say nothing about the pictures they will actually draw. The
    autoencoder is the clearest case: a tenth of the pixels of a decoded `randn` land outside
    `-1..=1`, where a real latent's barely do.

    So this runs the whole reference pipeline once, at 128 by 128 for eight steps, and takes its
    fixtures off that trajectory:

    | | what it is |
    |---|---|
    | `input_ids` | the template around the prompt, as the encoder reads it |
    | `hidden` | the twelve tapped layers, padding dropped |
    | `noise` | the latent the run starts from -- the one input that is noise because the model's is |
    | `sigmas` | the nine noise levels the reference's scheduler used |
    | `latent`, `timestep` | the latent entering step `STEP_AT`, and the sigma it enters at |
    | `velocity` | what the denoiser answers there |
    | `final_latent` | what the eight steps leave behind, which is what the decoder is handed |
    | `decoded` | the picture, from the autoencoder at full width |

    The prompt is padded here and not in the runtime, which is the point of the comparison: the
    reference runs the padded 512 and the tensors written are the rows that survive its mask, so a
    runtime that pads nothing has to produce exactly these.
    """
    from diffusers import Krea2Pipeline
    from diffusers.utils.torch_utils import randn_tensor

    ctx = Context("test_case")
    writer = TensorBag()

    def put(name: str, tensor: torch.Tensor) -> None:
        writer.write_tensor(ctx.with_subname(name), tensor, preserve_dtype=True)

    pipeline = Krea2Pipeline.from_pretrained(directory, torch_dtype=torch.bfloat16).to("cuda")

    with torch.no_grad():
        hidden, mask = pipeline.get_text_hidden_states([prompt], device="cuda")

        # The ids the runtime will feed, which are the reference's with the padding taken out.
        # Checked rather than assumed, because "the padding does not matter" is the claim this
        # whole comparison exists to test.
        template = pipeline.prompt_template_encode_prefix + prompt
        opening = pipeline.tokenizer(template, add_special_tokens=False).input_ids
        closing = pipeline.tokenizer(
            pipeline.prompt_template_encode_suffix, add_special_tokens=False).input_ids
        ids = opening[:512 + pipeline.prompt_template_encode_start_idx - len(closing)] + closing

        kept = mask[0].bool()
        if hidden[:, kept].shape[1] != len(ids) - pipeline.prompt_template_encode_start_idx:
            raise SystemExit(
                f"the reference kept {hidden[:, kept].shape[1]} tokens where the template makes "
                f"{len(ids) - pipeline.prompt_template_encode_start_idx}")

        writer.write_tensor(ctx.with_subname("input_ids"), torch.tensor([ids], dtype=torch.int64))
        put("hidden", hidden[:, kept].float())

        # The one thing that is allowed to be noise, because the model's own first input is:
        # generated here rather than inside the pipeline so that it can be written down and this
        # runtime can start its own walk from the same place.
        generator = torch.Generator().manual_seed(11)
        noise = randn_tensor(
            (1, TEST_CHANNELS, TEST_SIDE // 8, TEST_SIDE // 8),
            generator=generator,
            device=torch.device("cpu"),
            dtype=torch.float32)
        put("noise", noise)

        # The trajectory, run by the pipeline itself. The callback fires after each step, so
        # holding on to step `STEP_AT - 1`'s answer is holding the latent that step `STEP_AT`
        # reads.
        entering = {}

        def capture(pipe, index, timestep, kwargs):
            if index == STEP_AT - 1:
                entering["latent"] = kwargs["latents"].clone()
            return kwargs

        final = pipeline(
            prompt=prompt,
            height=TEST_SIDE,
            width=TEST_SIDE,
            num_inference_steps=TEST_STEPS,
            guidance_scale=0.0,
            latents=pipeline._pack_latents(
                noise.to("cuda", torch.bfloat16), 1, TEST_CHANNELS,
                TEST_SIDE // 8, TEST_SIDE // 8),
            output_type="latent",
            return_dict=False,
            callback_on_step_end=capture,
            callback_on_step_end_tensor_inputs=["latents"])[0]

        # The schedule it walked, which is nine numbers and the whole of what the sampler is.
        sigmas = pipeline.scheduler.sigmas.float().cpu()
        put("sigmas", sigmas)
        put("timestep", sigmas[STEP_AT : STEP_AT + 1].clone())

        unpack = lambda packed: pipeline._unpack_latents(packed, TEST_SIDE, TEST_SIDE)[:, :, 0]
        latent = entering["latent"]
        put("latent", unpack(latent).float())
        put("final_latent", unpack(final).float())

        # What the denoiser answers at that point of that trajectory. The prompt handed to it is
        # the padded one the reference carries, which is what makes the comparison a test of the
        # runtime's unpadded shortcut rather than of itself.
        positions = pipeline.prepare_position_ids(
            mask.shape[1], TEST_SIDE // 16, TEST_SIDE // 16, "cuda")
        velocity = pipeline.transformer(
            hidden_states=latent,
            encoder_hidden_states=hidden,
            timestep=sigmas[STEP_AT : STEP_AT + 1].to("cuda", torch.bfloat16),
            position_ids=positions,
            encoder_attention_mask=mask,
            return_dict=False)[0]
        put("velocity", unpack(velocity).float())

        # And the picture those eight steps stand for, straight out of the autoencoder -- at full
        # width, unlike everything above it. The autoencoder is half a gigabyte and the only part
        # of this model cheap enough to run in float32, so its reference is one: what the test
        # beside it then measures is this runtime's narrowing rather than the reference's.
        #
        # The decoder is called rather than `vae.decode`, which is the same arithmetic and one
        # thing more: `decode` clamps its answer to -1..=1. This runtime's decoder does not -- it
        # hands back what the convolutions produced and `to_rgb8` clamps where a picture becomes
        # bytes -- so a clamped fixture measures the clamp rather than the decoder. It is not a
        # large effect and it is not nothing: at this latent about one pixel in three hundred sits
        # against the ceiling, which is 1.4e-3 of difference that belongs to neither
        # implementation.
        vae = pipeline.vae.to(torch.float32)
        latents_mean = torch.tensor(vae.config.latents_mean).view(1, TEST_CHANNELS, 1, 1, 1)
        latents_std = torch.tensor(vae.config.latents_std).view(1, TEST_CHANNELS, 1, 1, 1)
        scaled = unpack(final).float().cpu()[:, :, None] * latents_std + latents_mean

        vae.clear_cache()
        moments = vae.post_quant_conv(scaled.to("cuda", torch.float32))
        decoded = vae.decoder(moments[:, :, 0:1], feat_cache=vae._feat_map, feat_idx=vae._conv_idx)
        vae.clear_cache()
        put("decoded", decoded[:, :, 0].float())

    return writer


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-model", required=True,
        help="the published release, as a directory: `transformer/`, `text_encoder/`, `vae/` and "
             "`tokenizer/` as `hf download krea/Krea-2-Turbo` leaves them.")
    parser.add_argument("-output", default="krea2.safetensors",
                        help="what to call the model. Names the weights and the manifest beside "
                             "them.")
    parser.add_argument(
        "-fp8", action="store_true",
        help="store the matrices as E4M3 with a scale per output channel: half the package and "
             "half the card, for about 2.6e-2 of relative error. See docs/fp8.md.")
    parser.add_argument(
        "-part-size", type=parse_size, default="4GB",
        help='split the package into parts of about this size, as in "4GB".')
    parser.add_argument(
        "-test_output", type=str, default=None,
        help="also write reference inputs and outputs here, for the runtime to be checked "
             "against. Runs the released model through diffusers, which wants a card with about "
             "forty gigabytes on it.")
    args = parser.parse_args()

    model = read_json(args.model, "transformer", "config.json")
    text_config = read_json(args.model, "text_encoder", "config.json")
    index = read_json(args.model, "model_index.json")
    vae_config = read_json(args.model, VAE_CONFIG)

    dit = read_weights(*transformer_files(args.model))
    text = read_weights(path.join(args.model, "text_encoder", "model.safetensors"))
    vae = read_weights(path.join(args.model, "vae", "diffusion_pytorch_model.safetensors"))

    shape = shape_of(model, text_config)
    shape["select_layers"] = index["text_encoder_select_layers"]
    shape["encoder_hidden_size"] = text_config["text_config"]["hidden_size"]
    check_shapes(shape, dit, text)
    count_stages(vae)

    print(f"{shape['num_blocks']} denoiser blocks, {shape['encoder_layers']} encoder layers, "
          f"{len(shape['select_layers'])} of them tapped")

    tokenizer = read_tokenizer(path.join(args.model, "tokenizer"))

    stem = stem_of(args.output)
    directory = path.dirname(path.abspath(args.output))

    tokenizer_file = stem + TOKENIZER_SUFFIX
    with open(path.join(directory, tokenizer_file), "wb") as fp:
        fp.write(tokenizer_json(tokenizer))
    print(f"wrote {tokenizer_file}")

    writer = open_weights(args.output, args.part_size)
    converter = Converter(writer, args.fp8)
    converter.export_dit(Context("krea2.dit"), dit, shape)
    converter.export_encoder(Context("krea2.text"), text, shape["encoder_layers"])
    converter.export_vae(Context("krea2.vae"), vae)

    config = generate_config(model, text_config, index, vae_config, shape, args.fp8)
    config["model"] = {"type": "krea2"}

    # What the card says to ask for, which is not what SDXL wants: eight steps, and a guidance of
    # one, which is this runtime's spelling of the reference's zero.
    suggested = {"steps": 8, "guidance": 1.0,
                 "sizes": [[1024, 1024], [1216, 832], [832, 1216]]}

    for name in writer.finish(config, suggested, {"tokenizer": tokenizer_file}):
        print(f"wrote {name}")
    print(f"{converter.written} tensors, {converter.widened} of them narrowed from bfloat16, "
          f"{converter.quantized} quantized")

    if args.test_output:
        # The weights are held twice over otherwise: once here and once inside diffusers.
        del dit, text, vae

        export_test_cases(args.model, TEST_PROMPT).save(args.test_output)
        print(f"wrote {args.test_output}")

        test_stem = stem_of(args.test_output)
        test_dir = path.dirname(path.abspath(args.test_output))
        corpus = tokenizer_corpus(encoder(tokenizer))
        name = test_stem + TOKENIZER_CORPUS
        with open(path.join(test_dir, name), "w", encoding="utf-8") as fp:
            fp.write(corpus)
        print(f"wrote {corpus.count(chr(10))} lines to {name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
