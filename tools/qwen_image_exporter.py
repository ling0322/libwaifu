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

"""Export Qwen-Image 2.1 to safetensors and the manifest that names them.

Qwen-Image 2.1 is a seven billion parameter single-stream DiT that reads the last hidden state of
a Qwen3-VL-8B text encoder and draws through a VAE of its own -- sixty-four latent channels,
sixteen times smaller than the picture on each side, four channels out. It is not Qwen-Image 1
and it is not the Qwen-Image VAE that Anima and Krea 2 share. `docs/qwen_image.md` is where the
whole of it is written down; read that first.

This reads the published diffusers layout as it stands, which is what `hf download` leaves:

    .venv/bin/python tools/qwen_image_exporter.py \\
        -model  ~/.cache/huggingface/hub/models--Qwen--Qwen-Image-2.1/snapshots/<rev> \\
        -output models/qwen-image-2.1.safetensors

`-fp8` writes the matrices as E4M3 with a scale per output channel: thirty gigabytes to sixteen.

`-controlnet` exports VideoX-Fun's ControlNet-Union for this model instead, as a package of its
own that the runtime reads beside the model's; the two have to agree on `-fp8`:

    .venv/bin/python tools/qwen_image_exporter.py \\
        -model  ~/.cache/huggingface/hub/models--Qwen--Qwen-Image-2.1/snapshots/<rev> \\
        -controlnet <Qwen-Image-2.1-Fun-Controlnet-Union.safetensors> \\
        -output models/qwen-image-2.1-controlnet.safetensors

`-test_output` also writes reference tensors off one real run of diffusers' `QwenImage21Pipeline`.
That needs a diffusers new enough to have it -- newer than `tools/requirements.txt` pins -- and a
CUDA torch; the run offloads layer by layer, so a sixteen gigabyte card is enough.

The released weights carry the Qwen Research License. Nothing here publishes anything; what it
writes is a package on your own disk, and where it goes afterwards is a decision this tool does
not make.
"""

from __future__ import annotations

import argparse
import glob
import json
from os import path

import torch
from safetensors import safe_open

from krea2_exporter import Converter as Krea2Converter
from krea2_exporter import count_blocks, fold_norm, read_json, read_weights
from model_exporter import Context, TensorBag, open_weights, parse_size, stem_of
from tokenizer_exporter import encoder, read_tokenizer, tokenizer_corpus, tokenizer_json

# The system turn the pipeline wraps every prompt in. Its tokens are read by the encoder and then
# dropped from what the denoiser is handed; the runtime counts them itself from this same text.
SYSTEM_PROMPT = "Comprehend and analyze the provided prompt."

# What the manifest calls the one tokenizer this model has, and the corpus beside the fixtures.
TOKENIZER_SUFFIX = ".tokenizer.json"
TOKENIZER_CORPUS = "_corpus.tsv"


class Converter(Krea2Converter):
    """Krea 2's converter, which already knows the Qwen3-VL text tower this model also reads,
    taught the denoiser and the autoencoder that are this model's own."""

    def _matrix(self, ctx: Context, tensor: torch.Tensor) -> None:
        """Krea 2's own `_matrix` now writes the tensor-scale FP8 format; this export has not
        moved to it yet, so `-fp8` here still writes one scale per row. See docs/fp8.md."""
        if not self._fp8:
            return self._write(ctx, tensor)

        self._writer.write_fp8_tensor(ctx, tensor.to(torch.float32))
        self._quantized += 1
        self._count += 1

    # ---- the denoiser -------------------------------------------------------------------

    def export_dit(self, ctx: Context, weights: dict, blocks: int) -> None:
        """The single-stream DiT. There is not one bias in it and not one norm with a weight in a
        block: the blocks' LayerNorms are affine-free and their modulation is the model's."""
        self._matrix(ctx.with_subname("img_in.weight"), weights["img_in.weight"])

        # The timestep path: a sinusoid, two linears with a swish between them.
        embedder = "time_text_embed.timestep_embedder"
        for name in ("linear_1", "linear_2"):
            self._matrix(ctx.with_subname(f"time_embed.{name}.weight"),
                         weights[f"{embedder}.{name}.weight"])

        # One modulation for every block: (scale, gate) for the attention and then for the feed
        # forward, 4 * D wide, computed once per timestep.
        self._matrix(ctx.with_subname("modulation.weight"), weights["modulation.1.weight"])

        # The prompt's way in. Its RMSNorm is zero centred; the two linears carry no biases.
        txt = ctx.with_subname("txt_in")
        self._write(txt.with_subname("norm.weight"), fold_norm(weights["txt_in.text_norm.weight"]))
        self._matrix(txt.with_subname("linear_1.weight"), weights["txt_in.in_layer.weight"])
        self._matrix(txt.with_subname("linear_2.weight"), weights["txt_in.out_layer.weight"])

        for index in range(blocks):
            self._export_block(ctx.with_subname(f"block{index}"), weights,
                               f"transformer_blocks.{index}.")

        final = ctx.with_subname("final")
        self._matrix(final.with_subname("linear.weight"), weights["norm_out.linear.weight"])
        self._matrix(final.with_subname("proj.weight"), weights["proj_out.weight"])

    def _export_block(self, ctx: Context, weights: dict, prefix: str) -> None:
        def at(name: str) -> torch.Tensor:
            return weights[prefix + name]

        self._fuse(ctx.with_subname("attn.qkv_proj.weight"),
                   at("attn.to_q.weight"), at("attn.to_k.weight"), at("attn.to_v.weight"))
        # diffusers' own RMSNorm, which stores the multiplier: not folded.
        self._write(ctx.with_subname("attn.q_norm.weight"), at("attn.norm_q.weight"))
        self._write(ctx.with_subname("attn.k_norm.weight"), at("attn.norm_k.weight"))
        self._matrix(ctx.with_subname("attn.out_proj.weight"), at("attn.to_out.0.weight"))

        # `out(silu(gate_layer(x)) * proj(x))`: the gate first, because that is the half flint's
        # swiglu puts the swish on.
        self._fuse(ctx.with_subname("ff.gate_up_proj.weight"),
                   at("img_mlp.gate_layer.weight"), at("img_mlp.proj.weight"))
        self._matrix(ctx.with_subname("ff.down.weight"), at("img_mlp.out.weight"))

    # ---- the VAE ------------------------------------------------------------------------

    def export_vae(self, ctx: Context, weights: dict) -> None:
        """Both halves, under diffusers' own names.

        Every convolution in this autoencoder is already two dimensional: it is the image
        specialization of Wan 2.2's, and its causal 3-D convolutions subclass `nn.Conv2d`. What is
        left out is the `time_conv` of each temporal resample, which a single frame never reaches
        -- the first chunk of a decode marks the cache "Rep" and skips it, and the first chunk of
        an encode stores itself and skips it. The norm scales are `F.normalize(x) * sqrt(C) *
        gamma`, which is an RMSNorm over the channels; the `gamma` is written flat and as
        `.weight`.
        """
        for name, tensor in sorted(weights.items()):
            if ".time_conv." in name:
                continue
            if name.endswith(".gamma"):
                name, tensor = name[:-len(".gamma")] + ".weight", tensor.reshape(-1)
            self._write(ctx.with_subname(name), tensor)

    # ---- the ControlNet -----------------------------------------------------------------

    def export_control(self, ctx: Context, weights: dict, blocks: int) -> None:
        """VideoX-Fun's ControlNet-Union: `blocks` more of the denoiser's own blocks, a projection
        into them and one out of each.

        The blocks are written the way the denoiser's are, so the runtime reads both with one
        function. `control_img_in` reads 129 channels, which is not a multiple of the sixteen an
        FP8 multiply reads at a time, so it is written at full width in every package; it is half
        a million numbers.
        """
        self._write(ctx.with_subname("img_in.weight"), weights["control_img_in.weight"])
        self._write(ctx.with_subname("img_in.bias"), weights["control_img_in.bias"])

        # Only the first block of the chain takes the denoiser's input in.
        before = ctx.with_subname("block0.before_proj")
        self._matrix(before.with_subname("weight"), weights["control_blocks.0.before_proj.weight"])
        self._write(before.with_subname("bias"), weights["control_blocks.0.before_proj.bias"])

        for index in range(blocks):
            prefix = f"control_blocks.{index}."
            block = ctx.with_subname(f"block{index}")
            self._export_block(block, weights, prefix)
            after = block.with_subname("after_proj")
            self._matrix(after.with_subname("weight"), weights[prefix + "after_proj.weight"])
            self._write(after.with_subname("bias"), weights[prefix + "after_proj.bias"])


def transformer_files(directory: str) -> list:
    index = path.join(directory, "transformer", "diffusion_pytorch_model.safetensors.index.json")
    with open(index, encoding="utf-8") as fp:
        names = sorted(set(json.load(fp)["weight_map"].values()))
    return [path.join(directory, "transformer", name) for name in names]


def encoder_weights(directory: str) -> dict:
    """The language half of the encoder, renamed the way Krea 2's converter reads it.

    Qwen3-VL-8B publishes it under `model.language_model.`; Krea 2's Qwen3-VL-4B under
    `language_model.`. The vision tower and the `lm_head` are never read and never loaded.
    """
    held = {}
    for file in sorted(glob.glob(path.join(directory, "text_encoder", "*.safetensors"))):
        with safe_open(file, framework="pt") as handle:
            for key in handle.keys():
                if key.startswith("model.language_model."):
                    held[key[len("model."):]] = handle.get_tensor(key)
    return held


def check_shapes(model: dict, text: dict, dit: dict, words: dict) -> None:
    """What the configurations claim, against what the tensors are."""
    hidden = model["num_attention_heads"] * model["attention_head_dim"]
    encoder = text["text_config"]
    checks = [
        ("img_in.weight", tuple(dit["img_in.weight"].shape),
         (hidden, model["in_channels"] * model["patch_size"] ** 2)),
        ("modulation.1.weight", tuple(dit["modulation.1.weight"].shape), (4 * hidden, hidden)),
        ("txt_in.in_layer.weight", tuple(dit["txt_in.in_layer.weight"].shape),
         (hidden, model["context_in_dim"])),
        ("img_mlp.proj.weight", tuple(dit["transformer_blocks.0.img_mlp.proj.weight"].shape),
         (hidden * model["mlp_ratio"], hidden)),
        ("embed_tokens.weight", tuple(words["language_model.embed_tokens.weight"].shape),
         (encoder["vocab_size"], encoder["hidden_size"])),
    ]
    for name, found, wanted in checks:
        if found != wanted:
            raise SystemExit(f"{name} is {found} where the configuration says {wanted}")

    if model["patch_size"] != 1:
        raise SystemExit(f"a patch of {model['patch_size']}: this runtime reads latents unpatched")
    if encoder["hidden_size"] != model["context_in_dim"]:
        raise SystemExit("the encoder is not as wide as the denoiser's prompt projection reads")
    if not model.get("causal_condition", False):
        raise SystemExit("this runtime modulates the prompt from t = 0, which needs "
                         "causal_condition; this release does not have it")

    counted = count_blocks(dit, r"transformer_blocks\.(\d+)\.")
    if counted != model["num_layers"]:
        raise SystemExit(f"{counted} blocks in the weights, {model['num_layers']} in the config")
    counted = count_blocks(words, r"language_model\.layers\.(\d+)\.")
    if counted != encoder["num_hidden_layers"]:
        raise SystemExit(f"{counted} encoder layers, {encoder['num_hidden_layers']} in the config")


def generate_config(model: dict, text: dict, vae: dict, scheduler: dict, fp8: bool) -> dict:
    encoder = text["text_config"]
    rope_theta = encoder.get("rope_theta") or encoder["rope_parameters"]["rope_theta"]

    if scheduler.get("time_shift_type") != "exponential" or not scheduler["use_dynamic_shifting"]:
        raise SystemExit("this scheduler does not bend its schedule the way the runtime does")

    config = {"qwen_image": {
        # the denoiser
        "num_blocks": str(model["num_layers"]),
        "hidden_size": str(model["num_attention_heads"] * model["attention_head_dim"]),
        "num_heads": str(model["num_attention_heads"]),
        "head_dim": str(model["attention_head_dim"]),
        "mlp_size": str(model["attention_head_dim"] * model["num_attention_heads"]
                        * model["mlp_ratio"]),
        "latent_channels": str(model["in_channels"]),
        "timestep_embed_dim": "256",
        # `QwenImage21Rope(theta=10000, ...)`: a constant of the class, in no config file.
        "rope_theta": "10000",
        "rope_axes": [int(axis) for axis in model["axes_dims_rope"]],
        "norm_eps": f"{model['eps']:g}",
        # the text encoder, of which only the last layer's output -- before the final norm -- is
        # read. `encoder_select_layers` says so the way Krea 2's package says which twelve.
        "encoder_num_layers": str(encoder["num_hidden_layers"]),
        "encoder_hidden_size": str(encoder["hidden_size"]),
        "encoder_vocab_size": str(encoder["vocab_size"]),
        "encoder_num_heads": str(encoder["num_attention_heads"]),
        "encoder_num_kv_heads": str(encoder["num_key_value_heads"]),
        "encoder_head_dim": str(encoder["head_dim"]),
        "encoder_mlp_size": str(encoder["intermediate_size"]),
        "encoder_rope_theta": f"{rope_theta:g}",
        "encoder_rms_norm_eps": f"{encoder['rms_norm_eps']:g}",
        "encoder_select_layers": [int(encoder["num_hidden_layers"])],
        # the VAE
        "vae_scale": str(vae["scale_factor_spatial"]),
        "image_channels": str(vae["out_channels"]),
        # Which of the decoder's upsampling stages were temporal ones, as 1 or 0: the encoder's
        # list backwards. No weight in the package says so -- the `time_conv` that would is left
        # out -- and the shortcut around each stage reads different channels either way.
        "vae_temporal_upsample": [int(flag) for flag in reversed(vae["temperal_downsample"])],
        "latents_mean": [float(f"{value:g}") for value in vae["latents_mean"]],
        "latents_std": [float(f"{value:g}") for value in vae["latents_std"]],
        # rectified flow, with a shift that depends on how many tokens are being drawn and a
        # schedule stretched to finish at `shift_terminal` rather than at zero.
        "prediction": "flow",
        "sampler_base_shift": f"{scheduler['base_shift']:g}",
        "sampler_max_shift": f"{scheduler['max_shift']:g}",
        "sampler_base_seq_len": str(scheduler["base_image_seq_len"]),
        "sampler_max_seq_len": str(scheduler["max_image_seq_len"]),
        "sampler_shift_terminal": f"{scheduler['shift_terminal']:g}",
        "system_prompt": SYSTEM_PROMPT,
    }}

    if fp8:
        config["qwen_image"]["weight_format"] = "fp8"

    return config


def generate_control_config(model: dict, weights: dict, fp8: bool) -> tuple:
    """The control package's manifest, and how many blocks its chain has.

    Nothing in the release says which of the denoiser's blocks a skip is added after: the
    checkpoint is one safetensors file and VideoX-Fun's config for it is not published. What
    `QwenImage21ControlTransformer2DModel` does without one is every second block from the first,
    which is also what the model card writes down -- `[0, 2, 4, .., 30]` -- and what sixteen blocks
    of chain come to over thirty-two.
    """
    blocks = count_blocks(weights, r"control_blocks\.(\d+)\.")
    layers = list(range(0, model["num_layers"], 2))
    if len(layers) != blocks:
        raise SystemExit(f"{blocks} control blocks, and every second of {model['num_layers']} "
                         f"denoiser blocks is {len(layers)}")

    in_channels = weights["control_img_in.weight"].shape[1]
    if in_channels != 2 * model["in_channels"] + 1:
        raise SystemExit(f"the control projection reads {in_channels} channels, not two latents "
                         f"and a mask")
    if "control_blocks.1.before_proj.weight" in weights:
        raise SystemExit("more than the first control block takes the denoiser's input in")

    section = {
        "control_layers": layers,
        "control_in_channels": str(in_channels),
    }
    if fp8:
        section["weight_format"] = "fp8"
    return {"model": {"type": "qwen_image_control"}, "qwen_image_control": section}, blocks


# What the reference outputs are computed for.
TEST_PROMPT = "a red fox sitting in fresh snow, golden hour, photorealistic"

# 256 by 256 is a 16 by 16 latent: 256 image tokens, 64 of the encoder's image slots. Ten steps of
# the forty the release wants, which is still a real trajectory and a quarter of the offloading.
TEST_SIDE = 256
TEST_STEPS = 10
STEP_AT = 5
TEST_CHANNELS = 64
VAE_SCALE = 16


def export_test_cases(directory: str, prompt: str) -> TensorBag:
    """Write what the reference makes of one prompt, taken off one real run of it.

    | | what it is |
    |---|---|
    | `input_ids` | the whole template around the prompt, system turn included |
    | `hidden` | what the denoiser reads: the last layer before the norm, system turn dropped |
    | `hidden_fp32` | the same, from the encoder at float32, which is what the runtime is held to |
    | `noise` | the latent the run starts from |
    | `sigmas` | the eleven noise levels the reference's scheduler walked |
    | `latent`, `timestep` | the latent entering step `STEP_AT`, and the timestep the model saw |
    | `velocity` | what the denoiser answers there |
    | `final_latent` | what the ten steps leave behind |
    | `decoded` | the picture, four channels, from the decoder at float32 and unclamped |

    The run is made without the reference's prefix cache, which is how this runtime runs: the
    prompt is recomputed at every step. diffusers says the two differ by rounding and nothing else.
    """
    from diffusers import AutoencoderKLQwenImage21, QwenImage21Pipeline

    ctx = Context("test_case")
    writer = TensorBag()

    def put(name: str, tensor: torch.Tensor) -> None:
        writer.write_tensor(ctx.with_subname(name), tensor, preserve_dtype=True)

    pipeline = QwenImage21Pipeline.from_pretrained(directory, torch_dtype=torch.bfloat16)
    pipeline.enable_sequential_cpu_offload()
    side = TEST_SIDE // VAE_SCALE

    with torch.no_grad():
        template = pipeline.prompt_template_t2i.format(prompt)
        ids = pipeline.processor(text=[template], return_tensors="pt").input_ids
        writer.write_tensor(ctx.with_subname("input_ids"), ids.to(torch.int64))

        embeds, mask, image_pad_mask = pipeline.encode_prompt(prompt, device="cuda")
        if mask is not None:
            raise SystemExit("one prompt came back padded")
        if embeds.shape[1] != ids.shape[1] - pipeline._drop_idx:
            raise SystemExit(f"the reference kept {embeds.shape[1]} states of {ids.shape[1]} ids "
                             f"and dropped {pipeline._drop_idx}")
        put("hidden", embeds.float())
        put("drop", torch.tensor([pipeline._drop_idx], dtype=torch.int64))

        generator = torch.Generator().manual_seed(11)
        noise = torch.randn((1, TEST_CHANNELS, side, side), generator=generator)
        put("noise", noise)

        def pack(latent):
            return latent.reshape(1, TEST_CHANNELS, side * side).transpose(1, 2)

        def unpack(packed):
            return packed.transpose(1, 2).reshape(1, TEST_CHANNELS, side, side)

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
            latents=pack(noise).to("cuda", torch.bfloat16),
            output_type="latent",
            return_dict=False,
            use_kv_cache=False,
            callback_on_step_end=capture,
            callback_on_step_end_tensor_inputs=["latents"])[0]

        sigmas = pipeline.scheduler.sigmas.float().cpu()
        put("sigmas", sigmas)

        # The timestep the model is handed is not the sigma: the pipeline casts `sigma * 1000` to
        # the latents' bfloat16 and divides by a thousand in bfloat16 again. Written as the value
        # it became, because that is what the velocity below was computed from.
        timestep = pipeline.scheduler.timesteps[STEP_AT].expand(1).to(torch.bfloat16) / 1000
        put("timestep", timestep.float().cpu())

        latent = entering["latent"]
        put("latent", unpack(latent).float())
        put("final_latent", unpack(final).float())

        slots = torch.ones(1, side * side // 4, dtype=torch.bool, device=image_pad_mask.device)
        velocity = pipeline.transformer(
            hidden_states=latent,
            encoder_hidden_states=embeds,
            timestep=timestep.to("cuda"),
            img_shapes=[[(1, side, side)]],
            img_mask=torch.cat([image_pad_mask, slots], dim=1),
            return_dict=False)[0]
        put("velocity", unpack(velocity[:, -side * side:]).float())

        # The encoder's states again, at float32 on the processor. The bfloat16 ones above are
        # the less exact of the two answers: the last layer before the norm carries values in the
        # hundreds, and bfloat16 puts the reference about 7e-2 from float32, which a float16
        # runtime is expected to beat rather than match.
        del pipeline
        from transformers import Qwen3VLForConditionalGeneration
        encoder = Qwen3VLForConditionalGeneration.from_pretrained(
            directory, subfolder="text_encoder", torch_dtype=torch.float32)
        text = encoder.model.language_model
        handle = text.norm.register_forward_hook(lambda module, args, output: args[0])
        states = text(input_ids=ids, output_hidden_states=True).hidden_states[-1]
        handle.remove()
        put("hidden_fp32", states[:, ids.shape[1] - embeds.shape[1]:].float())
        del encoder, text

        # The decoder alone, at float32, and without `decode`'s clamp to -1..=1: the runtime's
        # decoder hands back what the convolutions produced and clamps where a picture becomes
        # bytes, so a clamped fixture would measure the clamp.
        vae = AutoencoderKLQwenImage21.from_pretrained(
            directory, subfolder="vae", torch_dtype=torch.float32).to("cuda")
        mean = torch.tensor(vae.config.latents_mean).view(1, TEST_CHANNELS, 1, 1, 1)
        std = torch.tensor(vae.config.latents_std).view(1, TEST_CHANNELS, 1, 1, 1)
        scaled = unpack(final).float().cpu()[:, :, None] * std + mean

        vae.clear_cache()
        x = vae.post_quant_conv(scaled.to("cuda"))
        decoded = vae.decoder(x, feat_cache=vae._feat_map, feat_idx=vae._conv_idx,
                              first_chunk=True)
        vae.clear_cache()
        put("decoded", decoded[:, :, 0].float())

    return writer


def export_control_test_cases(directory: str, controlnet: str, control_image: str,
                              prompt: str) -> TensorBag:
    """Write what VideoX-Fun makes of one prompt steered by one control picture.

    | | what it is |
    |---|---|
    | `control_image` | the picture as the pipeline hands it to the encoder: three channels, `-1..=1` |
    | `control_latent` | what the encoder makes of it with an opaque alpha, at float32, normalized |
    | `control_context` | the 129 channels the chain reads: that latent, a zero mask, a zero latent |
    | `hidden` | the prompt's states, as the control pipeline encoded them |
    | `noise`, `sigmas` | where the run starts, and the noise levels it walked |
    | `latent`, `timestep` | the latent entering step `STEP_AT` of the controlled run, and its time |
    | `velocity` | the controlled answer there, the skips at scale one |
    | `velocity_half` | the same at scale one half |
    | `velocity_plain` | VideoX-Fun's own copy of the denoiser there, with no control at all |
    | `final_latent` | what the ten controlled steps leave behind |

    The upstream is VideoX-Fun, which trained the checkpoint and carries its own copies of the
    denoiser and the autoencoder rather than diffusers'. `velocity_plain` is what says its
    denoiser is the one the runtime was already held to; the encoder is checked against diffusers'
    here, at float32, before anything is written.

    Pure control, the recipe the model card leads with: no inpainting source and no mask, which
    the pipeline turns into a mask of zeros ("regenerate everywhere") and a latent of zeros.
    """
    from diffusers import AutoencoderKLQwenImage21 as DiffusersVae
    from diffusers import FlowMatchEulerDiscreteScheduler
    from PIL import Image
    from videox_fun.models import (AutoencoderKLQwenImage21, Qwen3VLForConditionalGeneration,
                                   Qwen3VLProcessor, QwenImage21ControlTransformer2DModel)
    from videox_fun.pipeline import QwenImage21ControlPipeline

    ctx = Context("test_case")
    writer = TensorBag()

    def put(name: str, tensor: torch.Tensor) -> None:
        writer.write_tensor(ctx.with_subname(name), tensor, preserve_dtype=True)

    import gc

    # The prompt first, and the encoder gone before the denoiser arrives: the offloaded run keeps
    # every model it was handed in host memory, and the encoder alone is sixteen gigabytes of it.
    processor = Qwen3VLProcessor.from_pretrained(directory, subfolder="processor")
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(directory, subfolder="scheduler")
    reader = QwenImage21ControlPipeline(
        vae=None,
        text_encoder=Qwen3VLForConditionalGeneration.from_pretrained(
            directory, subfolder="text_encoder", torch_dtype=torch.bfloat16),
        processor=processor,
        transformer=None,
        scheduler=scheduler,
    )
    with torch.no_grad():
        embeds, mask, image_pad_mask = reader.encode_prompt(prompt, device="cpu")
    if mask is not None:
        raise SystemExit("one prompt came back padded")
    del reader
    gc.collect()

    model = read_json(directory, "transformer", "config.json")
    weights = read_weights(controlnet)
    config, _ = generate_control_config(model, weights, fp8=False)
    section = config["qwen_image_control"]

    transformer = QwenImage21ControlTransformer2DModel.from_pretrained(
        directory, subfolder="transformer", low_cpu_mem_usage=True, torch_dtype=torch.bfloat16,
        transformer_additional_kwargs={"control_layers": section["control_layers"],
                                       "control_in_dim": int(section["control_in_channels"])},
    ).to(torch.bfloat16)
    missing, unexpected = transformer.load_state_dict(weights, strict=False)
    if unexpected or any(name.startswith("control") for name in missing):
        raise SystemExit(f"the checkpoint does not fit VideoX-Fun's control transformer: "
                         f"{len(unexpected)} unexpected, missing {[n for n in missing][:4]}")
    del weights
    gc.collect()

    pipeline = QwenImage21ControlPipeline(
        vae=AutoencoderKLQwenImage21.from_pretrained(directory, subfolder="vae").to(torch.bfloat16),
        text_encoder=None,
        processor=processor,
        transformer=transformer,
        scheduler=scheduler,
    )
    pipeline.enable_sequential_cpu_offload()
    side = TEST_SIDE // VAE_SCALE

    def pack(latent):
        channels = latent.shape[1]
        return latent.reshape(1, channels, side * side).transpose(1, 2)

    def unpack(packed):
        return packed.transpose(1, 2).reshape(1, packed.shape[2], side, side)

    with torch.no_grad():
        picture = Image.open(control_image).convert("RGB")
        control = pipeline.image_processor.preprocess(picture, height=TEST_SIDE, width=TEST_SIDE)
        put("control_image", control.float())
        put("hidden", embeds.float())
        embeds = embeds.to("cuda")
        image_pad_mask = image_pad_mask.to("cuda")

        generator = torch.Generator().manual_seed(11)
        noise = torch.randn((1, TEST_CHANNELS, side, side), generator=generator)
        put("noise", noise)

        entering = {}

        def capture(pipe, index, timestep, kwargs):
            if index == STEP_AT - 1:
                entering["latent"] = kwargs["latents"].clone()
            return kwargs

        final = pipeline(
            prompt_embeds=embeds,
            control_image=picture,
            control_context_scale=1.0,
            height=TEST_SIDE,
            width=TEST_SIDE,
            num_inference_steps=TEST_STEPS,
            latents=pack(noise).to("cuda", torch.bfloat16),
            output_type="latent",
            return_dict=False,
            use_kv_cache=False,
            callback_on_step_end=capture,
            callback_on_step_end_tensor_inputs=["latents"])[0]
        put("sigmas", pipeline.scheduler.sigmas.float().cpu())
        put("final_latent", unpack(final).float())

        timestep = pipeline.scheduler.timesteps[STEP_AT].expand(1).to(torch.bfloat16) / 1000
        put("timestep", timestep.float().cpu())
        latent = entering["latent"]
        put("latent", unpack(latent).float())

        # The encoder at float32, VideoX-Fun's and diffusers', which have to agree before either
        # is worth writing down.
        rgba = torch.cat([control, torch.ones_like(control[:, :1])], dim=1)[:, :, None]
        encoded = []
        for kind in (AutoencoderKLQwenImage21, DiffusersVae):
            vae = kind.from_pretrained(directory, subfolder="vae", torch_dtype=torch.float32)
            vae = vae.to("cuda")
            moments = vae.encode(rgba.to("cuda")).latent_dist.mode().cpu()
            mean = torch.tensor(vae.config.latents_mean).view(1, TEST_CHANNELS, 1, 1, 1)
            std = torch.tensor(vae.config.latents_std).view(1, TEST_CHANNELS, 1, 1, 1)
            encoded.append(((moments - mean) / std)[:, :, 0])
            del vae
            torch.cuda.empty_cache()
        apart = ((encoded[0] - encoded[1]).norm() / encoded[1].norm()).item()
        print(f"VideoX-Fun's encoder is {apart:.2e} from diffusers'")
        if apart > 1e-4:
            raise SystemExit("VideoX-Fun's autoencoder is not diffusers'")
        control_latent = encoded[0]
        put("control_latent", control_latent)

        control_context = torch.cat([
            control_latent,
            torch.zeros(1, 1, side, side),
            torch.zeros(1, TEST_CHANNELS, side, side),
        ], dim=1)
        put("control_context", control_context)

        slots = torch.ones(1, side * side // 4, dtype=torch.bool, device=image_pad_mask.device)
        arguments = dict(
            hidden_states=latent,
            encoder_hidden_states=embeds,
            timestep=timestep.to("cuda"),
            img_shapes=[[(1, side, side)]],
            img_mask=torch.cat([image_pad_mask, slots], dim=1),
            return_dict=False)
        packed = pack(control_context).to("cuda", torch.bfloat16)

        for name, scale in (("velocity", 1.0), ("velocity_half", 0.5)):
            velocity = transformer(**arguments, control_context=packed,
                                   control_context_scale=scale)[0]
            put(name, unpack(velocity[:, -side * side:]).float())
        velocity = transformer(**arguments)[0]
        put("velocity_plain", unpack(velocity[:, -side * side:]).float())

    return writer


def export_controlnet(args) -> int:
    """The ControlNet as a package of its own, which the runtime reads beside the model's."""
    model = read_json(args.model, "transformer", "config.json")

    if not args.test_only:
        weights = read_weights(args.controlnet)
        config, blocks = generate_control_config(model, weights, args.fp8)
        writer = open_weights(args.output, args.part_size)
        converter = Converter(writer, args.fp8)
        converter.export_control(Context("qwen_image.dit.control"), weights, blocks)
        for name in writer.finish(config):
            print(f"wrote {name}")
        print(f"{converter.written} tensors, {converter.widened} of them narrowed from bfloat16, "
              f"{converter.quantized} quantized")

    if args.test_output:
        if not args.control_image:
            raise SystemExit("the ControlNet's reference needs a -control_image")
        export_control_test_cases(args.model, args.controlnet, args.control_image,
                                  TEST_PROMPT).save(args.test_output)
        print(f"wrote {args.test_output}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-model", required=True,
        help="the published release, as a directory: `transformer/`, `text_encoder/`, `vae/`, "
             "`processor/` and `scheduler/` as `hf download Qwen/Qwen-Image-2.1` leaves them.")
    parser.add_argument("-output", default="qwen-image-2.1.safetensors",
                        help="what to call the model. Names the weights and the manifest beside "
                             "them.")
    parser.add_argument("-fp8", action="store_true",
                        help="store the matrices as E4M3 with a scale per output channel.")
    parser.add_argument("-part-size", type=parse_size, default="4GB",
                        help='split the package into parts of about this size, as in "4GB".')
    parser.add_argument("-test_output", type=str, default=None,
                        help="also write reference inputs and outputs here. Needs a diffusers "
                             "with QwenImage21Pipeline and a CUDA card.")
    parser.add_argument("-test_only", action="store_true",
                        help="write only the reference tensors, not the package.")
    parser.add_argument("-controlnet", type=str, default=None,
                        help="export this ControlNet-Union checkpoint -- VideoX-Fun's "
                             "`Qwen-Image-2.1-Fun-Controlnet-Union.safetensors` -- as a package of "
                             "its own instead of the model. `-model` is still the release it "
                             "steers. With `-test_output`, the reference is VideoX-Fun's.")
    parser.add_argument("-control_image", type=str, default=None,
                        help="the control picture the ControlNet's reference is computed for: a "
                             "real one, such as VideoX-Fun's `asset/pose.jpg`.")
    args = parser.parse_args()

    if args.controlnet:
        return export_controlnet(args)

    model = read_json(args.model, "transformer", "config.json")
    text_config = read_json(args.model, "text_encoder", "config.json")
    vae_config = read_json(args.model, "vae", "config.json")
    scheduler = read_json(args.model, "scheduler", "scheduler_config.json")

    if not args.test_only:
        dit = read_weights(*transformer_files(args.model))
        words = encoder_weights(args.model)
        vae = read_weights(path.join(args.model, "vae", "diffusion_pytorch_model.safetensors"))
        check_shapes(model, text_config, dit, words)

        tokenizer = read_tokenizer(path.join(args.model, "processor"))
        stem = stem_of(args.output)
        directory = path.dirname(path.abspath(args.output))
        tokenizer_file = stem + TOKENIZER_SUFFIX
        with open(path.join(directory, tokenizer_file), "wb") as fp:
            fp.write(tokenizer_json(tokenizer))
        print(f"wrote {tokenizer_file}")

        writer = open_weights(args.output, args.part_size)
        converter = Converter(writer, args.fp8)
        converter.export_dit(Context("qwen_image.dit"), dit, model["num_layers"])
        del dit
        converter.export_encoder(Context("qwen_image.text"), words,
                                 text_config["text_config"]["num_hidden_layers"])
        del words
        converter.export_vae(Context("qwen_image.vae"), vae)

        config = generate_config(model, text_config, vae_config, scheduler, args.fp8)
        config["model"] = {"type": "qwen_image"}

        # What the model card asks for: forty steps and no guidance -- `true_cfg_scale` defaults
        # to one and the pipeline calls the model "meant to be sampled without guidance". The
        # sizes are the card's own list at a quarter of the area, since the card's are two to
        # three megapixels and the runtime draws at one by default.
        suggested = {"steps": 40, "guidance": 1.0,
                     "sizes": [[1024, 1024], [1184, 896], [896, 1184], [1248, 832],
                               [832, 1248], [1376, 768], [768, 1376]]}

        for name in writer.finish(config, suggested, {"tokenizer": tokenizer_file}):
            print(f"wrote {name}")
        print(f"{converter.written} tensors, {converter.widened} of them narrowed from bfloat16, "
              f"{converter.quantized} quantized")

    if args.test_output:
        export_test_cases(args.model, TEST_PROMPT).save(args.test_output)
        print(f"wrote {args.test_output}")

        tokenizer = read_tokenizer(path.join(args.model, "processor"))
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
