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

"""Export See-through's two models to one libwaifu package.

See-through (https://github.com/shitagaki-lab/see-through) takes an anime illustration apart into
the layers it was drawn in -- hair, face, eyes, clothing -- and orders them by depth. It is two
diffusion models, both of them a U-Net that attends across the layers it is drawing as well as
across the picture:

  layers   SDXL (an Animagine XL 4.0 fine tune), eight channels in: the noisy latent and the
           picture's. Twenty-four layers in two passes, and a decoder that gives each its alpha.
  depth    Marigold on Stable Diffusion 2, twelve channels in: the picture, one layer, and the
           noisy depth. One depth map per layer, which is what orders them.

Neither reads a prompt a user writes. The layers model is conditioned on the name of each layer
-- "front hair", "irides" -- and the depth model on the empty prompt, so what both text encoders
would have made of those is computed here, once, and the package carries the result instead of
the encoders. The layers model's group embeddings, which tell it which pass it is on, are folded
into those same tensors for the same reason.

    PYTHONPATH=~/.cache/libwaifu/src/see-through/common \\
        .venv-see-through/bin/python tools/see_through_exporter.py \\
        -layers layerdifforg/seethroughv0.0.2_layerdiff3d \\
        -depth layerdifforg/seethroughv0.0.1_marigold \\
        -output models/see-through.safetensors -part-size 2GB
"""

from __future__ import annotations

import argparse
from os import path

import torch

from model_exporter import Context, Quant, open_weights
from sdxl_exporter import SdxlExporter

# The layers of the two passes, in the order the model was trained to draw them. The second pass
# is drawn from the head the first one found, cropped and enlarged. These are See-through's v3
# tags (`apply_layerdiff`), which is the version `seethroughv0.0.2_layerdiff3d` declares.
BODY_TAGS = [
    "front hair", "back hair", "head", "neck", "neckwear", "topwear", "handwear",
    "bottomwear", "legwear", "footwear", "tail", "wings", "objects"]
HEAD_TAGS = [
    "headwear", "face", "irides", "eyebrow", "eyewhite", "eyelash", "eyewear", "ears",
    "earwear", "nose", "mouth"]

# The slots the depth model is asked about, one depth map each, and the full page after them.
# Two are composites the layers model never draws on its own: `hair` is the two hairs and `eyes`
# the four parts of an eye, laid over each other in the order these list them.
DEPTH_TAGS = [
    "hair", "headwear", "face", "eyes", "eyewear", "ears", "earwear", "nose", "mouth",
    "neck", "neckwear", "topwear", "handwear", "bottomwear", "legwear", "footwear",
    "tail", "wings", "objects"]
DEPTH_COMPOSITES = {
    "hair": ["back hair", "front hair"],
    "eyes": ["eyewhite", "irides", "eyelash", "eyebrow"],
}

# What upstream runs at: the side of the square the picture is padded and scaled to, and the
# steps each sampler takes. `inference_psd.py`'s defaults.
LAYERS_RESOLUTION = 1280
LAYERS_STEPS = 30
DEPTH_RESOLUTION = 768
DEPTH_STEPS = 4


class SeeThroughExporter(SdxlExporter):
    """SDXL's exporter, with what See-through adds to its U-Nets and the alpha decoder."""

    def _write(self, ctx: Context, tensor: torch.Tensor) -> None:
        # The U-Nets are published in bfloat16, which the writer does not take; widened here and
        # narrowed to float16 by it. bfloat16 reaches further than float16 does, so a weight past
        # float16's range is refused rather than written as an infinity.
        if tensor.dtype == torch.bfloat16:
            tensor = tensor.float()
            if not self._float32 and tensor.abs().max() > torch.finfo(torch.float16).max:
                raise ValueError(f"{ctx.name} does not fit in float16")
        super()._write(ctx, tensor)

    def _export_frame_block(self, ctx: Context, block) -> None:
        """Attention across the layers: a gated feed forward, self attention, another feed forward.

        `CrossFrameTransformerBlock` with no cross attention, which is how See-through builds every
        one of them.
        """
        if block.attn2 is not None:
            raise ValueError("a frame block with cross attention, which none of these have")
        self.export_layer_norm(ctx.with_subname("norm_in"), block.norm_in)
        self._export_geglu(ctx.with_subname("ff_in.gate"), block.ff_in.net[0])
        self.export_linear(ctx.with_subname("ff_in.out_proj"), block.ff_in.net[2])
        self.export_layer_norm(ctx.with_subname("norm1"), block.norm1)
        self._export_attention(ctx.with_subname("attn1"), block.attn1)
        self.export_layer_norm(ctx.with_subname("norm3"), block.norm3)
        self._export_geglu(ctx.with_subname("ff.gate"), block.ff.net[0])
        self.export_linear(ctx.with_subname("ff.out_proj"), block.ff.net[2])

    def _export_transformer(self, ctx: Context, module) -> None:
        super()._export_transformer(ctx, module)
        for index, block in enumerate(getattr(module, "temporal_transformer_blocks", [])):
            self._export_frame_block(ctx.with_subname(f"frame{index}"), block)

    def _export_frame_unet(self, ctx: Context, unet) -> None:
        """A U-Net as SdxlExporter writes one, with or without SDXL's added embedding."""
        self._export_conv2d(ctx.with_subname("conv_in"), unet.conv_in)
        self.export_linear(ctx.with_subname("time_embd.linear1"), unet.time_embedding.linear_1)
        self.export_linear(ctx.with_subname("time_embd.linear2"), unet.time_embedding.linear_2)
        if getattr(unet, "add_embedding", None) is not None:
            self.export_linear(ctx.with_subname("add_embd.linear1"), unet.add_embedding.linear_1)
            self.export_linear(ctx.with_subname("add_embd.linear2"), unet.add_embedding.linear_2)

        for index, block in enumerate(unet.down_blocks):
            self._export_unet_block(ctx.with_subname(f"down{index}"), block)
        self._export_unet_block(ctx.with_subname("mid"), unet.mid_block)
        for index, block in enumerate(unet.up_blocks):
            self._export_unet_block(ctx.with_subname(f"up{index}"), block)

        self._export_group_norm(ctx.with_subname("conv_norm_out"), unet.conv_norm_out)
        self._export_conv2d(ctx.with_subname("conv_out"), unet.conv_out)

    def _export_alpha_decoder(self, ctx: Context, model) -> None:
        """The decoder that finds a layer's alpha: `UNet1024`, from LayerDiffuse.

        It reads the RGB the VAE decoded and the latent it was decoded from, and its blocks are
        the VAE's -- residual blocks with no timestep, attention with a group norm in front -- so
        the VAE's walk writes them.
        """
        self._export_conv2d(ctx.with_subname("conv_in"), model.conv_in)
        self._export_conv2d(ctx.with_subname("latent_conv_in"), model.latent_conv_in)
        for index, block in enumerate(model.down_blocks):
            self._export_vae_block(ctx.with_subname(f"down{index}"), block)
        self._export_vae_block(ctx.with_subname("mid"), model.mid_block)
        for index, block in enumerate(model.up_blocks):
            self._export_vae_block(ctx.with_subname(f"up{index}"), block)
        self._export_group_norm(ctx.with_subname("conv_norm_out"), model.conv_norm_out)
        self._export_conv2d(ctx.with_subname("conv_out"), model.conv_out)

    def _export_vae(self, ctx: Context, vae) -> None:
        """Both halves of an autoencoder, named as SdxlExporter names SDXL's."""
        self._export_vae_decoder(ctx, vae.decoder)
        self._export_conv2d(ctx.with_subname("post_quant_conv"), vae.post_quant_conv)
        self._export_vae_encoder(ctx.with_subname("encoder"), vae.encoder)
        self._export_conv2d(ctx.with_subname("encoder.quant_conv"), vae.quant_conv)

    def _constant(self, ctx: Context, tensor: torch.Tensor) -> None:
        """A tensor computed here rather than read from a checkpoint, kept at float32."""
        self._writer.write_tensor(ctx.with_quant(Quant.NONE), tensor.float(), preserve_dtype=True)

    def export_layers(self, ctx: Context, model) -> None:
        self._export_frame_unet(ctx.with_subname("unet"), model.unet)
        self._float32 = True
        self._export_vae(ctx.with_subname("vae"), model.vae)
        self._float32 = False
        # Half, unlike the autoencoder beside it: its activations peak near a hundred on real
        # layers, and in float16 it lands within a level of 255 of float32 at half the cost.
        self._export_alpha_decoder(ctx.with_subname("alpha"), model.alpha)
        for name, (context, pooled) in model.conditions.items():
            self._constant(ctx.with_subname(f"{name}.context"), context)
            self._constant(ctx.with_subname(f"{name}.pooled"), pooled)

    def export_depth(self, ctx: Context, model) -> None:
        self._export_frame_unet(ctx.with_subname("unet"), model.unet)
        self._float32 = True
        self._export_vae(ctx.with_subname("vae"), model.vae)
        self._float32 = False
        self._constant(ctx.with_subname("context"), model.context)


class Loaded:
    """The modules of one of the two models, and what its text encoders made of its prompts."""


def _repo(name_or_path: str) -> str:
    """A local directory for `name_or_path`, fetching it if it is a repository name."""
    if path.isdir(name_or_path):
        return name_or_path
    from huggingface_hub import snapshot_download
    return snapshot_download(name_or_path)


@torch.no_grad()
def load_layers(name_or_path: str) -> Loaded:
    from diffusers import AutoencoderKL
    from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer
    from modules.layerdiffuse.layerdiff3d import UNetFrameConditionModel
    from modules.layerdiffuse.vae import TransparentVAE

    root = _repo(name_or_path)
    model = Loaded()
    # The U-Net is published in bfloat16 and stays in it until the writer narrows it, which halves
    # what is held in memory against reading it as float32: it is four billion parameters.
    model.unet = UNetFrameConditionModel.from_pretrained(
        root, subfolder="unet", torch_dtype=torch.bfloat16)
    model.vae = AutoencoderKL.from_pretrained(root, subfolder="vae", torch_dtype=torch.float32)
    model.alpha = TransparentVAE.from_pretrained(
        root, subfolder="trans_vae", torch_dtype=torch.float32).decoder.model
    model.scheduler = _scheduler_config(root)

    if model.unet.get_tag_version() != "v3":
        raise ValueError(
            f"the layers model says tag version {model.unet.get_tag_version()!r}; the runtime "
            "draws the v3 layers and only those")

    # `encode_cropped_prompt_77tokens`: each tag on its own, padded to 77, the second to last
    # hidden state of both encoders side by side and the pooled vector of the second. In float32
    # here, where upstream caches the same thing in bfloat16.
    tokenizers = [CLIPTokenizer.from_pretrained(root, subfolder=f"tokenizer{s}") for s in ("", "_2")]
    encoders = [
        CLIPTextModel.from_pretrained(root, subfolder="text_encoder", torch_dtype=torch.float32),
        CLIPTextModelWithProjection.from_pretrained(
            root, subfolder="text_encoder_2", torch_dtype=torch.float32),
    ]

    def encode(tags):
        hidden, pooled = [], None
        for tokenizer, encoder in zip(tokenizers, encoders):
            ids = tokenizer(
                tags, padding="max_length", max_length=tokenizer.model_max_length,
                truncation=True, return_tensors="pt").input_ids
            out = encoder(ids, output_hidden_states=True, return_dict=False)
            pooled = out[0]
            hidden.append(out[-1][-2])
        return torch.cat(hidden, dim=-1), pooled

    # The group embeddings, folded in: `e + group_embeds2[g](e)` for what cross attention reads
    # and `t + group_embeds[g](t)` for the pooled vector, which is all the U-Net ever does with
    # them (`UNetFrameConditionModel.forward`, `get_aug_embed`).
    model.conditions = {}
    for group, (name, tags) in enumerate((("body", BODY_TAGS), ("head", HEAD_TAGS))):
        context, pooled = encode(tags)
        embed2 = model.unet.group_embeds2[group].float()
        embed = model.unet.group_embeds[group].float()
        model.conditions[name] = (context + embed2(context), pooled + embed(pooled))
    return model


@torch.no_grad()
def load_depth(name_or_path: str) -> Loaded:
    from diffusers import AutoencoderKL
    from transformers import CLIPTextModel, CLIPTokenizer
    from modules.layerdiffuse.layerdiff3d import UNetFrameConditionModel

    root = _repo(name_or_path)
    model = Loaded()
    model.unet = UNetFrameConditionModel.from_pretrained(
        root, subfolder="unet", torch_dtype=torch.bfloat16)
    model.vae = AutoencoderKL.from_pretrained(root, subfolder="vae", torch_dtype=torch.float32)
    model.scheduler = _scheduler_config(root)

    # The empty prompt, unpadded -- the start and end markers and nothing else -- through the
    # last hidden state (`MarigoldDepthPipeline.encode_empty_text`).
    tokenizer = CLIPTokenizer.from_pretrained(root, subfolder="tokenizer")
    encoder = CLIPTextModel.from_pretrained(
        root, subfolder="text_encoder", torch_dtype=torch.float32)
    ids = tokenizer("", padding="do_not_pad", return_tensors="pt").input_ids
    model.context = encoder(ids)[0]
    return model


def _scheduler_config(root: str) -> dict:
    import json
    with open(path.join(root, "scheduler", "scheduler_config.json")) as fp:
        return json.load(fp)


def _check_scheduler(name: str, scheduler: dict, expected: dict) -> None:
    """Refuse a schedule the runtime would build differently, rather than export it as another."""
    for key, want in expected.items():
        got = scheduler.get(key)
        if got != want:
            raise ValueError(f"the {name} scheduler wants {key}={got!r}; the runtime builds {want!r}")


def unet_config(prefix: str, unet) -> dict:
    """What `UnetConfig::from_section` reads, under keys starting `{prefix}_`."""
    config = unet.config
    attention = [
        int(n if "Attn" in block_type else 0)
        for n, block_type in zip(
            _per_level(config.transformer_layers_per_block, len(config.down_block_types)),
            config.down_block_types)]
    mid = unet.mid_block.attentions[0].transformer_blocks
    heads = config.attention_head_dim
    section = {
        f"{prefix}_in_channels": str(config.in_channels),
        f"{prefix}_block_out_channels": [int(c) for c in config.block_out_channels],
        f"{prefix}_layers_per_block": str(config.layers_per_block),
        f"{prefix}_transformer_layers_per_block": attention,
        f"{prefix}_mid_transformer_layers": str(len(mid)),
        f"{prefix}_attention_head_dim": [int(n) for n in _per_level(heads, len(attention))],
        f"{prefix}_norm_num_groups": str(config.norm_num_groups),
        f"{prefix}_cross_attention_dim": str(config.cross_attention_dim),
        f"{prefix}_frame_attention": "true",
    }
    if getattr(unet, "add_embedding", None) is not None:
        section[f"{prefix}_addition_time_embed_dim"] = str(config.addition_time_embed_dim)
        section[f"{prefix}_projection_class_embeddings_input_dim"] = str(
            config.projection_class_embeddings_input_dim)
    if int(config.out_channels) != 4:
        raise ValueError(f"a U-Net with {config.out_channels} channels out, where 4 are expected")
    return section


def _per_level(value, levels: int) -> list:
    return list(value) if isinstance(value, (list, tuple)) else [value] * levels


def vae_config(prefix: str, vae) -> dict:
    config = vae.config
    return {
        f"{prefix}_scaling_factor": str(config.scaling_factor),
        f"{prefix}_block_out_channels": [int(c) for c in config.block_out_channels],
        f"{prefix}_layers_per_block": str(config.layers_per_block),
        f"{prefix}_norm_num_groups": str(config.norm_num_groups),
    }


def alpha_config(alpha) -> dict:
    """UNet1024's shape. Its block types and widths are constructor defaults, not a config file."""
    down = [type(block).__name__ for block in alpha.down_blocks]
    attention = [int(name.startswith("Attn")) for name in down]
    first = alpha.down_blocks[-1].attentions[0]
    return {
        "alpha_block_out_channels": [int(block.resnets[0].out_channels) for block in alpha.down_blocks],
        "alpha_attention": attention,
        "alpha_layers_per_block": str(len(alpha.down_blocks[0].resnets)),
        "alpha_norm_num_groups": str(alpha.conv_norm_out.num_groups),
        "alpha_head_dim": str(first.inner_dim // first.heads),
        # The latent joins the picture before this down block, at an eighth of its size.
        "alpha_latent_level": "3",
    }


def generate_config(layers: Loaded, depth: Loaded) -> dict:
    _check_scheduler("layers", layers.scheduler, {
        "_class_name": "DPMSolverMultistepScheduler", "algorithm_type": "sde-dpmsolver++",
        "solver_order": 2, "solver_type": "midpoint", "final_sigmas_type": "zero",
        "lower_order_final": True, "prediction_type": "epsilon", "beta_schedule": "scaled_linear",
        "timestep_spacing": "leading", "use_karras_sigmas": False, "rescale_betas_zero_snr": False,
        "thresholding": False, "trained_betas": None})
    _check_scheduler("depth", depth.scheduler, {
        "_class_name": "DDIMScheduler", "prediction_type": "v_prediction",
        "beta_schedule": "scaled_linear", "timestep_spacing": "trailing",
        "rescale_betas_zero_snr": True, "set_alpha_to_one": False, "clip_sample": False,
        "thresholding": False, "trained_betas": None})

    section = {"latent_channels": "4"}
    section.update(unet_config("layers_unet", layers.unet))
    section.update(vae_config("layers_vae", layers.vae))
    section.update(alpha_config(layers.alpha))
    section.update(unet_config("depth_unet", depth.unet))
    section.update(vae_config("depth_vae", depth.vae))
    for prefix, scheduler in (("layers", layers.scheduler), ("depth", depth.scheduler)):
        section[f"{prefix}_scheduler_num_train_timesteps"] = str(scheduler["num_train_timesteps"])
        section[f"{prefix}_scheduler_beta_start"] = str(scheduler["beta_start"])
        section[f"{prefix}_scheduler_beta_end"] = str(scheduler["beta_end"])
        section[f"{prefix}_scheduler_steps_offset"] = str(scheduler.get("steps_offset", 0))

    section["body_tags"] = ",".join(BODY_TAGS)
    section["head_tags"] = ",".join(HEAD_TAGS)
    section["depth_tags"] = ",".join(DEPTH_TAGS)
    for name, parts in DEPTH_COMPOSITES.items():
        section[f"depth_composite_{name}"] = ",".join(parts)
    section["layers_resolution"] = str(LAYERS_RESOLUTION)
    section["layers_steps"] = str(LAYERS_STEPS)
    section["depth_resolution"] = str(DEPTH_RESOLUTION)
    section["depth_steps"] = str(DEPTH_STEPS)

    return {"model": {"type": "see_through"}, "see_through": section}


def export(layers: Loaded, depth: Loaded, writer) -> dict:
    exporter = SeeThroughExporter(writer)
    exporter.export_layers(Context("see_through.layers"), layers)
    exporter.export_depth(Context("see_through.depth"), depth)
    return generate_config(layers, depth)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="export See-through to the libwaifu format.")
    parser.add_argument("-layers", default="layerdifforg/seethroughv0.0.2_layerdiff3d",
                        help="the layers model: a repository name or a local copy of one.")
    parser.add_argument("-depth", default="layerdifforg/seethroughv0.0.1_marigold",
                        help="the depth model: a repository name or a local copy of one.")
    parser.add_argument("-output", default="see-through.safetensors",
                        help="what to call the package. Names the weights and the manifest.")
    parser.add_argument("-part-size", dest="part_size", type=lambda s: __import__(
        "model_writer").parse_size(s), default=None,
                        help="split the weights into files no larger than this, as 2GB.")
    args = parser.parse_args()

    layers = load_layers(args.layers)
    depth = load_depth(args.depth)
    writer = open_weights(args.output, args.part_size)
    config = export(layers, depth, writer)
    for name in writer.finish(config):
        print(f"wrote {name}")
