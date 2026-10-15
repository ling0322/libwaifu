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

"""The Anima reference, which is diffusers.

`sdxl_exporter.py` gets its reference outputs by running the diffusers pipeline. This is the same
arrangement for Anima: diffusers' own `AnimaTextConditioner`, `CosmosTransformer3DModel` and
`AutoencoderKLQwenImage`, the three models `AnimaModularPipeline` is built from, run on the same
weights the exporter reads. Nothing here is a transcription of the model. What is written here is
only how the published files' tensor names map onto those classes, and that is checked: every
load is strict, so a name this gets wrong is an error and not a randomly initialized layer.

Nothing here is used at run time. It is imported by `anima_exporter.py -test_output` and by
nothing else, and it needs diffusers 0.41 and a transformers new enough to have Qwen3, which the
pair in `tools/requirements.txt` is not. They go in a venv of their own:

    python3 -m venv .venv-anima
    .venv-anima/bin/pip install -r tools/requirements-anima.txt

    .venv-anima/bin/python tools/anima_exporter.py ... -test_output anima_test.waifupkg

Everything runs in float32 on the CPU. The residual stream of this model is wide enough that it
overflows half precision, so float32 throughout is what a reference owes it.
"""

from __future__ import annotations

import torch

# The rotary position extrapolation ratios, time first, as `rope_scale` in the transformer config
# of `circlestone-labs/Anima-Base-v1.0-Diffusers` and of diffusers' Cosmos-2.0 2B text-to-image
# entry, which Anima's denoiser is trained from. Each is an NTK factor and not a base -- the base
# is `10000 * ratio ** (dim / (dim - 2))` -- and they are a property of the model, not a choice
# made at run time.
T_EXTRAPOLATION_RATIO = 1.0
H_EXTRAPOLATION_RATIO = 4.0
W_EXTRAPOLATION_RATIO = 4.0

# The rest of that config. Only the width and the block count are read off the tensors.
DENOISER_CONFIG = dict(
    in_channels=16,
    out_channels=16,
    attention_head_dim=128,
    mlp_ratio=4.0,
    text_embed_dim=1024,
    adaln_lora_dim=256,
    max_size=(128, 240, 240),
    patch_size=(1, 2, 2),
    rope_scale=(T_EXTRAPOLATION_RATIO, H_EXTRAPOLATION_RATIO, W_EXTRAPOLATION_RATIO),
    concat_padding_mask=True,
    extra_pos_embed_type=None,
)

CONTEXT_LENGTH = 512

# How the denoiser's names in the published file become diffusers' names, applied in this order
# as substring replacements. The order matters: `blocks` has to become `transformer_blocks` before
# anything inside a block is renamed, and `final_layer.adaln_modulation` must not be caught by the
# per-block `adaln_modulation_*` patterns, which it is not because those carry an underscore.
DENOISER_RENAMES = (
    ("t_embedder.1", "time_embed.t_embedder"),
    ("t_embedding_norm", "time_embed.norm"),
    ("blocks", "transformer_blocks"),
    ("adaln_modulation_self_attn.1", "norm1.linear_1"),
    ("adaln_modulation_self_attn.2", "norm1.linear_2"),
    ("adaln_modulation_cross_attn.1", "norm2.linear_1"),
    ("adaln_modulation_cross_attn.2", "norm2.linear_2"),
    ("adaln_modulation_mlp.1", "norm3.linear_1"),
    ("adaln_modulation_mlp.2", "norm3.linear_2"),
    ("self_attn", "attn1"),
    ("cross_attn", "attn2"),
    ("q_proj", "to_q"),
    ("k_proj", "to_k"),
    ("v_proj", "to_v"),
    ("output_proj", "to_out.0"),
    ("q_norm", "norm_q"),
    ("k_norm", "norm_k"),
    ("mlp.layer1", "ff.net.0.proj"),
    ("mlp.layer2", "ff.net.2"),
    ("x_embedder.proj.1", "patch_embed.proj"),
    ("final_layer.adaln_modulation.1", "norm_out.linear_1"),
    ("final_layer.adaln_modulation.2", "norm_out.linear_2"),
    ("final_layer.linear", "proj_out"),
)

# Buffers and bookkeeping some releases carry beside the weights. diffusers rebuilds the position
# tables at construction, and the rest is training state.
DENOISER_SKIPPED = ("pos_embedder.", "accum_", "_extra_state")

ADAPTER_PREFIX = "llm_adapter."


def check() -> None:
    """Fail early, before anything is read, if this environment cannot run the reference."""
    try:
        import diffusers
        from diffusers import AnimaTextConditioner  # noqa: F401
        from transformers import Qwen3Model  # noqa: F401
    except ImportError as error:
        raise SystemExit(
            f"-test_output needs diffusers with Anima and transformers with Qwen3 ({error}). "
            f"They do not fit beside the pair tools/requirements.txt pins; install "
            f"tools/requirements-anima.txt into a venv of its own and run the exporter from it.")
    del diffusers


def _load(model, weights: dict, what: str):
    """Load and insist that the model and the checkpoint agree about every tensor.

    A reference that quietly runs with a randomly initialized layer is worse than no reference.
    """
    missing, unexpected = model.load_state_dict(weights, strict=False)
    if missing or unexpected:
        raise SystemExit(
            f"{what}: the diffusers model does not match this checkpoint -- "
            f"{len(missing)} missing, {len(unexpected)} unexpected\n"
            f"  missing: {missing[:4]}\n  unexpected: {unexpected[:4]}"
        )
    return model.eval().float()


class TextEncoder:
    """Qwen3-0.6B, read for its last hidden state rather than for a next token.

    The last hidden state, after the final norm, is what `AnimaTextEncoderStep` hands the adapter.
    """

    def __init__(self, weights: dict) -> None:
        from transformers import Qwen3Config, Qwen3Model

        weights = {name.removeprefix("model."): tensor for name, tensor in weights.items()
                   if not name.startswith("lm_head.")}
        vocab, hidden = weights["embed_tokens.weight"].shape
        head_dim = weights["layers.0.self_attn.q_norm.weight"].shape[0]
        config = Qwen3Config(
            vocab_size=vocab,
            hidden_size=hidden,
            intermediate_size=weights["layers.0.mlp.gate_proj.weight"].shape[0],
            num_hidden_layers=1 + max(int(name.split(".")[1]) for name in weights
                                      if name.startswith("layers.")),
            num_attention_heads=weights["layers.0.self_attn.q_proj.weight"].shape[0] // head_dim,
            num_key_value_heads=weights["layers.0.self_attn.k_proj.weight"].shape[0] // head_dim,
            head_dim=head_dim,
            max_position_embeddings=32768,
            rms_norm_eps=1e-6,
            rope_theta=1000000.0,
            attention_bias=False,
            tie_word_embeddings=False,
        )
        self.model = _load(Qwen3Model(config), weights, "the text encoder")

    def __call__(self, ids: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids=ids).last_hidden_state


class Denoiser:
    """The adapter and the transformer, which diffusers keeps as two models and the file as one.

    The adapter is reached both before padding, which is its own answer, and after, which is what
    the denoiser reads: the package carries both, and a runtime can go wrong in either.
    """

    def __init__(self, weights: dict, blocks: int, width: int) -> None:
        from diffusers import AnimaTextConditioner, CosmosTransformer3DModel

        adapter = {name.removeprefix(ADAPTER_PREFIX): tensor for name, tensor in weights.items()
                   if name.startswith(ADAPTER_PREFIX)}
        model_dim = adapter["blocks.0.self_attn.q_proj.weight"].shape[0]
        vocab, target_dim = adapter["embed.weight"].shape
        self.adapter = _load(AnimaTextConditioner(
            source_dim=adapter["blocks.0.cross_attn.k_proj.weight"].shape[1],
            target_dim=target_dim,
            model_dim=model_dim,
            num_layers=1 + max(int(name.split(".")[1]) for name in adapter
                               if name.startswith("blocks.")),
            num_attention_heads=model_dim // adapter["blocks.0.self_attn.q_norm.weight"].shape[0],
            target_vocab_size=vocab,
            min_sequence_length=CONTEXT_LENGTH,
        ), adapter, "the adapter")

        dit = {}
        for name, tensor in weights.items():
            if name.startswith(ADAPTER_PREFIX) or any(s in name for s in DENOISER_SKIPPED):
                continue
            for old, new in DENOISER_RENAMES:
                name = name.replace(old, new)
            if name in dit:
                raise SystemExit(f"the denoiser: two tensors were renamed to {name}")
            dit[name] = tensor
        config = dict(DENOISER_CONFIG, num_layers=blocks,
                      num_attention_heads=width // DENOISER_CONFIG["attention_head_dim"])
        self.model = _load(CosmosTransformer3DModel(**config), dit, "the denoiser")

    def adapt(self, hidden: torch.Tensor, t5_ids: torch.Tensor) -> torch.Tensor:
        # The conditioner pads its answer with zeros out to the context length at the very end,
        # so the rows before the padding are the adapter's own output.
        return self.adapt_padded(hidden, t5_ids)[:, : t5_ids.shape[1]]

    def adapt_padded(self, hidden: torch.Tensor, t5_ids: torch.Tensor) -> torch.Tensor:
        padded = self.adapter(source_hidden_states=hidden, target_input_ids=t5_ids)
        if padded.shape[1] != max(CONTEXT_LENGTH, t5_ids.shape[1]):
            raise SystemExit(f"the adapter returned {padded.shape[1]} rows of context")
        return padded

    def __call__(self, latent: torch.Tensor, timestep: torch.Tensor,
                 context: torch.Tensor) -> torch.Tensor:
        """One velocity, from a four dimensional latent. Anima draws stills, so time is one.

        `timestep` is the sigma itself, which is what `AnimaLoopBeforeDenoiser` passes: the
        scheduler's timestep divided by its thousand training steps. The padding mask is the
        all-zero one the pipeline builds, since nothing this runtime draws is padded.
        """
        n, _, h, w = latent.shape
        padding = latent.new_zeros(1, 1, h * 8, w * 8)
        out = self.model(hidden_states=latent.unsqueeze(2), timestep=timestep,
                         encoder_hidden_states=context, padding_mask=padding,
                         return_dict=False)[0]
        return out.squeeze(2)


def _vae_name(name: str) -> str:
    """A tensor name in the published Qwen-Image VAE to the one `AutoencoderKLQwenImage` uses."""

    def residual(name: str) -> str:
        for old, new in ((".residual.0.", ".norm1."), (".residual.2.", ".conv1."),
                         (".residual.3.", ".norm2."), (".residual.6.", ".conv2."),
                         (".shortcut.", ".conv_shortcut.")):
            name = name.replace(old, new)
        return name

    def middle(name: str) -> str:
        for old, new in ((".middle.0.", ".mid_block.resnets.0."),
                         (".middle.1.", ".mid_block.attentions.0."),
                         (".middle.2.", ".mid_block.resnets.1.")):
            name = name.replace(old, new)
        return residual(name)

    prefixes = (
        ("conv1.", "quant_conv."),
        ("conv2.", "post_quant_conv."),
        ("encoder.conv1.", "encoder.conv_in."),
        ("decoder.conv1.", "decoder.conv_in."),
        ("encoder.head.0.", "encoder.norm_out."),
        ("encoder.head.2.", "encoder.conv_out."),
        ("decoder.head.0.", "decoder.norm_out."),
        ("decoder.head.2.", "decoder.conv_out."),
    )
    for old, new in prefixes:
        if name.startswith(old):
            return new + name.removeprefix(old)
    if name.startswith("encoder.downsamples."):
        return residual("encoder.down_blocks." + name.removeprefix("encoder.downsamples."))
    if name.startswith("decoder.upsamples."):
        # Four to a resolution, flattened: three residual blocks and then the upsampler, except
        # the last resolution, which has no upsampler after it.
        index, rest = name.removeprefix("decoder.upsamples.").split(".", 1)
        index = int(index)
        if index in (3, 7, 11):
            return residual(f"decoder.up_blocks.{(index - 3) // 4}.upsamplers.0.{rest}")
        return residual(f"decoder.up_blocks.{index // 4}.resnets.{index % 4}.{rest}")
    if name.startswith(("encoder.middle.", "decoder.middle.")):
        return middle(name)
    return residual(name)


class Vae:
    """The Qwen-Image VAE, in the three dimensions it is actually written in.

    libwaifu folds every `Conv3d` here onto `W[:, :, -1]` at export time, which is exact when the
    temporal extent is one. Running the real thing in both directions is what makes the exported
    tensors a check on that folding rather than a restatement of it.

    Both halves, because both are in the package: the decoder is what a text to image run ends
    with, and the encoder is where an image to image run begins. One object rather than two, so
    that a round trip is the same weights going out and coming back.
    """

    def __init__(self, weights: dict, latents_mean, latents_std) -> None:
        from diffusers import AutoencoderKLQwenImage

        self.z_dim = 16
        renamed = {}
        for name, tensor in weights.items():
            new = _vae_name(name)
            if new in renamed:
                raise SystemExit(f"the VAE: two tensors were renamed to {new}")
            renamed[new] = tensor
        model = AutoencoderKLQwenImage(
            base_dim=weights["decoder.head.0.gamma"].shape[0],
            z_dim=self.z_dim,
            dim_mult=[1, 2, 4, 4],
            num_res_blocks=2,
            attn_scales=[],
            temperal_downsample=[False, True, True],
            dropout=0.0,
            input_channels=weights["encoder.conv1.weight"].shape[1],
            latents_mean=list(latents_mean),
            latents_std=list(latents_std),
        )
        self.model = _load(model, renamed, "the VAE")
        self.mean = torch.tensor(latents_mean, dtype=torch.float32).view(1, -1, 1, 1)
        self.std = torch.tensor(latents_std, dtype=torch.float32).view(1, -1, 1, 1)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """A `(N, 16, H, W)` latent to a `(N, 3, H * 8, W * 8)` picture.

        Sixteen channels normalized one at a time, where SDXL has a single factor. diffusers
        clamps the picture to [-1, 1] on the way out, and so the reference is a clamped picture:
        a runtime compared with it has to clamp its own the same way first.
        """
        unscaled = latent.float() * self.std + self.mean
        return self.model.decode(unscaled.unsqueeze(2), return_dict=False)[0].squeeze(2)

    def encode(self, image: torch.Tensor) -> torch.Tensor:
        """A `(N, 3, H, W)` picture to the `(N, 16, H / 8, W / 8)` latent that stands for it.

        The mean of the distribution rather than a draw from it: the draw is the one part of an
        encoder that two implementations have no way to agree on, and image to image wants the
        mean anyway. Scaled into the normalization the sampler works in, which is the decode
        above run backwards, so `encode(decode(x))` and `x` are comparable numbers.

        One frame, so the encoder runs a single chunk, and on the first chunk the cache is empty
        and the downsamplers never reach `time_conv` -- which is the branch the folded export
        assumes, and the reason those weights are not in the package.
        """
        posterior = self.model.encode(image.float().unsqueeze(2), return_dict=False)[0]
        mean = posterior.mode().squeeze(2)
        if mean.shape[1] != self.z_dim:
            raise SystemExit(f"the VAE encoder returned {mean.shape[1]} channels, not the "
                             f"{self.z_dim} of a latent")
        return (mean - self.mean) / self.std
