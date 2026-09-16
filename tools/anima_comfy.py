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

"""The Anima reference, which is ComfyUI itself.

`sdxl_exporter.py` gets its reference outputs by running the diffusers pipeline. This is the same
arrangement for Anima, against ComfyUI, which is the implementation Anima is published for and
the only one that can be called an authority on it.

There used to be a hand transcription here instead -- `anima_reference.py`, ~450 lines of torch
copied out of `comfy/ldm/cosmos/predict2.py` by eye. It was wrong, and the way it was wrong is
the argument against ever writing another one: it assumed every rotary extrapolation ratio was
1.0 where Anima asks for 4.0 on the two spatial axes, so every position in the picture rotated on
a base of 10000 instead of 42870.9. The velocity it produced was 13% off. It still drew perfectly
reasonable pictures, the runtime was built to match it, the test tensors were generated from it,
and the suite was green the whole time -- because every one of those measured agreement with the
transcription rather than with Anima. A reference that is not the upstream implementation is a
ruler nobody checks.

Nothing here is used at run time. It is imported by `anima_exporter.py -test_output` and by
nothing else, and it needs a ComfyUI checkout:

    git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git /path/to/ComfyUI

Do not install ComfyUI's requirements into `.venv`: it wants `transformers>=4.50.3` and
`tools/requirements.txt` pins 4.46.3 for the SDXL exporter's tokenizer. The handful of modules
these four stages need go in a directory of their own, which nothing else has to know about:

    .venv/bin/pip install --no-deps --target /path/to/pylibs \\
        comfy-aimdo==0.5.3 comfy-kitchen==0.2.33 einops torchvision torchaudio

    PYTHONPATH=/path/to/pylibs .venv/bin/python tools/anima_exporter.py ... \\
        -comfyui /path/to/ComfyUI -test_output anima_test.waifupkg

Everything runs in float32 on the CPU. The residual stream of this model is wide enough that
ComfyUI keeps it in float32 even when the rest is float16, so float32 throughout is what a
reference owes it.
"""

from __future__ import annotations

import sys

import torch

# What ComfyUI's `model_detection.py` asks for when `in_channels` is 16, which is what Anima's
# `x_embedder.proj.1.weight` of [2048, 68] comes to: `68 // 4 - 1`. The ratio is an NTK factor
# and not a base -- the base is `10000 * ratio ** (dim / (dim - 2))` -- and it is a property of
# the Cosmos-Predict2 2B this model was trained from, not a choice made at run time.
H_EXTRAPOLATION_RATIO = 4.0
W_EXTRAPOLATION_RATIO = 4.0
T_EXTRAPOLATION_RATIO = 1.0

# The rest of what `model_detection.py` derives for this architecture. Only two of these are read
# off the tensors -- the width and the block count -- and those are passed in.
DENOISER_CONFIG = dict(
    max_img_h=240, max_img_w=240, max_frames=128,
    in_channels=16, out_channels=16,
    patch_spatial=2, patch_temporal=1, concat_padding_mask=True,
    crossattn_emb_channels=1024,
    pos_emb_cls="rope3d", pos_emb_learnable=True, pos_emb_interpolation="crop",
    min_fps=1, max_fps=30,
    use_adaln_lora=True, adaln_lora_dim=256,
    num_heads=16,
    extra_per_block_abs_pos_emb=False,
    rope_h_extrapolation_ratio=H_EXTRAPOLATION_RATIO,
    rope_w_extrapolation_ratio=W_EXTRAPOLATION_RATIO,
    rope_t_extrapolation_ratio=T_EXTRAPOLATION_RATIO,
    extra_h_extrapolation_ratio=1.0, extra_w_extrapolation_ratio=1.0,
    extra_t_extrapolation_ratio=1.0,
    rope_enable_fps_modulation=False,
)

CONTEXT_LENGTH = 512


def use(comfyui: str) -> None:
    """Put a ComfyUI checkout on the path. Call this before anything else here."""
    # comfy.model_management reads argv at import and will take the exporter's flags for its own.
    sys.argv = [sys.argv[0], "--cpu"]
    if comfyui not in sys.path:
        sys.path.insert(0, comfyui)


def _ops():
    import comfy.ops
    return comfy.ops.disable_weight_init


def _load(model, weights: dict, what: str, ignore=()):
    """Load and insist that the config and the checkpoint agree about every tensor.

    A reference that quietly runs with a randomly initialized layer is worse than no reference,
    and `strict=False` is needed only for the position tables, which are buffers rebuilt at
    construction rather than weights.
    """
    missing, unexpected = model.load_state_dict(weights, strict=False)
    missing = [k for k in missing if not any(pattern in k for pattern in ignore)]
    if missing or unexpected:
        raise SystemExit(
            f"{what}: the config ComfyUI derives does not match this checkpoint -- "
            f"{len(missing)} missing, {len(unexpected)} unexpected\n"
            f"  missing: {missing[:4]}\n  unexpected: {unexpected[:4]}"
        )
    return model.eval().float()


class TextEncoder:
    """Qwen3-0.6B, read for its hidden states rather than for a next token."""

    def __init__(self, weights: dict) -> None:
        import comfy.text_encoders.llama as llama
        model = llama.Qwen3_06B({}, torch.float32, "cpu", _ops())
        self.model = _load(model, weights, "the text encoder")

    def __call__(self, ids: torch.Tensor) -> torch.Tensor:
        out = self.model(ids)
        return out[0] if isinstance(out, (tuple, list)) else out


class Denoiser:
    """The adapter and the transformer, which are one module in ComfyUI as they are in the file.

    `llm_adapter` is reached on its own as well as through `preprocess_text_embeds`, because the
    package carries the context both before padding, which is the adapter's own answer, and after,
    which is what the denoiser reads. A runtime can go wrong in either.
    """

    def __init__(self, weights: dict, blocks: int, width: int) -> None:
        import comfy.ldm.anima.model as anima
        model = anima.Anima(**DENOISER_CONFIG, num_blocks=blocks, model_channels=width,
                            image_model="anima", device="cpu", dtype=torch.float32,
                            operations=_ops())
        self.model = _load(model, weights, "the denoiser",
                           ignore=("pos_embedder", "_range"))

    def adapt(self, hidden: torch.Tensor, t5_ids: torch.Tensor) -> torch.Tensor:
        return self.model.llm_adapter(hidden, t5_ids)

    def adapt_padded(self, hidden: torch.Tensor, t5_ids: torch.Tensor) -> torch.Tensor:
        return self.model.preprocess_text_embeds(hidden, t5_ids)

    def __call__(self, latent: torch.Tensor, timestep: torch.Tensor,
                 context: torch.Tensor) -> torch.Tensor:
        """One velocity, from a four dimensional latent. Anima draws stills, so time is one.

        `_forward` rather than `forward`, which only wraps it in ComfyUI's patcher machinery.
        """
        out = self.model._forward(latent.unsqueeze(2), timestep, context,
                                  fps=None, padding_mask=None)
        return out.squeeze(2)


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
        import comfy.ldm.wan.vae as wan
        # The Wan 2.1 branch of comfy/sd.py's VAE detection, read off the same keys it reads.
        self.z_dim = 16
        config = dict(
            dim=weights["decoder.head.0.gamma"].shape[0],
            z_dim=self.z_dim,
            dim_mult=[1, 2, 4, 4],
            num_res_blocks=2,
            attn_scales=[],
            temperal_downsample=[False, True, True],
            image_channels=weights["encoder.conv1.weight"].shape[1],
            conv_out_channels=weights["decoder.head.2.weight"].shape[0],
            dropout=0.0,
        )
        self.model = _load(wan.WanVAE(**config), weights, "the VAE")
        self.mean = torch.tensor(latents_mean, dtype=torch.float32).view(1, -1, 1, 1)
        self.std = torch.tensor(latents_std, dtype=torch.float32).view(1, -1, 1, 1)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """A `(N, 16, H, W)` latent to a `(N, 3, H * 8, W * 8)` picture.

        Sixteen channels normalized one at a time, where SDXL has a single factor.
        """
        unscaled = latent.float() * self.std + self.mean
        return self.model.decode(unscaled.unsqueeze(2)).squeeze(2)

    def encode(self, image: torch.Tensor) -> torch.Tensor:
        """A `(N, 3, H, W)` picture to the `(N, 16, H / 8, W / 8)` latent that stands for it.

        The mean of the distribution rather than a draw from it: the draw is the one part of an
        encoder that two implementations have no way to agree on, and image to image wants the
        mean anyway. Scaled into the normalization the sampler works in, which is the decode
        above run backwards, so `encode(decode(x))` and `x` are comparable numbers.

        One frame, so `WanVAE.encode` runs a single chunk and passes `feat_cache=None` down,
        and with no cache `Resample`'s `downsample3d` branch never reaches `time_conv` -- which
        is the branch the folded export assumes, and the reason those weights are not in the
        package. `encode` returns mu itself upstream, but the shape is checked rather than
        trusted: a build that handed back the undivided moments would otherwise scale
        thirty-two channels by sixteen pairs and broadcast its way to a quiet, wrong answer.
        """
        moments = self.model.encode(image.float().unsqueeze(2))
        if isinstance(moments, (tuple, list)):
            moments = moments[0]
        moments = moments.squeeze(2)

        channels = moments.shape[1]
        if channels == 2 * self.z_dim:
            moments = moments.chunk(2, dim=1)[0]
        elif channels != self.z_dim:
            raise SystemExit(
                f"the VAE encoder returned {channels} channels, which is neither the "
                f"{self.z_dim} of a latent nor the {2 * self.z_dim} of its moments")

        return (moments - self.mean) / self.std
