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

"""What See-through computes on one real illustration, for `waifu/tests/see_through.rs`.

Upstream's own pipelines are run -- `KDiffusionStableDiffusionXLPipeline` for the layers and
`MarigoldDepthPipeline` for depth, on CUDA in bfloat16 as `inference_psd.py` runs them -- with
hooks that record what goes in and out of each model along the way. Every input a test hands the
runtime is one of those: a latent off the real trajectory, the layers the model really drew, the
picture really padded. Nothing here is `torch.randn` standing in for an activation.

Two things upstream leaves to an unseeded generator are pinned so the record means something: the
VAE encoding of the picture is its mean rather than a draw from it, and the noise the SDE sampler
adds at every step is drawn from a seeded generator and written down beside the step it went into.

Only the body pass is recorded; the head pass is the same model on a crop. The resolutions are
smaller than upstream's 1280 and 768 so that the record is a few tens of megabytes and a test
reads it in seconds; every block is exercised at either size.

    PYTHONPATH=~/.cache/libwaifu/src/see-through/common \\
        .venv-see-through/bin/python tools/see_through_reference.py \\
        -layers ~/.cache/libwaifu/hf/seethroughv0.0.2_layerdiff3d \\
        -depth ~/.cache/libwaifu/hf/seethroughv0.0.1_marigold \\
        -output models/see-through_test.safetensors
"""

from __future__ import annotations

import argparse
from os import path

import numpy as np
import torch
from PIL import Image

LAYERS_RESOLUTION = 512
DEPTH_RESOLUTION = 384
LAYERS_STEPS = 30
# The step whose U-Net input and output are kept, partway along so the latent is neither noise
# nor a finished picture.
RECORDED_STEP = 12
SEED = 42


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-layers", required=True)
    parser.add_argument("-depth", required=True)
    parser.add_argument("-image", default=None,
                        help="the illustration; upstream's common/assets/test_image.png by default")
    parser.add_argument("-output", required=True)
    args = parser.parse_args()

    import diffusers.models.autoencoders.vae as vae_module
    from modules.layerdiffuse.diffusers_kdiffusion_sdxl import KDiffusionStableDiffusionXLPipeline
    from modules.layerdiffuse.layerdiff3d import UNetFrameConditionModel
    from modules.layerdiffuse.vae import TransparentVAE
    from modules.marigold import MarigoldDepthPipeline
    from utils.cv import center_square_pad_resize, img_alpha_blending, smart_resize
    from utils.torchcv import pad_rgb_torch

    import modules.layerdiffuse as layerdiffuse_package
    source = path.dirname(path.dirname(path.dirname(layerdiffuse_package.__file__)))
    image_path = args.image or path.join(source, "assets", "test_image.png")

    out = {}

    def keep(name, tensor, dtype=torch.float32):
        out[name] = tensor.detach().to("cpu", dtype).contiguous()

    # The picture's latent is the mean, not a draw.
    vae_module.DiagonalGaussianDistribution.sample = lambda self, generator=None: self.mean

    torch.manual_seed(SEED)
    device = "cuda"
    # Float32 means float32: PyTorch otherwise multiplies and convolves it in TF32 on CUDA, which
    # is most of a percent from float32 on the autoencoders.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    # ---- layers ----------------------------------------------------------------------------
    trans_vae = TransparentVAE.from_pretrained(args.layers, subfolder="trans_vae")
    unet = UNetFrameConditionModel.from_pretrained(args.layers, subfolder="unet")
    pipe = KDiffusionStableDiffusionXLPipeline.from_pretrained(
        args.layers, trans_vae=trans_vae, unet=unet, scheduler=None)
    for module in (pipe.vae, pipe.trans_vae, pipe.unet, pipe.text_encoder, pipe.text_encoder_2):
        module.to(dtype=torch.bfloat16, device=device)
    pipe.cache_tag_embeds()
    pipe.set_progress_bar_config(disable=True)

    picture = np.array(Image.open(image_path).convert("RGBA"))
    page = center_square_pad_resize(picture, LAYERS_RESOLUTION)
    keep("layers.page", torch.from_numpy(page), torch.uint8)

    # Every U-Net call: its input, timestep and answer at the recorded step.
    calls = []
    unet_forward = pipe.unet.forward

    def recording_unet(sample, timestep, *a, **k):
        result = unet_forward(sample, timestep, *a, **k)
        if len(calls) == RECORDED_STEP:
            keep("layers.unet_input", sample[0])
            keep("layers.unet_timestep", torch.tensor([float(timestep)]))
            keep("layers.unet_output", result[0][0])
        if len(calls) == 0:
            keep("layers.page_latent", sample[0, 0, 4:])
        calls.append(float(timestep))
        return result

    pipe.unet.forward = recording_unet

    # The SDE noise, from a seeded generator, and the first two steps written out whole: the
    # first is first order and the second the midpoint correction.
    scheduler_step = pipe.scheduler.step
    noise_generator = torch.Generator("cpu").manual_seed(SEED + 1)
    steps = []

    def recording_step(model_output, timestep, sample, *a, **k):
        noise = torch.randn(model_output.shape, generator=noise_generator).to(
            device=model_output.device, dtype=torch.float32)
        index = len(steps)
        if index < 2:
            keep(f"layers.step{index}.eps", model_output[0])
            keep(f"layers.step{index}.sample", sample[0])
            keep(f"layers.step{index}.noise", noise[0])
        result = scheduler_step(model_output, timestep, sample, *a, variance_noise=noise, **k)
        if index < 2:
            keep(f"layers.step{index}.out", result[0][0])
        steps.append(index)
        return result

    pipe.scheduler.step = recording_step

    decoder_forward = pipe.trans_vae.decoder.forward
    final = []

    def recording_decoder(sd_vae, latent, *a, **k):
        final.append(latent)
        return decoder_forward(sd_vae, latent, *a, **k)

    pipe.trans_vae.decoder.forward = recording_decoder

    body_tags = ["front hair", "back hair", "head", "neck", "neckwear", "topwear", "handwear",
                 "bottomwear", "legwear", "footwear", "tail", "wings", "objects"]
    rng = torch.Generator(device=device).manual_seed(SEED)
    output = pipe(strength=1.0, num_inference_steps=LAYERS_STEPS, batch_size=1, generator=rng,
                  guidance_scale=1.0, prompt=body_tags, negative_prompt="", fullpage=page,
                  group_index=0)
    keep("layers.timesteps", torch.tensor(calls))
    # The final latents, scaled as the sampler leaves them, and what the decoder made of them.
    scaling = pipe.vae.config.scaling_factor
    keep("layers.final", torch.cat(final) * scaling)
    layers = {tag: image for tag, image in zip(body_tags, output.images)}
    keep("layers.rgba", torch.from_numpy(np.stack(output.images)), torch.uint8)
    print(f"layers: {len(calls)} steps, {len(layers)} layers")

    del pipe, unet, trans_vae
    torch.cuda.empty_cache()

    # The autoencoders again in float32, which is what the runtime runs them in: upstream's
    # bfloat16 is six percent from float32 on its own, which would drown any real disagreement.
    from diffusers import AutoencoderKL

    def encode32(root, images):
        vae = AutoencoderKL.from_pretrained(root, subfolder="vae", torch_dtype=torch.float32).to(device)
        with torch.no_grad():
            latents = [vae.encode(image[None].to(device)).latent_dist.mean * vae.config.scaling_factor
                       for image in images]
        del vae
        return torch.cat(latents)

    rgb = torch.from_numpy(page[..., :3]).float().permute(2, 0, 1) / 127.5 - 1
    keep("layers.page_latent_f32", encode32(args.layers, [rgb]))

    # ---- depth -----------------------------------------------------------------------------
    depth_unet = UNetFrameConditionModel.from_pretrained(args.depth, subfolder="unet")
    depth_pipe = MarigoldDepthPipeline.from_pretrained(args.depth, unet=depth_unet)
    depth_pipe.to(device=device, dtype=torch.bfloat16)
    depth_pipe.set_progress_bar_config(disable=True)

    # `apply_marigold`'s slots, from the layers just drawn; the head pass's are empty here.
    slots = ["hair", "headwear", "face", "eyes", "eyewear", "ears", "earwear", "nose", "mouth",
             "neck", "neckwear", "topwear", "handwear", "bottomwear", "legwear", "footwear",
             "tail", "wings", "objects"]
    composites = {"eyes": ["eyewhite", "irides", "eyelash", "eyebrow"],
                  "hair": ["back hair", "front hair"]}
    side = page.shape[0]
    empty = np.zeros((side, side, 4), dtype=np.uint8)

    def cleared(tag):
        layer = layers.get(tag, empty).copy()
        layer[..., -1][layer[..., -1] < 15] = 0
        return layer

    img_list = []
    for tag in slots:
        if tag in composites:
            img_list.append(img_alpha_blending([cleared(t) for t in composites[tag]],
                                               premultiplied=False))
        else:
            img_list.append(cleared(tag))
    blended = np.zeros((side, side), dtype=np.float32)
    for image in img_list:
        blended += image[..., -1].astype(np.float32) / 255
    full = page.copy()
    full[..., -1] = (np.clip(blended, 0, 1) * 255).astype(np.uint8)
    img_list.append(full)
    keep("depth.slots", torch.from_numpy(np.stack(img_list)), torch.uint8)

    img_list = [smart_resize(image, (DEPTH_RESOLUTION, DEPTH_RESOLUTION)) for image in img_list]
    keep("depth.slots_small", torch.from_numpy(np.stack(img_list)), torch.uint8)

    # What pad_rgb makes of the face and of the full page, the two most different slots.
    for name, index in (("face", slots.index("face")), ("full", len(slots))):
        argb = np.concatenate([img_list[index][..., 3:], img_list[index][..., :3]], axis=2)
        argb = torch.from_numpy(argb.astype(np.float32) / 255).permute(2, 0, 1)[None]
        keep(f"depth.pad_rgb.{name}", pad_rgb_torch(argb, return_format="rgb")[0])

    padded = []
    for image in img_list:
        argb = np.concatenate([image[..., 3:], image[..., :3]], axis=2)
        argb = torch.from_numpy(argb.astype(np.float32) / 255).permute(2, 0, 1)[None]
        padded.append(pad_rgb_torch(argb, return_format="rgb")[0] * 2 - 1)
    keep("depth.slot_latents_f32", encode32(args.depth, padded))

    depth_calls = []
    depth_forward = depth_pipe.unet.forward

    def recording_depth_unet(sample, timestep, *a, **k):
        result = depth_forward(sample, timestep, *a, **k)
        if len(depth_calls) == 0:
            keep("depth.condition", sample[0, :, :8])
            keep("depth.noise", sample[0, :, 8:])
        if len(depth_calls) == 1:
            keep("depth.unet_input", sample[0])
            keep("depth.unet_timestep", torch.tensor([float(timestep)]))
            keep("depth.unet_output", result.sample[0])
        depth_calls.append(float(timestep))
        return result

    depth_pipe.unet.forward = recording_depth_unet

    decode = depth_pipe.decode_depth
    depth_final = []

    def recording_decode(latent):
        depth_final.append(latent)
        return decode(latent)

    depth_pipe.decode_depth = recording_decode

    torch.manual_seed(SEED + 2)
    depth_out = depth_pipe(color_map=None, img_list=img_list, denoising_steps=-1)
    keep("depth.timesteps", torch.tensor(depth_calls))
    keep("depth.final", torch.cat(depth_final))
    keep("depth.maps", depth_out.depth_tensor.squeeze(1) if depth_out.depth_tensor.ndim == 4
         else depth_out.depth_tensor)
    print(f"depth: {len(depth_calls)} steps")

    from safetensors.torch import save_file
    save_file(out, args.output)
    total = sum(t.numel() * t.element_size() for t in out.values())
    print(f"wrote {args.output}: {len(out)} tensors, {total / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
