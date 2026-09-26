# Qwen-Image 2.1

Notes on running Qwen-Image 2.1 in libwaifu. The shapes are the published `config.json` files,
checked against the tensor headers by `tools/qwen_image_exporter.py`; the behaviour shapes do not
show -- the template, the attention mask, the positions, the schedule, the decoder's shortcuts --
is read off diffusers' `QwenImage21Pipeline`, `QwenImage21Transformer2DModel` and
`AutoencoderKLQwenImage21`, which are also what the reference tensors are computed with.

It is **not** Qwen-Image 1 with new weights and it does not draw through the Qwen-Image VAE that
Anima and Krea 2 share. It is a 7B single-stream DiT, a Qwen3-VL-8B text encoder read at its last
layer, and a Wan 2.2-style autoencoder with sixty-four latent channels and an alpha channel out.

## What is published

`Qwen/Qwen-Image-2.1`, ungated, under the Qwen Research License, in the diffusers layout:

| file | size | what |
|---|---|---|
| `transformer/*.safetensors` | 14.5 GB | the denoiser, two shards, bf16, no biases |
| `text_encoder/*.safetensors` | 17.5 GB | Qwen3-VL-8B-Instruct, vision tower and `lm_head` included |
| `vae/diffusion_pytorch_model.safetensors` | 0.8 GB | both halves, float32 |
| `processor/tokenizer.json` | 11 MB | Qwen2's byte-level BPE |

The package keeps the denoiser, the encoder's language half and the whole VAE: 30 GB in fp16,
about half that with `-fp8`.

## The denoiser

| | |
|---|---|
| blocks | 32 |
| hidden | 4096 = 32 heads x 128, no grouping |
| SwiGLU | 12288 |
| patch | none: 64 latent channels in and out, one token per latent pixel |
| rope | theta 10000, axes `(16, 56, 56)` over `(frame, h, w)`, adjacent pairs |
| norm eps | 1e-6 |

### How a block runs

```
scale1, gate1, scale2, gate2 = modulation(silu(temb))    # one (4, 4096) for the whole model
x = x + tanh(gate1) * attn((1 + scale1) * layer_norm(x))
x = x + tanh(gate2) * ff((1 + scale2) * layer_norm(x))
```

**Block-causal attention.** The prompt attends to itself causally and never to the picture; the
picture attends to the whole prompt and the whole of itself. The runtime runs this as two
attentions per block -- prompt queries over prompt keys with a causal mask, picture queries over
everything with none -- which is the same decomposition as diffusers' mask-free processor.

**The prompt is modulated at t = 0** (`causal_condition`). Text tokens read the modulation of
timestep zero, picture tokens that of the step, so the runtime carries the two as separate float32
residual streams and joins them only for the projections. The prompt's half of every block is
then the same at every step; the reference caches its keys and values after step one and this
runtime recomputes them, which costs a few percent at 1024 x 1024.

**No shift, and gates through tanh.** The modulation is scale and gate only; the final layer's is
a scale alone. The blocks' LayerNorms have no weights; `txt_in`'s RMSNorm is zero centred and
folded by the exporter; the per-head q/k norms store the multiplier and are not folded.

**Positions.** The prompt's n-th token is at `(n, n, n)`. The picture sits at frame `L` (the
prompt's length) with rows `-(H - H/2) .. H/2` and columns likewise -- centred, so the picture's
positions do not depend on the prompt's length.

**The timestep the model sees is not the sigma.** The pipeline casts `1000 * sigma` to bfloat16,
divides by a thousand in bfloat16, and the model multiplies by a thousand inside the sinusoid.
Near 1000 bfloat16 counts in fours, so at the sinusoid's fastest frequency that is up to two
radians. The runtime rounds the same way (`model_timestep` in `pipeline.rs`), because that is the
question the model was sampled with.

## The text side

```
<|im_start|>system
Comprehend and analyze the provided prompt.<|im_end|>
<|im_start|>user
{prompt}<|im_end|>
<|im_start|>assistant
```

Tokenized as one string. The encoder reads all of it; the system turn's states are dropped
afterwards and everything after it -- the prompt and the closing tokens -- is what the denoiser
reads. What is read is the **last layer's output before the final norm**: diffusers goes out of
its way to strip that norm on transformers 5, where `hidden_states[-1]` became normalized. The
runtime reuses Krea 2's Qwen3-VL text tower, tapped at `[36]` instead of twelve layers.

## The schedule

`sigmas = linspace(1, 1/N, N)`, bent by `exp(mu) / (exp(mu) + 1/t - 1)` with `mu` interpolated
linearly from 0.5 at 256 image tokens to 0.9 at 8192 (and extrapolated past it), then stretched so
the last step starts at `shift_terminal = 0.02`, then a final zero. The package stores the line
and the terminal rather than a shift, since the shift depends on the size.

Defaults are the card's: 40 steps, no guidance (`true_cfg_scale = 1`). A guidance above one runs
the second pass on the negative prompt, as the reference does.

## The autoencoder

Wan 2.2's, specialized to one frame: every convolution is already 2-D in the checkpoint. Base 96
in, 144 out, `dim_mult (1, 2, 4, 8, 8)`, 16x spatial, 64 latent channels, 4 image channels (RGBA).
What it has that the Qwen-Image VAE does not is a **shortcut around each upsampling stage**,
`DupUp3D`: repeat the input channels, fold them out into time and space, and keep the last time
slice (`first_chunk`). For one frame that is a fixed channel gather and a pixel shuffle -- a
nearest doubling for the first two stages, the odd channels for the third, a row-parity pick for
the fourth. Whether a stage was temporal changes which channels are read, and no exported weight
records it, so the manifest carries `vae_temporal_upsample`.

The runtime's `generate` composites RGBA over white, since everything downstream is RGB;
`generate_rgba` hands back all four channels.

## How close it is

Against diffusers' `QwenImage21Pipeline` on the same weights, 256 x 256, ten steps, float16 (or
fp8) here and bfloat16 there, relative RMSE (`waifu/tests/qwen_image.rs`):

| | float16 | fp8 |
|---|---|---|
| encoder, last layer before the norm | 9.0e-3 from float32; the bfloat16 reference is 7.3e-2 from it | 1.21e-1 from float32 |
| velocity at step six | 7.9e-3 | 2.15e-2 |
| decoder, RGBA, against float32 | 7.5e-4 | 7.5e-4 -- it is not quantized |
| the whole ten-step walk | 1.35e-2 | 1.19e-1 |
| schedule | the reference's eleven sigmas to 1e-5 | the same |

The encoder is held to float32 rather than to the bfloat16 reference because the reference is the
less exact of the two: the last layer before the norm carries activations in the hundreds, and
bfloat16 loses most of the first token the denoiser reads. float16 is the more exact of the two;
fp8 is not -- at 1.21e-1 it is well past the reference's own 7.3e-2 rounding, unlike every other
fp8 package here, where the float type is still most of the number. `-fp8` here still writes one
scale per row rather than one for the whole tensor (see "One scale for the whole tensor" in
`docs/fp8.md`) -- the same format Krea 2's fp8 package started with before moving off it -- and
this encoder's weights are evidently less forgiving of it than Krea 2's are. Nothing crashes or
NaNs; the picture is a real, working one, just a more noticeably different one than this runtime's
other fp8 packages draw.

## Memory

Measured on a 16 GB RTX 5060 Ti with 62 GB of RAM, fp16 package, `Residency::LowVram`, 1024 x
1024:

| | |
|---|---|
| card, reading the prompt | 1.7 GB |
| card, denoising and decoding | 3.2 GB at peak |
| host, the package mapped in | 27 GB resident, all of it page cache |

The weights stream through the card a layer at a time, so the card holds activations and
little else. What decides the speed is whether the 30 GB package stays in the page cache between
steps: with room for it, each step reads memory; without, each step reads the disk again, which
on this machine's disk is about 200 MB/s and most of the time a step takes. A machine with 48 GB
or more of RAM keeps it; `-fp8` halves the package -- 30 GB to 16 GB, the same ratio as its size on
disk, since `Residency::LowVram` moves whichever package it was given the same way.

## ControlNet

[`alibaba-pai/Qwen-Image-2.1-Fun-Controlnet-Union`](https://huggingface.co/alibaba-pai/Qwen-Image-2.1-Fun-Controlnet-Union)
steers the denoiser with a structure picture -- canny, depth, grayscale, HED, lineart, MLSD, pose
or scribble; one checkpoint, and nothing says which kind it was handed. It was trained with
VideoX-Fun, which is the reference here (`qwenimage21_transformer2d_control.py`,
`pipeline_qwenimage21_control.py`); diffusers has no class for it.

It is a package of its own, `qwen-image-2.1-controlnet`, 7.5 GB in fp16, read beside the model's
with `QwenImage::from_manifests`. Its tensors are named `qwen_image.dit.control.*` and it has to be
exported with the same `-fp8` as the model, because one graph reads both.

**What it is.** A second chain of sixteen of the denoiser's own blocks -- the same attention,
modulation, rope and block-causal split -- with a projection in and one out of each:

```
c = control_img_in(conditioning)          # (N, 129) -> (N, 4096) at the picture's positions,
                                          # zero at the prompt's
c = before_proj(c) + x                    # x: the denoiser's input to its first block
for k in 0..16:
    c = control_block[k](c)
    skip[k] = after_proj[k](c)

for i in 0..32:                           # the denoiser, afterwards
    x = block[i](x)
    if i is even: x = x + skip[i / 2] * scale
```

The chain reads the denoiser's stream once, at the start, and never its middle, so it runs to its
end first. The skips cover the whole sequence, prompt included: the prompt's half of the chain
starts as the prompt plus `before_proj`'s bias. `control_img_in` reads 129 channels, which an FP8
multiply cannot (it reads sixteen at a time), so it stays at full width in every package.

**The conditioning** is 129 channels on the latent grid: the control picture's latent, a mask
(one where the picture is kept), and the kept picture's latent. For a picture drawn from noise the
mask and the last latent are zeros. The latents are the VAE encoder's mode, normalized by the
package's `latents_mean` and `latents_std` like the sampler's. The picture is RGB in `-1..=1`
given an opaque alpha, since the encoder reads four channels.

**The encoder** is new here: the decoder was all a picture drawn from noise needed. It is Wan
2.2's, one frame. Each halving stage's shortcut, `AvgDown3D`, pads time in front to its temporal
factor before it averages, so on a temporal stage every even output channel is the mean of the
padded frame -- zero -- and every odd one the 2 x 2 mean of an input channel. The halving
convolution pads right and bottom only, like SDXL's.

## Not done

* Image editing and reference images. The model reads a condition picture both through the VAE
  and through Qwen3-VL's vision tower, which is not exported.
* The prefix KV cache.
* The ControlNet's inpainting (a mask and a kept picture), the webui, and the hub. The runtime
  takes a preprocessed control picture; making one out of a photograph -- a pose detector, a
  depth estimator -- is the caller's.

## Exporting and testing

```bash
hf download Qwen/Qwen-Image-2.1
.venv/bin/python tools/qwen_image_exporter.py -model <snapshot> -output models/qwen-image-2.1.safetensors
# reference tensors: needs a diffusers with QwenImage21Pipeline, transformers >= 5.17,
# torchvision and a CUDA torch; offloads layer by layer, so 16 GB is enough
python tools/qwen_image_exporter.py -test_only -model <snapshot> \
    -test_output models/qwen-image-2.1_test.safetensors
```

Add `-fp8` to the first command for the quantized package; it reads the same `-model` directory,
so both can be exported from one download.

The ControlNet, and its reference off VideoX-Fun (which needs VideoX-Fun on `PYTHONPATH`, and
`accelerate` beside the rest):

```bash
hf download alibaba-pai/Qwen-Image-2.1-Fun-Controlnet-Union
.venv/bin/python tools/qwen_image_exporter.py -model <snapshot> -controlnet <checkpoint> \
    -output models/qwen-image-2.1-controlnet.safetensors
python tools/qwen_image_exporter.py -test_only -model <snapshot> -controlnet <checkpoint> \
    -control_image <VideoX-Fun>/asset/pose.jpg \
    -test_output models/qwen-image-2.1-controlnet_test.safetensors
```

## Licensing

`license: other`, `license_name: qwen-research`, and the Qwen RESEARCH LICENSE AGREEMENT is
**non-commercial (research and evaluation) use only** -- unlike Krea 2's, there is no revenue
threshold that opens it up. Section 3 asks for, and this gets:

| | |
|---|---|
| §3.a a copy of the agreement | `LICENSE` ships in the package |
| §3.b modified files carry a notice | `NOTICE` lists every change: the renaming, the fusing, the norm folding, the narrowing, the quantization |
| §3.c the attribution notice, verbatim | `NOTICE` |
| §4.b "Built with Qwen" in the product documentation | said in `NOTICE` and on the card |
| §4.c not "Qwen" as the primary name | the repository id keeps this project's `libwaifu-` prefix |

The converted packages are published at
[ling0322/libwaifu-qwen-image-2.1](https://huggingface.co/ling0322/libwaifu-qwen-image-2.1) and
on ModelScope under the same name.
