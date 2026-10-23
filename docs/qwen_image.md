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
fp8 package here, where the float type is still most of the number. These numbers are for the
once-published `qwen-image:2.1-fp8` (since withdrawn for the Turbo package), which was exported
with one scale per row. `-fp8` now writes one scale for the whole tensor (`fp8_tensor_scale`, see "One scale for the whole tensor" in
`docs/fp8.md`) like Krea 2's, so a re-export will not reproduce them; the Turbo package is the
first published in that format. This encoder's weights were evidently less forgiving of the
per-row format than Krea 2's are. Nothing crashes or
NaNs; the picture is a real, working one, just a more noticeably different one than this runtime's
other fp8 packages draw.

## Memory

Measured on a 16 GB RTX 5060 Ti with 62 GB of RAM, fp16 package, `device_weight_bytes` 0 --
nothing kept on the card -- 1024 x 1024:

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
disk, since `Weights` moves whichever package it was given the same way.

## Turbo

`Qwen/Qwen-Image-2.1-Turbo` is the same architecture, distilled to **eight steps**. The weights
are laid out exactly as above; what differs is the schedule, which the checkpoint does not bend
but writes out in `model_index.json` as `sample_sigmas` (`1, 0.978453, .., 0.414568`), with a plain
scheduler (no dynamic shift, no `shift_terminal`). diffusers uses those levels as they stand and
appends the zero. The exporter copies them into the manifest as `sampler_sigmas`; when it is
present `QwenImage::sampler` builds `FlowSampler::from_sigmas` from it and **ignores the step
count and the picture's size**. Only the fp8 package is published, as `qwen-image:2.1-turbo-fp8`
in [ling0322/libwaifu-qwen-image-2.1-turbo](https://huggingface.co/ling0322/libwaifu-qwen-image-2.1-turbo).
Its VAE is a smaller file than the base release's (0.7 GB against 1.4 GB); it exports the same.

Against the reference, 256 x 256, eight steps (`waifu/tests/qwen_image.rs`,
`turbo_fp8_matches_the_reference`): velocity at step five 2.1e-2, decoder 8.0e-4, the whole walk
1.8e-1 -- the fp8 encoder again, with fewer steps to average it out.

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

## Pictures in the prompt

The model edits a picture, or draws from up to ten of them, by reading them as part of the prompt
(`QwenImage21Pipeline(image=...)`). Each picture is read twice -- by Qwen3-VL's vision tower, whose
rows the encoder reads in the picture's slots, and by the autoencoder's encoder, whose latent the
denoiser reads where those slots were -- and the picture drawn comes after all of it. The runtime's
door is `QwenImage::generate_with_pictures`; the C API's is a `<|name|>` in the prompt (see
`docs/ffi.md`), which the engine numbers `<image1>`, `<image2>` in the order the prompt names them.

**The template.** The user's turn opens with the pictures, then the prompt:

```
<|im_start|>user
<image1><|vision_start|><|image_pad|> x slots<|vision_end|> <image2><|vision_start|>...<|vision_end|>{prompt}<|im_end|>
```

`<imageN>` is plain text, not a token. A picture has one `<|image_pad|>` per 32 x 32 pixels.

**The size a picture is read at** is a megapixel at its own aspect ratio, each side rounded to a
multiple of 32 (`calculate_dimensions`, Python's rounding) -- whatever size the picture arrives at
and whatever size is being drawn (`QwenImage::picture_size`). The resize is Lanczos, the
reference's PIL one and the engine's `image` crate one. An RGBA picture is laid over white for the
vision tower; the autoencoder reads all four channels, and an RGB one is given an opaque alpha.

**The vision tower** (`vision.rs`) is `Qwen3VLVisionModel`: 27 blocks of 1152 over 16-pixel
patches, laid out a 2 x 2 block at a time so that four neighbours merge by a reshape into one
4096-wide row. Its patch embedding is a Conv3d two frames deep, because the tower also reads
video; a picture is two identical frames, which is one frame through the sum of the kernel's two
time slices. A learned 48 x 48 table of positions is interpolated bilinearly, corners aligned, to
the picture's grid; the attention rotates a quarter of each head by the row and a quarter by the
column. Blocks 8, 16 and 24 are merged a second way (fold four, then normalize: DeepStack). The
package has carried the tower, never quantized, since the Turbo release; a package without it
reads words only, and says so.

**The encoder** reads each slot's vision row in place of the `<|image_pad|>` embedding, and adds
the three DeepStack rows at the slots after its first three layers. Its rotary positions are
Qwen3-VL's interleaved mRoPE: frequency `i` of a head turns with the row where `i % 3 == 1` and
with the column where `i % 3 == 2` (twenty each), and with the frame otherwise. A word is at
`(n, n, n)`; a picture's slots are a grid at `(s, s + row, s + column)`, and the words after it
start `max(rows, columns)` on from `s` (`get_rope_index`). For a prompt of words alone this is the
table it always was.

**The denoiser** drops the encoder's states at the slots and puts each picture's latent there --
four latent tokens per slot, through the same `img_in` -- so the prompt it reads is runs of words
and pictures (`Segment`, `Pictures`). All of it is modulated at timestep zero, the pictures with
the words. The attention is causal between runs and whole within a picture: each run reads every
key up to its own end, causally for words and without a mask for a picture, which is exact
because flint's causal mask is aligned to the bottom right. A picture's positions are a centred
block at the frame the words reached, and the frame after it is `max(H, W)` further on, in latent
pixels this time. The graph is compiled per layout and the last four are kept: a guided run
alternates between the prompt's and the negative prompt's, which names the same pictures.

**How close it is.** Measured on Metal (Apple M5 Pro) with the published `qwen-image:2.1-turbo-fp8`
package, the 1024 x 704 cat of diffusers' documentation images read at 608 x 416, against
transformers' and diffusers' own modules at float32 on the CPU running the *same* weights -- the
fp8 ones dequantized -- so that what is measured is the runtime and not the quantization:

| | relative RMSE |
|---|---|
| the template, 284 ids | identical to `Qwen3VLProcessor`'s |
| vision rows | 3.8e-2 (float16 torch: 1.9e-2; the bfloat16 the pipeline runs: 1.9e-1) |
| DeepStack rows after blocks 8, 16, 24 | 3.8e-3, 7.8e-3, 1.5e-2 |
| encoder, the words' states after the picture | 7.8e-3 |
| velocity, step one and step five of a Turbo walk | 1.2e-3 and 7.6e-4 -- zeroing the picture's latent moves the reference 7.0e-1 |

The model test (`waifu/tests/qwen_image.rs`, `check_pictures`) asks the same questions of a real
`QwenImage21Pipeline` run on CUDA, with the words-only bounds of each package; its fixtures need
`-test_picture`.

## Not done

* Pictures in the CLI's `draw` command and the Windows app: the C API takes them and the Mac
  app asks for them (Add Picture, beside Generate), but these screens do not yet.
* The prefix KV cache, which with a megapixel picture in the prompt is most of the prompt.
* The ControlNet's inpainting (a mask and a kept picture), the CLI, and the hub. The runtime
  takes a preprocessed control picture; making one out of a photograph -- a pose detector, a
  depth estimator -- is the caller's.

## Exporting and testing

```bash
hf download Qwen/Qwen-Image-2.1
.venv/bin/python tools/qwen_image_exporter.py -model <snapshot> -output models/qwen-image-2.1.safetensors
# reference tensors: needs a diffusers with QwenImage21Pipeline, transformers >= 5.17,
# torchvision and a CUDA torch; offloads layer by layer, so 16 GB is enough. The photograph is
# what the reference edits, for the cases of a prompt that names a picture.
python tools/qwen_image_exporter.py -test_only -model <snapshot> \
    -test_picture <a photograph> -test_output models/qwen-image-2.1_test.safetensors
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

The converted package is published at
[ling0322/libwaifu-qwen-image-2.1-turbo](https://huggingface.co/ling0322/libwaifu-qwen-image-2.1-turbo)
and on ModelScope under the same name. The base release's packages,
`qwen-image:2.1` and `qwen-image:2.1-fp8`, are no longer in the catalog.
