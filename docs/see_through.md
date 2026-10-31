# See-through: img2layers

Notes on running [See-through](https://github.com/shitagaki-lab/see-through) (SIGGRAPH 2026) in
libwaifu. It takes an anime illustration apart into the layers it was drawn in -- up to
twenty-four of them, hair, face, eyes, clothing, what is held -- with what each hides drawn in, and
stacks them by depth into a Photoshop document. The task is `img2layers`; the code is
`waifu/src/see_through/`, and what it does is upstream's `inference/scripts/inference_psd.py`.

## What is published

| repository | licence | what |
|---|---|---|
| `layerdifforg/seethroughv0.0.2_layerdiff3d` | OpenRAIL++ | the layers model: an Animagine XL 4.0 U-Net, both CLIPs, SDXL's VAE, LayerDiffuse's alpha decoder |
| `layerdifforg/seethroughv0.0.1_marigold` | OpenRAIL++ | the depth model: Marigold on Stable Diffusion 2 |
| `shitagaki-lab/see-through` | Apache-2.0 | the code this is read off |

Both U-Nets are published in bfloat16. Neither is gated.

The package below is published as `ling0322/libwaifu-see-through` on Hugging Face and on
ModelScope, with both upstream licences and a `NOTICE` of what was changed. Its reference tensors
are `see-through_test.safetensors` in `ling0322/libwaifu_test_data`.

## The package

`tools/see_through_exporter.py` writes one package of both models, 11.4 GB in float16 with the
two autoencoders in float32, in parts under 2 GB:

```bash
uv venv -p 3.12 .venv-see-through
VIRTUAL_ENV=.venv-see-through uv pip install -r tools/requirements-see-through.txt --torch-backend cu128
git clone https://github.com/shitagaki-lab/see-through ~/.cache/libwaifu/src/see-through

PYTHONPATH=~/.cache/libwaifu/src/see-through/common:tools \
    .venv-see-through/bin/python tools/see_through_exporter.py \
    -layers layerdifforg/seethroughv0.0.2_layerdiff3d \
    -depth layerdifforg/seethroughv0.0.1_marigold \
    -output models/see-through.safetensors -part-size 2GB
```

| namespace | what |
|---|---|
| `see_through.layers.unet` | the layers U-Net, 4.1 B parameters |
| `see_through.layers.vae`, `.vae.encoder` | SDXL's autoencoder, both halves |
| `see_through.layers.alpha` | the alpha decoder |
| `see_through.layers.{body,head}.{context,pooled}` | what the two CLIPs make of each layer's name, group embeddings folded in |
| `see_through.depth.unet` | the depth U-Net |
| `see_through.depth.vae`, `.vae.encoder` | Stable Diffusion's autoencoder |
| `see_through.depth.context` | the empty prompt, `(1, 2, 1024)` |

**No text encoders.** Neither model reads a prompt anyone writes. The layers model is conditioned
on the name of each layer -- `"front hair"`, `"irides"` -- and the depth model on the empty prompt,
so what the encoders make of those is computed by the exporter and carried instead. The layers
model's group embeddings, which tell it which of its two passes it is on, are added to those same
tensors there (`e + group_embeds2[g](e)`, `t + group_embeds[g](t)`), since that is all the U-Net
ever does with them.

## The two U-Nets

Both are `UNetFrameConditionModel`: diffusers' U-Net with a block after transformer layers that
attends across the batch instead of across the picture. The batch is one picture's layers, so every
pixel of every layer sees the same pixel of the others -- the hair knows where the face is.

```
x  = (N, H*W, C)                    # N layers
x' = x.transpose(0, 1)              # (H*W, N, C): each pixel a batch, the layers its sequence
x' = x' + ff_in(norm_in(x'))        # GEGLU, 4C inside
x' = x' + attn1(norm1(x'))          # self attention over N, heads of 64, no positions
y  = ff(norm3(x'))                  # GEGLU; no residual -- the caller adds it
x  = x + y.transpose(0, 1)
```

One follows every layer of a transformer up to two deep, every second from three: SDXL's
ten-layer transformers carry five and its two-layer ones two, forty blocks in all.

The U-Net is `waifu/src/see_through/unet.rs`, a copy of `waifu/src/sdxl/unet.rs` with these
blocks added and SDXL left untouched: the two are different models, so their code is kept apart,
and each file names the other because a fix to a shared block likely belongs in both.

| | layers | depth |
|---|---|---|
| layout | SDXL, `[320, 640, 1280]` | Stable Diffusion 2, `[320, 640, 1280, 1280]` |
| in | 8: noisy latent, picture latent | 12: picture latent, layer latent, noisy depth |
| conditioned on | layer names, 2048 wide, and SDXL's added embedding | the empty prompt, 1024 wide; no added embedding |
| sampler | DPM-Solver++ 2M SDE, midpoint, 30 leading steps | DDIM, v-prediction, zero terminal SNR, 4 trailing steps |
| frames | 13 body, then 11 head | 19 slots and the full page |

The depth layout needed three things SDXL's did not: an input wider than the output, a middle
transformer as deep as it likes (one) where the narrowest level has none, and no added embedding.
They are `in_channels`, `mid_transformer_layers` and `projection_class_embeddings_input_dim: 0`.

## How a picture is taken apart

1. **Square.** The picture is centred in a square of transparent black and scaled to 1280 (area
   to shrink, linear to grow). Its colour, alpha dropped, is encoded -- the mean, not a draw.
2. **Body.** Thirteen layers, every one starting from the same noise, thirty steps; each step adds
   fresh noise per layer. Each latent is decoded by the VAE, then the alpha decoder reads that RGB
   and the latent and says `[alpha, R, G, B]`. Alpha is multiplied by the picture's own.
3. **Head.** The head layer's box, alpha above 15, is mapped back to the original picture, widened
   by a fifth either side unless it is already half of it, cut out, squared and scaled to 1280, and
   drawn as eleven layers of its own. They are scaled back and put where the head was.
4. **Depth.** Nineteen slots: each layer with alpha below 15 cleared, `hair` the two hairs and
   `eyes` the four parts of an eye laid over each other, and the page with its alpha the layers'
   summed. Each is scaled to 768, its transparent parts filled from around them by Lvmin's pyramid
   (`pad_rgb`), and encoded. Four DDIM steps; each map is the mean of the three channels decoded.
   A composite's parts take its depth where they show and the median of it where a nearer part
   covers them.
5. **Parts.** Each layer is cut to where alpha is above 10, with fewer than five such pixels in the
   top left nine tenths taken for nothing. With `split_pairs`, handwear, the four eye parts and the
   ears are split into their two largest pieces, `-r` the one further left. The nose and mouth
   take their colour from the picture. Each part's depth is its median; nose and mouth go in front
   of the face. Farthest first is the document's order.

The canvas the parts are on is the 1280 square, as upstream's is; `Decomposition::picture` says
where in it the picture is.

The document carries each part's depth too: a layer of it named `depth/<tag>`, gray (0 near, 255
far) under the part's own alpha, in a group `depth` that is hidden and closed and comes after every
part -- at the top of Photoshop's layer panel, out of the way, and out of the merged picture. It is
what the depth document (`to_depth_psd`) holds, in RGB, so that one file is enough to stratify.

## Stratification

One depth a part cannot say that it is in front of another part in one place and behind it in
another: a V collar whose back is behind the neck and whose flaps are in front, drawn whole as one
`topwear`, is laid over the whole neck and covers what of it the V shows. The paper calls this the
"sandwich" (section 4.2.2) and stratifies: each part's pixel depth, where it is drawn, is cut in two
by k-means (k = 2), and the near and far clusters become layers of their own. Its PSD export does
this to hair, hands, tops and bottoms. Upstream ships it only as a step to run by hand
(`heuristic_partseg.py seg_wdepth`, `cluster_inpaint_part` in `common/utils/torchcv.py`).

Here it is done to a document after img2layers wrote it, as upstream's `seg_wdepth` is, and not
as part of taking a picture apart: what img2layers writes is the model's, and what the layers are
cut into is for what they are used for. `see_through::stratify` (`waifu_stratify` in the C API,
and the `see_through_stratify` tool below) takes a document's layers with their depth and does it
to `front hair`, `back hair`, `handwear` and its `-l`/`-r`, `topwear` and `bottomwear`
(`waifu/src/see_through/stratify.rs`):

- `<tag>-0`, the near cluster: the part's colour, its alpha under the cluster's mask feathered by
  a 3 x 3 Gaussian. It takes the part's place in the document.
- `<tag>-1`, the far one: the whole part, its outline the part's own, with the near cluster's
  place, grown by a 7 x 7 ellipse, filled in with colour from around it, and its depth there the
  median of the rest. It lies under `-0`, which covers most of it, and goes down from the part's
  place past every layer right under it that is nearer than its own median depth.
- The rest of the document's order is kept, where upstream sorts every part by depth again: a
  document may have been put in order by hand.

Where upstream fills the far layer with LaMa, a model of its own that is not here, this takes
upstream's other branch, which fills with `cv2.inpaint`, and fills by Lvmin's pyramid (`pad_rgb`)
instead. What it fills lies under the near layer, so the document looks the same either way; it
shows only when the near layer is hidden or moved. The rest that differs from upstream:

- The k-means runs exactly, on a histogram of the 256 depths from both ends, where upstream's is
  seeded at random on 2000 samples.
- Specks are not clustered. A model's layer is scattered with faint pieces of a few pixels all
  over the canvas, at depths far from the part's own; clustered with it, they are the far half,
  and the part comes out as itself and a layer of specks. Pieces under 40 pixels, Anime2.5DRig's
  floor for the same specks, stay with the far layer and decide nothing.
- A side under a twentieth of what is clustered leaves the part whole. On hair and hands the far
  side was otherwise a hundredth or so: depth noise along an edge, cut off for nothing.
- The near layer keeps the part's own soft edge under its mask, and the far one the part's own
  outline. Upstream's near layer is its mask alone, and its far layer's alpha is filled in like
  its colour, which where the near cluster reaches the part's edge brings the transparency around
  the part in, and leaves a faint ghost of it.

From a terminal, a document is stratified into another, its depth kept in its group; one written
before documents kept their depth is read with its depth document:

```bash
cargo run --release -p waifu-tools --bin see_through_stratify -- \
    picture.psd stratified.psd [depth.psd]
```

On a Hatsune Miku bust with a V collar it cut only `topwear`, its far eighth -- the collar's back,
among the rest -- going behind the neck. Flattened against the picture it was taken from, the mean
error over the collar halved (16.3 to 7.6 levels of 255), and the whole picture's went from 4.93 to
4.70. It cuts by depth alone, which the paper says is not always a clean edge (section 6); one
picture is not a test of it.

## What is left out

- **Upstream's dead branches.** `further_extr` has an `eyes` split into eyes and brows and a `hair`
  split by k-means and LaMa inpainting; both look for parts by names the v3 model does not draw.
- **Upstream's unseeded randomness.** It draws the picture's latent from the VAE's distribution
  and its SDE noise from the global CUDA generator with no seed, so the same picture gives a
  different document each time. Here both come from the seed.
- **Precision.** Upstream runs everything in bfloat16. Here the U-Nets and the alpha decoder are
  float16 -- the alpha decoder's activations peak near 100 on real layers, and it lands within a
  level of 255 of float32 -- and the two autoencoders float32, since SDXL's overflows float16.

## Speed and memory

One 928 x 1232 illustration at the defaults, on an RTX 5060 Ti 16 GB:

| | libwaifu, no FlashAttention | libwaifu, `WITH_FLASH_ATTN=ON` | upstream, PyTorch |
|---|---|---|---|
| body, 30 steps | 308 s | 190 s | 153 s |
| head, 30 steps | 270 s | 164 s | 180 s |
| depth, 4 steps | 12 s | 6 s | 7 s |
| whole run | 722 s | 472 s | 398 s, loading included |

Both models are read onto the device at once, 11.4 GB. Decoding the thirteen body layers at 1280
peaks at 16.9 GB, past what a 16 GB card holds, and lets about 2.4 GB of weights go back to the host
for the rest of the run; upstream peaks at 15.1 GB, but loads the depth model only when it gets to
it. Most of what is left between the two is the decoding, whose autoencoder runs in float32 here.

A replay of upstream's recorded 1280 run -- its noise handed to the runtime step by step -- stays
with it to 2.6 % (body) and 3.9 % (head) of the latent at the last step, the drift of float16
against bfloat16, and gives the same layers.

## Tests

`waifu/tests/see_through.rs` is one test, because the package is eleven gigabytes. Its record is
`tools/see_through_reference.py`, which runs upstream's own pipelines on upstream's own
`test_image.png` at 512 and 384 and keeps what goes in and out of every model: a latent off the
real trajectory, the layers the model really drew.

**The bar is upstream's own precision.** Every model is run again on those inputs in float32
(TF32 off; the layers U-Net on the CPU) and in bfloat16, which upstream runs in. The test passes
where the runtime is no further from float32 than upstream's bfloat16 is:

```
error(runtime, torch float32)  <=  error(torch bfloat16, torch float32)
```

for the layers U-Net, the sampler's two orders, the picture's encoding, the autoencoder and the
alpha decoder, the depth model's slot encodings, its U-Net and its depth maps. The record keeps the
float32 answer as `<name>.f32` and the bound as `<name>.bf16_error`. As measured, the runtime is
within a seventh of that bound for both U-Nets and far inside it for the float32 autoencoders:

| | runtime vs float32 | bfloat16 vs float32 |
|---|---|---|
| layers U-Net, one step | 0.035 % | 0.26 % |
| depth U-Net, one step | 0.041 % | 0.25 % |
| autoencoder decode | 0.0004 % | 2.4 % |
| autoencoder + alpha decoder | 0.058 % | 0.54 % |

That bound is the floor, and for what runs in float32 here it is thousands of times too loose to
catch a float32 path going wrong, so those are held to float32 as well: the picture's latent within
1e-3, and each decoded layer, each slot's latent and each depth map on its own within 1e-4, 2e-3
and 1e-5 -- each, because one error over all twenty slots is the full page's and could hide an
empty slot gone wrong. And every layer `decode_layer` finishes, the picture's alpha multiplied in
and in eight bits, is compared with upstream's own: within 3 levels of 255 in alpha and 4 in red.

```bash
PYTHONPATH=~/.cache/libwaifu/src/see-through/common:tools \
    .venv-see-through/bin/python tools/see_through_reference.py \
    -layers ~/.cache/libwaifu/hf/seethroughv0.0.2_layerdiff3d \
    -depth ~/.cache/libwaifu/hf/seethroughv0.0.1_marigold \
    -output models/see-through_test.safetensors

cargo test --release --manifest-path waifu/Cargo.toml --no-fail-fast \
    --test see_through -- --ignored --test-threads=1
```
