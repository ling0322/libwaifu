# Anima

Notes towards running Anima in libwaifu. The shapes were read off the published tensor headers
over range requests, the conv folding was proved with a script that still runs, and the behaviour
that shapes alone do not show -- the text path, the position layout, the latent scaling -- came
from the reference implementations in ComfyUI (`comfy/ldm/anima/model.py`,
`comfy/ldm/cosmos/predict2.py`, `comfy/text_encoders/anima.py`) and diffusers.

Anima is **not an SDXL fine tune**, so none of `waifu/src/sdxl/` applies to it and the
`add-sdxl-model` skill does not describe this job. It is a diffusion transformer derived from
`nvidia/Cosmos-Predict2-2B-Text2Image`, with Qwen3-0.6B as the text encoder and the Qwen-Image
VAE. Three separate weight files, three architectures, none of them one we already have.

## What is published

`circlestone-labs/Anima`, under `split_files/`:

| file | size | what |
|---|---|---|
| `diffusion_models/anima-aesthetic-v1.1.safetensors` | 4.18 GB | the v1.1 denoiser, 30-50 steps at CFG 4-5 |
| `diffusion_models/anima-turbo-v1.1.safetensors` | 4.18 GB | distilled, **CFG 1 and 8-12 steps** |
| `text_encoders/qwen_3_06b_base.safetensors` | 1.19 GB | Qwen3-0.6B base, shared by every version |
| `vae/qwen_image_vae.safetensors` | 0.25 GB | the Qwen-Image VAE, shared |

Base and the v1.0 releases sit beside them; the repository holds nine denoisers in all and one
copy each of the encoder and VAE. All bf16.

Anima-2.9B is a third-party layer expansion, `Gazingstars123/Anima-2.9B`, one 5.84 GB file whose
`config.json` says `num_layers: 40` against Anima's 28. `expand_manifest.json` records how: twelve
blocks deep-copied from their neighbours and inserted at fixed positions, with
`adaln_modulation_*.2`, both `output_proj` and `mlp.layer2` zeroed so the model starts out
identical to the 28-block one. **Architecturally it is the same model with a different block
count** (verified: same tensor names, same shapes). One implementation covers both; read the
count from the package rather than baking in 28. Note its tensors are prefixed `net.` where
Anima's are `model.diffusion_model.` -- an exporter has to accept both.

## The denoiser (verified from the v1.1 tensor header)

28 blocks, hidden 2048, 685 tensors in the file. Nothing in the denoiser proper carries a bias;
the only biases anywhere are inside the adapter.

```
x_embedder.proj.1.weight            [2048, 68]      patchify
t_embedder.1.linear_1.weight        [2048, 2048]
t_embedder.1.linear_2.weight        [6144, 2048]
t_embedding_norm.weight             [2048]          RMSNorm
llm_adapter.embed.weight            [32128, 1024]   plus six blocks of its own, below
llm_adapter.norm.weight             [1024]
llm_adapter.out_proj.weight         [1024, 1024]
llm_adapter.out_proj.bias           [1024]
final_layer.adaln_modulation.1      [256, 2048]
final_layer.adaln_modulation.2      [4096, 256]     2 x 2048: shift, scale
final_layer.linear.weight           [64, 2048]      unpatchify
```

Count them carefully. A pattern of `blocks.<n>.` matches the adapter's blocks as well as the
denoiser's, and folding the two together is an easy way to get the depth wrong. Against exact
prefixes it comes to 28 x 20 + 6 x 19 + 11 = 685, which is the file.

and per block:

```
self_attn.{q,k,v}_proj              [2048, 2048]    16 heads x 128, plain MHA
self_attn.{q,k}_norm                [128]           QK RMSNorm, per head dim
self_attn.output_proj               [2048, 2048]
cross_attn.q_proj                   [2048, 2048]
cross_attn.{k,v}_proj               [2048, 1024]    1024 = Qwen3-0.6B hidden
cross_attn.{q,k}_norm               [128]
cross_attn.output_proj              [2048, 2048]
mlp.layer1                          [8192, 2048]    plain 4x MLP, not gated
mlp.layer2                          [2048, 8192]
adaln_modulation_{self_attn,cross_attn,mlp}.1  [256, 2048]
adaln_modulation_{self_attn,cross_attn,mlp}.2  [6144, 256]   3 x 2048: shift, scale, gate
```

Two things worth naming. The modulation is **AdaLN-LoRA**: 2048 -> 256 -> 6144 rather than a
direct 2048 -> 6144, three of them per block and a two-term one at the end. And `mlp.layer1`
widens to 8192 which `mlp.layer2` reads whole, so it is an ungated 4x MLP -- not the SwiGLU the
Qwen3 encoder uses. Getting that backwards costs nothing at load time and produces noise.

`final_layer.linear` emits 64 = 2 x 2 x 16, so patches are 2x2 over a 16-channel latent. The
patchify input of 68 is `(16 + 1) x 2 x 2`: Cosmos carries `concat_padding_mask=True`, which
appends one padding-mask channel to the latent before patchifying. Every image this runtime draws
is unpadded, so that channel is zeros -- but it has to be *there*, and it is not reconstructed on
the way out, which is why the output is 64 and the input 68.

## How a block actually runs

Shapes do not give any of this and most of it is a silent error. Read off
`comfy/ldm/cosmos/predict2.py`.

The timestep path is the one that reads backwards. `Timesteps` builds a sinusoid --
`exponent = -log(10000) * arange(half) / half`, then `cat([cos, sin])`, cosine first -- and hands
it to `TimestepEmbedding`, whose two linears produce **only the AdaLN-LoRA term**:

```python
emb = self.linear_2(self.activation(self.linear_1(sample)))
if self.use_adaln_lora:
    adaln_lora_B_T_3D = emb       # 6144 wide
    emb_B_T_D = sample            # the raw sinusoid, not the projection
```

So `t_embedder`'s linears never feed the modulation directly. What every block modulates on is
`t_embedding_norm(sinusoid)` -- the RMSNorm applied to the *input* of those linears -- and the
6144-wide LoRA rides alongside. Reading `emb` as the projection is the obvious mistake and it is
wrong.

Each block then, three times over:

```python
shift, scale, gate = (adaln_modulation_<part>(emb) + adaln_lora_B_T_3D).chunk(3, dim=-1)
normalized     = layer_norm(x) * (1 + scale) + shift
x              = x + gate * sublayer(normalized)          # torch.addcmul
```

The same `adaln_lora` is added to all three. `adaln_modulation_*` is
`Sequential(SiLU, Linear(2048, 256), Linear(256, 6144))`, which is why the checkpoint numbers
them `.1` and `.2`. The final layer is the same with `chunk(2)` and
`adaln_lora[:, :, : 2 * hidden]`, shift and scale but no gate.

The norms are **`LayerNorm(eps=1e-6, elementwise_affine=False)`** -- mean subtracting, and
carrying no weights, which is why no block has a norm tensor. They are not the RMSNorm used
everywhere else in this model.

In the attentions, `q_norm` and `k_norm` are `RMSNorm(head_dim, eps=1e-6)` applied per head; `v`
is not normalized. Nothing carries a bias. And **rotary position embedding is applied to self
attention only** -- the cross attention is explicitly excluded, which makes sense once the context
is 512 padded text tokens with no geometry, but it is one line in the reference and invisible
here. The MLP is a plain `Linear -> GELU -> Linear`.

### The residual stream has to be fp32

Straight from the reference, and the reason is not obvious:

> The residual stream for this model has large values. To make fp16 compute_dtype work, we keep
> the residual stream in fp32, but run attention and MLP modules in fp16. An alternate method that
> clamps fp16 values "works" in the sense that it makes coherent images, but there is noticeable
> quality degradation and visual artifacts.

`waifu::sdxl` runs fp16 throughout and has no reason not to. This one does. The weights are fine
narrow -- the exporter checks every one against fp16's range and none of turbo v1.1 comes close --
but the accumulating `x` between blocks is not.

And a wide stream is not enough on its own: **the places that step on and off it have to widen
before they multiply, not after.** There are two.

```python
x = torch.addcmul(x, gate.to(residual_dtype), result.to(residual_dtype))
```

Both operands are cast to float32 *before* the multiply. Gating in half and widening the product
is a different computation, and the difference is an infinity. The same goes for the modulation:
`norm(x) * (1 + scale) + shift` is done on the wide `x`, and only its result is narrowed for the
projections that follow.

Getting those two backwards cost a debugging session here, and every signpost pointed elsewhere.
It was not the timestep -- one step at sigma 0.75 was exact. It looked like the attention kernel,
because it appeared between 121 tokens and 144, which is where a 128 boundary would sit; attention
on its own is clean at every size tried. It was depth: twelve blocks survived, sixteen did not, and
a larger picture only reached the same cliff sooner. What a NaN at the end of twenty-eight blocks
tells you about where it started is almost nothing.

### Patchify and unpatchify are not inverses

The projection going in packs `b c (t r) (h m) (w n) -> b t h w (c r m n)`: channel first, then
the temporal and spatial offsets within the patch. The final layer coming out writes
`B T H W (p1 p2 t C) -> B C (T t) (H p1) (W p2)`: the offsets first and **the channel last**.

They are different orderings and the model's entire output rides on it. Unpacking the second as
though it mirrored the first type checks, keeps every shape, samples without complaint and returns
noise -- which is exactly what it did here before the ordering was read off the reference rather
than assumed.

## The text side

This is the part that looks like one encoder and is really three things.

Anima tokenizes the prompt **twice**, with two different tokenizers, and the denoiser never sees
the Qwen3 hidden states directly:

1. Qwen3-0.6B encodes the prompt into hidden states of width 1024. These are the *context*.
2. The same prompt is tokenized again with a **T5** tokenizer, and those ids -- vocabulary 32128,
   which is where that number comes from -- are embedded by `llm_adapter.embed` to form a *query*
   stream.
3. `llm_adapter` is a real six-block transformer at width 1024, 16 heads of 64, each block a
   self-attention, a cross-attention into the Qwen3 context, and a biased ungated 4x MLP, with
   three RMSNorms. It ends `norm(out_proj(x))`.
4. The result is multiplied by the per-token T5 weights -- this is what makes prompt weighting
   work, and why the model card says weights need to be pushed harder than for SDXL -- and then
   padded to **512 tokens**, which is the cross-attention context length.

Only that 512 x 1024 tensor reaches `cross_attn` in the denoiser blocks.

The reason for the shape is historical: Cosmos-Predict2 was pretrained against T5, and the adapter
is what lets a Qwen3 encoder stand in for it. It is not a training leftover and it cannot be
skipped.

Practically this means the package carries two tokenizers, in two named sections of one
`tokenizer.ini`, each section naming the entry that holds one model's published `tokenizer.json`.
`tools/tokenizer_exporter.py` puts them there and `waifu/src/tokenizer.rs` reads them back with the
`tokenizers` crate.

They are genuinely two different algorithms rather than two vocabularies for one, and the reason
this is worth saying is that it used to be a decision made here. The Qwen3 one is byte level BPE.
The T5 one is **not**: sentencepiece writes two kinds of model into the same file, a BPE one whose
scores are merge ranks and a unigram one whose scores are log probabilities, and T5's is unigram --
a walk through the text scoring highest overall, a dynamic program with one right answer, rather
than a sequence of merges. Reading it with the merge rule instead does not give a slightly worse
segmentation, it gives a different text: merging picks the *commonest* pair at each step, which is
the shortest, and walks away from the long pieces the vocabulary was built to have. Measured, it
disagreed on every prompt tried, and `masterpiece` came apart into eight pieces.

So this repository once carried a `bpe.rs` and a `unigram.rs`, and an exporter that read a
`trainer_spec` to decide which of the two a vocabulary wanted. None of that is here now. A
`tokenizer.json` names its own model type, so the question is answered by the file rather than
inferred, and the normalizer each of them applies first -- NFC for Qwen3, `nmt_nfkc` for T5 --
comes along with it. That last part was a real gap: the encoders written here normalized nothing,
so a fullwidth or ligatured prompt tokenized differently here than where the weights were trained.

Both are still checked against the published tokenizers token for token, over a corpus of 611 texts
each that the exporter writes into the test package. It is a narrower check than it was -- the
question is now whether the runtime reads back the file the exporter wrote, not whether an encoder
here walks a vocabulary correctly -- and it is still the only thing that can catch a package built
against the wrong revision, since no reference tensor can say what a vocabulary should produce.

## The text encoder (verified)

Stock Qwen3-0.6B: vocab 151936, hidden 1024, GQA with q `[2048, 1024]` and k/v `[1024, 1024]`
(16 query heads, 8 key/value heads, head dim 128), QK RMSNorm, SwiGLU MLP at 3072, RMSNorm
throughout.

Every one of those is an op flint already has -- `rmsNorm`, `swiglu`, `attention` (which takes k
and v already at `numKeyValueHeads`, so GQA needs no expansion), `rotaryEmbedding`. The deleted
`waifu/src/llama.rs` (last alive at `2e5f14f^`, 538 lines: `Mlp`, `Attention`, `DecodeLayer`,
`LlamaModel`) is the same shape of decoder and is the right thing to start from, minus the KV
cache and generation machinery -- a text encoder does one forward pass over the whole prompt and
keeps the hidden states.

## The VAE, and why it does not need conv3d

The Qwen-Image VAE is a **video** VAE: its convolution weights are five-dimensional,
`[out, in, 3, 3, 3]`, wrapped in a causal-in-time layer, with `gamma` RMS norms and an attention
block in the middle. flint has `conv2d` and no `conv3d`.

It does not need one. For a single image the temporal extent is 1 from end to end, and then the
causal convolution collapses exactly onto a 2-D one:

- `QwenImageCausalConv3d` pads the time axis by `2 * padding[0]` **at the front only**, through
  `F.pad` with its default constant zero. With `T = 1` the two padded frames are zeros, so of the
  three temporal taps only the last multiplies anything.
- Nothing in the decoder grows the time axis. `upsample3d` doubles it only on the
  `feat_cache is not None` path, and the first (and for an image, only) chunk takes the branch
  that marks the slot `"Rep"` and skips `time_conv` entirely. The spatial resamples are already
  `nn.Conv2d` applied per frame.

So every `Conv3d` in the decoder can be folded at export time to a `Conv2d` over
`W[:, :, -1]`, the last temporal slice. `tools/causal_conv3d_folding_test.py` checks exactly
that, against the reference semantics copied from
`diffusers.models.autoencoders.autoencoder_kl_qwenimage`, over float64:

```
3x3x3 pad(1,1,1)   max|diff| = 0.000e+00      # the common decoder conv
1x1x1 pad(0,0,0)   max|diff| = 0.000e+00      # conv1 / conv2
3x1x1 pad(1,0,0)   max|diff| = 0.000e+00      # upsample3d time_conv
```

Exact, not close. **This port needs no new flint kernels at all** -- conv2d, attention, rmsNorm,
swiglu, rotaryEmbedding, gelu, silu and softmax cover the whole of it.

The cost is that the exported VAE is image-only, which for this runtime it always was.

Three more things the weights do not say on their own:

- **The `time_conv` weights are dead for images.** Both `encoder.downsamples.N.time_conv` and the
  decoder's upsample counterparts are reached only on the `feat_cache` path, and the first chunk
  -- for an image, the only chunk -- takes the branch that caches and skips them. The encoder's is
  stride 2 in time with no causal padding, so at `T = 1` it could not even produce an output. Do
  not export them.
- **`QwenImageRMS_norm` is RMSNorm over the channel axis, not the last one.** It is written as
  `F.normalize(x, dim=1) * dim**0.5 * gamma`, and `x / (||x|| / sqrt(dim))` is exactly
  `x / sqrt(mean(x^2))` -- so it is RMSNorm, but along C of an `(N, C, H, W)` tensor where flint's
  `rmsNorm` works along the last dimension. It needs a permute either side, or the weights laid
  out to suit. `gamma` arrives shaped `[C, 1, 1, 1]` and wants flattening to `[C]`.
- The spatial `resample.1.weight` tensors are already 4-D `nn.Conv2d`, not causal 3-D, and need no
  folding at all.

## The sampler is the part that cannot be reused

Cosmos-Predict2 is **rectified flow**, not epsilon prediction. `waifu/src/sdxl/sampler.rs` builds
sigmas from `alphas_cumprod` and steps on the epsilon the model returns; none of that carries
over. A flow-matching Euler step is less code than what is there now, but it is new code, and it
is the one place where being wrong is silent -- the same trap the SDXL skill opens with.

Two schedules, because the distillation moved them: aesthetic v1.1 wants 30-50 steps at CFG 4-5,
turbo v1.1 wants **8-12 steps at CFG 1**, which is to say no classifier-free guidance and one
model evaluation per step instead of two.

## Shape of the work

Nothing here is unusually hard; there is just a lot of it, and almost none of it is shared with
what exists.

| | state | agrees with the reference to |
|---|---|---|
| `tools/anima_exporter.py` | done | -- |
| `tools/anima_comfy.py` | done; the reference is ComfyUI itself | -- |
| `waifu/src/anima/config.rs` | done | -- |
| `waifu/src/anima/text_encoder.rs` | done | 0.0029 |
| `waifu/src/anima/adapter.rs` | done | 0.0024 |
| `waifu/src/anima/dit.rs` | done | 0.0028 |
| `waifu/src/anima/vae.rs` | decoder done; no encoder, so no `draw -image` yet | 0.00072 |
| `waifu/src/anima/sampler.rs` | done | -- |
| the four of them together | draws a 512 by 512 picture in ten steps | -- |
| tokenizers | done; both in the package | token for token, over 611 texts each |
| `waifu/src/anima/pipeline.rs` | done | 0.0031, prompt to context |
| package + CLI | done. `waifu draw` reads `[model] type` and takes the package to whichever model it names |
| `hub.rs` `CATALOG` | done, once it was asked for: turbo v1.1 is published to both hubs as `anima:turbo`, under CircleStone Labs' licence rather than libwaifu's. See Licensing |

flint: nothing, as promised.

The RMSE numbers are relative, against `tools/anima_comfy.py`, which runs ComfyUI in float32 where the
runtime runs in half. The pipeline's is the encoder and the adapter composed, from a prompt as a
string through to the padded context the denoiser reads, so it lands a little above either of them
alone. The decoder is the closest of them, which is worth a second look: it is the
stage that appeared to need a kernel that does not exist, and folding a causal convolution over one
frame is exact rather than approximate, so what is left is only the half-precision arithmetic.

### What the exporter writes

`anima.dit`, `anima.adapter`, `anima.text` and `anima.vae`, 995 tensors for turbo v1.1, which is
`4 + 28 x 17 + 3` for the denoiser, `1 + 6 x 16 + 3` for the adapter, `1 + 28 x 8 + 1` for the
encoder and the VAE's 194 less the eight dead `time_conv` weights. (This file said 989 and
`6 x 15` until an export was counted: a block of the adapter writes sixteen tensors, not fifteen
-- two attentions of five apiece, four for the biased MLP and three norms.) Projections are fused where
the runtime wants them fused -- `qkv_proj` for both square attentions, `kv_proj` for the cross
attention that reads a narrower context, `gate_up_proj` for the encoder's SwiGLU, in that order,
because flint's `swiglu` gates on the first half.

Everything is narrowed from bfloat16 to fp16, which the parameter format can hold and bf16 it
cannot. That is a narrowing of *range* and not of precision -- bf16 keeps seven mantissa bits
where fp16 keeps ten, so every bf16 value inside fp16's exponent range converts exactly. Measured
over turbo v1.1: 649 of its 685 tensors come back bit for bit, the other 36 move by at most
3e-8, and those are weights small enough to fall through the bottom of fp16 rather than off the
top -- the smallest non-zero magnitude in the file is 2.5e-30. The largest is 123, against a
ceiling of 65504.

So the exporter's range check never fires for this model, and the same picture comes out of
narrowed weights as out of the originals. It is kept because the day a checkpoint does exceed the
range, what it would otherwise write is an infinity, and what that draws is noise.

### And the reference outputs

`-test_output` writes a second, small package the way `sdxl_exporter.py` does, holding what
`tools/anima_comfy.py` makes ComfyUI compute for one fixed prompt: the encoder's hidden states, the adapter's
context before and after padding, the velocity of one denoising step, and one decode. A 16 by 16
latent, so 64 patches and a 128 by 128 image -- every block exercised, 2.4 MB to carry.

The token ids were written into the exporter rather than tokenized at export time, so that the
reference outputs did not depend on having either tokenizer to hand. Four of the twenty were wrong
-- they decoded to `" Leonard Professional, best quality, 1spin, returning, ..."` -- and nothing
noticed, because the reference ran on the same wrong ids the runtime was then checked against. The
package now carries the tokenizers, so the exporter tokenizes with them and there is nowhere left
for a second copy of the answer to disagree. What a tokenizer does is still a separate question
from what the weights do, and the corpus is where it is asked.

The config is derived from the tensors themselves rather than written down, so the same exporter
produces the right `num_blocks` for the 40-block expansion without being told. The exceptions are
the things that are not in any tensor: the latent normalization, the rotary bases, and that this
is flow prediction rather than epsilon.

## Positions

Cosmos positions tokens on three axes rather than one sequence, and splits the head dimension
between them:

```
dim_h = head_dim // 6 * 2
dim_w = dim_h
dim_t = head_dim - 2 * dim_h
```

At `head_dim = 128` that is 42 for height, 42 for width and 44 for time, summing to 128. Each axis
gets its own base, `10000.0 * ntk_factor`, where the factor is
`extrapolation_ratio ** (dim / (dim - 2))`. The three are concatenated in the order t, h, w.
Rotation is NeoX `rotate_half`, which is the convention flint's `rotaryEmbedding` already
implements.

**The ratios are not all 1.0, and this file said for months that they were.** `model_detection.py`
branches on `in_channels`, and Anima's is 16 -- `x_embedder.proj.1.weight` is `[2048, 68]`, so
`68 // 4 - 1` -- which takes the branch asking for four times the spatial extent:

```
rope_h_extrapolation_ratio = 4.0
rope_w_extrapolation_ratio = 4.0
rope_t_extrapolation_ratio = 1.0
```

So height and width sit at `10000 * 4.0 ** (42 / 40)` = **42870.9**, and only time stays at 10000.
It is a trained-in property of the Cosmos-Predict2 2B this model descends from, not a runtime
choice, and it is the kind of error that draws a perfectly reasonable picture: measured on one
16x16 latent, a flat 10000 against the real bases is `max|d| = 5.5e-01` on the velocity, 13%
relative, at a cosine similarity of 0.9959. Close enough to look right, nowhere near right.

It was found by standing the hand transcription that used to live in `tools/anima_reference.py`
next to ComfyUI's own `comfy.ldm.anima.model.Anima`, on the same weights in float32. That
transcription is gone now, and this is why: it had been checked only by the fact that it drew a
coherent picture, the test tensors were generated from it, the runtime was built to match those,
and the suite was green throughout -- every one of those measured agreement with the
transcription rather than with Anima. **A reference that is not the upstream implementation is a
ruler nobody checks.** `tools/anima_comfy.py` replaces it by running ComfyUI itself, the same
arrangement `sdxl_exporter.py` has with diffusers, and `-test_output` now requires `-comfyui`.

Three things carried the assumption and moved together to be rid of it. `anima_exporter.py`
writes the ratios into the manifest as `rope_h_extrapolation_ratio` and its two neighbours --
written rather than assumed, because assuming them is how they were wrong -- and refuses a
checkpoint whose latent width it has no row for. `DitConfig::rope_bases` turns them into one base
per axis, which `waifu/src/anima/dit.rs` now gives to `frequencies` separately rather than
sharing `rope_theta` between all three. And the test tensors were regenerated, this time by
ComfyUI.

While the four stages were being put side by side, the other three were measured too, which had
never been done. They were right all along -- but nothing had established that, and the way the
rotary base was wrong is exactly how any of them could have been:

| stage | ComfyUI against the old transcription |
|---|---|
| Qwen3 text encoder | `6.1e-05` |
| LLM adapter | `0.0` exactly |
| denoiser | `8.9e-06` |
| VAE decoder | `2.1e-06` |

The last of those is worth its own line: libwaifu decodes through convolutions folded from three
dimensions to two, ComfyUI through the real `Conv3d`, and they agree to `2.1e-06` on the
published weights. The folding had only ever been proved on synthetic ones.

A model written before this has no ratios in its manifest and will not load: the reader asks
for the three keys and does not default them. That is deliberate. Defaulting to 1.0 would go back
to drawing the wrong picture in silence, and defaulting to 4.0 would pair a corrected base with
test tensors made against a flat one. Re-export it.

It needs no new kernel. flint indexes a `(maxPositions, 2 * headDim)` cache by token position, and
since every token's 3-D position is known before the pass, the whole axial layout can be baked
into that cache host-side and looked up with `positions = 0..numTokens`.

## Latents

The Qwen-Image VAE is `z_dim = 16` and normalizes per channel, not by one scalar the way SDXL
does:

```
latents_mean = [-0.7571, -0.7089, -0.9113,  0.1075, -0.1745,  0.9653, -0.1517,  1.5508,
                 0.4134, -0.0715,  0.5517, -0.3632, -0.1922, -0.9497,  0.2503, -0.2921]
latents_std  = [ 2.8184,  1.4541,  2.3275,  2.6558,  1.2196,  1.7708,  2.6052,  2.0743,
                 3.2687,  2.1526,  2.8652,  1.5579,  1.6382,  1.1253,  2.8251,  1.9160]
```

Sixteen pairs, applied channelwise. `SdxlConfig`'s single `scaling_factor` has nowhere to put
these, which is one more reason the config is its own type rather than a widened SDXL one.

## What was decided

Build against **anima-turbo-v1.1** first. At CFG 1 it runs one model evaluation per step instead
of two and wants 8-12 steps instead of 30-50, so a verification pass costs a fraction of what
aesthetic costs, and aesthetic is afterwards a matter of configuration rather than new code. The
classifier-free path still has to exist for aesthetic, but it does not have to exist first.

Export to `models/` and keep it there. Nothing about this goes to Hugging Face or ModelScope
without being asked for separately -- see Licensing.

## Licensing

`license: other`, `license_name: circlestone-labs-non-commercial-license`, with the text in the
repository's own `LICENSE.md`. It restricts commercial deployment of the model while leaving
generated images free to use. If any of this is ever published as a model, that license
travels with it and not libwaifu's MIT -- and Anima-2.9B is a third-party derivative, so its terms
want reading separately rather than assuming they match.
