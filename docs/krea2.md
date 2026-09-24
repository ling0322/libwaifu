# Krea 2

Notes on running Krea 2 in libwaifu. The shapes were read off the published `config.json` files
and checked against the tensor headers; the behaviour that shapes alone do not show -- the prompt
template, the padding, the position layout, the schedule -- came from the two reference
implementations, which are `krea-ai/krea-2` on GitHub (`mmdit.py`, `encoder.py`, `sampling.py`)
and diffusers' own `Krea2Pipeline` and `Krea2Transformer2DModel`.

Krea 2 is **neither an SDXL fine tune nor an Anima release**. It is a twelve billion parameter
single-stream MMDiT, conditioned on twelve tapped layers of a Qwen3-VL-4B text encoder, drawing
through the same Qwen-Image VAE Anima uses. Two of its four parts are new code; two are not.

## What is published

`krea/Krea-2-Turbo`, gated, in the diffusers layout:

| file | size | what |
|---|---|---|
| `transformer/*.safetensors` | 26.3 GB | the denoiser, in three shards, bf16 |
| `text_encoder/model.safetensors` | 8.88 GB | Qwen3-VL-4B-Instruct, vision tower and all |
| `vae/diffusion_pytorch_model.safetensors` | 0.51 GB | the Qwen-Image VAE, both halves |
| `tokenizer/tokenizer.json` | 11.4 MB | Qwen2's byte-level BPE |
| `turbo.safetensors` | 26.3 GB | the same denoiser under the research code's own names |

`krea/Krea-2-Raw` is the undistilled release the turbo one was distilled from. It is **not**
exportable by `tools/krea2_exporter.py` as it stands, and the exporter says so rather than
producing a package at the wrong schedule: its timestep shift is interpolated from the number of
image tokens, so it is a function of the resolution where a package carries one number. See
"The sampler" below.

`turbo.safetensors` is ignored here. It is the same weights as `transformer/`, named the way
`mmdit.py` names them; the diffusers names are the ones this exporter reads, because diffusers is
also what the reference outputs are computed with.

## The denoiser

| | |
|---|---|
| blocks | 28 |
| hidden | 6144 = 48 heads x 128 |
| key/value heads | 12 -- grouped four to one |
| SwiGLU | 16384 |
| patch | 2, over 16 latent channels, so 64 channels in and 64 out |
| timestep embedding | 256 wide before its two linears |
| rope | theta 1000, axes `(32, 48, 48)` over `(t, h, w)` |
| norm eps | 1e-5 |
| text fusion | 2560 wide, 20 heads, SwiGLU 6912, 2 layerwise + 2 refiner blocks |
| tapped layers | 12, at encoder layers 2, 5, 8, ... 35 |

11.8 billion parameters in the transformer, 3.9 billion in the encoder's language half.

### How a block runs

```
mod = temb_mod + block.scale_shift_table          # (6, 6144), one vector for the whole model
x = x + pregate  * attn((1 + prescale)  * norm1(x) + preshift)
x = x + postgate * ff((1 + postscale) * norm2(x) + postshift)
```

Five things here are easy to get wrong, and none of them fails loudly.

**The modulation is shared.** `time_mod_proj` produces one `6 x 6144` for the whole model and each
block adds its own learned table to it. There is no per-block projection of the timestep -- that is
the Cosmos transformer in `waifu/src/anima/`, and reading this one that way means looking for
weights that are not there.

**The attention is gated.** A fourth projection of the *normalized input* -- not of anything the
attention produced -- goes through a sigmoid and multiplies the attention output before the output
projection. It is as wide as the model rather than as the keys. The exporter fuses the query, key,
value and gate into one matrix in that order, so the runtime slices it at
`6144 | 1536 | 1536 | 6144`.

**The rotary pairing is adjacent.** `x[2i]` rotates against `x[2i + 1]`. Every other model here --
including the Qwen3-VL encoder that feeds this one -- pairs `x[i]` against `x[i + half]`. Both are
"rotary embeddings" and they are different permutations of the head; using one where the other
belongs draws a plausible and wrong picture. diffusers' `repeat_interleave_real=True` is what says
which, and `mmdit.py` builds the same thing as a stack of 2x2 rotations over `reshape(..., -1, 1, 2)`.

**The norm scales are zero centred.** Every RMSNorm in the denoiser multiplies by `1 + weight`.
`tools/krea2_exporter.py` folds the one in, so the package holds the multiplier itself and the
runtime's ordinary RMSNorm is right. The *encoder's* norms are not like this -- Qwen3 stores the
multiplier -- so the exporter folds one set and not the other.

**The GELUs are the tanh approximation.** All three of them: two in the timestep path and one in
the text projection. flint's `gelu` is the exact `x * Phi(x)`, which is right for every other model
here, so `krea2::dit::gelu_tanh` writes the approximation out. The difference is about 1e-3 around
the shoulder and it lands in the modulation of every block at every step.

### The residual stream

Kept in float32 while the sublayers run in half, the same as Anima's. The reference runs the whole
model in bfloat16, which has float32's exponent range; fp16 does not, and twelve billion
parameters of accumulated residual is not a thing to find out about at step six.

### Patchify and unpatchify *are* inverses

Unlike Anima's. Going in, `(1, C, H, W) -> (1, H/p * W/p, C*p*p)` with the channel leading inside a
patch; coming out, exactly backwards. There is also no mask channel: Anima's Cosmos denoiser reads
`(C + 1) * p * p` and this one reads `C * p * p`.

## The text side

The prompt is wrapped in the template the model was sampled with:

```
<|im_start|>system
Describe the image by detailing the color, shape, size, texture, quantity, text, spatial
relationships of the objects and background:<|im_end|>
<|im_start|>user
{prompt}<|im_end|>
<|im_start|>assistant
```

34 tokens in front, 5 behind. The encoder reads all of it; the 34 are then dropped from the
*states*, not from the ids -- they are what every state after them was computed against.

Then twelve of the encoder's thirty-six layers are stacked per token, `(L, 12, 2560)`, and that is
what the denoiser is told about the prompt. Inside it:

1. two blocks attend **across the twelve tapped layers**, one token at a time, with the sentence as
   the batch. The reshape is `(B, L, 12, D) -> (B * L, 12, D)`. Reading it the other way round --
   twelve as a batch and the sentence as the sequence -- type checks all the way to a picture;
2. a `12 -> 1` matrix collapses the stack;
3. two more blocks attend along the sentence;
4. an RMSNorm, a linear, a tanh GELU and another linear widen 2560 to 6144;
5. the result is concatenated *in front of* the image tokens, and every block reads one stream.

### The padding that is not here

The reference pads every prompt to 512 tokens **in the middle of the template** -- prefix, prompt,
padding, then the five closing tokens -- and carries a mask saying which are real. This runtime
pads nothing and runs the prompt at its own length. That is the same arithmetic, not an
approximation of it:

* **the encoder.** It is causal, and the reference masks the padding away as keys, so no real
  token ever reads one. Its positions are `cumsum(mask) - 1`, which counts only real tokens -- so
  the five closing tokens sit at positions 34+n .. 34+n+4 whether or not there is padding in
  between. diffusers says this in as many words: without it "the suffix gets a shifted mRoPE
  phase". The states of every real token are therefore identical either way.
* **the layerwise fusion blocks.** They attend across the twelve layers of one token. One token's
  padding cannot reach another's, whatever the mask says.
* **the refiner blocks and the twenty-eight stream blocks.** Every one of them takes the key
  padding mask, so a padded token is never a key. The rows a padded *query* produces are garbage
  in both implementations and are thrown away with the rest of the prompt at the output slice.

So the padding contributes to nothing that survives, and dropping it turns a 512-token prompt into
a 40-token one: at 1024x1024 that is 4136 tokens of attention rather than 4608, for the same
answer.

That is an argument, so it was measured. Running the reference's own text fusion two ways -- 512
rows and a mask, against the 19 real rows and no mask -- and comparing the rows that survive:

| | |
|---|---|
| in float32 | **4.0e-7**, which is float32 rounding and nothing else |
| in bfloat16 | 1.4e-3, which is bfloat16 rounding |

So the equivalence is exact and what is left is arithmetic order. By the end of all twenty-eight
blocks that 1.4e-3 has grown to 1.2e-2 in bfloat16 -- the depth amplifying its own rounding, the
same way the eight-step walk amplifies the difference between two float types.

`waifu/tests/krea2.rs` then checks the whole thing against the reference *with* its padding, which
is the only way the claim is worth anything end to end.

A prompt longer than `512 - 5` tokens is cut, which is what the reference's `truncation=True`
does at the same place.

## The text encoder

Qwen3-VL-4B-Instruct: 36 layers, hidden 2560, 32 query heads against 8 key heads of 128, SwiGLU
9728, rope theta 5e6, RMSNorm eps 1e-6, q/k norms per head. Structurally Anima's Qwen3-0.6B
encoder at six times the size, and `waifu/src/krea2/text_encoder.rs` is that file with three
differences.

**Only the language half is exported.** The published encoder is a vision-language model; its
vision tower is about two gigabytes that no prompt reaches.

**mRoPE is ordinary rope here.** Qwen3-VL interleaves three position axes through the head
(`mrope_section: [24, 20, 20]`). For a prompt with no picture in it all three carry the token's
index, so every frequency gets the same angle it would have got from a plain rotary embedding.
Nothing special is needed and nothing is being approximated.

**The taps are before the final norm and there is no lm_head.** `output_hidden_states` hands back
each layer's output as it leaves the residual stream; only the entry past the last layer has
`model.norm` applied. The highest tapped layer is 35 of 36, so `model.norm` is not in the package
at all.

## The VAE

The same Qwen-Image autoencoder Anima draws through, down to the sixteen latent means and
deviations -- so `waifu/src/qwen_vae.rs` is one file that both families re-export, and the note in
`docs/anima.md` about folding causal conv3d to conv2d applies here unchanged.

What differs is the names. Anima's package is exported from the original checkpoint and Krea 2
publishes the diffusers conversion, so `krea2_exporter.vae_name` rewrites one into the other:

| diffusers | the original, which the runtime reads |
|---|---|
| `post_quant_conv` | `conv2` |
| `quant_conv` | `conv1` |
| `decoder.conv_in` | `decoder.conv1` |
| `decoder.mid_block.resnets.{0,1}` | `decoder.middle.{0,2}` |
| `decoder.mid_block.attentions.0` | `decoder.middle.1` |
| `decoder.up_blocks.N.resnets.M` | `decoder.upsamples.K` -- one flat list |
| `decoder.up_blocks.N.upsamplers.0` | the entry after that stage's residual blocks |
| `resnet.{norm1,conv1,norm2,conv2,conv_shortcut}` | `residual.{0,2,3,6}`, `shortcut` |
| `decoder.norm_out`, `decoder.conv_out` | `decoder.head.0`, `decoder.head.2` |

Where each stage begins in the flat list is counted from the weights rather than written down.

Both halves are exported, as Anima's are. The encoder half has no layer to run it yet, so image to
image is refused with the same sentence.

One difference is not in the weights, and it cost two false explanations before it was pinned
down: `AutoencoderKLQwenImage.decode` **clamps** its answer to `-1..=1`, and this runtime's
decoder hands back what the convolutions produced, clamping where a picture becomes bytes. So a
fixture taken from `decode` measures the clamp. How much depends entirely on the latent:

| the latent fed in | pixels outside `-1..=1` | what the clamp was worth |
|---|---|---|
| `torch.randn` | a tenth of them | 4.8e-2 |
| the one eight real steps arrive at | two in a thousand | 1.4e-3 |

Neither is a property of either implementation. `tools/krea2_exporter.py` therefore calls the
decoder rather than `decode`, and the fixture is unclamped.

With that out of the way the decoder can be measured, and it is exactly the reference's:

| this runtime's decoder, against a float32 reference | |
|---|---|
| on the processor at full width, where neither side has a TF32 convolution | **1.9e-6** |
| on the card at full width, where both do | 7.3e-4 -- and TF32 costs the reference **7.3e-4** too |
| on the card in float16 | 1.7e-3 -- and diffusers' own float16 decode is **1.6e-3** |

The first line is the one that matters: the graph is the reference's arithmetic and not a near
copy of it. Everything after it is the width.

Half precision is also the right width for this autoencoder rather than a concession. Against
that same float32 reference, diffusers in **bfloat16** -- the dtype its own pipeline loads by
default -- is 9.2e-3, five times further out than this runtime's float16. Eight mantissa bits is
what costs; the decoder is where a picture stops being a tensor, and it wants the bits.

## The sampler

The same straight line Anima walks, which is why `waifu/src/flow.rs` is one file. The two
references write the bend two ways:

```
krea:   sigma = exp(mu) / (exp(mu) + (1 / t - 1))
anima:  sigma = shift * t / (1 + (shift - 1) * t)
```

and those are the same curve with `shift = exp(mu)`. The distilled release was trained at a fixed
`mu = 1.15`, so the package carries `sampler_shift: 3.15819`. The undistilled one interpolates mu
between 0.5 and 1.15 by image token count, which is a function of the resolution and does not fit
in one number -- hence the exporter's refusal above.

The timesteps are `linspace(1, 1/N, N)` bent by that shift, with a zero on the end. Anima's
sampler builds `1 - i/N` for `i` in `0..=N`, which is the same list. The model is handed the sigma
itself: its embedding multiplies by a thousand on the way in, so passing it 750 instead of 0.75
loads fine and produces noise.

## Guidance

Krea's reference computes `cond + scale * (cond - uncond)` and calls a scale of **zero** no
guidance. Every model in libwaifu spells that `uncond + scale * (cond - uncond)` with **one**
meaning none, which is the same formula one apart. So a Krea guidance of 4.5 is a 5.5 here, and
turbo's 0 is a 1. The pipeline says so where it does it.

For the distilled release that is not a number to start at but the only number there is. It was
trained to answer as though it had already been guided, so there is no second answer to push away
from: diffusers gives it blocks of its own with no guider on them and *no negative prompt
argument at all*, and the classic pipeline documents `negative_prompt` as ignored whenever
`guidance_scale <= 0`. Handing it either is asking for a picture nobody promised -- guidance on
top of distilled guidance is the usual burnt, over-contrasted one.

So the package says so, in its `suggested:` block:

```yaml
suggested:
  steps: 8
  guidance: 1.0
  takes_guidance: "false"
```

`guidance:` is where a dial would start; `takes_guidance:` says there is no dial. The web UI reads
it and stops drawing the CFG card and the negative prompt box, and `/api/generate` drops both from
a request that sends them anyway. It is one key rather than two because it is one fact: guidance
is the second pass and the negative prompt is what that pass is given, so a model without the one
has no use for the other.

Quoted, because `model_writer.py` puts quotes round anything YAML would otherwise read as a
boolean of its own. The reader takes `true/false`, `yes/no`, `on/off` and `1/0` in either case,
and a word it cannot read is no answer rather than a refusal -- the same as every other key here.

A manifest that says nothing leaves the answer to what the screen already believed about the kind
of model, which for `krea2` is that it takes no guidance, for the same reason its built-in
defaults are eight steps: the only release of it this exports is the distilled one. That is what
makes a package exported before this key existed behave correctly without being rewritten. The day
there is an undistilled package, its manifest says `takes_guidance: "true"` and gets the dial back.

## What the exporter writes

`krea2.dit`, `krea2.text` and `krea2.vae`, under the manifest's `krea2:` block.

| in the package | from |
|---|---|
| `dit.img_in.{weight,bias}` | `img_in` |
| `dit.time_embed.linear_{1,2}.{weight,bias}` | the same |
| `dit.time_mod_proj.{weight,bias}` | the same |
| `dit.text_fusion.{layerwise,refiner}{i}.*` | `text_fusion.{layerwise,refiner}_blocks.i.*` |
| `dit.text_fusion.projector.weight` | the same, never quantized |
| `dit.txt_in.{norm,linear_1,linear_2}` | the same |
| `dit.block{i}.attn.qkvg_proj.weight` | `to_q`, `to_k`, `to_v`, `to_gate`, concatenated |
| `dit.block{i}.attn.{q_norm,k_norm}.weight` | `attn.norm_{q,k}`, folded `+ 1` |
| `dit.block{i}.ff.gate_up_proj.weight` | `ff.gate`, `ff.up`, concatenated, gate first |
| `dit.block{i}.{norm1,norm2}.weight` | the same, folded `+ 1` |
| `dit.block{i}.scale_shift_table` | the same |
| `dit.final.*` | `final_layer.*`, its norm folded |
| `text.embed.weight` | `language_model.embed_tokens.weight` |
| `text.block{i}.attn.qkv_proj.weight` | `q_proj`, `k_proj`, `v_proj`, concatenated |
| `text.block{i}.mlp.gate_up_proj.weight` | `gate_proj`, `up_proj`, concatenated |
| `text.block{i}.{input_norm,post_attn_norm,attn.q_norm,attn.k_norm}` | **not** folded |
| `vae.*` | the table above |

`-fp8` quantizes everything the runtime multiplies by and nothing else: the package goes from
33.8 GB to 17.3 GB -- 279 matrices quantized, everything else as it was -- for about 2.6e-2 of
relative error per weight. The `12 -> 1` projector stays float whatever is asked for, because an
FP8 multiply wants its inner dimension in multiples of sixteen.

### And the reference outputs

`-test_output` runs the reference pipeline once -- the test prompt, 128 by 128, the eight steps
the distilled release is for -- and takes every fixture off that one trajectory:

| tensor | what it is |
|---|---|
| `input_ids` | the template around the prompt, as the encoder reads it |
| `hidden` | the twelve tapped layers, the padded rows dropped |
| `noise` | the latent the run starts from |
| `sigmas` | the nine levels `FlowMatchEulerDiscreteScheduler` actually walked |
| `latent`, `timestep` | the latent entering step 4, and the sigma it enters at |
| `velocity` | what the denoiser answers there |
| `final_latent` | what the eight steps leave behind |
| `decoded` | the picture, from the autoencoder at full width |

**None of these is `torch.randn`**, and that is the point rather than tidiness. A part of this
model is only worth comparing on activations it was trained to see: hand the denoiser white noise
at sigma 0.75, or the autoencoder a latent no denoiser produced, and both implementations will
answer *something* and the number between them means nothing. The autoencoder shows it plainly --
a tenth of the pixels of a decoded random latent land outside `-1..=1`, where none of a real
one's do.

The text tensors are the reference's *padded* run with the padded rows dropped, which is what
makes `waifu/tests/krea2.rs` a test of the padding argument rather than of itself -- and the walk
from `noise` to `final_latent` tests it over eight steps rather than one.

## What it was checked against

`tools/krea2_exporter.py -test_output` writes what diffusers makes of one prompt and
`waifu/tests/krea2*.rs` reads it back. On an A6000, against the released turbo weights:

Every input is one the model would really be handed -- see "And the reference outputs" above.

| | measured | the control it is read against |
|---|---|---|
| the twelve tapped states | 1.06e-2 | the reference's own bfloat16 is **1.05e-2** from float32, so nearly all of this is the reference's |
| one velocity, step 4 of 8 at sigma 0.7595 | 3.60e-2 | on bit-identical inputs. diffusers in fp16 against diffusers in bf16, same inputs, is **3.51e-2** -- so this is the float type and almost nothing else |
| one decode, of the latent those steps arrive at | 1.72e-3 | against a float32 reference. diffusers' own float16 decode is **1.64e-3** from it; at full width on the processor the two agree to **1.9e-6** |
| the nine sigmas | < 1e-7 | against `FlowMatchEulerDiscreteScheduler`'s own, which is what pins `shift = exp(mu)` |
| the whole eight-step walk | 9.94e-2 | diffusers in fp16 against diffusers in bf16, same everything, is **6.7e-2** -- a walk amplifies, and the reference does not agree with itself either |
| the prompt, end to end | 1.06e-2 | the pipeline's own tokenizing and template, against the reference's padded run |
| the tokenizer | exact | 614 texts, token for token, plus the template's 34 and 5 |

The walk is the one loose number and it is worth saying why rather than quoting it flat: eight
Euler steps amplify whatever one step disagrees by, and at these widths *changing nothing but the
float type inside diffusers itself* moves the final latent by 6.7e-2 and the picture it decodes
to by 1.2e-1. What the bound there catches is a schedule walked backwards or a step that does not
step, which are half a latent away rather than a tenth.

A 1024x1024 draw at the suggested eight steps takes about half a minute on that card once the
package is read, and the same seed draws the same picture twice. An fp8 package draws the same
picture as a float one.

## Shape of the work

The denoiser is 24 GB in fp16 and the encoder 7.8 GB, so a float package is 33.8 GB: that much on
the card under `cuda` (33.6 GB of it, measured). The host holds one file of the package at a time
while it is read -- each weight is moved as it arrives and the file let go of before the next -- so
reading it costs about 4 GB of host memory rather than the package. An fp8 one is 17.3 GB on disk
and 17.9 GB on the card, and draws the same picture.
`cuda_cpu_offload` draws on a smaller card by keeping the package page-locked on the host instead
-- 33.8 GB of it for the float package, which the driver locks for as long as the model is held --
and moves the whole model across the bus once per step, which at this size is not a small thing to
ask of it.

A test that reads the package pays that once per test, so `waifu/tests/krea2.rs` and
`waifu/tests/krea2_pipeline.rs` are one test each: the harness gives every test its own thread and
a thread-local fixture dies with it.

## Licensing

`license: other`, `license_name: krea-2-community-license`, and Krea's own repository is gated:
fetching the weights means agreeing to it, and there is an acceptable use policy alongside. Both
travel with any derivative, including a converted package -- libwaifu's MIT covers this code and
not those weights.

The converted packages are published at
[ling0322/libwaifu-krea2-turbo](https://huggingface.co/ling0322/libwaifu-krea2-turbo) and on
ModelScope under the same name. What §3 of the agreement asks of that, and what was done:

| | |
|---|---|
| §3.1(a) a copy of the agreement, recipients bound by it | `LICENSE.pdf` ships in the package and the card says that downloading is agreeing. The repository is **not** gated, unlike Krea's -- gating would 401 the anonymous fetch `waifu webui` makes |
| §3.1(b) "Krea" at the beginning of the model name | the card names the model *Krea 2 Turbo*; the repository id keeps this project's `libwaifu-` prefix |
| §3.1(c) the attribution notice, verbatim | `NOTICE` |
| §3.2(a) say that you modified it | `NOTICE` lists every change: the renaming and refusing, the conv folding, the norm folding, the narrowing, the quantization |
| §3.2(c) not official, not endorsed | said in `NOTICE` and on the card |
| §4.2 content filtering | an obligation on whoever *deploys* these weights; this runtime ships none, and the card says so |

`tools/krea2_exporter.py` itself publishes nothing: it writes a package onto your own disk.
