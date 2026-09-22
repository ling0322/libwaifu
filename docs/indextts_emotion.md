# IndexTTS-2.5's emotion path

Where the emotion vector comes from when nobody supplies one. A recording in, one row of 1280 out
— the row that sits beside the speaker's in [the GPT](indextts_gpt.md)'s prefix.

```rust
use waifu::indextts_emotion::{self, Config};

let config = Config::indextts();
let emotion = indextts_emotion::graph(
    &g, features, &config, frames, positions, dtype, device,
)?;
```

`features` is what [w2v-bert](w2v_bert.md) read off the reference audio — the same tensor
[the semantic codec](semantic_codec.md) is handed, standardized by the release's own mean and
deviation. `positions` is the sinusoid table, and there is a wrinkle about which copy of it to
use: see [below](#the-position-table-is-not-the-closed-form).

## Four stages

| stage | what it is | out |
| --- | --- | --- |
| `emo_conditioning_encoder` | a WeNet conformer: one convolution that halves the length, four blocks | `(T/2, 512)` |
| `emo_perceiver_encoder` | lucidrains' `PerceiverResampler`, with **one** latent | `(1, 1024)` |
| `emovec_layer` | a projection | `(1280)` |
| `emo_layer` | another | `(1280)` |

153 tensors and 163 M parameters, all under the checkpoint's own names — this module renames
nothing and folds nothing, for the reason [the GPT](indextts_gpt.md) gives.

The one latent is the whole trick of the middle stage. The speaker path's perceiver resamples a
recording to thirty-two latents and gets a sequence; this one asks for a single latent, and what
comes back is therefore a vector. A feeling is one thing about a recording, so it is one row.

## 134 M parameters in a single matrix

`emo_conditioning_encoder.embed.out.0.weight` is `(512, 261632)`. That is more than the conformer
it belongs to, and a fifth of everything the GPT's package holds.

It is not a transcription error. WeNet's `Conv2dSubsampling2` reads the 1024 features as an image
one channel deep, runs a 3×3 convolution at stride two over *both* axes, and then flattens
everything it produced at each output frame — 512 channels by the 511 feature columns that
survive 1024 — into one projection down to 512. Five hundred and eleven columns times five hundred
and twelve channels is the 261,632.

Worth knowing before anyone wonders why a four-block conformer weighs as much as a good part of
the twenty-four-layer stack it feeds, and worth checking at export, which
`tools/indextts_gpt_exporter.py` does: it is the one tensor here whose size a reader is most
likely to assume is wrong and helpfully "fix".

## Three things upstream does that are easy to get wrong

**The relative attention has no shift.** The score is Transformer-XL's — `(q + u) · k` plus
`(q + v) · p`, where `p` is the sinusoid table through `linear_pos` — but WeNet's `rel_shift`,
which is what would turn the second term's absolute positions into relative ones, is commented
out in the released source with a note that it is useless for speech. So the position term is an
ordinary matrix multiply, and a reimplementation that helpfully puts the shift back is wrong.

**The gated feed forward gates the second half.** `flint`'s `geglu` computes `gelu(first) *
second`; this `GEGLU` is `gelu(second) * first`. The halves are the other way round, so the module
writes the two slices out rather than calling the operator, and the weight stays laid out the way
the checkpoint holds it. Both use the exact GELU, which is the one thing about the two that does
agree.

**This conformer is not macaron and [w2v-bert's](w2v_bert.md) is.** There, two feed forwards are
added back at half weight; here there is one, at full. The two conformers in this one pipeline
differ on exactly that, which is how the mistake would arrive — and a scale copied from the wrong
one is wrong in every layer without being wrong in any shape.

## There is no mask

Upstream pads a batch of recordings out to the longest and carries a length mask through both
stages; `emo_cond_mask_pad` is the `ConstantPad1d` that prepends a `True` for the one latent. One
recording at a time — which is what a reading is — makes every position valid, so the masked
softmax is an ordinary one and there is nothing to pad.

That also makes the perceiver's context order irrelevant: concatenating the latents after the
frames rather than before permutes the keys and the values together, and a softmax does not care.
It would start to matter the moment a batch were padded, which is why the mask exists upstream and
why leaving it out is a statement about the caller rather than about the arithmetic.

## The position table is not the closed form

`emo_conditioning_encoder.embed.pos_enc.pe` is a registered buffer — five thousand rows of
sinusoid that depend on nothing but their own shape — so the obvious thing is to rebuild it at
load and save the ten megabytes. That turns out to be wrong, and it is the one surprise this
module turned up.

**The released table differs from the sinusoid at essentially every entry**, by 7.5e-4 on average
and 2.0e-3 at worst. Chasing that number down is worth the five minutes it takes, because the
first guess is wrong and the second one settles the design:

| hypothesis | verdict |
| --- | --- |
| every stored value is representable in `float16` | **no** |
| every stored value is representable in `bfloat16` | **yes** |
| `stored == sinusoid(...).to(float16)` | no, 2.0e-3 out |
| `stored == sinusoid(...).to(bfloat16)` | **yes, difference exactly zero** |

So it is not a mystery and not a corruption: somebody saved this model through `bfloat16` once,
and the buffer kept the scar. The worst case lands where it should: 2.0e-3 is 2^-9, exactly half
a `bfloat16` step for a value just under one, which is the most round-to-nearest can be out with a
7-bit significand.

And upstream runs with the rounded one. `pe` is persistent, so it is in the file;
`load_checkpoint` calls `load_state_dict(strict=False)`, which only forgives *absent* keys and
this one is present; so the exact table computed in `PositionalEncoding.__init__` is overwritten
the moment the checkpoint loads. A reimplementation that rebuilds the table from the formula is
therefore running a model the release does not run — by very little, but for no reason.

That leaves a genuine choice, and it could be made either way. The exporter writes the tensor out:
ten megabytes is 0.3% of the package, against a standing claim that a Rust loop over `f64` sines
lands on the same `bfloat16` grid `torch` reached through `float32`. True as far as it was tested,
and not worth having to keep true. The check at export is now a bit-for-bit equality rather than a
tolerance — a mystery that has been solved deserves a check that fails the moment it comes back.

`waifu::indextts_emotion::positional_encoding` still builds the unrounded table, for the tests and
for running without a package. What that costs is 0.13% of the position term after `linear_pos`,
below the noise floor of the half precision this model runs in on a card — worth writing down
rather than worth worrying about.

## What is checked

`waifu/tests/indextts_emotion.rs`, against the released source's own `ConformerEncoder` and
`PerceiverResampler` — imported and run, not transcribed. That distinction earns its keep here:
every one of the mistakes listed above produces a tensor of the right shape, and none would be
caught by comparing this against a second reading of the paper. The reference is
`tools/indextts_emotion_reference.py`, which needs no download and no checkpoint.

```bash
cargo test --manifest-path waifu/Cargo.toml --test indextts_emotion
```

Five probes: the sinusoid table, which both sides build for themselves and which fails before any
weight is involved; then the model cut where upstream cuts it, after `self.embed`, after the
blocks, after the perceiver, and after the two projections. A failure says which.

Nine plausible bugs were written into the module one at a time and every one of them fails, by
between 55 and 11,000 times the tolerance. **Getting there needed the weights drawn larger than
the other reference scripts here draw them.** At the usual 0.25 the activations land where GELU is
still nearly a straight line, `gelu(a) * b` and `gelu(b) * a` agree to a thousandth, and the
end-to-end probe did not notice the two halves being swapped at all. Drawing at 0.7 puts the
activations at the order of one, which is where the released model runs and where its
nonlinearities are nonlinear.

They are also drawn scattered rather than swept, for the reason `docs/indextts_gpt.md` sets out at
length: a swept parameter makes every key in a projection point nearly the same way, and the flat
softmax that follows is an average, which does not care what order its keys arrived in. Attention
is most of what this module is.

## What this does not do

`emo_matrix`, `spk_matrix` and the `qwen0.6bemo4-merge` text model in the release are the
*other* ways 2.5 can be told a feeling: eight named emotions with vectors to interpolate between,
and a small language model that reads an instruction like "say it sadly" and produces a mixture.
This is the path that takes the feeling off the reference recording, which is what
`inference_speech` does when it is given neither.
