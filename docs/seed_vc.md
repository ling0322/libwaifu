# Seed-VC v2

Somebody speaking and a recording of somebody else in; the first one's words, in the second one's
voice, out, at 22.05 kHz. [`Plachtaa/seed-vc`](https://github.com/Plachtaa/seed-vc)'s v2 model --
`v2/cfm_small.pth` and `v2/ar_base.pth` from `Plachta/Seed-VC`, with the ASTRAL tokenizers from
`Plachta/ASTRAL-quantization` -- under **GPL-3.0**, both the code and the weights.

```bash
cargo run --release --features gpl --example convert -- models/seed_vc.yaml source.wav voice.wav out.wav
cargo run --release --features gpl --example convert -- models/seed_vc.yaml source.wav voice.wav out.wav --style --seed 7
```

```rust
let vc = SeedVc::from_manifest(Device::Cuda, Residency::Device, &manifest)?;
let voice = vc.listen(&recording)?;                                    // once per voice
let sound = vc.convert(&source, &voice, &vc.conversion(), &mut report)?;
```

## Licence: GPL-3.0, behind a feature

Seed-VC and ASTRAL are GPL-3.0, and this port is written from their source, so it is GPL-3.0 too
-- `waifu/src/seed_vc/`, its test, the `convert` example and `tools/seed_vc_*.py`, each marked
`SPDX-License-Identifier: GPL-3.0-only`, with the licence in `LICENSE-GPL-3.0`. Everything else in
libwaifu is MIT.

It is compiled only with the `gpl` feature -- `-DENABLE_GPL=ON` to CMake -- which is off by
default. Without it none of the GPL code is built and what comes out is MIT; with it, the build as
a whole is covered by the GPL -- the same arrangement as FFmpeg's `--enable-gpl`. The weights are GPL-3.0 as
well, and are redistributed under it: the exported package carries the licence in its card.

## Where it runs

On the web page, as the speech2speech task, in a build with the `gpl` feature:

```bash
cargo run --release --manifest-path waifu/Cargo.toml --features cli,gpl -- -m seed-vc
```

The page takes a recording to convert (the first five minutes of it) and a recording of the voice
to convert it to (the first thirty seconds), decoded in the browser from whatever format it can
play; the steps, whether to convert the style too, and a seed. The bar counts the CFM's steps and
the vocoder's windows across the whole recording, and a run stops after the step it is on.

The page reaches the port through `waifu::Converter`, a trait in the MIT `speech` module that
`SeedVc` implements; the webui opens a `SeedVc` in one function that is compiled only with `gpl`.
A default build still knows the name `seed-vc`, and refuses it, before fetching anything, with the
reason. The speech2speech task itself is in every build, run by
[CosyVoice3](cosyvoice3.md#voice-conversion), which converts the voice but not the style.

The package is published as `ling0322/libwaifu-seed-vc` on Hugging Face and ModelScope, and
`waifu -m seed-vc` fetches it; or build it yourself, below.

## Building the package

```bash
PYTHONPATH=~/.cache/libwaifu/torchaudio-stub \
    .venv/bin/python tools/seed_vc_exporter.py -output models/seed_vc.safetensors
```

What comes out is `models/seed_vc.yaml` and two weight files, 553 M parameters, all float32:

| namespace | from | |
| --- | --- | --- |
| `seed_vc.hubert` | `facebook/hubert-large-ll60k` | the first 18 of 24 layers, weight norm folded |
| `seed_vc.wide`, `seed_vc.narrow` | `bsq2048_light.pth`, `bsq32_light.pth` | ConvNeXt V2 and BSQ's projection |
| `seed_vc.cfm`, `seed_vc.cfm_regulator` | `v2/cfm_small.pth` | the DiT and its rotary table; the token regulator |
| `seed_vc.ar`, `seed_vc.ar_regulator` | `v2/ar_base.pth` | the AR and its rotary table; the narrow tokens' embedding |
| `seed_vc.campplus` | `funasr/campplus` | by `campplus_exporter.py`, as for IndexTTS |
| `seed_vc.bigvgan` | `nvidia/bigvgan_v2_22khz_80band_256x` | by `bigvgan_exporter.py`, as for IndexTTS |

Upstream's code is fetched into `~/.cache/libwaifu/src/seed-vc`, pinned to `51383ef`, by
`tools/seed_vc_reference.py`, which the exporter imports for the AR's rotary table.

## The order

**Listening, once per voice.** The reference is brought to 22.05 kHz, cut to 25 seconds, and
brought from there to 16 kHz.

| | reads | gives |
| --- | --- | --- |
| [HuBERT-large](../waifu/src/seed_vc/hubert.rs), 18 layers | the 16 kHz audio | `(1, T, 1024)`, fifty frames a second |
| [ASTRAL](../waifu/src/seed_vc/astral.rs), wide and narrow | HuBERT's features | 2048- and 32-code tokens, one a frame |
| [CAMPPlus](campplus.md) | Kaldi energies of the 16 kHz audio | who is speaking, `(1, 192)` |
| the 22.05 kHz mel | the 22.05 kHz audio | the prompt the CFM continues |
| the CFM's regulator | the wide tokens | the prompt's condition, one row a mel frame |

**Converting.** The source is heard the same way, tokens only, and then:

- **Timbre only**, the default. The source's wide tokens are laid onto its own mel frames, so its
  timing is kept. The [CFM](../waifu/src/seed_vc/dit.rs) draws the mel after the reference's
  prompt -- thirty steps on a cosine schedule, guided twice over at 0.7 -- in windows of thirty
  seconds less the prompt, each overlapping the last by sixteen frames. BigVGAN sounds each
  window and the overlaps are crossfaded.
- **Style too**, `convert_style` / `--style`. Both speakers' narrow tokens have their runs
  collapsed, and the [AR](../waifu/src/seed_vc/ar.rs) continues the reference's wide tokens with
  the source's content, so accent and pacing follow the reference. The rest is as above.

## Things that are not what a reader would guess

**v2's DiT is not S2Mel.** IndexTTS's [S2Mel](s2mel.md) is Seed-VC v1's. This one has no WaveNet
and no skips; the timestep and the speaker go in front of the sequence as two extra tokens, and
its adaptive norms are DiT's six-way shift, scale and gate.

**Both rotary tables are read out of the package.** Upstream builds them in float32 and keeps them
in bfloat16, and the weights were trained with the rounded values; the exporter writes them as
upstream held them.

**ASTRAL's GRN normalizes over time.** ConvNeXt V2's response normalization, ported to one
dimension, takes the norm of each channel over the frames. A token depends on the whole piece it
was heard in, and long audio is heard in 30-second pieces overlapping by five, as upstream hears it.

**The AR is scored through `output`, not the embedding.** The configuration ties them; the
checkpoint's `output` differs, and generation uses it.

**The repetition penalty lands on the first token only.** Upstream hands its sampler
`previous_tokens[0]`. At the default penalty of 1.0 it does nothing either way; above it, this
does what upstream does.

**The style path's pieces are crossfaded as if they overlapped.** Only a source long enough to
need two readings of the AR meets it, and it loses about 0.19 s at the seam, as upstream does.

## What is checked

`waifu/tests/seed_vc.rs`, ignored, against a fixture `tools/seed_vc_reference.py` records by
running upstream's own classes through `convert_voice_with_streaming` on one of upstream's example
pairs (GLaDOS into Teio, six seconds each), in float32 on the CPU.

| stage | against upstream |
| --- | --- |
| HuBERT's 18th layer | 2.4e-6 of the largest value |
| wide and narrow tokens | 100% on both recordings |
| CAMPPlus, reference mel | 7.9e-6, 8.2e-5 |
| the CFM's regulator | 2.5e-6 of the largest value |
| one DiT call, all three guided rows | 1.2e-6 of the largest value |
| thirty steps from upstream's noise | 5.3e-5 largest, 1.4e-6 mean |
| the AR, teacher-forced 48 steps | 1.1e-5 in the scores; every argmax |
| BigVGAN | 88 dB |

End to end, CAMPPlus puts the conversion as close to the reference as upstream's is -- 0.81 both,
timbre only; 0.84 against upstream's 0.80 with style -- and Whisper large-v3 transcribes the
timbre conversion word for word as it does upstream's.

```bash
cargo test --release --manifest-path waifu/Cargo.toml --features gpl --no-fail-fast \
    --test seed_vc -- --ignored --test-threads=1
cargo test --manifest-path waifu/Cargo.toml --features gpl --lib seed_vc   # the unit tests
```

## What is not upstream's

The random numbers -- the CFM's noise and the AR's draws -- and the resampler: upstream reads
through `librosa.load` and soxr, this through the windowed sinc in
[`indextts::features`](indextts_features.md). The stage tests hand the runtime the samples upstream
held, so a resampler difference cannot hide in a model's error.

## Still missing

The anonymization mode (`anonymization_only`, which reads the AR without a reference), and fp16.
