# IndexTTS-2.5's audio front end

What a recording becomes before any of the models sees it.

```rust
use waifu::indextts::features;

let (features, pairs)  = indextts_features::w2v_bert(&wave_16k);  // (pairs, 160)
let (energies, frames) = indextts_features::campplus(&wave_16k);  // (frames, 80)
```

Two analyses, and they are the same analysis twice. Kaldi's filterbank — the same frames, the same
window, the same triangles — differing only in what is done to the result afterwards.

## The two differences, and neither is the obvious one

**The scaling is not one of them.** `SeamlessM4TFeatureExtractor` multiplies the waveform by 2¹⁵
before the transform, because Kaldi's conventions are 16-bit integers; `infer_v2_5.py` hands
CAMPPlus a float waveform in `[-1, 1]` and does not. That looks like it matters and almost never
does: a scale factor is a constant offset once the mel energies are logged, and *both* paths
remove a constant offset afterwards — one by subtracting each band's mean over time, the other by
standardizing each band. The one place it does matter is silence, where `mel_floor` clamps before
the log, so `Analysis::scale` is spelled out for each rather than assumed away.

**What actually differs is the last step.**

| | w2v-bert | CAMPPlus |
| --- | --- | --- |
| scaling | × 2¹⁵ | none |
| after the log | standardized per band, `ddof=1` | per-band mean removed |
| then | stacked in pairs | nothing |
| out | 160 wide at 50 fps | 80 wide at 100 fps |

The stacking is why w2v-bert's input is 160 and not 80: two consecutive frames side by side, which
halves the frame rate going in. An odd frame at the end is dropped, because a pair needs two --
where `SeamlessM4TFeatureExtractor` pads it with a zero frame and masks the pair out of
w2v-bert's attention instead. Twenty milliseconds, at the edge, and masked either way.

## Details that are each individually forgettable

- **The window is Povey's**, which is a Hann raised to 0.85 — not a Hann, not a Hamming — and it
  is **non-periodic**, so the denominator is `n - 1` and not `n`.
- **The frame's mean is removed before pre-emphasis**, not after. Removing it after removes a
  different number.
- **Pre-emphasis is walked backwards** so each sample reads the one before it *as it was*, and
  the first sample is scaled by `1 - 0.97` rather than filtered.
- **The triangles are built in mel space**, not in hertz. The corners are evenly spaced in mel
  either way; what changes is that the *slopes* are linear in mel too, so a filter is symmetric
  on the mel axis. This is `triangularize_in_mel_space`, and it is what `torchaudio` does.
- **`snip_edges`**: a frame that would run off the end is not taken, and nothing is padded.
- The deviation is the **sample** one, dividing by `n - 1`. That is `torch.var`'s default and not
  `numpy.var`'s.

Kaldi's mel scale and HTK's are the same function, incidentally — `2595 / ln(10)` is 1127.01 —
and even that last digit cancels, because the scale is only used to space corners linearly between
two of its own values. A constant factor divides out of every slope.

## Why this is on the host and not in a graph

[`waifu::audio`](audio.md) puts a short-time Fourier transform in the graph, and this does not use
it. The reason is the three things Kaldi does to a frame *before* it is windowed: the mean is
removed, a pre-emphasis filter is run along it, and that filter reads the sample before the one it
is writing. None of that is a matrix multiply, all of it is per-frame, and expressing it as graph
nodes would be several operators invented for one caller.

What it costs is nothing worth measuring. Fifteen seconds of speech is about 1,500 frames of 512
points; the transform is radix-2 and the whole analysis is tens of milliseconds, against a GPT
that will spend seconds deciding what to say.

## What is checked

`waifu/tests/indextts_features.rs`, against `SeamlessM4TFeatureExtractor` — the extractor
`infer_v2_5.py` itself calls, which is in the pinned transformers and needs no download.

```bash
cargo test --manifest-path waifu/Cargo.toml --test indextts_features
.venv/bin/python tools/indextts_features_reference.py
```

A front end deserves its own test more than most things do, because **nothing downstream of it can
tell you it is wrong**. A filterbank off by one bin, a periodic window, a pre-emphasis in the wrong
order — each produces a feature of exactly the right shape, full of plausible numbers, and the
first thing that notices is a voice that sounds like somebody else.

Six plausible bugs were written in one at a time and each fails: a periodic window (0.07 out), a
plain Hann (0.26), pre-emphasis walked forwards (0.96), the mean removed after pre-emphasis (6.0),
the population deviation (0.03), and triangles in hertz (1.9) — against a tolerance of 2e-3.

The CAMPPlus half's reference is built from `transformers.audio_utils` rather than torchaudio,
which is not installed here and should not be: it pins itself against a torch version, and this
venv's pins hold several models' reference tensors steady. The extractor's own docstring says it
is computing "mel-filter bank features using TorchAudio", so the two agree by construction.

## Resampling, and the other mel

Two more things a recording goes through, which the pipeline needed and which live here for the
same reason the filterbank does.

**`resample`** is `torchaudio.functional.resample`'s Hann-windowed sinc, six zero crossings wide,
in polyphase form -- one short dot product per output sample against one of `to / gcd` kernels.
[`waifu::audio`](audio.md) has a resampler too, and it is the textbook shape: stuff zeros up to the
common multiple, filter, keep every `down`th sample. For 44.1 kHz to 16 kHz that common multiple
is 160 times the input, and fifteen seconds of speech becomes a hundred million samples through a
filter fourteen thousand taps long, almost all of it computing samples that are thrown away.

**`reference_mel`** is the 22.05 kHz mel S2Mel conditions on -- `ref_mel` in `infer_v2_5.py`. It
is not Kaldi's: BigVGAN's `mel_spectrogram`, the signal reflected at each end, a *periodic* Hann,
the magnitude rather than the power, Slaney's filterbank, `ln(max(x, 1e-5))`.

Both are checked against upstream's own code: the resampler against torchaudio's two functions,
lifted out of its `functional.py` with `ast` and run, to 1e-5; the mel against IndexTTS's own
`s2mel/modules/audio.py`, imported. The mel's input is a sweep with noise under it, because a
sweep alone is narrowband and leaves most of a mel spectrogram on the `1e-5` floor, where any two
implementations agree whatever they do.
