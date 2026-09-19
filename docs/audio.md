# Audio primitives

Convolution along one axis, a short time Fourier transform, a mel filterbank, a resampling and a
vocoder's activation -- everything a speech model asks for that a diffusion model never did.

```rust
use waifu::audio::{conv1d, hann_window, stft, stft_basis, Padding};

let window = hann_window(1024);
let (real, imaginary) = stft(
    &g, wave, g.constant(stft_basis(1024, &window)?),
    1024, 256, Some(Padding::Reflect), DType::Float16, Device::Cuda,
)?;
```

Runs on every backend, needs no feature flag, and adds no kernel to `flint`.

## None of it is a kernel

The obvious reading of that list is six operators to write three times over, once for the
processor, once for CUDA and once for Metal. That is not what is here. None of the six is
irreducible, and each one is written as the operators `flint` already had:

| primitive | what it actually is |
| --- | --- |
| `conv1d` | `conv2d` over an image one pixel tall, plus the padding `conv2d` cannot express |
| `depthwise_conv1d` | `R` shifted slices, each scaled per channel, added up |
| `conv_transpose1d` | one `matmul`, then `ceil(kernel / stride)` shifted additions |
| `stft` | `conv1d` against a bank of windowed sinusoids |
| `istft` | `conv_transpose1d` against the matching bank, divided by the window's own overlap |
| `mel_filterbank` | a `matmul` |
| `resample` | a `conv_transpose1d` that stuffs zeros, a `conv1d` that low-passes, a stride |
| `snake` | arithmetic |

The one thing that was added is `sin` and `cos`, which the CPU and CUDA kernels already
implemented and the C interface simply did not export.

What this buys is three things. It runs wherever `conv2d` and `matmul` run, which is every
backend, on the day it is written rather than three ports later. It is correct wherever they are
correct, and they are tested. And a fused kernel can replace any one of these later without a
caller noticing, because what a caller sees is the function and not the seven nodes behind it.

### The padding is the whole trick of `conv1d`

`conv2d` takes a square padding, so a 1-D convolution cannot simply be a 2-D one over an
`(N, C, 1, L)` view: padding by `p` would make the height `2p + 1` and the output the wrong shape.
So `conv1d` pads the time axis itself, explicitly, and asks `conv2d` for none. Stride and dilation
need no such care -- a stride over one row leaves one row, and dilating a kernel one tall dilates
nothing.

### Why a transposed convolution is a matrix multiply

Read the operation as it is defined rather than as a convolution. Input position `l` contributes
`weight[:, :, k]` to output position `l * stride + k`, for every `k`. Every one of those
contributions at once is `(N, L, C) @ (C, K * R)`, and what is left is adding up the ones that
landed on the same place.

Which ones those are is fixed: `l * stride + k` and `l' * stride + k'` collide only when `k` and
`k'` are a whole stride apart. So cutting the kernel into `ceil(R / stride)` pieces of `stride`
each makes every piece collision-free within itself, and the whole operation a sum of that many
shifted copies. A BigVGAN's kernel is twice its stride, so it is two additions.

### What the transform costs

A bank of sinusoids makes an STFT an O(N²) matrix multiply per frame where a radix-2 transform is
O(N log N). At the 1024-point window these models use that is about a hundredfold more arithmetic.

It is worth saying plainly rather than hiding, and it is still the right trade here: the
arithmetic is shaped like a GEMM, which is the one shape this library is fastest at, and ten
seconds of audio is a couple of GFLOP. The vocoder is what costs in a speech pipeline, not the
transform in front of it. If that ever stops being true, `stft` is the function to replace, and
replacing it changes nothing above it.

## Reflect padding, without an operator that reads backwards

`torch.stft(center=True)` pads by mirroring, and every mel spectrogram in a speech model was
computed that way. Mirroring needs a dimension read in reverse, and nothing here does that.

It does not need to. A reversal is a permutation, a permutation is a matrix, and a matrix multiply
is the operator this library is built on: `reversal(n)` is the `n` by `n` anti-diagonal, and
`x @ reversal(n)` is `x` backwards. The two edge slices are written with constant bounds --
negative ones count from the back -- so neither needs the graph to know how long the signal is.

## What is checked

`waifu/tests/audio.rs` is fourteen tests on the processor, and none of them compares against a
recorded output. A composition is exactly the kind of thing that goes wrong in a way that still
runs and still produces a plausibly shaped tensor, so each test writes the definition out as a
loop over indices, in the least clever way available, and asks whether the graph agrees:

- `conv1d` against the sum it is defined to be, over six shapes: padded, strided, dilated to four,
  grouped, and depthwise.
- `conv_transpose1d` against the scatter it is defined to be, including a kernel that is not a
  whole number of strides and the 16-over-8 a vocoder upsamples with.
- `depthwise_conv1d` against `conv1d` with one group per channel, which is the thing it has to
  be a substitute for, at the 31 taps a conformer uses.
- `stft` against a direct discrete Fourier transform of each windowed frame.
- `istft` of `stft` back to the signal, which is the test that catches what the two banks disagree
  about -- a factor of `n_fft`, a bin doubled that should not have been, an overlap-add off by a
  hop.
- a mel spectrogram against the filterbank times the modulus, computed from the signal.
- a resampled sine, still that sine at the new rate and the same amplitude.

`waifu/tests/audio_cuda.rs` runs the same graphs on a card and compares them against the host,
because "it works on every backend" is either true on a second backend or it is not true at all.
Those are `#[ignore]`d the way every CUDA test here is:

```bash
cargo test --manifest-path waifu/Cargo.toml --test audio_cuda -- --ignored --test-threads=1
```

Both files together are well under a second. Nothing here loads a model.

## Grouped convolution, and why it stopped being a blocker

A grouped convolution. `flint`'s CUDA `conv2d` is CUTLASS's, `conv2d_cutlass.cu` throws on any
group count above one, and `cuda/conv2d.cc` calls it unconditionally:

```
InvalidArg: conv2d on CUTLASS: a grouped convolution is not implemented
```

This is not something the composition can paper over -- a grouped convolution is not a sum of
ungrouped ones at any reasonable node count, and a depthwise one over 512 channels least of all.
It is also not a build flag: `conv2d_cudnn.cc` beside it *does* set a group count, and cuDNN
supports this perfectly well, but that file is compiled into the benchmark rather than the
runtime, so reaching it means changing `conv2d.cc`.

There were two ways out, both of them a decision about `flint`'s CUDA convolution rather than
about audio: wire the cuDNN path into `cuda/conv2d.cc` and build with `WITH_CUDNN=ON`, or
implement the grouped case in `conv2d_cutlass.cu`.

Neither turned out to be necessary, because of what these models actually ask for. Every grouped
convolution in IndexTTS-2.5 is **depthwise** -- `groups == in_channels == out_channels`, one
channel per group and no channel multiplier -- and depthwise, unlike the general case, *is* a
composition:

```text
out[n, c, t] = sum over k of  weight[c, k] * x[n, c, t + k * dilation - padding]
```

Read as a sum over taps rather than over channels, that is `R` shifted copies of the input, each
scaled by one number per channel and added together. A shift is a slice, a per-channel scale is a
broadcast multiply. `depthwise_conv1d` is those `3R` nodes, and it runs wherever they do.

It is not free: `R` elementwise passes over the tensor rather than one fused kernel that reads it
once, so roughly `R`-fold the memory traffic for the same arithmetic. At a conformer's 31 taps
that is real, and it is still small beside the attention and the feed-forward in the same block.
A fused depthwise kernel would be the thing to write if it ever shows up in a profile; nothing
above `depthwise_conv1d` would have to change.

What remains genuinely unavailable on a card is a group count strictly between one and the
channel count. Nothing here uses one.

### What IndexTTS-2.5 does not use

`istft`. Its waveform comes out of BigVGAN, straight from a mel spectrogram, and the semantic
codec reconstructs w2v-bert features rather than audio -- `infer_v2_5.py` has no inverse transform
on any path. The forward `stft` is used, for the mel a vocoder is conditioned on.

It stays because an inverse transform is half of what a short time Fourier transform is for, a
Vocos *head* in some other model is exactly this, and `an_istft_gives_back_the_signal_an_stft_was_taken_of`
is the test that proves the forward bank is right. But it is not on this model's path, and a
reader deciding what to port first should know that.

### Who actually needs a group count

A conformer's convolution module, and only that. Checked against the reference implementation
rather than assumed:

| module | grouped? |
| --- | --- |
| `indextts/gpt/conformer_encoder.py`, the emotion conditioning encoder | **yes** -- `groups=channels` |
| w2v-bert-2.0, `Wav2Vec2BertConvolutionModule` | **yes** -- `groups=hidden_size` |
| BigVGAN, `bigvgan.py` | no |
| WaveNet postnet, the `WN` class | no |
| CAMPPlus, `DTDNN.py` and `campplus/layers.py` | no |
| S2Mel's diffusion transformer | no |

Both of the two are on the unconditional path. `infer_v2_5.py` builds a `Wav2Vec2BertModel` in
its constructor and runs it on the reference audio for every synthesis, and the emotion
conditioning encoder runs on every call too -- with no emotion prompt given it falls back to the
speaker audio rather than being skipped. So neither can be avoided by asking for less.

With `depthwise_conv1d` in place, all six rows above run on the card.

(One caveat on the table: `s2mel/modules/wavenet.py` also defines a `DDSConv` that is depthwise.
It is the VITS duration predictor's layer and does not appear to be on this model's path, but it
is in the file.)

## What this does not do yet

These are the primitives, not a speech model. What sits on top of them is not written.

For IndexTTS-2.5 specifically -- six models and about 5.5 GB -- what remains is roughly:

| piece | needs |
| --- | --- |
| GPT backbone, 1280d × 24L | the attention, RoPE and sampling that are already here, plus a conformer perceiver conditioner |
| semantic codec, 8192 entries | a Vocos *backbone*, which reconstructs w2v-bert features rather than audio -- no inverse transform |
| S2Mel, flow-matching DiT 13L × 512d | a WaveNet postnet, which is dilated `conv1d` |
| BigVGAN vocoder | `conv_transpose1d`, `snake`, and the anti-aliased resampling around them |
| Qwen3-0.6B emotion model | autoregressive generation; `anima::text_encoder` is the same architecture read a different way |
| w2v-bert-2.0, CAMPPlus | two encoders that are not even in the model's own repository -- they are fetched at runtime |

And outside the model graph: reading and writing audio files, a `.pth` to safetensors exporter for
four checkpoints, a `.tiktoken` vocabulary the `tokenizers` crate does not read, and a text
frontend for five languages.

None of that is blocked on a kernel, and since `depthwise_conv1d` none of it is blocked on the
CUDA convolution either.
