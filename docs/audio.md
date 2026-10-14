# Audio primitives

Convolution along one axis, a short time Fourier transform, a mel filterbank, a resampling and a
vocoder's activation -- everything a speech model asks for that a diffusion model never did.

```rust
use waifu::audio::{hann_window, stft, stft_basis, Padding};

let window = hann_window(1024);
let (real, imaginary) = stft(
    &g, wave, g.constant(stft_basis(1024, &window)?),
    1024, 256, Some(Padding::Reflect), DType::Float16, Device::Cuda,
)?;
```

Runs on every backend, needs no feature flag, and adds no kernel to `flint`.

## None of it is a kernel

A 1-D convolution is the one operator in that list: `Graph::conv1d`, which the CPU, CUDA, Metal
and Vulkan all implement, groups included. The obvious reading of the rest is more operators to
write three times over, once for the processor, once for CUDA and once for Metal. That is not
what is here. None of them is irreducible, and each one is written as the operators `flint`
already had:

| primitive | what it actually is |
| --- | --- |
| `conv_transpose1d` | one `matmul`, then `ceil(kernel / stride)` shifted additions |
| `stft` | `conv1d` against a bank of windowed sinusoids |
| `istft` | `conv_transpose1d` against the matching bank, divided by the window's own overlap |
| `mel_filterbank` | a `matmul` |
| `resample` | a `conv_transpose1d` that stuffs zeros, a `conv1d` that low-passes, a stride |
| `upsample1d`, `downsample1d` | the same pair at a fixed ratio, against a Kaiser-windowed sinc |
| `snake` | arithmetic |

The one thing that was added is `sin` and `cos`, which the CPU and CUDA kernels already
implemented and the C interface simply did not export.

What this buys is three things. It runs wherever `conv1d` and `matmul` run, which is every
backend, on the day it is written rather than three ports later. It is correct wherever they are
correct, and they are tested. And a fused kernel can replace any one of these later without a
caller noticing, because what a caller sees is the function and not the seven nodes behind it.

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

## The operators exist even though no backend does

Being a composition is how these are *computed*; it is not what they *are*. The operator set is
what the library says it can do, and a 1-D convolution should not be invisible there because it
happens to be assembled in Rust today.

So `convTranspose1d`, `snake`, `stft` and `istft` are declared in `Operators`, dispatched through
`F::`, exported by the C interface, and reachable from a graph as a single node. No backend
implements any of them, so such a node fails when the graph runs and the composition is what a
model should call. `conv1d` began as the fifth; every backend has since implemented it, and the
composition that stood in for it is gone. What the declarations buy is that a backend which writes one of these
kernels overrides a method rather than restructuring its callers, and that the contract it has to
meet -- shapes, layouts, what `centered` means, which of `alpha` and `beta` arrives already
exponentiated -- is written down where the kernel will go.

### They throw rather than abort

The sixty-one unimplemented operators already in `Operators` use `NOT_IMPL()`, which is
`LOG(FATAL)` followed by `abort()`. That is defensible for an operator some backend implements --
reaching it means asking the wrong device -- but these four have no implementation anywhere, so
every call would end the process.

They use `THROW(NotImplemented, ...)` instead. `lut::NotImplementedError` already existed; the C
interface's `guard` catches it like any other, and what comes back to Rust is an ordinary error
naming the operator and pointing at the composition that does work. That is also what makes them
testable: an operator that kills the process cannot be asked whether it is wired up.

### What is not an operator

- **`depthwiseConv1d`** -- depthwise is `conv1d` with `groups == in == out`, and one operation
  should not have two names at the interface. A backend wanting the specialized path tests
  `groups` against the channel count, which is what CUDA's does.
- **`mel_filterbank`, `resample`, `pad1d`** -- a constant built on the host and applied with
  `matmul`; two convolutions back to back; and `cat` against a zeros tensor. Nobody would write a
  kernel for any of the three.

## What is checked

`waifu/tests/audio.rs` is sixteen tests on the processor, and none of them compares against a
recorded output. A composition is exactly the kind of thing that goes wrong in a way that still
runs and still produces a plausibly shaped tensor, so each test writes the definition out as a
loop over indices, in the least clever way available, and asks whether the graph agrees:

- `Graph::conv1d` against the sum it is defined to be, over six shapes: padded, strided, dilated
  to four, grouped, and depthwise.
- `conv_transpose1d` against the scatter it is defined to be, including a kernel that is not a
  whole number of strides and the 16-over-8 a vocoder upsamples with.
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

## Grouped convolution

`flint`'s CUDA `conv2d` is CUTLASS's and throws on any group count above one. A grouped 1-D
convolution does not go through it: the `conv1d` node has a CUTLASS path of its own that takes
groups. The depthwise case -- `groups == in_channels == out_channels`, which is every grouped
convolution in IndexTTS-2.5 -- has a CUTLASS kernel of its own, and any other group count runs as
one ungrouped convolution per group.

### The upsampling either side of a snake is not `resample`

`upsample1d` and `downsample1d` are the pair a BigVGAN's anti-aliased activation is built on, and
they are not `resample` at a ratio of two. The window is a Kaiser one whose shape is derived from
the transition width rather than a Hann one, the padding is by replication rather than zeros, and
the trimming is asymmetric -- which is what makes the output exactly `ratio` times as long. All
three of those are `alias_free_activation/torch/resample.py`, and swapping in the other filter
changes the audio. Hence two resamplings here rather than one with a window argument.

`Padding::Replicate` arrived with them, and can pad by more than the signal is long, which
`Padding::Reflect` cannot.

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

With a grouped `conv1d`, all six rows above run on the card.

(One caveat on the table: `s2mel/modules/wavenet.py` also defines a `DDSConv` that is depthwise.
It is the VITS duration predictor's layer and does not appear to be on this model's path, but it
is in the file.)

## What this does not do yet

These are the primitives. What sits on top of them is now written, and what joins them up is not.

All six of IndexTTS-2.5's models -- about 5.5 GB between them -- are here, each with a document of
its own and a test against the implementation it came from:

| piece | |
| --- | --- |
| GPT backbone, 1280d × 24L | `waifu::indextts::gpt`, and [indextts_gpt.md](indextts_gpt.md) -- with the generation loop on top of it |
| semantic codec, 8192 entries | `waifu::indextts::semantic_codec`, and [semantic_codec.md](semantic_codec.md) |
| S2Mel, flow-matching DiT 13L × 512d | `waifu::indextts::s2mel`, and [s2mel.md](s2mel.md) |
| BigVGAN vocoder | `waifu::indextts::bigvgan`, and [bigvgan.md](bigvgan.md) |
| w2v-bert-2.0 | `waifu::indextts::w2v_bert`, and [w2v_bert.md](w2v_bert.md) |
| CAMPPlus | `waifu::indextts::campplus`, and [campplus.md](campplus.md) |

What remains is not a model. It is the emotion conformer and perceiver that produce `emo_vec`
when a recording is not handed one; a `.pth` to safetensors exporter for four of the six, and so
a package; a `.tiktoken` vocabulary the `tokenizers` crate does not read; a text frontend for five
languages; and the pipeline that runs the six in order behind a
[`Voice`](../waifu/src/speech.rs). Until that last one exists, the speech the program offers is still
the stand-in described in [speech.md](speech.md).

None of it is blocked on a kernel, and since `conv1d` took groups none of it is blocked on the CUDA
convolution either. Nor was any of the six: the vocoder, which was written first, needed two
compositions, one padding mode and nothing in `flint`, and the loop on the GPT needed no operator
that was not already here -- only the causal mask's alignment to the bottom right, which is what
lets one query read a whole history.
