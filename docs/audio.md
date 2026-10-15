# Audio primitives

Convolution along one axis and its transpose, a padding, a short time Fourier transform and its
inverse, a vocoder's activation, a clamp, a mel filterbank and a resampling -- everything a speech
model asks for that a diffusion model never did.

```rust
use waifu::audio::{halves, hann_window};
use waifu::flint::PadMode;

let window = g.constant(Tensor::from_f32(&[1024], &hann_window(1024))?.to_device(device)?);
let spectrum = g.stft(wave, window, 1024, 256, true);
let (real, imaginary) = halves(&g, spectrum, 1024);
```

The first seven are `flint` operators, each one node of a graph. The last two, and the numbers
every filter is made of, are `waifu::audio`.

## The operators

| node | CPU, CUDA and Vulkan | Metal |
| --- | --- | --- |
| `Graph::conv1d` | a kernel (CUTLASS on CUDA, groups included) | a kernel |
| `Graph::conv_transpose1d` | one GEMM, then a col2im kernel | one `matmul` and `ceil(R / stride)` shifted additions per group |
| `Graph::pad1d` (zero, reflect, replicate) | a kernel | `copy`, `fill`, and a `matmul` against the anti-diagonal for a mirror |
| `Graph::snake` | a kernel, one pass | `mul`, `sin`, `square`, `divTensor`, `add` |
| `Graph::clamp` | a kernel; a NaN stays a NaN | the `max` and `min` reductions of each element beside its bound, which pick rather than compute, and `exp(-\|x\|) * 0` added to keep a NaN |
| `Graph::stft` | a reflection `pad1d` and a `conv1d` against a bank of windowed sinusoids | the same |
| `Graph::istft` | a `conv_transpose1d` against the inverse bank, divided by the window's overlap in float32 whatever the type | the same |

Metal is the one device without these kernels today, and it loses nothing: the compositions live
in the base class, `flint/functional.cc`, so a backend without a kernel for one of these runs it out of the operators
it does have -- `matmul`, `copy`, `fill`, the elementwise family -- and overriding the method is
all it takes to give it a kernel later. Nothing above `Operators` can tell which it got.

These were compositions in `waifu::audio` until they were operators. Each was about six to
twenty-five graph nodes, and a vocoder runs them a lot: BigVGAN applies a snake inside every one of
109 anti-aliased activations and a transposed convolution in each of them, and the composed snake
was six passes over the tensor where the kernel is one.

### Why a transposed convolution is a GEMM

Read the operation as it is defined rather than as a convolution. Input position `l` contributes
`weight[:, :, r]` to output position `l * stride + r - padding`, for every `r`. Every one of those
contributions at once is `(N * L, C) @ (C, K * R)` -- `F::transposedConvolutionColumns`, which goes
through the device's own `matmul` and so its tuned GEMM. What is left is adding up the ones that
landed on the same place.

The CPU, CUDA and Vulkan do that in one pass, col2im: a thread per output position gathers the taps
congruent to it modulo the stride, `ceil(R / stride)` of them -- two for every vocoder here -- and
adds the bias. The composition gets there without a kernel: `l * stride + r` and `l' * stride + r'`
collide only when `r` and `r'` are a whole stride apart, so cutting the kernel into stride-long
pieces makes each piece collision-free within itself, and the whole a sum of that many shifted
copies.

### What the transform costs

A bank of sinusoids makes an STFT an O(N²) matrix multiply per frame where a radix-2 transform is
O(N log N). At the 1024-point window these models use that is about a hundredfold more arithmetic.

It is still the right trade here: the arithmetic is shaped like a GEMM, which is the one shape this
library is fastest at, and ten seconds of audio is a couple of GFLOP. The vocoder is what costs in
a speech pipeline, not the transform in front of it. The banks are built on the host from the
window the node is handed, which means reading the window back from the device once per call. A
backend with an FFT -- cuFFT, vDSP -- overrides `stft` and `istft`, and nothing above them changes.

### `pad1d` and `clamp` are nodes of their own

A clamp is not a `Unary`, because it carries two bounds, and a padding is not a view, because it
writes. So each is an `Op` of its own, `Op::Pad1d` and `Op::Clamp`, run through `F::pad1d` and
`F::clamp` and the C interface's `fl_pad1d` and `fl_clamp` -- the path `fl_conv_transpose1d`,
`fl_snake`, `fl_stft` and `fl_istft` take too, and every other operator.

## What is left in `waifu::audio`

What a model *designs* rather than computes:

- `hann_window`, `mel_filterbank` (HTK or Slaney, area-normalized or not), `resample_kernel` and
  `kaiser_sinc_filter` -- the numbers a filter is made of, worked out on the host once;
- `magnitude`, `apply_filterbank` and `halves`, which put two operators together the way every mel
  spectrogram here does;
- `resample`, a `conv_transpose1d` that stuffs zeros and a `conv1d` that low-passes and strides;
- `upsample1d` and `downsample1d`, the pair a BigVGAN's anti-aliased activation is built on.

### What is not an operator

- **`depthwiseConv1d`** -- depthwise is `conv1d` with `groups == in == out`, and one operation
  should not have two names at the interface. A backend wanting the specialized path tests
  `groups` against the channel count, which is what CUDA's does.
- **`mel_filterbank`, `resample`** -- a constant built on the host and applied with `matmul`, and
  two convolutions back to back. Nobody would write a kernel for either.
- **A maximum of two tensors.** Whisper's log mel raises everything to eight below its loudest
  value, a bound that is a value of the graph rather than a number, and `clamp` takes numbers. It
  is written `floor + relu(x - floor)` where it is used.

## What is checked

Each operator is checked against its own definition, written out as a loop over indices in the
least clever way available, and never against a recorded output -- an operator is exactly the
kind of thing that goes wrong in a way that still runs and still produces a plausibly shaped
tensor.

- `flint/cpu/signal_test.cc`: the CPU kernels *and* the compositions, each against the definition
  -- a transposed convolution with groups, output padding and a kernel shorter than the stride; a
  snake with and without beta; every padding mode, including a replication wider than the signal;
  a clamp on either side; an STFT against a direct DFT of each frame, and an ISTFT round trip.
- `flint/cuda/signal_test.cc`: the CUDA kernels against the CPU in float and half, and the
  compositions run on the GPU -- which is where reading the window back and moving the banks over
  is exercised before Metal relies on it.
- `flint/vulkan/operators_test.cc`: the Vulkan kernels in float and half against the CPU, and
  the compositions run on the same device -- what Metal relies on, tested from Linux.
- `waifu/tests/audio.rs`: every node from a graph, through `Op`, `ir` and the C interface, against
  its definition: `conv1d` over six shapes, `conv_transpose1d`, the paddings, the snake, the clamp,
  `stft` against a DFT, `istft` of `stft` back to the signal, a mel spectrogram against the
  filterbank times the modulus, and a resampled sine.
- `waifu/tests/audio_cuda.rs`: the same graphs on a card against the host. `#[ignore]`d the way
  every CUDA test here is:

```bash
./build/unittest "[signal]"
cargo test --manifest-path waifu/Cargo.toml --test audio_cuda -- --ignored --test-threads=1
```

All of it together is well under a second. Nothing here loads a model.

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

`PadMode::Replicate` arrived with them, and can pad by more than the signal is long, which
`PadMode::Reflect` cannot.

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

All six of IndexTTS-2.5's models -- about 5.5 GB between them -- are here, each with a test against
the implementation it came from, and described in [indextts.md](indextts.md):

| piece | |
| --- | --- |
| GPT backbone, 1280d × 24L | `waifu::indextts::gpt` -- with the generation loop on top of it |
| semantic codec, 8192 entries | `waifu::indextts::semantic_codec` |
| S2Mel, flow-matching DiT 13L × 512d | `waifu::indextts::s2mel` |
| BigVGAN vocoder | `waifu::indextts::bigvgan` |
| w2v-bert-2.0 | `waifu::indextts::w2v_bert` |
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
