# BigVGAN

The last model in a speech pipeline and the only one whose output is audio: a mel spectrogram in,
a waveform out.

```rust
use waifu::bigvgan::{BigVgan, BigVganConfig};

let vocoder = BigVgan::build(
    BigVganConfig::v2_22khz_80band_256x(), "", &weights, Device::Cuda, DType::Float,
)?;

let wave = vocoder.forward(&mel)?;   // (1, 80, frames) -> (1, 1, frames * 256)
```

`nvidia/bigvgan_v2_22khz_80band_256x` is the release IndexTTS-2.5 fetches on first run: 112 M
parameters, 80 mel bands, 22.05 kHz out, and one mel frame for every 256 samples. Nothing about
it is IndexTTS's -- it is NVIDIA's, published separately under the MIT licence and shared by
several speech models, which is why it is `waifu::bigvgan` and not `waifu::indextts::vocoder`.

## It adds no operator

Six stages of transposed convolution, each followed by three residual blocks whose outputs are
averaged, and a hundred and nine anti-aliased snakes. Every one of those is a function in
[`waifu::audio`](audio.md), and every one of those is `conv2d`, `matmul` and arithmetic. So this
model ran on all three backends the day it was written, and a card that can draw can also speak.

What it did need was two things that module did not have, both of them in the activation:
`Padding::Replicate`, and the Kaiser-windowed sinc pair `upsample1d` and `downsample1d`.

## The activation is the model

A BigVGAN is a HiFi-GAN with the activation replaced, and everything interesting is in the
replacement.

**It is periodic.** `x + sin(alpha x)^2 / beta` -- a *snake*. A ReLU has to learn that speech is
made of things that repeat; this is told. `alpha` sets the frequency and `beta` the depth, one
pair per channel, and they are trained.

**It is anti-aliased.** A periodic function applied at the signal's own rate generates components
above half that rate, and there is nowhere for them to go but back down into the audible band,
folded. So each activation upsamples by two, applies the snake at the higher rate where those
components fit, and filters on the way back down. That is the AMP -- anti-aliased
multi-periodicity -- in the block's name.

The resampling is the fiddly part, and it is fiddly in a way that is entirely about edges:

- The filter is a twelve-tap Kaiser-windowed sinc at a cutoff of a quarter and a transition width
  of 0.3, and the Kaiser beta is derived from those by the standard design formula. It is not
  [`resample_kernel`](audio.md), which is the Hann-windowed one that converts between two sample
  rates. The two do the same job with different windows and are not interchangeable.
- Padding is by *replication*, not zeros and not a mirror -- and by enough that the taper a
  transposed convolution leaves on each end lands entirely in the padding, which is then cut off.
  The upsampling pads by `kernel / ratio - 1` and trims `pad * ratio + (kernel - ratio) / 2` off
  the front and one more off the back when the kernel is odd. Those asymmetric halves are the
  reference's, and getting one of them wrong shifts the whole waveform by a sample.
- The reference convolves with `groups = C` and the same taps in every group. That is not a
  grouped convolution: reading `(N, C, L)` as `(N * C, 1, L)` makes each channel an item in a
  batch and one ungrouped convolution does all of them. Which is why neither this nor the
  transposed convolution needed the group count `audio::conv_transpose1d` does not have.

### What it costs

This is where the vocoder's time goes, and it is worth being plain about it.

The snake is written as arithmetic -- about six passes over the tensor plus a copy -- where a
fused kernel would read it once and write it once. There are eighteen of them per upsampling
stage, on tensors that have already been lengthened, and each one runs at *twice* the rate
because of the anti-aliasing. NVIDIA ships a CUDA kernel for exactly this
(`alias_free_activation/cuda`), which is what `use_cuda_kernel=True` reaches, and the fact that
they bothered is the measurement.

So if this model turns out to be slow, the activation is where to look first, and
`audio::snake` together with the two resamplings is the thing to replace. Replacing them changes
nothing above: what a caller sees is `BigVgan::forward`.

## The weights are the checkpoint's, folded

Every convolution in the reference is wrapped in `weight_norm`, which stores a direction and a
magnitude and multiplies them at each call. `IndexTTS2.__init__` calls `remove_weight_norm()` as
it loads the vocoder, so the folded weight is what the reference actually runs;
`tools/bigvgan_exporter.py` writes that product and the graph reads it.

Names are otherwise untouched -- `conv_pre`, `ups.0.0`, `resblocks.3.convs1.1`, `conv_post`,
`activations.2.act.alpha` -- so a tensor in the file and a load in the graph can be compared
directly when one of them is wrong.

The snake's `alpha` and `beta` are stored as logarithms and raised where they are read, which is
what the reference does and what keeps the package a copy rather than a transformation. It costs
one node over a vector as long as the channel count.

## How it is checked

`waifu/tests/bigvgan.rs` runs against numbers that came out of NVIDIA's implementation, not out
of a second reading of the paper. `tools/bigvgan_reference.py` downloads `bigvgan.py`, builds a
small model -- 8 mel bands, two upsamplings, 24 samples out -- and prints what it produced as
Rust constants.

Small, but not simplified: it has both kinds of convolution, four AMP blocks, dilations, and
seventeen anti-aliased snakes, which is every distinct thing the 112 M parameter release does. And
it is in the *fast* suite, because neither side reads a weight from the other -- both fill every
parameter from a hash of its own name, in float64 rounded once, so the two models are equal bit
for bit with no file passing between them. A vocoder that stops matching BigVGAN is noticed in
the same minute it stops, rather than the next time someone runs the ignored tests.

The tests that do need the real 460 MB checkpoint are `#[ignore]`d in the same file, and what
they compare is a chirp's mel spectrogram through the whole thing:

```bash
.venv/bin/python tools/bigvgan_exporter.py \
    -output models/bigvgan-22khz-80band.safetensors \
    -test_output models/bigvgan-22khz-80band_test.safetensors

cargo test --release --manifest-path waifu/Cargo.toml --test bigvgan -- --ignored
```

## What is not here

`resblock: "2"`, the cheaper AMP block with one convolution per dilation instead of two. No
released BigVGAN v2 uses it, and a configuration asking for it is refused rather than quietly run
as the other kind.

A kernel for the snake, and the fused CUDA activation that goes with it. See what it costs, above.

And the five models above this one. `docs/audio.md` lists what IndexTTS-2.5 still needs; with the
vocoder written, what is left is the GPT backbone, the semantic codec, S2Mel, the Qwen3 emotion
model, and the two encoders that are not in the model's own repository.
