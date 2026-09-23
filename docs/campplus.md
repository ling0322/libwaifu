# CAMPPlus

The speaker encoder IndexTTS-2.5 conditions on: 80-band Kaldi filterbank energies of the
reference audio in, one 192-dimensional vector out saying whose voice it is.

```rust
use waifu::indextts::campplus::{self, Config};

let embedding = campplus::graph(&g, features, &Config::indextts(), frames, dtype, device)?;
```

3D-Speaker's model under Apache 2.0, published as `funasr/campplus` and shared by several speech
systems — nothing about it is IndexTTS's. Note the width: IndexTTS builds
`CAMPPlus(feat_dim=80, embedding_size=192)`, not the class's own default of 512.

## Two halves

A small 2-D convolutional **front end** reads the spectrogram as a one-channel picture and walks
down the frequency axis, halving it three times, then folds what is left of frequency into the
channels — so a spectrogram becomes a sequence of `32 * feat_dim / 8` channels.

A densely connected **time-delay network** reads that sequence. Three blocks, twelve then
twenty-four then sixteen layers deep, and every layer concatenates its own output onto everything
before it, so the width grows by `growth_rate` per layer and a transition halves it again between
blocks. A statistics pooling turns however many frames there were into a mean and a standard
deviation, and one projection turns those into the embedding.

## What it needed that was not there

Nothing new in `flint`, but two pieces are worth pointing at.

### Batch normalization is not an operator

In `eval` a `BatchNorm` is a fixed per-channel affine and nothing more — the statistics are frozen
into the checkpoint. So `tools/campplus_exporter.py` folds each one to the two vectors that
describe it,

```
scale = weight / sqrt(running_var + eps)
shift = bias - running_mean * scale
```

which is exact rather than an approximation, and `campplus::batch_norm` is a multiply and an add.
The vectors are stored shaped `(C, 1)`, or `(C, 1, 1)` for the 2-D ones, because that is what
broadcasts against `(N, C, L)` and `(N, C, H, W)` **without a transpose** — `flint`'s binary
operations expand a size-one axis, so a per-channel vector needs no moving at all.

The normalization in `DenseLayer` is `affine=False` and has no weight to be the scale; the
statistics carry it instead, and it folds the same way.

### A stride down one axis only

Every downsampling in the front end is `stride=(2, 1)`: halve frequency, leave time alone.
`flint`'s `conv2d` takes one stride and applies it to both axes, so this is not expressible
directly.

`halve_frequency` convolves at stride one and then keeps every second row, by viewing `F` as
`(F / 2, 2)` and taking index zero of the new axis. Those are exactly the rows a strided
convolution would have centred on, which is what makes the two equal rather than merely similar.
It costs twice the arithmetic on an axis that is at most eighty long, which is not where this
model's time goes.

### The segment pooling

`CAMLayer` gates its local convolution by what the whole utterance and each hundred-frame segment
look like. The segment average is an ordinary windowed mean, except that the last window is short
— `ceil_mode` — and is averaged over the frames it actually has, so the divisor is a vector. Then
each window's value is stretched back across the frames it covered, which is a matrix multiply by
a row of ones: there is no operator that repeats an element, and this needs none.

## What is checked

`waifu/tests/campplus.rs`, in the fast suite, against 3D-Speaker's own implementation. The
constants come from `tools/campplus_reference.py`, which downloads the reference and runs it —
not a second opinion written from the same paper, which would agree with a bug as readily as with
the model.

No package and no download. Every parameter is filled from its own name, by the same FNV-1a hash
and the same sine in both languages, so the two sides need no file between them and cannot
disagree about the order they walk the model in.

The model under test is small — 16 bands, 16 initial channels, an 8-dimensional embedding — but
every *count* is the release's, because depth is what a dense block's bookkeeping gets wrong: an
off-by-one in the running width is invisible at layer one and fatal at layer twelve. The input is
420 frames so that the segment pooling sees three segments with a short one at the end, which is
the case `ceil_mode` exists for; anything shorter leaves one segment and tests nothing.

Three tests: the front end on its own, the embedding end to end, and one that asks the graph which
weights it wants and requires exactly the 573 the exporter writes. The last is there because a
numeric failure says only that something is wrong, where a missing layer should say which.

```bash
cargo test --manifest-path waifu/Cargo.toml --test campplus
.venv/bin/python tools/campplus_reference.py        # to regenerate the constants
```

## The real weights

```bash
.venv/bin/python tools/campplus_exporter.py -output models/campplus-192.safetensors
```

573 tensors, 6.85 M parameters. Not wired to a test yet: what exists is the graph and the
exporter, and an `#[ignore]`d test that runs the released checkpoint against a reference
embedding is the obvious next thing.
