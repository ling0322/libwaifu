# The semantic codec

The alphabet the GPT reads and writes, and the way back out of it. MaskGCT's codec, as
IndexTTS-2.5 vendors it.

```rust
// What the pipeline runs: tokens back into features.
let features = semantic_codec::decode(&g, codes, &Config::indextts(), tokens, dtype, device)?;

// What defines the alphabet, and what this release never calls.
let scores = semantic_codec::similarity(&g, features, &Config::indextts(), frames, dtype, device)?;
let tokens = semantic_codec::nearest(&scores, config.codebook_size as usize);
```

## The half that runs is the decoder, which is backwards from the guess

A text-to-speech pipeline sounds like it should encode. It does not. `infer_v2_5.py` calls

```python
S_infer = self.semantic_codec.decode(codes)
```

to turn the GPT's tokens into the features S2Mel's length regulator reads — and the line that
would run the encoder over the reference audio,

```python
# _, S_ref = self.semantic_codec.quantize(spk_cond_emb)
```

is **commented out in the released source**, replaced by `S_ref = self.get_emb(...)`. The
reference recording reaches the length regulator as w2v-bert's continuous 1024-wide features,
never having been near a codebook.

So `decode` is what a package carries and what a reading runs. The encoder stays in this module
because it is the definition of the alphabet — the thing that says what a token *means* — but
nothing here needs it to say a sentence, and `tools/semantic_codec_exporter.py` leaves it out.

An earlier version of this document had it the other way round. It was wrong.

## The shape of it

Encoding: a stride-two convolution halves the frame rate, then twelve ConvNeXt blocks read what is
left, then every frame is projected down to eight numbers and replaced by the nearest of 8192
codebook entries. The token is that entry's index.

Decoding is the mirror, at the same shapes: the entry is looked up, `out_project` widens those
eight numbers back to 1024, **a second backbone with its own weights** — structurally identical to
the encoder's, and not its transpose, because a codec is not an invertible function — reads them,
and then the frame rate is doubled back. One token becomes two frames, which is the ratio it stood
for.

Two asymmetries worth knowing, both of them the release's rather than this port's. `quantize` puts
a GELU after `down`; `decode` puts none after `out_project`. And the doubling is
`F.interpolate(scale_factor=2, mode="nearest")` followed by a width-three convolution — a repeat
*is* nearest interpolation at exactly two, so the graph concatenates the sequence with itself
along a new last axis and flattens it, which puts every frame beside its own copy.

A ConvNeXt block is a depthwise convolution seven frames wide that mixes no channels, then two
linears that mix channels and no frames. That separation is why this needs `depthwise_conv1d`
rather than a grouped `conv1d` — see `docs/audio.md` for why those are not the same thing on a
card.

## The nearest entry is found on the host

`flint` can take a maximum but cannot say *where* it was: there is no argmax. So the graph
computes the similarity — `(T, 8) @ (8, 8192)` — and `nearest` walks the result on the host.

That is less of a compromise than it sounds. Both sides are L2 normalized, which makes the
Euclidean distance `2 - 2 * cosine`, so the nearest entry is the largest dot product and the
whole search is one matrix multiply the graph is good at. What comes back is `T * 8192` floats
once per utterance, and walking them is short beside the twelve ConvNeXt blocks in front.

## What is checked

`waifu/tests/semantic_codec.rs`, against the reference — `tools/semantic_codec_reference.py`
downloads it and runs it. Twelve ConvNeXt layers as shipped, every width cut: a 32-wide feature
instead of 1024, a 64-entry codebook instead of 8192.

The search gets its own test rather than resting on the codes this small model picks. With
sixty-four entries and made-up weights it picks the same one for every frame, which would pass
against almost any indexing mistake. So the *similarity* is what checks the encoder, and
`nearest` is checked against a matrix whose answer is obvious by construction — including a row
that is negative throughout, which catches an implementation that starts its search from zero,
and a tie, which the earlier index has to win because that is what `torch.max` does.

```bash
cargo test --manifest-path waifu/Cargo.toml --test semantic_codec
.venv/bin/python tools/semantic_codec_reference.py
```

## A note on the reference harness

`vocos.py` imports two mel helpers from torchaudio at module scope, for a class `VocosBackbone`
never touches. torchaudio is not installed in this venv and should not be — it pins itself
against a torch version, and this venv's pins are what hold several models' reference tensors
steady. So `_torchaudio_shim` writes the two functions into the download cache instead, with
their real arithmetic rather than a stub that raises: a shim that lied would be worse than none.

## What is missing

The exporter. `codec.pth` is 607 MB and nothing here fetches it yet. The decoder half, which
training needs and inference does not.
