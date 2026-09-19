# The semantic codec

w2v-bert's continuous features in, one discrete token per two frames out — the alphabet the GPT
reads and writes. MaskGCT's codec, as IndexTTS-2.5 vendors it.

```rust
let scores = semantic_codec::similarity(&g, features, &Config::indextts(), frames, dtype, device)?;
let tokens = semantic_codec::nearest(&scores, config.codebook_size as usize);
```

At inference only half of it runs. `quantize` is the encoder and the codebook lookup; the decoder
is for training and is not ported.

## The shape of it

A stride-two convolution halves the frame rate, then twelve ConvNeXt blocks read what is left,
then every frame is projected down to eight numbers and replaced by the nearest of 8192 codebook
entries. The token is that entry's index.

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
