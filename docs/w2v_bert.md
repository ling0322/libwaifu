# w2v-bert-2.0

The conformer that reads speech into the features everything else conditions on. Meta's model,
shared by several speech systems; IndexTTS-2.5 runs it over the reference audio and hands the
result to [the semantic codec](semantic_codec.md).

## Sixteen layers of twenty-four

IndexTTS reads `hidden_states[17]` — the projection plus sixteen layers. **The last eight are
never used.** That is a third of a 580 M parameter model that does not have to run, or be
exported, at inference. `Config::USED_LAYERS` is that sixteen, and `graph` takes the count.

Stopping partway has its own test, because "the stack is right" and "reading it early lands in
the right place" are different claims and only the second is what `hidden_states[17]` rests on.

## A conformer layer is four things, and two of them are halved

Feed forward, attention, convolution, feed forward, then a normalization. The two feed forwards
are added back at **half weight** — `x * 0.5 + residual`, which is what "macaron" means. It is
not a detail: at full weight the residual stream doubles through every layer.

## Two things worth pointing at

### The depthwise convolution is causal

Every other depthwise convolution in this repository pads symmetrically. This one pads
`kernel - 1` on the left and nothing on the right, so a frame never sees the future.
`depthwise_conv1d` takes one padding for both sides, so the padding happens in front of it and
the convolution is asked for none.

### The positions are relative keys, not rotary

`position_embeddings_type` is `relative_key`: the score between two frames gets a term that
depends only on how far apart they are, clamped to sixty-four back and eight forward — so there
are seventy-three embeddings and the rest of the distance is thrown away.

Which embedding applies to which pair is a table of indices built on the host, and `lookup` turns
it into the `(T, T, head_dim)` the scores need. That makes the gather an operator that exists.
The term itself is one matrix multiply per query frame, with the frame as the batch axis:
`(T, B*H, D)` against `(T, D, T)`.

The `(T, T, head_dim)` is the memory to watch. It is the same for every layer and could be built
once and shared if that ever matters; at present each layer builds its own.

## What is checked

`waifu/tests/w2v_bert.rs`, in the fast suite. **No download of source or of weights** —
`Wav2Vec2BertModel` is in the pinned transformers already in the venv, so
`tools/w2v_bert_reference.py` imports the reference rather than fetching it.

Every width is cut, and so is the layer count — four rather than twenty-four — because a
conformer stack has no bookkeeping that turns on its depth: layer *n* reads layer *n-1* and that
is all. What is kept is the shape of one layer, which is where this model is unusual.

Three probes: the projection alone, two layers, and four.

```bash
cargo test --manifest-path waifu/Cargo.toml --test w2v_bert
.venv/bin/python tools/w2v_bert_reference.py
```

## What is missing

The exporter, and the feature extractor in front of the model — `SeamlessM4TFeatureExtractor`,
which turns audio into the 160-wide stacked mel features this reads. Padding masks: the reference
masks padded frames out of both the attention and the depthwise convolution, and a batch of one
at full length needs neither.
