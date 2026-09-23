# IndexTTS-2.5's GPT

Text and a voice in, the semantic tokens S2Mel reads out. The largest of the models here and the
one the rest of the pipeline is arranged around — and what it generates is neither audio nor mel,
but the codec's alphabet, one token per two frames of the speech to come.

## It is an ordinary GPT-2, fed unusually

Twenty-four layers, 1280 wide, twenty heads: a stock `GPT2Model`. Two things are taken out of it.

Its **token embedding is unused** — what goes in is embeddings this module assembles, not ids.

Its **position embedding is replaced by zeros**. The reference deletes `gpt.wpe` and puts
`null_position_embeddings` there. The positions this model uses are added earlier, by a learned
table applied to the text and to the mel separately. A reader who assumes GPT-2's own `wpe` is in
play gets a model that is wrong in every layer and still returns a tensor of the right shape,
which is why the backbone has a test of its own.

## What the prefix is made of

```
[ zero padding ][ speaker + emotion ][ 0 ][ 0 ][ text tokens ][ start mel ]
```

Three conditioning rows — one carrying the speaker vector from [CAMPPlus](campplus.md) plus an
emotion vector, then two rows of zeros that are not learned and carry nothing — and then the
text, each token embedded and given both a position and a language. The padding is on the
**left**, so generation always begins the same distance from the end.

## Two shapes that are not what they look like

**`Conv1D` is not `Linear`.** HuggingFace's GPT-2 uses its own `Conv1D`, whose weight is
`(in, out)` where `nn.Linear` stores `(out, in)`. `conv1d_linear` reads it as stored and does not
transpose. Using the ordinary `Linear` layer here would load the right numbers in the wrong order
and produce plausible nonsense. Note that `mel_head` *is* an `nn.Linear` and does transpose — the
two conventions sit ten lines apart.

**The activation is the tanh approximation.** `gelu_new`, not the exact GELU `flint` has. The two
differ by about a thousandth, which is small enough to look like noise in one layer and not small
enough to survive twenty-four. `gelu_new` writes it out: `0.5x(1 + tanh(√(2/π)(x + 0.044715x³)))`.

## What is checked

`waifu/tests/indextts_gpt.rs`, in the fast suite. **No download at all** — `GPT2Model` is in the
transformers this venv already pins, so `tools/indextts_gpt_reference.py` imports the reference
rather than fetching it, and zeroes `wpe` to match what the reference does by deleting it.

Every width is cut and so is the layer count, because a GPT-2 stack has no bookkeeping that turns
on its depth. Three probes kept apart so a failure says which part: the stack, the head on top,
and the embeddings in front.

```bash
cargo test --manifest-path waifu/Cargo.toml --test indextts_gpt
.venv/bin/python tools/indextts_gpt_reference.py
```

## The generation loop

`Gpt` is the model with weights behind it and the loop on top: two graphs, compiled once and run
many times.

| | reads | keeps |
| --- | --- | --- |
| `Gpt::prefill` | conditioning, text, start token — the whole prefix at once | every layer's keys and values |
| `Gpt::step` | one token | the same, one position longer |

`Gpt::generate` is the loop over the two, with `Sampling` deciding what to draw and when to stop.
Both are public, so a caller that wants a sampler this crate does not offer can run them itself.

### Why it is not paged attention

An earlier draft of this page said the cache should be built on `paged_attention` and
`store_kv_cache`, after the `KVCacheManager` pattern in the author's `libllm`. It is not, for two
reasons.

Those two are **eager** operators — `op.rs` says so in as many words, because an operation whose
result is a change to somebody else's tensor has no way to say that in a graph. A stack built on
them could not be one compiled `Ir`. And `paged_attention` rides on the FlashAttention kernels,
so a build without them has no paged attention at all and no portable fallback: the fast suite,
which is where this is tested, runs on the CPU.

What is here instead is the cache as ordinary graph values. A step concatenates its own keys onto
the history and attends over the lot, and `flint`'s causal mask is **aligned to the bottom right
of the score matrix** — so one query against a longer history sees all of it with `causal` left
true, and the same block serves the prefill and the step with no second path written. The
concatenation copies the history once per step, which is the same order of memory traffic the
attention beside it was always going to spend reading it.

What the cache saves is the rest: twenty-four layers of projections and a head eight thousand
wide, over the whole prefix, at every one of up to 1815 steps.

### Two places it is easy to be wrong

The head scores the **last position only**, taken before the matmul rather than after. A prefill
over a six-hundred-token sentence would otherwise compute six hundred rows of an eight-thousand-
wide projection and sample one of them.

The start token is a mel token at **position zero**, so the first token generated is at position
one. A loop that counts its own output from zero puts every token one place from where the model
was trained to find it — and that is the quiet one, because the reading still sounds like speech.

### What the loop is checked against

The three probes above compare against GPT-2. The loop cannot: there is no reference constant for
a reading. What a cached step has to satisfy instead is that it says what the whole sequence says,
read in one pass — an invariant that needs no second implementation to state.

Two things that cost time and are worth knowing before touching those tests:

- **The scores, not the tokens.** A token is an argmax, and an argmax over a forty-token alphabet
  survives a great deal of being wrong. Every mutation this was written against — the history
  concatenated on the wrong side, a mel position off by one, the prefix assembled in the wrong
  order — left the argmax alone and moved the scores.
- **The weights have to scatter.** `Filled` sweeps one sine per parameter, so every row of a
  projection samples the same curve, every key it produces points nearly the same way, and the
  softmax over them is flat. A flat softmax is an average, and an average does not care what order
  its keys arrived in: the whole cache test passed with the history concatenated backwards. The
  loop tests draw their weights by xorshift instead. Nothing there is compared against Python, so
  the only thing the numbers have to be is different from one another.

### And against the released weights

`tools/indextts_gpt_exporter.py` writes the 301 tensors this graph reads out of the release's
`gpt.pth` — and, since it is the same checkpoint, the 153 more that
[the emotion path](indextts_emotion.md) reads. It renames and folds nothing: both modules were
written against the checkpoint's own names, so a tensor that turns out to be wrong can be compared
with the one it came from. Of the 456 in the file, only `text_head` is left behind.

```bash
.venv/bin/python tools/indextts_gpt_exporter.py \
    -checkpoint ~/.cache/libwaifu/indextts25/gpt.pth \
    -output models/indextts25-gpt.safetensors
.venv/bin/python tools/indextts_gpt_real_reference.py

cargo test --release --manifest-path waifu/Cargo.toml \
    --test indextts_gpt -- --ignored --test-threads=1
```

That test is worth having for one reason the fast ones cannot cover: **weights made up from their
own names agree with any transposition of themselves that is self-consistent**, and the `Filled`
source answers every shape it is asked for. `lang_embedding` is the case that actually happened —
it was declared with nine rows against a table that has a hundred and seven, every fast test
passed, and only the real checkpoint said otherwise. 107 is `len(LANGUAGE_DICT) + 1`: Whisper's
list of 106 languages, most of which this model was never trained to say.

## Where `emo_vec` comes from

`prefill` takes the finished 1280-wide emotion vector directly, which is the path
`inference_speech` takes when it is given one. When it is not, the four stages that make one out
of the reference recording are [`waifu::indextts::emotion`](indextts_emotion.md) —
`emo_conditioning_encoder` → `emo_perceiver_encoder` → `emovec_layer` → `emo_layer`, 153 of the
checkpoint's tensors, exported by the same `tools/indextts_gpt_exporter.py` that writes this
model.

## What is still missing

- **Typical sampling**, which the reference offers and `flint` has no operator for. It is left out
  rather than approximated with one that is nearby.
