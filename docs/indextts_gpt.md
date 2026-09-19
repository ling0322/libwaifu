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

## What is missing: the generation loop

This is the prefill — every position at once. Turning it into speech needs the autoregressive
loop: one token at a time, each attending to everything before it, until `stop_mel_token` or
`max_mel_tokens`.

Done naively that is 1815 forward passes over a growing prefix, which is quadratic and not worth
writing. What it wants is a **key-value cache**, and the pattern for one is already in this
author's `libllm`: `llm/src/kv_cache.rs` has `KVCacheSpec` and a `KVCacheManager` that owns block
pools per layer, and `flint` here already exports the two operators they are built on —
`paged_attention` and `store_kv_cache`, with `paged_attention_available()` to say whether the
build has them.

So the remaining work is wiring, not invention:

1. a cache sized from the model's shape — twenty-four layers, twenty heads, head width 64;
2. a prefill that stores the prefix's keys and values rather than discarding them;
3. a step that runs one position against the cache through `paged_attention`;
4. sampling on top — `sample_with_params` and `repetition_penalty` are both already bound, and
   the reference also offers typical sampling, which is not.

Also missing: the exporter, and the emotion conformer and perceiver that produce `emo_vec` when
it is not supplied. `inference_speech` takes `emo_vec` directly, so that path can be fed from
outside while it is unwritten.
