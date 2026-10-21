# IndexTTS-2.5

A few seconds of somebody speaking and a sentence in; the sentence, in that voice, out.

```bash
cargo run --release -p waifu-tools --bin speak -- models/indextts25.yaml voice.wav "Hello there." out.wav
```

```rust
let tts = IndexTts::from_manifest(Device::Cuda, u64::MAX, &manifest)?;
let voice = tts.listen(&recording)?;               // once per voice
let sound = tts.say("今天天气很好。", &voice, &options, &mut report)?;
```

Seven models run in a row. Six are IndexTTS's alone and are described here: w2v-bert, the GPT, the
emotion path, the semantic codec, S2Mel and the BigVGAN vocoder. [CAMPPlus](campplus.md) is shared
with CosyVoice3 and has a document of its own, as do [the audio front end](audio_features.md)
and [the text normalizer](text_normalize.md).

## Building the package

Six exporters, a vocabulary, and one tool that puts them side by side under a namespace each:

```bash
.venv/bin/python tools/indextts_gpt_exporter.py -checkpoint ~/.cache/libwaifu/indextts25/gpt.pth \
    -output models/indextts25-gpt.safetensors
.venv/bin/python tools/w2v_bert_exporter.py -output models/w2v-bert-2.0.safetensors
.venv/bin/python tools/semantic_codec_exporter.py -output models/indextts25-codec.safetensors
.venv/bin/python tools/s2mel_exporter.py -output models/indextts25-s2mel.safetensors
.venv/bin/python tools/campplus_exporter.py -output models/campplus.safetensors
.venv/bin/python tools/bigvgan_exporter.py -output models/bigvgan-22khz-80band.safetensors
PYTHONPATH=~/.cache/libwaifu/pydeps .venv/bin/python tools/indextts_tokenizer_exporter.py \
    -output models/indextts25-tokenizer.json

cd tools && ../.venv/bin/python indextts_package.py -output ../models/indextts25.safetensors -models ../models
```

What comes out is `models/indextts25.yaml`, three weight files and a `tokenizer.json`: 1 367 M
parameters, all float32. Nothing is narrowed: upstream runs S2Mel and the vocoder with autocast
off, and a vocoder is where half a precision's error becomes audible.

Every exporter keeps the checkpoint's own tensor names and was written against the list the graph
loads, which `cargo run --release -p waifu-tools --bin parameters -- <model>` prints. Layers that
upstream wraps in `weight_norm` are exported folded, because upstream calls `remove_weight_norm()`
as it loads them.

## The order

**Listening, once per voice.** The recording is brought to 22.05 kHz and cut to fifteen seconds
-- upstream reads it with `librosa.load`, whose default rate that is -- and from there to 16 kHz.

| | reads | gives |
| --- | --- | --- |
| [w2v-bert](#w2v-bert-20) | the 16 kHz [features](audio_features.md), 160 wide | `(1, T, 1024)`, standardized |
| [CAMPPlus](campplus.md) | Kaldi energies of the 16 kHz audio | who is speaking, `(1, 192)` |
| [the emotion path](#the-emotion-path) | w2v-bert's features | how they sound, `(1, 1280)` |
| the 22.05 kHz mel | the 22.05 kHz audio | the prompt, `(1, 80, M)` |
| [S2Mel's length regulator](#s2mel) | w2v-bert's features, stretched onto `M` frames | the prompt's condition, `(1, M, 512)` |

**Saying, once per sentence.** The text is cleaned, normalized, prefixed with its language, cut
into segments of at most 120 tokens, and each segment goes through:

1. [The GPT](#the-gpt) -- speaker, emotion and text in, semantic tokens out.
2. [The codec's decoder](#the-semantic-codec) -- the tokens back to 1024-wide features, two per
   token.
3. The length regulator -- those stretched onto 1.72 mel frames each (22 050 / 256 frames a second
   over the codec's fifty), divided by the speed.
4. [S2Mel](#s2mel) -- twenty-five Euler steps with guidance at 0.7, the prompt's mel held in front
   and cut off the result afterwards.
5. [BigVGAN](#bigvgan) -- the mel to 22.05 kHz audio.

Segments are joined with 200 milliseconds of silence, as upstream's `interval_silence` does.

## What upstream does in the gaps

- **Punctuation is replaced before anything reads the text.** `char_rep_map` turns `，` into `,`,
  `。` into `.`, and every bracket and quotation mark into an apostrophe. Without it a Chinese
  sentence reaches the GPT ending in a `。` it has hardly seen, and it keeps talking. The table is
  applied first-match-wins, as upstream does, so `，，，` becomes three commas, not an ellipsis.
- **Ids 0 and 1 are taken out of the text.** `prepare_gpt_inputs` drops every start and stop token
  before putting one of each back, and those are also ordinary tokens of the vocabulary, so a
  character that encodes to either is silently not read -- by upstream, and so here.
- **The emotion comes from the same recording.** Upstream takes a separate emotion recording when
  given one and falls back to the speaker's.

## Where this differs from upstream, knowingly

- **No beam search.** Upstream samples inside a three-way beam search; this draws one sequence with
  the same temperature, top-k, top-p and repetition penalty. Typical sampling is not offered:
  `flint` has no operator for it.
- **The language is guessed from the script.** Kana is Japanese, any other CJK character Chinese,
  everything else English -- wrong for Spanish, which is what `IndexTts::say_in` is for.
- **The emotion cannot be set.** Upstream can also take eight named emotions (`emo_matrix`,
  `spk_matrix`) or read one out of an instruction with a small Qwen model. The voice here sounds
  the way its recording felt.
- **w2v-bert's features drop an odd frame at the end** where `SeamlessM4TFeatureExtractor` pads it
  and masks it out. Twenty milliseconds, masked either way.
- **No padding masks anywhere.** Upstream pads batches and carries length masks through w2v-bert,
  the emotion path and S2Mel. One recording and one segment at a time makes every position valid.

## The models

### w2v-bert-2.0

Meta's conformer, which reads speech into the features everything else conditions on.

- **Sixteen layers of twenty-four.** IndexTTS reads `hidden_states[17]` -- the projection plus
  sixteen layers -- so the last eight are neither exported nor run. `Config::USED_LAYERS` is that
  sixteen, and stopping early has a test of its own.
- **Macaron.** The two feed forwards are added back at half weight, `x * 0.5 + residual`.
- **The depthwise convolution is causal**, padding `kernel - 1` on the left and nothing on the
  right -- unlike every other depthwise convolution in this repository.
- **Positions are relative keys**, clamped to sixty-four back and eight forward: seventy-three
  embeddings. The index table is built on the host and `lookup` turns it into `(T, T, head_dim)`,
  which is the memory to watch.

### The GPT

A stock `GPT2Model` -- twenty-four layers, 1280 wide, twenty heads -- fed embeddings this module
assembles rather than ids.

```
[ zero padding ][ speaker + emotion ][ 0 ][ 0 ][ text tokens ][ start mel ]
```

- **Its position embedding is replaced by zeros.** Upstream deletes `gpt.wpe`; positions come from
  learned tables applied to the text and to the mel separately. Leaving GPT-2's own `wpe` in gives
  a model wrong in every layer and right in every shape.
- **`Conv1D` is not `Linear`.** HuggingFace's GPT-2 stores its weights `(in, out)` and is read
  without a transpose; `mel_head` is an `nn.Linear` and is transposed.
- **The activation is `gelu_new`**, the tanh approximation, not the exact GELU. They differ by a
  thousandth per layer, which does not survive twenty-four.
- **The cache is ordinary graph values, not paged attention.** `paged_attention` and
  `store_kv_cache` are eager operators and need the FlashAttention kernels, so a stack built on
  them could be neither one compiled graph nor tested on the CPU. A step concatenates its keys onto
  the history, and `flint`'s causal mask is aligned to the bottom right, so one block serves both
  `Gpt::prefill` and `Gpt::step`.
- **The head scores the last position only**, taken before the eight-thousand-wide projection.
- **The start token is at position zero**, so the first generated token is at position one. Off by
  one here still sounds like speech, which is what makes it easy to miss.

The fast tests compare against GPT-2 imported from the pinned transformers, every width and the
layer count cut, in three probes: the stack, the head, the embeddings. The loop is checked by an
invariant instead -- a cached step says what the whole sequence read in one pass says -- and two
lessons came out of that: compare **scores, not tokens** (an argmax survived every mutation
tried), and draw the weights **scattered** (xorshift), because swept weights make every key point
the same way and the resulting flat softmax does not notice the history being concatenated
backwards.

The released weights have a test too, because weights made up from their own names agree with any
self-consistent transposition: `lang_embedding` was declared with nine rows against a table of 107
(`len(LANGUAGE_DICT) + 1`), every fast test passed, and only the checkpoint said otherwise. The
exporter writes the GPT's 301 tensors and the emotion path's 153 out of the same `gpt.pth`; only
`text_head` is left behind.

### The emotion path

A recording in, the one 1280-wide row that sits beside the speaker's in the GPT's prefix.

| stage | what it is | out |
| --- | --- | --- |
| `emo_conditioning_encoder` | a WeNet conformer: a convolution that halves the length, four blocks | `(T/2, 512)` |
| `emo_perceiver_encoder` | lucidrains' `PerceiverResampler` with **one** latent, so a vector | `(1, 1024)` |
| `emovec_layer`, `emo_layer` | two projections | `(1280)` |

- **One matrix holds 134 M parameters.** `embed.out.0.weight` is `(512, 261632)`: WeNet's
  `Conv2dSubsampling2` convolves the 1024 features as an image and flattens 512 channels by 511
  columns into one projection. The exporter checks the shape, since it is the tensor a reader is
  most likely to "fix".
- **The relative attention has no shift.** WeNet's `rel_shift` is commented out in the released
  source, so the position term is an ordinary matrix multiply.
- **The gated feed forward gates the second half**: `gelu(second) * first`, the other way round
  from `flint`'s `geglu`, so the two slices are written out.
- **It is not macaron**, where w2v-bert is: one feed forward at full weight.
- **The position table is the checkpoint's, rounded through bfloat16.** The released `pos_enc.pe`
  equals the sinusoid cast to `bfloat16`, bit for bit, and upstream runs with it because loading
  overwrites the exact table. So the exporter writes the buffer out (ten megabytes) and checks that
  equality exactly.

The tests run the released `ConformerEncoder` and `PerceiverResampler`, in five probes cut where
upstream cuts. The weights are drawn at 0.7 rather than the usual 0.25: smaller, and GELU is so
nearly linear that swapping the gate's halves went unnoticed.

### The semantic codec

MaskGCT's codec, whose 8192 entries are the alphabet the GPT writes.

- **The half that runs is the decoder.** `infer_v2_5.py` decodes the GPT's tokens into features
  for the length regulator; the line that would quantize the reference recording is commented out.
  So the exporter leaves the encoder out, though the module keeps it, as the definition of what a
  token means.
- **The decoder is a second backbone**, structurally the encoder's, not its transpose. Two
  asymmetries are the release's: `quantize` has a GELU after `down` and `decode` none after
  `out_project`; and the doubling is a nearest interpolation, written as the sequence concatenated
  with itself on a new last axis and flattened.
- **The nearest entry is found on the host.** `flint` has no argmax, so the graph computes the
  similarity -- both sides L2 normalized, so nearest is the largest dot product -- and `nearest`
  walks it. Its own test covers an all-negative row and a tie, which the earlier index wins, as in
  `torch.max`.

The reference harness writes a two-function torchaudio shim into the download cache rather than
installing torchaudio, which would move the venv's pinned torch.

### S2Mel

A flow-matching model: given a partly denoised mel and a time it returns a direction, and
`solve_euler` integrates it. It is conditioned on the semantic tokens (`cond`), the prompt's mel
(`prompt_x`, zeros past it -- what makes it continue a voice), CAMPPlus's `style`, and the time.

A thirteen-layer transformer (RMS norm, rotary positions, SwiGLU, adaptive norms, U-net skips from
half the layers to the other half), then an eight-layer WaveNet.

- **The rotary embedding is interleaved**, pairing `2i` with `2i + 1`, not `flint`'s NeoX `i` with
  `i + D/2`; `rotate` does it out of cosine and sine tables, as Anima does.
- **The content is continuous whatever the configuration says.** `content_type: 'discrete'`
  builds a `cond_embedder`, but `forward` always uses `cond_projection`.
- **The final modulation is `(1 + scale) * norm(x) + shift`**, where every other adaptive norm is
  `weight * norm(x) + bias`.
- **The sampler holds the prompt**: its frames are forced back after every step. **Guidance is two
  passes**, `(1 + rate) * conditioned - rate * unconditioned`, rather than upstream's batch of two.
- **The length regulator resizes the way torch does**, with the scale computed and multiplied in
  float32, because an exact division picks a different frame for about one length in seventy.
- **Mish is written without a logarithm**: `tanh(softplus(x)) = (u² + 2u) / (u² + 2u + 2)` with
  `u = exp(x)`, exact, and safe because every call follows a group norm.

The tests keep every count and cut every width -- thirteen and eight layers, since the skip
connections are a function of depth -- and check the trunk, the denoiser and the regulator against
upstream, plus four closed forms: Mish, a constant velocity integrating to exactly one unit,
guidance as two passes, and no guidance as one.

### BigVGAN

`nvidia/bigvgan_v2_22khz_80band_256x`: 112 M parameters, 80 mel bands in, 22.05 kHz out, 256
samples per frame. NVIDIA's, MIT licensed; it sits under `indextts` because nothing else here runs
it, and moves beside [`waifu::audio`](audio.md) the day something does.

Six transposed convolutions, each followed by three averaged residual blocks, and 109 anti-aliased
snakes. The transposed convolution and the snake are flint operators with CPU, CUDA, Vulkan and
Metal kernels of their own, so it runs on every backend; the alias-free
resampling around each snake is `audio`'s, a padding and two convolutions.

- **The snake is `x + sin(alpha x)^2 / beta`**, applied at twice the rate and filtered back down
  so the harmonics it makes do not fold into the audible band.
- **The filter is a twelve-tap Kaiser-windowed sinc** (cutoff a quarter, transition 0.3), not the
  Hann-windowed `resample_kernel`; the two are not interchangeable.
- **Padding is by replication**, and the upsampling's asymmetric trims are the reference's: get one
  wrong and the waveform shifts by a sample.
- **`groups = C` with the same taps everywhere** is read as `(N * C, 1, L)` and one ungrouped
  convolution.
- **`alpha` and `beta` are stored as logarithms** and raised where read, as upstream does.

The activation is where the vocoder's time goes: about six passes over the tensor where a fused
kernel would read it once. NVIDIA ships a CUDA kernel for exactly this, which is the measurement.
`resblock: "2"` is refused rather than run as the other kind; no v2 release uses it.

## What it took from `flint`

Running in float32 on a card found three things the half-precision models never had, each with a
CUDA test of its own under `flint/cuda/`:

- **`lookup` failed to launch past 65 535 ids**, which w2v-bert reaches in ten seconds. The grid is
  one-dimensional now.
- **Reductions were half-only and rank three at most.** They read any rank as `(rows, last)` now.
- **The repetition penalty never ran on a card**: half logits only, a history of at most
  sixty-four, and a repeated token was two threads racing. It is a gather and a scatter now.

## What is checked

In the fast suite, each against upstream's own code -- imported or downloaded and run, not
transcribed -- with weights filled from a hash of each parameter's name on both sides:

| | |
| --- | --- |
| `waifu/tests/w2v_bert.rs` | the projection, two layers and four |
| `waifu/tests/indextts_gpt.rs` | the stack, the head, the embeddings, and the cached loop |
| `waifu/tests/indextts_emotion.rs` | the sinusoid table and four cuts through the path |
| `waifu/tests/semantic_codec.rs` | the similarity, and `nearest` |
| `waifu/tests/s2mel.rs` | trunk, denoiser, regulator, and four closed forms |
| `waifu/tests/bigvgan.rs` | a small model with every kind of layer the release has |
| `waifu/tests/indextts_features.rs` | the front end and resampling |
| `indextts::tests` | segment packing, punctuation, language ids |

The reference scripts are `tools/<model>_reference.py`, which print the Rust constants.

Ignored, needing the package or a checkpoint: the GPT and BigVGAN against their released weights,
and the whole pipeline, which needs two recordings:

```bash
PYTHONPATH=~/.cache/libwaifu/pydeps .venv/bin/python tools/indextts_reference_voices.py
cargo test --release --manifest-path waifu/Cargo.toml --no-fail-fast \
    --test indextts --test indextts_gpt -- --ignored --test-threads=1
```

The pipeline test asks that a sentence comes out a sentence long, sounds more like its own speaker
than another by CAMPPlus's embedding, repeats exactly on one seed, stops cleanly, and refuses with
a reason when given no recording. Whether the words are right needs Whisper, which was checked by
hand: it heard every sentence exactly, English and Chinese, numbers included ("25 dollars and 99
cents" as "$25.99"), except 界面 once heard as 见面. Upstream's CAMPPlus put each output at 0.85
to 0.89 from its own speaker and 0.32 to 0.63 from the others; two real recordings of one speaker
sit at 0.84.

On an RTX 5060 Ti the package reads in about a second and a half, a recording is listened to in
under a second, and a four-second sentence takes four seconds: about one for the GPT and three for
S2Mel and the vocoder.
