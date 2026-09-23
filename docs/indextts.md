# IndexTTS-2.5

A few seconds of somebody speaking and a sentence in; the sentence, in that voice, out.

```bash
waifu draw -voice models/indextts25.yaml          # the page's text2speech tab
cargo run --release --example speak -- models/indextts25.yaml voice.wav "Hello there." out.wav
```

```rust
let tts = IndexTts::from_manifest(Device::Cuda, Residency::Device, &manifest)?;
let voice = tts.listen(&recording)?;               // once per voice
let sound = tts.say("今天天气很好。", &voice, &options, &mut report)?;
```

Seven models, each with a module and a document of its own. This one is about the order they run
in, the shapes between them, and the handful of things upstream does in the gaps that no single
module owns.

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
parameters, all of them float32. Nothing is narrowed -- upstream runs S2Mel and the vocoder with
autocast off, and a vocoder is exactly where half a precision's error becomes audible -- so the
package is about the size of the release.

Every export was written against the list of tensors its graph actually loads, which
`cargo run --release --example parameters -- <model>` prints, and verified back against it.

## The order

**Listening, once per voice.** The recording is brought to 22.05 kHz and cut to fifteen seconds --
upstream reads it with `librosa.load`, whose default rate that is -- and from there to 16 kHz.

| | reads | gives |
| --- | --- | --- |
| [w2v-bert](w2v_bert.md) | the 16 kHz [features](indextts_features.md), 160 wide | `(1, T, 1024)`, standardized |
| [CAMPPlus](campplus.md) | Kaldi energies of the 16 kHz audio | who is speaking, `(1, 192)` |
| [the emotion path](indextts_emotion.md) | w2v-bert's features | how they sound, `(1, 1280)` |
| the 22.05 kHz mel | the 22.05 kHz audio | the prompt, `(1, 80, M)` |
| S2Mel's length regulator | w2v-bert's features, stretched onto `M` frames | the prompt's condition, `(1, M, 512)` |

**Saying, once per sentence.** The text is cleaned, normalized, prefixed with its language, cut
into segments of at most 120 tokens, and each segment goes through:

1. [The GPT](indextts_gpt.md) -- speaker, emotion and text in, semantic tokens out.
2. [The codec's decoder](semantic_codec.md) -- the tokens back to 1024-wide features, two per token.
3. The length regulator -- those stretched onto 1.72 mel frames each, 22 050 / 256 frames a
   second over the codec's fifty, divided by the speed.
4. [S2Mel](s2mel.md) -- twenty-five Euler steps with guidance at 0.7, the prompt's mel held in
   front and cut off the result afterwards.
5. [BigVGAN](bigvgan.md) -- the mel to 22.05 kHz audio.

Segments are joined with 200 milliseconds of silence, as upstream's `interval_silence` does.

## What upstream does in the gaps

- **Punctuation is replaced before anything reads the text.** `char_rep_map` turns `，` into `,`,
  `。` into `.`, and every bracket and quotation mark into an apostrophe. This is not cosmetic:
  without it a Chinese sentence reaches the GPT ending in a `。` it has hardly seen, and it keeps
  talking -- Whisper heard "今天天气很好，我们一起去公园散步吧" and then "法尔". With it, three seeds out
  of three said the sentence and stopped. The table is applied with upstream's own
  first-match-wins semantics, which is why `，，，` becomes three commas and not an ellipsis.
- **Ids 0 and 1 are taken out of the text.** `prepare_gpt_inputs` drops every start and stop token
  before putting one of each back, and those are also ordinary tokens of the vocabulary. So a
  character that encodes to either is silently not read -- by upstream, and so here.
- **The emotion comes from the same recording.** Upstream takes a separate emotion recording when
  given one and falls back to the speaker's; `merge_emovec` with the two the same is the speaker's
  own emotion vector.

## Where this differs from upstream, knowingly

- **No beam search.** Upstream samples inside a three-way beam search. This draws one sequence with
  the same temperature, top-k, top-p and repetition penalty.
- **The language is guessed from the script.** Upstream is told it. Kana is Japanese, any other CJK
  character is Chinese, everything else English -- wrong for Spanish, which is what
  `IndexTts::say_in` is for. The page has no language box yet.
- **The emotion cannot be set.** Upstream can also take eight named emotions as a vector, or read
  one out of an instruction with a small Qwen model. Neither is here; the voice sounds the way its
  recording felt.
- **w2v-bert's features drop an odd frame at the end** where `SeamlessM4TFeatureExtractor` pads it
  and masks it out. Twenty milliseconds, at the edge, masked either way.

## What it took from `flint`

Running in float32 on a card found three things the half-precision models never had:

- **`lookup` failed to launch past 65 535 ids.** The ids sat on the grid's y and z axes, which hold
  that many blocks each; w2v-bert asks for one per *pair* of frames, a quarter of a million for ten
  seconds. The grid is one-dimensional now, one thread per output element.
- **Reductions were half-only, and rank-three at most.** `sum` over a float32 tensor, or over any
  tensor of rank four, stopped with "not implemented". The kernel was always generic over the type;
  the dispatch was not. It reads any rank as `(rows, last)` now, with the rows on the x axis for
  the same reason as `lookup`.
- **The repetition penalty never ran on a card at all.** It was written for half logits, and the
  GPT casts its logits to float before penalizing them; it allowed a history of at most sixty-four
  tokens; and a token said twice was two threads racing on one score. It is a gather and then a
  scatter now, as the host's is, so a repeated token is penalized once.

Each has a CUDA test of its own under `flint/cuda/`.

## What is checked

| | |
| --- | --- |
| `waifu/tests/indextts_features.rs` | the front end, resampling and the 22.05 kHz mel against upstream's own code |
| `indextts::tests` | segment packing, punctuation against upstream's own regular expression, language ids |
| `waifu/tests/indextts.rs` | the whole pipeline on real weights -- ignored, see below |

The whole-pipeline test needs the package and two recordings:

```bash
PYTHONPATH=~/.cache/libwaifu/pydeps .venv/bin/python tools/indextts_reference_voices.py
cargo test --release --manifest-path waifu/Cargo.toml --test indextts -- --ignored --nocapture
```

It asks, in one pass: that a sentence comes out as a sound of a sentence's length; that it sounds
more like the speaker it was given than like another one, by CAMPPlus's own embedding; that one
seed says it identically twice; that a stop comes back with nothing; and that asking with no
recording is a sentence saying why.

What no test here can check is whether the words are right, since Whisper is not a thing this crate
runs. That was checked by hand while this was written, and the numbers are worth keeping:

- **Whisper heard every sentence exactly**: an English one, a Chinese one on three seeds, and a
  24-second passage that crossed several segments -- numbers included, "25 dollars and 99 cents"
  coming back as "$25.99". One Chinese sentence came back with 界面 heard as the near-homophone 见面.
- **It is the voice it was given.** Upstream's CAMPPlus, over three speakers from LibriSpeech:
  0.85, 0.85 and 0.89 between each output and its own recording, 0.32 to 0.63 against the others.
  Two real recordings of one speaker sit at 0.84.
- **A seed repeats**, across processes, byte for byte.

On an RTX 5060 Ti: the package reads in about a second and a half, a recording is listened to in
under a second, and a four-second sentence takes four seconds -- about one for the GPT and three for
S2Mel's fifty passes and the vocoder.
