# Fun-CosyVoice3

A few seconds of somebody speaking and a sentence in; the sentence, in that voice, out, at 24 kHz.
[`FunAudioLLM/Fun-CosyVoice3-0.5B-2512`](https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512),
under Apache 2.0.

```bash
waifu -m cosyvoice                                  # the text2speech page
cargo run --release -p waifu-tools --bin speak -- models/cosyvoice3.yaml voice.wav "你好。" out.wav
cargo run --release -p waifu-tools --bin speak -- models/cosyvoice3.yaml voice.wav "你好。" out.wav 7 "录音里说的话。"
```

```rust
let tts = CosyVoice3::from_manifest(Device::Cuda, u64::MAX, &manifest)?;
let voice = tts.listen(&recording)?;                               // once per voice
let sound = tts.say("今天天气很好。", &voice, &options, &mut report)?;  // no transcript
let sound = tts.say_after("今天天气很好。", "录音里说的话。", &voice, &options, &mut report)?;
```

It is published as `cosyvoice:v3` (`ling0322/libwaifu-cosyvoice-3` on Hugging Face and
ModelScope), so `waifu -m cosyvoice` fetches it; or build the package below and choose it by path.
The published package holds the language model and the DiT in half -- see
[In half](#in-half) -- so it needs a GPU; on the processor, export the float32 package below.

## Building the package

```bash
# Upstream's code, importable. Nothing of it goes into .venv, which is pinned.
git clone https://github.com/xingchensong/S3Tokenizer ~/.cache/libwaifu/src/S3Tokenizer
PYTHONPATH=~/.cache/libwaifu/src/S3Tokenizer:<onnx, onnxruntime> \
    .venv/bin/python tools/cosyvoice3_exporter.py -output models/cosyvoice3.safetensors
```

What comes out is `models/cosyvoice3.yaml`, three weight files and a `tokenizer.json`: 1 109 M
parameters, all float32, which is the precision upstream runs every one of these at -- and the
precision the reference tests hold to upstream's most tightly. Five namespaces:

| namespace | from | |
| --- | --- | --- |
| `cosyvoice3.lm` | `llm.pt` | Qwen2-0.5B, `speech_embedding`, `llm_decoder`; not Qwen's text head |
| `cosyvoice3.flow` | `flow.pt` | and `rand_noise`, the solver's fixed starting noise |
| `cosyvoice3.hift` | `hift.pt` | weight norm folded |
| `cosyvoice3.speech_tokenizer` | `speech_tokenizer_v3.onnx` | read by S3Tokenizer's `onnx2torch_v3` |
| `cosyvoice3.campplus` | `funasr/campplus` | the same network as `campplus.onnx`; see below |

The exporter checks the written `tokenizer.json` against the tokenizer upstream calls, and that
`<|endofprompt|>` comes out as 151646, which the language model asserts.

### In half

```bash
PYTHONPATH=~/.cache/libwaifu/src/S3Tokenizer:<onnx> .venv/bin/python tools/cosyvoice3_exporter.py \
    -output models/cosyvoice3.safetensors -part_size 2GB -half lm,flow
```

is the published package: the language model and the flow's DiT in half, 2.76 GB where float32
is 4.44 GB, in two files. `CosyVoice3::from_manifest` runs each model at the type its weights
are stored in -- there is no setting, and both packages load. It refuses a package that mixes
types within one model, or that holds anything else in half:

| | half | float32, always |
| --- | --- | --- |
| `cosyvoice3.lm` | all of it, `llm_decoder` too | |
| `cosyvoice3.flow` | `decoder.estimator.*`, the DiT | the token embedding, the look-ahead layer, the speaker projection, `rand_noise`; the Euler step and the guidance mix |
| `cosyvoice3.hift` | | all: its source integrates the pitch to a phase of 131 176 rad on this reading, past half's 65 504, and a quarter of `source_resblocks`' weights are below half's smallest normal |
| speech tokenizer, CAMPPlus | | all; they run once a voice |

What makes half safe in the DiT and the language model is in flint rather than here: attention
keeps its scores, their softmax and its accumulator in float -- the fused kernels, and the
composition (matmul, softmax, matmul) behind them on the CPU, CUDA and Metal alike -- and the
norms sum in float. The DiT's `gelu_new` and Mish overflow in half on their way to the right answer (x³ reaches
1.2e7; `tanh(±inf)` and `rsqrt(inf)` bring both back). Half needs a GPU: x64's processor has no
kernel taking a half weight against a float activation, so it keeps the float32 package.

## The order

**Listening, once per voice.** The recording is resampled to 16 kHz and to 24 kHz, as upstream's
`load_wav` does each, and cut to thirty seconds.

| | reads | gives |
| --- | --- | --- |
| [S3Tokenizer v3](../waifu/src/cosyvoice3/speech_tokenizer.rs) | Whisper's 128-band mel of the 16 kHz audio | speech tokens, 25 a second |
| [CAMPPlus](campplus.md) | Kaldi energies of the 16 kHz audio | who is speaking, `(192)` |
| [the 24 kHz mel](../waifu/src/cosyvoice3/features.rs) | the 24 kHz audio | 50 frames a second |

Tokens and mel are then cut to the same length, two frames a token.

**Saying, once per sentence.** The text is [normalized](text_normalize.md) and split into pieces
of 60 to 80 characters (Chinese) or tokens (anything else), as upstream's `text_normalize` does --
see [`frontend.rs`](../waifu/src/cosyvoice3/frontend.rs) -- and each piece goes through:

1. [The language model](../waifu/src/cosyvoice3/lm.rs): `[sos] text [task] prompt-tokens` in,
   speech tokens out, by repetition-aware nucleus sampling (top-p 0.8, top-k 25, a token said in
   the last ten is struck out and redrawn).
2. [The flow](../waifu/src/cosyvoice3/flow.rs): the prompt's tokens and the new ones, embedded and
   through a look-ahead convolution, twice over for two mel frames a token; then a 22-block DiT,
   ten Euler steps on a cosine schedule with guidance at 0.7, continuing the prompt's own mel.
3. [HiFT](../waifu/src/cosyvoice3/hift.rs): the mel to a pitch, the pitch to a harmonic
   excitation, and the mel and excitation through an upsampling filter to a 16-point spectrum,
   inverted to 24 kHz samples.

### With a transcript, without, and with a style

Given the words of the recording -- `SpeechOptions::reference_transcript`, a text-to-speech
request's `reference_transcript` over the C API, or `say_after` -- [`Voice::speak`](../waifu/src/speech.rs)
reads the way `inference_zero_shot` does: transcript and sentence as one text, continuing the
recording's own tokens. Handed a recording and no transcript, it reads the way
`inference_cross_lingual` does: the language model sees the system prompt and the sentence, and
nothing of the recording; the voice comes from the flow. Both sound like the recording; zero-shot
also carries on its pace and manner. Streaming reads either way.

Zero-shot is upstream's own way of cloning a voice in the same language. Its web UI will not read
"3s极速复刻" without the transcript, and it keeps "跨语种复刻", cross-lingual, for a sentence in
another language than the recording, which is what that reading is for. A page that offers a
recording should offer the box for its words beside it.

The transcript is the system prompt's to follow, as upstream's `prompt_text` is in its examples,
except that here the system prompt is put in front of it: a transcript holding a `<|...|>` marker
is refused rather than read with two. A recording over thirty seconds is cut there, and its
transcript should be of what is kept.

Given a style -- `SpeechOptions::style`, the Style card on the page -- it reads the way
`inference_instruct2` does. Upstream builds that as a zero-shot reading with the instruction where
the transcript would be and the recording's tokens taken out of the language model's input
(`frontend_instruct2`), so here the instruction goes inside the system prompt,

```text
You are a helpful assistant. 请非常开心地说一句话。<|endofprompt|>
```

encoded as it was given, apart from each sentence and not counted in the sentence's length limits;
the flow still continues the recording, so the voice is the recording's and the manner the
instruction's. [`CosyVoice3::STYLES`](../waifu/src/cosyvoice3/mod.rs) is upstream's `instruct_list`
word for word: happy, sad and angry, fast and slow, loud and soft, Peppa Pig, a robot, and
seventeen dialects. Those are what the model was taught; other wording is taken and is a guess. A
style and a transcript are refused together, before the recording is listened to: instruct2 drops
the recording's tokens from the language model, which leaves a transcript nothing to be the words
of, and upstream never combines them.

### Streaming

[`CosyVoice3::say_streaming`](../waifu/src/cosyvoice3/mod.rs) -- under it `Voice::speak_streaming`,
`Engine::speak_streaming` and a `WAIFU_PIPELINE_STREAMING_TEXT_TO_SPEECH` pipeline's
`waifu_pipeline_run`, read as `WAIFU_EVENT_AUDIO_PART` events ([ffi-abi6.md](ffi-abi6.md)) -- is
upstream's `stream=True`: the audio is handed over a hop of tokens at a time while the language
model goes on saying the rest, and the whole reading at the end.
[`stream`](../waifu/src/cosyvoice3/stream.rs) says when:

- **A hop** is 25 tokens, doubling each time to at most 100, kept across the sentences of a
  reading as upstream keeps `token_hop_len` (upstream keeps it across readings too; this starts
  every reading at 25). The first of a sentence is longer by however many tokens bring the
  recording's up to a multiple of the hop, so every hop starts on a block of the DiT's attention.
  A hop is drawn once three tokens past it are said: the look-ahead the flow's condition reads.
- **The flow** draws everything said in the sentence so far, prompt and all, with the DiT's
  attention in blocks of 50 frames counted from the prompt's first (`Flow::draw_in_blocks`, under
  it flint's `attention(q, k, v, 50)`), and keeps the frames after the last hop. A frame there is
  drawn from nothing after its own block, so what was drawn before does not move. The condition is
  the whole one's rows less the look-ahead -- the same rows upstream's `finalize=False` builds by
  handing the three tokens over as context. The last of a sentence is drawn every frame seeing
  every other: upstream's `tts` does not hand `token2wav` its `stream` for that one.
- **HiFT** sounds the sentence's mel so far and hands over what it had not, short of the last
  eight frames -- three its pitch reads ahead, four its first convolution, one its inverse
  transform -- which upstream's `finalize=False` cuts off and the next hop sounds. Every
  convolution after those reads only behind, so this is the whole run's start. Its noise is drawn
  once a sentence and only made longer, as upstream's `SineGen2` reads one buffer from its start.
- **Silent tokens**: `llm_job` hands the flow at most five of upstream's eleven silent and breath
  tokens in a row; the language model goes on reading the rest. Every reading does this, streaming
  or not.

It is another reading of the same tokens, not `say`'s samples: the blocks change what the DiT
draws. The flow redraws the whole sentence every hop, as upstream does -- at most a prompt of 30 s
and a sentence of about 20, so a hop takes at most about a second -- and the language model waits
while it does. On the RTX 5060 Ti, with the 9.1 s recording and 23 s of Chinese, the first 1.76 s
are handed over 0.59 s in, and every later part before the one ahead of it has finished playing;
the reading takes 8.8 s, where `say` takes 5.0. `say` takes 3.3 s to its first sentence, which a reading a sentence
at a time used to start with. On an M5 Pro, the half package, with a 5.0 s recording and 9.2 s of
Chinese: the first 0.88 s are handed over 0.75 s in, the second 0.2 s after the first has played,
and every later part before the one ahead of it has finished; the reading takes 5.3 s, where `say`
takes 3.4.

Streaming is refused before anything is said where it cannot be done: at a speed other than one,
which upstream refuses too, or where no attention kernel takes blocks -- anywhere but CUDA at half,
on Ampere or later, and Metal. It is not a reading a sentence at a time instead.

## Voice conversion

The same package converts one recording into the voice of another, as upstream's `inference_vc`
does: the source's speech tokens are what is said, and the flow draws them continuing the
reference -- its tokens, its mel, its speaker -- so the words and their timing are the source's and
the voice is the reference's. The language model is not run. It is `impl Converter for
CosyVoice3`, which the engine runs for `waifu_pipeline_run` on a voice conversion pipeline
([ffi-abi6.md](ffi-abi6.md)). The terminal's `waifu` draws only; the speech2speech page that ran this was
removed with the rest of the web page.

Upstream refuses a source over thirty seconds, which is as far as the speech tokenizer reads. A
longer one is cut into pieces of at most that, each at the quietest 20 ms of its last eight
seconds, so a cut falls between words; each piece is converted on its own and the pieces joined.
The page's steps are the flow's Euler steps (10 by default). There is no style conversion -- the
accent and pacing stay the source's -- so the page offers no box for it.

On the RTX 5060 Ti, upstream's `cross_lingual_prompt.wav` (13.75 s) into the voice of
`zero_shot_prompt.wav`, 10 steps: 1.3 s; the same source three times over (41.9 s, two pieces):
4.0 s. Speaker similarity by upstream's own CAMPPlus, and Whisper large-v3 on the result:

| | to the reference | to the source | Whisper |
| --- | --- | --- | --- |
| source itself | 0.100 | 1 | -- |
| converted, 13.8 s | 0.679 | 0.241 | the source's words exactly |
| converted, 41.9 s | 0.694 | 0.250 | the source's words exactly, three times |

## What was found on the way

- **CosyVoice's `campplus.onnx` is `funasr/campplus`.** Embeddings of one input agree to 4e-6, so
  the package carries IndexTTS-2.5's CAMPPlus export and the pipeline runs its port.
- **The DiT rotates one head.** F5-TTS's attention hands `x_transformers` the query before the
  head split, and that rotates the first 64 channels -- head zero of sixteen -- by adjacent pairs.
- **HiFT's randomness is not in `hift.pt`.** `SineGen2` draws a starting phase and seven million
  samples of noise at construction, from wherever torch's generator is by then. Here they are
  drawn from the reading's seed. The starting phase never reaches the output, upstream or here:
  it is added to the first sample of each frame, and the resampling after it reads samples 239
  and 240.
- **The minimum length does not stop the end token.** While fewer than twice the text's tokens
  have been said upstream masks index 6561, which in CosyVoice3 is `sos`, not the end token 6562;
  every index from 6561 up ends a reading. This does the same.
- **The flow's starting noise is upstream's, exactly.** `CausalConditionalCFM` seeds zero and
  draws once; the package holds that draw, three hundred seconds of it.

## How it is checked

`tools/cosyvoice3_reference.py` constructs upstream's own `CosyVoice3` from the release and runs it
on `asset/zero_shot_prompt.wav` -- the recording its examples use -- recording every stage into
`models/cosyvoice3_test.safetensors` -- published as `cosyvoice3_test.safetensors` in
`ling0322/libwaifu_test_data` on Hugging Face, so the tests need only that and the package.
`waifu/tests/cosyvoice3.rs` then checks each stage on the inputs upstream really handed it (on
this machine's RTX 5060 Ti):

(Upstream's frontend reaches wetext, which fetches its FSTs from ModelScope at construction; where
that fails it logs "no frontend is avaliable" and skips its normalizer. The sentence and the
transcript here have nothing for it to do -- no numbers, no symbols -- and the text-id test
agrees, but a run that reads anything else needs it.)

| stage | against upstream |
| --- | --- |
| resampling, Whisper mel, 24 kHz mel | 1.8e-7, 2.9e-5, 3.1e-5 |
| speech tokens | 87 of 87 |
| speaker embedding | 1.5e-5 |
| language model, teacher-forced over 231 tokens | log probability 3e-5, the same likeliest token in all 232 rows |
| flow: condition, first velocity, ten-step mel | 1.5e-5; 3.5e-3 of 14.4; 5.9e-3 max, 2.4e-4 mean |
| HiFT: pitch, excitation, filter, whole | 1.5e-3 Hz; 2.1e-4; 2.8e-5; 46.5 dB |
| reading length, eight seeds | mean 205 tokens against upstream's 212 |
| **half** language model, teacher-forced over 231 tokens | at the token said, 4.0e-3 mean and 1.7e-2 worst; the same likeliest token in 99.6% of rows |
| **half** flow, ten-step mel | 2.9e-1 max, 1.9e-2 mean |
| streaming: hops, and HiFT on upstream's mel a hop at a time | four parts of upstream's lengths to the sample; 71.8, 61.6, 51.3 and 34.0 dB |
| **half** streaming flow, the sentence's mel | 8.4e-1 max, 1.8e-2 mean |

The tests read the precision off `models/cosyvoice3.yaml` and hold each to its own bounds: a
float32 package to the rows above it, the published half one at 0.01 mean and 0.05 worst at the
token said, 97% agreement, and a mel of 0.04 mean and 1.5 worst.
Against this crate's float32 the half package's DiT moves the mel by 0.018 on average; through
the same vocoder on the same noise, the two readings heard again as mels are 0.09 apart, where
one mel on two draws of the vocoder's noise is 0.026 apart, and CAMPPlus puts both at the same
likeness to the recording (0.761 and 0.766). `waifu-tools`' `cosyvoice3_precision` measures all
of that for any two packages.

and then the whole pipeline. Whisper large-v3 transcribes the Chinese sentence exactly both ways
and the English one ("The train leaves at 7 tonight, and the ticket costs $25."); CAMPPlus puts
the output at 0.77 (cross-lingual) and 0.81 (zero-shot) of the recording's own voice, where
upstream's output is at 0.78 and a different speaker at 0.10.

```bash
cargo test --release --manifest-path waifu/Cargo.toml --no-fail-fast \
    --test cosyvoice3 -- --ignored --test-threads=1
```

## Speed

About real time: 9.4 s of audio in 8.2 s, of which the language model's 234 steps are 7 s. The
step concatenates every layer's cache each token rather than writing into a preallocated one;
that is the first thing to change if it needs to be faster.

In half, on the RTX 5060 Ti, against the float32 package in the same build: the language model
takes 5.9 ms a token where float32 takes 12.8, and the flow draws 11.9 s of audio in 339 ms where
float32 takes 861.

## Visemes

CosyVoice3 can also say the mouth shape of every 40 ms of what it reads -- one of `A E I O U @
MBP REST` for each speech token the flow is handed, laid on the audio that token became -- for a
face to move to as it plays: the C API's `TEXT_TO_SPEECH_VISEMES` ([ffi-abi6.md](ffi-abi6.md#what-61-adds)),
`Voice::speak_with_visemes` in Rust. `waifu/src/cosyvoice3/visemes.rs` is all of it.

What says them is a model of its own, attached to a pipeline that has CosyVoice3 loaded --
`waifu_pipeline_attach_model(pipeline, WAIFU_ATTACHED_MODEL_TEXT_TO_SPEECH_VISEMES, NULL, model)`,
`CosyVoice3::attach_visemes` in Rust -- so CosyVoice3's own package is not changed, and a better
viseme model is a few megabytes to fetch rather than the voice again. It is small and runs on the
host: a token and the token either side, as the flow's `input_embedding` reads them (3 × 80),
through one hidden layer of 256 to the eight. Its package is two files:

```yaml
# cosyvoice3-visemes.yaml
weights:
  - cosyvoice3-visemes.safetensors     # 2.5 MB: the model, and a copy of the flow's input_embedding
config:
  model:
    type: cosyvoice3_visemes
  cosyvoice3_visemes:
    speech_token_size: 6561
    visemes: [A, E, I, O, U, "@", MBP, REST]
```

The copy of the embedding is what lets it read nothing of CosyVoice3's weights. A package whose
classes are in another order, or for another number of tokens, is refused when it is attached.

### How it was trained

Nothing labels speech tokens with mouth shapes, so the labels are made from text. CosyVoice3 reads
sentences in several voices; S3Tokenizer hears each reading again, which gives its tokens at 25 a
second, lined up with the audio by construction; Qwen3-ForcedAligner puts each unit of the text on
the audio -- a Chinese character, a Japanese mora, an English word -- and the unit is spelled into
mouth shapes: pypinyin's initial and final, a kana table, CMUdict. Vowels, m/b/p and pauses are
anchors; any other consonant's frames run from the anchor before to the one after, which is what a
mouth does while it says one. A sound between two shapes is a mixture -- cat's æ is A 0.7 + E 0.3 --
so the labels are distributions, and the model is fitted to them.

```bash
# The readings: each sentence in each voice, and S3Tokenizer's tokens of it.
cargo run --release -p waifu-tools --bin viseme_corpus -- models/cosyvoice3.yaml zh.txt c_zh voice*.wav
# The labels: pypinyin, nagisa and Qwen3-ASR's aligner on the path (Qwen/Qwen3-ForcedAligner-0.6B).
python tools/cosyvoice3_visemes.py align c_zh -language Chinese        # Japanese, or English -cmudict
# Every voice group held out in turn, then the file.
python tools/cosyvoice3_visemes.py train c_zh c_ja c_en -package models/ -output models/cosyvoice3-visemes
```

The file in use came from 442 sentences -- 200 Chinese, 120 Japanese in kana, 122 English -- in
eight voices, four recordings and a pitch-shifted copy of each: 216 430 tokens, 5 555 of the 6 561.
Each recording with its copy held out in turn, on tokens whose label is at least 0.8 one class:

| | top-1 | Chinese / Japanese / English | m, b, p shut | vowels shut |
| --- | --- | --- | --- | --- |
| a token alone (`cosyvoice3_visemes.table`) | 0.59 | 0.62 / 0.60 / 0.50 | 0.90 | 1.7 % |
| with a token either side (what runs) | 0.68 | 0.71 / 0.69 / 0.59 | 0.81 | 1.9 % |

A, I and REST are right about four times in five; E, O, U and @ about half the time, mostly for
each other's neighbours. English is the weakest because the aligner times only its words, and a
word's sounds share its span by rule. The labels are made by rule and never checked by eye, so the
scores are of a model against labels that are themselves rough.

### Not yet

`VisemeSmoother` takes each frame's likeliest viseme, which flickers where two are close; Viterbi
over a transition matrix learnt from the labels, with a fixed lag for streaming, is the next step.
The tokens the model was trained on are S3Tokenizer's of the finished audio, not the language
model's own, which are what it reads at run time; the two are the same vocabulary for the same
sounds, but how far they agree is not measured.

## Not here

Streaming on the processor, whose attention has no block mask. A DiT that draws
only each new hop, keeping what it computed of the frames before, rather than all of them again.
A place in the apps to type the recording's transcript: the C API takes one, and neither app
sends it yet. Nor one that fills it in by itself, from Qwen3-ASR's transcription of the recording.
The instruct2 reading has no reference test beside the zero-shot and cross-lingual ones; its ids
are the zero-shot path's with a different prefix.
