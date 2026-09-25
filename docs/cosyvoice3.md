# Fun-CosyVoice3

A few seconds of somebody speaking and a sentence in; the sentence, in that voice, out, at 24 kHz.
[`FunAudioLLM/Fun-CosyVoice3-0.5B-2512`](https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512),
under Apache 2.0.

```bash
waifu webui -voice models/cosyvoice3.yaml          # the page's text2speech tab
cargo run --release --example speak -- models/cosyvoice3.yaml voice.wav "你好。" out.wav
cargo run --release --example speak -- models/cosyvoice3.yaml voice.wav "你好。" out.wav 7 "录音里说的话。"
```

```rust
let tts = CosyVoice3::from_manifest(Device::Cuda, Residency::Device, &manifest)?;
let voice = tts.listen(&recording)?;                               // once per voice
let sound = tts.say("今天天气很好。", &voice, &options, &mut report)?;  // no transcript
let sound = tts.say_after("今天天气很好。", "录音里说的话。", &voice, &options, &mut report)?;
```

It is not published to the hubs; build the package below and choose it by path.

## Building the package

```bash
# Upstream's code, importable. Nothing of it goes into .venv, which is pinned.
git clone https://github.com/xingchensong/S3Tokenizer ~/.cache/libwaifu/src/S3Tokenizer
PYTHONPATH=~/.cache/libwaifu/src/S3Tokenizer:<onnx, onnxruntime> \
    .venv/bin/python tools/cosyvoice3_exporter.py -output models/cosyvoice3.safetensors
```

What comes out is `models/cosyvoice3.yaml`, three weight files and a `tokenizer.json`: 1 109 M
parameters, all float32, which is the precision upstream runs every one of these at. Five
namespaces:

| namespace | from | |
| --- | --- | --- |
| `cosyvoice3.lm` | `llm.pt` | Qwen2-0.5B, `speech_embedding`, `llm_decoder`; not Qwen's text head |
| `cosyvoice3.flow` | `flow.pt` | and `rand_noise`, the solver's fixed starting noise |
| `cosyvoice3.hift` | `hift.pt` | weight norm folded |
| `cosyvoice3.speech_tokenizer` | `speech_tokenizer_v3.onnx` | read by S3Tokenizer's `onnx2torch_v3` |
| `cosyvoice3.campplus` | `funasr/campplus` | the same network as `campplus.onnx`; see below |

The exporter checks the written `tokenizer.json` against the tokenizer upstream calls, and that
`<|endofprompt|>` comes out as 151646, which the language model asserts.

## The order

**Listening, once per voice.** The recording is resampled to 16 kHz and to 24 kHz, as upstream's
`load_wav` does each, and cut to thirty seconds.

| | reads | gives |
| --- | --- | --- |
| [S3Tokenizer v3](../waifu/src/cosyvoice3/speech_tokenizer.rs) | Whisper's 128-band mel of the 16 kHz audio | speech tokens, 25 a second |
| [CAMPPlus](campplus.md) | Kaldi energies of the 16 kHz audio | who is speaking, `(192)` |
| [the 24 kHz mel](../waifu/src/cosyvoice3/features.rs) | the 24 kHz audio | 50 frames a second |

Tokens and mel are then cut to the same length, two frames a token.

**Saying, once per sentence.** The text is normalized and split into pieces of 60 to 80
characters (Chinese) or tokens (anything else), as upstream's `text_normalize` does -- see
[`frontend.rs`](../waifu/src/cosyvoice3/frontend.rs) -- and each piece goes through:

1. [The language model](../waifu/src/cosyvoice3/lm.rs): `[sos] text [task] prompt-tokens` in,
   speech tokens out, by repetition-aware nucleus sampling (top-p 0.8, top-k 25, a token said in
   the last ten is struck out and redrawn).
2. [The flow](../waifu/src/cosyvoice3/flow.rs): the prompt's tokens and the new ones, embedded and
   through a look-ahead convolution, twice over for two mel frames a token; then a 22-block DiT,
   ten Euler steps on a cosine schedule with guidance at 0.7, continuing the prompt's own mel.
3. [HiFT](../waifu/src/cosyvoice3/hift.rs): the mel to a pitch, the pitch to a harmonic
   excitation, and the mel and excitation through an upsampling filter to a 16-point spectrum,
   inverted to 24 kHz samples.

### With a transcript, and without

The page hands a voice a recording and no transcript, so [`Voice::speak`](../waifu/src/speech.rs)
reads the way upstream's `inference_cross_lingual` does: the language model sees the system prompt
and the sentence, and nothing of the recording; the voice comes from the flow. Given the words of
the recording, `say_after` reads the way `inference_zero_shot` does: transcript and sentence as one
text, continuing the recording's own tokens. Both sound like the recording; zero-shot also carries
on its pace and manner.

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
`models/cosyvoice3_test.safetensors`. `waifu/tests/cosyvoice3.rs` then checks each stage on the
inputs upstream really handed it (on this machine's RTX 5060 Ti):

| stage | against upstream |
| --- | --- |
| resampling, Whisper mel, 24 kHz mel | 1.8e-7, 2.9e-5, 3.1e-5 |
| speech tokens | 87 of 87 |
| speaker embedding | 1.5e-5 |
| language model, teacher-forced over 231 tokens | log probability 3e-5, the same likeliest token in all 232 rows |
| flow: condition, first velocity, ten-step mel | 1.5e-5; 3.5e-3 of 14.4; 5.9e-3 max, 2.4e-4 mean |
| HiFT: pitch, excitation, filter, whole | 1.5e-3 Hz; 2.1e-4; 2.8e-5; 46.5 dB |
| reading length, eight seeds | mean 205 tokens against upstream's 212 |

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

## Not here

Streaming, instruct mode (`inference_instruct2`), voice conversion, fp16, and a place on the
page to type the recording's transcript.
