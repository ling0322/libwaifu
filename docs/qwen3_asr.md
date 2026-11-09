# Qwen3-ASR

A recording in; what was said in it, and in which language, out. Thirty languages, told which or
left to find out.
[`Qwen/Qwen3-ASR-0.6B`](https://huggingface.co/Qwen/Qwen3-ASR-0.6B), under Apache 2.0. It is the
first model of the `speech2text` task: [`Transcriber`](../waifu/src/transcription.rs) is the seam,
[`Qwen3Asr`](../waifu/src/qwen3_asr/mod.rs) implements it, and the engine and the C API's
`waifu_pipeline_run_speech_to_text_async` ([ffi-abi5.md](ffi-abi5.md)) run it.

```bash
cargo run --release -p waifu-tools --bin transcribe -- models/qwen3-asr.yaml speech.wav
cargo run --release -p waifu-tools --bin transcribe -- models/qwen3-asr.yaml speech.wav English
cargo run --release -p waifu-tools --bin transcribe -- models/qwen3-asr.yaml speech.wav "" \
    --context="Qwen, libwaifu" --device=cpu
```

```rust
let asr = Qwen3Asr::from_manifest(Device::Cuda, u64::MAX, &manifest)?;
let said = asr.transcribe(&recording, &TranscriptionOptions::default(), &mut report)?;
// Some(Transcript { text: "希望你以后能够做得比我还好哟。", language: "Chinese" })
```

The package is published as `ling0322/libwaifu-qwen3-asr-0.6b` on Hugging Face and on
ModelScope, named `qwen3-asr` (`qwen3-asr:0.6b`) in the model manager's catalogue -- fetched by
name, and offered in no picker and not in `waifu_modelmanager_catalog_json`, which has no kind
for a transcriber yet. Its reference tensors are `qwen3-asr_test.safetensors` in
`ling0322/libwaifu_test_data`; with the package and that file in `models/`, the model test below
runs without exporting anything.

```bash
hf download ling0322/libwaifu-qwen3-asr-0.6b --local-dir models \
    --include "qwen3-asr.*"
hf download ling0322/libwaifu_test_data qwen3-asr_test.safetensors --local-dir models
```

## Building the package

```bash
uv venv .venv-qwen3-asr --python 3.12
uv pip install --python .venv-qwen3-asr/bin/python "torch==2.8.0+cu128" --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv-qwen3-asr/bin/python -r tools/requirements-qwen3-asr.txt
.venv-qwen3-asr/bin/python tools/qwen3_asr_exporter.py -output models/qwen3-asr.safetensors
```

What comes out is `models/qwen3-asr.yaml`, one weights file of 1.56 GB and a `tokenizer.json`:
782 M parameters in float16. `-repo Qwen/Qwen3-ASR-1.7B` exports the larger release, which is the
same network wider and deeper; the manifest carries the widths and the runtime reads them, but
only the 0.6B has been run.

| namespace | from | |
| --- | --- | --- |
| `qwen3_asr.audio_tower` | `thinker.audio_tower` | three convolutions, eighteen layers, two projections |
| `qwen3_asr.model` | `thinker.model` | Qwen3-0.6B's decoder |
| `qwen3_asr.lm_head` | `thinker.lm_head` | left out: the release's is the embedding table, bit for bit |

**Float16, from bfloat16.** The release is bfloat16 and flint has no bfloat16. Every bfloat16
value in half's range survives the narrowing whole (seven bits of mantissa against ten), so only
what is below half's smallest normal number is rounded; the exporter refuses a tensor with
anything past half's largest instead of writing an infinity. It runs at the device's default
float -- half on CUDA, Metal, Vulkan and, since #139, the processor.

## The order

1. **Resampled to 16 kHz**, with the host resampler CosyVoice3 and IndexTTS-2.5 use; upstream uses
   librosa's. A recording longer than twenty minutes (upstream's `MAX_ASR_INPUT_SECONDS`) is cut
   into pieces at the quietest tenth of a second within five seconds of each twenty-minute mark,
   and a piece shorter than half a second is padded with silence -- `split_audio_into_chunks`.
2. **Heard** as `WhisperFeatureExtractor` hears it: Whisper's 128-band log mel, 100 frames a
   second, [computed on the device](../waifu/src/qwen3_asr/features.rs).
3. **Encoded** by the [audio tower](../waifu/src/qwen3_asr/audio_encoder.rs). The mel is cut into
   one-second chunks, the last padded with zeros; three stride-two convolutions make thirteen
   tokens of each; a sinusoidal position table starts again in every chunk; and attention never
   crosses an eight-second window (104 tokens). One token per 80 ms, 1024 wide. On CUDA the
   windows are one attention under `MaskType::BlockDiagonal(104)`; elsewhere -- and on a card
   whose kernel refuses the mask, which is then remembered -- the whole windows are one batch of
   attention and the last another. The two agree, and the model test checks both.
4. **Written out** by the [decoder](../waifu/src/qwen3_asr/decoder.rs), greedily, after upstream's
   chat prompt with the tokens in place of `<|audio_pad|>`:

   ```text
   <|im_start|>system\n{context}<|im_end|>\n<|im_start|>user\n<|audio_start|>
   <|audio_pad|> x N <|audio_end|><|im_end|>\n<|im_start|>assistant\n[language X<asr_text>]
   ```

   The bracketed part is there only where a language is given; then what is written is the words
   alone. Otherwise it writes `language Chinese<asr_text>` first. It stops at `<|im_end|>` or
   `<|endoftext|>`, or at 4096 tokens.
5. **Parsed** as `parse_asr_output` parses it, with `detect_and_fix_repetitions`' cut of any run
   of twenty repeats to one, and the pieces' words and languages joined.

Upstream's text model rotates with interleaved MRoPE. For a sequence of audio and text its three
position axes are the same number -- `get_rope_index` is a running count, repeated three times --
and three equal axes interleaved are one, so the decoder uses the plain rotary table.

## The reference, and what the tests found

```bash
.venv-qwen3-asr/bin/python tools/qwen3_asr_reference.py -output models/qwen3-asr_test.safetensors \
    -recording zh=<CosyVoice>/asset/zero_shot_prompt.wav \
    -recording long=<CosyVoice>/asset/cross_lingual_prompt.wav \
    -recording en=models/cosyvoice3-export/cosyvoice3-english.wav
```

`qwen-asr`, upstream's package, is not run. The reference is put together from parts with
independent provenance, in float32: transformers' own `WhisperFeatureExtractor` for the features,
upstream's `Qwen3ASRAudioEncoder.forward` written out in plain torch for the encoder, and
transformers' `Qwen3ForCausalLM` with the release's weights for the decoder. Its transcripts are
right: `zero_shot_prompt.wav` is "希望你以后能够做得比我还好哟。", the two AISHELL recordings in
S3Tokenizer's assets come out as their transcripts, and CosyVoice3's English sentence comes out
word for word, its numbers spelt.

The reference is float32 on the processor. The same reference again in half on CUDA, on the
same inputs, is the floor: how far half alone moves each stage from float32. That needs a CUDA
build of torch, which the venv above installs, and the script refuses to run without one -- torch's
half on the processor keeps more of its arithmetic in float, and is not what runs here.

The test holds each stage to that floor on a measure of the whole of it, the relative RMSE: the
encoder's rows, and the decoder's final hidden state at every position of the sequence it was
teacher-forced over. Each may be 1.2 times as far from float32 as torch's half is. The decoder
is not held on the log probabilities of the tokens said: there are a few dozen with any real
chance (34 on `en`), each decided by where its logit, 20 to 40, happened to round to half, so
two equally good halves disagree on their mean by a quarter -- and torch's own by twice from 2.8
to 2.11. Every token said has to be the reference's.

| | zh, 3.5 s | long, 13.75 s at 22.05 kHz | en, 5.3 s |
| --- | --- | --- | --- |
| windows of attention | 1 | 2 | 1 |
| mel, relative error | 1.4e-5 | 1.5e-5 | 1.7e-5 |
| encoder, relative RMSE (torch's half) | 1.24e-3 (1.38e-3) | 1.28e-3 (1.46e-3) | 1.07e-3 (1.40e-3) |
| decoder's final state, relative RMSE (torch's half) | 4.34e-3 (4.64e-3) | 3.06e-3 (3.28e-3) | 4.12e-3 (4.68e-3) |
| tokens | the same 14 | the same 39 | the same 19 |

Two things put this runtime further from float32 than torch's half before they were found this
way, layer by layer against torch: a linear's bias added after its product was rounded to half
(1.3 times torch's half on the encoder; #176), and the half GEMM's split K handing its partial
sums from one slice to the next in half (1.13 to 1.37 times on the decoder's final state; #178).

On an RTX 5060 Ti the package reads in 0.6 s and writes the 13.75 s recording down in 0.36 s;
on the processor (AVX-512, half) in 2.5 s.

**Upstream's windows depend on its attention.** `Qwen3ASRAudioEncoder` keeps attention inside
the windows through `cu_seqlens`, which only FlashAttention reads: vLLM and
`attn_implementation="flash_attention_2"`, the two paths upstream recommends, attend window by
window. Its transformers backend under sdpa or eager -- the default without flash-attn -- is
handed no mask at all (`_prepare_attention_mask` is written and never called), so there the
encoder attends across the whole recording. This follows the windows. On the 13.75 s recording
the two encoders' rows differ by up to 76% of their largest, and the words written are the same.

```bash
cargo test --release --manifest-path waifu/Cargo.toml --no-fail-fast \
    --test qwen3_asr -- --ignored --test-threads=1
```

## What is not here

- **Timestamps.** Upstream gets them from a second model, Qwen3-ForcedAligner, which is not ported.
- **Streaming.** Upstream's streaming mode re-reads a growing recording a chunk at a time with
  vLLM; here a recording is written down whole.
- **Batching.** One piece at a time; upstream batches pieces of several recordings.
