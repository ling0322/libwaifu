#!/usr/bin/env python3
# The MIT License (MIT)
#
# Copyright (c) 2026 Xiaoyang Chen
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software
# and associated documentation files (the "Software"), to deal in the Software without
# restriction, including without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or
# substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""What upstream CosyVoice3 computes on one real reading, for `waifu/tests/cosyvoice3.rs`.

    .venv/bin/python tools/cosyvoice3_reference.py -output models/cosyvoice3_test.safetensors

with upstream importable -- `PYTHONPATH` holding checkouts of FunAudioLLM/CosyVoice,
xingchensong/S3Tokenizer and shivammehta25/Matcha-TTS, and the packages `requirements.txt` there
names that the export venv lacks (onnxruntime, onnx, x-transformers, einops, hyperpyyaml,
omegaconf, conformer, inflect, openai-whisper, wetext). torchaudio has no wheel for the venv's
torch; a stand-in holding torchaudio's own pure-Python `kaldi.py` and `functional.py` is enough,
since those are the only parts inference reaches.

Nothing here is a transcription. Upstream's `CosyVoice3` is constructed from the release and run:
its frontend on `asset/zero_shot_prompt.wav` -- the recording upstream's own example uses -- its
language model on a sentence, its flow on the tokens that language model actually said, and its
vocoder on the mel the flow actually drew. Every input a stage is handed is one it is really
handed in a reading.

# What is written

| name | what |
| --- | --- |
| `prompt_24k`, `prompt_16k` | the recording, and upstream's 16 kHz resampling of it |
| `whisper_mel` | `(1, 128, T)`, what the speech tokenizer reads |
| `prompt_tokens` | the speech tokenizer's tokens, from `speech_tokenizer_v3.onnx` |
| `speaker` | `(1, 192)`, `campplus.onnx`'s embedding |
| `prompt_feat` | `(1, F, 80)`, the 24 kHz mel, before the frontend trims it to twice the tokens |
| `text_ids` | the cross-lingual text: the system prompt and the sentence |
| `said` | the tokens the language model said for it, seed 1986 |
| `lm_logp` | `(len(said) + 1, 6761)`, teacher-forced log probabilities over `said` |
| `zero_shot_ids`, `zero_shot_logp` | the zero-shot prefix and its first row of log probabilities |
| `flow_mu` | `(1, 80, M)`, the flow's condition after the look-ahead layer and the repeat |
| `dit_x`, `dit_t`, `dit_velocity` | the estimator's first call: both halves of the guided batch |
| `mel` | `(1, 80, M - prompt)`, what the flow drew for `said` |
| `hift_*_noise`, `hift_rand_ini` | the noise HiFT's source was handed; see below |
| `f0`, `source`, `wave` | the vocoder's pitch, excitation and 24 kHz output |

# HiFT's noise is chosen here

`SineGen2` and `SourceModuleHnNSF` draw their noise once, at construction, from whatever state
torch's generator is in by then -- which depends on every module built before them. It is not a
weight and it is not in `hift.pt`. So this draws it from its own seed, puts it in place of
upstream's before the vocoder runs, and writes it out, and the runtime is handed the same.
"""

import argparse
import importlib.util
import logging
import os
import sys
import types

import numpy as np
import torch

# The system prompt a CosyVoice3 text has to carry; `Qwen2LM.inference` asserts its marker is there.
SYSTEM = "You are a helpful assistant.<|endofprompt|>"
PROMPT_TEXT = "希望你以后能够做的比我还好呦。"
TEXT = "收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐。"
SEED = 1986


def importable():
    """Upstream, with the modules it imports and inference never reaches stood in for."""
    # openai-whisper's package without its __init__, which imports numba for word timing.
    whisper = types.ModuleType("whisper")
    spec = importlib.util.find_spec("whisper")
    if spec is None or spec.submodule_search_locations is None:
        raise SystemExit("openai-whisper is not importable")
    whisper.__path__ = list(spec.submodule_search_locations)
    sys.modules["whisper"] = whisper
    import whisper.audio

    whisper.log_mel_spectrogram = whisper.audio.log_mel_spectrogram

    # Named by the yaml's training data pipeline, which is located but never run.
    sys.modules["pyworld"] = types.ModuleType("pyworld")

    # matcha.utils' __init__ pulls in hydra and lightning for training; inference reaches only
    # audio.py and pylogger.
    import matcha

    utils = types.ModuleType("matcha.utils")
    utils.__path__ = [os.path.join(os.path.dirname(matcha.__file__), "utils")]
    sys.modules["matcha.utils"] = utils
    pylogger = types.ModuleType("matcha.utils.pylogger")
    pylogger.get_pylogger = lambda name=__name__: logging.getLogger(name)
    sys.modules["matcha.utils.pylogger"] = pylogger


def record_calls(module, into):
    """Wrap `module.forward` so every call's arguments and result are appended to `into`.

    Copies, not references: `solve_euler` refills the same `x_in`, `t_in` and the rest in place on
    every step, so a reference kept from the first call reads as the last one's by the end.
    """
    original = module.forward

    def copied(value):
        return value.clone() if isinstance(value, torch.Tensor) else value

    def forward(*args, **kwargs):
        out = original(*args, **kwargs)
        into.append((tuple(copied(a) for a in args),
                     {k: copied(v) for k, v in kwargs.items()}, copied(out)))
        return out

    module.forward = forward
    return original


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-output", required=True)
    parser.add_argument("-checkpoint", help="the release; the HF cache otherwise")
    arguments = parser.parse_args()

    importable()
    from cosyvoice.cli.cosyvoice import CosyVoice3
    from cosyvoice.utils.common import set_all_random_seed

    source = arguments.checkpoint
    if source is None:
        from huggingface_hub import snapshot_download

        source = snapshot_download("FunAudioLLM/Fun-CosyVoice3-0.5B-2512")

    cosy = CosyVoice3(source)
    frontend, llm, flow, hift = cosy.frontend, cosy.model.llm, cosy.model.flow, cosy.model.hift
    out = {}

    import cosyvoice

    wav = os.path.join(os.path.dirname(os.path.dirname(cosyvoice.__file__)), "asset",
                       "zero_shot_prompt.wav")

    # The frontend, stage by stage, as `frontend_zero_shot` runs it.
    from cosyvoice.utils.file_utils import load_wav
    import whisper

    out["prompt_24k"] = load_wav(wav, 24000)[0]
    out["prompt_16k"] = load_wav(wav, 16000)[0]
    out["whisper_mel"] = whisper.log_mel_spectrogram(load_wav(wav, 16000), n_mels=128)
    prompt_tokens, _ = frontend._extract_speech_token(wav)
    out["prompt_tokens"] = prompt_tokens[0].long()
    out["speaker"] = frontend._extract_spk_embedding(wav).float()
    prompt_feat, _ = frontend._extract_speech_feat(wav)
    out["prompt_feat"] = prompt_feat.float()

    text = frontend.text_normalize(TEXT, split=True)
    assert len(text) == 1, text
    cross = frontend.frontend_cross_lingual(SYSTEM + text[0], wav, 24000, "")
    out["text_ids"] = cross["text"][0].long()

    # The language model, as `llm_job` runs it for a cross-lingual reading.
    set_all_random_seed(SEED)
    said = list(llm.inference(
        text=cross["text"], text_len=torch.tensor([cross["text"].shape[1]], dtype=torch.int32),
        prompt_text=torch.zeros(1, 0, dtype=torch.int32),
        prompt_text_len=torch.zeros(1, dtype=torch.int32),
        prompt_speech_token=torch.zeros(1, 0, dtype=torch.int32),
        prompt_speech_token_len=torch.zeros(1, dtype=torch.int32),
        embedding=torch.zeros(0, 192)))
    said = [int(token) for token in said]
    out["said"] = torch.tensor(said, dtype=torch.int64)
    print(f"said {len(said)} tokens for {cross['text'].shape[1]} text tokens")

    # The same reading teacher-forced: one row per position a token was drawn at, and the one after.
    with torch.inference_mode():
        embed = llm.llm.model.model.embed_tokens
        sos = llm.speech_embedding.weight[llm.sos].reshape(1, 1, -1)
        task = llm.speech_embedding.weight[llm.task_id].reshape(1, 1, -1)
        prefix = torch.cat([sos, embed(cross["text"]), task], dim=1)
        spoken = llm.speech_embedding(torch.tensor([said]))
        hidden, _ = llm.llm(torch.cat([prefix, spoken], dim=1),
                            torch.tensor([prefix.shape[1] + len(said)]))
        logp = llm.llm_decoder(hidden[0, prefix.shape[1] - 1:]).log_softmax(dim=-1)
    out["lm_logp"] = logp.float()

    # The zero-shot prefix: the transcript before the sentence and the prompt's tokens after task.
    zero = frontend.frontend_zero_shot(text[0], SYSTEM + PROMPT_TEXT, wav, 24000, "")
    ids = torch.cat([zero["prompt_text"], zero["text"]], dim=1)
    out["zero_shot_ids"] = ids[0].long()
    with torch.inference_mode():
        prompt = llm.speech_embedding(zero["llm_prompt_speech_token"])
        lm_input = torch.cat([sos, embed(ids), task, prompt], dim=1)
        hidden, _ = llm.llm(lm_input, torch.tensor([lm_input.shape[1]]))
        out["zero_shot_logp"] = llm.llm_decoder(hidden[0, -1:]).log_softmax(dim=-1).float()
    out["flow_prompt_tokens"] = cross["flow_prompt_speech_token"][0].long()
    out["flow_prompt_feat"] = cross["prompt_speech_feat"].float()

    # The flow, with its condition and the estimator's first call recorded on the way.
    decoder_calls, estimator_calls = [], []
    record_calls(flow.decoder, decoder_calls)
    record_calls(flow.decoder.estimator, estimator_calls)
    with torch.inference_mode():
        mel, _ = flow.inference(
            token=torch.tensor([said], dtype=torch.int32),
            token_len=torch.tensor([len(said)], dtype=torch.int32),
            prompt_token=cross["flow_prompt_speech_token"],
            prompt_token_len=torch.tensor([cross["flow_prompt_speech_token"].shape[1]],
                                          dtype=torch.int32),
            prompt_feat=cross["prompt_speech_feat"],
            prompt_feat_len=torch.tensor([cross["prompt_speech_feat"].shape[1]], dtype=torch.int32),
            embedding=cross["flow_embedding"], streaming=False, finalize=True)
    out["flow_mu"] = decoder_calls[0][1]["mu"].float()
    (x, mask, mu, t, spks, cond), _, velocity = estimator_calls[0]
    out["dit_x"], out["dit_t"], out["dit_velocity"] = x.float(), t.float(), velocity.float()
    out["dit_spks"], out["dit_cond"] = spks.float(), cond.float()
    out["mel"] = mel.float()

    # The vocoder, with noise of this script's choosing in place of what its constructor drew.
    length = mel.shape[2] * 480
    generator = torch.Generator().manual_seed(SEED)
    rand_ini = torch.rand(1, 9, generator=generator)
    rand_ini[:, 0] = 0
    sine_noise = torch.rand(1, length, 9, generator=generator)
    uv_noise = torch.rand(1, length, 1, generator=generator)
    sine = hift.m_source.l_sin_gen
    sine.rand_ini, sine.sine_waves, hift.m_source.uv = rand_ini, sine_noise, uv_noise
    out["hift_rand_ini"], out["hift_sine_noise"], out["hift_uv_noise"] = rand_ini, sine_noise, uv_noise

    f0_calls = []
    record_calls(hift.f0_predictor, f0_calls)
    with torch.inference_mode():
        wave, source = hift.inference(speech_feat=mel, finalize=True)
    out["f0"] = f0_calls[0][2].float()
    out["source"] = source.float()
    out["wave"] = wave.float()
    print(f"mel {tuple(mel.shape)}, wave {tuple(wave.shape)}")

    from safetensors.torch import save_file

    save_file({name: tensor.contiguous() for name, tensor in out.items()}, arguments.output,
              metadata={"text": TEXT, "prompt_text": PROMPT_TEXT, "system": SYSTEM})
    print("wrote", arguments.output)

    import soundfile

    listen = os.path.splitext(arguments.output)[0] + ".wav"
    soundfile.write(listen, wave[0].numpy(), 24000)
    print("wrote", listen)


if __name__ == "__main__":
    main()
