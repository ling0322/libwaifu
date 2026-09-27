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

"""What upstream Seed-VC v2 computes on one real conversion, for `waifu/tests/seed_vc.rs`.

    PYTHONPATH=~/.cache/libwaifu/torchaudio-stub \\
        .venv/bin/python tools/seed_vc_reference.py -output models/seed_vc_test.safetensors

Nothing here is a transcription. Upstream's own classes -- `VoiceConversionWrapper`, `CFM`,
`DiT`, `NaiveWrapper`, `AstralQuantizer`, `CAMPPlus`, its BigVGAN -- are built with the numbers
`configs/v2/vc_wrapper.yaml` gives them, loaded by the wrapper's own `load_checkpoints`, and run
through `convert_voice_with_streaming`, which is the path `inference_v2.py` and the web demo
take. What each stage was handed and what it gave back is recorded on the way through by
wrapping the stage, so every input a test hands the runtime is one upstream was really handed.

The checkout is `Plachtaa/seed-vc` at `COMMIT`, under `~/.cache/libwaifu/src/seed-vc`. hydra is
not needed: the yaml is small and its `_target_`s are written out below as the calls they make.

# The recordings

`examples/source/glados_0.wav` and `examples/reference/teio_0.wav` from that checkout, each cut to
its first `SECONDS` seconds and written as `seed_vc_test_source.wav` and
`seed_vc_test_reference.wav` beside the output. Upstream reads a file with `librosa.load(sr=22050)`
and resamples that to 16 kHz with `librosa.resample`; both are written here as upstream had them,
so that a stage test hands the runtime the same samples and a resampler difference cannot hide
in a model's error.

# Precision

float32 on the CPU. `inference_v2.py` runs under `autocast(float16)` on a card; that is a
narrowing upstream chose for speed, and the model is the float32 one.

# What is written

| name | what |
| --- | --- |
| `source_22k`, `reference_22k`, `source_16k`, `reference_16k` | the waves as upstream held them |
| `reference_mel` | `(1, 80, M)`, the reference's mel, the CFM's prompt |
| `hubert` | `(1, T, 1024)`, HuBERT's 18th layer on the source |
| `wide_projected`, `narrow_projected` | `(1, T, 11)` and `(1, T, 5)`, BSQ's input projection |
| `wide_source`, `wide_reference`, `narrow_source`, `narrow_reference` | the token ids |
| `style` | `(1, 192)`, CAMPPlus on the reference |
| `prompt_condition`, `condition` | the CFM length regulator on the reference's and source's tokens |
| `ar_condition`, `ar_said`, `ar_logits` | the AR's prefix, the tokens it said, and its raw scores |
| `dit_*` | one estimator call from the middle of the timbre trajectory: the batch of three |
| `cfm_noise`, `cfm_mel` | the timbre trajectory's starting noise and where it ended |
| `timbre_wave`, `style_wave` | what each path of the conversion produced |
"""

import argparse
import os
import subprocess
import sys
import types

import numpy as np
import torch

REPO = "https://github.com/Plachtaa/seed-vc"
COMMIT = "51383efd921027683c89e5348211d93ff12ac2a8"
CHECKOUT = os.path.join(os.path.expanduser("~"), ".cache", "libwaifu", "src", "seed-vc")

SOURCE = "examples/source/glados_0.wav"
REFERENCE = "examples/reference/teio_0.wav"
SECONDS = 6.0

# `inference_v2.py`'s defaults; `convert_style` is both.
STEPS = 30
CFG = (0.7, 0.7)
TOP_P, TEMPERATURE, REPETITION = 0.9, 1.0, 1.0

# Which estimator call is kept: one from the middle of the trajectory, where the mel is neither
# noise nor finished.
DIT_STEP = 12
# How many of the AR's steps keep their scores. Every step is checked by teacher forcing, but
# 2049 floats a step is more fixture than the check needs past the first few dozen.
AR_STEPS = 48


def checkout():
    """Upstream's source, at the commit this was written against."""
    if not os.path.isdir(CHECKOUT):
        subprocess.run(["git", "clone", "-q", REPO, CHECKOUT], check=True)
    head = subprocess.run(["git", "-C", CHECKOUT, "rev-parse", "HEAD"], check=True,
                          capture_output=True, text=True).stdout.strip()
    if head != COMMIT:
        subprocess.run(["git", "-C", CHECKOUT, "checkout", "-q", COMMIT], check=True)
    return CHECKOUT


def importable():
    """Upstream on the path, with the modules inference never reaches stood in for.

    `munch` is imported by `modules/commons.py` for training configs; `pydub` by the wrapper to
    encode the mp3 it streams, which is thrown away here.
    """
    root = checkout()
    if root not in sys.path:
        sys.path.insert(0, root)

    munch = types.ModuleType("munch")
    munch.Munch = dict
    sys.modules.setdefault("munch", munch)

    class AudioSegment:
        def __init__(self, *args, **kwargs):
            pass

        def export(self, *args, **kwargs):
            class Empty:
                def read(self):
                    return b""
            return Empty()

    pydub = types.ModuleType("pydub")
    pydub.AudioSegment = AudioSegment
    sys.modules.setdefault("pydub", pydub)
    return root


def build():
    """`configs/v2/vc_wrapper.yaml`, instantiated by hand, with the release loaded."""
    importable()

    from modules.audio import mel_spectrogram
    from modules.astral_quantization.bsq import BinarySphericalQuantize
    from modules.astral_quantization.convnext import ConvNeXtV2Stage
    from modules.astral_quantization.default_model import AstralQuantizer
    from modules.bigvgan.bigvgan import BigVGAN
    from modules.campplus.DTDNN import CAMPPlus
    from modules.v2.ar import NaiveModelArgs, NaiveTransformer, NaiveWrapper
    from modules.v2.cfm import CFM
    from modules.v2.dit_wrapper import DiT
    from modules.v2.length_regulator import InterpolateRegulator
    from modules.v2.vc_wrapper import VoiceConversionWrapper

    import functools

    mel_fn = functools.partial(mel_spectrogram, n_fft=1024, win_size=1024, hop_size=256,
                               num_mels=80, sampling_rate=22050, fmin=0, fmax=None, center=False)

    cfm = CFM(estimator=DiT(
        time_as_token=True, style_as_token=True, uvit_skip_connection=False, block_size=8192,
        depth=13, num_heads=8, hidden_dim=512, in_channels=80, content_dim=512,
        style_encoder_dim=192, class_dropout_prob=0.1, dropout_rate=0.0, attn_dropout_rate=0.0))
    cfm_regulator = InterpolateRegulator(channels=512, is_discrete=True, codebook_size=2048,
                                         sampling_ratios=[1, 1, 1, 1], f0_condition=False)
    ar = NaiveWrapper(model=NaiveTransformer(config=NaiveModelArgs(
        dropout=0.0, rope_base=10000.0, dim=768, head_dim=64, n_local_heads=2,
        intermediate_size=2304, n_head=12, n_layer=12, vocab_size=2049)))
    ar_regulator = InterpolateRegulator(channels=768, is_discrete=True, codebook_size=32,
                                        sampling_ratios=[], f0_condition=False)

    def quantizer(codebook_size, skip_ssl):
        return AstralQuantizer(
            tokenizer_name="openai/whisper-small",
            ssl_model_name="facebook/hubert-large-ll60k",
            ssl_output_layer=18,
            skip_ssl=skip_ssl,
            encoder=ConvNeXtV2Stage(dim=512, num_blocks=12, intermediate_dim=1536, dilation=1,
                                    input_dim=1024),
            quantizer=BinarySphericalQuantize(
                codebook_size=codebook_size, dim=512, entropy_loss_weight=0.1,
                diversity_gamma=1.0, spherical=True, enable_entropy_loss=True,
                soft_entropy_loss=True))

    wrapper = VoiceConversionWrapper(
        sr=22050, hop_size=256, mel_fn=mel_fn, cfm=cfm, cfm_length_regulator=cfm_regulator,
        content_extractor_narrow=quantizer(32, True),
        content_extractor_wide=quantizer(2048, False),
        ar_length_regulator=ar_regulator, ar=ar,
        style_encoder=CAMPPlus(feat_dim=80, embedding_size=192),
        vocoder=BigVGAN.from_pretrained("nvidia/bigvgan_v2_22khz_80band_256x",
                                        use_cuda_kernel=False))

    # The wrapper's own loader, which downloads through `hf_utils` into ./checkpoints; run it
    # from a scratch directory so that lands out of the tree.
    here = os.getcwd()
    scratch = os.path.join(os.path.expanduser("~"), ".cache", "libwaifu", "seed-vc-checkpoints")
    os.makedirs(scratch, exist_ok=True)
    os.chdir(scratch)
    try:
        wrapper.load_checkpoints()
    finally:
        os.chdir(here)

    wrapper.eval()
    wrapper.setup_ar_caches(max_batch_size=1, max_seq_len=4096, dtype=torch.float32,
                            device=torch.device("cpu"))
    return wrapper


def cut(source, target, seconds):
    """The first `seconds` of `source`, mixed to one channel, at its own rate, as a float WAV."""
    import soundfile

    wave, rate = soundfile.read(source, dtype="float32", always_2d=True)
    wave = wave.mean(axis=1)[: int(seconds * rate)]
    soundfile.write(target, wave, rate, subtype="FLOAT")


class Recorder:
    """Wraps callables so what they were handed and gave back is kept, as copies.

    Copies, not references: `solve_euler` zeroes the prompt's frames of `x` in place, and the AR's
    sampler scatters its repetition penalty into the scores it was handed.
    """

    def __init__(self):
        self.calls = {}

    @staticmethod
    def copied(value):
        if isinstance(value, torch.Tensor):
            return value.detach().clone()
        if isinstance(value, (list, tuple)):
            return type(value)(Recorder.copied(item) for item in value)
        if hasattr(value, "logits"):
            # `forward_generate`'s result, whose scores the sampler then edits in place.
            return types.SimpleNamespace(logits=value.logits.detach().clone())
        return value

    def wrap(self, owner, name, key):
        original = getattr(owner, name)
        calls = self.calls.setdefault(key, [])

        def recorded(*args, **kwargs):
            entry = {"args": self.copied(args), "kwargs": {k: self.copied(v)
                                                           for k, v in kwargs.items()}}
            out = original(*args, **kwargs)
            entry["out"] = self.copied(out)
            calls.append(entry)
            return out

        setattr(owner, name, recorded)
        return original


def convert(wrapper, source, reference, convert_style):
    """One run of `convert_voice_with_streaming`, with upstream's defaults, collected."""
    wave = None
    for _, full in wrapper.convert_voice_with_streaming(
            source_audio_path=source, target_audio_path=reference, diffusion_steps=STEPS,
            length_adjust=1.0, intelligebility_cfg_rate=CFG[0], similarity_cfg_rate=CFG[1],
            top_p=TOP_P, temperature=TEMPERATURE, repetition_penalty=REPETITION,
            convert_style=convert_style, anonymization_only=False,
            device=torch.device("cpu"), dtype=torch.float32, stream_output=True):
        if full is not None:
            wave = full[1]
    return torch.from_numpy(np.asarray(wave, dtype=np.float32))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-output", required=True, help="e.g. models/seed_vc_test.safetensors")
    parser.add_argument("-seed", type=int, default=1234)
    arguments = parser.parse_args()

    torch.set_grad_enabled(False)
    directory = os.path.dirname(os.path.abspath(arguments.output))
    stem = os.path.splitext(os.path.basename(arguments.output))[0]
    source = os.path.join(directory, stem + "_source.wav")
    reference = os.path.join(directory, stem + "_reference.wav")

    root = checkout()
    cut(os.path.join(root, SOURCE), source, SECONDS)
    cut(os.path.join(root, REFERENCE), reference, SECONDS)

    wrapper = build()

    import librosa

    out = {}
    source_22k = librosa.load(source, sr=22050)[0]
    reference_22k = librosa.load(reference, sr=22050)[0][: 22050 * (wrapper.dit_max_context_len - 5)]
    out["source_22k"] = torch.from_numpy(source_22k)
    out["reference_22k"] = torch.from_numpy(reference_22k)
    out["source_16k"] = torch.from_numpy(librosa.resample(source_22k, orig_sr=22050, target_sr=16000))
    out["reference_16k"] = torch.from_numpy(
        librosa.resample(reference_22k, orig_sr=22050, target_sr=16000))
    out["reference_mel"] = wrapper.mel_fn(out["reference_22k"][None])

    # ---- The timbre path: no AR. ----
    recorder = Recorder()
    wide, narrow = wrapper.content_extractor_wide, wrapper.content_extractor_narrow
    recorder.wrap(wide.ssl_model, "forward", "hubert")
    recorder.wrap(wide.quantizer.project_in, "forward", "wide_projected")
    recorder.wrap(narrow.quantizer.project_in, "forward", "narrow_projected")
    recorder.wrap(wide, "forward", "wide")
    recorder.wrap(narrow, "forward", "narrow")
    recorder.wrap(wrapper, "compute_style", "style")
    recorder.wrap(wrapper.cfm_length_regulator, "forward", "cfm_regulator")
    recorder.wrap(wrapper.cfm, "solve_euler", "solve_euler")
    recorder.wrap(wrapper.cfm.estimator, "forward", "estimator")
    recorder.wrap(wrapper.vocoder, "forward", "vocoder")

    torch.manual_seed(arguments.seed)
    out["timbre_wave"] = convert(wrapper, source, reference, False)
    calls = recorder.calls

    # `_process_content_features` runs the source first, then the reference.
    hubert = calls["hubert"][0]["out"].last_hidden_state
    out["hubert"] = hubert
    out["wide_projected"] = calls["wide_projected"][0]["out"]
    out["wide_source"] = calls["wide"][0]["out"][1].long()
    out["wide_reference"] = calls["wide"][1]["out"][1].long()
    out["style"] = calls["style"][0]["out"]
    out["prompt_condition"] = calls["cfm_regulator"][0]["out"][0]
    out["condition"] = calls["cfm_regulator"][1]["out"][0]

    euler = calls["solve_euler"][0]
    out["cfm_noise"] = euler["args"][0]
    out["cfm_mel"] = euler["out"]
    out["cfm_prompt_len"] = torch.tensor([euler["args"][2].size(-1)])

    x, prompt_x, x_lens, t, style, mu = calls["estimator"][DIT_STEP]["args"]
    out["dit_x"], out["dit_prompt_x"], out["dit_t"] = x, prompt_x, t
    out["dit_style"], out["dit_cond"] = style, mu
    out["dit_out"] = calls["estimator"][DIT_STEP]["out"]

    out["vocoder_mel"] = calls["vocoder"][-1]["args"][0]
    out["vocoder_wave"] = calls["vocoder"][-1]["out"]

    # ---- The style path: the AR in front. ----
    recorder = Recorder()
    recorder.wrap(narrow, "forward", "narrow")
    recorder.wrap(narrow.quantizer.project_in, "forward", "narrow_projected")
    recorder.wrap(wrapper.ar, "generate", "generate")
    recorder.wrap(wrapper.ar.model, "forward_generate", "forward_generate")

    torch.manual_seed(arguments.seed)
    out["style_wave"] = convert(wrapper, source, reference, True)
    calls = recorder.calls

    out["narrow_projected"] = calls["narrow_projected"][0]["out"]
    out["narrow_source"] = calls["narrow"][0]["out"][1].long()
    out["narrow_reference"] = calls["narrow"][1]["out"][1].long()

    generate = calls["generate"][0]
    out["ar_condition"] = generate["args"][0]
    out["ar_prompt"] = generate["args"][1].long()
    out["ar_said"] = generate["out"].long().reshape(-1)
    logits = [call["out"].logits.reshape(-1) for call in calls["forward_generate"][:AR_STEPS]]
    out["ar_logits"] = torch.stack(logits)

    from safetensors.torch import save_file

    save_file({name: tensor.contiguous().float() if tensor.is_floating_point()
               else tensor.contiguous() for name, tensor in out.items()}, arguments.output,
              metadata={"commit": COMMIT, "seed": str(arguments.seed)})

    import soundfile

    for name in ("timbre_wave", "style_wave"):
        soundfile.write(os.path.join(directory, f"{stem}_{name}.wav"), out[name].numpy(), 22050)

    for name, tensor in out.items():
        print(f"  {name}: {tuple(tensor.shape)} {tensor.dtype}")
    print(f"wrote {arguments.output}")


if __name__ == "__main__":
    main()
