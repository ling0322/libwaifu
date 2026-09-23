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

"""The numbers `waifu/tests/indextts_features.rs` checks the audio front end against.

    .venv/bin/python tools/indextts_features_reference.py

# The reference is the extractor IndexTTS actually calls

`SeamlessM4TFeatureExtractor` is in the pinned transformers in this venv, and it is the thing
`infer_v2_5.py` runs over the reference recording -- so the w2v-bert half of this needs no
download and no transcription. What comes out of it is compared against
`waifu::indextts_features::w2v_bert` sample for sample.

The CAMPPlus half is the same analysis with the scaling and the normalization changed, and
upstream reaches it through `torchaudio.compliance.kaldi.fbank`. torchaudio is not installed here
and should not be -- it pins itself against a torch version, and this venv's pins are what hold
several models' reference tensors steady. So that half is built from `transformers.audio_utils`
instead, which is the same `spectrogram` the extractor above uses, configured the way Kaldi's
defaults configure it. The extractor's own docstring says it is computing "mel-filter bank
features using TorchAudio", so the two agree by construction rather than by coincidence.

# Resampling and the 22.05 kHz mel come from upstream too

The reference recording is resampled by `torchaudio.transforms.Resample`, and torchaudio is not
here. Its resampler is two plain-torch functions, so they are lifted out of torchaudio's own
`functional.py` -- fetched once into `~/.cache/libwaifu/torchaudio-source` -- with `ast` and run
as written. The mel S2Mel conditions on is IndexTTS's own `s2mel/modules/audio.py`, imported
from the cached upstream tree; it needs only librosa, which is here.

# The waveform is a chirp, not noise

A sweep from 80 Hz to 6 kHz, which crosses most of the filterbank and lands different energy in
different bands at different times. Noise would exercise the same code and make every band look
alike, and a band that is wrong would then be hard to see.
"""

import ast
import math
import os
import sys
import types

import numpy as np
import torch

from transformers.audio_utils import mel_filter_bank, spectrogram, window_function
from transformers.models.seamless_m4t.feature_extraction_seamless_m4t import (
    SeamlessM4TFeatureExtractor,
)

RATE = 16000
SECONDS = 0.5
SAMPLES = int(RATE * SECONDS)

BINS = 80
FRAME_LENGTH = 400
HOP = 160
FFT_LENGTH = 512


UPSTREAM = os.path.expanduser("~/.cache/libwaifu/indextts-src")
TORCHAUDIO = os.path.expanduser("~/.cache/libwaifu/torchaudio-source/functional.py")
TORCHAUDIO_URL = (
    "https://raw.githubusercontent.com/pytorch/audio/main/src/torchaudio/functional/functional.py"
)

# The rates the resampler is checked between: a common recording rate onto both of the rates
# the pipeline needs.
SOURCE_RATE = 44100
MEL_RATE = 22050


def torchaudio_resample():
    """torchaudio's `_get_sinc_resample_kernel` and `_apply_sinc_resample_kernel`, as written."""
    if not os.path.exists(TORCHAUDIO):
        import urllib.request

        os.makedirs(os.path.dirname(TORCHAUDIO), exist_ok=True)
        urllib.request.urlretrieve(TORCHAUDIO_URL, TORCHAUDIO)

    tree = ast.parse(open(TORCHAUDIO).read())
    wanted = {"_get_sinc_resample_kernel", "_apply_sinc_resample_kernel"}
    body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in wanted]

    from typing import Optional

    namespace = {
        "torch": torch,
        "math": math,
        "warnings": __import__("warnings"),
        "Optional": Optional,
        "Tensor": torch.Tensor,
        "_CPU": torch.device("cpu"),
    }
    exec(compile(ast.Module(body=body, type_ignores=[]), TORCHAUDIO, "exec"), namespace)

    def resample(wave, orig, new):
        gcd = math.gcd(orig, new)
        kernel, width = namespace["_get_sinc_resample_kernel"](orig, new, gcd)
        return namespace["_apply_sinc_resample_kernel"](wave, orig, new, gcd, kernel, width)

    return resample


def upstream_mel():
    """IndexTTS's own `mel_spectrogram`, with S2Mel's parameters as `infer_v2_5.py` passes them."""
    sys.path.insert(0, UPSTREAM)
    from indextts.s2mel.modules.audio import mel_spectrogram

    def mel(wave):
        return mel_spectrogram(
            torch.from_numpy(wave).unsqueeze(0),
            n_fft=1024,
            num_mels=80,
            sampling_rate=MEL_RATE,
            hop_size=256,
            win_size=1024,
            fmin=0,
            fmax=None,
        )[0]

    return mel


def chirp_at(rate):
    """The same sweep as `chirp`, at another rate."""
    samples = int(rate * SECONDS)
    time = np.arange(samples) / rate
    sweep = 80.0 + (6000.0 - 80.0) * time / SECONDS
    phase = 2.0 * np.pi * np.cumsum(sweep) / rate

    return (0.1 * np.sin(phase)).astype(np.float32)


def broadband_at(rate):
    """The sweep with a little noise under it, so that every band has something in it.

    A sweep alone is narrowband: at any moment it is in one band, and every other cell of a mel
    spectrogram sits on the `1e-5` floor -- where two implementations agree whatever they do.
    The noise is a plain linear congruential generator in integers, which the Rust test repeats
    exactly rather than approximately.
    """
    wave = chirp_at(rate).astype(np.float64)
    state = 12345
    for index in range(wave.size):
        state = (1103515245 * state + 12345) % (1 << 31)
        wave[index] += (state / (1 << 31) - 0.5) * 0.1

    return wave.astype(np.float32)


def chirp():
    """A sweep from 80 Hz to 6 kHz, at a tenth of full scale."""
    time = np.arange(SAMPLES) / RATE
    sweep = 80.0 + (6000.0 - 80.0) * time / SECONDS
    phase = 2.0 * np.pi * np.cumsum(sweep) / RATE

    return (0.1 * np.sin(phase)).astype(np.float32)


def kaldi_filters():
    """The filterbank both halves share, as the extractor builds it."""
    filters = mel_filter_bank(
        num_frequency_bins=FFT_LENGTH // 2,
        num_mel_filters=BINS,
        min_frequency=20,
        max_frequency=RATE // 2,
        sampling_rate=RATE,
        norm=None,
        mel_scale="kaldi",
        triangularize_in_mel_space=True,
    )

    return np.pad(filters, ((0, 1), (0, 0)))


def log_mel(wave, scale):
    """Kaldi's filterbank: the shared core, with only the input scaling free."""
    return spectrogram(
        np.squeeze(wave) * scale,
        window_function(FRAME_LENGTH, "povey", periodic=False),
        frame_length=FRAME_LENGTH,
        hop_length=HOP,
        fft_length=FFT_LENGTH,
        power=2.0,
        center=False,
        preemphasis=0.97,
        mel_filters=kaldi_filters(),
        log_mel="log",
        mel_floor=1.192092955078125e-07,
        remove_dc_offset=True,
    ).T


def probe_indices(length, limit=8):
    step = max(length // limit, 1)
    return [(i * step + i * 7) % length for i in range(limit)]


def emit(name, array, comment):
    values = np.asarray(array).flatten().tolist()
    indices = probe_indices(len(values))

    print(f"/// `{name}`, {tuple(np.asarray(array).shape)} -- {comment}")
    print(f"///   8 of {len(values)}, at {indices}.")
    print(f"const {name.upper()}: [f32; 8] = [")
    for index in indices:
        print(f"    {values[index]:.6e},")
    print("];")
    print()


def main():
    wave = chirp()

    # w2v-bert: the extractor IndexTTS calls, whole.
    extractor = SeamlessM4TFeatureExtractor(feature_size=BINS, num_mel_bins=BINS, stride=2)
    extracted = extractor(wave, sampling_rate=RATE, return_tensors="np")
    features = extracted["input_features"][0]

    # CAMPPlus: the same analysis, unscaled, with each band's mean over time removed.
    energies = log_mel(wave, 1.0)
    centred = energies - energies.mean(axis=0, keepdims=True)

    print(f"// Generated by tools/indextts_features_reference.py -- {SAMPLES} samples at {RATE}.")
    print()
    emit("w2v_features", features, "standardized per band, then stacked in pairs")
    emit("campplus_features", centred, "each band's mean over time removed")

    # Resampling: a 44.1 kHz sweep onto both of the rates the pipeline needs.
    resample = torchaudio_resample()
    source = torch.from_numpy(chirp_at(SOURCE_RATE))
    to_16k = resample(source, SOURCE_RATE, RATE)
    to_22k = resample(source, SOURCE_RATE, MEL_RATE)
    emit("resampled_16k", to_16k, "44.1 kHz onto 16 kHz by torchaudio's own sinc")
    emit("resampled_22k", to_22k, "44.1 kHz onto 22.05 kHz")

    # The mel S2Mel conditions on, of a broadband signal made at 22.05 kHz directly.
    reference = upstream_mel()(broadband_at(MEL_RATE))
    emit("reference_mel", reference, "(bands, frames), IndexTTS's own mel_spectrogram")
    print(f"// reference mel cells on the 1e-5 floor: {int((reference <= math.log(1e-5) + 1e-3).sum())}")

    print(f"// resampled: {to_16k.shape[-1]} at 16 kHz, {to_22k.shape[-1]} at 22.05 kHz")
    print(f"// reference mel: {tuple(reference.shape)}")
    print(f"// w2v-bert: {features.shape[0]} pairs of {features.shape[1]} from {SAMPLES} samples")
    print(f"// campplus: {centred.shape[0]} frames of {centred.shape[1]}")


if __name__ == "__main__":
    main()
