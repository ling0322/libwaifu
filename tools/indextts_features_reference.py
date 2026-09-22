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

# The waveform is a chirp, not noise

A sweep from 80 Hz to 6 kHz, which crosses most of the filterbank and lands different energy in
different bands at different times. Noise would exercise the same code and make every band look
alike, and a band that is wrong would then be hard to see.
"""

import numpy as np

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

    print(f"// w2v-bert: {features.shape[0]} pairs of {features.shape[1]} from {SAMPLES} samples")
    print(f"// campplus: {centred.shape[0]} frames of {centred.shape[1]}")


if __name__ == "__main__":
    main()
