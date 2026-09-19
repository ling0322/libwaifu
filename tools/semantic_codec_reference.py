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

"""The numbers `waifu/tests/semantic_codec.rs` checks the codec against, from the codec itself.

It downloads IndexTTS's own `codec/models.py` and what it leans on, builds a small one, and
prints what `quantize` produces as Rust constants -- both the token each frame came out as and
the similarity the tokens were chosen from, since the choosing is the part done on the host.

Run it with the project venv, from the repository root:

    .venv/bin/python tools/semantic_codec_reference.py

# The model is small, but its count is the release's

Twelve ConvNeXt layers, as shipped. What is cut is every width: a 32-wide feature instead of
1024, a 64-entry codebook instead of 8192, a 24-wide trunk instead of 384. The codebook width
stays at eight, because that is already small and it is what "factorized" means here.

The frame count is even on purpose. `quantize` halves the frame rate with a stride-two
convolution and does not check, where `forward` drops the last frame when the count is odd; an
odd count here would be comparing two different lengths.
"""

import math
import os
import sys
import urllib.request

import torch

SOURCE_ROOT = "https://raw.githubusercontent.com/index-tts/index-tts/main/"
SOURCES = (
    "indextts/codec/models.py",
    "indextts/codec/kmeans/vocos.py",
    "indextts/codec/amphion_codec/quantize/__init__.py",
    "indextts/codec/amphion_codec/quantize/factorized_vector_quantize.py",
    "indextts/codec/amphion_codec/quantize/lookup_free_quantize.py",
    "indextts/codec/amphion_codec/quantize/residual_vq.py",
    "indextts/codec/amphion_codec/quantize/vector_quantize.py",
)

HIDDEN = 32
CODEBOOK_SIZE = 64
CODEBOOK_DIM = 8
VOCOS_DIM = 24
VOCOS_INTERMEDIATE = 48
VOCOS_LAYERS = 12
FRAMES = 20

WEIGHT_SCALE = 0.25
INPUT_SCALE = 0.6
PROBE = 8


def upstream(cache=None):
    """IndexTTS's codec, downloaded and importable."""
    cache = cache or os.path.join(os.path.expanduser("~"), ".cache", "libwaifu", "codec-source")

    for name in SOURCES:
        target = os.path.join(cache, *name.split("/"))
        os.makedirs(os.path.dirname(target), exist_ok=True)
        if not os.path.isfile(target):
            try:
                urllib.request.urlretrieve(SOURCE_ROOT + name, target)
            except urllib.error.HTTPError:
                # Not every name in the list has to exist; the package imports what it has.
                open(target, "w").close()

    for name in SOURCES:
        parts = name.split("/")[:-1]
        for depth in range(len(parts)):
            init = os.path.join(cache, *parts[: depth + 1], "__init__.py")
            if not os.path.isfile(init):
                open(init, "w").close()

    _torchaudio_shim(cache)

    if cache not in sys.path:
        sys.path.insert(0, cache)

    from indextts.codec import models

    return models


def _torchaudio_shim(cache):
    """The two mel helpers `vocos.py` imports from torchaudio, and nothing else.

    `vocos.py` does `from torchaudio.functional.functional import _hz_to_mel, _mel_to_hz` at
    module scope, for a mel spectrogram class further down the file that `VocosBackbone` never
    touches. torchaudio itself is not installed here and should not be: it pins itself against a
    torch version, and this venv's pins are what hold several models' reference tensors steady.

    So the two functions are written out instead. They are three lines each and this is their
    real arithmetic, not a stub that raises -- a shim that lies would be worse than no shim, and
    anything that did reach them would get the right answer.
    """
    root = os.path.join(cache, "torchaudio", "functional")
    os.makedirs(root, exist_ok=True)

    for init in (
        os.path.join(cache, "torchaudio", "__init__.py"),
        os.path.join(root, "__init__.py"),
    ):
        if not os.path.isfile(init):
            open(init, "w").close()

    target = os.path.join(root, "functional.py")
    if os.path.isfile(target):
        return

    with open(target, "w") as out:
        out.write(
            '''"""Written by tools/semantic_codec_reference.py -- see `_torchaudio_shim`."""

import math


def _hz_to_mel(freq: float, mel_scale: str = "htk") -> float:
    if mel_scale == "htk":
        return 2595.0 * math.log10(1.0 + (freq / 700.0))

    f_min, f_sp = 0.0, 200.0 / 3
    mels = (freq - f_min) / f_sp

    min_log_hz = 1000.0
    min_log_mel = (min_log_hz - f_min) / f_sp
    logstep = math.log(6.4) / 27.0
    if freq >= min_log_hz:
        mels = min_log_mel + math.log(freq / min_log_hz) / logstep

    return mels


def _mel_to_hz(mels, mel_scale: str = "htk"):
    if mel_scale == "htk":
        return 700.0 * (10.0 ** (mels / 2595.0) - 1.0)

    f_min, f_sp = 0.0, 200.0 / 3
    freqs = f_min + f_sp * mels

    min_log_hz = 1000.0
    min_log_mel = (min_log_hz - f_min) / f_sp
    logstep = math.log(6.4) / 27.0

    try:
        import torch

        if torch.is_tensor(mels):
            log_t = mels >= min_log_mel
            freqs[log_t] = min_log_hz * torch.exp(logstep * (mels[log_t] - min_log_mel))
            return freqs
    except ImportError:
        pass

    if mels >= min_log_mel:
        return min_log_hz * math.exp(logstep * (mels - min_log_mel))

    return freqs
'''
        )


def hash32(name: str) -> int:
    """FNV-1a over the parameter's name. The Rust test computes the same thing the same way."""
    value = 0x811C9DC5
    for byte in name.encode():
        value = ((value ^ byte) * 0x01000193) & 0xFFFFFFFF

    return value


def fill(name: str, count: int, scale: float) -> torch.Tensor:
    """`count` numbers that are this parameter's and no other's."""
    seed = (hash32(name) % 10007) * 0.001
    values = [math.sin(i * 0.7371 + seed) * scale for i in range(count)]

    return torch.tensor(values, dtype=torch.float32)


def codec(models):
    """A small EnhancedCodec, weight normalization folded and every parameter filled by name."""
    from munch import Munch

    cfg = Munch(
        codebook_size=CODEBOOK_SIZE,
        hidden_size=HIDDEN,
        codebook_dim=CODEBOOK_DIM,
        vocos_dim=VOCOS_DIM,
        vocos_intermediate_dim=VOCOS_INTERMEDIATE,
        vocos_num_layers=VOCOS_LAYERS,
        num_quantizers=1,
        downsample_scale=2,
    )

    model = models.EnhancedCodec(**cfg, cfg=cfg)
    model.eval()

    for child in model.modules():
        try:
            torch.nn.utils.remove_weight_norm(child)
        except (ValueError, RuntimeError, AttributeError):
            pass

    with torch.no_grad():
        for name, parameter in model.named_parameters():
            parameter.copy_(fill(name, parameter.numel(), WEIGHT_SCALE).reshape(parameter.shape))

    return model


def emit(name, tensor, limit, integer=False):
    """One probe as a Rust constant: `limit` numbers spread through it."""
    flat = tensor.reshape(-1).tolist()

    if limit >= len(flat):
        shown, where = flat, "all"
    else:
        step = max(1, len(flat) // limit)
        indices = [(i * step + i * 7) % len(flat) for i in range(limit)]
        shown, where = [flat[i] for i in indices], f"at {indices}"

    kind = "i32" if integer else "f32"
    print(f"/// `{name}`, {tuple(tensor.shape)} -- {len(shown)} of {len(flat)}, {where}.")
    print(f"const {name.upper()}: [{kind}; {len(shown)}] = [")
    for index in range(0, len(shown), 4):
        if integer:
            row = ", ".join(f"{int(value)}" for value in shown[index : index + 4])
        else:
            row = ", ".join(f"{value:.6e}" for value in shown[index : index + 4])
        print(f"    {row},")
    print("];")
    print()


def main():
    models = upstream()
    model = codec(models)

    x = fill("features", FRAMES * HIDDEN, INPUT_SCALE).reshape(1, FRAMES, HIDDEN)

    # The similarity the tokens are chosen from, which is all the graph can do; capturing it needs
    # the quantizer's own projection, so it is recomputed here the way `decode_latents` does.
    quantizer = model.quantizer.quantizers[0]

    with torch.no_grad():
        codes, quantized = model.quantize(x)

        # The same path `quantize` takes, stopped one step earlier.
        half = torch.nn.functional.gelu(model.down(x.transpose(1, 2)))
        encoded = model.encoder(half).transpose(1, 2)
        latents = quantizer.in_project(encoded)

        normed = torch.nn.functional.normalize(latents.transpose(1, 2), dim=-1)
        book = torch.nn.functional.normalize(quantizer.codebook.weight, dim=-1)
        similarity = normed @ book.t()

    print(f"// Generated by tools/semantic_codec_reference.py -- {FRAMES} frames.")
    print()
    emit("similarity", similarity, PROBE)
    emit("codes", codes, codes.numel(), integer=True)

    print("// shapes:", {"codes": tuple(codes.shape), "similarity": tuple(similarity.shape),
                         "quantized": tuple(quantized.shape)}, file=sys.stderr)


if __name__ == "__main__":
    main()
