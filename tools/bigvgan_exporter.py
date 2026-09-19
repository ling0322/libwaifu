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

"""BigVGAN's checkpoint as safetensors: `bigvgan_generator.pt` in, `waifu::bigvgan` out.

    .venv/bin/python tools/bigvgan_exporter.py -output models/bigvgan-22khz-80band.safetensors
    .venv/bin/python tools/bigvgan_exporter.py -test_output models/bigvgan-22khz-80band_test.safetensors

The vocoder IndexTTS-2.5 fetches at first run is `nvidia/bigvgan_v2_22khz_80band_256x`, and that
is what this exports with no arguments. Nothing about it is IndexTTS's -- it is NVIDIA's release
under the MIT licence, published separately and shared by several speech models.

# What is done to the weights

One thing: the weight normalization is folded out. Every convolution in a BigVGAN is wrapped in
`torch.nn.utils.weight_norm`, which stores a direction `weight_v` and a magnitude `weight_g` and
multiplies them each time the layer runs. That is a thing training wants and inference does not,
which is why `IndexTTS2.__init__` calls `remove_weight_norm()` on the vocoder as it loads it --
so the folded weight is what the reference itself runs, and what comes out here is a copy of it
rather than a transformation of the checkpoint.

Nothing else changes. Names, shapes and values are the checkpoint's, so `resblocks.3.convs1.1.weight`
in the file is `resblocks.3.convs1.1.weight` in the graph, and the two can be compared tensor for
tensor when one of them is wrong.

# The reference tensors

`-test_output` writes what `waifu/tests/bigvgan.rs` compares the runtime's audio against: a mel
spectrogram, and the waveform this model made of it. The mel is computed by the reference's own
`meldataset.mel_spectrogram` over a chirp, so the analysis in front of the vocoder is the one it
was trained with rather than one written here.
"""

import argparse
import contextlib
import math
import os
import sys

import torch

from huggingface_hub import hf_hub_download

from model_writer import save_tensors

# The vocoder `indextts/utils/model_download.py` names, and the only one this has been run
# against. Another BigVGAN v2 exports the same way; one with `resblock: "2"` does not, and the
# runtime refuses that configuration rather than running it as the other kind.
REPO = "nvidia/bigvgan_v2_22khz_80band_256x"

# The implementation, which is downloaded rather than vendored: what the runtime is checked
# against has to be the thing NVIDIA published, not a copy of it that has drifted.
SOURCES = (
    "activations.py",
    "bigvgan.py",
    "env.py",
    "utils.py",
    "meldataset.py",
    "alias_free_activation/torch/__init__.py",
    "alias_free_activation/torch/act.py",
    "alias_free_activation/torch/filter.py",
    "alias_free_activation/torch/resample.py",
)


def upstream():
    """NVIDIA's BigVGAN, downloaded and importable.

    `bigvgan.py` imports its neighbours by bare name -- `import activations`, `from utils import
    get_padding` -- so what goes on the path is the snapshot directory they all sit in.
    """
    root = None
    for name in SOURCES:
        root = hf_hub_download(repo_id=REPO, filename=name)[: -len(name)]

    if root not in sys.path:
        sys.path.insert(0, root)

    import bigvgan

    return bigvgan


def checkpoint(directory=None):
    """The generator's weights and the configuration beside them, from a directory or the hub."""
    names = ("config.json", "bigvgan_generator.pt")

    if directory is None:
        return [hf_hub_download(repo_id=REPO, filename=name) for name in names]

    return [os.path.join(directory, name) for name in names]


def vocoder(directory=None):
    """The model, loaded and folded flat."""
    bigvgan = upstream()

    config_path, weights_path = checkpoint(directory)
    config = bigvgan.load_hparams_from_json(config_path)

    if config.resblock != "1":
        raise SystemExit(f"resblock {config.resblock!r} is not the kind waifu::bigvgan reads")

    model = bigvgan.BigVGAN(config)

    # `weights_only` because a checkpoint is data and this one comes off the network; the file
    # holds the generator beside the optimizer state that trained it.
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state["generator"])

    with contextlib.redirect_stdout(sys.stderr):
        model.remove_weight_norm()

    model.eval()

    return model, config


def chirp(seconds: float, rate: int) -> torch.Tensor:
    """A sweep from 80 Hz to a quarter of the rate, which is a signal a mel can actually hold.

    Noise would exercise the same arithmetic and say less: a vocoder given the mel of a chirp
    should give back something that is still a chirp, so a reference waveform made this way can be
    looked at as well as compared.
    """
    samples = int(seconds * rate)
    low, high = 80.0, rate / 4.0

    # Constant relative rate of change, so the sweep spends as long per octave at the top as at
    # the bottom. The phase is the integral of that, which is where the exponential comes from.
    rise = math.log(high / low)
    phase = [
        2.0 * math.pi * low * seconds / rise * (math.exp(rise * i / samples) - 1.0)
        for i in range(samples)
    ]

    return torch.tensor([[math.sin(p) * 0.5 for p in phase]], dtype=torch.float32)


def reference(model, config) -> dict:
    """A mel and the waveform this vocoder makes of it, for `waifu/tests/bigvgan.rs`."""
    from meldataset import mel_spectrogram

    wave = chirp(0.25, config.sampling_rate)
    mel = mel_spectrogram(
        wave,
        config.n_fft,
        config.num_mels,
        config.sampling_rate,
        config.hop_size,
        config.win_size,
        config.fmin,
        config.fmax,
        center=False,
    )

    with torch.no_grad():
        waveform = model(mel)

    print(f"mel {tuple(mel.shape)} -> waveform {tuple(waveform.shape)}, "
          f"peak {waveform.abs().max().item():.4f}")

    # float32 both, `preserve_dtype` in `TensorBag`'s terms: the runtime reads this vocoder in
    # float32 on the processor, and a reference rounded to half would put a floor under the
    # comparison well above what the two implementations actually differ by.
    return {"mel": mel, "waveform": waveform}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "-checkpoint", type=str, default=None,
        help="a directory holding config.json and bigvgan_generator.pt; the hub by default")
    parser.add_argument(
        "-output", type=str, default=None,
        help="where to write the weights, as one safetensors file")
    parser.add_argument(
        "-test_output", type=str, default=None,
        help="where to write the reference mel and waveform")
    args = parser.parse_args()

    if args.output is None and args.test_output is None:
        parser.error("nothing to do: give -output, -test_output, or both")

    model, config = vocoder(args.checkpoint)

    if args.output is not None:
        weights = {name: value.detach() for name, value in model.state_dict().items()}
        save_tensors(args.output, weights)

        total = sum(value.numel() for value in weights.values())
        print(f"wrote {os.path.basename(args.output)}: "
              f"{len(weights)} tensors, {total / 1e6:.1f} M parameters")

    if args.test_output is not None:
        save_tensors(args.test_output, reference(model, config))
        print(f"wrote {os.path.basename(args.test_output)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
