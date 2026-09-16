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

"""Why the Qwen-Image VAE needs no conv3d for one image, in either direction.

The Qwen-Image VAE is a video VAE: its weights are <out, in, kt, kh, kw> and its convolutions are
causal in time. flint has conv2d and no conv3d. This shows the two agree exactly when the
temporal extent is one, which is what it is for every image this runtime draws, so the exporter
can fold each Conv3d onto W[:, :, -1] and hand the runtime a plain 2-D convolution.

The causal layer here is the semantics of QwenImageCausalConv3d in
diffusers.models.autoencoders.autoencoder_kl_qwenimage: pad the time axis by 2 * padding[0] at
the front only, through F.pad, whose default is a constant zero. With T = 1 those two frames are
zeros, so of the three temporal taps only the last one multiplies anything.

Nothing in the decoder grows the time axis either: upsample3d doubles it only on the
feat_cache path, and the first -- for an image, only -- chunk skips time_conv entirely.

The encoder is the same argument run backwards, and it matters now that the package carries the
encoder for image to image: its spatial halving is already an nn.Conv2d applied per frame, and
the only thing that would touch the time axis is downsample3d's time_conv, which the first chunk
skips. The last check below is why skipping it is the only thing to do rather than a shortcut --
at T = 1 that layer cannot produce an output at all.

Run it with the project venv:

    .venv/bin/python tools/causal_conv3d_folding_test.py
"""

import sys

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv3d(nn.Conv3d):
    """The reference layer, kept to its padding and nothing else."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self._padding = (
            self.padding[2],
            self.padding[2],
            self.padding[1],
            self.padding[1],
            2 * self.padding[0],
            0,
        )
        self.padding = (0, 0, 0)

    def forward(self, x, cache_x=None):
        return super().forward(F.pad(x, list(self._padding)))


def check(name, in_channels, out_channels, kernel_size, padding):
    """Run one frame through both paths and return the largest disagreement."""
    layer = CausalConv3d(in_channels, out_channels, kernel_size, padding=padding).double().eval()
    x = torch.randn(2, in_channels, 1, 9, 11).double()

    with torch.no_grad():
        reference = layer(x)
        folded = F.conv2d(
            x[:, :, 0],
            layer.weight[:, :, -1],
            layer.bias,
            padding=(padding[1], padding[2]),
        )

    if reference.shape[2] != 1:
        raise AssertionError(f"{name}: the time axis grew to {reference.shape[2]}")

    error = (reference[:, :, 0] - folded).abs().max().item()
    print(f"{name:<20} max|diff| = {error:.3e}  {'ok' if error == 0.0 else 'MISMATCH'}")
    return error


def encoder_time_conv_cannot_run():
    """The encoder's downsample3d time_conv, given the one frame an image is.

    It is stride 2 in time with no padding in front of it, so at T = 1 there is nothing for a
    three tap kernel to read: torch refuses it outright. That is the whole argument for the
    exporter dropping these weights rather than folding them -- there is no 2-D convolution to
    fold them onto, and a picture never reaches the branch that would want one.
    """
    layer = CausalConv3d(8, 8, (3, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0)).double().eval()
    x = torch.randn(2, 8, 1, 9, 11).double()

    try:
        with torch.no_grad():
            layer(x)
    except RuntimeError:
        print(f"{'3x1x1 stride 2 t':<20} unrunnable at T = 1  ok")
        return True

    print(f"{'3x1x1 stride 2 t':<20} ran at T = 1         UNEXPECTED")
    return False


def main():
    torch.manual_seed(0)
    worst = max(
        # the convolution most of both halves is made of
        check("3x3x3 pad(1,1,1)", 8, 12, (3, 3, 3), (1, 1, 1)),
        # the encoder's stem, which reads a three channel picture
        check("3x3x3 stem", 3, 12, (3, 3, 3), (1, 1, 1)),
        # conv1 and conv2, either side of the latent: the moments out of the encoder and the
        # sixteen channels into the decoder
        check("1x1x1 pad(0,0,0)", 32, 32, (1, 1, 1), (0, 0, 0)),
        check("1x1x1 pad(0,0,0)", 16, 16, (1, 1, 1), (0, 0, 0)),
        # the time_conv an upsample3d carries, which an image never reaches
        check("3x1x1 pad(1,0,0)", 8, 16, (3, 1, 1), (1, 0, 0)),
    )
    if worst != 0.0:
        print("folding is not exact; the exporter cannot rely on it", file=sys.stderr)
        return 1

    if not encoder_time_conv_cannot_run():
        print("the encoder's time_conv ran over one frame; the exporter drops weights it "
              "should not", file=sys.stderr)
        return 1

    print("\nfolding is exact: a Conv3d over one frame is Conv2d over W[:, :, -1]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
