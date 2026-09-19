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

"""CAMPPlus's checkpoint as safetensors: `campplus_cn_common.bin` in, `waifu::campplus` out.

    .venv/bin/python tools/campplus_exporter.py -output models/campplus-192.safetensors

CAMPPlus is the speaker encoder IndexTTS-2.5 conditions on: 80-band Kaldi filterbank energies of
the reference audio in, one 192-dimensional vector out, which is what tells the rest of the
pipeline whose voice it is. It is 3D-Speaker's under Apache 2.0, published as `funasr/campplus`
and shared by several speech systems; nothing about it is IndexTTS's.

Note the embedding: IndexTTS builds `CAMPPlus(feat_dim=80, embedding_size=192)`, not the class's
own default of 512.

# What is done to the weights

The batch normalizations are folded, and nothing else is touched.

A `BatchNorm1d` in `eval` mode is not a normalization at all -- it is a fixed per-channel affine,
`(x - mean) / sqrt(var + eps) * weight + bias`, with the statistics frozen into the checkpoint.
So each one is folded to the two vectors that describe it,

    scale = weight / sqrt(running_var + eps)
    shift = bias - running_mean * scale

which is exact rather than an approximation, and leaves the runtime with a multiply and an add in
place of an operator it does not have. `scale` and `shift` are written shaped `(C, 1)` for the
1-D ones and `(C, 1, 1)` for the 2-D, which is what broadcasts against `(N, C, L)` and
`(N, C, H, W)` without a transpose.

The normalizations in `DenseLayer` are `affine=False` -- no weight, no bias -- so their scale is
just the reciprocal deviation and their shift is what centring leaves behind. They are folded the
same way, and the arithmetic is the same.

Everything else keeps the checkpoint's name, shape and value, so `xvector.block1.tdnnd1.linear1.weight`
in the file is that in the graph, and the two can be compared tensor for tensor when one is wrong.
The folded pairs take their module's name -- `head.bn1` becomes `head.bn1.scale` and
`head.bn1.shift` -- so they are still traceable to what produced them.

# Why the weights are not what the fast test uses

They are 6.9 M parameters and a download. `waifu/tests/campplus.rs` builds a small model whose
every parameter comes from its own name, against numbers `tools/campplus_reference.py` printed
from this same upstream code; see that file. What this exports is for the `#[ignore]`d test that
checks the real release, and for running the model at all.
"""

import argparse
import os
import sys
import urllib.request

import torch

from huggingface_hub import hf_hub_download

REPO = "funasr/campplus"
CHECKPOINT = "campplus_cn_common.bin"

# The reference implementation, which is read rather than reimplemented. `DTDNN.py` imports its
# neighbour by the absolute package name, so the two files have to sit in that package for the
# import to resolve -- hence the directories below rather than a flat download.
SOURCE_ROOT = "https://raw.githubusercontent.com/index-tts/index-tts/main/"
SOURCES = (
    "indextts/s2mel/modules/campplus/layers.py",
    "indextts/s2mel/modules/campplus/DTDNN.py",
)

# What `torch.nn.BatchNorm1d` defaults to, and what this checkpoint was trained with.
BATCH_NORM_EPS = 1e-5


def upstream(cache=None):
    """3D-Speaker's CAMPPlus, downloaded and importable.

    Returns the `DTDNN` module. The files are laid out as the package they name themselves --
    `indextts/s2mel/modules/campplus/` -- because `DTDNN.py` imports its neighbour absolutely,
    and an `__init__.py` is written at every level so the import resolves.
    """
    cache = cache or os.path.join(
        os.path.expanduser("~"), ".cache", "libwaifu", "campplus-source"
    )

    for name in SOURCES:
        target = os.path.join(cache, *name.split("/"))
        os.makedirs(os.path.dirname(target), exist_ok=True)

        if not os.path.isfile(target):
            urllib.request.urlretrieve(SOURCE_ROOT + name, target)

    # Every directory on the way down has to be a package for the absolute import to work.
    parts = SOURCES[0].split("/")[:-1]
    for depth in range(len(parts)):
        init = os.path.join(cache, *parts[: depth + 1], "__init__.py")
        if not os.path.isfile(init):
            open(init, "w").close()

    if cache not in sys.path:
        sys.path.insert(0, cache)

    from indextts.s2mel.modules.campplus import DTDNN

    return DTDNN


def checkpoint(path=None):
    """The released weights, from a file or from the hub."""
    return torch.load(
        path or hf_hub_download(repo_id=REPO, filename=CHECKPOINT),
        map_location="cpu",
        weights_only=True,
    )


def fold_batch_norm(module, spatial: int):
    """The `(scale, shift)` a batch normalization is, once it has stopped normalizing.

    `spatial` is how many axes follow the channel -- one for a `(N, C, L)` tensor and two for
    `(N, C, H, W)` -- and is what the two vectors are shaped to broadcast against.
    """
    variance = module.running_var
    mean = module.running_mean
    eps = getattr(module, "eps", BATCH_NORM_EPS)

    scale = torch.rsqrt(variance + eps)
    if module.weight is not None:
        scale = scale * module.weight

    shift = -mean * scale
    if module.bias is not None:
        shift = shift + module.bias

    shape = (-1,) + (1,) * spatial

    return scale.reshape(shape).contiguous(), shift.reshape(shape).contiguous()


def parameters(model):
    """Every tensor the graph asks for, by the name it asks for it under.

    A walk of the modules rather than of the state dict, because what a batch normalization folds
    to depends on which kind it is, and only the module says.
    """
    out = {}

    for name, module in model.named_modules():
        kind = type(module).__name__

        if kind in ("BatchNorm1d", "BatchNorm2d"):
            scale, shift = fold_batch_norm(module, 1 if kind == "BatchNorm1d" else 2)
            out[f"{name}.scale"] = scale
            out[f"{name}.shift"] = shift

        elif kind in ("Conv1d", "Conv2d"):
            out[f"{name}.weight"] = module.weight.detach().contiguous()
            if module.bias is not None:
                out[f"{name}.bias"] = module.bias.detach().contiguous()

    return out


def build(DTDNN, feat_dim=80, embedding_size=192):
    """The released model, loaded and put in eval so the normalizations are affine."""
    model = DTDNN.CAMPPlus(feat_dim=feat_dim, embedding_size=embedding_size)
    model.load_state_dict(checkpoint())
    model.eval()

    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-output", help="where to write the exported weights")
    parser.add_argument("-feat_dim", type=int, default=80)
    parser.add_argument("-embedding_size", type=int, default=192)
    arguments = parser.parse_args()

    model = build(upstream(), arguments.feat_dim, arguments.embedding_size)
    tensors = parameters(model)

    total = sum(value.numel() for value in tensors.values())
    print(f"{len(tensors)} tensors, {total / 1e6:.2f} M parameters")

    if arguments.output:
        from safetensors.torch import save_file

        os.makedirs(os.path.dirname(arguments.output) or ".", exist_ok=True)
        save_file(tensors, arguments.output)
        print("wrote", arguments.output)


if __name__ == "__main__":
    main()
