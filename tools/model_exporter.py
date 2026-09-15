# The MIT License (MIT)
#
# Copyright (c) 2023 Xiaoyang Chen
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

from __future__ import annotations

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from torch import nn

import torch

# What a model is written as.
from model_writer import (
    MANIFEST_SUFFIX, WEIGHTS_SUFFIX, Context, Quant, Quantization, TensorBag, WeightsWriter,
    parse_size, part_names, read_manifest, save_tensors, stem_of, write_manifest)


def open_weights(output: str, part_size=None):
    """A writer for a model's weights, which the exporter then hands tensors to.

    `output` names the model: `sdxl-base.safetensors` writes that file and `sdxl-base.yaml` beside
    it, and with a `part_size` writes `sdxl-base-00001-of-00002.safetensors` and its neighbours
    instead.
    """
    return WeightsWriter(output, part_size)


class ModelExporter:
    def __init__(self, writer: WeightsWriter) -> None:
        self._writer = writer

    def _write(self, ctx: Context, tensor: torch.Tensor):
        self._writer.write_tensor(ctx, tensor)

    def export_embedding(self, ctx: Context, module: nn.Embedding):
        ctx = ctx.with_subname("weight")
        self._write(ctx, module.weight)

    def export_linear(self, ctx: Context, module, has_bias=True):
        self._write(ctx.with_subname("weight"), module.weight)
        if has_bias:
            self._write(ctx.with_subname("bias").with_quant(Quant.NONE), module.bias)

    def export_layer_norm(self, ctx: Context, module: nn.LayerNorm):
        self._write(ctx.with_subname("weight").with_quant(Quant.NONE), module.weight)
        self._write(ctx.with_subname("bias").with_quant(Quant.NONE), module.bias)
