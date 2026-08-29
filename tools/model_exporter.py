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

import binascii
import io
import os
import struct
import torch
import math
import sys
import zipfile
from os import path
import numpy as np
from enum import Enum
import torch.nn.functional as F

DTYPE_UNKNOWN = 0
DTYPE_FP32 = 1
DTYPE_INT64 = 2
DTYPE_UINT8 = 3
DTYPE_FP16 = 4
DTYPE_QINT4_32 = 5
DTYPE_INT8 = 6

class Quant(Enum):
    NONE = 0
    Q4 = 2

    @classmethod
    def parse(cls, quant: str) -> Quant:
        quant = quant.lower()
        if quant == "q4":
            return Quant.Q4
        elif quant == "none":
            return Quant.NONE
        else:
            raise NotImplementedError("unsupported quantization type: " + quant)

class Context:
    """stores the context of  module and tensor."""

    def __init__(self, name="", quant=Quant.NONE) -> None:
        self._ns = name
        self._quant = quant

    def _copy(self) -> Context:
        """get a cpy of current context."""
        ctx = Context()
        ctx._ns = self._ns
        ctx._quant = self._quant

        return ctx

    def _subname(self, name: str) -> str:
        return name if not self._ns else self._ns + '.' + name

    @property
    def name(self) -> str:
        return self._ns if self._ns else "<root>"
    
    @property
    def quant(self) -> Quant:
        return self._quant

    def with_subname(self, name: str) -> Context:
        """get the context object with a sub-namespace"""
        ctx = self._copy()
        ctx._ns = self._subname(name)
        return ctx

    def with_quant(self, quant: Quant) -> Context:
        """returns a context object the same as current context, the only difference is
        quantization setting."""
        ctx = self._copy()
        ctx._quant = quant
        return ctx

class Quantization:
    @classmethod
    def _pack_uint8_to_uint4x2(cls, tensor: torch.Tensor) -> torch.Tensor:
        assert tensor.dtype == torch.uint8 and tensor.dim() == 1
        assert torch.all(tensor <= 15)

        if tensor.shape[0] % 2 == 1:
            pad_value = torch.zeros((1, ), dtype=torch.uint8, device=tensor.device)
            tensor = torch.cat((tensor, pad_value))
        tensor = tensor.reshape(-1, 2)
        tensor = tensor[:, 0].type(torch.uint8) + tensor[:, 1].type(torch.uint8) * 16
        return tensor

    @classmethod
    def quantize_to_qint4x32(cls, tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """1D tensor to qdata (q4x2), scale (fp16), zero (fp16) """
        weights = tensor.reshape(-1, 32)
        num_group = weights.shape[0]

        min_value = torch.min(weights, 1).values
        max_value = torch.max(weights, 1).values

        scales = torch.clamp(max_value - min_value, min=1e-5) / 15
        zeros = -min_value

        qweights = torch.round((weights - min_value.reshape(num_group, 1)) / scales.reshape(num_group, 1))
        qweights = qweights.clamp(0, 15).reshape(-1).type(torch.uint8)
        qweights = cls._pack_uint8_to_uint4x2(qweights)

        return qweights, scales.type(torch.float16), zeros.type(torch.float16)

MODEL_INI = "model.ini"

# The key in model.ini that lists the parts of a model written as several files, and what the
# runtime looks for. A package without it is the whole model.
PARTS_KEY = "model_parts"

PACKAGE_SUFFIX = ".waifupkg"


def parse_size(text: str) -> int:
    """A size as a number of bytes, written plainly or with a unit: 4GB, 512MB, 2000000000."""
    units = {"": 1, "K": 1000, "KB": 1000, "M": 1000**2, "MB": 1000**2, "MIB": 1 << 20,
             "G": 1000**3, "GB": 1000**3, "GI": 1 << 30, "GIB": 1 << 30}
    text = text.strip().upper()
    digits = text.rstrip("ABGIKM")
    unit = text[len(digits):]
    if not digits or unit not in units:
        raise ValueError(f"{text!r} is not a size")
    return int(float(digits) * units[unit])


def part_names(stem: str, count: int) -> list:
    """What the parts of a model split `count` ways are called.

    A model that fits in one file keeps the name it was asked for; there is no `-00001-of-00001`,
    since a suffix saying "one of one" only invites the question of where the others are.
    """
    if count == 1:
        return [stem + PACKAGE_SUFFIX]
    return [f"{stem}-{i + 1:05d}-of-{count:05d}{PACKAGE_SUFFIX}" for i in range(count)]


class _CountingWriter:
    """Passes writes through and remembers how many bytes went by."""

    def __init__(self, fp) -> None:
        self._fp = fp
        self.written = 0

    def write(self, data) -> int:
        self.written += len(data)
        return self._fp.write(data)

    def close(self) -> None:
        self._fp.close()


class PackageWriter:
    """Writes a model as one package, or as several if a size limit is given.

    Seven gigabytes is an awkward size to publish and an awkward one to fetch: it cannot be
    downloaded in parallel, and a failed transfer starts over. Given `part_size` this rolls over
    to a new package whenever the current one reaches it, always between tensors and never inside
    one, so that each part is a whole parameter file in its own right -- its own header, its own
    records, its own end -- and can be read and checked alone.

    How many parts there will be is not known until the last tensor has been written, so they are
    written under temporary names and renamed at the end, when the count is finally known.

    The first part carries everything that is not parameters -- the configuration, the tokenizer
    -- and names the others. It is reopened to take them, since by then it has been closed.
    """

    def __init__(self, output: str, model_file: str, part_size=None) -> None:
        self._model_file = model_file
        self._part_size = part_size
        self._directory = path.dirname(path.abspath(output))
        name = path.basename(output)
        self._stem = name[: -len(PACKAGE_SUFFIX)] if name.endswith(PACKAGE_SUFFIX) else name

        self._temporary = []
        self._zip = None
        self._writer = None
        self._counter = None
        self._begin_part()

    def _begin_part(self) -> None:
        temporary = path.join(self._directory, f"{self._stem}-{len(self._temporary):05d}.part")
        self._zip = zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED)
        entry = self._zip.open(self._model_file, "w", force_zip64=True)
        self._counter = _CountingWriter(entry)
        self._writer = TensorWriter(self._counter)
        self._temporary.append(temporary)

    def _end_part(self) -> None:
        # TensorWriter closes what it was given, which closes the entry but not the archive.
        self._writer.__exit__(None, None, None)
        self._zip.close()

    def write_tensor(self, ctx, tensor, preserve_dtype=False) -> None:
        """What the exporter writes through. Rolls over first if this part has had enough."""
        if self._part_size is not None and self._counter.written >= self._part_size:
            self._end_part()
            self._begin_part()
        self._writer.write_tensor(ctx, tensor, preserve_dtype=preserve_dtype)

    def finish(self, config, extras=None) -> list:
        """Close the parts, name them, and put what is not parameters into the first.

        `config` is the model.ini to write, which is told about the parts when there is more than
        one of them. `extras` is called with the first part open for appending, for the entries
        that are neither the parameters nor the configuration.
        """
        self._end_part()

        names = part_names(self._stem, len(self._temporary))
        for temporary, name in zip(self._temporary, names):
            os.replace(temporary, path.join(self._directory, name))

        if len(names) > 1:
            config["model"][PARTS_KEY] = ",".join(names)

        first = path.join(self._directory, names[0])
        with zipfile.ZipFile(first, "a", compression=zipfile.ZIP_STORED) as package:
            with package.open(MODEL_INI, "w", force_zip64=True) as fp:
                config.write(io.TextIOWrapper(fp))
            if extras is not None:
                extras(package)

        return names


class TensorWriter:
    """write tensor to file with llyn tensor format."""

    def __init__(self, fp) -> None:
        self._fp = fp
        self._fp.write(b"llyn::tdicv2    ")
        self._fp.write(b"<d> ")

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        print("TensorWriter: __exit__")
        self._fp.write(b"</d>")
        self._fp.close()

    def _write_tensor_elem(self, tensor: torch.Tensor, dtype=DTYPE_UNKNOWN):
        if dtype == DTYPE_UNKNOWN:
            dtype = self._dtype_to_libwaifu_dtype(tensor.dtype)

        numel = tensor.numel()
        self._fp.write(struct.pack('<h', dtype))
        self._fp.write(struct.pack('<q', numel))

        np_tensor = tensor.cpu().detach().contiguous().numpy()
        assert np_tensor.dtype in {np.dtype(np.float32), np.dtype(np.float16), np.dtype(np.int64), np.dtype(np.int8), np.dtype(np.uint8)}

        self._fp.write(np_tensor.tobytes())
        self._fp.write(struct.pack('<h', 0x55aa))

    def _write_tensor_header(self, shape, num_slot=1):
        self._fp.write(b'tnsr')
        self._fp.write(struct.pack('<h', len(shape)))
        for size in shape:
            self._fp.write(struct.pack('<i', size))
        
        # header for data.
        self._fp.write(b"tdat")
        self._fp.write(struct.pack('<i', num_slot))


    def _write_tensor(self, tensor: torch.Tensor, preserve_dtype=False):
        if tensor.dtype == torch.float32 and not preserve_dtype:
            tensor = tensor.to(torch.float16)
        assert tensor.dtype in {torch.float32, torch.float16, torch.int64}

        self._write_tensor_header(tensor.shape)
        self._write_tensor_elem(tensor)
        

    def _dtype_to_libwaifu_dtype(self, dtype):
        if dtype == torch.float32:
            return DTYPE_FP32
        if dtype == torch.float16:
            return DTYPE_FP16
        if dtype == torch.int64:
            return DTYPE_INT64
        if dtype == torch.int8:
            return DTYPE_INT8
        if dtype == torch.uint8:
            return DTYPE_UINT8
        else:
            raise Exception("dtype not supported")

    def _write_tensor_qint4x32(self, tensor: torch.Tensor):
        """ quantize the pytorch tensor to q4 format (int4 asymmetric quantization, group size 32,
        scale format float16). Then write to self._fp.
        """
        if tensor.dtype == torch.float16:
            tensor = tensor.float()
        assert tensor.dtype in {torch.float32, torch.int64}

        self._write_tensor_header(tensor.shape)
        
        qdata, scale, zero = Quantization.quantize_to_qint4x32(tensor)
        assert qdata.dim() == 1 and scale.dim() == 1 and zero.dim() == 1
        assert qdata.dtype == torch.uint8 and scale.dtype == torch.float16 and zero.dtype == torch.float16

        num_group = int(qdata.shape[0] / 16)
        qdata = qdata.cpu().detach().contiguous().numpy()
        scale = scale.cpu().detach().contiguous().numpy()
        zero = zero.cpu().detach().contiguous().numpy()

        qdata = np.frombuffer(qdata.tobytes(), np.uint8).reshape(-1, 16)
        scale = np.frombuffer(scale.tobytes(), np.uint8).reshape(-1, 2)
        zero = np.frombuffer(zero.tobytes(), np.uint8).reshape(-1, 2)

        blocks = np.hstack((zero, scale, qdata))
        self._fp.write(struct.pack('<h', DTYPE_QINT4_32))

        numel = num_group * 32
        self._fp.write(struct.pack('<q', numel))

        bdata = blocks.tobytes()
        assert len(bdata) == num_group * 20
        self._fp.write(bdata)

        self._fp.write(struct.pack('<h', 0x55aa))

    def write_tensor(self, ctx: Context, tensor: torch.Tensor, preserve_dtype=False):
        print(f"write tensor {ctx.name}, shape={tensor.shape}, quant={ctx.quant}")
        self._fp.write(b"<r> ")

        if len(ctx.name) > 1024:
            raise Exception('name too long')
        name = ctx.name.encode('utf-8')
        self._fp.write(struct.pack('<h', len(name)))
        self._fp.write(name)

        if ctx.quant == Quant.NONE:
            self._write_tensor(tensor, preserve_dtype=preserve_dtype)
        elif ctx.quant == Quant.Q4:
            self._write_tensor_qint4x32(tensor)
        else:
            raise NotImplementedError(ctx.quant)

        self._fp.write(b"</r>")

class ModelExporter:
    def __init__(self, writer: TensorWriter) -> None:
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
