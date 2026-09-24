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

"""Writing a model: its weights, and the manifest that says what they are.

A model is a `<model-id>.yaml` and the files it names -- safetensors for the weights, one file per
tokenizer for the vocabularies. There used to be a zip of this project's own holding all of it, in
either of two layouts of this project's own, and none of it could be read by anything but
libwaifu. Now every file a model is made of is one something else already reads.

This is the writing half. `manifest.rs` and `param_file.rs` are the reading half, and
`tools/convert_model.py` is what turns a model written the old way into one written this way.
"""

from __future__ import annotations

from enum import Enum
import math
import os
from os import path

import torch
from safetensors.torch import save_file

MANIFEST_SUFFIX = ".yaml"

# The blocks, in the order they are written.
WEIGHTS_KEY = "weights"
TOKENIZERS_KEY = "tokenizers"
CONFIG_KEY = "config"
SUGGESTED_KEY = "suggested"

# What a file of weights is called. safetensors, because every other runtime reads it: a weight
# can be looked at with torch, with numpy, or with the viewer on the hub, and a format of one's
# own has to earn that cost.
WEIGHTS_SUFFIX = ".safetensors"

DATA_MAGIC = 0x55AA


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

    @classmethod
    def quantize_to_fp8(cls, tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """A 2D weight to E4M3 elements and one float32 scale per row.

        The same arithmetic the runtime's own quantizer does, and it has to be: a package written
        here is read by that, and `docs/fp8.md` is one format rather than two only if every
        producer agrees on the bytes.

            scale[r]   = amax(|x[r]|) / 448          # 448 is E4M3's largest finite magnitude
            data[r][j] = e4m3(x[r][j] / scale[r])

        so each row's largest element lands exactly on the top of the format's range. A row that is
        all zero gets a zero scale and quantizes to zeros rather than to a NaN.

        Three details are not free to change:

        - The division is float32 and round to nearest. One ulp moves elements onto the next code:
          on the CUDA side `--use_fast_math` turning this into a reciprocal approximation moved 2
          elements in 16896, which is why that kernel divides with `__fdiv_rn`.
        - The clamp is the saturation `__NV_SATFINITE` does. Without it torch sends what rounds
          past 448 to NaN, and the row maximum can land a hair above it.
        - `float8_e4m3fn` is the format: 448 max, no infinities, NaN at 0x7f. That is what
          `__NV_E4M3` is too.
        """
        if tensor.dim() != 2:
            raise ValueError(f"fp8 takes a matrix, not {tuple(tensor.shape)}")

        weights = tensor.detach().cpu().to(torch.float32)
        scales = weights.abs().amax(dim=1) / FP8_E4M3_MAX

        # An all zero row has no scale to speak of; dividing by one is what would make a NaN.
        divisor = torch.where(scales > 0, scales, torch.ones_like(scales))
        data = weights / divisor.unsqueeze(1)
        data = data.clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(torch.float8_e4m3fn)

        return data, scales

    @classmethod
    def quantize_to_fp8_tensor_scale(cls, tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """A 2D weight to E4M3 elements and one float32 scale for the whole matrix.

        The same rounding and the same saturation as `quantize_to_fp8`, over the whole tensor
        rather than one row at a time:

            scale      = amax(|x|) / 448          # 448 is E4M3's largest finite magnitude
            data[i][j] = e4m3(x[i][j] / scale)

        One `<float>` beside the elements instead of the `(out,)` `quantize_to_fp8` writes -- which
        is what `flint::gemmFp8TensorScale` reads, a single value its epilogue broadcasts rather
        than a row vector. The saving is smaller still to write and load; what it gives up is the
        row-by-row scaling `quantize_to_fp8` does: a single outlier row's magnitude sets the scale
        for every other row too, so a weight whose rows vary widely in magnitude loses more to this
        format than to the per-row one.

        An all zero tensor has no scale to speak of; dividing by one is what would make a NaN.
        """
        if tensor.dim() != 2:
            raise ValueError(f"fp8 takes a matrix, not {tuple(tensor.shape)}")

        weights = tensor.detach().cpu().to(torch.float32)
        scale = weights.abs().amax() / FP8_E4M3_MAX

        divisor = scale if scale > 0 else torch.ones_like(scale)
        data = weights / divisor
        data = data.clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(torch.float8_e4m3fn)

        return data, scale.reshape(1)


#: E4M3's largest finite magnitude, which is what a row's maximum is scaled onto.
FP8_E4M3_MAX = 448.0

#: What a package calls the scales beside a weight it stored quantized. The runtime looks for this
#: name exactly -- `flint::CHANNEL_SCALE_SUFFIX` -- and refuses an E4M3 tensor without one.
FP8_SCALE_SUFFIX = ".scale"


def dtype_code(dtype) -> int:
    """The number a torch element type is written as, which is what DType::from_code reads."""
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
    raise Exception("dtype not supported")


def tensor_bytes(tensor: torch.Tensor) -> bytes:
    """A tensor's elements, laid out as a package holds them."""
    array = tensor.cpu().detach().contiguous().numpy()
    assert array.dtype in {np.dtype(np.float32), np.dtype(np.float16), np.dtype(np.int64),
                           np.dtype(np.int8), np.dtype(np.uint8)}
    return array.tobytes()


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
    """What the weights of a model split `count` ways are called.

    A model that fits in one file keeps the name it was asked for; there is no `-00001-of-00001`,
    since a suffix saying "one of one" only invites the question of where the others are.
    """
    if count == 1:
        return [stem + WEIGHTS_SUFFIX]
    return [f"{stem}-{i + 1:05d}-of-{count:05d}{WEIGHTS_SUFFIX}" for i in range(count)]


def assign(items: list, lengths: list, part_size: int) -> list:
    """Which items go in which part, given how long each one is.

    The number of parts is what the limit calls for, and the items are then spread evenly over
    that many rather than filling each to the brim and leaving a small last one: two parts of
    three and a half gigabytes download better than one of five and one of two. Nothing is ever
    split across parts, and no part is left empty.
    """
    total = sum(lengths)
    count = max(1, math.ceil(total / part_size))
    target = math.ceil(total / count)

    parts = [[]]
    used = 0
    for item, length in zip(items, lengths):
        if used + length > target and parts[-1] and len(parts) < count:
            parts.append([])
            used = 0
        parts[-1].append(item)
        used += length

    return parts


def stem_of(output: str) -> str:
    """A file's name with its suffix off, which is what a model's files are named after."""
    name = path.basename(output)
    for suffix in (WEIGHTS_SUFFIX, MANIFEST_SUFFIX):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


# What a scalar may not begin with, if it is to be written without quotes: the characters YAML
# gives a meaning to at the start of a value -- an indicator, an anchor, a block marker -- which
# a value starting with one would be read as rather than as itself.
_INDICATORS = ",[]{}#&*!|>'\"%@`"

# The three that only mean something when a space follows, which is why they are not in the set
# above. `-0.7571,-0.7089,...` is a plain scalar and quoting it would only make the longest lines
# in a manifest harder to read; `- ` on its own would be a list item.
_INDICATORS_BEFORE_SPACE = "-?:"

# Words a plain scalar must not be, because a reader that resolves types would hand back a bool
# or a nothing rather than the text that was written. Everything in a manifest is text; nothing
# in it is worth the ambiguity.
_RESERVED = {"true", "false", "yes", "no", "on", "off", "null", "none", "~", ""}


def _quote(value: str) -> str:
    """One scalar, written so that it reads back as exactly this string.

    Plain where plain is unambiguous, which is nearly always: a hyperparameter is a number or a
    comma separated list of them, and quoting those would only make the file harder to read.
    Double quoted otherwise, which is the one form that can hold anything.
    """
    opens_badly = value[:1] in _INDICATORS or (
        value[:1] in _INDICATORS_BEFORE_SPACE and value[1:2] in ("", " "))

    plain = (value
             and value == value.strip()
             and not opens_badly
             and value.lower() not in _RESERVED
             and ": " not in value
             and " #" not in value
             and not value.endswith(":")
             and "\n" not in value
             and "\t" not in value
             and "\r" not in value)
    if plain:
        return value

    escaped = (value.replace("\\", "\\\\")
                    .replace('"', '\\"')
                    .replace("\n", "\\n")
                    .replace("\t", "\\t")
                    .replace("\r", "\\r"))
    return f'"{escaped}"'


def _value(value) -> str:
    """One value, as a scalar or as a flow list of them.

    A list is written `[a, b]` and may hold lists of its own, which is what `sizes` is: a model
    that draws well at several aspect ratios lists each as its own `[width, height]`.
    """
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_value(item) for item in value) + "]"
    return _quote(str(value))


def dump_manifest(weights: list, config, suggested=None, tokenizers=None) -> str:
    """A model's manifest, as the text of it.

    `config` is the whole of what the model is, as a ConfigParser or as a dict of dicts; every
    value in it is written as it stands, because what a setting means is the model's to say and
    this only has to hand it back unchanged. `tokenizers` is a name for each way this model turns
    text into ids, and the `tokenizer.json` it is in. `suggested` is what the model advises being
    asked for -- a prompt, the sizes it draws well at, how many steps, how much guidance -- and
    every part of it is optional.
    """
    sections = {name: dict(config[name]) for name in (
        config.sections() if hasattr(config, "sections") else config)}

    lines = ["# What this model is, and which files it is made of.",
             "#",
             "# The safetensors beside this file hold the weights and nothing else; everything",
             "# that says what they are is here.",
             "",
             f"{WEIGHTS_KEY}:"]
    for name in weights:
        lines.append(f"  - {_quote(name)}")

    # A tokenizer is one line, because a tokenizer is one file: everything about how this model's
    # text becomes ids is inside the `tokenizer.json` its authors published.
    if tokenizers:
        lines += ["", f"{TOKENIZERS_KEY}:"]
        for name, file in tokenizers.items():
            lines.append(f"  {_quote(name)}: {_quote(file)}")

    # `model` first: it is the block that says what the rest of them are.
    ordered = ([("model", sections["model"])] if "model" in sections else []) + \
              [(name, body) for name, body in sections.items() if name != "model"]

    lines += ["", f"{CONFIG_KEY}:"]
    for name, body in ordered:
        lines.append(f"  {_quote(name)}:")
        for key, value in body.items():
            lines.append(f"    {_quote(key)}: {_value(value)}")

    # A model with no advice to give has no `suggested` block at all, which is what every model
    # had before there was anywhere to say it. The sizes go on lines of their own, since there
    # are usually several and one to a line is how a person reads them.
    if suggested:
        lines += ["", f"{SUGGESTED_KEY}:"]
        for key, value in suggested.items():
            if key == "sizes" and isinstance(value, (list, tuple)):
                lines.append(f"  {_quote(key)}:")
                lines += [f"    - {_value(size)}" for size in value]
            else:
                lines.append(f"  {_quote(key)}: {_value(value)}")

    return "\n".join(lines) + "\n"


def write_manifest(directory: str, stem: str, config, weights: list, suggested=None,
                   tokenizers=None) -> str:
    """Write `<stem>.yaml` into `directory`, and say what it was called."""
    name = stem + MANIFEST_SUFFIX
    with open(path.join(directory, name), "w", encoding="utf-8") as fp:
        fp.write(dump_manifest(weights, config, suggested, tokenizers))
    print(f"wrote {name}: {len(weights)} weights file(s)")
    return name


def _unquote(text: str) -> str:
    """One scalar as it was written, which is the other half of `_quote`."""
    text = text.strip()
    if len(text) >= 2 and text[0] == text[-1] == "'":
        return text[1:-1].replace("''", "'")
    if len(text) >= 2 and text[0] == text[-1] == '"':
        body = text[1:-1]
        out, escaped = [], False
        for c in body:
            if escaped:
                out.append({"n": "\n", "t": "\t", "r": "\r"}.get(c, c))
                escaped = False
            elif c == "\\":
                escaped = True
            else:
                out.append(c)
        return "".join(out)

    # A `#` begins a comment only when something separates it from what came before.
    for index, c in enumerate(text):
        if c == "#" and (index == 0 or text[index - 1] == " "):
            return text[:index].rstrip()
    return text


def _read_value(text: str):
    """One value as it was written: a string, or a list of them, nested as deeply as it goes."""
    text = text.strip()
    if not text.startswith("["):
        return _unquote(text)

    items, depth, start, quote = [], 0, 1, None
    for index, c in enumerate(text):
        if quote:
            if c == quote:
                quote = None
            continue
        if c in "\"'":
            quote = c
        elif c == "[":
            depth += 1
            if depth == 1:
                start = index + 1
        elif c == "]":
            depth -= 1
            if depth == 0:
                piece = text[start:index].strip()
                if piece:
                    items.append(piece)
                break
        elif c == "," and depth == 1:
            items.append(text[start:index].strip())
            start = index + 1

    return [_read_value(item) for item in items if item]


def load_manifest(text: str) -> tuple:
    """A manifest read back: its weights, its tokenizers, its config, and its suggestions.

    The same subset `dump_manifest` writes and the runtime reads -- block mappings, block and flow
    sequences, and scalars, indented with spaces -- and nothing else in the language.
    """
    rows = []
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        indent = len(line) - len(line.lstrip(" "))
        rows.append((indent, line.strip(), number))

    weights, tokenizers, config, suggested = [], {}, {}, {}
    top, block, listing = None, None, None

    for indent, body, number in rows:
        if indent == 0:
            key = _unquote(body[:-1]) if body.endswith(":") else None
            if key is None:
                raise ValueError(f"line {number}: {body!r} is not one of the manifest's blocks")
            top, block, listing = key, None, None
        elif top == WEIGHTS_KEY:
            if not body.startswith("- "):
                raise ValueError(f"line {number}: {body!r} is not a weights file")
            weights.append(_unquote(body[2:]))
        elif top == CONFIG_KEY and indent == 2:
            block = _unquote(body[:-1]) if body.endswith(":") else None
            if block is None:
                raise ValueError(f"line {number}: {body!r} is not a block of settings")
            config.setdefault(block, {})
        elif body.startswith("- "):
            # An item of the list the key above it opened, which is how `sizes` is written.
            if listing is None:
                raise ValueError(f"line {number}: a list item outside a list")
            listing.append(_read_value(body[2:]))
        else:
            key, _, value = body.partition(":")
            where = {CONFIG_KEY: config.get(block), TOKENIZERS_KEY: tokenizers}.get(
                top, suggested)
            if where is None:
                raise ValueError(f"line {number}: {body!r} is outside any block")
            key = _unquote(key)
            if value.strip():
                where[key] = _read_value(value)
                listing = None
            else:
                listing = where.setdefault(key, [])

    return weights, tokenizers, config, suggested


def read_manifest(manifest: str) -> tuple:
    """`load_manifest` on a file."""
    with open(manifest, encoding="utf-8") as fp:
        return load_manifest(fp.read())


def save_tensors(where: str, tensors: dict) -> None:
    """Write a bag of named tensors as one safetensors file.

    Everything a model is written with goes through `WeightsWriter`. This is for the other thing
    an exporter writes: the reference tensors a test compares against, which are a bag of tensors
    rather than a model and have no manifest of their own.

    safetensors refuses tensors that share storage, because it writes each one's bytes once and
    two views of one buffer would come back as two tensors that are secretly one. A slice or a
    transpose off a model is exactly that, so everything is made contiguous and copied on the way
    in rather than being refused at the end of a long export.
    """
    ready = {name: tensor.detach().cpu().contiguous().clone()
             for name, tensor in tensors.items()}
    save_file(ready, where)


class TensorBag:
    """Collects named tensors and writes them as one safetensors file.

    Not a model -- it has no manifest and nothing builds a model out of it -- but the other thing
    an exporter writes: the reference tensors a test compares the runtime against. The interface
    is `WeightsWriter`'s so that the code writing them does not have to care which it is handing
    tensors to.
    """

    def __init__(self) -> None:
        self.tensors = {}

    def write_tensor(self, ctx, tensor: torch.Tensor, preserve_dtype=False) -> None:
        if tensor.dtype == torch.float32 and not preserve_dtype:
            tensor = tensor.to(torch.float16)
        self.tensors[ctx.name] = tensor

    def save(self, where: str) -> None:
        save_tensors(where, self.tensors)
        print(f"wrote {path.basename(where)}: {len(self.tensors)} reference tensors")


class WeightsWriter:
    """Collects a model's tensors and writes them as safetensors.

    Seven gigabytes is an awkward size to publish and an awkward one to fetch: it cannot be
    downloaded in parallel, and a failed transfer starts over. Given `part_size` this rolls over
    to a new file whenever the one being filled reaches it, so what comes out is
    `<stem>-00001-of-00002.safetensors` and its neighbours.

    A safetensors file is written in one go rather than streamed, so one part's worth of tensors
    is held in memory at a time and let go of as soon as it is written. That is the cost of a
    format whose header has to say where every tensor is before any of them are written, and it is
    why `part_size` bounds memory here as well as file size.
    """

    def __init__(self, output: str, part_size=None) -> None:
        self._part_size = part_size
        self._directory = path.dirname(path.abspath(output))
        self._stem = stem_of(output)

        self._parts = []
        self._holding = {}
        self._written = 0

    def _flush(self) -> None:
        """Write what has been collected as the next part, under a temporary name.

        How many parts there will be is not known until the last tensor has been written, so they
        are named at the end, when the count is finally known.
        """
        if not self._holding:
            return

        temporary = path.join(self._directory, f"{self._stem}-{len(self._parts):05d}.part")
        save_tensors(temporary, self._holding)
        self._parts.append(temporary)
        self._holding = {}
        self._written = 0

    def write_tensor(self, ctx, tensor: torch.Tensor, preserve_dtype=False) -> None:
        """What the exporter writes through. Rolls over first if this part has had enough.

        Everything that came out of the model in float32 is narrowed to float16 on the way in, as
        it always has been, unless the caller says the precision is the point.
        """
        if ctx.quant != Quant.NONE:
            raise NotImplementedError(
                f"{ctx.quant} belonged to a format that is gone, and no model this writes is "
                f"quantized")

        if tensor.dtype == torch.float32 and not preserve_dtype:
            tensor = tensor.to(torch.float16)
        if tensor.dtype not in (torch.float32, torch.float16, torch.int64):
            raise ValueError(f"{ctx.name}: {tensor.dtype} is not a type the runtime reads")

        self._hold(ctx.name, tensor)

    def write_fp8_tensor(self, ctx, tensor: torch.Tensor) -> None:
        """A matrix, quantized to E4M3 and written as the two tensors it is then made of.

        `<name>` holds the elements and `<name>.scale` the float32 scale of each row, which is the
        layout `docs/fp8.md` describes and the only one the runtime reads: safetensors cannot tie
        two tensors together, so the pair is a naming convention, and the reader refuses elements
        with no scales beside them.

        Half the file and half the bytes on the device, for about 2.6e-2 of relative error. Whether
        a model is worth that is the exporter's call, and the package has to say which it made:
        `weight_format: fp8` in the manifest is what makes the runtime build the quantized
        multiply, and a package that writes these tensors without it will not load.
        """
        data, scales = Quantization.quantize_to_fp8(tensor)

        self._hold(ctx.name, data)
        self._hold(ctx.name + FP8_SCALE_SUFFIX, scales)

    def write_fp8_tensor_scale(self, ctx, tensor: torch.Tensor) -> None:
        """A matrix, quantized to E4M3 with one scale for the whole tensor rather than one per row.

        The same two tensors `write_fp8_tensor` writes, under the same names -- `<name>` and
        `<name>.scale` -- so the pairing `docs/fp8.md` describes and `check_fp8_pairs` enforces is
        unchanged. `<name>.scale` is a single `<float>` here rather than one per row, which is what
        tells the runtime to build `fp8_matmul_tensor_scale` rather than `fp8_matmul`:
        `weight_format: fp8_tensor_scale` in the manifest is where that is said, the same way
        `weight_format: fp8` says the row-scaled format.
        """
        data, scale = Quantization.quantize_to_fp8_tensor_scale(tensor)

        self._hold(ctx.name, data)
        self._hold(ctx.name + FP8_SCALE_SUFFIX, scale)

    def _hold(self, name: str, tensor: torch.Tensor) -> None:
        """Keep one tensor for the part being filled, rolling over to the next when it is full."""
        if self._part_size is not None and self._written >= self._part_size:
            self._flush()

        if name in self._holding:
            raise ValueError(f"{name} is written twice")

        print(f"write tensor {name}, shape={tuple(tensor.shape)}, dtype={tensor.dtype}")
        self._holding[name] = tensor.detach().cpu().contiguous().clone()
        self._written += tensor.numel() * tensor.element_size()

    def finish(self, config, suggested=None, tokenizers=None) -> list:
        """Write what is left, name the parts, and write the manifest that describes them.

        `config` is what the model is, and becomes the `config:` of the manifest. `tokenizers` is
        a name and a `tokenizer.json` for each way the model turns text into ids. `suggested` is
        what it advises being asked for, and becomes its `suggested:` when there is any of it; a
        model with no advice to give has no such block at all.
        """
        self._flush()
        if not self._parts:
            raise ValueError("a model with no tensors in it is not a model")

        names = part_names(self._stem, len(self._parts))
        for temporary, name in zip(self._parts, names):
            os.replace(temporary, path.join(self._directory, name))

        write_manifest(self._directory, self._stem, config, names, suggested, tokenizers)
        return names
