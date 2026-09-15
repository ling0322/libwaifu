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

"""Convert a model written as a `.waifupkg` into one written as safetensors and a manifest.

A model used to be one zip of this project's own: the weights in a layout of this project's own,
a `model.ini` saying what they were, sometimes a `metadata.json`, and the tokenizer. None of it
could be read by anything but libwaifu, and the runtime no longer reads any of it. This turns one
of those into what a model is now:

    python tools/convert_model.py models/sdxl-base.waifupkg

writes `models/sdxl-base.safetensors`, `models/sdxl-base.tokenizer.json` and
`models/sdxl-base.yaml` beside it. A model written as several packages is named by its first -- the one that carried the
configuration and the `model_parts` list -- and all of them are read:

    python tools/convert_model.py models/sdxl-base-00001-of-00002.waifupkg

The `.waifupkg` files are not touched, so a conversion that goes wrong costs nothing but the disk
it wrote to. Delete them once the model reads.

A package written before the tokenizer rewrite is refused rather than half converted: its
vocabulary is in a format of this project's own, and what a model carries now is the
`tokenizer.json` its authors published. Those have to be exported again.

This is the only file left that knows either of the old layouts, and it exists to be run once per
model and then deleted along with them.
"""

import argparse
import configparser
import io
import json
import math
import os
import struct
import sys
import zipfile
from os import path

import numpy as np
import torch

sys.path.insert(0, path.dirname(path.abspath(__file__)))
from model_writer import (
    MANIFEST_SUFFIX, Context, WeightsWriter, parse_size, stem_of)

PACKAGE_SUFFIX = ".waifupkg"

MODEL_INI = "model.ini"
METADATA_JSON = "metadata.json"
TOKENIZER_INI = "tokenizer.ini"

# The one key a tokenizer section needs, since the rewrite that made a tokenizer one file.
FILE_KEY = "file"

# What those sections were called, and what the manifest calls them: the block says they are
# tokenizers, so the names no longer have to say it again.
TOKENIZER_NAMES = {"qwen3_tokenizer": "qwen3", "t5_tokenizer": "t5", "tokenizer": "tokenizer"}

# What the old configuration called the list of packages a model was written as, and the key that
# said which layout its tensors were in.
PARTS_KEY = "model_parts"
VERSION_KEY = "package_version"

# The one field a `metadata.json` ever held, and what it is called in a manifest.
OLD_PROMPT = "suggested_prompt"
NEW_PROMPT = "prompt"

# Version 0: one entry holding every tensor, as a `tdicv2` stream.
V0_MAGIC = b"llyn::tdicv2    "
DICT_BEGIN = b"<d> "
DICT_END = b"</d>"
RECORD_BEGIN = b"<r> "
RECORD_END = b"</r>"

# Version 1: one entry per tensor, under this prefix, each with this header.
TENSOR_DIR = "tensors/"
V1_MAGIC = b"waifu::tensor1  "

DATA_MAGIC = 0x55AA

# What one element of each type occupies, and what it is in torch. Mirrors DType in flint.
ELEMENT_SIZE = {1: 4, 2: 8, 3: 1, 4: 2, 6: 1}
TORCH_DTYPE = {
    1: torch.float32,
    2: torch.int64,
    3: torch.uint8,
    4: torch.float16,
    6: torch.int8,
}
NUMPY_DTYPE = {
    1: np.float32,
    2: np.int64,
    3: np.uint8,
    4: np.float16,
    6: np.int8,
}


def tokenizer_file(stem: str, name: str) -> str:
    """What a model's tokenizer is called, after the model and after what the manifest calls it.

    A model with one calls it `tokenizer`, and repeating that in the file name would only give
    `sdxl-base.tokenizer.tokenizer.json`.
    """
    return f"{stem}.tokenizer.json" if name == "tokenizer" \
        else f"{stem}.{name}.tokenizer.json"


def _read(fp, count: int) -> bytes:
    data = fp.read(count)
    if len(data) != count:
        raise ValueError(f"the file ends {count - len(data)} bytes early")
    return data


def _expect(fp, tag: bytes) -> None:
    got = _read(fp, len(tag))
    if got != tag:
        raise ValueError(f"expected {tag!r}, got {got!r}")


def _tensor(dtype: int, shape, data: bytes) -> torch.Tensor:
    """One tensor's bytes, as the tensor they are."""
    array = np.frombuffer(data, dtype=NUMPY_DTYPE[dtype]).copy()
    return torch.from_numpy(array).reshape(tuple(shape) if shape else ())


def read_v0(fp) -> dict:
    """Every tensor of a `tdicv2` stream, by name."""
    _expect(fp, V0_MAGIC)
    _expect(fp, DICT_BEGIN)

    tensors = {}
    while True:
        tag = _read(fp, 4)
        if tag == DICT_END:
            return tensors
        if tag != RECORD_BEGIN:
            raise ValueError(f"expected a record or the end of the stream, got {tag!r}")

        (name_length,) = struct.unpack("<h", _read(fp, 2))
        name = _read(fp, name_length).decode("utf-8")

        _expect(fp, b"tnsr")
        (rank,) = struct.unpack("<h", _read(fp, 2))
        shape = struct.unpack(f"<{rank}i", _read(fp, 4 * rank))

        _expect(fp, b"tdat")
        (slots,) = struct.unpack("<i", _read(fp, 4))
        if slots != 1:
            raise ValueError(f"{name} holds {slots} data slots, expected 1")

        (dtype,) = struct.unpack("<h", _read(fp, 2))
        (numel,) = struct.unpack("<q", _read(fp, 8))
        if dtype not in ELEMENT_SIZE:
            raise ValueError(f"{name} has element type {dtype}, which this tool does not know")
        if numel != (math.prod(shape) if shape else 1):
            raise ValueError(f"{name} holds {numel} elements but its shape calls for others")

        data = _read(fp, numel * ELEMENT_SIZE[dtype])
        (magic,) = struct.unpack("<h", _read(fp, 2))
        if magic != DATA_MAGIC:
            raise ValueError(f"{name} does not end where it should")
        _expect(fp, RECORD_END)

        tensors[name] = _tensor(dtype, shape, data)


def read_v1_entry(name: str, entry: bytes) -> torch.Tensor:
    """One tensor, as version 1 wrote it: its own entry, header and all."""
    if not entry.startswith(V1_MAGIC):
        raise ValueError(f"{name} does not start with {V1_MAGIC!r}")

    at = len(V1_MAGIC)
    (dtype,) = struct.unpack_from("<h", entry, at)
    (rank,) = struct.unpack_from("<h", entry, at + 2)
    shape = struct.unpack_from(f"<{rank}i", entry, at + 4)
    at += 4 + 4 * rank

    if dtype not in ELEMENT_SIZE:
        raise ValueError(f"{name} has element type {dtype}, which this tool does not know")

    numel = math.prod(shape) if shape else 1
    length = numel * ELEMENT_SIZE[dtype]
    data = entry[at:at + length]
    (magic,) = struct.unpack_from("<h", entry, at + length)
    if magic != DATA_MAGIC:
        raise ValueError(f"{name} does not end where it should")

    return _tensor(dtype, shape, data)


def read_tensors(packages: list, model_file: str | None) -> dict:
    """Every tensor of a model, out of whichever layout its packages are in.

    Read into one namespace, in the order the packages were named, which is what the runtime did
    with them too.
    """
    tensors = {}
    for source in packages:
        if model_file is not None:
            with source.open(model_file) as fp:
                found = read_v0(fp)
        else:
            found = {}
            for entry in source.namelist():
                if not entry.startswith(TENSOR_DIR):
                    continue
                name = entry[len(TENSOR_DIR):]
                found[name] = read_v1_entry(name, source.read(entry))

        for name, tensor in found.items():
            if name in tensors:
                raise ValueError(f"{name} is in more than one package of this model")
            tensors[name] = tensor

    return tensors


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("package", help="the .waifupkg to convert, or the first of several.")
    parser.add_argument("-output-dir", dest="output_dir", default=None,
                        help="where to write the model. Defaults to beside the package.")
    parser.add_argument("-part-size", dest="part_size", type=parse_size, default=None,
                        help="split the weights into files no larger than this, as 4GB. Left out, "
                             "they are written as one file however large it is.")
    parser.add_argument("-force", action="store_true",
                        help="overwrite a model that is already there.")
    args = parser.parse_args(argv)

    first = zipfile.ZipFile(args.package)
    if MODEL_INI not in first.namelist():
        print(f"{args.package} has no {MODEL_INI}, so it is not a model package", file=sys.stderr)
        return 1

    config = configparser.ConfigParser()
    config.read_string(first.read(MODEL_INI).decode("utf-8"))

    # Which files the model is in, and which layout their tensors are in. A package that did not
    # say listed only itself, and one that did not say a version was version 0.
    listed = config["model"].pop(PARTS_KEY, "")
    names = [name.strip() for name in listed.split(",") if name.strip()]
    if not names:
        names = [path.basename(args.package)]

    version = int(config["model"].pop(VERSION_KEY, 0))
    if version not in (0, 1):
        print(f"{args.package} is version {version} of a format this does not know", file=sys.stderr)
        return 1
    model_file = config["model"].pop("model_file", None) if version == 0 else None

    beside = path.dirname(path.abspath(args.package))
    directory = args.output_dir or beside
    os.makedirs(directory, exist_ok=True)

    # `sdxl-base-00001-of-00002` names one part; the model those parts make up is `sdxl-base`.
    stem = path.basename(names[0])
    stem = stem[: -len(PACKAGE_SUFFIX)] if stem.endswith(PACKAGE_SUFFIX) else stem
    pieces = stem.rsplit("-", 3)
    if len(pieces) == 4 and pieces[2] == "of" and pieces[1].isdigit() and pieces[3].isdigit():
        stem = pieces[0]

    manifest = path.join(directory, stem + MANIFEST_SUFFIX)
    if path.exists(manifest) and not args.force:
        print(f"{manifest} already exists; pass -force to overwrite", file=sys.stderr)
        return 1

    packages = []
    for name in names:
        where = path.join(beside, name)
        if not path.isfile(where):
            print(f"{name} is named as one of the packages and is not there", file=sys.stderr)
            return 1
        packages.append(first if path.basename(where) == path.basename(args.package)
                        else zipfile.ZipFile(where))

    # The tokenizers, which were entries of the first package and are files beside the manifest
    # now. A package written before the tokenizer rewrite holds a vocabulary in a format of this
    # project's own, and there is nothing to convert that into: what a model carries now is the
    # `tokenizer.json` its authors published, and only the exporter can produce one.
    tokenizers, missing = {}, []
    if TOKENIZER_INI in first.namelist():
        sections = configparser.ConfigParser()
        sections.read_string(first.read(TOKENIZER_INI).decode("utf-8"))

        for section in sections.sections():
            name = TOKENIZER_NAMES.get(section, section)
            entry = sections[section].get(FILE_KEY)
            if entry is None:
                # A vocabulary in the format this project used to write, which nothing reads any
                # more and which cannot be turned into a `tokenizer.json`: that file is the one
                # the model's authors published, and only the exporter can go and get it.
                missing.append(name)
                continue

            written = tokenizer_file(stem, name)
            with open(path.join(directory, written), "wb") as fp:
                fp.write(first.read(entry))
            print(f"wrote {written}")

            tokenizers[name] = written

    suggested = {}
    if METADATA_JSON in first.namelist():
        said = json.loads(first.read(METADATA_JSON).decode("utf-8"))
        prompt = said.get(OLD_PROMPT)
        if isinstance(prompt, str) and prompt.strip():
            suggested[NEW_PROMPT] = prompt.strip()

    print(f"reading {len(packages)} package(s) in version {version} of the old format")
    tensors = read_tensors(packages, model_file)
    print(f"{len(tensors)} tensors")

    writer = WeightsWriter(path.join(directory, stem), args.part_size)
    for name in list(tensors):
        # Taken out as it is handed over, so the two copies of a seven gigabyte model do not both
        # exist: what the writer is holding, and what has not been handed to it yet.
        #
        # Already the precision the model was exported at; narrowing again would be a second loss.
        writer.write_tensor(Context(name), tensors.pop(name), preserve_dtype=True)

    for name in writer.finish(config, suggested, tokenizers):
        print(f"wrote {name}")

    if missing:
        # Said at the end, where it is read, and said loudly: the weights are whole and the model
        # is one small file short of loading.
        print()
        print(f"the weights are converted, and {len(missing)} tokenizer(s) are not: "
              f"{', '.join(missing)}.")
        print("This package predates the rewrite that made a tokenizer the `tokenizer.json` its")
        print("authors published, and there is nothing here to convert into one. Write them with")
        print("`tools/tokenizer_exporter.py`'s `read_tokenizer`, put each beside the manifest, and")
        print(f"name it there:\n")
        print(f"  tokenizers:")
        for name in missing:
            print(f"    {name}: {tokenizer_file(stem, name)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
