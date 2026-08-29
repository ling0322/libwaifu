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

"""Split a model package into several, so that no one file is unreasonably large.

A seven gigabyte file is awkward to publish and awkward to fetch: it cannot be downloaded in
parallel, a failed transfer starts over, and some places will not take it at all. This rewrites
one package as several, each a package in its own right.

The split is between tensors, never inside one. Each part carries a whole parameter file -- its
own header, its own records, its own end -- so any part can be read and checked on its own. The
first part also carries everything that is not parameters (the configuration, the tokenizer) and
names the others, which is how the runtime finds them.

    python tools/split_package.py models/sdxl-base.waifupkg

writes `sdxl-base-00001-of-00002.waifupkg` and `sdxl-base-00002-of-00002.waifupkg` beside it.
Nothing needs re-exporting; the tensors are copied byte for byte.
"""

import argparse
import configparser
import io
import math
import os
import struct
import sys
import zipfile
from os import path

sys.path.insert(0, path.dirname(path.abspath(__file__)))
from model_exporter import (
    MODEL_INI, PACKAGE_SUFFIX, PARTS_KEY, parse_size, part_names)

# The parameter file's own framing, which this walks without interpreting.
MAGIC = b"llyn::tdicv2    "
DICT_BEGIN = b"<d> "
DICT_END = b"</d>"
RECORD_BEGIN = b"<r> "
RECORD_END = b"</r>"
DATA_MAGIC = 0x55AA

# What one element of each type occupies, which is all that is needed to find where a record
# ends. Mirrors DType::getTotalSize in flint/dtype.cc.
ELEMENT_SIZE = {1: 4, 2: 8, 3: 1, 4: 2, 6: 1, 7: 1, 8: 1, 9: 4}

DEFAULT_PART_SIZE = 4 * 1000**3


class Record:
    """One tensor in the parameter file, as a name and a run of bytes."""

    def __init__(self, name: str, offset: int, length: int) -> None:
        self.name = name
        self.offset = offset
        self.length = length


def _read(fp, count: int) -> bytes:
    data = fp.read(count)
    if len(data) != count:
        raise ValueError(f"the parameter file ends in the middle of a record")
    return data


def _expect(fp, tag: bytes) -> None:
    got = _read(fp, len(tag))
    if got != tag:
        raise ValueError(f"expected {tag!r} at {fp.tell() - len(tag)}, got {got!r}")


def index_records(fp) -> list:
    """Walk the parameter file and say where each record begins and how long it is.

    Only the headers are read; the data of each tensor is skipped rather than loaded, so this
    costs a seek per tensor rather than the size of the model.
    """
    _expect(fp, MAGIC)
    _expect(fp, DICT_BEGIN)

    records = []
    while True:
        begin = fp.tell()
        tag = _read(fp, 4)
        if tag == DICT_END:
            return records
        if tag != RECORD_BEGIN:
            raise ValueError(f"expected a record or the end of the file at {begin}, got {tag!r}")

        (name_length,) = struct.unpack("<h", _read(fp, 2))
        if name_length <= 0:
            raise ValueError(f"the record at {begin} has an empty name")
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
        expected = math.prod(shape) if shape else 1
        if numel != expected:
            raise ValueError(f"{name} holds {numel} elements but its shape calls for {expected}")

        fp.seek(numel * ELEMENT_SIZE[dtype], io.SEEK_CUR)
        (magic,) = struct.unpack("<h", _read(fp, 2))
        if magic != DATA_MAGIC:
            raise ValueError(f"{name} does not end where it should")
        _expect(fp, RECORD_END)

        records.append(Record(name, begin, fp.tell() - begin))


def assign(records: list, part_size: int) -> list:
    """Which records go in which part.

    The number of parts is what the limit calls for, and the records are then spread evenly over
    that many rather than filling each to the brim and leaving a small last one: two parts of
    three and a half gigabytes download better than one of five and one of two.
    """
    total = sum(record.length for record in records)
    count = max(1, math.ceil(total / part_size))
    target = math.ceil(total / count)

    parts = [[]]
    used = 0
    for record in records:
        # Never split a tensor, and never leave a part empty.
        if used + record.length > target and parts[-1] and len(parts) < count:
            parts.append([])
            used = 0
        parts[-1].append(record)
        used += record.length

    return parts


def copy_range(source, destination, offset: int, length: int) -> None:
    source.seek(offset)
    remaining = length
    while remaining:
        chunk = source.read(min(remaining, 1 << 22))
        if not chunk:
            raise ValueError("the parameter file ended early")
        destination.write(chunk)
        remaining -= len(chunk)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("package", help="the .waifupkg to split.")
    parser.add_argument("-part-size", dest="part_size", type=parse_size,
                        default=DEFAULT_PART_SIZE,
                        help="the largest a part may be, as 4GB, 512MB or a plain number of "
                             "bytes. Default 4GB.")
    parser.add_argument("-output-dir", dest="output_dir", default=None,
                        help="where to write the parts. Defaults to beside the input.")
    parser.add_argument("-force", action="store_true", help="overwrite parts that already exist.")
    args = parser.parse_args(argv)

    source = zipfile.ZipFile(args.package)
    entries = source.namelist()
    if MODEL_INI not in entries:
        print(f"{args.package} has no {MODEL_INI}, so it is not a model package", file=sys.stderr)
        return 1

    config = configparser.ConfigParser()
    config.read_string(source.read(MODEL_INI).decode("utf-8"))
    if PARTS_KEY in config["model"]:
        print(f"{args.package} is already split", file=sys.stderr)
        return 1
    model_file = config["model"]["model_file"]

    with source.open(model_file) as fp:
        if not fp.seekable():
            print(f"{model_file} is compressed, and a package should be stored", file=sys.stderr)
            return 1
        records = index_records(fp)

    parts = assign(records, args.part_size)
    stem = path.basename(args.package)
    stem = stem[: -len(PACKAGE_SUFFIX)] if stem.endswith(PACKAGE_SUFFIX) else stem
    directory = args.output_dir or path.dirname(path.abspath(args.package))
    names = part_names(stem, len(parts))

    for name in names:
        if path.exists(path.join(directory, name)) and not args.force:
            print(f"{name} already exists; pass -force to overwrite", file=sys.stderr)
            return 1

    # The first part carries everything that is not parameters, and names the others.
    config["model"][PARTS_KEY] = ",".join(names)
    extras = io.StringIO()
    config.write(extras)

    os.makedirs(directory, exist_ok=True)
    for index, (name, records_here) in enumerate(zip(names, parts)):
        written = 0
        with zipfile.ZipFile(path.join(directory, name), "w",
                             compression=zipfile.ZIP_STORED) as part:
            with part.open(model_file, "w", force_zip64=True) as out:
                out.write(MAGIC)
                out.write(DICT_BEGIN)
                with source.open(model_file) as fp:
                    for record in records_here:
                        copy_range(fp, out, record.offset, record.length)
                        written += record.length
                out.write(DICT_END)

            if index == 0:
                with part.open(MODEL_INI, "w", force_zip64=True) as out:
                    out.write(extras.getvalue().encode("utf-8"))
                for entry in entries:
                    if entry in (MODEL_INI, model_file):
                        continue
                    with part.open(entry, "w", force_zip64=True) as out:
                        out.write(source.read(entry))

        print(f"wrote {name}: {len(records_here)} tensors, {written / 1000**3:.2f} GB")

    return 0


if __name__ == "__main__":
    sys.exit(main())
