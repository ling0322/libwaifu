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

"""Rewrite a model's weights as several files, so that no one of them is unreasonably large.

Seven gigabytes is an awkward size to publish and an awkward one to fetch: it cannot be downloaded
in parallel, a failed transfer starts over, and some places will not take it at all. The exporter
takes a `-part-size` and writes the weights split to begin with; this does the same to a model
that is already written, without exporting it again.

    python tools/split_model.py models/sdxl-base.yaml -part-size 4GB

reads the weights the manifest names, writes `sdxl-base-00001-of-00002.safetensors` and its
neighbours, and rewrites the manifest to name them. The tokenizers and everything the manifest
says are carried across untouched -- what changes is which file each tensor is in, and a model
does not know that about itself.

The manifest is what this is given, rather than a weights file, because the manifest is the model:
it is what says which files there are and what has to be rewritten when that changes.
"""

import argparse
import os
import sys
from os import path

sys.path.insert(0, path.dirname(path.abspath(__file__)))
from model_exporter import (
    MANIFEST_SUFFIX, Context, WeightsWriter, parse_size, read_manifest)

from safetensors.torch import load_file

DEFAULT_PART_SIZE = 4 * 1000**3


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("manifest", help=f"the {MANIFEST_SUFFIX} of the model to split.")
    parser.add_argument("-part-size", dest="part_size", type=parse_size,
                        default=DEFAULT_PART_SIZE,
                        help="the largest a file may be, as 4GB, 512MB or a plain number of "
                             "bytes. Default 4GB.")
    parser.add_argument("-output-dir", dest="output_dir", default=None,
                        help="where to write the model. Defaults to beside the manifest.")
    args = parser.parse_args(argv)

    if not args.manifest.endswith(MANIFEST_SUFFIX):
        print(f"{args.manifest} is not a {MANIFEST_SUFFIX}; this takes the model's manifest "
              f"rather than one of its weights files", file=sys.stderr)
        return 1

    weights, tokenizers, config, suggested = read_manifest(args.manifest)

    beside = path.dirname(path.abspath(args.manifest))
    directory = args.output_dir or beside
    os.makedirs(directory, exist_ok=True)

    stem = path.basename(args.manifest)[: -len(MANIFEST_SUFFIX)]

    # Read in the order the manifest names them, which is the order they read into one namespace.
    tensors = {}
    for name in weights:
        where = path.join(beside, name)
        if not path.isfile(where):
            print(f"{name} is named by the manifest and is not there", file=sys.stderr)
            return 1

        print(f"reading {name}")
        for tensor_name, tensor in load_file(where).items():
            if tensor_name in tensors:
                print(f"{tensor_name} is in more than one file of this model", file=sys.stderr)
                return 1
            tensors[tensor_name] = tensor

    print(f"{len(tensors)} tensors")

    writer = WeightsWriter(path.join(directory, stem), args.part_size)
    for name in list(tensors):
        # Taken out as it is handed over, so the model is not held twice: what the writer has, and
        # what has not been handed to it yet.
        #
        # Already the precision it was exported at; narrowing again would be a second loss.
        writer.write_tensor(Context(name), tensors.pop(name), preserve_dtype=True)

    written = writer.finish(config, suggested, tokenizers)
    for name in written:
        print(f"wrote {name}")

    # The files that are no longer named, which are the ones the split replaced. Left on disk: a
    # run that ends here has written a whole model, and deleting what it read is the one step that
    # cannot be undone if any of it was wrong.
    stale = [name for name in weights if name not in written]
    if stale and directory == beside:
        print(f"{', '.join(stale)} {'is' if len(stale) == 1 else 'are'} no longer named by "
              f"{path.basename(args.manifest)}, and can be deleted once the model reads")
    return 0


if __name__ == "__main__":
    sys.exit(main())
