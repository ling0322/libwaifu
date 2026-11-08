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

"""Qwen3-ASR as one package: the release in, a manifest, its weights and its tokenizer out.

    .venv-qwen3-asr/bin/python tools/qwen3_asr_exporter.py -output models/qwen3-asr.safetensors

Reads `Qwen/Qwen3-ASR-0.6B` (from the HuggingFace cache, `-repo` for another size, or
`-checkpoint`) and writes what `waifu::qwen3_asr::Qwen3Asr::from_manifest` reads:

| namespace | from | what it is |
| --- | --- | --- |
| `qwen3_asr.audio_tower` | `thinker.audio_tower` | the audio encoder: three convolutions, 18 layers |
| `qwen3_asr.model` | `thinker.model` | Qwen3-0.6B's decoder |
| `qwen3_asr.lm_head` | `thinker.lm_head` | its head; left out where it is the embedding table |

and the Qwen vocabulary, with the audio markers and `<asr_text>`, as a `tokenizer.json`.

**Float16, from bfloat16.** The release is bfloat16 and the runtime has no bfloat16, so every
tensor is narrowed to half. Every bfloat16 value inside half's range keeps all of its bits there
(seven of mantissa against ten), so what is lost is only what falls below half's smallest normal
number; a tensor with anything past half's largest is refused rather than written as infinity.

**The architecture is in the manifest**, under `qwen3_asr_audio` and `qwen3_asr_text`, so that
the runtime reads its widths rather than knowing them -- the 1.7B release is the same network,
wider and deeper.
"""

import argparse
import json
import os
import sys
from os import path

import torch

sys.path.insert(0, path.dirname(path.abspath(__file__)))
from model_exporter import Context, open_weights, parse_size, stem_of  # noqa: E402

MODEL_TYPE = "qwen3_asr"
REPO = "Qwen/Qwen3-ASR-0.6B"

HALF_MAX = 65504.0


def checkpoint_dir(repo, given):
    if given:
        return given
    from huggingface_hub import snapshot_download

    return snapshot_download(repo)


def read_tensors(directory):
    from safetensors.torch import load_file

    tensors = {}
    names = sorted(name for name in os.listdir(directory) if name.endswith(".safetensors"))
    for name in names:
        for key, tensor in load_file(path.join(directory, name)).items():
            tensors[key.removeprefix("thinker.")] = tensor
    return tensors


def write_tokenizer(directory, output, config):
    """The release's vocabulary and added tokens, as the `tokenizer.json` the runtime reads."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(directory)
    for token, wanted in [("<|audio_start|>", config["audio_start_token_id"]),
                          ("<|audio_end|>", config["audio_end_token_id"]),
                          ("<|audio_pad|>", config["audio_token_id"])]:
        found = tokenizer.convert_tokens_to_ids(token)
        if found != wanted:
            raise SystemExit(f"{token} is {found}, and the model was trained with it at {wanted}")
    tokenizer.backend_tokenizer.save(output)

    # The file is what the runtime reads, so the file is what is checked: the same ids as the
    # tokenizer upstream's processor wraps, and the same text back from them.
    from tokenizers import Tokenizer

    written = Tokenizer.from_file(output)
    for probe in ["<|im_start|>system\n<|im_end|>\n<|im_start|>user\n<|audio_start|>"
                  "<|audio_pad|><|audio_pad|><|audio_end|><|im_end|>\n<|im_start|>assistant\n",
                  "language Chinese<asr_text>希望你以后能够做得比我还好哟。",
                  "language English<asr_text>The ticket costs twenty-five dollars, at 7:30."]:
        want = tokenizer(probe, add_special_tokens=False)["input_ids"]
        got = written.encode(probe, add_special_tokens=False).ids
        if want != got:
            raise SystemExit(f"the written tokenizer reads {probe!r} as {got}, not {want}")
        if written.decode(got, skip_special_tokens=True) != tokenizer.decode(
                want, skip_special_tokens=True):
            raise SystemExit(f"the written tokenizer decodes {probe!r} differently")

    return len(tokenizer)


def config_of(release):
    thinker = release["thinker_config"]
    audio = thinker["audio_config"]
    text = thinker["text_config"]

    mrope = (text.get("rope_scaling") or {}).get("mrope_section")
    return {
        "model": {"type": MODEL_TYPE},
        MODEL_TYPE: {
            "sample_rate": 16000,
            "audio_token_id": thinker["audio_token_id"],
            "audio_start_token_id": thinker["audio_start_token_id"],
            "audio_end_token_id": thinker["audio_end_token_id"],
            # `generate`'s: <|im_end|> and <|endoftext|>.
            "end_token_ids": "151645,151643",
            "languages": ",".join(release.get("support_languages", [])),
            # Upstream's MAX_ASR_INPUT_SECONDS: longer recordings are cut at a quiet point.
            "longest_chunk_seconds": 1200,
            "max_new_tokens": 4096,
        },
        "qwen3_asr_audio": {
            "mel_bins": audio["num_mel_bins"],
            "d_model": audio["d_model"],
            "layers": audio["encoder_layers"],
            "heads": audio["encoder_attention_heads"],
            "ffn_dim": audio["encoder_ffn_dim"],
            "downsample_hidden_size": audio["downsample_hidden_size"],
            "output_dim": audio["output_dim"],
            "n_window": audio["n_window"],
            "n_window_infer": audio["n_window_infer"],
            "max_source_positions": audio["max_source_positions"],
        },
        "qwen3_asr_text": {
            "hidden_size": text["hidden_size"],
            "layers": text["num_hidden_layers"],
            "heads": text["num_attention_heads"],
            "kv_heads": text["num_key_value_heads"],
            "head_dim": text["head_dim"],
            "mlp_size": text["intermediate_size"],
            "vocab_size": text["vocab_size"],
            "rope_theta": text["rope_theta"],
            "norm_eps": text["rms_norm_eps"],
            # Kept for whoever reads the package: with every axis at the same position, as in a
            # sequence of audio and text, the interleaved sections rotate as plain RoPE does.
            "mrope_section": ",".join(str(x) for x in mrope) if mrope else "",
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-output", required=True, help="the package, e.g. models/qwen3-asr.safetensors")
    parser.add_argument("-repo", default=REPO, help="the release on HuggingFace")
    parser.add_argument("-checkpoint", help="a local copy of the release; the HF cache otherwise")
    parser.add_argument("-part_size", default="2GB", help="how large one file of weights may be")
    arguments = parser.parse_args()

    source = checkpoint_dir(arguments.repo, arguments.checkpoint)
    with open(path.join(source, "config.json")) as f:
        release = json.load(f)
    directory = path.dirname(path.abspath(arguments.output))
    stem = stem_of(arguments.output)
    os.makedirs(directory, exist_ok=True)

    tokenizer_file = stem + ".tokenizer.json"
    vocabulary = write_tokenizer(source, path.join(directory, tokenizer_file),
                                 release["thinker_config"])
    print(f"wrote {tokenizer_file}: {vocabulary} tokens")

    tensors = read_tensors(source)
    unknown = [key for key in tensors
               if not key.startswith(("audio_tower.", "model.", "lm_head."))]
    if unknown:
        raise SystemExit(f"the release has tensors this does not know: {unknown[:5]}")

    # A tied head is the embedding table read again; the runtime loads it under that name.
    tied = torch.equal(tensors["lm_head.weight"], tensors["model.embed_tokens.weight"])
    if tied:
        del tensors["lm_head.weight"]
    print(f"lm_head {'is' if tied else 'is not'} the embedding table")

    writer = open_weights(arguments.output, parse_size(arguments.part_size))
    total = 0
    for key in sorted(tensors):
        tensor = tensors[key].detach()
        largest = float(tensor.float().abs().max())
        if largest > HALF_MAX:
            raise SystemExit(f"{key} holds {largest}, which is past half's largest")
        writer.write_tensor(Context(f"{MODEL_TYPE}.{key}"), tensor.to(torch.float16))
        total += tensor.numel()

    config = config_of(release)
    config[MODEL_TYPE]["tied_head"] = "true" if tied else "false"
    for name in writer.finish(config, None, {"tokenizer": tokenizer_file}):
        print(f"wrote {name}")
    print(f"{total / 1e6:.1f} M parameters")


if __name__ == "__main__":
    main()
