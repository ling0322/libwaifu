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

"""Fun-CosyVoice3-0.5B-2512 as one package: the release in, a manifest and its weights out.

    PYTHONPATH=<S3Tokenizer checkout> .venv/bin/python tools/cosyvoice3_exporter.py \\
        -output models/cosyvoice3.safetensors

Reads `FunAudioLLM/Fun-CosyVoice3-0.5B-2512` (from the HuggingFace cache, or `-checkpoint`) and
writes the five models `waifu::cosyvoice3::CosyVoice3::from_manifest` runs, each under a namespace
of its own:

| namespace | from | what it is |
| --- | --- | --- |
| `cosyvoice3.lm` | `llm.pt` | Qwen2-0.5B with a speech-token embedding and head |
| `cosyvoice3.flow` | `flow.pt` | token embedding, look-ahead convolution and the DiT |
| `cosyvoice3.hift` | `hift.pt` | the HiFT vocoder, weight norm folded away |
| `cosyvoice3.speech_tokenizer` | `speech_tokenizer_v3.onnx` | S3Tokenizer v3, for the prompt |
| `cosyvoice3.campplus` | `funasr/campplus` | the speaker encoder |

and the Qwen vocabulary, with the special tokens `CosyVoice3Tokenizer` adds, as a `tokenizer.json`.

# Where the weights are not upstream's files

**The speech tokenizer ships only as ONNX.** Its weights are read out of the graph by
S3Tokenizer's own `onnx2torch_v3`, the conversion its PyTorch port loads from. That port gives
the same 87 tokens onnxruntime does on upstream's `zero_shot_prompt.wav`, which is what makes
reading it this way safe -- and why S3Tokenizer has to be importable to run this.

**CAMPPlus is not read from `campplus.onnx`.** That graph and `funasr/campplus`'s
`campplus_cn_common.bin` are the same network -- their embeddings of one input agree to 4e-6 -- and
the second is what `tools/campplus_exporter.py` already exports for IndexTTS-2.5, into the layout
`waifu::indextts::campplus` reads. So it is exported by that tool's own functions.

**The flow's starting noise is a weight.** `CausalConditionalCFM` seeds torch with zero and draws
`randn(1, 80, 15000)` once, at construction, and every reading starts from a slice of it. It is
written here as `cosyvoice3.flow.rand_noise`, so that the runtime starts from the same point
upstream does rather than from its own draw of the same distribution.

# What is left out

`llm.model.lm_head` -- the Qwen text head, which a model that emits speech tokens never uses --
and HiFT's random buffers, which are not in `hift.pt` either (see `docs/cosyvoice3.md`).

# Nothing is narrowed

Every tensor is written at float32, which is the precision upstream runs every one of these
models at unless it is asked for `fp16`.
"""

import argparse
import os
import sys
from os import path

import torch

from model_exporter import Context, open_weights, parse_size, stem_of

MODEL_TYPE = "cosyvoice3"
REPO = "FunAudioLLM/Fun-CosyVoice3-0.5B-2512"

# `CosyVoice3Tokenizer.__init__`'s list, as written there, and in its order: the order is what
# numbers them. `<|endofprompt|>` has to come out as 151646, which `Qwen2LM.inference` asserts.
SPECIAL_TOKENS = [
    '<|im_start|>', '<|im_end|>', '<|endofprompt|>',
    '[breath]', '<strong>', '</strong>', '[noise]',
    '[laughter]', '[cough]', '[clucking]', '[accent]',
    '[quick_breath]',
    "<laughter>", "</laughter>",
    "[hissing]", "[sigh]", "[vocalized-noise]",
    "[lipsmack]", "[mn]", "<|endofsystem|>",
    "[AA]", "[AA0]", "[AA1]", "[AA2]", "[AE]", "[AE0]", "[AE1]", "[AE2]", "[AH]", "[AH0]", "[AH1]", "[AH2]",
    "[AO]", "[AO0]", "[AO1]", "[AO2]", "[AW]", "[AW0]", "[AW1]", "[AW2]", "[AY]", "[AY0]", "[AY1]", "[AY2]",
    "[B]", "[CH]", "[D]", "[DH]", "[EH]", "[EH0]", "[EH1]", "[EH2]", "[ER]", "[ER0]", "[ER1]", "[ER2]", "[EY]",
    "[EY0]", "[EY1]", "[EY2]", "[F]", "[G]", "[HH]", "[IH]", "[IH0]", "[IH1]", "[IH2]", "[IY]", "[IY0]", "[IY1]",
    "[IY2]", "[JH]", "[K]", "[L]", "[M]", "[N]", "[NG]", "[OW]", "[OW0]", "[OW1]", "[OW2]", "[OY]", "[OY0]",
    "[OY1]", "[OY2]", "[P]", "[R]", "[S]", "[SH]", "[T]", "[TH]", "[UH]", "[UH0]", "[UH1]", "[UH2]", "[UW]",
    "[UW0]", "[UW1]", "[UW2]", "[V]", "[W]", "[Y]", "[Z]", "[ZH]",
    "[a]", "[ai]", "[an]", "[ang]", "[ao]", "[b]", "[c]", "[ch]", "[d]", "[e]", "[ei]", "[en]", "[eng]", "[f]",
    "[g]", "[h]", "[i]", "[ian]", "[in]", "[ing]", "[iu]", "[ià]", "[iàn]", "[iàng]", "[iào]", "[iá]", "[ián]",
    "[iáng]", "[iáo]", "[iè]", "[ié]", "[iòng]", "[ióng]", "[iù]", "[iú]", "[iā]", "[iān]", "[iāng]", "[iāo]",
    "[iē]", "[iě]", "[iōng]", "[iū]", "[iǎ]", "[iǎn]", "[iǎng]", "[iǎo]", "[iǒng]", "[iǔ]", "[j]", "[k]", "[l]",
    "[m]", "[n]", "[o]", "[ong]", "[ou]", "[p]", "[q]", "[r]", "[s]", "[sh]", "[t]", "[u]", "[uang]", "[ue]",
    "[un]", "[uo]", "[uà]", "[uài]", "[uàn]", "[uàng]", "[uá]", "[uái]", "[uán]", "[uáng]", "[uè]", "[ué]", "[uì]",
    "[uí]", "[uò]", "[uó]", "[uā]", "[uāi]", "[uān]", "[uāng]", "[uē]", "[uě]", "[uī]", "[uō]", "[uǎ]", "[uǎi]",
    "[uǎn]", "[uǎng]", "[uǐ]", "[uǒ]", "[vè]", "[w]", "[x]", "[y]", "[z]", "[zh]", "[à]", "[ài]", "[àn]", "[àng]",
    "[ào]", "[á]", "[ái]", "[án]", "[áng]", "[áo]", "[è]", "[èi]", "[èn]", "[èng]", "[èr]", "[é]", "[éi]", "[én]",
    "[éng]", "[ér]", "[ì]", "[ìn]", "[ìng]", "[í]", "[ín]", "[íng]", "[ò]", "[òng]", "[òu]", "[ó]", "[óng]", "[óu]",
    "[ù]", "[ùn]", "[ú]", "[ún]", "[ā]", "[āi]", "[ān]", "[āng]", "[āo]", "[ē]", "[ēi]", "[ēn]", "[ēng]", "[ě]",
    "[ěi]", "[ěn]", "[ěng]", "[ěr]", "[ī]", "[īn]", "[īng]", "[ō]", "[ōng]", "[ōu]", "[ū]", "[ūn]", "[ǎ]", "[ǎi]",
    "[ǎn]", "[ǎng]", "[ǎo]", "[ǐ]", "[ǐn]", "[ǐng]", "[ǒ]", "[ǒng]", "[ǒu]", "[ǔ]", "[ǔn]", "[ǘ]", "[ǚ]", "[ǜ]"
]

END_OF_PROMPT = 151646

# What a reading is steered by, from `cosyvoice3.yaml` and the code that reads it. The Rust side
# has the same numbers as its defaults; these are here so a package says what it was exported as.
CONFIG = {
    "sample_rate": 24000,
    "speech_token_size": 6561,
    "token_mel_ratio": 2,
    "top_p": 0.8,
    "top_k": 25,
    "win_size": 10,
    "tau_r": 0.1,
    "min_token_text_ratio": 2,
    "max_token_text_ratio": 20,
    "diffusion_steps": 10,
    "cfg_rate": 0.7,
    "system_prompt": "You are a helpful assistant.<|endofprompt|>",
}

# How many tensors each namespace has to come out with; a count that moves is an export that
# changed shape, and cheaper to notice here than as a missing parameter at load.
COUNTS = {"lm": 292, "flow": 331, "hift": 246, "speech_tokenizer": 198, "campplus": 573}


def checkpoint_dir(given):
    if given:
        return given
    from huggingface_hub import snapshot_download

    return snapshot_download(REPO, allow_patterns=[
        "*.yaml", "llm.pt", "flow.pt", "hift.pt", "speech_tokenizer_v3.onnx",
        "CosyVoice-BlankEN/*"])


def lm_tensors(directory):
    """`llm.pt` as stored, less the text head. The Qwen names lose their `llm.model.model.`."""
    state = torch.load(path.join(directory, "llm.pt"), map_location="cpu", weights_only=True)
    out = {}
    for key, tensor in state.items():
        if key == "llm.model.lm_head.weight":
            continue
        out[key.removeprefix("llm.model.model.")] = tensor
    return out


def flow_tensors(directory):
    """`flow.pt` as stored, and the fixed noise the flow's solver starts from."""
    state = torch.load(path.join(directory, "flow.pt"), map_location="cpu", weights_only=True)
    out = dict(state)

    # `set_all_random_seed(0)` then `torch.randn([1, 80, 50 * 300])`, as the constructor does it.
    # The seed is reset immediately before the draw, so nothing constructed earlier moves it.
    generator = torch.Generator().manual_seed(0)
    out["rand_noise"] = torch.randn([1, 80, 50 * 300], generator=generator)
    return out


def fold_weight_norm(state):
    """`parametrizations.weight.original0/1` -- torch's `weight_norm` over dimension 0 -- as the
    one weight they stand for: `g * v / |v|`, the norm taken over everything but the first axis."""
    out = {}
    for key, tensor in state.items():
        if key.endswith(".parametrizations.weight.original1"):
            stem = key.removesuffix(".parametrizations.weight.original1")
            g = state[stem + ".parametrizations.weight.original0"]
            norm = tensor.flatten(1).norm(dim=1).reshape((-1,) + (1,) * (tensor.dim() - 1))
            out[stem + ".weight"] = (g * tensor / norm).contiguous()
        elif key.endswith(".parametrizations.weight.original0"):
            continue
        else:
            out[key] = tensor
    return out


def hift_tensors(directory):
    state = torch.load(path.join(directory, "hift.pt"), map_location="cpu", weights_only=True)
    return fold_weight_norm(state)


def speech_tokenizer_tensors(directory):
    from s3tokenizer.model_v3 import S3TokenizerV3

    model = S3TokenizerV3("speech_tokenizer_v3")
    model.init_from_onnx(path.join(directory, "speech_tokenizer_v3.onnx"))
    return {key: tensor.contiguous() for key, tensor in model.state_dict().items()}


def campplus_tensors():
    sys.path.insert(0, path.dirname(path.abspath(__file__)))
    import campplus_exporter

    return campplus_exporter.parameters(campplus_exporter.build(campplus_exporter.upstream()))


def write_tokenizer(directory, output):
    """The Qwen vocabulary with CosyVoice3's special tokens added, as `tokenizer.json`."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(path.join(directory, "CosyVoice-BlankEN"))
    tokenizer.add_special_tokens({
        "eos_token": "<|endoftext|>",
        "pad_token": "<|endoftext|>",
        "additional_special_tokens": SPECIAL_TOKENS,
    })

    found = tokenizer.convert_tokens_to_ids("<|endofprompt|>")
    if found != END_OF_PROMPT:
        raise SystemExit(f"<|endofprompt|> is {found}, and the model was trained with it at "
                         f"{END_OF_PROMPT}")

    tokenizer.backend_tokenizer.save(output)

    # The file is what the runtime reads, so the file is what is checked: the same ids as the
    # tokenizer upstream calls, on text with a special token and text without.
    from tokenizers import Tokenizer

    written = Tokenizer.from_file(output)
    for probe in ["You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。",
                  "Hello, world! [breath] 1984 年，I'm here.", "八百标兵奔北坡，北坡炮兵并排跑。"]:
        want = tokenizer([probe])["input_ids"][0]
        got = written.encode(probe, add_special_tokens=False).ids
        if want != got:
            raise SystemExit(f"the written tokenizer reads {probe!r} as {got}, not {want}")

    return len(tokenizer)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-output", required=True, help="the package, e.g. models/cosyvoice3.safetensors")
    parser.add_argument("-checkpoint", help="a local copy of the release; the HF cache otherwise")
    parser.add_argument("-part_size", default="2G", help="how large one file of weights may be")
    arguments = parser.parse_args()

    source = checkpoint_dir(arguments.checkpoint)
    directory = path.dirname(path.abspath(arguments.output))
    stem = stem_of(arguments.output)
    os.makedirs(directory, exist_ok=True)

    tokenizer_file = stem + ".tokenizer.json"
    vocabulary = write_tokenizer(source, path.join(directory, tokenizer_file))
    print(f"wrote {tokenizer_file}: {vocabulary} tokens")

    parts = [
        ("lm", lambda: lm_tensors(source)),
        ("flow", lambda: flow_tensors(source)),
        ("hift", lambda: hift_tensors(source)),
        ("speech_tokenizer", lambda: speech_tokenizer_tensors(source)),
        ("campplus", campplus_tensors),
    ]

    writer = open_weights(arguments.output, parse_size(arguments.part_size))
    total = 0
    for namespace, read in parts:
        tensors = read()
        if len(tensors) != COUNTS[namespace]:
            raise SystemExit(f"{namespace} came out as {len(tensors)} tensors, not "
                             f"{COUNTS[namespace]} -- the release or an exporter has changed")

        for key in sorted(tensors):
            tensor = tensors[key].detach()
            if tensor.dtype not in (torch.float32, torch.int64):
                tensor = tensor.float()
            writer.write_tensor(Context(f"{MODEL_TYPE}.{namespace}.{key}"), tensor,
                                preserve_dtype=True)
            total += tensor.numel()
        print(f"  {MODEL_TYPE}.{namespace}: {len(tensors)} tensors")

    config = {"model": {"type": MODEL_TYPE}, MODEL_TYPE: CONFIG}
    for name in writer.finish(config, None, {"tokenizer": tokenizer_file}):
        print(f"wrote {name}")
    print(f"{total / 1e6:.1f} M parameters")


if __name__ == "__main__":
    main()
