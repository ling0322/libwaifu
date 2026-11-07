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

"""What Qwen3-ASR-0.6B computes on real recordings, for `waifu/tests/qwen3_asr.rs`.

    .venv-qwen3-asr/bin/python tools/qwen3_asr_reference.py \\
        -output models/qwen3-asr_test.safetensors -recording zh=<a.wav> -recording en=<b.wav>

in the venv `tools/requirements-qwen3-asr.txt` describes. Reads `Qwen/Qwen3-ASR-0.6B` from the
HuggingFace cache (or `-checkpoint`), in float32.

# What the reference is

Not upstream's package. `qwen-asr` is a set of transformers classes of its own; what they compute
is put together here out of three parts:

- **The features** are transformers' own `WhisperFeatureExtractor`, built from the release's
  `preprocessor_config.json` -- the class upstream's processor wraps, called the way it calls it
  (`padding=True`, `truncation=False`, one recording at a time).
- **The audio encoder** is `audio_tower` below: upstream's `Qwen3ASRAudioEncoder.forward`
  written out in plain torch over the release's tensors -- the hundred-frame chunks, the three
  convolutions, the position table that starts again in every chunk, and attention that does not
  cross an eight-second window.
- **The decoder** is transformers' `Qwen3ForCausalLM` with the release's `thinker.model` and
  `thinker.lm_head`. Upstream's text model is Qwen3 with interleaved MRoPE; for a sequence of audio
  and text its three position axes are the same number (`get_rope_index` is a cumulative sum,
  repeated three times), and interleaving three equal axes is plain RoPE.

The recordings are real speech. Each is resampled to 16 kHz with librosa, as upstream's
`load_audio` does, and the 16 kHz wave is written too, so that the runtime is tested on exactly
what this read rather than on its own resampling of it.

# What is written, per recording `<name>`

| name | what |
| --- | --- |
| `<name>.wave` | the 16 kHz recording, float32 |
| `<name>.mel` | `(128, T)`, the features |
| `<name>.audio` | `(N, 1024)`, what the encoder hands the decoder |
| `<name>.prompt` | the ids of the prompt, with `N` audio placeholders |
| `<name>.said` | the ids greedy decoding said after it, up to and with the end token |
| `<name>.logp` | `(len(said), vocab)`, teacher-forced log probabilities of each said token's row |

and `<name>.text`'s ids under `<name>.text_ids` would be redundant: the test decodes `said`.
"""

import argparse
import json
import math
import os
import sys
from os import path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, path.dirname(path.abspath(__file__)))
from model_writer import TensorBag  # noqa: E402

REPO = "Qwen/Qwen3-ASR-0.6B"
SAMPLE_RATE = 16000

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
AUDIO_START = "<|audio_start|>"
AUDIO_END = "<|audio_end|>"
AUDIO_PAD = "<|audio_pad|>"
END_TOKENS = (151645, 151643)


class Name:
    """What `TensorBag.write_tensor` takes a name through."""

    def __init__(self, name):
        self.name = name


def checkpoint_dir(given):
    if given:
        return given
    from huggingface_hub import snapshot_download

    return snapshot_download(REPO)


def read_state(directory):
    from safetensors.torch import load_file

    state = load_file(path.join(directory, "model.safetensors"))
    return {key.removeprefix("thinker."): tensor.float() for key, tensor in state.items()}


def output_lengths(frames):
    """`_get_feat_extract_output_lengths`: what three stride-two convolutions leave of `frames`
    mel frames read in hundred-frame chunks, thirteen for every whole chunk."""
    rest = frames % 100
    after = (rest - 1) // 2 + 1
    return ((after - 1) // 2 + 1 - 1) // 2 + 1 + (frames // 100) * 13


def sinusoids(length, channels, max_timescale=10000):
    increment = np.log(max_timescale) / (channels // 2 - 1)
    inverse = torch.exp(-increment * torch.arange(channels // 2).float())
    scaled = torch.arange(length)[:, None] * inverse[None, :]
    return torch.cat([torch.sin(scaled), torch.cos(scaled)], dim=1)


def audio_tower(state, config, mel):
    """`Qwen3ASRAudioEncoder.forward` on one recording's `(128, T)` features: `(N, output_dim)`."""
    t = lambda name: state["audio_tower." + name]  # noqa: E731
    frames = mel.shape[1]
    chunk = config["n_window"] * 2

    # Hundred-frame chunks, the last one what is left, padded to the longest -- which is a
    # hundred unless the whole recording is shorter.
    count = math.ceil(frames / chunk)
    lengths = [chunk] * count
    if frames % chunk:
        lengths[-1] = frames % chunk
    pieces = mel.T.split(lengths, dim=0)
    padded = torch.nn.utils.rnn.pad_sequence(pieces, batch_first=True).transpose(1, 2)
    x = padded.unsqueeze(1)

    for name in ("conv2d1", "conv2d2", "conv2d3"):
        x = F.gelu(F.conv2d(x, t(name + ".weight"), t(name + ".bias"), stride=2, padding=1))
    b, c, f, w = x.shape
    x = F.linear(x.permute(0, 3, 1, 2).reshape(b, w, c * f), t("conv_out.weight"))
    x = x + sinusoids(config["max_source_positions"], config["d_model"])[:w].unsqueeze(0)

    keep = [output_lengths(length) for length in lengths]
    x = torch.cat([x[index, :length] for index, length in enumerate(keep)], dim=0)

    # Attention within windows of `n_window_infer` frames' worth of tokens, never across.
    total = x.shape[0]
    assert total == output_lengths(frames)
    window = w * (config["n_window_infer"] // chunk)
    bounds = list(range(0, total, window)) + [total]

    heads = config["encoder_attention_heads"]
    width = config["d_model"]
    for index in range(config["encoder_layers"]):
        p = f"layers.{index}."
        h = F.layer_norm(x, (width,), t(p + "self_attn_layer_norm.weight"),
                         t(p + "self_attn_layer_norm.bias"), 1e-5)
        q = F.linear(h, t(p + "self_attn.q_proj.weight"), t(p + "self_attn.q_proj.bias"))
        k = F.linear(h, t(p + "self_attn.k_proj.weight"), t(p + "self_attn.k_proj.bias"))
        v = F.linear(h, t(p + "self_attn.v_proj.weight"), t(p + "self_attn.v_proj.bias"))
        outs = []
        for start, end in zip(bounds[:-1], bounds[1:]):
            split = lambda z: z[start:end].reshape(end - start, heads, -1).transpose(0, 1)  # noqa
            out = F.scaled_dot_product_attention(split(q), split(k), split(v))
            outs.append(out.transpose(0, 1).reshape(end - start, width))
        attended = torch.cat(outs, dim=0)
        x = x + F.linear(attended, t(p + "self_attn.out_proj.weight"),
                         t(p + "self_attn.out_proj.bias"))

        h = F.layer_norm(x, (width,), t(p + "final_layer_norm.weight"),
                         t(p + "final_layer_norm.bias"), 1e-5)
        h = F.gelu(F.linear(h, t(p + "fc1.weight"), t(p + "fc1.bias")))
        x = x + F.linear(h, t(p + "fc2.weight"), t(p + "fc2.bias"))

    x = F.layer_norm(x, (width,), t("ln_post.weight"), t("ln_post.bias"), 1e-5)
    x = F.gelu(F.linear(x, t("proj1.weight"), t("proj1.bias")))
    return F.linear(x, t("proj2.weight"), t("proj2.bias"))


def decoder(state, config):
    from transformers import Qwen3Config, Qwen3ForCausalLM

    text = dict(config["text_config"])
    text.pop("rope_scaling", None)
    keep = {key: text[key] for key in (
        "vocab_size", "hidden_size", "intermediate_size", "num_hidden_layers",
        "num_attention_heads", "num_key_value_heads", "head_dim", "max_position_embeddings",
        "rms_norm_eps", "rope_theta", "attention_bias", "hidden_act")}
    model = Qwen3ForCausalLM(Qwen3Config(**keep, tie_word_embeddings=False))
    weights = {key: tensor for key, tensor in state.items()
               if key.startswith("model.") or key.startswith("lm_head.")}
    missing, unexpected = model.load_state_dict(weights, strict=False)
    missing = [key for key in missing if "rotary_emb" not in key]
    if missing or unexpected:
        raise SystemExit(f"the decoder did not take the release's tensors: {missing} {unexpected}")
    return model.eval()


def prompt_ids(tokenizer, count):
    """Upstream's chat template with no context and no language, the audio expanded to `count`
    placeholders: `<|im_start|>system\\n<|im_end|>\\n<|im_start|>user\\n<|audio_start|>
    <|audio_pad|>*count<|audio_end|><|im_end|>\\n<|im_start|>assistant\\n`."""
    text = (f"{IM_START}system\n{IM_END}\n{IM_START}user\n{AUDIO_START}"
            f"{AUDIO_PAD * count}{AUDIO_END}{IM_END}\n{IM_START}assistant\n")
    return tokenizer(text, add_special_tokens=False)["input_ids"]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-output", required=True)
    parser.add_argument("-checkpoint")
    parser.add_argument("-recording", action="append", required=True,
                        help="name=path of a recording; any rate, any length up to a minute")
    parser.add_argument("-max_new_tokens", type=int, default=256)
    arguments = parser.parse_args()

    import librosa
    from transformers import AutoTokenizer, WhisperFeatureExtractor

    torch.manual_seed(0)
    source = checkpoint_dir(arguments.checkpoint)
    with open(path.join(source, "config.json")) as f:
        config = json.load(f)["thinker_config"]
    audio_config = config["audio_config"]
    state = read_state(source)
    features = WhisperFeatureExtractor.from_pretrained(source)
    tokenizer = AutoTokenizer.from_pretrained(source)
    model = decoder(state, config)
    audio_pad = tokenizer.convert_tokens_to_ids(AUDIO_PAD)
    if audio_pad != config["audio_token_id"]:
        raise SystemExit(f"{AUDIO_PAD} is {audio_pad}, not {config['audio_token_id']}")

    bag = TensorBag()
    for item in arguments.recording:
        name, recording = item.split("=", 1)
        wave, _ = librosa.load(recording, sr=SAMPLE_RATE, mono=True)
        wave = wave.astype(np.float32)

        extracted = features([wave], sampling_rate=SAMPLE_RATE, padding=True, truncation=False,
                             return_attention_mask=True, return_tensors="pt")
        frames = int(extracted["attention_mask"][0].sum())
        mel = extracted["input_features"][0, :, :frames].float()

        with torch.no_grad():
            audio = audio_tower(state, audio_config, mel)
            ids = prompt_ids(tokenizer, audio.shape[0])
            ids_t = torch.tensor([ids])
            embeds = model.get_input_embeddings()(ids_t)
            embeds[ids_t == audio_pad] = audio

            # Greedy, as upstream's generation config asks.
            said = []
            past = None
            step = embeds
            for _ in range(arguments.max_new_tokens):
                out = model(inputs_embeds=step, past_key_values=past, use_cache=True)
                past = out.past_key_values
                token = int(out.logits[0, -1].argmax())
                said.append(token)
                if token in END_TOKENS:
                    break
                step = model.get_input_embeddings()(torch.tensor([[token]]))

            # Every row the said tokens were chosen from, in one pass over the whole sequence.
            whole = torch.cat([embeds, model.get_input_embeddings()(torch.tensor([said[:-1]]))],
                              dim=1)
            logits = model(inputs_embeds=whole).logits[0, len(ids) - 1:]
            logp = torch.log_softmax(logits.float(), dim=-1)
            assert logp.shape[0] == len(said)
            assert [int(i) for i in logp.argmax(-1)] == said, "the teacher-forced pass disagrees"

        text = tokenizer.decode(said, skip_special_tokens=True)
        print(f"{name}: {len(wave) / SAMPLE_RATE:.2f}s, {frames} frames, {audio.shape[0]} audio "
              f"tokens, said {len(said)}: {text!r}")

        bag.write_tensor(Name(f"{name}.wave"), torch.from_numpy(wave), preserve_dtype=True)
        bag.write_tensor(Name(f"{name}.mel"), mel.contiguous(), preserve_dtype=True)
        bag.write_tensor(Name(f"{name}.audio"), audio.contiguous(), preserve_dtype=True)
        bag.write_tensor(Name(f"{name}.prompt"), torch.tensor(ids, dtype=torch.int64))
        bag.write_tensor(Name(f"{name}.said"), torch.tensor(said, dtype=torch.int64))
        bag.write_tensor(Name(f"{name}.logp"), logp.to(torch.float16).contiguous())

    os.makedirs(path.dirname(path.abspath(arguments.output)), exist_ok=True)
    bag.save(arguments.output)


if __name__ == "__main__":
    main()
