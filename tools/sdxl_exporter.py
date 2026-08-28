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

"""Export an SDXL checkpoint to a libwaifu package.

The checkpoints these models ship as -- Illustrious, WAI, and every other SDXL fine tune on
Civitai -- are single safetensors files in the original LDM naming, with no tokenizer and no
config beside them. diffusers already knows how to read that layout and hand back a module tree,
so this walks the tree the way llama_exporter.py walks a LlamaForCausalLM rather than rewriting
several hundred lines of key mapping that would have no way to be checked.

The tokenizer is not in the checkpoint. Every SDXL derivative shares the base model's, and both of
its text encoders share one vocabulary of 49408, so one is read from the base repository and
written once.

    python sdxl_exporter.py -checkpoint waiIllustriousSDXL_v170.safetensors -output wai-v17.llmpkg
"""

from __future__ import annotations

import argparse
import configparser
import io
import sys
import zipfile
from os import path

import torch
from torch import nn

from bpe_exporter import read_clip_model
from model_exporter import Context, ModelExporter, Quant, TensorWriter

MODEL_BIN = "model.bin"
TEST_CASE_BIN = "test_case.bin"
TOKENIZER_CORPUS = "tokenizer_corpus.tsv"
MODEL_INI = "model.ini"
TOKENIZER_BIN = "tokenizer.bin"
TOKENIZER_INI = "tokenizer.ini"

# Where the tokenizer comes from when the checkpoint has none, which is the usual case.
BASE_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"


class SdxlExporter(ModelExporter):
    """Writes the four parts of an SDXL model: two text encoders, the U-Net, and the VAE.

    The names are this exporter's own, chosen to mirror the structure rather than to follow either
    diffusers or LDM, the same way llama_exporter.py names a Llama. Nothing reads them yet, so they
    are also the definition of what the runtime side will have to look for.
    """

    def __init__(self, writer: TensorWriter) -> None:
        super().__init__(writer)
        self._float32 = False

    def _write(self, ctx: Context, tensor: torch.Tensor) -> None:
        # Everything narrows to float16 except what is written while _float32 is set, which is the
        # VAE: it overflows float16 on real weights, and every implementation runs it wider.
        self._writer.write_tensor(ctx, tensor, preserve_dtype=self._float32)

    # ---- pieces shared by several parts -------------------------------------------------

    def _export_conv2d(self, ctx: Context, module: nn.Conv2d) -> None:
        self._write(ctx.with_subname("weight"), module.weight)
        if module.bias is not None:
            self._write(ctx.with_subname("bias").with_quant(Quant.NONE), module.bias)

    def _export_group_norm(self, ctx: Context, module: nn.GroupNorm) -> None:
        self._write(ctx.with_subname("weight").with_quant(Quant.NONE), module.weight)
        self._write(ctx.with_subname("bias").with_quant(Quant.NONE), module.bias)

    def _export_geglu(self, ctx: Context, module) -> None:
        """The gated linear unit of a transformer block's feed forward.

        diffusers splits the projection as `value, gate = proj(x).chunk(2)`, and libwaifu's geglu
        reads the first half as the gate, so the two halves are swapped here. Doing it at export
        keeps the runtime's swiglu and geglu reading their input the same way.
        """
        weight = module.proj.weight
        bias = module.proj.bias
        half = weight.shape[0] // 2

        self._write(
            ctx.with_subname("proj.weight"),
            torch.cat((weight[half:], weight[:half]), dim=0))
        self._write(
            ctx.with_subname("proj.bias").with_quant(Quant.NONE),
            torch.cat((bias[half:], bias[:half]), dim=0))

    def _export_attention(self, ctx: Context, module) -> None:
        """One attention block of a U-Net transformer.

        Self attention takes its keys and values from the same tensor as its queries, so all three
        projections have one input width and fuse into one. Cross attention reads keys and values
        from the text encoders, which are 2048 wide against the block's own width, so only those
        two fuse and the query stays on its own.
        """
        q = module.to_q.weight
        k = module.to_k.weight
        v = module.to_v.weight

        if q.shape[1] == k.shape[1]:
            self._write(ctx.with_subname("qkv_proj.weight"), torch.cat((q, k, v), dim=0))
        else:
            self._write(ctx.with_subname("q_proj.weight"), q)
            self._write(ctx.with_subname("kv_proj.weight"), torch.cat((k, v), dim=0))

        self._write(ctx.with_subname("out_proj.weight"), module.to_out[0].weight)
        self._write(
            ctx.with_subname("out_proj.bias").with_quant(Quant.NONE),
            module.to_out[0].bias)

    def _export_transformer_block(self, ctx: Context, block) -> None:
        self.export_layer_norm(ctx.with_subname("norm1"), block.norm1)
        self._export_attention(ctx.with_subname("attn1"), block.attn1)
        self.export_layer_norm(ctx.with_subname("norm2"), block.norm2)
        self._export_attention(ctx.with_subname("attn2"), block.attn2)
        self.export_layer_norm(ctx.with_subname("norm3"), block.norm3)
        self._export_geglu(ctx.with_subname("ff.gate"), block.ff.net[0])
        self.export_linear(ctx.with_subname("ff.out_proj"), block.ff.net[2])

    def _export_transformer(self, ctx: Context, module) -> None:
        self._export_group_norm(ctx.with_subname("norm"), module.norm)
        self.export_linear(ctx.with_subname("in_proj"), module.proj_in)
        for index, block in enumerate(module.transformer_blocks):
            self._export_transformer_block(ctx.with_subname(f"block{index}"), block)
        self.export_linear(ctx.with_subname("out_proj"), module.proj_out)

    def _export_resnet(self, ctx: Context, module) -> None:
        self._export_group_norm(ctx.with_subname("norm1"), module.norm1)
        self._export_conv2d(ctx.with_subname("conv1"), module.conv1)
        self.export_linear(ctx.with_subname("time_proj"), module.time_emb_proj)
        self._export_group_norm(ctx.with_subname("norm2"), module.norm2)
        self._export_conv2d(ctx.with_subname("conv2"), module.conv2)
        if module.conv_shortcut is not None:
            self._export_conv2d(ctx.with_subname("shortcut"), module.conv_shortcut)

    # ---- the U-Net ----------------------------------------------------------------------

    def _export_unet_block(self, ctx: Context, block) -> None:
        for index, resnet in enumerate(getattr(block, "resnets", [])):
            self._export_resnet(ctx.with_subname(f"resnet{index}"), resnet)
        for index, attn in enumerate(getattr(block, "attentions", [])):
            self._export_transformer(ctx.with_subname(f"attn{index}"), attn)

        for index, sampler in enumerate(getattr(block, "downsamplers", None) or []):
            self._export_conv2d(ctx.with_subname(f"downsample{index}"), sampler.conv)
        for index, sampler in enumerate(getattr(block, "upsamplers", None) or []):
            self._export_conv2d(ctx.with_subname(f"upsample{index}"), sampler.conv)

    def _export_unet(self, ctx: Context, unet) -> None:
        self._export_conv2d(ctx.with_subname("conv_in"), unet.conv_in)

        # The timestep embedding, and beside it the one SDXL adds for the pooled text embedding
        # and the size and crop the image was asked for.
        self.export_linear(ctx.with_subname("time_embd.linear1"), unet.time_embedding.linear_1)
        self.export_linear(ctx.with_subname("time_embd.linear2"), unet.time_embedding.linear_2)
        self.export_linear(ctx.with_subname("add_embd.linear1"), unet.add_embedding.linear_1)
        self.export_linear(ctx.with_subname("add_embd.linear2"), unet.add_embedding.linear_2)

        for index, block in enumerate(unet.down_blocks):
            self._export_unet_block(ctx.with_subname(f"down{index}"), block)
        self._export_unet_block(ctx.with_subname("mid"), unet.mid_block)
        for index, block in enumerate(unet.up_blocks):
            self._export_unet_block(ctx.with_subname(f"up{index}"), block)

        self._export_group_norm(ctx.with_subname("conv_norm_out"), unet.conv_norm_out)
        self._export_conv2d(ctx.with_subname("conv_out"), unet.conv_out)

    # ---- the VAE ------------------------------------------------------------------------

    def _export_vae_attention(self, ctx: Context, module) -> None:
        """The VAE's own attention, which is one head as wide as the whole channel count.

        libwaifu's flash attention only takes a head of 64, 128 or 256, so this one falls back to
        the written-out form. Its projections are 1x1 convolutions in the checkpoint but linear
        layers in the diffusers module, which is the shape written here.
        """
        self._export_group_norm(ctx.with_subname("norm"), module.group_norm)
        self._write(
            ctx.with_subname("qkv_proj.weight"),
            torch.cat((module.to_q.weight, module.to_k.weight, module.to_v.weight), dim=0))
        self._write(
            ctx.with_subname("qkv_proj.bias").with_quant(Quant.NONE),
            torch.cat((module.to_q.bias, module.to_k.bias, module.to_v.bias), dim=0))
        self.export_linear(ctx.with_subname("out_proj"), module.to_out[0])

    def _export_vae_block(self, ctx: Context, block) -> None:
        for index, resnet in enumerate(getattr(block, "resnets", [])):
            self._export_vae_resnet(ctx.with_subname(f"resnet{index}"), resnet)
        for index, attn in enumerate(getattr(block, "attentions", None) or []):
            self._export_vae_attention(ctx.with_subname(f"attn{index}"), attn)
        for index, sampler in enumerate(getattr(block, "upsamplers", None) or []):
            self._export_conv2d(ctx.with_subname(f"upsample{index}"), sampler.conv)
        for index, sampler in enumerate(getattr(block, "downsamplers", None) or []):
            self._export_conv2d(ctx.with_subname(f"downsample{index}"), sampler.conv)

    def _export_vae_resnet(self, ctx: Context, module) -> None:
        self._export_group_norm(ctx.with_subname("norm1"), module.norm1)
        self._export_conv2d(ctx.with_subname("conv1"), module.conv1)
        self._export_group_norm(ctx.with_subname("norm2"), module.norm2)
        self._export_conv2d(ctx.with_subname("conv2"), module.conv2)
        if module.conv_shortcut is not None:
            self._export_conv2d(ctx.with_subname("shortcut"), module.conv_shortcut)

    def _export_vae_decoder(self, ctx: Context, decoder) -> None:
        self._export_conv2d(ctx.with_subname("conv_in"), decoder.conv_in)
        self._export_vae_block(ctx.with_subname("mid"), decoder.mid_block)
        for index, block in enumerate(decoder.up_blocks):
            self._export_vae_block(ctx.with_subname(f"up{index}"), block)
        self._export_group_norm(ctx.with_subname("conv_norm_out"), decoder.conv_norm_out)
        self._export_conv2d(ctx.with_subname("conv_out"), decoder.conv_out)

    # ---- the text encoders --------------------------------------------------------------

    def _export_clip_layer(self, ctx: Context, layer) -> None:
        attn = layer.self_attn
        self.export_layer_norm(ctx.with_subname("input_norm"), layer.layer_norm1)
        self._write(
            ctx.with_subname("attn.qkv_proj.weight"),
            torch.cat((attn.q_proj.weight, attn.k_proj.weight, attn.v_proj.weight), dim=0))
        self._write(
            ctx.with_subname("attn.qkv_proj.bias").with_quant(Quant.NONE),
            torch.cat((attn.q_proj.bias, attn.k_proj.bias, attn.v_proj.bias), dim=0))
        self.export_linear(ctx.with_subname("attn.out_proj"), attn.out_proj)

        self.export_layer_norm(ctx.with_subname("post_attn_norm"), layer.layer_norm2)
        self.export_linear(ctx.with_subname("mlp.fc1"), layer.mlp.fc1)
        self.export_linear(ctx.with_subname("mlp.fc2"), layer.mlp.fc2)

    def _export_text_encoder(self, ctx: Context, encoder, projection=None) -> None:
        model = encoder.text_model
        self.export_embedding(ctx.with_subname("token_embd"), model.embeddings.token_embedding)
        self._write(
            ctx.with_subname("position_embd.weight").with_quant(Quant.NONE),
            model.embeddings.position_embedding.weight)

        for index, layer in enumerate(model.encoder.layers):
            self._export_clip_layer(ctx.with_subname(f"block{index}"), layer)

        self.export_layer_norm(ctx.with_subname("final_norm"), model.final_layer_norm)

        # Only the second encoder has one: SDXL takes its pooled conditioning from there.
        if projection is not None:
            self._write(ctx.with_subname("text_proj.weight"), projection.weight)

    # ---- the whole thing ----------------------------------------------------------------

    def _export(self, ctx: Context, pipeline) -> None:
        self._export_text_encoder(ctx.with_subname("text_encoder"), pipeline.text_encoder)
        self._export_text_encoder(
            ctx.with_subname("text_encoder2"),
            pipeline.text_encoder_2,
            projection=pipeline.text_encoder_2.text_projection)
        self._export_unet(ctx.with_subname("unet"), pipeline.unet)

        # Only the decoder: turning an image back into a latent is what the encoder is for, and
        # text to image never does that.
        self._float32 = True
        self._export_vae_decoder(ctx.with_subname("vae"), pipeline.vae.decoder)
        self._export_conv2d(ctx.with_subname("vae.post_quant_conv"), pipeline.vae.post_quant_conv)
        self._float32 = False

    @classmethod
    def generate_config(cls, pipeline, tokenizer) -> configparser.ConfigParser:
        unet = pipeline.unet.config
        vae = pipeline.vae.config
        text = pipeline.text_encoder.config
        text2 = pipeline.text_encoder_2.config

        config = configparser.ConfigParser()
        config["sdxl"] = {}
        section = config["sdxl"]

        section["latent_channels"] = str(vae.latent_channels)
        section["vae_scaling_factor"] = str(vae.scaling_factor)
        section["vae_block_out_channels"] = ",".join(str(c) for c in vae.block_out_channels)
        section["vae_layers_per_block"] = str(vae.layers_per_block)
        section["vae_norm_num_groups"] = str(vae.norm_num_groups)

        section["unet_block_out_channels"] = ",".join(str(c) for c in unet.block_out_channels)
        section["unet_layers_per_block"] = str(unet.layers_per_block)
        section["unet_transformer_layers_per_block"] = ",".join(
            str(n) for n in unet.transformer_layers_per_block)
        section["unet_attention_head_dim"] = ",".join(
            str(n) for n in unet.attention_head_dim) if isinstance(
                unet.attention_head_dim, (list, tuple)) else str(unet.attention_head_dim)
        section["unet_norm_num_groups"] = str(unet.norm_num_groups)
        section["unet_cross_attention_dim"] = str(unet.cross_attention_dim)
        section["unet_addition_time_embed_dim"] = str(unet.addition_time_embed_dim)
        section["unet_projection_class_embeddings_input_dim"] = str(
            unet.projection_class_embeddings_input_dim)

        # The two encoders differ in more than their width: the first activates with the
        # sigmoid approximation OpenAI's CLIP uses, the second with the ordinary GELU.
        for prefix, cfg in (("text", text), ("text2", text2)):
            section[f"{prefix}_hidden_size"] = str(cfg.hidden_size)
            section[f"{prefix}_intermediate_size"] = str(cfg.intermediate_size)
            section[f"{prefix}_num_layers"] = str(cfg.num_hidden_layers)
            section[f"{prefix}_num_heads"] = str(cfg.num_attention_heads)
            section[f"{prefix}_hidden_act"] = str(cfg.hidden_act)
            section[f"{prefix}_norm_eps"] = str(cfg.layer_norm_eps)

        # Both encoders share one vocabulary, and both stop at 77 because that is how many
        # positions their embedding table holds.
        section["context_length"] = str(tokenizer.model_max_length)
        section["vocab_size"] = str(tokenizer.vocab_size)
        section["bot_token_id"] = str(tokenizer.bos_token_id)
        section["eot_token_id"] = str(tokenizer.eos_token_id)

        # The padding that fills a prompt out to the context length is attended to like any other
        # position, so which token it is changes what comes out. The two encoders do not have to
        # agree on it, which is why both are written.
        section["pad_token_id"] = str(pipeline.tokenizer.pad_token_id)
        section["pad_token_id2"] = str(pipeline.tokenizer_2.pad_token_id)

        # SDXL conditions on the second to last hidden state of both encoders rather than the last.
        section["clip_skip"] = "2"

        return config

    @classmethod
    def export(cls, pipeline, tokenizer, fp) -> configparser.ConfigParser:
        ctx = Context("sdxl")
        with TensorWriter(fp) as writer:
            SdxlExporter(writer)._export(ctx, pipeline)

        config = cls.generate_config(pipeline, tokenizer)
        config["model"] = {}
        config["model"]["type"] = "sdxl"
        config["model"]["model_file"] = path.basename(MODEL_BIN)
        return config


# What the reference outputs are computed for. Short, so that the tensors stay small, and fixed so
# that a change in them is a change in the model rather than in the prompt.
TEST_PROMPT = "a photo of an astronaut riding a horse on mars"


def tokenizer_corpus(tokenizer) -> str:
    """Texts and the ids CLIP gives them, as one `text<TAB>id id id` line each.

    The text is stored as ftfy left it, not as it was generated. CLIPTokenizer runs its input
    through ftfy -- ligatures expanded, fullwidth folded, NFC applied -- before it matches its
    pattern, and libwaifu does not, so storing the text as it arrives at the pattern is what makes
    the two comparable. What is being checked here is the merging, not a mojibake repair library.
    """
    import random

    import ftfy

    random.seed(7)
    texts = []

    # The tags these models are actually prompted with.
    tags = [
        "1girl", "solo", "long_hair", "looking_at_viewer", "blush", "smile", "open_mouth",
        "bangs", "blue_eyes", "simple_background", "masterpiece", "best_quality",
        "highly_detailed", "absurdres", "hair_ornament", "school_uniform", "cherry_blossoms"]
    for _ in range(300):
        texts.append(", ".join(random.sample(tags, random.randint(1, 8))))

    # Prose, with the contractions the pattern lists ahead of its character classes.
    words = [
        "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog", "don't", "isn't",
        "we've", "I'll", "she'd", "astronaut", "riding", "horse", "mars", "photograph", "of"]
    for _ in range(300):
        text = " ".join(random.choice(words) for _ in range(random.randint(1, 20)))
        if random.random() < 0.3:
            text += random.choice(["!", "?", "...", "!!!", ".", ",", "?!"])
        if random.random() < 0.2:
            text = text.upper()
        if random.random() < 0.2:
            text = text.title()
        texts.append(text)

    # Numbers, which the pattern takes one digit at a time.
    for _ in range(150):
        texts.append(
            f"{random.randint(0, 999999)} and {random.random():.4f} "
            f"at {random.randint(1, 12)}:{random.randint(0, 59):02d}")

    # Text that is not ascii, which is where the byte level part of the vocabulary is exercised.
    unicode_texts = [
        "東方Project", "初音ミク", "霧雨魔理沙", "中文测试", "한국어", "café", "naïve",
        "Ünicode", "🐱🐶", "🌸 sakura 🌸", "Ω≈ç√∫", "ß"]
    for _ in range(250):
        texts.append(" ".join(
            random.choice(unicode_texts) for _ in range(random.randint(1, 5))))

    # Runs of punctuation, long words, whitespace, and the two names the pattern matches first.
    for _ in range(250):
        texts.append(random.choice([
            "a" * random.randint(1, 40),
            "".join(random.choice("!@#$%^&*()[]{}<>?/|-_=+~`\";:.,")
                    for _ in range(random.randint(1, 20))),
            "  ".join(["x"] * random.randint(1, 10)),
            "\t\n  mixed \r\n whitespace  ",
            "".join(random.choice("abc東🐱1!'") for _ in range(random.randint(1, 30))),
            "<|endoftext|>",
            "<|startoftext|> a cat <|endoftext|>",
            "</w>",
            "a</w>b",
        ]))

    lines = []
    for text in texts:
        text = ftfy.fix_text(text).replace("\t", " ").replace("\n", " ").replace("\r", " ")
        ids = tokenizer(text, add_special_tokens=False).input_ids
        lines.append(text + "\t" + " ".join(str(i) for i in ids))

    return "\n".join(lines) + "\n"


def export_test_cases(pipeline, fp) -> None:
    """Write what huggingface produces for one prompt, so the runtime can be checked against it.

    Only the text encoders: their output is what everything downstream is conditioned on, it is
    cheap to compute, and getting it right is most of getting the prompt right. The U-Net and the
    VAE need the sampler to be meaningful and are worth their own cases later.
    """
    ctx = Context("test_case")
    with TensorWriter(fp) as writer:
        ids = pipeline.tokenizer(
            TEST_PROMPT,
            padding="max_length",
            max_length=pipeline.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt").input_ids
        ids2 = pipeline.tokenizer_2(
            TEST_PROMPT,
            padding="max_length",
            max_length=pipeline.tokenizer_2.model_max_length,
            truncation=True,
            return_tensors="pt").input_ids

        writer.write_tensor(ctx.with_subname("input_ids"), ids.to(torch.int64))
        writer.write_tensor(ctx.with_subname("input_ids2"), ids2.to(torch.int64))

        with torch.no_grad():
            out = pipeline.text_encoder(ids, output_hidden_states=True)
            out2 = pipeline.text_encoder_2(ids2, output_hidden_states=True)

        # SDXL conditions on the second to last hidden state of both, and takes its pooled vector
        # from the second encoder.
        writer.write_tensor(
            ctx.with_subname("hidden"), out.hidden_states[-2], preserve_dtype=True)
        writer.write_tensor(
            ctx.with_subname("hidden2"), out2.hidden_states[-2], preserve_dtype=True)
        writer.write_tensor(
            ctx.with_subname("pooled2"), out2.text_embeds, preserve_dtype=True)


def load_pipeline(checkpoint: str, base_model: str, variant: str = None):
    """Read a checkpoint, whichever of the two shapes it comes in."""
    from diffusers import StableDiffusionXLPipeline

    if checkpoint.endswith(".safetensors") or checkpoint.endswith(".ckpt"):
        print(f"read single file checkpoint {checkpoint}")
        # A single file has no tokenizer beside it, so the base model supplies one.
        return StableDiffusionXLPipeline.from_single_file(
            checkpoint,
            torch_dtype=torch.float32,
            tokenizer=_base_tokenizer(base_model, ""),
            tokenizer_2=_base_tokenizer(base_model, "_2"))

    # Everything is read as float32 whatever it was stored as, because the exporter decides on
    # its own what to narrow: the U-Net and the text encoders to float16, the VAE not at all.
    print(f"read diffusers model {checkpoint}")
    return StableDiffusionXLPipeline.from_pretrained(
        checkpoint,
        torch_dtype=torch.float32,
        variant=variant)


def _base_tokenizer(base_model: str, suffix: str):
    from transformers import CLIPTokenizer

    return CLIPTokenizer.from_pretrained(base_model, subfolder=f"tokenizer{suffix}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="export an SDXL model to the libwaifu format.")
    parser.add_argument(
        "-checkpoint",
        type=str,
        required=True,
        help="a single file .safetensors checkpoint, or a diffusers model directory or name.")
    parser.add_argument(
        "-base_model",
        type=str,
        default=BASE_MODEL,
        help="where the tokenizer comes from when the checkpoint has none.")
    parser.add_argument("-output", type=str, default="sdxl.llmpkg", help="output file name.")
    parser.add_argument(
        "-test_output",
        type=str,
        default=None,
        help="where to write the reference outputs, if they are wanted.")
    parser.add_argument(
        "-variant",
        type=str,
        default=None,
        help='which weights of a diffusers model to read, as in "fp16".')
    args = parser.parse_args()

    pipeline = load_pipeline(args.checkpoint, args.base_model, args.variant)
    pipeline = pipeline.to("cpu")

    if pipeline.tokenizer.vocab_size != pipeline.tokenizer_2.vocab_size:
        print(
            "the two text encoders do not share a vocabulary, which this exporter assumes",
            file=sys.stderr)
        sys.exit(1)

    with zipfile.ZipFile(args.output, "w", compression=zipfile.ZIP_STORED) as package:
        libwaifu_tokenizer = read_clip_model(args.base_model, subfolder="tokenizer")

        with package.open(MODEL_BIN, "w", force_zip64=True) as fp:
            config = SdxlExporter.export(pipeline, pipeline.tokenizer, fp)

        with package.open(MODEL_INI, "w", force_zip64=True) as fp:
            config.write(io.TextIOWrapper(fp))

        with package.open(TOKENIZER_BIN, "w", force_zip64=True) as fp:
            libwaifu_tokenizer.save(fp)

        with package.open(TOKENIZER_INI, "w", force_zip64=True) as fp:
            libwaifu_tokenizer.get_config().to_ini(TOKENIZER_BIN).write(io.TextIOWrapper(fp))

    print(f"wrote {args.output}")

    if args.test_output:
        with zipfile.ZipFile(args.test_output, "w", compression=zipfile.ZIP_STORED) as package:
            with package.open(TEST_CASE_BIN, "w", force_zip64=True) as fp:
                export_test_cases(pipeline, fp)

            corpus = tokenizer_corpus(pipeline.tokenizer)
            with package.open(TOKENIZER_CORPUS, "w", force_zip64=True) as fp:
                fp.write(corpus.encode("utf-8"))
            print(f"wrote {corpus.count(chr(10))} corpus lines")
        print(f"wrote {args.test_output}")
