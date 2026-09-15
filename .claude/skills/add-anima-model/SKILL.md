---
name: add-anima-model
description: Export an Anima release -- a denoiser, the Qwen3-0.6B text encoder and the Qwen-Image VAE -- to safetensors and a manifest, publish it to Hugging Face and ModelScope, and wire it into the waifu CLI catalog. ONLY run when the user explicitly asks for this job by name or plainly asks to add, re-export or publish an Anima model -- never on your own initiative, and never because a task merely touches tools/anima_exporter.py, waifu/src/anima/, docs/anima.md, models/ or hub.rs. Anima's weights are someone else's under a non-commercial licence, and what a public repository gains is a human's decision every time. If you are guessing, ask instead of invoking.
---

# Adding a published Anima model

> **Do not run this skill unless the user explicitly asked for it.** It creates and changes public
> Hugging Face and ModelScope repositories, with weights that are not ours and whose licence
> restricts what may be done with them. `docs/anima.md` says it plainly: *nothing about this goes
> to Hugging Face or ModelScope without being asked for separately*. Working on the exporter, the
> runtime, the catalog or a model in `models/` is **not** a reason to start this, and neither is
> noticing that Anima is missing from `hub.rs` -- it is missing on purpose. If the request is
> anything short of "add/re-export/publish this Anima model", stop and ask.

**Read `docs/anima.md` first.** It is the whole of what this model is -- three architectures, a
text path that tokenizes twice, a rectified-flow sampler -- and several decisions below only make
sense against it. This skill is the job of *publishing* one; that file is the model.

**`add-sdxl-model` is a different job and almost none of it transfers.** Anima is not an SDXL fine
tune. What is genuinely shared is the shape of the six steps (check, build, export, verify,
publish, wire in), the rule that `suggested:` is written by hand, and the hubs. Everything else
differs:

| | SDXL | Anima |
|---|---|---|
| the checkpoint | one file | **three**: denoiser, text encoder, VAE |
| what it predicts | epsilon | rectified flow -- there is no `prediction_type` to check |
| the text path | two CLIP encoders, 77 tokens | Qwen3-0.6B, then a T5-tokenized adapter, **512** tokens |
| tokenizers in the package | one | **two**, and neither is published beside the weights |
| the silent error to fear | a v-prediction checkpoint | the wrong release's `steps`/`guidance`, and a manifest missing its rotary ratios |
| what a release costs on disk | ~7 GB in, ~7 GB out | ~5.6 GB in, ~5.6 GB out |

## 1. Check the checkpoint is one this runtime can draw with

### Which release, and why it is the thing to get right

The two published v1.1 denoisers are the same architecture and export identically. What separates
them is the numbers they want, and nothing in the weights says which you have:

| release | steps | guidance | what a step costs |
|---|---|---|---|
| `anima-turbo-*` | 8-12 | **1.0** -- no classifier-free guidance | one model evaluation |
| `anima-aesthetic-*` | 30-50 | 4-5 | two |

Draw an aesthetic release at turbo's eight steps and CFG 1 and you get a soft, unfinished picture;
draw a turbo release at thirty and five and you get a burnt, over-contrasted one. Neither reports
anything. The file name is the only evidence, so **carry it into the model's name and into
`suggested:`, and check the picture matches the release you think you exported** (§4).

### What the header says, without fetching the weights

Same two range requests the SDXL skill uses -- the header is a few hundred KB at the front of the
file:

```bash
curl -sL -r 0-7 "$URL" -o len.bin        # header length: u64 little endian at offset 0
python3 -c "import struct;print(struct.unpack('<Q',open('len.bin','rb').read())[0])"
curl -sL -r 8-<len+7> "$URL" -o hdr.json # the header itself
```

What to read out of it:

- **A prefix of `model.diffusion_model.` or `net.`** and nothing else. The official releases use
  the first, the community 2.9B expansion the second; the exporter strips both and refuses to
  find anything under a third.
- **`x_embedder.proj.1.weight` is `[hidden, 68]`.** `68 // 4 - 1` is 16 latent channels, which is
  the text-to-image row of ComfyUI's `model_detection.py` and **the only row the exporter has
  rotary extrapolation ratios for**. Anything else it refuses by name rather than guessing, which
  is right: guessing those ratios is exactly how they were wrong for months (`docs/anima.md`,
  "Positions").
- **`final_layer.linear.weight` is `[64, hidden]`** -- 2x2 patches over those 16 channels.
- **`blocks.<n>.` counted against an exact prefix** gives the depth: 28 for the official releases,
  40 for the 2.9B expansion. Count carefully -- `llm_adapter.blocks.<n>.` matches a loose pattern
  too, and folding the two together gets the depth wrong. The exporter reads the count off the
  tensors, so a correct count is not something you have to supply; it is something to check the
  file against what its card claims.
- **`blocks.0.cross_attn.k_proj.weight` is `[hidden, 1024]`** -- 1024 is Qwen3-0.6B's width. A
  denoiser expecting a different encoder is not one the shared text file will drive.

### The three files, and what does not change

`circlestone-labs/Anima` publishes under `split_files/`: nine denoisers in
`diffusion_models/`, one `text_encoders/qwen_3_06b_base.safetensors` (1.19 GB) and one
`vae/qwen_image_vae.safetensors` (0.25 GB). **The encoder and the VAE are shared by every
release** -- fetch them once and re-use them for the next model rather than pulling 1.4 GB again.
Only the denoiser (4.18 GB) is per release.

`Gazingstars123/Anima-2.9B` is a **third-party** layer expansion: 40 blocks, `net.` prefix, the
same tensor names and shapes. One implementation covers both. Its terms are its own and want
reading separately rather than assuming they match circlestone-labs'.

### The licence, which is not ours and not permissive

`license: other`, `license_name: circlestone-labs-non-commercial-license`, `license_link` to the
source repository's `LICENSE.md`. It restricts commercial *deployment of the model* while leaving
generated images free to use -- so say both in the card, in those words, rather than compressing
it to "non-commercial". Carry it into every card and every hub's repository metadata; libwaifu's
MIT covers the code and none of the weights.

**Redistribution is the user's call, every time.** Having the weights on disk is not consent to
mirror them.

### Finding what the author advises

The `suggested:` block is written by hand (§3), so finding what goes in it is part of the job, and
the rules are the SDXL skill's:

- **Read the card to the end.** Recommendations sit below the promo links.
- **An instruction can be negative**, and a parser cannot tell. Read what the card asks you *not*
  to do as carefully as what it asks for.
- **Prose about prompting is not evidence; the sample images are.** Anima is prompted through a
  Qwen3 encoder rather than CLIP, so whether it wants danbooru tags, sentences, or tags in
  sentences is a real question with a real answer, and the answer is in what the author actually
  ran. `docs/anima.md` also notes that prompt weighting has to be pushed harder here than for
  SDXL, because the weights multiply the adapter's query stream rather than the encoder's output.

## 2. Build

CUTLASS is not vendored and a CUDA build requires it. A verify run on CPU is not worth
attempting.

```bash
(cd third_party && ./install_cutlass.sh)   # a shallow clone, seconds

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DWITH_OPENMP=ON -DWITH_CUDA=ON
cmake --build build -j$(nproc)
```

**Do not build FlashAttention for this job, and do not believe the SDXL skill's trap row about
it.** Nothing Anima draws through reaches those kernels -- `attention` falls back when they are
absent -- and **a build without them breaks no test**: `stores_and_reads_a_paged_kv_cache` asks
`F::paged_attention_available()` and skips itself, and `flint/CMakeLists.txt` only compiles
`cuda/flash_attn_test.cc` when the option is on. That has been true since `dd804f1`.

The cost of not knowing it is real: `install_flash_attn.sh` compiles every kernel for `sm_80`,
`sm_86`, `sm_89`, `sm_90` *and* `sm_120`, which is tens of minutes of `nvcc` between you and a
picture. If you genuinely want the kernels, build them for this card alone --
`FLASH_ATTN_CUDA_ARCH=120 ./install_flash_attn.sh` on a Blackwell box -- and configure with
`-DWITH_FLASH_ATTN=ON`.

The exporter runs on CPU, so a CPU-only torch wheel is right for the venv:

```bash
python3 -m venv .venv
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install -r tools/requirements.txt
```

**A ComfyUI checkout is needed only for `-test_output`.** It is the reference the test tensors are
computed with -- the arrangement `sdxl_exporter.py` has with diffusers -- and
`tools/anima_comfy.py` says what to install and where. A plain export needs none of it.

## 3. Export

```bash
.venv/bin/python tools/anima_exporter.py \
  -dit    split_files/diffusion_models/anima-turbo-v1.1.safetensors \
  -text   split_files/text_encoders/qwen_3_06b_base.safetensors \
  -vae    split_files/vae/qwen_image_vae.safetensors \
  -output models/<stem>.safetensors \
  -part-size 2GB
```

It prints the three counts it derived (`28 denoiser blocks, 6 adapter blocks, 28 encoder layers`)
and how many tensors were narrowed from bfloat16. What lands beside the manifest:

| file | what |
|---|---|
| `<stem>-0000N-of-0000M.safetensors` | the weights, ~5.6 GB in fp16 -- three parts at `2GB` |
| `<stem>.yaml` | the manifest: `weights:`, `tokenizers:`, `config:`, and no `suggested:` |
| `<stem>.qwen3.tokenizer.json` | the text encoder's vocabulary |
| `<stem>.t5.tokenizer.json` | the adapter's -- a *different algorithm*, not a second vocabulary |

**Both tokenizers are downloaded from elsewhere.** Neither is published beside Anima's weights:
the encoder's is stock `Qwen/Qwen3-0.6B-Base` and the adapter's is `google/t5-v1_1-xxl`, which is
what Cosmos-Predict2 was pretrained against. On a box with no network the export dies here; pass
`-qwen_tokenizer` and `-t5_tokenizer` local paths (a directory, or a bare `spiece.model` for T5).

Narrowing bf16 to fp16 is a narrowing of *range*, not of precision, and the exporter checks every
tensor against fp16's ceiling of 65504 -- turbo v1.1's largest weight is 123. If it ever raises
`OverflowError`, that is the check doing its job: what it would otherwise write is an infinity,
and what an infinity draws is noise.

### The `suggested:` block, which the exporter does not write

`writer.finish(config, None, tokenizers)` -- the `None` is the suggestions, so a fresh manifest has
no `suggested:` at all. **Every part of it is written by hand**, into the `.yaml` the export left
beside the weights:

```yaml
# turbo v1.1: steps and guidance are the release's own, from the model card; the sizes and the
# prompt are filled in from what the card's samples were drawn at.
suggested:
  prompt: 1girl, hatsune miku, solo, twintails, masterpiece, best quality
  sizes:
    - [1024, 1024]
    - [832, 1216]
    - [1216, 832]
  steps: 8
  guidance: 1.0
```

The SDXL skill's rules hold: **write four of the five keys on every model** (`prompt`, `sizes`,
`steps`, `guidance`), `avoid` **only where the author gives one**, keep the author's own wording
verbatim, average a range and round (steps to a multiple of five, guidance to a whole number, down
at the halfway point), and **say in a comment above the block where the numbers came from**.
**Write the prompt about Hatsune Miku**, in the spelling this model's own samples use, so that two
models opened on the same prompt differ by the model.

Four things are Anima's own:

- **`steps` and `guidance` are the release, not the family.** 8 and 1.0 for turbo; the card's
  numbers for aesthetic. `guidance: 1.0` is not decoration -- the pipeline skips the whole
  unconditional pass at exactly 1.0, so it is half the work per step as well as the right picture.
- **Every size must be a multiple of 16.** The VAE scales by 8 and the denoiser patches 2x2 on top
  of it, and `Anima::check_size` refuses anything else. A size the screen offers that the model
  refuses is a box that errors when it is clicked, so check the list you write:
  `python3 -c "print([ (w,h) for w,h in SIZES if w%16 or h%16 ])"` should print nothing. Name at
  least three, the author's own first -- the first is what the screen opens on.
- **The ceiling is 512 tokens, not 77**, and both tokenizations are cut at it: the T5 side at 511
  plus its end marker, because that is the denoiser's context length, and the Qwen3 side alike.
  Nothing sane comes near it, so this is a check rather than a reason to shorten anything -- count
  it with the two files the export just wrote:

  ```python
  from tokenizers import Tokenizer
  t5 = Tokenizer.from_file("models/<stem>.t5.tokenizer.json")
  qwen = Tokenizer.from_file("models/<stem>.qwen3.tokenizer.json")
  len(t5.encode(text, add_special_tokens=False).ids) + 1   # the pipeline appends T5's end marker
  len(qwen.encode(text, add_special_tokens=False).ids)
  ```

- **An `avoid` on a turbo release is read only if someone raises the guidance.** At 1.0 there is
  no unconditional pass to steer away from. Write it anyway where the author gave one, and leave
  the key out where they did not -- the rule does not change, but do not go hunting for a negative
  prompt to fill the box with on a model that will not read it.

Naming, following what is already published:

| thing | form | example |
|---|---|---|
| model stem | lowercase, release and version | `anima-turbo-v1.1` |
| repo (both hubs) | `libwaifu-<stem>` | `ling0322/libwaifu-anima-turbo-v1.1` |
| CLI name | `anima:<release>:<version>` | `anima:turbo:v1.1` |
| CLI alias | `anima:<release>` | `anima:turbo` |

A version is spelled **the way its publisher spells it**: Anima v1.1 is `v1.1`, not `v11`. The
local test fixture is the exception and is deliberately not this -- see "Test data".

## 4. Verify before uploading, not after

```bash
cargo run --release --example load     -- models/<stem>.yaml cuda
cargo run --release --example generate -- models/<stem>.yaml "<prompt>" cuda
```

`load` builds the whole model from the manifest and encodes a prompt with it -- weights,
both tokenizers and the manifest agreeing, which is the cheapest thing that says the export is
readable. Run it *after* writing `suggested:`, since that is also what catches a block that does
not parse.

`generate` reads the kind out of the manifest and takes its numbers from the model's own defaults
and then from `suggested:` -- so a turbo release draws at eight steps and no guidance without
being told, and the line it prints (`1024 by 1024, 8 steps, guidance 1`) is a check on the block
you just wrote. It writes `generate.ppm`.

**You have to look at the picture**, and for Anima looking is necessary but not sufficient:

- Noise means something structural -- the patchify ordering, the residual stream, a missing key.
- **A perfectly reasonable picture can still be wrong.** A flat rotary base against the real ones
  is 13% off on the velocity at a cosine similarity of 0.9959, and it draws a picture nobody would
  question. Eyes cannot catch that. The numbers in `waifu/tests/anima*.rs` against a test package
  can, so if anything about the exporter or the runtime changed, run them (§7) rather than
  trusting the render.
- Ask for the model's own subject. Whether it draws Miku *as Miku* is a stronger check on the text
  path -- two tokenizers, an encoder and a six-block adapter -- than any norm.

Then delete the three source files. Their sha256s are in the card by now, which is what makes the
export reproducible without keeping six gigabytes around.

## 5. Publish

Only after the user has asked for it in so many words. Stage the model's files in a directory of
their own -- manifest, both tokenizers, every weight part, and the card -- and hard link rather
than copy (`ln -f`).

```bash
huggingface-cli upload ling0322/libwaifu-<stem> <stage-dir> . \
  --repo-type model --commit-message "..."
```

**Which CLI depends on which environment you are in, so check rather than assume.** The exporter's
venv pins `huggingface-hub==0.26.2` (`tools/requirements.txt`), where the CLI is
`huggingface-cli` and `hf` does not exist. A machine's own Python is usually far newer -- at 1.x
`huggingface-cli` still exists but *refuses to run*, printing "deprecated and no longer works",
and `hf` is the one that uploads. Use the newer one for the upload where there is one; it is the
half that speaks Xet. `0.00B transferred` is not a no-op -- Xet dedupes against the previous
revision -- so confirm against the API rather than the summary line:

```bash
curl -s "https://huggingface.co/api/models/<repo>?blobs=true" \
  | python3 -c "import json,sys;[print(f['rfilename'],f.get('size')) for f in json.load(sys.stdin)['siblings']]"
```

ModelScope mirror, same repo name and the same files:

```python
from modelscope.hub.api import HubApi
api = HubApi()
api.create_repo(repo_id="ling0322/libwaifu-<stem>", repo_type="model",
                visibility="public", license="other", exist_ok=True)
api.upload_folder(repo_id="ling0322/libwaifu-<stem>", repo_type="model",
                  folder_path="<stage-dir>", commit_message="...", disable_tqdm=True)
print([f["Path"] for f in api.get_model_files(model_id="ling0322/libwaifu-<stem>",
                                              recursive=True)])
```

Credentials live in `~/.modelscope/credentials`; neither hub's login prompt works without a TTY
(`getpass` raises `EOFError`), so ask the user to log in from a real terminal or pass `--token`.
**Never pipe an upload script through `tail`** -- both print what went wrong before the summary
line.

The card carries the **source** licence and names circlestone-labs, with `license: other`,
`license_name`, `license_link`, and a sentence saying that the restriction is on deploying the
model and not on the pictures. Add `not-for-all-audiences` if the samples warrant it.

## 6. Wire it into the CLI

`waifu/src/cli/hub.rs`, once the model is actually on both hubs and not before -- an entry naming
a manifest that is not published is a name that 404s for everyone who types it.

- a `Published` entry in `CATALOG`: `name`, `full_name` (what the picker shows), `repo`,
  `manifest`
- the unversioned alias in `ALIASES`
- `SUPERSEDED` only if a spelling that already went out is being replaced

**`every_model_is_named_the_way_the_others_are` asserts that the first field of every name is
`sdxl`.** It is the one test in that file that has to learn there is a second family; teach it the
set of families rather than dropping the check, since what it is really catching is a copied table
entry that was not finished. `names_are_listed_for_the_usage_text` names each model explicitly and
wants the new one too.

Nothing else in the fetch needs touching: `Manifest::files` already returns the weights *and*
every tokenizer a manifest names, so a two-tokenizer model downloads whole. Confirm the manifest
resolves on both hubs -- the branch names differ, which is the easy thing to get wrong:

```bash
curl -sIL -o /dev/null -w "%{http_code}\n" "https://huggingface.co/<repo>/resolve/main/<manifest>"
curl -sIL -o /dev/null -w "%{http_code}\n" "https://modelscope.cn/models/<repo>/resolve/master/<manifest>"
```

Then `README.md`: the model table, the `waifu draw -m` example list, and a `Recent updates` line.
The table's prose talks about `sdxl:*` names and the licences of SDXL fine tunes; a first Anima
entry makes that paragraph wrong as well as incomplete.

## 7. Test

```bash
./build/unittest
cargo test --release --manifest-path waifu/Cargo.toml --features cli -- --test-threads=1
```

`--features cli` is what compiles `src/cli/`, so without it the `hub.rs` tests you just changed do
not run at all.

The model tests are `#[ignore]`d and need a real package. **Do not run them unless the user asks**
(`CLAUDE.md`); say which were skipped and what runs them. When asked, and for a change that
reaches `waifu/src/anima/` or `tools/anima_*`:

```bash
cargo test --release --manifest-path waifu/Cargo.toml --no-fail-fast \
    --test anima --test anima_pipeline --test anima_tokenizer -- --ignored --test-threads=1
```

About 40 seconds for the three together on a free card, and they need `models/anima-turbo-v11.yaml`
and its test package present. One GPU, one job: a second worktree running these at the same time
produces `Aborted: out of memory`, which reads as a code failure and is not one.

## Traps

| symptom | cause | fix |
|---|---|---|
| `this exporter only knows the rotary extrapolation ratios for the 16 channel text-to-image branch` | the denoiser's latent width is not 16, so ComfyUI's `model_detection.py` would take a different row | not a bug to route around: find the ratios for that row and add it deliberately |
| the model will not load: a manifest key is missing | it was exported before the rotary ratios were written into manifests, and the reader does not default them | re-export. Defaulting them is how they were silently wrong for months |
| `-test_output` exits with "needs -comfyui" | the reference outputs are what ComfyUI computes; there is no second implementation here to ask | pass a ComfyUI checkout, per `tools/anima_comfy.py` |
| the export dies fetching a tokenizer | neither vocabulary is published beside Anima's weights -- they come from `Qwen/Qwen3-0.6B-Base` and `google/t5-v1_1-xxl` | pass `-qwen_tokenizer` / `-t5_tokenizer` local paths |
| `OverflowError: ... which fp16 cannot hold` | a weight outside fp16's range, which bf16 has and fp16 has not | the check is right; the format would otherwise write an infinity |
| a picture that is soft and unfinished, or burnt and over-contrasted | turbo's numbers on an aesthetic release, or the reverse | `suggested.steps`/`guidance` belong to the release, not to Anima |
| `... is not a multiple of 16, which is what this model draws in` | a `suggested.sizes` entry the VAE and the 2x2 patches cannot divide | every size is a multiple of `VAE_SCALE * patch_size` |
| `this model draws from a prompt but not yet from a picture` | Anima has no VAE *encoder* -- the weights are in the package, the layer is not written | expected; `draw -image` is SDXL only |
| the anima tokenizer tests cannot find their corpus | the corpus is named after the **test package**: `<stem>_test.safetensors` puts it in `<stem>_test_qwen3_corpus.tsv` | pass `-test_output models/<stem>_test.safetensors`, not some other stem |
| `hub.rs` tests fail on a name that is plainly right | `every_model_is_named_the_way_the_others_are` expects `sdxl` as the first field | §6 -- that test is part of adding a second family |
| every `anima*.rs` test fails with `Aborted: out of memory` | another job has the card, or `--test-threads=1` was left off | `nvidia-smi` before blaming a diff |
| `hf: command not found`, or `huggingface-cli` prints "deprecated and no longer works" | the two CLIs belong to different `huggingface-hub` versions, and the exporter's venv (0.26.2) is not the machine's Python | `huggingface-cli` in the exporter venv, `hf` in a 1.x environment -- check with `pip show huggingface-hub` |
| a hub login hangs or raises `EOFError` | no TTY: `getpass` cannot prompt | log in from a real terminal, or pass `--token` |

## Test data

The fixture the numerical tests open is its own model and is **not** the published one:
`models/anima-turbo-v11.yaml` -- no dot in the version -- with `models/anima-turbo-v11_test.safetensors`
beside it and the two corpora `anima-turbo-v11_test_qwen3_corpus.tsv` and
`anima-turbo-v11_test_t5_corpus.tsv`. `waifu/tests/anima*.rs` name those files literally, so the
published stem (`anima-turbo-v1.1`) is deliberately a different one. Do not rename the fixture to
match the published spelling.

```bash
.venv/bin/python tools/anima_exporter.py \
  -dit models/... -text models/... -vae models/... \
  -output models/anima-turbo-v11.safetensors \
  -test_output models/anima-turbo-v11_test.safetensors \
  -comfyui /path/to/ComfyUI
```

**Regenerate it whenever `export_test_cases` changes**, and whenever anything it depends on does:
the tensors are a 16x16 latent -- 64 patches, a 128x128 image, 2.4 MB -- covering the encoder's
hidden states, the adapter's context before and after padding, one step's velocity and one decode.
The corpora are 611 texts each and are what catch a package built against the wrong tokenizer
revision, which no reference tensor can.

The test export runs the whole model on CPU through ComfyUI and takes a few minutes.

## Disk

A release is ~4.2 GB of denoiser plus the 1.4 GB the encoder and VAE share, and the export writes
about as much again. The loop is: fetch, check the header, export, hand-write `suggested:`, load,
draw and look, delete the source denoiser, upload both hubs, verify both by API, delete the staged
files, next -- keeping the shared encoder and VAE, which the next release will want.

Build artifacts are the other several gigabytes: a debug `cargo test` compiles a whole second
dependency tree beside the release one CMake already built, so run the Rust suite with
`--release`. A full disk here does not announce itself -- every command fails before it runs,
because the harness cannot create the file it writes output to, and the failure before that one is
a link error that looks like a code problem.
