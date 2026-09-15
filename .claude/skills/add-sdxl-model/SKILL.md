---
name: add-sdxl-model
description: Export an SDXL checkpoint to safetensors and a manifest, publish it to Hugging Face and ModelScope, and wire it into the waifu CLI catalog. ONLY run when the user explicitly asks for this job by name or plainly asks to add, re-export, or publish a model -- never on your own initiative, and never because a task merely touches the exporter, the catalog, models/ or a model's files. Public repositories are not yours to add to or change -- every action that touches one is a human's decision, and this skill makes several. If you are guessing, ask instead of invoking.
---

# Adding a published SDXL model

> **Do not run this skill unless the user explicitly asked for it.** It creates and changes public
> Hugging Face and ModelScope repositories under the user's account. Whether a public repository
> gains a model, or an existing one is modified, is a human's call every single time -- not a
> conclusion you reach from the state of the code. Working on `tools/sdxl_exporter.py`,
> `hub::names()`, `models/` or anything else this skill touches is *not* a reason to start it.
> Neither is noticing that a model is missing, stale, or exportable. If the request is anything
> short of "add/re-export/publish this model", stop and ask -- an unwanted publish costs far more
> than one question.

The whole job is six steps: check the checkpoint, build, export, verify, publish, wire in. The
steps are cheap; the traps are in the environment and they cost hours if you meet them one at a
time. Read "Traps" first if you are only doing part of this.

## 1. Check the checkpoint is one this runtime can draw with

**It must be epsilon-prediction.** libwaifu's Euler sampler reads epsilon. A v-prediction
checkpoint is the same schedule parameterized differently -- it loads, exports and runs, and
produces noise. Nothing anywhere reports an error.

```bash
curl -sL "https://huggingface.co/<repo>/raw/main/scheduler/scheduler_config.json" | grep prediction_type
# want: "prediction_type": "epsilon"
```

Many anime fine tunes publish both (NoobAI has `noobai-XL-1.1` eps and `noobai-XL-Vpred-1.0`).
Pick deliberately. If there is no diffusers layout to read the config from -- a single file
checkpoint has none -- read the safetensors header instead, over two range requests, and look for
the `v_pred` marker a v-prediction checkpoint carries:

```bash
curl -sL -r 0-7 "$URL" -o len.bin        # header length: u64 little endian at offset 0
python3 -c "import struct;print(struct.unpack('<Q',open('len.bin','rb').read())[0])"
curl -sL -r 8-<len+7> "$URL" -o hdr.json # the header itself, a few hundred KB
```

No key containing `v_pred` means epsilon. The same header settles the architecture without
fetching the weights: `model.diffusion_model.input_blocks.0.0.weight` is `[320, 4, 3, 3]`,
`...input_blocks.4.1.transformer_blocks.0.attn2.to_k.weight` is `[640, 2048]` for a cross
attention dim of 2048, `label_emb.0.0.weight` is `[1280, 2816]`, and counting
`input_blocks.<n>.1.transformer_blocks.<m>.` gives depths `{4:2, 5:2, 7:10, 8:10}` -- stock SDXL,
where level 0 has no attention at all.

### Finding what the author advises

The `suggested:` block is written by hand (§3), so finding what goes in it is part of the job.

Read the card **to the end**. Recommendations sit below walls of promo links and group invites,
several thousand characters down; truncating at the first screen and concluding "the author gave
none" is wrong, and it is wrong silently.

**An instruction can be negative, and a parser cannot tell.** WAI's author asks for the opposite
of what every other card asks for: *"please do not add too many quality and aesthetic-related
tags, nor overly long negative prompts, as this will actually reduce image quality"*. A reader
looking for a recommended prompt finds nothing there and falls back to the family's
`masterpiece, best quality, ...`, which is precisely what the author said not to do. Read what the
card asks you **not** to do as carefully as what it asks for -- it is the half a machine misses,
and the reason the exporter no longer tries.

**Prose about prompting is not evidence; the sample images are.** Whether a model wants danbooru
tags or sentences decides a column of the README table, and authors describe it loosely -- "no
complicated tags needed" means *you need not pile them up*, not *write sentences*. Settle it from
what the author actually ran. On Civitai, `GET /api/v1/model-versions/<id>` carries `images[].meta`
with the real `prompt`, `steps`, `cfgScale`, `sampler` and `Size` of each sample; a comma
separated tag list, usually behind `masterpiece, best quality`, is a tag model. Those numbers beat
the stated ranges and often disagree with them -- put what was actually run in `suggested:`, and
quote both in the card, saying which is which. The sample prompts are also the honest source for a
`not-for-all-audiences` tag, whatever the model level `nsfw` flag claims.

### When the source is Civitai

Community fine tunes often live only on Civitai, which behaves nothing like the hub:

- **Downloading needs a token.** Anonymous is `401`. Pass one as
  `-H "Authorization: Bearer $TOKEN"` or `?token=$TOKEN`; ask the user for it, and tell them to
  rotate it afterwards if it went through the conversation.
- **The official sha256 is published, so verify against it.**
  `GET https://civitai.com/api/v1/model-versions/<versionId>` gives `files[].name`, `sizeKB` and
  `hashes.SHA256`. This is the reason to prefer the official file over a third-party re-upload on
  the hub: a re-upload converted to the diffusers layout has different bytes and can be checked
  against nothing.
- **There is no SPDX license.** `GET https://civitai.com/api/v1/models/<modelId>` returns flags
  instead: `allowNoCredit`, `allowDerivatives`, `allowDifferentLicense` and `allowCommercialUse`
  (a list, e.g. `['RentCivit','Image']` -- which is *not* general commercial use). Write them into
  the new card as `license: other` with a `license_name` and a `license_link` to the model page,
  name the author, and say plainly what is not allowed. `allowNoCredit: false` makes attribution a
  requirement rather than a courtesy.
- Cards link the `civitai.red` mirror; the API is on `civitai.com`.
- **The search endpoint does not return NSFW-flagged models.** `?query=` will swear a model does
  not exist while `/api/v1/models/<id>` hands it straight over, and `?username=` omits it from the
  author's own list. If a model is plainly published and search cannot find it, fetch it by id
  before concluding there is no official page.

**Find the official page, and check the mirror against it.** Anime fine tunes are re-uploaded to
the hub by people who are not their authors -- often the only copy that is downloadable -- and a
mirror restates the license from memory. WAI v17 is published on the hub by a third party under
`cdla-permissive-2.0`, while its author's own terms allow no general commercial use at all. Take
the license, the author's name and the recommendations from the official page; then verify that
the mirror's bytes are the author's by its published sha256 before using the file. A mirror whose
hash matches is the author's file with someone else's card wrapped round it, and only the file is
worth trusting.

**Redistribution is the user's call, every time.** These are one person's weights, published on a
site whose flags speak about derivatives rather than about mirroring. Ask before uploading, and do
not infer consent from being handed a download token.

Also confirm the architecture is stock SDXL: `unet/config.json` should have
`cross_attention_dim: 2048`, `transformer_layers_per_block: [1, 2, 10]`, `block_out_channels:
[320, 640, 1280]`.

## 2. Build, including the third_party prerequisites

CMake builds libunwind on its own, but CUTLASS is still a script, and the CUDA path needs more
than the README says:

```bash
(cd third_party && ./install_cutlass.sh)    # CUDA: required, not vendored
# FlashAttention is NOT needed: nothing here draws through it, and since dd804f1 a build
# without it breaks no test. See the traps table before running install_flash_attn.sh.

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DWITH_OPENMP=ON \
      -DWITH_CUDA=ON
cmake --build build -j$(nproc)
```

**Build with CUDA even if you only mean to verify.** A verify run is ~6 seconds on a GPU and
20+ minutes on CPU, because x64 has no half kernels and the package is widened from 7 GB of
float16 to 13.7 GB of float32 as it loads.

The exporter itself runs on CPU, so a CPU-only torch wheel is the right choice for the venv and
saves several GB:

```bash
python3 -m venv .venv
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install -r tools/requirements.txt
```

## 3. Export

```bash
.venv/bin/python tools/sdxl_exporter.py \
  -checkpoint <name>.safetensors \
  -output <stem>.safetensors \
  -part-size 2GB
```

The exporter writes no `suggested:` block. **Every part of it is written by hand**, into the
`.yaml` the export leaves beside the weights -- that is what the manifest being a small text file
rather than an entry inside a package is for. Reading a prompt out of a card was tried and taken
back out: what the author advises is scattered through English prose, and a parser that guesses at
it is wrong quietly, which is the worst way to be wrong.

So after the export, open the `.yaml` and write what the source says:

```yaml
suggested:
  prompt: masterpiece, best quality, amazing quality, very aesthetic, absurdres
  avoid: worst quality, low quality, jpeg artifacts, bad anatomy, watermark, signature
  sizes:
    - [832, 1216]
    - [1216, 832]
    - [1024, 1024]
  steps: 30
  guidance: 4.5
```

**Write four of the five keys on every model: `prompt`, `sizes`, `steps` and `guidance`.** They
are each optional to the reader -- a missing one means "this model has no opinion" and
falls back to the build's own defaults -- but a published model should have an opinion about all
four, because the person who fetched it cannot see the card and the screen is the only thing that
will ever tell them. A model that answers one question and stays silent on three has left the
other three to a default that was never chosen for it.

**Nothing in `suggested` may exceed 77 tokens** -- `prompt` and `avoid` alike. That is
`context_length` in the model's own config, and both encoders stop there: a longer one is not
refused, it is silently cut, and the tail the author cared about is the part that goes. Count it
rather than guessing, with the tokenizer the model ships:

```python
from transformers import CLIPTokenizerFast
tok = CLIPTokenizerFast.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0",
                                        subfolder="tokenizer")
len(tok(text)["input_ids"])          # BOS and EOS included, so this is the number that matters
```

Nothing published so far comes near it -- NoobAI's negative prompt is the longest at 47 -- so this
is a check to run, not a reason to shorten anything. **Where the author wrote one, keep it
verbatim**, padding and repetition and all: `worst quality, low quality, lowres` saying much the
same thing three times is how the author tuned it, and an editor who trims it to taste is guessing
at a model they did not train. Only a list that genuinely breaks 77 gets cut, and then from the
end.

**`avoid` is the exception among the five: write it only where the author gives one.** Unlike the other four it
has no sensible fallback, because a negative prompt is not generic -- NoobAI names `old`, `early`,
`anthro` and `feral`, which belong to a model trained on Danbooru *and* e621 and would be noise in
any other model's box. A made-up one would fill the box with tags nobody chose, and the reader
could not tell it from the author's. No `avoid` key leaves the box empty, which is the honest
answer when the card is silent.

Where the card says nothing, fall back to what the model's **base** gives, per field:

| key | when the card says it | when it does not |
|---|---|---|
| `prompt` | the author's own prefix, verbatim | the convention of the family the fine tune belongs to -- a quality-tag prefix for anything descended from Illustrious or NoobAI, a plain sentence for base SDXL |
| `avoid` | the author's negative prompt, verbatim | **leave the key out entirely** |
| `sizes` | the author's, theirs first | SDXL's buckets, which every derivative inherits |
| `steps` | the author's | SDXL's 30 |
| `guidance` | the author's | SDXL's 5.0 |

**Where the author gives a range, average it, then round: steps to a multiple of five, guidance to
a whole number, and down when the average falls exactly between.** `Steps: 25 ~ 30` averages 27.5
and is written `25`; `CFG 5 ~ 6` averages 5.5 and is written `5`. These are numbers a person will
read off a screen and type again, so they should look like numbers a person would have chosen --
`27` steps at a guidance of `5.5` reads as arithmetic rather than as advice, and tells the reader
nothing the round number does not. Rounding down also keeps the opening picture the cheaper one,
which is the right way to be wrong about a default.

**Write the prompt about Hatsune Miku**, whichever model it is and wherever the tags came from:
`1girl, hatsune miku, solo, twintails` and whatever else the model or the author's own examples
suggest, in the spelling that model wants -- tags for a booru fine tune, a sentence for base SDXL.
One subject across the catalogue makes the models comparable: open two of them on the same prompt
and the difference on screen is the model rather than the prompt. She is also the safest thing to
put in a box that opens by itself -- drawn by everything, owned by nobody, and no one's likeness.

**`sizes` must name at least three.** One size is not a choice and two is barely one; the screen
offers this list, and a model that narrows it to nothing has made the screen worse than the fixed
list it replaced. SDXL's buckets, to fill from: `1024x1024`, `1152x896`, `896x1152`, `1216x832`,
`832x1216`, `1344x768`, `768x1344`, `1536x640`, `640x1536`. Put the author's own first -- the
first is what the screen opens on.

**The `.yaml` carries no commentary of its own, with one exception: say where the `suggested:`
values came from, in a comment above the block.** Which of the four are the author's and which
were filled in from the base is the one thing a reader cannot recover from the numbers themselves,
and it belongs against the numbers rather than in the model card, where nobody is looking at the
moment they wonder. Everything else in the file is a record of what the model is and wants no
annotation at all -- and the model card needs no table tracing each number to its source either.

`-part-size 2GB` is the published convention. A 7.1 GB package becomes four parts
(2.02 / 2.01 / 2.01 / 1.06 GB) named `<stem>-0000N-of-00004.safetensors`. Use the exporter's own
`-part-size` rather than `tools/split_model.py` -- one step, same result.

Naming, matching what is already published:

| thing | form | example |
|---|---|---|
| model stem | lowercase, version suffixed | `noobai-xl-v11` |
| repo (both hubs) | `libwaifu-<stem>` | `ling0322/libwaifu-noobai-xl-v11` |
| CLI name | `sdxl:<short>:<version>` | `sdxl:noob:v1.1` |
| CLI alias | `sdxl:<short>` | `sdxl:noob` |

A version is spelled **the way its publisher spells it**: NoobAI-XL v1.1 is `v1.1`, not `v11`;
SDXL 1.0 is `v1.0`; WAI v17 and One Obsession v24 have no minor part and so carry no dot. Names
that went out under an older spelling stay answerable -- see `SUPERSEDED` in `hub.rs`, which
resolves them without listing them in `names()`.

## 4. Verify before uploading, not after

Two examples, in this order:

```bash
cargo run --release --example load     -- models/<stem>.yaml cuda
cargo run --release --example generate -- models/<stem>.yaml "<prompt>" cuda
```

`load` assembles the whole model from the manifest and encodes a prompt with it, which is the
cheapest thing that says the weights, the tokenizer and the manifest agree. It is also what
catches a `suggested:` block that does not parse, so run it after writing one.

`generate` draws, and **you have to look at the picture**. Statistics do not settle it: a
v-prediction checkpoint exported by mistake produces a full histogram of pure noise, and a picture
that is "not blank" by every measure can still be noise. Convert the PPM and open it. Ask for the
model's own subject -- every model here suggests Hatsune Miku, and whether it draws her *as Miku*,
turquoise twintails and all, is a stronger check on the text encoder than any norm: a model that
lost its conditioning still draws a girl.

Then delete the source checkpoint. Its sha256 is in the card by now, which is what makes the
export reproducible without keeping seven gigabytes around.

## 5. Publish

Stage the model's files in a directory of their own -- manifest, tokenizer, weights and the card
-- and upload that. Hard link rather than copy (`ln -f`): it costs no disk, and a seven gigabyte
model staged twice is what fills the box.

Hugging Face, in one commit, deleting what the old format left behind as it goes:

```bash
huggingface-cli upload ling0322/libwaifu-<stem> <stage-dir> . \
  --repo-type model --delete "*.waifupkg" --commit-message "..."
```

The CLI is `huggingface-cli`, not `hf`: `tools/requirements.txt` pins `huggingface-hub==0.26.2`
for the exporter, and `hf` arrived in 0.34.

`0.00B transferred` on a re-export is **not** a no-op: Xet content-defined chunking dedupes
against the previous revision, and unchanged weights transfer nothing. Always confirm against the
API rather than trusting the summary line:

```bash
curl -s "https://huggingface.co/api/models/<repo>?blobs=true" \
  | python3 -c "import json,sys;[print(f['rfilename'],f.get('size')) for f in json.load(sys.stdin)['siblings']]"
```

ModelScope mirror -- same repo name, same files. Delete **after** the upload, and check what the
repository holds afterwards rather than trusting the call:

```python
from modelscope.hub.api import HubApi
api = HubApi()
api.create_repo(repo_id="ling0322/libwaifu-<stem>", repo_type="model",
                visibility="public", license="<source license>", exist_ok=True)
api.upload_folder(repo_id="ling0322/libwaifu-<stem>", repo_type="model",
                  folder_path="<stage-dir>", commit_message="...", disable_tqdm=True)

# `delete_patterns`, not `path_patterns`: the wrong keyword raises TypeError, and a delete
# wrapped in a try/except swallows it and leaves both formats side by side.
api.delete_files(repo_id="ling0322/libwaifu-<stem>", repo_type="model",
                 delete_patterns=["*.waifupkg"], commit_message="...")
print([f["Path"] for f in api.get_model_files(model_id="ling0322/libwaifu-<stem>",
                                              recursive=True)])
```

Needs `pip install modelscope` (it does not disturb the exporter's pins); credentials live in
`~/.modelscope/credentials`. Neither hub's login prompt works without a TTY -- `getpass` raises
`EOFError` -- so ask the user to log in from a real terminal, or to pass `--token`.

**Never pipe an upload script through `tail`.** Both of these print the thing that went wrong
before the summary line, and `| tail -12` is how a swallowed `TypeError` stayed invisible through
a whole publish.

### Renaming a repository

Where a version's spelling changes, the repository follows, and the two hubs differ:

```python
HfApi().move_repo(from_id="ling0322/libwaifu-<old>", to_id="ling0322/libwaifu-<new>",
                  repo_type="model")   # leaves a redirect; the old address keeps working
```

ModelScope has no rename at all: create the new repository, upload, and delete the old one -- for
which there is no redirect, so the old address simply stops. Before deleting anything, list the
new repository and check every file is there **by name**. "The upload script said it worked" is
not the same as the model being there, and the delete is not reversible.

**Carry the source model's license, never libwaifu's MIT.** The weights are someone else's and
the fine tunes differ: base is `openrail++`, WAI is `cdla-permissive-2.0`, NoobAI is
`fair-ai-public-license-1.0-sd` which forbids commercial use. Copy `license`, `license_name`,
`license_link` and any `not-for-all-audiences` tag from the source card into the new one.

## 6. Wire it into the CLI

Only two edits in `waifu/src/cli/hub.rs` -- the picker and the `-m` usage text both build
themselves from `hub::names()`, so nothing else needs telling:

- add a `Published` entry to `CATALOG` (`name`, `repo`, `manifest`)
- add the unversioned alias to `ALIASES`
- where a name is replacing one that was already published, add the old spelling to `SUPERSEDED`,
  which answers it without listing it in `names()`

**Do not write the entry until the model is actually published.** An entry naming a manifest that
is not on the hub is a name that 404s for everyone who types it, and it is the sort of thing that
gets committed and forgotten. Publish, verify, then wire.

`manifest` must be the exact file name. Confirm every entry resolves, on both hubs -- they use
different branches, `main` against `master`, which is the easy thing to get wrong:

```bash
curl -sIL -o /dev/null -w "%{http_code}\n" "https://huggingface.co/<repo>/resolve/main/<manifest>"
curl -sIL -o /dev/null -w "%{http_code}\n" "https://modelscope.cn/models/<repo>/resolve/master/<manifest>"
```

Then `README.md`: the model table, the `waifu draw -m` example list, and a `Recent updates` line.

## 7. Test

```bash
./build/unittest
cargo test --manifest-path waifu/Cargo.toml --features cli -- --test-threads=1
```

Both flags matter -- see traps. To run the SDXL numerical tests you also need `models/`
populated; see "Test data" below.

## Traps

| symptom | cause | fix |
|---|---|---|
| configure stops on `invalid CUTLASS_ROOT` | `third_party/cutlass` is not vendored, and a CUDA build requires it | `third_party/install_cutlass.sh` |
| a CUDA build takes tens of minutes longer than expected | `install_flash_attn.sh` was run out of habit: it compiles every kernel for `sm_80`, `sm_86`, `sm_89`, `sm_90` and `sm_120` | skip it. Since `dd804f1` a build without it breaks nothing -- `stores_and_reads_a_paged_kv_cache` asks `F::paged_attention_available()` and skips itself, and `cuda/flash_attn_test.cc` is only compiled when the option is on. Wanted anyway? Pin `FLASH_ATTN_CUDA_ARCH` to this card's arch |
| every `sdxl.rs` test fails with `Aborted: out of memory` | `cargo test` runs one thread per core, each loading a 7 GB model onto one GPU | `-- --test-threads=1` |
| `hub.rs` tests never run | `cli` is a Cargo feature; plain `cargo test` compiles none of `src/cli/` | `--features cli` |
| CPU verify takes 20+ minutes | no CUDA in the build | build with CUDA |
| the export dies on a 106-byte "checkpoint" | a Civitai download without a valid token writes the JSON error body to the output file and `curl` still exits `0` | check it is not JSON (`file <name>.safetensors`) before exporting; verify the sha256 |
| a link error in `cargo test` that makes no sense | the disk filled: a debug `cargo test` builds a second dependency tree beside the release one cmake already built | `--release` on the Rust suite, and delete each checkpoint the moment its export finishes |
| every command fails with `ENOSPC ... open '/proc/self/fd/17/...'` | the disk is *completely* full, so the harness cannot create the file it writes command output to and nothing runs at all -- including the `rm` that would fix it | ask the user to free space from their own shell; there is no way out from in here |
| a model draws, but at 1024x1024 when its manifest says 832x1216 | the built you are running predates the picker reading `suggested.sizes` | rebuild; `App` carries the model's list now |
| `hf: command not found` | `huggingface-hub` is pinned at 0.26.2 for the exporter, and `hf` arrived in 0.34 | `huggingface-cli` |
| a hub login hangs or raises `EOFError` | no TTY: `getpass` cannot prompt | log in from a real terminal, or pass `--token` |
| ModelScope keeps both formats after a "successful" publish | `delete_files` takes `delete_patterns`; `path_patterns` raises `TypeError`, and a `try/except` around it hides that | pass the right keyword, and list the repo afterwards to confirm |
| Civitai swears a model does not exist | its search endpoint omits NSFW-flagged models, from `?query=` and from the author's own `?username=` listing | fetch it by id: `/api/v1/models/<id>` |

## Test data

`models/sdxl-base_test.safetensors` holds the reference tensors, published at
`ling0322/libwaifu_test_data`. **Regenerate it whenever `export_test_cases` in
`tools/sdxl_exporter.py` changes** -- it has its own history, separate from the weights:
`encoded` was added by `88fa345`, `round_trip` by `371be62`, and `waifu/tests/sdxl.rs:539` reads
`round_trip`, so an older test package fails there.

```bash
.venv/bin/python tools/sdxl_exporter.py -checkpoint sd_xl_base_1.0.safetensors \
  -output models/sdxl-base.safetensors -test_output models/sdxl-base_test.safetensors
```

The test export runs the reference diffusers pipeline on CPU, so it takes a few minutes. It
writes `<stem>_test_corpus.tsv` beside the tensors, and the tests read both.

The fixture the layer tests open is its own model, not the published one: stem `sdxl-base`, split
in two (`sdxl-base-00001-of-00002.safetensors` and its neighbour), named by `sdxl-base.yaml`. It
no longer has to be unsplit -- a manifest names every file it is made of, so the layer tests
follow it like anything else does. The published model is `sdxl-base-1.0`, which is a different
stem on purpose.

## Disk

A full pass wants ~14 GB per model (checkpoint in, weights out) and a 32 GB box holds **one** at
a time, not two. The loop is: download, verify the hash, export, hand-write `suggested:`, load,
draw and look, delete the checkpoint, upload both hubs, verify both by API, delete the staged
files, next. Nothing from a finished model stays on disk -- it is on two hubs by then, which is a
better place for it than a box that is about to run out.

**Build artifacts are the other seven gigabytes.** `cargo test` without `--release` compiles a
whole second dependency tree beside the release one CMake already built. On a box holding a model
that fills the disk, and a full disk here does not announce itself: the tool cannot even create
the file it writes command output to, so *every* command fails before it runs, and the failure
before that one is a linker error that looks like a code problem. Run the Rust suite as
`cargo test --release --manifest-path waifu/Cargo.toml --features cli`.
