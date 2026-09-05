---
name: add-sdxl-model
description: Export an SDXL checkpoint to a .waifupkg, publish it to Hugging Face and ModelScope, and wire it into the waifu CLI catalog. Use when adding a model to libwaifu's published list, re-exporting existing models after tools/sdxl_exporter.py changes, or regenerating the SDXL test package.
---

# Adding a published SDXL model

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
Pick deliberately. If there is no diffusers layout to read the config from, find out another way
before spending an hour on the export.

Also confirm the architecture is stock SDXL: `unet/config.json` should have
`cross_attention_dim: 2048`, `transformer_layers_per_block: [1, 2, 10]`, `block_out_channels:
[320, 640, 1280]`.

## 2. Build, including the third_party prerequisites

The README documents `install_unwind.sh` and `install_flash_attn.sh` but the CUDA path needs more
than it says:

```bash
(cd third_party && ./install_unwind.sh)     # always
(cd third_party && ./install_cutlass.sh)    # CUDA: required, see traps
(cd third_party && ./install_flash_attn.sh) # CUDA: slow, but skipping it breaks a test

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DWITH_OPENMP=ON \
      -DWITH_CUDA=ON -DWITH_CUTLASS=ON
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
  -output <stem>.waifupkg \
  -part-size 2GB
```

`-part-size 2GB` is the published convention. A 7.1 GB package becomes four parts
(2.02 / 2.01 / 2.01 / 1.06 GB) named `<stem>-0000N-of-00004.waifupkg`. Use the exporter's own
`-part-size` rather than `tools/split_package.py` -- one step, same result.

Naming, matching what is already published:

| thing | form | example |
|---|---|---|
| package stem | lowercase, version suffixed | `noobai-xl-v11` |
| repo (both hubs) | `libwaifu-<stem>` | `ling0322/libwaifu-noobai-xl-v11` |
| CLI name | `sdxl:<short>:<version>` | `sdxl:noob:v11` |
| CLI alias | `sdxl:<short>` | `sdxl:noob` |

## 4. Verify before uploading, not after

Loading proves the tensors parse and the VAE encoder is present; only drawing proves the model
works. Check both directions -- image-to-image is what exercises the encoder, and a package
exported before commit `88fa345` has no encoder at all.

Write a throwaway example that loads the package, runs `generate`, then `generate_from_image`,
and asserts the result is neither blank nor unchanged. Run it as
`cargo run --release --example <name> -- <first-part>.waifupkg cuda`. Delete it afterwards.

## 5. Publish

Hugging Face -- stage a directory and upload it as one commit:

```bash
hf upload ling0322/libwaifu-<stem> <stage-dir> . --commit-message "..."
```

`0.00B transferred` on a re-export is **not** a no-op: Xet content-defined chunking dedupes
against the previous revision, and unchanged weights transfer nothing. Always confirm against the
API rather than trusting the summary line:

```bash
curl -s "https://huggingface.co/api/models/<repo>?blobs=true" \
  | python3 -c "import json,sys;[print(f['rfilename'],f.get('size')) for f in json.load(sys.stdin)['siblings']]"
```

ModelScope mirror -- same repo name, same files, so the CLI needs only a different host:

```python
from modelscope.hub.api import HubApi
api = HubApi()
api.create_repo(repo_id="ling0322/libwaifu-<stem>", repo_type="model",
                visibility="public", license="<source license>", exist_ok=True)
api.upload_folder(repo_id="ling0322/libwaifu-<stem>", repo_type="model",
                  folder_path="<stage-dir>", commit_message="...", disable_tqdm=True)
```

Needs `pip install modelscope`; credentials live in `~/.modelscope/credentials`.

**Carry the source model's license, never libwaifu's MIT.** The weights are someone else's and
the fine tunes differ: base is `openrail++`, WAI is `cdla-permissive-2.0`, NoobAI is
`fair-ai-public-license-1.0-sd` which forbids commercial use. Copy `license`, `license_name`,
`license_link` and any `not-for-all-audiences` tag from the source card into the new one.

## 6. Wire it into the CLI

Only two edits in `waifu/src/cli/hub.rs` -- the picker and the `-m` usage text both build
themselves from `hub::names()`, so nothing else needs telling:

- add a `Published` entry to `CATALOG` (`name`, `repo`, `first_part`)
- add the unversioned alias to `ALIASES`

`first_part` must be the exact first file name, including `-00001-of-0000N`. Check the part count
did not change from what is already in the table. Confirm every entry resolves:

```bash
curl -sIL -o /dev/null -w "%{http_code}\n" "https://huggingface.co/<repo>/resolve/main/<first_part>"
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
| `Could not find UNWIND_LIB` | vendored libunwind not built | `third_party/install_unwind.sh` |
| `undefined reference to conv2dCutlass` | `WITH_CUDA=ON` alone does not link: `flint/cuda/conv2d.cc:45` calls it unconditionally but `flint/CMakeLists.txt:154` only compiles it under `WITH_CUTLASS` | add `-DWITH_CUTLASS=ON` |
| README says CUTLASS needs no download | it is wrong; `third_party/cutlass` is not vendored | `third_party/install_cutlass.sh` |
| `stores_and_reads_a_paged_kv_cache` SIGABRTs | built with `WITH_FLASH_ATTN=OFF`; `cuda_operators.cc:351` is a bare `NOT_IMPL()` → `abort()`, unlike `attention` beside it which falls back | build FlashAttention, or ignore that one test |
| every `sdxl.rs` test fails with `Aborted: out of memory` | `cargo test` runs one thread per core, each loading a 7 GB model onto one GPU | `-- --test-threads=1` |
| `hub.rs` tests never run | `cli` is a Cargo feature; plain `cargo test` compiles none of `src/cli/` | `--features cli` |
| CPU verify takes 20+ minutes | no CUDA in the build | build with CUDA |
| tensor "..." not found, but only in some tests | `sdxl_unet.rs:35` and the other layer tests (`sdxl_vae.rs:38`, `sdxl_text_encoder.rs:40`) read `model.bin` from **one** zip and do not follow `model_parts`; a split package only gives them part 1 | they need a whole unsplit package |

## Test data

`models/sdxl-base_test.waifupkg` holds the reference tensors, published at
`ling0322/libwaifu_test_data`. **Regenerate it whenever `export_test_cases` in
`tools/sdxl_exporter.py` changes** -- it has its own history, separate from the weights:
`encoded` was added by `88fa345`, `round_trip` by `371be62`, and `waifu/tests/sdxl.rs:539` reads
`round_trip`, so an older test package fails there.

```bash
.venv/bin/python tools/sdxl_exporter.py -checkpoint sd_xl_base_1.0.safetensors \
  -output models/sdxl-base.waifupkg -test_output models/sdxl-base_test.waifupkg
```

Export it **unsplit** -- the layer-level tests need one whole package (see traps). The test
export runs the reference diffusers pipeline on CPU, so it takes a few minutes.

## Disk

A full pass wants ~14 GB per model (checkpoint in, package out) and a 32 GB box holds about two
at once. Work one model at a time: download, export, verify, upload, delete, next. Delete each
checkpoint as soon as its export finishes -- record the sha256 in the model card first, which is
what makes it reproducible without keeping the file.
