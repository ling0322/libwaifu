# libwaifu

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DWITH_OPENMP=ON -DWITH_MLX=ON
cmake --build build -j$(sysctl -n hw.ncpu)
```

## Test

Before committing, both fast suites must pass:

```bash
./build/unittest
LIBWAIFU_LIB_DIR="$PWD/build" cargo test --manifest-path waifu/Cargo.toml
```

`LIBWAIFU_LIB_DIR` must be absolute because the Rust build script runs from `waifu/`.

### Model tests

Model tests are ignored and take over one minute. Run them only when explicitly requested; do not
ask whether to run them. Otherwise, report that they were skipped and provide the relevant command.
Select suites by test binary with `--test`, not by name filters such as `--skip sdxl`.

```bash
# waifu/src/anima/, tools/anima_*
cargo test --release --manifest-path waifu/Cargo.toml --no-fail-fast \
    --test anima --test anima_pipeline --test anima_tokenizer -- --ignored --test-threads=1

# waifu/src/layers.rs, waifu/src/sdxl/, flint/
cargo test --release --manifest-path waifu/Cargo.toml --no-fail-fast \
    --test sdxl --test sdxl_unet --test sdxl_vae --test sdxl_text_encoder \
    --test sdxl_tokenizer --test sdxl_sampler -- --ignored --test-threads=1
```

- Keep `--no-fail-fast` and `--test-threads=1`; each test loads a full model onto the GPU.
- Do not add `--features cli`.
- Run only one GPU job at a time. Check `nvidia-smi` before diagnosing OOM or timing failures.
- No individual test may exceed three minutes. Reuse model fixtures across tests instead of
  loading a model per test.

## Git

- Never modify, commit to, or push directly to `main`. Use a non-`main` branch and a pull request.
- Commit or push only when explicitly requested in the current conversation.
- Never push before the user has reviewed and requested the commit.
