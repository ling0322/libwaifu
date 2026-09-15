# libwaifu

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DWITH_OPENMP=ON -DWITH_MLX=ON
cmake --build build -j$(sysctl -n hw.ncpu)
```

## Test

Before every commit, run **both** test suites and make sure they pass:

```bash
./build/unittest                # C++ tests
cargo test --manifest-path waifu/Cargo.toml   # Rust tests (needs LIBWAIFU_LIB_DIR=build or a prior cmake build)
```

`LIBWAIFU_LIB_DIR` must be an absolute path: the build script runs with `waifu/` as its
working directory, so a relative `build` will not be found.

### Anything over a minute: do not run it unless asked

Every model test is `#[ignore]`d, because it needs a real package rather than weights written by
hand. They are also the slow ones, and **a test run that takes more than a minute does not happen
unless the user asks for it.** Not "ask and wait for a yes" — do not bring it up as a question at
all. Run the fast suites, then say which slow ones were skipped and what command runs them. The
user decides, on their own clock.

That covers every model suite here. None of them finishes in a minute.

When asked, run only what the change reaches. Selection is per *binary*: the names inside a
`tests/*.rs` are bare function names with no file prefix, so `--skip sdxl` matches nothing and
`--test` is the flag that works.

```bash
# waifu/src/anima/, tools/anima_*
cargo test --release --manifest-path waifu/Cargo.toml --no-fail-fast \
    --test anima --test anima_pipeline --test anima_tokenizer -- --ignored --test-threads=1

# waifu/src/layers.rs, waifu/src/sdxl/, flint/
cargo test --release --manifest-path waifu/Cargo.toml --no-fail-fast \
    --test sdxl --test sdxl_unet --test sdxl_vae --test sdxl_text_encoder \
    --test sdxl_tokenizer --test sdxl_sampler -- --ignored --test-threads=1
```

Neither `--test-threads=1` nor `--no-fail-fast` is optional: each test puts a whole model on the
card and cargo would otherwise start one per core, and cargo stops at the first test *binary*
that fails, so one failure in `sdxl.rs` would mean the rest never run.

Never add `--features cli` to a release model-test run. Nothing under `waifu/src/cli` is reached
from `tests/`, and asking for it drags hyper and rustls through a release compile first.

One GPU, one job. A second worktree running these at the same time produces `Aborted: out of
memory`, which reads as a code failure and is not one. Check `nvidia-smi` before blaming a diff.

### Never write a test that runs longer than three minutes

Three minutes is a hard ceiling on anything written here, and raising it is not one of the
options. A test that wants longer is not slow, it is wrong: shrink the latent, cut the steps,
load a smaller package, share one model load across the cases, or stop calling it a test.

Both suites meet it now, and what decided it was the fixture rather than the kernels. Measured on
an RTX 5060 Ti with the card otherwise free:

- the three anima binaries together: **40 s** (2026-09-13). `anima.rs` holds the package in a
  `OnceCell` and reads it once for the whole binary.
- the six sdxl binaries together: **228 s** (2026-09-14), of which `sdxl.rs` is 92 s. It used to
  be 877 s in `sdxl.rs` alone: `model()` opened the 7 GB package and uploaded it to the card once
  per `#[test]`, sixteen times over. It now holds it the way `anima.rs` holds its weights, so the
  package is read once per binary. What is left is real work -- one of those tests denoises on the
  CPU in float32, and that one is most of the 92 s.

Keep it that way. A fixture that builds a model per test is the one thing here that has ever put a
suite over the ceiling, and it does it quietly, because every individual test still looks fast.

Timings are worthless when the card is shared: the same anima suite measured 323 s while another
worktree held 5.5 GB. Check `nvidia-smi` before believing a number, and before filing a slow test.
This file used to claim the whole thing was "about two and a half minutes"; it was not true.

Do not commit if either suite fails.

## Git

Committing and pushing are the human's calls, in that order.

Do not run `git commit` on your own. The user reads the diff before it becomes a commit, and a
commit you made unasked takes that reading away. Finish the work, leave it in the working tree,
and say what changed and where; wait to be asked before recording it.

Do not run `git push` — to any remote, branch, or tag — unless the user has asked for that push
in this conversation. A commit that is ready to push stays unpushed until someone says so. Name
the branch instead, so the user can push it themselves.
