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

Do not commit if either fails.
