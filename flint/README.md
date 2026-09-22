# Flint

Flint is libwaifu's native tensor and kernel runtime. It provides the tensor storage, device
dispatch, CPU/CUDA operators, and stable C ABI used by the Rust bindings in
[`waifu::flint`](../waifu/src/flint).

Flint is intentionally focused on inference workloads rather than being a general-purpose tensor
framework. Its operators cover the paths needed by the model runtime, including matrix
multiplication, normalization, rotary embeddings, attention, paged KV cache updates, gated DeltaNet
linear attention, and temperature/top-k/top-p sampling.

## Architecture

```text
Rust waifu runtime
	|
	v
waifu::flint safe bindings
	|
	v
Flint C API (capi.h)
	|
	v
Tensor + Operators dispatch
	|             |
	v             v
  CPU backend    CUDA backend
```

- `Tensor` owns shape/stride metadata and shared storage.
- `Operators` dispatches an operation to the backend for the tensor's device.
- The CPU backend contains portable kernels plus AVX2, AVX-512, and ARM half-precision paths.
- The CUDA backend contains custom inference kernels and optional FlashAttention/CUTLASS paths.
- The C API owns no tensor storage directly; callers receive opaque handles and destroy them with
  `fl_tensor_destroy`.
- `lutil/` contains the small utility layer used by Flint and remains a separate CMake target.

## Directory layout

```text
flint/
|-- cpu/             CPU tensors and kernels
|-- cuda/            CUDA tensors and kernels
|-- lutil/           Utility code used by the native runtime
|-- bin/             Native test and benchmark entry points
|-- tensor.{h,cc}    Tensor metadata, views, and storage
|-- operators.{h,cc} Backend operator interface and the per-device instances
|-- capi.{h,cc}      Stable C ABI for language bindings
`-- CMakeLists.txt
```

## C++ API

Initialize the operator backends before creating tensors and release them after the last tensor
operation. Every operation is asked of an `Operators` instance -- the backend of one device --
which `getOperators()` hands out; nothing works out which device to use from the tensors it was
given.

```cpp
#include "flint/operators.h"
#include "flint/tensor.h"

int main() {
  fl::initOperators();

  {
    fl::Operators *cpu = fl::getOperators(fl::Device::kCpu);

    fl::Tensor a = fl::Tensor::create<float>({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    fl::Tensor b = fl::Tensor::create<float>({2, 2}, {4.0f, 3.0f, 2.0f, 1.0f});
    fl::Tensor sum = cpu->add(a, b);
    cpu->print(sum);
  }

  fl::destroyOperators();
}
```

A tensor that lives on the card is the CUDA operators' to compute with, and a reference computed
beside it on the host is the CPU operators': a test that checks one against the other holds both
instances and says which line runs where.

The main public C++ surfaces are:

- [`tensor.h`](tensor.h): tensor construction, metadata, slicing, views, and storage access.
- [`operators.h`](operators.h): the operations themselves, one instance per device.
- [`device.h`](device.h) and [`dtype.h`](dtype.h): device and element type definitions.
- [`memory.h`](memory.h): device memory statistics.

## C and Rust APIs

[`capi.h`](capi.h) exposes opaque tensor handles and status-returning functions for language
bindings. Call `fl_init()` once before using the C API, then `fl_operators_create()` for each
device you compute on: every operation takes that handle, as the C++ side does. A failing call
returns an error code, and the thread-local details are available through
`fl_get_last_error_code()` and `fl_get_last_error_message()`.

Rust applications should use the safe wrapper in [`waifu::flint`](../waifu/src/flint), not call the C
API directly. `waifu/build.rs` links the native `build/libflint.a` produced by CMake; CMake is what
drives the Rust build (see the top-level `CMakeLists.txt`'s `waifu-cli` target), so `cmake --build`
alone builds Flint and the CLI together.

## Build

Configure Flint from the repository root. A CPU-only build is:

```bash
cmake -S . -B build -DWITH_CUDA=OFF
cmake --build build --parallel
```

For CUDA, enable the CUDA backend:

```bash
cmake -S . -B build -DWITH_CUDA=ON -DCUDA_ARCH_NATIVE=ON
cmake --build build --parallel
```

FlashAttention is opt-in, since its kernels are slow to compile. Build them once with
`./third_party/install_flash_attn.sh` and add `-DWITH_FLASH_ATTN=ON` to use them.

Important native artifacts are:

```text
build/libflint.a          Native archive linked by the waifu crate
build/flint_link_flags.txt Additional libraries Cargo must link
build/unittest            Native test executable
build/benchmark           Native benchmark executable
```

`cmake --build build` (no `--target`) builds all of the above plus the `waifu` crate and its command
line binary, since the default target set includes the `waifu-cli` custom target that invokes
`cargo build`. Use
`--target flint`, `--target unittest`, or `--target benchmark` to build just one native piece.

## Tests and benchmarks

Build and run the full native test suite:

```bash
cmake --build build --target unittest --parallel
./build/unittest
```

Catch2 tags can select a narrower area. For example:

```bash
./build/unittest "[sampling][cuda]"
```

Build and run native benchmarks with:

```bash
cmake --build build --target benchmark --parallel
./build/benchmark
```

The Rust binding tests exercise the same native library. They read the link flags CMake already
wrote out rather than rebuilding it, so re-run `cmake --build build` first if you edited a kernel:

```bash
cargo test -p waifu --test tensor --test tensor_functional
```

## Adding an operator

An operator normally crosses these layers:

1. Declare the backend interface in `operators.h` and its default unsupported implementation in
   `operators.cc`.
2. Implement the CPU and/or CUDA backend and override the method in the backend `Operators`
   subclass.
3. Add C API and `waifu::flint` functions when the operation is needed outside C++. The C function
   takes an `fl_operators_t` first; the Rust wrapper gets it from the device of the tensor it
   reads, which is what `waifu::flint::operators` is for.
4. Add focused backend tests and run `./build/unittest`.
