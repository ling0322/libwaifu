# Flint

Flint is libwaifu's native tensor and kernel runtime. It provides the tensor storage, device
dispatch, CPU, CUDA, Metal and Vulkan operators, and stable C ABI used by the Rust bindings in
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
	|             |              |               |
	v             v              v               v
  CPU backend    CUDA backend   Metal backend   Vulkan backend
```

- `Tensor` owns shape/stride metadata and shared storage.
- `Operators` dispatches an operation to the backend for the tensor's device.
- The CPU backend contains portable kernels plus AVX2, AVX-512, and ARM half-precision paths.
- The CUDA backend contains custom inference kernels and optional FlashAttention/CUTLASS paths.
- The Metal backend runs on MLX.
- The Vulkan backend has GLSL compute kernels of its own, and runs on any GPU with a Vulkan 1.2
  driver. See [Vulkan](#vulkan).
- The C API owns no tensor storage directly; callers receive opaque handles and destroy them with
  `fl_tensor_destroy`.
- `lutil/` contains the small utility layer used by Flint and remains a separate CMake target.

## Directory layout

```text
flint/
|-- cpu/             CPU tensors and kernels
|-- cuda/            CUDA tensors and kernels
|-- metal/           Metal tensors and operators, through MLX
|-- vulkan/          Vulkan tensors and kernels; vulkan/shaders/ holds the GLSL
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

Vulkan is built by default on Linux and Windows, alone or beside CUDA; `-DWITH_VULKAN=OFF` leaves
it out. See [Vulkan](#vulkan).

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

## Vulkan

On by default on Linux and Windows, and off on macOS, where Metal is the GPU backend:

```bash
cmake -S . -B build                      # Linux or Windows: Vulkan included
cmake -S . -B build -DWITH_VULKAN=OFF    # without it
```

The build needs no Vulkan SDK, and no Vulkan loader either. Configuring fetches Vulkan-Headers, volk and VulkanMemoryAllocator
into `third_party/vulkan/`, and the build compiles glslang's `glslang` there too, unless a
`glslangValidator` is already on the `PATH`. The GLSL in `vulkan/shaders/` is compiled to SPIR-V at
build time and embedded in the library. `vulkan/shaders/shaders.cmake` lists the kernel variants,
one for each element type a kernel is built for.

Nothing links against a Vulkan loader. volk opens `libvulkan` when the operators are created, so
a Vulkan build still runs on a machine that has no loader, and there `fl_is_device_available`
reports the device as unavailable.

**What a device needs:** Vulkan 1.2 with buffer device addresses, 8- and 16-bit storage and 64-bit
integers in shaders. Kernels reach tensors through buffer device addresses passed as push
constants, so there are no descriptor sets. On devices that offer `VK_KHR_cooperative_matrix`
(16 x 16 x 16, half precision into float, subgroups of 32), half precision matmul and conv2d run on
the tensor cores. Every other device runs the same products on plain shader arithmetic.

**Operators:** the ones the image models use. That is the elementwise family, matmul, conv1d and
conv2d, the norms, softmax, lookup, rotary embedding, upsampling, the GLUs, the reductions,
rand/randn (the same Philox stream the CUDA operators draw, so a seed gives the same noise), cast,
copy and the memory statistics. Attention is the base class's composition of matmul and softmax.
The rest (paged attention, the KV cache, DeltaNet, sampling, fp8) is not implemented and says so.

**Environment variables:**

| Variable | Effect |
|---|---|
| `FLINT_VULKAN_DEVICE` | Pick the device by index or by part of its name. Needed to run on a software implementation such as lavapipe, which is never picked on its own. |
| `FLINT_VULKAN_COOPERATIVE_MATRIX=0` | Do not use cooperative matrices, to compare against the portable kernels. |
| `FLINT_VULKAN_PROFILE=1` | Time every command on the device, and print the time spent in each kernel when the process exits. |
| `FLINT_VULKAN_VALIDATION=1` | Enable the Khronos validation layer, where it is installed. |

**Tests:** `./build/unittest "[vulkan]"` checks every operator against the CPU. The hidden
`./build/unittest "[vulkan-benchmark]"` reports the throughput of matmul and conv2d on SDXL's
shapes.

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
