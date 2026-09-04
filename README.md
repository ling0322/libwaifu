# libwaifu: your local AI waifu

[![CI](https://github.com/ling0322/libwaifu/actions/workflows/ci.yml/badge.svg)](https://github.com/ling0322/libwaifu/actions/workflows/ci.yml)

libwaifu draws your waifu on your own machine. No API key, no cloud, no one else seeing what you
asked for -- she is painted by your own GPU, as fast as the hardware allows.

Underneath is a Rust runtime over optimized C++17/CUDA kernels: SDXL's two text encoders, its
U-Net and its autoencoder, the Euler schedule that walks between them, and classifier free
guidance. Draw from the `waifu` CLI, or embed the same pipeline in your own app through the Rust
API.

## Getting a model

Run `waifu draw` with no model and it offers the ones it knows, marking what is already on disk;
pick one and it is fetched. Ask for one by name to skip the list:

```bash
$ waifu draw
$ waifu draw -m sdxl:base
$ waifu draw -m sdxl:wai
```

The models `waifu` knows by name:

| name | model | published as |
|---|---|---|
| `sdxl:base` | SDXL 1.0 base, prompted with sentences | [libwaifu-sdxl-base-1.0](https://huggingface.co/ling0322/libwaifu-sdxl-base-1.0) |
| `sdxl:wai` | WAI Illustrious v17.0, an anime fine tune prompted with danbooru tags | [libwaifu-wai-illustrious-v17](https://huggingface.co/ling0322/libwaifu-wai-illustrious-v17) |

A name without a version follows whatever the current release is, so `sdxl:wai` keeps working
when a v18 arrives. `sdxl:base:v1` and `sdxl:wai:v17` name a release and keep meaning it. `waifu
draw -h` lists what this build knows.

What is fetched lands in `~/.cache/libwaifu/models` (`%LOCALAPPDATA%\libwaifu\models` on
Windows, or wherever `WAIFU_CACHE` points), so it is downloaded once and read from disk after
that. An interrupted download resumes where it stopped rather than starting over.

### Making one yourself

Any SDXL checkpoint becomes a package with the exporter in `tools/`. Any fine tune works --
Illustrious, WAI, or anything else that ships as a single safetensors file:

```bash
python3 -m venv .venv
.venv/bin/pip install -r tools/requirements.txt
.venv/bin/python tools/sdxl_exporter.py -checkpoint /path/to/checkpoint.safetensors -output sdxl.waifupkg
```

### Models too large for one file

A package is about seven gigabytes, which is an awkward size to publish and an awkward one to
fetch: it cannot be downloaded in parallel, and a failed transfer starts over. Pass a limit and
the model is written as several packages instead:

```bash
.venv/bin/python tools/sdxl_exporter.py ... -output sdxl.waifupkg -part-size 4GB
```

A package that is already written can be split without exporting it again, which copies the
tensors byte for byte:

```bash
.venv/bin/python tools/split_package.py sdxl.waifupkg -part-size 4GB
```

Either way you get `sdxl-00001-of-00002.waifupkg` and `sdxl-00002-of-00002.waifupkg`. Point
`waifu draw` at the first and it reads the rest: it holds the configuration and names the others.
Keep them in one directory -- the first names its neighbours by file name and will not follow a
path anywhere else -- and the picture they draw is identical, to the byte, to the one the whole
package draws.

The split is between tensors, never inside one, so each part is a parameter file in its own right
and can be read and checked alone.

## Drawing pictures

An SDXL package draws rather than talks, and `waifu draw` opens a terminal for it:

```bash
$ waifu draw
$ waifu draw -m sdxl:wai
$ waifu draw -m sdxl.waifupkg
```

`-m` takes either a published name from the table above or a package of your own. Left out, the
screen opens on the list of published models and fetches whichever one is picked.

The screen holds the prompt, what to steer away from, and the four numbers a run takes: how many
steps, how hard to push away from the unprompted answer, what size, and which seed.

```text
 tab move  enter draw  esc stop or quit
```

`tab` moves between the boxes, the arrows turn whichever number is under the cursor, `enter`
starts a run and `esc` stops one where it stands. Every finished picture is written into the
current directory as `waifu-0001.png` and listed on screen -- a terminal is no place to look at
one, so the file name is what comes back.

Drawing needs `Conv2d`, which either cuDNN or CUTLASS can answer -- see the build section below.

## Kernel support matrix

| OS       |  Platform | CUDA       |  avx2  |  avx512 | asimdhp |
|----------|-----------|------------|--------|---------|---------|
| Linux    | x64       | ✅         | ✅     | ✅       |         |
| Windows  | x64       | ✅         | ✅     | ✅       |         |
| macOS    | arm64     |            |        |         | ✅      |

## Recent updates

- [2026-08-30] Pick a model on screen: `waifu draw` with no `-m` lists them and fetches one.
- [2026-08-29] WAI Illustrious v17.0 is published too, as `sdxl:wai`.
- [2026-08-29] Ask for a model by name: `waifu draw -m sdxl:base` fetches it on first use.
- [2026-08-28] Draw pictures from a terminal.
- [2026-08-28] SDXL: a prompt in, an image out.

## Rust example

The Rust API reads a package and hands back an image:

```rust
use waifu::{to_rgb8, Device, GenerationOptions, Sdxl, ZipFile};

fn main() -> Result<(), waifu::Error> {
	let package = ZipFile::open("sdxl.waifupkg")?;
	let model = Sdxl::from_package(Device::Cuda, &package)?;

	let options = GenerationOptions {
		width: 1024,
		height: 1024,
		num_steps: 30,
		guidance_scale: 5.0,
		negative_prompt: String::new(),
		seed: Some(7),
	};

	let image = model.generate("a photo of an astronaut riding a horse on mars", &options)?;

	// Three bytes a pixel, row by row, ready for whatever writes the file.
	let pixels = to_rgb8(&image)?;
	println!("{} bytes, {} by {}", pixels.len(), options.width, options.height);
	Ok(())
}
```

`Sdxl::generate_reporting` is the same run with a reporter that hears how far along it is and can
stop it between steps, which is what the `draw` command is built on.

After completing the build steps below, run the complete example with:

```bash
cargo run --release -p waifu --example generate -- \
	sdxl.waifupkg \
	"a photo of an astronaut riding a horse on mars"
```

See [waifu/examples/generate.rs](waifu/examples/generate.rs) for the complete source.

## Build

CMake drives the whole build. Configuring picks the native Flint C++/CUDA options -- which
backends to compile, where CUDA lives, what the third_party prerequisites resolve to -- and
`cmake --build` does the rest: it builds `libflint.a`, then runs `cargo build` to link it into
`waifu` and the `waifu` binary.

Requirements:

- CMake 3.22 or newer
- A C++17 compiler
- Rust and Cargo
- OpenMP, unless configured with `-DWITH_OPENMP=OFF`
- The bundled libunwind build:

```bash
(cd third_party && ./install_unwind.sh)
```

### CPU build

A CPU build draws too, which it did not until the convolution and the two normalizations were
written for it. On a 32 thread machine 512 by 512 at 20 steps takes two and a half minutes, and
the model wants 13.7 GB rather than the 6.97 GB it is on disk: x64 has no half kernels, so the
weights are widened to float32 as they are read. That is also why it is the more accurate of the
two -- float32 throughout, against a float32 reference, is 1.3e-4 where the half path is 2.1e-2.

```bash
cmake -S . -B build -DWITH_CUDA=OFF
cmake --build build --parallel
```

The command-line executable is written to:

```text
build/waifu
```

### CUDA build

Install the CUDA Toolkit first. FlashAttention is enabled by default for CUDA builds and must be
built once before configuring libwaifu:

```bash
./third_party/install_flash_attn.sh
```

Then configure and build:

```bash
cmake -S . -B build \
	-DWITH_CUDA=ON \
	-DCUDA_ARCH_NATIVE=ON
cmake --build build --parallel
```

### Conv2d, through cuDNN or CUTLASS

`Conv2d` has two implementations and needs one of them. cuDNN is preferred where it is there;
`WITH_CUTLASS=ON` answers it too, and is enough on its own -- there is nothing to download for
it, since CUTLASS is a header library already in `third_party`. Convolving on CUTLASS costs about
3% on a whole image, which is what the layout costs: CUTLASS convolves in NHWC and the rest of
the library is NCHW, so the activations are permuted in and out around the kernel. What it does
not do is a grouped convolution, which nothing here asks for and which it refuses rather than
answers wrongly.

Set `LIBWAIFU_CONV=cudnn` or `=cutlass` to pick one for a run, whichever the build has.

cuDNN is the one library the CUDA Toolkit does not carry, so it is opt in. Point `CUDNN_ROOT` at
a directory holding `include/cudnn.h`:

```bash
cmake -S . -B build \
	-DWITH_CUDA=ON \
	-DWITH_CUDNN=ON \
	-DCUDNN_ROOT=/path/to/cudnn
```

Only the headers are needed to build. The library itself is resolved by name on first use, as
`libcudnn.so` and then `libcudnn.so.9`, so a build with cuDNN still runs where there is none --
`F::conv2d` reports that it is unavailable rather than failing to load. A `pip install
nvidia-cudnn-cu12` puts a usable `CUDNN_ROOT` under
`<venv>/lib/pythonX.Y/site-packages/nvidia/cudnn`, and ships only the versioned library, so its
`lib` directory has to be on `LD_LIBRARY_PATH` at run time.

`CUDA_ARCH_NATIVE=ON` builds only for GPUs installed in the current machine. Omit it when
building an artifact intended for several GPU generations. If CMake cannot find the intended
CUDA installation, add:

```text
-DCUDAToolkit_ROOT=/path/to/cuda
```

To build CUDA support without FlashAttention, configure with `-DWITH_FLASH_ATTN=OFF` instead of
running `install_flash_attn.sh`.

### macOS

Install OpenMP before configuring the CPU build:

```bash
brew install libomp
export OpenMP_ROOT="$(brew --prefix)/opt/libomp"

cmake -S . -B build -DWITH_CUDA=OFF
cmake --build build --parallel
```

### Tests

Run the native C++/CUDA test suite:

```bash
cmake --build build --target unittest --parallel
./build/unittest
```

Run the Rust tests. These read the link flags CMake already wrote out, so they work without
re-running `cmake --build` -- just be sure `build/` reflects the latest C++ if you edited a kernel:

```bash
cargo test -p waifu --features cli
```

The ignored Rust CUDA integration tests can be run on a CUDA machine with:

```bash
cargo test -p waifu --test tensor_cuda -- --ignored
```

Some `waifu` integration tests require the model and reference packages under `models/`.

### Custom native build directory

The Rust build uses `build/` by default. Point CMake at a different build directory and it builds the
CLI the same way:

```bash
cmake -S . -B out/native -DWITH_CUDA=ON
cmake --build out/native --parallel
```

To run `cargo` directly against a build directory that isn't the default `build/`, point it there
with `LIBWAIFU_LIB_DIR`:

```bash
LIBWAIFU_LIB_DIR="$PWD/out/native" cargo build -p waifu --features cli --release
```
