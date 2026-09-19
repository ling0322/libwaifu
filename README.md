# libwaifu: draw your waifu on your own GPU

[![CI](https://github.com/ling0322/libwaifu/actions/workflows/ci.yml/badge.svg)](https://github.com/ling0322/libwaifu/actions/workflows/ci.yml)

libwaifu is an image generator that runs start to finish on your own machine. A prompt in, a
picture out -- SDXL, Anima or Krea 2, painted by the GPU you already have, as fast as the
hardware allows. No API key, no cloud, no queue, and no one else seeing what you asked for.

## Supported models

| name | model | published as |
|---|---|---|
| `sdxl:base` | SDXL 1.0 base, prompted with sentences | [libwaifu-sdxl-base-1.0](https://huggingface.co/ling0322/libwaifu-sdxl-base-1.0) |
| `sdxl:illust` | Illustrious XL v2.0-STABLE, the official release the fine tunes below descend from | [libwaifu-illustrious-xl-v2.0](https://huggingface.co/ling0322/libwaifu-illustrious-xl-v2.0) |
| `sdxl:wai` | WAI-illustrious-SDXL v17.0, an anime fine tune prompted with danbooru tags | [libwaifu-wai-illustrious-v17](https://huggingface.co/ling0322/libwaifu-wai-illustrious-v17) |
| `sdxl:noob` | NoobAI-XL v1.1, an Illustrious fine tune trained on Danbooru and e621 | [libwaifu-noobai-xl-v1.1](https://huggingface.co/ling0322/libwaifu-noobai-xl-v1.1) |
| `sdxl:obsession` | One Obsession v24, an Illustrious fine tune that draws well at few steps | [libwaifu-one-obsession-v24](https://huggingface.co/ling0322/libwaifu-one-obsession-v24) |
| `anima:turbo` | Anima Turbo v1.1, a Cosmos-Predict2 transformer rather than an SDXL model, distilled for ten steps at no guidance | [libwaifu-anima-turbo-v1.1](https://huggingface.co/ling0322/libwaifu-anima-turbo-v1.1) |
| `krea2:turbo` | Krea 2 Turbo, twelve billion parameters of single-stream MMDiT conditioned on twelve tapped layers of a Qwen3-VL encoder, distilled for eight steps at no guidance | [libwaifu-krea2-turbo](https://huggingface.co/ling0322/libwaifu-krea2-turbo) |
| `krea2:turbo-fp8` | The same weights with the matrices quantized: half the package and half the card | the same repository |

`krea2:turbo` is 33.8 GB and wants that much card; `krea2:turbo-fp8` is 17.3 GB and wants about
18. The quantized one is not free -- it costs about four times the error in the text encoder, and
its eight-step trajectory ends somewhere measurably different -- so take it when the card is the
constraint rather than by default. [docs/krea2.md](docs/krea2.md) is what the model is, how it
differs from the two families above it, and what every one of those numbers is measured against.

Krea 2 carries the [Krea 2 Community License](https://krea.ai/krea-2-licensing) rather than this
repository's MIT: fetching it is agreeing to that, commercial use has a revenue threshold, and a
deployment is required to carry content filtering. The package here is a converted copy and is
neither official nor endorsed by Krea.


## Run

`draw` is the only command, and it opens a page in a browser rather than drawing and exiting:

```bash
$ waifu draw
waifu is at http://127.0.0.1:7860
```

![The page: the kind of run down the left, the model and the settings for it in the middle, and the picture it drew on the right](docs/libwaifu-webui.webp)

The page has three tabs: txt2img, img2img, and text2speech. The third one is a page ahead of its
model -- there is no published voice for libwaifu yet, and what reads a sentence out is a stand-in
built into the binary that makes a pitched tone where each syllable goes. It is not speech and the
page says so above every setting on that tab. What is real is everything around it: the box, the
recording to sound like, the settings, the bar, the clip on the disk and the player it comes back
in. [docs/speech.md](docs/speech.md) is what is a stand-in, what is not, and the five methods a
speech model implements to take its place.

## Recent updates

- [2026-09-19] A third tab: type a sentence and get a WAV. The voice behind it is a stand-in until
  a speech model is published -- see [docs/speech.md](docs/speech.md).
- [2026-09-18] Krea 2 Turbo draws here: a third architecture, exported from the gated release
  rather than published from this repository.
- [2026-09-15] The screen is a page in a browser rather than a screenful of terminal: txt2img and
  img2img, the models to fetch across the top, and the picture where it can actually be looked at.
- [2026-09-15] Anima Turbo v1.1 is published, as `anima:turbo` -- the first model here that is not
  an SDXL one.
- [2026-09-05] NoobAI-XL v1.1 is published, as `sdxl:noob`.
- [2026-09-04] Draw from a picture rather than from noise: `waifu draw -i photo.png`.
- [2026-08-30] Metal, through MLX: a macOS build draws on the GPU rather than the CPU.
- [2026-08-30] Pick a model on screen: `waifu draw` with no `-m` lists them and fetches one.
- [2026-08-29] WAI Illustrious v17.0 is published too, as `sdxl:wai`.
- [2026-08-29] Ask for a model by name: `waifu draw -m sdxl:base` fetches it on first use.
- [2026-08-28] Draw pictures from a terminal.
- [2026-08-28] SDXL: a prompt in, an image out.

## Supported platforms

| OS       |  Platform | CUDA       | Metal  |  avx2  |  avx512 | asimdhp | asimdfhm |
|----------|-----------|------------|--------|--------|---------|---------|----------|
| Linux    | x64       | ✅         |        | ✅     | ✅       |         |          |
| Windows  | x64       | ✅         |        | ✅     | ✅       |         |          |
| macOS    | arm64     |            | ✅     |        |         | ✅      | ✅        |

The GPU column a machine has is the one `waifu` picks on its own -- CUDA first, then Metal,
and the CPU kernels when there is neither. Metal is compiled in by `-DWITH_MLX=ON` and macOS is
the only host that configures with it.

The two aarch64 kernels are one choice, not two: FEAT_FHM is optional in ARMv8.2, so the half
GEMM has a kernel whether or not the part has `fmlal`, and the backend is picked at startup by
which one it finds. The x64 pair works the same way, AVX-512 where the ISA is there and AVX2
otherwise.

### Devices

| `-device` | what it means |
|---|---|
| `auto` | The first accelerator this build has -- CUDA, then Metal -- and the CPU where there is neither. The default, and never `cuda_cpu_offload`: that one is for a card the model does not fit on, which is not a thing to decide on someone's behalf. |
| `cpu` | The CPU kernels. |
| `cuda` | The card, weights moved once and kept there. |
| `cuda_cpu_offload` | Weights stay in host memory and each is moved onto the card as it is used, so a model larger than the card still draws. Slower: the whole model crosses the bus once per step. Also spelled `cuda-cpu-offload`. |
| `metal` | The GPU on Apple Silicon, through MLX. |

A build only has the devices it was configured with, so `cuda` on a `-DWITH_CUDA=OFF` build is a
device that is not there. See the matrix above and the build section below.

## Rust example

The Rust API reads a model and hands back an image:

```rust
use waifu::{to_rgb8, Device, GenerationOptions, Manifest, Residency, Sdxl};

fn main() -> Result<(), waifu::Error> {
	let manifest = Manifest::open("sdxl.yaml")?;
	let model = Sdxl::from_manifest(Device::Cuda, Residency::Device, &manifest)?;

	let options = GenerationOptions {
		width: 1024,
		height: 1024,
		num_steps: 30,
		guidance_scale: 5.0,
		seed: Some(7),
		..Default::default()
	};

	let image = model.generate("a photo of an astronaut riding a horse on mars", &options)?;

	// Three bytes a pixel, row by row, ready for whatever writes the file.
	let pixels = to_rgb8(&image)?;
	println!("{} bytes, {} by {}", pixels.len(), options.width, options.height);
	Ok(())
}
```

`Anima::from_manifest` and `Krea2::from_manifest` are the same call for the other two families,
and `manifest.section("model")` says which one a manifest holds, so a reader that handles all
three asks it first. What differs is the
numbers rather than the code: Anima's turbo release is distilled for few steps at no guidance and
comes out burnt at the thirty and five SDXL likes, so take a model's own `suggested:` block over
the defaults where it has one.

`Sdxl::generate_reporting` is the same run with a reporter that hears how far along it is and can
stop it between steps, which is what the `draw` command is built on.

After completing the build steps below, run the complete example with:

```bash
cargo run --release -p waifu --example generate -- \
	sdxl.yaml \
	"a photo of an astronaut riding a horse on mars"
```

It takes an optional third argument -- `cpu`, `cuda` or `metal` -- and takes the first accelerator
this build can reach when that is left out. See
[waifu/examples/generate.rs](waifu/examples/generate.rs) for the complete source.

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
- On Linux, `make` and a network connection the first time: CMake downloads and builds
  libunwind itself, into the build directory. The tarball is kept in `third_party/libunwind`,
  so later build directories reuse it rather than fetching it again.

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

Install the CUDA Toolkit first. CUTLASS is a build-time prerequisite -- it is a header library,
but not a vendored one, so clone it first:

```bash
cd third_party && ./install_cutlass.sh && cd ..

cmake -S . -B build \
	-DWITH_CUDA=ON \
	-DCUDA_ARCH_NATIVE=ON
cmake --build build --parallel
```

### FlashAttention

FlashAttention is off by default: its kernels cost several minutes of `nvcc` per architecture,
and a CUDA build without them still runs attention through the block-wise fallback. To use it,
build the kernels once, then configure with `WITH_FLASH_ATTN`:

```bash
./third_party/install_flash_attn.sh
cmake -S . -B build -DWITH_CUDA=ON -DWITH_FLASH_ATTN=ON
```

`WITH_FLASH_ATTN` requires `WITH_CUDA=ON`. Paged attention, which the KV cache uses, lives
only in the FlashAttention path.

### macOS

Install OpenMP first, then configure with `-DWITH_MLX=ON` for Metal. CMake builds MLX itself, as
an `ExternalProject`, and the Metal kernels end up inside the binary rather than beside it:

```bash
brew install libomp
export OpenMP_ROOT="$(brew --prefix)/opt/libomp"

cmake -S . -B build -DWITH_CUDA=OFF -DWITH_MLX=ON
cmake --build build --parallel
```

Leave `-DWITH_MLX=ON` out for a CPU-only build.

### Tests

Run the native C++/CUDA test suite:

```bash
cmake --build build --target unittest --parallel
./build/unittest
```

Run the Rust tests. These read the link flags CMake already wrote out, so they work without
re-running `cmake --build` -- just be sure `build/` reflects the latest C++ if you edited a kernel.
`--features cli` is what compiles `waifu/src/cli`, and without it none of the command line is
tested:

```bash
cargo test -p waifu --features cli
```

The ignored Rust CUDA integration tests can be run on a CUDA machine with:

```bash
cargo test -p waifu --test tensor_cuda -- --ignored
```

The model tests are `#[ignore]`d too, and are a different kind of slow: each wants a real published
model under `models/` and puts a whole one on the card. Run a family at a time, on a machine with
both:

```bash
# SDXL
cargo test --release -p waifu --no-fail-fast \
	--test sdxl --test sdxl_unet --test sdxl_vae --test sdxl_text_encoder \
	--test sdxl_tokenizer --test sdxl_sampler -- --ignored --test-threads=1

# Anima
cargo test --release -p waifu --no-fail-fast \
	--test anima --test anima_pipeline --test anima_tokenizer -- --ignored --test-threads=1

# Krea 2
cargo test --release -p waifu --no-fail-fast \
	--test krea2 --test krea2_sampler --test krea2_pipeline --test krea2_tokenizer \
	-- --ignored --test-threads=1
```

Neither flag is optional. `--test-threads=1` keeps one model on the card at a time, where cargo
would otherwise start one per core, and `--no-fail-fast` keeps a failure in one test binary from
stopping the ones after it. Leave `--features cli` off these: nothing under `waifu/src/cli` is
reached from `tests/`, and asking for it drags hyper and rustls through a release compile first.
One GPU means one job -- a second checkout running these at the same time reports `Aborted: out of
memory`, which reads as a code failure and is not one.
