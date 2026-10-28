# libwaifu: draw your waifu on your own GPU

[![CI](https://github.com/ling0322/libwaifu/actions/workflows/ci.yml/badge.svg)](https://github.com/ling0322/libwaifu/actions/workflows/ci.yml)

libwaifu is an image generator that runs start to finish on your own machine. A prompt in, a
picture out -- SDXL, Anima, Krea 2 or Qwen-Image, painted by the GPU you already have, as fast as the
hardware allows. No API key, no cloud, no queue, and no one else seeing what you asked for.

## Supported models

| name | model | published as |
|---|---|---|
| `sdxl:base` | SDXL 1.0 base | \[🤗 [HF](https://huggingface.co/ling0322/libwaifu-sdxl-base-1.0)\] \[[MS](https://modelscope.cn/models/ling0322/libwaifu-sdxl-base-1.0)\] |
| `sdxl:illust` | Illustrious XL v2.0-STABLE | \[🤗 [HF](https://huggingface.co/ling0322/libwaifu-illustrious-xl-v2.0)\] \[[MS](https://modelscope.cn/models/ling0322/libwaifu-illustrious-xl-v2.0)\] |
| `sdxl:wai` | WAI-illustrious-SDXL v17.0 | \[🤗 [HF](https://huggingface.co/ling0322/libwaifu-wai-illustrious-v17)\] \[[MS](https://modelscope.cn/models/ling0322/libwaifu-wai-illustrious-v17)\] |
| `sdxl:noob` | NoobAI-XL v1.1 | \[🤗 [HF](https://huggingface.co/ling0322/libwaifu-noobai-xl-v1.1)\] \[[MS](https://modelscope.cn/models/ling0322/libwaifu-noobai-xl-v1.1)\] |
| `sdxl:obsession` | One Obsession v24 | \[🤗 [HF](https://huggingface.co/ling0322/libwaifu-one-obsession-v24)\] \[[MS](https://modelscope.cn/models/ling0322/libwaifu-one-obsession-v24)\] |
| `anima:turbo` | Anima Turbo v1.1 | \[🤗 [HF](https://huggingface.co/ling0322/libwaifu-anima-turbo-v1.1)\] \[[MS](https://modelscope.cn/models/ling0322/libwaifu-anima-turbo-v1.1)\] |
| `anima:miaomiao` | MiaoMiao Harem Anima v1.6 | \[🤗 [HF](https://huggingface.co/ling0322/libwaifu-miaomiao-harem-v1.6)\] \[[MS](https://modelscope.cn/models/ling0322/libwaifu-miaomiao-harem-v1.6)\] |
| `krea2:turbo` | Krea 2 Turbo | \[🤗 [HF](https://huggingface.co/ling0322/libwaifu-krea2-turbo)\] \[[MS](https://modelscope.cn/models/ling0322/libwaifu-krea2-turbo)\] |
| `krea2:turbo-fp8` | Krea 2 Turbo, quantized to FP8 | \[🤗 [HF](https://huggingface.co/ling0322/libwaifu-krea2-turbo)\] \[[MS](https://modelscope.cn/models/ling0322/libwaifu-krea2-turbo)\] |
| `qwen-image:2.1-turbo-fp8` | Qwen-Image 2.1 Turbo (8 steps), quantized to FP8 | \[🤗 [HF](https://huggingface.co/ling0322/libwaifu-qwen-image-2.1-turbo)\] \[[MS](https://modelscope.cn/models/ling0322/libwaifu-qwen-image-2.1-turbo)\] |

## Supported voices

The voices run in the library and through its C API -- which the macOS app speaks with -- rather
than in the `waifu` terminal, which draws. From a checkout,
`cargo run --release -p waifu-tools --bin speak` says a sentence with either of them in the voice
of a recording.

| name | model | published as |
|---|---|---|
| `indextts` | IndexTTS 2.5 | \[🤗 [HF](https://huggingface.co/ling0322/libwaifu-indextts-2.5)\] \[[MS](https://modelscope.cn/models/ling0322/libwaifu-indextts-2.5)\] |
| `cosyvoice` | Fun-CosyVoice3 0.5B | \[🤗 [HF](https://huggingface.co/ling0322/libwaifu-cosyvoice-3)\] \[[MS](https://modelscope.cn/models/ling0322/libwaifu-cosyvoice-3)\] |

## Voice conversion

**Fun-CosyVoice3** turns one recording into another voice: a recording to convert, a recording of
the voice to convert it to, and a clip out -- the same package that reads text, through the C
API's voice conversion. It changes whose voice it is and keeps the words, timing and accent (see
[docs/cosyvoice3.md](docs/cosyvoice3.md#voice-conversion)).

## Layer decomposition

**See-through** takes an anime illustration apart into the layers it was drawn in -- hair, face,
eyes, clothing, up to twenty-four of them, with what each hides drawn in -- and stacks them by
depth into a Photoshop document: the `img2layers` task, in the library and through the C API's
`waifu_engine_layers_async` (see [docs/see_through.md](docs/see_through.md)).

| name | model | published as |
|---|---|---|
| `see-through` | See-through v0.0.2: LayerDiff 3D and its Marigold depth model | \[🤗 [HF](https://huggingface.co/ling0322/libwaifu-see-through)\] \[[MS](https://modelscope.cn/models/ling0322/libwaifu-see-through)\] |

## Low memory mode

`-device cuda`, `-device vulkan` and `-device metal` keep as many of the model's weights on the
GPU as fit -- the smallest first, up to 85% of the memory limit, which is the whole card (on a Mac,
the device's budget) where none is set and the "vram (GB)" slider in the Mac app -- and copy the
rest onto the GPU as each is used. A picture that needs more room than is left lets the largest
weights on the GPU go as it goes, and reads them across again when they are next used. So a
model larger than the card still draws, more slowly, rather than running out of memory. On a Mac
the weights left off the GPU stay in the model's files, each read off the disk as it is used, so a
model larger than the Mac's memory draws too.

The small weights -- norms, biases, scales -- are most of the moves and little of the bytes, so
what is left to copy across each step is the large projections.

## Run

`waifu` asks for a model in the terminal, fetches it under a progress bar if it is not on the disk
yet, and then puts up the drawing screen while the model is read onto the fastest device this
machine has: a prompt, what to steer away from, the steps, guidance, size and seed, and the device.
The terminal draws from a prompt (txt2img); drawing from a picture is the library's, for the
programs built over it. Enter on the device row lists every device this machine has; picking
another reads the model onto it in the background.

```bash
$ waifu
```

Every picture is written to `waifu-output/waifu-NNNN.png` (`-output` says where else), with the
prompt and the settings that drew it inside, in the `parameters` line other tools read. An empty
seed box is a new seed each run, and the one used is written down with the picture. Escape stops a
run between steps, and asks before leaving. The drawing screen does not choose or download a
model: to run another one, leave and run `waifu` again. The choosing can be skipped by naming the
model on the command line:

```bash
$ waifu -m sdxl:wai -device cuda
```

Fetched models go into `~/.cache/libwaifu/models` (`%LOCALAPPDATA%\libwaifu\models` on Windows).
The model screen shows where that is and where each model is, and `p` there changes it. The
choice is kept as `model_dir` in `config.toml`: next to `waifu.exe` on Windows, and in
`~/.config/libwaifu/` everywhere else. A relative `model_dir` is relative to the file, and
`WAIFU_CACHE`, when it is set, wins over both.

```toml
model_dir = 'D:\models'
```

## Recent updates

- [2026-10-28] See-through (`img2layers`): a picture taken apart into a layered, depth-sorted
  PSD, in the library and through the C API. Published as `see-through`; every reference check
  holds it as close to float32 as upstream's own bfloat16 run is, or closer.
- [2026-10-13] Drawing is back in the terminal, on the screen it had before the web page, and the
  web page is removed: `waifu` picks a task, a model and a device and then draws right there,
  writing every picture into `waifu-output`. Speech is the C API's and the macOS app's.
- [2026-10-11] Metal runs on flint's own kernels rather than MLX, which is removed: on an M5 Pro an
  SDXL image is 1.8 times as fast (9.2 s for 4 steps against 16.4 s), using the M5's neural
  accelerators where it has them, and a build runs on macOS 14. See [docs/metal.md](docs/metal.md).
- [2026-10-06] Seed-VC is removed, and libwaifu is MIT throughout again: there is no `gpl` feature
  or `-DENABLE_GPL`. Voice conversion is CosyVoice3's.
- [2026-10-04] Fun-CosyVoice3 converts voices on the speech2speech page:
  `waifu -task speech2speech -m cosyvoice`. See [docs/cosyvoice3.md](docs/cosyvoice3.md#voice-conversion).
- [2026-10-02] Fun-CosyVoice3-0.5B-2512 is published, as `cosyvoice:v3` -- the second voice, Apache 2.0.
  See [docs/cosyvoice3.md](docs/cosyvoice3.md).
- [2026-10-13] Qwen-Image 2.1 Turbo is published, as `qwen-image:2.1-turbo-fp8` -- the eight-step
  checkpoint, fp8 only. Its eight sigmas are fixed in the package, so the step count is ignored.
  `qwen-image:2.1` and `qwen-image:2.1-fp8` leave the catalog with it; a manifest already on the
  disk still loads with `-m <path>`.
- [2026-09-23] Qwen-Image 2.1 is published, as `qwen-image:2.1` and `qwen-image:2.1-fp8` -- a
  fourth architecture, Built with Qwen. Its licence is non-commercial (research and evaluation)
  only; see [docs/qwen_image.md](docs/qwen_image.md#licensing).
- [2026-09-23] IndexTTS-2.5 speaks here, as `indextts` -- the first published voice, and the
  text2speech tab's first real model.
- [2026-09-23] MiaoMiao Harem v1.6 is published, as `anima:miaomiao` -- an Anima fine tune, mirrored
  here with the author's permission for this specific conversion.
- [2026-09-18] Krea 2 Turbo draws here: a third architecture, exported from the gated release
  rather than published from this repository.
- [2026-09-15] The screen is a page in a browser rather than a screenful of terminal: txt2img and
  img2img, the models to fetch across the top, and the picture where it can actually be looked at.
- [2026-09-15] Anima Turbo v1.1 is published, as `anima:turbo` -- the first model here that is not
  an SDXL one.
- [2026-09-05] NoobAI-XL v1.1 is published, as `sdxl:noob`.
- [2026-08-30] Metal, through MLX: a macOS build draws on the GPU rather than the CPU.
- [2026-08-29] WAI Illustrious v17.0 is published too, as `sdxl:wai`.

## Supported platforms

| OS       |  Platform | CUDA       | Metal  | Vulkan |  avx2  |  avx512 | asimdhp | asimdfhm |
|----------|-----------|------------|--------|--------|--------|---------|---------|----------|
| Linux    | x64       | ✅         |        | ✅     | ✅     | ✅       |         |          |
| Windows  | x64       | ✅         |        |        | ✅     | ✅       |         |          |
| macOS    | arm64     |            | ✅     |        |        |         | ✅      | ✅        |

The GPU column a machine has is the one `waifu` picks on its own -- CUDA first, then Metal, then
Vulkan, and the CPU kernels when there is none. Metal is compiled in by default on macOS, the only
host that configures with it (`-DWITH_METAL=OFF` leaves it out). Vulkan is compiled in by default on Linux and Windows
(`-DWITH_VULKAN=OFF` leaves it out); it has only been run on Linux so far.

The two aarch64 kernels are one choice, not two: FEAT_FHM is optional in ARMv8.2, so the half
GEMM has a kernel whether or not the part has `fmlal`, and the backend is picked at startup by
which one it finds. The x64 pair works the same way, AVX-512 where the ISA is there and AVX2
otherwise.

### Devices

| `-device` | what it means |
|---|---|
| `auto` | The first accelerator this build has -- CUDA, then Metal, then Vulkan -- and the CPU where there is none. The default. |
| `cpu` | The CPU kernels. |
| `cuda` | The card. As many of the weights as fit stay on it, the smallest first, and the rest are moved onto it as they are used; a run that runs short lets weights on the card go, so a model larger than the card still draws. |
| `metal` | The GPU on Apple Silicon, through flint's own kernels -- and an M5's neural accelerators. As many of the weights as fit stay on it, the smallest first; the rest are read off the disk as they are used, so a model larger than the Mac's memory still draws. |
| `vulkan` | Any GPU with a Vulkan 1.2 driver, through flint's own kernels. Keeps the weights on it as `cuda` does. |

A build only has the devices it was configured with, so `cuda` on a `-DWITH_CUDA=OFF` build is a
device that is not there. See the matrix above and the build section below.

## Rust example

The Rust API reads a model and hands back an image:

```rust
use waifu::{to_rgb8, Device, GenerationOptions, Manifest, Sdxl};

fn main() -> Result<(), waifu::Error> {
	let manifest = Manifest::open("sdxl.yaml")?;
	let model = Sdxl::from_manifest(Device::Cuda, u64::MAX, &manifest)?; // as many weights on the card as fit

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

`Anima::from_manifest`, `Krea2::from_manifest` and `QwenImage::from_manifest` are the same call
for the other three families, and `manifest.section("model")` says which one a manifest holds, so
a reader that handles all four asks it first. What differs is the
numbers rather than the code: Anima's turbo release is distilled for few steps at no guidance and
comes out burnt at the thirty and five SDXL likes, so take a model's own `suggested:` block over
the defaults where it has one.

`Sdxl::generate_reporting` is the same run with a reporter that hears how far along it is and can
stop it between steps, which is what the `draw` command is built on.

After completing the build steps below, run the complete program with:

```bash
cargo run --release -p waifu-tools --bin generate -- \
	sdxl.yaml \
	"a photo of an astronaut riding a horse on mars"
```

It takes an optional third argument -- `cpu`, `cuda` or `metal` -- and takes the first accelerator
this build can reach when that is left out. See
[waifu-tools/src/bin/generate.rs](waifu-tools/src/bin/generate.rs) for the complete source.

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
cmake -S . -B build -DWITH_CUDA=OFF -DWITH_VULKAN=OFF
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

Nothing to install beyond Xcode. Metal is on by default: its kernels are compiled with Xcode's Metal
toolchain and end up inside the binary rather than beside it. OpenMP's runtime, which Xcode does not
ship, is LLVM's, fetched and built by the configure for the same macOS as the rest -- a dozen
seconds, once.

```bash
cmake -S . -B build -DWITH_CUDA=OFF
cmake --build build --parallel
```

`-DWITH_METAL=OFF` makes a CPU-only build. A build runs on macOS 14 and later. `-DLIBOMP_ROOT=...`
uses a `libomp` already built instead, such as Homebrew's (`$(brew --prefix libomp)`), which is
built for the macOS it was installed on and takes the binary with it. See
[docs/metal.md](docs/metal.md).

### Vulkan

On by default on Linux and Windows, beside CUDA or alone; `-DWITH_VULKAN=OFF` leaves it out. No
Vulkan SDK and no Vulkan loader is needed to build: CMake fetches the headers and builds the shader
compiler itself, which costs about half a minute the first time, and the binary opens the loader at
run time -- where there is none, the `vulkan` device is simply not available.
[flint/README.md](flint/README.md#vulkan) has the details.

### Tests

Run the native C++/CUDA test suite:

```bash
cmake --build build --target unittest --parallel
./build/unittest
```

Run the Rust tests. These read the link flags CMake already wrote out, so they work without
re-running `cmake --build` -- just be sure `build/` reflects the latest C++ if you edited a kernel.
The command line, `waifu-cli`, has both libraries linked into it statically and calls them through
their C APIs only:

```bash
LIBWAIFU_LIB_DIR="$PWD/build" cargo test -p waifu
cargo test -p waifu-modelmanager
LIBWAIFU_LIB_DIR="$PWD/build" cargo test -p waifu-cli
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

# Qwen-Image
cargo test --release -p waifu --no-fail-fast \
	--test qwen_image --test qwen_image_pipeline --test qwen_image_tokenizer \
	-- --ignored --test-threads=1
```

Neither flag is optional. `--test-threads=1` keeps one model on the card at a time, where cargo
would otherwise start one per core, and `--no-fail-fast` keeps a failure in one test binary from
stopping the ones after it. One GPU means one job -- a second checkout running these at the same time reports `Aborted: out of
memory`, which reads as a code failure and is not one.

## License

MIT (see `LICENSE`).
