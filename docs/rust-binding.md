# The Rust binding: `waifu`

Status: designed, not implemented. It is written against the C API's ABI 4
([ffi-abi4.md](ffi-abi4.md)), and lands after it.

`bindings/rust` is a Rust crate over libwaifu that compiles none of it: it opens a built
`libwaifu.so` (`libwaifu.dylib`, `waifu.dll`) when asked to, finds the functions `waifu.h`
declares, and calls them. It has no build script and no dependencies, so it builds wherever Rust
does and goes to crates.io; the engine goes as binaries.

## The package

| | |
|---|---|
| package | `libwaifu` -- what a `Cargo.toml` names and crates.io lists |
| library | `waifu` -- what code writes: `use waifu::...` |
| directory | `bindings/rust`, a Cargo workspace of its own |

`waifu` is the engine crate's name too, package and library, and the engine's library name is
what names `libwaifu.so`, so it stays. The two never meet: the root `Cargo.toml` excludes
`bindings/rust`, and `bindings/rust/Cargo.toml` has a `[workspace]` of its own.

```toml
# Cargo.toml
[workspace]
members = ["waifu", "waifu-cli", "waifu-modelmanager", "waifu-tools"]
exclude = ["bindings/rust"]

# bindings/rust/Cargo.toml
[package]
name = "libwaifu"
version = "0.1.0"
edition = "2021"
license = "MIT"

[lib]
name = "waifu"

[workspace]

[dev-dependencies]
# For tests/layout.rs alone, which holds every struct against the engine's. Named apart, the
# engine's crate being `waifu` too.
waifu_engine = { package = "waifu", path = "../../waifu" }
```

- The engine cannot go to crates.io: it links the `libflint.a` CMake builds, with CUDA, Vulkan and
  Metal behind it and 182 MB of CUTLASS. `waifu/Cargo.toml` gets `publish = false`.
- `cargo publish` drops a dev-dependency that has a path and no version, so the published binding
  depends on nothing.
- A lib named `waifu` with a dev-dependency renamed from a package named `waifu` builds and tests
  without a warning; it was tried on two crates of that shape.

Built and tested on its own:

```bash
cd bindings/rust
cargo build
LIBWAIFU_LIB_DIR="$PWD/../../build" cargo test   # layout.rs needs the engine; library.rs opens libwaifu.so there
cargo fmt
```

`cargo fmt --all` at the root and the fast suites in `CLAUDE.md` do not reach it, being another
workspace: `CLAUDE.md` gets a line for each. `CARGO_TARGET_DIR` pointed at the root's `target`
keeps its tests from building the engine a second time.

## Using it

```rust
use waifu::{Capability, Device, Event, ImageGeneration, ImageGenerationRequest, Model, Pipeline, Support};

waifu::open("/opt/waifu/libwaifu.so")?;

let model = Model::open("noob.yaml")?;
if let Support::No(why) = model.supports(Capability::ImageGeneration) {
    return Err(why.into());
}

let pipeline = Pipeline::<ImageGeneration>::new(Device::Cuda)?;
pipeline.load_model(&model)?.await?;

let image = pipeline
    .run(&ImageGenerationRequest { prompt: "a cat".into(), ..Default::default() })?
    .await?;
```

A streaming reading, its parts as they are made:

```rust
let pipeline = Pipeline::<StreamingTextToSpeech>::new(Device::Cuda)?;
pipeline.load_model(&voice)?.await?;   // fails here on a device that cannot stream

let mut job = pipeline.run(&request)?;
while let Some(event) = job.next().await {
    match event {
        Event::Progress(progress) => show(progress.stage),
        Event::Part(audio) => play(&audio),
        Event::Done(whole) => keep(whole?),
    }
}
```

A program with no async runtime waits with `waifu::block_on`:

```rust
let image = waifu::block_on(pipeline.run(&request)?)?;
```

## The API

```rust
// -- the library ---------------------------------------------------------------------------
pub fn open(path: impl AsRef<Path>) -> Result<()>;   // once a process; anything before it is ErrorCode::Library
pub const ABI_VERSION: u32 = 4;
pub fn version() -> Result<&'static str>;
pub fn revision() -> Result<Option<&'static str>>;

pub mod log {
    pub fn set_callback(f: impl Fn(&LogEvent) + Send + Sync + 'static);  // any thread; must not call in
    pub fn clear_callback();
    pub fn set_level(level: LogLevel);
}

// -- devices -------------------------------------------------------------------------------
pub enum Device { Auto, Cpu, Metal, Cuda, Vulkan }
impl Device {
    pub fn is_available(self) -> bool;
    pub fn memory_budget(self) -> Option<u64>;
    pub fn memory_held(self) -> Option<u64>;
    pub fn memory_available(self) -> Option<u64>;
    pub fn memory_limit(self) -> Option<u64>;
    pub fn set_memory_limit(self, bytes: u64) -> Result<()>;
}

// -- models --------------------------------------------------------------------------------
pub struct Model;                     // a WaifuModel, freed on drop
impl Model {
    pub fn open(manifest_path: impl AsRef<Path>) -> Result<Model>;
    pub fn supports(&self, capability: Capability) -> Support;
    pub fn size_on_disk(&self) -> Option<u64>;
    pub fn sample_rate(&self) -> Option<u32>;
    pub fn image_generation_width_height_multiple_of(&self) -> Option<u32>;
    pub fn image_generation_sampler(&self) -> Option<String>;
    pub fn image_generation_max_prompt_images(&self) -> Option<u32>;
    pub fn layer_decomposition_resolution_multiple_of(&self) -> Option<u32>;
    pub fn speech_to_text_languages(&self) -> Vec<String>;   // the comma-separated property, split
}

pub enum Capability {                 // WAIFU_MODEL_SUPPORTS_*
    ImageGeneration, TextToSpeech, StreamingTextToSpeech, SpeechToText, VoiceConversion,
    LayerDecomposition,
    ImageGenerationFromStartImage, ImageGenerationWithControlImage, ImageGenerationWithPromptImages,
    TextToSpeechLikeReference, TextToSpeechWithoutReference, TextToSpeechWithStyle,
    VoiceConversionOfStyle,
}
pub enum Support { Yes, No(String) }  // the C API's 1, or 0 and the string property that says why

// -- pipelines -----------------------------------------------------------------------------
pub trait Task {                      // what a pipeline is made for
    const TYPE: PipelineType;         // WAIFU_PIPELINE_*
    type Request;
    type Output;
    type Part;                        // what a streaming task hands over; Infallible for the rest
}
pub struct ImageGeneration;           // ImageGenerationRequest       -> Image
pub struct TextToSpeech;              // TextToSpeechRequest          -> Audio
pub struct StreamingTextToSpeech;     // StreamingTextToSpeechRequest -> Audio, parts of Audio
pub struct SpeechToText;              // SpeechToTextRequest          -> Transcript
pub struct VoiceConversion;           // VoiceConversionRequest       -> Audio
pub struct LayerDecomposition;        // LayerDecompositionRequest    -> DecomposedImage

#[derive(Clone)]
pub struct Pipeline<T: Task>;         // a WaifuPipeline; freed once it and every Job of it are dropped
impl<T: Task> Pipeline<T> {
    pub fn new(device: Device) -> Result<Self>;
    pub fn device(&self) -> Device;
    pub fn load_model(&self, model: &Model) -> Result<Job<()>>;   // Err at the call for a model the task is not supported by
    pub fn unload_model(&self) -> Result<Job<()>>;
    pub fn run(&self, request: &T::Request) -> Result<Job<T::Output, T::Part>>;
    pub fn cancel_all(&self);
}

// -- jobs ----------------------------------------------------------------------------------
pub struct Job<O, P = Infallible>;    // dropped: cancelled
impl<O, P> Job<O, P> {
    pub async fn next(&mut self) -> Option<Event<O, P>>;   // None after Done
    pub fn cancel(&self);
}
impl<O, P> IntoFuture for Job<O, P> { type Output = Result<O>; }   // .await: progress and parts skipped

pub enum Event<O, P = Infallible> {
    Progress(Progress),
    Part(P),                          // streaming text to speech only
    Done(Result<O>),                  // always the last
}

pub struct Progress {
    pub stage: Stage,                 // WAIFU_STAGE_*; Stage::Other(i32) for one this crate does not know
    pub fraction: Option<f64>,
    pub done: u64,
    pub total: u64,
    pub seconds: f64,
}

pub fn block_on<F: Future>(future: F) -> F::Output;   // for a program with no runtime; never inside async code

// -- errors --------------------------------------------------------------------------------
pub struct Error { pub code: ErrorCode, pub message: String }
pub enum ErrorCode {
    Library,                          // not opened, cannot be opened, another ABI, a function missing
    Cancelled, InvalidArgument, UnknownModel, Model, NotLoaded, Internal,
    Other(i32),
}
pub type Result<T, E = Error> = std::result::Result<T, E>;
```

Requests are the C API's as owned Rust values, each with a `Default`:
`ImageGenerationRequest` (with `start_image: Option<StartImage>`, `control_image:
Option<ControlImage>` and `prompt_images: Vec<PromptImage>`), `TextToSpeechRequest`,
`StreamingTextToSpeechRequest` (no `speed`), `SpeechToTextRequest`, `VoiceConversionRequest`,
`LayerDecompositionRequest`. Results are owned too, copied out of the callback: `Image`,
`RgbaImage`, `Audio`, `Transcript { text, language }`, `DecomposedImage { canvas, layers }`, each
`Layer` with its `LayerType`, `depth` and `stratified_from: Option<usize>`.

The C API's other enums come across as Rust enums with an `Other(i32)` for a number a newer
library has and this crate does not: `PipelineType`, `Stage`, `LayerType`, `LogLevel`.
`LogEvent<'a>` borrows its strings for the length of the callback.

`Pipeline<T>` is `Send` and `Sync`, and a `Job` is `Send`, so either can be handed to another
thread or spawned on a runtime's.

`waifu::sys` keeps the `#[repr(C)]` mirror of `waifu.h` and the table of its functions, for
whatever the crate does not wrap.

## Why

**Opened, not linked.** A crate that links the engine compiles flint, and with it CUDA, Vulkan or
Metal, before anything else builds. Opened at run time, the binding builds anywhere, and a program
using it runs on whichever build of libwaifu the machine has.

**Opened once.** The library keeps threads, thread-locals and device contexts that are not built
to be unloaded, so it never is: `waifu::open` is once a process, and nothing else takes a library
handle to carry around.

**Async, and `block_on`.** The C API hands a job's end to a callback; a future is that, and works
under any runtime, since only the standard library's `Waker` is involved. One set of methods, and
a program with no runtime blocks on them, rather than a second set that blocks.

**`Pipeline<T>`.** The C API has one pipeline type and refuses another task's run at the call;
the binding puts the task in the type, so another task's run does not compile, and `run` is one
method whose request and result the task decides.

**One `next()`.** Progress, a streaming reading's parts and the end come in the order they
happen, through one call. For a task that does not stream, `Part` is `Infallible`, and the arm
never matches.

## Not decided

- Whether the task marker types also get aliases, `type ImageGenerationPipeline =
  Pipeline<ImageGeneration>`.
- What a query does before `waifu::open`. `Device::memory_budget()` and its kind return an
  `Option`, and `None` there cannot say "not opened" apart from "the processor has none".
  Proposed: they panic, as forgetting `open` is a bug and not a state to handle; the alternative
  is `Result<Option<u64>>` on each.
- The `ErrorCode` of a refused call. A refusal at the call (`Model::open`, `Pipeline::new`,
  `run`) reaches the binding as a sentence only, so it is `InvalidArgument` whatever it was, until
  the C API gives a code with it (`waifu_last_error_code`, under "Not decided" in
  [ffi-abi4.md](ffi-abi4.md)).
