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
# layout.rs builds against the engine (LIBWAIFU_LIB_DIR, as the engine's own tests);
# library.rs loads the library it is pointed at (LIBWAIFU_PATH).
LIBWAIFU_LIB_DIR="$PWD/../../build" LIBWAIFU_PATH="$PWD/../../build/libwaifu.so" cargo test
cargo fmt
```

`cargo fmt --all` at the root and the fast suites in `CLAUDE.md` do not reach it, being another
workspace: `CLAUDE.md` gets a line for each. `CARGO_TARGET_DIR` pointed at the root's `target`
keeps its tests from building the engine a second time.

## Using it

```rust
use waifu::{Capability, Device, Event, ImageGeneration, ImageGenerationRequest, Model, Pipeline, Support};

waifu::init()?;   // finds libwaifu: see "Finding the library"

let model = Model::open("noob.yaml")?;
match model.supports(Capability::ImageGeneration) {
    Support::Yes => {}
    Support::No(why) => return Err(why.into()),
    Support::Unknown => return Err("this libwaifu is too old to say".into()),
}

let pipeline = Pipeline::<ImageGeneration>::new(Device::Cuda)?;
pipeline.load_model(&model)?.await?;

let image = pipeline
    .run(&ImageGenerationRequest { steps: 28, ..ImageGenerationRequest::new("a cat") })?
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

## Finding the library

`waifu::init()` loads the first of these that exists:

1. the file `LIBWAIFU_PATH` names, for a test, a developer, or a place the others would not find;
2. beside the running executable -- and on macOS also in `../Frameworks`, an app bundle's place
   -- which is where an app or a tool that ships the library puts it;
3. the platform loader's own search, handed the file's name alone (`libwaifu.so`,
   `libwaifu.dylib`, `waifu.dll`): what is installed, `LD_LIBRARY_PATH`, `PATH`.

It never looks in the current directory, which on Windows is how a planted DLL gets loaded. Where
it finds nothing, its error names every place it tried. `waifu::init_from_path(library_path)`
loads the file it is given and looks nowhere else.

Either may be called again: with the library already loaded, again or by another path to the same
file, it is `Ok` and does nothing; asked for a different file, it is `Err(ErrorCode::Library)`,
since one process has one libwaifu. So code that is not sure whether anything has initialized it
calls `init` rather than checking first -- checking and then initializing would race with
another thread doing the same, where `init` is one call behind a lock.

`waifu::is_initialized()` says whether a library is loaded, and `waifu::library_path()` which file
it is: the one `init` loaded, its full path, found where the loader had it (`dladdr`,
`GetModuleFileNameW`) when the loader did the searching. Neither reaches the library, so neither
returns a `Result`.

`LIBWAIFU_PATH` is a file at run time; `LIBWAIFU_LIB_DIR` stays what it is, the directory the
engine's build reads CMake's output from.

## The API

```rust
// -- the library ---------------------------------------------------------------------------
// Every function that reaches the library returns a Result: before `init`, Err(ErrorCode::Library).
pub fn init() -> Result<()>;                                           // finds libwaifu and loads it
pub fn init_from_path(library_path: impl AsRef<Path>) -> Result<()>;  // loads this file, and looks nowhere else
pub fn is_initialized() -> bool;                                       // whether init has loaded a library
pub fn library_path() -> Option<&'static Path>;                        // the file loaded; None before init
pub const ABI_VERSION: u32 = 4;
pub fn version() -> Result<&'static str>;
pub fn revision() -> Result<Option<&'static str>>;

pub mod log {
    pub fn set_callback(f: impl Fn(&LogEvent) + Send + Sync + 'static) -> Result<()>;  // any thread; must not call in
    pub fn clear_callback() -> Result<()>;
    pub fn set_level(level: LogLevel) -> Result<()>;
}

// -- devices -------------------------------------------------------------------------------
pub enum Device { Auto, Cpu, Metal, Cuda, Vulkan }
impl Device {
    pub fn is_available(self) -> Result<bool>;
    pub fn memory_budget(self) -> Result<Option<u64>>;
    pub fn memory_held(self) -> Result<Option<u64>>;
    pub fn memory_available(self) -> Result<Option<u64>>;
    pub fn memory_limit(self) -> Result<Option<u64>>;
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
pub enum Support {
    Yes,                              // the C API's 1
    No(String),                       // 0, and the string property of the same number: why
    Unknown,                          // absent: a capability this library is too old to know
}

// -- pipelines -----------------------------------------------------------------------------
// Sealed: implemented for the six tasks below and nothing else, so that a request, a result and
// a pipeline type cannot be paired by anyone but this crate.
pub trait Task: sealed::Sealed {      // what a pipeline is made for
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
    Library,                          // not initialized, not found, another ABI, a function missing
    Cancelled, InvalidArgument, UnknownModel, Model, NotLoaded, Internal, DeviceUnavailable,
    Other(i32),                       // a refused call's waifu_last_error_code or a job's status code alike
}
pub type Result<T, E = Error> = std::result::Result<T, E>;
```

Requests are the C API's as owned Rust values, with public fields. **No request has a
`Default`**: each is made by a `new` that takes what it cannot run without, and fills the rest
with values a run takes -- `ImageGenerationRequest::new(prompt)` (1024 x 1024, 20 steps),
`TextToSpeechRequest::new(text)`, `StreamingTextToSpeechRequest::new(text)` (no `speed`),
`SpeechToTextRequest::new(audio)`, `VoiceConversionRequest::new(source, reference)`,
`LayerDecompositionRequest::new(image)` -- and changed after with struct update syntax.
`ImageGenerationRequest` has `start_image: Option<StartImage>`, `control_image:
Option<ControlImage>` and `prompt_images: Vec<PromptImage>`. A model's own sizes and steps are
the caller's to read, not the request's to guess.

Results are owned too, copied out of the callback: `Image`, `RgbaImage`, `Audio`,
`Transcript { text, language }`, `DecomposedImage { canvas, layers }`. Each `Layer` has its
`LayerType`, `depth` and `stratified_from: Option<usize>`, and `DecomposedImage::stack()` and
`stratified_stack()` give the two stacks as indices into `layers`, by the C API's rule
([ffi-abi4.md](ffi-abi4.md#stratification)), so no caller writes the comparison.

The C API's other enums come across as Rust enums with an `Other(i32)` for a number a newer
library has and this crate does not: `PipelineType`, `Stage`, `LayerType`, `LogLevel`.
`LogEvent<'a>` borrows its strings for the length of the callback.

`Pipeline<T>` is `Send` and `Sync`, and a `Job` is `Send`, so either can be handed to another
thread or spawned on a runtime's.

`waifu::sys` keeps the mirror of `waifu.h` and the table of its functions, for whatever the crate
does not wrap. **Every C enum in it is a `c_int`**, never a Rust `enum`: a number a newer library
returns that a Rust enum has no variant for would be undefined behaviour before anything could
map it to `Other`. The safe types convert from the integer.

## Jobs, callbacks and memory

**A job's state outlives its `Job`.** Each run hands the library a box: the sending end of the
job's channel. The library calls back into it until `on_complete`, which takes the box back and
frees it -- a cancelled job and one that was still waiting included, since every job given an id
gets exactly one `on_complete`. Dropping a `Job` asks for a cancel and drops the receiving end;
the box lives on until the library is done with it. `Pipeline::drop`, for the last handle, calls
`waifu_pipeline_free`, which returns only once every job has had its `on_complete`.

**No caller code runs on a pipeline's thread.** The callbacks the crate hands the library only
copy what they are given and send it down the job's channel; nothing of the caller's -- no
closure, no `Pipeline`, no `Job` -- is reached from them. So a pipeline is never freed from its
own thread, and freeing it cannot wait on itself. The log callback is the one caller closure the
library calls, from any thread; it must not call into the crate.

**What waits for `next()`.** The library's callbacks must return at once, so a job keeps what
they hand it until the caller asks:

- progress is kept as the latest one only: a `Progress` replaces the one not yet read;
- a streaming reading's parts are all kept, in order -- they are what was asked for, and together
  are no more than the whole reading, which `Done` hands over anyway;
- a job that is awaited (`IntoFuture`) rather than walked keeps neither: only its end.

A callback never blocks on a slow reader.

## Why

**Opened, not linked.** A crate that links the engine compiles flint, and with it CUDA, Vulkan or
Metal, before anything else builds. Opened at run time, the binding builds anywhere, and a program
using it runs on whichever build of libwaifu the machine has.

**Initialized once.** The library keeps threads, thread-locals and device contexts that are not
built to be unloaded, so it never is: it is loaded once a process, by `init`, and nothing else
takes a library handle to carry around. It is `init`, not `open`: nothing is closed after it, and
`Model::open` opens a file. What reaches the library before it is an `Err(ErrorCode::Library)` --
every such function returns a `Result` -- and never a panic, nor a `None` that would look like an
answer.

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
