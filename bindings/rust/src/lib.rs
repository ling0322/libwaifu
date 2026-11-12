// The MIT License (MIT)
//
// Copyright (c) 2026 Xiaoyang Chen
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software
// and associated documentation files (the "Software"), to deal in the Software without
// restriction, including without limitation the rights to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all copies or
// substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
// BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
// DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

//! libwaifu from Rust, opened at run time rather than linked: this crate compiles none of the
//! engine. [`init`] finds a built `libwaifu.so` (`libwaifu.dylib`, `waifu.dll`) and loads it; the
//! rest calls the functions `waifu.h` declares, ABI 6. docs/rust-binding.md is the design.
//!
//! ```no_run
//! # async fn draw() -> waifu::Result<()> {
//! use waifu::{Capability, Device, ImageGeneration, ImageGenerationRequest, Model, Pipeline, Support};
//!
//! waifu::init()?;
//! let model = Model::open("noob.yaml")?;
//! if let Support::No(why) = model.supports(Capability::ImageGeneration) {
//!     panic!("{why}");
//! }
//! let pipeline = Pipeline::<ImageGeneration>::new(Device::Cuda)?;
//! pipeline.load_model(&model)?.await?;
//! let image = pipeline
//!     .run(&ImageGenerationRequest { steps: 28, ..ImageGenerationRequest::new("a cat") })?
//!     .await?;
//! # let _ = image;
//! # Ok(())
//! # }
//! ```

use std::ffi::{c_char, CStr};
use std::fmt;
use std::future::{Future, IntoFuture};
use std::path::Path;
use std::pin::pin;
use std::sync::Arc;
use std::task::{Context, Poll, Wake};

mod library;
pub mod log;
mod pipeline;
pub mod sys;

pub use library::{is_initialized, library_path};
pub use log::{LogEvent, LogLevel};
pub use pipeline::*;

// -- the library -----------------------------------------------------------------------------

/// The ABI this crate speaks, `major << 16 | minor`: a library of another major, or of an older
/// minor, is not loaded.
pub const ABI_VERSION: u32 = sys::WAIFU_ABI_VERSION;

/// Finds libwaifu and loads it: the file `LIBWAIFU_PATH` names; beside the running executable --
/// and on macOS in `../Frameworks` -- and then the platform loader's own search for the file's
/// name. Never the current directory. With a library loaded already, `Ok` and nothing else.
pub fn init() -> Result<()> {
    library::init()
}

/// Loads the library at `library_path`, and looks nowhere else. With that file loaded already,
/// `Ok`; with another, `Err(ErrorCode::Library)`, since one process has one libwaifu.
pub fn init_from_path(library_path: impl AsRef<Path>) -> Result<()> {
    library::init_from_path(library_path.as_ref())
}

/// "0.1.0": the library's version.
pub fn version() -> Result<&'static str> {
    let f = sys::functions()?;
    Ok(unsafe { static_text((f.waifu_version)()) }.unwrap_or_default())
}

/// The full commit the library was built from, or None for a build made without git.
pub fn revision() -> Result<Option<&'static str>> {
    let f = sys::functions()?;
    Ok(unsafe { static_text((f.waifu_revision)()) })
}

/// A string the library keeps forever.
unsafe fn static_text(pointer: *const c_char) -> Option<&'static str> {
    (!pointer.is_null())
        .then(|| CStr::from_ptr(pointer).to_str().ok())
        .flatten()
}

/// A string the library hands over for now, copied.
pub(crate) unsafe fn text(pointer: *const c_char) -> Option<String> {
    (!pointer.is_null()).then(|| CStr::from_ptr(pointer).to_string_lossy().into_owned())
}

// -- errors ----------------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Error {
    pub code: ErrorCode,
    pub message: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ErrorCode {
    /// Not initialized, not found, another ABI, a function missing.
    Library,
    Cancelled,
    InvalidArgument,
    UnknownModel,
    Model,
    NotLoaded,
    Internal,
    DeviceUnavailable,
    /// A request attribute the library does not know: it is older than this crate expects.
    UnknownAttribute,
    /// A refused call's `waifu_last_error_code` or a job's status code this crate does not know.
    Other(i32),
}

impl ErrorCode {
    /// The code a status of the library's names; never `Library`, and `Other(0)` for OK.
    pub(crate) fn from_status(code: sys::WaifuStatusCode) -> ErrorCode {
        match code {
            sys::WAIFU_CANCELLED => ErrorCode::Cancelled,
            sys::WAIFU_ERR_INVALID_ARGUMENT => ErrorCode::InvalidArgument,
            sys::WAIFU_ERR_UNKNOWN_MODEL => ErrorCode::UnknownModel,
            sys::WAIFU_ERR_MODEL => ErrorCode::Model,
            sys::WAIFU_ERR_NOT_LOADED => ErrorCode::NotLoaded,
            sys::WAIFU_ERR_INTERNAL => ErrorCode::Internal,
            sys::WAIFU_ERR_DEVICE_UNAVAILABLE => ErrorCode::DeviceUnavailable,
            sys::WAIFU_ERR_UNKNOWN_ATTRIBUTE => ErrorCode::UnknownAttribute,
            other => ErrorCode::Other(other),
        }
    }
}

impl Error {
    /// This thread's last refused call, as the library says it.
    pub(crate) fn refused(f: &sys::Functions) -> Error {
        let message = unsafe { text((f.waifu_last_error)()) }.unwrap_or_default();
        Error {
            code: ErrorCode::from_status(unsafe { (f.waifu_last_error_code)() }),
            message,
        }
    }

    pub(crate) fn invalid(message: impl Into<String>) -> Error {
        Error {
            code: ErrorCode::InvalidArgument,
            message: message.into(),
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.message.is_empty() {
            true => write!(f, "{:?}", self.code),
            false => f.write_str(&self.message),
        }
    }
}

impl std::error::Error for Error {}

pub type Result<T, E = Error> = std::result::Result<T, E>;

// -- devices ---------------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Device {
    Auto,
    Cpu,
    Metal,
    Cuda,
    Vulkan,
}

impl Device {
    pub(crate) fn raw(self) -> sys::WaifuDevice {
        match self {
            Device::Auto => sys::WAIFU_DEVICE_AUTO,
            Device::Cpu => sys::WAIFU_DEVICE_CPU,
            Device::Metal => sys::WAIFU_DEVICE_METAL,
            Device::Cuda => sys::WAIFU_DEVICE_CUDA,
            Device::Vulkan => sys::WAIFU_DEVICE_VULKAN,
        }
    }

    /// The device a number names; `Auto` for one this crate does not know, which a library of its
    /// ABI never hands back for a pipeline.
    pub(crate) fn from_raw(device: sys::WaifuDevice) -> Device {
        match device {
            sys::WAIFU_DEVICE_CPU => Device::Cpu,
            sys::WAIFU_DEVICE_METAL => Device::Metal,
            sys::WAIFU_DEVICE_CUDA => Device::Cuda,
            sys::WAIFU_DEVICE_VULKAN => Device::Vulkan,
            _ => Device::Auto,
        }
    }

    pub fn is_available(self) -> Result<bool> {
        let f = sys::functions()?;
        Ok(unsafe { (f.waifu_device_is_available)(self.raw()) })
    }

    fn memory(self, attribute: sys::WaifuDeviceAttribute) -> Result<Option<u64>> {
        let f = sys::functions()?;
        let mut value = 0i64;
        let there =
            unsafe { (f.waifu_device_get_int_attribute)(self.raw(), attribute, &mut value) };
        Ok(there.then(|| value.max(0) as u64))
    }

    /// The most this process may hold there: the memory limit, or the device's memory. None for
    /// the processor and a device that cannot be used. Asking CUDA builds a context on the card.
    pub fn memory_budget(self) -> Result<Option<u64>> {
        self.memory(sys::WAIFU_DEVICE_MEMORY_BUDGET)
    }

    /// What its allocator holds now, what it keeps for reuse included.
    pub fn memory_held(self) -> Result<Option<u64>> {
        self.memory(sys::WAIFU_DEVICE_MEMORY_HELD)
    }

    /// The budget less what is held, never below 0.
    pub fn memory_available(self) -> Result<Option<u64>> {
        self.memory(sys::WAIFU_DEVICE_MEMORY_AVAILABLE)
    }

    /// What [`set_memory_limit`](Device::set_memory_limit) set; None where nothing did.
    pub fn memory_limit(self) -> Result<Option<u64>> {
        self.memory(sys::WAIFU_DEVICE_MEMORY_LIMIT)
    }

    /// For the whole process, shared by every pipeline on the device: set it before a model is
    /// loaded. 0 for no limit but the device's. The processor has none, and is refused.
    pub fn set_memory_limit(self, bytes: u64) -> Result<()> {
        let f = sys::functions()?;
        match unsafe { (f.waifu_device_set_memory_limit)(self.raw(), bytes) } {
            true => Ok(()),
            false => Err(Error::refused(f)),
        }
    }
}

// -- models ----------------------------------------------------------------------------------

/// A model's manifest, read -- a few kilobytes, none of the weights -- and what can be asked of it.
pub struct Model {
    handle: *mut sys::WaifuModel,
    functions: &'static sys::Functions,
}

// What a model is answered from is read when it is opened, and never changes; any thread may ask.
unsafe impl Send for Model {}
unsafe impl Sync for Model {}

/// What a model can be asked for: `WAIFU_MODEL_SUPPORTS_*`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Capability {
    ImageGeneration,
    TextToSpeech,
    StreamingTextToSpeech,
    SpeechToText,
    VoiceConversion,
    LayerDecomposition,
    ImageGenerationFromStartImage,
    ImageGenerationWithControlImage,
    ImageGenerationWithPromptImages,
    TextToSpeechLikeReference,
    TextToSpeechWithoutReference,
    /// A reference without its transcript: CosyVoice3 does not, unless a style is given.
    TextToSpeechWithoutReferenceTranscript,
    TextToSpeechWithStyle,
    VoiceConversionOfStyle,
}

impl Capability {
    fn raw(self) -> sys::WaifuModelAttribute {
        match self {
            Capability::ImageGeneration => sys::WAIFU_MODEL_SUPPORTS_IMAGE_GENERATION,
            Capability::TextToSpeech => sys::WAIFU_MODEL_SUPPORTS_TEXT_TO_SPEECH,
            Capability::StreamingTextToSpeech => sys::WAIFU_MODEL_SUPPORTS_STREAMING_TEXT_TO_SPEECH,
            Capability::SpeechToText => sys::WAIFU_MODEL_SUPPORTS_SPEECH_TO_TEXT,
            Capability::VoiceConversion => sys::WAIFU_MODEL_SUPPORTS_VOICE_CONVERSION,
            Capability::LayerDecomposition => sys::WAIFU_MODEL_SUPPORTS_LAYER_DECOMPOSITION,
            Capability::ImageGenerationFromStartImage => {
                sys::WAIFU_MODEL_SUPPORTS_IMAGE_GENERATION_FROM_START_IMAGE
            }
            Capability::ImageGenerationWithControlImage => {
                sys::WAIFU_MODEL_SUPPORTS_IMAGE_GENERATION_WITH_CONTROL_IMAGE
            }
            Capability::ImageGenerationWithPromptImages => {
                sys::WAIFU_MODEL_SUPPORTS_IMAGE_GENERATION_WITH_PROMPT_IMAGES
            }
            Capability::TextToSpeechLikeReference => {
                sys::WAIFU_MODEL_SUPPORTS_TEXT_TO_SPEECH_LIKE_REFERENCE
            }
            Capability::TextToSpeechWithoutReference => {
                sys::WAIFU_MODEL_SUPPORTS_TEXT_TO_SPEECH_WITHOUT_REFERENCE
            }
            Capability::TextToSpeechWithoutReferenceTranscript => {
                sys::WAIFU_MODEL_SUPPORTS_TEXT_TO_SPEECH_WITHOUT_REFERENCE_TRANSCRIPT
            }
            Capability::TextToSpeechWithStyle => {
                sys::WAIFU_MODEL_SUPPORTS_TEXT_TO_SPEECH_WITH_STYLE
            }
            Capability::VoiceConversionOfStyle => {
                sys::WAIFU_MODEL_SUPPORTS_VOICE_CONVERSION_OF_STYLE
            }
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Support {
    /// The C API's 1.
    Yes,
    /// 0, and the string attribute of the same number: why.
    No(String),
    /// Absent: a capability the library is too old to know.
    Unknown,
}

impl Model {
    /// Refused with `UnknownModel` where there is no manifest at the path, `Model` where it cannot
    /// be read.
    pub fn open(manifest_path: impl AsRef<Path>) -> Result<Model> {
        let f = sys::functions()?;
        let path = manifest_path.as_ref();
        let asked = path
            .to_str()
            .ok_or_else(|| Error::invalid(format!("{} is not UTF-8", path.display())))?;
        let asked = std::ffi::CString::new(asked)
            .map_err(|_| Error::invalid("the manifest path has a NUL in it"))?;
        let handle = unsafe { (f.waifu_model_open)(asked.as_ptr()) };
        match handle.is_null() {
            true => Err(Error::refused(f)),
            false => Ok(Model {
                handle,
                functions: f,
            }),
        }
    }

    pub(crate) fn raw(&self) -> *const sys::WaifuModel {
        self.handle
    }

    fn int(&self, attribute: sys::WaifuModelAttribute) -> Option<i64> {
        let mut value = 0;
        unsafe {
            (self.functions.waifu_model_get_int_attribute)(self.handle, attribute, &mut value)
        }
        .then_some(value)
    }

    fn string(&self, attribute: sys::WaifuModelAttribute) -> Option<String> {
        unsafe {
            text((self.functions.waifu_model_get_string_attribute)(
                self.handle,
                attribute,
            ))
        }
    }

    pub fn supports(&self, capability: Capability) -> Support {
        match self.int(capability.raw()) {
            Some(1) => Support::Yes,
            Some(_) => Support::No(self.string(capability.raw()).unwrap_or_default()),
            None => Support::Unknown,
        }
    }

    /// Bytes; None until every file is here.
    pub fn size_on_disk(&self) -> Option<u64> {
        self.int(sys::WAIFU_MODEL_SIZE_ON_DISK)
            .map(|v| v.max(0) as u64)
    }

    /// Of what it says, converts or hears.
    pub fn sample_rate(&self) -> Option<u32> {
        self.int(sys::WAIFU_MODEL_SAMPLE_RATE).map(|v| v as u32)
    }

    pub fn image_generation_width_height_multiple_of(&self) -> Option<u32> {
        self.int(sys::WAIFU_MODEL_IMAGE_GENERATION_WIDTH_HEIGHT_MULTIPLE_OF)
            .map(|v| v as u32)
    }

    pub fn image_generation_sampler(&self) -> Option<String> {
        self.string(sys::WAIFU_MODEL_IMAGE_GENERATION_SAMPLER)
    }

    pub fn image_generation_max_prompt_images(&self) -> Option<u32> {
        self.int(sys::WAIFU_MODEL_IMAGE_GENERATION_MAX_PROMPT_IMAGES)
            .map(|v| v as u32)
    }

    pub fn layer_decomposition_resolution_multiple_of(&self) -> Option<u32> {
        self.int(sys::WAIFU_MODEL_LAYER_DECOMPOSITION_RESOLUTION_MULTIPLE_OF)
            .map(|v| v as u32)
    }

    /// The languages a speech to text request may name; empty for a model that does not hear.
    pub fn speech_to_text_languages(&self) -> Vec<String> {
        self.string(sys::WAIFU_MODEL_SPEECH_TO_TEXT_LANGUAGES)
            .map(|languages| {
                languages
                    .split(',')
                    .map(str::trim)
                    .filter(|language| !language.is_empty())
                    .map(str::to_string)
                    .collect()
            })
            .unwrap_or_default()
    }
}

impl Drop for Model {
    fn drop(&mut self) {
        unsafe { (self.functions.waifu_model_free)(self.handle) };
    }
}

impl fmt::Debug for Model {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Model").finish_non_exhaustive()
    }
}

// -- waiting without a runtime ---------------------------------------------------------------

/// Wakes the thread that waits.
struct Unpark(std::thread::Thread);

impl Wake for Unpark {
    fn wake(self: Arc<Self>) {
        self.0.unpark();
    }

    fn wake_by_ref(self: &Arc<Self>) {
        self.0.unpark();
    }
}

/// Runs `future` to its end on this thread, for a program with no async runtime: a future, or a
/// [`Job`] as it is. Never inside async code: it blocks the thread it is called on.
pub fn block_on<F: IntoFuture>(future: F) -> F::Output {
    let mut future = pin!(future.into_future());
    let waker = Arc::new(Unpark(std::thread::current())).into();
    let mut context = Context::from_waker(&waker);
    loop {
        match future.as_mut().poll(&mut context) {
            Poll::Ready(output) => return output,
            Poll::Pending => std::thread::park(),
        }
    }
}
