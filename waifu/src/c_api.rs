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

//! The C API, ABI 4: pipelines, models, devices and the log, as `extern "C"`. Everything outside
//! this crate goes through these. docs/ffi-abi4.md is the contract; it changes only with a human's
//! approval, and so does this.
//!
//! Nothing with any logic in it lives here. This converts -- C strings to `&str`, structs to the
//! engine's requests, what comes back to borrowed C views, panics to status codes -- and keeps
//! the promises the contract makes about when a callback can arrive.

#![allow(non_camel_case_types)]
#![allow(clippy::missing_safety_doc)]

use std::collections::BTreeMap;
use std::ffi::{c_char, c_void, CString};
use std::ptr;

mod support;

use support::{caught, refusable, required, status, string_in, Gate, Refused, UserData};
pub use support::{
    WaifuCallback, WaifuLog, WaifuLogCallback, WaifuLogLevel, WaifuProgress, WaifuProgressCallback,
    WaifuResult, WaifuStage, WaifuStatus, WaifuStatusCode, WAIFU_CANCELLED,
    WAIFU_ERR_DEVICE_UNAVAILABLE, WAIFU_ERR_INTERNAL, WAIFU_ERR_INVALID_ARGUMENT, WAIFU_ERR_MODEL,
    WAIFU_ERR_NOT_LOADED, WAIFU_ERR_UNKNOWN_MODEL, WAIFU_LOG_DEBUG, WAIFU_LOG_ERROR,
    WAIFU_LOG_FATAL, WAIFU_LOG_INFO, WAIFU_LOG_WARNING, WAIFU_OK,
    WAIFU_STAGE_IMAGE_GENERATION_DECODING_IMAGE, WAIFU_STAGE_IMAGE_GENERATION_DENOISING,
    WAIFU_STAGE_IMAGE_GENERATION_ENCODING_PROMPT,
    WAIFU_STAGE_LAYER_DECOMPOSITION_CUTTING_INTO_PARTS,
    WAIFU_STAGE_LAYER_DECOMPOSITION_DECODING_DEPTH,
    WAIFU_STAGE_LAYER_DECOMPOSITION_DECODING_HEAD_LAYERS,
    WAIFU_STAGE_LAYER_DECOMPOSITION_DECODING_LAYERS,
    WAIFU_STAGE_LAYER_DECOMPOSITION_DENOISING_DEPTH,
    WAIFU_STAGE_LAYER_DECOMPOSITION_DENOISING_HEAD_LAYERS,
    WAIFU_STAGE_LAYER_DECOMPOSITION_DENOISING_LAYERS,
    WAIFU_STAGE_LAYER_DECOMPOSITION_ENCODING_FOR_DEPTH,
    WAIFU_STAGE_LAYER_DECOMPOSITION_ENCODING_HEAD, WAIFU_STAGE_LAYER_DECOMPOSITION_ENCODING_IMAGE,
    WAIFU_STAGE_LOADING_WEIGHTS, WAIFU_STAGE_SPEECH_TO_TEXT_GENERATING_TOKENS,
    WAIFU_STAGE_SPEECH_TO_TEXT_READING_RECORDING,
    WAIFU_STAGE_STREAMING_TEXT_TO_SPEECH_GENERATING_TOKENS,
    WAIFU_STAGE_STREAMING_TEXT_TO_SPEECH_READING_TEXT,
    WAIFU_STAGE_STREAMING_TEXT_TO_SPEECH_SYNTHESIZING_AUDIO,
    WAIFU_STAGE_TEXT_TO_SPEECH_GENERATING_TOKENS, WAIFU_STAGE_TEXT_TO_SPEECH_READING_TEXT,
    WAIFU_STAGE_TEXT_TO_SPEECH_SYNTHESIZING_AUDIO, WAIFU_STAGE_VOICE_CONVERSION_DENOISING,
    WAIFU_STAGE_VOICE_CONVERSION_GENERATING_TOKENS,
    WAIFU_STAGE_VOICE_CONVERSION_READING_RECORDINGS,
    WAIFU_STAGE_VOICE_CONVERSION_SYNTHESIZING_AUDIO,
};

use crate::engine::{
    self, DrawRequest, Ended, Engine, EngineOptions, Failure, LayerKind, LayersRequest, RgbaImage,
    SpeakRequest, Stage, TranscribeRequest, VoiceConversionRequest,
};
use crate::flint::MemoryBudget;
use crate::model_info::{self, ModelInfo, Support};
use crate::runtime::{self, DeviceOption, Runtime};
use crate::wav::Sound;
use crate::Device;

/// What a caller compiled against this header expects. Bumped when anything in it changes in a
/// way an older caller would misread.
///
/// 4: pipelines made for one task, models and devices asked about by property, no JSON.
pub const WAIFU_ABI_VERSION: u32 = 4;

// -- enums -----------------------------------------------------------------------------------

pub type WaifuDevice = i32;
pub const WAIFU_DEVICE_AUTO: WaifuDevice = 0;
pub const WAIFU_DEVICE_CPU: WaifuDevice = 1;
pub const WAIFU_DEVICE_METAL: WaifuDevice = 2;
pub const WAIFU_DEVICE_CUDA: WaifuDevice = 3;
// 4 and 6 were CUDA_CPU_OFFLOAD and METAL_DISK_OFFLOAD, aliases kept for older headers: not reused.
pub const WAIFU_DEVICE_VULKAN: WaifuDevice = 5;

/// What a pipeline is made for.
pub type WaifuPipelineType = i32;
pub const WAIFU_PIPELINE_IMAGE_GENERATION: WaifuPipelineType = 1;
pub const WAIFU_PIPELINE_TEXT_TO_SPEECH: WaifuPipelineType = 2;
pub const WAIFU_PIPELINE_VOICE_CONVERSION: WaifuPipelineType = 3;
pub const WAIFU_PIPELINE_LAYER_DECOMPOSITION: WaifuPipelineType = 4;
pub const WAIFU_PIPELINE_SPEECH_TO_TEXT: WaifuPipelineType = 5;
pub const WAIFU_PIPELINE_STREAMING_TEXT_TO_SPEECH: WaifuPipelineType = 6;

/// What can be asked of a model. SUPPORTS_ properties are integers, 1 or 0; where 0, the string
/// property of the same number is why, in words to show, and is never NULL. A SUPPORTS_ property
/// this library does not know is absent, which is neither 1 nor 0. The rest are an integer or a
/// string, as marked. SUPPORTS_<task> has the number of WAIFU_PIPELINE_<task>.
pub type WaifuModelProperty = i32;
pub const WAIFU_MODEL_SUPPORTS_IMAGE_GENERATION: WaifuModelProperty = 1;
pub const WAIFU_MODEL_SUPPORTS_TEXT_TO_SPEECH: WaifuModelProperty = 2;
pub const WAIFU_MODEL_SUPPORTS_VOICE_CONVERSION: WaifuModelProperty = 3;
pub const WAIFU_MODEL_SUPPORTS_LAYER_DECOMPOSITION: WaifuModelProperty = 4;
pub const WAIFU_MODEL_SUPPORTS_SPEECH_TO_TEXT: WaifuModelProperty = 5;
/// The model streams; the device is checked at load.
pub const WAIFU_MODEL_SUPPORTS_STREAMING_TEXT_TO_SPEECH: WaifuModelProperty = 6;
/// img2img.
pub const WAIFU_MODEL_SUPPORTS_IMAGE_GENERATION_FROM_START_IMAGE: WaifuModelProperty = 100;
/// A ControlNet.
pub const WAIFU_MODEL_SUPPORTS_IMAGE_GENERATION_WITH_CONTROL_IMAGE: WaifuModelProperty = 101;
/// Images the prompt names as <|key|>.
pub const WAIFU_MODEL_SUPPORTS_IMAGE_GENERATION_WITH_PROMPT_IMAGES: WaifuModelProperty = 102;
/// In the voice of a recording.
pub const WAIFU_MODEL_SUPPORTS_TEXT_TO_SPEECH_LIKE_REFERENCE: WaifuModelProperty = 200;
/// CosyVoice3 does not.
pub const WAIFU_MODEL_SUPPORTS_TEXT_TO_SPEECH_WITHOUT_REFERENCE: WaifuModelProperty = 201;
/// Follows an instruction such as "speak happily".
pub const WAIFU_MODEL_SUPPORTS_TEXT_TO_SPEECH_WITH_STYLE: WaifuModelProperty = 202;
/// Reads the words of the reference recording -- a request's transcript -- and goes on from it.
pub const WAIFU_MODEL_SUPPORTS_TEXT_TO_SPEECH_WITH_TRANSCRIPT: WaifuModelProperty = 203;
/// The manner as well as the voice.
pub const WAIFU_MODEL_SUPPORTS_VOICE_CONVERSION_OF_STYLE: WaifuModelProperty = 300;
/// Integer, bytes; absent until every file is here.
pub const WAIFU_MODEL_SIZE_ON_DISK: WaifuModelProperty = 1000;
/// Integer, of what it says or converts.
pub const WAIFU_MODEL_SAMPLE_RATE: WaifuModelProperty = 1001;
/// Integer: 16, or 32 for SDXL and Qwen-Image.
pub const WAIFU_MODEL_IMAGE_GENERATION_WIDTH_HEIGHT_MULTIPLE_OF: WaifuModelProperty = 1100;
/// String: "Euler".
pub const WAIFU_MODEL_IMAGE_GENERATION_SAMPLER: WaifuModelProperty = 1101;
/// Integer.
pub const WAIFU_MODEL_IMAGE_GENERATION_MAX_PROMPT_IMAGES: WaifuModelProperty = 1102;
/// Integer: 64.
pub const WAIFU_MODEL_LAYER_DECOMPOSITION_RESOLUTION_MULTIPLE_OF: WaifuModelProperty = 1400;
/// String: "Chinese,English,...", comma separated.
pub const WAIFU_MODEL_SPEECH_TO_TEXT_LANGUAGES: WaifuModelProperty = 1500;

/// What can be asked of a device. All of it is this process's, not the card as other programs see
/// it.
pub type WaifuDeviceProperty = i32;
/// The most it may hold there: the memory limit, or the device's memory.
pub const WAIFU_DEVICE_MEMORY_BUDGET: WaifuDeviceProperty = 1;
/// What its allocator holds now, what it keeps for reuse included.
pub const WAIFU_DEVICE_MEMORY_HELD: WaifuDeviceProperty = 2;
/// BUDGET - HELD, never below 0.
pub const WAIFU_DEVICE_MEMORY_AVAILABLE: WaifuDeviceProperty = 3;
/// What waifu_device_set_memory_limit set; absent where nothing did.
pub const WAIFU_DEVICE_MEMORY_LIMIT: WaifuDeviceProperty = 4;

pub type WaifuLayerType = i32;
/// A part as the model drew it.
pub const WAIFU_LAYER_PART: WaifuLayerType = 0;
/// stratify_layers: the near half of a part.
pub const WAIFU_LAYER_STRATIFIED_NEAR: WaifuLayerType = 1;
/// stratify_layers: the far half of a part.
pub const WAIFU_LAYER_STRATIFIED_FAR: WaifuLayerType = 2;
/// The depth of the layer derived_from_id names, pixel by pixel.
pub const WAIFU_LAYER_DEPTH: WaifuLayerType = 3;

// -- the library -----------------------------------------------------------------------------

#[no_mangle]
pub extern "C" fn waifu_abi_version() -> u32 {
    WAIFU_ABI_VERSION
}

/// This thread's last refused call, as a sentence and as a code; both are set together by a
/// refused call and left alone by every other, and reading one does not change the other. The
/// string is valid until this thread's next refused call. "" and WAIFU_OK before any.
#[no_mangle]
pub extern "C" fn waifu_last_error() -> *const c_char {
    support::last_error()
}

#[no_mangle]
pub extern "C" fn waifu_last_error_code() -> WaifuStatusCode {
    support::last_error_code()
}

/// "0.1.0"; static.
#[no_mangle]
pub extern "C" fn waifu_version() -> *const c_char {
    concat!(env!("CARGO_PKG_VERSION"), "\0").as_ptr() as *const c_char
}

/// The full commit, or NULL for a build made without git; static.
#[no_mangle]
pub extern "C" fn waifu_revision() -> *const c_char {
    const REVISION: &str = concat!(env!("WAIFU_REVISION"), "\0");
    match REVISION.len() {
        1 => ptr::null(),
        _ => REVISION.as_ptr() as *const c_char,
    }
}

/// NULL: the console again. Waits for a line being handed to the callback before to finish: once
/// it returns, the callback and user_data before are never called again, and may be let go of.
#[no_mangle]
pub extern "C" fn waifu_set_log_callback(user_data: *mut c_void, callback: WaifuLogCallback) {
    caught((), || {
        crate::log::set_sink(support::log_sink(user_data, callback))
    })
}

/// INFO unless set. A level this library does not know changes nothing.
#[no_mangle]
pub extern "C" fn waifu_set_log_level(level: WaifuLogLevel) {
    caught((), || {
        if let Some(level) = support::log_level(level) {
            crate::log::set_level(level)
        }
    })
}

// -- devices ---------------------------------------------------------------------------------

/// The device a caller named, or None for a number this library does not know.
fn device_option(device: WaifuDevice) -> Option<DeviceOption> {
    match device {
        WAIFU_DEVICE_AUTO => Some(DeviceOption::Auto),
        WAIFU_DEVICE_CPU => Some(DeviceOption::Cpu),
        WAIFU_DEVICE_METAL => Some(DeviceOption::Metal),
        WAIFU_DEVICE_CUDA => Some(DeviceOption::Cuda),
        WAIFU_DEVICE_VULKAN => Some(DeviceOption::Vulkan),
        _ => None,
    }
}

/// A device as the header numbers it.
fn device_out(device: Device) -> WaifuDevice {
    match device {
        Device::Cpu => WAIFU_DEVICE_CPU,
        // Page-locked host memory is a place for CUDA's tensors, not a device a run goes to.
        Device::Cuda | Device::CudaHost => WAIFU_DEVICE_CUDA,
        Device::Metal => WAIFU_DEVICE_METAL,
        Device::Vulkan => WAIFU_DEVICE_VULKAN,
    }
}

/// The device `device` means here, where it is one this build and machine can use.
fn available(device: WaifuDevice) -> Option<Device> {
    let device = device_option(device)?.resolve().device();
    device.is_available().then_some(device)
}

#[no_mangle]
pub extern "C" fn waifu_device_is_available(device: WaifuDevice) -> bool {
    caught(false, || available(device).is_some())
}

/// false for the processor, a device that cannot be used, or a property this library does not
/// know. Asking CUDA builds a context on the card, a few hundred megabytes of it.
#[no_mangle]
pub unsafe extern "C" fn waifu_device_get_int_property(
    device: WaifuDevice,
    property: WaifuDeviceProperty,
    value: *mut i64,
) -> bool {
    caught(false, || {
        let Some(device) = available(device).filter(|device| *device != Device::Cpu) else {
            return false;
        };
        if value.is_null() {
            return false;
        }
        let measured = || {
            MemoryBudget::capture(device)
                .ok()
                .filter(|b| b.capacity > 0)
        };
        let answer = match property {
            WAIFU_DEVICE_MEMORY_BUDGET => measured().map(|budget| budget.capacity),
            WAIFU_DEVICE_MEMORY_HELD => measured().map(|budget| budget.held),
            WAIFU_DEVICE_MEMORY_AVAILABLE => {
                measured().map(|budget| (budget.capacity - budget.held).max(0))
            }
            WAIFU_DEVICE_MEMORY_LIMIT => {
                runtime::memory_limit(device).map(|bytes| bytes.min(i64::MAX as u64) as i64)
            }
            _ => None,
        };
        match answer {
            Some(answer) => {
                *value = answer;
                true
            }
            None => false,
        }
    })
}

/// For the whole process, one per device, shared by every pipeline on it: set it before a model
/// is loaded. 0 for no limit but the device's. The processor has none, and is refused.
#[no_mangle]
pub extern "C" fn waifu_device_set_memory_limit(device: WaifuDevice, bytes: u64) -> bool {
    refusable(false, || {
        let option = device_option(device)
            .ok_or_else(|| format!("{device} is not a device this library knows"))?;
        let resolved = option.resolve().device();
        if resolved == Device::Cpu {
            return Err("the processor takes no memory limit".into());
        }
        if !resolved.is_available() {
            return Err(Refused::new(
                WAIFU_ERR_DEVICE_UNAVAILABLE,
                format!("{} is not available on this machine", resolved.name()),
            ));
        }
        runtime::set_memory_limit(option, bytes)?;
        Ok(true)
    })
}

// -- models ----------------------------------------------------------------------------------

/// A model's manifest, read, and what can be asked of it, answered once when it is opened.
pub struct WaifuModel {
    /// What a load reads the weights from.
    path: String,
    ints: BTreeMap<WaifuModelProperty, i64>,
    /// Each lives until the model is freed.
    strings: BTreeMap<WaifuModelProperty, CString>,
}

impl WaifuModel {
    fn new(path: String, info: &ModelInfo) -> WaifuModel {
        let mut model = WaifuModel {
            path,
            ints: BTreeMap::new(),
            strings: BTreeMap::new(),
        };

        if let Some(bytes) = info.size_on_disk {
            model.int(WAIFU_MODEL_SIZE_ON_DISK, bytes.min(i64::MAX as u64) as i64);
        }
        if let Some(rate) = info.sample_rate {
            model.int(WAIFU_MODEL_SAMPLE_RATE, rate as i64);
        }

        // A detail of a task the model does not do is not supported either, for the same reason.
        let detail = |done: Option<&Support>, otherwise: &Support| match done {
            Some(support) => support.clone(),
            None => otherwise.clone(),
        };

        let image = task(&info.image_generation);
        model.supports(WAIFU_MODEL_SUPPORTS_IMAGE_GENERATION, &image);
        let generation = info.image_generation.as_ref().ok();
        model.supports(
            WAIFU_MODEL_SUPPORTS_IMAGE_GENERATION_FROM_START_IMAGE,
            &detail(generation.map(|g| &g.from_start_image), &image),
        );
        model.supports(
            WAIFU_MODEL_SUPPORTS_IMAGE_GENERATION_WITH_CONTROL_IMAGE,
            &detail(generation.map(|g| &g.with_control_image), &image),
        );
        model.supports(
            WAIFU_MODEL_SUPPORTS_IMAGE_GENERATION_WITH_PROMPT_IMAGES,
            &detail(generation.map(|g| &g.with_prompt_images), &image),
        );
        if let Some(generation) = generation {
            model.int(
                WAIFU_MODEL_IMAGE_GENERATION_WIDTH_HEIGHT_MULTIPLE_OF,
                generation.width_height_multiple_of as i64,
            );
            model.string(WAIFU_MODEL_IMAGE_GENERATION_SAMPLER, &generation.sampler);
            model.int(
                WAIFU_MODEL_IMAGE_GENERATION_MAX_PROMPT_IMAGES,
                generation.max_prompt_images as i64,
            );
        }

        let speech = task(&info.text_to_speech);
        model.supports(WAIFU_MODEL_SUPPORTS_TEXT_TO_SPEECH, &speech);
        let speaking = info.text_to_speech.as_ref().ok();
        model.supports(
            WAIFU_MODEL_SUPPORTS_TEXT_TO_SPEECH_LIKE_REFERENCE,
            &detail(speaking.map(|s| &s.like_reference), &speech),
        );
        model.supports(
            WAIFU_MODEL_SUPPORTS_TEXT_TO_SPEECH_WITHOUT_REFERENCE,
            &detail(speaking.map(|s| &s.without_reference), &speech),
        );
        model.supports(
            WAIFU_MODEL_SUPPORTS_TEXT_TO_SPEECH_WITH_STYLE,
            &detail(speaking.map(|s| &s.with_style), &speech),
        );
        model.supports(
            WAIFU_MODEL_SUPPORTS_TEXT_TO_SPEECH_WITH_TRANSCRIPT,
            &detail(speaking.map(|s| &s.with_transcript), &speech),
        );

        model.supports(
            WAIFU_MODEL_SUPPORTS_STREAMING_TEXT_TO_SPEECH,
            &info.streaming_text_to_speech,
        );

        let conversion = task(&info.voice_conversion);
        model.supports(WAIFU_MODEL_SUPPORTS_VOICE_CONVERSION, &conversion);
        model.supports(
            WAIFU_MODEL_SUPPORTS_VOICE_CONVERSION_OF_STYLE,
            &detail(
                info.voice_conversion.as_ref().ok().map(|c| &c.of_style),
                &conversion,
            ),
        );

        model.supports(
            WAIFU_MODEL_SUPPORTS_LAYER_DECOMPOSITION,
            &task(&info.layer_decomposition),
        );
        if let Ok(decomposition) = &info.layer_decomposition {
            model.int(
                WAIFU_MODEL_LAYER_DECOMPOSITION_RESOLUTION_MULTIPLE_OF,
                decomposition.resolution_multiple_of as i64,
            );
        }

        model.supports(
            WAIFU_MODEL_SUPPORTS_SPEECH_TO_TEXT,
            &task(&info.speech_to_text),
        );
        if let Ok(transcription) = &info.speech_to_text {
            model.string(
                WAIFU_MODEL_SPEECH_TO_TEXT_LANGUAGES,
                &transcription.languages.join(","),
            );
        }

        model
    }

    fn int(&mut self, property: WaifuModelProperty, value: i64) {
        self.ints.insert(property, value);
    }

    fn string(&mut self, property: WaifuModelProperty, value: &str) {
        self.strings.insert(property, support::c_string(value));
    }

    /// 1, or 0 and why.
    fn supports(&mut self, property: WaifuModelProperty, support: &Support) {
        match support {
            Ok(()) => self.int(property, 1),
            Err(why) => {
                self.int(property, 0);
                self.string(property, why);
            }
        }
    }

    /// Why the model cannot be loaded for `task`, where it cannot.
    fn unsupported(&self, task: WaifuPipelineType) -> Option<String> {
        match self.ints.get(&task) {
            Some(1) => None,
            _ => Some(
                self.strings
                    .get(&task)
                    .and_then(|why| why.to_str().ok())
                    .map_or_else(|| "the model does not do that".to_string(), str::to_string),
            ),
        }
    }
}

/// Whether a model does a task, and why not.
fn task<T>(done: &Result<T, String>) -> Support {
    match done {
        Ok(_) => Ok(()),
        Err(why) => Err(why.clone()),
    }
}

/// Reads the manifest at `manifest_path` -- a few kilobytes, and none of the weights. Refused --
/// NULL -- with WAIFU_ERR_UNKNOWN_MODEL where there is none, WAIFU_ERR_MODEL where it cannot be
/// read.
#[no_mangle]
pub unsafe extern "C" fn waifu_model_open(manifest_path: *const c_char) -> *mut WaifuModel {
    refusable(ptr::null_mut(), || {
        let path = required(manifest_path, "manifest_path")?;
        let info = model_info::model_info(&path)
            .map_err(|(failure, message)| Refused::new(code_of(failure), message))?;
        Ok(Box::into_raw(Box::new(WaifuModel::new(path, &info))))
    })
}

/// NULL is fine.
#[no_mangle]
pub unsafe extern "C" fn waifu_model_free(model: *mut WaifuModel) {
    if !model.is_null() {
        caught((), || drop(Box::from_raw(model)));
    }
}

/// false where the model has no such property or this library does not know it; *value untouched.
#[no_mangle]
pub unsafe extern "C" fn waifu_model_get_int_property(
    model: *const WaifuModel,
    property: WaifuModelProperty,
    value: *mut i64,
) -> bool {
    caught(false, || {
        let (Some(model), false) = (model.as_ref(), value.is_null()) else {
            return false;
        };
        match model.ints.get(&property) {
            Some(answer) => {
                *value = *answer;
                true
            }
            None => false,
        }
    })
}

/// NULL where there is none; valid until waifu_model_free.
#[no_mangle]
pub unsafe extern "C" fn waifu_model_get_string_property(
    model: *const WaifuModel,
    property: WaifuModelProperty,
) -> *const c_char {
    caught(ptr::null(), || {
        model
            .as_ref()
            .and_then(|model| model.strings.get(&property))
            .map_or(ptr::null(), |answer| answer.as_ptr())
    })
}

// -- pipelines -------------------------------------------------------------------------------

/// One task, one thread, at most one model: jobs run on it one at a time, in the order asked.
pub struct WaifuPipeline {
    engine: Engine,
    task: WaifuPipelineType,
}

/// A task as a sentence names it.
fn task_name(task: WaifuPipelineType) -> &'static str {
    match task {
        WAIFU_PIPELINE_IMAGE_GENERATION => "image generation",
        WAIFU_PIPELINE_TEXT_TO_SPEECH => "text to speech",
        WAIFU_PIPELINE_VOICE_CONVERSION => "voice conversion",
        WAIFU_PIPELINE_LAYER_DECOMPOSITION => "layer decomposition",
        WAIFU_PIPELINE_SPEECH_TO_TEXT => "speech to text",
        WAIFU_PIPELINE_STREAMING_TEXT_TO_SPEECH => "streaming text to speech",
        _ => "no task",
    }
}

/// Starts the pipeline's thread, with no model. Refused -- NULL, WAIFU_ERR_DEVICE_UNAVAILABLE --
/// where the device is not available.
#[no_mangle]
pub extern "C" fn waifu_pipeline_new(
    r#type: WaifuPipelineType,
    device: WaifuDevice,
) -> *mut WaifuPipeline {
    refusable(ptr::null_mut(), || {
        if task_name(r#type) == "no task" {
            return Err(format!("{} is not a pipeline type this library knows", r#type).into());
        }
        let option = device_option(device)
            .ok_or_else(|| format!("{device} is not a device this library knows"))?;
        let runtime: Runtime = option.resolve();
        if !runtime.device().is_available() {
            return Err(Refused::new(
                WAIFU_ERR_DEVICE_UNAVAILABLE,
                format!("{} is not available on this machine", runtime.name()),
            ));
        }
        let engine = Engine::new(EngineOptions { device: option })
            .map_err(|error| Refused::new(WAIFU_ERR_INTERNAL, error))?;
        Ok(Box::into_raw(Box::new(WaifuPipeline {
            engine,
            task: r#type,
        })))
    })
}

/// Stops what runs, cancels what waits, drops the model and joins the thread. Returns once every
/// job has had its on_complete. Must not be called from one of its own callbacks. NULL is fine.
#[no_mangle]
pub unsafe extern "C" fn waifu_pipeline_free(pipeline: *mut WaifuPipeline) {
    if !pipeline.is_null() {
        caught((), || drop(Box::from_raw(pipeline)));
    }
}

/// 0 for a NULL pipeline.
#[no_mangle]
pub unsafe extern "C" fn waifu_pipeline_get_type(
    pipeline: *const WaifuPipeline,
) -> WaifuPipelineType {
    pipeline.as_ref().map_or(0, |pipeline| pipeline.task)
}

/// Never AUTO, which is settled when it starts; AUTO only for a NULL pipeline.
#[no_mangle]
pub unsafe extern "C" fn waifu_pipeline_get_device(pipeline: *const WaifuPipeline) -> WaifuDevice {
    pipeline.as_ref().map_or(WAIFU_DEVICE_AUTO, |pipeline| {
        device_out(pipeline.engine.runtime().device())
    })
}

// -- jobs ------------------------------------------------------------------------------------

/// 0: refused at the call, with waifu_last_error and _code, and no callback is ever called. Any
/// other id: on_progress any number of times, then on_complete exactly once. Callbacks run on the
/// pipeline's thread and may start before the _async call has returned: keep a job's state in
/// user_data, not under its id. They must return quickly, and may call nothing but
/// waifu_pipeline_cancel.
pub type WaifuJob = u64;

/// After the step it is on, or before it starts.
#[no_mangle]
pub unsafe extern "C" fn waifu_pipeline_cancel(pipeline: *mut WaifuPipeline, job: WaifuJob) {
    if let Some(pipeline) = pipeline.as_ref() {
        caught((), || pipeline.engine.cancel(job));
    }
}

#[no_mangle]
pub unsafe extern "C" fn waifu_pipeline_cancel_all(pipeline: *mut WaifuPipeline) {
    if let Some(pipeline) = pipeline.as_ref() {
        caught((), || pipeline.engine.cancel_all());
    }
}

fn stage_out(stage: Stage) -> WaifuStage {
    match stage {
        Stage::LoadingWeights => WAIFU_STAGE_LOADING_WEIGHTS,
        Stage::ImageGenerationEncodingPrompt => WAIFU_STAGE_IMAGE_GENERATION_ENCODING_PROMPT,
        Stage::ImageGenerationDenoising => WAIFU_STAGE_IMAGE_GENERATION_DENOISING,
        Stage::ImageGenerationDecodingImage => WAIFU_STAGE_IMAGE_GENERATION_DECODING_IMAGE,
        Stage::TextToSpeechReadingText => WAIFU_STAGE_TEXT_TO_SPEECH_READING_TEXT,
        Stage::TextToSpeechGeneratingTokens => WAIFU_STAGE_TEXT_TO_SPEECH_GENERATING_TOKENS,
        Stage::TextToSpeechSynthesizingAudio => WAIFU_STAGE_TEXT_TO_SPEECH_SYNTHESIZING_AUDIO,
        Stage::VoiceConversionReadingRecordings => WAIFU_STAGE_VOICE_CONVERSION_READING_RECORDINGS,
        Stage::VoiceConversionGeneratingTokens => WAIFU_STAGE_VOICE_CONVERSION_GENERATING_TOKENS,
        Stage::VoiceConversionDenoising => WAIFU_STAGE_VOICE_CONVERSION_DENOISING,
        Stage::VoiceConversionSynthesizingAudio => WAIFU_STAGE_VOICE_CONVERSION_SYNTHESIZING_AUDIO,
        Stage::LayerDecompositionEncodingImage => WAIFU_STAGE_LAYER_DECOMPOSITION_ENCODING_IMAGE,
        Stage::LayerDecompositionDenoisingLayers => {
            WAIFU_STAGE_LAYER_DECOMPOSITION_DENOISING_LAYERS
        }
        Stage::LayerDecompositionDecodingLayers => WAIFU_STAGE_LAYER_DECOMPOSITION_DECODING_LAYERS,
        Stage::LayerDecompositionEncodingHead => WAIFU_STAGE_LAYER_DECOMPOSITION_ENCODING_HEAD,
        Stage::LayerDecompositionDenoisingHeadLayers => {
            WAIFU_STAGE_LAYER_DECOMPOSITION_DENOISING_HEAD_LAYERS
        }
        Stage::LayerDecompositionDecodingHeadLayers => {
            WAIFU_STAGE_LAYER_DECOMPOSITION_DECODING_HEAD_LAYERS
        }
        Stage::LayerDecompositionEncodingForDepth => {
            WAIFU_STAGE_LAYER_DECOMPOSITION_ENCODING_FOR_DEPTH
        }
        Stage::LayerDecompositionDenoisingDepth => WAIFU_STAGE_LAYER_DECOMPOSITION_DENOISING_DEPTH,
        Stage::LayerDecompositionDecodingDepth => WAIFU_STAGE_LAYER_DECOMPOSITION_DECODING_DEPTH,
        Stage::LayerDecompositionCuttingIntoParts => {
            WAIFU_STAGE_LAYER_DECOMPOSITION_CUTTING_INTO_PARTS
        }
        Stage::SpeechToTextReadingRecording => WAIFU_STAGE_SPEECH_TO_TEXT_READING_RECORDING,
        Stage::SpeechToTextGeneratingTokens => WAIFU_STAGE_SPEECH_TO_TEXT_GENERATING_TOKENS,
        Stage::StreamingTextToSpeechReadingText => {
            WAIFU_STAGE_STREAMING_TEXT_TO_SPEECH_READING_TEXT
        }
        Stage::StreamingTextToSpeechGeneratingTokens => {
            WAIFU_STAGE_STREAMING_TEXT_TO_SPEECH_GENERATING_TOKENS
        }
        Stage::StreamingTextToSpeechSynthesizingAudio => {
            WAIFU_STAGE_STREAMING_TEXT_TO_SPEECH_SYNTHESIZING_AUDIO
        }
    }
}

/// The engine's progress callback, calling the caller's.
fn progress_to(
    gate: &Gate,
    user_data: UserData,
    callback: WaifuProgressCallback,
) -> engine::OnProgress {
    let gate = gate.clone();
    Box::new(move |progress| {
        let Some(callback) = callback else {
            return;
        };
        gate.wait();
        let out = WaifuProgress {
            stage: stage_out(progress.stage),
            fraction: progress.fraction.unwrap_or(-1.0),
            done: progress.done,
            total: progress.total,
            seconds: progress.seconds,
        };
        let user_data = user_data;
        unsafe { callback(user_data.0, &out) };
    })
}

fn code_of(failure: Failure) -> WaifuStatusCode {
    match failure {
        Failure::UnknownModel => WAIFU_ERR_UNKNOWN_MODEL,
        Failure::Model => WAIFU_ERR_MODEL,
        Failure::NotLoaded => WAIFU_ERR_NOT_LOADED,
        Failure::Internal => WAIFU_ERR_INTERNAL,
    }
}

/// What an ending is called, and its message, as a WaifuStatus needs them -- with the CString kept
/// alive by the caller for as long as the status is.
fn status_of<T>(ended: &Ended<T>) -> (WaifuStatusCode, Option<CString>) {
    match ended {
        Ended::Done(_) => (WAIFU_OK, None),
        Ended::Cancelled => (WAIFU_CANCELLED, None),
        Ended::Failed(failure, message) => (code_of(*failure), Some(support::c_string(message))),
    }
}

fn completion_to(
    gate: &Gate,
    user_data: UserData,
    callback: WaifuCallback,
) -> engine::OnComplete<()> {
    let gate = gate.clone();
    Box::new(move |ended: Ended<()>| {
        let Some(callback) = callback else {
            return;
        };
        gate.wait();
        let (code, message) = status_of(&ended);
        let result = WaifuResult {
            status: status(code, &message),
        };
        let user_data = user_data;
        unsafe { callback(user_data.0, &result) };
    })
}

/// Runs an `_async` entry point: its id, or 0 with the refusal. The gate opens once the id is
/// ready to be handed back, which is when the job's callbacks may run.
fn submitted(gate: &Gate, submit: impl FnOnce() -> Result<WaifuJob, Refused>) -> WaifuJob {
    let job = refusable(0, submit);
    if job != 0 {
        gate.open();
    }
    job
}

/// The pipeline, where it is one made for `task`.
unsafe fn pipeline_for<'a>(
    pipeline: *mut WaifuPipeline,
    task: WaifuPipelineType,
) -> Result<&'a WaifuPipeline, Refused> {
    let pipeline = pipeline.as_ref().ok_or("the pipeline is NULL")?;
    if pipeline.task != task {
        return Err(format!(
            "this pipeline is made for {}, and {} is another task's",
            task_name(pipeline.task),
            task_name(task)
        )
        .into());
    }
    Ok(pipeline)
}

/// Reads the model's weights onto the pipeline's device, dropping the model before first, so two
/// are never on it at once. Refused at the call for a model the pipeline's task is not supported
/// by (WAIFU_MODEL_SUPPORTS_<task>). The call copies the manifest, not the weights, which the
/// pipeline's thread reads from the files: `model` may be freed as soon as it returns. A run may
/// be asked for straight after; it waits its turn. "tones", the stand-in voice the tests use,
/// opens as a model too.
#[no_mangle]
pub unsafe extern "C" fn waifu_pipeline_load_model_async(
    pipeline: *mut WaifuPipeline,
    model: *const WaifuModel,
    user_data: *mut c_void,
    on_progress: WaifuProgressCallback,
    on_complete: WaifuCallback,
) -> WaifuJob {
    let gate = Gate::default();
    submitted(&gate, || {
        let pipeline = pipeline.as_ref().ok_or("the pipeline is NULL")?;
        let model = model.as_ref().ok_or("the model is NULL")?;
        if let Some(why) = model.unsupported(pipeline.task) {
            return Err(format!(
                "a {} pipeline cannot load it: {why}",
                task_name(pipeline.task)
            )
            .into());
        }
        let user_data = UserData(user_data);
        let on_progress = progress_to(&gate, user_data, on_progress);
        let on_complete = completion_to(&gate, user_data, on_complete);
        Ok(match pipeline.task {
            WAIFU_PIPELINE_STREAMING_TEXT_TO_SPEECH => {
                pipeline
                    .engine
                    .load_for_streaming(&model.path, on_progress, on_complete)
            }
            _ => pipeline.engine.load(&model.path, on_progress, on_complete),
        })
    })
}

#[no_mangle]
pub unsafe extern "C" fn waifu_pipeline_unload_model_async(
    pipeline: *mut WaifuPipeline,
    user_data: *mut c_void,
    on_complete: WaifuCallback,
) -> WaifuJob {
    let gate = Gate::default();
    submitted(&gate, || {
        let pipeline = pipeline.as_ref().ok_or("the pipeline is NULL")?;
        Ok(pipeline
            .engine
            .unload(completion_to(&gate, UserData(user_data), on_complete)))
    })
}

/// A request struct, read only if it is at least as large as the one this library knows.
unsafe fn request_in<'a, T>(request: *const T, size: u32, what: &str) -> Result<&'a T, String> {
    let request = request
        .as_ref()
        .ok_or_else(|| format!("the {what} is NULL"))?;
    if (size as usize) < std::mem::size_of::<T>() {
        return Err(format!(
            "the {what} says it is {size} bytes, and this library reads {}: set struct_size to sizeof",
            std::mem::size_of::<T>()
        ));
    }
    Ok(request)
}

/// A request struct that has grown at its end since `least` bytes of it were all there was: read
/// if it is at least that large, with every field past what the caller's `size` holds zero -- the
/// NULL a caller built against the older header never set.
///
/// `T` must be one for which all-zero bytes are a value: numbers and pointers.
unsafe fn grown_request_in<T>(
    request: *const T,
    size: u32,
    least: usize,
    what: &str,
) -> Result<T, String> {
    if request.is_null() {
        return Err(format!("the {what} is NULL"));
    }
    if (size as usize) < least {
        return Err(format!(
            "the {what} says it is {size} bytes, and this library reads {}: set struct_size to sizeof",
            std::mem::size_of::<T>()
        ));
    }
    let mut out = std::mem::MaybeUninit::<T>::zeroed();
    let held = (size as usize).min(std::mem::size_of::<T>());
    ptr::copy_nonoverlapping(request as *const u8, out.as_mut_ptr() as *mut u8, held);
    Ok(out.assume_init())
}

/// A pointer to `values`, or NULL for none.
fn first<T>(values: &[T]) -> *const T {
    match values.is_empty() {
        true => ptr::null(),
        false => values.as_ptr(),
    }
}

// -- image generation ------------------------------------------------------------------------

#[repr(C)]
pub struct WaifuImage {
    pub width: u32,
    pub height: u32,
    /// width * height * 3, rows top to bottom.
    pub rgb: *const u8,
    pub len: usize,
}

#[repr(C)]
pub struct WaifuPromptImage {
    /// The <|key|> the prompt names it by: letters, digits and _.
    pub key: *const c_char,
    pub image: *const WaifuImage,
}

#[repr(C)]
pub struct WaifuImageGenerationRequest {
    pub struct_size: u32,
    pub prompt: *const c_char,
    /// NULL or "" for none.
    pub negative: *const c_char,
    /// 64..2048, a multiple of IMAGE_GENERATION_WIDTH_HEIGHT_MULTIPLE_OF.
    pub width: u32,
    pub height: u32,
    pub steps: u32,
    pub guidance: f32,
    pub seed: u64,
    /// img2img; NULL for none.
    pub start_image: *const WaifuImage,
    /// How far it walks away from it, 0..1.
    pub start_image_strength: f32,
    /// A ControlNet's condition; NULL for none.
    pub control_image: *const WaifuImage,
    /// How hard it holds to it.
    pub control_image_scale: f32,
    pub prompt_images: *const WaifuPromptImage,
    pub prompt_image_count: usize,
}

#[repr(C)]
pub struct WaifuImageGenerationResult {
    pub status: WaifuStatus,
    /// NULL unless WAIFU_OK.
    pub image: *const WaifuImage,
}

pub type WaifuImageGenerationCallback =
    Option<unsafe extern "C" fn(user_data: *mut c_void, result: *const WaifuImageGenerationResult)>;

unsafe fn image_in(image: *const WaifuImage, what: &str) -> Result<engine::Image, String> {
    let image = image.as_ref().ok_or_else(|| format!("{what} is NULL"))?;
    if image.rgb.is_null() && image.len > 0 {
        return Err(format!("{what} has no pixels"));
    }
    let rgb = match image.len {
        0 => Vec::new(),
        len => std::slice::from_raw_parts(image.rgb, len).to_vec(),
    };
    Ok(engine::Image {
        width: image.width,
        height: image.height,
        rgb,
    })
}

/// The images a request holds, as the engine keys them: the start and control images under the
/// engine's own keys, which a prompt image may not take.
unsafe fn images_in(
    request: &WaifuImageGenerationRequest,
) -> Result<Vec<(String, engine::Image)>, String> {
    let mut images = Vec::with_capacity(request.prompt_image_count + 2);
    if !request.start_image.is_null() {
        images.push((
            engine::START_FROM.to_string(),
            image_in(request.start_image, "start_image")?,
        ));
    }
    if !request.control_image.is_null() {
        images.push((
            engine::CONTROL.to_string(),
            image_in(request.control_image, "control_image")?,
        ));
    }
    if request.prompt_image_count > 0 {
        if request.prompt_images.is_null() {
            return Err("prompt_image_count is not 0 and prompt_images is NULL".to_string());
        }
        let given = std::slice::from_raw_parts(request.prompt_images, request.prompt_image_count);
        for input in given {
            let key = required(input.key, "a prompt image's key")?;
            if key == engine::START_FROM || key == engine::CONTROL {
                return Err(format!(
                    "\"{key}\" is not a key a prompt image can have here: give the image as \
                     start_image or control_image, or call it something else"
                ));
            }
            let image = image_in(input.image, &format!("the prompt image \"{key}\""))?;
            images.push((key, image));
        }
    }
    Ok(images)
}

/// Each run function is refused at the call on a pipeline made for another task.
#[no_mangle]
pub unsafe extern "C" fn waifu_pipeline_run_image_generation_async(
    pipeline: *mut WaifuPipeline,
    request: *const WaifuImageGenerationRequest,
    user_data: *mut c_void,
    on_progress: WaifuProgressCallback,
    on_complete: WaifuImageGenerationCallback,
) -> WaifuJob {
    let gate = Gate::default();
    submitted(&gate, || {
        let pipeline = pipeline_for(pipeline, WAIFU_PIPELINE_IMAGE_GENERATION)?;
        let size = request.as_ref().map_or(0, |request| request.struct_size);
        let request = request_in(request, size, "WaifuImageGenerationRequest")?;
        let request = DrawRequest {
            prompt: required(request.prompt, "prompt")?,
            negative: string_in(request.negative, "negative")?.unwrap_or_default(),
            images: images_in(request)?,
            width: request.width,
            height: request.height,
            steps: request.steps,
            guidance: request.guidance,
            seed: request.seed,
            strength: request.start_image_strength,
            control_scale: request.control_image_scale,
        };

        let user_data = UserData(user_data);
        let on_complete = {
            let gate = gate.clone();
            Box::new(move |ended: Ended<engine::Image>| {
                let Some(callback) = on_complete else {
                    return;
                };
                gate.wait();
                let (code, message) = status_of(&ended);
                let image = match &ended {
                    Ended::Done(image) => Some(WaifuImage {
                        width: image.width,
                        height: image.height,
                        rgb: image.rgb.as_ptr(),
                        len: image.rgb.len(),
                    }),
                    _ => None,
                };
                let result = WaifuImageGenerationResult {
                    status: status(code, &message),
                    image: image
                        .as_ref()
                        .map_or(ptr::null(), |image| image as *const _),
                };
                let user_data = user_data;
                unsafe { callback(user_data.0, &result) };
            })
        };
        Ok(pipeline.engine.draw(
            request,
            progress_to(&gate, user_data, on_progress),
            on_complete,
        )?)
    })
}

// -- text to speech and voice conversion -----------------------------------------------------

#[repr(C)]
pub struct WaifuAudio {
    /// Mono, -1..1.
    pub samples: *const f32,
    pub count: usize,
    pub sample_rate: u32,
}

#[repr(C)]
pub struct WaifuTextToSpeechRequest {
    pub struct_size: u32,
    pub text: *const c_char,
    /// A recording to sound like, or NULL.
    pub reference: *const WaifuAudio,
    pub speed: f32,
    pub temperature: f32,
    /// Any instruction; NULL or "" for the voice's own way.
    pub style: *const c_char,
    pub seed: u64,
    /// What the reference says, word for word, for a model that SUPPORTS_TEXT_TO_SPEECH_WITH_TRANSCRIPT;
    /// NULL or "" where it is not known. Last, and read only where struct_size holds it.
    pub transcript: *const c_char,
}

#[repr(C)]
pub struct WaifuVoiceConversionRequest {
    pub struct_size: u32,
    /// What is said, and how.
    pub source: *const WaifuAudio,
    /// Whose voice to say it in.
    pub reference: *const WaifuAudio,
    pub steps: u32,
    pub convert_style: bool,
    pub seed: u64,
}

/// Both speech tasks make audio.
#[repr(C)]
pub struct WaifuAudioResult {
    pub status: WaifuStatus,
    /// NULL unless WAIFU_OK.
    pub audio: *const WaifuAudio,
}

pub type WaifuAudioCallback =
    Option<unsafe extern "C" fn(user_data: *mut c_void, result: *const WaifuAudioResult)>;

unsafe fn audio_in(audio: *const WaifuAudio, what: &str) -> Result<Sound, String> {
    let audio = audio.as_ref().ok_or_else(|| format!("{what} is NULL"))?;
    if audio.samples.is_null() && audio.count > 0 {
        return Err(format!("{what} has no samples"));
    }
    if audio.sample_rate == 0 {
        return Err(format!("{what} has a sample rate of 0"));
    }
    let samples = match audio.count {
        0 => Vec::new(),
        count => std::slice::from_raw_parts(audio.samples, count).to_vec(),
    };
    Ok(Sound::new(samples, audio.sample_rate))
}

/// A reference recording, where there is one.
unsafe fn reference_in(audio: *const WaifuAudio) -> Result<Option<Sound>, String> {
    match audio.is_null() {
        true => Ok(None),
        false => Ok(Some(audio_in(audio, "reference")?)),
    }
}

/// `sound` as C reads it, for as long as `sound` is.
fn audio_out(sound: &Sound) -> WaifuAudio {
    WaifuAudio {
        samples: sound.samples.as_ptr(),
        count: sound.samples.len(),
        sample_rate: sound.rate,
    }
}

fn audio_completion_to(
    gate: &Gate,
    user_data: UserData,
    callback: WaifuAudioCallback,
) -> engine::OnComplete<Sound> {
    let gate = gate.clone();
    Box::new(move |ended: Ended<Sound>| {
        let Some(callback) = callback else {
            return;
        };
        gate.wait();
        let (code, message) = status_of(&ended);
        let audio = match &ended {
            Ended::Done(sound) => Some(audio_out(sound)),
            _ => None,
        };
        let result = WaifuAudioResult {
            status: status(code, &message),
            audio: audio
                .as_ref()
                .map_or(ptr::null(), |audio| audio as *const _),
        };
        let user_data = user_data;
        unsafe { callback(user_data.0, &result) };
    })
}

#[no_mangle]
pub unsafe extern "C" fn waifu_pipeline_run_text_to_speech_async(
    pipeline: *mut WaifuPipeline,
    request: *const WaifuTextToSpeechRequest,
    user_data: *mut c_void,
    on_progress: WaifuProgressCallback,
    on_complete: WaifuAudioCallback,
) -> WaifuJob {
    let gate = Gate::default();
    submitted(&gate, || {
        let pipeline = pipeline_for(pipeline, WAIFU_PIPELINE_TEXT_TO_SPEECH)?;
        let size = request.as_ref().map_or(0, |request| request.struct_size);
        let least = std::mem::offset_of!(WaifuTextToSpeechRequest, transcript);
        let request = grown_request_in(request, size, least, "WaifuTextToSpeechRequest")?;
        let request = SpeakRequest {
            text: required(request.text, "text")?,
            like: reference_in(request.reference)?,
            speed: request.speed,
            temperature: request.temperature,
            style: string_in(request.style, "style")?,
            transcript: string_in(request.transcript, "transcript")?,
            seed: request.seed,
        };
        let user_data = UserData(user_data);
        Ok(pipeline.engine.speak(
            request,
            progress_to(&gate, user_data, on_progress),
            audio_completion_to(&gate, user_data, on_complete),
        )?)
    })
}

#[no_mangle]
pub unsafe extern "C" fn waifu_pipeline_run_voice_conversion_async(
    pipeline: *mut WaifuPipeline,
    request: *const WaifuVoiceConversionRequest,
    user_data: *mut c_void,
    on_progress: WaifuProgressCallback,
    on_complete: WaifuAudioCallback,
) -> WaifuJob {
    let gate = Gate::default();
    submitted(&gate, || {
        let pipeline = pipeline_for(pipeline, WAIFU_PIPELINE_VOICE_CONVERSION)?;
        let size = request.as_ref().map_or(0, |request| request.struct_size);
        let request = request_in(request, size, "WaifuVoiceConversionRequest")?;
        let request = VoiceConversionRequest {
            source: audio_in(request.source, "source")?,
            reference: audio_in(request.reference, "reference")?,
            steps: request.steps,
            convert_style: request.convert_style,
            seed: request.seed,
        };
        let user_data = UserData(user_data);
        Ok(pipeline.engine.voice_conversion(
            request,
            progress_to(&gate, user_data, on_progress),
            audio_completion_to(&gate, user_data, on_complete),
        )?)
    })
}

// -- streaming text to speech ----------------------------------------------------------------

/// Text to speech with the audio handed over while the rest is said -- CosyVoice3's a hop of
/// tokens at a time, as upstream's stream=True -- for a caller that plays the start at once. A
/// load is refused at the call for a model that does not stream, and ends with WAIFU_ERR_MODEL
/// where the device cannot: anywhere but CUDA at half, on Ampere or later.
#[repr(C)]
pub struct WaifuStreamingTextToSpeechRequest {
    pub struct_size: u32,
    pub text: *const c_char,
    /// A recording to sound like, or NULL.
    pub reference: *const WaifuAudio,
    pub temperature: f32,
    /// Any instruction; NULL or "" for the voice's own way.
    pub style: *const c_char,
    /// No speed: a streaming reading is at the speed it is said, as upstream's is.
    pub seed: u64,
    /// As WaifuTextToSpeechRequest's.
    pub transcript: *const c_char,
}

/// A part of the reading's audio, in order; the samples are valid until it returns.
pub type WaifuAudioPartCallback =
    Option<unsafe extern "C" fn(user_data: *mut c_void, part: *const WaifuAudio)>;

/// The engine's part callback, calling the caller's.
fn audio_parts_to(
    gate: &Gate,
    user_data: UserData,
    callback: WaifuAudioPartCallback,
) -> engine::OnAudio {
    let gate = gate.clone();
    Box::new(move |part: &Sound| {
        let Some(callback) = callback else {
            return;
        };
        gate.wait();
        let audio = audio_out(part);
        let user_data = user_data;
        unsafe { callback(user_data.0, &audio) };
    })
}

/// on_part for each part as it is made; on_complete with the whole of it, the parts end to end.
#[no_mangle]
pub unsafe extern "C" fn waifu_pipeline_run_streaming_text_to_speech_async(
    pipeline: *mut WaifuPipeline,
    request: *const WaifuStreamingTextToSpeechRequest,
    user_data: *mut c_void,
    on_progress: WaifuProgressCallback,
    on_part: WaifuAudioPartCallback,
    on_complete: WaifuAudioCallback,
) -> WaifuJob {
    let gate = Gate::default();
    submitted(&gate, || {
        let pipeline = pipeline_for(pipeline, WAIFU_PIPELINE_STREAMING_TEXT_TO_SPEECH)?;
        let size = request.as_ref().map_or(0, |request| request.struct_size);
        let least = std::mem::offset_of!(WaifuStreamingTextToSpeechRequest, transcript);
        let request = grown_request_in(request, size, least, "WaifuStreamingTextToSpeechRequest")?;
        let request = SpeakRequest {
            text: required(request.text, "text")?,
            like: reference_in(request.reference)?,
            speed: 1.0,
            temperature: request.temperature,
            style: string_in(request.style, "style")?,
            transcript: string_in(request.transcript, "transcript")?,
            seed: request.seed,
        };
        let user_data = UserData(user_data);
        Ok(pipeline.engine.speak_streaming(
            request,
            progress_to(&gate, user_data, on_progress),
            audio_parts_to(&gate, user_data, on_part),
            audio_completion_to(&gate, user_data, on_complete),
        )?)
    })
}

// -- layer decomposition ---------------------------------------------------------------------

#[repr(C)]
pub struct WaifuRgbaImage {
    pub width: u32,
    pub height: u32,
    /// width * height * 4, not premultiplied.
    pub rgba: *const u8,
    pub len: usize,
}

#[repr(C)]
pub struct WaifuLayerDecompositionRequest {
    pub struct_size: u32,
    /// Any size, centred and scaled by the engine; transparent is fine.
    pub image: *const WaifuRgbaImage,
    /// 0 for the model's own.
    pub steps: u32,
    /// 0 for the model's own.
    pub depth_steps: u32,
    /// A multiple of LAYER_DECOMPOSITION_RESOLUTION_MULTIPLE_OF; 0 for 1280.
    pub resolution: u32,
    /// Hands, eyes, eyebrows, ears into a -l and a -r layer.
    pub split_pairs: bool,
    /// Hair, hands, top and bottom also cut into a near and a far half.
    pub stratify_layers: bool,
    pub seed: u64,
}

#[repr(C)]
pub struct WaifuLayer {
    pub r#type: WaifuLayerType,
    /// Unique within the result, from 1; it means nothing else.
    pub id: u32,
    /// "topwear"; its halves "topwear-0" and "topwear-1"; its depth "topwear_depth".
    pub name: *const c_char,
    /// Where its top left corner is on the canvas.
    pub left: i32,
    pub top: i32,
    /// Only as large as what is drawn. A depth layer: the same size and place as its layer, R, G
    /// and B the depth, 0 nearest to 255 farthest, under its layer's alpha.
    pub image: WaifuRgbaImage,
    /// How far away it is: 0 nearest, 1 farthest. A near half has its part's depth, not its
    /// pixels': that is what keeps it in its part's place in the stratified stack (see the
    /// result). A depth layer has its layer's.
    pub depth: f32,
    /// The id of the layer this one is made from: a half's part, a depth layer's layer; 0 for a
    /// part.
    pub derived_from_id: u32,
}

/// `layers` is in one order, which the library never changes: each part, bottom first -- which is
/// the stack, unstratified -- then, with stratify_layers, each part's halves, near before far;
/// then a depth layer for each of those, in their order. A part a half names in derived_from_id
/// is a split part.
///
/// The stratified stack is every part that was not split and every half, bottom first: from the
/// largest depth to the smallest; at one depth, a far half under a near one; then by their order
/// in `layers`. No depth layer is in either stack.
#[repr(C)]
pub struct WaifuLayerDecompositionResult {
    pub status: WaifuStatus,
    /// The side of the square canvas; 0 unless WAIFU_OK.
    pub canvas: u32,
    pub layers: *const WaifuLayer,
    pub layer_count: usize,
}

pub type WaifuLayerDecompositionCallback = Option<
    unsafe extern "C" fn(user_data: *mut c_void, result: *const WaifuLayerDecompositionResult),
>;

unsafe fn rgba_in(image: *const WaifuRgbaImage, what: &str) -> Result<RgbaImage, String> {
    let image = image.as_ref().ok_or_else(|| format!("{what} is NULL"))?;
    if image.rgba.is_null() && image.len > 0 {
        return Err(format!("{what} has no pixels"));
    }
    let rgba = match image.len {
        0 => Vec::new(),
        len => std::slice::from_raw_parts(image.rgba, len).to_vec(),
    };
    Ok(RgbaImage {
        width: image.width,
        height: image.height,
        rgba,
    })
}

#[no_mangle]
pub unsafe extern "C" fn waifu_pipeline_run_layer_decomposition_async(
    pipeline: *mut WaifuPipeline,
    request: *const WaifuLayerDecompositionRequest,
    user_data: *mut c_void,
    on_progress: WaifuProgressCallback,
    on_complete: WaifuLayerDecompositionCallback,
) -> WaifuJob {
    let gate = Gate::default();
    submitted(&gate, || {
        let pipeline = pipeline_for(pipeline, WAIFU_PIPELINE_LAYER_DECOMPOSITION)?;
        let size = request.as_ref().map_or(0, |request| request.struct_size);
        let request = request_in(request, size, "WaifuLayerDecompositionRequest")?;
        let request = LayersRequest {
            image: rgba_in(request.image, "image")?,
            steps: request.steps,
            depth_steps: request.depth_steps,
            resolution: request.resolution,
            split_pairs: request.split_pairs,
            stratify: request.stratify_layers,
            seed: request.seed,
        };

        let user_data = UserData(user_data);
        let on_complete = {
            let gate = gate.clone();
            Box::new(move |ended: Ended<engine::Layers>| {
                let Some(callback) = on_complete else {
                    return;
                };
                gate.wait();
                let (code, message) = status_of(&ended);
                let (canvas, layers) = match &ended {
                    Ended::Done(done) => (done.canvas, &done.layers[..]),
                    _ => (0, &[][..]),
                };
                // The names have to outlive the result, and be NUL terminated: kept here until the
                // callback returns.
                let names: Vec<CString> = layers
                    .iter()
                    .map(|layer| support::c_string(&layer.name))
                    .collect();
                let out: Vec<WaifuLayer> = names
                    .iter()
                    .zip(layers)
                    .map(|(name, layer)| WaifuLayer {
                        r#type: match layer.kind {
                            LayerKind::Part => WAIFU_LAYER_PART,
                            LayerKind::StratifiedNear => WAIFU_LAYER_STRATIFIED_NEAR,
                            LayerKind::StratifiedFar => WAIFU_LAYER_STRATIFIED_FAR,
                            LayerKind::Depth => WAIFU_LAYER_DEPTH,
                        },
                        id: layer.id,
                        name: name.as_ptr(),
                        left: layer.left,
                        top: layer.top,
                        image: WaifuRgbaImage {
                            width: layer.image.width,
                            height: layer.image.height,
                            rgba: layer.image.rgba.as_ptr(),
                            len: layer.image.rgba.len(),
                        },
                        depth: layer.depth,
                        derived_from_id: layer.derived_from_id.unwrap_or(0),
                    })
                    .collect();
                let result = WaifuLayerDecompositionResult {
                    status: status(code, &message),
                    canvas,
                    layers: first(&out),
                    layer_count: out.len(),
                };
                let user_data = user_data;
                unsafe { callback(user_data.0, &result) };
            })
        };
        Ok(pipeline.engine.layers(
            request,
            progress_to(&gate, user_data, on_progress),
            on_complete,
        )?)
    })
}

// -- speech to text --------------------------------------------------------------------------

#[repr(C)]
pub struct WaifuSpeechToTextRequest {
    pub struct_size: u32,
    /// Any rate, any length; resampled and cut into pieces by the engine.
    pub audio: *const WaifuAudio,
    /// The language it is in, one of SPEECH_TO_TEXT_LANGUAGES -- "English", "chinese" -- or NULL
    /// or "" for the model to tell.
    pub language: *const c_char,
    /// Words that help it hear: names, terms, the subject; NULL or "".
    pub context: *const c_char,
}

#[repr(C)]
pub struct WaifuSpeechToTextResult {
    pub status: WaifuStatus,
    /// What was said; "" for nothing said; NULL unless WAIFU_OK.
    pub text: *const c_char,
    /// The language it was said in, as the model names it -- "Chinese" -- or several, comma
    /// separated, for a recording that changes; "" where none was heard. NULL unless WAIFU_OK.
    pub language: *const c_char,
}

pub type WaifuSpeechToTextCallback =
    Option<unsafe extern "C" fn(user_data: *mut c_void, result: *const WaifuSpeechToTextResult)>;

#[no_mangle]
pub unsafe extern "C" fn waifu_pipeline_run_speech_to_text_async(
    pipeline: *mut WaifuPipeline,
    request: *const WaifuSpeechToTextRequest,
    user_data: *mut c_void,
    on_progress: WaifuProgressCallback,
    on_complete: WaifuSpeechToTextCallback,
) -> WaifuJob {
    let gate = Gate::default();
    submitted(&gate, || {
        let pipeline = pipeline_for(pipeline, WAIFU_PIPELINE_SPEECH_TO_TEXT)?;
        let size = request.as_ref().map_or(0, |request| request.struct_size);
        let request = request_in(request, size, "WaifuSpeechToTextRequest")?;
        let request = TranscribeRequest {
            recording: audio_in(request.audio, "audio")?,
            language: string_in(request.language, "language")?.filter(|l| !l.is_empty()),
            context: string_in(request.context, "context")?.unwrap_or_default(),
        };

        let user_data = UserData(user_data);
        let on_complete = {
            let gate = gate.clone();
            Box::new(move |ended: Ended<crate::Transcript>| {
                let Some(callback) = on_complete else {
                    return;
                };
                gate.wait();
                let (code, message) = status_of(&ended);
                // Kept here until the callback returns, NUL terminated.
                let strings = match &ended {
                    Ended::Done(transcript) => Some((
                        support::c_string(&transcript.text),
                        support::c_string(&transcript.language),
                    )),
                    _ => None,
                };
                let result = WaifuSpeechToTextResult {
                    status: status(code, &message),
                    text: strings
                        .as_ref()
                        .map_or(ptr::null(), |(text, _)| text.as_ptr()),
                    language: strings
                        .as_ref()
                        .map_or(ptr::null(), |(_, language)| language.as_ptr()),
                };
                let user_data = user_data;
                unsafe { callback(user_data.0, &result) };
            })
        };
        Ok(pipeline.engine.transcribe(
            request,
            progress_to(&gate, user_data, on_progress),
            on_complete,
        )?)
    })
}

#[cfg(test)]
mod tests;
