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

//! The C API, ABI 6: pipelines, requests, tasks, models, devices and the log, as `extern "C"`.
//! Everything outside this crate goes through these. docs/ffi-abi6.md is the contract; it changes
//! only with a human's approval, and so does this.
//!
//! Nothing with any logic in it lives here. This converts -- C strings to `&str`, a request's
//! attributes to the engine's requests, what comes back to views a task keeps, panics to status
//! codes. Nothing of a caller's runs on a pipeline's thread: the log's callback is the only one.

#![allow(non_camel_case_types)]
#![allow(clippy::missing_safety_doc)]

use std::collections::BTreeMap;
use std::ffi::{c_char, c_void, CString};
use std::ptr;

mod request;
mod support;
mod task;

pub use request::*;
use support::{caught, refusable, required, Refused};
pub use support::{
    WaifuLog, WaifuLogCallback, WaifuLogLevel, WaifuStage, WaifuStatusCode, WAIFU_CANCELLED,
    WAIFU_ERR_DEVICE_UNAVAILABLE, WAIFU_ERR_INTERNAL, WAIFU_ERR_INVALID_ARGUMENT, WAIFU_ERR_MODEL,
    WAIFU_ERR_NOT_LOADED, WAIFU_ERR_UNKNOWN_ATTRIBUTE, WAIFU_ERR_UNKNOWN_MODEL, WAIFU_LOG_DEBUG,
    WAIFU_LOG_ERROR, WAIFU_LOG_FATAL, WAIFU_LOG_INFO, WAIFU_LOG_WARNING, WAIFU_OK,
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

pub use task::*;

use crate::engine::{self, Engine, EngineOptions, Failure, RgbaImage, Stage};
use crate::flint::MemoryBudget;
use crate::model_info::{self, ModelInfo, Support};
use crate::runtime::{self, DeviceOption, Runtime};
use crate::wav::Sound;
use crate::Device;

/// The major version: bumped for whatever an older caller would misread -- a number that means
/// something else, a function or a type that changed or went, an attribute's default.
///
/// 4: pipelines made for one pipeline type, models and devices asked about by attribute, no JSON.
/// 5: the text-to-speech requests' reference_transcript, after reference.
/// 6: requests set an attribute at a time, tasks read rather than callbacks, a minor version.
pub const WAIFU_ABI_VERSION_MAJOR: u32 = 6;
/// The minor version: bumped for what an older caller never uses -- an attribute, a stage, an event
/// a request has to ask for, a function. Back to 0 with each major.
///
/// 6.1: the text-to-speech readings' visemes -- the VISEMES request and task attributes,
/// SUPPORTS_TEXT_TO_SPEECH_VISEMES, WaifuViseme, waifu_task_get_int_array_attribute.
pub const WAIFU_ABI_VERSION_MINOR: u32 = 1;
/// (major << 16) | minor, as waifu_abi_version says it.
pub const WAIFU_ABI_VERSION: u32 = (WAIFU_ABI_VERSION_MAJOR << 16) | WAIFU_ABI_VERSION_MINOR;

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

/// What can be asked of a model. SUPPORTS_ attributes are integers, 1 or 0; where 0, the string
/// attribute of the same number is why, in words to show, and is never NULL. A SUPPORTS_ attribute
/// this library does not know is absent, which is neither 1 nor 0. The rest are an integer or a
/// string, as marked. SUPPORTS_<type> has the number of WAIFU_PIPELINE_<type>.
pub type WaifuModelAttribute = i32;
pub const WAIFU_MODEL_SUPPORTS_IMAGE_GENERATION: WaifuModelAttribute = 1;
pub const WAIFU_MODEL_SUPPORTS_TEXT_TO_SPEECH: WaifuModelAttribute = 2;
pub const WAIFU_MODEL_SUPPORTS_VOICE_CONVERSION: WaifuModelAttribute = 3;
pub const WAIFU_MODEL_SUPPORTS_LAYER_DECOMPOSITION: WaifuModelAttribute = 4;
pub const WAIFU_MODEL_SUPPORTS_SPEECH_TO_TEXT: WaifuModelAttribute = 5;
/// The model streams; the device is checked at load.
pub const WAIFU_MODEL_SUPPORTS_STREAMING_TEXT_TO_SPEECH: WaifuModelAttribute = 6;
/// img2img.
pub const WAIFU_MODEL_SUPPORTS_IMAGE_GENERATION_FROM_START_IMAGE: WaifuModelAttribute = 100;
/// A ControlNet.
pub const WAIFU_MODEL_SUPPORTS_IMAGE_GENERATION_WITH_CONTROL_IMAGE: WaifuModelAttribute = 101;
/// Images the prompt names as <|key|>.
pub const WAIFU_MODEL_SUPPORTS_IMAGE_GENERATION_WITH_PROMPT_IMAGES: WaifuModelAttribute = 102;
/// In the voice of a recording.
pub const WAIFU_MODEL_SUPPORTS_TEXT_TO_SPEECH_LIKE_REFERENCE: WaifuModelAttribute = 200;
/// CosyVoice3 does not.
pub const WAIFU_MODEL_SUPPORTS_TEXT_TO_SPEECH_WITHOUT_REFERENCE: WaifuModelAttribute = 201;
/// Follows an instruction such as "speak happily".
pub const WAIFU_MODEL_SUPPORTS_TEXT_TO_SPEECH_WITH_STYLE: WaifuModelAttribute = 202;
/// A reference without its reference_transcript: CosyVoice3 does not, unless a style is given.
pub const WAIFU_MODEL_SUPPORTS_TEXT_TO_SPEECH_WITHOUT_REFERENCE_TRANSCRIPT: WaifuModelAttribute =
    203;
/// The visemes of what it says, with TEXT_TO_SPEECH_VISEMES or STREAMING_TEXT_TO_SPEECH_VISEMES:
/// a CosyVoice3 package with the viseme model. Since 6.1.
pub const WAIFU_MODEL_SUPPORTS_TEXT_TO_SPEECH_VISEMES: WaifuModelAttribute = 204;
/// The manner as well as the voice.
pub const WAIFU_MODEL_SUPPORTS_VOICE_CONVERSION_OF_STYLE: WaifuModelAttribute = 300;
/// Integer, bytes; absent until every file is here.
pub const WAIFU_MODEL_SIZE_ON_DISK: WaifuModelAttribute = 1000;
/// Integer, of what it says or converts.
pub const WAIFU_MODEL_SAMPLE_RATE: WaifuModelAttribute = 1001;
/// Integer: 16, or 32 for SDXL and Qwen-Image.
pub const WAIFU_MODEL_IMAGE_GENERATION_WIDTH_HEIGHT_MULTIPLE_OF: WaifuModelAttribute = 1100;
/// String: "Euler".
pub const WAIFU_MODEL_IMAGE_GENERATION_SAMPLER: WaifuModelAttribute = 1101;
/// Integer.
pub const WAIFU_MODEL_IMAGE_GENERATION_MAX_PROMPT_IMAGES: WaifuModelAttribute = 1102;
/// Integer: 64.
pub const WAIFU_MODEL_LAYER_DECOMPOSITION_RESOLUTION_MULTIPLE_OF: WaifuModelAttribute = 1400;
/// String: "Chinese,English,...", comma separated.
pub const WAIFU_MODEL_SPEECH_TO_TEXT_LANGUAGES: WaifuModelAttribute = 1500;

/// What can be asked of a device. All of it is this process's, not the card as other programs see
/// it.
pub type WaifuDeviceAttribute = i32;
/// The most it may hold there: the memory limit, or the device's memory.
pub const WAIFU_DEVICE_MEMORY_BUDGET: WaifuDeviceAttribute = 1;
/// What its allocator holds now, what it keeps for reuse included.
pub const WAIFU_DEVICE_MEMORY_HELD: WaifuDeviceAttribute = 2;
/// BUDGET - HELD, never below 0.
pub const WAIFU_DEVICE_MEMORY_AVAILABLE: WaifuDeviceAttribute = 3;
/// What waifu_device_set_memory_limit set; absent where nothing did.
pub const WAIFU_DEVICE_MEMORY_LIMIT: WaifuDeviceAttribute = 4;

pub type WaifuLayerType = i32;
/// A part as the model drew it.
pub const WAIFU_LAYER_PART: WaifuLayerType = 0;
/// stratify_layers: the near half of a part.
pub const WAIFU_LAYER_STRATIFIED_NEAR: WaifuLayerType = 1;
/// stratify_layers: the far half of a part.
pub const WAIFU_LAYER_STRATIFIED_FAR: WaifuLayerType = 2;
/// The depth of the layer derived_from_id names, pixel by pixel.
pub const WAIFU_LAYER_DEPTH: WaifuLayerType = 3;

/// A mouth shape, one for each frame of a reading's VISEMES: the sounds a face cannot tell apart
/// by looking are one.
pub type WaifuViseme = i32;
/// The five open vowels, as a Japanese face draws them.
pub const WAIFU_VISEME_A: WaifuViseme = 0;
pub const WAIFU_VISEME_E: WaifuViseme = 1;
pub const WAIFU_VISEME_I: WaifuViseme = 2;
pub const WAIFU_VISEME_O: WaifuViseme = 3;
pub const WAIFU_VISEME_U: WaifuViseme = 4;
/// The schwa, @: half open and not spread -- Mandarin's e and en, English's unstressed vowel.
pub const WAIFU_VISEME_SCHWA: WaifuViseme = 5;
/// The lips pressed shut: b, p and m.
pub const WAIFU_VISEME_MBP: WaifuViseme = 6;
/// Relaxed and lightly closed. A shape, not silence: whether anybody is speaking is the sound's.
pub const WAIFU_VISEME_REST: WaifuViseme = 7;

// -- the library -----------------------------------------------------------------------------

/// (major << 16) | minor. A caller built against 6.m runs on any 6.n with n >= m.
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

/// false for the processor, a device that cannot be used, or an attribute this library does not
/// know. Asking CUDA builds a context on the card, a few hundred megabytes of it.
#[no_mangle]
pub unsafe extern "C" fn waifu_device_get_int_attribute(
    device: WaifuDevice,
    attribute: WaifuDeviceAttribute,
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
        let answer = match attribute {
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
    ints: BTreeMap<WaifuModelAttribute, i64>,
    /// Each lives until the model is freed.
    strings: BTreeMap<WaifuModelAttribute, CString>,
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
            WAIFU_MODEL_SUPPORTS_TEXT_TO_SPEECH_WITHOUT_REFERENCE_TRANSCRIPT,
            &detail(speaking.map(|s| &s.without_reference_transcript), &speech),
        );
        model.supports(
            WAIFU_MODEL_SUPPORTS_TEXT_TO_SPEECH_WITH_STYLE,
            &detail(speaking.map(|s| &s.with_style), &speech),
        );
        model.supports(
            WAIFU_MODEL_SUPPORTS_TEXT_TO_SPEECH_VISEMES,
            &detail(speaking.map(|s| &s.with_visemes), &speech),
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

    fn int(&mut self, attribute: WaifuModelAttribute, value: i64) {
        self.ints.insert(attribute, value);
    }

    fn string(&mut self, attribute: WaifuModelAttribute, value: &str) {
        self.strings.insert(attribute, support::c_string(value));
    }

    /// 1, or 0 and why.
    fn supports(&mut self, attribute: WaifuModelAttribute, support: &Support) {
        match support {
            Ok(()) => self.int(attribute, 1),
            Err(why) => {
                self.int(attribute, 0);
                self.string(attribute, why);
            }
        }
    }

    /// Why the model cannot be loaded for a pipeline of type `task`, where it cannot.
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

/// Whether a model does what a pipeline type is made for, and why not.
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

/// false where the model has no such attribute or this library does not know it; *value untouched.
#[no_mangle]
pub unsafe extern "C" fn waifu_model_get_int_attribute(
    model: *const WaifuModel,
    attribute: WaifuModelAttribute,
    value: *mut i64,
) -> bool {
    caught(false, || {
        let (Some(model), false) = (model.as_ref(), value.is_null()) else {
            return false;
        };
        match model.ints.get(&attribute) {
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
pub unsafe extern "C" fn waifu_model_get_string_attribute(
    model: *const WaifuModel,
    attribute: WaifuModelAttribute,
) -> *const c_char {
    caught(ptr::null(), || {
        model
            .as_ref()
            .and_then(|model| model.strings.get(&attribute))
            .map_or(ptr::null(), |answer| answer.as_ptr())
    })
}

// -- pipelines -------------------------------------------------------------------------------

/// One pipeline type, one thread, at most one model: its tasks run one at a time, in the order
/// asked.
pub struct WaifuPipeline {
    engine: Engine,
    task: WaifuPipelineType,
}

/// A pipeline type as a sentence names it.
pub(crate) fn task_name(task: WaifuPipelineType) -> &'static str {
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
/// task of the pipeline has ended -- WAIFU_CANCELLED for what had not -- and the tasks are still
/// the caller's to free. NULL is fine.
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

/// Every task of the pipeline, waiting or running, is cancelled.
#[no_mangle]
pub unsafe extern "C" fn waifu_pipeline_cancel_all(pipeline: *mut WaifuPipeline) {
    if let Some(pipeline) = pipeline.as_ref() {
        caught((), || pipeline.engine.cancel_all());
    }
}

pub(crate) fn stage_out(stage: Stage) -> WaifuStage {
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

pub(crate) fn code_of(failure: Failure) -> WaifuStatusCode {
    match failure {
        Failure::UnknownModel => WAIFU_ERR_UNKNOWN_MODEL,
        Failure::Model => WAIFU_ERR_MODEL,
        Failure::NotLoaded => WAIFU_ERR_NOT_LOADED,
        Failure::Internal => WAIFU_ERR_INTERNAL,
    }
}

/// Reads the model's weights onto the pipeline's device, dropping the model before first, so two
/// are never on it at once. Refused -- NULL -- for a model the pipeline's type is not supported by
/// (WAIFU_MODEL_SUPPORTS_<type>). The call copies the manifest, not the weights, which the
/// pipeline's thread reads from the files: `model` may be freed as soon as it returns. A run may
/// be asked for straight after; it waits its turn. "tones", the stand-in voice the tests use,
/// opens as a model too.
#[no_mangle]
pub unsafe extern "C" fn waifu_pipeline_load_model(
    pipeline: *mut WaifuPipeline,
    model: *const WaifuModel,
) -> *mut WaifuTask {
    refusable(ptr::null_mut(), || {
        let pipeline = pipeline.as_ref().ok_or("the pipeline is NULL")?;
        let model = model.as_ref().ok_or("the model is NULL")?;
        if let Some(why) = model.unsupported(pipeline.task) {
            return Err(format!(
                "a {} pipeline cannot load it: {why}",
                task_name(pipeline.task)
            )
            .into());
        }
        let started = task::Started::new();
        let (on_progress, on_complete) = (started.on_progress(), started.on_loaded());
        let id = match pipeline.task {
            WAIFU_PIPELINE_STREAMING_TEXT_TO_SPEECH => {
                pipeline
                    .engine
                    .load_for_streaming(&model.path, on_progress, on_complete)
            }
            _ => pipeline.engine.load(&model.path, on_progress, on_complete),
        };
        Ok(started.task(id, pipeline.engine.canceller(), None))
    })
}

/// Drops the model; the pipeline then has none.
#[no_mangle]
pub unsafe extern "C" fn waifu_pipeline_unload_model(
    pipeline: *mut WaifuPipeline,
) -> *mut WaifuTask {
    refusable(ptr::null_mut(), || {
        let pipeline = pipeline.as_ref().ok_or("the pipeline is NULL")?;
        let started = task::Started::new();
        let id = pipeline.engine.unload(started.on_loaded());
        Ok(started.task(id, pipeline.engine.canceller(), None))
    })
}

/// Runs the request on the pipeline's thread, in its turn, and returns at once. Refused -- NULL,
/// with the last error and its code -- for a request of another pipeline type, a required attribute
/// unset, or an attribute that contradicts another; a run with no model loaded is not refused, and
/// ends with WAIFU_ERR_NOT_LOADED.
#[no_mangle]
pub unsafe extern "C" fn waifu_pipeline_run(
    pipeline: *mut WaifuPipeline,
    request: *const WaifuRequest,
) -> *mut WaifuTask {
    refusable(ptr::null_mut(), || {
        let pipeline = pipeline.as_ref().ok_or("the pipeline is NULL")?;
        let request = request.as_ref().ok_or("the request is NULL")?;
        if request.task() != pipeline.task {
            return Err(format!(
                "this pipeline is made for {}, and the request is for {}",
                task_name(pipeline.task),
                task_name(request.task())
            )
            .into());
        }
        let engine = &pipeline.engine;
        let started = task::Started::new();
        let progress = started.on_progress();
        let id = match pipeline.task {
            WAIFU_PIPELINE_IMAGE_GENERATION => {
                engine.draw(request.draw()?, progress, started.on_image())?
            }
            WAIFU_PIPELINE_TEXT_TO_SPEECH => engine.speak_with(
                request.speak()?,
                progress,
                None,
                request.wants_visemes().then(|| started.on_visemes()),
                started.on_audio(),
            )?,
            WAIFU_PIPELINE_STREAMING_TEXT_TO_SPEECH => engine.speak_with(
                request.speak_streaming()?,
                progress,
                Some(started.on_part()),
                request.wants_visemes().then(|| started.on_visemes()),
                started.on_audio(),
            )?,
            WAIFU_PIPELINE_VOICE_CONVERSION => engine.voice_conversion(
                request.voice_conversion()?,
                progress,
                started.on_audio(),
            )?,
            WAIFU_PIPELINE_LAYER_DECOMPOSITION => {
                engine.layers(request.layers()?, progress, started.on_layers())?
            }
            WAIFU_PIPELINE_SPEECH_TO_TEXT => {
                engine.transcribe(request.transcribe()?, progress, started.on_transcript())?
            }
            _ => return Err("the pipeline is of a type this library does not know".into()),
        };
        Ok(started.task(id, engine.canceller(), Some(pipeline.task)))
    })
}

// -- the four structs ------------------------------------------------------------------------

// Frozen: a change to any of these is a major version. WaifuLog is in support.rs.

#[repr(C)]
pub struct WaifuImage {
    pub width: u32,
    pub height: u32,
    /// width * height * 3, rows top to bottom.
    pub rgb: *const u8,
    pub len: usize,
}

#[repr(C)]
pub struct WaifuRgbaImage {
    pub width: u32,
    pub height: u32,
    /// width * height * 4, not premultiplied.
    pub rgba: *const u8,
    pub len: usize,
}

#[repr(C)]
pub struct WaifuAudio {
    /// Mono, -1..1.
    pub samples: *const f32,
    pub count: usize,
    pub sample_rate: u32,
}

pub(crate) unsafe fn image_in(
    image: *const WaifuImage,
    what: &str,
) -> Result<engine::Image, String> {
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

pub(crate) unsafe fn rgba_in(
    image: *const WaifuRgbaImage,
    what: &str,
) -> Result<RgbaImage, String> {
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

pub(crate) unsafe fn audio_in(audio: *const WaifuAudio, what: &str) -> Result<Sound, String> {
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

#[cfg(test)]
mod tests;
