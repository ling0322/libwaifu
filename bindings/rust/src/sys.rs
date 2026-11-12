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

//! The mirror of `waifu.h`, ABI 6, and the table of its functions, for whatever the crate does not
//! wrap.
//!
//! Every C enum here is a `c_int`, never a Rust `enum`: a number a newer library returns that a
//! Rust enum has no variant for would be undefined behaviour before anything could map it. The
//! safe types convert from the integer. tests/layout.rs holds every struct against the engine's.

#![allow(non_camel_case_types)]

use std::ffi::{c_char, c_int, c_void};

/// The major version this crate speaks: a library of another major is refused.
pub const WAIFU_ABI_VERSION_MAJOR: u32 = 6;
/// The minor version this crate was written against: a library of an older minor is refused.
pub const WAIFU_ABI_VERSION_MINOR: u32 = 1;
pub const WAIFU_ABI_VERSION: u32 = (WAIFU_ABI_VERSION_MAJOR << 16) | WAIFU_ABI_VERSION_MINOR;

// -- enums -----------------------------------------------------------------------------------

pub type WaifuLogLevel = c_int;
pub const WAIFU_LOG_DEBUG: WaifuLogLevel = 0;
pub const WAIFU_LOG_INFO: WaifuLogLevel = 1;
pub const WAIFU_LOG_WARNING: WaifuLogLevel = 2;
pub const WAIFU_LOG_ERROR: WaifuLogLevel = 3;
pub const WAIFU_LOG_FATAL: WaifuLogLevel = 4;

pub type WaifuDevice = c_int;
pub const WAIFU_DEVICE_AUTO: WaifuDevice = 0;
pub const WAIFU_DEVICE_CPU: WaifuDevice = 1;
pub const WAIFU_DEVICE_METAL: WaifuDevice = 2;
pub const WAIFU_DEVICE_CUDA: WaifuDevice = 3;
pub const WAIFU_DEVICE_VULKAN: WaifuDevice = 5;

pub type WaifuPipelineType = c_int;
pub const WAIFU_PIPELINE_IMAGE_GENERATION: WaifuPipelineType = 1;
pub const WAIFU_PIPELINE_TEXT_TO_SPEECH: WaifuPipelineType = 2;
pub const WAIFU_PIPELINE_VOICE_CONVERSION: WaifuPipelineType = 3;
pub const WAIFU_PIPELINE_LAYER_DECOMPOSITION: WaifuPipelineType = 4;
pub const WAIFU_PIPELINE_SPEECH_TO_TEXT: WaifuPipelineType = 5;
pub const WAIFU_PIPELINE_STREAMING_TEXT_TO_SPEECH: WaifuPipelineType = 6;

pub type WaifuStatusCode = c_int;
pub const WAIFU_OK: WaifuStatusCode = 0;
pub const WAIFU_CANCELLED: WaifuStatusCode = 1;
pub const WAIFU_ERR_INVALID_ARGUMENT: WaifuStatusCode = 2;
pub const WAIFU_ERR_UNKNOWN_MODEL: WaifuStatusCode = 3;
pub const WAIFU_ERR_MODEL: WaifuStatusCode = 5;
pub const WAIFU_ERR_NOT_LOADED: WaifuStatusCode = 6;
pub const WAIFU_ERR_INTERNAL: WaifuStatusCode = 7;
pub const WAIFU_ERR_DEVICE_UNAVAILABLE: WaifuStatusCode = 8;
pub const WAIFU_ERR_UNKNOWN_ATTRIBUTE: WaifuStatusCode = 9;

pub type WaifuStage = c_int;
pub const WAIFU_STAGE_LOADING_WEIGHTS: WaifuStage = 1;
pub const WAIFU_STAGE_IMAGE_GENERATION_ENCODING_PROMPT: WaifuStage = 100;
pub const WAIFU_STAGE_IMAGE_GENERATION_DENOISING: WaifuStage = 101;
pub const WAIFU_STAGE_IMAGE_GENERATION_DECODING_IMAGE: WaifuStage = 102;
pub const WAIFU_STAGE_TEXT_TO_SPEECH_READING_TEXT: WaifuStage = 200;
pub const WAIFU_STAGE_TEXT_TO_SPEECH_GENERATING_TOKENS: WaifuStage = 201;
pub const WAIFU_STAGE_TEXT_TO_SPEECH_SYNTHESIZING_AUDIO: WaifuStage = 202;
pub const WAIFU_STAGE_VOICE_CONVERSION_READING_RECORDINGS: WaifuStage = 300;
pub const WAIFU_STAGE_VOICE_CONVERSION_GENERATING_TOKENS: WaifuStage = 301;
pub const WAIFU_STAGE_VOICE_CONVERSION_DENOISING: WaifuStage = 302;
pub const WAIFU_STAGE_VOICE_CONVERSION_SYNTHESIZING_AUDIO: WaifuStage = 303;
pub const WAIFU_STAGE_LAYER_DECOMPOSITION_ENCODING_IMAGE: WaifuStage = 400;
pub const WAIFU_STAGE_LAYER_DECOMPOSITION_DENOISING_LAYERS: WaifuStage = 401;
pub const WAIFU_STAGE_LAYER_DECOMPOSITION_DECODING_LAYERS: WaifuStage = 402;
pub const WAIFU_STAGE_LAYER_DECOMPOSITION_ENCODING_HEAD: WaifuStage = 403;
pub const WAIFU_STAGE_LAYER_DECOMPOSITION_DENOISING_HEAD_LAYERS: WaifuStage = 404;
pub const WAIFU_STAGE_LAYER_DECOMPOSITION_DECODING_HEAD_LAYERS: WaifuStage = 405;
pub const WAIFU_STAGE_LAYER_DECOMPOSITION_ENCODING_FOR_DEPTH: WaifuStage = 406;
pub const WAIFU_STAGE_LAYER_DECOMPOSITION_DENOISING_DEPTH: WaifuStage = 407;
pub const WAIFU_STAGE_LAYER_DECOMPOSITION_DECODING_DEPTH: WaifuStage = 408;
pub const WAIFU_STAGE_LAYER_DECOMPOSITION_CUTTING_INTO_PARTS: WaifuStage = 409;
pub const WAIFU_STAGE_SPEECH_TO_TEXT_READING_RECORDING: WaifuStage = 500;
pub const WAIFU_STAGE_SPEECH_TO_TEXT_GENERATING_TOKENS: WaifuStage = 501;
pub const WAIFU_STAGE_STREAMING_TEXT_TO_SPEECH_READING_TEXT: WaifuStage = 600;
pub const WAIFU_STAGE_STREAMING_TEXT_TO_SPEECH_GENERATING_TOKENS: WaifuStage = 601;
pub const WAIFU_STAGE_STREAMING_TEXT_TO_SPEECH_SYNTHESIZING_AUDIO: WaifuStage = 602;

pub type WaifuModelAttribute = c_int;
pub const WAIFU_MODEL_SUPPORTS_IMAGE_GENERATION: WaifuModelAttribute = 1;
pub const WAIFU_MODEL_SUPPORTS_TEXT_TO_SPEECH: WaifuModelAttribute = 2;
pub const WAIFU_MODEL_SUPPORTS_VOICE_CONVERSION: WaifuModelAttribute = 3;
pub const WAIFU_MODEL_SUPPORTS_LAYER_DECOMPOSITION: WaifuModelAttribute = 4;
pub const WAIFU_MODEL_SUPPORTS_SPEECH_TO_TEXT: WaifuModelAttribute = 5;
pub const WAIFU_MODEL_SUPPORTS_STREAMING_TEXT_TO_SPEECH: WaifuModelAttribute = 6;
pub const WAIFU_MODEL_SUPPORTS_IMAGE_GENERATION_FROM_START_IMAGE: WaifuModelAttribute = 100;
pub const WAIFU_MODEL_SUPPORTS_IMAGE_GENERATION_WITH_CONTROL_IMAGE: WaifuModelAttribute = 101;
pub const WAIFU_MODEL_SUPPORTS_IMAGE_GENERATION_WITH_PROMPT_IMAGES: WaifuModelAttribute = 102;
pub const WAIFU_MODEL_SUPPORTS_TEXT_TO_SPEECH_LIKE_REFERENCE: WaifuModelAttribute = 200;
pub const WAIFU_MODEL_SUPPORTS_TEXT_TO_SPEECH_WITHOUT_REFERENCE: WaifuModelAttribute = 201;
pub const WAIFU_MODEL_SUPPORTS_TEXT_TO_SPEECH_WITH_STYLE: WaifuModelAttribute = 202;
pub const WAIFU_MODEL_SUPPORTS_TEXT_TO_SPEECH_WITHOUT_REFERENCE_TRANSCRIPT: WaifuModelAttribute =
    203;
pub const WAIFU_MODEL_SUPPORTS_TEXT_TO_SPEECH_VISEMES: WaifuModelAttribute = 204;
pub const WAIFU_MODEL_SUPPORTS_VOICE_CONVERSION_OF_STYLE: WaifuModelAttribute = 300;
pub const WAIFU_MODEL_SIZE_ON_DISK: WaifuModelAttribute = 1000;
pub const WAIFU_MODEL_SAMPLE_RATE: WaifuModelAttribute = 1001;
pub const WAIFU_MODEL_IMAGE_GENERATION_WIDTH_HEIGHT_MULTIPLE_OF: WaifuModelAttribute = 1100;
pub const WAIFU_MODEL_IMAGE_GENERATION_SAMPLER: WaifuModelAttribute = 1101;
pub const WAIFU_MODEL_IMAGE_GENERATION_MAX_PROMPT_IMAGES: WaifuModelAttribute = 1102;
pub const WAIFU_MODEL_LAYER_DECOMPOSITION_RESOLUTION_MULTIPLE_OF: WaifuModelAttribute = 1400;
pub const WAIFU_MODEL_SPEECH_TO_TEXT_LANGUAGES: WaifuModelAttribute = 1500;

pub type WaifuDeviceAttribute = c_int;
pub const WAIFU_DEVICE_MEMORY_BUDGET: WaifuDeviceAttribute = 1;
pub const WAIFU_DEVICE_MEMORY_HELD: WaifuDeviceAttribute = 2;
pub const WAIFU_DEVICE_MEMORY_AVAILABLE: WaifuDeviceAttribute = 3;
pub const WAIFU_DEVICE_MEMORY_LIMIT: WaifuDeviceAttribute = 4;

pub type WaifuLayerType = c_int;
pub const WAIFU_LAYER_PART: WaifuLayerType = 0;
pub const WAIFU_LAYER_STRATIFIED_NEAR: WaifuLayerType = 1;
pub const WAIFU_LAYER_STRATIFIED_FAR: WaifuLayerType = 2;
pub const WAIFU_LAYER_DEPTH: WaifuLayerType = 3;

pub type WaifuViseme = c_int;
pub const WAIFU_VISEME_A: WaifuViseme = 0;
pub const WAIFU_VISEME_E: WaifuViseme = 1;
pub const WAIFU_VISEME_I: WaifuViseme = 2;
pub const WAIFU_VISEME_O: WaifuViseme = 3;
pub const WAIFU_VISEME_U: WaifuViseme = 4;
pub const WAIFU_VISEME_SCHWA: WaifuViseme = 5;
pub const WAIFU_VISEME_MBP: WaifuViseme = 6;
pub const WAIFU_VISEME_REST: WaifuViseme = 7;

pub type WaifuEventType = c_int;
pub const WAIFU_EVENT_NONE: WaifuEventType = 0;
pub const WAIFU_EVENT_PROGRESS: WaifuEventType = 1;
pub const WAIFU_EVENT_AUDIO_PART: WaifuEventType = 2;
pub const WAIFU_EVENT_END: WaifuEventType = 3;

pub const WAIFU_WAIT_FOREVER: u32 = u32::MAX;

pub type WaifuRequestAttribute = c_int;
pub const WAIFU_REQUEST_IMAGE_GENERATION_PROMPT: WaifuRequestAttribute = 100;
pub const WAIFU_REQUEST_IMAGE_GENERATION_NEGATIVE: WaifuRequestAttribute = 101;
pub const WAIFU_REQUEST_IMAGE_GENERATION_WIDTH: WaifuRequestAttribute = 102;
pub const WAIFU_REQUEST_IMAGE_GENERATION_HEIGHT: WaifuRequestAttribute = 103;
pub const WAIFU_REQUEST_IMAGE_GENERATION_STEPS: WaifuRequestAttribute = 104;
pub const WAIFU_REQUEST_IMAGE_GENERATION_GUIDANCE: WaifuRequestAttribute = 105;
pub const WAIFU_REQUEST_IMAGE_GENERATION_SEED: WaifuRequestAttribute = 106;
pub const WAIFU_REQUEST_IMAGE_GENERATION_START_IMAGE: WaifuRequestAttribute = 107;
pub const WAIFU_REQUEST_IMAGE_GENERATION_START_IMAGE_STRENGTH: WaifuRequestAttribute = 108;
pub const WAIFU_REQUEST_IMAGE_GENERATION_CONTROL_IMAGE: WaifuRequestAttribute = 109;
pub const WAIFU_REQUEST_IMAGE_GENERATION_CONTROL_IMAGE_SCALE: WaifuRequestAttribute = 110;
pub const WAIFU_REQUEST_IMAGE_GENERATION_PROMPT_IMAGES: WaifuRequestAttribute = 111;
pub const WAIFU_REQUEST_TEXT_TO_SPEECH_TEXT: WaifuRequestAttribute = 200;
pub const WAIFU_REQUEST_TEXT_TO_SPEECH_REFERENCE: WaifuRequestAttribute = 201;
pub const WAIFU_REQUEST_TEXT_TO_SPEECH_REFERENCE_TRANSCRIPT: WaifuRequestAttribute = 202;
pub const WAIFU_REQUEST_TEXT_TO_SPEECH_SPEED: WaifuRequestAttribute = 203;
pub const WAIFU_REQUEST_TEXT_TO_SPEECH_TEMPERATURE: WaifuRequestAttribute = 204;
pub const WAIFU_REQUEST_TEXT_TO_SPEECH_STYLE: WaifuRequestAttribute = 205;
pub const WAIFU_REQUEST_TEXT_TO_SPEECH_SEED: WaifuRequestAttribute = 206;
pub const WAIFU_REQUEST_TEXT_TO_SPEECH_VISEMES: WaifuRequestAttribute = 207;
pub const WAIFU_REQUEST_VOICE_CONVERSION_SOURCE: WaifuRequestAttribute = 300;
pub const WAIFU_REQUEST_VOICE_CONVERSION_REFERENCE: WaifuRequestAttribute = 301;
pub const WAIFU_REQUEST_VOICE_CONVERSION_STEPS: WaifuRequestAttribute = 302;
pub const WAIFU_REQUEST_VOICE_CONVERSION_CONVERT_STYLE: WaifuRequestAttribute = 303;
pub const WAIFU_REQUEST_VOICE_CONVERSION_SEED: WaifuRequestAttribute = 304;
pub const WAIFU_REQUEST_LAYER_DECOMPOSITION_IMAGE: WaifuRequestAttribute = 400;
pub const WAIFU_REQUEST_LAYER_DECOMPOSITION_STEPS: WaifuRequestAttribute = 401;
pub const WAIFU_REQUEST_LAYER_DECOMPOSITION_DEPTH_STEPS: WaifuRequestAttribute = 402;
pub const WAIFU_REQUEST_LAYER_DECOMPOSITION_RESOLUTION: WaifuRequestAttribute = 403;
pub const WAIFU_REQUEST_LAYER_DECOMPOSITION_SPLIT_PAIRS: WaifuRequestAttribute = 404;
pub const WAIFU_REQUEST_LAYER_DECOMPOSITION_STRATIFY_LAYERS: WaifuRequestAttribute = 405;
pub const WAIFU_REQUEST_LAYER_DECOMPOSITION_SEED: WaifuRequestAttribute = 406;
pub const WAIFU_REQUEST_SPEECH_TO_TEXT_AUDIO: WaifuRequestAttribute = 500;
pub const WAIFU_REQUEST_SPEECH_TO_TEXT_LANGUAGE: WaifuRequestAttribute = 501;
pub const WAIFU_REQUEST_SPEECH_TO_TEXT_CONTEXT: WaifuRequestAttribute = 502;
pub const WAIFU_REQUEST_STREAMING_TEXT_TO_SPEECH_TEXT: WaifuRequestAttribute = 600;
pub const WAIFU_REQUEST_STREAMING_TEXT_TO_SPEECH_REFERENCE: WaifuRequestAttribute = 601;
pub const WAIFU_REQUEST_STREAMING_TEXT_TO_SPEECH_REFERENCE_TRANSCRIPT: WaifuRequestAttribute = 602;
pub const WAIFU_REQUEST_STREAMING_TEXT_TO_SPEECH_TEMPERATURE: WaifuRequestAttribute = 603;
pub const WAIFU_REQUEST_STREAMING_TEXT_TO_SPEECH_STYLE: WaifuRequestAttribute = 604;
pub const WAIFU_REQUEST_STREAMING_TEXT_TO_SPEECH_SEED: WaifuRequestAttribute = 605;
pub const WAIFU_REQUEST_STREAMING_TEXT_TO_SPEECH_VISEMES: WaifuRequestAttribute = 606;

pub type WaifuTaskAttribute = c_int;
pub const WAIFU_TASK_STATUS: WaifuTaskAttribute = 1;
pub const WAIFU_TASK_STATUS_MESSAGE: WaifuTaskAttribute = 2;
pub const WAIFU_TASK_PROGRESS_STAGE: WaifuTaskAttribute = 10;
pub const WAIFU_TASK_PROGRESS_DONE: WaifuTaskAttribute = 11;
pub const WAIFU_TASK_PROGRESS_TOTAL: WaifuTaskAttribute = 12;
pub const WAIFU_TASK_PROGRESS_FRACTION: WaifuTaskAttribute = 13;
pub const WAIFU_TASK_PROGRESS_SECONDS: WaifuTaskAttribute = 14;
pub const WAIFU_TASK_IMAGE_GENERATION_IMAGE: WaifuTaskAttribute = 100;
pub const WAIFU_TASK_TEXT_TO_SPEECH_AUDIO: WaifuTaskAttribute = 200;
pub const WAIFU_TASK_TEXT_TO_SPEECH_VISEMES: WaifuTaskAttribute = 201;
pub const WAIFU_TASK_TEXT_TO_SPEECH_VISEME_FRAME_MS: WaifuTaskAttribute = 202;
pub const WAIFU_TASK_VOICE_CONVERSION_AUDIO: WaifuTaskAttribute = 300;
pub const WAIFU_TASK_LAYER_DECOMPOSITION_CANVAS: WaifuTaskAttribute = 400;
pub const WAIFU_TASK_LAYER_DECOMPOSITION_LAYER_COUNT: WaifuTaskAttribute = 401;
pub const WAIFU_TASK_LAYER_DECOMPOSITION_LAYER_TYPE: WaifuTaskAttribute = 402;
pub const WAIFU_TASK_LAYER_DECOMPOSITION_LAYER_ID: WaifuTaskAttribute = 403;
pub const WAIFU_TASK_LAYER_DECOMPOSITION_LAYER_NAME: WaifuTaskAttribute = 404;
pub const WAIFU_TASK_LAYER_DECOMPOSITION_LAYER_LEFT: WaifuTaskAttribute = 405;
pub const WAIFU_TASK_LAYER_DECOMPOSITION_LAYER_TOP: WaifuTaskAttribute = 406;
pub const WAIFU_TASK_LAYER_DECOMPOSITION_LAYER_IMAGE: WaifuTaskAttribute = 407;
pub const WAIFU_TASK_LAYER_DECOMPOSITION_LAYER_DEPTH: WaifuTaskAttribute = 408;
pub const WAIFU_TASK_LAYER_DECOMPOSITION_LAYER_DERIVED_FROM_ID: WaifuTaskAttribute = 409;
pub const WAIFU_TASK_SPEECH_TO_TEXT_TEXT: WaifuTaskAttribute = 500;
pub const WAIFU_TASK_SPEECH_TO_TEXT_LANGUAGE: WaifuTaskAttribute = 501;
pub const WAIFU_TASK_STREAMING_TEXT_TO_SPEECH_AUDIO_PART: WaifuTaskAttribute = 600;
pub const WAIFU_TASK_STREAMING_TEXT_TO_SPEECH_AUDIO: WaifuTaskAttribute = 601;
pub const WAIFU_TASK_STREAMING_TEXT_TO_SPEECH_VISEMES: WaifuTaskAttribute = 602;
pub const WAIFU_TASK_STREAMING_TEXT_TO_SPEECH_VISEME_FRAME_MS: WaifuTaskAttribute = 603;

// -- the four structs ------------------------------------------------------------------------

#[repr(C)]
pub struct WaifuLog {
    pub level: WaifuLogLevel,
    pub source: *const c_char,
    pub message: *const c_char,
    pub time: *const c_char,
}

pub type WaifuLogCallback =
    Option<unsafe extern "C" fn(user_data: *mut c_void, log: *const WaifuLog)>;

#[repr(C)]
pub struct WaifuImage {
    pub width: u32,
    pub height: u32,
    pub rgb: *const u8,
    pub len: usize,
}

#[repr(C)]
pub struct WaifuRgbaImage {
    pub width: u32,
    pub height: u32,
    pub rgba: *const u8,
    pub len: usize,
}

#[repr(C)]
pub struct WaifuAudio {
    pub samples: *const f32,
    pub count: usize,
    pub sample_rate: u32,
}

// -- the handles -----------------------------------------------------------------------------

#[repr(C)]
pub struct WaifuModel {
    _opaque: [u8; 0],
}

#[repr(C)]
pub struct WaifuPipeline {
    _opaque: [u8; 0],
}

#[repr(C)]
pub struct WaifuRequest {
    _opaque: [u8; 0],
}

#[repr(C)]
pub struct WaifuTask {
    _opaque: [u8; 0],
}

// -- the functions ---------------------------------------------------------------------------

/// Declares [`Functions`], one field per function of the header, and how each is found.
macro_rules! functions {
    ($($name:ident: fn($($argument:ty),*) $(-> $out:ty)?;)*) => {
        /// Every function `waifu.h` declares, found in the library [`crate::init`] loaded.
        pub struct Functions {
            $(pub $name: unsafe extern "C" fn($($argument),*) $(-> $out)?,)*
        }

        impl Functions {
            /// Each function, by name, from `find`; the first that is missing, otherwise.
            pub(crate) fn find(
                find: impl Fn(&str) -> Option<*mut c_void>,
            ) -> std::result::Result<Functions, &'static str> {
                Ok(Functions {
                    $($name: {
                        let found = find(stringify!($name)).ok_or(stringify!($name))?;
                        // The symbol is the function of that name the header declares.
                        unsafe {
                            std::mem::transmute::<
                                *mut c_void,
                                unsafe extern "C" fn($($argument),*) $(-> $out)?,
                            >(found)
                        }
                    },)*
                })
            }
        }
    };
}

functions! {
    waifu_abi_version: fn() -> u32;
    waifu_last_error: fn() -> *const c_char;
    waifu_last_error_code: fn() -> WaifuStatusCode;
    waifu_version: fn() -> *const c_char;
    waifu_revision: fn() -> *const c_char;
    waifu_set_log_callback: fn(*mut c_void, WaifuLogCallback);
    waifu_set_log_level: fn(WaifuLogLevel);

    waifu_device_is_available: fn(WaifuDevice) -> bool;
    waifu_device_get_int_attribute: fn(WaifuDevice, WaifuDeviceAttribute, *mut i64) -> bool;
    waifu_device_set_memory_limit: fn(WaifuDevice, u64) -> bool;

    waifu_model_open: fn(*const c_char) -> *mut WaifuModel;
    waifu_model_free: fn(*mut WaifuModel);
    waifu_model_get_int_attribute: fn(*const WaifuModel, WaifuModelAttribute, *mut i64) -> bool;
    waifu_model_get_string_attribute: fn(*const WaifuModel, WaifuModelAttribute) -> *const c_char;

    waifu_pipeline_new: fn(WaifuPipelineType, WaifuDevice) -> *mut WaifuPipeline;
    waifu_pipeline_free: fn(*mut WaifuPipeline);
    waifu_pipeline_get_type: fn(*const WaifuPipeline) -> WaifuPipelineType;
    waifu_pipeline_get_device: fn(*const WaifuPipeline) -> WaifuDevice;
    waifu_pipeline_cancel_all: fn(*mut WaifuPipeline);
    waifu_pipeline_load_model: fn(*mut WaifuPipeline, *const WaifuModel) -> *mut WaifuTask;
    waifu_pipeline_unload_model: fn(*mut WaifuPipeline) -> *mut WaifuTask;
    waifu_pipeline_run: fn(*mut WaifuPipeline, *const WaifuRequest) -> *mut WaifuTask;

    waifu_request_new: fn(WaifuPipelineType) -> *mut WaifuRequest;
    waifu_request_free: fn(*mut WaifuRequest);
    waifu_request_get_type: fn(*const WaifuRequest) -> WaifuPipelineType;
    waifu_request_set_int_attribute: fn(*mut WaifuRequest, WaifuRequestAttribute, i64) -> bool;
    waifu_request_set_float_attribute: fn(*mut WaifuRequest, WaifuRequestAttribute, f64) -> bool;
    waifu_request_set_string_attribute: fn(
        *mut WaifuRequest,
        WaifuRequestAttribute,
        *const c_char
    ) -> bool;
    waifu_request_set_image_attribute: fn(
        *mut WaifuRequest,
        WaifuRequestAttribute,
        *const WaifuImage
    ) -> bool;
    waifu_request_set_rgba_image_attribute: fn(
        *mut WaifuRequest,
        WaifuRequestAttribute,
        *const WaifuRgbaImage
    ) -> bool;
    waifu_request_set_audio_attribute: fn(
        *mut WaifuRequest,
        WaifuRequestAttribute,
        *const WaifuAudio
    ) -> bool;
    waifu_request_add_prompt_image: fn(*mut WaifuRequest, *const c_char, *const WaifuImage) -> bool;
    waifu_request_unset_attribute: fn(*mut WaifuRequest, WaifuRequestAttribute) -> bool;

    waifu_task_next_event: fn(*mut WaifuTask, u32) -> WaifuEventType;
    waifu_task_wait: fn(*mut WaifuTask, u32) -> bool;
    waifu_task_cancel: fn(*mut WaifuTask);
    waifu_task_free: fn(*mut WaifuTask);
    waifu_task_get_int_attribute: fn(*const WaifuTask, WaifuTaskAttribute, *mut i64) -> bool;
    waifu_task_get_float_attribute: fn(*const WaifuTask, WaifuTaskAttribute, *mut f64) -> bool;
    waifu_task_get_string_attribute: fn(*const WaifuTask, WaifuTaskAttribute) -> *const c_char;
    waifu_task_get_image_attribute: fn(*const WaifuTask, WaifuTaskAttribute, *mut WaifuImage) -> bool;
    waifu_task_get_rgba_image_attribute: fn(
        *const WaifuTask,
        WaifuTaskAttribute,
        *mut WaifuRgbaImage
    ) -> bool;
    waifu_task_get_audio_attribute: fn(*const WaifuTask, WaifuTaskAttribute, *mut WaifuAudio) -> bool;
    waifu_task_get_int_attribute_at: fn(*const WaifuTask, WaifuTaskAttribute, usize, *mut i64) -> bool;
    waifu_task_get_float_attribute_at: fn(
        *const WaifuTask,
        WaifuTaskAttribute,
        usize,
        *mut f64
    ) -> bool;
    waifu_task_get_string_attribute_at: fn(*const WaifuTask, WaifuTaskAttribute, usize) -> *const c_char;
    waifu_task_get_rgba_image_attribute_at: fn(
        *const WaifuTask,
        WaifuTaskAttribute,
        usize,
        *mut WaifuRgbaImage
    ) -> bool;
    waifu_task_get_int_array_attribute: fn(
        *const WaifuTask,
        WaifuTaskAttribute,
        *mut usize
    ) -> *const i32;
}

/// The functions of the library [`crate::init`] loaded, or why there are none.
pub fn functions() -> crate::Result<&'static Functions> {
    crate::library::functions()
}
