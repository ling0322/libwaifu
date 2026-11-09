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

//! The mirror of `waifu.h`, ABI 4, and the table of its functions, for whatever the crate does not
//! wrap.
//!
//! Every C enum here is a `c_int`, never a Rust `enum`: a number a newer library returns that a
//! Rust enum has no variant for would be undefined behaviour before anything could map it. The
//! safe types convert from the integer. tests/layout.rs holds every struct against the engine's.

#![allow(non_camel_case_types)]

use std::ffi::{c_char, c_int, c_void};

pub const WAIFU_ABI_VERSION: u32 = 4;

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

pub type WaifuModelProperty = c_int;
pub const WAIFU_MODEL_SUPPORTS_IMAGE_GENERATION: WaifuModelProperty = 1;
pub const WAIFU_MODEL_SUPPORTS_TEXT_TO_SPEECH: WaifuModelProperty = 2;
pub const WAIFU_MODEL_SUPPORTS_VOICE_CONVERSION: WaifuModelProperty = 3;
pub const WAIFU_MODEL_SUPPORTS_LAYER_DECOMPOSITION: WaifuModelProperty = 4;
pub const WAIFU_MODEL_SUPPORTS_SPEECH_TO_TEXT: WaifuModelProperty = 5;
pub const WAIFU_MODEL_SUPPORTS_STREAMING_TEXT_TO_SPEECH: WaifuModelProperty = 6;
pub const WAIFU_MODEL_SUPPORTS_IMAGE_GENERATION_FROM_START_IMAGE: WaifuModelProperty = 100;
pub const WAIFU_MODEL_SUPPORTS_IMAGE_GENERATION_WITH_CONTROL_IMAGE: WaifuModelProperty = 101;
pub const WAIFU_MODEL_SUPPORTS_IMAGE_GENERATION_WITH_PROMPT_IMAGES: WaifuModelProperty = 102;
pub const WAIFU_MODEL_SUPPORTS_TEXT_TO_SPEECH_LIKE_REFERENCE: WaifuModelProperty = 200;
pub const WAIFU_MODEL_SUPPORTS_TEXT_TO_SPEECH_WITHOUT_REFERENCE: WaifuModelProperty = 201;
pub const WAIFU_MODEL_SUPPORTS_TEXT_TO_SPEECH_WITH_STYLE: WaifuModelProperty = 202;
pub const WAIFU_MODEL_SUPPORTS_VOICE_CONVERSION_OF_STYLE: WaifuModelProperty = 300;
pub const WAIFU_MODEL_SIZE_ON_DISK: WaifuModelProperty = 1000;
pub const WAIFU_MODEL_SAMPLE_RATE: WaifuModelProperty = 1001;
pub const WAIFU_MODEL_IMAGE_GENERATION_WIDTH_HEIGHT_MULTIPLE_OF: WaifuModelProperty = 1100;
pub const WAIFU_MODEL_IMAGE_GENERATION_SAMPLER: WaifuModelProperty = 1101;
pub const WAIFU_MODEL_IMAGE_GENERATION_MAX_PROMPT_IMAGES: WaifuModelProperty = 1102;
pub const WAIFU_MODEL_LAYER_DECOMPOSITION_RESOLUTION_MULTIPLE_OF: WaifuModelProperty = 1400;
pub const WAIFU_MODEL_SPEECH_TO_TEXT_LANGUAGES: WaifuModelProperty = 1500;

pub type WaifuDeviceProperty = c_int;
pub const WAIFU_DEVICE_MEMORY_BUDGET: WaifuDeviceProperty = 1;
pub const WAIFU_DEVICE_MEMORY_HELD: WaifuDeviceProperty = 2;
pub const WAIFU_DEVICE_MEMORY_AVAILABLE: WaifuDeviceProperty = 3;
pub const WAIFU_DEVICE_MEMORY_LIMIT: WaifuDeviceProperty = 4;

pub type WaifuLayerType = c_int;
pub const WAIFU_LAYER_PART: WaifuLayerType = 0;
pub const WAIFU_LAYER_STRATIFIED_NEAR: WaifuLayerType = 1;
pub const WAIFU_LAYER_STRATIFIED_FAR: WaifuLayerType = 2;
pub const WAIFU_LAYER_DEPTH: WaifuLayerType = 3;

// -- the library -----------------------------------------------------------------------------

#[repr(C)]
pub struct WaifuLog {
    pub level: WaifuLogLevel,
    pub source: *const c_char,
    pub message: *const c_char,
    pub time: *const c_char,
}

pub type WaifuLogCallback =
    Option<unsafe extern "C" fn(user_data: *mut c_void, log: *const WaifuLog)>;

// -- models and pipelines --------------------------------------------------------------------

#[repr(C)]
pub struct WaifuModel {
    _opaque: [u8; 0],
}

#[repr(C)]
pub struct WaifuPipeline {
    _opaque: [u8; 0],
}

pub type WaifuJob = u64;

#[repr(C)]
pub struct WaifuProgress {
    pub stage: WaifuStage,
    pub fraction: f64,
    pub done: u64,
    pub total: u64,
    pub seconds: f64,
}

pub type WaifuProgressCallback =
    Option<unsafe extern "C" fn(user_data: *mut c_void, progress: *const WaifuProgress)>;

#[repr(C)]
pub struct WaifuStatus {
    pub code: WaifuStatusCode,
    pub message: *const c_char,
}

#[repr(C)]
pub struct WaifuResult {
    pub status: WaifuStatus,
}

pub type WaifuCallback =
    Option<unsafe extern "C" fn(user_data: *mut c_void, result: *const WaifuResult)>;

// -- image generation ------------------------------------------------------------------------

#[repr(C)]
pub struct WaifuImage {
    pub width: u32,
    pub height: u32,
    pub rgb: *const u8,
    pub len: usize,
}

#[repr(C)]
pub struct WaifuPromptImage {
    pub key: *const c_char,
    pub image: *const WaifuImage,
}

#[repr(C)]
pub struct WaifuImageGenerationRequest {
    pub struct_size: u32,
    pub prompt: *const c_char,
    pub negative: *const c_char,
    pub width: u32,
    pub height: u32,
    pub steps: u32,
    pub guidance: f32,
    pub seed: u64,
    pub start_image: *const WaifuImage,
    pub start_image_strength: f32,
    pub control_image: *const WaifuImage,
    pub control_image_scale: f32,
    pub prompt_images: *const WaifuPromptImage,
    pub prompt_image_count: usize,
}

#[repr(C)]
pub struct WaifuImageGenerationResult {
    pub status: WaifuStatus,
    pub image: *const WaifuImage,
}

pub type WaifuImageGenerationCallback =
    Option<unsafe extern "C" fn(user_data: *mut c_void, result: *const WaifuImageGenerationResult)>;

// -- text to speech and voice conversion -----------------------------------------------------

#[repr(C)]
pub struct WaifuAudio {
    pub samples: *const f32,
    pub count: usize,
    pub sample_rate: u32,
}

#[repr(C)]
pub struct WaifuTextToSpeechRequest {
    pub struct_size: u32,
    pub text: *const c_char,
    pub reference: *const WaifuAudio,
    pub speed: f32,
    pub temperature: f32,
    pub style: *const c_char,
    pub seed: u64,
    pub reference_transcript: *const c_char,
}

#[repr(C)]
pub struct WaifuVoiceConversionRequest {
    pub struct_size: u32,
    pub source: *const WaifuAudio,
    pub reference: *const WaifuAudio,
    pub steps: u32,
    pub convert_style: bool,
    pub seed: u64,
}

#[repr(C)]
pub struct WaifuAudioResult {
    pub status: WaifuStatus,
    pub audio: *const WaifuAudio,
}

pub type WaifuAudioCallback =
    Option<unsafe extern "C" fn(user_data: *mut c_void, result: *const WaifuAudioResult)>;

#[repr(C)]
pub struct WaifuStreamingTextToSpeechRequest {
    pub struct_size: u32,
    pub text: *const c_char,
    pub reference: *const WaifuAudio,
    pub temperature: f32,
    pub style: *const c_char,
    pub seed: u64,
    pub reference_transcript: *const c_char,
}

pub type WaifuAudioPartCallback =
    Option<unsafe extern "C" fn(user_data: *mut c_void, part: *const WaifuAudio)>;

// -- layer decomposition ---------------------------------------------------------------------

#[repr(C)]
pub struct WaifuRgbaImage {
    pub width: u32,
    pub height: u32,
    pub rgba: *const u8,
    pub len: usize,
}

#[repr(C)]
pub struct WaifuLayerDecompositionRequest {
    pub struct_size: u32,
    pub image: *const WaifuRgbaImage,
    pub steps: u32,
    pub depth_steps: u32,
    pub resolution: u32,
    pub split_pairs: bool,
    pub stratify_layers: bool,
    pub seed: u64,
}

#[repr(C)]
pub struct WaifuLayer {
    pub r#type: WaifuLayerType,
    pub id: u32,
    pub name: *const c_char,
    pub left: i32,
    pub top: i32,
    pub image: WaifuRgbaImage,
    pub depth: f32,
    pub derived_from_id: u32,
}

#[repr(C)]
pub struct WaifuLayerDecompositionResult {
    pub status: WaifuStatus,
    pub canvas: u32,
    pub layers: *const WaifuLayer,
    pub layer_count: usize,
}

pub type WaifuLayerDecompositionCallback = Option<
    unsafe extern "C" fn(user_data: *mut c_void, result: *const WaifuLayerDecompositionResult),
>;

// -- speech to text --------------------------------------------------------------------------

#[repr(C)]
pub struct WaifuSpeechToTextRequest {
    pub struct_size: u32,
    pub audio: *const WaifuAudio,
    pub language: *const c_char,
    pub context: *const c_char,
}

#[repr(C)]
pub struct WaifuSpeechToTextResult {
    pub status: WaifuStatus,
    pub text: *const c_char,
    pub language: *const c_char,
}

pub type WaifuSpeechToTextCallback =
    Option<unsafe extern "C" fn(user_data: *mut c_void, result: *const WaifuSpeechToTextResult)>;

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
    waifu_device_get_int_property: fn(WaifuDevice, WaifuDeviceProperty, *mut i64) -> bool;
    waifu_device_set_memory_limit: fn(WaifuDevice, u64) -> bool;

    waifu_model_open: fn(*const c_char) -> *mut WaifuModel;
    waifu_model_free: fn(*mut WaifuModel);
    waifu_model_get_int_property: fn(*const WaifuModel, WaifuModelProperty, *mut i64) -> bool;
    waifu_model_get_string_property: fn(*const WaifuModel, WaifuModelProperty) -> *const c_char;

    waifu_pipeline_new: fn(WaifuPipelineType, WaifuDevice) -> *mut WaifuPipeline;
    waifu_pipeline_free: fn(*mut WaifuPipeline);
    waifu_pipeline_get_type: fn(*const WaifuPipeline) -> WaifuPipelineType;
    waifu_pipeline_get_device: fn(*const WaifuPipeline) -> WaifuDevice;
    waifu_pipeline_cancel: fn(*mut WaifuPipeline, WaifuJob);
    waifu_pipeline_cancel_all: fn(*mut WaifuPipeline);
    waifu_pipeline_load_model_async: fn(
        *mut WaifuPipeline,
        *const WaifuModel,
        *mut c_void,
        WaifuProgressCallback,
        WaifuCallback
    ) -> WaifuJob;
    waifu_pipeline_unload_model_async: fn(*mut WaifuPipeline, *mut c_void, WaifuCallback) -> WaifuJob;
    waifu_pipeline_run_image_generation_async: fn(
        *mut WaifuPipeline,
        *const WaifuImageGenerationRequest,
        *mut c_void,
        WaifuProgressCallback,
        WaifuImageGenerationCallback
    ) -> WaifuJob;
    waifu_pipeline_run_text_to_speech_async: fn(
        *mut WaifuPipeline,
        *const WaifuTextToSpeechRequest,
        *mut c_void,
        WaifuProgressCallback,
        WaifuAudioCallback
    ) -> WaifuJob;
    waifu_pipeline_run_voice_conversion_async: fn(
        *mut WaifuPipeline,
        *const WaifuVoiceConversionRequest,
        *mut c_void,
        WaifuProgressCallback,
        WaifuAudioCallback
    ) -> WaifuJob;
    waifu_pipeline_run_streaming_text_to_speech_async: fn(
        *mut WaifuPipeline,
        *const WaifuStreamingTextToSpeechRequest,
        *mut c_void,
        WaifuProgressCallback,
        WaifuAudioPartCallback,
        WaifuAudioCallback
    ) -> WaifuJob;
    waifu_pipeline_run_layer_decomposition_async: fn(
        *mut WaifuPipeline,
        *const WaifuLayerDecompositionRequest,
        *mut c_void,
        WaifuProgressCallback,
        WaifuLayerDecompositionCallback
    ) -> WaifuJob;
    waifu_pipeline_run_speech_to_text_async: fn(
        *mut WaifuPipeline,
        *const WaifuSpeechToTextRequest,
        *mut c_void,
        WaifuProgressCallback,
        WaifuSpeechToTextCallback
    ) -> WaifuJob;
}

/// The functions of the library [`crate::init`] loaded, or why there are none.
pub fn functions() -> crate::Result<&'static Functions> {
    crate::library::functions()
}
