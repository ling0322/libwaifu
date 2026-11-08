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

//! Every struct and every number of `waifu::sys` held against the engine's own: the mirror is the
//! header, field for field. Built against the engine (LIBWAIFU_LIB_DIR), and runs nothing of it.

use std::mem::{align_of, offset_of, size_of};

use waifu::sys as mine;
use waifu_engine::c_api as theirs;

/// The same size and alignment, and each field at the same place.
macro_rules! same_layout {
    ($name:ident { $($field:ident),* $(,)? }) => {{
        assert_eq!(size_of::<mine::$name>(), size_of::<theirs::$name>(), stringify!($name));
        assert_eq!(align_of::<mine::$name>(), align_of::<theirs::$name>(), stringify!($name));
        $(
            assert_eq!(
                offset_of!(mine::$name, $field),
                offset_of!(theirs::$name, $field),
                "{}.{}",
                stringify!($name),
                stringify!($field),
            );
        )*
    }};
}

macro_rules! same_numbers {
    ($($name:ident),* $(,)?) => {
        $(assert_eq!(mine::$name as i64, theirs::$name as i64, stringify!($name));)*
    };
}

#[test]
fn every_struct_is_the_header_s() {
    same_layout!(WaifuLog {
        level,
        source,
        message,
        time
    });
    same_layout!(WaifuProgress {
        stage,
        fraction,
        done,
        total,
        seconds
    });
    same_layout!(WaifuStatus { code, message });
    same_layout!(WaifuResult { status });
    same_layout!(WaifuImage {
        width,
        height,
        rgb,
        len
    });
    same_layout!(WaifuPromptImage { key, image });
    same_layout!(WaifuImageGenerationRequest {
        struct_size,
        prompt,
        negative,
        width,
        height,
        steps,
        guidance,
        seed,
        start_image,
        start_image_strength,
        control_image,
        control_image_scale,
        prompt_images,
        prompt_image_count,
    });
    same_layout!(WaifuImageGenerationResult { status, image });
    same_layout!(WaifuAudio {
        samples,
        count,
        sample_rate
    });
    same_layout!(WaifuTextToSpeechRequest {
        struct_size,
        text,
        reference,
        speed,
        temperature,
        style,
        seed,
    });
    same_layout!(WaifuVoiceConversionRequest {
        struct_size,
        source,
        reference,
        steps,
        convert_style,
        seed,
    });
    same_layout!(WaifuAudioResult { status, audio });
    same_layout!(WaifuStreamingTextToSpeechRequest {
        struct_size,
        text,
        reference,
        temperature,
        style,
        seed,
    });
    same_layout!(WaifuRgbaImage {
        width,
        height,
        rgba,
        len
    });
    same_layout!(WaifuLayerDecompositionRequest {
        struct_size,
        image,
        steps,
        depth_steps,
        resolution,
        split_pairs,
        stratify_layers,
        seed,
    });
    same_layout!(WaifuLayer {
        r#type,
        id,
        name,
        left,
        top,
        image,
        depth,
        derived_from_id,
    });
    same_layout!(WaifuLayerDecompositionResult {
        status,
        canvas,
        layers,
        layer_count
    });
    same_layout!(WaifuSpeechToTextRequest {
        struct_size,
        audio,
        language,
        context
    });
    same_layout!(WaifuSpeechToTextResult {
        status,
        text,
        language
    });
}

#[test]
fn every_number_is_the_header_s() {
    assert_eq!(mine::WAIFU_ABI_VERSION, theirs::WAIFU_ABI_VERSION);
    same_numbers!(
        WAIFU_LOG_DEBUG,
        WAIFU_LOG_INFO,
        WAIFU_LOG_WARNING,
        WAIFU_LOG_ERROR,
        WAIFU_LOG_FATAL,
        WAIFU_DEVICE_AUTO,
        WAIFU_DEVICE_CPU,
        WAIFU_DEVICE_METAL,
        WAIFU_DEVICE_CUDA,
        WAIFU_DEVICE_VULKAN,
        WAIFU_PIPELINE_IMAGE_GENERATION,
        WAIFU_PIPELINE_TEXT_TO_SPEECH,
        WAIFU_PIPELINE_VOICE_CONVERSION,
        WAIFU_PIPELINE_LAYER_DECOMPOSITION,
        WAIFU_PIPELINE_SPEECH_TO_TEXT,
        WAIFU_PIPELINE_STREAMING_TEXT_TO_SPEECH,
        WAIFU_OK,
        WAIFU_CANCELLED,
        WAIFU_ERR_INVALID_ARGUMENT,
        WAIFU_ERR_UNKNOWN_MODEL,
        WAIFU_ERR_MODEL,
        WAIFU_ERR_NOT_LOADED,
        WAIFU_ERR_INTERNAL,
        WAIFU_ERR_DEVICE_UNAVAILABLE,
        WAIFU_STAGE_LOADING_WEIGHTS,
        WAIFU_STAGE_IMAGE_GENERATION_ENCODING_PROMPT,
        WAIFU_STAGE_IMAGE_GENERATION_DENOISING,
        WAIFU_STAGE_IMAGE_GENERATION_DECODING_IMAGE,
        WAIFU_STAGE_TEXT_TO_SPEECH_READING_TEXT,
        WAIFU_STAGE_TEXT_TO_SPEECH_GENERATING_TOKENS,
        WAIFU_STAGE_TEXT_TO_SPEECH_SYNTHESIZING_AUDIO,
        WAIFU_STAGE_VOICE_CONVERSION_READING_RECORDINGS,
        WAIFU_STAGE_VOICE_CONVERSION_GENERATING_TOKENS,
        WAIFU_STAGE_VOICE_CONVERSION_DENOISING,
        WAIFU_STAGE_VOICE_CONVERSION_SYNTHESIZING_AUDIO,
        WAIFU_STAGE_LAYER_DECOMPOSITION_ENCODING_IMAGE,
        WAIFU_STAGE_LAYER_DECOMPOSITION_DENOISING_LAYERS,
        WAIFU_STAGE_LAYER_DECOMPOSITION_DECODING_LAYERS,
        WAIFU_STAGE_LAYER_DECOMPOSITION_ENCODING_HEAD,
        WAIFU_STAGE_LAYER_DECOMPOSITION_DENOISING_HEAD_LAYERS,
        WAIFU_STAGE_LAYER_DECOMPOSITION_DECODING_HEAD_LAYERS,
        WAIFU_STAGE_LAYER_DECOMPOSITION_ENCODING_FOR_DEPTH,
        WAIFU_STAGE_LAYER_DECOMPOSITION_DENOISING_DEPTH,
        WAIFU_STAGE_LAYER_DECOMPOSITION_DECODING_DEPTH,
        WAIFU_STAGE_LAYER_DECOMPOSITION_CUTTING_INTO_PARTS,
        WAIFU_STAGE_SPEECH_TO_TEXT_READING_RECORDING,
        WAIFU_STAGE_SPEECH_TO_TEXT_GENERATING_TOKENS,
        WAIFU_STAGE_STREAMING_TEXT_TO_SPEECH_READING_TEXT,
        WAIFU_STAGE_STREAMING_TEXT_TO_SPEECH_GENERATING_TOKENS,
        WAIFU_STAGE_STREAMING_TEXT_TO_SPEECH_SYNTHESIZING_AUDIO,
        WAIFU_MODEL_SUPPORTS_IMAGE_GENERATION,
        WAIFU_MODEL_SUPPORTS_TEXT_TO_SPEECH,
        WAIFU_MODEL_SUPPORTS_VOICE_CONVERSION,
        WAIFU_MODEL_SUPPORTS_LAYER_DECOMPOSITION,
        WAIFU_MODEL_SUPPORTS_SPEECH_TO_TEXT,
        WAIFU_MODEL_SUPPORTS_STREAMING_TEXT_TO_SPEECH,
        WAIFU_MODEL_SUPPORTS_IMAGE_GENERATION_FROM_START_IMAGE,
        WAIFU_MODEL_SUPPORTS_IMAGE_GENERATION_WITH_CONTROL_IMAGE,
        WAIFU_MODEL_SUPPORTS_IMAGE_GENERATION_WITH_PROMPT_IMAGES,
        WAIFU_MODEL_SUPPORTS_TEXT_TO_SPEECH_LIKE_REFERENCE,
        WAIFU_MODEL_SUPPORTS_TEXT_TO_SPEECH_WITHOUT_REFERENCE,
        WAIFU_MODEL_SUPPORTS_TEXT_TO_SPEECH_WITH_STYLE,
        WAIFU_MODEL_SUPPORTS_VOICE_CONVERSION_OF_STYLE,
        WAIFU_MODEL_SIZE_ON_DISK,
        WAIFU_MODEL_SAMPLE_RATE,
        WAIFU_MODEL_IMAGE_GENERATION_WIDTH_HEIGHT_MULTIPLE_OF,
        WAIFU_MODEL_IMAGE_GENERATION_SAMPLER,
        WAIFU_MODEL_IMAGE_GENERATION_MAX_PROMPT_IMAGES,
        WAIFU_MODEL_LAYER_DECOMPOSITION_RESOLUTION_MULTIPLE_OF,
        WAIFU_MODEL_SPEECH_TO_TEXT_LANGUAGES,
        WAIFU_DEVICE_MEMORY_BUDGET,
        WAIFU_DEVICE_MEMORY_HELD,
        WAIFU_DEVICE_MEMORY_AVAILABLE,
        WAIFU_DEVICE_MEMORY_LIMIT,
        WAIFU_LAYER_PART,
        WAIFU_LAYER_STRATIFIED_NEAR,
        WAIFU_LAYER_STRATIFIED_FAR,
        WAIFU_LAYER_DEPTH,
    );
}
