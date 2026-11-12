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

//! A request: what one run of one task is asked to do, set an attribute at a time, and each
//! attribute's type, range and default -- the one place a default is chosen.

#![allow(clippy::missing_safety_doc)]

use std::collections::BTreeMap;
use std::ffi::c_char;
use std::ptr;

use super::support::{
    caught, refusable, required, string_in, Refused, WAIFU_ERR_UNKNOWN_ATTRIBUTE,
};
use super::{
    audio_in, image_in, rgba_in, task_name, WaifuAudio, WaifuImage, WaifuPipelineType,
    WaifuRgbaImage, WAIFU_PIPELINE_IMAGE_GENERATION, WAIFU_PIPELINE_LAYER_DECOMPOSITION,
    WAIFU_PIPELINE_SPEECH_TO_TEXT, WAIFU_PIPELINE_STREAMING_TEXT_TO_SPEECH,
    WAIFU_PIPELINE_TEXT_TO_SPEECH, WAIFU_PIPELINE_VOICE_CONVERSION,
};
use crate::engine::{
    self, DrawRequest, LayersRequest, RgbaImage, SpeakRequest, TranscribeRequest,
    VoiceConversionRequest,
};
use crate::wav::Sound;

/// A request's attributes, numbered by pipeline type as the stages are. The type after each is the
/// setter it takes; "required" is refused at waifu_pipeline_run when unset. A boolean is an int, 0
/// or 1.
pub type WaifuRequestAttribute = i32;
/// String, required.
pub const WAIFU_REQUEST_IMAGE_GENERATION_PROMPT: WaifuRequestAttribute = 100;
/// String; none.
pub const WAIFU_REQUEST_IMAGE_GENERATION_NEGATIVE: WaifuRequestAttribute = 101;
/// Int, 64..2048, a multiple of the model's IMAGE_GENERATION_WIDTH_HEIGHT_MULTIPLE_OF; 1024.
pub const WAIFU_REQUEST_IMAGE_GENERATION_WIDTH: WaifuRequestAttribute = 102;
/// Int, as WIDTH; 1024.
pub const WAIFU_REQUEST_IMAGE_GENERATION_HEIGHT: WaifuRequestAttribute = 103;
/// Int, 1..; 20.
pub const WAIFU_REQUEST_IMAGE_GENERATION_STEPS: WaifuRequestAttribute = 104;
/// Float, 0..; 5.0, ignored by a model without guidance.
pub const WAIFU_REQUEST_IMAGE_GENERATION_GUIDANCE: WaifuRequestAttribute = 105;
/// Int, read as uint64; 0.
pub const WAIFU_REQUEST_IMAGE_GENERATION_SEED: WaifuRequestAttribute = 106;
/// Image; none (img2img).
pub const WAIFU_REQUEST_IMAGE_GENERATION_START_IMAGE: WaifuRequestAttribute = 107;
/// Float, 0..1, how far it walks away from it; 0.8.
pub const WAIFU_REQUEST_IMAGE_GENERATION_START_IMAGE_STRENGTH: WaifuRequestAttribute = 108;
/// Image; none (a ControlNet's condition).
pub const WAIFU_REQUEST_IMAGE_GENERATION_CONTROL_IMAGE: WaifuRequestAttribute = 109;
/// Float, 0.., how hard it holds to it; 1.0.
pub const WAIFU_REQUEST_IMAGE_GENERATION_CONTROL_IMAGE_SCALE: WaifuRequestAttribute = 110;
/// waifu_request_add_prompt_image; none.
pub const WAIFU_REQUEST_IMAGE_GENERATION_PROMPT_IMAGES: WaifuRequestAttribute = 111;

/// String, required.
pub const WAIFU_REQUEST_TEXT_TO_SPEECH_TEXT: WaifuRequestAttribute = 200;
/// Audio, a recording to sound like; none.
pub const WAIFU_REQUEST_TEXT_TO_SPEECH_REFERENCE: WaifuRequestAttribute = 201;
/// String, what REFERENCE says, word for word; none. CosyVoice3 reads on from it, as upstream's
/// zero-shot does, and takes it or a style, not both; a model that reads none ignores it.
pub const WAIFU_REQUEST_TEXT_TO_SPEECH_REFERENCE_TRANSCRIPT: WaifuRequestAttribute = 202;
/// Float, above 0; 1.0.
pub const WAIFU_REQUEST_TEXT_TO_SPEECH_SPEED: WaifuRequestAttribute = 203;
/// Float, 0..; 1.0.
pub const WAIFU_REQUEST_TEXT_TO_SPEECH_TEMPERATURE: WaifuRequestAttribute = 204;
/// String, any instruction; the voice's own way.
pub const WAIFU_REQUEST_TEXT_TO_SPEECH_STYLE: WaifuRequestAttribute = 205;
/// Int, read as uint64; 0.
pub const WAIFU_REQUEST_TEXT_TO_SPEECH_SEED: WaifuRequestAttribute = 206;

/// Audio, what is said and how; required.
pub const WAIFU_REQUEST_VOICE_CONVERSION_SOURCE: WaifuRequestAttribute = 300;
/// Audio, whose voice to say it in; required.
pub const WAIFU_REQUEST_VOICE_CONVERSION_REFERENCE: WaifuRequestAttribute = 301;
/// Int, 1..; 10.
pub const WAIFU_REQUEST_VOICE_CONVERSION_STEPS: WaifuRequestAttribute = 302;
/// Boolean; 0.
pub const WAIFU_REQUEST_VOICE_CONVERSION_CONVERT_STYLE: WaifuRequestAttribute = 303;
/// Int, read as uint64; 0.
pub const WAIFU_REQUEST_VOICE_CONVERSION_SEED: WaifuRequestAttribute = 304;

/// RGBA image, any size, centred and scaled by the engine; required.
pub const WAIFU_REQUEST_LAYER_DECOMPOSITION_IMAGE: WaifuRequestAttribute = 400;
/// Int, 1..; the model's own.
pub const WAIFU_REQUEST_LAYER_DECOMPOSITION_STEPS: WaifuRequestAttribute = 401;
/// Int, 1..; the model's own.
pub const WAIFU_REQUEST_LAYER_DECOMPOSITION_DEPTH_STEPS: WaifuRequestAttribute = 402;
/// Int, a multiple of the model's LAYER_DECOMPOSITION_RESOLUTION_MULTIPLE_OF; 1280.
pub const WAIFU_REQUEST_LAYER_DECOMPOSITION_RESOLUTION: WaifuRequestAttribute = 403;
/// Boolean, hands, eyes, eyebrows, ears into a -l and a -r layer; 0.
pub const WAIFU_REQUEST_LAYER_DECOMPOSITION_SPLIT_PAIRS: WaifuRequestAttribute = 404;
/// Boolean, hair, hands, top and bottom also cut into a near and a far half; 0.
pub const WAIFU_REQUEST_LAYER_DECOMPOSITION_STRATIFY_LAYERS: WaifuRequestAttribute = 405;
/// Int, read as uint64; 0.
pub const WAIFU_REQUEST_LAYER_DECOMPOSITION_SEED: WaifuRequestAttribute = 406;

/// Audio, any rate and length; required.
pub const WAIFU_REQUEST_SPEECH_TO_TEXT_AUDIO: WaifuRequestAttribute = 500;
/// String, one of the model's SPEECH_TO_TEXT_LANGUAGES -- "English", "chinese"; the model tells.
pub const WAIFU_REQUEST_SPEECH_TO_TEXT_LANGUAGE: WaifuRequestAttribute = 501;
/// String, words that help it hear: names, terms, the subject; none.
pub const WAIFU_REQUEST_SPEECH_TO_TEXT_CONTEXT: WaifuRequestAttribute = 502;

/// String, required.
pub const WAIFU_REQUEST_STREAMING_TEXT_TO_SPEECH_TEXT: WaifuRequestAttribute = 600;
/// Audio; none.
pub const WAIFU_REQUEST_STREAMING_TEXT_TO_SPEECH_REFERENCE: WaifuRequestAttribute = 601;
/// String, as TEXT_TO_SPEECH_REFERENCE_TRANSCRIPT; none.
pub const WAIFU_REQUEST_STREAMING_TEXT_TO_SPEECH_REFERENCE_TRANSCRIPT: WaifuRequestAttribute = 602;
/// Float, 0..; 1.0.
pub const WAIFU_REQUEST_STREAMING_TEXT_TO_SPEECH_TEMPERATURE: WaifuRequestAttribute = 603;
/// String; the voice's own way. No speed: a streaming reading is at the speed it is said, as
/// upstream's is.
pub const WAIFU_REQUEST_STREAMING_TEXT_TO_SPEECH_STYLE: WaifuRequestAttribute = 604;
/// Int, read as uint64; 0.
pub const WAIFU_REQUEST_STREAMING_TEXT_TO_SPEECH_SEED: WaifuRequestAttribute = 605;

/// What an attribute holds, and what it may hold.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Kind {
    Int {
        min: i64,
        max: i64,
    },
    Bool,
    /// Any 64 bits, read as a uint64.
    Seed,
    Float {
        min: f64,
        max: f64,
    },
    Text,
    Image,
    Rgba,
    Audio,
    /// Set with waifu_request_add_prompt_image.
    PromptImages,
}

impl Kind {
    /// The setter a caller is told to use.
    fn setter(self) -> &'static str {
        match self {
            Kind::Int { .. } | Kind::Bool | Kind::Seed => "waifu_request_set_int_attribute",
            Kind::Float { .. } => "waifu_request_set_float_attribute",
            Kind::Text => "waifu_request_set_string_attribute",
            Kind::Image => "waifu_request_set_image_attribute",
            Kind::Rgba => "waifu_request_set_rgba_image_attribute",
            Kind::Audio => "waifu_request_set_audio_attribute",
            Kind::PromptImages => "waifu_request_add_prompt_image",
        }
    }
}

const SIDE: Kind = Kind::Int { min: 64, max: 2048 };
const COUNT: Kind = Kind::Int {
    min: 1,
    max: u32::MAX as i64,
};
const AT_LEAST_ZERO: Kind = Kind::Float {
    min: 0.0,
    max: f64::MAX,
};

/// An attribute's pipeline type, name and kind, or None for a number this library does not know.
fn attribute(attribute: WaifuRequestAttribute) -> Option<(WaifuPipelineType, &'static str, Kind)> {
    use Kind::*;
    let image = WAIFU_PIPELINE_IMAGE_GENERATION;
    let speech = WAIFU_PIPELINE_TEXT_TO_SPEECH;
    let conversion = WAIFU_PIPELINE_VOICE_CONVERSION;
    let layers = WAIFU_PIPELINE_LAYER_DECOMPOSITION;
    let hearing = WAIFU_PIPELINE_SPEECH_TO_TEXT;
    let streaming = WAIFU_PIPELINE_STREAMING_TEXT_TO_SPEECH;
    Some(match attribute {
        WAIFU_REQUEST_IMAGE_GENERATION_PROMPT => (image, "IMAGE_GENERATION_PROMPT", Text),
        WAIFU_REQUEST_IMAGE_GENERATION_NEGATIVE => (image, "IMAGE_GENERATION_NEGATIVE", Text),
        WAIFU_REQUEST_IMAGE_GENERATION_WIDTH => (image, "IMAGE_GENERATION_WIDTH", SIDE),
        WAIFU_REQUEST_IMAGE_GENERATION_HEIGHT => (image, "IMAGE_GENERATION_HEIGHT", SIDE),
        WAIFU_REQUEST_IMAGE_GENERATION_STEPS => (image, "IMAGE_GENERATION_STEPS", COUNT),
        WAIFU_REQUEST_IMAGE_GENERATION_GUIDANCE => {
            (image, "IMAGE_GENERATION_GUIDANCE", AT_LEAST_ZERO)
        }
        WAIFU_REQUEST_IMAGE_GENERATION_SEED => (image, "IMAGE_GENERATION_SEED", Seed),
        WAIFU_REQUEST_IMAGE_GENERATION_START_IMAGE => {
            (image, "IMAGE_GENERATION_START_IMAGE", Image)
        }
        WAIFU_REQUEST_IMAGE_GENERATION_START_IMAGE_STRENGTH => (
            image,
            "IMAGE_GENERATION_START_IMAGE_STRENGTH",
            Float { min: 0.0, max: 1.0 },
        ),
        WAIFU_REQUEST_IMAGE_GENERATION_CONTROL_IMAGE => {
            (image, "IMAGE_GENERATION_CONTROL_IMAGE", Image)
        }
        WAIFU_REQUEST_IMAGE_GENERATION_CONTROL_IMAGE_SCALE => {
            (image, "IMAGE_GENERATION_CONTROL_IMAGE_SCALE", AT_LEAST_ZERO)
        }
        WAIFU_REQUEST_IMAGE_GENERATION_PROMPT_IMAGES => {
            (image, "IMAGE_GENERATION_PROMPT_IMAGES", PromptImages)
        }

        WAIFU_REQUEST_TEXT_TO_SPEECH_TEXT => (speech, "TEXT_TO_SPEECH_TEXT", Text),
        WAIFU_REQUEST_TEXT_TO_SPEECH_REFERENCE => (speech, "TEXT_TO_SPEECH_REFERENCE", Audio),
        WAIFU_REQUEST_TEXT_TO_SPEECH_REFERENCE_TRANSCRIPT => {
            (speech, "TEXT_TO_SPEECH_REFERENCE_TRANSCRIPT", Text)
        }
        WAIFU_REQUEST_TEXT_TO_SPEECH_SPEED => (
            speech,
            "TEXT_TO_SPEECH_SPEED",
            Float {
                min: f64::MIN_POSITIVE,
                max: f64::MAX,
            },
        ),
        WAIFU_REQUEST_TEXT_TO_SPEECH_TEMPERATURE => {
            (speech, "TEXT_TO_SPEECH_TEMPERATURE", AT_LEAST_ZERO)
        }
        WAIFU_REQUEST_TEXT_TO_SPEECH_STYLE => (speech, "TEXT_TO_SPEECH_STYLE", Text),
        WAIFU_REQUEST_TEXT_TO_SPEECH_SEED => (speech, "TEXT_TO_SPEECH_SEED", Seed),

        WAIFU_REQUEST_VOICE_CONVERSION_SOURCE => (conversion, "VOICE_CONVERSION_SOURCE", Audio),
        WAIFU_REQUEST_VOICE_CONVERSION_REFERENCE => {
            (conversion, "VOICE_CONVERSION_REFERENCE", Audio)
        }
        WAIFU_REQUEST_VOICE_CONVERSION_STEPS => (conversion, "VOICE_CONVERSION_STEPS", COUNT),
        WAIFU_REQUEST_VOICE_CONVERSION_CONVERT_STYLE => {
            (conversion, "VOICE_CONVERSION_CONVERT_STYLE", Bool)
        }
        WAIFU_REQUEST_VOICE_CONVERSION_SEED => (conversion, "VOICE_CONVERSION_SEED", Seed),

        WAIFU_REQUEST_LAYER_DECOMPOSITION_IMAGE => (layers, "LAYER_DECOMPOSITION_IMAGE", Rgba),
        WAIFU_REQUEST_LAYER_DECOMPOSITION_STEPS => (layers, "LAYER_DECOMPOSITION_STEPS", COUNT),
        WAIFU_REQUEST_LAYER_DECOMPOSITION_DEPTH_STEPS => {
            (layers, "LAYER_DECOMPOSITION_DEPTH_STEPS", COUNT)
        }
        WAIFU_REQUEST_LAYER_DECOMPOSITION_RESOLUTION => {
            (layers, "LAYER_DECOMPOSITION_RESOLUTION", COUNT)
        }
        WAIFU_REQUEST_LAYER_DECOMPOSITION_SPLIT_PAIRS => {
            (layers, "LAYER_DECOMPOSITION_SPLIT_PAIRS", Bool)
        }
        WAIFU_REQUEST_LAYER_DECOMPOSITION_STRATIFY_LAYERS => {
            (layers, "LAYER_DECOMPOSITION_STRATIFY_LAYERS", Bool)
        }
        WAIFU_REQUEST_LAYER_DECOMPOSITION_SEED => (layers, "LAYER_DECOMPOSITION_SEED", Seed),

        WAIFU_REQUEST_SPEECH_TO_TEXT_AUDIO => (hearing, "SPEECH_TO_TEXT_AUDIO", Audio),
        WAIFU_REQUEST_SPEECH_TO_TEXT_LANGUAGE => (hearing, "SPEECH_TO_TEXT_LANGUAGE", Text),
        WAIFU_REQUEST_SPEECH_TO_TEXT_CONTEXT => (hearing, "SPEECH_TO_TEXT_CONTEXT", Text),

        WAIFU_REQUEST_STREAMING_TEXT_TO_SPEECH_TEXT => {
            (streaming, "STREAMING_TEXT_TO_SPEECH_TEXT", Text)
        }
        WAIFU_REQUEST_STREAMING_TEXT_TO_SPEECH_REFERENCE => {
            (streaming, "STREAMING_TEXT_TO_SPEECH_REFERENCE", Audio)
        }
        WAIFU_REQUEST_STREAMING_TEXT_TO_SPEECH_REFERENCE_TRANSCRIPT => (
            streaming,
            "STREAMING_TEXT_TO_SPEECH_REFERENCE_TRANSCRIPT",
            Text,
        ),
        WAIFU_REQUEST_STREAMING_TEXT_TO_SPEECH_TEMPERATURE => (
            streaming,
            "STREAMING_TEXT_TO_SPEECH_TEMPERATURE",
            AT_LEAST_ZERO,
        ),
        WAIFU_REQUEST_STREAMING_TEXT_TO_SPEECH_STYLE => {
            (streaming, "STREAMING_TEXT_TO_SPEECH_STYLE", Text)
        }
        WAIFU_REQUEST_STREAMING_TEXT_TO_SPEECH_SEED => {
            (streaming, "STREAMING_TEXT_TO_SPEECH_SEED", Seed)
        }
        _ => return None,
    })
}

/// A value an attribute was set to, copied from what the caller handed over.
#[derive(Clone, Debug)]
enum Value {
    Int(i64),
    Float(f64),
    Text(String),
    Image(engine::Image),
    Rgba(RgbaImage),
    Audio(Sound),
}

/// What one run on a pipeline of one type is asked to do. Not shared between threads while it is
/// being set; run as often as liked, with attributes changed in between: waifu_pipeline_run copies
/// it.
pub struct WaifuRequest {
    task: WaifuPipelineType,
    values: BTreeMap<WaifuRequestAttribute, Value>,
    /// IMAGE_GENERATION_PROMPT_IMAGES, in the order they were added.
    prompt_images: Vec<(String, engine::Image)>,
}

impl WaifuRequest {
    pub fn task(&self) -> WaifuPipelineType {
        self.task
    }

    /// The attribute's kind, where it is one of this request's pipeline type's.
    fn kind(&self, number: WaifuRequestAttribute) -> Result<(&'static str, Kind), Refused> {
        let Some((task, name, kind)) = attribute(number) else {
            return Err(Refused::new(
                WAIFU_ERR_UNKNOWN_ATTRIBUTE,
                format!("{number} is not a request attribute this library knows"),
            ));
        };
        if task != self.task {
            return Err(Refused::new(
                WAIFU_ERR_UNKNOWN_ATTRIBUTE,
                format!(
                    "{name} is an attribute of {}, and this is a request for {}",
                    task_name(task),
                    task_name(self.task)
                ),
            ));
        }
        Ok((name, kind))
    }

    fn set(&mut self, number: WaifuRequestAttribute, value: Option<Value>) {
        match value {
            Some(value) => {
                self.values.insert(number, value);
            }
            None => {
                self.values.remove(&number);
            }
        }
    }

    fn int(&self, number: WaifuRequestAttribute) -> Option<i64> {
        match self.values.get(&number) {
            Some(Value::Int(value)) => Some(*value),
            _ => None,
        }
    }

    fn float(&self, number: WaifuRequestAttribute) -> Option<f64> {
        match self.values.get(&number) {
            Some(Value::Float(value)) => Some(*value),
            _ => None,
        }
    }

    fn text(&self, number: WaifuRequestAttribute) -> Option<String> {
        match self.values.get(&number) {
            Some(Value::Text(value)) => Some(value.clone()),
            _ => None,
        }
    }

    fn image(&self, number: WaifuRequestAttribute) -> Option<engine::Image> {
        match self.values.get(&number) {
            Some(Value::Image(value)) => Some(value.clone()),
            _ => None,
        }
    }

    fn audio(&self, number: WaifuRequestAttribute) -> Option<Sound> {
        match self.values.get(&number) {
            Some(Value::Audio(value)) => Some(value.clone()),
            _ => None,
        }
    }

    /// The name of an attribute, for a sentence saying it is not set.
    fn missing(number: WaifuRequestAttribute) -> Refused {
        let name = attribute(number).map_or("an attribute", |(_, name, _)| name);
        format!("{name} is not set, and the request cannot be run without it").into()
    }

    fn required_text(&self, number: WaifuRequestAttribute) -> Result<String, Refused> {
        self.text(number).ok_or_else(|| Self::missing(number))
    }

    fn required_audio(&self, number: WaifuRequestAttribute) -> Result<Sound, Refused> {
        self.audio(number).ok_or_else(|| Self::missing(number))
    }

    fn u32_or(&self, number: WaifuRequestAttribute, default: u32) -> u32 {
        self.int(number).map_or(default, |value| value as u32)
    }

    fn f32_or(&self, number: WaifuRequestAttribute, default: f32) -> f32 {
        self.float(number).map_or(default, |value| value as f32)
    }

    fn seed(&self, number: WaifuRequestAttribute) -> u64 {
        self.int(number).map_or(0, |value| value as u64)
    }

    fn flag(&self, number: WaifuRequestAttribute) -> bool {
        self.int(number) == Some(1)
    }

    pub fn draw(&self) -> Result<DrawRequest, Refused> {
        let mut images = Vec::with_capacity(self.prompt_images.len() + 2);
        if let Some(image) = self.image(WAIFU_REQUEST_IMAGE_GENERATION_START_IMAGE) {
            images.push((engine::START_FROM.to_string(), image));
        }
        if let Some(image) = self.image(WAIFU_REQUEST_IMAGE_GENERATION_CONTROL_IMAGE) {
            images.push((engine::CONTROL.to_string(), image));
        }
        images.extend(self.prompt_images.iter().cloned());
        Ok(DrawRequest {
            prompt: self.required_text(WAIFU_REQUEST_IMAGE_GENERATION_PROMPT)?,
            negative: self
                .text(WAIFU_REQUEST_IMAGE_GENERATION_NEGATIVE)
                .unwrap_or_default(),
            images,
            width: self.u32_or(WAIFU_REQUEST_IMAGE_GENERATION_WIDTH, 1024),
            height: self.u32_or(WAIFU_REQUEST_IMAGE_GENERATION_HEIGHT, 1024),
            steps: self.u32_or(WAIFU_REQUEST_IMAGE_GENERATION_STEPS, 20),
            guidance: self.f32_or(WAIFU_REQUEST_IMAGE_GENERATION_GUIDANCE, 5.0),
            seed: self.seed(WAIFU_REQUEST_IMAGE_GENERATION_SEED),
            strength: self.f32_or(WAIFU_REQUEST_IMAGE_GENERATION_START_IMAGE_STRENGTH, 0.8),
            control_scale: self.f32_or(WAIFU_REQUEST_IMAGE_GENERATION_CONTROL_IMAGE_SCALE, 1.0),
        })
    }

    pub fn speak(&self) -> Result<SpeakRequest, Refused> {
        Ok(SpeakRequest {
            text: self.required_text(WAIFU_REQUEST_TEXT_TO_SPEECH_TEXT)?,
            like: self.audio(WAIFU_REQUEST_TEXT_TO_SPEECH_REFERENCE),
            speed: self.f32_or(WAIFU_REQUEST_TEXT_TO_SPEECH_SPEED, 1.0),
            temperature: self.f32_or(WAIFU_REQUEST_TEXT_TO_SPEECH_TEMPERATURE, 1.0),
            style: self.text(WAIFU_REQUEST_TEXT_TO_SPEECH_STYLE),
            reference_transcript: self.text(WAIFU_REQUEST_TEXT_TO_SPEECH_REFERENCE_TRANSCRIPT),
            seed: self.seed(WAIFU_REQUEST_TEXT_TO_SPEECH_SEED),
        })
    }

    pub fn speak_streaming(&self) -> Result<SpeakRequest, Refused> {
        Ok(SpeakRequest {
            text: self.required_text(WAIFU_REQUEST_STREAMING_TEXT_TO_SPEECH_TEXT)?,
            like: self.audio(WAIFU_REQUEST_STREAMING_TEXT_TO_SPEECH_REFERENCE),
            speed: 1.0,
            temperature: self.f32_or(WAIFU_REQUEST_STREAMING_TEXT_TO_SPEECH_TEMPERATURE, 1.0),
            style: self.text(WAIFU_REQUEST_STREAMING_TEXT_TO_SPEECH_STYLE),
            reference_transcript: self
                .text(WAIFU_REQUEST_STREAMING_TEXT_TO_SPEECH_REFERENCE_TRANSCRIPT),
            seed: self.seed(WAIFU_REQUEST_STREAMING_TEXT_TO_SPEECH_SEED),
        })
    }

    pub fn voice_conversion(&self) -> Result<VoiceConversionRequest, Refused> {
        Ok(VoiceConversionRequest {
            source: self.required_audio(WAIFU_REQUEST_VOICE_CONVERSION_SOURCE)?,
            reference: self.required_audio(WAIFU_REQUEST_VOICE_CONVERSION_REFERENCE)?,
            steps: self.u32_or(WAIFU_REQUEST_VOICE_CONVERSION_STEPS, 10),
            convert_style: self.flag(WAIFU_REQUEST_VOICE_CONVERSION_CONVERT_STYLE),
            seed: self.seed(WAIFU_REQUEST_VOICE_CONVERSION_SEED),
        })
    }

    pub fn layers(&self) -> Result<LayersRequest, Refused> {
        let image = match self.values.get(&WAIFU_REQUEST_LAYER_DECOMPOSITION_IMAGE) {
            Some(Value::Rgba(image)) => image.clone(),
            _ => return Err(Self::missing(WAIFU_REQUEST_LAYER_DECOMPOSITION_IMAGE)),
        };
        // 0 is the engine's "the model's own" for each of the three.
        Ok(LayersRequest {
            image,
            steps: self.u32_or(WAIFU_REQUEST_LAYER_DECOMPOSITION_STEPS, 0),
            depth_steps: self.u32_or(WAIFU_REQUEST_LAYER_DECOMPOSITION_DEPTH_STEPS, 0),
            resolution: self.u32_or(WAIFU_REQUEST_LAYER_DECOMPOSITION_RESOLUTION, 0),
            split_pairs: self.flag(WAIFU_REQUEST_LAYER_DECOMPOSITION_SPLIT_PAIRS),
            stratify: self.flag(WAIFU_REQUEST_LAYER_DECOMPOSITION_STRATIFY_LAYERS),
            seed: self.seed(WAIFU_REQUEST_LAYER_DECOMPOSITION_SEED),
        })
    }

    pub fn transcribe(&self) -> Result<TranscribeRequest, Refused> {
        Ok(TranscribeRequest {
            recording: self.required_audio(WAIFU_REQUEST_SPEECH_TO_TEXT_AUDIO)?,
            language: self
                .text(WAIFU_REQUEST_SPEECH_TO_TEXT_LANGUAGE)
                .filter(|language| !language.is_empty()),
            context: self
                .text(WAIFU_REQUEST_SPEECH_TO_TEXT_CONTEXT)
                .unwrap_or_default(),
        })
    }
}

/// A request for a pipeline of `type`, with every attribute unset. Refused -- NULL,
/// WAIFU_ERR_INVALID_ARGUMENT -- for a type this library does not know.
#[no_mangle]
pub extern "C" fn waifu_request_new(r#type: WaifuPipelineType) -> *mut WaifuRequest {
    refusable(ptr::null_mut(), || {
        if task_name(r#type) == "no task" {
            return Err(format!("{} is not a pipeline type this library knows", r#type).into());
        }
        Ok(Box::into_raw(Box::new(WaifuRequest {
            task: r#type,
            values: BTreeMap::new(),
            prompt_images: Vec::new(),
        })))
    })
}

/// NULL is fine.
#[no_mangle]
pub unsafe extern "C" fn waifu_request_free(request: *mut WaifuRequest) {
    if !request.is_null() {
        caught((), || drop(Box::from_raw(request)));
    }
}

/// 0 for NULL.
#[no_mangle]
pub unsafe extern "C" fn waifu_request_get_type(request: *const WaifuRequest) -> WaifuPipelineType {
    request.as_ref().map_or(0, |request| request.task)
}

/// Runs a setter: the request, the attribute's name and kind, and what the setter makes of them.
unsafe fn setter(
    request: *mut WaifuRequest,
    number: WaifuRequestAttribute,
    set: impl FnOnce(&str, Kind) -> Result<Option<Value>, Refused>,
) -> bool {
    refusable(false, || {
        let request = request.as_mut().ok_or("the request is NULL")?;
        let (name, kind) = request.kind(number)?;
        let value = set(name, kind)?;
        request.set(number, value);
        Ok(true)
    })
}

fn wrong_type(name: &str, kind: Kind, given: &str) -> Refused {
    format!("{name} is not {given}: set it with {}", kind.setter()).into()
}

/// Each sets one attribute and copies what it is handed: a string, an image's pixels, a recording's
/// samples. false, with the last error and its code, for an attribute this library does not know or
/// that is another pipeline type's (WAIFU_ERR_UNKNOWN_ATTRIBUTE), and for a value of another type,
/// out of the attribute's range, or a string that is not UTF-8 (WAIFU_ERR_INVALID_ARGUMENT); the
/// attribute is then as it was. A NULL string, image or audio unsets the attribute.
#[no_mangle]
pub unsafe extern "C" fn waifu_request_set_int_attribute(
    request: *mut WaifuRequest,
    attribute: WaifuRequestAttribute,
    value: i64,
) -> bool {
    setter(request, attribute, |name, kind| match kind {
        Kind::Int { min, max } if (min..=max).contains(&value) => Ok(Some(Value::Int(value))),
        Kind::Int { min, max } => {
            Err(format!("{name} is {value}: it has to be from {min} to {max}").into())
        }
        Kind::Bool if value == 0 || value == 1 => Ok(Some(Value::Int(value))),
        Kind::Bool => Err(format!("{name} is a boolean, 0 or 1, and {value} is neither").into()),
        Kind::Seed => Ok(Some(Value::Int(value))),
        _ => Err(wrong_type(name, kind, "an integer")),
    })
}

/// As waifu_request_set_int_attribute. The value is kept, and checked against the attribute's
/// range, as the float the model is handed: one too large for a float is refused, not infinity.
#[no_mangle]
pub unsafe extern "C" fn waifu_request_set_float_attribute(
    request: *mut WaifuRequest,
    attribute: WaifuRequestAttribute,
    value: f64,
) -> bool {
    // The engine is handed an f32, so that is what is checked and kept: a double too large for one
    // would be handed over as infinity, and one too small as 0.
    let narrow = value as f32;
    let kept = narrow as f64;
    setter(request, attribute, |name, kind| match kind {
        Kind::Float { min, max } if narrow.is_finite() && kept >= min && kept <= max => {
            Ok(Some(Value::Float(kept)))
        }
        Kind::Float { .. } if value.is_finite() && !narrow.is_finite() => Err(format!(
            "{name} is {value}: it has to be a float's, at most {:e}",
            f32::MAX
        )
        .into()),
        Kind::Float { min, .. } if min == f64::MIN_POSITIVE => {
            Err(format!("{name} is {value}: it has to be above 0").into())
        }
        Kind::Float { min, max } if max == f64::MAX => {
            Err(format!("{name} is {value}: it has to be {min} or more").into())
        }
        Kind::Float { min, max } => {
            Err(format!("{name} is {value}: it has to be from {min} to {max}").into())
        }
        _ => Err(wrong_type(name, kind, "a float")),
    })
}

/// As waifu_request_set_int_attribute.
#[no_mangle]
pub unsafe extern "C" fn waifu_request_set_string_attribute(
    request: *mut WaifuRequest,
    attribute: WaifuRequestAttribute,
    value: *const c_char,
) -> bool {
    setter(request, attribute, |name, kind| match kind {
        Kind::Text => Ok(string_in(value, name)?.map(Value::Text)),
        _ => Err(wrong_type(name, kind, "a string")),
    })
}

/// As waifu_request_set_int_attribute.
#[no_mangle]
pub unsafe extern "C" fn waifu_request_set_image_attribute(
    request: *mut WaifuRequest,
    attribute: WaifuRequestAttribute,
    value: *const WaifuImage,
) -> bool {
    setter(request, attribute, |name, kind| match kind {
        Kind::Image if value.is_null() => Ok(None),
        Kind::Image => Ok(Some(Value::Image(image_in(value, name)?))),
        _ => Err(wrong_type(name, kind, "an RGB image")),
    })
}

/// As waifu_request_set_int_attribute.
#[no_mangle]
pub unsafe extern "C" fn waifu_request_set_rgba_image_attribute(
    request: *mut WaifuRequest,
    attribute: WaifuRequestAttribute,
    value: *const WaifuRgbaImage,
) -> bool {
    setter(request, attribute, |name, kind| match kind {
        Kind::Rgba if value.is_null() => Ok(None),
        Kind::Rgba => Ok(Some(Value::Rgba(rgba_in(value, name)?))),
        _ => Err(wrong_type(name, kind, "an RGBA image")),
    })
}

/// As waifu_request_set_int_attribute.
#[no_mangle]
pub unsafe extern "C" fn waifu_request_set_audio_attribute(
    request: *mut WaifuRequest,
    attribute: WaifuRequestAttribute,
    value: *const WaifuAudio,
) -> bool {
    setter(request, attribute, |name, kind| match kind {
        Kind::Audio if value.is_null() => Ok(None),
        Kind::Audio => Ok(Some(Value::Audio(audio_in(value, name)?))),
        _ => Err(wrong_type(name, kind, "audio")),
    })
}

/// IMAGE_GENERATION_PROMPT_IMAGES: one more image the prompt names as <|key|>; key is letters,
/// digits and _. A key given twice is refused. Unsetting the attribute drops them all.
#[no_mangle]
pub unsafe extern "C" fn waifu_request_add_prompt_image(
    request: *mut WaifuRequest,
    key: *const c_char,
    image: *const WaifuImage,
) -> bool {
    refusable(false, || {
        let request = request.as_mut().ok_or("the request is NULL")?;
        request.kind(WAIFU_REQUEST_IMAGE_GENERATION_PROMPT_IMAGES)?;
        let key = required(key, "a prompt image's key")?;
        if key == engine::START_FROM || key == engine::CONTROL {
            return Err(format!(
                "\"{key}\" is not a key a prompt image can have here: set the image as \
                 IMAGE_GENERATION_START_IMAGE or IMAGE_GENERATION_CONTROL_IMAGE, or call it \
                 something else"
            )
            .into());
        }
        if request.prompt_images.iter().any(|(given, _)| *given == key) {
            return Err(format!("a prompt image is already called \"{key}\"").into());
        }
        let image = image_in(image, &format!("the prompt image \"{key}\""))?;
        request.prompt_images.push((key, image));
        Ok(true)
    })
}

/// Back to the attribute's default. false, as a setter's, for an attribute this library does not
/// know.
#[no_mangle]
pub unsafe extern "C" fn waifu_request_unset_attribute(
    request: *mut WaifuRequest,
    attribute: WaifuRequestAttribute,
) -> bool {
    refusable(false, || {
        let request = request.as_mut().ok_or("the request is NULL")?;
        request.kind(attribute)?;
        if attribute == WAIFU_REQUEST_IMAGE_GENERATION_PROMPT_IMAGES {
            request.prompt_images.clear();
        }
        request.values.remove(&attribute);
        Ok(true)
    })
}
