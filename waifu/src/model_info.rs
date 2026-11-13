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

//! What a model can do, from its manifest alone: what the C API's `WaifuModel` answers, one
//! property at a time (docs/ffi-abi5.md).
//!
//! A task the model does not do is an `Err` with the reason, in words to show; what a task's
//! request may hold is in its own struct, there only for a model that does the task.

use std::path::Path;

use crate::describe::{self, ConverterKind, TranscriberKind, VoiceKind, TONES};
use crate::engine::Failure;
use crate::see_through::SeeThrough;
use crate::Manifest;

/// Whether something is done, and why not where it is not.
pub type Support = Result<(), String>;

/// What a model can be asked for.
#[derive(Clone, Debug, PartialEq)]
pub struct ModelInfo {
    /// Bytes of every file the manifest names, where all of them are here; `None` before.
    pub size_on_disk: Option<u64>,
    /// The rate of what it says, converts or hears.
    pub sample_rate: Option<u32>,
    pub image_generation: Result<ImageGeneration, String>,
    pub text_to_speech: Result<TextToSpeech, String>,
    /// Whether it can stream a reading at all; the device is checked when it is loaded.
    pub streaming_text_to_speech: Support,
    pub speech_to_text: Result<SpeechToText, String>,
    pub voice_conversion: Result<VoiceConversion, String>,
    pub layer_decomposition: Result<LayerDecomposition, String>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ImageGeneration {
    pub width_height_multiple_of: u32,
    pub sampler: String,
    pub from_start_image: Support,
    pub with_control_image: Support,
    pub with_prompt_images: Support,
    pub max_prompt_images: u32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TextToSpeech {
    pub like_reference: Support,
    pub without_reference: Support,
    /// A recording without its words: CosyVoice3 does not, unless a style is given.
    pub without_reference_transcript: Support,
    pub with_style: Support,
    /// The mouth shape of every frame of what it says: a CosyVoice3 package with the viseme model.
    pub with_visemes: Support,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SpeechToText {
    /// The languages a request may name.
    pub languages: Vec<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct VoiceConversion {
    pub of_style: Support,
}

#[derive(Clone, Debug, PartialEq)]
pub struct LayerDecomposition {
    pub resolution_multiple_of: u32,
}

/// What a See-through package's layers are drawn at a multiple of: both of its models halve that
/// often. `SeeThrough::decompose` refuses any other.
const LAYER_DECOMPOSITION_RESOLUTION_MULTIPLE_OF: u32 = 64;

/// What the model at `asked` -- the path of a manifest, or "tones" -- can be asked for, read from
/// its manifest and none of its weights. `UnknownModel` where there is no manifest there, `Model`
/// where there is one that cannot be read.
pub fn model_info(asked: &str) -> Result<ModelInfo, (Failure, String)> {
    if asked == TONES {
        let voice = describe::look_at_voice(asked);
        return Ok(ModelInfo {
            size_on_disk: Some(0),
            sample_rate: Some(voice.rate),
            image_generation: not("image generation", &voice.full_name),
            text_to_speech: Ok(text_to_speech(&voice)),
            streaming_text_to_speech: Ok(()),
            speech_to_text: not("speech to text", &voice.full_name),
            voice_conversion: not("voice conversion", &voice.full_name),
            layer_decomposition: not("layer decomposition", &voice.full_name),
        });
    }

    let path = Path::new(asked);
    if !path.is_file() {
        return Err((Failure::UnknownModel, describe::no_manifest_at(asked)));
    }
    let manifest = Manifest::open(path).map_err(|error| (Failure::Model, error.to_string()))?;
    let size_on_disk = describe::bytes_on_disk(asked);

    if let Some(kind) = describe::voice_kind(path) {
        let voice = describe::look_at_voice(asked);
        let name = voice.full_name.clone();
        let converts = describe::converter_kind(path) == Some(ConverterKind::CosyVoice3);
        let converter = converts.then(|| describe::look_at_converter(asked));
        return Ok(ModelInfo {
            size_on_disk,
            sample_rate: Some(voice.rate),
            image_generation: not("image generation", &name),
            text_to_speech: Ok(text_to_speech(&voice)),
            streaming_text_to_speech: match kind {
                VoiceKind::CosyVoice3 => Ok(()),
                VoiceKind::IndexTts => Err(format!(
                    "{name} does not stream: it hands over a reading only when the whole of it is made"
                )),
            },
            speech_to_text: not("speech to text", &name),
            voice_conversion: match converter {
                Some(converter) => Ok(VoiceConversion {
                    of_style: support(converter.no_style_because),
                }),
                None => not("voice conversion", &name),
            },
            layer_decomposition: not("layer decomposition", &name),
        });
    }

    if describe::transcriber_kind(path) == Some(TranscriberKind::Qwen3Asr) {
        let transcriber = describe::look_at_transcriber(asked);
        let name = transcriber.full_name.clone();
        return Ok(ModelInfo {
            size_on_disk,
            sample_rate: Some(transcriber.rate),
            image_generation: not("image generation", &name),
            text_to_speech: not("text to speech", &name),
            streaming_text_to_speech: not("streaming text to speech", &name),
            speech_to_text: Ok(SpeechToText {
                languages: transcriber.languages,
            }),
            voice_conversion: not("voice conversion", &name),
            layer_decomposition: not("layer decomposition", &name),
        });
    }

    let model_type = manifest
        .section("model")
        .and_then(|model| Ok(model.get_str("type")?.to_string()))
        .unwrap_or_default();
    if model_type == SeeThrough::MODEL_TYPE {
        let name = describe::shown_name(asked);
        return Ok(ModelInfo {
            size_on_disk,
            sample_rate: None,
            image_generation: not("image generation", &name),
            text_to_speech: not("text to speech", &name),
            streaming_text_to_speech: not("streaming text to speech", &name),
            speech_to_text: not("speech to text", &name),
            voice_conversion: not("voice conversion", &name),
            layer_decomposition: Ok(LayerDecomposition {
                resolution_multiple_of: LAYER_DECOMPOSITION_RESOLUTION_MULTIPLE_OF,
            }),
        });
    }

    if describe::converter_kind(path) == Some(ConverterKind::CosyVoice3) {
        let converter = describe::look_at_converter(asked);
        let name = converter.full_name.clone();
        return Ok(ModelInfo {
            size_on_disk,
            sample_rate: Some(converter.rate),
            image_generation: not("image generation", &name),
            text_to_speech: not("text to speech", &name),
            streaming_text_to_speech: not("streaming text to speech", &name),
            speech_to_text: not("speech to text", &name),
            voice_conversion: Ok(VoiceConversion {
                of_style: support(converter.no_style_because),
            }),
            layer_decomposition: not("layer decomposition", &name),
        });
    }

    // Anything else draws, as the engine reads it: SDXL where nothing says otherwise.
    let chosen = describe::look_at(asked);
    let name = chosen.full_name.clone();
    Ok(ModelInfo {
        size_on_disk,
        sample_rate: None,
        image_generation: Ok(ImageGeneration {
            width_height_multiple_of: chosen.alignment.max(1) as u32,
            sampler: chosen.sampler.to_string(),
            from_start_image: support(chosen.no_picture_because),
            with_control_image: Err(format!("{name} has no ControlNet here")),
            with_prompt_images: match chosen.prompt_pictures {
                0 => Err(format!("{name} reads words only")),
                _ => Ok(()),
            },
            max_prompt_images: chosen.prompt_pictures as u32,
        }),
        text_to_speech: not("text to speech", &name),
        streaming_text_to_speech: not("streaming text to speech", &name),
        speech_to_text: not("speech to text", &name),
        voice_conversion: not("voice conversion", &name),
        layer_decomposition: not("layer decomposition", &name),
    })
}

fn not<T>(task: &str, name: &str) -> Result<T, String> {
    Err(format!("{name} does not do {task}"))
}

fn support(why_not: Option<String>) -> Support {
    match why_not {
        None => Ok(()),
        Some(why) => Err(why),
    }
}

fn text_to_speech(voice: &describe::Spoken) -> TextToSpeech {
    TextToSpeech {
        like_reference: support(voice.no_likeness_because.clone()),
        without_reference: match voice.needs_a_recording {
            false => Ok(()),
            true => Err(format!(
                "{} speaks only in the voice of a recording: give it one",
                voice.full_name
            )),
        },
        without_reference_transcript: match voice.needs_a_reference_transcript {
            false => Ok(()),
            true => Err(format!(
                "{} reads on from what the recording says: give its transcript with it, \
                 unless a style is given",
                voice.full_name
            )),
        },
        with_style: match voice.styles.is_empty() {
            true => Err(format!(
                "{} follows no instruction for how to speak",
                voice.full_name
            )),
            false => Ok(()),
        },
        with_visemes: match voice.says_visemes {
            true => Ok(()),
            false => Err(format!(
                "{} does not say which mouth shape goes with what it says",
                voice.full_name
            )),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cosyvoice3::CosyVoice3;
    use crate::engine::tests::a_manifest;

    #[test]
    fn a_picture_model_draws_and_says_what_it_cannot_take() {
        let sdxl = model_info(&a_manifest("info-sdxl", "sdxl")).unwrap();
        let image = sdxl.image_generation.unwrap();
        assert_eq!(image.width_height_multiple_of, 32);
        assert_eq!(image.from_start_image, Ok(()));
        assert!(image.with_control_image.is_err());
        assert!(image.with_prompt_images.is_err());
        assert!(sdxl.text_to_speech.is_err());
        assert!(sdxl.layer_decomposition.is_err());
        assert_eq!(sdxl.sample_rate, None);

        let anima = model_info(&a_manifest("info-anima", crate::Anima::MODEL_TYPE)).unwrap();
        let image = anima.image_generation.unwrap();
        assert_eq!(image.width_height_multiple_of, 16);
        assert!(image.from_start_image.is_err());
    }

    #[test]
    fn the_stand_in_voice_speaks_and_streams_without_a_recording() {
        let tones = model_info(TONES).unwrap();
        let speech = tones.text_to_speech.unwrap();
        assert_eq!(speech.without_reference, Ok(()));
        assert_eq!(speech.without_reference_transcript, Ok(()));
        assert_eq!(tones.streaming_text_to_speech, Ok(()));
        assert!(tones.image_generation.is_err());
        assert_eq!(tones.size_on_disk, Some(0));
        assert!(tones.sample_rate.unwrap() > 0);
    }

    #[test]
    fn cosyvoice3_speaks_streams_and_converts_but_needs_a_recording() {
        let voice = model_info(&a_manifest("info-cosyvoice3", CosyVoice3::MODEL_TYPE)).unwrap();
        let speech = voice.text_to_speech.unwrap();
        assert!(speech.without_reference.is_err());
        assert!(speech.without_reference_transcript.is_err());
        assert_eq!(speech.with_style, Ok(()));
        assert_eq!(voice.streaming_text_to_speech, Ok(()));
        assert!(voice.voice_conversion.is_ok());
        assert!(voice.image_generation.is_err());
        // A package without the viseme model does not say them.
        assert!(speech.with_visemes.is_err());
    }

    #[test]
    fn a_cosyvoice3_package_with_the_viseme_model_says_so_in_its_manifest() {
        let with = |stem: &str, listed: &str| {
            let path = a_manifest(stem, CosyVoice3::MODEL_TYPE);
            let text = std::fs::read_to_string(&path).unwrap();
            std::fs::write(
                &path,
                format!("{text}  cosyvoice3:\n    sample_rate: 24000\n    visemes: {listed}\n"),
            )
            .unwrap();
            model_info(&path)
                .unwrap()
                .text_to_speech
                .unwrap()
                .with_visemes
        };
        assert_eq!(
            with("info-visemes", r#"[A, E, I, O, U, "@", MBP, REST]"#),
            Ok(())
        );
        // Classes in another order are another model's, which this does not read.
        assert!(with(
            "info-visemes-reordered",
            r#"[E, A, I, O, U, "@", MBP, REST]"#
        )
        .is_err());
        // Tones says none.
        let tones = model_info(TONES).unwrap();
        assert!(tones.text_to_speech.unwrap().with_visemes.is_err());
    }

    #[test]
    fn a_path_with_no_manifest_is_unknown() {
        let (failure, why) = model_info("no-such-model:v9").unwrap_err();
        assert_eq!(failure, Failure::UnknownModel);
        assert!(why.contains("no-such-model:v9"), "{why}");
    }
}
