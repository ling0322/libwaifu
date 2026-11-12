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

//! Pipelines, jobs, and what they are asked for and hand back.
//!
//! A job is the library's task, read to its end by a thread of this crate's: the library calls
//! nothing of a caller's, so that thread is what wakes a `Job` waiting for its next event. It
//! frees the task once it has read its end.

use std::collections::{HashSet, VecDeque};
use std::convert::Infallible;
use std::ffi::CString;
use std::future::{Future, IntoFuture};
use std::marker::PhantomData;
use std::pin::Pin;
use std::ptr;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll, Waker};

use crate::{sys, text, Device, Error, ErrorCode, Model, Result};

// -- enums -----------------------------------------------------------------------------------

/// What a pipeline is made for: `WAIFU_PIPELINE_*`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PipelineType {
    ImageGeneration,
    TextToSpeech,
    VoiceConversion,
    LayerDecomposition,
    SpeechToText,
    StreamingTextToSpeech,
    /// A type a newer library has and this crate does not know.
    Other(i32),
}

impl PipelineType {
    fn raw(self) -> sys::WaifuPipelineType {
        match self {
            PipelineType::ImageGeneration => sys::WAIFU_PIPELINE_IMAGE_GENERATION,
            PipelineType::TextToSpeech => sys::WAIFU_PIPELINE_TEXT_TO_SPEECH,
            PipelineType::VoiceConversion => sys::WAIFU_PIPELINE_VOICE_CONVERSION,
            PipelineType::LayerDecomposition => sys::WAIFU_PIPELINE_LAYER_DECOMPOSITION,
            PipelineType::SpeechToText => sys::WAIFU_PIPELINE_SPEECH_TO_TEXT,
            PipelineType::StreamingTextToSpeech => sys::WAIFU_PIPELINE_STREAMING_TEXT_TO_SPEECH,
            PipelineType::Other(other) => other,
        }
    }
}

impl From<sys::WaifuPipelineType> for PipelineType {
    fn from(raw: sys::WaifuPipelineType) -> PipelineType {
        match raw {
            sys::WAIFU_PIPELINE_IMAGE_GENERATION => PipelineType::ImageGeneration,
            sys::WAIFU_PIPELINE_TEXT_TO_SPEECH => PipelineType::TextToSpeech,
            sys::WAIFU_PIPELINE_VOICE_CONVERSION => PipelineType::VoiceConversion,
            sys::WAIFU_PIPELINE_LAYER_DECOMPOSITION => PipelineType::LayerDecomposition,
            sys::WAIFU_PIPELINE_SPEECH_TO_TEXT => PipelineType::SpeechToText,
            sys::WAIFU_PIPELINE_STREAMING_TEXT_TO_SPEECH => PipelineType::StreamingTextToSpeech,
            other => PipelineType::Other(other),
        }
    }
}

/// What a job is doing: `WAIFU_STAGE_*`. `done` and `total` in [`Progress`] count what the C API
/// says each counts.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Stage {
    LoadingWeights,
    ImageGenerationEncodingPrompt,
    ImageGenerationDenoising,
    ImageGenerationDecodingImage,
    TextToSpeechReadingText,
    TextToSpeechGeneratingTokens,
    TextToSpeechSynthesizingAudio,
    VoiceConversionReadingRecordings,
    VoiceConversionGeneratingTokens,
    VoiceConversionDenoising,
    VoiceConversionSynthesizingAudio,
    LayerDecompositionEncodingImage,
    LayerDecompositionDenoisingLayers,
    LayerDecompositionDecodingLayers,
    LayerDecompositionEncodingHead,
    LayerDecompositionDenoisingHeadLayers,
    LayerDecompositionDecodingHeadLayers,
    LayerDecompositionEncodingForDepth,
    LayerDecompositionDenoisingDepth,
    LayerDecompositionDecodingDepth,
    LayerDecompositionCuttingIntoParts,
    SpeechToTextReadingRecording,
    SpeechToTextGeneratingTokens,
    StreamingTextToSpeechReadingText,
    StreamingTextToSpeechGeneratingTokens,
    StreamingTextToSpeechSynthesizingAudio,
    /// A stage a newer library has and this crate does not know.
    Other(i32),
}

impl From<sys::WaifuStage> for Stage {
    fn from(raw: sys::WaifuStage) -> Stage {
        match raw {
            sys::WAIFU_STAGE_LOADING_WEIGHTS => Stage::LoadingWeights,
            sys::WAIFU_STAGE_IMAGE_GENERATION_ENCODING_PROMPT => {
                Stage::ImageGenerationEncodingPrompt
            }
            sys::WAIFU_STAGE_IMAGE_GENERATION_DENOISING => Stage::ImageGenerationDenoising,
            sys::WAIFU_STAGE_IMAGE_GENERATION_DECODING_IMAGE => Stage::ImageGenerationDecodingImage,
            sys::WAIFU_STAGE_TEXT_TO_SPEECH_READING_TEXT => Stage::TextToSpeechReadingText,
            sys::WAIFU_STAGE_TEXT_TO_SPEECH_GENERATING_TOKENS => {
                Stage::TextToSpeechGeneratingTokens
            }
            sys::WAIFU_STAGE_TEXT_TO_SPEECH_SYNTHESIZING_AUDIO => {
                Stage::TextToSpeechSynthesizingAudio
            }
            sys::WAIFU_STAGE_VOICE_CONVERSION_READING_RECORDINGS => {
                Stage::VoiceConversionReadingRecordings
            }
            sys::WAIFU_STAGE_VOICE_CONVERSION_GENERATING_TOKENS => {
                Stage::VoiceConversionGeneratingTokens
            }
            sys::WAIFU_STAGE_VOICE_CONVERSION_DENOISING => Stage::VoiceConversionDenoising,
            sys::WAIFU_STAGE_VOICE_CONVERSION_SYNTHESIZING_AUDIO => {
                Stage::VoiceConversionSynthesizingAudio
            }
            sys::WAIFU_STAGE_LAYER_DECOMPOSITION_ENCODING_IMAGE => {
                Stage::LayerDecompositionEncodingImage
            }
            sys::WAIFU_STAGE_LAYER_DECOMPOSITION_DENOISING_LAYERS => {
                Stage::LayerDecompositionDenoisingLayers
            }
            sys::WAIFU_STAGE_LAYER_DECOMPOSITION_DECODING_LAYERS => {
                Stage::LayerDecompositionDecodingLayers
            }
            sys::WAIFU_STAGE_LAYER_DECOMPOSITION_ENCODING_HEAD => {
                Stage::LayerDecompositionEncodingHead
            }
            sys::WAIFU_STAGE_LAYER_DECOMPOSITION_DENOISING_HEAD_LAYERS => {
                Stage::LayerDecompositionDenoisingHeadLayers
            }
            sys::WAIFU_STAGE_LAYER_DECOMPOSITION_DECODING_HEAD_LAYERS => {
                Stage::LayerDecompositionDecodingHeadLayers
            }
            sys::WAIFU_STAGE_LAYER_DECOMPOSITION_ENCODING_FOR_DEPTH => {
                Stage::LayerDecompositionEncodingForDepth
            }
            sys::WAIFU_STAGE_LAYER_DECOMPOSITION_DENOISING_DEPTH => {
                Stage::LayerDecompositionDenoisingDepth
            }
            sys::WAIFU_STAGE_LAYER_DECOMPOSITION_DECODING_DEPTH => {
                Stage::LayerDecompositionDecodingDepth
            }
            sys::WAIFU_STAGE_LAYER_DECOMPOSITION_CUTTING_INTO_PARTS => {
                Stage::LayerDecompositionCuttingIntoParts
            }
            sys::WAIFU_STAGE_SPEECH_TO_TEXT_READING_RECORDING => {
                Stage::SpeechToTextReadingRecording
            }
            sys::WAIFU_STAGE_SPEECH_TO_TEXT_GENERATING_TOKENS => {
                Stage::SpeechToTextGeneratingTokens
            }
            sys::WAIFU_STAGE_STREAMING_TEXT_TO_SPEECH_READING_TEXT => {
                Stage::StreamingTextToSpeechReadingText
            }
            sys::WAIFU_STAGE_STREAMING_TEXT_TO_SPEECH_GENERATING_TOKENS => {
                Stage::StreamingTextToSpeechGeneratingTokens
            }
            sys::WAIFU_STAGE_STREAMING_TEXT_TO_SPEECH_SYNTHESIZING_AUDIO => {
                Stage::StreamingTextToSpeechSynthesizingAudio
            }
            other => Stage::Other(other),
        }
    }
}

/// What a layer of a decomposed image is: `WAIFU_LAYER_*`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum LayerType {
    /// A part as the model drew it.
    Part,
    /// `stratify_layers`: the near half of a part.
    StratifiedNear,
    /// `stratify_layers`: the far half of a part.
    StratifiedFar,
    /// The depth of the layer `derived_from_id` names, pixel by pixel.
    Depth,
    /// A type a newer library has and this crate does not know.
    Other(i32),
}

impl From<sys::WaifuLayerType> for LayerType {
    fn from(raw: sys::WaifuLayerType) -> LayerType {
        match raw {
            sys::WAIFU_LAYER_PART => LayerType::Part,
            sys::WAIFU_LAYER_STRATIFIED_NEAR => LayerType::StratifiedNear,
            sys::WAIFU_LAYER_STRATIFIED_FAR => LayerType::StratifiedFar,
            sys::WAIFU_LAYER_DEPTH => LayerType::Depth,
            other => LayerType::Other(other),
        }
    }
}

// -- what goes in and comes out --------------------------------------------------------------

/// A picture as rows of RGB, three bytes a pixel, top to bottom.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Image {
    pub width: u32,
    pub height: u32,
    pub rgb: Vec<u8>,
}

/// A picture as rows of RGBA, four bytes a pixel, not premultiplied.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RgbaImage {
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
}

/// Mono samples, -1..1.
#[derive(Clone, Debug, PartialEq)]
pub struct Audio {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
}

/// What a recording says.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Transcript {
    /// "" for nothing said.
    pub text: String,
    /// As the model names it -- "Chinese" -- or several, comma separated, for a recording that
    /// changes; "" where none was heard.
    pub language: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Layer {
    pub layer_type: LayerType,
    /// Unique within the result, from 1; it means nothing else.
    pub id: u32,
    /// "topwear"; its halves "topwear-0" and "topwear-1"; its depth "topwear_depth".
    pub name: String,
    /// Where its top left corner is on the canvas.
    pub left: i32,
    pub top: i32,
    /// Only as large as what is drawn. A depth layer: its layer's size and place, R, G and B the
    /// depth, 0 nearest to 255 farthest, under its layer's alpha.
    pub image: RgbaImage,
    /// 0 nearest, 1 farthest. A near half has its part's depth, not its pixels'; a depth layer
    /// has its layer's.
    pub depth: f32,
    /// The id of the layer this one is made from: a half's part, a depth layer's layer; None for
    /// a part.
    pub derived_from_id: Option<u32>,
}

/// An image taken apart. `layers` is in the library's one order: each part, bottom first, then,
/// with `stratify_layers`, each part's halves, near before far; then a depth layer for each.
#[derive(Clone, Debug, PartialEq)]
pub struct DecomposedImage {
    /// The side of the square canvas.
    pub canvas: u32,
    pub layers: Vec<Layer>,
}

impl DecomposedImage {
    /// The stack unstratified, bottom first: every part, as indices into `layers`.
    pub fn stack(&self) -> Vec<usize> {
        (0..self.layers.len())
            .filter(|&index| self.layers[index].layer_type == LayerType::Part)
            .collect()
    }

    /// The stratified stack, bottom first, as indices into `layers`: every part that was not split
    /// and every half -- from the largest depth to the smallest; at one depth, a far half under a
    /// near one; then by their order in `layers`. No depth layer is in it.
    pub fn stratified_stack(&self) -> Vec<usize> {
        let is_half = |layer: &Layer| {
            matches!(
                layer.layer_type,
                LayerType::StratifiedNear | LayerType::StratifiedFar
            )
        };
        let split: HashSet<u32> = self
            .layers
            .iter()
            .filter(|layer| is_half(layer))
            .filter_map(|layer| layer.derived_from_id)
            .collect();
        let mut stack: Vec<usize> = (0..self.layers.len())
            .filter(|&index| {
                let layer = &self.layers[index];
                match layer.layer_type {
                    LayerType::Part => !split.contains(&layer.id),
                    _ => is_half(layer),
                }
            })
            .collect();
        let lower = |layer_type: LayerType| match layer_type {
            LayerType::StratifiedFar => 0,
            _ => 1,
        };
        stack.sort_by(|&a, &b| {
            let (a_layer, b_layer) = (&self.layers[a], &self.layers[b]);
            b_layer
                .depth
                .total_cmp(&a_layer.depth)
                .then(lower(a_layer.layer_type).cmp(&lower(b_layer.layer_type)))
                .then(a.cmp(&b))
        });
        stack
    }
}

/// The picture a run starts from: img2img.
#[derive(Clone, Debug, PartialEq)]
pub struct StartImage {
    pub image: Image,
    /// How far it walks away from it, 0..1.
    pub strength: f32,
}

/// A ControlNet's condition.
#[derive(Clone, Debug, PartialEq)]
pub struct ControlImage {
    pub image: Image,
    /// How hard it holds to it.
    pub scale: f32,
}

/// An image the prompt names as `<|key|>`.
#[derive(Clone, Debug, PartialEq)]
pub struct PromptImage {
    /// Letters, digits and _.
    pub key: String,
    pub image: Image,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ImageGenerationRequest {
    pub prompt: String,
    /// "" for none.
    pub negative: String,
    /// 64..2048, a multiple of the model's `image_generation_width_height_multiple_of`.
    pub width: u32,
    pub height: u32,
    pub steps: u32,
    pub guidance: f32,
    pub seed: u64,
    pub start_image: Option<StartImage>,
    pub control_image: Option<ControlImage>,
    pub prompt_images: Vec<PromptImage>,
}

impl ImageGenerationRequest {
    /// 1024 x 1024, 20 steps, guidance 5, seed 0, from noise.
    pub fn new(prompt: impl Into<String>) -> ImageGenerationRequest {
        ImageGenerationRequest {
            prompt: prompt.into(),
            negative: String::new(),
            width: 1024,
            height: 1024,
            steps: 20,
            guidance: 5.0,
            seed: 0,
            start_image: None,
            control_image: None,
            prompt_images: Vec::new(),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct TextToSpeechRequest {
    pub text: String,
    /// A recording to sound like.
    pub reference: Option<Audio>,
    /// What `reference` says, word for word; None where it is not known. CosyVoice3 reads on
    /// from it, as upstream's zero-shot does; a model that reads none ignores it.
    pub reference_transcript: Option<String>,
    pub speed: f32,
    pub temperature: f32,
    /// Any instruction; None for the voice's own way. CosyVoice3 takes a style or a
    /// reference_transcript, not both.
    pub style: Option<String>,
    pub seed: u64,
}

impl TextToSpeechRequest {
    /// At speed 1, temperature 1, seed 0, in the voice's own way and voice.
    pub fn new(text: impl Into<String>) -> TextToSpeechRequest {
        TextToSpeechRequest {
            text: text.into(),
            reference: None,
            reference_transcript: None,
            speed: 1.0,
            temperature: 1.0,
            style: None,
            seed: 0,
        }
    }
}

/// No speed: a streaming reading is at the speed it is said.
#[derive(Clone, Debug, PartialEq)]
pub struct StreamingTextToSpeechRequest {
    pub text: String,
    pub reference: Option<Audio>,
    /// As TextToSpeechRequest's.
    pub reference_transcript: Option<String>,
    pub temperature: f32,
    pub style: Option<String>,
    pub seed: u64,
}

impl StreamingTextToSpeechRequest {
    /// At temperature 1, seed 0, in the voice's own way and voice.
    pub fn new(text: impl Into<String>) -> StreamingTextToSpeechRequest {
        StreamingTextToSpeechRequest {
            text: text.into(),
            reference: None,
            reference_transcript: None,
            temperature: 1.0,
            style: None,
            seed: 0,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct VoiceConversionRequest {
    /// What is said, and how.
    pub source: Audio,
    /// Whose voice to say it in.
    pub reference: Audio,
    pub steps: u32,
    pub convert_style: bool,
    pub seed: u64,
}

impl VoiceConversionRequest {
    /// 10 steps, the voice only, seed 0.
    pub fn new(source: Audio, reference: Audio) -> VoiceConversionRequest {
        VoiceConversionRequest {
            source,
            reference,
            steps: 10,
            convert_style: false,
            seed: 0,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct LayerDecompositionRequest {
    /// Any size, centred and scaled by the library; transparent is fine.
    pub image: RgbaImage,
    /// 0 for the model's own.
    pub steps: u32,
    /// 0 for the model's own.
    pub depth_steps: u32,
    /// A multiple of the model's `layer_decomposition_resolution_multiple_of`; 0 for 1280.
    pub resolution: u32,
    /// Hands, eyes, eyebrows, ears into a -l and a -r layer.
    pub split_pairs: bool,
    /// Hair, hands, top and bottom also cut into a near and a far half.
    pub stratify_layers: bool,
    pub seed: u64,
}

impl LayerDecompositionRequest {
    /// The model's own steps and resolution, nothing split or stratified, seed 0.
    pub fn new(image: RgbaImage) -> LayerDecompositionRequest {
        LayerDecompositionRequest {
            image,
            steps: 0,
            depth_steps: 0,
            resolution: 0,
            split_pairs: false,
            stratify_layers: false,
            seed: 0,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct SpeechToTextRequest {
    /// Any rate, any length.
    pub audio: Audio,
    /// One of the model's `speech_to_text_languages`, or None for the model to tell.
    pub language: Option<String>,
    /// Words that help it hear: names, terms, the subject; "" for none.
    pub context: String,
}

impl SpeechToTextRequest {
    /// The language for the model to tell, and no context.
    pub fn new(audio: Audio) -> SpeechToTextRequest {
        SpeechToTextRequest {
            audio,
            language: None,
            context: String::new(),
        }
    }
}

// -- jobs ------------------------------------------------------------------------------------

/// How far along a job is.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Progress {
    pub stage: Stage,
    /// Of the whole job, 0..1, where anything can say.
    pub fraction: Option<f64>,
    /// Of the stage, done already.
    pub done: u64,
    pub total: u64,
    /// Since the job started.
    pub seconds: f64,
}

/// What a job hands over, in the order it happens.
#[derive(Debug)]
pub enum Event<O, P = Infallible> {
    Progress(Progress),
    /// Streaming text to speech only.
    Part(P),
    /// Always the last.
    Done(Result<O>),
}

/// What a job's reader has handed over and its `Job` has not read yet.
struct State<O, P> {
    events: VecDeque<Event<O, P>>,
    waker: Option<Waker>,
    /// Walked with `next()`, which keeps progress and parts; awaited whole, which keeps neither.
    walked: bool,
}

/// What a job's reader writes into and its `Job` reads.
struct Shared<O, P> {
    state: Mutex<State<O, P>>,
}

impl<O, P> Shared<O, P> {
    fn new() -> Arc<Shared<O, P>> {
        Arc::new(Shared {
            state: Mutex::new(State {
                events: VecDeque::new(),
                waker: None,
                walked: true,
            }),
        })
    }

    fn state(&self) -> std::sync::MutexGuard<'_, State<O, P>> {
        self.state.lock().unwrap_or_else(|held| held.into_inner())
    }

    /// Adds `event` the way its kind is kept, and wakes whoever waits.
    fn hand_over(&self, event: Event<O, P>) {
        let waker = {
            let mut state = self.state();
            match event {
                Event::Progress(_) if !state.walked => return,
                Event::Part(_) if !state.walked => return,
                // A progress replaces the one not yet read.
                Event::Progress(_) => state
                    .events
                    .retain(|kept| !matches!(kept, Event::Progress(_))),
                _ => {}
            }
            state.events.push_back(event);
            state.waker.take()
        };
        if let Some(waker) = waker {
            waker.wake();
        }
    }
}

/// The library's task, until its reader has read it to the end and freed it. Cancelling and
/// freeing are both done holding the lock: a task must not be freed while another call is on it.
struct TaskHandle {
    task: Mutex<Option<TaskPointer>>,
    functions: &'static sys::Functions,
}

#[derive(Clone, Copy)]
struct TaskPointer(*mut sys::WaifuTask);

// The library's task may be read, waited on and cancelled from any thread.
unsafe impl Send for TaskPointer {}

impl TaskPointer {
    /// The pointer, taken by a method so that a closure moves the whole `TaskPointer` -- which is
    /// `Send` -- and not the field alone, which is not.
    fn get(self) -> *mut sys::WaifuTask {
        self.0
    }
}

impl TaskHandle {
    fn cancel(&self) {
        let task = self.task.lock().unwrap_or_else(|held| held.into_inner());
        if let Some(TaskPointer(task)) = *task {
            unsafe { (self.functions.waifu_task_cancel)(task) };
        }
    }

    fn free(&self) {
        let mut task = self.task.lock().unwrap_or_else(|held| held.into_inner());
        if let Some(TaskPointer(task)) = task.take() {
            unsafe { (self.functions.waifu_task_free)(task) };
        }
    }
}

/// What a task's attributes say, read once it has ended or as each event comes.
mod read {
    use super::*;

    pub unsafe fn int(
        f: &sys::Functions,
        task: *mut sys::WaifuTask,
        attribute: i32,
    ) -> Option<i64> {
        let mut value = 0;
        (f.waifu_task_get_int_attribute)(task, attribute, &mut value).then_some(value)
    }

    pub unsafe fn float(
        f: &sys::Functions,
        task: *mut sys::WaifuTask,
        attribute: i32,
    ) -> Option<f64> {
        let mut value = 0.0;
        (f.waifu_task_get_float_attribute)(task, attribute, &mut value).then_some(value)
    }

    pub unsafe fn string(
        f: &sys::Functions,
        task: *mut sys::WaifuTask,
        attribute: i32,
    ) -> Option<String> {
        text((f.waifu_task_get_string_attribute)(task, attribute))
    }

    pub unsafe fn audio(
        f: &sys::Functions,
        task: *mut sys::WaifuTask,
        attribute: i32,
    ) -> Option<Audio> {
        let mut audio = sys::WaifuAudio {
            samples: ptr::null(),
            count: 0,
            sample_rate: 0,
        };
        (f.waifu_task_get_audio_attribute)(task, attribute, &mut audio).then(|| Audio {
            samples: match audio.count {
                0 => Vec::new(),
                count => std::slice::from_raw_parts(audio.samples, count).to_vec(),
            },
            sample_rate: audio.sample_rate,
        })
    }

    pub unsafe fn image(
        f: &sys::Functions,
        task: *mut sys::WaifuTask,
        attribute: i32,
    ) -> Option<Image> {
        let mut image = sys::WaifuImage {
            width: 0,
            height: 0,
            rgb: ptr::null(),
            len: 0,
        };
        (f.waifu_task_get_image_attribute)(task, attribute, &mut image).then(|| Image {
            width: image.width,
            height: image.height,
            rgb: match image.len {
                0 => Vec::new(),
                len => std::slice::from_raw_parts(image.rgb, len).to_vec(),
            },
        })
    }

    pub unsafe fn int_at(
        f: &sys::Functions,
        task: *mut sys::WaifuTask,
        attribute: i32,
        index: usize,
    ) -> Option<i64> {
        let mut value = 0;
        (f.waifu_task_get_int_attribute_at)(task, attribute, index, &mut value).then_some(value)
    }

    pub unsafe fn layer(
        f: &sys::Functions,
        task: *mut sys::WaifuTask,
        index: usize,
    ) -> Option<Layer> {
        let int = |attribute| int_at(f, task, attribute, index);
        let mut depth = 0.0;
        let mut image = sys::WaifuRgbaImage {
            width: 0,
            height: 0,
            rgba: ptr::null(),
            len: 0,
        };
        let found = (f.waifu_task_get_float_attribute_at)(
            task,
            sys::WAIFU_TASK_LAYER_DECOMPOSITION_LAYER_DEPTH,
            index,
            &mut depth,
        ) && (f.waifu_task_get_rgba_image_attribute_at)(
            task,
            sys::WAIFU_TASK_LAYER_DECOMPOSITION_LAYER_IMAGE,
            index,
            &mut image,
        );
        if !found {
            return None;
        }
        let derived_from_id = int(sys::WAIFU_TASK_LAYER_DECOMPOSITION_LAYER_DERIVED_FROM_ID)?;
        Some(Layer {
            layer_type: LayerType::from(int(sys::WAIFU_TASK_LAYER_DECOMPOSITION_LAYER_TYPE)? as i32),
            id: int(sys::WAIFU_TASK_LAYER_DECOMPOSITION_LAYER_ID)? as u32,
            name: text((f.waifu_task_get_string_attribute_at)(
                task,
                sys::WAIFU_TASK_LAYER_DECOMPOSITION_LAYER_NAME,
                index,
            ))
            .unwrap_or_default(),
            left: int(sys::WAIFU_TASK_LAYER_DECOMPOSITION_LAYER_LEFT)? as i32,
            top: int(sys::WAIFU_TASK_LAYER_DECOMPOSITION_LAYER_TOP)? as i32,
            image: RgbaImage {
                width: image.width,
                height: image.height,
                rgba: match image.len {
                    0 => Vec::new(),
                    len => std::slice::from_raw_parts(image.rgba, len).to_vec(),
                },
            },
            depth: depth as f32,
            derived_from_id: (derived_from_id != 0).then_some(derived_from_id as u32),
        })
    }

    /// The progress the last PROGRESS event said.
    pub unsafe fn progress(f: &sys::Functions, task: *mut sys::WaifuTask) -> Option<Progress> {
        Some(Progress {
            stage: Stage::from(int(f, task, sys::WAIFU_TASK_PROGRESS_STAGE)? as i32),
            fraction: float(f, task, sys::WAIFU_TASK_PROGRESS_FRACTION),
            done: int(f, task, sys::WAIFU_TASK_PROGRESS_DONE)? as u64,
            total: int(f, task, sys::WAIFU_TASK_PROGRESS_TOTAL)? as u64,
            seconds: float(f, task, sys::WAIFU_TASK_PROGRESS_SECONDS).unwrap_or(0.0),
        })
    }

    /// How an ended task ended: `Ok` for WAIFU_OK, the error otherwise.
    pub unsafe fn status(f: &sys::Functions, task: *mut sys::WaifuTask) -> Result<()> {
        match int(f, task, sys::WAIFU_TASK_STATUS) {
            Some(code) if code as i32 == sys::WAIFU_OK => Ok(()),
            Some(code) => Err(Error {
                code: ErrorCode::from_status(code as i32),
                message: string(f, task, sys::WAIFU_TASK_STATUS_MESSAGE).unwrap_or_default(),
            }),
            None => Err(Error {
                code: ErrorCode::Internal,
                message: "the library said the job had ended, and had no status for it".into(),
            }),
        }
    }
}

fn missing(what: &str) -> Error {
    Error {
        code: ErrorCode::Internal,
        message: format!("the library said the job was done, and handed over no {what}"),
    }
}

/// How a job's reader reads its output and its parts.
struct Reading<O, P> {
    output: unsafe fn(&sys::Functions, *mut sys::WaifuTask) -> Result<O>,
    part: unsafe fn(&sys::Functions, *mut sys::WaifuTask) -> Option<P>,
}

unsafe fn no_part(_: &sys::Functions, _: *mut sys::WaifuTask) -> Option<Infallible> {
    None
}

unsafe fn nothing(_: &sys::Functions, _: *mut sys::WaifuTask) -> Result<()> {
    Ok(())
}

/// Reads `task` to its end on a thread of its own, handing each event to `shared`, then frees it:
/// the library calls nothing of ours, so something has to wait on it for a `Job`'s waker.
fn read_to_the_end<O: Send + 'static, P: Send + 'static>(
    handle: Arc<TaskHandle>,
    task: TaskPointer,
    shared: Arc<Shared<O, P>>,
    reading: Reading<O, P>,
) {
    let f = handle.functions;
    let (told, kept) = (Arc::clone(&shared), Arc::clone(&handle));
    let read = move || {
        let task = task.get();
        loop {
            match unsafe { (f.waifu_task_next_event)(task, sys::WAIFU_WAIT_FOREVER) } {
                sys::WAIFU_EVENT_PROGRESS => {
                    if let Some(progress) = unsafe { read::progress(f, task) } {
                        shared.hand_over(Event::Progress(progress));
                    }
                }
                sys::WAIFU_EVENT_AUDIO_PART => {
                    if let Some(part) = unsafe { (reading.part)(f, task) } {
                        shared.hand_over(Event::Part(part));
                    }
                }
                sys::WAIFU_EVENT_END => {
                    let done =
                        unsafe { read::status(f, task).and_then(|()| (reading.output)(f, task)) };
                    handle.free();
                    shared.hand_over(Event::Done(done));
                    return;
                }
                // NONE never comes without a timeout; anything else is a type this crate did not
                // ask for, and is let go of.
                _ => {}
            }
        }
    };
    if let Err(error) = std::thread::Builder::new()
        .name("waifu-job".to_string())
        .spawn(read)
    {
        // Without a reader the job would never be told its end: it is told now, and the task
        // let go of.
        kept.cancel();
        kept.free();
        told.hand_over(Event::Done(Err(Error {
            code: ErrorCode::Internal,
            message: format!("could not start a thread to wait on the job: {error}"),
        })));
    }
}

/// A job a pipeline was asked for. Dropped before its end, it is cancelled.
///
/// Walk it with [`next`](Job::next) for its progress -- and a streaming reading's parts -- as they
/// come, or `.await` it for its end alone.
pub struct Job<O, P = Infallible> {
    /// Kept so that the pipeline outlives every job of it.
    _pipeline: Arc<Inner>,
    handle: Arc<TaskHandle>,
    shared: Arc<Shared<O, P>>,
    /// Whether `Done` has been read.
    ended: bool,
}

impl<O, P> Job<O, P> {
    /// The next thing the job hands over; None after `Done`.
    pub async fn next(&mut self) -> Option<Event<O, P>> {
        std::future::poll_fn(|context| self.poll_next(context)).await
    }

    fn poll_next(&mut self, context: &mut Context<'_>) -> Poll<Option<Event<O, P>>> {
        if self.ended {
            return Poll::Ready(None);
        }
        let mut state = self.shared.state();
        match state.events.pop_front() {
            Some(event) => {
                if let Event::Done(_) = event {
                    self.ended = true;
                }
                Poll::Ready(Some(event))
            }
            None => {
                state.waker = Some(context.waker().clone());
                Poll::Pending
            }
        }
    }

    /// Stops it: after the step it is on where it runs, before it starts where it waits.
    pub fn cancel(&self) {
        self.handle.cancel();
    }
}

impl<O, P> Drop for Job<O, P> {
    fn drop(&mut self) {
        if !self.ended {
            self.cancel();
        }
    }
}

impl<O, P> std::fmt::Debug for Job<O, P> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Job")
            .field("ended", &self.ended)
            .finish_non_exhaustive()
    }
}

/// What `.await` on a [`Job`] polls. Not nameable outside the crate: a job is awaited, not this.
mod awaited {
    use super::*;

    /// A job awaited whole: its end, and none of its progress or parts.
    pub struct JobFuture<O, P = Infallible> {
        pub(super) job: Job<O, P>,
    }

    impl<O, P> Future for JobFuture<O, P> {
        type Output = Result<O>;

        fn poll(self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Result<O>> {
            let job = &mut self.get_mut().job;
            if job.ended {
                return Poll::Ready(Err(Error {
                    code: ErrorCode::Internal,
                    message: "the job's end was read already".to_string(),
                }));
            }
            let mut state = job.shared.state();
            state.walked = false;
            let done = state.events.drain(..).find_map(|event| match event {
                Event::Done(done) => Some(done),
                _ => None,
            });
            match done {
                Some(done) => {
                    drop(state);
                    job.ended = true;
                    Poll::Ready(done)
                }
                None => {
                    state.waker = Some(context.waker().clone());
                    Poll::Pending
                }
            }
        }
    }

    impl<O, P> IntoFuture for Job<O, P> {
        type Output = Result<O>;
        type IntoFuture = JobFuture<O, P>;

        fn into_future(self) -> JobFuture<O, P> {
            JobFuture { job: self }
        }
    }
}

// -- pipelines -------------------------------------------------------------------------------

/// The library's pipeline, freed when the last `Pipeline` and the last `Job` of it are dropped.
struct Inner {
    handle: *mut sys::WaifuPipeline,
    functions: &'static sys::Functions,
}

// Every function may be called on a pipeline from any thread; freeing it is ordered by the `Arc`.
unsafe impl Send for Inner {}
unsafe impl Sync for Inner {}

impl Drop for Inner {
    fn drop(&mut self) {
        // Returns once every task has ended; each job's reader then reads its end and frees it.
        unsafe { (self.functions.waifu_pipeline_free)(self.handle) };
    }
}

/// What a request is made of on its way to the library. Public in a private module: a sealed
/// trait names it, and nothing outside the crate can.
mod raw {
    use super::*;

    /// A request of the library's, set an attribute at a time and freed when this is dropped.
    pub struct RawRequest {
        pub(super) handle: *mut sys::WaifuRequest,
        f: &'static sys::Functions,
    }

    impl Drop for RawRequest {
        fn drop(&mut self) {
            unsafe { (self.f.waifu_request_free)(self.handle) };
        }
    }

    impl RawRequest {
        pub(super) fn new(f: &'static sys::Functions, r#type: PipelineType) -> Result<RawRequest> {
            let handle = unsafe { (f.waifu_request_new)(r#type.raw()) };
            match handle.is_null() {
                true => Err(Error::refused(f)),
                false => Ok(RawRequest { handle, f }),
            }
        }

        pub(super) fn checked(&self, set: bool) -> Result<()> {
            match set {
                true => Ok(()),
                false => Err(Error::refused(self.f)),
            }
        }

        pub(super) fn int(&self, attribute: sys::WaifuRequestAttribute, value: i64) -> Result<()> {
            self.checked(unsafe {
                (self.f.waifu_request_set_int_attribute)(self.handle, attribute, value)
            })
        }

        pub(super) fn uint(&self, attribute: sys::WaifuRequestAttribute, value: u32) -> Result<()> {
            self.int(attribute, value as i64)
        }

        pub(super) fn seed(&self, attribute: sys::WaifuRequestAttribute, value: u64) -> Result<()> {
            self.int(attribute, value as i64)
        }

        pub(super) fn flag(
            &self,
            attribute: sys::WaifuRequestAttribute,
            value: bool,
        ) -> Result<()> {
            self.int(attribute, value as i64)
        }

        pub(super) fn float(
            &self,
            attribute: sys::WaifuRequestAttribute,
            value: f32,
        ) -> Result<()> {
            self.checked(unsafe {
                (self.f.waifu_request_set_float_attribute)(self.handle, attribute, value as f64)
            })
        }

        pub(super) fn string(
            &self,
            attribute: sys::WaifuRequestAttribute,
            value: &str,
            what: &str,
        ) -> Result<()> {
            let value = c_text(value, what)?;
            self.checked(unsafe {
                (self.f.waifu_request_set_string_attribute)(self.handle, attribute, value.as_ptr())
            })
        }

        pub(super) fn optional_string(
            &self,
            attribute: sys::WaifuRequestAttribute,
            value: &Option<String>,
            what: &str,
        ) -> Result<()> {
            match value {
                Some(value) => self.string(attribute, value, what),
                None => Ok(()),
            }
        }

        pub(super) fn image(
            &self,
            attribute: sys::WaifuRequestAttribute,
            image: &Image,
        ) -> Result<()> {
            let raw = image_in(image);
            self.checked(unsafe {
                (self.f.waifu_request_set_image_attribute)(self.handle, attribute, &raw)
            })
        }

        pub(super) fn audio(
            &self,
            attribute: sys::WaifuRequestAttribute,
            audio: &Audio,
        ) -> Result<()> {
            let raw = audio_in(audio);
            self.checked(unsafe {
                (self.f.waifu_request_set_audio_attribute)(self.handle, attribute, &raw)
            })
        }

        pub(super) fn optional_audio(
            &self,
            attribute: sys::WaifuRequestAttribute,
            audio: &Option<Audio>,
        ) -> Result<()> {
            match audio {
                Some(audio) => self.audio(attribute, audio),
                None => Ok(()),
            }
        }
    }
}

use raw::RawRequest;

mod sealed {
    use super::*;

    /// How a task's run is asked of the library and read back. Not nameable outside this crate,
    /// so the six tasks below are all there are.
    pub trait Sealed {
        /// The library's request for `request`, every attribute set.
        fn request(
            f: &'static sys::Functions,
            request: &<Self as Task>::Request,
        ) -> Result<RawRequest>
        where
            Self: Task;

        /// The output of a task that ended well.
        ///
        /// # Safety
        /// `task` is a live task of this task's run that has ended.
        unsafe fn output(
            f: &sys::Functions,
            task: *mut sys::WaifuTask,
        ) -> Result<<Self as Task>::Output>
        where
            Self: Task;

        /// The part the last AUDIO_PART event said, for a task that has parts.
        ///
        /// # Safety
        /// `task` is a live task of this task's run.
        unsafe fn part(
            f: &sys::Functions,
            task: *mut sys::WaifuTask,
        ) -> Option<<Self as Task>::Part>
        where
            Self: Task;
    }
}

/// What a pipeline is made for. Implemented for the six tasks of this crate and nothing else, so
/// that a request, a result and a pipeline type cannot be paired by anyone but it.
pub trait Task: sealed::Sealed + 'static {
    /// `WAIFU_PIPELINE_*`.
    const TYPE: PipelineType;
    type Request;
    type Output: Send + 'static;
    /// What a streaming task hands over; `Infallible` for the rest.
    type Part: Send + 'static;
}

/// `ImageGenerationRequest` -> `Image`.
#[derive(Clone, Copy, Debug)]
pub struct ImageGeneration;
/// `TextToSpeechRequest` -> `Audio`.
#[derive(Clone, Copy, Debug)]
pub struct TextToSpeech;
/// `StreamingTextToSpeechRequest` -> `Audio`, and parts of `Audio`.
#[derive(Clone, Copy, Debug)]
pub struct StreamingTextToSpeech;
/// `SpeechToTextRequest` -> `Transcript`.
#[derive(Clone, Copy, Debug)]
pub struct SpeechToText;
/// `VoiceConversionRequest` -> `Audio`.
#[derive(Clone, Copy, Debug)]
pub struct VoiceConversion;
/// `LayerDecompositionRequest` -> `DecomposedImage`.
#[derive(Clone, Copy, Debug)]
pub struct LayerDecomposition;

impl Task for ImageGeneration {
    const TYPE: PipelineType = PipelineType::ImageGeneration;
    type Request = ImageGenerationRequest;
    type Output = Image;
    type Part = Infallible;
}

impl Task for TextToSpeech {
    const TYPE: PipelineType = PipelineType::TextToSpeech;
    type Request = TextToSpeechRequest;
    type Output = Audio;
    type Part = Infallible;
}

impl Task for StreamingTextToSpeech {
    const TYPE: PipelineType = PipelineType::StreamingTextToSpeech;
    type Request = StreamingTextToSpeechRequest;
    type Output = Audio;
    type Part = Audio;
}

impl Task for SpeechToText {
    const TYPE: PipelineType = PipelineType::SpeechToText;
    type Request = SpeechToTextRequest;
    type Output = Transcript;
    type Part = Infallible;
}

impl Task for VoiceConversion {
    const TYPE: PipelineType = PipelineType::VoiceConversion;
    type Request = VoiceConversionRequest;
    type Output = Audio;
    type Part = Infallible;
}

impl Task for LayerDecomposition {
    const TYPE: PipelineType = PipelineType::LayerDecomposition;
    type Request = LayerDecompositionRequest;
    type Output = DecomposedImage;
    type Part = Infallible;
}

/// A string as the library takes it.
fn c_text(text: &str, what: &str) -> Result<CString> {
    CString::new(text).map_err(|_| Error::invalid(format!("{what} has a NUL in it")))
}

fn image_in(image: &Image) -> sys::WaifuImage {
    sys::WaifuImage {
        width: image.width,
        height: image.height,
        rgb: image.rgb.as_ptr(),
        len: image.rgb.len(),
    }
}

fn audio_in(audio: &Audio) -> sys::WaifuAudio {
    sys::WaifuAudio {
        samples: audio.samples.as_ptr(),
        count: audio.samples.len(),
        sample_rate: audio.sample_rate,
    }
}

unsafe fn audio_output(
    f: &sys::Functions,
    task: *mut sys::WaifuTask,
    attribute: sys::WaifuTaskAttribute,
) -> Result<Audio> {
    read::audio(f, task, attribute).ok_or_else(|| missing("audio"))
}

impl sealed::Sealed for ImageGeneration {
    fn request(f: &'static sys::Functions, request: &ImageGenerationRequest) -> Result<RawRequest> {
        let raw = RawRequest::new(f, Self::TYPE)?;
        raw.string(
            sys::WAIFU_REQUEST_IMAGE_GENERATION_PROMPT,
            &request.prompt,
            "the prompt",
        )?;
        if !request.negative.is_empty() {
            raw.string(
                sys::WAIFU_REQUEST_IMAGE_GENERATION_NEGATIVE,
                &request.negative,
                "the negative prompt",
            )?;
        }
        raw.uint(sys::WAIFU_REQUEST_IMAGE_GENERATION_WIDTH, request.width)?;
        raw.uint(sys::WAIFU_REQUEST_IMAGE_GENERATION_HEIGHT, request.height)?;
        raw.uint(sys::WAIFU_REQUEST_IMAGE_GENERATION_STEPS, request.steps)?;
        raw.float(
            sys::WAIFU_REQUEST_IMAGE_GENERATION_GUIDANCE,
            request.guidance,
        )?;
        raw.seed(sys::WAIFU_REQUEST_IMAGE_GENERATION_SEED, request.seed)?;
        if let Some(start) = &request.start_image {
            raw.image(
                sys::WAIFU_REQUEST_IMAGE_GENERATION_START_IMAGE,
                &start.image,
            )?;
            raw.float(
                sys::WAIFU_REQUEST_IMAGE_GENERATION_START_IMAGE_STRENGTH,
                start.strength,
            )?;
        }
        if let Some(control) = &request.control_image {
            raw.image(
                sys::WAIFU_REQUEST_IMAGE_GENERATION_CONTROL_IMAGE,
                &control.image,
            )?;
            raw.float(
                sys::WAIFU_REQUEST_IMAGE_GENERATION_CONTROL_IMAGE_SCALE,
                control.scale,
            )?;
        }
        for prompt_image in &request.prompt_images {
            let key = c_text(&prompt_image.key, "a prompt image's key")?;
            let image = image_in(&prompt_image.image);
            raw.checked(unsafe {
                (f.waifu_request_add_prompt_image)(raw.handle, key.as_ptr(), &image)
            })?;
        }
        Ok(raw)
    }

    unsafe fn output(f: &sys::Functions, task: *mut sys::WaifuTask) -> Result<Image> {
        read::image(f, task, sys::WAIFU_TASK_IMAGE_GENERATION_IMAGE).ok_or_else(|| missing("image"))
    }

    unsafe fn part(_: &sys::Functions, _: *mut sys::WaifuTask) -> Option<Infallible> {
        None
    }
}

impl sealed::Sealed for TextToSpeech {
    fn request(f: &'static sys::Functions, request: &TextToSpeechRequest) -> Result<RawRequest> {
        let raw = RawRequest::new(f, Self::TYPE)?;
        raw.string(
            sys::WAIFU_REQUEST_TEXT_TO_SPEECH_TEXT,
            &request.text,
            "the text",
        )?;
        raw.optional_audio(
            sys::WAIFU_REQUEST_TEXT_TO_SPEECH_REFERENCE,
            &request.reference,
        )?;
        raw.optional_string(
            sys::WAIFU_REQUEST_TEXT_TO_SPEECH_REFERENCE_TRANSCRIPT,
            &request.reference_transcript,
            "the reference transcript",
        )?;
        raw.float(sys::WAIFU_REQUEST_TEXT_TO_SPEECH_SPEED, request.speed)?;
        raw.float(
            sys::WAIFU_REQUEST_TEXT_TO_SPEECH_TEMPERATURE,
            request.temperature,
        )?;
        raw.optional_string(
            sys::WAIFU_REQUEST_TEXT_TO_SPEECH_STYLE,
            &request.style,
            "the style",
        )?;
        raw.seed(sys::WAIFU_REQUEST_TEXT_TO_SPEECH_SEED, request.seed)?;
        Ok(raw)
    }

    unsafe fn output(f: &sys::Functions, task: *mut sys::WaifuTask) -> Result<Audio> {
        audio_output(f, task, sys::WAIFU_TASK_TEXT_TO_SPEECH_AUDIO)
    }

    unsafe fn part(_: &sys::Functions, _: *mut sys::WaifuTask) -> Option<Infallible> {
        None
    }
}

impl sealed::Sealed for StreamingTextToSpeech {
    fn request(
        f: &'static sys::Functions,
        request: &StreamingTextToSpeechRequest,
    ) -> Result<RawRequest> {
        let raw = RawRequest::new(f, Self::TYPE)?;
        raw.string(
            sys::WAIFU_REQUEST_STREAMING_TEXT_TO_SPEECH_TEXT,
            &request.text,
            "the text",
        )?;
        raw.optional_audio(
            sys::WAIFU_REQUEST_STREAMING_TEXT_TO_SPEECH_REFERENCE,
            &request.reference,
        )?;
        raw.optional_string(
            sys::WAIFU_REQUEST_STREAMING_TEXT_TO_SPEECH_REFERENCE_TRANSCRIPT,
            &request.reference_transcript,
            "the reference transcript",
        )?;
        raw.float(
            sys::WAIFU_REQUEST_STREAMING_TEXT_TO_SPEECH_TEMPERATURE,
            request.temperature,
        )?;
        raw.optional_string(
            sys::WAIFU_REQUEST_STREAMING_TEXT_TO_SPEECH_STYLE,
            &request.style,
            "the style",
        )?;
        raw.seed(
            sys::WAIFU_REQUEST_STREAMING_TEXT_TO_SPEECH_SEED,
            request.seed,
        )?;
        Ok(raw)
    }

    unsafe fn output(f: &sys::Functions, task: *mut sys::WaifuTask) -> Result<Audio> {
        audio_output(f, task, sys::WAIFU_TASK_STREAMING_TEXT_TO_SPEECH_AUDIO)
    }

    unsafe fn part(f: &sys::Functions, task: *mut sys::WaifuTask) -> Option<Audio> {
        read::audio(f, task, sys::WAIFU_TASK_STREAMING_TEXT_TO_SPEECH_AUDIO_PART)
    }
}

impl sealed::Sealed for SpeechToText {
    fn request(f: &'static sys::Functions, request: &SpeechToTextRequest) -> Result<RawRequest> {
        let raw = RawRequest::new(f, Self::TYPE)?;
        raw.audio(sys::WAIFU_REQUEST_SPEECH_TO_TEXT_AUDIO, &request.audio)?;
        raw.optional_string(
            sys::WAIFU_REQUEST_SPEECH_TO_TEXT_LANGUAGE,
            &request.language,
            "the language",
        )?;
        if !request.context.is_empty() {
            raw.string(
                sys::WAIFU_REQUEST_SPEECH_TO_TEXT_CONTEXT,
                &request.context,
                "the context",
            )?;
        }
        Ok(raw)
    }

    unsafe fn output(f: &sys::Functions, task: *mut sys::WaifuTask) -> Result<Transcript> {
        match read::string(f, task, sys::WAIFU_TASK_SPEECH_TO_TEXT_TEXT) {
            Some(said) => Ok(Transcript {
                text: said,
                language: read::string(f, task, sys::WAIFU_TASK_SPEECH_TO_TEXT_LANGUAGE)
                    .unwrap_or_default(),
            }),
            None => Err(missing("transcript")),
        }
    }

    unsafe fn part(_: &sys::Functions, _: *mut sys::WaifuTask) -> Option<Infallible> {
        None
    }
}

impl sealed::Sealed for VoiceConversion {
    fn request(f: &'static sys::Functions, request: &VoiceConversionRequest) -> Result<RawRequest> {
        let raw = RawRequest::new(f, Self::TYPE)?;
        raw.audio(sys::WAIFU_REQUEST_VOICE_CONVERSION_SOURCE, &request.source)?;
        raw.audio(
            sys::WAIFU_REQUEST_VOICE_CONVERSION_REFERENCE,
            &request.reference,
        )?;
        raw.uint(sys::WAIFU_REQUEST_VOICE_CONVERSION_STEPS, request.steps)?;
        raw.flag(
            sys::WAIFU_REQUEST_VOICE_CONVERSION_CONVERT_STYLE,
            request.convert_style,
        )?;
        raw.seed(sys::WAIFU_REQUEST_VOICE_CONVERSION_SEED, request.seed)?;
        Ok(raw)
    }

    unsafe fn output(f: &sys::Functions, task: *mut sys::WaifuTask) -> Result<Audio> {
        audio_output(f, task, sys::WAIFU_TASK_VOICE_CONVERSION_AUDIO)
    }

    unsafe fn part(_: &sys::Functions, _: *mut sys::WaifuTask) -> Option<Infallible> {
        None
    }
}

impl sealed::Sealed for LayerDecomposition {
    fn request(
        f: &'static sys::Functions,
        request: &LayerDecompositionRequest,
    ) -> Result<RawRequest> {
        let raw = RawRequest::new(f, Self::TYPE)?;
        let image = sys::WaifuRgbaImage {
            width: request.image.width,
            height: request.image.height,
            rgba: request.image.rgba.as_ptr(),
            len: request.image.rgba.len(),
        };
        raw.checked(unsafe {
            (f.waifu_request_set_rgba_image_attribute)(
                raw.handle,
                sys::WAIFU_REQUEST_LAYER_DECOMPOSITION_IMAGE,
                &image,
            )
        })?;
        // 0 is this crate's "the model's own", which an unset attribute is.
        for (attribute, value) in [
            (sys::WAIFU_REQUEST_LAYER_DECOMPOSITION_STEPS, request.steps),
            (
                sys::WAIFU_REQUEST_LAYER_DECOMPOSITION_DEPTH_STEPS,
                request.depth_steps,
            ),
            (
                sys::WAIFU_REQUEST_LAYER_DECOMPOSITION_RESOLUTION,
                request.resolution,
            ),
        ] {
            if value != 0 {
                raw.uint(attribute, value)?;
            }
        }
        raw.flag(
            sys::WAIFU_REQUEST_LAYER_DECOMPOSITION_SPLIT_PAIRS,
            request.split_pairs,
        )?;
        raw.flag(
            sys::WAIFU_REQUEST_LAYER_DECOMPOSITION_STRATIFY_LAYERS,
            request.stratify_layers,
        )?;
        raw.seed(sys::WAIFU_REQUEST_LAYER_DECOMPOSITION_SEED, request.seed)?;
        Ok(raw)
    }

    unsafe fn output(f: &sys::Functions, task: *mut sys::WaifuTask) -> Result<DecomposedImage> {
        let canvas = read::int(f, task, sys::WAIFU_TASK_LAYER_DECOMPOSITION_CANVAS)
            .ok_or_else(|| missing("canvas"))?;
        let count = read::int(f, task, sys::WAIFU_TASK_LAYER_DECOMPOSITION_LAYER_COUNT)
            .ok_or_else(|| missing("layer count"))?;
        let layers = (0..count as usize)
            .map(|index| read::layer(f, task, index).ok_or_else(|| missing("layer")))
            .collect::<Result<Vec<_>>>()?;
        Ok(DecomposedImage {
            canvas: canvas as u32,
            layers,
        })
    }

    unsafe fn part(_: &sys::Functions, _: *mut sys::WaifuTask) -> Option<Infallible> {
        None
    }
}

/// A pipeline made for task `T` on one device: one thread in the library, at most one model, jobs
/// run one at a time in the order asked. Cloned, it is the same pipeline.
pub struct Pipeline<T: Task> {
    inner: Arc<Inner>,
    task: PhantomData<fn() -> T>,
}

impl<T: Task> Clone for Pipeline<T> {
    fn clone(&self) -> Pipeline<T> {
        Pipeline {
            inner: Arc::clone(&self.inner),
            task: PhantomData,
        }
    }
}

impl<T: Task> std::fmt::Debug for Pipeline<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Pipeline")
            .field("type", &T::TYPE)
            .field("device", &self.device())
            .finish()
    }
}

impl<T: Task> Pipeline<T> {
    /// Starts the pipeline's thread, with no model. Refused with `DeviceUnavailable` where the
    /// device cannot be used.
    pub fn new(device: Device) -> Result<Pipeline<T>> {
        let f = sys::functions()?;
        let handle = unsafe { (f.waifu_pipeline_new)(T::TYPE.raw(), device.raw()) };
        match handle.is_null() {
            true => Err(Error::refused(f)),
            false => Ok(Pipeline {
                inner: Arc::new(Inner {
                    handle,
                    functions: f,
                }),
                task: PhantomData,
            }),
        }
    }

    /// What it runs on: never `Auto`, which is settled when it starts.
    pub fn device(&self) -> Device {
        let raw = unsafe { (self.inner.functions.waifu_pipeline_get_device)(self.inner.handle) };
        Device::from_raw(raw)
    }

    /// The task the library started, kept as a `Job` with a reader of its own; refused where the
    /// library returned none.
    fn job<O: Send + 'static, P: Send + 'static>(
        &self,
        task: *mut sys::WaifuTask,
        reading: Reading<O, P>,
    ) -> Result<Job<O, P>> {
        let f = self.inner.functions;
        if task.is_null() {
            return Err(Error::refused(f));
        }
        let handle = Arc::new(TaskHandle {
            task: Mutex::new(Some(TaskPointer(task))),
            functions: f,
        });
        let shared = Shared::<O, P>::new();
        read_to_the_end(
            Arc::clone(&handle),
            TaskPointer(task),
            Arc::clone(&shared),
            reading,
        );
        Ok(Job {
            _pipeline: Arc::clone(&self.inner),
            handle,
            shared,
            ended: false,
        })
    }

    /// Reads the model's weights onto the device, dropping the model before first. Refused at the
    /// call for a model the task is not supported by. `model` may be dropped as soon as this
    /// returns.
    pub fn load_model(&self, model: &Model) -> Result<Job<()>> {
        let (f, handle) = (self.inner.functions, self.inner.handle);
        let task = unsafe { (f.waifu_pipeline_load_model)(handle, model.raw()) };
        self.job(
            task,
            Reading {
                output: nothing,
                part: no_part,
            },
        )
    }

    pub fn unload_model(&self) -> Result<Job<()>> {
        let (f, handle) = (self.inner.functions, self.inner.handle);
        let task = unsafe { (f.waifu_pipeline_unload_model)(handle) };
        self.job(
            task,
            Reading {
                output: nothing,
                part: no_part,
            },
        )
    }

    /// Runs the task. Refused at the call for a request that cannot be run; the request is copied,
    /// and may be dropped as soon as this returns.
    pub fn run(&self, request: &T::Request) -> Result<Job<T::Output, T::Part>> {
        let (f, handle) = (self.inner.functions, self.inner.handle);
        let raw = T::request(f, request)?;
        let task = unsafe { (f.waifu_pipeline_run)(handle, raw.handle) };
        self.job(
            task,
            Reading {
                output: T::output,
                part: T::part,
            },
        )
    }

    /// Stops what runs after its step and cancels what waits.
    pub fn cancel_all(&self) {
        unsafe { (self.inner.functions.waifu_pipeline_cancel_all)(self.inner.handle) };
    }
}
