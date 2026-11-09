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
//! A job's state outlives its [`Job`]: each run hands the library an `Arc` of it, which the
//! library's callbacks only copy into -- no caller code runs on a pipeline's thread -- and which
//! `on_complete`, called exactly once for every job given an id, takes back.

use std::collections::{HashSet, VecDeque};
use std::convert::Infallible;
use std::ffi::{c_char, c_void, CString};
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
    pub speed: f32,
    pub temperature: f32,
    /// Any instruction; None for the voice's own way.
    pub style: Option<String>,
    pub seed: u64,
}

impl TextToSpeechRequest {
    /// At speed 1, temperature 1, seed 0, in the voice's own way and voice.
    pub fn new(text: impl Into<String>) -> TextToSpeechRequest {
        TextToSpeechRequest {
            text: text.into(),
            reference: None,
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

/// What a job's callbacks have handed over and its `Job` has not read yet.
struct State<O, P> {
    events: VecDeque<Event<O, P>>,
    waker: Option<Waker>,
    /// Walked with `next()`, which keeps progress and parts; awaited whole, which keeps neither.
    walked: bool,
}

/// What the library's callbacks write into: one `Arc` of it is the library's until `on_complete`.
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

/// The user data a job's callbacks are handed: its `Shared`, the library's reference to it.
fn user_data<O, P>(shared: &Arc<Shared<O, P>>) -> *mut c_void {
    Arc::into_raw(Arc::clone(shared)) as *mut c_void
}

/// Takes back the library's reference: for `on_complete`, and for a call that was refused.
unsafe fn taken_back<O, P>(user_data: *mut c_void) -> Arc<Shared<O, P>> {
    Arc::from_raw(user_data as *const Shared<O, P>)
}

unsafe fn shared<'a, O, P>(user_data: *mut c_void) -> &'a Shared<O, P> {
    &*(user_data as *const Shared<O, P>)
}

unsafe extern "C" fn on_progress<O, P>(
    user_data: *mut c_void,
    progress: *const sys::WaifuProgress,
) {
    let progress = &*progress;
    shared::<O, P>(user_data).hand_over(Event::Progress(Progress {
        stage: Stage::from(progress.stage),
        fraction: (progress.fraction >= 0.0).then_some(progress.fraction),
        done: progress.done,
        total: progress.total,
        seconds: progress.seconds,
    }));
}

unsafe fn audio_out(audio: &sys::WaifuAudio) -> Audio {
    Audio {
        samples: match audio.count {
            0 => Vec::new(),
            count => std::slice::from_raw_parts(audio.samples, count).to_vec(),
        },
        sample_rate: audio.sample_rate,
    }
}

unsafe fn rgba_out(image: &sys::WaifuRgbaImage) -> RgbaImage {
    RgbaImage {
        width: image.width,
        height: image.height,
        rgba: match image.len {
            0 => Vec::new(),
            len => std::slice::from_raw_parts(image.rgba, len).to_vec(),
        },
    }
}

unsafe extern "C" fn on_part(user_data: *mut c_void, part: *const sys::WaifuAudio) {
    shared::<Audio, Audio>(user_data).hand_over(Event::Part(audio_out(&*part)));
}

/// What a status says: `Ok` for WAIFU_OK, the error otherwise.
unsafe fn status_of(status: &sys::WaifuStatus) -> Result<()> {
    match status.code {
        sys::WAIFU_OK => Ok(()),
        code => Err(Error {
            code: ErrorCode::from_status(code),
            message: text(status.message).unwrap_or_default(),
        }),
    }
}

/// Ends the job with `done`, giving back the library's reference.
unsafe fn finish<O, P>(user_data: *mut c_void, done: Result<O>) {
    taken_back::<O, P>(user_data).hand_over(Event::Done(done));
}

unsafe extern "C" fn on_unit_complete(user_data: *mut c_void, result: *const sys::WaifuResult) {
    finish::<(), Infallible>(user_data, status_of(&(*result).status));
}

unsafe extern "C" fn on_image_complete(
    user_data: *mut c_void,
    result: *const sys::WaifuImageGenerationResult,
) {
    let result = &*result;
    let done = status_of(&result.status).and_then(|()| match result.image.as_ref() {
        Some(image) => Ok(Image {
            width: image.width,
            height: image.height,
            rgb: match image.len {
                0 => Vec::new(),
                len => std::slice::from_raw_parts(image.rgb, len).to_vec(),
            },
        }),
        None => Err(missing("an image")),
    });
    finish::<Image, Infallible>(user_data, done);
}

unsafe extern "C" fn on_audio_complete<P>(
    user_data: *mut c_void,
    result: *const sys::WaifuAudioResult,
) {
    let result = &*result;
    let done = status_of(&result.status).and_then(|()| match result.audio.as_ref() {
        Some(audio) => Ok(audio_out(audio)),
        None => Err(missing("audio")),
    });
    finish::<Audio, P>(user_data, done);
}

unsafe extern "C" fn on_layers_complete(
    user_data: *mut c_void,
    result: *const sys::WaifuLayerDecompositionResult,
) {
    let result = &*result;
    let done = status_of(&result.status).map(|()| {
        let layers = match result.layer_count {
            0 => &[][..],
            count => std::slice::from_raw_parts(result.layers, count),
        };
        DecomposedImage {
            canvas: result.canvas,
            layers: layers
                .iter()
                .map(|layer| Layer {
                    layer_type: LayerType::from(layer.r#type),
                    id: layer.id,
                    name: text(layer.name).unwrap_or_default(),
                    left: layer.left,
                    top: layer.top,
                    image: rgba_out(&layer.image),
                    depth: layer.depth,
                    derived_from_id: (layer.derived_from_id != 0).then_some(layer.derived_from_id),
                })
                .collect(),
        }
    });
    finish::<DecomposedImage, Infallible>(user_data, done);
}

unsafe extern "C" fn on_transcript_complete(
    user_data: *mut c_void,
    result: *const sys::WaifuSpeechToTextResult,
) {
    let result = &*result;
    let done = status_of(&result.status).and_then(|()| match text(result.text) {
        Some(said) => Ok(Transcript {
            text: said,
            language: text(result.language).unwrap_or_default(),
        }),
        None => Err(missing("a transcript")),
    });
    finish::<Transcript, Infallible>(user_data, done);
}

fn missing(what: &str) -> Error {
    Error {
        code: ErrorCode::Internal,
        message: format!("the library said the job was done, and handed over no {what}"),
    }
}

/// A job a pipeline was asked for. Dropped before its end, it is cancelled.
///
/// Walk it with [`next`](Job::next) for its progress -- and a streaming reading's parts -- as they
/// come, or `.await` it for its end alone.
pub struct Job<O, P = Infallible> {
    pipeline: Arc<Inner>,
    id: sys::WaifuJob,
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
        unsafe { (self.pipeline.functions.waifu_pipeline_cancel)(self.pipeline.handle, self.id) };
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
            .field("id", &self.id)
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
        // Returns once every job has had its on_complete, which runs nothing of the caller's.
        unsafe { (self.functions.waifu_pipeline_free)(self.handle) };
    }
}

mod sealed {
    use super::*;

    /// How a task's run is asked of the library. Not nameable outside this crate, so the six tasks
    /// below are all there are.
    pub trait Sealed {
        /// Asks for the run, with `user_data` for its callbacks: the job's id, 0 where refused.
        ///
        /// # Safety
        /// `user_data` is a `Shared<Self::Output, Self::Part>` the library's callbacks may take.
        unsafe fn start(
            functions: &sys::Functions,
            pipeline: *mut sys::WaifuPipeline,
            request: &<Self as Task>::Request,
            user_data: *mut c_void,
        ) -> Result<sys::WaifuJob>
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
    type Output;
    /// What a streaming task hands over; `Infallible` for the rest.
    type Part;
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

fn c_optional(text: &Option<String>, what: &str) -> Result<Option<CString>> {
    text.as_deref().map(|text| c_text(text, what)).transpose()
}

fn c_pointer(text: &Option<CString>) -> *const c_char {
    text.as_ref().map_or(ptr::null(), |text| text.as_ptr())
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

fn optional<T>(value: &Option<T>) -> *const T {
    value
        .as_ref()
        .map_or(ptr::null(), |value| value as *const T)
}

fn struct_size<T>() -> u32 {
    std::mem::size_of::<T>() as u32
}

impl sealed::Sealed for ImageGeneration {
    unsafe fn start(
        f: &sys::Functions,
        pipeline: *mut sys::WaifuPipeline,
        request: &ImageGenerationRequest,
        user_data: *mut c_void,
    ) -> Result<sys::WaifuJob> {
        let prompt = c_text(&request.prompt, "the prompt")?;
        let negative = c_text(&request.negative, "the negative prompt")?;
        let start = request
            .start_image
            .as_ref()
            .map(|start| image_in(&start.image));
        let control = request
            .control_image
            .as_ref()
            .map(|control| image_in(&control.image));
        let keys = request
            .prompt_images
            .iter()
            .map(|prompt_image| c_text(&prompt_image.key, "a prompt image's key"))
            .collect::<Result<Vec<_>>>()?;
        let images: Vec<sys::WaifuImage> = request
            .prompt_images
            .iter()
            .map(|prompt_image| image_in(&prompt_image.image))
            .collect();
        let prompt_images: Vec<sys::WaifuPromptImage> = keys
            .iter()
            .zip(&images)
            .map(|(key, image)| sys::WaifuPromptImage {
                key: key.as_ptr(),
                image,
            })
            .collect();
        let raw = sys::WaifuImageGenerationRequest {
            struct_size: struct_size::<sys::WaifuImageGenerationRequest>(),
            prompt: prompt.as_ptr(),
            negative: negative.as_ptr(),
            width: request.width,
            height: request.height,
            steps: request.steps,
            guidance: request.guidance,
            seed: request.seed,
            start_image: optional(&start),
            start_image_strength: request.start_image.as_ref().map_or(0.0, |s| s.strength),
            control_image: optional(&control),
            control_image_scale: request.control_image.as_ref().map_or(0.0, |c| c.scale),
            prompt_images: match prompt_images.is_empty() {
                true => ptr::null(),
                false => prompt_images.as_ptr(),
            },
            prompt_image_count: prompt_images.len(),
        };
        Ok((f.waifu_pipeline_run_image_generation_async)(
            pipeline,
            &raw,
            user_data,
            Some(on_progress::<Image, Infallible>),
            Some(on_image_complete),
        ))
    }
}

impl sealed::Sealed for TextToSpeech {
    unsafe fn start(
        f: &sys::Functions,
        pipeline: *mut sys::WaifuPipeline,
        request: &TextToSpeechRequest,
        user_data: *mut c_void,
    ) -> Result<sys::WaifuJob> {
        let said = c_text(&request.text, "the text")?;
        let style = c_optional(&request.style, "the style")?;
        let reference = request.reference.as_ref().map(audio_in);
        let raw = sys::WaifuTextToSpeechRequest {
            struct_size: struct_size::<sys::WaifuTextToSpeechRequest>(),
            text: said.as_ptr(),
            reference: optional(&reference),
            speed: request.speed,
            temperature: request.temperature,
            style: c_pointer(&style),
            seed: request.seed,
        };
        Ok((f.waifu_pipeline_run_text_to_speech_async)(
            pipeline,
            &raw,
            user_data,
            Some(on_progress::<Audio, Infallible>),
            Some(on_audio_complete::<Infallible>),
        ))
    }
}

impl sealed::Sealed for StreamingTextToSpeech {
    unsafe fn start(
        f: &sys::Functions,
        pipeline: *mut sys::WaifuPipeline,
        request: &StreamingTextToSpeechRequest,
        user_data: *mut c_void,
    ) -> Result<sys::WaifuJob> {
        let said = c_text(&request.text, "the text")?;
        let style = c_optional(&request.style, "the style")?;
        let reference = request.reference.as_ref().map(audio_in);
        let raw = sys::WaifuStreamingTextToSpeechRequest {
            struct_size: struct_size::<sys::WaifuStreamingTextToSpeechRequest>(),
            text: said.as_ptr(),
            reference: optional(&reference),
            temperature: request.temperature,
            style: c_pointer(&style),
            seed: request.seed,
        };
        Ok((f.waifu_pipeline_run_streaming_text_to_speech_async)(
            pipeline,
            &raw,
            user_data,
            Some(on_progress::<Audio, Audio>),
            Some(on_part),
            Some(on_audio_complete::<Audio>),
        ))
    }
}

impl sealed::Sealed for SpeechToText {
    unsafe fn start(
        f: &sys::Functions,
        pipeline: *mut sys::WaifuPipeline,
        request: &SpeechToTextRequest,
        user_data: *mut c_void,
    ) -> Result<sys::WaifuJob> {
        let audio = audio_in(&request.audio);
        let language = c_optional(&request.language, "the language")?;
        let context = c_text(&request.context, "the context")?;
        let raw = sys::WaifuSpeechToTextRequest {
            struct_size: struct_size::<sys::WaifuSpeechToTextRequest>(),
            audio: &audio,
            language: c_pointer(&language),
            context: context.as_ptr(),
        };
        Ok((f.waifu_pipeline_run_speech_to_text_async)(
            pipeline,
            &raw,
            user_data,
            Some(on_progress::<Transcript, Infallible>),
            Some(on_transcript_complete),
        ))
    }
}

impl sealed::Sealed for VoiceConversion {
    unsafe fn start(
        f: &sys::Functions,
        pipeline: *mut sys::WaifuPipeline,
        request: &VoiceConversionRequest,
        user_data: *mut c_void,
    ) -> Result<sys::WaifuJob> {
        let (source, reference) = (audio_in(&request.source), audio_in(&request.reference));
        let raw = sys::WaifuVoiceConversionRequest {
            struct_size: struct_size::<sys::WaifuVoiceConversionRequest>(),
            source: &source,
            reference: &reference,
            steps: request.steps,
            convert_style: request.convert_style,
            seed: request.seed,
        };
        Ok((f.waifu_pipeline_run_voice_conversion_async)(
            pipeline,
            &raw,
            user_data,
            Some(on_progress::<Audio, Infallible>),
            Some(on_audio_complete::<Infallible>),
        ))
    }
}

impl sealed::Sealed for LayerDecomposition {
    unsafe fn start(
        f: &sys::Functions,
        pipeline: *mut sys::WaifuPipeline,
        request: &LayerDecompositionRequest,
        user_data: *mut c_void,
    ) -> Result<sys::WaifuJob> {
        let image = sys::WaifuRgbaImage {
            width: request.image.width,
            height: request.image.height,
            rgba: request.image.rgba.as_ptr(),
            len: request.image.rgba.len(),
        };
        let raw = sys::WaifuLayerDecompositionRequest {
            struct_size: struct_size::<sys::WaifuLayerDecompositionRequest>(),
            image: &image,
            steps: request.steps,
            depth_steps: request.depth_steps,
            resolution: request.resolution,
            split_pairs: request.split_pairs,
            stratify_layers: request.stratify_layers,
            seed: request.seed,
        };
        Ok((f.waifu_pipeline_run_layer_decomposition_async)(
            pipeline,
            &raw,
            user_data,
            Some(on_progress::<DecomposedImage, Infallible>),
            Some(on_layers_complete),
        ))
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

    /// Asks the library with a fresh job state, keeping it as a `Job` where the call was taken.
    fn job<O, P>(
        &self,
        start: impl FnOnce(*mut c_void) -> Result<sys::WaifuJob>,
    ) -> Result<Job<O, P>> {
        let shared = Shared::<O, P>::new();
        let data = user_data(&shared);
        let id = match start(data) {
            Ok(id) => id,
            Err(error) => {
                drop(unsafe { taken_back::<O, P>(data) });
                return Err(error);
            }
        };
        if id == 0 {
            // Refused: no callback is ever called, and the reference is ours again.
            drop(unsafe { taken_back::<O, P>(data) });
            return Err(Error::refused(self.inner.functions));
        }
        Ok(Job {
            pipeline: Arc::clone(&self.inner),
            id,
            shared,
            ended: false,
        })
    }

    /// Reads the model's weights onto the device, dropping the model before first. Refused at the
    /// call for a model the task is not supported by. `model` may be dropped as soon as this
    /// returns.
    pub fn load_model(&self, model: &Model) -> Result<Job<()>> {
        let (f, handle) = (self.inner.functions, self.inner.handle);
        self.job(|data| {
            Ok(unsafe {
                (f.waifu_pipeline_load_model_async)(
                    handle,
                    model.raw(),
                    data,
                    Some(on_progress::<(), Infallible>),
                    Some(on_unit_complete),
                )
            })
        })
    }

    pub fn unload_model(&self) -> Result<Job<()>> {
        let (f, handle) = (self.inner.functions, self.inner.handle);
        self.job(|data| {
            Ok(unsafe {
                (f.waifu_pipeline_unload_model_async)(handle, data, Some(on_unit_complete))
            })
        })
    }

    /// Runs the task. Refused at the call for a request that cannot be run; the request is copied,
    /// and may be dropped as soon as this returns.
    pub fn run(&self, request: &T::Request) -> Result<Job<T::Output, T::Part>> {
        let (f, handle) = (self.inner.functions, self.inner.handle);
        self.job(|data| unsafe { T::start(f, handle, request, data) })
    }

    /// Stops what runs after its step and cancels what waits.
    pub fn cancel_all(&self) {
        unsafe { (self.inner.functions.waifu_pipeline_cancel_all)(self.inner.handle) };
    }
}
