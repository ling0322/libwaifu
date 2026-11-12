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

//! One thread that holds a model and runs jobs on it, for a program that is not the command line.
//!
//! A tensor never leaves the thread that made it, so a model cannot be handed to whoever asks: it
//! is made, used and dropped on the engine's thread, and every call from outside is a message to
//! that thread. The calls return at once; what comes of a job arrives through its callbacks --
//! progress any number of times, then exactly one completion -- on the engine's thread.
//!
//! This is what the C API in [`crate::c_api`] is a thin skin over; see docs/ffi.md for the contract it
//! keeps. One model at a time: two on one card double the memory, and on Metal nothing can say
//! beforehand whether they would fit.

use std::collections::HashSet;
use std::ops::ControlFlow;
use std::panic::{self, AssertUnwindSafe};
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::Instant;

use crate::cosyvoice3::CosyVoice3;
use crate::describe::{
    self, converter_kind, transcriber_kind, voice_kind, Chosen, ConverterKind, TranscriberKind,
    VoiceKind, TONES,
};
use crate::image_model::ImageModel;
use crate::indextts::IndexTts;
use crate::progress::{Convert, Doing, Run, Say, Transcribing, VOCODER_RATIO};
use crate::qwen3_asr::Qwen3Asr;
use crate::runtime::{DeviceOption, Runtime};
use crate::see_through::{
    Image as SeeThroughImage, SeeThrough, SeeThroughOptions, SeeThroughProgress,
};
use crate::wav::Sound;
use crate::{
    from_rgb8, to_rgb8, ConversionOptions, ConversionProgress, Converter, Device,
    GenerationOptions, GenerationProgress, Manifest, SpeechOptions, SpeechProgress, Tones,
    Transcriber, Transcript, TranscriptionOptions, TranscriptionProgress, Voice,
};

/// A job, as the engine numbers them. Never 0, which the C API keeps for a refusal.
pub type JobId = u64;

/// The keys an image is given under that are not names a prompt can write, but say what the image
/// is for.
pub const START_FROM: &str = "start_from";
pub const CONTROL: &str = "control";
const KEPT_KEYS: &[&str] = &[START_FROM, CONTROL];

/// How an engine is started: where it runs. How much of the device it may use is the process's,
/// not an engine's: see [`crate::runtime::set_memory_limit`].
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct EngineOptions {
    pub device: DeviceOption,
}

/// A picture as rows of RGB, three bytes a pixel, top to bottom.
#[derive(Clone, Debug, PartialEq)]
pub struct Image {
    pub width: u32,
    pub height: u32,
    pub rgb: Vec<u8>,
}

/// What a draw is asked for. See `WaifuDrawRequest` in docs/ffi.md, which is this.
#[derive(Clone, Debug)]
pub struct DrawRequest {
    /// Words, and `<|key|>` where an image in `images` is to be read.
    pub prompt: String,
    pub negative: String,
    pub images: Vec<(String, Image)>,
    pub width: u32,
    pub height: u32,
    pub steps: u32,
    pub guidance: f32,
    pub seed: u64,
    /// With an image under "start_from": how far the run walks away from it.
    pub strength: f32,
    /// With an image under "control": how hard it holds to it.
    pub control_scale: f32,
}

#[derive(Clone, Debug)]
pub struct SpeakRequest {
    pub text: String,
    pub like: Option<Sound>,
    pub speed: f32,
    pub temperature: f32,
    pub style: Option<String>,
    /// What `like` says, word for word; `None` where it is not known. A voice that reads none
    /// ignores it.
    pub reference_transcript: Option<String>,
    pub seed: u64,
}

#[derive(Clone, Debug)]
pub struct VoiceConversionRequest {
    pub source: Sound,
    pub reference: Sound,
    pub steps: u32,
    pub convert_style: bool,
    pub seed: u64,
}

/// A picture as rows of RGBA, four bytes a pixel, not premultiplied.
#[derive(Clone, Debug, PartialEq)]
pub struct RgbaImage {
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
}

/// What writing down a recording is asked for. See `WaifuTranscribeRequest` in docs/ffi.md.
#[derive(Clone, Debug)]
pub struct TranscribeRequest {
    pub recording: Sound,
    /// The language it is in, by a name the model's description lists; `None` for the model to
    /// tell.
    pub language: Option<String>,
    /// Words that help it hear the recording -- names, terms, what it is about; empty for none.
    pub context: String,
}

/// What taking a picture apart is asked for. See `WaifuLayersRequest` in docs/ffi.md.
#[derive(Clone, Debug)]
pub struct LayersRequest {
    pub image: RgbaImage,
    /// 0 for the model's own.
    pub steps: u32,
    /// 0 for the model's own.
    pub depth_steps: u32,
    /// The side of the square the layers are drawn at, a multiple of 64; 0 for the model's own.
    pub resolution: u32,
    /// Split what comes in pairs -- hands, eyes, ears -- into a left and a right layer.
    pub split_pairs: bool,
    /// Cut hair, hands, top and bottom into a near and a far half by their own depth, and append
    /// the halves: see [`Layers`].
    pub stratify: bool,
    pub seed: u64,
}

/// What a layer of a picture taken apart is.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LayerKind {
    /// A part as the model drew it.
    Part,
    /// The near half of a stratified part.
    StratifiedNear,
    /// The far half of a stratified part.
    StratifiedFar,
    /// The depth of the layer `derived_from_id` names, pixel by pixel.
    Depth,
}

/// One layer of a picture taken apart.
#[derive(Clone, Debug, PartialEq)]
pub struct LayerOut {
    pub kind: LayerKind,
    /// Unique within the result, from 1; it means nothing else.
    pub id: u32,
    /// A depth layer's is its layer's with `_depth` after it: `topwear_depth`.
    pub name: String,
    /// Where its top left corner is on the canvas.
    pub left: i32,
    pub top: i32,
    /// A depth layer's: R, G and B the depth, 0 near to 255 far, under its layer's alpha.
    pub image: RgbaImage,
    /// How far away it is, 0 near to 1 far. A near half has its part's depth, not its own pixels':
    /// that is what keeps it in its part's place in the stratified stack. A depth layer has its
    /// layer's.
    pub depth: f32,
    /// A half: the id of the part it was cut from; a depth layer: of the layer it is the depth
    /// of; a part: None.
    pub derived_from_id: Option<u32>,
}

/// A picture taken apart. `layers` is in one order, never changed: every part, bottom first --
/// the stack, unstratified -- then, where stratification was asked for, each split part's halves,
/// near before far; then a depth layer for each of those, in their order. See
/// [`Layers::stratified_stack`] for the other stack.
#[derive(Clone, Debug, PartialEq)]
pub struct Layers {
    /// The side of the square canvas the layers are on.
    pub canvas: u32,
    pub layers: Vec<LayerOut>,
}

impl Layers {
    /// The stratified stack, bottom first, as indices into `layers`: every part that was not split
    /// and every half -- no depth layer -- ordered by depth from the largest; at one depth, a far
    /// half under a near one; then by index. docs/ffi-abi5.md, "Stratification".
    pub fn stratified_stack(&self) -> Vec<usize> {
        let split: HashSet<u32> = self
            .layers
            .iter()
            .filter(|layer| {
                matches!(
                    layer.kind,
                    LayerKind::StratifiedNear | LayerKind::StratifiedFar
                )
            })
            .filter_map(|layer| layer.derived_from_id)
            .collect();
        let mut stack: Vec<usize> = (0..self.layers.len())
            .filter(|&index| {
                let layer = &self.layers[index];
                layer.kind != LayerKind::Depth && !split.contains(&layer.id)
            })
            .collect();
        let lower = |kind: LayerKind| match kind {
            LayerKind::StratifiedFar => 0,
            _ => 1,
        };
        stack.sort_by(|&a, &b| {
            let (a_layer, b_layer) = (&self.layers[a], &self.layers[b]);
            b_layer
                .depth
                .total_cmp(&a_layer.depth)
                .then(lower(a_layer.kind).cmp(&lower(b_layer.kind)))
                .then(a.cmp(&b))
        });
        stack
    }
}

/// Which part of a job a progress report is about: one for each thing a job does, by task, as the
/// C API's `WaifuStage` numbers them (docs/ffi-abi5.md). `done` and `total` count what each says,
/// and are 0 and 0 for the rest.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Stage {
    LoadingWeights,

    ImageGenerationEncodingPrompt,
    /// Steps.
    ImageGenerationDenoising,
    ImageGenerationDecodingImage,

    TextToSpeechReadingText,
    /// Tokens; the total is an estimate, and can be passed.
    TextToSpeechGeneratingTokens,
    TextToSpeechSynthesizingAudio,

    VoiceConversionReadingRecordings,
    /// Tokens; the total is an estimate.
    VoiceConversionGeneratingTokens,
    /// Steps.
    VoiceConversionDenoising,
    /// The vocoder's windows.
    VoiceConversionSynthesizingAudio,

    LayerDecompositionEncodingImage,
    /// Steps.
    LayerDecompositionDenoisingLayers,
    /// Layers.
    LayerDecompositionDecodingLayers,
    LayerDecompositionEncodingHead,
    /// Steps.
    LayerDecompositionDenoisingHeadLayers,
    /// Layers.
    LayerDecompositionDecodingHeadLayers,
    /// Layers.
    LayerDecompositionEncodingForDepth,
    /// Steps.
    LayerDecompositionDenoisingDepth,
    /// Layers.
    LayerDecompositionDecodingDepth,
    LayerDecompositionCuttingIntoParts,

    /// Pieces the recording is cut into.
    SpeechToTextReadingRecording,
    /// Tokens; the total is an estimate, and can be passed.
    SpeechToTextGeneratingTokens,

    StreamingTextToSpeechReadingText,
    /// Tokens; the total is an estimate.
    StreamingTextToSpeechGeneratingTokens,
    /// Parts handed over.
    StreamingTextToSpeechSynthesizingAudio,
}

/// How far along a job is: what `on_progress` is told.
#[derive(Clone, Debug, PartialEq)]
pub struct Progress {
    pub stage: Stage,
    /// Of the whole job, 0 to 1, or None where nothing can say -- reading weights.
    pub fraction: Option<f64>,
    /// What the stage counts: steps or tokens. 0 and 0 where it counts nothing.
    pub done: u64,
    pub total: u64,
    /// "step 3 of 8", for a status line.
    pub words: String,
    /// Since the job started.
    pub seconds: f64,
}

/// Why a job failed, in the C API's terms.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Failure {
    /// Neither the stand-in nor a manifest that exists.
    UnknownModel,
    Model,
    NotLoaded,
    Internal,
}

/// How a job ended: what its completion is handed.
#[derive(Debug, PartialEq)]
pub enum Ended<T> {
    Done(T),
    Cancelled,
    Failed(Failure, String),
}

impl<T> Ended<T> {
    fn failed(failure: Failure, message: impl Into<String>) -> Ended<T> {
        Ended::Failed(failure, message.into())
    }
}

pub type OnProgress = Box<dyn FnMut(&Progress) + Send>;
pub type OnComplete<T> = Box<dyn FnOnce(Ended<T>) + Send>;
/// A part of a reading's audio as soon as it is made: see [`Engine::speak_streaming`].
pub type OnAudio = Box<dyn FnMut(&Sound) + Send>;

/// The engine: a thread, a queue to it, and how to stop what is on it.
///
/// Dropping it stops what is running after the step it is on, cancels everything waiting --
/// every one of them still gets its completion, as cancelled -- drops the model and joins the
/// thread.
pub struct Engine {
    /// What it runs on, `auto` resolved.
    runtime: Runtime,
    commands: Sender<Command>,
    shared: Arc<Shared>,
    thread: Option<JoinHandle<()>>,
}

/// What the callers and the engine's thread both see.
struct Shared {
    next: AtomicU64,
    /// The job on the thread now, if any.
    running: Mutex<Option<JobId>>,
    /// Asked of the running job between its steps.
    stop: AtomicBool,
    /// Jobs asked to stop, waiting or running; the thread skips a waiting one when it comes to it.
    cancelled: Mutex<HashSet<JobId>>,
    /// Jobs not yet finished, for cancelling all of them.
    open: Mutex<HashSet<JobId>>,
}

enum Command {
    Job(JobId, Work),
    Quit,
}

enum Work {
    Load {
        model: String,
        /// For readings handed over a part at a time: a voice that cannot stream on this device
        /// is let go of again, and the load fails.
        streaming: bool,
        on_progress: OnProgress,
        on_complete: OnComplete<()>,
    },
    Unload {
        on_complete: OnComplete<()>,
    },
    Draw {
        request: DrawRequest,
        on_progress: OnProgress,
        on_complete: OnComplete<Image>,
    },
    Speak {
        request: SpeakRequest,
        on_progress: OnProgress,
        /// Where each part goes as it is made, for a streaming reading.
        on_audio: Option<OnAudio>,
        on_complete: OnComplete<Sound>,
    },
    VoiceConversion {
        request: VoiceConversionRequest,
        on_progress: OnProgress,
        on_complete: OnComplete<Sound>,
    },
    Layers {
        request: LayersRequest,
        on_progress: OnProgress,
        on_complete: OnComplete<Layers>,
    },
    Transcribe {
        request: TranscribeRequest,
        on_progress: OnProgress,
        on_complete: OnComplete<Transcript>,
    },
}

impl Work {
    /// Ends this job without running it.
    fn cancel(self) {
        match self {
            Work::Load { on_complete, .. } | Work::Unload { on_complete } => {
                on_complete(Ended::Cancelled)
            }
            Work::Draw { on_complete, .. } => on_complete(Ended::Cancelled),
            Work::Speak { on_complete, .. } | Work::VoiceConversion { on_complete, .. } => {
                on_complete(Ended::Cancelled)
            }
            Work::Layers { on_complete, .. } => on_complete(Ended::Cancelled),
            Work::Transcribe { on_complete, .. } => on_complete(Ended::Cancelled),
        }
    }
}

impl Engine {
    /// Starts an engine the way `options` says. Refused where the device is not one this build
    /// can use on this machine.
    pub fn new(options: EngineOptions) -> Result<Engine, String> {
        let runtime = options.device.resolve();
        if !runtime.device().is_available() {
            return Err(format!(
                "{} is not available on this machine",
                runtime.name()
            ));
        }

        let (commands, queue) = mpsc::channel();
        let shared = Arc::new(Shared {
            next: AtomicU64::new(1),
            running: Mutex::new(None),
            stop: AtomicBool::new(false),
            cancelled: Mutex::new(HashSet::new()),
            open: Mutex::new(HashSet::new()),
        });
        let thread = thread::Builder::new()
            .name("waifu-engine".to_string())
            .spawn({
                let shared = Arc::clone(&shared);
                move || work(runtime, &shared, queue)
            })
            .map_err(|error| format!("could not start the engine's thread: {error}"))?;

        Ok(Engine {
            runtime,
            commands,
            shared,
            thread: Some(thread),
        })
    }

    /// What it runs on: never `auto`, which is settled when it starts.
    pub fn runtime(&self) -> Runtime {
        self.runtime
    }

    fn submit(&self, work: Work) -> JobId {
        let job = self.shared.next.fetch_add(1, Ordering::Relaxed);
        lock(&self.shared.open).insert(job);
        // The thread is there until this is dropped, so the send only fails if it died -- and a
        // job it never sees still has to be ended.
        if let Err(mpsc::SendError(Command::Job(_, work))) =
            self.commands.send(Command::Job(job, work))
        {
            lock(&self.shared.open).remove(&job);
            work.cancel();
        }
        job
    }

    /// Reads the model whose manifest is at `model` onto the device, dropping the one before.
    /// `model` is a path -- fetching a published model and saying where it went is the model
    /// manager's -- or the stand-in voice, "tones".
    pub fn load(&self, model: &str, on_progress: OnProgress, on_complete: OnComplete<()>) -> JobId {
        self.submit(Work::Load {
            model: model.to_string(),
            streaming: false,
            on_progress,
            on_complete,
        })
    }

    /// As [`load`](Engine::load), for a voice that is to stream: one that cannot on this device --
    /// CosyVoice3 on the processor, or on CUDA other than at half on Ampere or later -- is let go
    /// of again, and the load
    /// fails with [`Failure::Model`] saying why.
    pub fn load_for_streaming(
        &self,
        model: &str,
        on_progress: OnProgress,
        on_complete: OnComplete<()>,
    ) -> JobId {
        self.submit(Work::Load {
            model: model.to_string(),
            streaming: true,
            on_progress,
            on_complete,
        })
    }

    pub fn unload(&self, on_complete: OnComplete<()>) -> JobId {
        self.submit(Work::Unload { on_complete })
    }

    /// Refused at the call, with the reason, for a request no model could run: see
    /// [`check_draw`].
    pub fn draw(
        &self,
        request: DrawRequest,
        on_progress: OnProgress,
        on_complete: OnComplete<Image>,
    ) -> Result<JobId, String> {
        check_draw(&request)?;
        Ok(self.submit(Work::Draw {
            request,
            on_progress,
            on_complete,
        }))
    }

    pub fn speak(
        &self,
        request: SpeakRequest,
        on_progress: OnProgress,
        on_complete: OnComplete<Sound>,
    ) -> Result<JobId, String> {
        if request.text.trim().is_empty() {
            return Err("there is nothing to say: the text is empty".to_string());
        }
        Ok(self.submit(Work::Speak {
            request,
            on_progress,
            on_audio: None,
            on_complete,
        }))
    }

    /// As [`speak`](Engine::speak), with each part of the audio handed to `on_audio` as soon as
    /// it is made, while the rest is still being said -- see [`Voice::speak_streaming`] -- and the
    /// whole of it, the parts end to end, to `on_complete`. A voice that cannot stream is refused
    /// there. Cancelling stops it after the part it is on; the parts handed over are all there
    /// will be, and `on_complete` is told it was cancelled.
    pub fn speak_streaming(
        &self,
        request: SpeakRequest,
        on_progress: OnProgress,
        on_audio: OnAudio,
        on_complete: OnComplete<Sound>,
    ) -> Result<JobId, String> {
        if request.text.trim().is_empty() {
            return Err("there is nothing to say: the text is empty".to_string());
        }
        Ok(self.submit(Work::Speak {
            request,
            on_progress,
            on_audio: Some(on_audio),
            on_complete,
        }))
    }

    pub fn voice_conversion(
        &self,
        request: VoiceConversionRequest,
        on_progress: OnProgress,
        on_complete: OnComplete<Sound>,
    ) -> Result<JobId, String> {
        if request.source.samples.is_empty() || request.reference.samples.is_empty() {
            return Err("both recordings are needed, and one of them is empty".to_string());
        }
        Ok(self.submit(Work::VoiceConversion {
            request,
            on_progress,
            on_complete,
        }))
    }

    /// Takes a picture apart into the layers it was drawn in, with a model that does that loaded.
    pub fn layers(
        &self,
        request: LayersRequest,
        on_progress: OnProgress,
        on_complete: OnComplete<Layers>,
    ) -> Result<JobId, String> {
        let image = &request.image;
        if image.width == 0 || image.height == 0 {
            return Err("there is nothing to take apart: the picture is empty".to_string());
        }
        let expected = image.width as u64 * image.height as u64 * 4;
        if image.rgba.len() as u64 != expected {
            return Err(format!(
                "a {} by {} picture is {expected} bytes of RGBA, and this one is {}",
                image.width,
                image.height,
                image.rgba.len()
            ));
        }
        Ok(self.submit(Work::Layers {
            request,
            on_progress,
            on_complete,
        }))
    }

    /// Writes down what is said in a recording, with a model that does that loaded.
    pub fn transcribe(
        &self,
        request: TranscribeRequest,
        on_progress: OnProgress,
        on_complete: OnComplete<Transcript>,
    ) -> Result<JobId, String> {
        if request.recording.samples.is_empty() {
            return Err("there is nothing to write down: the recording is empty".to_string());
        }
        Ok(self.submit(Work::Transcribe {
            request,
            on_progress,
            on_complete,
        }))
    }

    /// Stops `job`: after the step it is on where it is running, before it starts where it waits.
    pub fn cancel(&self, job: JobId) {
        self.canceller().cancel(job)
    }

    /// What stops this engine's jobs, without holding the engine: a job's handle in the C API
    /// outlives its pipeline, and must not keep the thread alive.
    pub fn canceller(&self) -> Canceller {
        Canceller(Arc::clone(&self.shared))
    }

    pub fn cancel_all(&self) {
        let open: Vec<JobId> = lock(&self.shared.open).iter().copied().collect();
        lock(&self.shared.cancelled).extend(open);
        if lock(&self.shared.running).is_some() {
            self.shared.stop.store(true, Ordering::Relaxed);
        }
    }
}

/// Stops one of an engine's jobs, as [`Engine::cancel`] does. Once the engine is gone it has no job
/// left to stop, and does nothing.
#[derive(Clone)]
pub struct Canceller(Arc<Shared>);

impl Canceller {
    pub fn cancel(&self, job: JobId) {
        let shared = &self.0;
        if !lock(&shared.open).contains(&job) {
            return;
        }
        lock(&shared.cancelled).insert(job);
        if *lock(&shared.running) == Some(job) {
            shared.stop.store(true, Ordering::Relaxed);
        }
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        self.cancel_all();
        let _ = self.commands.send(Command::Quit);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

/// A poisoned lock is taken as it is: what these hold is bookkeeping a panic on the other side
/// cannot have left half-written in a way that matters more than refusing every call after it.
fn lock<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|held| held.into_inner())
}

/// Refuses a draw that cannot be run whatever the model: an image no placeholder names, a
/// placeholder no image answers, a key that is not a name, and the like. The model's own
/// refusals -- one that reads words only, one with no ControlNet -- come at the run.
pub fn check_draw(request: &DrawRequest) -> Result<(), String> {
    if request.prompt.trim().is_empty() {
        return Err("there is nothing to draw: the prompt is empty".to_string());
    }
    // The range only. What a side has to be a multiple of is the model's -- sixteen for the
    // transformers, thirty-two for SDXL -- and its run refuses a size it cannot draw at, saying
    // which.
    for (side, name) in [(request.width, "width"), (request.height, "height")] {
        if !(64..=2048).contains(&side) {
            return Err(format!(
                "the {name} is {side}: it has to be from 64 to 2048"
            ));
        }
    }
    if request.steps == 0 {
        return Err("a run needs at least one step".to_string());
    }
    if request.negative.contains("<|") {
        return Err("the negative prompt is words only: it cannot name an image".to_string());
    }

    let mut keys = HashSet::new();
    for (key, image) in &request.images {
        if key.is_empty() || !key.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
            return Err(format!(
                "\"{key}\" is not a key: letters, digits and _ only"
            ));
        }
        if !keys.insert(key.as_str()) {
            return Err(format!("two images are given under \"{key}\""));
        }
        if image.width == 0
            || image.height == 0
            || image.rgb.len() != (image.width * image.height * 3) as usize
        {
            return Err(format!(
                "the image under \"{key}\" is {}x{} and {} bytes, which is not that many rows of RGB",
                image.width,
                image.height,
                image.rgb.len()
            ));
        }
    }

    let named = placeholders(&request.prompt)?;
    for name in &named {
        if KEPT_KEYS.contains(&name.as_str()) {
            return Err(format!("<|{name}|> names an image that is drawn with, not read: \"{name}\" is not for the prompt"));
        }
        if !keys.contains(name.as_str()) {
            return Err(format!("<|{name}|> names no image"));
        }
    }
    for key in &keys {
        if !KEPT_KEYS.contains(key) && !named.iter().any(|name| name == key) {
            return Err(format!("the image under \"{key}\" is not named in the prompt: write <|{key}|> where it is to be read"));
        }
    }
    Ok(())
}

/// Every `<|name|>` in a prompt, in order. Anything else between `<|` and `|>` is refused: the
/// model would read it as one of its own markers.
fn placeholders(prompt: &str) -> Result<Vec<String>, String> {
    let mut found = Vec::new();
    let mut rest = prompt;
    while let Some(at) = rest.find("<|") {
        let after = &rest[at + 2..];
        let Some(end) = after.find("|>") else {
            return Err("the prompt opens a <| it does not close".to_string());
        };
        let name = &after[..end];
        if name.is_empty() || !name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
            return Err(format!("<|{name}|> is not a name of an image, and the model would read it as one of its own markers"));
        }
        found.push(name.to_string());
        rest = &after[end + 2..];
    }
    Ok(found)
}

/// The prompt with each `<|name|>` written the way Qwen-Image 2.1 names its pictures --
/// `<image1>`, `<image2>`, in the order the names first appear -- and that order, which is the
/// order the pictures go in front of the prompt. A name written twice is one picture.
///
/// Called on a prompt [`check_draw`] passed, so every placeholder is a well-formed name.
fn numbered(prompt: &str) -> (String, Vec<String>) {
    let mut order: Vec<String> = Vec::new();
    let mut written = String::with_capacity(prompt.len());
    let mut rest = prompt;
    while let Some(at) = rest.find("<|") {
        let after = &rest[at + 2..];
        let Some(end) = after.find("|>") else {
            break;
        };
        let name = &after[..end];
        let index = match order.iter().position(|seen| seen == name) {
            Some(index) => index,
            None => {
                order.push(name.to_string());
                order.len() - 1
            }
        };
        written.push_str(&rest[..at]);
        written.push_str(&format!("<image{}>", index + 1));
        rest = &after[end + 2..];
    }
    written.push_str(rest);
    (written, order)
}

/// Whether the manifest at `path` is a See-through package, which is read before the models that
/// draw are asked: it has no prompt, and none of them could read it.
fn is_see_through(path: &Path) -> bool {
    Manifest::open(path)
        .and_then(|manifest| Ok(manifest.section("model")?.get_str("type")?.to_string()))
        .is_ok_and(|kind| kind == SeeThrough::MODEL_TYPE)
}

// -- the thread ------------------------------------------------------------------------------------

/// What the thread has read.
enum Loaded {
    Nothing,
    Image {
        model: ImageModel,
        chosen: Chosen,
    },
    Voice(Box<dyn Voice>),
    /// Both a voice and a converter: one model, used as either.
    CosyVoice(CosyVoice3),
    /// What takes a picture apart into layers.
    Layers(Box<SeeThrough>),
    /// What writes down what a recording says.
    Transcriber(Box<dyn Transcriber>),
}

/// What the thread keeps between jobs.
struct Worker {
    runtime: Runtime,
    loaded: Loaded,
    /// The name it was asked for, for the parameters of what it makes.
    name: String,
    /// How long the last reading's vocoder took against everything before it.
    vocoder_ratio: f64,
}

fn work(runtime: Runtime, shared: &Shared, queue: Receiver<Command>) {
    let mut worker = Worker {
        runtime,
        loaded: Loaded::Nothing,
        name: String::new(),
        vocoder_ratio: VOCODER_RATIO,
    };

    while let Ok(Command::Job(job, work)) = queue.recv() {
        let cancelled = lock(&shared.cancelled).remove(&job);
        if cancelled {
            lock(&shared.open).remove(&job);
            work.cancel();
            continue;
        }

        shared.stop.store(false, Ordering::Relaxed);
        *lock(&shared.running) = Some(job);
        // A cancel can land between the check above and running being set; it is honoured here.
        if lock(&shared.cancelled).contains(&job) {
            shared.stop.store(true, Ordering::Relaxed);
        }

        worker.run(shared, work);

        *lock(&shared.running) = None;
        lock(&shared.cancelled).remove(&job);
        lock(&shared.open).remove(&job);
    }
    // Quit, or every sender gone: the model goes with the thread.
}

/// Runs `body`, turning a panic into a failure, so that a job always ends.
fn guarded<T>(body: impl FnOnce() -> Ended<T>) -> Ended<T> {
    match panic::catch_unwind(AssertUnwindSafe(body)) {
        Ok(ended) => ended,
        Err(cause) => {
            let said = cause
                .downcast_ref::<&str>()
                .map(|said| said.to_string())
                .or_else(|| cause.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "it panicked".to_string());
            Ended::failed(Failure::Internal, format!("a bug in the library: {said}"))
        }
    }
}

/// Builds the progress a report is turned into, and hands it on.
struct Reporter<'a> {
    on_progress: &'a mut OnProgress,
    started: Instant,
}

impl Reporter<'_> {
    fn tell(&mut self, doing: &Doing) {
        let count = |n: i32| n.max(0) as u64;
        let (stage, done, total) = match doing {
            Doing::Nothing => return,
            Doing::Reading { .. } => (Stage::LoadingWeights, 0, 0),
            Doing::Drawing(run) => match run.progress {
                GenerationProgress::Encoding => (Stage::ImageGenerationEncodingPrompt, 0, 0),
                GenerationProgress::Step { done, total } => {
                    (Stage::ImageGenerationDenoising, count(done), count(total))
                }
                GenerationProgress::Decoding => (Stage::ImageGenerationDecodingImage, 0, 0),
            },
            // A streaming reading is a task of its own, with stages of its own.
            Doing::Speaking(say) => match (say.streaming, say.progress) {
                (false, SpeechProgress::Reading) => (Stage::TextToSpeechReadingText, 0, 0),
                (false, SpeechProgress::Saying { done, expected }) => (
                    Stage::TextToSpeechGeneratingTokens,
                    count(done),
                    count(expected),
                ),
                (false, SpeechProgress::Sounding) => (Stage::TextToSpeechSynthesizingAudio, 0, 0),
                (true, SpeechProgress::Reading) => (Stage::StreamingTextToSpeechReadingText, 0, 0),
                (true, SpeechProgress::Saying { done, expected }) => (
                    Stage::StreamingTextToSpeechGeneratingTokens,
                    count(done),
                    count(expected),
                ),
                (true, SpeechProgress::Sounding) => (
                    Stage::StreamingTextToSpeechSynthesizingAudio,
                    say.parts as u64,
                    0,
                ),
            },
            Doing::Converting(convert) => match convert.progress {
                ConversionProgress::Listening => (Stage::VoiceConversionReadingRecordings, 0, 0),
                ConversionProgress::Saying { done, expected } => (
                    Stage::VoiceConversionGeneratingTokens,
                    count(done),
                    count(expected),
                ),
                ConversionProgress::Drawing { done, total } => {
                    (Stage::VoiceConversionDenoising, count(done), count(total))
                }
                ConversionProgress::Sounding { done, total } => (
                    Stage::VoiceConversionSynthesizingAudio,
                    count(done),
                    count(total),
                ),
            },
            // Pass 0 is the whole picture and pass 1 the head.
            Doing::Layering { progress, .. } => match *progress {
                SeeThroughProgress::Encoding { pass: 0 } => {
                    (Stage::LayerDecompositionEncodingImage, 0, 0)
                }
                SeeThroughProgress::Encoding { .. } => {
                    (Stage::LayerDecompositionEncodingHead, 0, 0)
                }
                SeeThroughProgress::Layers {
                    pass: 0,
                    done,
                    total,
                } => (
                    Stage::LayerDecompositionDenoisingLayers,
                    count(done),
                    count(total),
                ),
                SeeThroughProgress::Layers { done, total, .. } => (
                    Stage::LayerDecompositionDenoisingHeadLayers,
                    count(done),
                    count(total),
                ),
                SeeThroughProgress::Decoding {
                    pass: 0,
                    done,
                    total,
                } => (
                    Stage::LayerDecompositionDecodingLayers,
                    count(done),
                    count(total),
                ),
                SeeThroughProgress::Decoding { done, total, .. } => (
                    Stage::LayerDecompositionDecodingHeadLayers,
                    count(done),
                    count(total),
                ),
                SeeThroughProgress::DepthEncoding { done, total } => (
                    Stage::LayerDecompositionEncodingForDepth,
                    count(done),
                    count(total),
                ),
                SeeThroughProgress::Depth { done, total } => (
                    Stage::LayerDecompositionDenoisingDepth,
                    count(done),
                    count(total),
                ),
                SeeThroughProgress::DepthDecoding { done, total } => (
                    Stage::LayerDecompositionDecodingDepth,
                    count(done),
                    count(total),
                ),
                SeeThroughProgress::Assembling => (Stage::LayerDecompositionCuttingIntoParts, 0, 0),
            },
            Doing::Transcribing(transcribing) => match transcribing.progress {
                TranscriptionProgress::Listening { done, total } => (
                    Stage::SpeechToTextReadingRecording,
                    count(done),
                    count(total),
                ),
                TranscriptionProgress::Writing { done, expected } => (
                    Stage::SpeechToTextGeneratingTokens,
                    count(done),
                    count(expected),
                ),
            },
        };

        (self.on_progress)(&Progress {
            stage,
            fraction: doing.fraction(),
            done,
            total,
            words: doing.words(),
            seconds: self.started.elapsed().as_secs_f64(),
        });
    }
}

impl Worker {
    fn run(&mut self, shared: &Shared, work: Work) {
        let stop = || shared.stop.load(Ordering::Relaxed);
        match work {
            Work::Load {
                model,
                streaming,
                mut on_progress,
                on_complete,
            } => {
                let ended = guarded(|| self.load(&model, streaming, &mut on_progress));
                on_complete(ended);
            }
            Work::Unload { on_complete } => {
                self.let_go();
                on_complete(Ended::Done(()));
            }
            Work::Draw {
                request,
                mut on_progress,
                on_complete,
            } => {
                let ended = guarded(|| self.draw(&request, &mut on_progress, &stop));
                // What a draw needed at its peak -- the decoder's, on Metal some 3 GB past the
                // model -- is the device's again once it is over, done, stopped or failed, rather
                // than the process's until the next. Only the model stays.
                crate::flint::release_memory();
                on_complete(ended);
            }
            Work::Speak {
                request,
                mut on_progress,
                mut on_audio,
                on_complete,
            } => {
                let ended =
                    guarded(|| self.speak(&request, &mut on_progress, on_audio.as_mut(), &stop));
                on_complete(ended);
            }
            Work::VoiceConversion {
                request,
                mut on_progress,
                on_complete,
            } => {
                let ended = guarded(|| self.convert(&request, &mut on_progress, &stop));
                on_complete(ended);
            }
            Work::Layers {
                request,
                mut on_progress,
                on_complete,
            } => {
                let ended = guarded(|| self.layers(&request, &mut on_progress, &stop));
                crate::flint::release_memory();
                on_complete(ended);
            }
            Work::Transcribe {
                request,
                mut on_progress,
                on_complete,
            } => {
                let ended = guarded(|| self.transcribe(&request, &mut on_progress, &stop));
                crate::flint::release_memory();
                on_complete(ended);
            }
        }
    }

    fn load(&mut self, asked: &str, streaming: bool, on_progress: &mut OnProgress) -> Ended<()> {
        let mut reporter = Reporter {
            on_progress,
            started: Instant::now(),
        };

        // Let go of what is there first, so that two models are never on one card at once.
        self.let_go();

        if asked == TONES {
            self.loaded = Loaded::Voice(Box::new(Tones::new()));
            self.name = asked.to_string();
            return Ended::Done(());
        }
        let path = std::path::PathBuf::from(asked);
        if !path.is_file() {
            return Ended::failed(Failure::UnknownModel, describe::no_manifest_at(asked));
        }
        let shown = describe::shown_name(asked);

        info(&format!("reading {shown} onto {}", self.runtime.name()));
        reporter.tell(&Doing::Reading {
            model: shown.clone(),
        });
        let (device, device_weight_bytes) =
            (self.runtime.device(), self.runtime.device_weight_bytes());
        let read = || -> crate::Result<Loaded> {
            if let Some(kind) = voice_kind(&path) {
                let manifest = Manifest::open(&path)?;
                return Ok(match kind {
                    VoiceKind::CosyVoice3 => Loaded::CosyVoice(CosyVoice3::from_manifest(
                        device,
                        device_weight_bytes,
                        &manifest,
                    )?),
                    VoiceKind::IndexTts => Loaded::Voice(Box::new(IndexTts::from_manifest(
                        device,
                        device_weight_bytes,
                        &manifest,
                    )?)),
                });
            }
            if let Some(TranscriberKind::Qwen3Asr) = transcriber_kind(&path) {
                let manifest = Manifest::open(&path)?;
                return Ok(Loaded::Transcriber(Box::new(Qwen3Asr::from_manifest(
                    device,
                    device_weight_bytes,
                    &manifest,
                )?)));
            }
            if is_see_through(&path) {
                let manifest = Manifest::open(&path)?;
                return Ok(Loaded::Layers(Box::new(SeeThrough::from_manifest(
                    device,
                    device_weight_bytes,
                    &manifest,
                )?)));
            }
            if let Some(ConverterKind::CosyVoice3) = converter_kind(&path) {
                let manifest = Manifest::open(&path)?;
                return Ok(Loaded::CosyVoice(CosyVoice3::from_manifest(
                    device,
                    device_weight_bytes,
                    &manifest,
                )?));
            }
            let model = ImageModel::from_manifest(&path, device, device_weight_bytes)?;
            let mut chosen = describe::look_at(asked);
            chosen.no_picture_because = model.no_picture_because().map(str::to_string);
            chosen.on_disk = true;
            chosen.in_memory = true;
            Ok(Loaded::Image { model, chosen })
        };
        let read = || -> crate::Result<Loaded> {
            let loaded = read()?;
            if let (true, Loaded::CosyVoice(voice)) = (streaming, &loaded) {
                voice.streams()?;
            }
            Ok(loaded)
        };
        match read() {
            Ok(loaded) => {
                self.loaded = loaded;
                self.name = asked.to_string();
                info(&format!(
                    "{shown} read onto {} in {:.1}s; {}",
                    self.runtime.name(),
                    reporter.started.elapsed().as_secs_f64(),
                    crate::flint::memory_of(device)
                ));
                Ended::Done(())
            }
            Err(error) => {
                // What was read before it failed is dropped with the error; its memory goes too.
                crate::flint::release_memory();
                Ended::failed(Failure::Model, error.to_string())
            }
        }
    }

    /// Drops the model, and hands the memory it was in back to the system: on Metal a dropped
    /// tensor's buffer is otherwise kept for reuse, and the next model is read on top of it.
    fn let_go(&mut self) {
        if !self.name.is_empty() {
            info(&format!("let go of {}", describe::shown_name(&self.name)));
        }
        self.loaded = Loaded::Nothing;
        self.name.clear();
        crate::flint::release_memory();
    }

    fn draw(
        &mut self,
        request: &DrawRequest,
        on_progress: &mut OnProgress,
        stop: &dyn Fn() -> bool,
    ) -> Ended<Image> {
        let Loaded::Image { model, chosen } = &self.loaded else {
            return Ended::failed(Failure::NotLoaded, "no model that draws is loaded");
        };

        // What the model cannot be given, said before anything runs.
        let named = request
            .images
            .iter()
            .filter(|(key, _)| !KEPT_KEYS.contains(&key.as_str()))
            .collect::<Vec<_>>();
        if let Some((key, _)) = named.first() {
            let most = model.prompt_pictures();
            if most == 0 {
                return Ended::failed(
                    Failure::Model,
                    format!(
                        "{} reads words only: it cannot be shown <|{key}|>",
                        chosen.full_name
                    ),
                );
            }
            if named.len() > most {
                return Ended::failed(
                    Failure::Model,
                    format!(
                        "{} reads at most {most} pictures in a prompt, and this one names {}",
                        chosen.full_name,
                        named.len()
                    ),
                );
            }
        }
        if request.images.iter().any(|(key, _)| key == CONTROL) {
            return Ended::failed(
                Failure::Model,
                format!(
                    "{} has no ControlNet to hold to a control image",
                    chosen.full_name
                ),
            );
        }
        let start_from = request
            .images
            .iter()
            .find(|(key, _)| key == START_FROM)
            .map(|(_, image)| image);
        if start_from.is_some() {
            if let Some(why) = model.no_picture_because() {
                return Ended::failed(Failure::Model, why);
            }
        }

        // As asked: what to start a model's boxes at, guidance included, is the caller's to have
        // decided, and the request is what it decided.
        let options = GenerationOptions {
            width: request.width as i32,
            height: request.height as i32,
            num_steps: request.steps as i32,
            guidance_scale: request.guidance,
            negative_prompt: request.negative.clone(),
            seed: Some(request.seed),
            strength: request.strength.clamp(0.0, 1.0),
        };

        info(&format!(
            "drawing {}x{}{}, {} steps, guidance {:.1}, seed {}",
            request.width,
            request.height,
            match start_from {
                Some(_) => format!(" from a picture at strength {:.2}", options.strength),
                None => String::new(),
            },
            request.steps,
            request.guidance,
            request.seed
        ));
        let device = self.runtime.device();
        let _ = crate::flint::MemorySnapshot::reset_peak_stats(device);

        let started = Instant::now();
        let mut reporter = Reporter {
            on_progress,
            started,
        };
        let mut run = Run {
            progress: GenerationProgress::Encoding,
            steps: options.num_steps,
            started,
        };
        reporter.tell(&Doing::Drawing(run.clone()));
        let mut report = |progress| {
            run.progress = progress;
            reporter.tell(&Doing::Drawing(run.clone()));
            match stop() {
                true => ControlFlow::Break(()),
                false => ControlFlow::Continue(()),
            }
        };

        let drawn = match start_from {
            Some(image) => {
                let scaled = scaled(image, request.width, request.height);
                let tensor = match from_rgb8(request.width as i32, request.height as i32, &scaled) {
                    Ok(tensor) => tensor,
                    Err(error) => return Ended::failed(Failure::Model, error.to_string()),
                };
                model.generate_from_image_reporting(&tensor, &request.prompt, &options, &mut report)
            }
            None if !named.is_empty() => {
                // The model's own names for them, in the order the prompt first names them, and
                // each picture at the size the model reads one at.
                let (prompt, order) = numbered(&request.prompt);
                let mut pictures = Vec::with_capacity(order.len());
                for name in &order {
                    let Some((_, image)) = named.iter().find(|(key, _)| key == name) else {
                        return Ended::failed(Failure::Model, format!("<|{name}|> names no image"));
                    };
                    let (width, height) = model
                        .picture_size(image.width as i32, image.height as i32)
                        .expect("a model that reads pictures says what size it reads them at");
                    let scaled = scaled(image, width as u32, height as u32);
                    match from_rgb8(width, height, &scaled) {
                        Ok(tensor) => pictures.push(tensor),
                        Err(error) => return Ended::failed(Failure::Model, error.to_string()),
                    }
                }
                model.generate_with_pictures_reporting(&prompt, &pictures, &options, &mut report)
            }
            None => model.generate_reporting(&request.prompt, &options, &mut report),
        };
        let tensor = match drawn {
            Ok(Some(tensor)) => tensor,
            Ok(None) => {
                info("stopped before the picture was finished");
                return Ended::Cancelled;
            }
            Err(error) => return Ended::failed(Failure::Model, error.to_string()),
        };

        let seconds = started.elapsed().as_secs_f64();
        let peak = match crate::flint::MemorySnapshot::capture(device) {
            Ok(memory) if memory.total > 0 => {
                format!(", {} MB at most", memory.peak_allocated / 1_000_000)
            }
            _ => String::new(),
        };
        info(&format!(
            "drawn in {seconds:.1}s, {:.2}s a step{peak}; {}",
            seconds / request.steps.max(1) as f64,
            crate::flint::memory_of(device)
        ));

        let shape = tensor.shape();
        match to_rgb8(&tensor) {
            Ok(rgb) => Ended::Done(Image {
                width: shape[3] as u32,
                height: shape[2] as u32,
                rgb,
            }),
            Err(error) => Ended::failed(Failure::Model, error.to_string()),
        }
    }

    fn layers(
        &mut self,
        request: &LayersRequest,
        on_progress: &mut OnProgress,
        stop: &dyn Fn() -> bool,
    ) -> Ended<Layers> {
        let Loaded::Layers(model) = &self.loaded else {
            return Ended::failed(
                Failure::NotLoaded,
                "no model that takes a picture apart into layers is loaded",
            );
        };
        let config = model.config();
        let options = SeeThroughOptions {
            seed: request.seed,
            steps: (request.steps > 0).then_some(request.steps as i32),
            depth_steps: (request.depth_steps > 0).then_some(request.depth_steps as i32),
            split_pairs: request.split_pairs,
            resolution: (request.resolution > 0).then_some(request.resolution as usize),
            depth_resolution: None,
        };
        let (steps, depth_steps) = (
            options.steps.unwrap_or(config.steps),
            options.depth_steps.unwrap_or(config.depth_steps),
        );
        info(&format!(
            "taking a {}x{} picture apart, {steps} and {depth_steps} steps, seed {}",
            request.image.width, request.image.height, request.seed
        ));

        let started = Instant::now();
        let mut reporter = Reporter {
            on_progress,
            started,
        };
        let mut report = |progress| {
            reporter.tell(&Doing::Layering {
                progress,
                steps,
                depth_steps,
            });
            match stop() {
                true => ControlFlow::Break(()),
                false => ControlFlow::Continue(()),
            }
        };

        let picture = SeeThroughImage::from_data(
            request.image.width as usize,
            request.image.height as usize,
            4,
            request.image.rgba.clone(),
        );
        let decomposition = match model.decompose(&picture, &options, &mut report) {
            Ok(Some(decomposition)) => decomposition,
            Ok(None) => {
                info("stopped before the picture was taken apart");
                return Ended::Cancelled;
            }
            Err(error) => return Ended::failed(Failure::Model, error.to_string()),
        };
        info(&format!(
            "{} layers in {:.1}s",
            decomposition.parts.len(),
            started.elapsed().as_secs_f64()
        ));

        // Each layer with what its depth layer is drawn from: the parts first, then their halves,
        // near before far. The near half takes its part's depth, so that sorting by depth keeps it
        // in its part's place; the far half keeps its own.
        let mut drawn: Vec<(LayerKind, crate::see_through::Part, f32, Option<u32>)> = decomposition
            .parts
            .iter()
            .map(|part| (LayerKind::Part, part.clone(), part.depth_median, None))
            .collect();
        if request.stratify {
            for (index, part) in decomposition.parts.iter().enumerate() {
                if !crate::see_through::STRATIFIED.contains(&part.tag.as_str()) {
                    continue;
                }
                if let Some((near, far)) = crate::see_through::split(part) {
                    let id = Some(index as u32 + 1);
                    drawn.push((LayerKind::StratifiedNear, near, part.depth_median, id));
                    let far_depth = far.depth_median;
                    drawn.push((LayerKind::StratifiedFar, far, far_depth, id));
                }
            }
        }

        let place = |kind: LayerKind,
                     id: usize,
                     name: String,
                     part: &crate::see_through::Part,
                     rgba: Vec<u8>,
                     depth: f32,
                     derived_from_id: Option<u32>| LayerOut {
            kind,
            id: id as u32,
            name,
            left: part.left as i32,
            top: part.top as i32,
            image: RgbaImage {
                width: part.pixels.width as u32,
                height: part.pixels.height as u32,
                rgba,
            },
            depth,
            derived_from_id,
        };
        let mut layers: Vec<LayerOut> = drawn
            .iter()
            .enumerate()
            .map(|(index, (kind, part, depth, from))| {
                place(
                    *kind,
                    index + 1,
                    part.tag.clone(),
                    part,
                    part.pixels.data.clone(),
                    *depth,
                    *from,
                )
            })
            .collect();
        // Then each one's depth, in the same order: the depth in R, G and B, under its alpha.
        let count = drawn.len();
        for (index, (_, part, depth, _)) in drawn.iter().enumerate() {
            let alpha = part.pixels.data.chunks(4).map(|pixel| pixel[3]);
            let rgba: Vec<u8> = part
                .depth
                .data
                .iter()
                .zip(alpha)
                .flat_map(|(&d, a)| [d, d, d, a])
                .collect();
            layers.push(place(
                LayerKind::Depth,
                count + index + 1,
                format!("{}_depth", part.tag),
                part,
                rgba,
                *depth,
                Some(index as u32 + 1),
            ));
        }
        Ended::Done(Layers {
            canvas: decomposition.canvas as u32,
            layers,
        })
    }

    fn transcribe(
        &mut self,
        request: &TranscribeRequest,
        on_progress: &mut OnProgress,
        stop: &dyn Fn() -> bool,
    ) -> Ended<Transcript> {
        let Loaded::Transcriber(model) = &self.loaded else {
            return Ended::failed(
                Failure::NotLoaded,
                "no model that writes down a recording is loaded",
            );
        };

        let options = TranscriptionOptions {
            language: request
                .language
                .as_ref()
                .map(|language| language.trim().to_string())
                .filter(|language| !language.is_empty()),
            context: request.context.clone(),
        };
        let seconds = request.recording.samples.len() as f64 / request.recording.rate as f64;
        info(&format!(
            "writing down {seconds:.1}s of recording{}",
            match &options.language {
                Some(language) => format!(" in {language}"),
                None => String::new(),
            }
        ));

        let started = Instant::now();
        let mut reporter = Reporter {
            on_progress,
            started,
        };
        let mut transcribing = Transcribing {
            progress: TranscriptionProgress::Listening { done: 0, total: 1 },
        };
        reporter.tell(&Doing::Transcribing(transcribing.clone()));
        let mut report = |progress| {
            transcribing.progress = progress;
            reporter.tell(&Doing::Transcribing(transcribing.clone()));
            match stop() {
                true => ControlFlow::Break(()),
                false => ControlFlow::Continue(()),
            }
        };

        match model.transcribe(&request.recording, &options, &mut report) {
            Ok(Some(transcript)) => {
                info(&format!(
                    "written down in {:.1}s: {} characters{}",
                    started.elapsed().as_secs_f64(),
                    transcript.text.chars().count(),
                    match transcript.language.is_empty() {
                        true => String::new(),
                        false => format!(", {}", transcript.language),
                    }
                ));
                Ended::Done(transcript)
            }
            Ok(None) => {
                info("stopped before the recording was written down");
                Ended::Cancelled
            }
            Err(error) => Ended::failed(Failure::Model, error.to_string()),
        }
    }

    fn voice(&self) -> Option<&dyn Voice> {
        match &self.loaded {
            Loaded::Voice(voice) => Some(voice.as_ref()),
            Loaded::CosyVoice(model) => Some(model),
            _ => None,
        }
    }

    fn speak(
        &mut self,
        request: &SpeakRequest,
        on_progress: &mut OnProgress,
        on_audio: Option<&mut OnAudio>,
        stop: &dyn Fn() -> bool,
    ) -> Ended<Sound> {
        let ratio = self.vocoder_ratio;
        let Some(voice) = self.voice() else {
            return Ended::failed(Failure::NotLoaded, "no voice is loaded");
        };
        if request.like.is_some() {
            if let Some(why) = voice.no_likeness_because() {
                return Ended::failed(Failure::Model, why);
            }
        }
        let style = request
            .style
            .as_ref()
            .map(|style| style.trim().to_string())
            .filter(|style| !style.is_empty());
        if style.is_some() && voice.styles().is_empty() {
            return Ended::failed(Failure::Model, format!("{} takes no style", voice.name()));
        }
        let reference_transcript = request
            .reference_transcript
            .as_ref()
            .map(|words| words.trim().to_string())
            .filter(|words| !words.is_empty());
        if reference_transcript.is_some() && request.like.is_none() {
            return Ended::failed(
                Failure::Model,
                "a reference transcript is the words of the reference recording, and there is none",
            );
        }

        let options = SpeechOptions {
            speed: request.speed.clamp(0.25, 4.0),
            temperature: request.temperature.clamp(0.0, 2.0),
            seed: Some(request.seed),
            style,
            reference_transcript,
        };

        let started = Instant::now();
        let mut reporter = Reporter {
            on_progress,
            started,
        };
        // A streaming reading takes turns between saying and sounding a hop at a time, so
        // the vocoder's share of it is no measure of a reading that says everything first, and
        // its bar is drawn from the tokens alone.
        let streaming = on_audio.is_some();
        let mut say = Say {
            started,
            ..Say::new(0, ratio, streaming)
        };
        reporter.tell(&Doing::Speaking(say.clone()));
        // Parts handed over so far, for a streaming reading's stage: counted where they are
        // handed over and read where progress is told.
        let parts = std::cell::Cell::new(0u32);
        let mut report = |progress| {
            say.told(progress);
            say.parts = parts.get();
            reporter.tell(&Doing::Speaking(say.clone()));
            match stop() {
                true => ControlFlow::Break(()),
                false => ControlFlow::Continue(()),
            }
        };

        let like = request.like.as_ref();
        let spoken = match on_audio {
            None => voice.speak(&request.text, like, &options, &mut report),
            Some(on_audio) => {
                let mut heard = |part: &Sound| {
                    on_audio(part);
                    parts.set(parts.get() + 1);
                    match stop() {
                        true => ControlFlow::Break(()),
                        false => ControlFlow::Continue(()),
                    }
                };
                voice.speak_streaming(&request.text, like, &options, &mut report, &mut heard)
            }
        };
        let measured = say.measured_ratio(Instant::now());
        match spoken {
            Ok(Some(sound)) => {
                if let (Some(ratio), false) = (measured, streaming) {
                    self.vocoder_ratio = ratio;
                }
                Ended::Done(sound)
            }
            Ok(None) => Ended::Cancelled,
            Err(error) => Ended::failed(Failure::Model, error.to_string()),
        }
    }

    fn convert(
        &mut self,
        request: &VoiceConversionRequest,
        on_progress: &mut OnProgress,
        stop: &dyn Fn() -> bool,
    ) -> Ended<Sound> {
        let converter: &dyn Converter = match &self.loaded {
            Loaded::CosyVoice(model) => model,
            _ => {
                return Ended::failed(
                    Failure::NotLoaded,
                    "no model that converts voices is loaded",
                )
            }
        };
        if request.convert_style {
            if let Some(why) = converter.no_style_conversion_because() {
                return Ended::failed(Failure::Model, why);
            }
        }

        let options = ConversionOptions {
            steps: request.steps.clamp(1, 100) as i32,
            convert_style: request.convert_style,
            seed: Some(request.seed),
        };

        let mut reporter = Reporter {
            on_progress,
            started: Instant::now(),
        };
        let mut convert = Convert::new(options.convert_style);
        reporter.tell(&Doing::Converting(convert.clone()));
        let mut report = |progress| {
            convert.heard(progress);
            reporter.tell(&Doing::Converting(convert.clone()));
            match stop() {
                true => ControlFlow::Break(()),
                false => ControlFlow::Continue(()),
            }
        };

        match converter.convert(&request.source, &request.reference, &options, &mut report) {
            Ok(Some(sound)) => Ended::Done(sound),
            Ok(None) => Ended::Cancelled,
            Err(error) => Ended::failed(Failure::Model, error.to_string()),
        }
    }
}

/// One line about what the engine is doing, for whoever is listening: see [`crate::log`]. Said
/// to come from where this was called.
#[track_caller]
fn info(message: &str) {
    crate::log::write(crate::log::Level::Info, message);
}

/// `image` stretched to `width` by `height`, Lanczos, as the draw command scales a picture to start from.
fn scaled(image: &Image, width: u32, height: u32) -> Vec<u8> {
    if image.width == width && image.height == height {
        return image.rgb.clone();
    }
    let Some(source) = image::RgbImage::from_raw(image.width, image.height, image.rgb.clone())
    else {
        return vec![0; (width * height * 3) as usize];
    };
    image::imageops::resize(
        &source,
        width,
        height,
        image::imageops::FilterType::Lanczos3,
    )
    .into_raw()
}

/// Which device a name means, as the C API numbers them.
pub fn device_option(device: Device) -> DeviceOption {
    match device {
        Device::Cpu => DeviceOption::Cpu,
        Device::Cuda => DeviceOption::Cuda,
        // Page-locked host memory is a place for CUDA's tensors, not a device a run goes to.
        Device::CudaHost => DeviceOption::Cuda,
        Device::Metal => DeviceOption::Metal,
        Device::Vulkan => DeviceOption::Vulkan,
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    use std::sync::mpsc::RecvTimeoutError;
    use std::time::Duration;

    fn a_request(prompt: &str, images: Vec<(&str, Image)>) -> DrawRequest {
        DrawRequest {
            prompt: prompt.to_string(),
            negative: String::new(),
            images: images
                .into_iter()
                .map(|(key, image)| (key.to_string(), image))
                .collect(),
            width: 64,
            height: 64,
            steps: 1,
            guidance: 5.0,
            seed: 7,
            strength: 0.8,
            control_scale: 1.0,
        }
    }

    fn a_pixel() -> Image {
        Image {
            width: 1,
            height: 1,
            rgb: vec![0, 0, 0],
        }
    }

    #[test]
    fn names_the_pictures_the_way_qwen_image_does() {
        let (prompt, order) = numbered("put the girl in <|girl|> into the room in <|room|>");
        assert_eq!(prompt, "put the girl in <image1> into the room in <image2>");
        assert_eq!(order, ["girl", "room"]);

        // Numbered by where a name first appears; a name written twice is one picture.
        let (prompt, order) = numbered("<|b|> beside <|a|>, and <|b|> again");
        assert_eq!(prompt, "<image1> beside <image2>, and <image1> again");
        assert_eq!(order, ["b", "a"]);

        let (prompt, order) = numbered("no pictures at all");
        assert_eq!(prompt, "no pictures at all");
        assert!(order.is_empty());
    }

    #[test]
    fn a_draw_is_refused_at_the_call_for_what_no_model_could_run() {
        assert!(check_draw(&a_request("a cat", vec![])).is_ok());
        assert!(check_draw(&a_request(
            "the girl in <|girl|>",
            vec![("girl", a_pixel())]
        ))
        .is_ok());
        assert!(check_draw(&a_request("a cat", vec![(START_FROM, a_pixel())])).is_ok());

        for (request, says) in [
            (a_request("   ", vec![]), "empty"),
            (a_request("<|girl|>", vec![]), "names no image"),
            (
                a_request("a cat", vec![("girl", a_pixel())]),
                "not named in the prompt",
            ),
            (
                a_request("<|start_from|>", vec![(START_FROM, a_pixel())]),
                "not for the prompt",
            ),
            (a_request("a <|endofprompt|> cat", vec![]), "names no image"),
            (a_request("a <|end of|> cat", vec![]), "own markers"),
            (a_request("a <| cat", vec![]), "does not close"),
            (
                a_request("<|a|> <|a|>", vec![("a", a_pixel()), ("a", a_pixel())]),
                "two images",
            ),
            (a_request("<|a-b|>", vec![("a-b", a_pixel())]), "not a key"),
            (
                a_request(
                    "<|a|>",
                    vec![(
                        "a",
                        Image {
                            width: 2,
                            height: 2,
                            rgb: vec![0; 3],
                        },
                    )],
                ),
                "rows of RGB",
            ),
        ] {
            let refused = check_draw(&request).expect_err(&request.prompt);
            assert!(refused.contains(says), "{}: {refused}", request.prompt);
        }

        let mut huge = a_request("a cat", vec![]);
        huge.width = 4096;
        assert!(check_draw(&huge).unwrap_err().contains("from 64 to 2048"));

        // A size only some models draw at is theirs to refuse: Anima's cards name 912.
        let mut sixteenth = a_request("a cat", vec![]);
        sixteenth.width = 912;
        check_draw(&sixteenth).unwrap();
    }

    /// What a test hears of one job: its progress and its one end, over a channel, so that it can
    /// wait for them on its own thread.
    fn listen<T: Send + 'static>() -> (OnProgress, OnComplete<T>, Receiver<Ended<T>>) {
        let (ended, heard) = mpsc::channel();
        (
            Box::new(|_| {}),
            Box::new(move |end| ended.send(end).unwrap()),
            heard,
        )
    }

    fn an_engine() -> Engine {
        Engine::new(EngineOptions {
            device: DeviceOption::Cpu,
        })
        .expect("an engine on the CPU")
    }

    fn a_speaking() -> SpeakRequest {
        SpeakRequest {
            text: "hello there".to_string(),
            like: None,
            speed: 1.0,
            temperature: 0.8,
            style: None,
            reference_transcript: None,
            seed: 7,
        }
    }

    const WAIT: Duration = Duration::from_secs(20);

    #[test]
    fn jobs_run_in_order_and_each_ends_exactly_once() {
        let engine = an_engine();

        // Nothing is loaded: a job for a model of a kind that is not there fails, and says so.
        let (progress, complete, heard) = listen::<Sound>();
        engine.speak(a_speaking(), progress, complete).unwrap();
        match heard.recv_timeout(WAIT).unwrap() {
            Ended::Failed(Failure::NotLoaded, _) => {}
            other => panic!("{other:?}"),
        }

        let (progress, complete, loaded) = listen::<()>();
        engine.load(TONES, progress, complete);
        let (progress, complete, spoken) = listen::<Sound>();
        engine.speak(a_speaking(), progress, complete).unwrap();

        assert_eq!(loaded.recv_timeout(WAIT).unwrap(), Ended::Done(()));
        match spoken.recv_timeout(WAIT).unwrap() {
            Ended::Done(sound) => assert!(!sound.samples.is_empty()),
            other => panic!("{other:?}"),
        }
        // And nothing more: one end each.
        assert!(matches!(
            loaded.recv_timeout(Duration::from_millis(50)),
            Err(RecvTimeoutError::Timeout | RecvTimeoutError::Disconnected)
        ));
        assert!(matches!(
            spoken.recv_timeout(Duration::from_millis(50)),
            Err(RecvTimeoutError::Timeout | RecvTimeoutError::Disconnected)
        ));

        // A picture asked of a voice is the wrong kind.
        let (progress, complete, drawn) = listen::<Image>();
        engine
            .draw(a_request("a cat", vec![]), progress, complete)
            .unwrap();
        assert!(matches!(
            drawn.recv_timeout(WAIT).unwrap(),
            Ended::Failed(Failure::NotLoaded, _)
        ));
    }

    #[test]
    fn a_name_that_is_no_manifest_is_unknown() {
        // A published name included: it is the model manager's to turn into a path.
        let engine = an_engine();
        let (progress, complete, heard) = listen::<()>();
        engine.load("sdxl:base", progress, complete);
        assert!(matches!(
            heard.recv_timeout(WAIT).unwrap(),
            Ended::Failed(Failure::UnknownModel, _)
        ));
    }

    #[test]
    fn a_streaming_reading_hands_over_its_parts_and_then_the_whole() {
        let engine = an_engine();
        let (progress, complete, loaded) = listen::<()>();
        engine.load(TONES, progress, complete);
        assert_eq!(loaded.recv_timeout(WAIT).unwrap(), Ended::Done(()));

        // The parts, in order, before the end -- which is the whole, the parts end to end.
        let (part, parts) = mpsc::channel::<Sound>();
        let (progress, complete, heard) = listen::<Sound>();
        engine
            .speak_streaming(
                a_speaking(),
                progress,
                Box::new(move |sound| part.send(sound.clone()).unwrap()),
                complete,
            )
            .unwrap();
        let Ended::Done(whole) = heard.recv_timeout(WAIT).unwrap() else {
            panic!("the reading did not finish");
        };
        let parts: Vec<Sound> = parts.try_iter().collect();
        assert!(!parts.is_empty());
        let joined: Vec<f32> = parts.iter().flat_map(|p| p.samples.clone()).collect();
        assert_eq!(joined, whole.samples);

        // And it is the reading `speak` makes of the same request.
        let (progress, complete, heard) = listen::<Sound>();
        engine.speak(a_speaking(), progress, complete).unwrap();
        assert_eq!(heard.recv_timeout(WAIT).unwrap(), Ended::Done(whole));

        // Nothing to say is refused at the call, as `speak` refuses it.
        let empty = SpeakRequest {
            text: "  ".to_string(),
            ..a_speaking()
        };
        let (progress, complete, _) = listen::<Sound>();
        assert!(engine
            .speak_streaming(empty, progress, Box::new(|_| {}), complete)
            .is_err());
    }

    #[test]
    fn a_reference_transcript_needs_a_recording_and_a_voice_that_reads_none_ignores_it() {
        let engine = an_engine();
        let (progress, complete, loaded) = listen::<()>();
        engine.load(TONES, progress, complete);
        assert_eq!(loaded.recv_timeout(WAIT).unwrap(), Ended::Done(()));

        let ended = |request: SpeakRequest| -> Ended<Sound> {
            let (progress, complete, heard) = listen::<Sound>();
            engine.speak(request, progress, complete).unwrap();
            heard.recv_timeout(WAIT).unwrap()
        };
        let words = Some("the words of the recording".to_string());
        let recording = Some(Sound::new(vec![0.0; 1600], 16000));

        // The words of a recording that is not there.
        match ended(SpeakRequest {
            reference_transcript: words.clone(),
            ..a_speaking()
        }) {
            Ended::Failed(Failure::Model, why) => assert!(why.contains("there is none"), "{why}"),
            other => panic!("{other:?}"),
        }

        // The stand-in hears a recording's pitch and not its words: it reads as it would without.
        let without = ended(SpeakRequest {
            like: recording.clone(),
            ..a_speaking()
        });
        let with = ended(SpeakRequest {
            like: recording,
            reference_transcript: words,
            ..a_speaking()
        });
        assert!(matches!(without, Ended::Done(_)));
        assert_eq!(with, without);

        // A blank one is none at all.
        let blank = ended(SpeakRequest {
            reference_transcript: Some("  ".to_string()),
            ..a_speaking()
        });
        assert!(matches!(blank, Ended::Done(_)));
    }

    #[test]
    fn a_job_cancelled_while_it_waits_ends_cancelled_without_running() {
        let engine = an_engine();
        let (progress, complete, loaded) = listen::<()>();
        engine.load(TONES, progress, complete);

        // Held up behind a job that blocks until it is told to go, so that the next ones wait.
        let (go, wait) = mpsc::channel::<()>();
        let wait = Mutex::new(wait);
        let (_, complete, first) = listen::<Sound>();
        let mut blocked = false;
        engine
            .speak(
                a_speaking(),
                Box::new(move |_| {
                    if !blocked {
                        blocked = true;
                        let _ = wait.lock().unwrap().recv_timeout(WAIT);
                    }
                }),
                complete,
            )
            .unwrap();
        let (progress, complete, second) = listen::<Sound>();
        let waiting = engine.speak(a_speaking(), progress, complete).unwrap();

        engine.cancel(waiting);
        go.send(()).unwrap();

        assert_eq!(loaded.recv_timeout(WAIT).unwrap(), Ended::Done(()));
        assert!(matches!(first.recv_timeout(WAIT).unwrap(), Ended::Done(_)));
        assert_eq!(second.recv_timeout(WAIT).unwrap(), Ended::Cancelled);
    }

    #[test]
    fn dropping_the_engine_ends_every_job_still_waiting() {
        let (progress, complete, loaded) = listen::<()>();
        let mut ends = Vec::new();
        {
            let engine = an_engine();
            engine.load(TONES, progress, complete);
            for _ in 0..3 {
                let (progress, complete, heard) = listen::<Sound>();
                engine.speak(a_speaking(), progress, complete).unwrap();
                ends.push(heard);
            }
        }
        // Each got its one end, whatever it was: dropped engines leave nobody waiting.
        assert!(loaded.recv_timeout(WAIT).is_ok());
        for heard in ends {
            assert!(heard.recv_timeout(WAIT).is_ok());
        }
    }

    #[test]
    fn the_stratified_stack_leaves_out_split_parts_and_every_depth_layer() {
        let layer = |kind, id, depth, derived_from_id| LayerOut {
            kind,
            id,
            name: String::new(),
            left: 0,
            top: 0,
            image: RgbaImage {
                width: 0,
                height: 0,
                rgba: Vec::new(),
            },
            depth,
            derived_from_id,
        };
        // Back hair split in two, a top that was not; then a depth layer for each.
        let layers = Layers {
            canvas: 64,
            layers: vec![
                layer(LayerKind::Part, 1, 0.95, None),
                layer(LayerKind::Part, 2, 0.6, None),
                layer(LayerKind::StratifiedNear, 3, 0.95, Some(1)),
                layer(LayerKind::StratifiedFar, 4, 0.95, Some(1)),
                layer(LayerKind::Depth, 5, 0.95, Some(1)),
                layer(LayerKind::Depth, 6, 0.6, Some(2)),
                layer(LayerKind::Depth, 7, 0.95, Some(3)),
                layer(LayerKind::Depth, 8, 0.95, Some(4)),
            ],
        };
        // The top has a depth layer naming it, and is not split by it.
        assert_eq!(layers.stratified_stack(), [3, 2, 1]);
    }

    #[test]
    fn a_reading_reports_progress_before_it_ends() {
        let engine = an_engine();
        let (progress, complete, loaded) = listen::<()>();
        engine.load(TONES, progress, complete);
        assert_eq!(loaded.recv_timeout(WAIT).unwrap(), Ended::Done(()));

        let (told, heard_progress) = mpsc::channel();
        let (_, complete, spoken) = listen::<Sound>();
        engine
            .speak(
                a_speaking(),
                Box::new(move |progress| told.send(progress.clone()).unwrap()),
                complete,
            )
            .unwrap();
        assert!(matches!(spoken.recv_timeout(WAIT).unwrap(), Ended::Done(_)));

        let reports: Vec<Progress> = heard_progress.try_iter().collect();
        assert!(!reports.is_empty());
        assert_eq!(reports[0].stage, Stage::TextToSpeechReadingText);
        assert!(reports.iter().all(|report| !report.words.is_empty()));
        // A reading said all at once has none of a streaming reading's stages.
        assert!(reports.iter().all(|report| !matches!(
            report.stage,
            Stage::StreamingTextToSpeechReadingText
                | Stage::StreamingTextToSpeechGeneratingTokens
                | Stage::StreamingTextToSpeechSynthesizingAudio
        )));
    }

    #[test]
    fn a_transcription_is_refused_at_the_call_or_run_by_a_transcriber() {
        let engine = an_engine();
        let request = |samples: Vec<f32>| TranscribeRequest {
            recording: Sound::new(samples, 16000),
            language: None,
            context: String::new(),
        };

        // Nothing to write down is refused before it is queued.
        let (progress, complete, _) = listen::<Transcript>();
        assert!(engine
            .transcribe(request(Vec::new()), progress, complete)
            .unwrap_err()
            .contains("empty"));

        // A recording with no transcriber loaded -- here, a voice -- fails, saying so.
        let (progress, complete, loaded) = listen::<()>();
        engine.load(TONES, progress, complete);
        assert_eq!(loaded.recv_timeout(WAIT).unwrap(), Ended::Done(()));
        let (progress, complete, written) = listen::<Transcript>();
        engine
            .transcribe(request(vec![0.0; 16000]), progress, complete)
            .unwrap();
        match written.recv_timeout(WAIT).unwrap() {
            Ended::Failed(Failure::NotLoaded, why) => assert!(why.contains("recording"), "{why}"),
            other => panic!("{other:?}"),
        }
    }

    /// A manifest of `kind` in a directory of its own, naming one weights file that is not there:
    /// what a manifest fetched on its own, or bundled with an app, looks like.
    pub(crate) fn a_manifest(stem: &str, kind: &str) -> String {
        let directory =
            std::env::temp_dir().join(format!("waifu-manifest-{}-{stem}", std::process::id()));
        std::fs::create_dir_all(&directory).unwrap();
        let path = directory.join(format!("{stem}.yaml"));
        let text =
            format!("weights:\n  - {stem}.safetensors\nconfig:\n  model:\n    type: {kind}\n");
        std::fs::write(&path, text).unwrap();
        path.to_str().unwrap().to_string()
    }
}
