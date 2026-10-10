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
//! This is what the C API in waifu-ffi is a thin skin over; see docs/ffi.md for the contract it
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
use crate::describe::{self, converter_kind, voice_kind, Chosen, ConverterKind, VoiceKind, TONES};
use crate::hub;
use crate::image_model::ImageModel;
use crate::indextts::IndexTts;
use crate::progress::{Convert, Doing, Fetch, Run, Say, VOCODER_RATIO};
use crate::runtime::{DeviceOption, Runtime};
use crate::wav::Sound;
use crate::{
    from_rgb8, to_rgb8, ConversionOptions, ConversionProgress, Converter, Device,
    GenerationOptions, GenerationProgress, Manifest, SpeechOptions, SpeechProgress, Tones, Voice,
};

/// A job, as the engine numbers them. Never 0, which the C API keeps for a refusal.
pub type JobId = u64;

/// The keys an image is given under that are not names a prompt can write, but say what the image
/// is for.
pub const START_FROM: &str = "start_from";
pub const CONTROL: &str = "control";
const KEPT_KEYS: &[&str] = &[START_FROM, CONTROL];

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

/// Which part of a job a progress report is about.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Stage {
    Fetching,
    Reading,
    Encoding,
    Drawing,
    Decoding,
    Listening,
    Saying,
    Sounding,
}

/// How far along a job is: what `on_progress` is told.
#[derive(Clone, Debug, PartialEq)]
pub struct Progress {
    pub stage: Stage,
    /// Of the whole job, 0 to 1, or None where nothing can say -- reading weights.
    pub fraction: Option<f64>,
    /// What the stage counts: bytes, steps or tokens. 0 and 0 where it counts nothing.
    pub done: u64,
    pub total: u64,
    /// Fetching only: which file of how many.
    pub part: u32,
    pub parts: u32,
    pub file: Option<String>,
    /// "step 3 of 8", for a status line.
    pub words: String,
    /// Since the job started.
    pub seconds: f64,
}

/// Why a job failed, in the C API's terms.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Failure {
    UnknownModel,
    Fetch,
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

/// The engine: a thread, a queue to it, and how to stop what is on it.
///
/// Dropping it stops what is running after the step it is on, cancels everything waiting --
/// every one of them still gets its completion, as cancelled -- drops the model and joins the
/// thread.
pub struct Engine {
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
        on_complete: OnComplete<Sound>,
    },
    VoiceConversion {
        request: VoiceConversionRequest,
        on_progress: OnProgress,
        on_complete: OnComplete<Sound>,
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
        }
    }
}

impl Engine {
    /// Starts an engine on `device`. Refused where the device is not one this build can use on
    /// this machine.
    pub fn new(device: DeviceOption) -> Result<Engine, String> {
        let runtime = device.resolve();
        if !runtime.device().is_available() {
            return Err(format!("{} is not available on this machine", runtime.name()));
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
            commands,
            shared,
            thread: Some(thread),
        })
    }

    fn submit(&self, work: Work) -> JobId {
        let job = self.shared.next.fetch_add(1, Ordering::Relaxed);
        lock(&self.shared.open).insert(job);
        // The thread is there until this is dropped, so the send only fails if it died -- and a
        // job it never sees still has to be ended.
        if let Err(mpsc::SendError(Command::Job(_, work))) = self.commands.send(Command::Job(job, work)) {
            lock(&self.shared.open).remove(&job);
            work.cancel();
        }
        job
    }

    /// Fetches `model` if it has to, and reads it onto the device, dropping the one before.
    pub fn load(&self, model: &str, on_progress: OnProgress, on_complete: OnComplete<()>) -> JobId {
        self.submit(Work::Load {
            model: model.to_string(),
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

    /// Stops `job`: after the step it is on where it is running, before it starts where it waits.
    pub fn cancel(&self, job: JobId) {
        if !lock(&self.shared.open).contains(&job) {
            return;
        }
        lock(&self.shared.cancelled).insert(job);
        if *lock(&self.shared.running) == Some(job) {
            self.shared.stop.store(true, Ordering::Relaxed);
        }
    }

    pub fn cancel_all(&self) {
        let open: Vec<JobId> = lock(&self.shared.open).iter().copied().collect();
        lock(&self.shared.cancelled).extend(open);
        if lock(&self.shared.running).is_some() {
            self.shared.stop.store(true, Ordering::Relaxed);
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
            return Err(format!("the {name} is {side}: it has to be from 64 to 2048"));
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
            return Err(format!("\"{key}\" is not a key: letters, digits and _ only"));
        }
        if !keys.insert(key.as_str()) {
            return Err(format!("two images are given under \"{key}\""));
        }
        if image.width == 0 || image.height == 0 || image.rgb.len() != (image.width * image.height * 3) as usize {
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

// -- the thread ------------------------------------------------------------------------------------

/// What the thread has read.
enum Loaded {
    Nothing,
    Image { model: ImageModel, chosen: Chosen },
    Voice(Box<dyn Voice>),
    /// Both a voice and a converter: one model, used as either.
    CosyVoice(CosyVoice3),
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
        let (stage, done, total, part, parts, file) = match doing {
            Doing::Nothing => return,
            Doing::Fetching(fetch) => (
                Stage::Fetching,
                fetch.done,
                fetch.total.unwrap_or(0),
                fetch.part as u32,
                fetch.parts as u32,
                (!fetch.file.is_empty()).then(|| fetch.file.clone()),
            ),
            Doing::Reading { .. } => (Stage::Reading, 0, 0, 0, 0, None),
            Doing::Drawing(run) => match run.progress {
                GenerationProgress::Encoding => (Stage::Encoding, 0, 0, 0, 0, None),
                GenerationProgress::Step { done, total } => {
                    (Stage::Drawing, done as u64, total as u64, 0, 0, None)
                }
                GenerationProgress::Decoding => (Stage::Decoding, 0, 0, 0, 0, None),
            },
            Doing::Speaking(say) => match say.progress {
                SpeechProgress::Reading => (Stage::Encoding, 0, 0, 0, 0, None),
                SpeechProgress::Saying { done, expected } => {
                    (Stage::Saying, done.max(0) as u64, expected.max(0) as u64, 0, 0, None)
                }
                SpeechProgress::Sounding => (Stage::Sounding, 0, 0, 0, 0, None),
            },
            Doing::Converting(convert) => match convert.progress {
                ConversionProgress::Listening => (Stage::Listening, 0, 0, 0, 0, None),
                ConversionProgress::Saying { done, expected } => {
                    (Stage::Saying, done.max(0) as u64, expected.max(0) as u64, 0, 0, None)
                }
                ConversionProgress::Drawing { done, total } => {
                    (Stage::Drawing, done.max(0) as u64, total.max(0) as u64, 0, 0, None)
                }
                ConversionProgress::Sounding { done, total } => {
                    (Stage::Sounding, done.max(0) as u64, total.max(0) as u64, 0, 0, None)
                }
            },
        };

        (self.on_progress)(&Progress {
            stage,
            fraction: doing.fraction(),
            done,
            total,
            part,
            parts,
            file,
            words: doing.words(),
            seconds: self.started.elapsed().as_secs_f64(),
        });
    }
}

/// The name a screen shows for `asked`: the file name of a path, since the whole of one does not
/// fit, and the name itself otherwise.
fn shown_name(asked: &str) -> String {
    match asked.rsplit_once('/') {
        Some((_, file)) if !file.is_empty() => file.to_string(),
        _ => asked.to_string(),
    }
}

/// Fetches the published model `asked` names, if it is not here already, and hands back its
/// manifest; a manifest's own path is handed back as it is. On the caller's thread: what
/// `waifu_modelmanager_fetch_async` runs on a thread of its own, and what a load starts with.
pub fn fetch(asked: &str, on_progress: &mut OnProgress, stop: &dyn Fn() -> bool) -> Ended<std::path::PathBuf> {
    let mut reporter = Reporter {
        on_progress,
        started: Instant::now(),
    };
    fetch_reporting(asked, &mut reporter, stop)
}

/// Fetches the manifest of every published model not yet here, and none of their packages: see
/// [`hub::fetch_manifests`]. Each is reported as a part of the whole, under the model name
/// "model metadata".
pub fn fetch_manifests(on_progress: &mut OnProgress, stop: &dyn Fn() -> bool) -> Ended<()> {
    let mut reporter = Reporter {
        on_progress,
        started: Instant::now(),
    };
    let mut fetch = Fetch {
        model: "model metadata".to_string(),
        hub: None,
        file: String::new(),
        done: 0,
        total: None,
        part: 0,
        parts: 0,
    };
    let fetched = hub::fetch_manifests(
        &mut |progress| {
            match progress {
                hub::Progress::From { hub } => fetch.hub = Some(hub),
                hub::Progress::Fetching {
                    file,
                    done,
                    total,
                    part,
                    parts,
                } => {
                    fetch.file = file.to_string();
                    (fetch.done, fetch.total, fetch.part, fetch.parts) = (done, total, part, parts);
                }
                hub::Progress::Fetched {
                    file,
                    bytes,
                    part,
                    parts,
                } => {
                    fetch.file = file.to_string();
                    (fetch.done, fetch.total, fetch.part, fetch.parts) = (bytes, Some(bytes), part, parts);
                }
            }
            reporter.tell(&Doing::Fetching(fetch.clone()));
        },
        stop,
    );
    match fetched {
        Ok(()) => Ended::Done(()),
        Err(error) if hub::stopped(&error) => Ended::Cancelled,
        Err(error) => Ended::failed(Failure::Fetch, error.to_string()),
    }
}

fn fetch_reporting(asked: &str, reporter: &mut Reporter, stop: &dyn Fn() -> bool) -> Ended<std::path::PathBuf> {
    if hub::full_name(asked).is_none() && !Path::new(asked).is_file() {
        return Ended::failed(
            Failure::UnknownModel,
            format!("there is no model called \"{asked}\": neither a published name nor a manifest that exists"),
        );
    }

    let mut fetch = Fetch {
        model: shown_name(asked),
        hub: None,
        file: String::new(),
        done: 0,
        total: None,
        part: 0,
        parts: 0,
    };
    reporter.tell(&Doing::Fetching(fetch.clone()));

    let fetched = hub::resolve_reporting(
        asked,
        &mut |progress| {
            match progress {
                hub::Progress::From { hub } => fetch.hub = Some(hub),
                hub::Progress::Fetching {
                    file,
                    done,
                    total,
                    part,
                    parts,
                } => {
                    fetch.file = file.to_string();
                    (fetch.done, fetch.total, fetch.part, fetch.parts) = (done, total, part, parts);
                }
                hub::Progress::Fetched {
                    file,
                    bytes,
                    part,
                    parts,
                } => {
                    fetch.file = file.to_string();
                    (fetch.done, fetch.total, fetch.part, fetch.parts) = (bytes, Some(bytes), part, parts);
                }
            }
            reporter.tell(&Doing::Fetching(fetch.clone()));
        },
        stop,
    );
    match fetched {
        Ok(path) => Ended::Done(path),
        Err(error) if hub::stopped(&error) => Ended::Cancelled,
        Err(error) => Ended::failed(Failure::Fetch, error.to_string()),
    }
}

impl Worker {
    fn run(&mut self, shared: &Shared, work: Work) {
        let stop = || shared.stop.load(Ordering::Relaxed);
        match work {
            Work::Load {
                model,
                mut on_progress,
                on_complete,
            } => {
                let ended = guarded(|| self.load(&model, &mut on_progress, &stop));
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
                on_complete(ended);
            }
            Work::Speak {
                request,
                mut on_progress,
                on_complete,
            } => {
                let ended = guarded(|| self.speak(&request, &mut on_progress, &stop));
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
        }
    }

    fn load(&mut self, asked: &str, on_progress: &mut OnProgress, stop: &dyn Fn() -> bool) -> Ended<()> {
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
        let path = match fetch_reporting(asked, &mut reporter, stop) {
            Ended::Done(path) => path,
            Ended::Cancelled => return Ended::Cancelled,
            Ended::Failed(failure, message) => return Ended::Failed(failure, message),
        };
        let shown = shown_name(asked);

        reporter.tell(&Doing::Reading { model: shown });
        let (device, residency) = (self.runtime.device(), self.runtime.residency());
        let read = || -> crate::Result<Loaded> {
            if let Some(kind) = voice_kind(&path) {
                let manifest = Manifest::open(&path)?;
                return Ok(match kind {
                    VoiceKind::CosyVoice3 => {
                        Loaded::CosyVoice(CosyVoice3::from_manifest(device, residency, &manifest)?)
                    }
                    VoiceKind::IndexTts => {
                        Loaded::Voice(Box::new(IndexTts::from_manifest(device, residency, &manifest)?))
                    }
                });
            }
            if let Some(ConverterKind::CosyVoice3) = converter_kind(&path) {
                let manifest = Manifest::open(&path)?;
                return Ok(Loaded::CosyVoice(CosyVoice3::from_manifest(device, residency, &manifest)?));
            }
            let model = ImageModel::from_manifest(&path, device, residency)?;
            let mut chosen = describe::look_at(asked);
            chosen.no_picture_because = model.no_picture_because().map(str::to_string);
            chosen.on_disk = true;
            chosen.in_memory = true;
            Ok(Loaded::Image { model, chosen })
        };
        match read() {
            Ok(loaded) => {
                self.loaded = loaded;
                self.name = asked.to_string();
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
        self.loaded = Loaded::Nothing;
        self.name.clear();
        crate::flint::release_memory();
    }

    fn draw(&mut self, request: &DrawRequest, on_progress: &mut OnProgress, stop: &dyn Fn() -> bool) -> Ended<Image> {
        let Loaded::Image { model, chosen } = &self.loaded else {
            return Ended::failed(Failure::NotLoaded, "no model that draws is loaded");
        };

        // What the model cannot be given, said before anything runs.
        if let Some((key, _)) = request.images.iter().find(|(key, _)| !KEPT_KEYS.contains(&key.as_str())) {
            return Ended::failed(
                Failure::Model,
                format!("{} reads words only: it cannot be shown <|{key}|>", chosen.full_name),
            );
        }
        if request.images.iter().any(|(key, _)| key == CONTROL) {
            return Ended::failed(Failure::Model, format!("{} has no ControlNet to hold to a control image", chosen.full_name));
        }
        let start_from = request.images.iter().find(|(key, _)| key == START_FROM).map(|(_, image)| image);
        if start_from.is_some() {
            if let Some(why) = model.no_picture_because() {
                return Ended::failed(Failure::Model, why);
            }
        }

        // Both or neither, and neither for a model with no second pass to steer.
        let (guidance, negative) = match chosen.takes_guidance {
            true => (request.guidance, request.negative.clone()),
            false => (chosen.defaults.guidance_scale, String::new()),
        };
        let options = GenerationOptions {
            width: request.width as i32,
            height: request.height as i32,
            num_steps: request.steps as i32,
            guidance_scale: guidance,
            negative_prompt: negative,
            seed: Some(request.seed),
            strength: request.strength.clamp(0.0, 1.0),
        };

        let started = Instant::now();
        let mut reporter = Reporter { on_progress, started };
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
            None => model.generate_reporting(&request.prompt, &options, &mut report),
        };
        let tensor = match drawn {
            Ok(Some(tensor)) => tensor,
            Ok(None) => return Ended::Cancelled,
            Err(error) => return Ended::failed(Failure::Model, error.to_string()),
        };

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

    fn voice(&self) -> Option<&dyn Voice> {
        match &self.loaded {
            Loaded::Voice(voice) => Some(voice.as_ref()),
            Loaded::CosyVoice(model) => Some(model),
            _ => None,
        }
    }

    fn speak(&mut self, request: &SpeakRequest, on_progress: &mut OnProgress, stop: &dyn Fn() -> bool) -> Ended<Sound> {
        let ratio = self.vocoder_ratio;
        let Some(voice) = self.voice() else {
            return Ended::failed(Failure::NotLoaded, "no voice is loaded");
        };
        if request.like.is_some() {
            if let Some(why) = voice.no_likeness_because() {
                return Ended::failed(Failure::Model, why);
            }
        }
        let style = request.style.as_ref().map(|style| style.trim().to_string()).filter(|style| !style.is_empty());
        if style.is_some() && voice.styles().is_empty() {
            return Ended::failed(Failure::Model, format!("{} takes no style", voice.name()));
        }

        let options = SpeechOptions {
            speed: request.speed.clamp(0.25, 4.0),
            temperature: request.temperature.clamp(0.0, 2.0),
            seed: Some(request.seed),
            style,
        };

        let started = Instant::now();
        let mut reporter = Reporter { on_progress, started };
        let mut say = Say {
            progress: SpeechProgress::Reading,
            expected: 0,
            started,
            sounding: None,
            ratio,
        };
        reporter.tell(&Doing::Speaking(say.clone()));
        let mut report = |progress| {
            say.progress = progress;
            match progress {
                SpeechProgress::Saying { expected, .. } => say.expected = expected,
                SpeechProgress::Sounding if say.sounding.is_none() => say.sounding = Some(Instant::now()),
                _ => {}
            }
            reporter.tell(&Doing::Speaking(say.clone()));
            match stop() {
                true => ControlFlow::Break(()),
                false => ControlFlow::Continue(()),
            }
        };

        let spoken = voice.speak(&request.text, request.like.as_ref(), &options, &mut report);
        let measured = say.measured_ratio(Instant::now());
        match spoken {
            Ok(Some(sound)) => {
                if let Some(ratio) = measured {
                    self.vocoder_ratio = ratio;
                }
                Ended::Done(sound)
            }
            Ok(None) => Ended::Cancelled,
            Err(error) => Ended::failed(Failure::Model, error.to_string()),
        }
    }

    fn convert(&mut self, request: &VoiceConversionRequest, on_progress: &mut OnProgress, stop: &dyn Fn() -> bool) -> Ended<Sound> {
        let converter: &dyn Converter = match &self.loaded {
            Loaded::CosyVoice(model) => model,
            _ => return Ended::failed(Failure::NotLoaded, "no model that converts voices is loaded"),
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

/// `image` stretched to `width` by `height`, Lanczos, as webui scales a picture to start from.
fn scaled(image: &Image, width: u32, height: u32) -> Vec<u8> {
    if image.width == width && image.height == height {
        return image.rgb.clone();
    }
    let Some(source) = image::RgbImage::from_raw(image.width, image.height, image.rgb.clone()) else {
        return vec![0; (width * height * 3) as usize];
    };
    image::imageops::resize(&source, width, height, image::imageops::FilterType::Lanczos3).into_raw()
}

// -- what is there, as JSON ------------------------------------------------------------------

/// A float as a box can show it: four places, which is finer than any setting steps.
fn showable(value: f32) -> f64 {
    (f64::from(value) * 10_000.0).round() / 10_000.0
}

/// The published models, aliases only: `waifu_modelmanager_catalog_json`.
pub fn catalog_json() -> serde_json::Value {
    let entry = |listed: hub::Listed, kind: &str| {
        serde_json::json!({
            "name": listed.name,
            "full_name": listed.full_name,
            "kind": kind,
            "cached": listed.cached,
            // Whether its manifest is here, with or without its packages: what it suggests can
            // be read from it. See waifu_modelmanager_fetch_manifests_async.
            "manifest_here": hub::local_manifest(listed.name).is_some(),
            "bytes_on_disk": listed.bytes,
            "explicit": listed.explicit,
        })
    };
    let models: Vec<serde_json::Value> = hub::listed()
        .into_iter()
        .map(|listed| entry(listed, "image"))
        .chain(hub::listed_voices().into_iter().map(|listed| entry(listed, "speech")))
        .chain(hub::listed_conversions().into_iter().map(|listed| entry(listed, "conversion")))
        .collect();
    serde_json::json!({ "models": models })
}

/// What a model is, without reading it: `waifu_modelmanager_describe_json`. None for a name that
/// is neither published nor a manifest on the disk.
pub fn describe_json(asked: &str) -> Option<serde_json::Value> {
    if asked != TONES && hub::full_name(asked).is_none() && !Path::new(asked).is_file() {
        return None;
    }

    let mut kinds = Vec::new();
    if describe::is_a_voice(asked) {
        kinds.push("speech");
    }
    if describe::is_a_converter(asked) {
        kinds.push("conversion");
    }

    // What a converter is described as, alone or beside the voice of a model that is both.
    let conversion = || {
        let converter = describe::look_at_converter(asked);
        serde_json::json!({
            "name": converter.name,
            "full_name": converter.full_name,
            "on_disk": converter.on_disk,
            "steps": converter.defaults.steps,
            "rate": converter.rate,
            "converts_style": converter.no_style_because.is_none(),
            "no_style_because": converter.no_style_because,
        })
    };

    if kinds.first() == Some(&"speech") {
        let voice = describe::look_at_voice(asked);
        let mut described = serde_json::json!({
            "kind": "speech",
            "kinds": kinds,
            "name": voice.name,
            "full_name": voice.full_name,
            "on_disk": voice.on_disk,
            "speed": showable(voice.defaults.speed),
            "temperature": showable(voice.defaults.temperature),
            "rate": voice.rate,
            "takes_a_recording": voice.no_likeness_because.is_none(),
            "no_likeness_because": voice.no_likeness_because,
            "styles": voice.styles.iter().map(|style| serde_json::json!({
                "label": style.label,
                "instruction": style.instruction,
            })).collect::<Vec<_>>(),
            "not_a_voice_because": voice.not_a_voice_because,
        });
        // CosyVoice3 reads and converts, with one model: what a window offers for the second.
        if kinds.contains(&"conversion") {
            described["conversion"] = conversion();
        }
        return Some(described);
    }
    if kinds.first() == Some(&"conversion") {
        let converter = describe::look_at_converter(asked);
        return Some(serde_json::json!({
            "kind": "conversion",
            "kinds": kinds,
            "name": converter.name,
            "full_name": converter.full_name,
            "on_disk": converter.on_disk,
            "steps": converter.defaults.steps,
            "rate": converter.rate,
            "converts_style": converter.no_style_because.is_none(),
            "no_style_because": converter.no_style_because,
        }));
    }

    let chosen = describe::look_at(asked);
    let mut image_keys = Vec::new();
    let mut why_not = serde_json::Map::new();
    match &chosen.no_picture_because {
        None => image_keys.push(START_FROM),
        Some(why) => {
            why_not.insert(START_FROM.to_string(), why.clone().into());
        }
    }
    why_not.insert(
        CONTROL.to_string(),
        format!("{} has no ControlNet here", chosen.full_name).into(),
    );
    why_not.insert(
        "prompt_images".to_string(),
        format!("{} reads words only", chosen.full_name).into(),
    );
    Some(serde_json::json!({
        "kind": "image",
        "kinds": ["image"],
        "name": chosen.name,
        "full_name": chosen.full_name,
        "on_disk": chosen.on_disk,
        "width": chosen.defaults.width,
        "height": chosen.defaults.height,
        "steps": chosen.defaults.num_steps,
        "guidance": showable(chosen.defaults.guidance_scale),
        "takes_guidance": chosen.takes_guidance,
        "sampler": chosen.sampler,
        "sizes": chosen.sizes.iter().map(|(w, h)| serde_json::json!([w, h])).collect::<Vec<_>>(),
        "alignment": chosen.alignment,
        "prompt": chosen.suggested_prompt,
        "avoid": chosen.suggested_avoid,
        "prompt_images": 0,
        "image_keys": image_keys,
        "why_not": why_not,
    }))
}

/// The machine: `waifu_machine_json`. Measured for the device `auto` would choose.
pub fn machine_json() -> serde_json::Value {
    crate::machine::describe(DeviceOption::Auto.resolve())
}

/// Which device a name means, as the C API numbers them.
pub fn device_option(device: Device) -> DeviceOption {
    match device {
        Device::Cpu => DeviceOption::Cpu,
        Device::Cuda => DeviceOption::Cuda,
        Device::CudaHost => DeviceOption::CudaCpuOffload,
        Device::Metal => DeviceOption::Metal,
        Device::Vulkan => DeviceOption::Vulkan,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::sync::mpsc::RecvTimeoutError;
    use std::time::Duration;

    fn a_request(prompt: &str, images: Vec<(&str, Image)>) -> DrawRequest {
        DrawRequest {
            prompt: prompt.to_string(),
            negative: String::new(),
            images: images.into_iter().map(|(key, image)| (key.to_string(), image)).collect(),
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
        Image { width: 1, height: 1, rgb: vec![0, 0, 0] }
    }

    #[test]
    fn a_draw_is_refused_at_the_call_for_what_no_model_could_run() {
        assert!(check_draw(&a_request("a cat", vec![])).is_ok());
        assert!(check_draw(&a_request("the girl in <|girl|>", vec![("girl", a_pixel())])).is_ok());
        assert!(check_draw(&a_request("a cat", vec![(START_FROM, a_pixel())])).is_ok());

        for (request, says) in [
            (a_request("   ", vec![]), "empty"),
            (a_request("<|girl|>", vec![]), "names no image"),
            (a_request("a cat", vec![("girl", a_pixel())]), "not named in the prompt"),
            (a_request("<|start_from|>", vec![(START_FROM, a_pixel())]), "not for the prompt"),
            (a_request("a <|endofprompt|> cat", vec![]), "names no image"),
            (a_request("a <|end of|> cat", vec![]), "own markers"),
            (a_request("a <| cat", vec![]), "does not close"),
            (a_request("<|a|> <|a|>", vec![("a", a_pixel()), ("a", a_pixel())]), "two images"),
            (a_request("<|a-b|>", vec![("a-b", a_pixel())]), "not a key"),
            (a_request("<|a|>", vec![("a", Image { width: 2, height: 2, rgb: vec![0; 3] })]), "rows of RGB"),
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
        (Box::new(|_| {}), Box::new(move |end| ended.send(end).unwrap()), heard)
    }

    fn an_engine() -> Engine {
        Engine::new(DeviceOption::Cpu).expect("an engine on the CPU")
    }

    fn a_speaking() -> SpeakRequest {
        SpeakRequest {
            text: "hello there".to_string(),
            like: None,
            speed: 1.0,
            temperature: 0.8,
            style: None,
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
        assert!(matches!(loaded.recv_timeout(Duration::from_millis(50)), Err(RecvTimeoutError::Timeout | RecvTimeoutError::Disconnected)));
        assert!(matches!(spoken.recv_timeout(Duration::from_millis(50)), Err(RecvTimeoutError::Timeout | RecvTimeoutError::Disconnected)));

        // A picture asked of a voice is the wrong kind.
        let (progress, complete, drawn) = listen::<Image>();
        engine.draw(a_request("a cat", vec![]), progress, complete).unwrap();
        assert!(matches!(drawn.recv_timeout(WAIT).unwrap(), Ended::Failed(Failure::NotLoaded, _)));
    }

    #[test]
    fn a_model_nobody_published_is_unknown() {
        let engine = an_engine();
        let (progress, complete, heard) = listen::<()>();
        engine.load("no-such-model:v9", progress, complete);
        assert!(matches!(heard.recv_timeout(WAIT).unwrap(), Ended::Failed(Failure::UnknownModel, _)));
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
    fn a_reading_reports_progress_before_it_ends() {
        let engine = an_engine();
        let (progress, complete, loaded) = listen::<()>();
        engine.load(TONES, progress, complete);
        assert_eq!(loaded.recv_timeout(WAIT).unwrap(), Ended::Done(()));

        let (told, heard_progress) = mpsc::channel();
        let (_, complete, spoken) = listen::<Sound>();
        engine
            .speak(a_speaking(), Box::new(move |progress| told.send(progress.clone()).unwrap()), complete)
            .unwrap();
        assert!(matches!(spoken.recv_timeout(WAIT).unwrap(), Ended::Done(_)));

        let reports: Vec<Progress> = heard_progress.try_iter().collect();
        assert!(!reports.is_empty());
        assert_eq!(reports[0].stage, Stage::Encoding);
        assert!(reports.iter().all(|report| !report.words.is_empty()));
    }

    #[test]
    fn what_is_published_and_what_a_model_is_come_back_as_json() {
        let catalog = catalog_json();
        let models = catalog["models"].as_array().unwrap();
        assert!(models.iter().any(|model| model["name"] == "sdxl:base" && model["kind"] == "image"));
        assert!(models.iter().any(|model| model["kind"] == "speech"));

        let sdxl = describe_json("sdxl:base").unwrap();
        assert_eq!(sdxl["kind"], "image");
        assert_eq!(sdxl["takes_guidance"], true);
        assert_eq!(sdxl["prompt_images"], 0);
        assert_eq!(sdxl["image_keys"], serde_json::json!(["start_from"]));

        let anima = describe_json("anima:turbo").unwrap();
        assert_eq!(anima["image_keys"], serde_json::json!([]));
        assert!(anima["why_not"]["start_from"].as_str().unwrap().contains("Anima"));
        assert_eq!(sdxl["alignment"], 32);
        assert_eq!(anima["alignment"], 16);

        let tones = describe_json(TONES).unwrap();
        assert_eq!(tones["kind"], "speech");

        let cosyvoice = describe_json("cosyvoice").unwrap();
        assert_eq!(cosyvoice["kinds"], serde_json::json!(["speech", "conversion"]));
        assert!(cosyvoice["conversion"]["steps"].as_i64().unwrap() > 0, "{cosyvoice}");

        assert!(describe_json("no-such-model:v9").is_none());
    }
}
