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

//! A task: one load, unload or run on a pipeline, as the caller holds it. The engine's callbacks
//! write what happens into it, on the pipeline's thread; the caller reads it back -- the events in
//! order, then the status and the result -- on whichever thread it likes. Nothing of the caller's
//! ever runs on the pipeline's thread.

#![allow(clippy::missing_safety_doc)]

use std::collections::VecDeque;
use std::ffi::{c_char, CString};
use std::ptr;
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use super::support::{c_string, caught, WaifuStage, WaifuStatusCode, WAIFU_CANCELLED, WAIFU_OK};
use super::{
    code_of, stage_out, WaifuAudio, WaifuImage, WaifuLayerType, WaifuPipelineType, WaifuRgbaImage,
    WAIFU_LAYER_DEPTH, WAIFU_LAYER_PART, WAIFU_LAYER_STRATIFIED_FAR, WAIFU_LAYER_STRATIFIED_NEAR,
    WAIFU_PIPELINE_IMAGE_GENERATION, WAIFU_PIPELINE_LAYER_DECOMPOSITION,
    WAIFU_PIPELINE_SPEECH_TO_TEXT, WAIFU_PIPELINE_STREAMING_TEXT_TO_SPEECH,
    WAIFU_PIPELINE_TEXT_TO_SPEECH, WAIFU_PIPELINE_VOICE_CONVERSION,
};
use crate::engine::{self, Canceller, Ended, JobId, LayerKind, Progress};
use crate::speech::Viseme;
use crate::wav::Sound;
use crate::Transcript;

/// What waifu_task_next_event says happened. A task is sent only the types its request asked for
/// or that were there in 6.0, so a caller is never handed one it does not know.
pub type WaifuEventType = i32;
/// Nothing before the timeout.
pub const WAIFU_EVENT_NONE: WaifuEventType = 0;
/// The PROGRESS_ attributes say where the task is now.
pub const WAIFU_EVENT_PROGRESS: WaifuEventType = 1;
/// Streaming text to speech: STREAMING_TEXT_TO_SPEECH_AUDIO_PART is the next part.
pub const WAIFU_EVENT_AUDIO_PART: WaifuEventType = 2;
/// STATUS is there; and the result, for WAIFU_OK.
pub const WAIFU_EVENT_END: WaifuEventType = 3;

/// waifu_task_next_event and waifu_task_wait: no timeout.
pub const WAIFU_WAIT_FOREVER: u32 = u32::MAX;

/// What a task can be asked. The type after each is the getter that reads it; "indexed" ones are
/// read with the _at getters, 0 .. LAYER_DECOMPOSITION_LAYER_COUNT - 1.
pub type WaifuTaskAttribute = i32;
/// Int, a WaifuStatusCode; absent until the task has ended.
pub const WAIFU_TASK_STATUS: WaifuTaskAttribute = 1;
/// String, why; absent for WAIFU_OK and WAIFU_CANCELLED.
pub const WAIFU_TASK_STATUS_MESSAGE: WaifuTaskAttribute = 2;

/// Int, a WaifuStage: the progress the last PROGRESS event said; absent before one.
pub const WAIFU_TASK_PROGRESS_STAGE: WaifuTaskAttribute = 10;
/// Int, of the stage, done already.
pub const WAIFU_TASK_PROGRESS_DONE: WaifuTaskAttribute = 11;
/// Int.
pub const WAIFU_TASK_PROGRESS_TOTAL: WaifuTaskAttribute = 12;
/// Float, 0..1 of the whole task; absent where nothing can say.
pub const WAIFU_TASK_PROGRESS_FRACTION: WaifuTaskAttribute = 13;
/// Float, seconds since the task started.
pub const WAIFU_TASK_PROGRESS_SECONDS: WaifuTaskAttribute = 14;

/// Image: there once STATUS is WAIFU_OK, as each task's result is.
pub const WAIFU_TASK_IMAGE_GENERATION_IMAGE: WaifuTaskAttribute = 100;

/// Audio.
pub const WAIFU_TASK_TEXT_TO_SPEECH_AUDIO: WaifuTaskAttribute = 200;
/// Int array, read with waifu_task_get_int_array_attribute: a WaifuViseme for each
/// TEXT_TO_SPEECH_VISEME_FRAME_MS of AUDIO, frame i being milliseconds [i, i + 1) * FRAME_MS of it,
/// ceil(milliseconds / FRAME_MS) of them; for a request with TEXT_TO_SPEECH_VISEMES. Since 6.1.
pub const WAIFU_TASK_TEXT_TO_SPEECH_VISEMES: WaifuTaskAttribute = 201;
/// Int: how many milliseconds a frame of VISEMES holds -- 40 for CosyVoice3, a speech token's.
/// Since 6.1.
pub const WAIFU_TASK_TEXT_TO_SPEECH_VISEME_FRAME_MS: WaifuTaskAttribute = 202;

/// Audio.
pub const WAIFU_TASK_VOICE_CONVERSION_AUDIO: WaifuTaskAttribute = 300;

/// Int, the side of the square canvas.
pub const WAIFU_TASK_LAYER_DECOMPOSITION_CANVAS: WaifuTaskAttribute = 400;
/// Int. The layers are in one order, which the library never changes: each part, bottom first --
/// which is the stack, unstratified -- then, with STRATIFY_LAYERS, each part's halves, near before
/// far; then a depth layer for each of those, in their order. The stratified stack is every part
/// that was not split and every half, bottom first: from the largest depth to the smallest; at one
/// depth, a far half under a near one; then by their order. No depth layer is in either stack.
pub const WAIFU_TASK_LAYER_DECOMPOSITION_LAYER_COUNT: WaifuTaskAttribute = 401;
/// Int, indexed: a WaifuLayerType.
pub const WAIFU_TASK_LAYER_DECOMPOSITION_LAYER_TYPE: WaifuTaskAttribute = 402;
/// Int, indexed: unique within the result, from 1; it means nothing else.
pub const WAIFU_TASK_LAYER_DECOMPOSITION_LAYER_ID: WaifuTaskAttribute = 403;
/// String, indexed: "topwear"; its halves "topwear-0" and "topwear-1"; its depth "topwear_depth".
pub const WAIFU_TASK_LAYER_DECOMPOSITION_LAYER_NAME: WaifuTaskAttribute = 404;
/// Int, indexed: where its top left corner is on the canvas.
pub const WAIFU_TASK_LAYER_DECOMPOSITION_LAYER_LEFT: WaifuTaskAttribute = 405;
/// Int, indexed.
pub const WAIFU_TASK_LAYER_DECOMPOSITION_LAYER_TOP: WaifuTaskAttribute = 406;
/// RGBA image, indexed: only as large as what is drawn. A depth layer: the same size and place as
/// its layer, R, G and B the depth, 0 nearest to 255 farthest, under its layer's alpha.
pub const WAIFU_TASK_LAYER_DECOMPOSITION_LAYER_IMAGE: WaifuTaskAttribute = 407;
/// Float, indexed: 0 nearest, 1 farthest. A near half has its part's depth, not its pixels': that
/// is what keeps it in its part's place in the stratified stack. A depth layer has its layer's.
pub const WAIFU_TASK_LAYER_DECOMPOSITION_LAYER_DEPTH: WaifuTaskAttribute = 408;
/// Int, indexed: the id of the layer this one is made from -- a half's part, a depth layer's
/// layer; 0 for a part.
pub const WAIFU_TASK_LAYER_DECOMPOSITION_LAYER_DERIVED_FROM_ID: WaifuTaskAttribute = 409;

/// String: what was said; "" for nothing said.
pub const WAIFU_TASK_SPEECH_TO_TEXT_TEXT: WaifuTaskAttribute = 500;
/// String: the language it was said in, as the model names it -- "Chinese" -- or several, comma
/// separated, for a recording that changes; "" where none was heard.
pub const WAIFU_TASK_SPEECH_TO_TEXT_LANGUAGE: WaifuTaskAttribute = 501;

/// Audio: the part the last AUDIO_PART event said, from the moment it is said, before the task
/// ends.
pub const WAIFU_TASK_STREAMING_TEXT_TO_SPEECH_AUDIO_PART: WaifuTaskAttribute = 600;
/// Audio: every part, end to end.
pub const WAIFU_TASK_STREAMING_TEXT_TO_SPEECH_AUDIO: WaifuTaskAttribute = 601;
/// Int array, as TEXT_TO_SPEECH_VISEMES, frame 0 the start of the first part: the frames settled
/// by the part the last AUDIO_PART event said, from the moment it is said -- never past the end of
/// that part, and only ever more of them with each part -- and every frame once the task has ended
/// well. For a request with STREAMING_TEXT_TO_SPEECH_VISEMES. Since 6.1.
pub const WAIFU_TASK_STREAMING_TEXT_TO_SPEECH_VISEMES: WaifuTaskAttribute = 602;
/// Int, as TEXT_TO_SPEECH_VISEME_FRAME_MS. Since 6.1.
pub const WAIFU_TASK_STREAMING_TEXT_TO_SPEECH_VISEME_FRAME_MS: WaifuTaskAttribute = 603;

/// A progress report, as the PROGRESS_ attributes read it.
#[derive(Clone, Copy, Debug)]
struct Report {
    stage: WaifuStage,
    done: u64,
    total: u64,
    fraction: Option<f64>,
    seconds: f64,
}

/// An event not yet read.
enum Pending {
    Progress(Report),
    /// The index of the part in `State::parts`.
    Part(usize),
}

/// A layer as the getters read it, its name kept NUL terminated.
struct Layer {
    r#type: WaifuLayerType,
    id: u32,
    name: CString,
    left: i32,
    top: i32,
    image: engine::RgbaImage,
    depth: f32,
    derived_from_id: u32,
}

/// What a task made.
enum Output {
    Nothing,
    Image(engine::Image),
    Audio(Sound),
    Layers { canvas: u32, layers: Vec<Layer> },
    Transcript { text: CString, language: CString },
}

/// How a task ended. Once set it is never changed, so pointers into it live as long as the task.
struct End {
    code: WaifuStatusCode,
    message: Option<CString>,
    output: Output,
}

#[derive(Default)]
struct State {
    queue: VecDeque<Pending>,
    /// The progress the last PROGRESS event read said.
    progress: Option<Report>,
    /// Every part handed over, in order, with the visemes settled by then: each lives as long as
    /// the task, the visemes a copy of their own so that a pointer into them does too.
    parts: Vec<(Sound, Vec<i32>)>,
    /// Every viseme handed over so far, as WaifuVisemes; read through once the task has ended,
    /// when nothing moves it any more.
    visemes: Vec<i32>,
    /// What a frame of them holds, once the first has come.
    viseme_frame_ms: Option<u32>,
    /// The part the last AUDIO_PART event read said.
    part: Option<usize>,
    end: Option<End>,
}

/// What the engine's callbacks and the caller both hold.
#[derive(Default)]
struct Shared {
    state: Mutex<State>,
    changed: Condvar,
}

impl Shared {
    fn lock(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(|held| held.into_inner())
    }

    fn push(&self, event: Pending) {
        let mut state = self.lock();
        // Progress not yet read is the latest: a report behind another report is let go of.
        if let (Pending::Progress(_), Some(Pending::Progress(_))) = (&event, state.queue.back()) {
            state.queue.pop_back();
        }
        state.queue.push_back(event);
        self.changed.notify_all();
    }

    fn end<T>(&self, ended: Ended<T>, output: impl FnOnce(T) -> Output) {
        let end = match ended {
            Ended::Done(made) => End {
                code: WAIFU_OK,
                message: None,
                output: output(made),
            },
            Ended::Cancelled => End {
                code: WAIFU_CANCELLED,
                message: None,
                output: Output::Nothing,
            },
            Ended::Failed(failure, message) => End {
                code: code_of(failure),
                message: Some(c_string(&message)),
                output: Output::Nothing,
            },
        };
        let mut state = self.lock();
        if state.end.is_none() {
            state.end = Some(end);
        }
        self.changed.notify_all();
    }
}

/// One load, unload or run on a pipeline, as the caller holds it: its events are read with
/// waifu_task_next_event, and its status and result with the getters, on any thread.
pub struct WaifuTask {
    shared: Arc<Shared>,
    id: JobId,
    canceller: Canceller,
    /// The pipeline type of a run, whose result it says; None for a load or an unload.
    run: Option<WaifuPipelineType>,
}

/// The callbacks a task is started with, and the task, once the engine has given it an id.
pub(crate) struct Started {
    shared: Arc<Shared>,
}

impl Started {
    pub fn new() -> Started {
        Started {
            shared: Arc::default(),
        }
    }

    pub fn on_progress(&self) -> engine::OnProgress {
        let shared = Arc::clone(&self.shared);
        Box::new(move |progress: &Progress| {
            shared.push(Pending::Progress(Report {
                stage: stage_out(progress.stage),
                done: progress.done,
                total: progress.total,
                fraction: progress.fraction,
                seconds: progress.seconds,
            }))
        })
    }

    pub fn on_part(&self) -> engine::OnAudio {
        let shared = Arc::clone(&self.shared);
        Box::new(move |part: &Sound| {
            let mut state = shared.lock();
            let settled = state.visemes.clone();
            state.parts.push((part.clone(), settled));
            let index = state.parts.len() - 1;
            drop(state);
            shared.push(Pending::Part(index));
        })
    }

    /// The visemes, as they come: before the part they lie in, for a streaming reading.
    pub fn on_visemes(&self) -> engine::OnVisemes {
        let shared = Arc::clone(&self.shared);
        Box::new(move |frame_ms: u32, said: &[Viseme]| {
            let mut state = shared.lock();
            state.viseme_frame_ms = Some(frame_ms);
            state
                .visemes
                .extend(said.iter().map(|viseme| *viseme as i32));
        })
    }

    pub fn on_loaded(&self) -> engine::OnComplete<()> {
        let shared = Arc::clone(&self.shared);
        Box::new(move |ended| shared.end(ended, |()| Output::Nothing))
    }

    pub fn on_image(&self) -> engine::OnComplete<engine::Image> {
        let shared = Arc::clone(&self.shared);
        Box::new(move |ended| shared.end(ended, Output::Image))
    }

    pub fn on_audio(&self) -> engine::OnComplete<Sound> {
        let shared = Arc::clone(&self.shared);
        Box::new(move |ended| shared.end(ended, Output::Audio))
    }

    pub fn on_layers(&self) -> engine::OnComplete<engine::Layers> {
        let shared = Arc::clone(&self.shared);
        Box::new(move |ended| {
            shared.end(ended, |made| Output::Layers {
                canvas: made.canvas,
                layers: made.layers.into_iter().map(layer_out).collect(),
            })
        })
    }

    pub fn on_transcript(&self) -> engine::OnComplete<Transcript> {
        let shared = Arc::clone(&self.shared);
        Box::new(move |ended| {
            shared.end(ended, |made: Transcript| Output::Transcript {
                text: c_string(&made.text),
                language: c_string(&made.language),
            })
        })
    }

    /// The task the engine numbered `id`.
    pub fn task(
        self,
        id: JobId,
        canceller: Canceller,
        run: Option<WaifuPipelineType>,
    ) -> *mut WaifuTask {
        Box::into_raw(Box::new(WaifuTask {
            shared: self.shared,
            id,
            canceller,
            run,
        }))
    }
}

fn layer_out(layer: engine::LayerOut) -> Layer {
    Layer {
        r#type: match layer.kind {
            LayerKind::Part => WAIFU_LAYER_PART,
            LayerKind::StratifiedNear => WAIFU_LAYER_STRATIFIED_NEAR,
            LayerKind::StratifiedFar => WAIFU_LAYER_STRATIFIED_FAR,
            LayerKind::Depth => WAIFU_LAYER_DEPTH,
        },
        id: layer.id,
        name: c_string(&layer.name),
        left: layer.left,
        top: layer.top,
        image: layer.image,
        depth: layer.depth,
        derived_from_id: layer.derived_from_id.unwrap_or(0),
    }
}

/// Waits on `shared` until `done` says so or `timeout_ms` has gone by: the state, and whether
/// `done` said so.
fn wait_until<'a>(
    shared: &'a Shared,
    timeout_ms: u32,
    mut done: impl FnMut(&mut State) -> bool,
) -> (MutexGuard<'a, State>, bool) {
    let deadline = (timeout_ms != WAIFU_WAIT_FOREVER)
        .then(|| Instant::now() + Duration::from_millis(timeout_ms as u64));
    let mut state = shared.lock();
    loop {
        if done(&mut state) {
            return (state, true);
        }
        state = match deadline {
            None => shared
                .changed
                .wait(state)
                .unwrap_or_else(|held| held.into_inner()),
            Some(deadline) => {
                let left = deadline.saturating_duration_since(Instant::now());
                if left.is_zero() {
                    return (state, false);
                }
                shared
                    .changed
                    .wait_timeout(state, left)
                    .unwrap_or_else(|held| held.into_inner())
                    .0
            }
        };
    }
}

/// The next thing that happened to the task, in order, waiting up to timeout_ms for it: 0 only
/// looks. NONE where nothing did in time. END once it has ended, and END for every call after.
/// Progress not yet read is one event, the latest; audio parts are never merged or dropped. One
/// thread at a time asks a task for events.
#[no_mangle]
pub unsafe extern "C" fn waifu_task_next_event(
    task: *mut WaifuTask,
    timeout_ms: u32,
) -> WaifuEventType {
    let Some(task) = task.as_ref() else {
        return WAIFU_EVENT_NONE;
    };
    caught(WAIFU_EVENT_NONE, || {
        let mut event = WAIFU_EVENT_NONE;
        let _ = wait_until(&task.shared, timeout_ms, |state| {
            match state.queue.pop_front() {
                Some(Pending::Progress(report)) => {
                    state.progress = Some(report);
                    event = WAIFU_EVENT_PROGRESS;
                }
                Some(Pending::Part(index)) => {
                    state.part = Some(index);
                    event = WAIFU_EVENT_AUDIO_PART;
                }
                None if state.end.is_some() => event = WAIFU_EVENT_END,
                None => return false,
            }
            true
        });
        event
    })
}

/// Waits up to timeout_ms for the task to end, without reading its events: true once it has. Any
/// number of threads may wait on one task.
#[no_mangle]
pub unsafe extern "C" fn waifu_task_wait(task: *mut WaifuTask, timeout_ms: u32) -> bool {
    let Some(task) = task.as_ref() else {
        return false;
    };
    caught(false, || {
        wait_until(&task.shared, timeout_ms, |state| state.end.is_some()).1
    })
}

/// After the step it is on, or before it starts; it then ends with WAIFU_CANCELLED, waking anyone
/// waiting. Nothing once it has ended.
#[no_mangle]
pub unsafe extern "C" fn waifu_task_cancel(task: *mut WaifuTask) {
    if let Some(task) = task.as_ref() {
        caught((), || task.canceller.cancel(task.id));
    }
}

/// A task that has not ended is cancelled, and the library lets it go when it does; this returns
/// at once. Every pointer the task handed out is then gone. NULL is fine.
#[no_mangle]
pub unsafe extern "C" fn waifu_task_free(task: *mut WaifuTask) {
    if task.is_null() {
        return;
    }
    caught((), || {
        let task = Box::from_raw(task);
        if task.shared.lock().end.is_none() {
            task.canceller.cancel(task.id);
        }
    })
}

/// What a getter found: a value, or nothing there.
enum Found {
    Int(i64),
    Float(f64),
    Text(*const c_char),
    Image(WaifuImage),
    Rgba(WaifuRgbaImage),
    Audio(WaifuAudio),
    Ints(*const i32, usize),
}

fn ints_found(values: &[i32]) -> Found {
    Found::Ints(values.as_ptr(), values.len())
}

fn audio_found(sound: &Sound) -> Found {
    Found::Audio(WaifuAudio {
        samples: sound.samples.as_ptr(),
        count: sound.samples.len(),
        sample_rate: sound.rate,
    })
}

fn rgba_found(image: &engine::RgbaImage) -> Found {
    Found::Rgba(WaifuRgbaImage {
        width: image.width,
        height: image.height,
        rgba: image.rgba.as_ptr(),
        len: image.rgba.len(),
    })
}

impl WaifuTask {
    /// What `attribute` is now, or None; `index` for the indexed ones, None for the rest.
    fn find(&self, attribute: WaifuTaskAttribute, index: Option<usize>) -> Option<Found> {
        let state = self.shared.lock();
        let indexed = (WAIFU_TASK_LAYER_DECOMPOSITION_LAYER_TYPE
            ..=WAIFU_TASK_LAYER_DECOMPOSITION_LAYER_DERIVED_FROM_ID)
            .contains(&attribute);
        if indexed != index.is_some() {
            return None;
        }

        let report = state.progress;
        match attribute {
            WAIFU_TASK_STATUS => return state.end.as_ref().map(|end| Found::Int(end.code as i64)),
            WAIFU_TASK_STATUS_MESSAGE => {
                return state
                    .end
                    .as_ref()?
                    .message
                    .as_ref()
                    .map(|message| Found::Text(message.as_ptr()))
            }
            WAIFU_TASK_PROGRESS_STAGE => return report.map(|r| Found::Int(r.stage as i64)),
            WAIFU_TASK_PROGRESS_DONE => return report.map(|r| Found::Int(r.done as i64)),
            WAIFU_TASK_PROGRESS_TOTAL => return report.map(|r| Found::Int(r.total as i64)),
            WAIFU_TASK_PROGRESS_FRACTION => return report?.fraction.map(Found::Float),
            WAIFU_TASK_PROGRESS_SECONDS => return report.map(|r| Found::Float(r.seconds)),
            WAIFU_TASK_STREAMING_TEXT_TO_SPEECH_AUDIO_PART => {
                if self.run != Some(WAIFU_PIPELINE_STREAMING_TEXT_TO_SPEECH) {
                    return None;
                }
                return state.part.map(|index| audio_found(&state.parts[index].0));
            }
            WAIFU_TASK_TEXT_TO_SPEECH_VISEME_FRAME_MS
            | WAIFU_TASK_STREAMING_TEXT_TO_SPEECH_VISEME_FRAME_MS => {
                let streaming = attribute == WAIFU_TASK_STREAMING_TEXT_TO_SPEECH_VISEME_FRAME_MS;
                let run = match streaming {
                    true => WAIFU_PIPELINE_STREAMING_TEXT_TO_SPEECH,
                    false => WAIFU_PIPELINE_TEXT_TO_SPEECH,
                };
                if self.run != Some(run) {
                    return None;
                }
                return state.viseme_frame_ms.map(|ms| Found::Int(ms as i64));
            }
            WAIFU_TASK_STREAMING_TEXT_TO_SPEECH_VISEMES => {
                if self.run != Some(WAIFU_PIPELINE_STREAMING_TEXT_TO_SPEECH) {
                    return None;
                }
                state.viseme_frame_ms?;
                if state.end.as_ref().is_some_and(|end| end.code == WAIFU_OK) {
                    return Some(ints_found(&state.visemes));
                }
                return state.part.map(|index| ints_found(&state.parts[index].1));
            }
            _ => {}
        }

        // Each task's result, once it has ended well: the attribute has to be this run's.
        let end = state.end.as_ref().filter(|end| end.code == WAIFU_OK)?;
        let run = self.run?;
        match (run, attribute, &end.output) {
            (
                WAIFU_PIPELINE_IMAGE_GENERATION,
                WAIFU_TASK_IMAGE_GENERATION_IMAGE,
                Output::Image(image),
            ) => Some(Found::Image(WaifuImage {
                width: image.width,
                height: image.height,
                rgb: image.rgb.as_ptr(),
                len: image.rgb.len(),
            })),
            (
                WAIFU_PIPELINE_TEXT_TO_SPEECH,
                WAIFU_TASK_TEXT_TO_SPEECH_AUDIO,
                Output::Audio(sound),
            )
            | (
                WAIFU_PIPELINE_VOICE_CONVERSION,
                WAIFU_TASK_VOICE_CONVERSION_AUDIO,
                Output::Audio(sound),
            )
            | (
                WAIFU_PIPELINE_STREAMING_TEXT_TO_SPEECH,
                WAIFU_TASK_STREAMING_TEXT_TO_SPEECH_AUDIO,
                Output::Audio(sound),
            ) => Some(audio_found(sound)),
            (
                WAIFU_PIPELINE_TEXT_TO_SPEECH,
                WAIFU_TASK_TEXT_TO_SPEECH_VISEMES,
                Output::Audio(_),
            ) => {
                state.viseme_frame_ms?;
                Some(ints_found(&state.visemes))
            }
            (
                WAIFU_PIPELINE_SPEECH_TO_TEXT,
                WAIFU_TASK_SPEECH_TO_TEXT_TEXT,
                Output::Transcript { text, .. },
            ) => Some(Found::Text(text.as_ptr())),
            (
                WAIFU_PIPELINE_SPEECH_TO_TEXT,
                WAIFU_TASK_SPEECH_TO_TEXT_LANGUAGE,
                Output::Transcript { language, .. },
            ) => Some(Found::Text(language.as_ptr())),
            (WAIFU_PIPELINE_LAYER_DECOMPOSITION, _, Output::Layers { canvas, layers }) => {
                match (attribute, index) {
                    (WAIFU_TASK_LAYER_DECOMPOSITION_CANVAS, None) => {
                        Some(Found::Int(*canvas as i64))
                    }
                    (WAIFU_TASK_LAYER_DECOMPOSITION_LAYER_COUNT, None) => {
                        Some(Found::Int(layers.len() as i64))
                    }
                    (_, Some(index)) => {
                        let layer = layers.get(index)?;
                        match attribute {
                            WAIFU_TASK_LAYER_DECOMPOSITION_LAYER_TYPE => {
                                Some(Found::Int(layer.r#type as i64))
                            }
                            WAIFU_TASK_LAYER_DECOMPOSITION_LAYER_ID => {
                                Some(Found::Int(layer.id as i64))
                            }
                            WAIFU_TASK_LAYER_DECOMPOSITION_LAYER_NAME => {
                                Some(Found::Text(layer.name.as_ptr()))
                            }
                            WAIFU_TASK_LAYER_DECOMPOSITION_LAYER_LEFT => {
                                Some(Found::Int(layer.left as i64))
                            }
                            WAIFU_TASK_LAYER_DECOMPOSITION_LAYER_TOP => {
                                Some(Found::Int(layer.top as i64))
                            }
                            WAIFU_TASK_LAYER_DECOMPOSITION_LAYER_IMAGE => {
                                Some(rgba_found(&layer.image))
                            }
                            WAIFU_TASK_LAYER_DECOMPOSITION_LAYER_DEPTH => {
                                Some(Found::Float(layer.depth as f64))
                            }
                            WAIFU_TASK_LAYER_DECOMPOSITION_LAYER_DERIVED_FROM_ID => {
                                Some(Found::Int(layer.derived_from_id as i64))
                            }
                            _ => None,
                        }
                    }
                    _ => None,
                }
            }
            _ => None,
        }
    }
}

/// Runs a getter: what it found of the type it reads, or `otherwise`.
unsafe fn getter<T>(
    task: *const WaifuTask,
    attribute: WaifuTaskAttribute,
    index: Option<usize>,
    otherwise: T,
    read: impl FnOnce(Found) -> Option<T>,
) -> T {
    let Some(task) = task.as_ref() else {
        return otherwise;
    };
    caught(None, || task.find(attribute, index).and_then(read)).unwrap_or(otherwise)
}

/// Writes `found` to `out`, where there is somewhere to write it.
unsafe fn put<T>(out: *mut T, found: Option<T>) -> bool {
    match (out.is_null(), found) {
        (false, Some(found)) => {
            *out = found;
            true
        }
        _ => false,
    }
}

/// false, or NULL, where the attribute is not there -- not yet, of another pipeline type, or not
/// known to this library -- and *value untouched; nothing is set. A pointer, and every pointer in
/// what an image or audio getter fills in, is valid until waifu_task_free.
#[no_mangle]
pub unsafe extern "C" fn waifu_task_get_int_attribute(
    task: *const WaifuTask,
    attribute: WaifuTaskAttribute,
    value: *mut i64,
) -> bool {
    let found = getter(task, attribute, None, None, |found| match found {
        Found::Int(found) => Some(Some(found)),
        _ => None,
    });
    put(value, found)
}

/// As waifu_task_get_int_attribute.
#[no_mangle]
pub unsafe extern "C" fn waifu_task_get_float_attribute(
    task: *const WaifuTask,
    attribute: WaifuTaskAttribute,
    value: *mut f64,
) -> bool {
    let found = getter(task, attribute, None, None, |found| match found {
        Found::Float(found) => Some(Some(found)),
        _ => None,
    });
    put(value, found)
}

/// As waifu_task_get_int_attribute.
#[no_mangle]
pub unsafe extern "C" fn waifu_task_get_string_attribute(
    task: *const WaifuTask,
    attribute: WaifuTaskAttribute,
) -> *const c_char {
    getter(task, attribute, None, ptr::null(), |found| match found {
        Found::Text(found) => Some(found),
        _ => None,
    })
}

/// As waifu_task_get_int_attribute.
#[no_mangle]
pub unsafe extern "C" fn waifu_task_get_image_attribute(
    task: *const WaifuTask,
    attribute: WaifuTaskAttribute,
    value: *mut WaifuImage,
) -> bool {
    let found = getter(task, attribute, None, None, |found| match found {
        Found::Image(found) => Some(Some(found)),
        _ => None,
    });
    put(value, found)
}

/// As waifu_task_get_int_attribute.
#[no_mangle]
pub unsafe extern "C" fn waifu_task_get_rgba_image_attribute(
    task: *const WaifuTask,
    attribute: WaifuTaskAttribute,
    value: *mut WaifuRgbaImage,
) -> bool {
    let found = getter(task, attribute, None, None, |found| match found {
        Found::Rgba(found) => Some(Some(found)),
        _ => None,
    });
    put(value, found)
}

/// As waifu_task_get_int_attribute.
#[no_mangle]
pub unsafe extern "C" fn waifu_task_get_audio_attribute(
    task: *const WaifuTask,
    attribute: WaifuTaskAttribute,
    value: *mut WaifuAudio,
) -> bool {
    let found = getter(task, attribute, None, None, |found| match found {
        Found::Audio(found) => Some(Some(found)),
        _ => None,
    });
    put(value, found)
}

/// The arrays: a pointer to *count values, valid until waifu_task_free, or NULL and *count
/// untouched where the attribute is not there, as waifu_task_get_int_attribute. Since 6.1.
#[no_mangle]
pub unsafe extern "C" fn waifu_task_get_int_array_attribute(
    task: *const WaifuTask,
    attribute: WaifuTaskAttribute,
    count: *mut usize,
) -> *const i32 {
    let found = getter(task, attribute, None, None, |found| match found {
        Found::Ints(values, len) => Some(Some((values, len))),
        _ => None,
    });
    match (count.is_null(), found) {
        (false, Some((values, len))) => {
            *count = len;
            // An empty array is there, and is not NULL.
            match values.is_null() {
                true => ptr::NonNull::<i32>::dangling().as_ptr(),
                false => values,
            }
        }
        _ => ptr::null(),
    }
}

/// The indexed attributes: the layers of a decomposition. false, or NULL, past the count as well.
#[no_mangle]
pub unsafe extern "C" fn waifu_task_get_int_attribute_at(
    task: *const WaifuTask,
    attribute: WaifuTaskAttribute,
    index: usize,
    value: *mut i64,
) -> bool {
    let found = getter(task, attribute, Some(index), None, |found| match found {
        Found::Int(found) => Some(Some(found)),
        _ => None,
    });
    put(value, found)
}

/// As waifu_task_get_int_attribute_at.
#[no_mangle]
pub unsafe extern "C" fn waifu_task_get_float_attribute_at(
    task: *const WaifuTask,
    attribute: WaifuTaskAttribute,
    index: usize,
    value: *mut f64,
) -> bool {
    let found = getter(task, attribute, Some(index), None, |found| match found {
        Found::Float(found) => Some(Some(found)),
        _ => None,
    });
    put(value, found)
}

/// As waifu_task_get_int_attribute_at.
#[no_mangle]
pub unsafe extern "C" fn waifu_task_get_string_attribute_at(
    task: *const WaifuTask,
    attribute: WaifuTaskAttribute,
    index: usize,
) -> *const c_char {
    getter(
        task,
        attribute,
        Some(index),
        ptr::null(),
        |found| match found {
            Found::Text(found) => Some(found),
            _ => None,
        },
    )
}

/// As waifu_task_get_int_attribute_at.
#[no_mangle]
pub unsafe extern "C" fn waifu_task_get_rgba_image_attribute_at(
    task: *const WaifuTask,
    attribute: WaifuTaskAttribute,
    index: usize,
    value: *mut WaifuRgbaImage,
) -> bool {
    let found = getter(task, attribute, Some(index), None, |found| match found {
        Found::Rgba(found) => Some(Some(found)),
        _ => None,
    });
    put(value, found)
}
