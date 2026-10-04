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

//! What the program is doing, and the one lock the threads meet over.
//!
//! The model sits on a thread of its own -- a tensor never leaves the thread that made it -- and
//! the requests arrive on whichever thread the server handed them to. What moves between them is
//! a value under a lock: the worker writes what it is doing as it does it, and a request reads
//! what it needs. The lock is held for the length of a field copy and never across anything that
//! draws, which is the only rule this arrangement has.
//!
//! There is one of everything here. Nothing knows who is asking: a job is found by its id, and
//! what a request is answered with is the same whoever sent it. The jobs themselves, and what they
//! made, are kept by the [`Store`]; this is what only the running program knows -- which job is
//! running, how far along it is, and which are waiting their turn.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, MutexGuard};
use std::time::{Duration, Instant};

use serde_json::{json, Value};

use crate::cli::args::Runtime;
use crate::cli::task::Task;
use crate::cli::webui::store::{Cancelled, Job, Kind, Refused, Status, Store};
use crate::{GenerationDefaults, GenerationProgress, SpeechDefaults, SpeechProgress, Style};

/// How much of a run the parts that are not steps are worth, when a bar is drawn from them.
///
/// Weighed rather than counted. Reading the prompt costs about a step and turning the finished
/// latent into pixels costs several, so a bar drawn from the steps alone would fill up and then
/// sit there for the slowest part of the run.
const ENCODING: f64 = 1.0;
const DECODING: f64 = 3.0;

/// And the same weighing for a run that speaks rather than draws.
///
/// Reading the text is a tokenizer and is quick. The vocoder at the end is a pass over the whole
/// utterance and is not: it is the part of a speech run that a bar drawn from the tokens alone
/// would sit in front of, saying nothing, for a second or two.
const TEXT: f64 = 1.0;
const VOCODER: f64 = 4.0;

/// The model the page is set to draw with, as the screen describes it.
///
/// Chosen is not loaded. Picking one is a click and reading one is minutes, so what is here is
/// what could be known when it was picked -- guessed from the name, for a model that may not
/// even be on the disk yet -- and it is described again out of the package the first time a run
/// needs the weights. Everything on the screen that is about the model reads this: the boxes it
/// fills in, and whether image to image is offered at all.
#[derive(Clone)]
pub struct Chosen {
    /// What to ask for when the time comes to read it: the catalogue name, or the path that was
    /// given. Not for showing -- a path does not fit on the screen -- but it is what a run is
    /// posted with, and what the worker hands to the hub.
    pub name: String,
    /// What it is called on screen: the catalogue's full name, or the file name for a path.
    pub full_name: String,
    /// Whether every package of it is already fetched. A model that is not is still a model that
    /// can be chosen; the download happens at the first run, where the bar can say so.
    pub on_disk: bool,
    /// Whether its weights are read and on the device. False until the first run needs them.
    pub in_memory: bool,
    /// What the boxes start at. Guessed from the name until the package is read, and read out of
    /// the package after that -- a distilled release answers both of these differently.
    pub defaults: GenerationDefaults,
    /// The sizes to offer: the model's own where its manifest names enough of them.
    pub sizes: Vec<(i32, i32)>,
    /// What walks the noise back, named. One kind of model has one of these here, so it is
    /// something the screen reports rather than something it offers -- but a picture that came
    /// out unlike another tool's is a question about this first, and a screen that does not say
    /// is a screen that cannot be asked.
    pub sampler: &'static str,
    /// What the package's own card suggests be typed in the boxes. Known once it has been read,
    /// which is why the boxes take it whenever it arrives rather than only when a page opens.
    pub suggested_prompt: Option<String>,
    pub suggested_avoid: Option<String>,
    /// Why this one cannot be handed a picture to start from, or None where it can. A page that
    /// offers what the model will refuse is a refusal someone finds out about after waiting, and
    /// one that is simply greyed out is a screen that cannot be asked why.
    pub no_picture_because: Option<String>,
    /// Whether to offer guidance and a negative prompt at all. False for a distilled release,
    /// which has no second pass for either to reach. Guessed from the name like the rest of this
    /// and answered by the package once it has been read.
    pub takes_guidance: bool,
}

/// The voice the page is set to speak with, as the screen describes it.
///
/// What [`Chosen`] is for a picture model, and for the same reasons: it is what could be known
/// before any weights were read, and it is described again out of the voice itself once one is
/// loaded. There is one of these even before anything has been asked for, because the boxes on
/// the speech tab are a voice's numbers and have to start somewhere.
#[derive(Clone)]
pub struct Spoken {
    /// What to ask for when the time comes to read it. It travels with a run the way a model's
    /// name does, so that a run is of the voice that was chosen when the button was pressed.
    pub name: String,
    /// What it is called on screen.
    pub full_name: String,
    /// Whether the package is already here, which is the difference between a first reading that
    /// takes seconds and one that starts with a download of several gigabytes.
    pub on_disk: bool,
    /// Whether it is read and on the device. The stand-in, `tones`, holds nothing, so for it this
    /// is true from the first run rather than after a fetch.
    pub in_memory: bool,
    /// What the boxes start at, which is the voice's to say.
    pub defaults: SpeechDefaults,
    /// The rate it writes at, which is a property of the voice and not a setting. On the screen
    /// because a clip that came out at the wrong pitch is a question about this first.
    pub rate: u32,
    /// Why it cannot be handed a recording to sound like, or None where it can.
    pub no_likeness_because: Option<String>,
    /// The ways of saying something it was taught, for the page to list. Empty where it takes no
    /// style, which is where the page offers none.
    pub styles: &'static [Style],
    /// Why what comes out is not speech, for as long as that is true.
    ///
    /// The one sentence this whole tab is built around saying. It comes out of the voice itself
    /// rather than being written on the page, so the day a real model is what is loaded the
    /// warning goes away on its own rather than by somebody remembering to delete it.
    pub not_a_voice_because: Option<String>,
}

/// Which package of a model is being fetched, and how far into it.
#[derive(Clone)]
pub struct Fetch {
    pub model: String,
    /// Which hub it is coming from, once the fetch has settled that. Unknown for the moment
    /// before: it is decided by a probe, and a screen that guessed would be naming a hub while
    /// the probe is still out.
    pub hub: Option<&'static str>,
    pub file: String,
    pub done: u64,
    pub total: Option<u64>,
    pub part: usize,
    /// How many packages there are in all, or zero while that is still unknown.
    pub parts: usize,
}

/// A run in flight.
#[derive(Clone)]
pub struct Run {
    pub progress: GenerationProgress,
    /// How many steps this run was asked for, which the bar needs after the last one.
    pub steps: i32,
    pub started: Instant,
}

/// A reading in flight.
#[derive(Clone)]
pub struct Say {
    pub progress: SpeechProgress,
    /// How many tokens this reading is expected to take, which the bar needs after the last one
    /// and before the first.
    ///
    /// Unlike a picture's step count this is not known when the run is posted -- it is the
    /// model's estimate, and it arrives with the first token. Until then it is what the length of
    /// the text suggests, so that the bar has somewhere to start rather than jumping once the
    /// model has an opinion.
    pub expected: i32,
    pub started: Instant,
}

/// What the worker is busy with, which is the only thing on the screen that moves on its own.
#[derive(Clone)]
pub enum Doing {
    /// Nothing: either no model has been asked for yet, or the last thing asked for has finished.
    Nothing,
    Fetching(Fetch),
    /// Reading a model off the disk and onto the device, which is minutes for a large one.
    Reading {
        model: String,
    },
    Drawing(Run),
    Speaking(Say),
}

impl Doing {
    /// Whether the worker is busy, which is what makes a second request wait its turn.
    pub fn is_busy(&self) -> bool {
        !matches!(self, Doing::Nothing)
    }

    /// How far along, between nothing and all of it, or `None` for work with no end in sight.
    ///
    /// A fetch knows how many bytes it is; reading a model does not know anything -- it is one
    /// call into the tensor library that returns when it returns -- and a bar drawn for it would
    /// be a bar making something up.
    fn fraction(&self) -> Option<f64> {
        match self {
            Doing::Nothing | Doing::Reading { .. } => None,
            Doing::Fetching(fetch) => fetch
                .total
                .filter(|total| *total > 0)
                .map(|total| fetch.done as f64 / total as f64),
            Doing::Drawing(run) => Some(drawn(run.progress, run.steps)),
            Doing::Speaking(say) => Some(spoken(say.progress, say.expected)),
        }
    }

    /// What it is busy with, in the words the bar carries.
    pub fn words(&self) -> String {
        match self {
            Doing::Nothing => String::new(),
            Doing::Fetching(fetch) if fetch.file.is_empty() => format!("fetching {}", fetch.model),
            Doing::Fetching(fetch) => match fetch.parts {
                0 => format!("fetching {} -- {}", fetch.model, fetch.file),
                parts => format!(
                    "fetching {} -- {} ({} of {parts})",
                    fetch.model, fetch.file, fetch.part
                ),
            },
            Doing::Reading { model } => format!("reading {model}"),
            Doing::Drawing(run) => match run.progress {
                GenerationProgress::Encoding => "reading the prompt".to_string(),
                GenerationProgress::Step { done, total } => format!("step {done} of {total}"),
                GenerationProgress::Decoding => "making the picture".to_string(),
            },
            Doing::Speaking(say) => match say.progress {
                SpeechProgress::Reading => "reading the text".to_string(),
                // "of about", because the number on the right is worked out from the length of
                // the text rather than known: a model that decides what to say one token at a
                // time does not know how many there will be until it stops. A bar that said "of"
                // and then went past it would be a bar that had been lying.
                SpeechProgress::Saying { done, expected } => {
                    format!("token {done} of about {expected}")
                }
                SpeechProgress::Sounding => "making the sound".to_string(),
            },
        }
    }
}

/// How far along a run is, as a bar can show it.
fn drawn(progress: GenerationProgress, steps: i32) -> f64 {
    let all = |steps: i32| ENCODING + f64::from(steps.max(1)) + DECODING;
    match progress {
        GenerationProgress::Encoding => 0.0,
        GenerationProgress::Step { done, total } => (ENCODING + f64::from(done)) / all(total),
        GenerationProgress::Decoding => (ENCODING + f64::from(steps.max(1))) / all(steps),
    }
}

/// How far along a reading is, as a bar can show it.
///
/// The same weighing the picture bar uses, over the three stages a speech run has. The token
/// count is an estimate, so `done` can pass `expected` on a reading that runs long; what that has
/// to not do is run the bar off the end and back round, so it is held at the last token instead
/// -- which is where a run that is taking longer than expected actually is.
fn spoken(progress: SpeechProgress, expected: i32) -> f64 {
    let all = |expected: f64| TEXT + expected + VOCODER;
    let expected = f64::from(expected.max(1));

    match progress {
        SpeechProgress::Reading => 0.0,
        SpeechProgress::Saying { done, .. } => {
            (TEXT + f64::from(done.max(0)).min(expected)) / all(expected)
        }
        SpeechProgress::Sounding => (TEXT + expected) / all(expected),
    }
}

/// A setting, as a number a box can hold.
///
/// JSON has one kind of number and it is a double, so an `f32` widened into one arrives at the
/// page as what that `f32` actually was: a temperature of 0.8 is 0.800000011920929, and a box
/// filled in from it shows fifteen digits of the float's own representation rather than the
/// setting somebody chose.
///
/// Rounded to four places, which is finer than any box here steps and coarser than the error.
/// What goes back the other way is read as an `f32` again, so nothing is lost that was not
/// already lost on the way in.
fn showable(value: f32) -> f64 {
    (f64::from(value) * 10_000.0).round() / 10_000.0
}

/// What a finished picture was asked for, which is written into the PNG and kept with its job.
///
/// What was asked for is kept beside it because a picture that came out well is asked about
/// later, and by then the seed it came from is the first thing nobody remembers. It is the same
/// line the browser shows under the picture and the same line that would let somebody draw it
/// again.
pub struct Picture {
    pub width: usize,
    pub height: usize,
    pub prompt: String,
    pub negative: String,
    pub seed: u64,
    pub steps: i32,
    pub guidance: f32,
    pub model: String,
    /// Whether it started from a picture rather than from noise, and how far it walked from it.
    pub from_image: Option<f32>,
    pub elapsed: Duration,
}

impl Picture {
    /// The line under the picture, written the way every other tool writes it, so that it can be
    /// pasted somewhere that reads them.
    pub fn parameters(&self) -> String {
        let mut lines = vec![self.prompt.clone()];
        if !self.negative.trim().is_empty() {
            lines.push(format!("Negative prompt: {}", self.negative));
        }

        let mut settings = format!(
            "Steps: {}, CFG scale: {}, Seed: {}, Size: {}x{}, Model: {}",
            self.steps, self.guidance, self.seed, self.width, self.height, self.model,
        );
        if let Some(strength) = self.from_image {
            settings.push_str(&format!(", Denoising strength: {strength}"));
        }
        lines.push(settings);

        lines.join("\n")
    }

    pub fn json(&self) -> Value {
        json!({
            "width": self.width,
            "height": self.height,
            "prompt": self.prompt,
            "negative": self.negative,
            "seed": self.seed,
            "steps": self.steps,
            "guidance": showable(self.guidance),
            "model": self.model,
            "strength": self.from_image.map(showable),
            "seconds": self.elapsed.as_secs_f64(),
            "parameters": self.parameters(),
        })
    }
}

/// What a finished clip was asked for, kept with its job.
///
/// The same thing beside a clip that [`Picture::parameters`] is beside a picture, and kept for
/// the same reason: a reading that came out well is asked about later, and by then the seed and
/// the speed are the first two things nobody remembers.
pub struct Clip {
    pub text: String,
    pub seconds: f64,
    pub rate: u32,
    pub speed: f32,
    pub temperature: f32,
    pub seed: u64,
    pub voice: String,
    /// Whether it was given a recording to sound like. Not the recording itself -- that is
    /// megabytes, and it is held once where the page can ask for it.
    pub from_a_recording: bool,
    /// The instruction it was read with, where it was given one.
    pub style: Option<String>,
    pub elapsed: Duration,
}

impl Clip {
    /// The line under the clip, in the same shape the line under a picture is in: what was said
    /// first, and everything it would take to say it again after it.
    pub fn parameters(&self) -> String {
        let mut settings = format!(
            "Speed: {}, Temperature: {}, Seed: {}, Rate: {} Hz, Voice: {}",
            self.speed, self.temperature, self.seed, self.rate, self.voice,
        );
        if self.from_a_recording {
            settings.push_str(", From a recording: yes");
        }
        if let Some(style) = &self.style {
            settings.push_str(&format!(", Style: {style}"));
        }

        format!("{}\n{settings}", self.text)
    }

    pub fn json(&self) -> Value {
        json!({
            "text": self.text,
            "length": self.seconds,
            "rate": self.rate,
            "speed": showable(self.speed),
            "temperature": showable(self.temperature),
            "seed": self.seed,
            "voice": self.voice,
            "from_a_recording": self.from_a_recording,
            "style": self.style,
            "seconds": self.elapsed.as_secs_f64(),
            "parameters": self.parameters(),
        })
    }
}

/// What the program is doing, which is the same whoever is asking.
pub struct World {
    /// What the program draws with, for a program started to draw.
    pub model: Option<Chosen>,
    /// What it speaks with, for a program started to speak.
    pub voice: Option<Spoken>,
    pub doing: Doing,
    /// The job the worker is running, if it is running one. Set by the store as it hands the
    /// job over, under the store's own lock: see [`Store::take_next`].
    pub running: Option<String>,
    /// What the read before the server was up came to. The terminal reads it; a job's own
    /// failure is written on the job.
    pub note: Option<String>,
}

/// The world, the store, the flag that stops a run, and what about the process cannot change.
///
/// Where both locks are taken, the store's is taken first: [`Store::take_next`] tells the world
/// which job is running while it holds its own. Anything that wants both takes what it needs from
/// the store before it takes the world.
pub struct Shared {
    world: Mutex<World>,
    /// Set while a job is running to ask it to stop between steps, which is the only place it
    /// can be asked: a step, once started, is a kernel launch that nothing here can call back.
    cancel: AtomicBool,
    /// Whether the read before the server was up is still going.
    reading: AtomicBool,
    /// Where runs go, chosen in the terminal before the page opened.
    runtime: Runtime,
    /// What the program is for, chosen in the terminal as well: which kind of job it takes.
    task: Task,
    /// Every job and upload, kept on the disk.
    store: Store,
    /// The upload `-i` named, which the page offers to start from.
    starting_picture: Option<String>,
}

impl Shared {
    pub fn new(task: Task, runtime: Runtime, store: Store) -> Shared {
        Shared {
            world: Mutex::new(World {
                model: None,
                voice: None,
                doing: Doing::Nothing,
                running: None,
                note: None,
            }),
            cancel: AtomicBool::new(false),
            reading: AtomicBool::new(false),
            runtime,
            task,
            store,
            starting_picture: None,
        }
    }

    /// Keeps `bytes` as an upload that the page offers to start from, which is what `-i` asks.
    pub fn start_holding(&mut self, bytes: &[u8]) -> Result<(), Refused> {
        self.starting_picture = Some(self.store.upload(bytes)?.id);
        Ok(())
    }

    /// The world, for as long as the returned guard lives.
    ///
    /// A poisoned lock is taken as it is rather than unwrapped into a panic of this thread's own.
    /// What the lock holds is a description of what is happening, not an invariant anything else
    /// depends on: a worker that died partway through writing it leaves a stale line on a screen,
    /// and a server that then refuses every request leaves nothing on it at all.
    pub fn world(&self) -> MutexGuard<'_, World> {
        self.world.lock().unwrap_or_else(|held| held.into_inner())
    }

    /// Changes the world under its lock.
    pub fn change<T>(&self, change: impl FnOnce(&mut World) -> T) -> T {
        change(&mut self.world())
    }

    pub fn store(&self) -> &Store {
        &self.store
    }

    pub fn runtime(&self) -> Runtime {
        self.runtime
    }

    /// What kind of job this program takes.
    pub fn kind(&self) -> Kind {
        match self.task.speaks() {
            true => Kind::Speech,
            false => Kind::Image,
        }
    }

    /// Says something about the read before the server was up, in the terminal.
    pub fn say(&self, said: impl Into<String>, bad: bool) {
        let said = said.into();
        crate::cli::webui::log::line(format_args!(
            "{}: {said}",
            if bad { "error" } else { "note" }
        ));
        self.world().note = Some(said);
    }

    /// Marks the read before the server was up as started or finished.
    pub fn reading(&self, reading: bool) {
        self.reading.store(reading, Ordering::Release);
    }

    /// Whether anything is happening: the read before the server is up, or a job.
    pub fn is_busy(&self) -> bool {
        // The store's before the world's, as everywhere.
        let queued = self.store.queued();
        let world = self.world();
        self.reading.load(Ordering::Acquire)
            || queued > 0
            || world.running.is_some()
            || world.doing.is_busy()
    }

    // -- the queue ------------------------------------------------------------------------------

    /// The job that has waited longest, marked as running, waiting for one where there is none.
    /// The world is told which it is, and the stop flag cleared, before anything can ask it to
    /// stop: a stop asked for after the last job finished is not this one's.
    pub fn next(&self) -> Job {
        self.store.take_next(|job| {
            self.world().running = Some(job.id.clone());
            self.cancel.store(false, Ordering::Relaxed);
        })
    }

    /// Marks the running job as finished with, whichever way it ended.
    pub fn finished(&self) {
        let mut world = self.world();
        world.running = None;
        world.doing = Doing::Nothing;
    }

    /// Asks a job to stop: marked cancelled where it is waiting, asked to stop between steps where
    /// it is running. There is one worker, so a running job is the one it is running.
    pub fn cancel(&self, id: &str) -> Cancelled {
        self.store
            .cancel(id, || self.cancel.store(true, Ordering::Relaxed))
    }

    /// Whether the running job has been asked to stop.
    pub fn interrupted(&self) -> bool {
        self.cancel.load(Ordering::Relaxed)
    }

    // -- what is answered with ------------------------------------------------------------------

    /// A job as a request is answered with it: what is kept, and what only the running program
    /// knows -- where it is in line, and how far along it is.
    pub fn describe_job(&self, job: &Job) -> Value {
        let mut described = job.json();
        match job.status {
            Status::Queued => {
                if let Some(at) = self.store.position(&job.id) {
                    described["position"] = json!(at);
                }
            }
            Status::Running => {
                let world = self.world();
                if world.running.as_deref() == Some(job.id.as_str()) {
                    described["progress"] = progress(&world.doing);
                    described["stopping"] = json!(self.interrupted());
                }
            }
            _ => {}
        }

        described
    }

    /// What the worker is doing, for anybody: whether it is busy, which job, how far along, and
    /// how many are waiting.
    pub fn describe_worker(&self) -> Value {
        let busy = self.is_busy();
        let queued = self.store.queued();
        let world = self.world();
        json!({
            "busy": busy,
            "running": world.running,
            "queued": queued,
            "progress": progress(&world.doing),
        })
    }

    /// The model this program serves, and what a job of it can be asked for.
    pub fn describe_model(&self) -> Value {
        let (used, limit) = (self.store.used(), self.store.limit());
        let world = self.world();
        let model = world.model.as_ref().map(|model| {
            json!({
                "name": model.name,
                "full_name": model.full_name,
                "on_disk": model.on_disk,
                "in_memory": model.in_memory,
                "width": model.defaults.width,
                "height": model.defaults.height,
                "steps": model.defaults.num_steps,
                "guidance": showable(model.defaults.guidance_scale),
                // Not the number, but whether there is a number to ask for. A distilled model
                // runs one pass and the second prompt is never encoded, so the box for it would
                // be a box whose contents go nowhere.
                "takes_guidance": model.takes_guidance,
                "sampler": model.sampler,
                "sizes": model.sizes.iter().map(|(w, h)| json!([w, h])).collect::<Vec<_>>(),
                "prompt": model.suggested_prompt,
                "avoid": model.suggested_avoid,
                "draws_from_a_picture": model.no_picture_because.is_none(),
                "no_picture_because": model.no_picture_because,
            })
        });

        let voice = world.voice.as_ref().map(|voice| {
            json!({
                "name": voice.name,
                "full_name": voice.full_name,
                "on_disk": voice.on_disk,
                "in_memory": voice.in_memory,
                "speed": showable(voice.defaults.speed),
                "temperature": showable(voice.defaults.temperature),
                "rate": voice.rate,
                "takes_a_recording": voice.no_likeness_because.is_none(),
                "no_likeness_because": voice.no_likeness_because,
                // What the style list offers, in the model's own words beside the label. Empty
                // for a voice that takes none, which is how the page knows to show no list.
                "styles": voice.styles.iter().map(|style| json!({
                    "label": style.label,
                    "instruction": style.instruction,
                })).collect::<Vec<_>>(),
                // The sentence the speech tab is built around. Null for a real model, which is
                // how the warning on that tab goes away by itself the day there is one.
                "not_a_voice_because": voice.not_a_voice_because,
            })
        });

        json!({
            "built_from": crate::cli::REVISION,
            "task": self.task.name(),
            "kind": self.kind().name(),
            "device": self.runtime.name(),
            "model": model,
            "voice": voice,
            "starting_picture": self.starting_picture,
            "output": { "used": used, "limit": limit },
        })
    }
}

/// How far along the worker is, as a bar draws it.
fn progress(doing: &Doing) -> Value {
    json!({
        "doing": doing.words(),
        "fraction": doing.fraction(),
        "seconds": match doing {
            Doing::Drawing(run) => Some(run.started.elapsed().as_secs_f64()),
            Doing::Speaking(say) => Some(say.started.elapsed().as_secs_f64()),
            _ => None,
        },
        "fetching": matches!(doing, Doing::Fetching(_)),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::cli::args::DeviceOption;

    /// A program with nothing loaded, keeping its jobs in a directory of the calling test's own.
    fn a_program(called: &str) -> Shared {
        let root =
            std::env::temp_dir().join(format!("libwaifu-state-{called}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        Shared::new(
            Task::Txt2Img,
            DeviceOption::Cpu.resolve(),
            Store::open(&root, None).expect("a store"),
        )
    }

    fn a_drawing() -> Doing {
        Doing::Drawing(Run {
            progress: GenerationProgress::Step { done: 3, total: 8 },
            steps: 8,
            started: Instant::now(),
        })
    }

    fn a_picture(seed: u64) -> Picture {
        Picture {
            width: 1024,
            height: 1024,
            prompt: "a cat".to_string(),
            negative: String::new(),
            seed,
            steps: 30,
            guidance: 5.0,
            model: "sdxl:base".to_string(),
            from_image: None,
            elapsed: Duration::from_secs(9),
        }
    }

    fn a_clip(seed: u64) -> Clip {
        Clip {
            text: "hello there".to_string(),
            seconds: 1.5,
            rate: 24_000,
            speed: 1.0,
            temperature: 0.8,
            seed,
            voice: "tones".to_string(),
            from_a_recording: false,
            style: None,
            elapsed: Duration::from_millis(120),
        }
    }

    #[test]
    fn the_bar_weighs_the_parts_that_are_not_steps() {
        // Reading the prompt costs about a step and the decode costs several, so a run that has
        // finished every step is not a run that has finished.
        let all_the_steps = drawn(GenerationProgress::Decoding, 30);
        assert!(all_the_steps < 1.0, "{all_the_steps}");
        assert!(all_the_steps > 0.8, "{all_the_steps}");

        assert_eq!(drawn(GenerationProgress::Encoding, 30), 0.0);
        assert!(
            drawn(
                GenerationProgress::Step {
                    done: 15,
                    total: 30
                },
                30
            ) > 0.4,
            "half the steps is about half way"
        );
    }

    #[test]
    fn a_run_of_no_steps_does_not_divide_by_nothing() {
        // Not something the page can ask for -- the box has a minimum -- and the arithmetic has
        // to hold anyway, since what it would do instead is produce an infinity and paint with it.
        for progress in [
            GenerationProgress::Encoding,
            GenerationProgress::Step { done: 0, total: 0 },
            GenerationProgress::Decoding,
        ] {
            assert!(drawn(progress, 0).is_finite(), "{progress:?}");
        }
    }

    #[test]
    fn reading_a_model_does_not_claim_to_know_how_far_along_it_is() {
        // It is one call into the tensor library that returns when it returns. A fraction here
        // would be a number nothing measured.
        let reading = Doing::Reading {
            model: "sdxl:base".to_string(),
        };
        assert_eq!(reading.fraction(), None);
        assert!(reading.is_busy());
        assert!(reading.words().contains("sdxl:base"));
    }

    #[test]
    fn a_fetch_of_an_unknown_length_has_no_fraction_either() {
        // Which is what a server that did not say how long the file is leaves, and it is also
        // what the moment before the first byte looks like.
        let mut fetch = Fetch {
            model: "sdxl:base".to_string(),
            hub: None,
            file: "sdxl-base.waifupkg".to_string(),
            done: 512,
            total: None,
            part: 1,
            parts: 3,
        };
        assert_eq!(Doing::Fetching(fetch.clone()).fraction(), None);

        fetch.total = Some(1024);
        assert_eq!(Doing::Fetching(fetch.clone()).fraction(), Some(0.5));

        // And the words say which model, not only which file: the file names in a package are not
        // something anybody recognises.
        let words = Doing::Fetching(fetch).words();
        assert!(words.contains("sdxl:base"), "{words}");
        assert!(words.contains("1 of 3"), "{words}");
    }

    #[test]
    fn nothing_happening_is_not_busy() {
        assert!(!Doing::Nothing.is_busy());
        assert_eq!(Doing::Nothing.fraction(), None);
        assert_eq!(Doing::Nothing.words(), "");
    }

    #[test]
    fn what_a_picture_was_asked_for_is_written_where_it_can_be_read_back() {
        // The line every other tool of this kind writes, so that it can be pasted somewhere that
        // reads them -- and so that the seed is beside the picture it drew.
        let said = a_picture(7).parameters();
        assert!(said.starts_with("a cat"), "{said}");
        assert!(said.contains("Seed: 7"), "{said}");
        assert!(said.contains("Size: 1024x1024"), "{said}");
        assert!(said.contains("Model: sdxl:base"), "{said}");

        // An empty negative prompt is left out rather than written as an empty line: what it says
        // is that there was nothing to steer away from, which the absence says too.
        assert!(!said.contains("Negative prompt"), "{said}");
    }

    #[test]
    fn a_run_from_a_picture_says_how_far_it_walked_from_it() {
        let mut picture = a_picture(7);
        picture.negative = "blurry".to_string();
        picture.from_image = Some(0.6);

        let said = picture.parameters();
        assert!(said.contains("Negative prompt: blurry"), "{said}");
        assert!(said.contains("Denoising strength: 0.6"), "{said}");
    }

    #[test]
    fn the_speech_bar_weighs_the_vocoder_at_the_end() {
        // The same weighing the picture bar has, and for the same reason: a bar drawn from the
        // tokens alone fills up and then sits there while the vocoder runs.
        let all_the_tokens = spoken(SpeechProgress::Sounding, 40);
        assert!(all_the_tokens < 1.0, "{all_the_tokens}");
        assert!(all_the_tokens > 0.8, "{all_the_tokens}");

        assert_eq!(spoken(SpeechProgress::Reading, 40), 0.0);
        let half = spoken(
            SpeechProgress::Saying {
                done: 20,
                expected: 40,
            },
            40,
        );
        assert!(half > 0.4 && half < 0.6, "{half}");
    }

    #[test]
    fn a_reading_that_runs_long_does_not_run_the_bar_off_the_end() {
        // The token count is an estimate, so a reading can pass it -- which is a bar that fills
        // up and starts again if nothing holds it.
        let past = spoken(
            SpeechProgress::Saying {
                done: 400,
                expected: 40,
            },
            40,
        );
        assert!(past <= 1.0, "{past}");
        assert_eq!(past, spoken(SpeechProgress::Sounding, 40));

        // And a reading with no estimate yet does not divide by one that is not there.
        for expected in [0, -1] {
            for progress in [
                SpeechProgress::Reading,
                SpeechProgress::Saying {
                    done: 0,
                    expected: 0,
                },
                SpeechProgress::Sounding,
            ] {
                assert!(spoken(progress, expected).is_finite(), "{progress:?}");
            }
        }
    }

    #[test]
    fn what_is_being_said_is_reported_without_claiming_to_know_how_long_it_will_be() {
        // "of about", because the number on the right is worked out from the text rather than
        // known. A bar that said "of" and then went past it would have been lying.
        let saying = Doing::Speaking(Say {
            progress: SpeechProgress::Saying {
                done: 3,
                expected: 40,
            },
            expected: 40,
            started: Instant::now(),
        });

        assert!(saying.is_busy());
        let words = saying.words();
        assert!(words.contains("3 of about 40"), "{words}");
        assert!(saying.fraction().is_some());
    }

    #[test]
    fn what_a_clip_was_asked_for_is_written_where_it_can_be_read_back() {
        // The same line a picture gets, beside the clip it belongs to: a reading that came out
        // well is asked about later, and by then the seed is the first thing nobody remembers.
        let said = a_clip(7).parameters();
        assert!(said.starts_with("hello there"), "{said}");
        assert!(said.contains("Seed: 7"), "{said}");
        assert!(said.contains("Speed: 1"), "{said}");
        assert!(said.contains("Voice: tones"), "{said}");
        // Not said where it was not true, rather than written as a "no".
        assert!(!said.contains("From a recording"), "{said}");

        let mut from_one = a_clip(7);
        from_one.from_a_recording = true;
        assert!(from_one.parameters().contains("From a recording: yes"));
    }

    /// A job posted a moment after the last, so that which came first is the order they were
    /// posted in: two in the same millisecond go in an order of their own, by id.
    fn a_job(shared: &Shared) -> Job {
        std::thread::sleep(std::time::Duration::from_millis(2));
        shared.store().submit(Kind::Image, json!({})).unwrap()
    }

    #[test]
    fn jobs_are_run_in_the_order_they_were_asked_for() {
        let shared = a_program("order");
        let first = a_job(&shared);
        let second = a_job(&shared);

        assert_eq!(shared.describe_job(&first)["position"], 0);
        assert_eq!(shared.describe_job(&second)["position"], 1);
        assert_eq!(shared.describe_worker()["queued"], 2);

        assert_eq!(shared.next().id, first.id);
        assert_eq!(shared.world().running.as_deref(), Some(first.id.as_str()));
        assert_eq!(shared.store().position(&second.id), Some(0));
        assert!(shared.is_busy());

        shared.finished();
        assert_eq!(shared.next().id, second.id);
        shared.finished();
        assert!(!shared.is_busy());
    }

    #[test]
    fn a_waiting_job_is_taken_out_of_line_and_a_running_one_asked_to_stop() {
        let shared = a_program("cancel");
        let running = a_job(&shared);
        let waiting = a_job(&shared);
        shared.next();

        assert_eq!(shared.cancel(&waiting.id), Cancelled::Dequeued);
        assert_eq!(shared.store().position(&waiting.id), None);
        assert_eq!(
            shared.store().find(&waiting.id).unwrap().status,
            Status::Cancelled
        );
        assert_eq!(shared.store().queued(), 0);

        assert!(!shared.interrupted());
        assert_eq!(shared.cancel(&running.id), Cancelled::Stopping);
        assert!(shared.interrupted());
        shared.world().doing = a_drawing();
        let described = shared.describe_job(&shared.store().find(&running.id).unwrap());
        assert_eq!(described["stopping"], true);
        assert!(described["progress"]["doing"]
            .as_str()
            .unwrap()
            .contains("3 of 8"));

        assert_eq!(shared.cancel(&"0".repeat(32)), Cancelled::NotGoing);
        assert_eq!(shared.cancel(&waiting.id), Cancelled::NotGoing);
    }

    #[test]
    fn a_stop_left_over_from_the_last_job_is_not_the_next_one_s() {
        let shared = a_program("stale-stop");
        let first = a_job(&shared);
        shared.next();
        shared.cancel(&first.id);
        assert!(shared.interrupted());
        shared.store().cancelled(&first.id);
        shared.finished();

        a_job(&shared);
        shared.next();
        assert!(!shared.interrupted());
    }

    #[test]
    fn the_worker_waits_for_a_job_when_there_is_none() {
        let shared = std::sync::Arc::new(a_program("wait"));

        let waiting = std::thread::spawn({
            let shared = std::sync::Arc::clone(&shared);
            move || shared.next()
        });
        std::thread::sleep(std::time::Duration::from_millis(20));
        assert!(!waiting.is_finished());

        let job = a_job(&shared);
        assert_eq!(waiting.join().unwrap().id, job.id);
    }

    #[test]
    fn the_model_says_what_built_it_what_it_takes_and_how_full_the_output_is() {
        // Whichever screen is up when something goes wrong is the one that ends up in the
        // screenshot, and a screenshot that cannot say which code it came from is worth much less
        // than one that can.
        let shared = a_program("model");
        let described = shared.describe_model();
        assert_eq!(described["built_from"], crate::cli::REVISION);
        assert_eq!(described["device"], "cpu");
        assert_eq!(described["kind"], "image");
        assert!(described["model"].is_null());
        assert_eq!(described["output"]["used"], 0);
        assert!(described["starting_picture"].is_null());
    }

    #[test]
    fn the_picture_minus_i_named_is_an_upload_the_page_is_offered() {
        let mut shared = a_program("minus-i");
        shared.start_holding(&[0x89, b'P', b'N', b'G', 1]).unwrap();
        let id = shared.describe_model()["starting_picture"]
            .as_str()
            .unwrap()
            .to_string();
        assert_eq!(shared.store().find_upload(&id).unwrap().mime, "image/png");
    }

    #[test]
    fn a_setting_reaches_the_page_as_the_number_somebody_chose() {
        // JSON has one kind of number and it is a double. A temperature of 0.8 widened straight
        // into one is 0.800000011920929, and a box filled in from that shows the float's own
        // representation rather than the setting.
        assert_eq!(showable(0.8), 0.8);
        assert_eq!(showable(5.3), 5.3);
        assert_eq!(showable(7.0), 7.0);
        assert_eq!(showable(0.0), 0.0);

        let clip = a_clip(1).json();
        assert_eq!(clip["temperature"], 0.8);
        assert_eq!(clip["speed"], 1.0);
    }
}
