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
use std::time::Duration;

use serde_json::{json, Value};

use crate::cli::args::Runtime;
use crate::cli::task::Task;
use crate::cli::webui::store::{Cancelled, Job, Kind, Refused, Status, Store};
// In the library since the C API needed them; named here as well, where they used to be.
pub use crate::describe::{Chosen, ChosenConverter, Spoken};
pub use crate::progress::{Convert, Doing, Fetch, Run, Say, VOCODER_RATIO};

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

/// What a finished conversion was asked for, kept with its job: what [`Clip`] is for a reading.
pub struct Converted {
    pub seconds: f64,
    pub rate: u32,
    pub steps: i32,
    pub style: bool,
    pub seed: u64,
    pub converter: String,
    pub elapsed: Duration,
}

impl Converted {
    /// The line under the clip. There are no words to put first: what was said is the source's.
    pub fn parameters(&self) -> String {
        format!(
            "Steps: {}, Style: {}, Seed: {}, Rate: {} Hz, Converter: {}",
            self.steps,
            if self.style { "yes" } else { "no" },
            self.seed,
            self.rate,
            self.converter,
        )
    }

    pub fn json(&self) -> Value {
        json!({
            "length": self.seconds,
            "rate": self.rate,
            "steps": self.steps,
            "style": self.style,
            "seed": self.seed,
            "converter": self.converter,
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
    /// What it converts with, for a program started to convert.
    pub converter: Option<ChosenConverter>,
    /// How long the vocoder took against everything before it, the last time anything was said.
    /// What the speech bar fills its last part by; see [`Say::ratio`].
    pub vocoder_ratio: f64,
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
                converter: None,
                vocoder_ratio: VOCODER_RATIO,
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
        match self.task {
            Task::Text2Speech => Kind::Speech,
            Task::Speech2Speech => Kind::Conversion,
            Task::Txt2Img | Task::Img2Img => Kind::Image,
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

        let converter = world.converter.as_ref().map(|converter| {
            json!({
                "name": converter.name,
                "full_name": converter.full_name,
                "on_disk": converter.on_disk,
                "in_memory": converter.in_memory,
                "steps": converter.defaults.steps,
                "rate": converter.rate,
                "converts_style": converter.no_style_because.is_none(),
                "no_style_because": converter.no_style_because,
            })
        });
        json!({
            "built_from": crate::cli::REVISION,
            "task": self.task.name(),
            "kind": self.kind().name(),
            "device": self.runtime.name(),
            "model": model,
            "voice": voice,
            "converter": converter,
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
            Doing::Converting(convert) => Some(convert.started.elapsed().as_secs_f64()),
            _ => None,
        },
        "fetching": matches!(doing, Doing::Fetching(_)),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::time::Instant;

    use crate::cli::args::DeviceOption;
    use crate::GenerationProgress;

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
