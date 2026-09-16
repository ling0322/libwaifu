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

//! What the browser is looking at, and the one lock the two threads meet over.
//!
//! The model sits on a thread of its own -- a tensor never leaves the thread that made it -- and
//! the requests arrive on whichever thread the server handed them to. Between the two there used
//! to be a pair of channels, which worked when one screen read them in a loop; a request cannot
//! read a channel, because it arrives once and has to answer with everything at that moment
//! rather than with whatever has been posted since it last looked.
//!
//! So what moves is a value under a lock. The worker writes what it is doing as it does it, and
//! every request reads the whole of it. The lock is held for the length of a field copy and never
//! across anything that draws, which is the only rule this arrangement has.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, MutexGuard};
use std::time::{Duration, Instant};

use serde_json::{json, Value};

use crate::cli::args::Runtime;
use crate::{GenerationDefaults, GenerationProgress};

/// How much of a run the parts that are not steps are worth, when a bar is drawn from them.
///
/// Weighed rather than counted. Reading the prompt costs about a step and turning the finished
/// latent into pixels costs several, so a bar drawn from the steps alone would fill up and then
/// sit there for the slowest part of the run.
const ENCODING: f64 = 1.0;
const DECODING: f64 = 3.0;

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
        }
    }

    /// What it is busy with, in the words the bar carries.
    fn words(&self) -> String {
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

/// A finished picture: the file it was written to, and what it was asked for.
///
/// What was asked for is kept beside it because a picture that came out well is asked about
/// later, and by then the seed it came from is the first thing nobody remembers. It is the same
/// line the browser shows under the picture and the same line that would let somebody draw it
/// again.
pub struct Picture {
    pub file: String,
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

    fn json(&self) -> Value {
        json!({
            "file": self.file,
            "width": self.width,
            "height": self.height,
            "prompt": self.prompt,
            "negative": self.negative,
            "seed": self.seed,
            "steps": self.steps,
            "guidance": self.guidance,
            "model": self.model,
            "strength": self.from_image,
            "seconds": self.elapsed.as_secs_f64(),
            "parameters": self.parameters(),
        })
    }
}

/// Something worth saying, under the bar. `bad` is what colours it.
#[derive(Clone)]
pub struct Note {
    pub said: String,
    pub bad: bool,
}

/// Everything the browser can ask about, behind one lock.
pub struct Session {
    /// What the page is set to draw with, whether or not its weights have been read.
    pub model: Option<Chosen>,
    pub doing: Doing,
    pub note: Option<Note>,
    /// What has been drawn this session, newest first. The files themselves are on the disk where
    /// the program was started; this is the list of what may be asked for by name, which is also
    /// what keeps a request from asking for a file that was never written here.
    pub gallery: Vec<Picture>,
    /// Counted up every time anything here changes, so that a browser polling for news can tell
    /// "nothing has happened" from "everything happened and finished" without comparing the whole
    /// of it. A run that starts and ends between two polls still moves this.
    pub revision: u64,
}

impl Session {
    fn new() -> Session {
        Session {
            model: None,
            doing: Doing::Nothing,
            note: None,
            gallery: Vec::new(),
            revision: 0,
        }
    }
}

/// The lock, the flag that stops a run, and the one thing about the process that cannot change.
pub struct Shared {
    session: Mutex<Session>,
    /// Set while a run is in flight to ask it to stop between steps, which is the only place it
    /// can be asked: a step, once started, is a kernel launch that nothing here can call back.
    cancel: AtomicBool,
    /// Whether a command has been posted to the worker and not yet finished.
    ///
    /// [`Doing`] is not that. It says what the worker has picked up, which is a moment later than
    /// when a request handed it over -- and in that moment a second request reads "nothing is
    /// happening" and posts a second run. Two clicks on a button are exactly that far apart, and
    /// what came of it was two pictures for one press.
    working: AtomicBool,
    /// The picture an image to image run would start from, as the bytes it arrived as.
    ///
    /// One of them rather than a list: the page has one such box, and a second file dropped on it
    /// replaces the first. Behind its own lock, away from the session, because the session is
    /// copied out into JSON several times a second and this is megabytes.
    upload: Mutex<Option<Vec<u8>>>,
    /// Where runs go. Named on the command line to begin with and changed from the page after
    /// that, which is why it is behind a lock: a request reads it to describe the program while
    /// the worker is reading it to decide where to put the next model.
    ///
    /// Changed only by the worker, and only with nothing loaded -- weights live on the device
    /// they were read onto, and a model that is on a card the runtime no longer names is a model
    /// nothing can be asked of.
    runtime: Mutex<Runtime>,
}

impl Shared {
    pub fn new(runtime: Runtime) -> Shared {
        Shared {
            session: Mutex::new(Session::new()),
            cancel: AtomicBool::new(false),
            working: AtomicBool::new(false),
            upload: Mutex::new(None),
            runtime: Mutex::new(runtime),
        }
    }

    /// The session, for as long as the returned guard lives.
    ///
    /// A poisoned lock is taken as it is rather than unwrapped into a panic of this thread's own.
    /// What the lock holds is a description of what is happening, not an invariant anything else
    /// depends on: a worker that died partway through writing it leaves a stale line on a screen,
    /// and a server that then refuses every request leaves nothing on it at all.
    pub fn session(&self) -> MutexGuard<'_, Session> {
        self.session.lock().unwrap_or_else(|held| held.into_inner())
    }

    /// Changes the session and marks it changed, which is the only way it is changed.
    pub fn change<T>(&self, change: impl FnOnce(&mut Session) -> T) -> T {
        let mut session = self.session();
        let answer = change(&mut session);
        session.revision += 1;

        answer
    }

    pub fn say(&self, said: impl Into<String>, bad: bool) {
        self.change(|session| {
            session.note = Some(Note {
                said: said.into(),
                bad,
            })
        });
    }

    /// Takes the worker, if it is free. True means it is this caller's to post a command to.
    ///
    /// The one place a request decides whether the program is busy, and the deciding and the
    /// claiming are the same instruction on purpose: asked and answered separately, two requests
    /// both get told yes.
    pub fn claim(&self) -> bool {
        self.working
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
    }

    /// Gives it back, which the worker does as it finishes each command -- and which a request
    /// does itself when the command it claimed for could not be posted after all.
    pub fn release(&self) {
        self.working.store(false, Ordering::Release);
    }

    /// Whether anything is happening, which is a wider question than what the worker has picked
    /// up: a command posted a moment ago has not been picked up and is still something happening.
    pub fn is_busy(&self) -> bool {
        self.working.load(Ordering::Acquire) || self.session().doing.is_busy()
    }

    pub fn hold_upload(&self, bytes: Vec<u8>) {
        *self.held() = Some(bytes);
        // Counted as news: the page shows whether it has a picture to draw from, and a run
        // started from `-i` puts one there before any page has opened.
        self.change(|_| ());
    }

    /// A copy of the picture being held, for whoever is about to do something with it.
    ///
    /// Copied rather than borrowed: what wants it is the worker, on the far side of a channel,
    /// and holding this lock while a run decodes and scales the thing would be holding it for as
    /// long as that takes.
    pub fn upload(&self) -> Option<Vec<u8>> {
        self.held().clone()
    }

    pub fn forget_upload(&self) {
        *self.held() = None;
        self.change(|_| ());
    }

    /// Whether there is a picture to draw from, which is the part of it the page is told about.
    pub fn holding_a_picture(&self) -> bool {
        self.held().is_some()
    }

    fn held(&self) -> MutexGuard<'_, Option<Vec<u8>>> {
        self.upload.lock().unwrap_or_else(|held| held.into_inner())
    }

    pub fn runtime(&self) -> Runtime {
        *self.runtime.lock().unwrap_or_else(|held| held.into_inner())
    }

    /// Sends what is loaded next somewhere else. The worker's to call, once it has let go of
    /// whatever was on the old device.
    pub fn use_runtime(&self, runtime: Runtime) {
        *self.runtime.lock().unwrap_or_else(|held| held.into_inner()) = runtime;
        self.change(|_| ());
    }

    /// Asks whatever is running to stop where it is.
    pub fn interrupt(&self) {
        self.cancel.store(true, Ordering::Relaxed);
    }

    /// Whether a stop has been asked for. Cleared by the worker as each run begins: a stop asked
    /// for after the last run finished is not the next run's business.
    pub fn interrupted(&self) -> bool {
        self.cancel.load(Ordering::Relaxed)
    }

    pub fn carry_on(&self) {
        self.cancel.store(false, Ordering::Relaxed);
    }

    /// The little of it a browser drawing a progress bar needs, which it asks for twice a second.
    pub fn progress(&self) -> Value {
        // Read before the lock is taken, since asking it takes the lock too.
        let busy = self.is_busy();

        let session = self.session();
        json!({
            "revision": session.revision,
            "busy": busy,
            "drawing": matches!(session.doing, Doing::Drawing(_)),
            "fraction": session.doing.fraction(),
            "doing": session.doing.words(),
            "seconds": match &session.doing {
                Doing::Drawing(run) => Some(run.started.elapsed().as_secs_f64()),
                _ => None,
            },
            // Only while there is something to stop. The flag stays set between a run that was
            // stopped and the next one that clears it, and a bar saying "stopping" over an idle
            // program would be reporting the flag rather than what is happening.
            "interrupting": self.interrupted() && matches!(session.doing, Doing::Drawing(_)),
        })
    }

    /// The whole of it, which is what a browser asks for when it opens and after anything lands.
    pub fn describe(&self, models: Value) -> Value {
        let session = self.session();
        let model = session.model.as_ref().map(|model| {
            json!({
                "name": model.name,
                "full_name": model.full_name,
                "on_disk": model.on_disk,
                "in_memory": model.in_memory,
                "width": model.defaults.width,
                "height": model.defaults.height,
                "steps": model.defaults.num_steps,
                "guidance": model.defaults.guidance_scale,
                "sampler": model.sampler,
                "sizes": model.sizes.iter().map(|(w, h)| json!([w, h])).collect::<Vec<_>>(),
                "prompt": model.suggested_prompt,
                "avoid": model.suggested_avoid,
                "draws_from_a_picture": model.no_picture_because.is_none(),
                "no_picture_because": model.no_picture_because,
            })
        });

        json!({
            "revision": session.revision,
            "built_from": crate::cli::REVISION,
            "device": self.runtime().name(),
            // What else this machine could be asked for. A list rather than a flag, because the
            // answer is the machine's and not the build's.
            "devices": Runtime::available()
                .into_iter()
                .map(|runtime| runtime.name())
                .collect::<Vec<_>>(),
            "holding_a_picture": self.holding_a_picture(),
            "models": models,
            "model": model,
            "note": session.note.as_ref().map(|note| json!({ "said": note.said, "bad": note.bad })),
            "gallery": session.gallery.iter().map(Picture::json).collect::<Vec<_>>(),
        })
    }

    /// Where a file the browser asks for by name is, if this session wrote it.
    ///
    /// The gallery is the whole of what can be asked for. A path out of a request is otherwise a
    /// way to read any file the process can, and a server on the loopback address is still
    /// reachable by anything else running on the machine, a browser tab on another site included.
    pub fn wrote(&self, file: &str) -> Option<PathBuf> {
        let session = self.session();
        session
            .gallery
            .iter()
            .find(|picture| picture.file == file)
            .map(|picture| PathBuf::from(&picture.file))
    }

    /// Takes a picture out of the gallery, which is what makes its name unaskable-for again.
    ///
    /// Said separately from deleting the file: the row goes whether or not the file was still
    /// there to delete, because a row for a file that is gone is a thumbnail that cannot load.
    pub fn forget_picture(&self, file: &str) {
        self.change(|session| session.gallery.retain(|picture| picture.file != file));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::cli::args::DeviceOption;

    fn a_session() -> Shared {
        Shared::new(DeviceOption::Cpu.resolve())
    }

    fn a_picture(seed: u64) -> Picture {
        Picture {
            file: format!("waifu-{seed:04}.png"),
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
    fn every_change_is_news() {
        // The page polls a handful of numbers and asks for the whole state only when this moves,
        // so a change that did not move it is a change no open page would ever see.
        let shared = a_session();
        let first = shared.session().revision;

        shared.say("ready", false);
        let second = shared.session().revision;
        assert!(second > first);

        shared.change(|session| session.gallery.push(a_picture(1)));
        assert!(shared.session().revision > second);
    }

    #[test]
    fn only_what_this_session_drew_can_be_asked_for_by_name() {
        // The gallery is the whole of what is readable. Anything else is a path out of a request,
        // and this server is reachable by everything else running on the machine.
        let shared = a_session();
        shared.change(|session| session.gallery.push(a_picture(1)));

        assert!(shared.wrote("waifu-0001.png").is_some());
        assert!(shared.wrote("waifu-0002.png").is_none());
        assert!(shared.wrote("../../etc/passwd").is_none());
        assert!(shared.wrote("/etc/passwd").is_none());
    }

    #[test]
    fn a_picture_that_has_been_deleted_is_no_longer_one_of_this_session_s() {
        // Out of the gallery is out of everything: the list the page draws, and the names a
        // request may ask for.
        let shared = a_session();
        shared.change(|session| session.gallery.push(a_picture(1)));
        shared.change(|session| session.gallery.push(a_picture(2)));
        let before = shared.session().revision;

        shared.forget_picture("waifu-0001.png");
        assert!(shared.wrote("waifu-0001.png").is_none());
        assert!(shared.wrote("waifu-0002.png").is_some());
        // News, so that a second tab open on the same address stops showing it too.
        assert!(shared.session().revision > before);

        // Deleting what is not there leaves the rest alone rather than failing.
        shared.forget_picture("waifu-0001.png");
        assert_eq!(shared.session().gallery.len(), 1);
    }

    #[test]
    fn a_stop_is_only_reported_while_there_is_something_to_stop() {
        // The flag stays set between a run that was stopped and the next one that clears it, and
        // a bar saying "stopping" over an idle program would be reporting the flag rather than
        // what is happening.
        let shared = a_session();
        shared.interrupt();
        assert!(shared.interrupted());
        assert_eq!(shared.progress()["interrupting"], false);

        shared.change(|session| {
            session.doing = Doing::Drawing(Run {
                progress: GenerationProgress::Encoding,
                steps: 8,
                started: Instant::now(),
            })
        });
        assert_eq!(shared.progress()["interrupting"], true);

        shared.carry_on();
        assert!(!shared.interrupted());
    }

    #[test]
    fn the_picture_to_draw_from_is_held_until_it_is_let_go_of() {
        let shared = a_session();
        assert!(!shared.holding_a_picture());

        shared.hold_upload(vec![1, 2, 3]);
        assert!(shared.holding_a_picture());
        assert_eq!(shared.upload(), Some(vec![1, 2, 3]));

        // Held where the page can see it, so that a picture named on the command line is one an
        // already-open page finds out about.
        assert_eq!(shared.describe(Value::Null)["holding_a_picture"], true);

        shared.forget_upload();
        assert!(!shared.holding_a_picture());
        assert_eq!(shared.upload(), None);
    }

    #[test]
    fn the_whole_state_says_what_built_it_and_where_runs_go() {
        // Whichever screen is up when something goes wrong is the one that ends up in the
        // screenshot, and a screenshot that cannot say which code it came from is worth much less
        // than one that can.
        let described = a_session().describe(Value::Null);
        assert_eq!(described["built_from"], crate::cli::REVISION);
        assert_eq!(described["device"], "cpu");
        assert!(described["model"].is_null());
        assert_eq!(described["gallery"].as_array().unwrap().len(), 0);
    }
}
