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
use crate::cli::task::Task;
use crate::{GenerationDefaults, GenerationProgress, SpeechDefaults, SpeechProgress};

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
            "guidance": showable(self.guidance),
            "model": self.model,
            "strength": self.from_image.map(showable),
            "seconds": self.elapsed.as_secs_f64(),
            "parameters": self.parameters(),
        })
    }
}

/// A finished clip: the file it was written to, and what it was asked for.
///
/// The same thing beside a clip that [`Picture::parameters`] is beside a picture, and kept for
/// the same reason: a reading that came out well is asked about later, and by then the seed and
/// the speed are the first two things nobody remembers.
pub struct Clip {
    pub file: String,
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

        format!("{}\n{settings}", self.text)
    }

    fn json(&self) -> Value {
        json!({
            "file": self.file,
            "text": self.text,
            "length": self.seconds,
            "rate": self.rate,
            "speed": showable(self.speed),
            "temperature": showable(self.temperature),
            "seed": self.seed,
            "voice": self.voice,
            "from_a_recording": self.from_a_recording,
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
    /// What the page is set to speak with. Filled in before any page opens, because the boxes on
    /// the speech tab are a voice's numbers and there is nothing else to fill them from.
    pub voice: Option<Spoken>,
    pub doing: Doing,
    pub note: Option<Note>,
    /// What has been drawn this session, newest first. The files themselves are on the disk where
    /// the program was started; this is the list of what may be asked for by name, which is also
    /// what keeps a request from asking for a file that was never written here.
    pub gallery: Vec<Picture>,
    /// What has been said this session, newest first.
    ///
    /// Beside the pictures rather than mixed in with them. It is one gallery in the sense that
    /// matters -- one list of what this session made, and one place that decides which file names
    /// a request may ask for -- and two lists in the sense that also matters, which is that a
    /// clip has a length and a picture has a shape and neither has the other. The screen shows
    /// one kind at a time, so a single list would be a list the page filtered on every draw.
    pub clips: Vec<Clip>,
    /// Counted up every time anything here changes, so that a browser polling for news can tell
    /// "nothing has happened" from "everything happened and finished" without comparing the whole
    /// of it. A run that starts and ends between two polls still moves this.
    pub revision: u64,
    /// The [`Session::revision`] the held picture last changed at, and the held recording.
    ///
    /// What the page puts on the end of the address it shows each one from, so that the browser
    /// fetches it again when it is a different picture or recording and not otherwise. It used
    /// the session's own revision for that, which moves on every token of a reading and every
    /// step of a drawing -- and the recording's player reloaded twice a second for as long as a
    /// run went on, which is what somebody listening to it noticed.
    pub picture_revision: u64,
    pub recording_revision: u64,
}

impl Session {
    fn new() -> Session {
        Session {
            model: None,
            voice: None,
            doing: Doing::Nothing,
            note: None,
            gallery: Vec::new(),
            clips: Vec::new(),
            revision: 0,
            picture_revision: 0,
            recording_revision: 0,
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
    /// The recording a reading should sound like, as the WAV bytes it arrived as.
    ///
    /// Its own lock beside the picture's, for the same reason that one is not in the session: it
    /// is megabytes and the session is copied into JSON several times a second. Separate from the
    /// picture rather than one box for both, because they are two different runs' inputs and
    /// dropping a voice should not throw away the picture somebody is about to redraw.
    ///
    /// WAV rather than whatever was dropped on the page. The browser has a decoder for every
    /// format it will play and this program has one for none of them, so the page decodes what it
    /// is given and posts the samples -- which means what arrives here is always something
    /// [`crate::wav::read`] can read.
    recording: Mutex<Option<Vec<u8>>>,
    /// Where runs go, chosen in the terminal before the page opened. Not something the page can
    /// change: weights live on the device they were read onto, and moving them is a question the
    /// terminal asks, before anything has been read.
    runtime: Runtime,
    /// What the page is for, chosen in the terminal as well: which tab it opens on, and which of
    /// the two kinds of model this session runs.
    task: Task,
}

impl Shared {
    pub fn new(task: Task, runtime: Runtime) -> Shared {
        Shared {
            session: Mutex::new(Session::new()),
            cancel: AtomicBool::new(false),
            working: AtomicBool::new(false),
            upload: Mutex::new(None),
            recording: Mutex::new(None),
            runtime,
            task,
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
        self.change(|session| session.picture_revision = session.revision + 1);
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
        self.change(|session| session.picture_revision = session.revision + 1);
    }

    /// Whether there is a picture to draw from, which is the part of it the page is told about.
    pub fn holding_a_picture(&self) -> bool {
        self.held().is_some()
    }

    fn held(&self) -> MutexGuard<'_, Option<Vec<u8>>> {
        self.upload.lock().unwrap_or_else(|held| held.into_inner())
    }

    pub fn hold_recording(&self, bytes: Vec<u8>) {
        *self.recorded() = Some(bytes);
        self.change(|session| session.recording_revision = session.revision + 1);
    }

    /// A copy of the recording being held, for the run that is about to be given it.
    pub fn recording(&self) -> Option<Vec<u8>> {
        self.recorded().clone()
    }

    pub fn forget_recording(&self) {
        *self.recorded() = None;
        self.change(|session| session.recording_revision = session.revision + 1);
    }

    /// Whether there is a recording to sound like, which is the part of it the page is told.
    pub fn holding_a_recording(&self) -> bool {
        self.recorded().is_some()
    }

    fn recorded(&self) -> MutexGuard<'_, Option<Vec<u8>>> {
        self.recording
            .lock()
            .unwrap_or_else(|held| held.into_inner())
    }

    pub fn runtime(&self) -> Runtime {
        self.runtime
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
            // Told apart from drawing, because the two buttons that stop them are on different
            // tabs and each one is live only while its own kind of run is going.
            "speaking": matches!(session.doing, Doing::Speaking(_)),
            // And told apart from both, because the same button stops a fetch but what it says
            // it will do is not the same thing: a run that is stopped keeps the picture so far,
            // and a fetch that is stopped keeps the packages so far.
            "fetching": matches!(session.doing, Doing::Fetching(_)),
            "fraction": session.doing.fraction(),
            "doing": session.doing.words(),
            "seconds": match &session.doing {
                Doing::Drawing(run) => Some(run.started.elapsed().as_secs_f64()),
                Doing::Speaking(say) => Some(say.started.elapsed().as_secs_f64()),
                _ => None,
            },
            // Only while there is something to stop. The flag stays set between a run that was
            // stopped and the next one that clears it, and a bar saying "stopping" over an idle
            // program would be reporting the flag rather than what is happening.
            //
            // A fetch is one of the things there is to stop; reading the weights onto the device
            // is not, and the flag set during that one is a stop that will be taken by whatever
            // comes after it rather than by the read.
            "interrupting": self.interrupted()
                && matches!(
                    session.doing,
                    Doing::Drawing(_) | Doing::Speaking(_) | Doing::Fetching(_)
                ),
        })
    }

    /// The whole of it, which is what a browser asks for when it opens and after anything lands.
    pub fn describe(&self) -> Value {
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

        let voice = session.voice.as_ref().map(|voice| {
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
                // The sentence the speech tab is built around. Null for a real model, which is
                // how the warning on that tab goes away by itself the day there is one.
                "not_a_voice_because": voice.not_a_voice_because,
            })
        });

        json!({
            "revision": session.revision,
            "built_from": crate::cli::REVISION,
            "task": self.task.name(),
            "device": self.runtime.name(),
            "holding_a_picture": self.holding_a_picture(),
            "holding_a_recording": self.holding_a_recording(),
            "picture_revision": session.picture_revision,
            "recording_revision": session.recording_revision,
            "model": model,
            "voice": voice,
            "note": session.note.as_ref().map(|note| json!({ "said": note.said, "bad": note.bad })),
            "gallery": session.gallery.iter().map(Picture::json).collect::<Vec<_>>(),
            "clips": session.clips.iter().map(Clip::json).collect::<Vec<_>>(),
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

    /// Where a clip the browser asks for by name is, if this session said it.
    ///
    /// The same door as a picture and the same answer through it: a name that is not in the list
    /// of what this session made is not a file this program will open, whatever it is a path to.
    /// Two lists and two doors rather than one of each, so that a request for a picture cannot
    /// reach a clip and be handed one under the wrong content type.
    pub fn spoke(&self, file: &str) -> Option<PathBuf> {
        let session = self.session();
        session
            .clips
            .iter()
            .find(|clip| clip.file == file)
            .map(|clip| PathBuf::from(&clip.file))
    }

    /// Takes a clip out of the list, which is what makes its name unaskable-for again.
    pub fn forget_clip(&self, file: &str) {
        self.change(|session| session.clips.retain(|clip| clip.file != file));
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
        Shared::new(Task::Txt2Img, DeviceOption::Cpu.resolve())
    }

    fn a_clip(seed: u64) -> Clip {
        Clip {
            file: format!("waifu-{seed:04}.wav"),
            text: "hello there".to_string(),
            seconds: 1.5,
            rate: 24_000,
            speed: 1.0,
            temperature: 0.8,
            seed,
            voice: "tones".to_string(),
            from_a_recording: false,
            elapsed: Duration::from_millis(120),
        }
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

        // A fetch is the other thing there is to stop, and the page is told which of the two it
        // is looking at: the same button stops both, and what it promises to keep is not the
        // same.
        shared.change(|session| {
            session.doing = Doing::Fetching(Fetch {
                model: "sdxl:base".to_string(),
                hub: None,
                file: "unet.safetensors".to_string(),
                done: 1,
                total: Some(2),
                part: 1,
                parts: 3,
            })
        });
        assert_eq!(shared.progress()["interrupting"], true);
        assert_eq!(shared.progress()["fetching"], true);
        assert_eq!(shared.progress()["drawing"], false);

        // Reading the weights onto the device is neither: it is one call into the tensor library
        // that returns when it returns, so there is nothing there to ask to stop and the bar does
        // not offer to.
        shared.change(|session| {
            session.doing = Doing::Reading {
                model: "sdxl:base".to_string(),
            }
        });
        assert_eq!(shared.progress()["interrupting"], false);
        assert_eq!(shared.progress()["fetching"], false);

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
        assert_eq!(shared.describe()["holding_a_picture"], true);

        shared.forget_upload();
        assert!(!shared.holding_a_picture());
        assert_eq!(shared.upload(), None);
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

    #[test]
    fn only_what_this_session_said_can_be_asked_for_by_name_either() {
        // The same rule the pictures are under, through its own door: a request for a picture
        // cannot reach a clip and be handed one under the wrong content type.
        let shared = a_session();
        shared.change(|session| session.clips.push(a_clip(1)));
        shared.change(|session| session.gallery.push(a_picture(1)));

        assert!(shared.spoke("waifu-0001.wav").is_some());
        assert!(shared.spoke("waifu-0002.wav").is_none());
        assert!(shared.spoke("../../etc/passwd").is_none());
        assert!(shared.spoke("/etc/passwd").is_none());

        // And neither list is a door into the other.
        assert!(shared.spoke("waifu-0001.png").is_none());
        assert!(shared.wrote("waifu-0001.wav").is_none());
    }

    #[test]
    fn a_clip_that_has_been_deleted_is_no_longer_one_of_this_session_s() {
        let shared = a_session();
        shared.change(|session| session.clips.push(a_clip(1)));
        shared.change(|session| session.clips.push(a_clip(2)));
        let before = shared.session().revision;

        shared.forget_clip("waifu-0001.wav");
        assert!(shared.spoke("waifu-0001.wav").is_none());
        assert!(shared.spoke("waifu-0002.wav").is_some());
        assert!(shared.session().revision > before);

        // Deleting what is not there leaves the rest alone rather than failing.
        shared.forget_clip("waifu-0001.wav");
        assert_eq!(shared.session().clips.len(), 1);
    }

    /// What the page puts on the end of the recording's address moves when the recording does and
    /// not when anything else does. A reading reports every token, and the player beside it
    /// reloaded on every one while the address carried the session's revision instead.
    #[test]
    fn a_held_file_is_only_news_when_it_changes() {
        let shared = a_session();

        shared.hold_recording(vec![1]);
        shared.hold_upload(vec![2]);
        let (recording, picture) = {
            let session = shared.session();
            (session.recording_revision, session.picture_revision)
        };

        // A run's progress, many times over.
        for _ in 0..5 {
            shared.change(|session| session.note = None);
        }
        let described = shared.describe();
        assert_eq!(described["recording_revision"], recording);
        assert_eq!(described["picture_revision"], picture);

        // Each moves with its own file, and neither with the other's.
        shared.hold_recording(vec![3]);
        assert!(shared.session().recording_revision > recording);
        assert_eq!(shared.session().picture_revision, picture);

        let recording = shared.session().recording_revision;
        shared.forget_upload();
        assert!(shared.session().picture_revision > picture);
        assert_eq!(shared.session().recording_revision, recording);
    }

    #[test]
    fn the_recording_and_the_picture_are_two_boxes_rather_than_one() {
        // They are two different runs' inputs. Dropping a recording should not throw away the
        // picture somebody is about to redraw, and the other way round.
        let shared = a_session();
        assert!(!shared.holding_a_recording());

        shared.hold_upload(vec![1, 2, 3]);
        shared.hold_recording(vec![4, 5, 6]);
        assert_eq!(shared.upload(), Some(vec![1, 2, 3]));
        assert_eq!(shared.recording(), Some(vec![4, 5, 6]));
        assert_eq!(shared.describe()["holding_a_recording"], true);

        shared.forget_recording();
        assert!(!shared.holding_a_recording());
        assert!(shared.holding_a_picture());
    }

    #[test]
    fn a_stop_is_reported_while_something_is_being_said_as_well() {
        // The button is on both tabs and stops whichever kind of run is going.
        let shared = a_session();
        shared.interrupt();
        assert_eq!(shared.progress()["interrupting"], false);

        shared.change(|session| {
            session.doing = Doing::Speaking(Say {
                progress: SpeechProgress::Reading,
                expected: 0,
                started: Instant::now(),
            })
        });
        assert_eq!(shared.progress()["interrupting"], true);
        assert_eq!(shared.progress()["speaking"], true);
        assert_eq!(shared.progress()["drawing"], false);
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

        let described = a_session().describe();
        assert_eq!(described["voice"], Value::Null);

        let shared = a_session();
        shared.change(|session| session.clips.push(a_clip(1)));
        let clip = &shared.describe()["clips"][0];
        assert_eq!(clip["temperature"], 0.8);
        assert_eq!(clip["speed"], 1.0);
    }

    #[test]
    fn the_whole_state_says_what_built_it_and_where_runs_go() {
        // Whichever screen is up when something goes wrong is the one that ends up in the
        // screenshot, and a screenshot that cannot say which code it came from is worth much less
        // than one that can.
        let described = a_session().describe();
        assert_eq!(described["built_from"], crate::cli::REVISION);
        assert_eq!(described["device"], "cpu");
        assert!(described["model"].is_null());
        assert_eq!(described["gallery"].as_array().unwrap().len(), 0);
    }
}
