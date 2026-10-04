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

//! The thread that owns the model.
//!
//! A tensor never leaves the thread that made it, so the model cannot be touched from a request:
//! a request arrives on whichever of the server's threads was free, and there are several of
//! them. What the requests do instead is keep a job and put its id in line, and this one thread
//! takes the ids off the line in order, runs each, and says what it is doing in [`Shared`] as it
//! goes.
//!
//! It is also why the answer to "draw this" is where the job is rather than a picture. A run is
//! minutes long; a request that waited for one would hold a socket open across it, and the
//! browser would have nothing to show in the meantime.

use std::io::Cursor;
use std::ops::ControlFlow;
use std::path::{Path, PathBuf};
use std::time::Instant;

use serde_json::Value;

use crate::cli::args::Runtime;
use crate::cli::hub;
use crate::cli::webui::log;
use crate::cli::webui::state::{Chosen, Clip, Doing, Fetch, Picture, Run, Say, Shared, Spoken};
use crate::cli::webui::store::{self, Kind};
use crate::cosyvoice3::{self, CosyVoice3};
use crate::flint::Tensor;
use crate::indextts::{self, IndexTts};
use crate::wav;
use crate::{
    from_rgb8, to_rgb8, Anima, GenerationDefaults, GenerationOptions, GenerationProgress, Krea2,
    Manifest, QwenImage, Sdxl, SpeechOptions, SpeechProgress, Tones, Voice,
};

type Error = Box<dyn std::error::Error>;

/// The sizes offered for a model that does not name its own, which are the ones SDXL was trained
/// at. Every one of them is a multiple of the 32 pixels the U-Net's own halvings need.
pub const SIZES: &[(i32, i32)] = &[
    (512, 512),
    (640, 640),
    (768, 768),
    (896, 896),
    (1024, 1024),
    (832, 1216),
    (1216, 832),
    (768, 1344),
    (1344, 768),
];

/// Why Anima cannot be handed a picture to start from. True of the kind rather than of any one
/// package, which is what lets it be said before a package has been looked at.
const ANIMA_DRAWS_FROM_NO_PICTURE: &str = "Anima cannot start from a picture yet: its package \
     carries an image encoder, but the layer that reads one is not written";

/// And why Krea 2 cannot either, which is the same reason: it draws through the same Qwen-Image
/// autoencoder, and the half of it that reads a picture has no layer to run it.
const KREA2_DRAWS_FROM_NO_PICTURE: &str = "Krea 2 cannot start from a picture yet: its package \
     carries an image encoder, but the layer that reads one is not written";

/// And why Qwen-Image 2.1 cannot, for a different reason: its model edits pictures by reading
/// them as well as starting from them, and neither half of that is written here.
const QWEN_IMAGE_DRAWS_FROM_NO_PICTURE: &str = "Qwen-Image 2.1 cannot start from a picture yet: \
     it edits by reading the picture through its text encoder too, and that is not written";

/// What the built-in stand-in voice is called where a name is asked for.
///
/// A constant rather than a catalogue entry, because it is not published anywhere and cannot be
/// fetched: it is in the binary. Only ever had by asking for it with `-m tones` -- it makes a
/// noise where the syllables are, which is for checking the page and not for listening to.
pub const TONES: &str = "tones";

/// How many sizes a model has to name before its list is used instead of the one above.
///
/// One size is not a choice and two is barely one. A model that names fewer has not replaced the
/// list, it has emptied it, and the built-in one is the better thing to offer.
const ENOUGH_SIZES: usize = 3;

/// What the worker reads before the server is up: the one model this program serves.
pub enum Load {
    Model(String),
    Voice(String),
}

/// Reads the model, then runs jobs off the queue for as long as the program runs.
///
/// The model is read once, before the page is served, and it stays on the device until the
/// program stops. That is what the program is for: it is started for one model, chosen in the
/// terminal, and every job anybody posts is run with it. Nothing a request does takes it off the
/// card -- a stop included. A stop ends one job, and the next one in line starts straight away.
pub fn work(shared: &Shared, load: Load) {
    let mut model: Option<Model> = None;
    let mut voice: Option<Box<dyn Voice>> = None;

    match load {
        Load::Model(asked) => {
            read_model(shared, &asked, &mut model);
        }
        Load::Voice(asked) => {
            read_voice(shared, &asked, &mut voice);
        }
    }
    shared.reading(false);

    loop {
        let job = shared.next();
        run(shared, model.as_ref(), voice.as_deref(), &job);
        shared.finished();
    }
}

/// How a job ended, where it did not fail.
enum Ran {
    /// With a file, and a description of it to keep beside it.
    Made {
        bytes: Vec<u8>,
        made: Value,
    },
    Stopped,
}

/// Runs one job, already marked as running, and writes down how it went.
fn run(shared: &Shared, model: Option<&Model>, voice: Option<&dyn Voice>, job: &store::Job) {
    let store = shared.store();
    let id = job.id.as_str();

    // Asked to stop in the moment between being handed over and here.
    let ran = if shared.interrupted() {
        Ok(Ran::Stopped)
    } else {
        match (job.kind, model, voice) {
            (Kind::Image, Some(model), _) => draw(shared, model, job),
            (Kind::Speech, _, Some(voice)) => say(shared, voice, job),
            // Not reachable through the API, which refuses a job of the other kind -- but a
            // job is a job, and one with nothing to run it is said to have failed.
            (Kind::Image, None, _) => Err("this program was not started to draw".into()),
            (Kind::Speech, _, None) => Err("this program was not started to speak".into()),
        }
    };

    match ran {
        Ok(Ran::Made { bytes, made }) => {
            if let Err(error) = store.done(id, &bytes, made) {
                log::line(format_args!("error: job {id}: {error}"));
                store.failed(id, format!("what it made could not be kept: {error}"));
            }
        }
        Ok(Ran::Stopped) => {
            log::line(format_args!("job {id} stopped where it was"));
            store.cancelled(id);
        }
        Err(error) => {
            log::line(format_args!("error: job {id}: {error}"));
            store.failed(id, error.to_string());
        }
    }
}

/// A job's settings, read back out of what was kept. Every one is there: the request was given
/// the model's defaults for whatever it left out before it was kept.
fn generation_options(asked: &Value) -> GenerationOptions {
    let whole = |field: &str| asked.get(field).and_then(Value::as_i64).unwrap_or(0) as i32;
    let decimal = |field: &str| asked.get(field).and_then(Value::as_f64).unwrap_or(0.0) as f32;
    GenerationOptions {
        width: whole("width"),
        height: whole("height"),
        num_steps: whole("steps"),
        guidance_scale: decimal("guidance"),
        negative_prompt: text(asked, "negative"),
        seed: seed(asked),
        strength: decimal("strength"),
    }
}

fn speech_options(asked: &Value) -> SpeechOptions {
    let decimal = |field: &str| asked.get(field).and_then(Value::as_f64).unwrap_or(0.0) as f32;
    SpeechOptions {
        speed: decimal("speed"),
        temperature: decimal("temperature"),
        seed: seed(asked),
    }
}

fn text(asked: &Value, field: &str) -> String {
    asked
        .get(field)
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string()
}

/// A seed is sixty-four bits and is kept as a string: a JSON number is a double.
fn seed(asked: &Value) -> Option<u64> {
    asked.get("seed").and_then(Value::as_str)?.parse().ok()
}

/// The bytes of the upload a job's settings name under `field`, if they name one.
fn upload_named(shared: &Shared, asked: &Value, field: &str) -> Result<Option<Vec<u8>>, Error> {
    let Some(id) = asked.get(field).and_then(Value::as_str) else {
        return Ok(None);
    };
    let (_, path) = shared
        .store()
        .upload_file(id)
        .ok_or_else(|| format!("the upload it was to start from ({id}) has been deleted"))?;
    Ok(Some(std::fs::read(path)?))
}

/// Reads the model `asked` names into `model`, and says on screen how it went. True where the
/// weights are now in memory.
///
/// Whatever was in `model` has already been let go of by here. The cost of that is that a name
/// which turns out to be wrong leaves nothing loaded rather than leaving what was there; the
/// gain is that two models are never on one card at once, and a swap that runs out of memory
/// would lose the model that was working as well as the one that was asked for.
fn read_model(shared: &Shared, asked: &str, model: &mut Option<Model>) -> bool {
    match load(shared, asked) {
        Ok((read, opened)) => {
            *model = Some(opened);
            shared.change(|world| {
                // Only where it is still the chosen one. A choice made while this was reading is
                // the newer answer to "what is the page set to draw with", and describing the
                // model that has just been read over the top of it would undo somebody's click.
                if world.model.as_ref().is_none_or(|one| one.name == read.name) {
                    world.model = Some(read);
                }
                world.doing = Doing::Nothing;
            });
            shared.say("ready", false);
            true
        }
        Err(error) => {
            shared.change(|world| world.doing = Doing::Nothing);
            // A stop is not a failure. It is the button doing what it says, and what it has to
            // say for itself -- which package is left on the disk, what the next fetch will not
            // have to bring down again -- belongs on the bar in the ordinary way rather than as a
            // complaint about something going wrong.
            shared.say(error.to_string(), !hub::stopped(&error));
            false
        }
    }
}

/// The voice `asked` names, as the screen describes it, without reading any of it.
///
/// What [`look_at`] is for a picture model. [`TONES`] is [`Tones`], built into the binary, which
/// can be constructed and asked about itself for nothing. Anything else is a package -- a manifest
/// on the disk, or a published name -- and what can be said of one before minutes of reading is
/// what its kind says: IndexTTS-2.5's rate and starting values, whether it is on the disk, and
/// that it is not in memory. The read itself is [`read_voice`], at the first run that wants it,
/// and it says so if the package turns out to be something else.
pub fn look_at_voice(asked: &str) -> Spoken {
    if asked == TONES {
        return describe_voice(asked, &Tones::new(), false);
    }

    let path = on_disk(asked);
    let (kind_name, defaults, rate) = match path.as_deref().and_then(voice_kind) {
        Some(VoiceKind::CosyVoice3) => (CosyVoice3::NAME, CosyVoice3::DEFAULTS, cosyvoice3::RATE),
        _ => (IndexTts::NAME, IndexTts::DEFAULTS, indextts::RATE),
    };

    Spoken {
        name: asked.to_string(),
        // The catalogue's name for it where it has one, and the kind's where it was named by path.
        full_name: hub::full_name(asked).unwrap_or(kind_name).to_string(),
        on_disk: path.is_some(),
        in_memory: false,
        defaults,
        rate,
        no_likeness_because: None,
        not_a_voice_because: None,
    }
}

/// Whether `asked` is a voice: the stand-in, a published voice, or a manifest on the disk whose
/// `model.type` is one of the speech models. What `-m` alone is taken to mean text2speech by.
pub fn is_a_voice(asked: &str) -> bool {
    asked == TONES
        || hub::is_voice(asked)
        || on_disk(asked).as_deref().and_then(voice_kind).is_some()
}

/// The speech models a package can be.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum VoiceKind {
    IndexTts,
    CosyVoice3,
}

/// Which speech model the manifest at `path` describes, by its `model.type`; `None` where it
/// cannot be read or is not a speech model this reads.
fn voice_kind(path: &std::path::Path) -> Option<VoiceKind> {
    let manifest = Manifest::open(path).ok()?;
    match manifest.section("model").ok()?.get_str("type").ok()? {
        IndexTts::MODEL_TYPE => Some(VoiceKind::IndexTts),
        CosyVoice3::MODEL_TYPE => Some(VoiceKind::CosyVoice3),
        _ => None,
    }
}

/// What a voice says about itself, in the words the screen shows, under the name it was asked for.
fn describe_voice(name: &str, voice: &dyn Voice, in_memory: bool) -> Spoken {
    Spoken {
        name: name.to_string(),
        // The catalogue's name first, so the button does not change its wording when it is read.
        full_name: hub::full_name(name).unwrap_or(voice.name()).to_string(),
        // Built in, or just read off the disk: either way there is nothing left to fetch.
        on_disk: true,
        in_memory,
        defaults: voice.defaults(),
        rate: voice.rate(),
        no_likeness_because: voice.no_likeness_because().map(str::to_string),
        not_a_voice_because: voice.not_a_voice_because().map(str::to_string),
    }
}

/// Whether the voice `asked` names has weights behind it -- which is to say, whether it can share
/// the card with a picture model. [`Tones`] is two floats and can; a package cannot.
fn voice_holds_weights(asked: &str) -> bool {
    asked != TONES
}

/// Reads the voice `asked` names into `voice`, and says on screen how it went. True where it is
/// now ready to speak with.
///
/// The mirror of [`read_model`]: a package is fetched if it is a name and read onto the device,
/// reporting through the same [`Doing::Fetching`] and [`Doing::Reading`] the picture models
/// report through, and nothing above this function finds out which kind of voice it was.
fn read_voice(shared: &Shared, asked: &str, voice: &mut Option<Box<dyn Voice>>) -> bool {
    let read: Result<Box<dyn Voice>, Error> = match asked == TONES {
        true => Ok(Box::new(Tones::new())),
        false => load_voice(shared, asked),
    };

    match read {
        Ok(read) => {
            let described = describe_voice(asked, read.as_ref(), true);
            *voice = Some(read);

            shared.change(|world| {
                world.voice = Some(described);
                world.doing = Doing::Nothing;
            });
            if voice_holds_weights(asked) {
                shared.say("ready", false);
            }
            true
        }
        Err(error) => {
            shared.change(|world| world.doing = Doing::Nothing);
            // A stop is not a failure -- see `read_model`, which says the same thing.
            shared.say(error.to_string(), !hub::stopped(&error));
            false
        }
    }
}

/// Fetches the voice package `asked` names, if it is a name, and reads it onto the device.
fn load_voice(shared: &Shared, asked: &str) -> Result<Box<dyn Voice>, Error> {
    // The same rule `load` keeps for a picture model: the name on the screen is the tail of a
    // path rather than all of it.
    let name = match asked.rsplit_once('/') {
        Some((_, file)) if !file.is_empty() => file.to_string(),
        _ => asked.to_string(),
    };

    shared.change(|world| {
        world.doing = Doing::Fetching(Fetch {
            model: name.clone(),
            hub: None,
            file: String::new(),
            done: 0,
            total: None,
            part: 0,
            parts: 0,
        })
    });

    let path = hub::resolve_reporting(
        asked,
        &mut |progress| report_fetch(shared, &name, progress),
        &|| shared.interrupted(),
    )?;

    shared.change(|world| {
        world.doing = Doing::Reading {
            model: name.clone(),
        }
    });

    let runtime = shared.runtime();
    let manifest = Manifest::open(&path)?;
    let (device, residency) = (runtime.device(), runtime.residency());

    // Whichever speech model the package says it is. Anything else is IndexTTS-2.5's to refuse,
    // which it does by naming the kind it found.
    Ok(match voice_kind(&path) {
        Some(VoiceKind::CosyVoice3) => {
            Box::new(CosyVoice3::from_manifest(device, residency, &manifest)?)
        }
        _ => Box::new(IndexTts::from_manifest(device, residency, &manifest)?),
    })
}

/// What a kind of model implies for the screen before any of its weights are read: what to ask
/// it for, what walks the noise back, why it cannot start from a picture, and whether guidance is
/// a thing to offer at all.
///
/// One table for the three kinds, read twice -- once from a name that has not been fetched yet,
/// and once from the `type` of a manifest that is already on the disk.
///
/// The last of the four is false for Krea 2 for the same reason its defaults are eight steps at a
/// guidance of one: the only release of it this exports is the distilled one, and the distilled
/// one has no second pass. A package's own `suggested:` overrides this, so the day there is an
/// undistilled one it says `takes_guidance: "true"` and gets the dial back.
fn about(kind: &str) -> (GenerationDefaults, &'static str, Option<&'static str>, bool) {
    match kind {
        Anima::MODEL_TYPE => (
            Anima::DEFAULTS,
            "Flow match Euler",
            Some(ANIMA_DRAWS_FROM_NO_PICTURE),
            true,
        ),
        Krea2::MODEL_TYPE => (
            Krea2::DEFAULTS,
            "Flow match Euler",
            Some(KREA2_DRAWS_FROM_NO_PICTURE),
            false,
        ),
        // Sampled without guidance by default, and able to take it: the reference's
        // `true_cfg_scale` is a second pass on a negative prompt, which is this runtime's.
        QwenImage::MODEL_TYPE => (
            QwenImage::DEFAULTS,
            "Flow match Euler",
            Some(QWEN_IMAGE_DRAWS_FROM_NO_PICTURE),
            true,
        ),
        // Everything else is SDXL, which is what `Model::from_manifest` decides too.
        _ => (GenerationDefaults::default(), "Euler", None, true),
    }
}

/// What can be said about a model that has not been read, which is what a choice is made from.
///
/// Guessed from the name where there is nothing else to go on, and read out of the manifest where
/// the package is already here. Neither is the last word -- the moment a run reads the package,
/// [`load`] describes it again out of what is actually in there, wrong guesses included.
pub fn look_at(asked: &str) -> Chosen {
    // The kinds are told apart by the catalogue's own naming. A package somebody exported
    // themselves and named by path is taken as SDXL until the manifest below says otherwise,
    // because the one thing this decides -- whether image to image is offered -- is a thing SDXL
    // packages do.
    let guessed = match asked {
        name if name.starts_with("anima") => Anima::MODEL_TYPE,
        name if name.starts_with("krea") => Krea2::MODEL_TYPE,
        name if name.starts_with("qwen") => QwenImage::MODEL_TYPE,
        _ => "",
    };
    let (defaults, sampler, no_picture, takes_guidance) = about(guessed);

    let mut chosen = Chosen {
        name: asked.to_string(),
        // The catalogue's name for it, or the file name where a path was given: the whole path
        // does not fit on screen and its tail is the part that identifies it.
        full_name: hub::full_name(asked)
            .map(str::to_string)
            .unwrap_or_else(|| match asked.rsplit_once('/') {
                Some((_, file)) if !file.is_empty() => file.to_string(),
                _ => asked.to_string(),
            }),
        on_disk: hub::is_cached(asked),
        in_memory: false,
        defaults,
        sizes: SIZES.to_vec(),
        sampler,
        // Not known until the package is read: what a card suggests be typed is written in it.
        suggested_prompt: None,
        suggested_avoid: None,
        no_picture_because: no_picture.map(str::to_string),
        takes_guidance,
    };

    // And where the package is already here, what it says about itself. The manifest is a few
    // kilobytes of text beside gigabytes of weights, and reading it is what keeps the numbers in
    // the boxes from changing under somebody later: a model described twice, once from its name
    // and once from its package, is a model whose settings move while they are being used.
    //
    // The weights are still not touched. Whether this one can start from a picture is the guess
    // above until a run opens the package.
    let Some(path) = on_disk(asked) else {
        return chosen;
    };
    let Ok(manifest) = Manifest::open(path) else {
        return chosen;
    };

    // What kind it is, which the manifest states and a name only hints at. It matters most for a
    // package named by its path, which is every package somebody exported themselves: `krea2` is
    // not in the catalogue at all, so this is the only place its kind is ever read.
    if let Ok(kind) = manifest
        .section(Sdxl::MODEL_SECTION)
        .and_then(|section| section.get_str("type").map(str::to_string))
    {
        let (defaults, sampler, no_picture, takes_guidance) = about(&kind);
        chosen.defaults = defaults;
        chosen.sampler = sampler;
        chosen.no_picture_because = no_picture.map(str::to_string);
        chosen.takes_guidance = takes_guidance;
    }

    let suggested = manifest.suggested();
    chosen.defaults = suggested.over(chosen.defaults);
    chosen.suggested_prompt = suggested.prompt.clone();
    chosen.suggested_avoid = suggested.avoid.clone();
    // Only where it says. A manifest written before there was a key for this is silent, and
    // silence leaves the line above's answer alone rather than arguing with it.
    chosen.takes_guidance = suggested.takes_guidance.unwrap_or(chosen.takes_guidance);
    if suggested.sizes.len() >= ENOUGH_SIZES {
        chosen.sizes = suggested.sizes.clone();
    }

    chosen
}

/// The manifest of a model that can be read without fetching anything: one already in the cache,
/// or one that was named by the path of its manifest in the first place.
fn on_disk(asked: &str) -> Option<PathBuf> {
    hub::cached_manifest(asked).or_else(|| {
        let path = PathBuf::from(asked);
        path.is_file().then_some(path)
    })
}

/// Fetches and reads the model `asked` names, saying how it is getting on as it goes.
fn load(shared: &Shared, asked: &str) -> Result<(Chosen, Model), Error> {
    // The name the screen shows. As typed when it was named, because that is what someone would
    // type again; the file name when a path was given, because the whole path does not fit and
    // its tail is the part that identifies it.
    let name = match asked.rsplit_once('/') {
        Some((_, file)) if !file.is_empty() => file.to_string(),
        _ => asked.to_string(),
    };

    shared.change(|world| {
        world.doing = Doing::Fetching(Fetch {
            model: name.clone(),
            hub: None,
            file: String::new(),
            done: 0,
            total: None,
            part: 0,
            parts: 0,
        })
    });

    // Stoppable from here to the last byte of the last package. What follows it is not: reading
    // the weights onto the device is one call into the tensor library that returns when it
    // returns, and the screen says which of the two is happening.
    let path = hub::resolve_reporting(
        asked,
        &mut |progress| report_fetch(shared, &name, progress),
        &|| shared.interrupted(),
    )?;

    shared.change(|world| {
        world.doing = Doing::Reading {
            model: name.clone(),
        }
    });

    // Read before the model is: this is a directory entry and one small file, and the model
    // behind it is minutes of reading. What is wrong with it is reported once, by the read below,
    // with the error that says why -- so every way this can fail leaves the boxes at their
    // defaults rather than saying anything of its own.
    let suggested = Manifest::open(&path)
        .map(|manifest| manifest.suggested().clone())
        .unwrap_or_default();
    let sizes = match suggested.sizes.len() >= ENOUGH_SIZES {
        true => suggested.sizes.clone(),
        false => SIZES.to_vec(),
    };

    let opened = Model::from_manifest(&path, shared.runtime())?;
    let read = Chosen {
        defaults: suggested.over(opened.defaults()),
        sampler: opened.sampler(),
        sizes,
        suggested_prompt: suggested.prompt.clone(),
        suggested_avoid: suggested.avoid.clone(),
        // Now asked of the package rather than guessed from the name, which is the whole
        // difference between this and what was on the screen a moment ago.
        no_picture_because: opened.no_picture_because().map(str::to_string),
        // Whether to offer guidance is not restated here: `look_at` below reads it out of this
        // same manifest, and a package that is being opened is a package that is on the disk for
        // it to read.
        on_disk: true,
        in_memory: true,
        ..look_at(asked)
    };

    Ok((read, opened))
}

/// Copies what a fetch has to say into the world: how far along, which is progress and not news.
fn report_fetch(shared: &Shared, model: &str, progress: hub::Progress) {
    shared.change(|world| {
        // Whatever it said last, so that a line about one field does not blank the others: the
        // hub is said once, before the first byte, and every line after it is about a file.
        let mut fetch = match &world.doing {
            Doing::Fetching(fetch) => fetch.clone(),
            _ => Fetch {
                model: model.to_string(),
                hub: None,
                file: String::new(),
                done: 0,
                total: None,
                part: 0,
                parts: 0,
            },
        };

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
                fetch.done = done;
                fetch.total = total;
                fetch.part = part;
                fetch.parts = parts;
            }
            hub::Progress::Fetched {
                file,
                bytes,
                part,
                parts,
            } => {
                fetch.file = file.to_string();
                fetch.done = bytes;
                fetch.total = Some(bytes);
                fetch.part = part;
                fetch.parts = parts;
            }
        }

        world.doing = Doing::Fetching(fetch);
    });
}

/// Draws one job's picture, and says what came of it.
fn draw(shared: &Shared, model: &Model, job: &store::Job) -> Result<Ran, Error> {
    let prompt = text(&job.asked, "prompt");
    let options = generation_options(&job.asked);
    let from = upload_named(shared, &job.asked, "init_image")?;

    let started = Instant::now();
    shared.change(|world| {
        world.doing = Doing::Drawing(Run {
            progress: GenerationProgress::Encoding,
            steps: options.num_steps,
            started,
        })
    });
    log::line(format_args!(
        "job {}: drawing {}x{}, {} steps, seed {}{} -- \"{}\"",
        job.id,
        options.width,
        options.height,
        options.num_steps,
        options.seed.unwrap_or_default(),
        match &from {
            Some(_) => format!(", from a picture at strength {}", options.strength),
            None => String::new(),
        },
        log::clipped(&prompt, 80)
    ));

    let mut report = |progress| {
        shared.change(|world| {
            if let Doing::Drawing(run) = &mut world.doing {
                run.progress = progress;
            }
        });

        match shared.interrupted() {
            true => ControlFlow::Break(()),
            false => ControlFlow::Continue(()),
        }
    };

    // The picture to start from is read and scaled before the run, so that one that cannot be
    // read is one failure rather than a run that fails partway.
    let image = match &from {
        Some(bytes) => {
            let started_from = read_image(bytes, options.width, options.height)
                .map_err(|error| format!("the picture to draw from: {error}"))?;
            model.generate_from_image_reporting(&started_from, &prompt, &options, &mut report)?
        }
        None => model.generate_reporting(&prompt, &options, &mut report)?,
    };
    let Some(image) = image else {
        return Ok(Ran::Stopped);
    };

    let shape = image.shape();
    let picture = Picture {
        width: shape[3] as usize,
        height: shape[2] as usize,
        prompt,
        negative: options.negative_prompt.clone(),
        seed: options.seed.unwrap_or_default(),
        steps: options.num_steps,
        guidance: options.guidance_scale,
        model: shared
            .world()
            .model
            .as_ref()
            .map(|chosen| chosen.name.clone())
            .unwrap_or_default(),
        from_image: from.as_ref().map(|_| options.strength),
        elapsed: started.elapsed(),
    };
    log::line(format_args!(
        "job {}: drew it in {:.1}s",
        job.id,
        picture.elapsed.as_secs_f64()
    ));

    // With the line that says what drew it written into the file, so that a picture saved out of
    // the page still says so wherever it ends up.
    Ok(Ran::Made {
        bytes: png(&image, &picture.parameters())?,
        made: picture.json(),
    })
}

/// Reads one job's text out, and says what came of it.
fn say(shared: &Shared, voice: &dyn Voice, job: &store::Job) -> Result<Ran, Error> {
    let said = text(&job.asked, "text");
    let options = speech_options(&job.asked);
    let like = match upload_named(shared, &job.asked, "reference")? {
        Some(bytes) => Some(
            wav::read(&bytes).map_err(|error| format!("the recording to sound like: {error}"))?,
        ),
        None => None,
    };

    let started = Instant::now();
    shared.change(|world| {
        world.doing = Doing::Speaking(Say {
            progress: SpeechProgress::Reading,
            // Not known until the model has an opinion, which arrives with the first token. The
            // bar reads it as "no idea yet" and sits at the start, which is where the run is.
            expected: 0,
            started,
            sounding: None,
            ratio: world.vocoder_ratio,
        })
    });
    log::line(format_args!(
        "job {}: speaking, seed {}{} -- \"{}\"",
        job.id,
        options.seed.unwrap_or_default(),
        match &like {
            Some(_) => ", like a recording",
            None => "",
        },
        log::clipped(&said, 80)
    ));

    let mut report = |progress| {
        shared.change(|world| {
            if let Doing::Speaking(say) = &mut world.doing {
                say.progress = progress;
                // The model's own estimate, taken as it arrives rather than once: a model that
                // revises it upward mid-reading is a bar that should follow it rather than one
                // that pins itself to the first guess.
                match progress {
                    SpeechProgress::Saying { expected, .. } => say.expected = expected,
                    // Once: a model that says it is still sounding after each sentence is still
                    // in the stage that started at the first.
                    SpeechProgress::Sounding if say.sounding.is_none() => {
                        say.sounding = Some(Instant::now())
                    }
                    _ => {}
                }
            }
        });

        match shared.interrupted() {
            true => ControlFlow::Break(()),
            false => ControlFlow::Continue(()),
        }
    };

    let Some(sound) = voice.speak(&said, like.as_ref(), &options, &mut report)? else {
        return Ok(Ran::Stopped);
    };

    // What the vocoder took this time is what the bar expects of it next time.
    let finished = Instant::now();
    shared.change(|world| {
        if let Doing::Speaking(say) = &world.doing {
            if let Some(ratio) = say.measured_ratio(finished) {
                world.vocoder_ratio = ratio;
            }
        }
    });

    let clip = Clip {
        text: said,
        seconds: sound.seconds(),
        rate: voice.rate(),
        speed: options.speed,
        temperature: options.temperature,
        seed: options.seed.unwrap_or_default(),
        voice: shared
            .world()
            .voice
            .as_ref()
            .map(|spoken| spoken.name.clone())
            .unwrap_or_default(),
        from_a_recording: like.is_some(),
        elapsed: started.elapsed(),
    };
    log::line(format_args!(
        "job {}: said {:.1}s of sound in {:.1}s",
        job.id,
        clip.seconds,
        clip.elapsed.as_secs_f64()
    ));

    Ok(Ran::Made {
        bytes: wav::write(&sound),
        made: clip.json(),
    })
}

pub enum Model {
    Sdxl(Sdxl),
    Anima(Anima),
    Krea2(Krea2),
    QwenImage(QwenImage),
}

impl Model {
    /// What kind of model a manifest describes, without reading the model.
    ///
    /// One line of a file that is a couple of kilobytes. The alternative is to try each kind in
    /// turn and keep the one that does not complain, which reads gigabytes to answer a question
    /// the manifest states.
    ///
    /// Either model's `MODEL_SECTION` would do -- the block that says what a model is, is the
    /// same block whatever it turns out to be, which is the point of it.
    fn kind(manifest: &Manifest) -> crate::Result<String> {
        Ok(manifest
            .section(Sdxl::MODEL_SECTION)?
            .get_str("type")?
            .to_string())
    }

    fn from_manifest(model_path: &Path, runtime: Runtime) -> crate::Result<Model> {
        let manifest = Manifest::open(model_path)?;

        match Self::kind(&manifest)?.as_str() {
            Anima::MODEL_TYPE => Ok(Model::Anima(Anima::from_manifest(
                runtime.device(),
                runtime.residency(),
                &manifest,
            )?)),
            Krea2::MODEL_TYPE => Ok(Model::Krea2(Krea2::from_manifest(
                runtime.device(),
                runtime.residency(),
                &manifest,
            )?)),
            QwenImage::MODEL_TYPE => Ok(Model::QwenImage(QwenImage::from_manifest(
                runtime.device(),
                runtime.residency(),
                &manifest,
            )?)),
            // Everything else goes to SDXL, which says what it makes of it. A model of a kind
            // nobody here has heard of is its complaint to make rather than this one's, since it
            // is the one that knows what it can read.
            _ => Ok(Model::Sdxl(Sdxl::from_manifest(
                runtime.device(),
                runtime.residency(),
                &manifest,
            )?)),
        }
    }

    /// What this build believes about the kind of model, which is what the manifest's own
    /// suggestions are laid over. A distilled release wants eight steps at no guidance and comes
    /// out burnt at the thirty and five SDXL likes.
    fn defaults(&self) -> GenerationDefaults {
        match self {
            Model::Sdxl(_) => GenerationDefaults::default(),
            Model::Anima(_) => Anima::DEFAULTS,
            Model::Krea2(_) => Krea2::DEFAULTS,
            Model::QwenImage(_) => QwenImage::DEFAULTS,
        }
    }

    /// What walks the noise back, as the model's own reference implementation names it.
    fn sampler(&self) -> &'static str {
        match self {
            Model::Sdxl(_) => "Euler",
            Model::Anima(_) | Model::Krea2(_) | Model::QwenImage(_) => "Flow match Euler",
        }
    }

    /// Why it cannot be handed a picture to start from, or None where it can.
    ///
    /// A sentence rather than a bool, because both answers of no are somebody's next move: one
    /// is fetched away by taking the model down and bringing it back, and the other is not.
    fn no_picture_because(&self) -> Option<&'static str> {
        match self {
            // Not a property of SDXL but of the copy on this disk: the encoder's weights are in
            // packages exported since image to image landed, and in none written before it.
            Model::Sdxl(model) => (!model.draws_from_a_picture()).then_some(
                "this copy was packaged before image to image existed, so it carries no VAE \
                 encoder. Deleting it below and fetching it again brings one down",
            ),
            // The weights for one are in the package; the layer that reads them is not written.
            // See docs/anima.md and docs/krea2.md -- it is the same autoencoder and the same
            // missing half.
            Model::Anima(_) => Some(ANIMA_DRAWS_FROM_NO_PICTURE),
            Model::Krea2(_) => Some(KREA2_DRAWS_FROM_NO_PICTURE),
            Model::QwenImage(_) => Some(QWEN_IMAGE_DRAWS_FROM_NO_PICTURE),
        }
    }

    fn generate_reporting(
        &self,
        prompt: &str,
        options: &GenerationOptions,
        report: &mut dyn FnMut(GenerationProgress) -> ControlFlow<()>,
    ) -> crate::Result<Option<Tensor>> {
        match self {
            Model::Sdxl(model) => model.generate_reporting(prompt, options, report),
            Model::Anima(model) => model.generate_reporting(prompt, options, report),
            Model::Krea2(model) => model.generate_reporting(prompt, options, report),
            Model::QwenImage(model) => model.generate_reporting(prompt, options, report),
        }
    }

    fn generate_from_image_reporting(
        &self,
        image: &Tensor,
        prompt: &str,
        options: &GenerationOptions,
        report: &mut dyn FnMut(GenerationProgress) -> ControlFlow<()>,
    ) -> crate::Result<Option<Tensor>> {
        match self {
            Model::Sdxl(model) => {
                model.generate_from_image_reporting(image, prompt, options, report)
            }
            // The same sentence the door turns a run away with, for the caller that is not the
            // door: one wording for one refusal.
            Model::Anima(_) | Model::Krea2(_) | Model::QwenImage(_) => Err(crate::Error::model(
                self.no_picture_because()
                    .unwrap_or("this model cannot start from a picture"),
            )),
        }
    }
}

/// A picture, as the `(1, 3, height, width)` tensor a run can start from.
///
/// Scaled to the size that was asked for rather than kept at its own: the model works at
/// multiples of 64, and a picture off a camera or a phone is not one. Stretched to it rather than
/// cropped, so that the whole of what was handed in is what the run sees -- the sizes on offer
/// include the portrait and landscape ones SDXL was trained at, which is where to say what shape
/// the picture is.
///
/// Lanczos, because this is the direction that loses pixels -- a photograph is larger than
/// anything SDXL draws -- and a cheaper filter leaves stair steps that the run then faithfully
/// keeps.
fn read_image(bytes: &[u8], width: i32, height: i32) -> Result<Tensor, Error> {
    // Guessed from the bytes rather than trusted from the request. What a browser says a dropped
    // file is, is what the file was called, and a run refused over a wrong extension is a run
    // refused over a name.
    let opened = image::ImageReader::new(Cursor::new(bytes))
        .with_guessed_format()?
        .decode()?;

    let scaled = opened.resize_exact(
        width as u32,
        height as u32,
        image::imageops::FilterType::Lanczos3,
    );

    Ok(from_rgb8(width, height, scaled.to_rgb8().as_raw())?)
}

/// The tensor a run ends with, as a PNG, with `parameters` written into it.
fn png(image: &Tensor, parameters: &str) -> Result<Vec<u8>, Error> {
    use image::ImageEncoder;

    let pixels = to_rgb8(image)?;
    let shape = image.shape();
    let (width, height) = (shape[3] as u32, shape[2] as u32);

    let mut encoded = Vec::new();
    image::codecs::png::PngEncoder::new(&mut encoded).write_image(
        &pixels,
        width,
        height,
        image::ExtendedColorType::Rgb8,
    )?;

    Ok(with_text(&encoded, "parameters", parameters))
}

/// `png` with a text chunk saying `keyword` is `text`, straight after the header.
///
/// `parameters` is the keyword every other tool of this kind writes its generation line under,
/// so that a picture dropped into one of them says what drew it. An `iTXt` chunk rather than a
/// `tEXt`, because `tEXt` is Latin-1 and a prompt is whatever somebody typed -- in Chinese, as
/// often as not.
///
/// Written here rather than asked of the encoder: the one `image` wraps does not take text, and a
/// chunk is a length, four letters, the bytes and a checksum.
fn with_text(png: &[u8], keyword: &str, text: &str) -> Vec<u8> {
    // The eight-byte signature, then the header chunk: its length, its type, thirteen bytes and
    // its checksum. Anything that does not start that way is left as it is.
    const AFTER_THE_HEADER: usize = 8 + 4 + 4 + 13 + 4;
    if png.len() < AFTER_THE_HEADER || &png[12..16] != b"IHDR" {
        return png.to_vec();
    }

    // The keyword, then no compression, no method, an empty language and an empty translation,
    // each ended by a zero -- and the text itself, which is not.
    let mut data = keyword.as_bytes().to_vec();
    data.extend_from_slice(&[0, 0, 0, 0, 0]);
    data.extend_from_slice(text.as_bytes());

    let mut chunk = b"iTXt".to_vec();
    chunk.extend_from_slice(&data);
    let checksum = crc32(&chunk);

    let mut whole = Vec::with_capacity(png.len() + chunk.len() + 8);
    whole.extend_from_slice(&png[..AFTER_THE_HEADER]);
    whole.extend_from_slice(&(data.len() as u32).to_be_bytes());
    whole.extend_from_slice(&chunk);
    whole.extend_from_slice(&checksum.to_be_bytes());
    whole.extend_from_slice(&png[AFTER_THE_HEADER..]);

    whole
}

/// The CRC-32 a PNG chunk ends with, over its type and its data. A bit at a time rather than from
/// a table: a chunk of a few hundred bytes once per picture does not need the table.
fn crc32(bytes: &[u8]) -> u32 {
    let mut crc = !0u32;
    for byte in bytes {
        crc ^= u32::from(*byte);
        for _ in 0..8 {
            crc = match crc & 1 {
                1 => (crc >> 1) ^ 0xedb8_8320,
                _ => crc >> 1,
            };
        }
    }

    !crc
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_stop_ends_one_job_and_the_voice_is_still_there_for_the_next() {
        use crate::cli::args::DeviceOption;
        use crate::cli::task::Task;
        use crate::cli::webui::store::{Status, Store};

        let root =
            std::env::temp_dir().join(format!("libwaifu-worker-stop-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let shared = Shared::new(
            Task::Text2Speech,
            DeviceOption::Cpu.resolve(),
            Store::open(&root, None).unwrap(),
        );
        shared.change(|world| world.voice = Some(look_at_voice(TONES)));
        let mut voice = None;
        assert!(read_voice(&shared, TONES, &mut voice));

        let asked = serde_json::json!({"text": "hello there", "speed": 1.0, "temperature": 0.8, "seed": "7"});

        // Stopped after it was handed to the worker and before it ran: it does not run.
        let stopped = shared.store().submit(Kind::Speech, asked.clone()).unwrap();
        let job = shared.next();
        assert_eq!(job.id, stopped.id);
        shared.cancel(&job.id);
        run(&shared, None, voice.as_deref(), &job);
        shared.finished();
        assert_eq!(
            shared.store().find(&job.id).unwrap().status,
            Status::Cancelled
        );

        // Nothing was put down: the next job runs with what is loaded, and the stop before it is
        // not its own.
        assert!(voice.is_some());
        assert!(shared.world().voice.as_ref().unwrap().in_memory);
        shared.store().submit(Kind::Speech, asked).unwrap();
        let job = shared.next();
        run(&shared, None, voice.as_deref(), &job);
        shared.finished();
        let done = shared.store().find(&job.id).unwrap();
        assert_eq!(done.status, Status::Done, "{:?}", done.error);
        assert_eq!(done.made.unwrap()["seed"], 7);
    }

    #[test]
    fn the_checksum_is_the_one_a_png_reader_checks() {
        // The check value every CRC-32 is published with.
        assert_eq!(crc32(b"123456789"), 0xcbf4_3926);
    }

    #[test]
    fn a_picture_says_what_drew_it_and_is_still_a_picture() {
        // One pixel, as the encoder writes it, and then with the line written into it -- in a
        // script Latin-1 has no letters for, which is why the chunk is iTXt.
        let mut plain = Vec::new();
        image::ImageEncoder::write_image(
            image::codecs::png::PngEncoder::new(&mut plain),
            &[255, 0, 0],
            1,
            1,
            image::ExtendedColorType::Rgb8,
        )
        .unwrap();
        let said = "一只猫\nSteps: 20, Seed: 7";
        let written = with_text(&plain, "parameters", said);

        // A reader that checks every chunk's checksum still reads it, and reads the same pixel.
        let read = image::load_from_memory(&written)
            .expect("still a PNG")
            .to_rgb8();
        assert_eq!(read.get_pixel(0, 0).0, [255, 0, 0]);

        // And the line is in it, after the header and before the pixels, under the keyword.
        let at = written
            .windows(4)
            .position(|four| four == b"iTXt")
            .expect("a text chunk");
        assert_eq!(at, 8 + 4 + 4 + 13 + 4 + 4);
        let after = &written[at + 4..];
        assert!(after.starts_with(b"parameters\0\0\0\0\0"));
        assert!(after[15..].starts_with(said.as_bytes()));

        // Something that is not a PNG is handed back as it came.
        assert_eq!(with_text(b"GIF89a", "parameters", said), b"GIF89a");
    }

    #[test]
    fn a_job_is_run_with_the_settings_it_was_kept_with() {
        let asked = serde_json::json!({
            "prompt": "a cat",
            "negative": "blurry",
            "width": 832,
            "height": 1216,
            "steps": 28,
            "guidance": 5.5,
            "strength": 0.6,
            "seed": "18446744073709551615",
        });
        let options = generation_options(&asked);
        assert_eq!((options.width, options.height), (832, 1216));
        assert_eq!(options.num_steps, 28);
        assert_eq!(options.guidance_scale, 5.5);
        assert_eq!(options.strength, 0.6);
        assert_eq!(options.negative_prompt, "blurry");
        // All sixty-four bits of it, which is why it is kept as a string.
        assert_eq!(options.seed, Some(u64::MAX));

        let spoken =
            speech_options(&serde_json::json!({"speed": 1.25, "temperature": 0.7, "seed": "7"}));
        assert_eq!(spoken.speed, 1.25);
        assert_eq!(spoken.seed, Some(7));
    }

    /// A manifest on the disk holding `body`, and the path it was written to.
    ///
    /// Named after the test that asked for it so that two of them never collide, and left behind:
    /// it is a few hundred bytes in the temporary directory, and a file removed on the way out of
    /// a test that failed is the file somebody wanted to look at.
    fn a_manifest(called: &str, body: &str) -> PathBuf {
        let path =
            std::env::temp_dir().join(format!("libwaifu-{called}-{}.yaml", std::process::id()));
        std::fs::write(&path, body).expect("somewhere to write a manifest");
        path
    }

    #[test]
    fn a_distilled_model_is_known_to_take_no_guidance_before_it_is_read() {
        // From the name alone, which is what the boxes are filled in from while the package is
        // still being fetched. Krea 2 is distilled here or it is not exported at all, so the kind
        // answers this the same way it answers eight steps and a guidance of one.
        let turbo = look_at("krea2:turbo:v1.0");
        assert!(!turbo.takes_guidance);
        assert_eq!(turbo.defaults.num_steps, 8);

        // And the models that do have a second pass, which is every other kind.
        assert!(look_at("sdxl:base:v1.0").takes_guidance);
        assert!(look_at("anima:turbo:v1.1").takes_guidance);
    }

    #[test]
    fn a_package_may_say_for_itself_whether_it_takes_guidance() {
        // Silence is not a no. A package exported before there was a key for this says nothing in
        // its `suggested:` block, and what fills the gap is what the kind already answered --
        // which is what lets one keep working here without being rewritten.
        let quiet = a_manifest(
            "quiet",
            "weights:\n  - krea2.0.safetensors\nconfig:\n  model:\n    type: krea2\n\
             suggested:\n  steps: 8\n  guidance: 1.0\n",
        );
        assert!(!look_at(quiet.to_str().expect("a path")).takes_guidance);

        // And a package that says so plainly is believed over the kind, in both directions: the
        // day there is an undistilled Krea 2, this is the line it gets its dial back with.
        let undistilled = a_manifest(
            "undistilled",
            "weights:\n  - krea2.0.safetensors\nconfig:\n  model:\n    type: krea2\n\
             suggested:\n  steps: 28\n  guidance: 5.5\n  takes_guidance: \"true\"\n",
        );
        let raw = look_at(undistilled.to_str().expect("a path"));
        assert!(raw.takes_guidance);
        assert_eq!(raw.defaults.num_steps, 28);

        // The other direction, for a distilled release of a kind that is not otherwise one.
        let distilled_sdxl = a_manifest(
            "distilled-sdxl",
            "weights:\n  - sdxl.0.safetensors\nconfig:\n  model:\n    type: sdxl\n\
             suggested:\n  steps: 4\n  takes_guidance: \"false\"\n",
        );
        assert!(!look_at(distilled_sdxl.to_str().expect("a path")).takes_guidance);
    }
}
