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
//! them. What the requests do instead is post a command down a channel to this one thread, which
//! reads them in the order they were posted and says what it is doing in [`Shared`] as it goes.
//!
//! It is also why the answer to "draw this" is an empty acknowledgement rather than a picture.
//! A run is minutes long; a request that waited for one would hold a socket open across it, and
//! the browser would have nothing to show in the meantime.

use std::io::Cursor;
use std::ops::ControlFlow;
use std::path::{Path, PathBuf};
use std::sync::mpsc::Receiver;
use std::time::Instant;

use crate::cli::args::Runtime;
use crate::cli::hub;
use crate::cli::webui::state::{Chosen, Doing, Fetch, Picture, Run, Shared};
use crate::flint::Tensor;
use crate::{
    from_rgb8, to_rgb8, Anima, GenerationDefaults, GenerationOptions, GenerationProgress, Krea2,
    Manifest, Sdxl,
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

/// How many sizes a model has to name before its list is used instead of the one above.
///
/// One size is not a choice and two is barely one. A model that names fewer has not replaced the
/// list, it has emptied it, and the built-in one is the better thing to offer.
const ENOUGH_SIZES: usize = 3;

/// A prompt to draw.
pub struct Job {
    /// The model to draw it with, as it was chosen: read now if the weights in memory are of
    /// another one, or of none.
    pub model: String,
    pub prompt: String,
    /// Always with a seed in it. A run asked for without one is given a fresh one here rather
    /// than left to whatever the device's generator was last used for, so that the picture that
    /// comes out can be drawn a second time -- which is most of what a seed is for.
    pub options: GenerationOptions,
    /// The picture to start from, as the bytes it arrived as, where the run starts from one.
    ///
    /// Bytes rather than a path. One of these was dropped on the page and never had a name on
    /// this machine at all, and the other was read where `-i` named it, back when there was still
    /// a terminal to complain to about a file that is not there.
    pub from: Option<Vec<u8>>,
}

/// What the requests can ask the worker for.
pub enum Command {
    /// Fetch this model if it is not on the disk, read it onto the device, and hold it.
    Use(String),
    /// Send what is loaded, and everything after it, to another device.
    UseDevice(Runtime),
    Draw(Job),
}

/// Reads commands and carries them out, until the channel is closed.
pub fn work(shared: &Shared, commands: &Receiver<Command>) {
    // What is in memory. Held here rather than in the session, which describes the model without
    // holding it: the weights are the one thing in this program that cannot cross a thread.
    let mut model: Option<Model> = None;

    // Which model the weights in there are. What the page has chosen may be another one by now,
    // and the two are compared at each run: this is the name that was opened, not the name that
    // is wanted.
    let mut in_memory: Option<String> = None;

    for command in commands {
        match command {
            // Asked for by `-m`, which says to have it ready rather than to wait for a run.
            Command::Use(asked) => {
                model = None;
                in_memory = None;
                if read_model(shared, &asked, &mut model) {
                    in_memory = Some(asked);
                }
            }

            // The weights live on the device they were read onto, so the way to another one is
            // through letting go of them. Nothing is read back here: the next run reads what it
            // needs, which is the whole of what this program does with a model now.
            Command::UseDevice(runtime) => {
                model = None;
                in_memory = None;
                forget_the_weights(shared);
                shared.use_runtime(runtime);
                shared.say(format!("runs go to {} now", runtime.name()), false);
            }

            Command::Draw(job) => {
                // Read here rather than when it was chosen. Choosing is a click and reading is
                // minutes -- a fetch of several gigabytes, on a model that is not here yet -- and
                // somebody still deciding what to type should not be waiting for either.
                if in_memory.as_deref() != Some(job.model.as_str()) {
                    model = None;
                    in_memory = None;
                    forget_the_weights(shared);

                    if read_model(shared, &job.model, &mut model) {
                        in_memory = Some(job.model.clone());
                    }
                }

                match &model {
                    Some(model) => draw(shared, model, job),
                    // Nothing to say: the read that just failed said why, in the words of
                    // whatever went wrong with it.
                    None => (),
                }
            }
        }

        // Whatever it was, it is finished with, and the next request can have the worker. Every
        // way out of the two arms above comes through here -- which is the point of releasing it
        // in one place rather than at each of the dozen places a command can end.
        shared.release();
    }
}

/// Says on screen that there are no weights in memory, leaving what is chosen chosen.
///
/// The choice outlives the weights. Somebody who changed the device, or whose last run was of
/// another model, has not un-chosen anything -- the next run reads what they chose again.
fn forget_the_weights(shared: &Shared) {
    shared.change(|session| {
        if let Some(chosen) = &mut session.model {
            chosen.in_memory = false;
        }
    });
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
            shared.change(|session| {
                // Only where it is still the chosen one. A choice made while this was reading is
                // the newer answer to "what is the page set to draw with", and describing the
                // model that has just been read over the top of it would undo somebody's click.
                if session
                    .model
                    .as_ref()
                    .is_none_or(|one| one.name == read.name)
                {
                    session.model = Some(read);
                }
                session.doing = Doing::Nothing;
            });
            shared.say("ready", false);
            true
        }
        Err(error) => {
            shared.change(|session| session.doing = Doing::Nothing);
            shared.say(error.to_string(), true);
            false
        }
    }
}

/// What a kind of model implies for the screen before any of its weights are read: what to ask
/// it for, what walks the noise back, and why it cannot start from a picture.
///
/// One table for the three kinds, read twice -- once from a name that has not been fetched yet,
/// and once from the `type` of a manifest that is already on the disk.
fn about(kind: &str) -> (GenerationDefaults, &'static str, Option<&'static str>) {
    match kind {
        Anima::MODEL_TYPE => (
            Anima::DEFAULTS,
            "Flow match Euler",
            Some(ANIMA_DRAWS_FROM_NO_PICTURE),
        ),
        Krea2::MODEL_TYPE => (
            Krea2::DEFAULTS,
            "Flow match Euler",
            Some(KREA2_DRAWS_FROM_NO_PICTURE),
        ),
        // Everything else is SDXL, which is what `Model::from_manifest` decides too.
        _ => (GenerationDefaults::default(), "Euler", None),
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
        _ => "",
    };
    let (defaults, sampler, no_picture) = about(guessed);

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
        let (defaults, sampler, no_picture) = about(&kind);
        chosen.defaults = defaults;
        chosen.sampler = sampler;
        chosen.no_picture_because = no_picture.map(str::to_string);
    }

    let suggested = manifest.suggested();
    chosen.defaults = suggested.over(chosen.defaults);
    chosen.suggested_prompt = suggested.prompt.clone();
    chosen.suggested_avoid = suggested.avoid.clone();
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

    shared.change(|session| {
        session.doing = Doing::Fetching(Fetch {
            model: name.clone(),
            hub: None,
            file: String::new(),
            done: 0,
            total: None,
            part: 0,
            parts: 0,
        })
    });

    let path =
        hub::resolve_reporting(asked, &mut |progress| report_fetch(shared, &name, progress))?;

    shared.change(|session| {
        session.doing = Doing::Reading {
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
        on_disk: true,
        in_memory: true,
        ..look_at(asked)
    };

    Ok((read, opened))
}

/// Copies what a fetch has to say into the session.
fn report_fetch(shared: &Shared, model: &str, progress: hub::Progress) {
    shared.change(|session| {
        // Whatever it said last, so that a line about one field does not blank the others: the
        // hub is said once, before the first byte, and every line after it is about a file.
        let mut fetch = match &session.doing {
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

        session.doing = Doing::Fetching(fetch);
    });
}

/// Draws one picture and puts it in the gallery.
fn draw(shared: &Shared, model: &Model, job: Job) {
    // A stop asked for after the last run finished is not this run's business.
    shared.carry_on();
    let started = Instant::now();
    shared.change(|session| {
        session.doing = Doing::Drawing(Run {
            progress: GenerationProgress::Encoding,
            steps: job.options.num_steps,
            started,
        });
        session.note = None;
    });

    let mut report = |progress| {
        shared.change(|session| {
            if let Doing::Drawing(run) = &mut session.doing {
                run.progress = progress;
            }
        });

        match shared.interrupted() {
            true => ControlFlow::Break(()),
            false => ControlFlow::Continue(()),
        }
    };

    // The picture to start from is read and scaled before the run, so that one that cannot be
    // read is one message rather than a run that fails partway.
    let started_from = match &job.from {
        Some(bytes) => match read_image(bytes, job.options.width, job.options.height) {
            Ok(image) => Some(image),
            Err(error) => {
                shared.change(|session| session.doing = Doing::Nothing);
                return shared.say(format!("the picture to draw from: {error}"), true);
            }
        },
        None => None,
    };

    let drawn = match &started_from {
        Some(image) => {
            model.generate_from_image_reporting(image, &job.prompt, &job.options, &mut report)
        }
        None => model.generate_reporting(&job.prompt, &job.options, &mut report),
    };

    let elapsed = started.elapsed();
    let kept = match drawn {
        // Written here rather than handed back, because the pixels are three megabytes that
        // nothing on the other side would do anything with but hand to a file.
        Ok(Some(image)) => keep(&image),
        Ok(None) => {
            shared.change(|session| session.doing = Doing::Nothing);
            return shared.say("stopped where it was", false);
        }
        Err(error) => Err(error.into()),
    };

    let (file, width, height) = match kept {
        Ok(kept) => kept,
        Err(error) => {
            shared.change(|session| session.doing = Doing::Nothing);
            return shared.say(error.to_string(), true);
        }
    };

    let model_name = shared
        .session()
        .model
        .as_ref()
        .map(|loaded| loaded.name.clone())
        .unwrap_or_default();

    shared.change(|session| {
        session.gallery.insert(
            0,
            Picture {
                file,
                width,
                height,
                prompt: job.prompt.clone(),
                negative: job.options.negative_prompt.clone(),
                // Always there: a job is given one before it is posted.
                seed: job.options.seed.unwrap_or_default(),
                steps: job.options.num_steps,
                guidance: job.options.guidance_scale,
                model: model_name,
                from_image: job.from.as_ref().map(|_| job.options.strength),
                elapsed,
            },
        );
        session.doing = Doing::Nothing;
        session.note = None;
    });
}

/// The model a run draws with, whichever kind of package was opened.
///
/// The two share no code and no weights. What they share is the question -- here is a prompt and
/// here is how big -- and this is where that question stops being asked of a particular model.
///
/// One of these exists, on the worker's stack, for as long as a model is loaded, and the two
/// variants differ by a couple of hundred bytes against the gigabytes of weights they point at.
/// Boxing the larger to even them up would buy an indirection and nothing else.
#[allow(clippy::large_enum_variant)]
pub enum Model {
    Sdxl(Sdxl),
    Anima(Anima),
    Krea2(Krea2),
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
        }
    }

    /// What walks the noise back, as the model's own reference implementation names it.
    fn sampler(&self) -> &'static str {
        match self {
            Model::Sdxl(_) => "Euler",
            Model::Anima(_) | Model::Krea2(_) => "Flow match Euler",
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
            Model::Anima(_) | Model::Krea2(_) => Err(crate::Error::model(
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

/// Writes the tensor a run ends with into the first `waifu-NNNN.png` here that nothing else has.
fn keep(image: &Tensor) -> Result<(String, usize, usize), Error> {
    let pixels = to_rgb8(image)?;
    let shape = image.shape();
    let (width, height) = (shape[3] as usize, shape[2] as usize);

    for number in 1..10_000 {
        let name = format!("waifu-{number:04}.png");
        if Path::new(&name).exists() {
            continue;
        }

        image::save_buffer(
            &name,
            &pixels,
            width as u32,
            height as u32,
            image::ColorType::Rgb8,
        )?;
        return Ok((name, width, height));
    }

    Err("there are already ten thousand pictures in this directory".into())
}
