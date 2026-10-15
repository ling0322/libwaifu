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

//! The draw command: what the session is for, settled on the command line or on the terminal's
//! screens, and then the drawing screen for it.
//!
//! Everything here happens before a model is read. A flag that is wrong, a picture that is not
//! there, a directory that cannot be made or a model that cannot do what was asked of it are all
//! said now, while the terminal is still a terminal and before anything has been fetched -- not a
//! minute later, over the top of a screen. What comes after is [`tui::draw`].

use std::path::{Path, PathBuf};

use crate::args::Args;
use crate::engine::{self, Chosen};
use crate::manager::{self, Fetch, Word};
use crate::task::{Launch, Task};
use crate::tui;

type Error = Box<dyn std::error::Error>;

fn print_usage() {
    eprintln!("Usage: waifu draw [OPTIONS]");
    eprintln!();
    eprintln!("Options:");
    crate::args::print_options();
    eprintln!();
}

/// Says what was wrong before printing the usage, which is the order the Go tool prints them in.
fn with_usage<T, E: std::fmt::Display>(result: Result<T, E>) -> Result<T, E> {
    if let Err(error) = &result {
        eprintln!("{error}\n");
        print_usage();
    }
    result
}

pub fn main(arguments: &[String]) -> Result<(), Error> {
    let args = match Args::parse(arguments) {
        Ok(args) => args,
        Err(error) => {
            eprintln!("{error}\n");
            print_usage();
            return Err(error.into());
        }
    };
    if args.wants_help() {
        print_usage();
        return Ok(());
    }

    let model = with_usage(args.model())?.map(str::to_string);
    let asked_task = with_usage(args.task())?;
    let device = with_usage(args.device())?;
    let output = PathBuf::from(args.output());
    let image = args.image();

    // Piped or redirected -- a script, a CI job -- there is nobody at a keyboard to type a prompt,
    // and `ratatui::init` on something that is not a terminal panics rather than reporting it.
    // Asked first, so that a session that cannot draw does not fetch gigabytes on its way to
    // finding that out.
    if !tui::can_take_the_screen() {
        return with_usage(Err(
            "there is no terminal to draw in: waifu draws on a screen in the terminal it is run from"
                .into(),
        ));
    }

    // Looked at now rather than when a run begins, so that a path that is not there is said here
    // -- and before the terminal is taken over by the screens, which would hide it.
    if let Some(image) = image {
        if !Path::new(image).is_file() {
            return Err(format!("could not read {image}: there is no such file").into());
        }
        if asked_task == Some(Task::Txt2Img) {
            return with_usage(Err(
                "-i names a picture to start from, which is img2img: leave -task out, or make it \
                 img2img"
                    .into(),
            ));
        }
    }
    // A picture named is a picture to start from, which is where the task list starts too.
    let task = asked_task.or(image.map(|_| Task::Img2Img));

    // Made now for the same reason: a directory that cannot be written to is something to find
    // out before a model has been fetched, not after the first picture has been drawn.
    std::fs::create_dir_all(&output)
        .map_err(|error| format!("could not make {}: {error}", output.display()))?;

    // What this machine can run on and how much room its cards have, asked once and before any
    // screen is up: the first question put to a device has the tensor library say what it found.
    let machine = tui::Machine::measure(device);

    let (task, model) = match model {
        // Named on the command line: nothing to ask. Fetched here, in the terminal, on a line
        // that rewrites itself rather than under a bar.
        Some(model) => {
            let task = task.unwrap_or(Task::Txt2Img);
            // What can be told before anything is fetched: a voice is a voice by the catalogue,
            // and a manifest already here says what it can do.
            if manager::is_a_voice(&model) {
                return with_usage(Err(is_a_voice(&model).into()));
            }
            if let Ok(manifest) = manager::manifest_path(&model) {
                with_usage(fits(task, &model, &engine::look_at(&manifest)?))?;
            }
            (task, model)
        }
        // Not named: the terminal asks, and fetches what is picked under its own bar.
        None => match tui::choose(task)? {
            Some(chosen) => chosen,
            // Somebody who looked at the lists and left. Not a failure.
            None => return Ok(()),
        },
    };

    // After the fetch, since it is the packages on the disk that are measured.
    let manifest = manifest_of(&model)?;
    let chosen = engine::look_at(&manifest)?;
    with_usage(fits(task, &model, &chosen))?;
    let (runtime, why) = machine.start_for(chosen.bytes);
    let launch = Launch {
        task,
        model: manifest.to_string_lossy().into_owned(),
        label: label_of(&model),
        runtime,
    };
    tui::draw(&launch, why, &chosen, image, &output, &machine)
}

/// What the screens call a model: as typed when it was a name, because that is what someone
/// would type again, and the file name when it was a path, because the whole path does not fit
/// and its tail is the part that identifies it.
pub fn label_of(model: &str) -> String {
    match model.rsplit_once(['/', '\\']) {
        Some((_, file)) if !file.is_empty() => file.to_string(),
        _ => model.to_string(),
    }
}

fn is_a_voice(model: &str) -> String {
    format!(
        "{model} is a voice, and the terminal draws pictures: speech is the engine's, for the \
         programs built over libwaifu"
    )
}

/// Whether `model` can do `task`, as its manifest says before any of its weights are read: a
/// voice or a converter cannot draw, and a picture model can start from a picture or not by its
/// kind.
fn fits(task: Task, model: &str, chosen: &Chosen) -> Result<(), String> {
    if !chosen.draws {
        return Err(is_a_voice(model));
    }
    if task == Task::Img2Img {
        if let Some(why) = &chosen.no_picture_because {
            return Err(format!("{model} cannot do img2img: {why}"));
        }
    }

    Ok(())
}

/// Whether a picture model can start from a picture, as far as is known without fetching it:
/// what the terminal's img2img list is filtered on. Asked of its manifest where one is here --
/// fetched, or bundled -- and taken as yes where none is, which a run says otherwise of later.
pub fn draws_from_a_picture(model: &str) -> bool {
    match manager::manifest_path(model) {
        Ok(manifest) => engine::look_at(&manifest).is_ok_and(|chosen| chosen.no_picture_because.is_none()),
        Err(_) => true,
    }
}

/// The manifest of `model`, fetching the model first where it is a published name that is not on
/// the disk yet -- saying how it is going on one line that rewrites itself, the command line's
/// version of the model list's bar.
///
/// Nothing is fetched for a model already here, which is asked of the catalogue first: fetching
/// one that is here would still go looking for which hub to ask, and a machine that is offline
/// would wait on that for nothing.
fn manifest_of(model: &str) -> Result<PathBuf, Error> {
    let here = Path::new(model).is_file()
        || manager::catalog()
            .iter()
            .any(|listed| listed.name == model && listed.cached);
    if here {
        return Ok(manager::manifest_path(model)?);
    }

    let fetch = Fetch::start(model)?;
    let ended = loop {
        let Ok(word) = fetch.news.recv() else {
            break Word::Failed("the fetch went away".to_string());
        };
        let Word::Fetching {
            hub,
            file,
            done,
            total,
            part,
            parts,
        } = word
        else {
            break word;
        };
        if file.is_empty() {
            continue;
        }

        let from = hub.map(|hub| format!(" from {hub}")).unwrap_or_default();
        let of = match parts {
            0 => String::new(),
            parts => format!(" ({part} of {parts})"),
        };
        let how_far = match total {
            Some(total) => format!(
                "{:.0}% of {:.2} GB",
                done as f64 / total as f64 * 100.0,
                total as f64 / 1e9
            ),
            None => format!("{:.2} GB", done as f64 / 1e9),
        };
        eprint!("\r\x1b[2Kfetching {file}{from}{of}: {how_far}");
    };
    eprintln!();

    match ended {
        Word::Done => Ok(manager::manifest_path(model)?),
        Word::Stopped(said) | Word::Failed(said) => Err(said.into()),
        Word::Fetching { .. } => unreachable!("the loop ends on anything else"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn a_manifest(stem: &str, kind: &str) -> PathBuf {
        let path = std::env::temp_dir().join(format!("waifu-draw-cmd-{}-{stem}.yaml", std::process::id()));
        std::fs::write(
            &path,
            format!("weights:\n  - w.safetensors\nconfig:\n  model:\n    type: {kind}\n"),
        )
        .unwrap();
        path
    }

    #[test]
    fn a_voice_is_not_something_to_draw_with() {
        let voice = engine::look_at(&a_manifest("voice", "indextts2")).unwrap();
        let error = fits(Task::Txt2Img, "indextts", &voice).unwrap_err();
        assert!(error.contains("is a voice"), "{error}");

        let sdxl = engine::look_at(&a_manifest("sdxl", "sdxl")).unwrap();
        assert!(fits(Task::Txt2Img, "sdxl:base", &sdxl).is_ok());
    }

    #[test]
    fn img2img_is_refused_for_a_model_that_cannot_start_from_a_picture() {
        // Said of the kind, before a package has been looked at: Anima's layer that reads a
        // picture is not written.
        let anima = engine::look_at(&a_manifest("anima", "anima")).unwrap();
        let error = fits(Task::Img2Img, "anima:turbo", &anima).unwrap_err();
        assert!(error.contains("cannot do img2img"), "{error}");

        let sdxl = engine::look_at(&a_manifest("sdxl2", "sdxl")).unwrap();
        assert!(fits(Task::Img2Img, "sdxl:base", &sdxl).is_ok());
        // And txt2img asks nothing of it.
        assert!(fits(Task::Txt2Img, "anima:turbo", &anima).is_ok());
    }

    #[test]
    fn a_manifest_named_by_its_path_is_its_own_manifest() {
        let path = a_manifest("own", "sdxl");
        // Nothing to fetch, so nothing goes looking for a hub.
        assert_eq!(manifest_of(path.to_str().unwrap()).unwrap(), path);
        assert!(draws_from_a_picture(path.to_str().unwrap()));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn a_model_is_called_by_its_name_or_its_file() {
        assert_eq!(label_of("sdxl:base"), "sdxl:base");
        assert_eq!(label_of("/models/sdxl-base.yaml"), "sdxl-base.yaml");
    }
}
