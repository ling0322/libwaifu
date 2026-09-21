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

//! The requests, and what each one answers with.
//!
//! Three of them serve the page itself, and the rest are the conversation the page holds with the
//! worker: what models there are, what is loaded, what is happening, draw this, stop. None of
//! them waits for a model -- a request that did would hold its socket open for minutes -- so each
//! one posts a command and answers with what is true at that moment. The page asks again.

use std::io::{Cursor, Read};
use std::sync::mpsc::Sender;
use std::sync::Arc;

use serde_json::{json, Value};
use tiny_http::{Header, Method, Request, Response};

use crate::cli::args::Runtime;
use crate::cli::hub;
use crate::cli::webui::machine;
use crate::cli::webui::state::{Doing, Shared};
use crate::cli::webui::worker::{self, Command, Job, SayJob};
use crate::{GenerationOptions, SpeechOptions};

/// The page, built into the binary. There is no directory of files to find at runtime and no
/// order the program has to be started from: a single executable is the whole of it.
const INDEX: &str = include_str!("assets/index.html");
const STYLE: &str = include_str!("assets/style.css");
const SCRIPT: &str = include_str!("assets/app.js");

/// The three files the page is drawn with: React, its renderer, and htm, which stands in for JSX.
/// They are here rather than fetched from anywhere so that the page opens with the network
/// unplugged, and unbuilt so that what is served is what is in the repository.
const VENDOR: [(&str, &str); 3] = [
    ("/vendor/react.js", include_str!("assets/vendor/react.js")),
    (
        "/vendor/react-dom.js",
        include_str!("assets/vendor/react-dom.js"),
    ),
    ("/vendor/htm.js", include_str!("assets/vendor/htm.js")),
];

/// How much of a posted picture to read before giving up on it.
///
/// A photograph off a phone is a few megabytes and this is a good deal more than that. What it is
/// for is the request that is not a picture at all: the body is read into memory before anything
/// looks at it, and without a limit "read it all" is as long as whatever is sending it likes.
const MOST_OF_A_PICTURE: u64 = 64 << 20;

/// And how much of a posted recording.
///
/// Smaller than a picture on purpose. What a speech model wants to hear of somebody is seconds,
/// not minutes -- the page trims to half a minute before it posts -- and what arrives here is
/// uncompressed samples rather than a compressed file, so the number that is generous for a
/// recording is smaller than the one that is generous for a photograph.
const MOST_OF_A_RECORDING: u64 = 16 << 20;

/// What every answer is: bytes, with a type on them. One shape rather than a dozen, because a
/// route that could answer with any of several would otherwise need them boxed.
type Reply = Response<Cursor<Vec<u8>>>;

/// Reads one request and answers it.
pub fn answer(shared: &Arc<Shared>, commands: &Sender<Command>, request: &mut Request) -> Reply {
    // What was asked for, without anything after a question mark. Nothing here reads a query
    // string -- what a request carries, it carries in its body -- but a browser appends one to
    // defeat its own cache, and a path that kept it would match nothing.
    let path = request.url().split('?').next().unwrap_or("/").to_string();

    match (request.method(), path.as_str()) {
        (Method::Get, "/") => page(INDEX, "text/html; charset=utf-8"),
        (Method::Get, "/style.css") => page(STYLE, "text/css; charset=utf-8"),
        (Method::Get, "/app.js") => page(SCRIPT, "text/javascript; charset=utf-8"),

        (Method::Get, "/api/state") => json(shared.describe(models())),
        (Method::Get, "/api/progress") => json(shared.progress()),
        // Apart from the state rather than inside it, because it answers a different question.
        // The state is what this session has done and is a new one every time anything happens;
        // what the machine has left changes on its own, with nothing here having done anything,
        // and a revision counted up for every megabyte of memory somebody else's browser took
        // would have the page reading the gallery back several times a second.
        (Method::Get, "/api/machine") => json(machine::describe(shared.runtime())),

        (Method::Post, "/api/model") => choose_model(shared, request),
        (Method::Post, "/api/device") => use_device(shared, commands, request),
        (Method::Delete, "/api/model") => forget_model(shared, request),
        (Method::Post, "/api/generate") => generate(shared, commands, request),
        (Method::Post, "/api/speak") => speak(shared, commands, request),
        (Method::Post, "/api/interrupt") => {
            shared.interrupt();
            json(json!({ "ok": true }))
        }

        (Method::Delete, "/api/picture") => forget_picture(shared, request),
        (Method::Delete, "/api/clip") => forget_clip(shared, request),

        (Method::Post, "/api/upload") => upload(shared, request),
        (Method::Get, "/api/upload") => held_picture(shared),
        (Method::Delete, "/api/upload") => {
            shared.forget_upload();
            json(json!({ "ok": true }))
        }

        // The recording a reading is to sound like. The same three doors the picture has, and
        // its own box behind them: they are two different runs' inputs, and dropping one should
        // not throw away the other.
        (Method::Post, "/api/voice") => hold_recording(shared, request),
        (Method::Get, "/api/voice") => held_recording(shared),
        (Method::Delete, "/api/voice") => {
            shared.forget_recording();
            json(json!({ "ok": true }))
        }

        // What is left of a GET: the libraries the page is drawn with, and the pictures this
        // session drew. A picture goes by the name it was written under and no other -- a path
        // out of a request is otherwise a way to read any file this process can, and a server on
        // the loopback address is still reachable by anything else running on the machine.
        (Method::Get, path) => match VENDOR.iter().find(|(name, _)| *name == path) {
            Some((_, script)) => page(script, "text/javascript; charset=utf-8"),
            None => match (path.strip_prefix("/picture/"), path.strip_prefix("/clip/")) {
                (Some(file), _) => picture(shared, file),
                (_, Some(file)) => clip(shared, file),
                _ => refused(404, "no such page"),
            },
        },

        _ => refused(404, "no such page"),
    }
}

/// Sets what the page draws with. Nothing is read here.
///
/// The weights are the expensive part and they are read by the run that needs them -- this is a
/// name, what the name says about the model, and nothing else. So it takes no worker and waits
/// for nothing: somebody can change their mind about which model to use while the one before it
/// is still being fetched.
fn choose_model(shared: &Arc<Shared>, request: &mut Request) -> Reply {
    let asked = match body(request) {
        Ok(body) => body,
        Err(error) => return refused(400, &error),
    };
    // A null is "nothing is chosen", which the page sends when the kind of run changes: what is
    // worth drawing with is a question about the run, and the answer to the old one is not the
    // answer to the new one. Told apart from a missing field, which is a request that forgot to
    // say anything -- one is a decision and the other is a mistake.
    if matches!(asked.get("model"), Some(Value::Null)) {
        shared.change(|session| session.model = None);
        shared.say("pick a model to begin", false);
        return json(json!({ "ok": true }));
    }

    let Some(name) = asked.get("model").and_then(Value::as_str) else {
        return refused(400, "no model was named");
    };
    if name.trim().is_empty() {
        return refused(400, "no model was named");
    }

    let chosen = worker::look_at(name.trim());
    shared.change(|session| session.model = Some(chosen));
    shared.say(format!("{} is what runs will use", name.trim()), false);

    json(json!({ "ok": true }))
}

/// Sends what runs next to another device, once whatever is in front of it has finished.
fn use_device(shared: &Arc<Shared>, commands: &Sender<Command>, request: &mut Request) -> Reply {
    let asked = match body(request) {
        Ok(body) => body,
        Err(error) => return refused(400, &error),
    };
    let Some(name) = asked.get("device").and_then(Value::as_str) else {
        return refused(400, "no device was named");
    };

    // Against what this machine has, not against what the names spell: a build with CUDA in it
    // on a machine with no card knows both words and can carry out neither.
    let Some(runtime) = Runtime::named(name).filter(|one| Runtime::available().contains(one))
    else {
        let names: Vec<&str> = Runtime::available().iter().map(|one| one.name()).collect();
        return refused(
            400,
            &format!(
                "no device here is called \"{name}\": this machine has {}",
                names.join(", ")
            ),
        );
    };
    if runtime == shared.runtime() {
        return json(json!({ "ok": true }));
    }

    // The same claim a load takes, because this is one: the model on the old device is let go of
    // and read again on the new one.
    if !shared.claim() {
        return refused(409, &already(shared));
    }

    post(shared, commands, Command::UseDevice(runtime))
}

/// Throws away what has been fetched of a model, which is the only way this program gives disk
/// space back.
///
/// A model is several gigabytes and the list says which are on the disk, so the list is where
/// somebody notices they are keeping four of them. What is lost is a download and nothing more:
/// nothing outside the cache is touched, and a package somebody exported themselves is not
/// something the catalogue can reach.
fn forget_model(shared: &Arc<Shared>, request: &mut Request) -> Reply {
    let asked = match body(request) {
        Ok(body) => body,
        Err(error) => return refused(400, &error),
    };
    let Some(name) = asked.get("model").and_then(Value::as_str) else {
        return refused(400, "no model was named");
    };

    // A model with its weights in memory is reading them off those files as it draws -- a step at
    // a time, from a mapping -- so deleting them out from under it is deleting what the next step
    // needs. One that has only been chosen is holding nothing open, and may go.
    if shared
        .session()
        .model
        .as_ref()
        .is_some_and(|model| model.in_memory && model.name == name)
    {
        return refused(409, "that model is in memory: pick another one first");
    }
    if shared.is_busy() {
        return refused(409, &already(shared));
    }

    match hub::remove(name) {
        Ok(()) => {
            shared.say(format!("{name} is no longer on the disk"), false);
            json(json!({ "ok": true }))
        }
        Err(error) => refused(400, &error.to_string()),
    }
}

/// Starts a run, if there is a model and nothing else is happening.
fn generate(shared: &Arc<Shared>, commands: &Sender<Command>, request: &mut Request) -> Reply {
    let asked = match body(request) {
        Ok(body) => body,
        Err(error) => return refused(400, &error),
    };

    let prompt = asked
        .get("prompt")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();
    if prompt.trim().is_empty() {
        return refused(400, "there is nothing to draw: type a prompt");
    }

    // The worker is taken before anything else is asked of the state, so that what a second
    // request is told is what is actually happening -- "no model is loaded" is a true sentence
    // while one is being read and a baffling thing to be told at that moment.
    if !shared.claim() {
        return refused(409, &already(shared));
    }

    let Some(chosen) = shared.session().model.clone() else {
        return give_up(
            shared,
            409,
            "no model is chosen: pick one from the list of models",
        );
    };

    // Whether this run starts from a picture, which is the img2img tab having one in it. Asked of
    // the request and then of what is actually held: a page that was left open while the picture
    // was cleared would otherwise ask for a run from a picture there is none of.
    let from_a_picture = asked
        .get("from_picture")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let from = match from_a_picture {
        true => match shared.upload() {
            Some(bytes) => Some(bytes),
            None => return give_up(shared, 400, "there is no picture to draw from"),
        },
        false => None,
    };
    if let (true, Some(why)) = (from.is_some(), &chosen.no_picture_because) {
        return give_up(shared, 409, why);
    }

    let options = GenerationOptions {
        width: pixels(asked.get("width"), chosen.defaults.width),
        height: pixels(asked.get("height"), chosen.defaults.height),
        num_steps: whole(asked.get("steps"), chosen.defaults.num_steps).clamp(1, 150),
        guidance_scale: decimal(asked.get("guidance"), chosen.defaults.guidance_scale)
            .clamp(1.0, 30.0),
        negative_prompt: asked
            .get("negative")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string(),
        // Never left open. A run asked for without a seed is given a fresh one rather than left
        // to whatever the device's generator was last used for, so that what comes out says which
        // number drew it and can be drawn again.
        seed: Some(seed(asked.get("seed"))),
        strength: decimal(asked.get("strength"), 0.8).clamp(0.0, 1.0),
    };

    post(
        shared,
        commands,
        Command::Draw(Job {
            // Which model to draw with travels with the run rather than being read off the
            // session by the worker: what was chosen when the button was pressed is what this
            // run is of, whatever gets chosen while it waits its turn.
            model: chosen.name,
            prompt,
            options,
            from,
        }),
    )
}

/// Starts a reading, if there is anything to read and nothing else is happening.
///
/// The mirror of [`generate`], including the order it refuses in: what is wrong with an empty
/// page is that there is nothing to say, and that is the thing somebody can act on.
fn speak(shared: &Arc<Shared>, commands: &Sender<Command>, request: &mut Request) -> Reply {
    let asked = match body(request) {
        Ok(body) => body,
        Err(error) => return refused(400, &error),
    };

    let text = asked
        .get("text")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();
    if text.trim().is_empty() {
        return refused(400, "there is nothing to say: type something to read out");
    }

    // Taken before anything else is asked of the state, for the same reason a picture's run takes
    // it first: what a second request is told should be what is actually happening.
    if !shared.claim() {
        return refused(409, &already(shared));
    }

    let Some(voice) = shared.session().voice.clone() else {
        return give_up(shared, 409, "there is no voice to speak with");
    };

    // Whether this reading is given a recording to sound like. Asked of the request and then of
    // what is actually held, so that a page left open while the recording was cleared does not
    // ask for a run from one there is none of.
    let from_a_recording = asked
        .get("from_recording")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let like = match from_a_recording {
        true => match shared.recording() {
            Some(bytes) => Some(bytes),
            None => return give_up(shared, 400, "there is no recording to sound like"),
        },
        false => None,
    };
    if let (true, Some(why)) = (like.is_some(), &voice.no_likeness_because) {
        return give_up(shared, 409, why);
    }

    let options = SpeechOptions {
        speed: decimal(asked.get("speed"), voice.defaults.speed).clamp(0.25, 4.0),
        temperature: decimal(asked.get("temperature"), voice.defaults.temperature).clamp(0.0, 2.0),
        // Never left open, for the same reason a picture's is not: what comes out should say
        // which number said it, so that it can be said again.
        seed: Some(seed(asked.get("seed"))),
    };

    post(
        shared,
        commands,
        Command::Speak(SayJob {
            voice: voice.name,
            text,
            options,
            like,
        }),
    )
}

/// Holds the recording a reading is to sound like.
///
/// Posted as the bytes of a WAV file and nothing else. The page decodes whatever was dropped on
/// it -- the browser has a decoder for every format it will play, and this program has one for
/// none of them -- so what arrives is samples this program can actually read, whatever the file
/// on the far side was.
fn hold_recording(shared: &Arc<Shared>, request: &mut Request) -> Reply {
    let mut bytes = Vec::new();
    if let Err(error) = request
        .as_reader()
        .take(MOST_OF_A_RECORDING)
        .read_to_end(&mut bytes)
    {
        return refused(400, &format!("the recording did not arrive whole: {error}"));
    }
    if bytes.is_empty() {
        return refused(400, "the recording was empty");
    }

    // Read here rather than at the run, alone among the things this server is posted. A picture
    // that is not one is found out about by an image decoder that says so in a sentence; a WAV
    // that is not one is found out about here, where there is still a request to refuse -- and
    // refusing at the door is how somebody who dropped the wrong file learns it now rather than
    // after pressing the button.
    if let Err(error) = crate::wav::read(&bytes) {
        return refused(400, &error.to_string());
    }

    shared.hold_recording(bytes);
    json(json!({ "ok": true }))
}

/// Hands back the recording being held, so that a page opening fresh can play what is in the box.
fn held_recording(shared: &Arc<Shared>) -> Reply {
    match shared.recording() {
        Some(bytes) => reply(200, "audio/wav", bytes),
        None => refused(404, "no recording is being held"),
    }
}

/// One of the clips this session said.
fn clip(shared: &Arc<Shared>, file: &str) -> Reply {
    let Some(path) = shared.spoke(file) else {
        return refused(404, "this session did not say that");
    };

    match std::fs::read(&path) {
        Ok(bytes) => reply(200, "audio/wav", bytes),
        Err(error) => refused(410, &format!("{file} can no longer be read: {error}")),
    }
}

/// Deletes one of the clips this session said, file and row alike.
///
/// The same door as a picture's and the same rule through it: by the name it was written under
/// and no other.
fn forget_clip(shared: &Arc<Shared>, request: &mut Request) -> Reply {
    let asked = match body(request) {
        Ok(body) => body,
        Err(error) => return refused(400, &error),
    };
    let Some(file) = asked.get("file").and_then(Value::as_str) else {
        return refused(400, "no clip was named");
    };
    let Some(path) = shared.spoke(file) else {
        return refused(404, "this session did not say that");
    };

    match std::fs::remove_file(&path) {
        Ok(()) => {}
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return refused(500, &format!("{file} could not be deleted: {error}")),
    }

    shared.forget_clip(file);
    json(json!({ "ok": true }))
}

/// Deletes one of the pictures this session drew, file and row alike.
///
/// By the name it was written under and no other, like every other way this program is asked for
/// a picture: the gallery is the whole of what a request may name, and a path that came out of a
/// request is otherwise a way to delete any file this process can reach.
fn forget_picture(shared: &Arc<Shared>, request: &mut Request) -> Reply {
    let asked = match body(request) {
        Ok(body) => body,
        Err(error) => return refused(400, &error),
    };
    let Some(file) = asked.get("file").and_then(Value::as_str) else {
        return refused(400, "no picture was named");
    };
    let Some(path) = shared.wrote(file) else {
        return refused(404, "this session did not draw that");
    };

    // A file that is already gone is not an error: that is the state being asked for, and it is
    // already the state. Anything else -- a permission, a directory read-only -- is worth saying,
    // and the row stays so that what it says can be acted on.
    match std::fs::remove_file(&path) {
        Ok(()) => {}
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return refused(500, &format!("{file} could not be deleted: {error}")),
    }

    shared.forget_picture(file);
    json(json!({ "ok": true }))
}

/// Holds the picture an img2img run starts from.
///
/// Posted as the bytes of the file and nothing else -- no form, no boundary, no field names. What
/// crosses is one file, the page knows which, and a multipart body would be a parser this program
/// owns in order to unwrap something it already has.
fn upload(shared: &Arc<Shared>, request: &mut Request) -> Reply {
    let mut bytes = Vec::new();
    if let Err(error) = request
        .as_reader()
        .take(MOST_OF_A_PICTURE)
        .read_to_end(&mut bytes)
    {
        return refused(400, &format!("the picture did not arrive whole: {error}"));
    }
    if bytes.is_empty() {
        return refused(400, "the picture was empty");
    }

    shared.hold_upload(bytes);
    json(json!({ "ok": true }))
}

/// Hands back the picture being held, so that a page opening fresh can show what `-i` named.
fn held_picture(shared: &Arc<Shared>) -> Reply {
    match shared.upload() {
        // Read off the first bytes rather than off a name: what is held arrived either as a file
        // this program opened or as a body a browser posted, and only one of those ever had a
        // name on it.
        Some(bytes) => reply(200, looks_like(&bytes), bytes),
        None => refused(404, "no picture is being held"),
    }
}

/// Which of the two formats a picture is in, from the bytes every file of it starts with.
fn looks_like(bytes: &[u8]) -> &'static str {
    match bytes.starts_with(&[0x89, b'P', b'N', b'G']) {
        true => "image/png",
        false => "image/jpeg",
    }
}

/// One of the pictures this session drew.
fn picture(shared: &Arc<Shared>, file: &str) -> Reply {
    let Some(path) = shared.wrote(file) else {
        return refused(404, "this session did not draw that");
    };

    match std::fs::read(&path) {
        Ok(bytes) => reply(200, "image/png", bytes),
        // It was written, and now it is not readable: deleted from under the program, most
        // likely, which is a thing someone tidying a directory does.
        Err(error) => refused(410, &format!("{file} can no longer be read: {error}")),
    }
}

/// Every model this build can be asked for, and whether it is on the disk already.
fn models() -> Value {
    hub::listed()
        .into_iter()
        .map(|model| {
            json!({
                "name": model.name,
                "full_name": model.full_name,
                "cached": model.cached,
                "bytes": model.bytes,
            })
        })
        .collect()
}

/// What is happening, for a request that has just been told it cannot have the worker.
///
/// The commonest of these is a second click on a button, where "there is already a picture being
/// drawn" is the whole answer. The others are worth naming: waiting on a fetch of several
/// gigabytes looks exactly like a program that has ignored the click.
fn already(shared: &Arc<Shared>) -> String {
    match &shared.session().doing {
        Doing::Drawing(_) => "there is already a picture being drawn".to_string(),
        Doing::Speaking(_) => "there is already something being said".to_string(),
        Doing::Reading { model } => format!("{model} is still being read"),
        Doing::Fetching(fetch) => format!("{} is still being fetched", fetch.model),
        // Claimed, and not yet picked up -- the moment between a request posting a command and
        // the worker starting on it.
        Doing::Nothing => "something is already happening: wait for it, or stop it".to_string(),
    }
}

/// Posts a claimed command to the worker, or says that there is no longer one to post to.
fn post(shared: &Arc<Shared>, commands: &Sender<Command>, command: Command) -> Reply {
    match commands.send(command) {
        Ok(()) => json(json!({ "ok": true })),
        // The worker gives the claim back as it finishes each command; one it never received is
        // one nothing would ever give back.
        Err(_) => give_up(shared, 500, "the model thread is no longer there"),
    }
}

/// Refuses, and hands the worker back to whoever asks next.
///
/// For the refusals that happen after the claim rather than before it: what they have found is a
/// request that cannot be carried out, not a program that is busy.
fn give_up(shared: &Arc<Shared>, status: u16, said: &str) -> Reply {
    shared.release();

    refused(status, said)
}

/// The JSON a request carried, or what was wrong with it.
fn body(request: &mut Request) -> Result<Value, String> {
    let mut said = String::new();
    request
        .as_reader()
        .take(MOST_OF_A_PICTURE)
        .read_to_string(&mut said)
        .map_err(|error| format!("the request did not arrive whole: {error}"))?;

    serde_json::from_str(&said).map_err(|error| format!("that is not a request: {error}"))
}

/// A number of pixels, rounded to something the model can be asked for.
///
/// The U-Net halves its input three times and the decoder eight, so a side that is not a multiple
/// of sixty-four is a shape that does not survive the round trip. The page offers a list of sizes
/// and every one of them is already such a multiple; this is for a request that did not come from
/// the page.
fn pixels(value: Option<&Value>, whether: i32) -> i32 {
    let asked = whole(value, whether).clamp(64, 2048);

    (asked / 64).max(1) * 64
}

fn whole(value: Option<&Value>, whether: i32) -> i32 {
    value
        .and_then(Value::as_i64)
        .and_then(|number| i32::try_from(number).ok())
        .unwrap_or(whether)
}

fn decimal(value: Option<&Value>, whether: f32) -> f32 {
    value
        .and_then(Value::as_f64)
        .map(|number| number as f32)
        .filter(|number| number.is_finite())
        .unwrap_or(whether)
}

/// The seed a run should use: the one that was asked for, or a fresh one.
///
/// Empty, absent, or the minus one every other tool spells "surprise me" all mean the same thing.
/// A seed is sixty-four bits and arrives as a string for that reason -- JSON numbers are doubles,
/// and the largest seeds there are do not survive one.
fn seed(value: Option<&Value>) -> u64 {
    let asked = match value {
        Some(Value::String(said)) => said.trim().parse::<u64>().ok(),
        Some(Value::Number(number)) => number.as_u64(),
        _ => None,
    };

    asked.unwrap_or_else(fresh_seed)
}

/// A seed nobody chose.
///
/// From the hasher the standard library seeds from the operating system, which is the one source
/// of randomness in std. Every `RandomState` is built with different keys, so finishing an empty
/// one is a different number each time it is asked -- which is the whole of what is wanted here.
fn fresh_seed() -> u64 {
    use std::hash::{BuildHasher, Hasher};

    std::collections::hash_map::RandomState::new()
        .build_hasher()
        .finish()
}

fn page(said: &str, mime: &str) -> Reply {
    reply(200, mime, said.as_bytes().to_vec())
}

fn json(value: Value) -> Reply {
    reply(200, "application/json", value.to_string().into_bytes())
}

/// An answer that is not the one that was asked for, with the reason in the same shape everything
/// else answers in, so that the page can show it without knowing which request it came from.
fn refused(status: u16, said: &str) -> Reply {
    reply(
        status,
        "application/json",
        json!({ "error": said }).to_string().into_bytes(),
    )
}

fn reply(status: u16, mime: &str, body: Vec<u8>) -> Reply {
    let mut response = Response::from_data(body).with_status_code(status);
    if let Ok(header) = Header::from_bytes(&b"Content-Type"[..], mime.as_bytes()) {
        response.add_header(header);
    }

    // Nothing here is worth keeping between requests: the page is rebuilt from the state on every
    // load, and a picture's name is only ever used once. A browser that cached either would show
    // yesterday's screen to someone who restarted the program.
    if let Ok(header) = Header::from_bytes(&b"Cache-Control"[..], &b"no-store"[..]) {
        response.add_header(header);
    }

    response
}

#[cfg(test)]
mod tests {
    use super::*;

    use serde_json::json;

    #[test]
    fn a_seed_survives_being_sixty_four_bits_wide() {
        // Which is why it arrives as a string. A JSON number is a double, and the largest seeds
        // there are would come back a few thousand off what was typed.
        assert_eq!(seed(Some(&json!("18446744073709551615"))), u64::MAX);
        assert_eq!(seed(Some(&json!("7"))), 7);
        assert_eq!(seed(Some(&json!(" 7 "))), 7);

        // A number is taken as well, for a request that did not come from the page.
        assert_eq!(seed(Some(&json!(7))), 7);
    }

    #[test]
    fn asking_for_no_seed_asks_for_a_fresh_one() {
        // Every way of spelling "surprise me": nothing typed, the minus one every other tool of
        // this kind takes, and no field at all.
        for asked in [json!(""), json!("-1"), json!("  "), json!(null)] {
            let first = seed(Some(&asked));
            let second = seed(Some(&asked));
            assert_ne!(first, second, "{asked} drew the same seed twice");
        }

        assert_ne!(seed(None), seed(None));
    }

    #[test]
    fn a_size_is_rounded_to_something_the_model_can_be_asked_for() {
        // The U-Net halves its input three times and the decoder eight, so a side that is not a
        // multiple of sixty-four is a shape that does not survive the round trip.
        assert_eq!(pixels(Some(&json!(1024)), 512), 1024);
        assert_eq!(pixels(Some(&json!(1000)), 512), 960);
        assert_eq!(pixels(Some(&json!(-8)), 512), 64);
        assert_eq!(pixels(Some(&json!(99999)), 512), 2048);

        // Absent, or a number that is not one, leaves what the model asked for.
        assert_eq!(pixels(None, 832), 832);
        assert_eq!(pixels(Some(&json!("wide")), 832), 832);
    }

    #[test]
    fn a_number_that_is_not_one_leaves_the_default_where_it_was() {
        assert_eq!(whole(Some(&json!(20)), 30), 20);
        assert_eq!(whole(Some(&json!("20")), 30), 30);
        assert_eq!(whole(None, 30), 30);

        assert_eq!(decimal(Some(&json!(7.5)), 5.0), 7.5);
        assert_eq!(decimal(Some(&json!(7)), 5.0), 7.0);
        assert_eq!(decimal(None, 5.0), 5.0);

        // An exponent past what a double holds. Whether it arrives as an infinity or does not
        // parse at all is serde_json's business; what matters here is that an infinity never
        // reaches the sampler as a guidance scale, and this is where that is checked.
        let enormous: Value = serde_json::from_str("1e400").unwrap_or(Value::Null);
        assert_eq!(decimal(Some(&enormous), 5.0), 5.0);
    }

    #[test]
    fn a_picture_is_recognised_by_what_it_starts_with() {
        // Rather than by what a browser said it was called: that is the name of a file on
        // somebody else's machine, and it is the one part of an upload that can be wrong.
        assert_eq!(looks_like(&[0x89, b'P', b'N', b'G', 13, 10]), "image/png");
        assert_eq!(looks_like(&[0xff, 0xd8, 0xff]), "image/jpeg");
        assert_eq!(looks_like(&[]), "image/jpeg");
    }
}
