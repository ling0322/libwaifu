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
//! A REST API over two kinds of thing, jobs and uploads, and three things there is one of:
//!
//! | method | path                       |                                                   |
//! | ------ | -------------------------- | ------------------------------------------------- |
//! | GET    | `/api/model`               | the model this program serves, and its defaults   |
//! | GET    | `/api/machine`             | the processor, the memory, the card               |
//! | GET    | `/api/worker`              | whether it is busy, how far along, how many wait  |
//! | POST   | `/api/jobs`                | a new job: 202, and where to ask after it         |
//! | GET    | `/api/jobs`                | every job, or `?ids=a,b` for those                |
//! | GET    | `/api/jobs/{id}`           | one job: where it is in line, how far along       |
//! | GET    | `/api/jobs/{id}/output`    | what it made                                      |
//! | POST   | `/api/jobs/{id}/cancel`    | out of line, or stopped after the step it is on   |
//! | DELETE | `/api/jobs/{id}`           | gone, and what it made with it                    |
//! | POST   | `/api/uploads`             | a picture or a WAV for a job to start from: 201   |
//! | GET    | `/api/uploads/{id}`        | the file                                          |
//! | DELETE | `/api/uploads/{id}`        | gone                                              |
//!
//! None of them waits for a model -- a request that did would hold its socket open for minutes --
//! so a job is posted and answered with where it is, and asked after until it is done.
//!
//! Nothing here knows who is asking. There are no sessions and no users: an id is 128 random bits
//! and whoever holds one can read and delete what it names. Which ids are whose is for whatever
//! sits in front of this to keep -- the page keeps its own in the browser. What is checked is only
//! that a request came from this machine's own pages and not from some other site open in a
//! browser on it, which a server on the loopback address is otherwise open to: see
//! [`from_here`].

use std::io::{Cursor, Read};
use std::sync::Arc;

use serde_json::{json, Value};
use tiny_http::{Header, Method, Request, Response};

use crate::cli::webui::machine;
use crate::cli::webui::state::{Chosen, ChosenConverter, Shared, Spoken};
use crate::cli::webui::store::{Cancelled, Kind, Refused, Status};

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

/// How much of a posted file to read before giving up on it.
///
/// A photograph off a phone is a few megabytes and this is a good deal more than that. What it is
/// for is the request that is not a picture at all: the body is read into memory before anything
/// looks at it, and without a limit "read it all" is as long as whatever is sending it likes.
const MOST_OF_A_FILE: u64 = 64 << 20;

/// How much of a JSON body to read. A job's settings are a few hundred bytes; a prompt that is a
/// megabyte long is not one anybody typed.
const MOST_OF_A_REQUEST: u64 = 1 << 20;

/// How many jobs may wait at once. Past this a new one is refused rather than taken: a line of
/// hundreds is somebody's script gone wrong, and every one of them would be kept on the disk.
const MOST_QUEUED: usize = 64;

/// How long an instruction may be. The ones a model was taught are a dozen characters; a page of
/// them is not a style.
const LONGEST_STYLE: usize = 200;

/// What every answer is: bytes, with a type on them. One shape rather than a dozen, because a
/// route that could answer with any of several would otherwise need them boxed.
type Reply = Response<Cursor<Vec<u8>>>;

/// Reads one request and answers it.
pub fn answer(shared: &Arc<Shared>, request: &mut Request) -> Reply {
    // What was asked for, and the query after it -- which only the list of jobs reads. A browser
    // appends one to defeat its own cache elsewhere, and a path that kept it would match nothing.
    let url = request.url().to_string();
    let (path, query) = url.split_once('?').unwrap_or((&url, ""));

    if *request.method() == Method::Get {
        if let Some(reply) = static_file(path) {
            return reply;
        }
    }
    if let Err(refusal) = from_here(request) {
        return refusal;
    }

    let parts: Vec<&str> = path.trim_end_matches('/').split('/').collect();
    match (request.method(), parts.as_slice()) {
        (Method::Get, [""]) => page(INDEX, "text/html; charset=utf-8"),

        (Method::Get, ["", "api", "model"]) => json(200, shared.describe_model()),
        (Method::Get, ["", "api", "worker"]) => json(200, shared.describe_worker()),
        // Apart from the rest because it answers a different question: what the machine has
        // left changes on its own, with nothing here having done anything.
        (Method::Get, ["", "api", "machine"]) => json(200, machine::describe(shared.runtime())),

        (Method::Get, ["", "api", "jobs"]) => list_jobs(shared, query),
        (Method::Post, ["", "api", "jobs"]) => submit(shared, request),
        (Method::Get, ["", "api", "jobs", id]) => match shared.store().find(id) {
            Some(job) => json(200, shared.describe_job(&job)),
            None => no_such_job(),
        },
        (Method::Get, ["", "api", "jobs", id, "output"]) => output(shared, id, range_of(request)),
        (Method::Post, ["", "api", "jobs", id, "cancel"]) => cancel(shared, id),
        (Method::Delete, ["", "api", "jobs", id]) => delete_job(shared, id),

        (Method::Post, ["", "api", "uploads"]) => upload(shared, request),
        (Method::Get, ["", "api", "uploads", id]) => match shared.store().upload_file(id) {
            Some((mime, path)) => match std::fs::read(path) {
                Ok(bytes) => media(mime, bytes, range_of(request).as_deref()),
                Err(error) => refused(410, &format!("that upload can no longer be read: {error}")),
            },
            None => refused(404, "there is no upload by that id"),
        },
        (Method::Delete, ["", "api", "uploads", id]) => match shared.store().delete_upload(id) {
            true => no_content(),
            false => refused(404, "there is no upload by that id"),
        },

        _ => refused(404, "no such page"),
    }
}

/// One of the files the page is drawn from, if that is what `path` names.
fn static_file(path: &str) -> Option<Reply> {
    let (said, mime) = match path {
        "/style.css" => (STYLE, "text/css; charset=utf-8"),
        "/app.js" => (SCRIPT, "text/javascript; charset=utf-8"),
        path => (
            VENDOR.iter().find(|(name, _)| *name == path)?.1,
            "text/javascript; charset=utf-8",
        ),
    };

    Some(page(said, mime))
}

/// Refuses a request that did not come from a page this program served.
///
/// Not a question of who is asking -- nothing here knows that -- but of where from. The server
/// listens on the loopback address, and every site open in a browser on this machine can send a
/// request there: a form on some page posting a job, or a name that resolves to 127.0.0.1 after
/// the page on it has loaded. Three things between them close that:
///
/// - `Host` has to be the loopback address by name or number, which a name some other site
///   rebound to it is not;
/// - `Origin`, where a browser sent one, has to be this server's own;
/// - a POST has to say what it carries, and say something a plain form cannot. A browser only
///   sends another site's request without asking first when it is one of those, and this server
///   never answers the asking -- so the request is never sent.
fn from_here(request: &Request) -> Result<(), Reply> {
    let header = |name: &'static str| {
        request
            .headers()
            .iter()
            .find(|header| header.field.equiv(name))
            .map(|header| header.value.as_str().to_string())
    };

    let host = header("Host").unwrap_or_default();
    if !is_loopback(&host) {
        return Err(refused(
            403,
            "this server only answers requests addressed to this machine",
        ));
    }
    if let Some(origin) = header("Origin") {
        if origin != format!("http://{host}") {
            return Err(refused(403, "this server only answers its own pages"));
        }
    }

    if *request.method() == Method::Post {
        let kind = header("Content-Type").unwrap_or_default();
        let essence = kind
            .split(';')
            .next()
            .unwrap_or("")
            .trim()
            .to_ascii_lowercase();
        let plain = [
            "",
            "text/plain",
            "application/x-www-form-urlencoded",
            "multipart/form-data",
        ];
        if plain.contains(&essence.as_str()) {
            return Err(refused(
                415,
                "say what the body is: application/json for a job, the file's own type for an upload",
            ));
        }
    }

    Ok(())
}

/// Whether a `Host` header names this machine: `localhost`, `127.0.0.1` or `[::1]`, with a port
/// or without one.
fn is_loopback(host: &str) -> bool {
    let name = match host.strip_prefix('[') {
        Some(rest) => rest.split(']').next().unwrap_or(""),
        None => host.split(':').next().unwrap_or(""),
    };
    matches!(name, "localhost" | "127.0.0.1" | "::1")
}

/// The jobs, oldest first: every one, or those `?ids=` names, in the order they were made.
fn list_jobs(shared: &Arc<Shared>, query: &str) -> Reply {
    let wanted: Option<Vec<&str>> = query
        .split('&')
        .find_map(|pair| pair.strip_prefix("ids="))
        .map(|ids| ids.split(',').filter(|id| !id.is_empty()).collect());

    let jobs: Vec<Value> = shared
        .store()
        .jobs()
        .iter()
        .filter(|job| {
            wanted
                .as_ref()
                .is_none_or(|wanted| wanted.contains(&job.id.as_str()))
        })
        .map(|job| shared.describe_job(job))
        .collect();

    json(200, json!({ "jobs": jobs }))
}

/// Takes a job: checks what it asks for, fills in what it left to the model, keeps it, and puts it
/// in line. Answered with 202 and where to ask after it.
fn submit(shared: &Arc<Shared>, request: &mut Request) -> Reply {
    let asked = match body(request) {
        Ok(body) => body,
        Err(error) => return refused(400, &error),
    };

    let kind = shared.kind();
    if let Some(named) = asked.get("kind").and_then(Value::as_str) {
        if Kind::named(named) != Some(kind) {
            return refused(
                400,
                &format!("this program runs {} jobs, not {named}", kind.name()),
            );
        }
    }

    let settled = match kind {
        Kind::Image => match shared.world().model.clone() {
            Some(chosen) => image_settings(shared, &asked, &chosen),
            None => Err("this program has no model to draw with".to_string()),
        },
        Kind::Speech => match shared.world().voice.clone() {
            Some(voice) => speech_settings(shared, &asked, &voice),
            None => Err("this program has no voice to speak with".to_string()),
        },
        Kind::Conversion => match shared.world().converter.clone() {
            Some(converter) => conversion_settings(shared, &asked, &converter),
            None => Err("this program has nothing to convert with".to_string()),
        },
    };
    let settled = match settled {
        Ok(settled) => settled,
        Err(error) => return refused(400, &error),
    };

    if shared.store().queued() >= MOST_QUEUED {
        return refused(
            503,
            &format!("{MOST_QUEUED} jobs are already waiting: ask again when fewer are"),
        );
    }

    let job = match shared.store().submit(kind, settled) {
        Ok(job) => job,
        Err(error) => return refused(500, &format!("the job could not be kept: {error}")),
    };

    let mut reply = json(202, shared.describe_job(&job));
    add_header(&mut reply, "Location", &format!("/api/jobs/{}", job.id));
    reply
}

/// A picture job's settings, every one of them: what was asked for, checked, and the model's own
/// for what was not.
fn image_settings(shared: &Shared, asked: &Value, chosen: &Chosen) -> Result<Value, String> {
    let prompt = asked
        .get("prompt")
        .and_then(Value::as_str)
        .unwrap_or_default();
    if prompt.trim().is_empty() {
        return Err("there is nothing to draw: give it a prompt".to_string());
    }

    let init_image = match asked.get("init_image").and_then(Value::as_str) {
        Some(id) => {
            let upload = shared
                .store()
                .find_upload(id)
                .ok_or_else(|| format!("there is no upload {id} to draw from"))?;
            if !upload.mime.starts_with("image/") {
                return Err(format!("upload {id} is not a picture"));
            }
            if let Some(why) = &chosen.no_picture_because {
                return Err(why.clone());
            }
            Some(upload.id)
        }
        None => None,
    };

    let (guidance, negative) = steering(asked, chosen);
    Ok(json!({
        "prompt": prompt,
        "negative": negative,
        "width": pixels(asked.get("width"), chosen.defaults.width),
        "height": pixels(asked.get("height"), chosen.defaults.height),
        "steps": whole(asked.get("steps"), chosen.defaults.num_steps).clamp(1, 150),
        "guidance": guidance,
        // Never left open. A job asked for without a seed is given a fresh one rather than left
        // to whatever the device's generator was last used for, so that what comes out says
        // which number drew it and can be drawn again. A string: it is sixty-four bits.
        "seed": seed(asked.get("seed")).to_string(),
        "strength": decimal(asked.get("strength"), 0.8).clamp(0.0, 1.0),
        "init_image": init_image,
        "model": chosen.name,
    }))
}

/// The same, for something to read out.
fn speech_settings(shared: &Shared, asked: &Value, voice: &Spoken) -> Result<Value, String> {
    let text = asked
        .get("text")
        .and_then(Value::as_str)
        .unwrap_or_default();
    if text.trim().is_empty() {
        return Err("there is nothing to say: give it some text to read out".to_string());
    }

    let reference = match asked.get("reference").and_then(Value::as_str) {
        Some(id) => {
            let upload = shared
                .store()
                .find_upload(id)
                .ok_or_else(|| format!("there is no upload {id} to sound like"))?;
            if upload.mime != "audio/wav" {
                return Err(format!("upload {id} is not a WAV recording"));
            }
            if let Some(why) = &voice.no_likeness_because {
                return Err(why.clone());
            }
            Some(upload.id)
        }
        None => None,
    };

    let style = speech_style(asked, voice)?;

    Ok(json!({
        "text": text,
        "style": style,
        "speed": decimal(asked.get("speed"), voice.defaults.speed).clamp(0.25, 4.0),
        "temperature": decimal(asked.get("temperature"), voice.defaults.temperature).clamp(0.0, 2.0),
        "seed": seed(asked.get("seed")).to_string(),
        "reference": reference,
        "voice": voice.name,
    }))
}

/// The same, for a recording to say again in another voice.
fn conversion_settings(
    shared: &Shared,
    asked: &Value,
    converter: &ChosenConverter,
) -> Result<Value, String> {
    // Both recordings are needed, and both have to be WAVs: the page decodes whatever was dropped
    // and posts the samples, and anything else posting here is asked for the same.
    let recording = |field: &str, what: &str| -> Result<String, String> {
        let id = asked
            .get(field)
            .and_then(Value::as_str)
            .ok_or_else(|| format!("there is nothing to convert without {what}"))?;
        let upload = shared
            .store()
            .find_upload(id)
            .ok_or_else(|| format!("there is no upload {id} to use as {what}"))?;
        if upload.mime != "audio/wav" {
            return Err(format!("upload {id} is not a WAV recording"));
        }
        Ok(upload.id)
    };
    let source = recording("source", "a recording to convert")?;
    let reference = recording("reference", "a recording of the voice to convert it to")?;

    Ok(json!({
        "source": source,
        "reference": reference,
        "steps": whole(asked.get("steps"), converter.defaults.steps).clamp(1, 100),
        "style": asked.get("style").and_then(Value::as_bool).unwrap_or(false),
        "seed": seed(asked.get("seed")).to_string(),
        "converter": converter.name,
    }))
}

/// The instruction a reading is asked to follow, or `None` for the voice's own way of reading.
///
/// Free text rather than one of the list's: the list is what the model was taught, and the page
/// says so, but an instruction written in the same manner is the caller's to try.
fn speech_style(asked: &Value, voice: &Spoken) -> Result<Option<String>, String> {
    match asked.get("style") {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(style)) if style.trim().is_empty() => Ok(None),
        Some(Value::String(style)) => {
            if voice.styles.is_empty() {
                return Err(format!("{} takes no style", voice.full_name));
            }
            if style.chars().count() > LONGEST_STYLE {
                return Err(format!(
                    "a style is a sentence to follow, at most {LONGEST_STYLE} characters"
                ));
            }
            // The model's own marker would end the instruction early and read the rest aloud.
            if style.contains("<|") {
                return Err("a style cannot hold a marker such as <|endofprompt|>".to_string());
            }
            Ok(Some(style.trim().to_string()))
        }
        Some(_) => Err("a style is a string".to_string()),
    }
}

/// What a finished job made.
fn output(shared: &Arc<Shared>, id: &str, range: Option<String>) -> Reply {
    let Some(job) = shared.store().find(id) else {
        return no_such_job();
    };
    let Some((mime, path)) = shared.store().output_file(id) else {
        return refused(
            404,
            &format!("it has not made anything: it is {}", job.status.name()),
        );
    };

    match std::fs::read(path) {
        Ok(bytes) => media(mime, bytes, range.as_deref()),
        Err(error) => refused(410, &format!("what it made can no longer be read: {error}")),
    }
}

/// Takes a waiting job out of line, or asks a running one to stop after the step it is on.
fn cancel(shared: &Arc<Shared>, id: &str) -> Reply {
    let Some(job) = shared.store().find(id) else {
        return no_such_job();
    };

    match shared.cancel(id) {
        // Answered with how it is now, which for a job taken out of line is cancelled.
        Cancelled::Dequeued => json(
            200,
            shared.describe_job(&shared.store().find(id).unwrap_or(job)),
        ),
        // Asked, and not done yet: the job says it is stopping until it has.
        Cancelled::Stopping => json(202, shared.describe_job(&job)),
        Cancelled::NotGoing => refused(
            409,
            &format!("there is nothing to stop: it is {}", job.status.name()),
        ),
    }
}

/// Deletes a job and what it made, stopping it first if it has not finished.
fn delete_job(shared: &Arc<Shared>, id: &str) -> Reply {
    let Some(job) = shared.store().find(id) else {
        return no_such_job();
    };
    if matches!(job.status, Status::Queued | Status::Running) {
        shared.cancel(id);
    }

    match shared.store().delete(id) {
        true => no_content(),
        false => no_such_job(),
    }
}

/// Keeps a picture or a recording for a job to start from.
///
/// Posted as the bytes of the file and nothing else -- no form, no boundary, no field names. What
/// crosses is one file, the page knows which, and a multipart body would be a parser this program
/// owns in order to unwrap something it already has. What kind of file it is is read off the
/// bytes rather than taken from the request: that is the one part of it that cannot be wrong.
fn upload(shared: &Arc<Shared>, request: &mut Request) -> Reply {
    let mut bytes = Vec::new();
    if let Err(error) = request
        .as_reader()
        .take(MOST_OF_A_FILE + 1)
        .read_to_end(&mut bytes)
    {
        return refused(400, &format!("the file did not arrive whole: {error}"));
    }
    if bytes.len() as u64 > MOST_OF_A_FILE {
        return refused(413, "that file is bigger than anything a job starts from");
    }
    if bytes.is_empty() {
        return refused(400, "the file was empty");
    }

    // A recording is read here rather than at the job, alone among the things this server keeps:
    // a WAV that is not one is found out about where there is still a request to refuse, rather
    // than as a job that fails after waiting its turn.
    if bytes.starts_with(b"RIFF") {
        if let Err(error) = crate::wav::read(&bytes) {
            return refused(400, &error.to_string());
        }
    }

    match shared.store().upload(&bytes) {
        Ok(upload) => {
            let mut reply = json(201, upload.json());
            add_header(
                &mut reply,
                "Location",
                &format!("/api/uploads/{}", upload.id),
            );
            reply
        }
        Err(Refused::NotAFile) => refused(415, &Refused::NotAFile.to_string()),
        Err(error) => refused(500, &format!("the file could not be kept: {error}")),
    }
}

fn no_such_job() -> Reply {
    refused(404, "there is no job by that id")
}

fn no_content() -> Reply {
    Response::from_data(Vec::new()).with_status_code(204)
}

fn add_header(reply: &mut Reply, name: &str, value: &str) {
    if let Ok(header) = Header::from_bytes(name.as_bytes(), value.as_bytes()) {
        reply.add_header(header);
    }
}

/// The JSON a request carried, or what was wrong with it.
fn body(request: &mut Request) -> Result<Value, String> {
    let mut said = String::new();
    request
        .as_reader()
        .take(MOST_OF_A_REQUEST)
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

/// How hard to push towards the prompt and what to push away from, for a run of this model.
///
/// Both or neither. Guidance is the second pass through the denoiser and the negative prompt is
/// what that pass is given, so a model distilled to answer in one has nowhere to put either: what
/// arrived is dropped, and the run is the plain one the model was trained to do.
///
/// Dropped here and not only left off the screen. The page does not show the boxes, but the page
/// is not the only thing that can post to this, and a run that kept a negative prompt it never
/// encoded would go on to say so on the card under the picture -- which is the line somebody
/// reads to find out what drew it.
fn steering(asked: &Value, chosen: &Chosen) -> (f32, String) {
    if !chosen.takes_guidance {
        return (chosen.defaults.guidance_scale, String::new());
    }

    let guidance = decimal(asked.get("guidance"), chosen.defaults.guidance_scale).clamp(1.0, 30.0);
    let negative = asked
        .get("negative")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();

    (guidance, negative)
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

fn json(status: u16, value: Value) -> Reply {
    reply(status, "application/json", value.to_string().into_bytes())
}

/// An answer that is not the one that was asked for, with the reason in the same shape everything
/// else answers in, so that the page can show it without knowing which request it came from.
///
/// Said in the terminal as well, on the line before the request's own: a status code says that a
/// request was turned away and nothing about why.
fn refused(status: u16, said: &str) -> Reply {
    crate::cli::webui::log::line(format_args!("refused ({status}): {said}"));
    reply(
        status,
        "application/json",
        json!({ "error": said }).to_string().into_bytes(),
    )
}

/// What a request's `Range` header asked for, if it sent one.
fn range_of(request: &Request) -> Option<String> {
    request
        .headers()
        .iter()
        .find(|header| header.field.equiv("Range"))
        .map(|header| header.value.as_str().to_string())
}

/// Something an `<audio>` element plays: sent with its length, and in pieces when asked for them.
///
/// An ordinary [`reply`] is not enough for one. `tiny_http` sends anything over 32 KB with chunked
/// transfer encoding and no `Content-Length`, which a picture does not mind and a media element
/// does: it asks for byte ranges so that it can find the length and seek, and a response of
/// unknown length that ignores the range is one it gives up on -- the player on the speech tab
/// showed an error over a clip that was a perfectly good WAV. `Tones` never found this out,
/// because its clips were small and the tab was hidden.
///
/// So the length is always sent, `Accept-Ranges` says ranges may be asked for, and a single range
/// -- `bytes=a-b`, `bytes=a-` or `bytes=-n`, which is all a browser sends a media file -- comes
/// back as a `206` with its `Content-Range`. A range that cannot be satisfied is a `416`.
fn media(mime: &str, body: Vec<u8>, range: Option<&str>) -> Reply {
    let length = body.len();

    let (status, body, span) = match range.map(|range| byte_range(range, length)) {
        None => (200, body, None),
        Some(Some((first, last))) => (206, body[first..=last].to_vec(), Some((first, last))),
        Some(None) => {
            let mut response = reply(416, mime, Vec::new());
            if let Ok(header) =
                Header::from_bytes(&b"Content-Range"[..], format!("bytes */{length}").as_bytes())
            {
                response.add_header(header);
            }
            return response.with_chunked_threshold(usize::MAX);
        }
    };

    let mut response = reply(status, mime, body).with_chunked_threshold(usize::MAX);
    if let Ok(header) = Header::from_bytes(&b"Accept-Ranges"[..], &b"bytes"[..]) {
        response.add_header(header);
    }
    if let Some((first, last)) = span {
        if let Ok(header) = Header::from_bytes(
            &b"Content-Range"[..],
            format!("bytes {first}-{last}/{length}").as_bytes(),
        ) {
            response.add_header(header);
        }
    }

    response
}

/// The inclusive byte span a `Range` header asks for out of `length`, or `None` where it asks for
/// nothing this can give -- another unit, several ranges, or a span that starts past the end.
///
/// An end past the last byte is cut to it rather than refused, which is what the standard says to
/// do and what a browser relies on when it asks for `bytes=0-` of something it has not seen yet.
fn byte_range(range: &str, length: usize) -> Option<(usize, usize)> {
    let spec = range.trim().strip_prefix("bytes=")?;
    if spec.contains(',') || length == 0 {
        return None;
    }

    let (first, last) = spec.split_once('-')?;
    let (first, last) = (first.trim(), last.trim());

    let (first, last) = match (first.is_empty(), last.is_empty()) {
        // `bytes=-n`: the last n bytes.
        (true, false) => {
            let suffix: usize = last.parse().ok()?;
            if suffix == 0 {
                return None;
            }
            (length.saturating_sub(suffix), length - 1)
        }
        // `bytes=a-`: from a to the end.
        (false, true) => (first.parse().ok()?, length - 1),
        // `bytes=a-b`.
        (false, false) => {
            let (first, last): (usize, usize) = (first.parse().ok()?, last.parse().ok()?);
            (first, last.min(length - 1))
        }
        (true, true) => return None,
    };

    (first <= last && first < length).then_some((first, last))
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

    use crate::cli::webui::worker;

    /// The three forms a browser sends a media file, and what each asks for out of 1000 bytes.
    #[test]
    fn a_range_is_the_bytes_it_names() {
        assert_eq!(byte_range("bytes=0-", 1000), Some((0, 999)));
        assert_eq!(byte_range("bytes=100-199", 1000), Some((100, 199)));
        assert_eq!(byte_range("bytes=-10", 1000), Some((990, 999)));

        // An end past the last byte is cut to it, which is what `bytes=0-` relies on too.
        assert_eq!(byte_range("bytes=900-5000", 1000), Some((900, 999)));
        // A suffix longer than the whole is the whole.
        assert_eq!(byte_range("bytes=-5000", 1000), Some((0, 999)));
    }

    #[test]
    fn a_range_that_cannot_be_given_is_none() {
        assert_eq!(byte_range("bytes=1000-", 1000), None);
        assert_eq!(byte_range("bytes=500-100", 1000), None);
        assert_eq!(byte_range("bytes=0-1,5-9", 1000), None);
        assert_eq!(byte_range("items=0-1", 1000), None);
        assert_eq!(byte_range("bytes=-0", 1000), None);
        assert_eq!(byte_range("bytes=-", 1000), None);
        assert_eq!(byte_range("bytes=0-", 0), None);
    }

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
    fn a_model_with_no_second_pass_is_asked_for_neither_of_the_two() {
        // Both of them in the request, which is what a page talking to an older build would send
        // and what anything posting here by hand may send at any time.
        let asked = json!({"guidance": 7.5, "negative": "worst quality, bad hands"});

        // Krea 2 Turbo is distilled: one pass, already as though it had been guided. Whatever
        // arrived, the run is the plain one -- at the guidance the model itself asked for, which
        // is this runtime's spelling of the reference's zero.
        let turbo = worker::look_at("krea2:turbo:v1.0");
        assert!(!turbo.takes_guidance);
        let (guidance, negative) = steering(&asked, &turbo);
        assert_eq!(guidance, turbo.defaults.guidance_scale);
        assert_eq!(negative, "");

        // And the same request to a model that has a second pass, where both are what was asked
        // for. This is the half that would still pass if the two were simply never read.
        let sdxl = worker::look_at("sdxl:base:v1.0");
        assert!(sdxl.takes_guidance);
        let (guidance, negative) = steering(&asked, &sdxl);
        assert_eq!(guidance, 7.5);
        assert_eq!(negative, "worst quality, bad hands");

        // A request that says neither leaves that model where its own card put it.
        let (guidance, negative) = steering(&json!({}), &sdxl);
        assert_eq!(guidance, sdxl.defaults.guidance_scale);
        assert_eq!(negative, "");
    }

    #[test]
    fn a_style_is_taken_only_by_a_voice_that_offers_them() {
        let cosyvoice = worker::look_at_voice("cosyvoice:v3");
        assert!(!cosyvoice.styles.is_empty());
        let style = |asked: Value, voice| speech_style(&asked, voice);

        assert_eq!(
            style(json!({"style": " 请非常开心地说一句话。 "}), &cosyvoice),
            Ok(Some("请非常开心地说一句话。".to_string()))
        );
        // Not asking, asking for nothing, and asking with nothing in it are all the plain reading.
        assert_eq!(style(json!({}), &cosyvoice), Ok(None));
        assert_eq!(style(json!({"style": null}), &cosyvoice), Ok(None));
        assert_eq!(style(json!({"style": "  "}), &cosyvoice), Ok(None));

        // What would reach the model as something other than words to follow is refused here,
        // where the caller is told, rather than in the worker after the job has queued.
        assert!(style(json!({"style": "开心<|endofprompt|>"}), &cosyvoice).is_err());
        assert!(style(json!({"style": "长".repeat(LONGEST_STYLE + 1)}), &cosyvoice).is_err());
        assert!(style(json!({"style": 7}), &cosyvoice).is_err());

        // And a voice with no list says so, rather than reading the sentence as if it had.
        let indextts = worker::look_at_voice("indextts");
        assert!(indextts.styles.is_empty());
        let refused = style(json!({"style": "请非常开心地说一句话。"}), &indextts).unwrap_err();
        assert!(refused.contains("takes no style"), "{refused}");
    }
}
