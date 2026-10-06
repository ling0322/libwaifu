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

//! The webui command: a page in a browser, a model behind it, and a file at the end.
//!
//! Two kinds of file, since the page grew a third tab: a PNG from a diffusion model, and a WAV
//! from a voice. Everything below this line is the same for both -- one worker thread, one lock
//! it reports through, one server -- which is the reason speech went into this page rather than
//! into a second command with a second copy of all of it.
//!
//! A run takes minutes rather than milliseconds, which is what the shape of this is about. The
//! model sits on a thread of its own -- a tensor never leaves the thread that made it -- and the
//! browser talks to it through a server: a command goes one way down a channel, and how far along
//! the run is, is read back out of a lock the worker writes as it goes.
//!
//! It is a browser rather than the terminal for one reason, which is that the thing being made is
//! a picture. A terminal can say that one was written and where; it cannot show it. What comes
//! before the page -- which task, which model, which device, and the fetch -- is still asked in
//! the terminal (see [`tui`](crate::cli::tui)), and the page is served for the answers: it offers
//! no model list, no download and no device box of its own.

mod http;
mod log;
mod machine;
mod state;
mod store;
mod worker;

use std::io;
use std::net::{Ipv4Addr, SocketAddr, TcpListener};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::cli::args::Args;
use crate::cli::hub;
use crate::cli::task::{Launch, Task};
use crate::cli::tui;
use crate::cli::webui::state::Shared;
use crate::cli::webui::store::Store;
use crate::cli::webui::worker::Load;

type Error = Box<dyn std::error::Error>;

/// Where the server listens. The loopback address and nothing else: this serves a page that will
/// read a file off the disk and start a run on the card, which is not a thing to offer the
/// network a laptop is on. Somebody who wants it further than this machine has an ssh tunnel.
const HOST: Ipv4Addr = Ipv4Addr::LOCALHOST;

/// The port to try first, which is the one every other tool of this kind uses -- so that a
/// bookmark, a browser's history and whatever muscle memory someone arrived with all still work.
const PORT: u16 = 7860;

/// How many ports past that one to try before giving up.
///
/// A port already taken is most often this program still running in another window, or the tool
/// this one borrowed the number from. Neither is a reason to refuse to start.
const NEARBY: u16 = 16;

/// How many requests can be answered at once.
///
/// Small on purpose. Nothing here is slow -- the slow thing is on the far side of a channel --
/// and what these threads actually do is read a little JSON and copy a file. Two would very
/// nearly do; four means a browser that has opened several at once never waits.
const ANSWERERS: usize = 4;

fn print_usage() {
    eprintln!("Usage: waifu webui [OPTIONS]");
    eprintln!();
    eprintln!("Options:");
    crate::cli::args::print_options();
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
    let wanted_port = with_usage(args.port())?;
    let output = PathBuf::from(args.output());
    let limit = with_usage(args.output_limit())?;

    // Read now rather than when a run begins, so that a path that is not there is said here --
    // and before the terminal is taken over by the screens, which would hide it.
    let picture = match args.image() {
        Some(picture) => Some(
            std::fs::read(picture).map_err(|error| format!("could not read {picture}: {error}"))?,
        ),
        None => None,
    };

    // Made now for the same reason: a directory that cannot be written to is something to find
    // out before a model has been fetched, not after the first picture has been drawn into it.
    std::fs::create_dir_all(&output)
        .map_err(|error| format!("could not make {}: {error}", output.display()))?;

    let launch = match model {
        // Named on the command line: nothing to ask, and the task is what the model is where it
        // was not named. Fetched here, in the terminal, so that the page never has a download.
        Some(model) => {
            let task = match asked_task {
                Some(task) => task,
                None => task_for(&model, picture.is_some()),
            };
            with_usage(fits(task, &model))?;
            fetch_in_the_terminal(&model)?;
            // After the fetch, since it is the packages on the disk that are measured.
            let runtime = device.resolve_for(hub::model_bytes(&model));
            Launch {
                task,
                model,
                runtime,
            }
        }
        // Not named: the terminal asks, where there is a terminal to ask on.
        None => {
            if !tui::can_take_the_screen() {
                return with_usage(Err(
                    "no model was named, and there is no terminal to pick one on: name one with -m"
                        .into(),
                ));
            }
            match tui::choose(asked_task, device)? {
                Some(launch) => launch,
                // Somebody who looked at the lists and left. Not a failure.
                None => return Ok(()),
            }
        }
    };

    serve_the_page(launch, picture, wanted_port, &output, limit)
}

/// The task `-m` asks for when `-task` does not say: a voice reads, a converter converts, and a
/// picture model draws -- from the picture, where `-i` named one. A manifest on the disk is a
/// voice or a converter where its `model.type` names one.
fn task_for(model: &str, from_a_picture: bool) -> Task {
    if worker::is_a_voice(model) {
        Task::Text2Speech
    } else if worker::is_a_converter(model) {
        Task::Speech2Speech
    } else if from_a_picture {
        Task::Img2Img
    } else {
        Task::Txt2Img
    }
}

/// Whether `model` can do `task`, as far as can be told before reading it: a published model is
/// a voice or a picture model by its name, and a picture model can start from a picture or not by
/// its kind. A manifest on the disk is taken at its word, and says so at the first run if not.
fn fits(task: Task, model: &str) -> Result<(), String> {
    let published = hub::full_name(model).is_some() || model == worker::TONES;
    let voice = model == worker::TONES || hub::is_voice(model);
    let converter = hub::is_conversion(model);

    let fitting = match task {
        Task::Text2Speech => voice,
        Task::Speech2Speech => converter,
        Task::Txt2Img | Task::Img2Img => !voice && !converter,
    };
    if published && !fitting {
        let is = match (voice, converter) {
            (true, _) => "is a voice",
            (_, true) => "converts voices",
            _ => "draws pictures",
        };
        let needs = match task {
            Task::Text2Speech => "a voice",
            Task::Speech2Speech => "a converter",
            Task::Txt2Img | Task::Img2Img => "a model that draws",
        };
        return Err(format!("{model} {is}: {} needs {needs}", task.name()));
    }
    if task == Task::Img2Img && !voice {
        if let Some(why) = worker::look_at(model).no_picture_because {
            return Err(format!("{model} cannot do img2img: {why}"));
        }
    }

    Ok(())
}

/// Whether a picture model can start from a picture, as far as is known without reading it: what
/// the terminal's img2img list is filtered on.
pub fn draws_from_a_picture(model: &str) -> bool {
    worker::look_at(model).no_picture_because.is_none()
}

/// Fetches a published model that is not on the disk yet, saying how it is going on one line
/// that rewrites itself -- the command line's version of the terminal's bar, for `-m`.
///
/// Nothing for a model already here, which is asked of the cache first: fetching one that is
/// here would still go looking for which hub to ask, and a machine that is offline would wait on
/// that for nothing. A path is checked for being there, which is the whole of what fetching one
/// means.
fn fetch_in_the_terminal(model: &str) -> Result<(), Error> {
    if model == worker::TONES || hub::is_cached(model) {
        return Ok(());
    }

    let mut from = String::new();
    let mut report = |progress: hub::Progress| {
        let (file, done, total, part, parts) = match progress {
            hub::Progress::From { hub } => {
                from = format!(" from {hub}");
                return;
            }
            hub::Progress::Fetching {
                file,
                done,
                total,
                part,
                parts,
            } => (file, done, total, part, parts),
            hub::Progress::Fetched {
                file,
                bytes,
                part,
                parts,
            } => (file, bytes, Some(bytes), part, parts),
        };

        let of = match parts {
            0 => String::new(),
            parts => format!(" ({part} of {parts})"),
        };
        let how_far = match total {
            Some(total) if total > 0 => format!(
                "{:.0}% of {:.2} GB",
                done as f64 / total as f64 * 100.0,
                total as f64 / 1e9
            ),
            _ => format!("{:.2} GB", done as f64 / 1e9),
        };
        eprint!("\r\x1b[2Kfetching {file}{from}{of}: {how_far}");
    };

    let fetched = hub::resolve_reporting(model, &mut report, &|| false);
    eprintln!();
    fetched.map(|_| ())
}

/// Serves the page for what was launched, until the process is stopped.
fn serve_the_page(
    launch: Launch,
    picture: Option<Vec<u8>>,
    wanted_port: Option<u16>,
    output: &Path,
    limit: Option<u64>,
) -> Result<(), Error> {
    // Taken before the model is read, and not answered on until it has been. A port that is taken
    // is a session that cannot go anywhere, and finding that out after a minute of reading a model
    // is finding it out late; nothing is served from it, and no address is printed, until the
    // model is in memory.
    let (listener, address) = listen(wanted_port)?;

    let store = Store::open(output, limit)
        .map_err(|error| format!("could not use {}: {error}", output.display()))?;
    let mut shared = Shared::new(launch.task, launch.runtime, store);
    if let Some(bytes) = picture {
        shared
            .start_holding(&bytes)
            .map_err(|error| format!("could not keep the picture -i named: {error}"))?;
    }

    // Read onto the device before the page opens, on the worker -- the thread the weights have to
    // live on -- with this one waiting for it in the terminal. The page never opens on a model
    // that is still being read, and one that cannot be read is said here rather than on a page.
    let load = match launch.task {
        Task::Text2Speech => {
            shared.change(|world| world.voice = Some(worker::look_at_voice(&launch.model)));
            Load::Voice(launch.model.clone())
        }
        Task::Speech2Speech => {
            shared.change(|world| world.converter = Some(worker::look_at_converter(&launch.model)));
            Load::Converter(launch.model.clone())
        }
        Task::Txt2Img | Task::Img2Img => {
            shared.change(|world| world.model = Some(worker::look_at(&launch.model)));
            Load::Model(launch.model.clone())
        }
    };
    shared.reading(true);
    let shared = Arc::new(shared);

    let worker = std::thread::spawn({
        let shared = Arc::clone(&shared);
        move || worker::work(&shared, load)
    });
    read_in_the_terminal(&shared, &launch)?;

    println!(
        "waifu is at http://{address} -- {} with {} on {}",
        launch.task.name(),
        launch.model,
        launch.runtime.name()
    );
    println!(
        "Jobs and what they made are kept in {} ({}).",
        output.display(),
        match shared.store().limit() {
            Some(limit) => format!("up to {}, oldest deleted first", size(limit)),
            None => "with no limit".to_string(),
        }
    );
    println!("Press ctrl-c to stop.");

    serve(listener, shared)?;
    let _ = worker.join();

    Ok(())
}

/// Waits for the worker to finish reading what `launch` names, saying how long it has been on one
/// line that rewrites itself, and says why where it could not be read.
fn read_in_the_terminal(shared: &Shared, launch: &Launch) -> Result<(), Error> {
    let started = std::time::Instant::now();
    while shared.is_busy() {
        let doing = shared.world().doing.words();
        let doing = match doing.is_empty() {
            true => format!("reading {}", launch.model),
            false => doing,
        };
        eprint!(
            "\r\x1b[2K{doing} onto {} -- {}s",
            launch.runtime.name(),
            started.elapsed().as_secs()
        );
        std::thread::sleep(std::time::Duration::from_millis(200));
    }
    eprintln!();

    let world = shared.world();
    let read = match launch.task {
        Task::Text2Speech => world.voice.as_ref().is_some_and(|voice| voice.in_memory),
        Task::Speech2Speech => world
            .converter
            .as_ref()
            .is_some_and(|converter| converter.in_memory),
        Task::Txt2Img | Task::Img2Img => world.model.as_ref().is_some_and(|model| model.in_memory),
    };
    if read {
        eprintln!(
            "{} is on {}, read in {:.1}s",
            launch.model,
            launch.runtime.name(),
            started.elapsed().as_secs_f64()
        );
        return Ok(());
    }

    let why = world
        .note
        .clone()
        .unwrap_or_else(|| "it stopped without saying why".to_string());
    Err(format!("could not read {}: {why}", launch.model).into())
}

/// A number of bytes, in the unit that says it in a few digits.
fn size(bytes: u64) -> String {
    match bytes {
        bytes if bytes >= 1 << 30 => format!("{:.1} GB", bytes as f64 / (1u64 << 30) as f64),
        bytes if bytes >= 1 << 20 => format!("{:.0} MB", bytes as f64 / (1u64 << 20) as f64),
        bytes => format!("{bytes} bytes"),
    }
}

/// Takes a port to listen on: the one that was asked for, or the first free one near it.
///
/// A port someone named is taken as meant -- they have something pointed at it -- so a refusal
/// there is a refusal. It is only the default that walks, because the default is a number this
/// program chose rather than one anybody asked for.
fn listen(wanted: Option<u16>) -> Result<(TcpListener, SocketAddr), Error> {
    let first = wanted.unwrap_or(PORT);
    let past = match wanted {
        Some(_) => first,
        None => first.saturating_add(NEARBY),
    };

    let mut refused = None;
    for port in first..=past {
        match TcpListener::bind(SocketAddr::from((HOST, port))) {
            Ok(listener) => {
                let address = listener.local_addr()?;
                return Ok((listener, address));
            }
            Err(error) if error.kind() == io::ErrorKind::AddrInUse => refused = Some(error),
            Err(error) => return Err(error.into()),
        }
    }

    Err(match refused {
        Some(error) => format!("nothing is free between port {first} and {past}: {error}").into(),
        None => format!("could not listen on port {first}").into(),
    })
}

/// Answers requests until the process is stopped.
///
/// There is no way out of this short of ctrl-c, which is the honest shape: the page is the
/// program, and a page that could shut the server down from a button is a page any other tab on
/// the machine could shut down too.
fn serve(listener: TcpListener, shared: Arc<Shared>) -> Result<(), Error> {
    // Mapped rather than passed on: what comes back is a `Send + Sync` box, which is not the
    // plain box everything here reports with and does not convert into one on its own.
    let server = tiny_http::Server::from_listener(listener, None)
        .map_err(|error| format!("could not answer on that port: {error}"))?;
    let server = Arc::new(server);

    let answerers: Vec<_> = (0..ANSWERERS)
        .map(|_| {
            let server = Arc::clone(&server);
            let shared = Arc::clone(&shared);

            std::thread::spawn(move || {
                while let Ok(mut request) = server.recv() {
                    let method = request.method().to_string();
                    let path = request.url().split('?').next().unwrap_or("/").to_string();
                    let started = std::time::Instant::now();
                    let reply = http::answer(&shared, &mut request);
                    log::request(&method, &path, reply.status_code().0, started.elapsed());

                    // A browser that navigated away mid-request is not this program's problem,
                    // and it is the commonest way for this to fail.
                    let _ = request.respond(reply);
                }
            })
        })
        .collect();

    for answerer in answerers {
        let _ = answerer.join();
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::io::{Read, Write};
    use std::net::TcpStream;

    use serde_json::{json, Value};

    use crate::cli::args::DeviceOption;

    /// A port nothing is on, as far as anything can tell: taken from the operating system and
    /// given straight back. Racy in principle and settled in practice, since what asks for one
    /// next is the line after this.
    fn a_free_port() -> u16 {
        TcpListener::bind(SocketAddr::from((HOST, 0)))
            .expect("a port")
            .local_addr()
            .expect("its number")
            .port()
    }

    #[test]
    fn a_port_somebody_named_and_cannot_have_is_refused() {
        // Named, so they have something pointed at it; a quiet walk to the next one along would
        // put the page somewhere other than where they are about to look for it.
        let port = a_free_port();
        let _held = TcpListener::bind(SocketAddr::from((HOST, port))).expect("to hold it");

        let refused = listen(Some(port)).expect_err("the port is taken");
        assert!(refused.to_string().contains(&port.to_string()));
    }

    #[test]
    fn a_model_named_on_the_command_line_says_which_task_it_is_for() {
        // A voice reads, and a picture model draws -- from the picture, where -i named one.
        assert_eq!(task_for("indextts", false), Task::Text2Speech);
        assert_eq!(task_for(worker::TONES, true), Task::Text2Speech);
        assert_eq!(task_for("sdxl:base", false), Task::Txt2Img);
        assert_eq!(task_for("sdxl:base", true), Task::Img2Img);
        // A manifest that is not there, or says nothing of a speech model, draws until -task says
        // otherwise.
        assert_eq!(task_for("/somewhere/else.yaml", false), Task::Txt2Img);
    }

    #[test]
    fn a_task_the_model_cannot_do_is_refused_before_anything_is_fetched() {
        assert!(fits(Task::Txt2Img, "sdxl:base").is_ok());
        assert!(fits(Task::Img2Img, "sdxl:base").is_ok());
        assert!(fits(Task::Text2Speech, "indextts").is_ok());
        assert!(fits(Task::Text2Speech, worker::TONES).is_ok());

        let refused = fits(Task::Text2Speech, "sdxl:base").unwrap_err();
        assert!(refused.contains("needs a voice"), "{refused}");
        let refused = fits(Task::Txt2Img, "indextts").unwrap_err();
        assert!(refused.contains("is a voice"), "{refused}");

        // What the img2img list in the terminal is filtered on, said for a model named instead.
        let refused = fits(Task::Img2Img, "anima:turbo").unwrap_err();
        assert!(refused.contains("cannot do img2img"), "{refused}");
        assert!(!draws_from_a_picture("krea2:turbo"));
        assert!(draws_from_a_picture("sdxl:base"));

        // A manifest on the disk is taken at its word.
        assert!(fits(Task::Text2Speech, "/somewhere/voice.yaml").is_ok());

        // Speech2speech is for a converter.
        let refused = fits(Task::Speech2Speech, "indextts").unwrap_err();
        assert!(refused.contains("needs a converter"), "{refused}");

        // CosyVoice3 does both: it reads, and it converts.
        assert!(fits(Task::Text2Speech, "cosyvoice").is_ok());
        assert!(fits(Task::Speech2Speech, "cosyvoice").is_ok());
        assert_eq!(task_for("cosyvoice", false), Task::Text2Speech);
    }

    /// A program for `task`, keeping its jobs in a directory of the calling test's own, with
    /// `model` described the way `main` describes what the terminal chose -- and a voice, the
    /// stand-in, for one that speaks.
    fn a_program(called: &str, task: Task, model: Option<&str>) -> Shared {
        let root =
            std::env::temp_dir().join(format!("libwaifu-webui-{called}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let shared = Shared::new(
            task,
            DeviceOption::Cpu.resolve(),
            Store::open(&root, None).expect("a store"),
        );
        match task {
            Task::Text2Speech => shared.change(|world| {
                world.voice = Some(worker::look_at_voice(model.unwrap_or(worker::TONES)))
            }),
            Task::Speech2Speech => shared.change(|world| {
                world.converter = Some(worker::look_at_converter(model.unwrap_or("cosyvoice")))
            }),
            Task::Txt2Img | Task::Img2Img => {
                if let Some(model) = model {
                    shared.change(|world| world.model = Some(worker::look_at(model)));
                }
            }
        }
        shared
    }

    /// `shared`, answering on a port of its own, with no worker behind it: what is posted waits
    /// in line, which is what most of these are about. Left running for the rest of the test
    /// process, which is the same shape the program has.
    fn a_server(shared: Shared) -> (SocketAddr, Arc<Shared>) {
        let (listener, address) = listen(Some(a_free_port())).expect("somewhere to listen");
        let shared = Arc::new(shared);
        std::thread::spawn({
            let shared = Arc::clone(&shared);
            move || {
                let _ = serve(listener, shared).is_ok();
            }
        });
        (address, shared)
    }

    /// The same, with a worker behind it that has read the stand-in voice: `tones` holds no
    /// weights, so a job of it is a real job that takes no time.
    fn a_speaking_server(called: &str) -> (SocketAddr, Arc<Shared>) {
        let shared = a_program(called, Task::Text2Speech, None);
        shared.reading(true);
        let (address, shared) = a_server(shared);
        std::thread::spawn({
            let shared = Arc::clone(&shared);
            move || worker::work(&shared, Load::Voice(worker::TONES.to_string()))
        });
        while shared.is_busy() {
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        (address, shared)
    }

    /// What came back from one request.
    struct Answer {
        status: u16,
        head: String,
        body: Vec<u8>,
    }

    impl Answer {
        fn json(&self) -> Value {
            serde_json::from_slice(&self.body).expect("an answer in JSON")
        }

        fn text(&self) -> String {
            String::from_utf8_lossy(&self.body).to_string()
        }

        fn header(&self, name: &str) -> Option<String> {
            self.head.lines().find_map(|line| {
                let (field, value) = line.split_once(':')?;
                field
                    .eq_ignore_ascii_case(name)
                    .then(|| value.trim().to_string())
            })
        }
    }

    /// One request, as the bytes of one, with whatever extra header lines `extra` carries.
    ///
    /// Written out rather than asked of a client library, because what is being tested is the
    /// wire -- a client that rewrote the request on the way out would be testing itself.
    fn ask(address: SocketAddr, line: &str, extra: &str, body: &[u8]) -> Answer {
        let mut socket = TcpStream::connect(address).expect("the server");
        let head = format!(
            "{line} HTTP/1.1\r\nHost: {address}\r\n{extra}Content-Length: {}\r\nConnection: close\r\n\r\n",
            body.len()
        );
        socket.write_all(head.as_bytes()).expect("to ask");
        socket.write_all(body).expect("to ask");

        let mut whole = Vec::new();
        socket.read_to_end(&mut whole).expect("an answer");
        let at = whole
            .windows(4)
            .position(|four| four == b"\r\n\r\n")
            .expect("a head");
        let head = String::from_utf8_lossy(&whole[..at]).to_string();
        let status = head
            .split_whitespace()
            .nth(1)
            .and_then(|code| code.parse().ok())
            .unwrap_or(0);

        Answer {
            status,
            head,
            body: whole[at + 4..].to_vec(),
        }
    }

    fn get(address: SocketAddr, path: &str) -> Answer {
        ask(address, &format!("GET {path}"), "", b"")
    }

    fn delete(address: SocketAddr, path: &str) -> Answer {
        ask(address, &format!("DELETE {path}"), "", b"")
    }

    /// A POST of JSON, the way the page sends one.
    fn post(address: SocketAddr, path: &str, body: &str) -> Answer {
        ask(
            address,
            &format!("POST {path}"),
            "Content-Type: application/json\r\n",
            body.as_bytes(),
        )
    }

    /// A POST of a file, the way the page uploads one.
    fn post_file(address: SocketAddr, mime: &str, bytes: &[u8]) -> Answer {
        ask(
            address,
            "POST /api/uploads",
            &format!("Content-Type: {mime}\r\n"),
            bytes,
        )
    }

    /// A job, asked after until it has finished.
    fn once_it_is_done(address: SocketAddr, id: &str) -> Value {
        for _ in 0..500 {
            let job = get(address, &format!("/api/jobs/{id}")).json();
            if !matches!(job["status"].as_str(), Some("queued" | "running")) {
                return job;
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        panic!("job {id} never finished");
    }

    /// A WAV of a tenth of a second of silence.
    fn a_recording() -> Vec<u8> {
        crate::wav::write(&crate::wav::Sound::new(vec![0.0; 2400], 24_000))
    }

    const PNG: &[u8] = &[0x89, b'P', b'N', b'G', 13, 10, 26, 10];

    #[test]
    fn the_page_and_everything_it_asks_for_is_served() {
        let (address, _) = a_server(a_program("page", Task::Txt2Img, None));

        for path in [
            "/",
            "/style.css",
            "/app.js",
            // The page is drawn in React, and a missing one of these is a blank window rather
            // than a broken corner: every script the frame names has to be here.
            "/vendor/react.js",
            "/vendor/react-dom.js",
            "/vendor/htm.js",
        ] {
            let answer = get(address, path);
            assert_eq!(answer.status, 200, "{path}");
            assert!(!answer.body.is_empty(), "{path} came back empty");
        }

        // A browser appends a query to defeat its own cache. A path that kept it would match
        // nothing, which is the kind of failure that only shows up on the second load.
        assert_eq!(get(address, "/api/model?4").status, 200);
    }

    #[test]
    fn the_model_the_terminal_chose_is_described_before_a_byte_of_it_is_read() {
        // What the boxes on the page are filled in from: what the name says about the model.
        let (address, _) = a_server(a_program("sdxl", Task::Txt2Img, Some("sdxl:base")));
        let described = get(address, "/api/model").json();
        assert_eq!(described["kind"], "image");
        assert_eq!(described["task"], "txt2img");
        assert_eq!(described["device"], "cpu");
        let chosen = &described["model"];
        assert_eq!(chosen["name"], "sdxl:base");
        assert_eq!(chosen["full_name"], "Stable Diffusion XL Base 1.0");
        assert_eq!(chosen["in_memory"], false);
        assert_eq!(chosen["sampler"], "Euler");
        // An SDXL package is taken to draw from a picture until one says otherwise, and has a
        // second pass, so the page draws the CFG card and the negative prompt box.
        assert_eq!(chosen["draws_from_a_picture"], true);
        assert_eq!(chosen["takes_guidance"], true);

        // An Anima one is known not to start from a picture before anything of it is read.
        let (address, _) = a_server(a_program("anima", Task::Txt2Img, Some("anima:turbo")));
        let chosen = &get(address, "/api/model").json()["model"];
        assert_eq!(chosen["draws_from_a_picture"], false);
        assert!(
            chosen["no_picture_because"]
                .as_str()
                .expect("a reason")
                .contains("Anima"),
            "{chosen}"
        );

        // Krea 2 answers in a single pass, and the page stops drawing the two things that steer
        // a second one.
        let (address, _) = a_server(a_program("krea2", Task::Txt2Img, Some("krea2:turbo")));
        let chosen = &get(address, "/api/model").json()["model"];
        assert_eq!(chosen["takes_guidance"], false);
        assert_eq!(chosen["guidance"], 1.0);
        assert_eq!(chosen["steps"], 8);
    }

    #[test]
    fn the_model_says_what_will_speak_and_that_it_is_not_a_voice() {
        // The one sentence the speech tab is built around. A page that did not get it would be a
        // page claiming a voice this program has not got.
        let (address, _) = a_server(a_program("tones", Task::Text2Speech, None));
        let described = get(address, "/api/model").json();
        assert_eq!(described["kind"], "speech");
        let voice = &described["voice"];

        assert_eq!(voice["name"], "tones");
        assert_eq!(voice["rate"], 24_000);
        assert_eq!(voice["takes_a_recording"], true);
        assert!(
            voice["not_a_voice_because"]
                .as_str()
                .expect("the sentence")
                .contains("not a speech model"),
            "{voice}"
        );
    }

    #[test]
    fn a_published_voice_is_described_the_way_a_model_is() {
        // Described and not read or fetched. It is a real voice, so the tab has no apology to
        // make, and it says what it is called and whether it is in memory yet.
        let shared = a_program("indextts", Task::Text2Speech, Some("indextts"));
        let described = shared.describe_model();

        assert_eq!(described["task"], "text2speech");
        let voice = &described["voice"];
        assert_eq!(voice["name"], "indextts");
        assert_eq!(voice["full_name"], "IndexTTS 2.5");
        assert_eq!(voice["in_memory"], false);
        assert!(voice["on_disk"].is_boolean(), "{voice}");
        assert_eq!(voice["not_a_voice_because"], Value::Null);
    }

    #[test]
    fn the_page_cannot_choose_fetch_or_move_a_model() {
        // All three are the terminal's. A page that could would be a second place to change what
        // the first had settled, and the doors are not there to be knocked on.
        let (address, shared) = a_server(a_program("no-doors", Task::Txt2Img, Some("sdxl:base")));

        for path in ["/api/model", "/api/fetch", "/api/device"] {
            assert_eq!(
                post(address, path, r#"{"model":"sdxl:noob"}"#).status,
                404,
                "{path}"
            );
        }
        assert_eq!(delete(address, "/api/model").status, 404);
        assert!(!shared.is_busy(), "something was started anyway");
    }

    #[test]
    fn a_job_is_taken_put_in_line_and_can_be_found_again() {
        let (address, _) = a_server(a_program("submit", Task::Txt2Img, Some("sdxl:base")));

        let first = post(
            address,
            "/api/jobs",
            r#"{"prompt":"a cat","width":1000,"seed":"7"}"#,
        );
        assert_eq!(first.status, 202, "{}", first.text());
        let job = first.json();
        let id = job["id"].as_str().unwrap().to_string();
        assert_eq!(first.header("Location"), Some(format!("/api/jobs/{id}")));
        assert_eq!(job["status"], "queued");
        assert_eq!(job["kind"], "image");
        assert_eq!(job["position"], 0);

        // Every setting spelled out: what was asked, rounded to what the model can be asked for,
        // and the model's own for the rest.
        let asked = &job["asked"];
        assert_eq!(asked["prompt"], "a cat");
        assert_eq!(asked["width"], 960);
        assert_eq!(asked["height"], 1024);
        assert_eq!(asked["seed"], "7");
        assert_eq!(asked["model"], "sdxl:base");

        // A second one waits behind it, and is given a seed of its own where it asked for none. A
        // moment later: two posted in the same millisecond go in the order of their ids.
        std::thread::sleep(std::time::Duration::from_millis(2));
        let second = post(address, "/api/jobs", r#"{"prompt":"a dog"}"#).json();
        assert_eq!(second["position"], 1);
        let seed = second["asked"]["seed"].as_str().unwrap();
        assert!(seed.parse::<u64>().is_ok(), "{seed}");
        let second_id = second["id"].as_str().unwrap();

        // Found again by id, by a list of ids, and in the list of all of them.
        assert_eq!(
            get(address, &format!("/api/jobs/{id}")).json()["asked"]["prompt"],
            "a cat"
        );
        let some = get(address, &format!("/api/jobs?ids={second_id}")).json();
        assert_eq!(some["jobs"].as_array().unwrap().len(), 1);
        assert_eq!(some["jobs"][0]["id"], second_id);
        assert_eq!(
            get(address, "/api/jobs").json()["jobs"]
                .as_array()
                .unwrap()
                .len(),
            2
        );

        let worker = get(address, "/api/worker").json();
        assert_eq!(worker["queued"], 2);
        assert_eq!(worker["busy"], true);

        // Nothing made yet.
        assert_eq!(get(address, &format!("/api/jobs/{id}/output")).status, 404);
        assert_eq!(
            get(address, &format!("/api/jobs/{}", "0".repeat(32))).status,
            404
        );
        assert_eq!(get(address, "/api/jobs/../../etc/passwd").status, 404);
    }

    #[test]
    fn a_job_that_cannot_be_run_is_refused_with_the_reason() {
        let (address, _) = a_server(a_program("refused", Task::Txt2Img, Some("anima:turbo")));

        for (body, reason) in [
            (r#"{"prompt":"   "}"#, "give it a prompt"),
            ("not json at all", "not a request"),
            (r#"{"kind":"speech","text":"hello"}"#, "runs image jobs"),
            (
                r#"{"prompt":"a cat","init_image":"00000000000000000000000000000000"}"#,
                "no upload",
            ),
        ] {
            let answer = post(address, "/api/jobs", body);
            assert_eq!(answer.status, 400, "{body}");
            assert!(answer.text().contains(reason), "{body}: {}", answer.text());
        }

        // A picture to start from, for a model that cannot: refused with the model's own reason.
        let upload = post_file(address, "image/png", PNG).json();
        let answer = post(
            address,
            "/api/jobs",
            &json!({"prompt": "a cat", "init_image": upload["id"]}).to_string(),
        );
        assert_eq!(answer.status, 400);
        assert!(answer.text().contains("Anima"), "{}", answer.text());

        // Nothing refused went in line.
        assert_eq!(get(address, "/api/worker").json()["queued"], 0);
    }

    #[test]
    fn a_waiting_job_is_cancelled_and_then_deleted() {
        let (address, _) = a_server(a_program("cancel", Task::Txt2Img, Some("sdxl:base")));
        let id = post(address, "/api/jobs", r#"{"prompt":"a cat"}"#).json()["id"]
            .as_str()
            .unwrap()
            .to_string();

        let cancelled = post(address, &format!("/api/jobs/{id}/cancel"), "{}");
        assert_eq!(cancelled.status, 200);
        assert_eq!(cancelled.json()["status"], "cancelled");
        assert_eq!(get(address, "/api/worker").json()["queued"], 0);

        // Nothing to stop the second time.
        assert_eq!(
            post(address, &format!("/api/jobs/{id}/cancel"), "{}").status,
            409
        );

        assert_eq!(delete(address, &format!("/api/jobs/{id}")).status, 204);
        assert_eq!(get(address, &format!("/api/jobs/{id}")).status, 404);
        assert_eq!(delete(address, &format!("/api/jobs/{id}")).status, 404);
    }

    #[test]
    fn deleting_a_waiting_job_takes_it_out_of_line() {
        let (address, _) = a_server(a_program("delete-queued", Task::Txt2Img, Some("sdxl:base")));
        let id = post(address, "/api/jobs", r#"{"prompt":"a cat"}"#).json()["id"]
            .as_str()
            .unwrap()
            .to_string();

        assert_eq!(delete(address, &format!("/api/jobs/{id}")).status, 204);
        assert_eq!(get(address, "/api/worker").json()["queued"], 0);
    }

    #[test]
    fn a_file_goes_up_and_comes_back_as_it_went() {
        let (address, _) = a_server(a_program("uploads", Task::Txt2Img, None));

        let up = post_file(address, "image/png", PNG);
        assert_eq!(up.status, 201, "{}", up.text());
        let upload = up.json();
        let id = upload["id"].as_str().unwrap();
        assert_eq!(up.header("Location"), Some(format!("/api/uploads/{id}")));
        assert_eq!(upload["type"], "image/png");

        let back = get(address, &format!("/api/uploads/{id}"));
        assert_eq!(back.status, 200);
        assert_eq!(back.body, PNG);
        assert_eq!(back.header("Content-Type").as_deref(), Some("image/png"));

        // A recording, which is read at the door.
        let recording = post_file(address, "audio/wav", &a_recording());
        assert_eq!(recording.status, 201, "{}", recording.text());
        assert_eq!(recording.json()["type"], "audio/wav");

        assert_eq!(delete(address, &format!("/api/uploads/{id}")).status, 204);
        assert_eq!(get(address, &format!("/api/uploads/{id}")).status, 404);
        assert_eq!(delete(address, &format!("/api/uploads/{id}")).status, 404);
    }

    #[test]
    fn a_file_a_job_cannot_start_from_is_refused_at_the_door() {
        let (address, _) = a_server(a_program("bad-uploads", Task::Txt2Img, None));

        assert_eq!(post_file(address, "image/png", b"").status, 400);
        assert_eq!(
            post_file(address, "application/x-sh", b"#!/bin/sh").status,
            415
        );

        // Says it is a WAV and is not one: found out now, rather than after waiting in line.
        let answer = post_file(address, "audio/wav", b"RIFF\0\0\0\0WAVEjunk");
        assert_eq!(answer.status, 400, "{}", answer.text());
    }

    #[test]
    fn only_this_machine_s_own_pages_are_answered() {
        let (address, _) = a_server(a_program("from-here", Task::Txt2Img, Some("sdxl:base")));
        let job = r#"{"prompt":"a cat"}"#.as_bytes();

        // What a form on some other site can send without the browser asking first: no type, or
        // a plain one. Refused, and not taken.
        for extra in [
            "",
            "Content-Type: text/plain\r\n",
            "Content-Type: application/x-www-form-urlencoded\r\n",
            "Content-Type: multipart/form-data; boundary=x\r\n",
        ] {
            let answer = ask(address, "POST /api/jobs", extra, job);
            assert_eq!(answer.status, 415, "{extra:?}");
        }

        // Another site's page, and a name some other site has pointed at this machine.
        let answer = ask(
            address,
            "POST /api/jobs",
            "Content-Type: application/json\r\nOrigin: http://evil.example\r\n",
            job,
        );
        assert_eq!(answer.status, 403);
        let mut socket = TcpStream::connect(address).unwrap();
        socket
            .write_all(b"GET /api/jobs HTTP/1.1\r\nHost: evil.example\r\nConnection: close\r\n\r\n")
            .unwrap();
        let mut whole = String::new();
        socket.read_to_string(&mut whole).unwrap();
        assert!(whole.starts_with("HTTP/1.1 403"), "{whole}");

        assert_eq!(get(address, "/api/worker").json()["queued"], 0);

        // And this server's own page is answered, Origin and all.
        let answer = ask(
            address,
            "POST /api/jobs",
            &format!("Content-Type: application/json\r\nOrigin: http://{address}\r\n"),
            job,
        );
        assert_eq!(answer.status, 202, "{}", answer.text());
    }

    #[test]
    fn a_conversion_is_taken_with_both_recordings_and_refused_without_either() {
        let (address, _) = a_server(a_program("convert", Task::Speech2Speech, Some("cosyvoice")));

        let described = get(address, "/api/model").json();
        assert_eq!(described["task"], "speech2speech");
        assert_eq!(described["kind"], "conversion");
        assert_eq!(described["converter"]["full_name"], "Fun-CosyVoice3 0.5B");
        assert_eq!(
            described["converter"]["steps"],
            crate::cosyvoice3::CosyVoice3::CONVERSION_DEFAULTS.steps
        );

        let source = post_file(address, "audio/wav", &a_recording()).json()["id"].clone();
        let reference = post_file(address, "audio/wav", &a_recording()).json()["id"].clone();
        let picture = post_file(address, "image/png", PNG).json()["id"].clone();

        for (body, reason) in [
            (json!({"reference": reference}), "nothing to convert"),
            (json!({"source": source}), "nothing to convert"),
            (
                json!({"source": picture, "reference": reference}),
                "not a WAV",
            ),
            (
                json!({"kind": "speech", "text": "hello"}),
                "runs conversion jobs",
            ),
        ] {
            let answer = post(address, "/api/jobs", &body.to_string());
            assert_eq!(answer.status, 400, "{body}");
            assert!(answer.text().contains(reason), "{body}: {}", answer.text());
        }

        let posted = post(
            address,
            "/api/jobs",
            &json!({"kind": "conversion", "source": source, "reference": reference, "steps": 500})
                .to_string(),
        );
        assert_eq!(posted.status, 202, "{}", posted.text());
        let asked = &posted.json()["asked"];
        assert_eq!(asked["source"], source);
        assert_eq!(asked["reference"], reference);
        // Held to what a run can be asked for, and the rest filled in.
        assert_eq!(asked["steps"], 100);
        assert_eq!(asked["style"], false);
        assert!(asked["seed"].as_str().unwrap().parse::<u64>().is_ok());
        assert_eq!(asked["converter"], "cosyvoice");
    }

    #[test]
    fn a_page_that_is_not_there_says_so_in_the_shape_everything_else_answers_in() {
        let (address, _) = a_server(a_program("missing", Task::Txt2Img, None));
        let answer = get(address, "/nowhere");
        assert_eq!(answer.status, 404);
        assert!(answer.json()["error"].is_string());
    }

    #[test]
    fn a_job_runs_to_the_end_and_what_it_made_is_kept_until_it_is_deleted() {
        let (address, shared) = a_speaking_server("runs");

        let reference = post_file(address, "audio/wav", &a_recording()).json();
        let posted = post(
            address,
            "/api/jobs",
            &json!({"text": "hello there", "seed": "7", "reference": reference["id"]}).to_string(),
        );
        assert_eq!(posted.status, 202, "{}", posted.text());
        let id = posted.json()["id"].as_str().unwrap().to_string();

        let job = once_it_is_done(address, &id);
        assert_eq!(job["status"], "done", "{job}");
        let output = &job["output"];
        assert_eq!(output["type"], "audio/wav");
        assert_eq!(output["made"]["seed"], 7);
        assert_eq!(output["made"]["from_a_recording"], true);

        let made = get(address, output["url"].as_str().unwrap());
        assert_eq!(made.status, 200);
        assert!(made.body.starts_with(b"RIFF"));
        assert_eq!(made.body.len() as u64, output["bytes"].as_u64().unwrap());

        // On the disk, where a restart would find it, until it is deleted.
        let directory = std::env::temp_dir()
            .join(format!("libwaifu-webui-runs-{}", std::process::id()))
            .join("jobs")
            .join(&id);
        assert!(directory.join("output.wav").is_file());
        assert_eq!(delete(address, &format!("/api/jobs/{id}")).status, 204);
        assert!(!directory.exists());
        assert!(!shared.is_busy());
    }
}
