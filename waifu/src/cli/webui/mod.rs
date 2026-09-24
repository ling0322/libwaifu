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
//! It is a browser rather than the terminal it used to be for one reason, which is that the thing
//! being made is a picture. A terminal can say that one was written and where; it cannot show it.

mod http;
mod machine;
mod state;
mod worker;

use std::io;
use std::net::{Ipv4Addr, SocketAddr, TcpListener};
use std::sync::mpsc::channel;
use std::sync::Arc;

use crate::cli::args::Args;
use crate::cli::webui::state::Shared;
use crate::cli::webui::worker::Command;

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
    let voice = args.voice().unwrap_or(worker::VOICE).to_string();
    let runtime = with_usage(args.device())?.resolve();
    let wanted_port = with_usage(args.port())?;

    let shared = Arc::new(Shared::new(runtime));
    let (commands, waiting) = channel::<Command>();

    // The voice the speech tab reads with, described before any page has opened. The boxes on
    // that tab are a voice's own numbers and there is nothing else to fill them from. Described
    // rather than read: a voice with a package behind it is read at the first reading that wants
    // it, exactly as a picture model is at the first picture.
    shared.change(|session| session.voice = Some(worker::look_at_voice(&voice)));

    // A picture named on the command line only fills the box. Everything about a run is
    // changeable between runs, and this is no different: it is where to start, not what to be
    // stuck with. Read now rather than when a run begins, so that a path that is not there is
    // said here, where there is still a terminal to say it on.
    if let Some(picture) = args.image() {
        let bytes =
            std::fs::read(picture).map_err(|error| format!("could not read {picture}: {error}"))?;
        shared.hold_upload(bytes);
    }

    // Bound before the worker starts and before anything is fetched. A port that is taken is a
    // run that cannot go anywhere, and finding that out after ten minutes of downloading a model
    // is finding it out at the worst possible moment.
    let (listener, address) = listen(wanted_port)?;

    let worker = std::thread::spawn({
        let shared = Arc::clone(&shared);
        move || worker::work(&shared, &waiting)
    });

    // Asked for here rather than left to the page, so that `waifu webui -m sdxl:base` starts
    // fetching and reading while the browser is still being opened -- which is what someone who
    // named a model on the command line asked for.
    if let Some(model) = model {
        // Chosen straight away, so that the page opens with the boxes already filled in for it,
        // and read straight away as well -- which is what naming one on the command line asks
        // for, rather than waiting for the first run the way a model picked on the page does.
        shared.change(|session| session.model = Some(worker::look_at(&model)));
        shared.claim();
        let _ = commands.send(Command::Use(model));
    } else {
        shared.say("pick a model to begin", false);
    }

    // Not "drawing at" any more: the page has a tab that says things rather than draws them, and
    // a line that names one of the two is a line somebody reads as the whole of what is there.
    println!("waifu is at http://{address}");
    println!("Press ctrl-c to stop.");

    serve(listener, shared, commands)?;
    let _ = worker.join();

    Ok(())
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
fn serve(
    listener: TcpListener,
    shared: Arc<Shared>,
    commands: std::sync::mpsc::Sender<Command>,
) -> Result<(), Error> {
    // Mapped rather than passed on: what comes back is a `Send + Sync` box, which is not the
    // plain box everything here reports with and does not convert into one on its own.
    let server = tiny_http::Server::from_listener(listener, None)
        .map_err(|error| format!("could not answer on that port: {error}"))?;
    let server = Arc::new(server);
    let commands = Arc::new(std::sync::Mutex::new(commands));

    let answerers: Vec<_> = (0..ANSWERERS)
        .map(|_| {
            let server = Arc::clone(&server);
            let shared = Arc::clone(&shared);
            let commands = Arc::clone(&commands);

            std::thread::spawn(move || {
                while let Ok(mut request) = server.recv() {
                    // One at a time, which is what the channel wants and costs nothing: posting a
                    // command is a pointer move, and the work it asks for happens elsewhere.
                    let reply = {
                        let commands = commands.lock().unwrap_or_else(|held| held.into_inner());
                        http::answer(&shared, &commands, &mut request)
                    };

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
    use std::sync::mpsc::Receiver;

    use serde_json::Value;

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

    /// A server with no model behind it, answering on a port of its own.
    ///
    /// The receiving end of the channel comes back with it: dropped, every command posted to it
    /// would be refused as "the model thread is no longer there", which is not what these are
    /// about.
    fn a_server() -> (SocketAddr, Receiver<Command>) {
        let (listener, address) = listen(Some(a_free_port())).expect("somewhere to listen");
        let shared = Arc::new(Shared::new(DeviceOption::Cpu.resolve()));
        let (commands, waiting) = channel::<Command>();

        // The same thing `main` does before it opens a browser: there is one voice and it is
        // described before any page reads the state. A server without it is a server no run of
        // this program produces.
        shared.change(|session| session.voice = Some(worker::look_at_voice(worker::VOICE)));

        // Left running for the rest of the test process. There is no way to stop it short of the
        // process ending, which is the same shape the program has. What it answers with is
        // dropped here rather than carried out: an error in it is a box that is not `Send`, and
        // nothing on this side would do anything with one anyway.
        std::thread::spawn(move || {
            let _ = serve(listener, shared, commands).is_ok();
        });

        (address, waiting)
    }

    /// One request, as the bytes of one, and what came back: the status and the body.
    ///
    /// Written out rather than asked of a client library, because what is being tested is the
    /// wire -- a client that rewrote the request on the way out would be testing itself.
    fn asked(address: SocketAddr, line: &str, body: &str) -> (u16, String) {
        posted(address, line, body.as_bytes())
    }

    /// The same, for a body that is not text.
    ///
    /// A WAV file is bytes, and most of them are not characters: sent as a string it arrives as
    /// whatever the replacement character made of every one that was not valid UTF-8, which is a
    /// file that no longer parses and a test that was checking the wrong refusal.
    fn posted(address: SocketAddr, line: &str, body: &[u8]) -> (u16, String) {
        let mut socket = TcpStream::connect(address).expect("the server");
        let head = format!(
            "{line} HTTP/1.1\r\nHost: localhost\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
            body.len()
        );
        socket.write_all(head.as_bytes()).expect("to ask");
        socket.write_all(body).expect("to ask");

        // Read as bytes, since what comes back can be a file too. Only the status line and a
        // refusal's JSON are ever looked at, and both are text.
        let mut whole = Vec::new();
        socket.read_to_end(&mut whole).expect("an answer");
        let answer = String::from_utf8_lossy(&whole).to_string();

        let status = answer
            .split_whitespace()
            .nth(1)
            .and_then(|code| code.parse().ok())
            .unwrap_or(0);
        let body = answer
            .split_once("\r\n\r\n")
            .map(|(_, body)| body)
            .unwrap_or("");

        (status, body.to_string())
    }

    fn json(address: SocketAddr, line: &str, body: &str) -> Value {
        serde_json::from_str(&asked(address, line, body).1).expect("an answer in JSON")
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
    fn the_page_and_everything_it_asks_for_is_served() {
        let (address, _commands) = a_server();

        for (line, mime) in [
            ("GET /", "text/html"),
            ("GET /style.css", "text/css"),
            ("GET /app.js", "text/javascript"),
            // The page is drawn in React, and a missing one of these is a blank window rather
            // than a broken corner: every script the frame names has to be here.
            ("GET /vendor/react.js", "text/javascript"),
            ("GET /vendor/react-dom.js", "text/javascript"),
            ("GET /vendor/htm.js", "text/javascript"),
        ] {
            let (status, body) = asked(address, line, "");
            assert_eq!(status, 200, "{line}");
            assert!(!body.is_empty(), "{line} came back empty");
            assert!(mime.starts_with("text/"), "{mime}");
        }

        // A browser appends a query to defeat its own cache. A path that kept it would match
        // nothing, which is the kind of failure that only shows up on the second load.
        assert_eq!(asked(address, "GET /api/state?4", "").0, 200);
    }

    #[test]
    fn the_state_says_there_is_no_model_rather_than_failing_to_say_anything() {
        // The page opens before anything is loaded, every time. Half of what it draws has to hold
        // for a program that has not been asked to do anything yet.
        let (address, _commands) = a_server();
        let state = json(address, "GET /api/state", "");

        assert!(state["model"].is_null());
        assert!(!state["models"]
            .as_array()
            .expect("the catalogue")
            .is_empty());
        assert_eq!(state["device"], "cpu");
        assert_eq!(state["holding_a_picture"], false);
    }

    #[test]
    fn the_list_says_which_models_draw_explicit_pictures() {
        // The page leaves those out until somebody asks for them, and it has nothing to go on but
        // this: a catalogue that answered without the label would be a page showing everything.
        let (address, _commands) = a_server();
        let state = json(address, "GET /api/state", "");
        let models = state["models"].as_array().expect("the catalogue");

        let said = |name: &str| {
            models
                .iter()
                .find(|model| model["name"] == name)
                .unwrap_or_else(|| panic!("{name} is offered"))["explicit"]
                .as_bool()
                .expect("a yes or a no about every model")
        };

        assert!(said("sdxl:noob"));
        assert!(!said("sdxl:base"));
    }

    #[test]
    fn a_run_asked_for_with_no_model_is_refused_with_the_reason() {
        let (address, _commands) = a_server();

        let (status, body) = asked(address, "POST /api/generate", r#"{"prompt":"a cat"}"#);
        assert_eq!(status, 409);
        assert!(body.contains("no model is chosen"), "{body}");
    }

    #[test]
    fn a_run_with_nothing_to_draw_is_refused_before_anything_else_is_looked_at() {
        // Before the model is, because "type a prompt" is the more useful of the two things that
        // are wrong with an empty page, and it is the one the person can act on.
        let (address, _commands) = a_server();

        let (status, body) = asked(address, "POST /api/generate", r#"{"prompt":"   "}"#);
        assert_eq!(status, 400);
        assert!(body.contains("type a prompt"), "{body}");
    }

    #[test]
    fn what_is_not_a_request_is_said_to_be_one_rather_than_dropped() {
        let (address, _commands) = a_server();

        let (status, body) = asked(address, "POST /api/generate", "not json at all");
        assert_eq!(status, 400);
        assert!(body.contains("not a request"), "{body}");
    }

    #[test]
    fn choosing_a_model_reads_nothing_and_says_what_it_would_read() {
        // Choosing is a click; reading is minutes and several gigabytes. So nothing is posted to
        // the worker here -- what comes back is what the name says about the model, which is what
        // the boxes on the page are filled in from until a run reads the package itself.
        let (address, commands) = a_server();

        assert_eq!(
            asked(address, "POST /api/model", r#"{"model":"sdxl:base"}"#).0,
            200
        );
        assert!(commands.try_recv().is_err(), "the worker was told anyway");

        let chosen = &json(address, "GET /api/state", "")["model"];
        assert_eq!(chosen["name"], "sdxl:base");
        assert_eq!(chosen["full_name"], "Stable Diffusion XL Base 1.0");
        assert_eq!(chosen["in_memory"], false);
        assert_eq!(chosen["sampler"], "Euler");
        // An SDXL package is taken to draw from a picture until one says otherwise, and an Anima
        // one is known not to before anything of it has been fetched.
        assert_eq!(chosen["draws_from_a_picture"], true);
        // And it has a second pass, so the page draws the CFG card and the negative prompt box.
        assert_eq!(chosen["takes_guidance"], true);

        assert_eq!(
            asked(address, "POST /api/model", r#"{"model":"anima:turbo"}"#).0,
            200
        );
        let chosen = &json(address, "GET /api/state", "")["model"];
        assert_eq!(chosen["name"], "anima:turbo");
        assert_eq!(chosen["draws_from_a_picture"], false);
        assert!(
            chosen["no_picture_because"]
                .as_str()
                .expect("a reason")
                .contains("Anima"),
            "{chosen}"
        );

        // Krea 2 is the one that answers in a single pass. The page reads this and stops drawing
        // the two things that steer a second one, before a byte of the package has been fetched:
        // a control that appeared once a download finished and then vanished would be a screen
        // whose settings move while somebody is using them.
        assert_eq!(
            asked(address, "POST /api/model", r#"{"model":"krea2:turbo"}"#).0,
            200
        );
        let chosen = &json(address, "GET /api/state", "")["model"];
        assert_eq!(chosen["name"], "krea2:turbo");
        assert_eq!(chosen["takes_guidance"], false);
        // What it would be guided at if it were, which is this runtime's spelling of none. The
        // number is still described, because it is what a run of this model is posted with.
        assert_eq!(chosen["guidance"], 1.0);
        assert_eq!(chosen["steps"], 8);

        // A null un-chooses, which is what the page sends when the kind of run changes.
        assert_eq!(
            asked(address, "POST /api/model", r#"{"model":null}"#).0,
            200
        );
        assert!(json(address, "GET /api/state", "")["model"].is_null());

        // And a request that names nothing is refused rather than remembered as an empty name,
        // which the first run would then fail to fetch a model called "". A field that is missing
        // is a request that forgot to say anything; the null above is a decision.
        let (status, body) = asked(address, "POST /api/model", r#"{"model":"  "}"#);
        assert_eq!(status, 400);
        assert!(body.contains("no model was named"), "{body}");
        let (status, body) = asked(address, "POST /api/model", r#"{}"#);
        assert_eq!(status, 400);
        assert!(body.contains("no model was named"), "{body}");
    }

    #[test]
    fn a_device_this_machine_does_not_have_is_refused_by_name() {
        // Every build knows all four names and no machine has all four devices. What decides is
        // what is here, and the refusal says what that is rather than only that this was wrong.
        let (address, _commands) = a_server();

        let (status, body) = asked(address, "POST /api/device", r#"{"device":"nonsense"}"#);
        assert_eq!(status, 400);
        assert!(body.contains("no device here is called"), "{body}");
        assert!(body.contains("cpu"), "{body}");

        let (status, body) = asked(address, "POST /api/device", r#"{"model":"cpu"}"#);
        assert_eq!(status, 400);
        assert!(body.contains("no device was named"), "{body}");
    }

    #[test]
    fn the_device_already_in_use_is_not_asked_for_again() {
        // The page sends what the box says, and the box says the device it is already on until
        // somebody changes it. Taking the worker for that would unload the model to load it
        // again onto the device it is already on.
        let (address, commands) = a_server();

        let device = json(address, "GET /api/state", "")["device"]
            .as_str()
            .expect("a device")
            .to_string();
        let sent = format!(r#"{{"device":"{device}"}}"#);
        assert_eq!(asked(address, "POST /api/device", &sent).0, 200);
        assert!(commands.try_recv().is_err(), "the worker was told anyway");
    }

    #[test]
    fn a_second_thing_asked_for_before_the_first_has_started_is_refused() {
        // Two clicks on a button are a few tens of milliseconds apart, and the worker picks a
        // command up some time after it is posted. Asked of what the worker is doing, both
        // requests read "nothing", and one press drew two pictures.
        let (address, commands) = a_server();

        assert_eq!(
            asked(address, "POST /api/model", r#"{"model":"sdxl:base"}"#).0,
            200
        );
        let run = r#"{"prompt":"a cat"}"#;
        assert_eq!(asked(address, "POST /api/generate", run).0, 200);
        let (status, body) = asked(address, "POST /api/generate", run);
        assert_eq!(status, 409);
        assert!(body.contains("already happening"), "{body}");

        // One command, not two: the second never reached the channel.
        assert!(commands.try_recv().is_ok());
        assert!(commands.try_recv().is_err());
    }

    #[test]
    fn a_command_that_could_not_be_posted_gives_the_worker_back() {
        // A refusal after the claim has found a request that could not be carried out, not a
        // program that is busy. Keeping the claim would leave every request after it refused as
        // "already happening" by something that never started.
        let (address, commands) = a_server();
        assert_eq!(
            asked(address, "POST /api/model", r#"{"model":"sdxl:base"}"#).0,
            200
        );
        drop(commands);

        for _ in 0..2 {
            let (status, body) = asked(address, "POST /api/generate", r#"{"prompt":"a cat"}"#);
            assert_eq!(status, 500);
            assert!(body.contains("no longer there"), "{body}");
        }
    }

    #[test]
    fn a_picture_this_session_did_not_draw_cannot_be_deleted_through_it_either() {
        // The same door as reading one, and the same answer through it: a name that is not in the
        // gallery is not a file this program will touch, whatever it is a path to.
        let (address, _commands) = a_server();

        let (status, body) = asked(
            address,
            "DELETE /api/picture",
            r#"{"file":"../../etc/passwd"}"#,
        );
        assert_eq!(status, 404);
        assert!(body.contains("did not draw that"), "{body}");

        let (status, body) = asked(address, "DELETE /api/picture", r#"{}"#);
        assert_eq!(status, 400);
        assert!(body.contains("no picture was named"), "{body}");
    }

    #[test]
    fn a_picture_this_session_did_not_draw_cannot_be_read_through_it() {
        // The gallery is the whole of what is readable. This server is on the loopback address,
        // which every other program on the machine can reach, a browser tab on another site
        // included.
        let (address, _commands) = a_server();

        for path in [
            "/picture/../../../etc/passwd",
            "/picture//etc/passwd",
            "/picture/waifu-0001.png",
        ] {
            let (status, body) = asked(address, &format!("GET {path}"), "");
            assert_eq!(status, 404, "{path}");
            assert!(body.contains("did not draw"), "{path}: {body}");
        }
    }

    #[test]
    fn the_picture_to_draw_from_goes_up_and_comes_back_as_it_went() {
        // The page posts the bytes of a file and nothing else -- no form, no boundary, no field
        // names -- and asks for them back to show what a run started from `-i` is drawing from.
        let (address, _commands) = a_server();
        assert_eq!(asked(address, "GET /api/upload", "").0, 404);

        let png = "\u{89}PNG\r\n\u{1a}\nand the rest of it";
        assert_eq!(asked(address, "POST /api/upload", png).0, 200);

        let (status, body) = asked(address, "GET /api/upload", "");
        assert_eq!(status, 200);
        assert_eq!(body, png);
        assert_eq!(
            json(address, "GET /api/state", "")["holding_a_picture"],
            true
        );

        assert_eq!(asked(address, "DELETE /api/upload", "").0, 200);
        assert_eq!(asked(address, "GET /api/upload", "").0, 404);
    }

    #[test]
    fn an_empty_picture_is_not_a_picture() {
        let (address, _commands) = a_server();

        let (status, body) = asked(address, "POST /api/upload", "");
        assert_eq!(status, 400);
        assert!(body.contains("empty"), "{body}");
    }

    #[test]
    fn a_page_that_is_not_there_says_so_in_the_shape_everything_else_answers_in() {
        // So that the page can show what went wrong without knowing which request it came from.
        let (address, _commands) = a_server();

        let (status, body) = asked(address, "GET /nope", "");
        assert_eq!(status, 404);
        assert!(body.contains("\"error\""), "{body}");
    }

    #[test]
    fn the_state_says_what_will_speak_and_that_it_is_not_a_voice() {
        // The one sentence the speech tab is built around. A page that did not get it would be a
        // page claiming a voice this program has not got.
        let (address, _commands) = a_server();
        let state = json(address, "GET /api/state", "");
        let voice = &state["voice"];

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

        assert_eq!(state["holding_a_recording"], false);
        assert_eq!(state["clips"].as_array().expect("the clips").len(), 0);
    }

    #[test]
    fn a_reading_with_nothing_to_read_is_refused_with_the_reason() {
        let (address, _commands) = a_server();

        let (status, body) = asked(address, "POST /api/speak", r#"{"text":"   "}"#);
        assert_eq!(status, 400);
        assert!(body.contains("nothing to say"), "{body}");

        let (status, body) = asked(address, "POST /api/speak", "not json at all");
        assert_eq!(status, 400);
        assert!(body.contains("not a request"), "{body}");
    }

    #[test]
    fn a_reading_is_posted_to_the_worker_once_however_often_it_is_asked_for() {
        // The same two clicks that used to draw two pictures. Nothing about the speech tab makes
        // that different: the worker is picked up some time after a command is posted to it.
        let (address, commands) = a_server();

        let run = r#"{"text":"hello there"}"#;
        assert_eq!(asked(address, "POST /api/speak", run).0, 200);
        let (status, body) = asked(address, "POST /api/speak", run);
        assert_eq!(status, 409);
        assert!(body.contains("already"), "{body}");

        assert!(matches!(commands.try_recv(), Ok(Command::Speak(_))));
        assert!(commands.try_recv().is_err());
    }

    #[test]
    fn a_reading_from_a_recording_there_is_none_of_is_refused_rather_than_run() {
        // A page left open while the recording was cleared asks for exactly this, and what it
        // must not do is take the worker and then quietly read without one.
        let (address, commands) = a_server();

        let (status, body) = asked(
            address,
            "POST /api/speak",
            r#"{"text":"hello","from_recording":true}"#,
        );
        assert_eq!(status, 400);
        assert!(body.contains("no recording"), "{body}");
        assert!(commands.try_recv().is_err(), "the worker was told anyway");

        // And the worker was handed back, rather than left claimed by a run that never started.
        assert_eq!(
            asked(address, "POST /api/speak", r#"{"text":"hello"}"#).0,
            200
        );
    }

    #[test]
    fn the_recording_to_sound_like_goes_up_and_comes_back_as_it_went() {
        let (address, _commands) = a_server();
        assert_eq!(asked(address, "GET /api/voice", "").0, 404);

        // A real WAV file, because this door reads what it is handed rather than holding bytes
        // that a run would find out about later. Posted as bytes: most of a WAV file is not
        // characters, and sent as a string it would arrive as something that no longer parses.
        let wav = crate::wav::write(&crate::Sound::new(vec![0.0; 64], 16_000));
        assert_eq!(posted(address, "POST /api/voice", &wav).0, 200);

        assert_eq!(asked(address, "GET /api/voice", "").0, 200);
        assert_eq!(
            json(address, "GET /api/state", "")["holding_a_recording"],
            true
        );

        // And the picture is a different box: clearing one leaves the other alone.
        assert_eq!(asked(address, "POST /api/upload", "a picture").0, 200);
        assert_eq!(asked(address, "DELETE /api/voice", "").0, 200);
        assert_eq!(asked(address, "GET /api/voice", "").0, 404);
        assert_eq!(asked(address, "GET /api/upload", "").0, 200);
    }

    #[test]
    fn a_file_that_is_not_a_recording_is_refused_at_the_door() {
        // Rather than at the run, which is after a button was pressed and a wait. It is the one
        // thing posted to this server that is checked as it arrives, because it is the one where
        // the check is a header rather than a decoder.
        let (address, _commands) = a_server();

        let (status, body) = asked(address, "POST /api/voice", "ID3 this is an mp3");
        assert_eq!(status, 400);
        assert!(body.contains("not a WAV file"), "{body}");

        let (status, body) = asked(address, "POST /api/voice", "");
        assert_eq!(status, 400);
        assert!(body.contains("empty"), "{body}");
    }

    #[test]
    fn a_clip_this_session_did_not_say_cannot_be_read_or_deleted_through_it() {
        // The same door the pictures have, and it has to be the same door: this server is on the
        // loopback address, which every other program on the machine can reach.
        let (address, _commands) = a_server();

        for path in [
            "/clip/../../../etc/passwd",
            "/clip//etc/passwd",
            "/clip/waifu-0001.wav",
        ] {
            let (status, body) = asked(address, &format!("GET {path}"), "");
            assert_eq!(status, 404, "{path}");
            assert!(body.contains("did not say"), "{path}: {body}");
        }

        let (status, body) = asked(
            address,
            "DELETE /api/clip",
            r#"{"file":"../../etc/passwd"}"#,
        );
        assert_eq!(status, 404);
        assert!(body.contains("did not say"), "{body}");

        let (status, body) = asked(address, "DELETE /api/clip", r#"{}"#);
        assert_eq!(status, 400);
        assert!(body.contains("no clip was named"), "{body}");

        // And a clip is not reachable through the door pictures come out of, which would hand a
        // WAV back under an image's content type.
        assert_eq!(asked(address, "GET /picture/waifu-0001.wav", "").0, 404);
    }

    #[test]
    fn stopping_is_asked_for_whether_or_not_there_is_anything_to_stop() {
        // A run that finished between the button being pressed and the request arriving is not an
        // error: the flag is cleared by the next run before it starts.
        let (address, _commands) = a_server();
        assert_eq!(asked(address, "POST /api/interrupt", "").0, 200);
    }
}
