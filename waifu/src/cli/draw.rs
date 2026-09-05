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

//! The draw command: a screenful of knobs, a diffusion model behind it, and a PNG at the end.
//!
//! A run takes minutes rather than milliseconds, which is what the shape of this is about. The
//! model sits on a thread of its own -- a tensor never leaves the thread that made it, so the
//! model cannot be touched from the drawing loop -- and the two talk through a pair of channels:
//! a prompt goes one way, and how far along the run is comes back the other.
//!
//! What comes back is a file name rather than a picture. A terminal is a poor place to look at
//! one, so every finished run is written out where an image viewer can open it, and the screen
//! keeps the list of what it wrote.

use std::io;
use std::ops::ControlFlow;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::Arc;
use std::time::{Duration, Instant};

use ratatui::crossterm::event::{self, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Color, Style, Stylize};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, BorderType, Gauge, Paragraph, Wrap};
use ratatui::{DefaultTerminal, Frame};

use crate::cli::args::Args;
use crate::cli::ask::{self, Answer};
use crate::cli::centred;
use crate::cli::field::{edit_text, TextField};
use crate::cli::files::{FilePicker, Outcome};
use crate::cli::picker;
use crate::cli::{built_from, hub};
use crate::flint::Tensor;
use crate::{from_rgb8, to_rgb8, Device, GenerationOptions, GenerationProgress, Sdxl, ZipFile};

type Error = Box<dyn std::error::Error>;

/// The sizes on offer, which are the ones SDXL was trained at. Every one of them is a multiple of
/// the 32 pixels the U-Net's own halvings need.
/// How tall the prompt boxes are: two rows to type in and a border round them.
///
/// Stacked rather than side by side, and this is why. A prompt here is a list of tags rather than
/// a sentence, and a long one in a box half the screen wide wraps every few words -- the cursor
/// ends up somewhere down the middle of a paragraph with no line long enough to find it on. Full
/// width and two rows is a box a tag list can be steered around in, and it is a row shorter than
/// the three it used to have.
const TEXT_BOX: u16 = 4;

/// How big the button is: a block rather than a stripe, and no wider than it needs to be.
///
/// Drawn across the whole screen it read as a rule between two halves of the screen rather than as
/// something to press, which is the shape a button is supposed to have all of.
const BUTTON: u16 = 3;
const BUTTON_WIDTH: u16 = 24;

const SIZES: &[(i32, i32)] = &[
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

/// What a fresh screen asks for: the square SDXL is happiest at.
const DEFAULT_SIZE: usize = 4;

/// How far a run that starts from a picture walks away from it, before anyone says otherwise.
const DEFAULT_STRENGTH: f32 = 0.8;

/// What the picture picker offers, which is what the image crate is built to read.
const PICTURES: &[&str] = &["png", "jpg", "jpeg"];

/// Says what was wrong before printing the usage, which is the order the Go tool prints them in.
fn with_usage<T, E: std::fmt::Display>(result: Result<T, E>) -> Result<T, E> {
    if let Err(error) = &result {
        eprintln!("{error}\n");
        print_usage();
    }
    result
}

/// Puts the terminal back the way it was, from inside the tensor library's last breath.
///
/// A check that fails there prints what went wrong and calls `abort()`, and the message is worth
/// nothing written across a screen full of boxes -- which is where it lands, since by then the
/// alternate screen is up, the cursor is hidden and the terminal is in raw mode. So the last thing
/// that happens before the message is this.
///
/// It runs on whichever thread failed, which may not be the one drawing. Racing a half-finished
/// frame is not a concern: nothing after this draws another, and a frame torn on the way out is
/// better than a message nobody can read.
extern "C" fn give_the_screen_back() {
    ratatui::restore();

    // And the cursor, which restore() does not do: hiding it is the Terminal's and showing it
    // again is what its Drop does, and abort() runs no destructors. Without this the shell that
    // gets the terminal back has no cursor in it.
    let _ = ratatui::crossterm::execute!(io::stdout(), ratatui::crossterm::cursor::Show);
}

/// Takes the terminal and paints over whatever was on it.
///
/// The clear is the whole reason this is a function. ratatui writes only what changed since the
/// frame before, and the first frame is compared against a blank one, so a cell that is blank in
/// the drawing is a cell it never writes -- it is relying on the alternate screen it just entered
/// being empty. Which it is, unless the terminal was already in one: entering it a second time
/// changes nothing, and then every blank cell shows whatever the last program left there.
///
/// That is not a rare shape. It is what a program that took the screen and died without giving it
/// back leaves behind, which until a moment ago was what a failed check inside the tensor library
/// did, and one of those can leave every run after it looking corrupted.
fn take_the_screen() -> Result<DefaultTerminal, Error> {
    let mut terminal = ratatui::init();
    terminal.clear()?;

    Ok(terminal)
}

fn print_usage() {
    eprintln!("Usage: waifu draw [OPTIONS]");
    eprintln!();
    eprintln!("Options:");
    crate::cli::args::print_options();
    eprintln!();
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

    // Before the screen goes up, and it stays for the rest of the run: both screens take the
    // terminal, and a check that fails inside the tensor library can come from either.
    crate::flint::on_fatal(give_the_screen_back);

    let model = with_usage(args.model())?.map(str::to_string);
    let mut device = with_usage(args.device())?.resolve();

    // Either a path, or a name like "sdxl:base" that is fetched into the cache first. Done here,
    // before the screen is set up, because fetching a model prints as it goes and the terminal is
    // still the terminal at this point. Without a name, the screen goes up early and offers the
    // published models instead -- a fetch there is minutes long and wants a progress bar rather
    // than a scrolling line.
    // What the screen calls the model. As typed when it was named, because that is what someone
    // would type again; the file name when a path was given, because the whole path does not fit
    // and its tail is the part that identifies it.
    let mut model_name = model.clone().map(|model| match model.rsplit_once('/') {
        Some((_, file)) if !file.is_empty() => file.to_string(),
        _ => model,
    });

    let model_path = match model {
        Some(model) => hub::resolve(&model)?,
        None => {
            // The screen offers the device as well, starting on whatever `-d` resolved to, so
            // what comes back may not be what went in.
            let mut terminal = take_the_screen()?;
            let chosen = picker::choose(&mut terminal, device);
            ratatui::restore();

            match chosen? {
                Some(chosen) => {
                    device = chosen.device;
                    model_name = Some(chosen.name);
                    chosen.path
                }
                None => return Ok(()),
            }
        }
    };

    // Before the path goes to the painter, which takes it with it. A path with no name behind it
    // still has to say something, and its file name is the part that tells one apart from another.
    let model_name = model_name.unwrap_or_else(|| {
        model_path
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_else(|| "a model".to_string())
    });

    // Set while a run is in flight to ask it to stop between steps, which is the only place it
    // can be asked: a step, once started, is a kernel launch that nothing here can call back.
    let cancel = Arc::new(AtomicBool::new(false));
    let (jobs, waiting) = channel::<Job>();
    let (updates, arriving) = channel::<Update>();

    let painter = std::thread::spawn({
        let cancel = Arc::clone(&cancel);
        move || paint(&model_path, device, &waiting, &updates, &cancel)
    });

    // Waited for out here rather than on the screen, because the tensor library says what
    // hardware it found by printing it, and what it prints it to is the terminal the screen is
    // about to be drawn on. Everything it has to say it says while the model is being read, so
    // letting that finish first is all it takes to keep the two apart.
    println!("Reading the model. This takes a moment.");
    match arriving.recv() {
        Ok(Update::Ready) => {}
        Ok(Update::Failed(error)) => return Err(error.into()),
        _ => return Err("the model was never read".into()),
    }

    let terminal = take_the_screen()?;
    let mut app = App::new(model_name, device);

    // -i only fills the box. Everything about a run is changeable between runs, and a picture
    // named on the command line is no different -- it is where to start, not what to be stuck
    // with.
    if let Some(image) = args.image() {
        app.from = TextField::new(image);
    }

    let outcome = run(terminal, app, &jobs, &arriving, &cancel);
    ratatui::restore();

    // The painter is left where it is rather than joined. It may be halfway through the decode,
    // which is the one part of a run that cannot be cut short, and there is nothing it holds that
    // the end of the process does not release.
    cancel.store(true, Ordering::Relaxed);
    drop(jobs);
    drop(painter);

    outcome
}

/// Draws, reads the keyboard, and passes what the two channels carry between them.
fn run(
    mut terminal: DefaultTerminal,
    mut app: App,
    jobs: &Sender<Job>,
    updates: &Receiver<Update>,
    cancel: &AtomicBool,
) -> Result<(), Error> {
    while !app.quit {
        terminal.draw(|frame| app.render(frame))?;

        // Long enough that an idle screen costs nothing, short enough that the bar moves.
        if event::poll(Duration::from_millis(100))? {
            if let Event::Key(key) = event::read()? {
                if key.kind == KeyEventKind::Press {
                    app.key(key, cancel);
                }
            }
        }

        // Until there is nothing more to hear, or nobody left to hear it from -- a painter that
        // gave up has already said why.
        while let Ok(update) = updates.try_recv() {
            app.update(update);
        }

        if let Some(job) = app.pending.take() {
            if jobs.send(job).is_err() {
                app.failed("the model is no longer there");
            }
        }
    }

    Ok(())
}

/// A prompt to draw.
struct Job {
    prompt: String,
    options: GenerationOptions,
    /// The picture to start from, where the run is not starting from noise. Handed over as the
    /// path it was typed as: reading it is minutes of work away and belongs on the thread that
    /// does the work, not on the one drawing the screen.
    from: Option<PathBuf>,
}

/// A finished picture, as the file it was written to.
struct Kept {
    path: PathBuf,
    width: usize,
    height: usize,
}

/// What the painter has to say about it.
enum Update {
    /// The model is loaded and a prompt can be sent.
    Ready,
    Progress(GenerationProgress),
    Done {
        kept: Kept,
        elapsed: Duration,
    },
    /// The run gave up where it was, because it was asked to.
    Stopped,
    Failed(String),
}

/// The thread that owns the model: it reads jobs, and says what it is doing as it does it.
fn paint(
    model_path: &Path,
    device: Device,
    jobs: &Receiver<Job>,
    updates: &Sender<Update>,
    cancel: &AtomicBool,
) {
    let model = match load(model_path, device) {
        Ok(model) => model,
        Err(error) => {
            let _ = updates.send(Update::Failed(error.to_string()));
            return;
        }
    };
    if updates.send(Update::Ready).is_err() {
        return;
    }

    for job in jobs {
        // A stop asked for after the last run finished is not this run's business.
        cancel.store(false, Ordering::Relaxed);
        let started = Instant::now();

        let mut report = |progress| {
            let _ = updates.send(Update::Progress(progress));
            if cancel.load(Ordering::Relaxed) {
                ControlFlow::Break(())
            } else {
                ControlFlow::Continue(())
            }
        };

        // A picture to start from is read and scaled here, before the run, so that a path that
        // is not there is one message rather than a run that fails partway.
        let started_from = match &job.from {
            Some(path) => match read_image(path, job.options.width, job.options.height) {
                Ok(image) => Some(image),
                Err(error) => {
                    let _ = updates.send(Update::Failed(format!("{}: {error}", path.display())));
                    continue;
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

        let update = match drawn {
            // Written here rather than back on the drawing thread: the pixels are three megabytes
            // that nothing on the other side would do anything with but hand to a file.
            Ok(Some(image)) => match keep(&image) {
                Ok(kept) => Update::Done {
                    kept,
                    elapsed: started.elapsed(),
                },
                Err(error) => Update::Failed(error.to_string()),
            },
            Ok(None) => Update::Stopped,
            Err(error) => Update::Failed(error.to_string()),
        };

        if updates.send(update).is_err() {
            return;
        }
    }
}

fn load(model_path: &Path, device: Device) -> crate::Result<Sdxl> {
    let package = ZipFile::open(model_path)?;
    Sdxl::from_package(device, &package)
}

/// A picture from a file, as the `(1, 3, height, width)` tensor a run can start from.
///
/// Scaled to the size the screen asks for rather than kept at its own: the model works at
/// multiples of 64, and a picture off a camera or a phone is not one. Stretched to it rather than
/// cropped, so that the whole of what was handed in is what the run sees -- the sizes on offer
/// include the portrait and landscape ones SDXL was trained at, which is where to say what shape
/// the picture is.
///
/// Lanczos, because this is the direction that loses pixels -- a photograph is larger than
/// anything SDXL draws -- and a cheaper filter leaves stair steps that the run then faithfully
/// keeps.
fn read_image(path: &Path, width: i32, height: i32) -> Result<Tensor, Error> {
    let opened = image::ImageReader::open(path)?
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
fn keep(image: &Tensor) -> Result<Kept, Error> {
    let pixels = to_rgb8(image)?;
    let shape = image.shape();
    let (width, height) = (shape[3] as usize, shape[2] as usize);

    for number in 1..10_000 {
        let path = PathBuf::from(format!("waifu-{number:04}.png"));
        if path.exists() {
            continue;
        }

        image::save_buffer(
            &path,
            &pixels,
            width as u32,
            height as u32,
            image::ColorType::Rgb8,
        )?;
        return Ok(Kept {
            path,
            width,
            height,
        });
    }

    Err(io::Error::other("there are already ten thousand pictures in this directory").into())
}

/// The boxes on the screen, in the order tab walks them.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Field {
    Prompt,
    Negative,
    From,
    Steps,
    Guidance,
    Size,
    Strength,
    Seed,
    /// The one that is not a box to fill in but the thing all of them are for.
    Generate,
}

const FIELDS: [Field; 9] = [
    Field::Prompt,
    Field::Negative,
    Field::From,
    Field::Steps,
    Field::Guidance,
    Field::Size,
    Field::Strength,
    Field::Seed,
    Field::Generate,
];

/// The boxes as they sit on the screen, row by row.
///
/// Tab walks [`FIELDS`] in order and does not care how they are arranged; up and down do, because
/// what is above the row of numbers is the box above it and not the number to its left. The two
/// lists hold the same boxes, which [`the_grid_and_the_tab_order_hold_the_same_boxes`] checks.
const ROWS: &[&[Field]] = &[
    &[Field::Prompt],
    &[Field::Negative],
    &[Field::From],
    &[
        Field::Steps,
        Field::Guidance,
        Field::Size,
        Field::Strength,
        Field::Seed,
    ],
    &[Field::Generate],
];

impl Field {
    /// Which row it is on and how far along, which is where up, down, left and right start from.
    fn at(self) -> (usize, usize) {
        for (row, boxes) in ROWS.iter().enumerate() {
            if let Some(column) = boxes.iter().position(|box_| *box_ == self) {
                return (row, column);
            }
        }

        unreachable!("every box is somewhere on the screen")
    }

    /// Whether this is a box typed into, which decides what left and right do while in it.
    ///
    /// Everywhere else they move between boxes, which is what the row of numbers wants: they sit
    /// side by side and there is nothing else those two keys could be pointing at. In a box of
    /// text there is -- the cursor -- so there they keep meaning that, and tab and up and down
    /// are how to leave.
    fn takes_text(self) -> bool {
        matches!(
            self,
            Field::Prompt | Field::Negative | Field::From | Field::Seed
        )
    }
}

/// The box `by` places along the row from `field`, stopping at either end of it.
fn along_from(field: Field, by: isize) -> Field {
    let (row, column) = field.at();
    let column = (column as isize + by).clamp(0, ROWS[row].len() as isize - 1) as usize;

    ROWS[row][column]
}

/// A box on top of the screen asking for one value, and which box the answer goes into.
enum Asking {
    Number(Field, ask::Number),
    Size(ask::Choice),
}

/// What the screen is waiting for.
enum Status {
    Ready,
    Running {
        progress: GenerationProgress,
        /// How many steps this run was asked for, which the bar needs after the last one.
        steps: i32,
        started: Instant,
    },
}

struct App {
    prompt: TextField,
    negative: TextField,
    /// The picture to draw from. Empty is the ordinary case: a run that starts from noise.
    from: TextField,
    /// Empty means a different picture every time; a number means the same one every time.
    seed: TextField,
    steps: i32,
    guidance: f32,
    size: usize,
    /// How far a run that starts from a picture walks away from it, by deciding how much noise
    /// it starts under. Not how many steps run -- that is the box beside it, either way. Read by
    /// nothing when the box above is empty.
    strength: f32,
    focus: usize,
    /// The column up and down aim for, which is not always the column they land in: a row with
    /// one box in it takes every column, and leaving it again should go back to where it came
    /// from rather than to the near end. What a text editor remembers for the same two keys.
    column: usize,

    status: Status,
    /// The last thing worth saying, under the bar.
    message: String,
    /// Whether that thing was bad news.
    unhappy: bool,
    /// What has been written out this session, newest first.
    written: Vec<String>,
    /// The job the loop is about to hand to the painter.
    pending: Option<Job>,
    /// The file picker, while it is up. Everything else on the screen keeps its state behind it,
    /// so closing one puts the screen back exactly as it was.
    browsing: Option<FilePicker>,
    /// Where the picker was last looking, so that closing it without picking anything and opening
    /// it again does not walk back to where the program was started from.
    browsed: Option<PathBuf>,
    /// The box asking for one value, while it is up.
    asking: Option<Asking>,
    /// The box asking whether to leave, while it is up.
    leaving: Option<ask::Confirm>,
    quit: bool,

    /// What the run was pointed at. Shown rather than kept, because a picture that came out wrong
    /// is asked about later, and by then which model and which device made it is the first thing
    /// nobody remembers.
    model: String,
    device: Device,
}

impl App {
    fn new(model: String, device: Device) -> App {
        App {
            model,
            device,
            prompt: TextField::default(),
            negative: TextField::default(),
            from: TextField::default(),
            seed: TextField::default(),
            steps: 30,
            guidance: 5.0,
            size: DEFAULT_SIZE,
            // What every image to image interface starts at: enough to redraw the picture in the
            // style asked for, not so much that its composition is gone.
            strength: DEFAULT_STRENGTH,
            focus: 0,
            column: 0,
            status: Status::Ready,
            message: "type a prompt, then press enter on generate".to_string(),
            unhappy: false,
            written: Vec::new(),
            pending: None,
            browsing: None,
            browsed: None,
            asking: None,
            leaving: None,
            quit: false,
        }
    }

    fn focused(&self) -> Field {
        FIELDS[self.focus]
    }

    fn running(&self) -> bool {
        matches!(self.status, Status::Running { .. })
    }

    /// Whether a box is up over the screen, which is where the keys are going.
    fn covered(&self) -> bool {
        self.browsing.is_some() || self.asking.is_some() || self.leaving.is_some()
    }

    fn failed(&mut self, message: impl Into<String>) {
        self.message = message.into();
        self.unhappy = true;
    }

    fn said(&mut self, message: impl Into<String>) {
        self.message = message.into();
        self.unhappy = false;
    }

    // -- what the painter says ------------------------------------------------------------

    fn update(&mut self, update: Update) {
        match update {
            // Said once, and heard before there was a screen to say it on.
            Update::Ready => self.status = Status::Ready,
            Update::Progress(reported) => {
                if let Status::Running { progress, .. } = &mut self.status {
                    *progress = reported;

                    // A refusal shown on the bar's label stands until the run moves, which is a
                    // step or two: long enough to be read, short enough that the count it is
                    // standing in front of is not gone for the rest of the run.
                    self.message.clear();
                    self.unhappy = false;
                }
            }
            Update::Done { kept, elapsed } => {
                self.status = Status::Ready;
                self.said(format!("written to {}", kept.path.display()));
                self.written.insert(
                    0,
                    format!(
                        "{}   {} by {}   {:.1}s",
                        kept.path.display(),
                        kept.width,
                        kept.height,
                        elapsed.as_secs_f64()
                    ),
                );
            }
            Update::Stopped => {
                self.status = Status::Ready;
                self.said("stopped");
            }
            Update::Failed(error) => {
                self.status = Status::Ready;
                self.failed(error);
            }
        }
    }

    // -- what the keyboard says -----------------------------------------------------------

    fn key(&mut self, key: KeyEvent, cancel: &AtomicBool) {
        let control = key.modifiers.contains(KeyModifiers::CONTROL);

        // Ctrl-C still ends the program from inside any box on top of the screen: it is the key
        // that always means that, and a box is no reason for it to stop meaning it.
        if self.covered() && key.code == KeyCode::Char('c') && control {
            cancel.store(true, Ordering::Relaxed);
            self.quit = true;
            return;
        }

        if let Some(leaving) = &mut self.leaving {
            match leaving.key(key) {
                Answer::Open => {}
                Answer::Cancelled | Answer::Given(false) => self.leaving = None,
                Answer::Given(true) => {
                    cancel.store(true, Ordering::Relaxed);
                    self.quit = true;
                }
            }
            return;
        }

        if let Some(asking) = &mut self.asking {
            match asking {
                Asking::Number(field, box_) => match box_.key(key) {
                    Answer::Open => {}
                    Answer::Cancelled => self.asking = None,
                    Answer::Given(value) => {
                        match field {
                            Field::Steps => self.steps = value as i32,
                            Field::Guidance => self.guidance = value as f32,
                            Field::Strength => self.strength = value as f32,
                            _ => {}
                        }
                        self.asking = None;
                    }
                },
                Asking::Size(box_) => match box_.key(key) {
                    Answer::Open => {}
                    Answer::Cancelled => self.asking = None,
                    Answer::Given(index) => {
                        self.size = index;
                        self.asking = None;
                    }
                },
            }
            return;
        }

        if let Some(browsing) = &mut self.browsing {
            let outcome = browsing.key(key);
            if outcome != Outcome::Open {
                self.browsed = Some(browsing.directory().to_path_buf());
                self.browsing = None;
            }
            if let Outcome::Picked(path) = outcome {
                self.from = TextField::new(&path.to_string_lossy());
            }
            return;
        }

        match key.code {
            KeyCode::Char('c') if control => {
                cancel.store(true, Ordering::Relaxed);
                self.quit = true;
            }
            KeyCode::Esc => {
                // While a run is in flight this ends the run rather than the program, so that
                // one wrong prompt does not cost a reload of the model.
                if self.running() {
                    cancel.store(true, Ordering::Relaxed);
                    self.said("stopping after this step");
                } else {
                    // Asked rather than done. Leaving costs the minute it takes to read the model
                    // back in, and escape is the key pressed to get out of everything else here.
                    self.leaving = Some(ask::Confirm::ask("leave waifu?"));
                }
            }
            // Enter is what the box the cursor is in does, and for the boxes that do nothing of
            // their own that is still to draw.
            KeyCode::Enter => self.enter(),

            // Tab walks every box in order, wherever it is; the arrows walk the screen as it is
            // laid out, which is what someone looking at it is walking.
            KeyCode::Tab => self.step_focus(1),
            KeyCode::BackTab => self.step_focus(-1),
            KeyCode::Up => self.move_row(-1),
            KeyCode::Down => self.move_row(1),

            // Along the row, except where the box has its own use for them.
            KeyCode::Left if !self.focused().takes_text() => {
                self.focus_on(along_from(self.focused(), -1));
            }
            KeyCode::Right if !self.focused().takes_text() => {
                self.focus_on(along_from(self.focused(), 1));
            }

            code => self.edit(code),
        }
    }

    /// Moves the cursor `by` boxes along the tab order, round either end.
    fn step_focus(&mut self, by: isize) {
        let count = FIELDS.len() as isize;
        self.focus = ((self.focus as isize + by).rem_euclid(count)) as usize;
    }

    /// Puts the cursor in a named box, and aims up and down at the column it is in.
    fn focus_on(&mut self, field: Field) {
        self.focus = FIELDS
            .iter()
            .position(|box_| *box_ == field)
            .expect("every box is in the tab order");
        self.column = field.at().1;
    }

    /// Moves the cursor `by` rows, as far along the row it lands on as there is room for.
    ///
    /// The column it aims for is not read off the box it is leaving: a row with one box in it
    /// takes every column, and a cursor that walked up out of the size box and back down should
    /// land on the size box rather than on the near end of the row.
    fn move_row(&mut self, by: isize) {
        let column = self.column;
        let (row, _) = self.focused().at();
        let row = (row as isize + by).clamp(0, ROWS.len() as isize - 1) as usize;

        let landed = ROWS[row][column.min(ROWS[row].len() - 1)];
        self.focus = FIELDS
            .iter()
            .position(|box_| *box_ == landed)
            .expect("every box is in the tab order");
        self.column = column;
    }

    /// The keys that mean something to the box the cursor is in.
    ///
    /// Only the boxes typed into have any: a value is changed by opening a box for it, which is
    /// what enter does, rather than by nudging it one arrow press at a time.
    fn edit(&mut self, code: KeyCode) {
        match self.focused() {
            Field::Prompt => edit_text(&mut self.prompt, code, |_| true),
            Field::Negative => edit_text(&mut self.negative, code, |_| true),
            Field::From => edit_text(&mut self.from, code, |_| true),
            Field::Seed => edit_text(&mut self.seed, code, char::is_ascii_digit),
            _ => {}
        }
    }

    /// What enter does, which is whatever the box the cursor is in is for.
    fn enter(&mut self) {
        match self.focused() {
            Field::From => self.browse(),
            Field::Steps => {
                self.asking = Some(Asking::Number(
                    Field::Steps,
                    ask::Number::ask("steps", &self.steps.to_string(), 1.0, 150.0, true),
                ));
            }
            Field::Guidance => {
                self.asking = Some(Asking::Number(
                    Field::Guidance,
                    ask::Number::ask(
                        "guidance",
                        &format!("{:.1}", self.guidance),
                        1.0,
                        20.0,
                        false,
                    ),
                ));
            }
            // Down to zero, which keeps the picture and only sends it through the autoencoder,
            // and up to one, which keeps nothing of it and is the same walk as from noise.
            Field::Strength => {
                self.asking = Some(Asking::Number(
                    Field::Strength,
                    ask::Number::ask(
                        "strength",
                        &format!("{:.2}", self.strength),
                        0.0,
                        1.0,
                        false,
                    ),
                ));
            }
            // A handful someone chose in advance rather than a range, so it is a list.
            Field::Size => {
                let sizes = SIZES
                    .iter()
                    .map(|(width, height)| format!("{width} x {height}"))
                    .collect();
                self.asking = Some(Asking::Size(ask::Choice::ask("size", sizes, self.size)));
            }
            Field::Generate => self.start(),

            // Nothing. A run is minutes long and starts from one place, which is the button --
            // enter in the box a prompt is being typed into is a key pressed on the way to
            // something else as often as it is one meant.
            Field::Prompt | Field::Negative | Field::Seed => {}
        }
    }

    /// What the painter is about to be asked for.
    fn options(&self) -> GenerationOptions {
        let (width, height) = SIZES[self.size];
        GenerationOptions {
            width,
            height,
            num_steps: self.steps,
            guidance_scale: self.guidance,
            negative_prompt: self.negative.text(),
            // Nothing in the box means a new picture every time, which is what leaving the seed
            // out asks the model for.
            seed: self.seed.text().trim().parse().ok(),
            strength: self.strength,
        }
    }

    /// Opens the picker where the box is already pointing, or in this directory when it is empty.
    fn browse(&mut self) {
        // Somewhere that is there, whatever is in the box. What is typed wins where it exists --
        // someone who typed a path means that one -- then where the picker was last left, and a
        // path that is not there at all is not somewhere to open: a typo in the box would
        // otherwise put a list up that says only that it cannot be read.
        let start = [self.from(), self.browsed.clone()]
            .into_iter()
            .flatten()
            .find(|path| path.exists())
            .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));
        self.browsing = Some(FilePicker::open("a picture to draw from", &start, PICTURES));
    }

    /// The picture this run starts from, or None where it starts from noise.
    fn from(&self) -> Option<PathBuf> {
        let typed = self.from.text();
        let typed = typed.trim();
        (!typed.is_empty()).then(|| PathBuf::from(typed))
    }

    /// Hands the prompt over, if there is one and there is nothing already being drawn.
    fn start(&mut self) {
        if self.running() {
            return self.failed("there is already a picture being drawn");
        }

        let prompt = self.prompt.text();
        if prompt.trim().is_empty() {
            return self.failed("there is no prompt to draw");
        }

        let options = self.options();
        self.status = Status::Running {
            progress: GenerationProgress::Encoding,
            steps: options.num_steps,
            started: Instant::now(),
        };
        self.said(String::new());
        self.pending = Some(Job {
            prompt,
            options,
            from: self.from(),
        });
    }

    // -- the screen -----------------------------------------------------------------------

    fn render(&self, frame: &mut Frame) {
        let [heading, prompt, negative, from, numbers, generate, status, written, keys] =
            Layout::vertical([
                Constraint::Length(1),
                // The same height, because the two are the same kind of thing: what to draw and
                // what to keep out of it. One box shorter than the other reads as the shorter one
                // mattering less, which is not what the difference was ever about.
                Constraint::Length(TEXT_BOX),
                Constraint::Length(TEXT_BOX),
                // A path is one line however long it is, so this box is the height of its border.
                Constraint::Length(3),
                Constraint::Length(3),
                Constraint::Length(BUTTON),
                Constraint::Length(3),
                Constraint::Min(0),
                Constraint::Length(1),
            ])
            .areas(frame.area());

        let fifths = [Constraint::Ratio(1, 5); 5];
        let [steps, guidance, size, strength, seed] = Layout::horizontal(fifths).areas(numbers);

        // What made this picture. A screenshot of a run says nothing about which code drew it,
        // and that is the first thing worth knowing about one that came out wrong.
        // The model and the device go to the right, away from what built it: one pair says which
        // code, the other says what it was pointed at, and neither answers for the other.
        let pointed = format!("{}  {}  ", self.model, self.device.name());
        let [built, pointed_at] =
            Layout::horizontal([Constraint::Min(0), Constraint::Length(pointed.len() as u16)])
                .areas(heading);

        frame.render_widget(Paragraph::new(built_from()), built);
        frame.render_widget(
            Paragraph::new(Line::from(vec![
                Span::raw(&self.model).bold(),
                Span::raw("  "),
                Span::styled(self.device.name(), Style::new().fg(Color::DarkGray)),
                Span::raw("  "),
            ])),
            pointed_at,
        );

        self.render_text(frame, prompt, Field::Prompt, "prompt", &self.prompt);
        self.render_text(
            frame,
            negative,
            Field::Negative,
            "away from",
            &self.negative,
        );
        // The one box whose title says what its key does, because a box you type into does not
        // otherwise look like a box you can also open a list from.
        self.render_text(
            frame,
            from,
            Field::From,
            match self.focused() {
                Field::From => "draw from (enter to browse)",
                _ => "draw from",
            },
            &self.from,
        );
        self.render_number(frame, steps, Field::Steps, "steps", self.steps.to_string());
        self.render_number(
            frame,
            guidance,
            Field::Guidance,
            "guidance",
            format!("{:.1}", self.guidance),
        );
        let (width, height) = SIZES[self.size];
        self.render_number(
            frame,
            size,
            Field::Size,
            "size",
            format!("{width}x{height}"),
        );
        // Greyed out where the box above it is empty, since a run from noise never reads it.
        self.render_number(
            frame,
            strength,
            Field::Strength,
            "strength",
            match self.from() {
                Some(_) => format!("{:.2}", self.strength),
                None => "-".to_string(),
            },
        );
        self.render_number(
            frame,
            seed,
            Field::Seed,
            "seed",
            match self.seed.text().trim() {
                "" => "any".to_string(),
                seed => seed.to_string(),
            },
        );

        self.render_generate(frame, generate);
        self.render_status(frame, status);
        self.render_written(frame, written);

        // A box on top carries its own line of key hints, and while one is up these are not what
        // the keys do. Two rows saying different things about the same keys is worse than one.
        //
        // Last, so that it is on top of everything above rather than under it.
        if let Some(browsing) = &self.browsing {
            browsing.render(frame, frame.area());
            return;
        }
        if let Some(asking) = &self.asking {
            match asking {
                Asking::Number(_, box_) => box_.render(frame, frame.area()),
                Asking::Size(box_) => box_.render(frame, frame.area()),
            }
            return;
        }
        if let Some(leaving) = &self.leaving {
            leaving.render(frame, frame.area());
            return;
        }

        frame.render_widget(
            Paragraph::new(Line::from(vec![
                Span::raw(" tab").bold(),
                Span::raw(" or "),
                Span::raw("arrows").bold(),
                Span::raw(" move  "),
                Span::raw("enter").bold(),
                Span::raw(" open  "),
                Span::raw("esc").bold(),
                Span::raw(" stop or leave"),
            ]))
            .style(Style::new().fg(Color::DarkGray)),
            keys,
        );
    }

    /// The one thing on the screen that is not a box to fill in.
    ///
    /// Drawn as a bar of colour rather than as another bordered box, because it is not the same
    /// kind of thing as the boxes above it and should not have to be read to find that out. It is
    /// still in the order tab walks, so it is reachable the way everything else is, and enter on
    /// any of the boxes with nothing else to do still starts a run.
    fn render_generate(&self, frame: &mut Frame, area: Rect) {
        let area = centred(area, BUTTON_WIDTH, area.height);
        let focused = self.focused() == Field::Generate;
        let (label, style) = match (self.running(), focused) {
            // Nothing to press while one is already being drawn, and the bar says so rather than
            // sitting there green and inviting a second press that would be refused.
            (true, _) => (
                "drawing".to_string(),
                Style::new().fg(Color::Black).bg(Color::DarkGray),
            ),
            (false, true) => (
                "> generate <".to_string(),
                Style::new().fg(Color::Black).bg(Color::LightGreen).bold(),
            ),
            (false, false) => (
                "generate".to_string(),
                Style::new().fg(Color::Black).bg(Color::Green),
            ),
        };

        // The label on the middle row of the block, with a row of colour above and below it, so
        // that what is on the screen is a button with a word in it rather than a coloured word.
        let lines = vec![
            Line::raw(""),
            Line::from(Span::styled(label, style)),
            Line::raw(""),
        ];
        frame.render_widget(Paragraph::new(lines).centered().style(style), area);
    }

    /// The border a box gets, which is where the cursor being in it shows.
    fn box_of(&self, title: &str, field: Field) -> Block<'static> {
        let focused = self.focused() == field;
        let colour = if focused {
            Color::Yellow
        } else {
            Color::DarkGray
        };

        Block::bordered()
            .border_type(if focused {
                BorderType::Thick
            } else {
                BorderType::Rounded
            })
            .border_style(Style::new().fg(colour))
            .title(Span::styled(format!(" {title} "), Style::new().fg(colour)))
    }

    fn render_text(
        &self,
        frame: &mut Frame,
        area: Rect,
        field: Field,
        title: &str,
        text: &TextField,
    ) {
        let outline = self.box_of(title, field);
        let inner = outline.inner(area);
        frame.render_widget(outline, area);
        if inner.is_empty() {
            return;
        }

        let (lines, (row, column)) = text.wrapped(inner.width as usize);
        // Scrolled by whole lines, only as far as it takes to keep the cursor in the box.
        let height = inner.height as usize;
        let first = (row + 1).saturating_sub(height);

        let shown: Vec<Line> = lines
            .iter()
            .skip(first)
            .take(height)
            .map(|line| Line::raw(line.clone()))
            .collect();
        frame.render_widget(Paragraph::new(shown), inner);

        if self.focused() == field {
            frame.set_cursor_position((inner.x + column as u16, inner.y + (row - first) as u16));
        }
    }

    fn render_number(
        &self,
        frame: &mut Frame,
        area: Rect,
        field: Field,
        title: &str,
        value: String,
    ) {
        let outline = self.box_of(title, field);
        let inner = outline.inner(area);
        frame.render_widget(outline, area);

        // Nothing but the value. There used to be a pair of arrows here saying which keys moved
        // it, which is not what they do any more; that the border is lit is what says the cursor
        // is here, and the line at the foot is what says enter opens it.
        let line = if self.focused() == field {
            Line::from(Span::raw(value).bold())
        } else {
            Line::raw(value)
        };
        frame.render_widget(Paragraph::new(line.centered()), inner);
    }

    fn render_status(&self, frame: &mut Frame, area: Rect) {
        let outline = Block::bordered()
            .border_type(BorderType::Rounded)
            .border_style(Style::new().fg(Color::DarkGray))
            .title(Span::styled(
                " what it is doing ",
                Style::new().fg(Color::DarkGray),
            ));
        let inner = outline.inner(area);
        frame.render_widget(outline, area);
        if inner.is_empty() {
            return;
        }

        // One row rather than a bar with a line under it. The two never had anything to say at
        // the same time: while a run is on, the bar's own label says where it is, and the last
        // thing that happened is only worth reading once there is no bar.
        match &self.status {
            Status::Ready => {
                let colour = if self.unhappy {
                    Color::Red
                } else if self.message.is_empty() {
                    Color::DarkGray
                } else {
                    Color::Gray
                };
                let said = if self.message.is_empty() {
                    "waiting for a prompt"
                } else {
                    &self.message
                };
                frame.render_widget(
                    Paragraph::new(said)
                        .style(Style::new().fg(colour))
                        .wrap(Wrap { trim: true }),
                    inner,
                );
            }
            Status::Running {
                progress,
                steps,
                started,
            } => {
                // A message raised while a run is on goes in the label, which is the one place
                // there is room for it. Turning down a second prompt is the case that matters:
                // saying nothing would read as the key not having worked.
                let label = if self.unhappy {
                    self.message.clone()
                } else {
                    format!(
                        "{} ({:.0}s)",
                        doing(*progress),
                        started.elapsed().as_secs_f64()
                    )
                };
                frame.render_widget(
                    Gauge::default()
                        .ratio(fraction(*progress, *steps))
                        .label(label)
                        // Named so that the label reads against the bar: ratatui draws it by
                        // swapping these two, and there is nothing to swap without a background.
                        .gauge_style(Style::new().fg(Color::Magenta).bg(Color::Black)),
                    inner,
                );
            }
        }
    }

    /// What this session has drawn, which is the only place a finished picture shows up: a
    /// terminal is no place to look at one, so the file name is what is worth keeping on screen.
    fn render_written(&self, frame: &mut Frame, area: Rect) {
        let outline = Block::bordered()
            .border_type(BorderType::Rounded)
            .border_style(Style::new().fg(Color::DarkGray))
            .title(Span::styled(
                " pictures written ",
                Style::new().fg(Color::DarkGray),
            ));
        let inner = outline.inner(area);
        frame.render_widget(outline, area);
        if inner.is_empty() {
            return;
        }

        if self.written.is_empty() {
            frame.render_widget(
                Paragraph::new(Line::raw("nothing yet").fg(Color::DarkGray)),
                inner,
            );
            return;
        }

        // Newest first, and only as many as there is room for.
        let lines: Vec<Line> = self
            .written
            .iter()
            .take(inner.height as usize)
            .enumerate()
            .map(|(index, written)| {
                let colour = if index == 0 {
                    Color::Green
                } else {
                    Color::DarkGray
                };
                Line::raw(written.clone()).fg(colour)
            })
            .collect();
        frame.render_widget(Paragraph::new(lines), inner);
    }
}

/// How far along a run is, as a bar can show it.
///
/// Weighed rather than counted. Reading the prompt costs about a step and turning the finished
/// latent into pixels costs several, so a bar drawn from the steps alone would fill up and then
/// sit there for the slowest part of the run.
fn fraction(progress: GenerationProgress, steps: i32) -> f64 {
    const ENCODING: f64 = 1.0;
    const DECODING: f64 = 3.0;

    let all = |steps: i32| ENCODING + f64::from(steps.max(1)) + DECODING;
    match progress {
        GenerationProgress::Encoding => 0.0,
        GenerationProgress::Step { done, total } => (ENCODING + f64::from(done)) / all(total),
        GenerationProgress::Decoding => (ENCODING + f64::from(steps.max(1))) / all(steps),
    }
}

/// What a run is busy with, in words.
fn doing(progress: GenerationProgress) -> String {
    match progress {
        GenerationProgress::Encoding => "reading the prompt".to_string(),
        GenerationProgress::Step { done, total } => format!("step {done} of {total}"),
        GenerationProgress::Decoding => "making the picture".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use ratatui::backend::TestBackend;

    use super::*;

    fn press(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, KeyModifiers::NONE)
    }

    /// A screen as it is when the model has been read, which is as it starts.
    fn ready() -> App {
        App::new("sdxl:wai".to_string(), Device::Cpu)
    }

    fn type_in(app: &mut App, text: &str, cancel: &AtomicBool) {
        for character in text.chars() {
            app.key(press(KeyCode::Char(character)), cancel);
        }
    }

    #[test]
    fn every_size_is_one_the_model_can_be_asked_for() {
        // The U-Net halves the picture twice on top of the eight the autoencoder stands for, so
        // anything not a multiple of 32 is refused by the pipeline rather than rounded.
        for (width, height) in SIZES {
            assert_eq!(width % 32, 0, "{width}");
            assert_eq!(height % 32, 0, "{height}");
        }
        assert_eq!(SIZES[DEFAULT_SIZE], (1024, 1024));
    }

    #[test]
    fn typing_goes_to_the_box_the_cursor_is_in() {
        let cancel = AtomicBool::new(false);
        let mut app = ready();

        type_in(&mut app, "a cat", &cancel);
        app.key(press(KeyCode::Tab), &cancel);
        type_in(&mut app, "blurry", &cancel);

        assert_eq!(app.prompt.text(), "a cat");
        assert_eq!(app.negative.text(), "blurry");
    }

    /// Presses the button, which is the one place a run starts from.
    fn generate(app: &mut App, cancel: &AtomicBool) {
        focus_on(app, Field::Generate);
        app.key(press(KeyCode::Enter), cancel);
    }

    /// Puts the cursor in a named box, rather than counting tab presses to it: which number a
    /// box is changes whenever one is added, and the tests are not about the order.
    ///
    /// The screen's own, so that a test starts a box the way the keys leave it -- with up and
    /// down aimed at the column it is in.
    fn focus_on(app: &mut App, field: Field) {
        app.focus_on(field);
    }

    #[test]
    fn tab_goes_round_and_back_again() {
        let cancel = AtomicBool::new(false);
        let mut app = ready();

        for _ in 0..FIELDS.len() {
            app.key(press(KeyCode::Tab), &cancel);
        }
        assert_eq!(app.focused(), Field::Prompt, "all the way round");

        app.key(press(KeyCode::BackTab), &cancel);
        assert_eq!(app.focused(), Field::Generate, "and one back off the end");
    }

    #[test]
    fn the_grid_and_the_tab_order_hold_the_same_boxes() {
        // Two lists of the same boxes, and a box in one and not the other is either unreachable
        // by tab or nowhere the arrows can go.
        let mut on_screen: Vec<Field> = ROWS.iter().flat_map(|row| row.iter().copied()).collect();
        let mut in_order = FIELDS.to_vec();
        on_screen.sort_by_key(|field| format!("{field:?}"));
        in_order.sort_by_key(|field| format!("{field:?}"));

        assert_eq!(on_screen, in_order);
    }

    #[test]
    fn up_and_down_walk_the_screen_as_it_is_laid_out() {
        let cancel = AtomicBool::new(false);
        let mut app = ready();

        // Above the row of numbers is the box above it, not the number to its left.
        focus_on(&mut app, Field::Size);
        app.key(press(KeyCode::Up), &cancel);
        assert_eq!(app.focused(), Field::From);

        // And coming back lands on the one it left from, so looking around does not lose it.
        app.key(press(KeyCode::Down), &cancel);
        assert_eq!(app.focused(), Field::Size);

        // The three boxes above it are stacked, one to a row, so they are walked one at a time
        // however far along the row below the cursor started.
        focus_on(&mut app, Field::From);
        app.key(press(KeyCode::Up), &cancel);
        assert_eq!(app.focused(), Field::Negative);
        app.key(press(KeyCode::Up), &cancel);
        assert_eq!(app.focused(), Field::Prompt);

        focus_on(&mut app, Field::Seed);
        app.key(press(KeyCode::Up), &cancel);
        assert_eq!(app.focused(), Field::From, "the row above has one box");

        // Below the numbers is the button, and there is nothing below that.
        focus_on(&mut app, Field::Strength);
        app.key(press(KeyCode::Down), &cancel);
        assert_eq!(app.focused(), Field::Generate);
        app.key(press(KeyCode::Down), &cancel);
        assert_eq!(app.focused(), Field::Generate, "down walked off the screen");

        // And nothing above the top.
        focus_on(&mut app, Field::Prompt);
        app.key(press(KeyCode::Up), &cancel);
        assert_eq!(app.focused(), Field::Prompt);
    }

    #[test]
    fn the_arrows_move_between_boxes_except_where_a_box_wants_them() {
        let cancel = AtomicBool::new(false);
        let mut app = ready();

        // In the row of numbers, left and right are the only thing they could be pointing at.
        focus_on(&mut app, Field::Guidance);
        app.key(press(KeyCode::Left), &cancel);
        assert_eq!(app.focused(), Field::Steps);
        app.key(press(KeyCode::Right), &cancel);
        app.key(press(KeyCode::Right), &cancel);
        assert_eq!(app.focused(), Field::Size);

        // They stop at the ends of the row rather than wrapping onto another one: a row is a row
        // and walking off it sideways is not something the screen shows happening.
        focus_on(&mut app, Field::Steps);
        app.key(press(KeyCode::Left), &cancel);
        assert_eq!(app.focused(), Field::Steps);
        focus_on(&mut app, Field::Seed);
        app.key(press(KeyCode::Right), &cancel);
        assert_eq!(app.focused(), Field::Seed);

        // And they do not touch the values on the way past, which is what they used to do.
        assert_eq!(app.steps, 30);
        assert_eq!(app.guidance, 5.0);

        // In a box of text there is something else they mean, so they keep meaning it.
        focus_on(&mut app, Field::Prompt);
        type_in(&mut app, "cat", &cancel);
        app.key(press(KeyCode::Left), &cancel);
        app.key(press(KeyCode::Left), &cancel);
        assert_eq!(
            app.focused(),
            Field::Prompt,
            "left walked out of a text box"
        );
        type_in(&mut app, "-", &cancel);
        assert_eq!(app.prompt.text(), "c-at", "left did not move the cursor");
    }

    /// Opens the box on `field`, types `text` into it, and presses enter.
    fn set(app: &mut App, field: Field, text: &str, cancel: &AtomicBool) {
        focus_on(app, field);
        app.key(press(KeyCode::Enter), cancel);
        assert!(
            app.asking.is_some(),
            "enter did not open a box for {field:?}"
        );

        for _ in 0..12 {
            app.key(press(KeyCode::Backspace), cancel);
        }
        type_in(app, text, cancel);
        app.key(press(KeyCode::Enter), cancel);
    }

    #[test]
    fn a_number_is_typed_into_a_box_rather_than_nudged() {
        let cancel = AtomicBool::new(false);
        let mut app = ready();

        set(&mut app, Field::Steps, "42", &cancel);
        assert!(app.asking.is_none(), "the box stayed open");
        assert_eq!(app.steps, 42);

        set(&mut app, Field::Guidance, "7.5", &cancel);
        assert_eq!(app.guidance, 7.5);

        set(&mut app, Field::Strength, "0.35", &cancel);
        assert!((app.strength - 0.35).abs() < 1e-6);
    }

    #[test]
    fn the_box_starts_on_the_value_it_is_changing() {
        // So that moving a number by one is two keys and not the whole number again.
        let cancel = AtomicBool::new(false);
        let mut app = ready();

        focus_on(&mut app, Field::Steps);
        app.key(press(KeyCode::Enter), &cancel);
        let Some(Asking::Number(_, number)) = &app.asking else {
            panic!("no number box");
        };
        assert_eq!(number.typed(), "30");
    }

    #[test]
    fn a_number_outside_the_range_is_refused_and_the_box_stays_open() {
        let cancel = AtomicBool::new(false);
        let mut app = ready();

        focus_on(&mut app, Field::Steps);
        app.key(press(KeyCode::Enter), &cancel);
        for _ in 0..12 {
            app.key(press(KeyCode::Backspace), &cancel);
        }
        type_in(&mut app, "999", &cancel);
        app.key(press(KeyCode::Enter), &cancel);

        assert!(app.asking.is_some(), "999 steps was taken");
        assert_eq!(app.steps, 30, "and it changed the value anyway");

        // What is wrong with it is on the box, and esc leaves the value alone.
        let drawn = screen(&app, 90, 28);
        assert!(drawn.contains("1 to 150"), "{drawn}");
        app.key(press(KeyCode::Esc), &cancel);
        assert!(app.asking.is_none());
        assert_eq!(app.steps, 30);
    }

    #[test]
    fn the_size_is_picked_off_a_list_rather_than_typed() {
        // A handful someone chose in advance is a list, not a range.
        let cancel = AtomicBool::new(false);
        let mut app = ready();

        focus_on(&mut app, Field::Size);
        app.key(press(KeyCode::Enter), &cancel);
        assert!(matches!(app.asking, Some(Asking::Size(_))));

        let drawn = screen(&app, 90, 28);
        assert!(drawn.contains("832 x 1216"), "{drawn}");

        app.key(press(KeyCode::Down), &cancel);
        app.key(press(KeyCode::Enter), &cancel);
        assert!(app.asking.is_none());
        assert_eq!(app.size, DEFAULT_SIZE + 1);
    }

    #[test]
    fn the_seed_takes_digits_and_nothing_else() {
        let cancel = AtomicBool::new(false);
        let mut app = ready();
        focus_on(&mut app, Field::Seed);

        type_in(&mut app, "12a3", &cancel);
        assert_eq!(app.seed.text(), "123");
        assert_eq!(app.options().seed, Some(123));

        for _ in 0..3 {
            app.key(press(KeyCode::Backspace), &cancel);
        }
        assert_eq!(app.options().seed, None, "an empty box means any seed");
    }

    #[test]
    fn enter_on_the_draw_from_box_opens_the_picker_rather_than_drawing() {
        let cancel = AtomicBool::new(false);
        let mut app = ready();
        type_in(&mut app, "a cat", &cancel);

        // The box says what enter does there, and only while the cursor is in it.
        assert!(!screen(&app, 90, 26).contains("enter to browse"));
        focus_on(&mut app, Field::From);
        assert!(screen(&app, 90, 26).contains("enter to browse"));

        app.key(press(KeyCode::Enter), &cancel);
        assert!(app.browsing.is_some(), "the picker did not open");
        assert!(app.pending.is_none(), "it started a run instead");

        // And what is on screen is the picker, over everything that was there.
        let drawn = screen(&app, 90, 26);
        assert!(drawn.contains("a picture to draw from"), "{drawn}");

        // Esc takes it down again, leaving everything behind it as it was.
        app.key(press(KeyCode::Esc), &cancel);
        assert!(app.browsing.is_none(), "esc did not close the picker");
        assert!(!app.quit, "esc closed the program rather than the picker");
        assert_eq!(app.prompt.text(), "a cat");
    }

    #[test]
    fn the_picker_opens_somewhere_that_is_there_whatever_is_in_the_box() {
        let cancel = AtomicBool::new(false);
        let mut app = ready();

        // A typo in the box would otherwise put up a list saying only that it cannot be read.
        focus_on(&mut app, Field::From);
        type_in(&mut app, "/no/such/place/at/all.png", &cancel);
        app.key(press(KeyCode::Enter), &cancel);

        let here = std::env::current_dir().unwrap();
        assert_eq!(app.browsing.as_ref().unwrap().directory(), here);

        // And a path that is there is the one it opens on.
        app.key(press(KeyCode::Esc), &cancel);
        focus_on(&mut app, Field::From);
        for _ in 0..40 {
            app.key(press(KeyCode::Backspace), &cancel);
        }
        type_in(&mut app, here.parent().unwrap().to_str().unwrap(), &cancel);
        app.key(press(KeyCode::Enter), &cancel);
        assert_eq!(
            app.browsing.as_ref().unwrap().directory(),
            here.parent().unwrap()
        );
    }

    #[test]
    fn what_the_picker_hands_back_goes_into_the_box() {
        let cancel = AtomicBool::new(false);
        let mut app = ready();

        // A directory of its own with one picture in it, so that what is picked is known.
        let root = std::env::temp_dir().join("waifu-draw-picker-test");
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join("only.png"), b"x").unwrap();

        focus_on(&mut app, Field::From);
        type_in(&mut app, root.to_str().unwrap(), &cancel);
        app.key(press(KeyCode::Enter), &cancel);

        // Past "..", onto the one picture, and take it.
        app.key(press(KeyCode::Down), &cancel);
        app.key(press(KeyCode::Enter), &cancel);

        assert!(app.browsing.is_none(), "the picker stayed open");
        assert_eq!(app.from(), Some(root.join("only.png")));

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn while_the_picker_is_up_the_keys_behind_it_are_left_alone() {
        let cancel = AtomicBool::new(false);
        let mut app = ready();
        type_in(&mut app, "a cat", &cancel);

        focus_on(&mut app, Field::From);
        app.key(press(KeyCode::Enter), &cancel);

        // Tab would move the cursor and typing would go into a box, were the picker not there.
        let focus = app.focus;
        app.key(press(KeyCode::Tab), &cancel);
        type_in(&mut app, "zzz", &cancel);

        assert_eq!(app.focus, focus, "tab moved the screen behind the picker");
        assert_eq!(app.from.text(), "", "what was typed went into a box");
        assert!(app.browsing.is_some());

        // Ctrl-C still means what it always means.
        app.key(
            KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL),
            &cancel,
        );
        assert!(
            app.quit,
            "ctrl-c did not end the program from inside the picker"
        );
    }

    #[test]
    fn a_picture_off_the_disk_becomes_a_tensor_at_the_size_asked_for() {
        // Through a real file rather than a buffer: the point of this one is that what the image
        // crate decodes, scales and hands back lines up with what from_rgb8 expects, which a
        // tensor built by hand would not check.
        let path = std::env::temp_dir().join("waifu-read-image-test.png");
        let pixels: Vec<u8> = (0..32 * 32)
            .flat_map(|i| [i as u8, 255 - i as u8, 7])
            .collect();
        image::save_buffer(&path, &pixels, 32, 32, image::ColorType::Rgb8).unwrap();

        // A size that is neither the file's nor a whole multiple of it, since the run decides the
        // size and a picture off a camera never happens to be it.
        let image = read_image(&path, 192, 128).unwrap();
        std::fs::remove_file(&path).unwrap();

        assert_eq!(image.shape(), vec![1, 3, 128, 192]);

        // Scaling is allowed to move a value; leaving the range the model reads is not.
        let values = image.to_vec_f32().unwrap();
        assert!(
            values.iter().all(|x| (-1.0..=1.0).contains(x)),
            "a scaled picture left [-1, 1]"
        );

        // The third channel was one value everywhere, so scaling cannot have made it anything
        // else -- which is the check that the planes did not get shuffled on the way through.
        let plane = 128 * 192;
        let blue = 7.0 / 127.5 - 1.0;
        assert!(
            values[2 * plane..].iter().all(|x| (x - blue).abs() < 1e-3),
            "the channels came back in the wrong order"
        );
    }

    #[test]
    fn a_picture_that_is_not_there_is_said_so_rather_than_drawn_over() {
        let missing = std::env::temp_dir().join("waifu-no-such-picture.png");
        assert!(read_image(&missing, 64, 64).is_err());
    }

    #[test]
    fn an_empty_draw_from_box_is_a_run_from_noise() {
        let cancel = AtomicBool::new(false);
        let mut app = ready();
        type_in(&mut app, "a cat", &cancel);
        app.start();

        let job = app.pending.take().unwrap();
        assert!(
            job.from.is_none(),
            "nothing was typed, so there is nothing to draw from"
        );

        // And the strength box says so rather than showing a number that nothing reads.
        assert!(
            screen(&app, 90, 24).contains(" - "),
            "{}",
            screen(&app, 90, 24)
        );
    }

    #[test]
    fn a_path_in_the_draw_from_box_is_handed_over_with_the_job() {
        let cancel = AtomicBool::new(false);
        let mut app = ready();
        type_in(&mut app, "a cat", &cancel);

        focus_on(&mut app, Field::From);
        type_in(&mut app, "  cat.png  ", &cancel);
        set(&mut app, Field::Strength, "0.75", &cancel);
        app.start();

        let job = app.pending.take().unwrap();
        // Trimmed, because a path typed with a space either side is the path.
        assert_eq!(job.from, Some(PathBuf::from("cat.png")));
        assert!((job.options.strength - 0.75).abs() < 1e-6);

        // The number is on the screen now that something reads it.
        assert!(
            screen(&app, 90, 28).contains("0.75"),
            "{}",
            screen(&app, 90, 28)
        );
    }

    #[test]
    fn enter_hands_over_what_the_boxes_say() {
        let cancel = AtomicBool::new(false);
        let mut app = ready();
        type_in(&mut app, "a cat", &cancel);
        generate(&mut app, &cancel);

        let job = app.pending.take().expect("a job to draw");
        assert_eq!(job.prompt, "a cat");
        assert_eq!(job.options.num_steps, 30);
        assert_eq!((job.options.width, job.options.height), (1024, 1024));
        assert!(app.running());
    }

    #[test]
    fn a_second_prompt_while_the_first_is_being_drawn_is_refused() {
        let cancel = AtomicBool::new(false);
        let mut app = ready();
        type_in(&mut app, "a cat", &cancel);

        generate(&mut app, &cancel);
        assert!(app.pending.take().is_some());

        // Pressing it again is a mistake rather than a queue: there is one model, and it is busy.
        app.key(press(KeyCode::Enter), &cancel);
        assert!(app.pending.is_none());
        assert!(app.unhappy);
    }

    #[test]
    fn an_empty_prompt_is_not_sent() {
        let cancel = AtomicBool::new(false);
        let mut app = ready();
        type_in(&mut app, "   ", &cancel);

        generate(&mut app, &cancel);
        assert!(app.pending.is_none());
        assert!(app.unhappy);
        assert!(!app.running());
    }

    #[test]
    fn escape_stops_a_run_but_asks_before_leaving_an_idle_one() {
        let cancel = AtomicBool::new(false);
        let mut app = ready();
        type_in(&mut app, "a cat", &cancel);
        generate(&mut app, &cancel);

        app.key(press(KeyCode::Esc), &cancel);
        assert!(cancel.load(Ordering::Relaxed), "the run was asked to stop");
        assert!(!app.quit, "and the program was not");
        assert!(app.leaving.is_none(), "and it did not ask");

        // Idle, it asks. Escape is the key pressed to get out of everything else on this screen,
        // and leaving costs the minute it takes to read the model back in.
        app.update(Update::Stopped);
        app.key(press(KeyCode::Esc), &cancel);
        assert!(!app.quit, "escape left without asking");
        assert!(app.leaving.is_some());

        let drawn = screen(&app, 90, 30);
        assert!(drawn.contains("leave waifu?"), "{drawn}");
    }

    #[test]
    fn the_question_about_leaving_starts_on_no_and_takes_either_answer() {
        let cancel = AtomicBool::new(false);
        let mut app = ready();

        // No, by escaping out of the question, which is not an answer of yes.
        app.key(press(KeyCode::Esc), &cancel);
        app.key(press(KeyCode::Esc), &cancel);
        assert!(!app.quit);
        assert!(app.leaving.is_none());

        // No, by pressing enter on where it starts. What it is asked before cannot be taken back.
        app.key(press(KeyCode::Esc), &cancel);
        assert!(!app.leaving.as_ref().unwrap().on_yes());
        app.key(press(KeyCode::Enter), &cancel);
        assert!(!app.quit);
        assert!(app.leaving.is_none());

        // Yes, by walking to it.
        app.key(press(KeyCode::Esc), &cancel);
        app.key(press(KeyCode::Left), &cancel);
        app.key(press(KeyCode::Enter), &cancel);
        assert!(app.quit);
        assert!(cancel.load(Ordering::Relaxed), "the painter was let go");
    }

    #[test]
    fn the_letters_answer_the_question_outright() {
        let cancel = AtomicBool::new(false);
        let mut app = ready();

        app.key(press(KeyCode::Esc), &cancel);
        app.key(press(KeyCode::Char('n')), &cancel);
        assert!(!app.quit);
        assert!(app.leaving.is_none());

        app.key(press(KeyCode::Esc), &cancel);
        app.key(press(KeyCode::Char('y')), &cancel);
        assert!(app.quit);
    }

    #[test]
    fn enter_draws_from_the_button_and_nowhere_else() {
        // A run is minutes long and starts from one place. Enter in the box a prompt is being
        // typed into is a key pressed on the way to something else as often as it is one meant.
        let cancel = AtomicBool::new(false);
        let mut app = ready();
        type_in(&mut app, "a cat", &cancel);

        for field in [Field::Prompt, Field::Negative, Field::Seed] {
            focus_on(&mut app, field);
            app.key(press(KeyCode::Enter), &cancel);
            assert!(app.pending.is_none(), "enter on {field:?} started a run");
            assert!(!app.running());
        }

        generate(&mut app, &cancel);
        assert!(app.pending.is_some());
    }

    #[test]
    fn control_c_quits_whatever_is_happening() {
        let cancel = AtomicBool::new(false);
        let mut app = ready();
        type_in(&mut app, "a cat", &cancel);
        app.key(press(KeyCode::Enter), &cancel);

        app.key(
            KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL),
            &cancel,
        );
        assert!(app.quit);
        assert!(cancel.load(Ordering::Relaxed));
    }

    #[test]
    fn a_finished_run_is_listed_and_a_failed_one_leaves_the_reason() {
        let mut app = ready();
        app.update(Update::Done {
            kept: Kept {
                path: PathBuf::from("waifu-0001.png"),
                width: 1024,
                height: 1024,
            },
            elapsed: Duration::from_secs(2),
        });
        assert!(!app.unhappy);
        assert!(app.message.contains("waifu-0001.png"), "{}", app.message);
        assert_eq!(app.written.len(), 1);
        assert!(
            app.written[0].contains("1024 by 1024"),
            "{}",
            app.written[0]
        );

        app.update(Update::Failed("out of memory".to_string()));
        assert!(app.unhappy);
        assert_eq!(app.message, "out of memory");
        assert_eq!(app.written.len(), 1, "what was drawn is still on the list");
    }

    #[test]
    fn the_newest_picture_is_at_the_top_of_the_list() {
        let mut app = ready();
        for number in 1..=3 {
            app.update(Update::Done {
                kept: Kept {
                    path: PathBuf::from(format!("waifu-{number:04}.png")),
                    width: 512,
                    height: 512,
                },
                elapsed: Duration::from_secs(1),
            });
        }

        assert_eq!(app.written.len(), 3);
        let names: Vec<&str> = app
            .written
            .iter()
            .map(|line| line.split_whitespace().next().unwrap())
            .collect();
        assert_eq!(
            names,
            ["waifu-0003.png", "waifu-0002.png", "waifu-0001.png"]
        );
    }

    #[test]
    fn the_bar_leaves_room_for_the_parts_that_are_not_steps() {
        let steps = 10;
        let start = fraction(GenerationProgress::Encoding, steps);
        let last = fraction(
            GenerationProgress::Step {
                done: steps,
                total: steps,
            },
            steps,
        );
        let decoding = fraction(GenerationProgress::Decoding, steps);

        assert_eq!(start, 0.0);
        assert!(last < 1.0, "the last step is not the end of the run");
        assert_eq!(last, decoding, "which is where the decode picks up");
        assert!(decoding > 0.75, "but it is nearly the end: {decoding}");

        // Halfway through the steps is somewhere near halfway along the bar.
        let middle = fraction(
            GenerationProgress::Step {
                done: 5,
                total: steps,
            },
            steps,
        );
        assert!((0.35..0.55).contains(&middle), "{middle}");
    }

    /// The whole screen, drawn into a buffer, as one long string of what it says.
    fn screen(app: &App, width: u16, height: u16) -> String {
        let mut terminal = ratatui::Terminal::new(TestBackend::new(width, height)).unwrap();
        terminal.draw(|frame| app.render(frame)).unwrap();

        terminal
            .backend()
            .buffer()
            .content()
            .iter()
            .map(|cell| cell.symbol())
            .collect()
    }

    #[test]
    fn the_screen_says_what_built_it() {
        let drawn = screen(&ready(), 80, 30);

        // Both halves, because either alone is useless: the name without the revision does not
        // say which code, and a bare hash on a screenshot does not say what it is a hash of.
        assert!(drawn.contains("libwaifu"), "{drawn}");
        assert!(drawn.contains(crate::cli::REVISION), "{drawn}");
    }

    #[test]
    fn the_screen_says_what_it_was_pointed_at() {
        let drawn = screen(&ready(), 80, 30);

        // What made the picture is two questions, not one: which code, and which model on which
        // device. A screenshot that answers only the first still cannot be reproduced from.
        assert!(drawn.contains("sdxl:wai"), "{drawn}");
        assert!(drawn.contains("cpu"), "{drawn}");
    }

    #[test]
    fn every_part_of_the_screen_is_on_it() {
        let cancel = AtomicBool::new(false);
        let mut app = ready();
        type_in(&mut app, "a cat", &cancel);
        app.update(Update::Done {
            kept: Kept {
                path: PathBuf::from("waifu-0007.png"),
                width: 1024,
                height: 1024,
            },
            elapsed: Duration::from_secs(42),
        });

        // One row taller than it used to need, which is the button's. A shorter terminal still
        // draws -- the list of pictures is what gives the row up -- but this is the census, and a
        // census wants a screen with room for everything on it.
        let screen = screen(&app, 90, 28);
        for wanted in [
            "prompt",
            "a cat",
            "away from",
            "steps",
            "30",
            "guidance",
            "5.0",
            "size",
            "1024x1024",
            "seed",
            "any",
            "draw from",
            "strength",
            "generate",
            "pictures written",
            "waifu-0007.png",
            "tab or arrows move",
        ] {
            assert!(
                screen.contains(wanted),
                "the screen does not say {wanted:?}"
            );
        }
    }

    #[test]
    fn the_two_prompt_boxes_are_the_same_height() {
        let drawn = screen(&ready(), 80, 30);

        // Counted off the screen rather than off the constant, so that a layout that stops giving
        // them the same run of rows is caught here rather than looked at.
        let lines: Vec<String> = (0..30)
            .map(|row| drawn.chars().skip(row * 80).take(80).collect())
            .collect();
        let top = |needle: &str| lines.iter().position(|line| line.contains(needle)).unwrap();
        let bottom = |from: usize| {
            lines[from + 1..]
                .iter()
                .position(|line| line.starts_with('┗') || line.starts_with('╰'))
                .unwrap()
        };

        let prompt = top(" prompt ");
        let negative = top(" away from ");
        assert_eq!(bottom(prompt), bottom(negative), "{drawn}");
    }

    #[test]
    fn what_it_is_doing_takes_one_row_and_still_says_both_things() {
        // Idle: the last thing that happened, where the bar would be.
        let mut app = ready();
        app.said("written to a.png");
        let drawn = screen(&app, 80, 30);
        assert!(drawn.contains("written to a.png"), "{drawn}");

        // Running: the bar, with where it has got to on it.
        let cancel = AtomicBool::new(false);
        type_in(&mut app, "a cat", &cancel);
        generate(&mut app, &cancel);
        let drawn = screen(&app, 80, 30);
        assert!(drawn.contains("reading the prompt"), "{drawn}");

        // Running and turned down: the refusal takes the label, because saying nothing would read
        // as the key not having worked.
        app.key(press(KeyCode::Enter), &cancel);
        let drawn = screen(&app, 80, 30);
        assert!(drawn.contains("there is already a picture"), "{drawn}");

        // And it gives the label back as soon as the run moves.
        app.update(Update::Progress(GenerationProgress::Step {
            done: 1,
            total: 30,
        }));
        let drawn = screen(&app, 80, 30);
        assert!(drawn.contains("step 1 of 30"), "{drawn}");
    }

    #[test]
    fn a_screen_with_no_room_on_it_still_draws() {
        // Every box is laid out from what is left after the one above it, so a terminal too
        // small for them is the case where a width or a height comes out zero.
        for (width, height) in [(1, 1), (10, 3), (40, 8), (200, 60)] {
            screen(&ready(), width, height);
        }
    }

    #[test]
    fn what_it_is_doing_is_said_in_words() {
        assert_eq!(doing(GenerationProgress::Encoding), "reading the prompt");
        assert_eq!(
            doing(GenerationProgress::Step { done: 3, total: 30 }),
            "step 3 of 30"
        );
    }
}
