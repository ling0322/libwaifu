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
use crate::cli::field::TextField;
use crate::cli::hub;
use crate::cli::picker;
use crate::cli::png;
use crate::flint::Tensor;
use crate::{to_rgb8, Device, GenerationOptions, GenerationProgress, Sdxl, ZipFile};

type Error = Box<dyn std::error::Error>;

/// The sizes on offer, which are the ones SDXL was trained at. Every one of them is a multiple of
/// the 32 pixels the U-Net's own halvings need.
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

/// Says what was wrong before printing the usage, which is the order the Go tool prints them in.
fn with_usage<T, E: std::fmt::Display>(result: Result<T, E>) -> Result<T, E> {
    if let Err(error) = &result {
        eprintln!("{error}\n");
        print_usage();
    }
    result
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

    let model = with_usage(args.model())?.map(str::to_string);
    let device = with_usage(args.device())?.resolve();

    // Either a path, or a name like "sdxl:base" that is fetched into the cache first. Done here,
    // before the screen is set up, because fetching a model prints as it goes and the terminal is
    // still the terminal at this point. Without a name, the screen goes up early and offers the
    // published models instead -- a fetch there is minutes long and wants a progress bar rather
    // than a scrolling line.
    let model_path = match model {
        Some(model) => hub::resolve(&model)?,
        None => {
            let mut terminal = ratatui::init();
            let chosen = picker::choose(&mut terminal);
            ratatui::restore();

            match chosen? {
                Some(path) => path,
                None => return Ok(()),
            }
        }
    };

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

    let terminal = ratatui::init();
    let outcome = run(terminal, &jobs, &arriving, &cancel);
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
    jobs: &Sender<Job>,
    updates: &Receiver<Update>,
    cancel: &AtomicBool,
) -> Result<(), Error> {
    let mut app = App::new();

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

        let update = match model.generate_reporting(&job.prompt, &job.options, &mut report) {
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

        std::fs::write(&path, png::encode(width as u32, height as u32, &pixels))?;
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
    Steps,
    Guidance,
    Size,
    Seed,
}

const FIELDS: [Field; 6] = [
    Field::Prompt,
    Field::Negative,
    Field::Steps,
    Field::Guidance,
    Field::Size,
    Field::Seed,
];

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
    /// Empty means a different picture every time; a number means the same one every time.
    seed: TextField,
    steps: i32,
    guidance: f32,
    size: usize,
    focus: usize,

    status: Status,
    /// The last thing worth saying, under the bar.
    message: String,
    /// Whether that thing was bad news.
    unhappy: bool,
    /// What has been written out this session, newest first.
    written: Vec<String>,
    /// The job the loop is about to hand to the painter.
    pending: Option<Job>,
    quit: bool,
}

impl App {
    fn new() -> App {
        App {
            prompt: TextField::default(),
            negative: TextField::default(),
            seed: TextField::default(),
            steps: 30,
            guidance: 5.0,
            size: DEFAULT_SIZE,
            focus: 0,
            status: Status::Ready,
            message: "type a prompt and press enter".to_string(),
            unhappy: false,
            written: Vec::new(),
            pending: None,
            quit: false,
        }
    }

    fn focused(&self) -> Field {
        FIELDS[self.focus]
    }

    fn running(&self) -> bool {
        matches!(self.status, Status::Running { .. })
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
                    self.quit = true;
                }
            }
            KeyCode::Enter => self.start(),
            KeyCode::Tab | KeyCode::Down => self.focus = (self.focus + 1) % FIELDS.len(),
            KeyCode::BackTab | KeyCode::Up => {
                self.focus = (self.focus + FIELDS.len() - 1) % FIELDS.len();
            }
            code => self.edit(code),
        }
    }

    /// The keys that mean something to the box the cursor is in.
    fn edit(&mut self, code: KeyCode) {
        match self.focused() {
            Field::Prompt => edit_text(&mut self.prompt, code, |_| true),
            Field::Negative => edit_text(&mut self.negative, code, |_| true),
            Field::Seed => edit_text(&mut self.seed, code, char::is_ascii_digit),
            Field::Steps => match code {
                KeyCode::Left => self.steps = (self.steps - 1).max(1),
                KeyCode::Right => self.steps = (self.steps + 1).min(150),
                _ => {}
            },
            Field::Guidance => match code {
                KeyCode::Left => self.guidance = (self.guidance - 0.5).max(1.0),
                KeyCode::Right => self.guidance = (self.guidance + 0.5).min(20.0),
                _ => {}
            },
            Field::Size => match code {
                KeyCode::Left => self.size = self.size.saturating_sub(1),
                KeyCode::Right => self.size = (self.size + 1).min(SIZES.len() - 1),
                _ => {}
            },
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
        }
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
        self.pending = Some(Job { prompt, options });
    }

    // -- the screen -----------------------------------------------------------------------

    fn render(&self, frame: &mut Frame) {
        let [prompt, negative, numbers, status, written, keys] = Layout::vertical([
            Constraint::Length(5),
            Constraint::Length(4),
            Constraint::Length(3),
            Constraint::Length(4),
            Constraint::Min(0),
            Constraint::Length(1),
        ])
        .areas(frame.area());

        let quarters = [Constraint::Percentage(25); 4];
        let [steps, guidance, size, seed] = Layout::horizontal(quarters).areas(numbers);

        self.render_text(frame, prompt, Field::Prompt, "prompt", &self.prompt);
        self.render_text(
            frame,
            negative,
            Field::Negative,
            "away from",
            &self.negative,
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

        self.render_status(frame, status);
        self.render_written(frame, written);
        frame.render_widget(
            Paragraph::new(Line::from(vec![
                Span::raw(" tab").bold(),
                Span::raw(" move  "),
                Span::raw("enter").bold(),
                Span::raw(" draw  "),
                Span::raw("esc").bold(),
                Span::raw(" stop or quit"),
            ]))
            .style(Style::new().fg(Color::DarkGray)),
            keys,
        );
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

        // The arrows are only worth drawing on the box they would move.
        let line = if self.focused() == field {
            Line::from(vec![
                Span::styled("< ", Style::new().fg(Color::Yellow)),
                Span::raw(value).bold(),
                Span::styled(" >", Style::new().fg(Color::Yellow)),
            ])
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

        let [bar, message] =
            Layout::vertical([Constraint::Length(1), Constraint::Min(0)]).areas(inner);

        match &self.status {
            Status::Ready => frame.render_widget(
                Paragraph::new(Line::raw("waiting for a prompt").fg(Color::DarkGray)),
                bar,
            ),
            Status::Running {
                progress,
                steps,
                started,
            } => {
                frame.render_widget(
                    Gauge::default()
                        .ratio(fraction(*progress, *steps))
                        .label(format!(
                            "{} ({:.0}s)",
                            doing(*progress),
                            started.elapsed().as_secs_f64()
                        ))
                        .gauge_style(Style::new().fg(Color::Magenta)),
                    bar,
                );
            }
        }

        let colour = if self.unhappy {
            Color::Red
        } else {
            Color::Gray
        };
        frame.render_widget(
            Paragraph::new(self.message.clone())
                .style(Style::new().fg(colour))
                .wrap(Wrap { trim: true }),
            message,
        );
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

/// The keys that edit a box of text, taking only the characters `accepts` allows.
fn edit_text(field: &mut TextField, code: KeyCode, accepts: fn(&char) -> bool) {
    match code {
        KeyCode::Char(character) if accepts(&character) => field.insert(character),
        KeyCode::Backspace => field.backspace(),
        KeyCode::Delete => field.delete(),
        KeyCode::Left => field.left(),
        KeyCode::Right => field.right(),
        KeyCode::Home => field.home(),
        KeyCode::End => field.end(),
        _ => {}
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
        App::new()
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

    #[test]
    fn tab_goes_round_and_back_again() {
        let cancel = AtomicBool::new(false);
        let mut app = ready();

        for _ in 0..FIELDS.len() {
            app.key(press(KeyCode::Tab), &cancel);
        }
        assert_eq!(app.focused(), Field::Prompt, "all the way round");

        app.key(press(KeyCode::BackTab), &cancel);
        assert_eq!(app.focused(), Field::Seed, "and one back off the end");
    }

    #[test]
    fn the_arrows_turn_the_knobs_and_stop_at_their_ends() {
        let cancel = AtomicBool::new(false);
        let mut app = ready();
        app.focus = 2;
        assert_eq!(app.focused(), Field::Steps);

        app.key(press(KeyCode::Right), &cancel);
        assert_eq!(app.steps, 31);
        for _ in 0..40 {
            app.key(press(KeyCode::Left), &cancel);
        }
        assert_eq!(app.steps, 1, "a run of no steps is not a run");

        app.focus = 3;
        app.key(press(KeyCode::Left), &cancel);
        assert_eq!(app.guidance, 4.5);

        app.focus = 4;
        for _ in 0..SIZES.len() * 2 {
            app.key(press(KeyCode::Right), &cancel);
        }
        assert_eq!(app.size, SIZES.len() - 1);
    }

    #[test]
    fn the_seed_takes_digits_and_nothing_else() {
        let cancel = AtomicBool::new(false);
        let mut app = ready();
        app.focus = 5;

        type_in(&mut app, "12a3", &cancel);
        assert_eq!(app.seed.text(), "123");
        assert_eq!(app.options().seed, Some(123));

        for _ in 0..3 {
            app.key(press(KeyCode::Backspace), &cancel);
        }
        assert_eq!(app.options().seed, None, "an empty box means any seed");
    }

    #[test]
    fn enter_hands_over_what_the_boxes_say() {
        let cancel = AtomicBool::new(false);
        let mut app = ready();
        type_in(&mut app, "a cat", &cancel);
        app.key(press(KeyCode::Enter), &cancel);

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

        app.key(press(KeyCode::Enter), &cancel);
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

        app.key(press(KeyCode::Enter), &cancel);
        assert!(app.pending.is_none());
        assert!(app.unhappy);
        assert!(!app.running());
    }

    #[test]
    fn escape_stops_a_run_but_quits_an_idle_one() {
        let cancel = AtomicBool::new(false);
        let mut app = ready();
        type_in(&mut app, "a cat", &cancel);
        app.key(press(KeyCode::Enter), &cancel);

        app.key(press(KeyCode::Esc), &cancel);
        assert!(cancel.load(Ordering::Relaxed), "the run was asked to stop");
        assert!(!app.quit, "and the program was not");

        app.update(Update::Stopped);
        app.key(press(KeyCode::Esc), &cancel);
        assert!(app.quit);
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

        let screen = screen(&app, 90, 24);
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
            "pictures written",
            "waifu-0007.png",
            "tab move",
        ] {
            assert!(
                screen.contains(wanted),
                "the screen does not say {wanted:?}"
            );
        }
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
