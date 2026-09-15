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

//! Choosing a model when the command line did not name one.
//!
//! `waifu draw` used to insist on `-m`, which is a poor thing to insist on the first time someone
//! runs it: the answer is a name they have not read yet, for a file they do not have. This is that
//! list, with what is already on disk marked, and the fetch that follows if what they picked is
//! not.
//!
//! The fetch is minutes long, so it runs on a thread of its own and says how it is getting on down
//! a channel, the same shape the drawing screen uses for a run. The screen stays live while it
//! happens: a download that cannot be watched is indistinguishable from one that has hung.

use std::path::PathBuf;
use std::sync::mpsc::{channel, Receiver, TryRecvError};
use std::time::{Duration, Instant};

use ratatui::crossterm::event::{self, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Color, Style, Stylize};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, BorderType, Paragraph};
use ratatui::{DefaultTerminal, Frame};

use crate::cli::args::Runtime;
use crate::cli::ask::{Answer, Confirm};
use crate::cli::files::{self, FilePicker};
use crate::cli::{bar, built_from, hub};

type Error = Box<dyn std::error::Error>;

/// How often the screen looks for a key or for news of the fetch.
const TICK: Duration = Duration::from_millis(100);

/// How far apart two byte counts have to be before they are worth a rate. Shorter than this and
/// the answer is mostly the buffer size divided by how long one read happened to take.
const SPEED_WINDOW: Duration = Duration::from_millis(500);

/// What a model's manifest is called, which is what the list offers to look for.
///
/// The packages beside it are not offered: a model is its manifest, and the weights are files it
/// names rather than files anybody picks.
const MANIFEST_SUFFIX_NAME: &str = "yaml";

/// One row of the list.
enum Entry {
    /// A model this build knows how to fetch.
    Published {
        name: &'static str,
        /// Human-readable name shown beside the short name.
        full_name: &'static str,
        /// Whether every package of it is already in the cache.
        cached: bool,
        /// What is on disk for it, which is most of a model for one that was interrupted.
        bytes: u64,
    },
    /// Not a model but a way out of the list: a manifest already somewhere on the disk, which is
    /// where anything exported here rather than published lives.
    OnDisk,
}

impl Entry {
    fn name(&self) -> &str {
        match self {
            Entry::Published { name, .. } => name,
            Entry::OnDisk => "a file...",
        }
    }

    fn describe(&self) -> String {
        match self {
            Entry::Published {
                cached: true,
                bytes,
                ..
            } => format!("on disk, {}", gigabytes(*bytes)),
            Entry::Published { bytes, .. } if *bytes > 0 => {
                format!("part fetched, {}", gigabytes(*bytes))
            }
            Entry::Published { .. } => "not fetched".to_string(),
            Entry::OnDisk => format!("a .{MANIFEST_SUFFIX_NAME} you exported yourself"),
        }
    }

    /// Whether the row reads as ready to use, which is what lights it up.
    fn ready(&self) -> bool {
        match self {
            Entry::Published { cached, .. } => *cached,
            // Nothing to fetch, so nothing to be waiting for.
            Entry::OnDisk => true,
        }
    }
}

/// What a manifest picked off the disk becomes.
///
/// It is called by the file it is in rather than by a name from the list. The whole path is too
/// long for the heading the drawing screen puts it in and its tail is the part that identifies it,
/// which is the same call the command line makes for a `-m` that names a path.
///
/// There used to be a check here for a later part of a split model, because a model was its
/// packages and four of them sat in a directory with nothing in the names to say which one to
/// open. A model is its manifest now, and there is only ever one of those.
fn from_disk(path: PathBuf, runtime: Runtime) -> Result<Chosen, String> {
    Ok(Chosen {
        name: path
            .file_stem()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string(),
        path,
        runtime,
    })
}

/// A place the run could go, and whether this build can go there.
struct Choice {
    runtime: Runtime,
    /// Asked once, at the start. The answer cannot change while the screen is up, and asking is a
    /// call across the C boundary that a redraw has no business making.
    available: bool,
}

/// Which box of the screen the keys are talking to.
#[derive(Clone, Copy, Eq, PartialEq)]
enum Pane {
    Models,
    Devices,
}

/// Everything the screen shows that is not the fetch.
struct Choices {
    entries: Vec<Entry>,
    selected: usize,
    devices: Vec<Choice>,
    device: usize,
    pane: Pane,
}

impl Choices {
    fn chosen_device(&self) -> &Choice {
        &self.devices[self.device]
    }
}

/// What the fetching thread has to say.
enum Word {
    /// Which hub the packages are coming from, said once the fetch has settled it.
    From(&'static str),
    /// A file, how far into it, how long it is when that is known, and which package of how many
    /// it is -- `parts` being zero until the first package has named the others.
    Fetching {
        file: String,
        done: u64,
        total: Option<u64>,
        part: usize,
        parts: usize,
    },
    /// Every package is in the cache; the model is at this path.
    Done(std::path::PathBuf),
    Failed(String),
}

/// What came out of the channel this pass, once the whole backlog has been read.
enum Outcome {
    Nothing,
    Done(std::path::PathBuf),
    Failed(String),
}

/// What the screen is doing.
enum Doing {
    Choosing,
    /// Fetching, with the last word from the thread and the channel it comes down.
    Fetching {
        news: Receiver<Word>,
        /// Which hub is being asked, once the fetch has said. Unknown for the moment before that:
        /// it is worked out on the far side of a thread, and a screen that guessed at it would be
        /// naming a hub while the probe that decides is still out.
        from: Option<&'static str>,
        file: String,
        done: u64,
        total: Option<u64>,
        part: usize,
        parts: usize,
        /// Bytes a second, smoothed, once two samples far enough apart have been seen. What makes
        /// a bar that has not moved in a second still look alive.
        speed: Option<f64>,
        /// The moment and the byte count the last speed was worked out from.
        sample: (Instant, u64),
    },
}

/// A model and somewhere to run it, chosen.
#[derive(Debug)]
pub struct Chosen {
    pub path: std::path::PathBuf,
    /// What it was called in the list, which is shorter and steadier than its path and is what
    /// the drawing screen shows. Owned rather than static, since a file off the disk is called
    /// whatever it is called.
    pub name: String,
    pub runtime: Runtime,
}

/// Run the picker until a model is chosen and on disk, or the user leaves.
///
/// Returns where the chosen model is, or `None` if they quit. The terminal is the caller's: this
/// borrows it and gives it back as it found it.
pub fn choose(terminal: &mut DefaultTerminal, runtime: Runtime) -> Result<Option<Chosen>, Error> {
    // Versioned names are left out. Someone who wants `sdxl:base:v1` in particular can type it,
    // and a list is for someone who does not yet know what to type.
    let mut entries: Vec<Entry> = hub::names()
        .into_iter()
        .filter(|name| name.matches(':').count() == 1)
        .map(|name| Entry::Published {
            name,
            full_name: hub::full_name(name).unwrap_or(""),
            cached: hub::is_cached(name),
            bytes: hub::cached_bytes(name),
        })
        .collect();
    if entries.is_empty() {
        return Err("this build knows no models to offer".into());
    }

    // Last, under the published ones: it is the answer for someone who already has a package, and
    // the list above is for someone who does not yet know what to ask for.
    entries.push(Entry::OnDisk);

    let devices: Vec<Choice> = Runtime::ALL
        .into_iter()
        .map(|runtime| Choice {
            available: runtime.is_available(),
            runtime,
        })
        .collect();

    // Something already on disk is the one most likely to be wanted, so start there.
    let mut choices = Choices {
        selected: entries
            .iter()
            .position(|entry| matches!(entry, Entry::Published { cached: true, .. }))
            .unwrap_or(0),
        entries,
        // Whatever the command line resolved to, so that the box says what would have happened
        // rather than making someone who passed -device set it a second time.
        device: devices
            .iter()
            .position(|choice| choice.runtime == runtime)
            .unwrap_or(0),
        devices,
        pane: Pane::Models,
    };
    let mut doing = Doing::Choosing;
    let mut failure: Option<String> = None;
    let mut browsing: Option<FilePicker> = None;
    // Which model the question on screen is about, kept beside the question: the cursor can only
    // be answered where it was asked, and a name is what the answer needs rather than a row that
    // may have been walked away from.
    let mut deleting: Option<(&'static str, Confirm)> = None;

    loop {
        terminal.draw(|frame| {
            draw(frame, &choices, &doing, failure.as_deref());
            if let Some(browsing) = &browsing {
                browsing.render(frame, frame.area());
            }
            if let Some((_, asking)) = &deleting {
                asking.render(frame, frame.area());
            }
        })?;

        let mut outcome = Outcome::Nothing;
        if let Doing::Fetching {
            news,
            from,
            file,
            done,
            total,
            part,
            parts,
            speed,
            sample,
        } = &mut doing
        {
            // The whole backlog, not one word a tick. The thread speaks several times a second
            // and this loop turns over ten, so taking one at a time would show a number that
            // falls further behind the download the longer it runs.
            loop {
                match news.try_recv() {
                    Ok(Word::From(hub)) => *from = Some(hub),
                    Ok(Word::Fetching {
                        file: name,
                        done: at,
                        total: length,
                        part: which,
                        parts: many,
                    }) => {
                        if *file != name {
                            *file = name;
                            *speed = None;
                            *sample = (Instant::now(), at);
                        }
                        *done = at;
                        *total = length;
                        *part = which;
                        *parts = many;

                        // Over half a second rather than between two words, which are a tenth of
                        // a second apart and would make a rate that jumps around with the buffer.
                        let waited = sample.0.elapsed();
                        if waited >= SPEED_WINDOW && at > sample.1 {
                            let rate = (at - sample.1) as f64 / waited.as_secs_f64();
                            *speed = Some(match *speed {
                                Some(before) => before * 0.6 + rate * 0.4,
                                None => rate,
                            });
                            *sample = (Instant::now(), at);
                        }
                    }
                    Ok(Word::Done(path)) => {
                        outcome = Outcome::Done(path);
                        break;
                    }
                    Ok(Word::Failed(message)) => {
                        outcome = Outcome::Failed(message);
                        break;
                    }
                    Err(TryRecvError::Empty) => break,
                    Err(TryRecvError::Disconnected) => {
                        outcome = Outcome::Failed("the fetch stopped without saying why".into());
                        break;
                    }
                }
            }
        }

        match outcome {
            Outcome::Nothing => {}
            // The device and residency the fetch was started with, which are still the ones
            // showing: the keys do nothing but quit while one is running.
            Outcome::Done(path) => {
                return Ok(Some(Chosen {
                    path,
                    name: choices.entries[choices.selected].name().to_string(),
                    runtime: choices.chosen_device().runtime,
                }))
            }
            Outcome::Failed(message) => {
                failure = Some(format!("could not fetch: {message}"));
                doing = Doing::Choosing;
                read_the_cache_again(&mut choices.entries);
            }
        }

        if !event::poll(TICK)? {
            continue;
        }
        let Event::Key(key) = event::read()? else {
            continue;
        };
        if key.kind != KeyEventKind::Press {
            continue;
        }

        // A question on screen has the keys before anything behind it, the file picker included:
        // it is the thing that was just asked for, and it is answered before anything else is.
        if let Some((name, asking)) = &mut deleting {
            if key.code == KeyCode::Char('c') && key.modifiers.contains(KeyModifiers::CONTROL) {
                return Ok(None);
            }

            match asking.key(key) {
                Answer::Open => {}
                Answer::Cancelled | Answer::Given(false) => deleting = None,
                Answer::Given(true) => {
                    let name = *name;
                    deleting = None;
                    match hub::remove(name) {
                        // Read back off the disk rather than assumed: what the row says about a
                        // model is what is there, and this is the same call that first said it.
                        Ok(()) => read_the_cache_again(&mut choices.entries),
                        Err(gone_wrong) => {
                            failure = Some(format!("could not delete: {gone_wrong}"))
                        }
                    }
                }
            }
            continue;
        }

        // While the file picker is up it has the keys, except the one that always ends the
        // program. Nothing behind it is running -- a fetch and the picker cannot both be on --
        // so there is nothing else for a key to reach.
        if let Some(open) = &mut browsing {
            if key.code == KeyCode::Char('c') && key.modifiers.contains(KeyModifiers::CONTROL) {
                return Ok(None);
            }

            match open.key(key) {
                files::Outcome::Open => {}
                files::Outcome::Cancelled => browsing = None,
                files::Outcome::Picked(path) => {
                    browsing = None;
                    match from_disk(path, choices.chosen_device().runtime) {
                        Ok(chosen) => return Ok(Some(chosen)),
                        Err(message) => failure = Some(message),
                    }
                }
            }
            continue;
        }

        // A fetch is not interruptible here: the thread is inside a read that this cannot reach
        // into, and leaving would abandon a `.part` that the next run resumes anyway. So while one
        // is running the only key that does anything is the one that quits the program.
        if let Doing::Fetching { .. } = doing {
            if quits(&key) {
                return Ok(None);
            }
            continue;
        }

        match key.code {
            KeyCode::Up | KeyCode::Char('k') => {
                match choices.pane {
                    Pane::Models => choices.selected = choices.selected.saturating_sub(1),
                    Pane::Devices => choices.device = choices.device.saturating_sub(1),
                }
                failure = None;
            }
            KeyCode::Down | KeyCode::Char('j') => {
                match choices.pane {
                    Pane::Models => {
                        choices.selected = (choices.selected + 1).min(choices.entries.len() - 1);
                    }
                    Pane::Devices => {
                        choices.device = (choices.device + 1).min(choices.devices.len() - 1);
                    }
                }
                failure = None;
            }
            // Delete, or the letter for anyone whose hands are on the vim keys. Nothing is thrown
            // away on the press: what it opens is the question, and the question starts on no.
            KeyCode::Delete | KeyCode::Char('d') if choices.pane == Pane::Models => {
                failure = None;
                match &choices.entries[choices.selected] {
                    // Nothing fetched is nothing to delete, and a question about deleting it
                    // would be a question with no answer worth giving.
                    Entry::Published { bytes: 0, .. } => {
                        failure = Some("nothing of that one is on disk yet".to_string());
                    }
                    Entry::Published { name, bytes, .. } => {
                        deleting = Some((
                            name,
                            Confirm::ask(&format!("delete {name}? {} on disk", gigabytes(*bytes))),
                        ));
                    }
                    // A package of someone's own, in a directory of their choosing. The list did
                    // not put it there and has no business taking it away.
                    Entry::OnDisk => {
                        failure =
                            Some("that row is a file of your own, not a fetched model".into());
                    }
                }
            }
            // Two boxes, side by side, and three keys that move between them: the two arrows
            // that point at them and the tab that goes round.
            KeyCode::Left | KeyCode::Char('h') => choices.pane = Pane::Models,
            KeyCode::Right | KeyCode::Char('l') => choices.pane = Pane::Devices,
            KeyCode::Tab | KeyCode::BackTab => {
                choices.pane = match choices.pane {
                    Pane::Models => Pane::Devices,
                    Pane::Devices => Pane::Models,
                };
            }
            KeyCode::Enter => {
                // Said here rather than by refusing to put the cursor on it. A device this build
                // has no operators for is worth listing -- it is how someone finds out the build
                // is a CPU one -- and a cursor that skips a row without saying why is worse than
                // a line of text that says it.
                let chosen = choices.chosen_device();
                if !chosen.available {
                    failure = Some(format!(
                        "{} is not available in this build",
                        chosen.runtime.name()
                    ));
                    continue;
                }

                failure = None;

                // The row that is not a model: nothing to fetch, so what opens is the disk.
                let Entry::Published { name, .. } = choices.entries[choices.selected] else {
                    browsing = Some(FilePicker::open(
                        "a model manifest",
                        &std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")),
                        &[MANIFEST_SUFFIX_NAME],
                    ));
                    continue;
                };

                let (say, news) = channel::<Word>();
                std::thread::spawn(move || {
                    let mut report = |progress: hub::Progress| {
                        let word = match progress {
                            hub::Progress::From { hub } => Word::From(hub),
                            hub::Progress::Fetching {
                                file,
                                done,
                                total,
                                part,
                                parts,
                            } => Word::Fetching {
                                file: file.to_string(),
                                done,
                                total,
                                part,
                                parts,
                            },
                            hub::Progress::Fetched {
                                file,
                                bytes,
                                part,
                                parts,
                            } => Word::Fetching {
                                file: file.to_string(),
                                done: bytes,
                                total: Some(bytes),
                                part,
                                parts,
                            },
                        };
                        let _ = say.send(word);
                    };

                    let word = match hub::resolve_reporting(name, &mut report) {
                        Ok(path) => Word::Done(path),
                        Err(error) => Word::Failed(error.to_string()),
                    };
                    let _ = say.send(word);
                });

                doing = Doing::Fetching {
                    news,
                    from: None,
                    file: name.to_string(),
                    done: 0,
                    total: None,
                    part: 1,
                    parts: 0,
                    speed: None,
                    sample: (Instant::now(), 0),
                };
            }
            _ if quits(&key) => return Ok(None),
            _ => {}
        }
    }
}

/// Ask the cache again what it holds, for every row that is a model this build can fetch.
///
/// Called after anything that changes what is on disk -- a fetch that stopped partway, a model
/// deleted -- rather than working the new state out from what just happened: the row says what is
/// there, and the disk is the only thing that knows.
fn read_the_cache_again(entries: &mut [Entry]) {
    for entry in entries {
        if let Entry::Published {
            name,
            cached,
            bytes,
            ..
        } = entry
        {
            *cached = hub::is_cached(name);
            *bytes = hub::cached_bytes(name);
        }
    }
}

fn quits(key: &KeyEvent) -> bool {
    matches!(key.code, KeyCode::Esc | KeyCode::Char('q'))
        || (key.code == KeyCode::Char('c') && key.modifiers.contains(KeyModifiers::CONTROL))
}

fn draw(frame: &mut Frame, choices: &Choices, doing: &Doing, failure: Option<&str>) {
    let [built, told, middle, foot] = Layout::vertical([
        Constraint::Length(1),
        Constraint::Length(1),
        Constraint::Min(3),
        Constraint::Length(3),
    ])
    .areas(frame.area());

    // Wide enough for the longest line of the right column and no wider: the model names and what
    // is on disk are what the eye is looking for, and what is beside them is one switch set once.
    // The box is given exactly its rows rather than the height of the column, so that four short
    // lines do not sit inside a frame reaching to the foot of the screen.
    let [models, settings] =
        Layout::horizontal([Constraint::Min(40), Constraint::Length(40)]).areas(middle);
    let [devices, _] = Layout::vertical([
        Constraint::Length(choices.devices.len() as u16 + 2),
        Constraint::Min(0),
    ])
    .areas(settings);

    // The same line the drawing screen carries, and for the same reason. What used to be here was
    // a bordered box saying what to do, which is two rows of frame around one row of text that
    // the key hints at the foot already cover.
    frame.render_widget(Paragraph::new(built_from()), built);
    frame.render_widget(
        Paragraph::new(
            Line::from(
                " Pick a model and where to run it. The model is fetched the first time it is used.",
            )
            .dim(),
        ),
        told,
    );

    let full_name_width = choices
        .entries
        .iter()
        .map(|e| match e {
            Entry::Published { full_name, .. } => full_name.len(),
            Entry::OnDisk => 0,
        })
        .max()
        .unwrap_or(0);

    let rows: Vec<Line> = choices
        .entries
        .iter()
        .enumerate()
        .map(|(index, entry)| {
            let chosen = index == choices.selected;
            let mut spans = vec![
                Span::raw(if chosen { "> " } else { "  " }),
                Span::styled(
                    format!("{:<14}", entry.name()),
                    row_style(chosen, choices.pane == Pane::Models, entry.ready()),
                ),
            ];
            if let Entry::Published { full_name, .. } = entry {
                spans.push(Span::raw(format!(
                    "  {:<width$}",
                    full_name,
                    width = full_name_width
                )));
            }
            spans.push(Span::raw("  "));
            spans.push(Span::raw(entry.describe()).dim());
            Line::from(spans)
        })
        .collect();
    frame.render_widget(
        Paragraph::new(rows).block(focused(" models ", choices.pane == Pane::Models)),
        models,
    );

    let rows: Vec<Line> = choices
        .devices
        .iter()
        .enumerate()
        .map(|(index, choice)| {
            let chosen = index == choices.device;
            Line::from(vec![
                Span::raw(if chosen { "> " } else { "  " }),
                Span::styled(
                    // As wide as the longest name, which is the offload one. The names are not
                    // the same length and a column that wanders is harder to read down than one
                    // with a gap in it.
                    format!("{:<16}", choice.runtime.name()),
                    row_style(chosen, choices.pane == Pane::Devices, choice.available),
                ),
                Span::raw("  "),
                Span::raw(if choice.available {
                    choice.runtime.about()
                } else {
                    "not in this build"
                })
                .dim(),
            ])
        })
        .collect();
    frame.render_widget(
        Paragraph::new(rows).block(focused(" device ", choices.pane == Pane::Devices)),
        devices,
    );

    let enter = match choices.entries.get(choices.selected) {
        Some(Entry::OnDisk) => "look for one",
        _ => "fetch and use",
    };
    draw_foot(frame, foot, doing, enter, failure);
}

/// How a row reads: marked when it is the one chosen, and only lit up while its half of the screen
/// is the half the keys are talking to, so that two cursors on screen do not both look live.
fn row_style(chosen: bool, focused: bool, ready: bool) -> Style {
    if chosen && focused {
        Style::default().fg(Color::Black).bg(Color::Cyan)
    } else if chosen {
        Style::default().fg(Color::Cyan)
    } else if ready {
        Style::default().fg(Color::Green)
    } else {
        Style::default()
    }
}

/// A box that shows whether the keys are talking to it, the way the drawing screen marks its
/// fields.
fn focused(title: &str, focused: bool) -> Block<'_> {
    let block = bordered(title);
    if focused {
        block
            .border_type(BorderType::Thick)
            .border_style(Style::new().fg(Color::Yellow))
    } else {
        block
    }
}

/// The bar at the foot: the fetch while there is one, and otherwise the keys or what went wrong.
///
/// `enter` is what enter does on the row the cursor is on, which is not the same on all of them,
/// and `failure` is shown as it was given -- whoever set it says what it was, since by the time it
/// reaches here a fetch that failed and a file that cannot be opened look alike.
fn draw_foot(frame: &mut Frame, area: Rect, doing: &Doing, enter: &str, failure: Option<&str>) {
    match doing {
        Doing::Fetching {
            from,
            file,
            done,
            total,
            part,
            parts,
            speed,
            ..
        } => {
            let ratio = match total {
                Some(total) if *total > 0 => (*done as f64 / *total as f64).clamp(0.0, 1.0),
                _ => 0.0,
            };

            // The file name lives in the border rather than in the bar. It is longer than the
            // numbers are and it does not change while one is being fetched, and a label wide
            // enough to cover the bar hides the one thing the bar is there to show.
            let of = if *parts > 0 {
                format!(" part {part} of {parts} ")
            } else {
                String::new()
            };

            // Which hub, and only which hub: the address it is fetched by is the fetch's business
            // and nobody watching a bar move needs it read out to them. It appears the moment the
            // fetch has settled on one, which is a moment after the bar does.
            let hub = match from {
                Some(hub) => format!(" from {hub}"),
                None => String::new(),
            };
            let title = format!(" fetching {file}{hub}{of}");

            let mut label = match total {
                Some(total) if *total > 0 => format!(
                    "{:.0}%  {} of {}",
                    ratio * 100.0,
                    gigabytes(*done),
                    gigabytes(*total)
                ),
                _ => gigabytes(*done),
            };
            if let Some(rate) = speed {
                label.push_str(&format!("   {}", per_second(*rate)));

                // The rate is what says a stalled-looking bar is still moving, and the time left
                // is what says whether to wait for it. Both are guesses from the last few seconds
                // and neither is worth showing to more figures than that supports.
                if let Some(total) = total {
                    label.push_str(&format!(
                        "   {}",
                        remaining(total.saturating_sub(*done), *rate)
                    ));
                }
            }

            // Cyan is what the chosen row in the list is marked with, so the bar and the list
            // read as one screen.
            let outline = bordered(&title);
            let inner = outline.inner(area);
            frame.render_widget(outline, area);
            bar(frame, inner, ratio, &label, Color::Cyan);
        }
        Doing::Choosing => {
            let line = match failure {
                Some(message) => Line::from(Span::styled(
                    message.to_string(),
                    Style::default().fg(Color::Red),
                )),
                None => Line::from(
                    format!(" up/down choose   tab switch   enter {enter}   d delete   esc quit")
                        .dim(),
                ),
            };
            frame.render_widget(Paragraph::new(line).block(bordered("")), area);
        }
    }
}

fn bordered(title: &str) -> Block<'_> {
    Block::bordered()
        .border_type(BorderType::Rounded)
        .title(title)
}

/// A rate, in the same units the sizes are shown in.
fn per_second(bytes: f64) -> String {
    if bytes >= 1e9 {
        format!("{:.2} GB/s", bytes / 1e9)
    } else {
        format!("{:.0} MB/s", bytes / 1e6)
    }
}

/// How long `bytes` more will take at `rate`, said the way someone waiting would say it.
fn remaining(bytes: u64, rate: f64) -> String {
    if rate <= 0.0 {
        return String::new();
    }

    let seconds = (bytes as f64 / rate).round() as u64;
    if seconds >= 3600 {
        format!("{}h{:02}m left", seconds / 3600, (seconds % 3600) / 60)
    } else if seconds >= 60 {
        format!("{}m{:02}s left", seconds / 60, seconds % 60)
    } else {
        format!("{seconds}s left")
    }
}

fn gigabytes(bytes: u64) -> String {
    if bytes >= 1_000_000_000 {
        format!("{:.2} GB", bytes as f64 / 1e9)
    } else {
        format!("{:.0} MB", bytes as f64 / 1e6)
    }
}

#[cfg(test)]
mod tests {
    use ratatui::backend::TestBackend;

    use super::*;
    use crate::{Device, Residency};

    /// The foot, drawn into a buffer, as one long string of what it says.
    fn foot(doing: &Doing) -> String {
        let mut terminal = ratatui::Terminal::new(TestBackend::new(100, 3)).unwrap();
        terminal
            .draw(|frame| draw_foot(frame, frame.area(), doing, "fetch and use", None))
            .unwrap();

        terminal
            .backend()
            .buffer()
            .content()
            .iter()
            .map(|cell| cell.symbol())
            .collect()
    }

    fn fetching(done: u64, total: Option<u64>, speed: Option<f64>) -> Doing {
        let (_say, news) = channel::<Word>();
        Doing::Fetching {
            news,
            from: Some("Hugging Face"),
            file: "wai-illustrious-v17-00001-of-00004.waifupkg".to_string(),
            done,
            total,
            part: 2,
            parts: 4,
            speed,
            sample: (Instant::now(), done),
        }
    }

    /// The whole screen, drawn into a buffer, as one long string of what it says.
    fn screen(choices: &Choices) -> String {
        let mut terminal = ratatui::Terminal::new(TestBackend::new(100, 15)).unwrap();
        terminal
            .draw(|frame| draw(frame, choices, &Doing::Choosing, None))
            .unwrap();

        terminal
            .backend()
            .buffer()
            .content()
            .iter()
            .map(|cell| cell.symbol())
            .collect()
    }

    fn choices(pane: Pane) -> Choices {
        Choices {
            entries: vec![
                Entry::Published {
                    name: "sdxl:wai",
                    full_name: "WAI Illustrious v17",
                    cached: true,
                    bytes: 6_970_000_000,
                },
                Entry::OnDisk,
            ],
            selected: 0,
            devices: Runtime::ALL
                .into_iter()
                .map(|runtime| Choice {
                    // Said rather than asked, so that what the box draws is the same on a
                    // machine with a card and on one without.
                    available: runtime.device() != Device::Metal,
                    runtime,
                })
                .collect(),
            device: 0,
            pane,
        }
    }

    #[test]
    fn the_list_offers_a_file_off_the_disk_under_the_published_ones() {
        let drawn = screen(&choices(Pane::Models));

        // The published models first, since the list is for someone who does not yet know what
        // to ask for, and the way to a file they already have under them.
        assert!(drawn.contains("sdxl:wai"), "{drawn}");
        assert!(drawn.contains("a file..."), "{drawn}");
        assert!(drawn.contains("yaml"), "{drawn}");

        let choices = choices(Pane::Models);
        let names: Vec<&str> = choices.entries.iter().map(Entry::name).collect();
        assert_eq!(names.last(), Some(&"a file..."));
    }

    #[test]
    fn where_the_weights_wait_is_a_device_and_not_a_box_of_its_own() {
        let drawn = screen(&choices(Pane::Models));

        // One list, with the offload one in it. There is no second box: where the weights wait
        // is only ever a question about one device, and asked beside the device it was a box
        // whose rows were struck out three times out of four.
        for runtime in Runtime::ALL {
            assert!(drawn.contains(runtime.name()), "{} {drawn}", runtime.name());
        }
        assert!(!drawn.contains("weights"), "{drawn}");
        assert!(!drawn.contains("low vram"), "{drawn}");
        assert!(!drawn.contains("on the card"), "{drawn}");
    }

    #[test]
    fn the_offload_row_says_what_it_costs() {
        // The one row anybody has to be told something about. The others are a device name and
        // nothing else to know; this one is the slow answer, and reads as a mistake to anyone
        // who picks it without being told.
        let drawn = screen(&choices(Pane::Models));
        assert!(drawn.contains("slower"), "{drawn}");

        // And a device this build has no operators for says so rather than being left out: a row
        // that says why is better than a row that is not there.
        assert!(drawn.contains("not in this build"), "{drawn}");
    }

    #[test]
    fn the_device_box_is_what_it_hands_back() {
        let mut choices = choices(Pane::Devices);

        assert_eq!(choices.chosen_device().runtime, Runtime::ALL[0]);
        assert_eq!(
            choices.chosen_device().runtime.residency(),
            Residency::Device
        );

        // The offload row, which is the only one that answers it the other way.
        choices.device = Runtime::ALL
            .iter()
            .position(|runtime| *runtime == Runtime::CUDA_CPU_OFFLOAD)
            .unwrap();
        assert_eq!(
            choices.chosen_device().runtime.residency(),
            Residency::LowVram
        );
        assert_eq!(choices.chosen_device().runtime.device(), Device::Cuda);
    }

    #[test]
    fn the_list_says_nothing_about_where_a_model_would_come_from() {
        // Which hub answers is settled by a probe on the first fetch, so at this point there is
        // no true answer to show -- and an address is not what anyone wanted anyway.
        let drawn = screen(&choices(Pane::Models));
        assert!(!drawn.contains("downloaded from"), "{drawn}");
        assert!(!drawn.contains("https://"), "{drawn}");
    }

    #[test]
    fn a_fetch_says_which_hub_it_is_talking_to_and_not_which_address() {
        // The one thing worth knowing about where seven gigabytes is coming from, in the words
        // the hub is called by, over the bar that shows how it is going.
        let drawn = foot(&fetching(1_000_000_000, Some(2_000_000_000), None));
        assert!(drawn.contains("from Hugging Face"), "{drawn}");
        assert!(!drawn.contains("https://"), "{drawn}");
        assert!(!drawn.contains("huggingface.co"), "{drawn}");

        // And before the fetch has settled on one, the bar says everything else and not that.
        let (_say, news) = channel::<Word>();
        let waiting = Doing::Fetching {
            news,
            from: None,
            file: "wai-illustrious-v17-00001-of-00004.waifupkg".to_string(),
            done: 0,
            total: None,
            part: 1,
            parts: 0,
            speed: None,
            sample: (Instant::now(), 0),
        };
        let drawn = foot(&waiting);
        assert!(drawn.contains("fetching wai-illustrious"), "{drawn}");
        assert!(!drawn.contains("from"), "{drawn}");
    }

    #[test]
    fn the_foot_says_what_enter_does_on_the_row_the_cursor_is_on() {
        // The rows do not all do the same thing, and a bar that says "fetch" over a row that
        // opens a list is worse than no bar at all.
        let foot = |selected: usize| {
            let mut choices = choices(Pane::Models);
            choices.selected = selected;

            let mut terminal = ratatui::Terminal::new(TestBackend::new(90, 18)).unwrap();
            terminal
                .draw(|frame| draw(frame, &choices, &Doing::Choosing, None))
                .unwrap();
            terminal
                .backend()
                .buffer()
                .content()
                .iter()
                .map(|cell| cell.symbol())
                .collect::<String>()
        };

        assert!(foot(0).contains("enter fetch and use"), "{}", foot(0));
        assert!(foot(1).contains("enter look for one"), "{}", foot(1));
    }

    #[test]
    fn what_went_wrong_is_shown_as_it_was_given() {
        // A fetch that failed and a file that cannot be opened both land here, so the message
        // says which it was rather than the bar assuming.
        let mut terminal = ratatui::Terminal::new(TestBackend::new(90, 18)).unwrap();
        terminal
            .draw(|frame| {
                draw(
                    frame,
                    &choices(Pane::Models),
                    &Doing::Choosing,
                    Some("00002-of-00004 is a later part of a model"),
                )
            })
            .unwrap();
        let drawn: String = terminal
            .backend()
            .buffer()
            .content()
            .iter()
            .map(|cell| cell.symbol())
            .collect();

        assert!(drawn.contains("is a later part of a model"), "{drawn}");
        assert!(!drawn.contains("could not fetch"), "{drawn}");
    }

    #[test]
    fn the_row_for_a_file_is_not_waiting_on_a_fetch() {
        // The other rows are lit up by whether they are on disk yet. This one has nothing to
        // fetch, so it reads as ready rather than as never fetched.
        assert!(Entry::OnDisk.ready());
        assert!(!Entry::Published {
            name: "sdxl:base",
            full_name: "Stable Diffusion XL Base 1.0",
            cached: false,
            bytes: 0,
        }
        .ready());
    }

    #[test]
    fn a_model_off_the_disk_is_called_by_its_file_name() {
        let chosen = from_disk(
            PathBuf::from("/home/someone/models/wai-illustrious-v17.yaml"),
            Runtime::CUDA_CPU_OFFLOAD,
        )
        .unwrap();

        // The name, not the path: the whole path does not fit in the heading it goes into, and
        // the suffix says nothing that the screen it is on does not already say.
        assert_eq!(chosen.name, "wai-illustrious-v17");
        assert_eq!(chosen.runtime, Runtime::CUDA_CPU_OFFLOAD);
        assert_eq!(
            chosen.path,
            PathBuf::from("/home/someone/models/wai-illustrious-v17.yaml")
        );
    }

    #[test]
    fn a_model_is_its_manifest_however_many_packages_it_has() {
        // There used to be a check here for a later part of a split model, because a model was
        // its packages: four of them sat in a directory and nothing in the names said which one
        // to open. A model is one manifest now, so there is nothing left to get wrong.
        let chosen = from_disk(
            PathBuf::from("/c/wai-illustrious-v17.yaml"),
            Runtime::ALL[0],
        )
        .expect("a manifest is a model");
        assert_eq!(chosen.name, "wai-illustrious-v17");
    }

    #[test]
    fn this_screen_says_what_built_it_too() {
        let drawn = screen(&choices(Pane::Models));

        // Whichever screen is up when something goes wrong is the one that gets screenshotted, so
        // both of them carry it.
        assert!(drawn.contains("libwaifu"), "{drawn}");
        assert!(drawn.contains(crate::cli::REVISION), "{drawn}");
    }

    #[test]
    fn the_device_is_offered_beside_the_models() {
        let drawn = screen(&choices(Pane::Models));

        assert!(drawn.contains("models"), "{drawn}");
        assert!(drawn.contains("device"), "{drawn}");
        assert!(drawn.contains("sdxl:wai"), "{drawn}");
        assert!(drawn.contains("cpu"), "{drawn}");

        // A device this build has no operators for is listed and says so, rather than being left
        // out: that is how someone finds out which kind of build they are running.
        assert!(drawn.contains("cuda"), "{drawn}");
        assert!(drawn.contains("not in this build"), "{drawn}");
    }

    #[test]
    fn only_the_half_the_keys_talk_to_is_lit_up() {
        // The marked model row either way, found rather than counted to: the row it lands on is
        // the layout's business and moves whenever the heading does.
        let model_row = |pane| {
            let mut terminal = ratatui::Terminal::new(TestBackend::new(100, 15)).unwrap();
            let choices = choices(pane);
            terminal
                .draw(|frame| draw(frame, &choices, &Doing::Choosing, None))
                .unwrap();

            let buffer = terminal.backend().buffer().clone();
            let (x, y) = (0..12)
                .flat_map(|y| (0..100).map(move |x| (x, y)))
                .find(|at| buffer[*at].symbol() == "s" && buffer[(at.0 + 1, at.1)].symbol() == "d")
                .expect("the model name is on the screen");
            buffer[(x, y)].bg
        };

        // Two cursors are on screen at once, and both looking live would say the keys go to both.
        assert_eq!(model_row(Pane::Models), Color::Cyan);
        assert_ne!(model_row(Pane::Devices), Color::Cyan);
    }

    #[test]
    fn the_name_is_in_the_border_and_the_numbers_are_in_the_bar() {
        let drawn = foot(&fetching(223_000_000, Some(2_020_000_000), None));

        // The name is long and the bar is what the eye is on, so one is in the border and the
        // other is not: a label wide enough to cover the bar hides what the bar is for.
        assert!(
            drawn.contains("wai-illustrious-v17-00001-of-00004.waifupkg"),
            "{drawn}"
        );
        assert!(drawn.contains("part 2 of 4"), "{drawn}");
        assert!(drawn.contains("11%"), "{drawn}");
        assert!(drawn.contains("223 MB of 2.02 GB"), "{drawn}");
    }

    #[test]
    fn the_label_reads_against_the_bar_it_sits_on() {
        let mut terminal = ratatui::Terminal::new(TestBackend::new(100, 3)).unwrap();
        let doing = fetching(1_240_000_000, Some(1_740_000_000), Some(29e6));
        terminal
            .draw(|frame| draw_foot(frame, frame.area(), &doing, "fetch and use", None))
            .unwrap();

        // The first digit of the label, which at 71% is well inside the filled part of the bar.
        let buffer = terminal.backend().buffer().clone();
        let row = 1;
        let at = (0..100)
            .find(|x| buffer[(*x, row)].symbol() == "7")
            .expect("the label is on the middle row");

        // Swapped against the bar rather than drawn over it: the blocks are cyan on black, so the
        // text on them has to be black on cyan or it cannot be read.
        let cell = &buffer[(at, row)];
        assert_eq!(cell.fg, Color::Black, "{:?}", cell);
        assert_eq!(cell.bg, Color::Cyan, "{:?}", cell);
    }

    #[test]
    fn the_bar_is_painted_rather_than_spelled_out_in_blocks() {
        let mut terminal = ratatui::Terminal::new(TestBackend::new(100, 3)).unwrap();
        let doing = fetching(1_240_000_000, Some(1_740_000_000), None);
        terminal
            .draw(|frame| draw_foot(frame, frame.area(), &doing, "fetch and use", None))
            .unwrap();
        let buffer = terminal.backend().buffer().clone();

        // Every cell of the bar is a background colour on a space. A row that is part block glyph
        // and part coloured space agrees with itself on the grid and not on the screen: the glyph
        // does not fill its cell in every terminal font, and the bar comes out ribbed.
        for x in 1..99 {
            let cell = &buffer[(x, 1)];
            assert_ne!(cell.symbol(), "\u{2588}", "column {x}");
        }

        // Filled at the left and empty at the right, at 71% of the way along.
        assert_eq!(buffer[(1, 1)].bg, Color::Cyan);
        assert_eq!(buffer[(98, 1)].bg, Color::Black);
    }

    #[test]
    fn a_known_rate_says_how_fast_and_how_long_is_left() {
        let drawn = foot(&fetching(1_940_000_000, Some(2_020_000_000), Some(40e6)));

        assert!(drawn.contains("40 MB/s"), "{drawn}");
        assert!(drawn.contains("2s left"), "{drawn}");
    }

    #[test]
    fn without_a_length_there_is_no_percentage_to_show() {
        let drawn = foot(&fetching(223_000_000, None, Some(40e6)));

        assert!(drawn.contains("223 MB"), "{drawn}");
        assert!(!drawn.contains('%'), "{drawn}");
        assert!(!drawn.contains("left"), "{drawn}");
    }

    #[test]
    fn time_left_is_said_the_way_someone_waiting_would_say_it() {
        assert_eq!(remaining(40_000_000, 40e6), "1s left");
        assert_eq!(remaining(4_000_000_000, 40e6), "1m40s left");
        assert_eq!(remaining(400_000_000_000, 40e6), "2h46m left");

        // A rate of nothing gives no answer rather than an infinite one.
        assert_eq!(remaining(1_000, 0.0), "");
    }

    #[test]
    fn a_rate_reads_in_the_units_the_sizes_do() {
        assert_eq!(per_second(40e6), "40 MB/s");
        assert_eq!(per_second(1.5e9), "1.50 GB/s");
    }
}
