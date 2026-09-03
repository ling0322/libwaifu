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

use std::sync::mpsc::{channel, Receiver, TryRecvError};
use std::time::{Duration, Instant};

use ratatui::crossterm::event::{self, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Color, Style, Stylize};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, BorderType, Gauge, Paragraph};
use ratatui::{DefaultTerminal, Frame};

use crate::cli::{built_from, hub};
use crate::Device;

type Error = Box<dyn std::error::Error>;

/// How often the screen looks for a key or for news of the fetch.
const TICK: Duration = Duration::from_millis(100);

/// How far apart two byte counts have to be before they are worth a rate. Shorter than this and
/// the answer is mostly the buffer size divided by how long one read happened to take.
const SPEED_WINDOW: Duration = Duration::from_millis(500);

/// One row of the list.
struct Entry {
    name: &'static str,
    /// Whether every package of it is already in the cache.
    cached: bool,
    /// What is on disk for it, which is most of a model for one that was interrupted.
    bytes: u64,
}

impl Entry {
    fn describe(&self) -> String {
        if self.cached {
            format!("on disk, {}", gigabytes(self.bytes))
        } else if self.bytes > 0 {
            format!("part fetched, {}", gigabytes(self.bytes))
        } else {
            "not fetched".to_string()
        }
    }
}

/// A device the run could go on, and whether it can.
struct Choice {
    device: Device,
    /// Asked once, at the start. The answer cannot change while the screen is up, and asking is a
    /// call across the C boundary that a redraw has no business making.
    available: bool,
}

/// Which half of the screen the keys are talking to.
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

/// Run the picker until a model is chosen and on disk, or the user leaves.
///
/// Returns where the chosen model is, or `None` if they quit. The terminal is the caller's: this
/// borrows it and gives it back as it found it.
/// A model and a device, chosen.
pub struct Chosen {
    pub path: std::path::PathBuf,
    /// What it was called in the list, which is shorter and steadier than its path and is what
    /// the drawing screen shows.
    pub name: &'static str,
    pub device: Device,
}

pub fn choose(
    terminal: &mut DefaultTerminal,
    device: Device,
) -> Result<Option<Chosen>, Error> {
    // Versioned names are left out. Someone who wants `sdxl:base:v1` in particular can type it,
    // and a list is for someone who does not yet know what to type.
    let entries: Vec<Entry> = hub::names()
        .into_iter()
        .filter(|name| name.matches(':').count() == 1)
        .map(|name| Entry {
            name,
            cached: hub::is_cached(name),
            bytes: hub::cached_bytes(name),
        })
        .collect();
    if entries.is_empty() {
        return Err("this build knows no models to offer".into());
    }

    let devices: Vec<Choice> = [Device::Cpu, Device::Cuda]
        .into_iter()
        .map(|device| Choice {
            device,
            available: device.is_available(),
        })
        .collect();

    // Something already on disk is the one most likely to be wanted, so start there.
    let mut choices = Choices {
        selected: entries.iter().position(|entry| entry.cached).unwrap_or(0),
        entries,
        // Whatever the command line resolved to, so that the box says what would have happened
        // rather than making someone who passed -d set it a second time.
        device: devices
            .iter()
            .position(|choice| choice.device == device)
            .unwrap_or(0),
        devices,
        pane: Pane::Models,
    };
    let mut doing = Doing::Choosing;
    let mut failure: Option<String> = None;

    loop {
        terminal.draw(|frame| draw(frame, &choices, &doing, failure.as_deref()))?;

        let mut outcome = Outcome::Nothing;
        if let Doing::Fetching {
            news,
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
            // The device the fetch was started with, which is still the one showing: the keys do
            // nothing but quit while one is running.
            Outcome::Done(path) => {
                return Ok(Some(Chosen {
                    path,
                    name: choices.entries[choices.selected].name,
                    device: choices.chosen_device().device,
                }))
            }
            Outcome::Failed(message) => {
                failure = Some(message);
                doing = Doing::Choosing;
                for entry in &mut choices.entries {
                    entry.cached = hub::is_cached(entry.name);
                    entry.bytes = hub::cached_bytes(entry.name);
                }
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
                        chosen.device.name()
                    ));
                    continue;
                }

                let name = choices.entries[choices.selected].name;
                failure = None;

                let (say, news) = channel::<Word>();
                std::thread::spawn(move || {
                    let mut report = |progress: hub::Progress| {
                        let word = match progress {
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

    // Wide enough for the longest device line and no wider: the model names and what is on disk
    // are what the eye is looking for, and the device is a two-way switch set once.
    let [models, devices] =
        Layout::horizontal([Constraint::Min(20), Constraint::Length(30)]).areas(middle);

    // The same line the drawing screen carries, and for the same reason. What used to be here was
    // a bordered box saying what to do, which is two rows of frame around one row of text that
    // the key hints at the foot already cover.
    frame.render_widget(Paragraph::new(built_from()), built);
    frame.render_widget(
        Paragraph::new(
            Line::from(" Pick a model and a device. The model is fetched the first time it is used.")
                .dim(),
        ),
        told,
    );

    let rows: Vec<Line> = choices
        .entries
        .iter()
        .enumerate()
        .map(|(index, entry)| {
            let chosen = index == choices.selected;
            Line::from(vec![
                Span::raw(if chosen { "> " } else { "  " }),
                Span::styled(
                    format!("{:<14}", entry.name),
                    row_style(chosen, choices.pane == Pane::Models, entry.cached),
                ),
                Span::raw("  "),
                Span::raw(entry.describe()).dim(),
            ])
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
                    format!("{:<6}", choice.device.name()),
                    row_style(chosen, choices.pane == Pane::Devices, choice.available),
                ),
                Span::raw("  "),
                Span::raw(if choice.available {
                    "ready"
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

    draw_foot(frame, foot, doing, failure);
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

fn draw_foot(frame: &mut Frame, area: Rect, doing: &Doing, failure: Option<&str>) {
    match doing {
        Doing::Fetching {
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
            let title = format!(" fetching {file}{of}");

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
                    label.push_str(&format!("   {}", remaining(total.saturating_sub(*done), *rate)));
                }
            }

            frame.render_widget(
                Gauge::default()
                    .block(bordered(&title))
                    .ratio(ratio)
                    // Named rather than left to default, because the label sits on top of the bar
                    // and ratatui draws it by swapping these two: without a colour to swap there
                    // is nothing to tell the text from the blocks under it. Cyan is what the
                    // chosen row in the list is marked with, so the two read as one screen.
                    .gauge_style(Style::default().fg(Color::Cyan).bg(Color::Black))
                    .label(label),
                area,
            );
        }
        Doing::Choosing => {
            let line = match failure {
                Some(message) => Line::from(Span::styled(
                    format!("could not fetch: {message}"),
                    Style::default().fg(Color::Red),
                )),
                None => Line::from(
                    " up/down choose   left/right or tab switch   enter fetch and use   esc quit"
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

    /// The foot, drawn into a buffer, as one long string of what it says.
    fn foot(doing: &Doing) -> String {
        let mut terminal = ratatui::Terminal::new(TestBackend::new(100, 3)).unwrap();
        terminal
            .draw(|frame| draw_foot(frame, frame.area(), doing, None))
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
        let mut terminal = ratatui::Terminal::new(TestBackend::new(100, 12)).unwrap();
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
            entries: vec![Entry {
                name: "sdxl:wai",
                cached: true,
                bytes: 6_970_000_000,
            }],
            selected: 0,
            devices: vec![
                Choice {
                    device: Device::Cpu,
                    available: true,
                },
                Choice {
                    device: Device::Cuda,
                    available: false,
                },
            ],
            device: 0,
            pane,
        }
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
            let mut terminal = ratatui::Terminal::new(TestBackend::new(100, 12)).unwrap();
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
        assert!(drawn.contains("wai-illustrious-v17-00001-of-00004.waifupkg"), "{drawn}");
        assert!(drawn.contains("part 2 of 4"), "{drawn}");
        assert!(drawn.contains("11%"), "{drawn}");
        assert!(drawn.contains("223 MB of 2.02 GB"), "{drawn}");
    }

    #[test]
    fn the_label_reads_against_the_bar_it_sits_on() {
        let mut terminal = ratatui::Terminal::new(TestBackend::new(100, 3)).unwrap();
        let doing = fetching(1_240_000_000, Some(1_740_000_000), Some(29e6));
        terminal
            .draw(|frame| draw_foot(frame, frame.area(), &doing, None))
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
