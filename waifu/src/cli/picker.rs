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
use std::time::Duration;

use ratatui::crossterm::event::{self, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Color, Style, Stylize};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, BorderType, Gauge, Paragraph};
use ratatui::{DefaultTerminal, Frame};

use crate::cli::hub;

type Error = Box<dyn std::error::Error>;

/// How often the screen looks for a key or for news of the fetch.
const TICK: Duration = Duration::from_millis(100);

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

/// What the fetching thread has to say.
enum Word {
    /// A file, how far into it, and how long it is when that is known.
    Fetching(String, u64, Option<u64>),
    /// Every package is in the cache; the model is at this path.
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
    },
}

/// Run the picker until a model is chosen and on disk, or the user leaves.
///
/// Returns where the chosen model is, or `None` if they quit. The terminal is the caller's: this
/// borrows it and gives it back as it found it.
pub fn choose(terminal: &mut DefaultTerminal) -> Result<Option<std::path::PathBuf>, Error> {
    // Versioned names are left out. Someone who wants `sdxl:base:v1` in particular can type it,
    // and a list is for someone who does not yet know what to type.
    let mut entries: Vec<Entry> = hub::names()
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

    // Something already on disk is the one most likely to be wanted, so start there.
    let mut selected = entries.iter().position(|entry| entry.cached).unwrap_or(0);
    let mut doing = Doing::Choosing;
    let mut failure: Option<String> = None;

    loop {
        terminal.draw(|frame| draw(frame, &entries, selected, &doing, failure.as_deref()))?;

        if let Doing::Fetching {
            news,
            file,
            done,
            total,
        } = &mut doing
        {
            match news.try_recv() {
                Ok(Word::Fetching(name, at, length)) => {
                    *file = name;
                    *done = at;
                    *total = length;
                }
                Ok(Word::Done(path)) => return Ok(Some(path)),
                Ok(Word::Failed(message)) => {
                    failure = Some(message);
                    doing = Doing::Choosing;
                    for entry in &mut entries {
                        entry.cached = hub::is_cached(entry.name);
                        entry.bytes = hub::cached_bytes(entry.name);
                    }
                }
                Err(TryRecvError::Empty) => {}
                Err(TryRecvError::Disconnected) => {
                    failure = Some("the fetch stopped without saying why".to_string());
                    doing = Doing::Choosing;
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
                selected = selected.saturating_sub(1);
                failure = None;
            }
            KeyCode::Down | KeyCode::Char('j') => {
                selected = (selected + 1).min(entries.len() - 1);
                failure = None;
            }
            KeyCode::Enter => {
                let name = entries[selected].name;
                failure = None;

                let (say, news) = channel::<Word>();
                std::thread::spawn(move || {
                    let mut report = |progress: hub::Progress| {
                        let word = match progress {
                            hub::Progress::Fetching { file, done, total } => {
                                Word::Fetching(file.to_string(), done, total)
                            }
                            hub::Progress::Fetched { file, bytes } => {
                                Word::Fetching(file.to_string(), bytes, Some(bytes))
                            }
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

fn draw(frame: &mut Frame, entries: &[Entry], selected: usize, doing: &Doing, failure: Option<&str>) {
    let [heading, list, foot] = Layout::vertical([
        Constraint::Length(3),
        Constraint::Min(3),
        Constraint::Length(3),
    ])
    .areas(frame.area());

    frame.render_widget(
        Paragraph::new("Pick a model. One will be fetched the first time it is used.")
            .block(bordered(" waifu ")),
        heading,
    );

    let rows: Vec<Line> = entries
        .iter()
        .enumerate()
        .map(|(index, entry)| {
            let marker = if index == selected { "> " } else { "  " };
            let style = if index == selected {
                Style::default().fg(Color::Black).bg(Color::Cyan)
            } else if entry.cached {
                Style::default().fg(Color::Green)
            } else {
                Style::default()
            };
            Line::from(vec![
                Span::raw(marker),
                Span::styled(format!("{:<14}", entry.name), style),
                Span::raw("  "),
                Span::raw(entry.describe()).dim(),
            ])
        })
        .collect();
    frame.render_widget(Paragraph::new(rows).block(bordered(" models ")), list);

    draw_foot(frame, foot, doing, failure);
}

fn draw_foot(frame: &mut Frame, area: Rect, doing: &Doing, failure: Option<&str>) {
    match doing {
        Doing::Fetching {
            file, done, total, ..
        } => {
            let ratio = match total {
                Some(total) if *total > 0 => (*done as f64 / *total as f64).clamp(0.0, 1.0),
                _ => 0.0,
            };
            let label = match total {
                Some(total) if *total > 0 => {
                    format!("{file}  {} of {}", gigabytes(*done), gigabytes(*total))
                }
                _ => format!("{file}  {}", gigabytes(*done)),
            };
            frame.render_widget(
                Gauge::default()
                    .block(bordered(" fetching "))
                    .ratio(ratio)
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
                None => Line::from(" up/down choose   enter fetch and use   esc quit".dim()),
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

fn gigabytes(bytes: u64) -> String {
    if bytes >= 1_000_000_000 {
        format!("{:.2} GB", bytes as f64 / 1e9)
    } else {
        format!("{:.0} MB", bytes as f64 / 1e6)
    }
}
