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

//! The second screen: which model, and the fetch that follows if it is not on the disk.
//!
//! The list is the task's: picture models to draw with, only the ones that can start from a
//! picture for img2img, and voices to read with. What is already on disk is marked, what is not
//! is fetched when it is picked, and the next screen does not come up until it is all here -- so
//! the page that opens at the end never has a download to do.
//!
//! The fetch is minutes long, so it runs on a thread of its own and says how it is getting on down
//! a channel. The screen stays live while it happens: a download that cannot be watched is
//! indistinguishable from one that has hung, and one that cannot be stopped is a download somebody
//! ends with ctrl-c and a `.part` file.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{channel, Receiver, TryRecvError};
use std::sync::Arc;
use std::time::{Duration, Instant};

use ratatui::crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Color, Style, Stylize};
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;
use ratatui::{DefaultTerminal, Frame};

use crate::cli::hub;
use crate::cli::task::Task;
use crate::cli::tui::ask::{Answer, Confirm};
use crate::cli::tui::files::{self, FilePicker};
use crate::cli::tui::{bar, bordered, gigabytes, heading, quits, row_style, Picked, Step};

type Error = Box<dyn std::error::Error>;

/// How often the screen looks for a key or for news of the fetch.
const TICK: Duration = Duration::from_millis(100);

/// How far apart two byte counts have to be before they are worth a rate. Shorter than this and
/// the answer is mostly the buffer size divided by how long one read happened to take.
const SPEED_WINDOW: Duration = Duration::from_millis(500);

/// What a model's manifest is called, which is what the file row offers to look for.
///
/// The packages beside it are not offered: a model is its manifest, and the weights are files it
/// names rather than files anybody picks.
const MANIFEST_SUFFIX_NAME: &str = "yaml";

/// One row of the list.
enum Entry {
    /// A model this build knows how to fetch.
    Published {
        name: &'static str,
        full_name: &'static str,
        /// Whether every package of it is already in the cache.
        cached: bool,
        /// What is on disk for it, which is most of a model for one that was interrupted.
        bytes: u64,
        /// Whether it draws explicit pictures readily, which keeps it off the list until somebody
        /// says they want to see those.
        explicit: bool,
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

    fn explicit(&self) -> bool {
        matches!(self, Entry::Published { explicit: true, .. })
    }
}

/// The published models a task can run, in the catalogue's order, and the file row under them.
fn entries_for(task: Task) -> Vec<Entry> {
    let listed = match task.speaks() {
        true => hub::listed_voices(),
        false => hub::listed(),
    };

    let mut entries: Vec<Entry> = listed
        .into_iter()
        // What the img2img page would only refuse. Asked of the kind and of the manifest where it
        // is here, which is as much as can be known without reading the weights.
        .filter(|model| {
            task != Task::Img2Img || crate::cli::webui::draws_from_a_picture(model.name)
        })
        .map(|model| Entry::Published {
            name: model.name,
            full_name: model.full_name,
            cached: model.cached,
            bytes: model.bytes,
            explicit: model.explicit,
        })
        .collect();

    // Last, under the published ones: it is the answer for someone who already has a package, and
    // the list above is for someone who does not yet know what to ask for.
    entries.push(Entry::OnDisk);
    entries
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
    /// Every package is in the cache.
    Done,
    /// It was asked to stop, and did; what is said is what was kept.
    Stopped(String),
    Failed(String),
}

/// What the screen is doing.
enum Doing {
    Choosing,
    /// Fetching, with the last word from the thread and the channel it comes down.
    Fetching {
        news: Receiver<Word>,
        /// What the thread asks between one piece of work and the next.
        stop: Arc<AtomicBool>,
        /// Which model, so that it can be handed on when the fetch is done: the cursor could not
        /// move while it ran, but the name is what is wanted and not a row.
        name: &'static str,
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

/// Everything the screen shows that is not the fetch.
struct Choices {
    task: Task,
    /// Every row there is, hidden ones included; [`Choices::shown`] is what is on screen.
    entries: Vec<Entry>,
    /// Where the cursor is, counted over the shown rows.
    selected: usize,
    /// Whether the models that draw explicit pictures are on the list. Off until somebody asks,
    /// and asked about before it is turned on.
    explicit: bool,
}

impl Choices {
    /// The rows on screen, as indices into `entries`.
    fn shown(&self) -> Vec<usize> {
        (0..self.entries.len())
            .filter(|index| self.explicit || !self.entries[*index].explicit())
            .collect()
    }

    fn hidden(&self) -> usize {
        match self.explicit {
            true => 0,
            false => self.entries.iter().filter(|entry| entry.explicit()).count(),
        }
    }

    fn current(&self) -> &Entry {
        let shown = self.shown();
        &self.entries[shown[self.selected.min(shown.len() - 1)]]
    }

    /// Puts the cursor on the row `name` is on, where it is on screen.
    fn select(&mut self, name: &str) {
        if let Some(at) = self
            .shown()
            .iter()
            .position(|index| self.entries[*index].name() == name)
        {
            self.selected = at;
        }
    }

    /// Asks the cache again what it holds, for every row that is a model this build can fetch.
    ///
    /// Called after anything that changes what is on disk -- a fetch that stopped partway, a model
    /// deleted -- rather than working the new state out from what just happened: the row says
    /// what is there, and the disk is the only thing that knows.
    fn read_the_cache_again(&mut self) {
        for entry in &mut self.entries {
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
}

/// Runs the list until a model is chosen and on the disk, or somebody goes back or leaves.
pub fn choose(terminal: &mut DefaultTerminal, task: Task) -> Result<Step<Picked>, Error> {
    let mut choices = Choices {
        task,
        entries: entries_for(task),
        selected: 0,
        explicit: false,
    };

    // Something already on disk is the one most likely to be wanted, so start there.
    choices.selected = choices
        .shown()
        .iter()
        .position(|index| {
            matches!(
                choices.entries[*index],
                Entry::Published { cached: true, .. }
            )
        })
        .unwrap_or(0);

    let mut doing = Doing::Choosing;
    let mut failure: Option<String> = None;
    let mut browsing: Option<FilePicker> = None;
    // What the question on screen is about, kept beside the question: a name for a delete, and
    // nothing for the question about showing the explicit models.
    let mut asking: Option<(Option<&'static str>, Confirm)> = None;

    loop {
        terminal.draw(|frame| {
            draw(frame, &choices, &doing, failure.as_deref());
            if let Some(browsing) = &browsing {
                browsing.render(frame, frame.area());
            }
            if let Some((_, question)) = &asking {
                question.render(frame, frame.area());
            }
        })?;

        if let Doing::Fetching { .. } = doing {
            match listen(&mut doing) {
                None => {}
                Some(Word::Done) => {
                    let Doing::Fetching { name, .. } = doing else {
                        unreachable!("only a fetch is listened to");
                    };
                    return Ok(Step::Next(published(name)));
                }
                Some(Word::Stopped(said)) => {
                    failure = Some(said);
                    doing = Doing::Choosing;
                    choices.read_the_cache_again();
                }
                Some(Word::Failed(message)) => {
                    failure = Some(format!("could not fetch: {message}"));
                    doing = Doing::Choosing;
                    choices.read_the_cache_again();
                }
                Some(_) => unreachable!("listen hands back only an ending"),
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

        // A question on screen has the keys before anything behind it: it is the thing that was
        // just asked for, and it is answered before anything else is.
        if let Some((about, question)) = &mut asking {
            if key.code == KeyCode::Char('c') && key.modifiers.contains(KeyModifiers::CONTROL) {
                return Ok(Step::Quit);
            }

            match (question.key(key), *about) {
                (Answer::Open, _) => {}
                (Answer::Cancelled | Answer::Given(false), _) => asking = None,
                (Answer::Given(true), None) => {
                    asking = None;
                    let on = choices.current().name().to_string();
                    choices.explicit = true;
                    choices.select(&on);
                }
                (Answer::Given(true), Some(name)) => {
                    asking = None;
                    match hub::remove(name) {
                        // Read back off the disk rather than assumed: what the row says about a
                        // model is what is there, and this is the same call that first said it.
                        Ok(()) => choices.read_the_cache_again(),
                        Err(gone_wrong) => {
                            failure = Some(format!("could not delete: {gone_wrong}"))
                        }
                    }
                }
            }
            continue;
        }

        // While the file picker is up it has the keys, except the one that always ends the
        // program. Nothing behind it is running -- a fetch and the picker cannot both be on.
        if let Some(open) = &mut browsing {
            if key.code == KeyCode::Char('c') && key.modifiers.contains(KeyModifiers::CONTROL) {
                return Ok(Step::Quit);
            }

            match open.key(key) {
                files::Outcome::Open => {}
                files::Outcome::Cancelled => browsing = None,
                files::Outcome::Picked(path) => return Ok(Step::Next(from_disk(path))),
            }
            continue;
        }

        // While a fetch is running the keys stop it: escape stops it and stays, and the keys
        // that leave stop it on the way out. What has come down is kept either way, and the next
        // fetch of it carries on from there.
        if let Doing::Fetching { stop, .. } = &doing {
            if quits(&key) {
                stop.store(true, Ordering::Relaxed);
                return Ok(Step::Quit);
            }
            if key.code == KeyCode::Esc {
                stop.store(true, Ordering::Relaxed);
            }
            continue;
        }

        let last = choices.shown().len() - 1;
        match key.code {
            KeyCode::Up | KeyCode::Char('k') => {
                choices.selected = choices.selected.saturating_sub(1);
                failure = None;
            }
            KeyCode::Down | KeyCode::Char('j') => {
                choices.selected = (choices.selected + 1).min(last);
                failure = None;
            }
            // Delete, or the letter for anyone whose hands are on the vim keys. Nothing is thrown
            // away on the press: what it opens is the question, and the question starts on no.
            KeyCode::Delete | KeyCode::Char('d') => {
                failure = None;
                match choices.current() {
                    // Nothing fetched is nothing to delete, and a question about deleting it
                    // would be a question with no answer worth giving.
                    Entry::Published { bytes: 0, .. } => {
                        failure = Some("nothing of that one is on disk yet".to_string());
                    }
                    Entry::Published { name, bytes, .. } => {
                        asking = Some((
                            Some(name),
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
            // All of them, or back to the ones that do not draw explicit pictures. Showing them is
            // asked about first and hiding them is not: one of the two is a thing somebody might
            // not want on their screen, and the other is the list as it opened.
            KeyCode::Char('a') if !choices.task.speaks() => {
                failure = None;
                if choices.explicit {
                    let on = choices.current().name().to_string();
                    choices.explicit = false;
                    choices.selected = 0;
                    choices.select(&on);
                } else if choices.hidden() > 0 {
                    asking = Some((
                        None,
                        Confirm::ask("show models that draw explicit pictures?"),
                    ));
                }
            }
            KeyCode::Enter | KeyCode::Right => {
                failure = None;

                // The row that is not a model: nothing to fetch, so what opens is the disk.
                let (name, cached) = match *choices.current() {
                    Entry::Published { name, cached, .. } => (name, cached),
                    Entry::OnDisk => {
                        browsing = Some(FilePicker::open(
                            "a model manifest",
                            &std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")),
                            &[MANIFEST_SUFFIX_NAME],
                        ));
                        continue;
                    }
                };

                // Here already: nothing to wait for, and nothing to show a bar for.
                if cached {
                    return Ok(Step::Next(published(name)));
                }

                doing = start_fetching(name);
            }
            KeyCode::Esc | KeyCode::Left => return Ok(Step::Back),
            _ if quits(&key) => return Ok(Step::Quit),
            _ => {}
        }
    }
}

/// A published model, as the screens after this one call it: by its name, which is short and
/// steady, rather than by the full name, which is neither.
fn published(name: &str) -> Picked {
    Picked {
        model: name.to_string(),
        label: name.to_string(),
    }
}

/// What a manifest picked off the disk becomes.
///
/// It is called by the file it is in rather than by a name from the list. The whole path is too
/// long for the heading it goes into and its tail is the part that identifies it.
fn from_disk(path: PathBuf) -> Picked {
    Picked {
        label: path
            .file_stem()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string(),
        model: path.to_string_lossy().to_string(),
    }
}

/// Starts fetching `name` on a thread of its own.
fn start_fetching(name: &'static str) -> Doing {
    let (say, news) = channel::<Word>();
    let stop = Arc::new(AtomicBool::new(false));

    std::thread::spawn({
        let stop = Arc::clone(&stop);
        move || {
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

            let stopping = || stop.load(Ordering::Relaxed);
            let word = match hub::resolve_reporting(name, &mut report, &stopping) {
                Ok(_) => Word::Done,
                Err(error) if hub::stopped(&error) => Word::Stopped(error.to_string()),
                Err(error) => Word::Failed(error.to_string()),
            };
            let _ = say.send(word);
        }
    });

    Doing::Fetching {
        news,
        stop,
        name,
        from: None,
        file: name.to_string(),
        done: 0,
        total: None,
        part: 1,
        parts: 0,
        speed: None,
        sample: (Instant::now(), 0),
    }
}

/// Reads everything the fetching thread has said since the last look, and hands back how it ended
/// if it has: [`Word::Done`], [`Word::Stopped`] or [`Word::Failed`].
///
/// The whole backlog, not one word a tick. The thread speaks several times a second and the
/// screen turns over ten, so taking one at a time would show a number that falls further behind
/// the download the longer it runs.
fn listen(doing: &mut Doing) -> Option<Word> {
    let Doing::Fetching {
        news,
        from,
        file,
        done,
        total,
        part,
        parts,
        speed,
        sample,
        ..
    } = doing
    else {
        return None;
    };

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

                // Over half a second rather than between two words, which are a tenth of a second
                // apart and would make a rate that jumps around with the buffer.
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
            Ok(ending) => return Some(ending),
            Err(TryRecvError::Empty) => return None,
            Err(TryRecvError::Disconnected) => {
                return Some(Word::Failed("the fetch stopped without saying why".into()))
            }
        }
    }
}

fn draw(frame: &mut Frame, choices: &Choices, doing: &Doing, failure: Option<&str>) {
    let [top, told, list, foot] = Layout::vertical([
        Constraint::Length(1),
        Constraint::Length(1),
        Constraint::Min(3),
        Constraint::Length(3),
    ])
    .areas(frame.area());

    let what = match choices.task.speaks() {
        true => "voice",
        false => "model",
    };

    frame.render_widget(Paragraph::new(heading(&[choices.task.name()])), top);
    frame.render_widget(
        Paragraph::new(
            Line::from(format!(
                " Pick a {what}. One that is not on the disk yet is fetched now, before the page \
                 opens."
            ))
            .dim(),
        ),
        told,
    );

    let shown = choices.shown();
    let name_width = shown
        .iter()
        .map(|index| choices.entries[*index].name().len())
        .max()
        .unwrap_or(0)
        .max(14);
    let full_name_width = shown
        .iter()
        .map(|index| match &choices.entries[*index] {
            Entry::Published { full_name, .. } => full_name.len(),
            Entry::OnDisk => 0,
        })
        .max()
        .unwrap_or(0);

    let rows: Vec<Line> = shown
        .iter()
        .enumerate()
        .map(|(at, index)| {
            let entry = &choices.entries[*index];
            let chosen = at == choices.selected;
            let mut spans = vec![
                Span::raw(if chosen { "> " } else { "  " }),
                Span::styled(
                    format!("{:<name_width$}", entry.name()),
                    row_style(chosen, entry.ready()),
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
            if entry.explicit() {
                spans.push(Span::styled("  explicit", Style::new().fg(Color::Magenta)));
            }
            Line::from(spans)
        })
        .collect();

    // How many are left off, in the frame where the list is: a list that silently holds some of
    // itself back is a list somebody concludes is all there is.
    let title = match choices.hidden() {
        0 => format!(" {what}s "),
        1 => format!(" {what}s -- 1 more draws explicit pictures: a shows it "),
        hidden => format!(" {what}s -- {hidden} more draw explicit pictures: a shows them "),
    };
    frame.render_widget(Paragraph::new(rows).block(bordered(&title)), list);

    let enter = match choices.current() {
        Entry::OnDisk => "look for one",
        entry if entry.ready() => "use",
        _ => "fetch and use",
    };
    let explicit = match (
        choices.task.speaks(),
        choices.explicit,
        choices.hidden() > 0,
    ) {
        (false, true, _) => "   a hide explicit",
        (false, false, true) => "   a show all",
        _ => "",
    };
    draw_foot(frame, foot, doing, enter, explicit, failure);
}

/// The bar at the foot: the fetch while there is one, and otherwise the keys or what went wrong.
///
/// `enter` is what enter does on the row the cursor is on, which is not the same on all of them,
/// and `failure` is shown as it was given -- whoever set it says what it was, since by the time it
/// reaches here a fetch that failed and a file that cannot be opened look alike.
fn draw_foot(
    frame: &mut Frame,
    area: Rect,
    doing: &Doing,
    enter: &str,
    explicit: &str,
    failure: Option<&str>,
) {
    match doing {
        Doing::Fetching {
            stop,
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
                format!(" part {part} of {parts}")
            } else {
                String::new()
            };

            // Which hub, and only which hub: the address it is fetched by is the fetch's business
            // and nobody watching a bar move needs it read out to them.
            let hub = match from {
                Some(hub) => format!(" from {hub}"),
                None => String::new(),
            };
            let keys = match stop.load(Ordering::Relaxed) {
                true => "stopping...",
                false => "esc stops it",
            };
            let title = format!(" fetching {file}{hub}{of} -- {keys} ");

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
                    format!(" {message}"),
                    Style::default().fg(Color::Red),
                )),
                None => Line::from(
                    format!(
                        " up/down choose   enter {enter}   d delete{explicit}   esc back   q quit"
                    )
                    .dim(),
                ),
            };
            frame.render_widget(Paragraph::new(line).block(bordered("")), area);
        }
    }
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

#[cfg(test)]
mod tests {
    use ratatui::backend::TestBackend;

    use super::*;
    use crate::cli::tui::screen_text;

    /// The foot, drawn into a buffer, as one long string of what it says.
    fn foot(doing: &Doing) -> String {
        let mut terminal = ratatui::Terminal::new(TestBackend::new(110, 3)).unwrap();
        terminal
            .draw(|frame| draw_foot(frame, frame.area(), doing, "fetch and use", "", None))
            .unwrap();
        screen_text(&terminal)
    }

    fn fetching(done: u64, total: Option<u64>, speed: Option<f64>) -> Doing {
        let (_say, news) = channel::<Word>();
        Doing::Fetching {
            news,
            stop: Arc::new(AtomicBool::new(false)),
            name: "sdxl:wai",
            from: Some("Hugging Face"),
            file: "wai-illustrious-v17-00001-of-00004.safetensors".to_string(),
            done,
            total,
            part: 2,
            parts: 4,
            speed,
            sample: (Instant::now(), done),
        }
    }

    fn row(name: &'static str, cached: bool, explicit: bool) -> Entry {
        Entry::Published {
            name,
            full_name: "Some Model v1",
            cached,
            bytes: if cached { 6_970_000_000 } else { 0 },
            explicit,
        }
    }

    fn choices(task: Task) -> Choices {
        Choices {
            task,
            entries: vec![
                row("sdxl:wai", true, false),
                row("sdxl:noob", false, true),
                row("sdxl:base", false, false),
                Entry::OnDisk,
            ],
            selected: 0,
            explicit: false,
        }
    }

    /// The whole screen, drawn into a buffer, as one long string of what it says.
    fn screen(choices: &Choices) -> String {
        let mut terminal = ratatui::Terminal::new(TestBackend::new(120, 15)).unwrap();
        terminal
            .draw(|frame| draw(frame, choices, &Doing::Choosing, None))
            .unwrap();
        screen_text(&terminal)
    }

    #[test]
    fn the_explicit_models_are_left_off_until_they_are_asked_for_and_the_frame_says_so() {
        let mut choices = choices(Task::Txt2Img);
        let drawn = screen(&choices);
        assert!(!drawn.contains("sdxl:noob"), "{drawn}");
        assert!(drawn.contains("1 more draws explicit pictures"), "{drawn}");
        assert!(drawn.contains("a show all"), "{drawn}");

        choices.explicit = true;
        let drawn = screen(&choices);
        assert!(drawn.contains("sdxl:noob"), "{drawn}");
        assert!(drawn.contains("explicit"), "{drawn}");
        assert!(drawn.contains("a hide explicit"), "{drawn}");
        assert!(!drawn.contains("more draws"), "{drawn}");
    }

    #[test]
    fn the_cursor_is_counted_over_what_is_shown() {
        // With the explicit model hidden, the second row on screen is the third in the list.
        let mut choices = choices(Task::Txt2Img);
        choices.selected = 1;
        assert_eq!(choices.current().name(), "sdxl:base");

        // And showing it keeps the cursor on the model it was on, not on the row number.
        choices.explicit = true;
        choices.select("sdxl:base");
        assert_eq!(choices.current().name(), "sdxl:base");
        assert_eq!(choices.selected, 2);
    }

    #[test]
    fn the_list_offers_a_file_off_the_disk_under_the_published_ones() {
        let drawn = screen(&choices(Task::Txt2Img));
        assert!(drawn.contains("sdxl:wai"), "{drawn}");
        assert!(drawn.contains("a file..."), "{drawn}");
        assert!(drawn.contains("yaml"), "{drawn}");

        let choices = choices(Task::Txt2Img);
        assert!(matches!(choices.entries.last(), Some(Entry::OnDisk)));
    }

    #[test]
    fn the_list_is_the_tasks_own() {
        // Voices for reading, and no picture model among them.
        let voices = entries_for(Task::Text2Speech);
        let names: Vec<&str> = voices.iter().map(Entry::name).collect();
        assert!(names.contains(&"indextts"), "{names:?}");
        assert!(!names.contains(&"sdxl:base"), "{names:?}");

        // Pictures for drawing, and no voice among them.
        let pictures = entries_for(Task::Txt2Img);
        let names: Vec<&str> = pictures.iter().map(Entry::name).collect();
        assert!(names.contains(&"sdxl:base"), "{names:?}");
        assert!(!names.contains(&"indextts"), "{names:?}");

        // And only what can start from a picture, for the task that starts from one. Anima
        // cannot, and says so out of its kind, before a package has been looked at.
        let from_pictures = entries_for(Task::Img2Img);
        let names: Vec<&str> = from_pictures.iter().map(Entry::name).collect();
        assert!(names.contains(&"sdxl:base"), "{names:?}");
        assert!(
            !names.iter().any(|name| name.starts_with("anima")),
            "{names:?}"
        );
    }

    #[test]
    fn the_catalogue_says_which_models_draw_explicit_pictures() {
        // The list leaves those out until somebody asks, and it has nothing to go on but this: a
        // catalogue that answered without the label would be a list showing everything.
        let entries = entries_for(Task::Txt2Img);
        let explicit = |name: &str| {
            entries
                .iter()
                .find(|entry| entry.name() == name)
                .unwrap_or_else(|| panic!("{name} is offered"))
                .explicit()
        };

        assert!(explicit("sdxl:noob"));
        assert!(!explicit("sdxl:base"));
    }

    #[test]
    fn a_voice_list_has_nothing_explicit_to_show() {
        let mut choices = choices(Task::Text2Speech);
        choices.entries = vec![row("indextts", false, false), Entry::OnDisk];
        let drawn = screen(&choices);
        assert!(drawn.contains("voices"), "{drawn}");
        assert!(!drawn.contains("show all"), "{drawn}");
    }

    #[test]
    fn the_foot_says_what_enter_does_on_the_row_the_cursor_is_on() {
        let foot = |selected: usize| {
            let mut choices = choices(Task::Txt2Img);
            choices.selected = selected;
            screen(&choices)
        };

        assert!(foot(0).contains("enter use"), "{}", foot(0));
        assert!(foot(1).contains("enter fetch and use"), "{}", foot(1));
        assert!(foot(2).contains("enter look for one"), "{}", foot(2));
    }

    #[test]
    fn a_fetch_says_which_hub_it_is_talking_to_and_how_to_stop_it() {
        let drawn = foot(&fetching(1_000_000_000, Some(2_000_000_000), None));
        assert!(drawn.contains("from Hugging Face"), "{drawn}");
        assert!(drawn.contains("esc stops it"), "{drawn}");
        assert!(!drawn.contains("https://"), "{drawn}");

        let stopping = fetching(1_000_000_000, Some(2_000_000_000), None);
        if let Doing::Fetching { stop, .. } = &stopping {
            stop.store(true, Ordering::Relaxed);
        }
        assert!(foot(&stopping).contains("stopping"), "{}", foot(&stopping));
    }

    #[test]
    fn the_name_is_in_the_border_and_the_numbers_are_in_the_bar() {
        let drawn = foot(&fetching(223_000_000, Some(2_020_000_000), None));
        assert!(
            drawn.contains("wai-illustrious-v17-00001-of-00004.safetensors"),
            "{drawn}"
        );
        assert!(drawn.contains("part 2 of 4"), "{drawn}");
        assert!(drawn.contains("11%"), "{drawn}");
        assert!(drawn.contains("223 MB of 2.02 GB"), "{drawn}");
    }

    #[test]
    fn the_bar_is_painted_rather_than_spelled_out_in_blocks() {
        let mut terminal = ratatui::Terminal::new(TestBackend::new(100, 3)).unwrap();
        let doing = fetching(1_240_000_000, Some(1_740_000_000), None);
        terminal
            .draw(|frame| draw_foot(frame, frame.area(), &doing, "fetch and use", "", None))
            .unwrap();
        let buffer = terminal.backend().buffer().clone();

        for x in 1..99 {
            assert_ne!(buffer[(x, 1)].symbol(), "\u{2588}", "column {x}");
        }
        assert_eq!(buffer[(1, 1)].bg, Color::Cyan);
        assert_eq!(buffer[(98, 1)].bg, Color::Black);
    }

    #[test]
    fn a_known_rate_says_how_fast_and_how_long_is_left() {
        let drawn = foot(&fetching(1_940_000_000, Some(2_020_000_000), Some(40e6)));
        assert!(drawn.contains("40 MB/s"), "{drawn}");
        assert!(drawn.contains("2s left"), "{drawn}");

        let drawn = foot(&fetching(223_000_000, None, Some(40e6)));
        assert!(!drawn.contains('%'), "{drawn}");
        assert!(!drawn.contains("left"), "{drawn}");
    }

    #[test]
    fn time_left_is_said_the_way_someone_waiting_would_say_it() {
        assert_eq!(remaining(40_000_000, 40e6), "1s left");
        assert_eq!(remaining(4_000_000_000, 40e6), "1m40s left");
        assert_eq!(remaining(400_000_000_000, 40e6), "2h46m left");
        assert_eq!(remaining(1_000, 0.0), "");
        assert_eq!(per_second(1.5e9), "1.50 GB/s");
    }

    #[test]
    fn a_model_off_the_disk_is_called_by_its_file_name() {
        let picked = from_disk(PathBuf::from(
            "/home/someone/models/wai-illustrious-v17.yaml",
        ));
        assert_eq!(picked.label, "wai-illustrious-v17");
        assert_eq!(
            picked.model,
            "/home/someone/models/wai-illustrious-v17.yaml"
        );
    }

    #[test]
    fn what_went_wrong_is_shown_as_it_was_given() {
        let mut terminal = ratatui::Terminal::new(TestBackend::new(120, 15)).unwrap();
        terminal
            .draw(|frame| {
                draw(
                    frame,
                    &choices(Task::Txt2Img),
                    &Doing::Choosing,
                    Some("stopped -- what has come down is kept"),
                )
            })
            .unwrap();
        let drawn = screen_text(&terminal);
        assert!(drawn.contains("what has come down is kept"), "{drawn}");
        assert!(!drawn.contains("could not fetch"), "{drawn}");
    }
}
