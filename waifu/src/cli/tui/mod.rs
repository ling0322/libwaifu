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

//! The terminal half of the tool: what to do, with which model, and where.
//!
//! Three screens, one after the other -- a task, then a model for it, then a device -- and then
//! the terminal is given back and the page is served for what was picked. The page picks none of
//! the three. A model is gigabytes and a device is where they go, and both are settled here,
//! where a fetch can be watched on a bar and stopped with a key, before anything is listening.
//!
//! Escape goes back a screen, which on the first one is out; `q` and ctrl-c leave from anywhere.

mod ask;
mod devices;
mod files;
mod models;
mod tasks;

use std::io::{self, IsTerminal};

use ratatui::layout::Rect;
use ratatui::style::{Color, Style, Stylize};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, BorderType};
use ratatui::{DefaultTerminal, Frame};

use crate::cli::args::Runtime;
use crate::cli::task::{Launch, Task};

type Error = Box<dyn std::error::Error>;

/// Where a screen was left: on to the next one with what it chose, back to the one before, or
/// out of the program altogether.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Step<T> {
    Next(T),
    Back,
    Quit,
}

/// A model chosen on the second screen, and on the disk.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Picked {
    /// What to hand the worker: a catalogue name, or the path of a manifest.
    pub model: String,
    /// What the screens after it call it. A path does not fit in the heading.
    pub label: String,
}

/// Whether there is a terminal to put the screens on.
///
/// Piped or redirected -- a script, a CI job, a shell completion -- there is nobody at a keyboard
/// to pick anything with, and `ratatui::init` on something that is not a terminal panics rather
/// than reporting it.
pub fn can_take_the_screen() -> bool {
    io::stdin().is_terminal() && io::stdout().is_terminal()
}

/// Asks for a task, a model and a device, and fetches the model if it is not here.
///
/// `task` is where the task list starts, which is what `-task` said, and `runtime` is where the
/// device list starts, which is what `-device` said or what `auto` came to. Both are only where the
/// cursor starts: the screens are still shown, since the model between them is not yet chosen.
/// `None` is somebody leaving, which is not a failure.
pub fn choose(task: Option<Task>, runtime: Runtime) -> Result<Option<Launch>, Error> {
    // Asked before the screen is taken, not when the device list comes up. The first question
    // put to a device starts the tensor library, which says what hardware it found on stderr --
    // straight over whatever is on the screen at that moment.
    let available: Vec<Runtime> = Runtime::ALL
        .into_iter()
        .filter(|runtime| runtime.device().is_available())
        .collect();

    let mut terminal = take_the_screen()?;
    let chosen = walk(&mut terminal, task, runtime, &available);
    ratatui::restore();

    chosen
}

/// The three screens, with the way back through them.
fn walk(
    terminal: &mut DefaultTerminal,
    task: Option<Task>,
    runtime: Runtime,
    available: &[Runtime],
) -> Result<Option<Launch>, Error> {
    let mut task_at = task
        .and_then(|task| Task::ALL.iter().position(|one| *one == task))
        .unwrap_or(0);
    loop {
        let task = match tasks::choose(terminal, task_at)? {
            Step::Next(task) => task,
            Step::Back | Step::Quit => return Ok(None),
        };
        task_at = Task::ALL.iter().position(|one| *one == task).unwrap_or(0);

        // The model list is built again each time it is come back to, so that it says what is on
        // the disk now: a model fetched on the way through is one that is here on the way back.
        loop {
            let picked = match models::choose(terminal, task)? {
                Step::Next(picked) => picked,
                Step::Back => break,
                Step::Quit => return Ok(None),
            };

            match devices::choose(terminal, task, &picked.label, runtime, available)? {
                Step::Next(runtime) => {
                    return Ok(Some(Launch {
                        task,
                        model: picked.model,
                        runtime,
                    }))
                }
                Step::Back => continue,
                Step::Quit => return Ok(None),
            }
        }
    }
}

/// Takes the terminal and paints over whatever was on it.
///
/// The clear is the whole reason this is a function. ratatui writes only what changed since the
/// frame before, and the first frame is compared against a blank one, so a cell that is blank in
/// the drawing is a cell it never writes -- it is relying on the alternate screen it just entered
/// being empty. Which it is, unless the terminal was already in one: entering it a second time
/// changes nothing, and then every blank cell shows whatever the last program left there.
fn take_the_screen() -> Result<DefaultTerminal, Error> {
    let mut terminal = ratatui::init();
    terminal.clear()?;

    Ok(terminal)
}

/// What built this, and how far through the three screens this one is, as every screen says it.
///
/// What built it is on all of them rather than one: whichever is on screen when something goes
/// wrong is the one that ends up in a screenshot, and a screenshot that cannot say which code it
/// came from is worth much less than one that can.
fn heading(trail: &[&str]) -> Line<'static> {
    let mut spans = vec![
        Span::raw(" libwaifu").bold(),
        Span::raw("  "),
        Span::styled(crate::cli::REVISION, Style::new().fg(Color::DarkGray)),
    ];
    for step in trail {
        spans.push(Span::styled("  >  ", Style::new().fg(Color::DarkGray)));
        spans.push(Span::styled(step.to_string(), Style::new().fg(Color::Cyan)));
    }

    Line::from(spans)
}

/// A rectangle of at most `width` by `height`, in the middle of `area`.
///
/// Where every box that opens on top of a screen puts itself.
fn centred(area: Rect, width: u16, height: u16) -> Rect {
    let width = width.min(area.width);
    let height = height.min(area.height);

    Rect {
        x: area.x + (area.width - width) / 2,
        y: area.y + (area.height - height) / 2,
        width,
        height,
    }
}

fn bordered(title: &str) -> Block<'_> {
    Block::bordered()
        .border_type(BorderType::Rounded)
        .title(title)
}

/// A progress bar: a run of coloured cells, with its label centred on it.
///
/// Written out rather than left to ratatui's `Gauge`, which paints the filled part with `\u{2588}`
/// glyphs and the label's own patch with a background colour instead. The two agree on the grid
/// and not on the screen: in macOS Terminal the block glyph does not quite fill its cell, so the
/// bar comes out ribbed and steps up where the label sits on it. Colouring the background of
/// every cell the same way leaves a font nothing to disagree about.
///
/// `area` is the inside of whatever box the bar goes in; the caller draws the box.
fn bar(frame: &mut Frame, area: Rect, ratio: f64, label: &str, colour: Color) {
    if area.is_empty() {
        return;
    }

    let filled = (ratio.clamp(0.0, 1.0) * f64::from(area.width)).round() as u16;
    let text: Vec<char> = label.chars().take(area.width as usize).collect();
    let start = area.x + (area.width - text.len() as u16) / 2;
    let row = area.y + area.height / 2;

    let buffer = frame.buffer_mut();
    for y in area.top()..area.bottom() {
        for x in area.left()..area.right() {
            buffer[(x, y)].set_char(' ').set_bg(if x - area.x < filled {
                colour
            } else {
                Color::Black
            });
        }
    }

    // The label reads against whatever it lands on: black where the bar has reached it, and the
    // bar's own colour on the empty part, which is the only pair that stays legible on both.
    for (at, letter) in text.into_iter().enumerate() {
        let x = start + at as u16;
        buffer[(x, row)]
            .set_char(letter)
            .set_fg(if x - area.x < filled {
                Color::Black
            } else {
                colour
            });
    }
}

/// How a row reads: marked when it is the one chosen, and green when it is ready to use.
fn row_style(chosen: bool, ready: bool) -> Style {
    if chosen {
        Style::default().fg(Color::Black).bg(Color::Cyan)
    } else if ready {
        Style::default().fg(Color::Green)
    } else {
        Style::default()
    }
}

fn gigabytes(bytes: u64) -> String {
    if bytes >= 1_000_000_000 {
        format!("{:.2} GB", bytes as f64 / 1e9)
    } else {
        format!("{:.0} MB", bytes as f64 / 1e6)
    }
}

/// Whether a key is one that leaves the program from any screen: `q`, or ctrl-c, which is what
/// someone who wants out of a terminal program presses before reading about either.
fn quits(key: &ratatui::crossterm::event::KeyEvent) -> bool {
    use ratatui::crossterm::event::{KeyCode, KeyModifiers};

    key.code == KeyCode::Char('q')
        || (key.code == KeyCode::Char('c') && key.modifiers.contains(KeyModifiers::CONTROL))
}

/// Everything a screen drew, as one long string: what the tests read a screen with.
#[cfg(test)]
fn screen_text(terminal: &ratatui::Terminal<ratatui::backend::TestBackend>) -> String {
    terminal
        .backend()
        .buffer()
        .content()
        .iter()
        .map(|cell| cell.symbol())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    use ratatui::backend::TestBackend;
    use ratatui::widgets::Paragraph;

    #[test]
    fn the_heading_says_what_built_it_and_where_the_screens_have_got_to() {
        let mut terminal = ratatui::Terminal::new(TestBackend::new(80, 1)).unwrap();
        terminal
            .draw(|frame| {
                frame.render_widget(
                    Paragraph::new(heading(&["txt2img", "sdxl:base"])),
                    frame.area(),
                )
            })
            .unwrap();
        let drawn = screen_text(&terminal);

        assert!(drawn.contains("libwaifu"), "{drawn}");
        assert!(drawn.contains(crate::cli::REVISION), "{drawn}");
        assert!(drawn.contains("txt2img  >  sdxl:base"), "{drawn}");
    }

    #[test]
    fn a_box_is_never_larger_than_the_room_it_is_put_in() {
        let area = Rect::new(0, 0, 10, 4);
        let inside = centred(area, 50, 20);
        assert_eq!((inside.width, inside.height), (10, 4));

        let inside = centred(Rect::new(0, 0, 40, 10), 20, 4);
        assert_eq!((inside.x, inside.y), (10, 3));
    }

    #[test]
    fn sizes_read_in_the_units_a_model_is_counted_in() {
        assert_eq!(gigabytes(6_970_000_000), "6.97 GB");
        assert_eq!(gigabytes(223_000_000), "223 MB");
    }
}
