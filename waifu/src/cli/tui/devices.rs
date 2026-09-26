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

//! The third screen: where the model runs.
//!
//! Last, because it is the one question with an answer already -- whatever `-device` said, or what
//! `auto` makes of this machine -- and most of the time enter is all it takes. Every device there
//! is has a row. One this machine cannot use says so rather than being left off: that is how
//! somebody finds out which kind of build they are running.

use ratatui::crossterm::event::{self, Event, KeyCode, KeyEvent, KeyEventKind};
use ratatui::layout::{Constraint, Layout};
use ratatui::style::{Color, Style, Stylize};
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;
use ratatui::{DefaultTerminal, Frame};

use crate::cli::args::Runtime;
use crate::cli::task::Task;
use crate::cli::tui::ask::Answer;
use crate::cli::tui::{bordered, heading, quits, row_style, Step};

type Error = Box<dyn std::error::Error>;

/// A place a run could go, and whether this machine can go there.
struct Choice {
    runtime: Runtime,
    /// Asked once, before the screens went up. The answer cannot change while they are up, and
    /// asking is a call across the C boundary that a redraw has no business making.
    available: bool,
}

/// The screen: the rows, the cursor, and why the last enter did nothing.
struct Devices {
    /// What the heading says this is for.
    task: Task,
    model: String,
    choices: Vec<Choice>,
    selected: usize,
    refused: Option<String>,
}

/// Offers every device, starting on `runtime`, and waits for one of `available`.
pub fn choose(
    terminal: &mut DefaultTerminal,
    task: Task,
    model: &str,
    runtime: Runtime,
    available: &[Runtime],
) -> Result<Step<Runtime>, Error> {
    let choices: Vec<Choice> = Runtime::ALL
        .into_iter()
        .map(|runtime| Choice {
            available: available.contains(&runtime),
            runtime,
        })
        .collect();
    let mut screen = Devices {
        task,
        model: model.to_string(),
        selected: choices
            .iter()
            .position(|choice| choice.runtime == runtime)
            .unwrap_or(0),
        choices,
        refused: None,
    };

    loop {
        terminal.draw(|frame| screen.render(frame))?;

        let Event::Key(key) = event::read()? else {
            continue;
        };
        if key.kind != KeyEventKind::Press {
            continue;
        }

        match screen.key(key) {
            Answer::Open => {}
            Answer::Cancelled if quits(&key) => return Ok(Step::Quit),
            Answer::Cancelled => return Ok(Step::Back),
            Answer::Given(runtime) => return Ok(Step::Next(runtime)),
        }
    }
}

impl Devices {
    fn key(&mut self, key: KeyEvent) -> Answer<Runtime> {
        let last = self.choices.len() - 1;
        match key.code {
            KeyCode::Up | KeyCode::Char('k') => self.selected = self.selected.saturating_sub(1),
            KeyCode::Down | KeyCode::Char('j') => self.selected = (self.selected + 1).min(last),
            KeyCode::Enter | KeyCode::Right => {
                // Said here rather than by refusing to put the cursor on it. A device this build
                // has no operators for is worth listing, and a cursor that skips a row without
                // saying why is worse than a line of text that says it.
                let chosen = &self.choices[self.selected];
                if !chosen.available {
                    self.refused = Some(format!(
                        "{} is not available on this machine",
                        chosen.runtime.name()
                    ));
                    return Answer::Open;
                }
                return Answer::Given(chosen.runtime);
            }
            KeyCode::Esc | KeyCode::Left => return Answer::Cancelled,
            _ if quits(&key) => return Answer::Cancelled,
            _ => {}
        }

        self.refused = None;
        Answer::Open
    }

    fn render(&self, frame: &mut Frame) {
        let [top, told, devices, _, foot] = Layout::vertical([
            Constraint::Length(1),
            Constraint::Length(1),
            Constraint::Length(self.choices.len() as u16 + 2),
            Constraint::Min(0),
            Constraint::Length(3),
        ])
        .areas(frame.area());

        frame.render_widget(
            Paragraph::new(heading(&[self.task.name(), &self.model])),
            top,
        );
        frame.render_widget(
            Paragraph::new(
                Line::from(" Where should it run? The page opens once this is chosen.").dim(),
            ),
            told,
        );

        let rows: Vec<Line> = self
            .choices
            .iter()
            .enumerate()
            .map(|(index, choice)| {
                let chosen = index == self.selected;
                Line::from(vec![
                    Span::raw(if chosen { "> " } else { "  " }),
                    Span::styled(
                        // As wide as the longest name, which is the offload one. A column that
                        // wanders is harder to read down than one with a gap in it.
                        format!("{:<16}", choice.runtime.name()),
                        row_style(chosen, choice.available),
                    ),
                    Span::raw("  "),
                    Span::raw(match choice.available {
                        true => choice.runtime.about(),
                        false => "not available on this machine",
                    })
                    .dim(),
                ])
            })
            .collect();
        frame.render_widget(Paragraph::new(rows).block(bordered(" device ")), devices);

        let line = match &self.refused {
            Some(refused) => Line::from(Span::styled(
                format!(" {refused}"),
                Style::default().fg(Color::Red),
            )),
            None => Line::from(" up/down choose   enter open the page   esc back   q quit".dim()),
        };
        frame.render_widget(Paragraph::new(line).block(bordered("")), foot);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use ratatui::backend::TestBackend;
    use ratatui::crossterm::event::KeyModifiers;

    use crate::cli::tui::screen_text;
    use crate::Device;

    fn press(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, KeyModifiers::NONE)
    }

    fn devices() -> Devices {
        Devices {
            task: Task::Txt2Img,
            model: "sdxl:base".to_string(),
            choices: Runtime::ALL
                .into_iter()
                .map(|runtime| Choice {
                    // Said rather than asked, so that what the screen draws is the same on a
                    // machine with a card and on one without.
                    available: runtime.device() != Device::Metal,
                    runtime,
                })
                .collect(),
            selected: 0,
            refused: None,
        }
    }

    fn screen(devices: &Devices) -> String {
        let mut terminal = ratatui::Terminal::new(TestBackend::new(100, 15)).unwrap();
        terminal.draw(|frame| devices.render(frame)).unwrap();
        screen_text(&terminal)
    }

    #[test]
    fn every_device_has_a_row_and_one_that_is_not_here_says_so() {
        let drawn = screen(&devices());
        for runtime in Runtime::ALL {
            assert!(drawn.contains(runtime.name()), "{} {drawn}", runtime.name());
        }
        assert!(drawn.contains("not available on this machine"), "{drawn}");
        assert!(drawn.contains("slower"), "{drawn}");

        // And the heading says what it is choosing for.
        assert!(drawn.contains("txt2img  >  sdxl:base"), "{drawn}");
    }

    #[test]
    fn enter_on_a_device_that_is_not_here_is_refused_with_the_reason() {
        let mut devices = devices();
        devices.selected = Runtime::ALL
            .iter()
            .position(|runtime| runtime.device() == Device::Metal)
            .unwrap();

        assert_eq!(devices.key(press(KeyCode::Enter)), Answer::Open);
        let drawn = screen(&devices);
        assert!(drawn.contains("metal is not available"), "{drawn}");

        // Moving off it clears the complaint, which was about that row.
        devices.key(press(KeyCode::Up));
        assert!(devices.refused.is_none());
    }

    #[test]
    fn enter_hands_back_the_device_the_cursor_is_on() {
        let mut devices = devices();
        devices.selected = Runtime::ALL
            .iter()
            .position(|runtime| *runtime == Runtime::CUDA_CPU_OFFLOAD)
            .unwrap();
        assert_eq!(
            devices.key(press(KeyCode::Enter)),
            Answer::Given(Runtime::CUDA_CPU_OFFLOAD)
        );
    }

    #[test]
    fn escape_goes_back_and_the_keys_that_leave_leave() {
        assert_eq!(devices().key(press(KeyCode::Esc)), Answer::Cancelled);
        assert_eq!(devices().key(press(KeyCode::Char('q'))), Answer::Cancelled);
        assert!(quits(&press(KeyCode::Char('q'))));
        assert!(!quits(&press(KeyCode::Esc)));
    }
}
