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

//! The first screen: what the session is for.
//!
//! Asked first because everything after it depends on the answer. The models offered next are the
//! ones that can do it -- a voice cannot draw, and not every picture model can start from a
//! picture -- and the page that opens at the end is that task's page.

use ratatui::crossterm::event::{self, Event, KeyCode, KeyEvent, KeyEventKind};
use ratatui::layout::{Constraint, Layout};
use ratatui::style::{Color, Style, Stylize};
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;
use ratatui::{DefaultTerminal, Frame};

use crate::cli::task::Task;
use crate::cli::tui::ask::Answer;
use crate::cli::tui::{bordered, heading, quits, Step};

type Error = Box<dyn std::error::Error>;

/// Offers the tasks and waits for one to be picked, starting with the cursor on `at`.
pub fn choose(terminal: &mut DefaultTerminal, at: usize) -> Result<Step<Task>, Error> {
    let mut menu = Menu {
        selected: at.min(Task::ALL.len() - 1),
    };

    loop {
        terminal.draw(|frame| menu.render(frame))?;

        // Read rather than polled: nothing on this screen moves on its own, so there is nothing
        // for a tick to come back and redraw.
        let Event::Key(key) = event::read()? else {
            continue;
        };
        if key.kind != KeyEventKind::Press {
            continue;
        }

        match menu.key(key) {
            Answer::Open => {}
            // There is no screen before this one, so back is out.
            Answer::Cancelled => return Ok(Step::Quit),
            Answer::Given(index) => return Ok(Step::Next(Task::ALL[index])),
        }
    }
}

/// The screen: a cursor on the list, and nothing else to remember.
struct Menu {
    selected: usize,
}

impl Menu {
    fn key(&mut self, key: KeyEvent) -> Answer<usize> {
        let last = Task::ALL.len() - 1;
        match key.code {
            KeyCode::Enter | KeyCode::Right => return Answer::Given(self.selected),

            KeyCode::Up | KeyCode::Char('k') => self.selected = self.selected.saturating_sub(1),
            KeyCode::Down | KeyCode::Char('j') => self.selected = (self.selected + 1).min(last),
            KeyCode::Home => self.selected = 0,
            KeyCode::End => self.selected = last,

            KeyCode::Esc => return Answer::Cancelled,
            _ if quits(&key) => return Answer::Cancelled,
            _ => {}
        }

        Answer::Open
    }

    fn render(&self, frame: &mut Frame) {
        let [top, told, tasks, _, foot] = Layout::vertical([
            Constraint::Length(1),
            Constraint::Length(1),
            // Its rows and its frame, rather than the room there is: three lines inside a box
            // that reaches the foot of the screen reads as a screen that failed to fill itself.
            Constraint::Length(Task::ALL.len() as u16 + 2),
            Constraint::Min(0),
            Constraint::Length(3),
        ])
        .areas(frame.area());

        frame.render_widget(Paragraph::new(heading(&[])), top);
        frame.render_widget(
            Paragraph::new(Line::from(" What would you like to do?").dim()),
            told,
        );

        // As wide as the longest name, so that what is beside the names is a column.
        let width = Task::ALL
            .iter()
            .map(|task| task.name().len())
            .max()
            .unwrap_or(0);
        let rows: Vec<Line> = Task::ALL
            .iter()
            .enumerate()
            .map(|(index, task)| {
                let chosen = index == self.selected;
                Line::from(vec![
                    Span::raw(if chosen { "> " } else { "  " }),
                    Span::styled(
                        format!("{:<width$}", task.name()),
                        match chosen {
                            true => Style::default().fg(Color::Black).bg(Color::Cyan),
                            false => Style::default().fg(Color::Green),
                        },
                    ),
                    Span::raw("  "),
                    Span::raw(task.about()).dim(),
                ])
            })
            .collect();

        frame.render_widget(Paragraph::new(rows).block(bordered(" tasks ")), tasks);
        frame.render_widget(
            Paragraph::new(Line::from(" up/down choose   enter next   esc quit".dim()))
                .block(bordered("")),
            foot,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use ratatui::backend::TestBackend;
    use ratatui::crossterm::event::KeyModifiers;

    use crate::cli::tui::screen_text;

    fn press(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, KeyModifiers::NONE)
    }

    fn screen(menu: &Menu) -> String {
        let mut terminal = ratatui::Terminal::new(TestBackend::new(100, 15)).unwrap();
        terminal.draw(|frame| menu.render(frame)).unwrap();
        screen_text(&terminal)
    }

    #[test]
    fn the_list_offers_every_task_and_says_what_each_one_does() {
        let drawn = screen(&Menu { selected: 0 });
        for task in Task::ALL {
            assert!(
                drawn.contains(task.name()),
                "{} missing: {drawn}",
                task.name()
            );
            assert!(
                drawn.contains(task.about()),
                "{} missing: {drawn}",
                task.about()
            );
        }
        for hint in ["up/down", "enter", "esc"] {
            assert!(drawn.contains(hint), "{hint} missing from {drawn}");
        }
    }

    #[test]
    fn enter_hands_back_the_row_the_cursor_is_on() {
        let mut menu = Menu { selected: 0 };
        assert_eq!(menu.key(press(KeyCode::Down)), Answer::Open);
        assert_eq!(menu.key(press(KeyCode::Enter)), Answer::Given(1));
    }

    #[test]
    fn the_cursor_stays_on_the_list() {
        // What comes back is indexed into the list, so a cursor that walked off either end would
        // be a panic rather than a wrong row.
        let mut menu = Menu { selected: 0 };
        for _ in 0..Task::ALL.len() + 3 {
            menu.key(press(KeyCode::Down));
        }
        assert_eq!(
            menu.key(press(KeyCode::Enter)),
            Answer::Given(Task::ALL.len() - 1)
        );

        for _ in 0..Task::ALL.len() + 3 {
            menu.key(press(KeyCode::Up));
        }
        assert_eq!(menu.key(press(KeyCode::Enter)), Answer::Given(0));
    }

    #[test]
    fn leaving_is_an_answer_of_its_own() {
        for key in [
            press(KeyCode::Esc),
            press(KeyCode::Char('q')),
            KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL),
        ] {
            assert_eq!(Menu { selected: 0 }.key(key), Answer::Cancelled);
        }
    }
}
