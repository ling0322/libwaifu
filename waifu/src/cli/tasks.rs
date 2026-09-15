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

//! What the tool can be asked to do, and the box that asks when the command line did not say.
//!
//! `waifu` on its own used to print the usage and exit failing, which is a fair answer to someone
//! who mistyped a command and a poor one to someone who has just installed it: the thing they came
//! for is one word away and they have been told off instead of offered it. So the words are
//! offered, on the same kind of screen the model list is offered on, and what is picked is what
//! runs.
//!
//! [`ALL`] is the one place the commands are written down. The usage prints from it, the command
//! line dispatches through it, and this box lists it, so a command cannot be in one of the three
//! and missing from the others.

use ratatui::crossterm::event::{self, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use ratatui::layout::{Constraint, Layout};
use ratatui::style::{Color, Style, Stylize};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, BorderType, Paragraph};
use ratatui::{DefaultTerminal, Frame};

use crate::cli::ask::Answer;
use crate::cli::{built_from, draw};

type Error = Box<dyn std::error::Error>;

/// One thing the tool can be asked to do: a word on the command line, a row in the box.
pub struct Task {
    /// What it is called, which is both what is typed and what the row says.
    pub name: &'static str,
    /// What it does, in the one line the usage and the row each have room for.
    pub about: &'static str,
    /// The command itself, handed everything that was not its name.
    main: fn(&[String]) -> Result<(), Error>,
}

impl Task {
    pub fn run(&self, arguments: &[String]) -> Result<(), Error> {
        (self.main)(arguments)
    }
}

/// Everything the tool can be asked to do, in the order the usage and the box list them.
///
/// One entry long for now, and a list rather than a special case for `draw` anyway: the screen
/// below is written for however many there turn out to be, and the day a second command lands is
/// not the day to go looking for the three places that named the first.
///
/// A `static` and not a `const`: a const is inlined at each use, so the table read here and the
/// table read by [`named`] would be two tables that happen to say the same thing, and the row
/// handed back would not be the row the caller can recognise.
pub static ALL: &[Task] = &[Task {
    name: "draw",
    about: "Draw a picture with your waifu",
    main: draw::main,
}];

/// The task a word names, if it names one.
pub fn named(word: &str) -> Option<&'static Task> {
    ALL.iter().find(|task| task.name == word)
}

/// Offers the tasks and waits for one to be picked.
///
/// `None` is someone leaving rather than something going wrong: they opened the list, read it and
/// pressed escape, which is not a run that failed.
pub fn choose(terminal: &mut DefaultTerminal) -> Result<Option<&'static Task>, Error> {
    let mut menu = Menu::new();

    loop {
        terminal.draw(|frame| menu.render(frame))?;

        // Read rather than polled, which is where this screen differs from the other two: nothing
        // on it moves on its own, so there is nothing for a tick to come back and redraw.
        let Event::Key(key) = event::read()? else {
            continue;
        };
        if key.kind != KeyEventKind::Press {
            continue;
        }

        match menu.key(key) {
            Answer::Open => {}
            Answer::Cancelled => return Ok(None),
            Answer::Given(index) => return Ok(Some(&ALL[index])),
        }
    }
}

/// The screen: a cursor on the list, and nothing else to remember.
struct Menu {
    selected: usize,
}

impl Menu {
    fn new() -> Menu {
        Menu { selected: 0 }
    }

    /// The same three answers the boxes in [`ask`](crate::cli::ask) give, and the same keys.
    fn key(&mut self, key: KeyEvent) -> Answer<usize> {
        let last = ALL.len().saturating_sub(1);
        match key.code {
            KeyCode::Enter => return Answer::Given(self.selected),

            KeyCode::Up | KeyCode::Left => self.selected = self.selected.saturating_sub(1),
            KeyCode::Down | KeyCode::Right => self.selected = (self.selected + 1).min(last),
            KeyCode::Home => self.selected = 0,
            KeyCode::End => self.selected = last,

            // Escape and q, as the model list takes them, and ctrl-c, which is what someone who
            // wants out of a terminal program presses before reading about either.
            KeyCode::Esc | KeyCode::Char('q') => return Answer::Cancelled,
            KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                return Answer::Cancelled
            }
            _ => {}
        }

        Answer::Open
    }

    fn render(&self, frame: &mut Frame) {
        let [built, told, tasks, _, foot] = Layout::vertical([
            Constraint::Length(1),
            Constraint::Length(1),
            // Its rows and its frame, rather than the room there is: one line inside a box that
            // reaches the foot of the screen reads as a screen that failed to fill itself.
            Constraint::Length(ALL.len() as u16 + 2),
            Constraint::Min(0),
            Constraint::Length(3),
        ])
        .areas(frame.area());

        frame.render_widget(Paragraph::new(built_from()), built);
        frame.render_widget(
            Paragraph::new(Line::from(" What would you like to do?").dim()),
            told,
        );

        // As wide as the longest name, so that what is beside the names is a column.
        let width = ALL.iter().map(|task| task.name.len()).max().unwrap_or(0);
        let rows: Vec<Line> = ALL
            .iter()
            .enumerate()
            .map(|(index, task)| {
                let chosen = index == self.selected;
                Line::from(vec![
                    Span::raw(if chosen { "> " } else { "  " }),
                    Span::styled(
                        format!("{:<width$}", task.name),
                        match chosen {
                            true => Style::default().fg(Color::Black).bg(Color::Cyan),
                            false => Style::default().fg(Color::Green),
                        },
                    ),
                    Span::raw("  "),
                    Span::raw(task.about).dim(),
                ])
            })
            .collect();

        frame.render_widget(Paragraph::new(rows).block(bordered(" tasks ")), tasks);
        frame.render_widget(
            Paragraph::new(Line::from(" up/down choose   enter start   esc quit".dim()))
                .block(bordered("")),
            foot,
        );
    }
}

fn bordered(title: &str) -> Block<'_> {
    Block::bordered()
        .border_type(BorderType::Rounded)
        .title(title)
}

#[cfg(test)]
mod tests {
    use super::*;

    use ratatui::backend::TestBackend;

    fn press(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, KeyModifiers::NONE)
    }

    /// The whole screen, drawn into a buffer, as one long string of what it says.
    fn screen(menu: &Menu) -> String {
        let mut terminal = ratatui::Terminal::new(TestBackend::new(100, 15)).unwrap();
        terminal.draw(|frame| menu.render(frame)).unwrap();

        terminal
            .backend()
            .buffer()
            .content()
            .iter()
            .map(|cell| cell.symbol())
            .collect()
    }

    #[test]
    fn the_box_offers_every_task_and_says_what_each_one_does() {
        let drawn = screen(&Menu::new());
        for task in ALL {
            assert!(
                drawn.contains(task.name),
                "{} missing from {drawn}",
                task.name
            );
            assert!(
                drawn.contains(task.about),
                "{} missing from {drawn}",
                task.about
            );
        }
    }

    #[test]
    fn the_box_says_which_keys_it_takes() {
        // It is the only thing on the terminal at this point, so what it does not say about
        // itself there is nowhere else to find out.
        let drawn = screen(&Menu::new());
        for hint in ["up/down", "enter", "esc"] {
            assert!(drawn.contains(hint), "{hint} missing from {drawn}");
        }
    }

    #[test]
    fn this_screen_says_what_built_it_too() {
        // The same line the other two carry, and for the same reason: whichever screen is up when
        // something goes wrong is the one that ends up in the screenshot.
        assert!(screen(&Menu::new()).contains("libwaifu"));
    }

    #[test]
    fn enter_hands_back_the_row_the_cursor_is_on() {
        let mut menu = Menu::new();
        assert_eq!(menu.key(press(KeyCode::End)), Answer::Open);
        assert_eq!(
            menu.key(press(KeyCode::Enter)),
            Answer::Given(ALL.len() - 1)
        );
    }

    #[test]
    fn the_cursor_stays_on_the_list() {
        // What comes back is indexed into ALL, so a cursor that walked off either end would be a
        // panic rather than a wrong row.
        let mut menu = Menu::new();
        for _ in 0..ALL.len() + 3 {
            menu.key(press(KeyCode::Down));
        }
        assert_eq!(
            menu.key(press(KeyCode::Enter)),
            Answer::Given(ALL.len() - 1)
        );

        for _ in 0..ALL.len() + 3 {
            menu.key(press(KeyCode::Up));
        }
        assert_eq!(menu.key(press(KeyCode::Enter)), Answer::Given(0));
    }

    #[test]
    fn leaving_is_an_answer_of_its_own() {
        // Not the first row, and not an error either: escape from a list of things to do is
        // someone who has decided to do none of them.
        for key in [press(KeyCode::Esc), press(KeyCode::Char('q'))] {
            assert_eq!(Menu::new().key(key), Answer::Cancelled);
        }
        assert_eq!(
            Menu::new().key(KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL)),
            Answer::Cancelled
        );
    }

    #[test]
    fn every_task_is_a_word_the_command_line_takes() {
        // The list on the screen is the dispatch table. A row that can be picked but not typed
        // would be a command that exists only on the screen.
        for task in ALL {
            assert!(
                std::ptr::eq(named(task.name).unwrap(), task),
                "{}",
                task.name
            );
        }
        assert!(named("sing").is_none());
    }
}
