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

//! A question with two answers, as a box that opens over whatever screen wanted it.
//!
//! The same shape as the file picker next door and for the same reason: no terminal, no loop, no
//! thread. The host keeps one in an `Option`, hands it the keys while it is there, gives it an
//! area to draw in, and takes the answer out of [`Answer`].
//!
//! The drawing screen that used to live beside this asked for numbers and text in boxes of the
//! same kind. That screen is a page in a browser now, and what is left to ask in the terminal is
//! whether somebody is sure.

use ratatui::crossterm::event::{KeyCode, KeyEvent};
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Color, Style, Stylize};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, BorderType, Clear, Paragraph};
use ratatui::Frame;

use crate::cli::tui::centred;

/// How wide the box is drawn, before the room it is given decides.
///
/// Wide enough for the longest line it puts at the foot, with a little to spare: a key hint that
/// runs off the end of its own box says nothing about the key it was naming.
const WIDTH: u16 = 52;

/// What a key press left a box in. Shared by every screen here, since each of them is closed the
/// same three ways.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Answer<T> {
    /// Still open. The host keeps drawing it and keeps handing it keys.
    Open,
    /// Closed with nothing answered.
    Cancelled,
    /// Closed on this answer.
    Given(T),
}

/// A question with two answers, for something worth being sure about.
pub struct Confirm {
    question: String,
    /// What is said under the question when it needs more than its title, wrapped to the box.
    body: Option<String>,
    /// Which answer the cursor is on. Starts on no, since what this is asked before is something
    /// that cannot be taken back and a key pressed by accident should not do it.
    yes: bool,
}

impl Confirm {
    pub fn ask(question: &str) -> Confirm {
        Confirm {
            question: question.to_string(),
            body: None,
            yes: false,
        }
    }

    /// The same question with something said under it, for when the title alone would not tell
    /// somebody what they are agreeing to.
    pub fn saying(mut self, body: &str) -> Confirm {
        self.body = Some(body.to_string());
        self
    }

    /// Which answer the cursor is on. Only the tests look; the answer comes back on enter.
    #[cfg(test)]
    pub fn on_yes(&self) -> bool {
        self.yes
    }

    pub fn key(&mut self, key: KeyEvent) -> Answer<bool> {
        match key.code {
            // Escaping out of a question is not an answer of yes.
            KeyCode::Esc => return Answer::Cancelled,
            KeyCode::Enter => return Answer::Given(self.yes),

            // The letters answer it outright, which is what someone who knows the question is
            // going to type rather than walking to the answer and pressing enter on it.
            KeyCode::Char('y') | KeyCode::Char('Y') => return Answer::Given(true),
            KeyCode::Char('n') | KeyCode::Char('N') => return Answer::Given(false),

            KeyCode::Left | KeyCode::Right | KeyCode::Tab | KeyCode::BackTab => {
                self.yes = !self.yes
            }
            _ => {}
        }

        Answer::Open
    }

    pub fn render(&self, frame: &mut Frame, area: Rect) {
        // Wrapped by hand rather than by the paragraph, since the box has to know how tall to be
        // before anything is drawn in it. A blank line keeps the body off the answers.
        let body = match &self.body {
            Some(body) => wrap(body, (WIDTH - 4) as usize),
            None => Vec::new(),
        };
        let said = match body.len() {
            0 => 0,
            lines => lines as u16 + 1,
        };
        let box_area = centred(area, WIDTH, 4 + said);
        frame.render_widget(Clear, box_area);

        let outline = Block::bordered()
            .border_type(BorderType::Thick)
            .border_style(Style::new().fg(Color::Yellow))
            .title(Span::styled(
                format!(" {} ", self.question),
                Style::new().fg(Color::Yellow),
            ));
        let inner = outline.inner(box_area);
        frame.render_widget(outline, box_area);
        if inner.height < 2 + said {
            return;
        }

        let [told, answers, foot] = Layout::vertical([
            Constraint::Length(said),
            Constraint::Length(1),
            Constraint::Length(1),
        ])
        .areas(inner);

        let lines: Vec<Line> = body.into_iter().map(Line::from).collect();
        frame.render_widget(Paragraph::new(lines).centered(), told);

        let answer = |label: &'static str, on: bool| {
            Span::styled(
                format!("  {label}  "),
                match on {
                    true => Style::default().fg(Color::Black).bg(Color::Cyan),
                    false => Style::default(),
                },
            )
        };
        frame.render_widget(
            Paragraph::new(Line::from(vec![
                answer("yes", self.yes),
                Span::raw("   "),
                answer("no", !self.yes),
            ]))
            .centered(),
            answers,
        );

        frame.render_widget(
            Paragraph::new(Line::from(
                "y or n   left/right choose   enter takes it".fg(Color::DarkGray),
            ))
            .centered(),
            foot,
        );
    }
}

/// `text` broken between words into lines no wider than `width`. A word longer than that is
/// left whole on a line of its own and cut off by the box, which is less wrong than splitting it.
fn wrap(text: &str, width: usize) -> Vec<String> {
    let mut lines = Vec::new();
    let mut line = String::new();
    for word in text.split_whitespace() {
        if !line.is_empty() && line.chars().count() + 1 + word.chars().count() > width {
            lines.push(std::mem::take(&mut line));
        }
        if !line.is_empty() {
            line.push(' ');
        }
        line.push_str(word);
    }
    if !line.is_empty() {
        lines.push(line);
    }
    lines
}

#[cfg(test)]
mod tests {
    use super::*;

    use ratatui::backend::TestBackend;
    use ratatui::crossterm::event::KeyModifiers;

    fn press(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, KeyModifiers::NONE)
    }

    fn drawn(render: impl Fn(&mut Frame)) -> String {
        let mut terminal = ratatui::Terminal::new(TestBackend::new(70, 12)).unwrap();
        terminal.draw(|frame| render(frame)).unwrap();
        terminal
            .backend()
            .buffer()
            .content()
            .iter()
            .map(|cell| cell.symbol())
            .collect()
    }

    #[test]
    fn a_question_starts_on_no_and_either_key_answers_it() {
        // What it is asked before cannot be taken back, so a key pressed by accident should not
        // do it: enter where it starts is no.
        let mut box_ = Confirm::ask("delete sdxl:base?");
        assert!(!box_.on_yes());
        assert_eq!(box_.key(press(KeyCode::Enter)), Answer::Given(false));

        // Walked to, and either arrow walks: there are two answers and they are side by side.
        let mut box_ = Confirm::ask("delete sdxl:base?");
        box_.key(press(KeyCode::Left));
        assert!(box_.on_yes());
        assert_eq!(box_.key(press(KeyCode::Enter)), Answer::Given(true));

        // Or answered outright, which is what someone who knows the question will do.
        let mut box_ = Confirm::ask("delete sdxl:base?");
        assert_eq!(box_.key(press(KeyCode::Char('y'))), Answer::Given(true));
        let mut box_ = Confirm::ask("delete sdxl:base?");
        assert_eq!(box_.key(press(KeyCode::Char('n'))), Answer::Given(false));

        let mut box_ = Confirm::ask("delete sdxl:base?");
        assert_eq!(box_.key(press(KeyCode::Esc)), Answer::Cancelled);
    }

    #[test]
    fn a_question_shows_both_answers_and_how_to_give_one() {
        let box_ = Confirm::ask("delete sdxl:base?");
        let screen = drawn(|frame| box_.render(frame, frame.area()));

        assert!(screen.contains("delete sdxl:base?"), "{screen}");
        assert!(screen.contains("yes"), "{screen}");
        assert!(screen.contains("no"), "{screen}");
        assert!(screen.contains("y or n"), "{screen}");
        assert!(screen.contains("enter takes it"), "{screen}");
    }

    #[test]
    fn what_is_said_under_the_question_is_wrapped_inside_the_box() {
        let box_ = Confirm::ask("not-for-all-audiences").saying(
            "The author of sdxl:noob marked it as not-for-all-audiences. It may contain potentially \
             harmful or sensitive content. Continue?",
        );
        let screen = drawn(|frame| box_.render(frame, frame.area()));

        assert!(screen.contains("not-for-all-audiences"), "{screen}");
        assert!(screen.contains("potentially"), "{screen}");
        assert!(screen.contains("Continue?"), "{screen}");
        assert!(screen.contains("enter takes it"), "{screen}");

        assert_eq!(wrap("a bb ccc", 4), vec!["a bb", "ccc"]);
        assert_eq!(wrap("", 4), Vec::<String>::new());
    }

    #[test]
    fn a_box_with_no_room_for_it_still_draws() {
        let confirm = Confirm::ask("delete sdxl:base?").saying("and everything in it");
        for (width, height) in [(1u16, 1u16), (6, 2), (20, 3)] {
            let mut terminal = ratatui::Terminal::new(TestBackend::new(width, height)).unwrap();
            terminal
                .draw(|frame| confirm.render(frame, frame.area()))
                .unwrap();
        }
    }
}
