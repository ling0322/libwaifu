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

//! Asking for one value, as a box that opens over whatever screen wanted it.
//!
//! The same shape as the file picker next door and for the same reason: no terminal, no loop, no
//! thread. The host keeps one in an `Option`, hands it the keys while it is there, gives it an
//! area to draw in, and takes the answer out of [`Answer`].
//!
//! Four of them. [`Number`] is a box to type in, for a knob whose range is too wide to walk -- a
//! step count, a guidance. [`Text`] is the same box for a value that is not a number this can do
//! arithmetic on: a seed is sixty-four bits and may be nothing at all, and neither survives a
//! trip through an f64. [`Choice`] is a short list to pick from, for a value whose settings are a
//! handful someone chose in advance. And [`Confirm`] is the one that is not a value at all but a
//! question with two answers.

use ratatui::crossterm::event::KeyCode;
use ratatui::crossterm::event::KeyEvent;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Color, Style, Stylize};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, BorderType, Clear, Paragraph};
use ratatui::Frame;

use crate::cli::centred;
use crate::cli::field::{edit_text, TextField};

/// How wide these boxes are drawn, before the room they are given decides.
///
/// Wide enough for the longest line either of them puts at the foot, with a little to spare: a
/// key hint that runs off the end of its own box says nothing about the key it was naming.
const WIDTH: u16 = 48;

/// What a key press left the box in.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Answer<T> {
    /// Still open. The host keeps drawing it and keeps handing it keys.
    Open,
    /// Closed with nothing answered.
    Cancelled,
    /// Closed on this answer.
    Given(T),
}

/// The frame both of them are drawn in.
fn outline(title: &str) -> Block<'static> {
    Block::bordered()
        .border_type(BorderType::Thick)
        .border_style(Style::new().fg(Color::Yellow))
        .title(Span::styled(
            format!(" {title} "),
            Style::new().fg(Color::Yellow),
        ))
}

/// A number typed in, refused until it is one and in range.
pub struct Number {
    title: String,
    typed: TextField,
    low: f64,
    high: f64,
    /// Whether a fraction is an answer. A step count is not divisible; a guidance is.
    whole: bool,
    /// Why the last answer was not taken, if it was not.
    trouble: Option<String>,
}

impl Number {
    /// A box asking for a number between `low` and `high`, starting on `now`.
    ///
    /// `now` is what is in the box already, so that an answer close to the current one is a
    /// character or two of typing rather than the whole number again.
    pub fn ask(title: &str, now: &str, low: f64, high: f64, whole: bool) -> Number {
        Number {
            title: title.to_string(),
            typed: TextField::new(now),
            low,
            high,
            whole,
            trouble: None,
        }
    }

    /// What is typed so far. Only the tests look, since the box on screen already shows it.
    #[cfg(test)]
    pub fn typed(&self) -> String {
        self.typed.text()
    }

    /// The answer, or why what is typed is not one.
    fn read(&self) -> Result<f64, String> {
        let text = self.typed.text();
        let text = text.trim();
        if text.is_empty() {
            return Err("there is no number here".to_string());
        }

        let Ok(value) = text.parse::<f64>() else {
            return Err(format!("{text:?} is not a number"));
        };
        // Short enough to fit the box it is drawn in, which is why it is not a sentence.
        if !(self.low..=self.high).contains(&value) {
            return Err(format!("{value} is outside {}", self.range()));
        }
        if self.whole && value.fract() != 0.0 {
            return Err(format!("{value} is not a whole number"));
        }

        Ok(value)
    }

    fn range(&self) -> String {
        match self.whole {
            true => format!("{} to {}", self.low, self.high),
            false => format!("{:.2} to {:.2}", self.low, self.high),
        }
    }

    pub fn key(&mut self, key: KeyEvent) -> Answer<f64> {
        match key.code {
            KeyCode::Esc => return Answer::Cancelled,
            KeyCode::Enter => match self.read() {
                Ok(value) => return Answer::Given(value),
                Err(trouble) => self.trouble = Some(trouble),
            },

            // Only what a number is made of, so that a stray letter is refused as it is typed
            // rather than saved up to be complained about when enter is pressed.
            _ => {
                edit_text(&mut self.typed, key.code, |character| {
                    character.is_ascii_digit() || *character == '.' || *character == '-'
                });
                self.trouble = None;
            }
        }

        Answer::Open
    }

    pub fn render(&self, frame: &mut Frame, area: Rect) {
        let box_area = centred(area, WIDTH, 4);
        frame.render_widget(Clear, box_area);

        let frame_ = outline(&self.title);
        let inner = frame_.inner(box_area);
        frame.render_widget(frame_, box_area);
        if inner.height < 2 {
            return;
        }

        let [typed, foot] =
            Layout::vertical([Constraint::Length(1), Constraint::Length(1)]).areas(inner);

        frame.render_widget(
            Paragraph::new(Line::from(vec![
                Span::raw(self.typed.text()).bold(),
                // Where the next character goes, since a box this small draws no real cursor.
                Span::styled("_", Style::new().fg(Color::Yellow)),
            ])),
            typed,
        );

        let line = match &self.trouble {
            Some(trouble) => Line::from(Span::styled(
                trouble.clone(),
                Style::default().fg(Color::Red),
            )),
            None => {
                Line::from(format!("{}   enter set   esc close", self.range()).fg(Color::DarkGray))
            }
        };
        frame.render_widget(Paragraph::new(line), foot);
    }
}

/// Something typed in that is not a number to do arithmetic on.
///
/// A seed is the case this exists for. It is sixty-four bits wide, which is more than an f64
/// carries exactly, and it may be empty -- neither of those is a thing [`Number`] can hand back,
/// and both are things a run reads straight off the characters.
pub struct Text {
    title: String,
    typed: TextField,
    /// What the box will take, which is also all it can be wrong about: whatever is in it when
    /// enter is pressed is the answer.
    accepts: fn(&char) -> bool,
    /// What belongs in it, said under the box.
    hint: String,
}

impl Text {
    pub fn ask(title: &str, now: &str, hint: &str, accepts: fn(&char) -> bool) -> Text {
        Text {
            title: title.to_string(),
            typed: TextField::new(now),
            accepts,
            hint: hint.to_string(),
        }
    }

    /// What is typed so far. Only the tests look; the box on screen already shows it.
    #[cfg(test)]
    pub fn typed(&self) -> String {
        self.typed.text()
    }

    pub fn key(&mut self, key: KeyEvent) -> Answer<String> {
        match key.code {
            KeyCode::Esc => Answer::Cancelled,
            KeyCode::Enter => Answer::Given(self.typed.text().trim().to_string()),
            _ => {
                edit_text(&mut self.typed, key.code, self.accepts);
                Answer::Open
            }
        }
    }

    pub fn render(&self, frame: &mut Frame, area: Rect) {
        let box_area = centred(area, WIDTH, 4);
        frame.render_widget(Clear, box_area);

        let frame_ = outline(&self.title);
        let inner = frame_.inner(box_area);
        frame.render_widget(frame_, box_area);
        if inner.height < 2 {
            return;
        }

        let [typed, foot] =
            Layout::vertical([Constraint::Length(1), Constraint::Length(1)]).areas(inner);

        frame.render_widget(
            Paragraph::new(Line::from(vec![
                Span::raw(self.typed.text()).bold(),
                Span::styled("_", Style::new().fg(Color::Yellow)),
            ])),
            typed,
        );
        frame.render_widget(
            Paragraph::new(Line::from(
                format!("{}   enter set   esc close", self.hint).fg(Color::DarkGray),
            )),
            foot,
        );
    }
}

/// One of a few values, picked off a list.
pub struct Choice {
    title: String,
    options: Vec<String>,
    selected: usize,
}

impl Choice {
    pub fn ask(title: &str, options: Vec<String>, selected: usize) -> Choice {
        Choice {
            title: title.to_string(),
            selected: selected.min(options.len().saturating_sub(1)),
            options,
        }
    }

    /// Which row the cursor is on. Only the tests look; the answer comes back on enter.
    #[cfg(test)]
    pub fn selected(&self) -> usize {
        self.selected
    }

    pub fn key(&mut self, key: KeyEvent) -> Answer<usize> {
        let last = self.options.len().saturating_sub(1);
        match key.code {
            KeyCode::Esc => return Answer::Cancelled,
            KeyCode::Enter => return Answer::Given(self.selected),

            KeyCode::Up | KeyCode::Left => self.selected = self.selected.saturating_sub(1),
            KeyCode::Down | KeyCode::Right => self.selected = (self.selected + 1).min(last),
            KeyCode::Home => self.selected = 0,
            KeyCode::End => self.selected = last,
            _ => {}
        }

        Answer::Open
    }

    pub fn render(&self, frame: &mut Frame, area: Rect) {
        // As tall as the list, so a handful of sizes is a handful of rows.
        let rows = self.options.len().clamp(1, 14) as u16;
        let box_area = centred(area, WIDTH, rows + 3);
        frame.render_widget(Clear, box_area);

        let frame_ = outline(&self.title);
        let inner = frame_.inner(box_area);
        frame.render_widget(frame_, box_area);
        if inner.height < 2 {
            return;
        }

        let [list, foot] =
            Layout::vertical([Constraint::Min(1), Constraint::Length(1)]).areas(inner);

        // Scrolled only as far as it takes to keep the cursor on screen, the same way the file
        // picker's list is, which for a list this short is never.
        let height = list.height as usize;
        let first = (self.selected + 1).saturating_sub(height);
        let lines: Vec<Line> = self
            .options
            .iter()
            .enumerate()
            .skip(first)
            .take(height)
            .map(|(index, option)| {
                let on = index == self.selected;
                Line::from(vec![
                    Span::raw(if on { "> " } else { "  " }),
                    Span::styled(
                        option.clone(),
                        match on {
                            true => Style::default().fg(Color::Black).bg(Color::Cyan),
                            false => Style::default(),
                        },
                    ),
                ])
            })
            .collect();

        frame.render_widget(Paragraph::new(lines), list);
        frame.render_widget(
            Paragraph::new(Line::from(
                "up/down choose   enter set   esc close".fg(Color::DarkGray),
            )),
            foot,
        );
    }
}

/// A question with two answers, for something worth being sure about.
pub struct Confirm {
    question: String,
    /// Which answer the cursor is on. Starts on no, since what this is asked before is something
    /// that cannot be taken back and a key pressed by accident should not do it.
    yes: bool,
}

impl Confirm {
    pub fn ask(question: &str) -> Confirm {
        Confirm {
            question: question.to_string(),
            yes: false,
        }
    }

    /// Which answer the cursor is on. Only the tests look; the answer comes back on enter.
    #[cfg(test)]
    pub fn on_yes(&self) -> bool {
        self.yes
    }

    pub fn key(&mut self, key: KeyEvent) -> Answer<bool> {
        match key.code {
            // Escaping out of a question about leaving is not an answer of yes.
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
        let box_area = centred(area, WIDTH, 4);
        frame.render_widget(Clear, box_area);

        let frame_ = outline(&self.question);
        let inner = frame_.inner(box_area);
        frame.render_widget(frame_, box_area);
        if inner.height < 2 {
            return;
        }

        let [answers, foot] =
            Layout::vertical([Constraint::Length(1), Constraint::Length(1)]).areas(inner);

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

#[cfg(test)]
mod tests {
    use super::*;

    use ratatui::backend::TestBackend;
    use ratatui::crossterm::event::KeyModifiers;

    fn press(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, KeyModifiers::NONE)
    }

    fn type_in<T>(keys: &mut dyn FnMut(KeyEvent) -> Answer<T>, text: &str) {
        for character in text.chars() {
            keys(press(KeyCode::Char(character)));
        }
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
    fn a_number_is_taken_when_it_is_one_and_in_range() {
        let mut box_ = Number::ask("steps", "30", 1.0, 150.0, true);

        // It starts on what it is changing, so a nudge is two keys and not the whole number.
        assert_eq!(box_.typed(), "30");
        assert_eq!(box_.key(press(KeyCode::Backspace)), Answer::Open);
        type_in(&mut |key| box_.key(key), "5");
        assert_eq!(box_.key(press(KeyCode::Enter)), Answer::Given(35.0));
    }

    #[test]
    fn what_is_not_a_number_never_gets_into_the_box() {
        // Refused as it is typed rather than saved up to be complained about at the end.
        let mut box_ = Number::ask("guidance", "", 1.0, 20.0, false);
        type_in(&mut |key| box_.key(key), "7a.b5x");
        assert_eq!(box_.typed(), "7.5");
    }

    #[test]
    fn a_number_out_of_range_is_refused_and_says_the_range() {
        let mut box_ = Number::ask("steps", "999", 1.0, 150.0, true);
        assert_eq!(box_.key(press(KeyCode::Enter)), Answer::Open);

        let screen = drawn(|frame| box_.render(frame, frame.area()));
        assert!(screen.contains("1 to 150"), "{screen}");
        assert!(screen.contains("999 is outside"), "{screen}");
    }

    #[test]
    fn a_whole_number_box_refuses_a_fraction_and_a_fractional_one_does_not() {
        let mut steps = Number::ask("steps", "12.5", 1.0, 150.0, true);
        assert_eq!(steps.key(press(KeyCode::Enter)), Answer::Open);
        assert!(drawn(|frame| steps.render(frame, frame.area())).contains("not a whole number"));

        let mut guidance = Number::ask("guidance", "12.5", 1.0, 20.0, false);
        assert_eq!(guidance.key(press(KeyCode::Enter)), Answer::Given(12.5));
    }

    #[test]
    fn an_empty_box_is_not_an_answer() {
        let mut box_ = Number::ask("steps", "", 1.0, 150.0, true);
        assert_eq!(box_.key(press(KeyCode::Enter)), Answer::Open);
        assert!(drawn(|frame| box_.render(frame, frame.area())).contains("no number here"));
    }

    #[test]
    fn typing_again_clears_what_was_wrong_with_the_last_answer() {
        // The complaint is about what was typed, so it goes as soon as that changes.
        let mut box_ = Number::ask("steps", "999", 1.0, 150.0, true);
        box_.key(press(KeyCode::Enter));
        assert!(drawn(|frame| box_.render(frame, frame.area())).contains("is outside"));

        box_.key(press(KeyCode::Backspace));
        assert!(!drawn(|frame| box_.render(frame, frame.area())).contains("is outside"));
    }

    #[test]
    fn esc_leaves_the_value_alone() {
        let mut box_ = Number::ask("steps", "30", 1.0, 150.0, true);
        assert_eq!(box_.key(press(KeyCode::Esc)), Answer::Cancelled);
    }

    #[test]
    fn a_text_box_hands_back_what_is_in_it_whatever_that_is() {
        // Nothing it could be is wrong, which is the difference from Number: a seed is sixty-four
        // bits and may be empty, and neither survives a trip through an f64.
        let mut box_ = Text::ask("seed", "", "empty is a new one", char::is_ascii_digit);
        type_in(&mut |key| box_.key(key), "18446744073709551615");
        assert_eq!(
            box_.key(press(KeyCode::Enter)),
            Answer::Given("18446744073709551615".to_string())
        );
        assert_eq!(
            "18446744073709551615".parse::<u64>().unwrap(),
            u64::MAX,
            "which is a number that has to come back whole"
        );

        // Empty is an answer.
        let mut box_ = Text::ask("seed", "123", "empty is a new one", char::is_ascii_digit);
        for _ in 0..4 {
            box_.key(press(KeyCode::Backspace));
        }
        assert_eq!(
            box_.key(press(KeyCode::Enter)),
            Answer::Given(String::new())
        );
    }

    #[test]
    fn a_text_box_takes_only_what_it_was_told_to() {
        let mut box_ = Text::ask("seed", "", "empty is a new one", char::is_ascii_digit);
        type_in(&mut |key| box_.key(key), "1a2-3.4");
        assert_eq!(box_.typed(), "1234");

        let screen = drawn(|frame| box_.render(frame, frame.area()));
        assert!(screen.contains("seed"), "{screen}");
        assert!(screen.contains("empty is a new one"), "{screen}");
        assert!(screen.contains("enter set   esc close"), "{screen}");
    }

    #[test]
    fn esc_out_of_a_text_box_leaves_the_value_alone() {
        let mut box_ = Text::ask("seed", "123", "empty is a new one", char::is_ascii_digit);
        assert_eq!(box_.key(press(KeyCode::Esc)), Answer::Cancelled);
    }

    #[test]
    fn a_choice_is_picked_off_the_list_and_stops_at_both_ends() {
        let options = vec!["512".to_string(), "768".to_string(), "1024".to_string()];
        let mut box_ = Choice::ask("size", options, 1);
        assert_eq!(box_.selected(), 1);

        for _ in 0..5 {
            box_.key(press(KeyCode::Down));
        }
        assert_eq!(box_.selected(), 2);
        for _ in 0..5 {
            box_.key(press(KeyCode::Up));
        }
        assert_eq!(box_.selected(), 0);

        box_.key(press(KeyCode::End));
        assert_eq!(box_.key(press(KeyCode::Enter)), Answer::Given(2));
    }

    #[test]
    fn a_choice_shows_what_there_is_to_choose_from() {
        let options = vec!["512".to_string(), "1024".to_string()];
        let box_ = Choice::ask("size", options, 0);

        let screen = drawn(|frame| box_.render(frame, frame.area()));
        assert!(screen.contains("size"), "{screen}");
        assert!(screen.contains("512"), "{screen}");
        assert!(screen.contains("1024"), "{screen}");
        assert!(screen.contains("esc close"), "{screen}");
    }

    #[test]
    fn the_key_hints_fit_inside_the_box_they_are_drawn_in() {
        // A hint clipped by its own border names a key nobody can read. The widest foot either
        // box has is a fractional range beside them, which is the one measured here.
        let number = Number::ask("strength", "0.80", 0.0, 1.0, false);
        let screen = drawn(|frame| number.render(frame, frame.area()));
        assert!(screen.contains("0.00 to 1.00"), "{screen}");
        assert!(screen.contains("enter set   esc close"), "{screen}");

        let choice = Choice::ask("size", vec!["1024 x 1024".to_string()], 0);
        let screen = drawn(|frame| choice.render(frame, frame.area()));
        assert!(
            screen.contains("up/down choose   enter set   esc close"),
            "{screen}"
        );
    }

    #[test]
    fn a_choice_starting_past_the_end_of_its_list_is_brought_back_to_it() {
        let box_ = Choice::ask("size", vec!["only".to_string()], 9);
        assert_eq!(box_.selected(), 0);
    }

    #[test]
    fn a_question_starts_on_no_and_either_key_answers_it() {
        // What it is asked before cannot be taken back, so a key pressed by accident should not
        // do it: enter where it starts is no.
        let mut box_ = Confirm::ask("leave waifu?");
        assert!(!box_.on_yes());
        assert_eq!(box_.key(press(KeyCode::Enter)), Answer::Given(false));

        // Walked to, and either arrow walks: there are two answers and they are side by side.
        let mut box_ = Confirm::ask("leave waifu?");
        box_.key(press(KeyCode::Left));
        assert!(box_.on_yes());
        assert_eq!(box_.key(press(KeyCode::Enter)), Answer::Given(true));

        // Or answered outright, which is what someone who knows the question will do.
        let mut box_ = Confirm::ask("leave waifu?");
        assert_eq!(box_.key(press(KeyCode::Char('y'))), Answer::Given(true));
        let mut box_ = Confirm::ask("leave waifu?");
        assert_eq!(box_.key(press(KeyCode::Char('n'))), Answer::Given(false));

        // And escaping out of a question about leaving is not an answer of yes.
        let mut box_ = Confirm::ask("leave waifu?");
        assert_eq!(box_.key(press(KeyCode::Esc)), Answer::Cancelled);
    }

    #[test]
    fn a_question_shows_both_answers_and_how_to_give_one() {
        let box_ = Confirm::ask("leave waifu?");
        let screen = drawn(|frame| box_.render(frame, frame.area()));

        assert!(screen.contains("leave waifu?"), "{screen}");
        assert!(screen.contains("yes"), "{screen}");
        assert!(screen.contains("no"), "{screen}");
        assert!(screen.contains("y or n"), "{screen}");
        assert!(screen.contains("enter takes it"), "{screen}");
    }

    #[test]
    fn a_box_with_no_room_for_it_still_draws() {
        let number = Number::ask("steps", "30", 1.0, 150.0, true);
        let choice = Choice::ask("size", vec!["512".to_string()], 0);
        let confirm = Confirm::ask("leave waifu?");
        let text = Text::ask("seed", "123", "empty is a new one", char::is_ascii_digit);
        for (width, height) in [(1u16, 1u16), (6, 2), (20, 3)] {
            let mut terminal = ratatui::Terminal::new(TestBackend::new(width, height)).unwrap();
            terminal
                .draw(|frame| number.render(frame, frame.area()))
                .unwrap();
            terminal
                .draw(|frame| choice.render(frame, frame.area()))
                .unwrap();
            terminal
                .draw(|frame| confirm.render(frame, frame.area()))
                .unwrap();
            terminal
                .draw(|frame| text.render(frame, frame.area()))
                .unwrap();
        }
    }
}
