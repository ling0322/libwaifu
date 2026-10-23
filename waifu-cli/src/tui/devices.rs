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

//! Where the model runs, as a box that opens over the drawing screen from its device row.
//!
//! Not a screen of its own any more. It is the one question with an answer already -- whatever
//! `-device` said, or what `auto` makes of this machine and the model -- so the drawing screen
//! opens on that answer, and this is how to change it afterwards. Every device there is has a
//! row. One this machine cannot use says so rather than being left off: that is how somebody finds
//! out which kind of build they are running.
//!
//! The same shape as the other boxes here: the host keeps one in an `Option`, hands it the keys,
//! gives it an area, and takes the device out of [`Answer`].

use ratatui::crossterm::event::{KeyCode, KeyEvent};
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Color, Style, Stylize};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, BorderType, Clear, Paragraph};
use ratatui::Frame;

use crate::args::Runtime;
use crate::tui::ask::Answer;
use crate::tui::{centred, row_style};

/// How wide the box is drawn, before the room it is given decides: a name, and the longest thing
/// said beside one.
const WIDTH: u16 = 84;

/// A place a run could go, and whether this machine can go there.
struct Choice {
    runtime: Runtime,
    /// Asked once, before the screens went up. The answer cannot change while they are up, and
    /// asking is a call across the C boundary that a redraw has no business making.
    available: bool,
}

/// The box: the rows, the cursor, and why the last enter did nothing.
pub struct DevicePicker {
    choices: Vec<Choice>,
    selected: usize,
    refused: Option<String>,
}

impl DevicePicker {
    /// Every device, with the cursor on `now`. One of `available` is what it hands back.
    ///
    /// None is too small for a model: libwaifu keeps on a card what fits of it and lets weights go
    /// when a run runs short, so there is nothing to warn about before one is taken.
    pub fn open(now: Runtime, available: &[Runtime]) -> DevicePicker {
        let choices: Vec<Choice> = Runtime::ALL
            .into_iter()
            .map(|runtime| Choice {
                available: available.contains(&runtime),
                runtime,
            })
            .collect();

        DevicePicker {
            selected: choices
                .iter()
                .position(|choice| choice.runtime == now)
                .unwrap_or(0),
            choices,
            refused: None,
        }
    }

    pub fn key(&mut self, key: KeyEvent) -> Answer<Runtime> {
        let last = self.choices.len() - 1;
        match key.code {
            KeyCode::Up | KeyCode::Char('k') => self.selected = self.selected.saturating_sub(1),
            KeyCode::Down | KeyCode::Char('j') => self.selected = (self.selected + 1).min(last),
            KeyCode::Home => self.selected = 0,
            KeyCode::End => self.selected = last,
            KeyCode::Enter => {
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
            KeyCode::Esc => return Answer::Cancelled,
            _ => {}
        }

        self.refused = None;
        Answer::Open
    }

    pub fn render(&self, frame: &mut Frame, area: Rect) {
        // The rows, a line under them for the keys or what was refused, and the frame.
        let box_area = centred(area, WIDTH, self.choices.len() as u16 + 3);
        frame.render_widget(Clear, box_area);

        let outline = Block::bordered()
            .border_type(BorderType::Thick)
            .border_style(Style::new().fg(Color::Yellow))
            .title(Span::styled(
                " device: the model is read onto it again ",
                Style::new().fg(Color::Yellow),
            ));
        let inner = outline.inner(box_area);
        frame.render_widget(outline, box_area);
        if inner.height < 2 {
            return;
        }

        let [list, foot] =
            Layout::vertical([Constraint::Min(1), Constraint::Length(1)]).areas(inner);

        let rows: Vec<Line> = self
            .choices
            .iter()
            .enumerate()
            .map(|(index, choice)| {
                let chosen = index == self.selected;
                Line::from(vec![
                    Span::raw(if chosen { "> " } else { "  " }),
                    Span::styled(
                        // Wider than the longest name. A column that wanders is harder to read
                        // down than one with a gap in it.
                        format!("{:<16}", choice.runtime.name()),
                        row_style(chosen, choice.available),
                    ),
                    Span::raw("  "),
                    match choice.available {
                        false => Span::raw("not available on this machine").dim(),
                        true => Span::raw(choice.runtime.about()).dim(),
                    },
                ])
            })
            .collect();
        frame.render_widget(Paragraph::new(rows), list);

        let line = match &self.refused {
            Some(refused) => Line::from(Span::styled(
                refused.clone(),
                Style::default().fg(Color::Red),
            )),
            None => {
                Line::from("up/down choose   enter read it there   esc close".fg(Color::DarkGray))
            }
        };
        frame.render_widget(Paragraph::new(line), foot);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use ratatui::backend::TestBackend;
    use ratatui::crossterm::event::KeyModifiers;

    use crate::runtime::Device;
    use crate::tui::screen_text;

    fn press(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, KeyModifiers::NONE)
    }

    /// Every device but Metal, said rather than asked, so that what the box draws is the same on a
    /// machine with a card and on one without.
    fn here() -> Vec<Runtime> {
        Runtime::ALL
            .into_iter()
            .filter(|runtime| runtime.device() != Device::Metal)
            .collect()
    }

    fn devices() -> DevicePicker {
        DevicePicker::open(Runtime::ALL[0], &here())
    }

    fn at(device: Runtime) -> usize {
        Runtime::ALL
            .iter()
            .position(|runtime| *runtime == device)
            .unwrap()
    }

    #[test]
    fn it_opens_on_the_device_the_model_is_on() {
        let vulkan = Runtime::on(Device::Vulkan);
        let picker = DevicePicker::open(vulkan, &here());
        assert_eq!(picker.selected, at(vulkan));
    }

    fn screen(devices: &DevicePicker) -> String {
        let mut terminal = ratatui::Terminal::new(TestBackend::new(100, 15)).unwrap();
        terminal
            .draw(|frame| devices.render(frame, frame.area()))
            .unwrap();
        screen_text(&terminal)
    }

    #[test]
    fn every_device_has_a_row_and_one_that_is_not_here_says_so() {
        let drawn = screen(&devices());
        for runtime in Runtime::ALL {
            assert!(drawn.contains(runtime.name()), "{} {drawn}", runtime.name());
        }
        assert!(drawn.contains("not available on this machine"), "{drawn}");

        // And it says what choosing one costs: the model is read again.
        assert!(drawn.contains("read onto it again"), "{drawn}");
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
    fn enter_hands_back_the_device_the_cursor_is_on_and_escape_closes() {
        let mut devices = devices();
        let vulkan = Runtime::on(Device::Vulkan);
        devices.selected = at(vulkan);
        assert_eq!(devices.key(press(KeyCode::Enter)), Answer::Given(vulkan));
        assert_eq!(devices.key(press(KeyCode::Esc)), Answer::Cancelled);
    }

    #[test]
    fn a_box_with_no_room_for_it_still_draws() {
        let devices = devices();
        for (width, height) in [(1u16, 1u16), (10, 3), (40, 6)] {
            let mut terminal = ratatui::Terminal::new(TestBackend::new(width, height)).unwrap();
            terminal
                .draw(|frame| devices.render(frame, frame.area()))
                .unwrap();
        }
    }
}
