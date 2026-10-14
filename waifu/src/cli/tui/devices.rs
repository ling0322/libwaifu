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

use crate::cli::args::{Room, Runtime};
use crate::cli::tui::ask::{Answer, Confirm};
use crate::cli::tui::{centred, gigabytes, row_style};

/// How wide the box is drawn, before the room it is given decides: a name, and the longest thing
/// said beside one, which is how short of memory a card is.
const WIDTH: u16 = 84;

/// A place a run could go, and whether this machine can go there.
struct Choice {
    runtime: Runtime,
    /// Asked once, before the screens went up. The answer cannot change while they are up, and
    /// asking is a call across the C boundary that a redraw has no business making.
    available: bool,
    /// What the model wants of this device's card and what the card had free, where the card is
    /// too small for it. Measured before anything was put on the card.
    short: Option<(u64, u64)>,
}

/// The box: the rows, the cursor, and why the last enter did nothing.
pub struct DevicePicker {
    /// What the warning says wants the memory.
    model: String,
    choices: Vec<Choice>,
    selected: usize,
    refused: Option<String>,
    /// Open over the list after enter on a device the model is likely to run out of memory on.
    warning: Option<Confirm>,
}

impl DevicePicker {
    /// Every device, with the cursor on `now`. One of `available` is what it hands back; one a
    /// model of `weights` bytes does not fit on, by `room`, asks before it is taken.
    pub fn open(
        model: &str,
        now: Runtime,
        available: &[Runtime],
        weights: Option<u64>,
        room: Room,
    ) -> DevicePicker {
        let choices: Vec<Choice> = Runtime::ALL
            .into_iter()
            .map(|runtime| Choice {
                available: available.contains(&runtime),
                short: runtime.short_of(weights, room),
                runtime,
            })
            .collect();

        DevicePicker {
            model: model.to_string(),
            selected: choices
                .iter()
                .position(|choice| choice.runtime == now)
                .unwrap_or(0),
            choices,
            refused: None,
            warning: None,
        }
    }

    pub fn key(&mut self, key: KeyEvent) -> Answer<Runtime> {
        // The question has every key while it is open. Escape in it is no to the question, not
        // out of the list: what somebody escaping out of a warning wants is the list it came up
        // over.
        if let Some(warning) = &mut self.warning {
            return match warning.key(key) {
                Answer::Open => Answer::Open,
                Answer::Given(true) => {
                    self.warning = None;
                    Answer::Given(self.choices[self.selected].runtime)
                }
                Answer::Given(false) | Answer::Cancelled => {
                    self.warning = None;
                    Answer::Open
                }
            };
        }

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
                // Asked rather than refused. The figure is a guess at what a run makes beside the
                // weights, and the card may have been busy with something else when it was read.
                if let Some((wanted, free)) = chosen.short {
                    self.warning = Some(self.warn(chosen.runtime, wanted, free));
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

    /// The question put before a run on `runtime`, whose card has `free` bytes of the `wanted`.
    fn warn(&self, runtime: Runtime, wanted: u64, free: u64) -> Confirm {
        let name = runtime.name();
        // The way out, where this machine has one: the same card with the weights on the host.
        let instead = self
            .choices
            .iter()
            .find(|choice| {
                choice.available
                    && choice.short.is_none()
                    && choice.runtime == Runtime::CUDA_CPU_OFFLOAD
                    && choice.runtime.device() == runtime.device()
            })
            .map(|choice| {
                format!(
                    " {} keeps the weights on the host and finishes.",
                    choice.runtime.name()
                )
            })
            .unwrap_or_default();
        Confirm::ask("may run out of memory").saying(&format!(
            "{} wants about {} on the card and {} of it is free, so a run on {name} is likely to \
             abort.{instead} Run on {name} anyway?",
            self.model,
            gigabytes(wanted),
            gigabytes(free),
        ))
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
                        // As wide as the longest name, which is the offload one. A column that
                        // wanders is harder to read down than one with a gap in it.
                        format!("{:<16}", choice.runtime.name()),
                        row_style(chosen, choice.available),
                    ),
                    Span::raw("  "),
                    match (choice.available, choice.short) {
                        (false, _) => Span::raw("not available on this machine").dim(),
                        (true, Some((wanted, free))) => Span::styled(
                            format!(
                                "may run out of memory: {} wanted, {} free",
                                gigabytes(wanted),
                                gigabytes(free)
                            ),
                            Style::default().fg(Color::Yellow),
                        ),
                        (true, None) => Span::raw(choice.runtime.about()).dim(),
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

        if let Some(warning) = &self.warning {
            warning.render(frame, area);
        }
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

    /// Every device but Metal, said rather than asked, so that what the box draws is the same on a
    /// machine with a card and on one without.
    fn here() -> Vec<Runtime> {
        Runtime::ALL
            .into_iter()
            .filter(|runtime| runtime.device() != Device::Metal)
            .collect()
    }

    fn devices() -> DevicePicker {
        DevicePicker::open("sdxl:base", Runtime::ALL[0], &here(), None, Room::default())
    }

    fn at(device: Runtime) -> usize {
        Runtime::ALL
            .iter()
            .position(|runtime| *runtime == device)
            .unwrap()
    }

    /// The list for a model of 15 GB on a card with 8 free, which is short on cuda and vulkan.
    fn too_big() -> DevicePicker {
        const GB: u64 = 1_000_000_000;
        let room = Room {
            cuda: Some(8 * GB),
            vulkan: Some(8 * GB),
        };
        let cuda = *Runtime::ALL
            .iter()
            .find(|runtime| runtime.device() == Device::Cuda)
            .unwrap();
        DevicePicker::open("sdxl:base", cuda, &here(), Some(15 * GB), room)
    }

    #[test]
    fn it_opens_on_the_device_the_model_is_on() {
        let picker = DevicePicker::open(
            "sdxl:base",
            Runtime::CUDA_CPU_OFFLOAD,
            &here(),
            None,
            Room::default(),
        );
        assert_eq!(picker.selected, at(Runtime::CUDA_CPU_OFFLOAD));
    }

    #[test]
    fn a_device_the_model_does_not_fit_on_says_so_and_asks_before_it_is_taken() {
        let mut devices = too_big();
        let drawn = screen(&devices);
        assert!(
            drawn.contains("may run out of memory: 18.65 GB wanted"),
            "{drawn}"
        );

        // Enter puts the question up instead of taking it, and says what to do instead.
        assert_eq!(devices.key(press(KeyCode::Enter)), Answer::Open);
        let drawn = screen(&devices);
        assert!(drawn.contains("may run out of memory ━"), "{drawn}");
        assert!(drawn.contains("a run on cuda is"), "{drawn}");
        assert!(drawn.contains("cuda_cpu_offload keeps the"), "{drawn}");
        assert!(drawn.contains("anyway?"), "{drawn}");

        // No, or escape, closes it and leaves the list where it was.
        assert_eq!(devices.key(press(KeyCode::Char('n'))), Answer::Open);
        assert!(devices.warning.is_none());
        devices.key(press(KeyCode::Enter));
        assert_eq!(devices.key(press(KeyCode::Esc)), Answer::Open);
        assert!(devices.warning.is_none());

        // Yes takes it all the same.
        devices.key(press(KeyCode::Enter));
        assert_eq!(
            devices.key(press(KeyCode::Char('y'))),
            Answer::Given(Runtime::ALL[devices.selected])
        );
    }

    #[test]
    fn a_device_the_model_fits_on_is_taken_without_asking() {
        let mut devices = too_big();
        devices.selected = at(Runtime::CUDA_CPU_OFFLOAD);
        assert_eq!(
            devices.key(press(KeyCode::Enter)),
            Answer::Given(Runtime::CUDA_CPU_OFFLOAD)
        );
        assert!(devices.warning.is_none());
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
        assert!(drawn.contains("slower"), "{drawn}");

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
        devices.selected = at(Runtime::CUDA_CPU_OFFLOAD);
        assert_eq!(
            devices.key(press(KeyCode::Enter)),
            Answer::Given(Runtime::CUDA_CPU_OFFLOAD)
        );
        assert_eq!(devices.key(press(KeyCode::Esc)), Answer::Cancelled);
    }

    #[test]
    fn a_box_with_no_room_for_it_still_draws() {
        let devices = too_big();
        for (width, height) in [(1u16, 1u16), (10, 3), (40, 6)] {
            let mut terminal = ratatui::Terminal::new(TestBackend::new(width, height)).unwrap();
            terminal
                .draw(|frame| devices.render(frame, frame.area()))
                .unwrap();
        }
    }
}
