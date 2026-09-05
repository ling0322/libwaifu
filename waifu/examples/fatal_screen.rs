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

//! What a failed check looks like on a screen that has been taken over, with and without
//! [`waifu::flint::on_fatal`].
//!
//! There is no test for this. A check that fails calls `abort()`, which takes the test harness
//! with it, and what is being looked at is the state a terminal is left in rather than a value
//! anything can compare. So it is an example: run it both ways and look.
//!
//!     cargo run --features cli --example fatal_screen              # the message over the screen
//!     cargo run --features cli --example fatal_screen -- --hooked  # the screen given back first
//!
//! The first leaves the terminal in the alternate screen with the cursor hidden and raw mode on,
//! with the message written across the boxes. The second prints it to a terminal that has been put
//! back. Told apart without eyes by the escape sequences, which is what makes this checkable:
//!
//!     script -qec "cargo run -q --features cli --example fatal_screen -- --hooked" /dev/null \
//!         | cat -v | grep -o '\^\[\[?[0-9]*[hl]'
//!
//! Hooked, every sequence is balanced -- `?1049h` and `?1049l`, `?25l` and `?25h` -- and the two
//! that undo the screen come before the message. Unhooked, only the two that take it are there.

use std::io;

use ratatui::crossterm::{cursor, execute};
use ratatui::text::Line;
use ratatui::widgets::{Block, Paragraph};

use waifu::flint::{functional as F, DType, Device, Tensor};

/// The same handler the draw command installs.
extern "C" fn give_the_screen_back() {
    ratatui::restore();
    let _ = execute!(io::stdout(), cursor::Show);
}

fn main() {
    let hooked = std::env::args().any(|argument| argument == "--hooked");
    if hooked {
        waifu::flint::on_fatal(give_the_screen_back);
    }

    let mut terminal = ratatui::init();
    terminal
        .draw(|frame| {
            frame.render_widget(
                Paragraph::new(vec![
                    Line::from(""),
                    Line::from("  This is the screen the program drew."),
                    Line::from("  A check inside the tensor library is about to fail underneath"),
                    Line::from("  it, and what you can read afterwards is the whole point."),
                    Line::from(""),
                    Line::from(match hooked {
                        true => "  on_fatal is installed: the screen goes back first.",
                        false => "  on_fatal is not installed: the message lands on top of this.",
                    }),
                ])
                .block(Block::bordered().title(" waifu ")),
                frame.area(),
            );
        })
        .expect("the screen is drawn once before anything goes wrong");

    // Two tensors on two devices meeting at a convolution, which is the check that started this:
    // the draw command handed a picture read off the disk, on the host, to an encoder on the GPU.
    let input = F::rand(&[1, 3, 8, 8], DType::Float, Device::Cpu).expect("a host tensor");
    let weight = Tensor::from_f32(&[1, 3, 3, 3], &[0.0; 27])
        .and_then(|weight| weight.to_device(Device::Cuda))
        .expect("a device tensor, which needs a CUDA build to fail the way this is about");

    let _ = F::conv2d(&input, &weight, None, 1, 1, 1, 1);

    unreachable!("the check above ends the process");
}
