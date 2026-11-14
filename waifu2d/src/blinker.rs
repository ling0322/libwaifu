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

//! How open the eyes are from moment to moment: Anime2.5DRig's blink. A port of Rig.swift's
//! `Blinker`.

/// Closed in 80 ms, held 340 ms, open in 160 ms; the next one 1.6 to 5.4 s after this one began,
/// or -- one time in six or so -- right after it; the first 1.8 s in. Every change is eased by a
/// low-pass of 71 ms, and an eye just opened gives its iris a small bounce.
#[derive(Clone, Debug)]
pub struct Blinker {
    /// Seconds since it began, at its last step.
    pub now: f64,
    next_blink: f64,
    blinking: Option<f64>,
    bounced: bool,
    bounce_time: Option<f64>,
    /// How open the eyes are, 0 shut to 1 open, eased.
    pub openness: f64,
    /// The iris's scale across and down, from its bounce.
    pub bounce: [f64; 2],
}

impl Default for Blinker {
    fn default() -> Blinker {
        Blinker {
            now: 0.0,
            next_blink: 1.8,
            blinking: None,
            bounced: false,
            bounce_time: None,
            openness: 1.0,
            bounce: [1.0, 1.0],
        }
    }
}

impl Blinker {
    /// Blinks once, now, whatever the clock says.
    pub fn blink_now(&mut self) {
        if self.blinking.is_none() {
            self.blinking = Some(0.0);
            self.bounced = false;
        }
    }

    /// Where the blink is `seconds` into it: how open the eyes are, or `None` once it is over.
    pub fn shape(seconds: f64) -> Option<f64> {
        let hold = 0.34;
        if seconds < 0.08 {
            Some(1.0 - seconds / 0.08)
        } else if seconds < 0.08 + hold {
            Some(0.0)
        } else if seconds < 0.24 + hold {
            Some((seconds - 0.08 - hold) / 0.16)
        } else {
            None
        }
    }

    /// Moves on by `dt` seconds. `open` is as open as the eyes may be; `automatic` blinks on its
    /// own; `random` is 0 to 1.
    pub fn step(&mut self, dt: f64, open: f64, automatic: bool, random: &mut dyn FnMut() -> f64) {
        let dt = dt.min(0.25).max(0.0);
        self.now += dt;
        if automatic && self.blinking.is_none() && self.now > self.next_blink {
            self.blinking = Some(0.0);
            self.bounced = false;
            self.next_blink = self.now + 1.6 + random() * 3.8;
            if random() < 0.18 {
                self.next_blink = self.now + 0.28;
            }
        }
        let mut target = open;
        if let Some(mut t) = self.blinking {
            t += dt;
            if let Some(v) = Blinker::shape(t) {
                self.blinking = Some(t);
                target = target.min(v);
                if t >= 0.08 + 0.34 && v > 0.12 && !self.bounced {
                    self.bounced = true;
                    self.bounce_time = Some(0.0);
                }
            } else {
                self.blinking = None;
            }
        }
        self.openness += (target - self.openness) * (1.0 - (-dt * 14.0).exp());

        self.bounce = [1.0, 1.0];
        if let Some(mut t) = self.bounce_time {
            t += dt;
            let p = t / 0.52;
            if p >= 1.0 {
                self.bounce_time = None;
            } else {
                self.bounce_time = Some(t);
                let damp = (-2.2 * p).exp();
                let scale = 1.0 + 0.18 * (4.0 * std::f64::consts::PI * p).sin() * damp;
                let squash = 0.10
                    * (4.0 * std::f64::consts::PI * p + std::f64::consts::PI / 2.0).sin()
                    * damp;
                self.bounce = [scale * (1.0 + squash), scale * (1.0 - squash)];
            }
        }
    }
}
