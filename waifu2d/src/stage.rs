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

//! A rig on a page: what waifu-app/swiftui's RigView.Driver and waifu-app/wpf's Waifu2dView each
//! did by hand -- fitting the canvas into the view, turning the pointer into -1 to 1 over it,
//! picking the part under a press and dragging it, the babble, the clock -- once, for both.
//!
//! The view is measured in whatever unit the host's pointer comes in (points, DIPs), origin top
//! left, y down; the renderer is told pixels. The fit is the same at any scale, so the two never
//! need converting into each other.

use std::sync::Arc;

use crate::motion::{Controls, Frame, Motion};
use crate::render::{self, DrawList};
use crate::rig::{PartId, Rig};
use crate::speech::Speech;

/// A part being dragged: which, and where on the canvas it was taken hold of and is now.
#[derive(Clone, Copy, Debug)]
struct Drag {
    part: PartId,
    start: [f32; 2],
    now: [f32; 2],
}

pub struct Stage {
    rig: Arc<Rig>,
    motion: Motion,
    controls: Controls,
    view: [f64; 2],
    /// -1 to 1 across the canvas and down it, clamped to ±1.5; `None` while it is out of the view.
    pointer: Option<[f32; 2]>,
    drag: Option<Drag>,
    /// Seconds stepped, for the babble.
    clock: f64,
    /// The reading the mouth says, and where in it the sound being heard is; `None`, nowhere.
    speech: Option<Speech>,
    heard: Option<f64>,
}

impl Stage {
    /// `seed` drives look-around and the blinks.
    pub fn new(rig: Arc<Rig>, seed: u64) -> Stage {
        let motion = Motion::new(&rig, seed);
        Stage {
            rig,
            motion,
            controls: Controls::default(),
            view: [0.0, 0.0],
            pointer: None,
            drag: None,
            clock: 0.0,
            speech: None,
            heard: None,
        }
    }

    pub fn rig(&self) -> &Arc<Rig> {
        &self.rig
    }

    pub fn controls(&self) -> &Controls {
        &self.controls
    }

    pub fn controls_mut(&mut self) -> &mut Controls {
        &mut self.controls
    }

    pub fn motion(&self) -> &Motion {
        &self.motion
    }

    /// Where the pointer is over the canvas, as [`Motion::step`] is told it.
    pub fn pointer(&self) -> Option<[f32; 2]> {
        self.pointer
    }

    /// The part being dragged, if one is.
    pub fn dragging(&self) -> Option<PartId> {
        self.drag.map(|d| d.part)
    }

    /// The view's size, in the unit the pointer is given in.
    pub fn set_view_size(&mut self, width: f32, height: f32) {
        self.view = [width.max(0.0) as f64, height.max(0.0) as f64];
    }

    /// The pointer in the view: origin top left, y down.
    pub fn pointer_moved(&mut self, x: f32, y: f32) {
        self.pointer = self.over_canvas(x, y);
        if let (Some(drag), Some(at)) = (
            self.drag.as_mut(),
            Self::canvas_at(&self.rig, self.view, x, y),
        ) {
            drag.now = at;
        }
    }

    pub fn pointer_left(&mut self) {
        self.pointer = None;
    }

    /// Takes hold of the part under (x, y) -- a swinging one if there is one under it -- which
    /// follows the pointer until [`Stage::pointer_up`]. False where there is none.
    pub fn pointer_down(&mut self, x: f32, y: f32) -> bool {
        self.pointer = self.over_canvas(x, y);
        let Some(at) = Self::canvas_at(&self.rig, self.view, x, y) else {
            return false;
        };
        let Some(part) = self.pick(at) else {
            return false;
        };
        self.drag = Some(Drag {
            part,
            start: at,
            now: at,
        });
        self.motion.grab(Some(part), [0.0, 0.0]);
        true
    }

    pub fn pointer_up(&mut self) {
        if self.drag.take().is_some() {
            self.motion.grab(None, [0.0, 0.0]);
        }
    }

    pub fn blink_now(&mut self) {
        self.motion.blink_now();
    }

    pub fn jump(&mut self) {
        self.motion.jump();
    }

    pub fn shake(&mut self) {
        self.motion.shake();
    }

    pub fn nod_once(&mut self) {
        self.motion.nod_once();
    }

    /// The mouth shape to make from now on; `None` is the drawn mouth, at rest.
    pub fn set_viseme(&mut self, viseme: Option<crate::mouths::Viseme>) {
        self.motion.set_viseme(viseme);
    }

    /// A reading for the mouth to say from now on, at `rate` samples a second, with none of its
    /// sound yet: it says it as `speech_at` says the sound is heard. Replaces the one before.
    pub fn speak(&mut self, rate: u32) {
        self.speech = Some(Speech::new(rate));
        self.heard = None;
    }

    /// The reading's next samples, mono, -1 to 1.
    pub fn speech_samples(&mut self, samples: &[f32]) {
        if let Some(speech) = &mut self.speech {
            speech.push(samples);
        }
    }

    /// How loud the reading is, in dB a frame of `frame_ms`: libwaifu's LOUDNESS, all of it so far.
    /// Where it is given, the samples need not be.
    pub fn speech_loudness(&mut self, db: &[f32], frame_ms: u32) {
        if let Some(speech) = &mut self.speech {
            speech.set_loudness(db, frame_ms);
        }
    }

    /// The shapes its voice said, every one from its start, a frame of `frame_ms` each.
    pub fn speech_visemes(&mut self, visemes: &[i32], frame_ms: u32) {
        if let Some(speech) = &mut self.speech {
            speech.set_visemes(visemes, frame_ms);
        }
    }

    /// Where in the reading the sound being heard is, in seconds; `None` where none of it is --
    /// paused, or not begun -- and the mouth is at rest.
    pub fn speech_at(&mut self, seconds: Option<f64>) {
        self.heard = seconds.filter(|s| s.is_finite());
    }

    /// No reading any more: the mouth at rest, and the page's own shape, if it sets one.
    pub fn silence(&mut self) {
        if self.speech.take().is_some() {
            self.motion.set_mouth(None);
        }
        self.heard = None;
    }

    /// The reading the mouth says, if there is one.
    pub fn speech(&self) -> Option<&Speech> {
        self.speech.as_ref()
    }

    /// Moves on by `dt` seconds: the drag, the babble, then the motion.
    pub fn step(&mut self, dt: f64) {
        let dt = dt.max(0.0);
        self.clock += dt;
        if let Some(drag) = self.drag {
            let pull = [drag.now[0] - drag.start[0], drag.now[1] - drag.start[1]];
            self.motion.grab(Some(drag.part), pull);
        }
        let mut controls = self.controls;
        if controls.switches.auto_talk {
            controls.talk_level = controls.talk_level.max(babble(self.clock));
        }
        // A reading says the mouth: its shape as open as its sound, where the sound is heard.
        if let Some(speech) = &self.speech {
            self.motion
                .set_mouth(self.heard.and_then(|at| speech.mouth(at)));
        }
        self.motion.step(&self.rig, &controls, self.pointer, dt);
    }

    pub fn frame(&self) -> Frame {
        self.motion.frame(&self.rig, self.controls.switches.physics)
    }

    /// This frame as a backend draws it into `width` by `height` pixels.
    pub fn draw_list(&self, width: u32, height: u32) -> DrawList {
        render::draw_list(&self.rig, self.frame(), [width, height])
    }

    /// The canvas fitted into the view.
    fn fitted(rig: &Rig, view: [f64; 2]) -> Option<crate::Rect> {
        if view[0] <= 0.0 || view[1] <= 0.0 {
            return None;
        }
        let fitted = render::fit([rig.width as f64, rig.height as f64], view);
        (fitted.width > 0.0 && fitted.height > 0.0).then_some(fitted)
    }

    /// A view point as -1 to 1 across the drawing and down it, clamped to ±1.5.
    fn over_canvas(&self, x: f32, y: f32) -> Option<[f32; 2]> {
        let fitted = Self::fitted(&self.rig, self.view)?;
        let u = (x as f64 - fitted.x) / fitted.width * 2.0 - 1.0;
        let v = (y as f64 - fitted.y) / fitted.height * 2.0 - 1.0;
        Some([u.clamp(-1.5, 1.5) as f32, v.clamp(-1.5, 1.5) as f32])
    }

    /// A view point in canvas pixels, the space the rig's vertices are in.
    fn canvas_at(rig: &Rig, view: [f64; 2], x: f32, y: f32) -> Option<[f32; 2]> {
        let fitted = Self::fitted(rig, view)?;
        Some([
            ((x as f64 - fitted.x) / fitted.width * rig.width as f64) as f32,
            ((y as f64 - fitted.y) / fitted.height * rig.height as f64) as f32,
        ])
    }

    /// The part under a canvas point, topmost first: a swinging one -- or a skirt -- if there is
    /// one under it, else the topmost of any. By the box of its grid at rest, as Waifu2dView did.
    fn pick(&self, at: [f32; 2]) -> Option<PartId> {
        let mut any = None;
        for (i, part) in self.rig.parts.iter().enumerate().rev() {
            if part.mesh.rest.is_empty() {
                continue;
            }
            let (mut lo, mut hi) = ([f32::MAX; 2], [f32::MIN; 2]);
            for p in &part.mesh.rest {
                lo = [lo[0].min(p[0]), lo[1].min(p[1])];
                hi = [hi[0].max(p[0]), hi[1].max(p[1])];
            }
            if at[0] < lo[0] || at[0] > hi[0] || at[1] < lo[1] || at[1] > hi[1] {
                continue;
            }
            let id = PartId(i as u32);
            if part.swing.is_some() || part.slot == "bottomwear" {
                return Some(id);
            }
            any.get_or_insert(id);
        }
        any
    }
}

/// Made-up talk at `t` seconds, 0 to 1: bursts of babble with gaps, as speech comes and goes.
/// Waifu2dView's, which ran off the wall clock; this runs off the stage's.
fn babble(t: f64) -> f32 {
    let gate = ((t * 0.9).sin() * 0.6 + (t * 0.31 + 1.0).sin() * 0.5).max(0.0);
    let babble = ((t * 21.0).sin() * 0.55 + (t * 33.0 + 0.7).sin() * 0.45).abs();
    (gate * babble) as f32
}
