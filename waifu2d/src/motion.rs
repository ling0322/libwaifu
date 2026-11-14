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

//! How a rig moves, frame by frame, as Anime2.5DRig moves one (lib/app.js, MIT, Copyright (c) 2026
//! hakoniwa): what it is told to do -- where it looks, how open its eyes are, how it breathes --
//! and every vertex of every layer moved by that.
//!
//! The head turns about the neck and every layer of it moves by how far in front of the face it
//! is, which is what makes it look turned rather than slid: front hair more than the face, back
//! hair less. The body follows the head a little and leans about its foot. Hair swings in strands,
//! each a stiff spring at the root and a soft one at the tip, after the head; an earring or a tail
//! sways.
//!
//! A port of waifu-app/swiftui's RigMotion.swift, with what waifu-app/wpf's RigMotion.cs added to
//! it -- the mouth, smile, brows, twist, winks, wind, the jump, shake, nod and grab -- each of
//! which does nothing at its default: at its defaults this is the Mac's motion, number for number.

use std::f32::consts::PI;

use crate::blinker::Blinker;
use crate::geom::smooth;
use crate::rig::{Group, PartId, Phys, Rig, Role, Side};
use crate::rng::Rng;

/// What moves on its own. Every default is the Mac's.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Switches {
    pub follow_pointer: bool,
    /// Sways a little, all the time.
    pub idle: bool,
    /// Looks somewhere else now and then.
    pub look_around: bool,
    /// The hair swings.
    pub physics: bool,
    /// Blinks on its own.
    pub blink: bool,
    /// The wind in the hair. The Mac has no switch of its own for it: it sets it with `idle`.
    pub wind: bool,
    /// Babble: the mouth moving with no sound.
    pub auto_talk: bool,
    /// The Windows port's wind: gusts that come and go over `wind_strength`, and a skirt's hem
    /// (a `bottomwear` layer) swung by them and by the body. Off, as the Mac has it; on, the wind
    /// is never the Mac's, whatever its strength.
    pub gusts: bool,
}

impl Default for Switches {
    fn default() -> Switches {
        Switches {
            follow_pointer: true,
            idle: true,
            look_around: true,
            physics: true,
            blink: true,
            wind: true,
            auto_talk: false,
            gusts: false,
        }
    }
}

/// Everything the host's settings say. Every default is what the Mac does today; the ones after
/// `body_follows` are the Windows port's and do nothing at their defaults.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Controls {
    pub switches: Switches,
    /// As open as the eyes may be, 0 shut to 1 open.
    pub eyes_open: f32,
    /// How soon the open eye gives way to the closed one, 0 to 1: Anime2.5DRig's "closes easily".
    pub closes_at: f32,
    /// How far the head turns up and down: 1 as the rig turns it, 0 not at all; to 2.
    pub nod: f32,
    /// How far the body turns and leans after the head: 1 as the rig turns it; 0 to 2.
    pub body_follows: f32,
    /// The microphone's loudness, 0 to 1, which the host measures.
    pub talk_level: f32,
    /// The mouth as open as this, set by hand, 0 to 1.
    pub mouth_open: f32,
    /// -1 to 1.
    pub smile: f32,
    /// -1 to 1.
    pub brows: f32,
    /// -1 to 1.
    pub twist: f32,
    /// 0 to 1: how far the eye is winked shut.
    pub wink_left: f32,
    pub wink_right: f32,
    /// How gusty the wind is, 0 to 3.
    pub wind_strength: f32,
}

impl Default for Controls {
    fn default() -> Controls {
        Controls {
            switches: Switches::default(),
            eyes_open: 1.0,
            closes_at: 0.3,
            nod: 1.0,
            body_follows: 1.0,
            talk_level: 0.0,
            mouth_open: 0.0,
            smile: 0.0,
            brows: 0.0,
            twist: 0.0,
            wink_left: 0.0,
            wink_right: 0.0,
            wind_strength: 1.0,
        }
    }
}

/// What the rig is told to do, this frame. Angles and the eyes' look are -1 to 1.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Pose {
    pub angle_x: f32,
    pub angle_y: f32,
    pub angle_z: f32,
    pub body: f32,
    pub eye_x: f32,
    pub eye_y: f32,
    pub openness: f32,
    /// The iris's scale across and down, from its bounce after a blink.
    pub bounce: [f32; 2],
    /// 0 to 1: the chest's breath, and the head's, which lags it.
    pub breath: f32,
    pub breath_head: f32,
    /// How soon the open eye gives way to the closed one, 0 to 1.
    pub ease: f32,
    /// How far the head turns up and down, 1 as the rig turns it.
    pub nod: f32,
    /// How far the upper body has turned after the head, -1 to 1 as the head's turn is.
    pub body_turn: f32,
    // The Windows port's, each nothing at its default.
    pub mouth: f32,
    pub smile: f32,
    pub brow: f32,
    pub wink_left: f32,
    pub wink_right: f32,
    pub twist: f32,
    pub jump_y: f32,
    pub squash: f32,
    pub grab: [f32; 2],
}

impl Default for Pose {
    fn default() -> Pose {
        Pose {
            angle_x: 0.0,
            angle_y: 0.0,
            angle_z: 0.0,
            body: 0.0,
            eye_x: 0.0,
            eye_y: 0.0,
            openness: 1.0,
            bounce: [1.0, 1.0],
            breath: 0.0,
            breath_head: 0.0,
            ease: 0.3,
            nod: 1.0,
            body_turn: 0.0,
            mouth: 0.0,
            smile: 0.0,
            brow: 0.0,
            wink_left: 0.0,
            wink_right: 0.0,
            twist: 0.0,
            jump_y: 0.0,
            squash: 1.0,
            grab: [0.0, 0.0],
        }
    }
}

/// A spring after a target: Anime2.5DRig's, semi-implicit Euler in steps of at most 1/120 s, so
/// that it is the same at 10 frames a second as at 240.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Spring {
    pub x: f32,
    pub v: f32,
}

impl Spring {
    pub fn step(&mut self, target: f32, k: f32, damping: f32, dt: f32) {
        let count = ((dt * 120.0).ceil() as i64).max(1);
        let h = dt / count as f32;
        for _ in 0..count {
            self.v += (-k * (self.x - target) - damping * self.v) * h;
            self.x += self.v * h;
        }
    }
}

/// A strand's two springs, and how far each lags its target.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StrandSprings {
    pub stiff: Spring,
    pub soft: Spring,
    pub phase: f32,
    pub stiff_lag: f32,
    pub soft_lag: f32,
}

impl StrandSprings {
    fn new(phase: f32) -> StrandSprings {
        StrandSprings {
            stiff: Spring::default(),
            soft: Spring::default(),
            phase,
            stiff_lag: 0.0,
            soft_lag: 0.0,
        }
    }
}

/// Where it looks next, when it looks around: the angles, the body and the eyes.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct Look {
    angle_x: f32,
    angle_y: f32,
    angle_z: f32,
    body: f32,
    eye_x: f32,
    eye_y: f32,
}

/// The moving rig's state from one frame to the next.
#[derive(Clone, Debug)]
pub struct Motion {
    /// Seconds it has moved for.
    pub time: f32,
    /// Where the pose is, eased toward where it is told to be.
    pub pose: Pose,
    pub blinker: Blinker,
    rng: Rng,
    /// Each part's strands' springs, part by part: empty for one that does not swing.
    strands: Vec<Vec<StrandSprings>>,
    bust: Spring,
    bust_lag: f32,

    look: Look,
    next_look: f32,

    // The Windows port's, each of which does nothing until it is told to.
    /// How much the hem of a skirt swings, a spring like the bust's.
    hem: Spring,
    hem_lag: f32,
    /// The Windows port's gusts, and the hem they and the body swing. Off, as the Mac has it.
    gusts: bool,
    gust_phase: f32,
    /// A jump: how far up the rig is, negative above the ground, and how fast it is going down.
    jump_y: f32,
    jump_v: f32,
    /// The squash a landing gives, 1 at rest, and how fast it is changing.
    squash: f32,
    squash_v: f32,
    /// Seconds into a shake or a nod told to happen once, or negative while there is none.
    shake_t: f32,
    nod_t: f32,
    /// A part being dragged about, and the pull in canvas pixels.
    grabbed: Option<PartId>,
    grab_pull: [f32; 2],
}

impl Motion {
    /// `seed` drives look-around and the blinker, so a test sees the same frames every run.
    pub fn new(rig: &Rig, seed: u64) -> Motion {
        let strands = rig
            .parts
            .iter()
            .enumerate()
            .map(|(index, part)| {
                let count = part.swing.as_ref().map_or(0, |swing| swing.strands.len());
                (0..count)
                    .map(|s| StrandSprings::new(s as f32 * 1.37 + index as f32))
                    .collect()
            })
            .collect();
        Motion {
            time: 0.0,
            pose: Pose::default(),
            blinker: Blinker::default(),
            rng: Rng::new(seed),
            strands,
            bust: Spring::default(),
            bust_lag: 0.0,
            look: Look::default(),
            next_look: 0.0,
            hem: Spring::default(),
            hem_lag: 0.0,
            gusts: false,
            gust_phase: 2.1,
            jump_y: 0.0,
            jump_v: 0.0,
            squash: 1.0,
            squash_v: 0.0,
            shake_t: -1.0,
            nod_t: -1.0,
            grabbed: None,
            grab_pull: [0.0, 0.0],
        }
    }

    /// Moves on by `dt` seconds, clamped to 0.05. `pointer` is where the pointer is over the
    /// canvas, -1 to 1 across it and down it and clamped to ±1.5, or `None` where it is not over it.
    pub fn step(&mut self, rig: &Rig, controls: &Controls, pointer: Option<[f32; 2]>, dt: f64) {
        let switches = controls.switches;
        // Kept for `deform`, which is not told the controls.
        self.gusts = switches.gusts;
        let pointer = pointer.map(|[x, y]| [x.max(-1.5).min(1.5), y.max(-1.5).min(1.5)]);
        let dt = dt.min(0.05).max(0.0) as f32;
        self.time += dt;
        let t = self.time;
        let mut target = Pose::default();

        // Where it is told to look: after the pointer, or wherever it looks next.
        if switches.follow_pointer {
            if let Some(m) = pointer {
                target.angle_x = clamp(m[0] * 0.9);
                target.angle_y = clamp(-m[1] * 0.7);
                target.eye_x = clamp(m[0] * 1.2);
                target.eye_y = clamp(-m[1] * 0.8);
            }
        }
        if switches.idle {
            target.angle_x += 0.13 * (t * 0.42).sin() + 0.05 * (t * 1.13).sin();
            target.angle_y += 0.08 * (t * 0.31 + 1.7).sin();
            target.angle_z += 0.07 * (t * 0.23 + 0.5).sin();
            target.body += 0.10 * (t * 0.19 + 2.1).sin();
        }
        // Anime2.5DRig looks around over the pointer too; here it does not while it follows one,
        // which it would only wander from.
        if switches.look_around && !(switches.follow_pointer && pointer.is_some()) {
            if t > self.next_look {
                let rng = &mut self.rng;
                self.next_look = t + 1.4 + rng.next_f32() * 2.6;
                self.look.angle_x = (rng.next_f32() * 2.0 - 1.0) * 0.55;
                self.look.angle_y = (rng.next_f32() * 2.0 - 1.0) * 0.40;
                self.look.angle_z = (rng.next_f32() * 2.0 - 1.0) * 0.35;
                self.look.body = (rng.next_f32() * 2.0 - 1.0) * 0.30;
                self.look.eye_x = (rng.next_f32() * 2.0 - 1.0) * 0.60;
                self.look.eye_y = (rng.next_f32() * 2.0 - 1.0) * 0.35;
            }
            target.angle_x += self.look.angle_x;
            target.angle_y += self.look.angle_y;
            target.angle_z += self.look.angle_z;
            target.body += self.look.body;
            target.eye_x += self.look.eye_x;
            target.eye_y += self.look.eye_y;
        }

        // The body leans the way the head turns, as Anime2.5DRig's does after a tracked face.
        let follow = controls.body_follows;
        target.body += clamp(target.angle_x) * 0.25 * follow;

        let rng = &mut self.rng;
        self.blinker.step(
            dt as f64,
            controls.eyes_open as f64,
            switches.blink,
            &mut || rng.next_f64(),
        );

        // Eased, 71 ms; the body after the head, over 170.
        let k = 1.0 - (-dt * 14.0).exp();
        let pose = &mut self.pose;
        pose.angle_x += (clamp(target.angle_x) - pose.angle_x) * k;
        pose.angle_y += (clamp(target.angle_y) - pose.angle_y) * k;
        pose.angle_z += (clamp(target.angle_z) - pose.angle_z) * k;
        pose.body += (clamp(target.body) - pose.body) * (1.0 - (-dt * 6.0).exp());
        // How far the body has turned after the head: the head's turn, eased the body's way.
        pose.body_turn +=
            (clamp(target.angle_x) * follow - pose.body_turn) * (1.0 - (-dt * 6.0).exp());
        pose.eye_x += (clamp(target.eye_x) - pose.eye_x) * k;
        pose.eye_y += (clamp(target.eye_y) - pose.eye_y) * k;
        pose.openness = self.blinker.openness as f32;
        pose.bounce = [self.blinker.bounce[0] as f32, self.blinker.bounce[1] as f32];
        pose.breath = 0.5 + 0.5 * (t * 2.0 * PI / 3.4).sin();
        pose.breath_head = 0.5 + 0.5 * (t * 2.0 * PI / 3.4 - 0.6).sin();
        pose.ease = controls.closes_at;
        pose.nod = controls.nod;

        // The mouth, as loud as the talk is unless set by hand: fast to open, slow to shut.
        let mouth_target = controls.talk_level.max(controls.mouth_open);
        let mk = 1.0
            - (-dt
                * if mouth_target > pose.mouth {
                    30.0
                } else {
                    10.0
                })
            .exp();
        pose.mouth += (mouth_target - pose.mouth) * mk;
        let ek = 1.0 - (-dt * 10.0).exp();
        pose.smile += (controls.smile - pose.smile) * ek;
        pose.brow += (controls.brows - pose.brow) * ek;
        pose.twist += (controls.twist - pose.twist) * ek;
        pose.wink_left += (controls.wink_left - pose.wink_left) * ek;
        pose.wink_right += (controls.wink_right - pose.wink_right) * ek;

        // A jump: up off the body pivot, down with gravity, squashing as it lands and stretching
        // back.
        if self.jump_y < 0.0 || self.jump_v != 0.0 {
            self.jump_y += self.jump_v * dt;
            self.jump_v += 900.0 * rig.face_scale as f32 * dt;
            if self.jump_y >= 0.0 {
                self.jump_y = 0.0;
                self.jump_v = 0.0;
                self.squash_v -= 4.5;
            }
        }
        // The squash, critically not damped: it overshoots, as a landing does.
        self.squash_v += (1.0 - self.squash) * 220.0 * dt;
        self.squash_v *= (-dt * 9.0).exp();
        self.squash += self.squash_v * dt;
        pose.jump_y = self.jump_y;
        pose.squash = self.squash;

        // A shake or a nod told to happen once, decaying away.
        if self.shake_t >= 0.0 {
            self.shake_t += dt;
            let s = self.shake_t;
            pose.angle_x += (s * 34.0).sin() * (-s * 3.5).exp() * 0.55;
            if s > 1.6 {
                self.shake_t = -1.0;
            }
        }
        if self.nod_t >= 0.0 {
            self.nod_t += dt;
            let s = self.nod_t;
            pose.angle_y += (s * 12.0).sin() * (-s * 2.2).exp() * 0.5 * pose.nod;
            if s > 1.4 {
                self.nod_t = -1.0;
            }
        }

        // Where a grabbed part is pulled to, easing after the pointer; let go, back to nothing.
        pose.grab = if self.grabbed.is_some() {
            let g = 1.0 - (-dt * 18.0).exp();
            [
                pose.grab[0] + (self.grab_pull[0] - pose.grab[0]) * g,
                pose.grab[1] + (self.grab_pull[1] - pose.grab[1]) * g,
            ]
        } else {
            let g = (-dt * 10.0).exp();
            [pose.grab[0] * g, pose.grab[1] * g]
        };

        self.step_springs(rig, dt, switches.wind, controls.wind_strength);
    }

    /// Each strand's springs after where the head or the body has gone, and the wind; and the
    /// bust's, and a skirt's hem.
    fn step_springs(&mut self, rig: &Rig, dt: f32, wind: bool, strength: f32) {
        let time = self.time;
        // The wind as strong as it is told to be; with the Windows port's gusts, coming and going
        // over that in slow noise.
        let mut gust = strength.max(0.0);
        if self.gusts {
            gust *= 0.6
                + 0.4 * (time * 0.53 + self.gust_phase).sin()
                + 0.25 * (time * 1.31 + 2.0).sin();
        }
        let a = &rig.anchors;
        let pose = &self.pose;
        let fs = rig.face_scale as f32;
        let head_dx = (pose.angle_x * 14.0
            + pose.angle_z * 0.07 * (a.neck_pivot.y - a.face_center.y) as f32)
            * fs;
        let body_dx = pose.body * 0.028 * (a.body_pivot.y - a.neck_pivot.y) as f32 * 0.35
            + pose.angle_x * 14.0 * 0.16 * fs;
        for (part, springs) in rig.parts.iter().zip(self.strands.iter_mut()) {
            if springs.is_empty() {
                continue;
            }
            let base = if part.group == Group::Head {
                head_dx
            } else {
                body_dx
            };
            let sways = part.swing.as_ref().map(|swing| swing.kind) == Some(Phys::Sway);
            let wind_scale: f32 = if sways && part.group != Group::Head {
                2.5
            } else {
                1.0
            };
            for strand in springs.iter_mut() {
                let phase = strand.phase;
                let w = if wind {
                    (1.8 * (time * 0.8 + phase).sin() + 1.0 * (time * 1.9 + phase * 2.3).sin())
                        * wind_scale
                        * gust
                } else {
                    0.0
                };
                let target = base + w * fs;
                strand.stiff.step(target, 70.0, 9.0, dt);
                strand.stiff_lag = -(strand.stiff.x - target) * 2.2;
                strand.soft.step(target, 16.0, 1.3, dt);
                strand.soft_lag = -(strand.soft.x - target) * 3.0;
            }
        }
        let bust_target = (pose.breath * 3.0 - pose.angle_y * 6.0 + pose.body * 4.0) * fs;
        self.bust.step(bust_target, 140.0, 4.2, dt);
        self.bust_lag = -(self.bust.x - bust_target) * 3.0;

        // A skirt's hem, swung like the strands but as one: after the body and the gusts.
        let hem_target = body_dx * 1.4
            + if wind {
                gust * 2.6 * (time * 0.9 + 1.2).sin() * fs
            } else {
                0.0
            };
        self.hem.step(hem_target, 26.0, 2.0, dt);
        self.hem_lag = -(self.hem.x - hem_target) * 2.4;
    }

    pub fn blink_now(&mut self) {
        self.blinker.blink_now();
    }

    /// Told to jump: an upward speed, which gravity turns into a hop with a squashy landing.
    pub fn jump(&mut self) {
        if self.jump_y >= 0.0 {
            self.jump_v = -300.0;
        }
    }

    /// Told to shake the head once.
    pub fn shake(&mut self) {
        self.shake_t = 0.0;
    }

    /// Told to nod once.
    pub fn nod_once(&mut self) {
        self.nod_t = 0.0;
    }

    /// The part pulled `pull` canvas pixels from where it was grabbed; `None` lets go.
    pub fn grab(&mut self, part: Option<PartId>, pull: [f32; 2]) {
        // Let go: fling the part's soft springs back with the pull it was under.
        if part.is_none() {
            if let Some(springs) = self.grabbed.and_then(|g| self.strands.get_mut(g.index())) {
                for strand in springs.iter_mut() {
                    strand.soft.v -= self.pose.grab[0] * 4.0;
                }
            }
        }
        self.grabbed = part;
        self.grab_pull = pull;
    }

    pub fn pose(&self) -> &Pose {
        &self.pose
    }

    /// Part `part`'s strands' springs: empty for one that does not swing.
    pub fn strands(&self, part: PartId) -> &[StrandSprings] {
        &self.strands[part.index()]
    }

    /// How far the bust lags its target.
    pub fn bust_lag(&self) -> f32 {
        self.bust_lag
    }

    /// Every part where it is now, and how much of it shows.
    pub fn frame(&self, rig: &Rig, physics: bool) -> Frame {
        let ids = (0..rig.parts.len()).map(|i| PartId(i as u32));
        Frame::from_parts(
            ids.clone()
                .map(|id| self.deform(rig, id, physics))
                .collect(),
            ids.map(|id| Motion::alpha(rig, id, &self.pose)).collect(),
        )
    }

    /// How much of a part shows: an open eye's parts fading into its closed eye as it shuts, or as
    /// it is winked.
    pub fn alpha(rig: &Rig, part: PartId, pose: &Pose) -> f32 {
        let part = rig.part(part);
        let opacity = part.opacity as f32;
        // The long closed eye, for a long blink, is not blinked with.
        if part.slot == "eye_close2" {
            return 0.0;
        }
        if let Some(side) = part.side {
            if rig.eye(side).is_some() {
                // A wink shuts its own eye, as `deform` does: faded by the openness winked.
                let v = pose.openness * (1.0 - wink(pose, side));
                let opened = smooth32((v - (0.10 + pose.ease * 0.45)) / 0.15);
                match part.role {
                    Role::EyeWhite | Role::Irides | Role::EyeLash => return opacity * opened,
                    Role::EyeClose => return opacity * (1.0 - opened),
                    _ => {}
                }
            }
        }
        opacity
    }

    /// Where each vertex of part `part` is now, on the canvas: Anime2.5DRig's deform, in its order
    /// -- the eyes, the head turned, the breath, the chest, the strands, and the body's lean last;
    /// and among them the Windows port's mouth, brows, twist, hem, grab and jump.
    pub fn deform(&self, rig: &Rig, part: PartId, physics: bool) -> Vec<[f32; 2]> {
        let index = part.index();
        let part = rig.part(part);
        let e = &self.pose;
        let a = &rig.anchors;
        let fs = rig.face_scale as f32;
        let np = [a.neck_pivot.x as f32, a.neck_pivot.y as f32];
        let bp = [a.body_pivot.x as f32, a.body_pivot.y as f32];
        let face_center_y = a.face_center.y as f32;
        // The face's surface, which the head turns up and down on: an ellipse over the face's box
        // about its middle, as high in the middle as 0.3 of its width and nothing at its edge.
        let surface_center = [a.face_center.x as f32, face_center_y];
        let surface_radius = [
            (a.face.width as f32 / 2.0).max(1.0),
            (a.face.height as f32 / 2.0).max(1.0),
        ];
        let surface_height = a.face.width as f32 * 0.3;
        let surface_width = a.face.width as f32;
        let az = e.angle_z * 0.07;
        let (cz, sz) = (az.cos(), az.sin());
        let ab = e.body * 0.028;
        let (cb, sb) = (ab.cos(), ab.sin());
        let eye = part.side.and_then(|side| rig.eye(side));
        let eye_role = if eye.is_some() {
            part.role
        } else {
            Role::Other
        };
        // How open the eyes are, and a wink shuts its own eye over that.
        let mut v = e.openness;
        if let Some(side) = part.side {
            v *= 1.0 - wink(e, side);
        }
        let chest = [a.chest.x as f32, a.chest.y as f32];
        let reach = [a.chest_reach.width as f32, a.chest_reach.height as f32];
        let neck_top = a.neck_top as f32;
        let neck_bottom = a.neck_bottom as f32;
        let is_top = part.slot == "topwear";

        // The mouth's own box, for opening and smiling: its mesh's rest bounds. Moved only while
        // it is open or smiling, so that at rest it is exactly where the Mac has it.
        let mouth = (part.slot == "mouth_open" || part.slot == "mouth")
            && !part.mesh.rest.is_empty()
            && (e.mouth != 0.0 || e.smile != 0.0);
        let (mut mouth_min, mut mouth_size, mut mouth_center) =
            ([0.0f32; 2], [1.0f32; 2], [0.0f32; 2]);
        if mouth {
            let (lo, hi) = bounds(&part.mesh.rest);
            mouth_min = lo;
            mouth_size = [(hi[0] - lo[0]).max(1.0), (hi[1] - lo[1]).max(1.0)];
            mouth_center = [(lo[0] + hi[0]) / 2.0, (lo[1] + hi[1]) / 2.0];
        }

        // A skirt's hem, top to bottom of this part's own box, with the Windows port's gusts.
        let hem = self.gusts && physics && part.slot == "bottomwear";
        let (hem_top, hem_bottom) = if hem {
            let (lo, hi) = bounds(&part.mesh.rest);
            (lo[1], hi[1])
        } else {
            (0.0, 0.0)
        };

        // How the strands swing it.
        let swing = if physics { part.swing.as_ref() } else { None };
        let springs = &self.strands[index];
        let (amp_k, soft_k, power): (f32, f32, f32) = match swing {
            Some(swing) if swing.kind == Phys::Sway => (1.0 * 1.4, 0.8, 1.2),
            Some(swing) if swing.kind == Phys::Hair && swing.front => (2.0, 0.4, 1.8),
            _ => (2.0, 2.0, 2.1),
        };
        let grabbed = self.grabbed.map(PartId::index) == Some(index);

        let mut out = part.mesh.rest.clone();
        for i in 0..out.len() {
            let [mut x, mut y] = out[i];

            // An eye: the white and lashes squashed toward the line it closes onto, the iris
            // bounced, looking, and round until the eye is nearly shut.
            if let Some(eye) = eye {
                let close_y = eye.close_y as f32;
                match eye_role {
                    Role::Irides => {
                        let (cx, cy) = (eye.iris.x as f32, eye.iris.y as f32);
                        x = cx + (x - cx) * e.bounce[0];
                        y = cy + (y - cy) * e.bounce[1];
                        // Anime2.5DRig moves it down for a look up; here it goes where it looks.
                        x += e.eye_x * 11.0 * fs;
                        y -= e.eye_y * 6.0 * fs;
                        let shut = smooth32((0.32 - v) / 0.32);
                        y = close_y + (y - close_y) * (1.0 - 0.80 * shut);
                    }
                    Role::EyeWhite | Role::EyeLash => {
                        y = close_y + (y - close_y) * (1.0 - 0.85 * (1.0 - v));
                    }
                    Role::EyeClose => {
                        y -= v * 3.0;
                    }
                    _ => {}
                }
            }
            // A brow drops as the eye shuts, and goes up or down as it is told.
            if part.role == Role::Eyebrow {
                y += ((1.0 - v) * 3.5 - e.brow * 9.0) * fs;
            }

            // The mouth: opened to speak, its corners up with a smile -- about its own middle, so
            // it does not wander off the face.
            if mouth {
                // 0 left to 1 right, 0 top to 1 bottom.
                let mu = (x - mouth_min[0]) / mouth_size[0];
                let mv = (y - mouth_min[1]) / mouth_size[1];
                // Open: the lower lip down and everything a little narrower.
                let open_w = (mv - 0.25).max(0.0) / 0.75;
                y += e.mouth * mouth_size[1] * 0.9 * open_w * open_w;
                x = mouth_center[0] + (x - mouth_center[0]) * (1.0 - e.mouth * 0.25);
                // Smile: the corners up and a little out, more as they near the edge.
                let corner = ((mu - 0.5).abs() * 2.0).powf(1.6);
                y -= e.smile * mouth_size[1] * (0.55 * corner - 0.10 * (1.0 - corner));
                x += e.smile * mouth_size[0] * 0.10 * (mu - 0.5) * corner;
            }

            // The head turned about the neck: all of the head, a little of the body, the neck more
            // the higher up it.
            let mut hw: f32 = if part.group == Group::Head { 1.0 } else { 0.16 };
            if part.slot == "neck" {
                hw = 0.55 * smooth32((neck_bottom - y) / (neck_bottom - neck_top).max(1.0));
            }
            if hw > 0.0 {
                let (rx, ry) = (x - np[0], y - np[1]);
                x += (rx * cz - ry * sz - rx) * hw;
                y += (rx * sz + ry * cz - ry) * hw;
                let d = part.depth;
                x += hw
                    * fs
                    * (e.angle_x * (14.0 + 40.0 * (d - 1.0)) + e.angle_x * (np[1] - y) * 0.028);
                if part.group == Group::Head {
                    // Up and down, the head as a ball: every vertex as near as the face's surface
                    // is there, and its layer in front of that, turned about the face's middle.
                    // The middle of the face is nearest, so the eyes, nose and mouth go farther
                    // than its outline, and on an arc; the head is shortened; and a layer goes as
                    // far as it is in front -- the same as Anime2.5DRig's 30 a unit of depth,
                    // which slides the head up and down rather than turning it.
                    let pitch = e.angle_y * 0.35 * e.nod;
                    let u = (x - surface_center[0]) / surface_radius[0];
                    let w = (y - surface_center[1]) / surface_radius[1];
                    let z = surface_height * (1.0 - u * u - w * w).max(0.0).sqrt()
                        + (d - 1.0) * 0.3 * surface_width;
                    y = surface_center[1] + (y - surface_center[1]) * pitch.cos()
                        - pitch.sin() * z
                        - e.angle_y * e.nod * 9.0 * fs;
                } else {
                    y += hw
                        * fs
                        * (-e.angle_y * (9.0 + 30.0 * (d - 1.0))
                            - e.angle_y * (d - 1.0) * (y - face_center_y) * 0.05);
                }
            }
            // The upper body turned after the head: the more the higher up, nothing at the
            // canvas's foot, and each layer as far as it is in front -- the tie past the top, the
            // arms behind it.
            if part.group == Group::Body {
                let up = smooth32((bp[1] - y) / (bp[1] - np[1]).max(1.0));
                x += e.body_turn * fs * (6.0 + 40.0 * (part.depth - 0.9)) * up;
                // Twisted: the shoulders one way, the hips the other, the middle not at all.
                if e.twist != 0.0 {
                    x += e.twist * fs * 26.0 * (up - 0.5);
                }
            }

            // The breath: the body rises with it, the head after; the shoulders rise and the top
            // widens.
            y -= (if part.group == Group::Body {
                e.breath * 2.0
            } else {
                e.breath_head * 1.6
            }) * fs;
            if is_top {
                if y < chest[1] {
                    y -= e.breath * 2.2 * fs * smooth32((chest[1] - y) / (reach[1] * 2.0));
                }
                x = np[0] + (x - np[0]) * (1.0 + e.breath * 0.003);
                let gx = (x - chest[0]) / reach[0];
                let gy = (y - (chest[1] + 70.0 * fs)) / reach[1];
                y += self.bust_lag * 2.5 * (-gx * gx - gy * gy).exp();
            }

            // The strands: a stiff root and a soft tip, more of the soft the farther down.
            if let Some(swing) = swing {
                if !springs.is_empty() {
                    let along = swing.along[i];
                    let u = if swing.front {
                        (along * 1.6).min(1.0)
                    } else {
                        along
                    };
                    let amp = u.powf(power) * amp_k;
                    let soft_mix = (u.powf(1.2) * soft_k).min(1.0);
                    let mut dx: f32 = 0.0;
                    let n = springs.len();
                    for (s, strand) in springs.iter().enumerate() {
                        let w = swing.weights[i * n + s];
                        if w < 0.001 {
                            continue;
                        }
                        dx +=
                            w * (strand.stiff_lag * (1.0 - soft_mix) + strand.soft_lag * soft_mix);
                    }
                    x += dx * amp;
                    y += dx.abs() * amp * 0.12;
                }
            }

            // A skirt's hem: the lower it is, the more it swings with the body and the wind.
            if hem {
                let edge =
                    smooth32((y - hem_top) / (hem_bottom - hem_top).max(1.0) - 0.45).powf(1.6);
                x += self.hem_lag * edge;
            }

            // A part being pulled: the tip of its strands goes all the way, the root not.
            if grabbed {
                let gw = match swing {
                    Some(swing) if i < swing.along.len() => 0.15 + 0.85 * swing.along[i],
                    _ => 1.0,
                };
                x += e.grab[0] * gw;
                y += e.grab[1] * gw;
            }
            out[i] = [x, y];
        }

        // The body's lean about its foot, which everything leans with.
        if ab.abs() > 1e-4 {
            for p in out.iter_mut() {
                let r = [p[0] - bp[0], p[1] - bp[1]];
                *p = [
                    bp[0] + (r[0] * cb - r[1] * sb),
                    bp[1] + (r[0] * sb + r[1] * cb),
                ];
            }
        }

        // In the air or squashing down from a jump: the whole rig as one piece.
        if e.jump_y != 0.0 || (e.squash - 1.0).abs() > 1e-3 {
            for p in out.iter_mut() {
                let y = bp[1] + (p[1] - bp[1]) * e.squash + e.jump_y;
                let x = np[0] + (p[0] - np[0]) * (2.0 - e.squash);
                *p = [x, y];
            }
        }
        out
    }
}

/// How far `side`'s eye is winked shut.
fn wink(pose: &Pose, side: Side) -> f32 {
    match side {
        Side::Left => pose.wink_left,
        Side::Right => pose.wink_right,
    }
}

/// The least and the greatest of a part's vertices, across and down.
fn bounds(points: &[[f32; 2]]) -> ([f32; 2], [f32; 2]) {
    let mut lo = [f32::MAX, f32::MAX];
    let mut hi = [f32::MIN, f32::MIN];
    for p in points {
        lo = [lo[0].min(p[0]), lo[1].min(p[1])];
        hi = [hi[0].max(p[0]), hi[1].max(p[1])];
    }
    (lo, hi)
}

/// Smoothstep of an `f32`, worked in `f64` as the Swift app's `smooth` is.
fn smooth32(t: f32) -> f32 {
    smooth(t as f64) as f32
}

fn clamp(v: f32) -> f32 {
    v.max(-1.0).min(1.0)
}

/// One frame of a rig: every part's vertices where they are now, on the canvas, and how much of
/// each shows.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Frame {
    /// All parts' vertices, one part after another.
    pub positions: Vec<[f32; 2]>,
    /// `parts.len() + 1` of them: part `i` is `positions[offsets[i]..offsets[i + 1]]`.
    pub offsets: Vec<u32>,
    /// Part by part.
    pub alphas: Vec<f32>,
}

impl Frame {
    /// A frame from each part's vertices, part by part, and their alphas.
    pub fn from_parts(parts: Vec<Vec<[f32; 2]>>, alphas: Vec<f32>) -> Frame {
        let mut offsets = Vec::with_capacity(parts.len() + 1);
        let mut positions = Vec::with_capacity(parts.iter().map(Vec::len).sum());
        offsets.push(0);
        for part in parts {
            positions.extend_from_slice(&part);
            offsets.push(positions.len() as u32);
        }
        Frame {
            positions,
            offsets,
            alphas,
        }
    }

    pub fn part_count(&self) -> usize {
        self.alphas.len()
    }

    /// Part `part`'s vertices.
    pub fn vertices(&self, part: PartId) -> &[[f32; 2]] {
        let i = part.index();
        &self.positions[self.offsets[i] as usize..self.offsets[i + 1] as usize]
    }

    pub fn alpha(&self, part: PartId) -> f32 {
        self.alphas[part.index()]
    }
}
