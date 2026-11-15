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

//! The rig and its motion against the Mac app's own, frame for frame.
//!
//! fixtures/swift_frames.json is what waifu-app/swiftui's Rig.swift and RigMotion.swift -- as they
//! were before this crate took their place, at 2565c0f -- made of common::figure_psd() moved
//! through the script below: frames of every part's vertices and alphas. The script draws nothing
//! random: look-around and automatic blinks are off, and one blink is asked for at step 90. A
//! change here that moves a vertex is a change to how the Mac's drawings move.

mod common;

use serde_json::Value;
use waifu2d::rig::{Phys, Rig};
use waifu2d::{Controls, Motion, PartId, Switches};

/// What one step of the script does: the pointer, how open the eyes may be, how far the head
/// nods, how far the body follows.
struct Step {
    dt: f64,
    pointer: Option<[f32; 2]>,
    open: f32,
    nod: f32,
    follow: f32,
    blink: bool,
}

/// The script the Swift frames were recorded under, step by step.
fn script(i: usize) -> Step {
    let pointer = (60..180).contains(&i).then(|| {
        let t = i as f64;
        [
            ((t * 0.05).sin() * 1.2) as f32,
            ((t * 0.07).cos() * 0.8) as f32,
        ]
    });
    Step {
        dt: if i % 7 == 0 { 1.0 / 30.0 } else { 1.0 / 60.0 },
        pointer,
        open: if i >= 200 { 0.5 } else { 1.0 },
        nod: if i >= 120 { 1.5 } else { 1.0 },
        follow: if i >= 150 { 0.5 } else { 1.0 },
        blink: i == 90,
    }
}

#[test]
fn the_mac_app_moved_the_figure_as_this_does() {
    let recorded: Value = serde_json::from_str(include_str!("fixtures/swift_frames.json"))
        .expect("the fixture is JSON");
    let rig = Rig::read_psd(&common::figure_psd()).expect("the figure reads");
    let names: Vec<&str> = recorded["parts"]
        .as_array()
        .unwrap()
        .iter()
        .map(|n| n.as_str().unwrap())
        .collect();
    assert_eq!(
        rig.parts
            .iter()
            .map(|p| p.name.as_str())
            .collect::<Vec<_>>(),
        names
    );

    let mut frames = recorded["frames"].as_array().unwrap().iter().peekable();
    let mut motion = Motion::new(&rig, 0);
    let mut controls = Controls {
        switches: Switches {
            follow_pointer: true,
            idle: true,
            look_around: false,
            physics: true,
            blink: false,
            // The Mac's wind blew with its sway.
            wind: true,
            auto_talk: false,
            gusts: false,
        },
        ..Controls::default()
    };
    let (mut worst, mut compared) = (0f32, 0usize);
    // Vertices on the foot that the Mac lifted off it.
    let mut lifted = 0usize;
    // Vertices of hair, and of them the back of the head's.
    let (mut hair, mut still) = (0usize, 0usize);
    for i in 0..240 {
        let step = script(i);
        if step.blink {
            motion.blink_now();
        }
        controls.eyes_open = step.open;
        controls.nod = step.nod;
        controls.body_follows = step.follow;
        motion.step(&rig, &controls, step.pointer, step.dt);

        let Some(expected) = frames.next_if(|f| f["step"].as_u64() == Some(i as u64)) else {
            continue;
        };
        let frame = motion.frame(&rig, controls.switches.physics);
        let counts: Vec<usize> = expected["counts"]
            .as_array()
            .unwrap()
            .iter()
            .map(|c| c.as_u64().unwrap() as usize)
            .collect();
        assert_eq!(counts.len(), rig.parts.len(), "step {i}: the parts");
        // Where each part's vertices begin among the Mac's.
        let mut theirs_at = vec![0usize];
        for c in &counts {
            theirs_at.push(theirs_at.last().unwrap() + c);
        }
        let positions = expected["positions"].as_array().unwrap();
        assert_eq!(positions.len(), theirs_at[counts.len()] * 2, "step {i}");
        let unswung = motion.frame(&rig, false);
        for (p, part) in rig.parts.iter().enumerate() {
            let id = PartId(p as u32);
            let ours = frame.vertices(id);
            // Hair is drawn on a finer grid and swings from where it leaves the head, by how far
            // along the hair it is (Rig::build); the Mac swung it from the top of each strand's
            // column, and swung the back of the head with it. Where it does not swing now, it is
            // where the head took it.
            if let Some(swing) = part.swing.as_ref().filter(|s| s.kind == Phys::Hair) {
                for (j, v) in ours.iter().enumerate() {
                    hair += 1;
                    if swing.along[j] == 0.0 {
                        assert_eq!(*v, unswung.vertices(id)[j], "step {i}: part {p} swung");
                        still += 1;
                    }
                }
                continue;
            }
            assert_eq!(ours.len(), counts[p], "step {i}: part {p}'s vertex count");
            for (j, v) in ours.iter().enumerate() {
                let k = theirs_at[p] + j;
                let theirs = |c: usize| positions[k * 2 + c].as_f64().unwrap() as f32;
                // Where the Mac lifted a layer cut off by the canvas's foot off that foot, this
                // holds it down: such a vertex goes no farther up or down than the Mac's, and not
                // at all on the foot itself.
                if let Some(rest_y) = held_at_foot(&rig, p, j) {
                    let (ours, mac) = (v[1] - rest_y, theirs(1) - rest_y);
                    assert!(
                        ours.abs() <= mac.abs() + 0.01 && ours * mac >= -1e-4,
                        "step {i}: part {p} vertex {j} moved {ours} up or down, the Mac {mac}"
                    );
                    if rest_y >= rig.height as f32 {
                        lifted += usize::from(mac.abs() > 0.1);
                        assert_eq!(
                            v[1], rest_y,
                            "step {i}: part {p} vertex {j} is off the canvas's foot"
                        );
                    }
                    continue;
                }
                for (c, value) in v.iter().enumerate() {
                    worst = worst.max((value - theirs(c)).abs());
                    assert!(
                        // The two differ by a rounding step of an f32 at most: 3e-5 at a few
                        // hundred.
                        (value - theirs(c)).abs() <= 1e-4,
                        "step {i}: part {p} vertex {j} {} is {value}, the Mac had {}",
                        ["x", "y"][c],
                        theirs(c)
                    );
                }
            }
        }
        let alphas = expected["alphas"].as_array().unwrap();
        for (p, a) in frame.alphas.iter().enumerate() {
            let theirs = alphas[p].as_f64().unwrap() as f32;
            assert!(
                (a - theirs).abs() <= 1e-6,
                "step {i}: part {p}'s alpha is {a}, the Mac had {theirs}"
            );
        }
        compared += 1;
    }
    assert_eq!(compared, 7, "every recorded frame compared");
    assert!(lifted > 0, "the Mac lifted the top off the canvas's foot");
    assert!(
        hair > 0 && still > 0,
        "{hair} vertices of hair, {still} of them not swinging"
    );
    println!("{compared} frames; the farthest a vertex was from the Mac's: {worst} px");
}

/// Vertex `j` of part `p`, if it is one Motion::deform holds to the canvas's foot and the Mac did
/// not: of a layer cut off by the foot, as near it as the chest reaches down. Its rest, down the
/// canvas.
fn held_at_foot(rig: &Rig, p: usize, j: usize) -> Option<f32> {
    let part = &rig.parts[p];
    let foot = rig.height as f32;
    let rest_y = part.mesh.rest[j][1];
    let cut = (part.rect.y + part.rect.height) as f32 >= foot - 1.0;
    let near = foot - rest_y < rig.anchors.chest_reach.height as f32;
    (cut && near).then_some(rest_y)
}
