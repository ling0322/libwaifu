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
use waifu2d::rig::Rig;
use waifu2d::{Controls, Motion, Switches};

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
        let counts: Vec<u64> = expected["counts"]
            .as_array()
            .unwrap()
            .iter()
            .map(|c| c.as_u64().unwrap())
            .collect();
        let ours: Vec<u64> = frame
            .offsets
            .windows(2)
            .map(|w| (w[1] - w[0]) as u64)
            .collect();
        assert_eq!(ours, counts, "step {i}: the parts' vertex counts");

        let positions = expected["positions"].as_array().unwrap();
        assert_eq!(positions.len(), frame.positions.len() * 2, "step {i}");
        for (k, v) in frame.positions.iter().enumerate() {
            for (c, value) in v.iter().enumerate() {
                let theirs = positions[k * 2 + c].as_f64().unwrap() as f32;
                worst = worst.max((value - theirs).abs());
                assert!(
                    // The two differ by a rounding step of an f32 at most: 3e-5 at a few hundred.
                    (value - theirs).abs() <= 1e-4,
                    "step {i}: vertex {k} {} is {value}, the Mac had {theirs}",
                    ["x", "y"][c]
                );
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
    println!("{compared} frames; the farthest a vertex was from the Mac's: {worst} px");
}
