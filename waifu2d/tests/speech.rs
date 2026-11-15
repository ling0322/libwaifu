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

//! A reading said by the mouth: as open as its sound is loud, in the shape its voice said.

mod common;

use std::sync::Arc;

use waifu2d::mouths::{self, Viseme};
use waifu2d::rig::Rig;
use waifu2d::speech::{Speech, LEAD};
use waifu2d::Stage;

const RATE: u32 = 24_000;

/// `seconds` of a tone, `loud` 0 to 1.
fn tone(seconds: f64, loud: f32) -> Vec<f32> {
    (0..(seconds * RATE as f64) as usize)
        .map(|i| loud * (i as f32 * 2.0 * std::f32::consts::PI * 220.0 / RATE as f32).sin())
        .collect()
}

/// Loud, quiet, loud: a syllable, the gap between, the next.
fn syllables() -> Vec<f32> {
    let mut s = tone(0.2, 0.5);
    s.extend(tone(0.1, 0.005));
    s.extend(tone(0.2, 0.5));
    s
}

#[test]
fn the_mouth_is_as_open_as_the_sound_is_loud() {
    let mut speech = Speech::new(RATE);
    speech.push(&syllables());
    let at = |s: f64| speech.openness(s).unwrap();
    assert!(at(0.1) > 0.95, "a syllable: {}", at(0.1));
    assert!(at(0.25) < 0.05, "between: {}", at(0.25));
    assert!(at(0.4) > 0.95, "the next: {}", at(0.4));
    // Nothing past what has been heard.
    assert!(speech.openness(0.6).is_none());
    assert!(speech.openness(-0.1).is_none());
}

/// The loudness libwaifu hands over for `syllables`, a frame each 10 ms: -9 dB, a sine of 0.5's,
/// and -49 between.
fn syllables_loudness() -> Vec<f32> {
    let mut db = vec![-9.0; 20];
    db.extend([-49.0; 10]);
    db.extend([-9.0; 20]);
    db
}

#[test]
fn the_loudness_handed_over_opens_it_as_the_one_heard_does() {
    let mut given = Speech::new(RATE);
    given.set_loudness(&syllables_loudness(), 10);
    let mut heard = Speech::new(RATE);
    heard.push(&syllables());
    for at in [0.05, 0.1, 0.22, 0.25, 0.28, 0.4] {
        let (g, h) = (given.openness(at).unwrap(), heard.openness(at).unwrap());
        assert!((g - h).abs() < 0.1, "at {at}: given {g}, heard {h}");
    }
    assert!(given.openness(0.25).unwrap() < 0.05);
    assert!(given.openness(0.5).is_none(), "past the last frame");
}

#[test]
fn once_the_loudness_is_given_the_samples_are_not_listened_to() {
    let mut speech = Speech::new(RATE);
    speech.set_loudness(&[-9.0; 10], 10);
    // Silence, which would shut it, were it heard.
    speech.push(&vec![0.0; RATE as usize]);
    assert!(speech.openness(0.05).unwrap() > 0.95);
    assert!(speech.openness(0.2).is_none());
}

#[test]
fn the_voice_says_the_shape_and_mbp_and_rest_shut_it() {
    let mut speech = Speech::new(RATE);
    speech.push(&tone(0.5, 0.5));
    // A for the first 0.2 s, then MBP, then REST.
    let mut v = vec![Viseme::A as i32; 5];
    v.extend([Viseme::Mbp as i32; 3]);
    v.extend([Viseme::Rest as i32; 4]);
    speech.set_visemes(&v, 40);
    let open = speech.mouth(0.05 - LEAD).unwrap();
    assert!(open[Viseme::A as usize] > 0.95, "{open:?}");
    let shut = speech.mouth(0.25 - LEAD).unwrap();
    assert_eq!(shut[Viseme::Mbp as usize], 1.0, "loud, and MBP: {shut:?}");
    let rest = speech.mouth(0.41 - LEAD).unwrap();
    assert_eq!(rest[Viseme::Rest as usize], 1.0);
    // Past the shapes said so far, at rest until they are.
    assert!(speech.mouth(0.49 - LEAD).is_none());
}

#[test]
fn a_shape_held_one_frame_is_a_stumble_but_mbp_is_not() {
    let mut speech = Speech::new(RATE);
    speech.push(&tone(0.4, 0.5));
    let (a, i, m) = (Viseme::A as i32, Viseme::I as i32, Viseme::Mbp as i32);
    speech.set_visemes(&[a, a, i, a, a, m, a, a], 40);
    assert_eq!(speech.viseme(0.09), Some(Viseme::A), "the one-frame I");
    assert_eq!(speech.viseme(0.21), Some(Viseme::Mbp), "the one-frame MBP");
}

#[test]
fn with_no_shapes_an_open_mouth_is_as() {
    let mut speech = Speech::new(RATE);
    speech.push(&tone(0.3, 0.5));
    let w = speech.mouth(0.1).unwrap();
    assert!(w[Viseme::A as usize] > 0.95, "{w:?}");
}

#[test]
fn a_stage_says_a_reading_and_falls_silent() {
    let rig = Arc::new(mouths::with_drawn_mouths(
        Rig::read_psd(&common::figure_psd()).unwrap(),
    ));
    let mut stage = Stage::new(rig, 1);
    stage.speak(RATE);
    stage.speech_samples(&syllables());
    stage.speech_visemes(&[Viseme::O as i32; 13], 40);
    let rest = Viseme::Rest as usize;
    // Heard in the middle of a syllable, the mouth opens into O.
    for _ in 0..10 {
        stage.speech_at(Some(0.1));
        stage.step(1.0 / 60.0);
    }
    let w = stage.motion().pose().visemes;
    assert!(w[Viseme::O as usize] > 0.9 && w[rest] < 0.1, "{w:?}");
    // Between syllables, it shuts.
    for _ in 0..10 {
        stage.speech_at(Some(0.25 - LEAD));
        stage.step(1.0 / 60.0);
    }
    assert!(stage.motion().pose().visemes[rest] > 0.9);
    // Paused, and gone: at rest.
    stage.speech_at(Some(0.1));
    stage.step(1.0 / 60.0);
    stage.silence();
    for _ in 0..30 {
        stage.step(1.0 / 60.0);
    }
    assert!(stage.speech().is_none());
    assert_eq!(stage.motion().pose().visemes, Viseme::Rest.alone());
}
