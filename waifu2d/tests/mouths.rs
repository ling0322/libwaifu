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

//! The made mouth: hidden at rest, open in its shapes, and in between them on the way.

mod common;

use waifu2d::mouths::{self, Viseme};
use waifu2d::rig::{PartId, Rig};
use waifu2d::Motion;

fn figure() -> (Rig, Rig) {
    let plain = Rig::read_psd(&common::figure_psd()).unwrap();
    let made = mouths::with_drawn_mouths(plain.clone());
    (plain, made)
}

/// How far across and down part `p`'s vertices reach.
fn extent(points: &[[f32; 2]]) -> (f32, f32) {
    let (lo, hi) = points
        .iter()
        .fold(([f32::MAX; 2], [f32::MIN; 2]), |(lo, hi), p| {
            (
                [lo[0].min(p[0]), lo[1].min(p[1])],
                [hi[0].max(p[0]), hi[1].max(p[1])],
            )
        });
    (hi[0] - lo[0], hi[1] - lo[1])
}

#[test]
fn at_rest_the_drawing_is_as_it_was() {
    let (plain, made) = figure();
    let m = made
        .mouth
        .clone()
        .expect("the figure has a mouth to make one over");
    assert_eq!(made.parts.len(), plain.parts.len() + 4);
    assert_eq!(made.part(m.drawn).name, "mouth");
    let plain_frame = Motion::new(&plain, 0).frame(&plain, false);
    let made_frame = Motion::new(&made, 0).frame(&made, false);
    let ours = [m.inside, m.teeth, m.tongue, m.outline];
    for p in ours {
        assert_eq!(made_frame.alpha(p), 0.0, "{}", made.part(p).name);
    }
    // Every other part where it was and as it was: the made ones come after the drawn mouth.
    let mut k = 0;
    for i in 0..made.parts.len() {
        let id = PartId(i as u32);
        if ours.contains(&id) {
            continue;
        }
        assert_eq!(made_frame.vertices(id), plain_frame.vertices(PartId(k)));
        assert_eq!(made_frame.alpha(id), plain_frame.alpha(PartId(k)));
        k += 1;
    }
}

#[test]
fn the_shapes_open_and_shut() {
    let (_, rig) = figure();
    let m = rig.mouth.clone().unwrap();
    let inside = |v: Viseme| extent(&mouths::shaped(&rig, m.inside, &v.alone()).unwrap());
    let (aw, ah) = inside(Viseme::A);
    let (ow, oh) = inside(Viseme::O);
    let (iw, ih) = inside(Viseme::I);
    let (_, shut) = inside(Viseme::Mbp);
    assert!(
        ah > oh && oh > ih && ih > 0.0,
        "A {ah}, O {oh}, I {ih} open"
    );
    assert!(iw > aw && aw > ow, "I {iw}, A {aw}, O {ow} wide");
    // Shut, the inside is the bend of the line and no more.
    assert!(shut < 0.2 * ah, "MBP {shut} against A {ah}");

    // Shown, the drawn mouth is gone and the made one whole.
    let mut motion = Motion::new(&rig, 0);
    motion.pose.visemes = Viseme::A.alone();
    let frame = motion.frame(&rig, false);
    assert_eq!(frame.alpha(m.drawn), 0.0);
    assert_eq!(frame.alpha(m.inside), 1.0);
}

#[test]
fn between_two_shapes_the_mouth_is_between_them() {
    let (_, rig) = figure();
    let m = rig.mouth.clone().unwrap();
    let inside = |w: [f32; 8]| extent(&mouths::shaped(&rig, m.inside, &w).unwrap());
    let (aw, ah) = inside(Viseme::A.alone());
    let (ow, oh) = inside(Viseme::O.alone());
    let mut half = [0.0; 8];
    half[Viseme::A as usize] = 0.5;
    half[Viseme::O as usize] = 0.5;
    let (hw, hh) = inside(half);
    assert!(ow < hw && hw < aw, "{ow} < {hw} < {aw}");
    assert!(oh < hh && hh < ah, "{oh} < {hh} < {ah}");
    // The topology never changes: every shape is the same vertices, moved.
    for p in [m.inside, m.teeth, m.tongue, m.outline] {
        let n = rig.part(p).mesh.rest.len();
        for v in Viseme::ALL {
            assert_eq!(mouths::shaped(&rig, p, &v.alone()).unwrap().len(), n);
        }
    }
}

#[test]
fn the_teeth_and_tongue_are_inside_the_mouth() {
    let (_, rig) = figure();
    let m = rig.mouth.clone().unwrap();
    let mask = rig.masks.iter().find(|d| d.name == "mouth").unwrap();
    assert_eq!(mask.parts, vec![m.inside]);
    assert_eq!(mask.clips, vec![m.teeth, m.tongue]);
    // The eyes' masks still name their own parts, after the parts made before them moved on.
    for d in rig.masks.iter().filter(|d| d.name.starts_with("eyewhite")) {
        assert!(d.parts.iter().all(|p| rig.part(*p).slot == "eyewhite"));
        assert!(d.clips.iter().all(|p| rig.part(*p).slot == "irides"));
    }
}

#[test]
fn a_rig_without_a_mouth_is_left_as_it_is() {
    let rig = Rig::read_psd(&common::face_psd()).unwrap();
    let parts = rig.parts.len();
    let same = mouths::with_drawn_mouths(rig);
    assert!(same.mouth.is_none());
    assert_eq!(same.parts.len(), parts);
}
