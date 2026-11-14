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

//! A Photoshop document read, and the rig made of it: the Mac app's PSDTests.swift and the
//! rig-building half of its RigTests.swift. How the rig moves is tests/motion.rs'.

use waifu2d::psd::PsdDocument;
use waifu2d::rig::{self, MaskId, PartId, Phys, Pixels, Rig, RigMesh, Role, Side, Strand};
use waifu2d::{Error, Rect, RgbaImage};

mod common;

use common::{layer, layer_where, Layer, Writer};

fn read(width: i64, height: i64, layers: &[Layer]) -> PsdDocument {
    PsdDocument::read(&Writer::document(width, height, layers)).expect("the document reads")
}

fn assert_near(a: f64, b: f64, accuracy: f64) {
    assert!(
        (a - b).abs() <= accuracy,
        "{a} is not within {accuracy} of {b}"
    );
}

// ---- reading ------------------------------------------------------------------------------------

#[test]
fn a_document_is_read_layer_by_layer() {
    let mut fringe = layer("前髪", (1, 2, 3, 2), [200, 100, 50, 128]);
    fringe.pack_bits = true;
    let mut hidden = layer("hidden one", (0, 0, 2, 2), [9, 9, 9, 255]);
    hidden.hidden = true;
    let document = read(
        8,
        6,
        &[
            layer("back", (0, 0, 8, 6), [10, 20, 30, 255]),
            fringe,
            hidden,
            // A group, closed then opened going up, hidden: what is in it is hidden too.
            Layer {
                name: "</Layer group>".into(),
                divider: 3,
                ..Layer::default()
            },
            layer("in a hidden group", (2, 2, 2, 2), [1, 2, 3, 255]),
            Layer {
                name: "group".into(),
                divider: 1,
                hidden: true,
                ..Layer::default()
            },
        ],
    );
    assert_eq!((document.width, document.height), (8, 6));
    let names: Vec<&str> = document.layers.iter().map(|l| l.name.as_str()).collect();
    assert_eq!(names, ["back", "前髪", "hidden one", "in a hidden group"]);
    let visible: Vec<bool> = document.layers.iter().map(|l| l.visible).collect();
    assert_eq!(visible, [true, true, false, false]);

    let fringe = &document.layers[1];
    assert_eq!([fringe.left, fringe.top], [1, 2]);
    assert_eq!([fringe.image.width, fringe.image.height], [3, 2]);
    assert_eq!(
        fringe.image.rgba[..4],
        [200, 100, 50, 128],
        "PackBits, alpha its own channel"
    );
    assert_near(fringe.opacity, 1.0, 1e-12);
    let back = &document.layers[0].image.rgba;
    assert_eq!(back[back.len() - 4..], [10, 20, 30, 255]);
}

#[test]
fn what_is_not_a_photoshop_document_says_so() {
    assert_eq!(
        PsdDocument::read(b"not a psd"),
        Err(Error::Psd("this is not a Photoshop document".into()))
    );
    let mut big = Writer::document(1, 1, &[]);
    big[5] = 2;
    match PsdDocument::read(&big) {
        Err(Error::Psd(message)) => assert!(message.contains(".psb"), "{message}"),
        other => panic!("{other:?}"),
    }
    let whole = Writer::document(4, 4, &[layer("face", (0, 0, 4, 4), [1, 2, 3, 255])]);
    assert_eq!(
        PsdDocument::read(&whole[..whole.len() - 20]),
        Err(Error::Psd(
            "the document ends where more of it was expected".into()
        ))
    );
}

#[test]
fn the_depth_is_kept_apart_from_the_layers() {
    let mut depth = layer(&rig_depth_name("face"), (0, 0, 8, 8), [40, 40, 40, 255]);
    depth.hidden = true;
    let document = read(
        8,
        8,
        &[
            // 8 by 8: anything under 40 pixels is a speck the rig clears.
            layer("face", (0, 0, 8, 8), [250, 220, 200, 255]),
            Layer {
                name: "</Layer group>".into(),
                divider: 3,
                hidden: true,
                ..Layer::default()
            },
            depth,
            Layer {
                name: "depth".into(),
                divider: 2,
                hidden: true,
                ..Layer::default()
            },
        ],
    );
    let (layers, depths) = document.split_depth();
    assert_eq!(
        layers.iter().map(|l| l.name.as_str()).collect::<Vec<_>>(),
        ["face"]
    );
    assert_eq!(
        depths.iter().map(|l| l.name.as_str()).collect::<Vec<_>>(),
        ["depth/face"]
    );
    assert_eq!(document.depth("face").map(|l| l.image.rgba[0]), Some(40));
    assert!(document.depth("hair").is_none());
    // A depth is hidden, so the rig does not draw it.
    let rig = Rig::build(&document);
    assert_eq!(
        rig.parts
            .iter()
            .map(|p| p.name.as_str())
            .collect::<Vec<_>>(),
        ["face"]
    );
}

fn rig_depth_name(name: &str) -> String {
    waifu2d::psd::depth_name(name)
}

// ---- names --------------------------------------------------------------------------------------

#[test]
fn layers_are_known_by_their_names() {
    assert_eq!(rig::role_of("eyewhite").0, Role::EyeWhite);
    assert_eq!(rig::role_of("白目").0, Role::EyeWhite);
    assert_eq!(rig::role_of("Eyelash_2").0, Role::EyeLash);
    assert_eq!(rig::role_of("eyelash のコピー").0, Role::EyeLash);
    assert_eq!(rig::role_of("front hair").0, Role::Other);
    // See-through's sides are the character's: its right eye is on the picture's left.
    assert_eq!(rig::role_of("irides-r"), (Role::Irides, Some(Side::Left)));
    assert_eq!(
        rig::role_of("eyebrow-l"),
        (Role::Eyebrow, Some(Side::Right))
    );
    assert_eq!(rig::role_of("右白目"), (Role::EyeWhite, Some(Side::Left)));
    assert_eq!(rig::role_of("face"), (Role::Face, None));
}

#[test]
fn slots_are_known_by_their_names() {
    assert_eq!(rig::slot_of("前髪2").0, "front hair");
    assert_eq!(rig::slot_of("front hair-0").0, "front hair");
    assert_eq!(rig::slot_of("mouth").0, "mouth_open");
    assert_eq!(rig::slot_of("mouth_c").0, "mouth_close");
    assert_eq!(rig::slot_of("eyelash_c").0, "eye_close");
    assert_eq!(rig::slot_of("Skirt").0, "bottomwear");
    assert_eq!(
        rig::slot_of("earwear-l"),
        ("earwear".to_string(), Some(Side::Right))
    );
    assert_eq!(rig::slot_of("something").0, "something");
    assert_eq!(rig::slot_named("front hair").map(|s| s.depth), Some(1.28));
    assert_eq!(
        rig::slot_named("front hair").and_then(|s| s.phys),
        Some(Phys::Hair)
    );
    assert!(rig::slot_named("something").is_none());
}

#[test]
fn names_are_folded_as_foundation_folds_them() {
    // Full and half width, as NFKC has them.
    assert_eq!(rig::slot_of("前髪２").0, "front hair");
    assert_eq!(rig::slot_of("ＥｙｅＷｈｉｔｅ").0, "eyewhite");
    assert_eq!(rig::fold("ｶﾁｭｰｼｬ"), "カチューシャ");
    assert_eq!(rig::slot_of("ｶﾁｭｰｼｬ").0, "headwear");
    // A voicing mark apart is put back on its kana: "まつげ" as macOS spells a file's name.
    assert_eq!(rig::slot_of("まつけ\u{3099}").0, "eyelash");
    assert_eq!(rig::fold("ﾌﾞﾋﾟ"), "ブピ");
    // A copy's suffix, with its number or without.
    assert_eq!(rig::fold("Eyelash copy"), "eyelash");
    assert_eq!(rig::fold("Eyelash  copy 2"), "eyelash");
    assert_eq!(rig::fold("eyelashcopy"), "eyelashcopy");
    assert_eq!(rig::fold("copy"), "copy");
    assert_eq!(rig::fold("eye copy copy"), "eyecopy");
    // Spaces trimmed, a line's end kept, the marks between words left out.
    assert_eq!(rig::fold("\u{3000} front_hair・a.b "), "fronthairab");
    assert!(rig::is_numbered("前髪 3"));
    assert!(rig::is_numbered("前髪②"));
    assert!(!rig::is_numbered("前髪"));
}

// ---- building -----------------------------------------------------------------------------------

/// A face 100 wide in the middle of a 200 by 200 canvas, and both eyes on it drawn as one layer
/// each: whites, irides, lashes.
fn face() -> PsdDocument {
    let both = |x: i64, _| !(30..50).contains(&x);
    read(
        200,
        200,
        &[
            layer("face", (50, 40, 100, 120), [250, 220, 200, 255]),
            layer_where("eyewhite", (65, 80, 80, 20), [255, 255, 255, 255], both),
            layer_where("irides", (70, 82, 70, 16), [40, 90, 200, 255], |x, _| {
                !(20..50).contains(&x)
            }),
            layer_where("eyelash", (63, 76, 84, 6), [60, 30, 30, 255], |x, _| {
                !(32..52).contains(&x)
            }),
            layer("front hair", (40, 20, 120, 30), [80, 160, 160, 255]),
        ],
    )
}

#[test]
fn both_eyes_of_one_layer_each_are_found_and_given_a_closed_eye() {
    let rig = Rig::build(&face());
    assert!(rig.notes.is_empty(), "{:?}", rig.notes);
    let left = rig.eye(Side::Left).expect("a left eye");
    let right = rig.eye(Side::Right).expect("a right eye");
    // Cut at the middle of the face, x = 100: each white is its own 30 by 20.
    assert_eq!(left.white, Rect::new(65.0, 80.0, 30.0, 20.0));
    assert_eq!(right.white, Rect::new(115.0, 80.0, 30.0, 20.0));
    assert_near(left.close_y, 80.0 + 0.62 * 20.0, 1e-9);
    assert_near(left.iris.y, 90.0, 1e-9);

    // A closed eye each, over the lashes and under the hair, 1.1 times the white and
    // lash-coloured.
    let names: Vec<(Role, Option<Side>)> = rig.parts.iter().map(|p| (p.role, p.side)).collect();
    use Role::*;
    let (l, r) = (Some(Side::Left), Some(Side::Right));
    assert_eq!(
        names,
        [
            (Face, None),
            (EyeWhite, l),
            (EyeWhite, r),
            (Irides, l),
            (Irides, r),
            (EyeLash, l),
            (EyeLash, r),
            (EyeClose, l),
            (EyeClose, r),
            (Other, None),
        ]
    );
    let closed = rig
        .parts
        .iter()
        .find(|p| p.role == EyeClose && p.side == l)
        .expect("a closed left eye");
    assert_eq!(closed.name, "eye_close (made)");
    assert_near(closed.rect.width, 33.0, 1.0);
    assert_near(closed.rect.mid_x(), left.white.mid_x(), 1.0);
    assert_near(rig.face_scale, 100.0 / 333.0, 1e-9);

    // The parts are the grids they are drawn on, a picture the size of each.
    for part in &rig.parts {
        assert_eq!(part.pixels.width as f64, part.rect.width);
        assert_eq!(part.pixels.height as f64, part.rect.height);
        assert_eq!(part.mesh.rest.len(), part.mesh.uv.len());
    }
    assert_eq!(rig.white(Side::Left), Some(PartId(1)));
    assert_eq!(rig.white(Side::Right), Some(PartId(2)));
}

#[test]
fn each_white_is_a_mask_its_iris_is_cut_by() {
    let rig = Rig::build(&face());
    assert_eq!(rig.masks.len(), 2);
    let (left, right) = (&rig.masks[0], &rig.masks[1]);
    assert_eq!((left.id, left.name.as_str()), (MaskId(0), "eyewhite-left"));
    assert_eq!(left.parts, [PartId(1)]);
    assert_eq!(left.clips, [PartId(3)]);
    assert_eq!(
        (right.id, right.name.as_str()),
        (MaskId(1), "eyewhite-right")
    );
    assert_eq!(right.parts, [PartId(2)]);
    assert_eq!(right.clips, [PartId(4)]);

    // One eye's whites in two layers make one mask of both; an eye without a white, none.
    let rig = Rig::build(&read(
        200,
        200,
        &[
            layer("face", (50, 40, 100, 120), [250, 220, 200, 255]),
            layer("eyewhite-r", (65, 80, 30, 20), [255, 255, 255, 255]),
            layer("eyewhite_2-r", (65, 100, 30, 10), [255, 255, 255, 255]),
            layer("irides-r", (70, 82, 20, 16), [40, 90, 200, 255]),
            layer("irides-l", (115, 82, 20, 16), [40, 90, 200, 255]),
        ],
    ));
    assert_eq!(rig.masks.len(), 1);
    assert_eq!(rig.masks[0].id, MaskId(0));
    assert_eq!(rig.masks[0].parts, [PartId(1), PartId(2)]);
    assert_eq!(rig.masks[0].clips, [PartId(3)]);
    assert!(rig.eye(Side::Right).is_none());
}

#[test]
fn a_closed_eye_is_painted_the_colour_of_the_lashes() {
    let eye = Rig::closed_eye(Side::Left, 88.0, (0.5, 0.25, 0.25)).expect("a closed eye");
    assert_eq!((eye.width, eye.height), (88, 42));
    let colour = Pixels::mean_colour(&eye).expect("something drawn");
    // Around the tint: each pixel as light as it was against the drawing's mean.
    assert_near(colour.0 / colour.1, 2.0, 0.1);
    assert_near(colour.1, colour.2, 0.01);
    let right = Rig::closed_eye(Side::Right, 47.0, (0.2, 0.15, 0.15)).expect("a closed eye");
    assert_eq!((right.width, right.height), (47, 16));
    assert!(Rig::closed_eye(Side::Left, 0.5, (0.2, 0.15, 0.15)).is_none());
}

#[test]
fn a_picture_without_eyes_says_why() {
    let rig = Rig::build(&read(
        10,
        10,
        &[layer("body", (0, 0, 10, 10), [1, 1, 1, 255])],
    ));
    assert!(rig.eyes.iter().all(Option::is_none));
    assert_eq!(rig.notes.len(), 1);
    assert!(rig.masks.is_empty());
}

#[test]
fn a_grid_covers_its_layer() {
    // 42-pixel cells at a 768 canvas: a 100 by 50 layer is 2 by 2 of them at least.
    let mesh = RigMesh::grid(Rect::new(10.0, 20.0, 100.0, 50.0), 768, false);
    assert_eq!(mesh.rest.len(), 3 * 3);
    assert_eq!(mesh.indices.len(), 2 * 2 * 6);
    assert_eq!(mesh.rest.first(), Some(&[10.0, 20.0]));
    assert_eq!(mesh.rest.last(), Some(&[110.0, 70.0]));
    assert_eq!(mesh.uv.last(), Some(&[1.0, 1.0]));
    // What swings is cut finer.
    let fine = RigMesh::grid(Rect::new(0.0, 0.0, 300.0, 300.0), 768, true);
    assert_eq!(fine.rest.len(), 11 * 11);
}

#[test]
fn hair_hangs_in_a_strand_at_each_tip() {
    // Three locks 60 wide, 100, 160 and 100 long: a strand at the bottom of each.
    let (w, h) = (180, 170);
    let mut rgba = vec![0u8; w * h * 4];
    for x in 0..w {
        let long = [100, 160, 100][x / 60];
        for y in 0..long {
            rgba[(y * w + x) * 4 + 3] = 255;
        }
    }
    let strands = Rig::strands(&RgbaImage::new(w, h, rgba), 3, 30);
    assert_eq!(strands.len(), 3);
    assert!(
        strands
            .iter()
            .any(|s| (s.x - 90.0).abs() <= 2.0 && s.tip_y == 159.0 && s.root_y == 0.0),
        "{strands:?}"
    );
    let xs: Vec<f32> = strands.iter().map(|s| s.x).collect();
    let mut sorted = xs.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    assert_eq!(xs, sorted);

    let mesh = RigMesh::grid(Rect::new(0.0, 0.0, w as f64, h as f64), 768, true);
    let swing = Rig::swing(Phys::Hair, true, &strands, &mesh);
    assert_eq!(swing.weights.len(), mesh.rest.len() * 3);
    for i in 0..mesh.rest.len() {
        let sum: f32 = swing.weights[i * 3..(i + 1) * 3].iter().sum();
        assert!((sum - 1.0).abs() <= 1e-4, "{sum}");
    }
    assert_eq!(swing.along.first(), Some(&0.0));
    assert_eq!(swing.along.last(), Some(&1.0));
}

#[test]
fn one_strand_takes_every_vertex() {
    let strand = Strand {
        x: 10.0,
        root_y: 0.0,
        tip_y: 100.0,
    };
    let mesh = RigMesh::grid(Rect::new(0.0, 0.0, 20.0, 100.0), 768, true);
    let swing = Rig::swing(Phys::Sway, false, &[strand], &mesh);
    assert!(swing.weights.iter().all(|&w| w == 1.0));
    assert_eq!(swing.along[0], 0.0);
}

/// A face with a mouth, a neck under it, hair in front and behind.
fn head() -> Rig {
    Rig::build(&read(
        300,
        400,
        &[
            layer("back hair", (60, 40, 180, 260), [60, 40, 40, 255]),
            layer("neck", (130, 200, 40, 60), [240, 210, 190, 255]),
            layer("topwear", (60, 250, 180, 150), [40, 60, 120, 255]),
            layer("face", (100, 60, 100, 150), [250, 220, 200, 255]),
            layer("mouth", (135, 160, 30, 12), [150, 40, 50, 255]),
            layer("front hair", (80, 30, 140, 70), [60, 40, 40, 255]),
        ],
    ))
}

#[test]
fn the_head_turns_on_its_neck_and_its_hair_swings() {
    let rig = head();
    assert_near(rig.anchors.neck_pivot.y, 200.0 + 0.85 * 59.0, 1.0);
    let part = |slot: &str| rig.parts.iter().find(|p| p.slot == slot).expect(slot);
    assert!(part("front hair").swing.as_ref().is_some_and(|s| s.front));
    assert!(part("back hair").swing.as_ref().is_some_and(|s| !s.front));
    assert!(part("topwear").swing.is_none());
    assert_eq!(part("mouth_open").name, "mouth");
    assert_eq!(part("front hair").depth, 1.28);
    // The neck is the body's, the hair the head's.
    assert_eq!(part("neck").group, rig::Group::Body);
    assert_eq!(part("back hair").group, rig::Group::Head);
    assert!(rig.anchors.chest.y > rig.anchors.neck_bottom);
    assert_eq!(rig.anchors.body_pivot.y, 400.0);
}

#[test]
fn every_rig_has_its_own_id() {
    let (a, b) = (head(), head());
    assert_ne!(a.id, b.id);
}

#[test]
fn a_real_document() {
    // An img2layers document, where one is named by LIBWAIFU_RIG_PSD: what the page would make of
    // it.
    let Some(path) = std::env::var_os("LIBWAIFU_RIG_PSD") else {
        eprintln!("skipped: set LIBWAIFU_RIG_PSD to an img2layers document");
        return;
    };
    let bytes = std::fs::read(&path).expect("the document");
    let document = PsdDocument::read(&bytes).expect("a Photoshop document");
    let rig = Rig::build(&document);
    let names: Vec<&str> = document.layers.iter().map(|l| l.name.as_str()).collect();
    println!("{} x {}: {names:?}", document.width, document.height);
    for part in &rig.parts {
        println!(
            "{} {:?} {:?} {:?}",
            part.name, part.role, part.side, part.rect
        );
    }
    println!("eyes: {:?} notes: {:?}", rig.eyes, rig.notes);
    assert!(rig.eyes.iter().all(Option::is_some));
}
