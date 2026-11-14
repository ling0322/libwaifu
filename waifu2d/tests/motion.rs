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

//! How a rig moves: RigTests.swift's springs, head, hair and blink, and the Windows port's mouth,
//! wink, jump and grab -- each of which does nothing at its default. The rigs are made by hand, a
//! grid over each layer's box, so that these are the motion's tests and nothing else's.

use waifu2d::blinker::Blinker;
use waifu2d::motion::{Controls, Frame, Motion, Pose, Spring, Switches};
use waifu2d::rig::{
    next_rig_id, Anchors, EyeAnchor, Group, PartId, Phys, Rig, RigMesh, RigPart, Role, Side,
    Strand, Swing,
};
use waifu2d::{Point, Rect, RgbaImage, Size};

const DT: f64 = 1.0 / 60.0;

/// Rig.swift's `RigMesh.grid`: cells of 42 pixels at a 768 canvas, 30 for what swings, at least
/// two a side.
fn grid(rect: Rect, canvas: u32, swings: bool) -> RigMesh {
    let cell = (if swings { 30.0 } else { 42.0 }) * (canvas as f64 / 768.0).max(0.6);
    let nx = ((rect.width / cell).round() as usize).max(2);
    let ny = ((rect.height / cell).round() as usize).max(2);
    let mut rest = Vec::new();
    let mut uv = Vec::new();
    for j in 0..=ny {
        for i in 0..=nx {
            let (u, v) = (i as f32 / nx as f32, j as f32 / ny as f32);
            uv.push([u, v]);
            rest.push([
                rect.min_x() as f32 + u * rect.width as f32,
                rect.min_y() as f32 + v * rect.height as f32,
            ]);
        }
    }
    let mut indices = Vec::new();
    for j in 0..ny {
        for i in 0..nx {
            let a = (j * (nx + 1) + i) as u32;
            let b = a + 1;
            let c = a + (nx + 1) as u32;
            let d = c + 1;
            indices.extend_from_slice(&[a, b, c, b, d, c]);
        }
    }
    RigMesh { rest, uv, indices }
}

/// What a slot's layer is to the rig, as Rig.swift has it.
fn role(slot: &str) -> Role {
    match slot {
        "eyewhite" => Role::EyeWhite,
        "irides" => Role::Irides,
        "eyelash" => Role::EyeLash,
        "eye_close" | "eye_close2" => Role::EyeClose,
        "eyebrow" => Role::Eyebrow,
        "face" => Role::Face,
        _ => Role::Other,
    }
}

/// A layer over `(x, y, width, height)` on a 400 canvas, drawn on a plain grid.
fn part(
    slot: &str,
    (x, y, width, height): (f64, f64, f64, f64),
    depth: f32,
    group: Group,
) -> RigPart {
    let rect = Rect::new(x, y, width, height);
    RigPart {
        name: slot.to_string(),
        slot: slot.to_string(),
        role: role(slot),
        side: None,
        rect,
        pixels: RgbaImage::transparent(width as usize, height as usize),
        opacity: 1.0,
        depth,
        group,
        mesh: grid(rect, 400, false),
        swing: None,
    }
}

fn sided(mut part: RigPart, side: Side) -> RigPart {
    part.side = Some(side);
    part
}

/// The layer swinging in `count` strands, evenly across it and its whole height long: each
/// vertex weighted toward the strands nearest it across.
fn swinging(mut part: RigPart, kind: Phys, front: bool, count: usize) -> RigPart {
    let rect = part.rect;
    part.mesh = grid(rect, 400, true);
    let strands: Vec<Strand> = (0..count)
        .map(|s| Strand {
            x: (rect.min_x() + rect.width * (s as f64 + 0.5) / count as f64) as f32,
            root_y: rect.min_y() as f32,
            tip_y: rect.max_y() as f32,
        })
        .collect();
    let mut weights = Vec::new();
    let mut along = Vec::new();
    for p in &part.mesh.rest {
        let near: Vec<f32> = strands
            .iter()
            .map(|s| 1.0 / (1.0 + (p[0] - s.x).abs()))
            .collect();
        let sum: f32 = near.iter().sum();
        weights.extend(near.iter().map(|w| w / sum));
        along.push(((p[1] - rect.min_y() as f32) / rect.height as f32).clamp(0.0, 1.0));
    }
    part.swing = Some(Swing {
        kind,
        front,
        strands,
        weights,
        along,
    });
    part
}

fn rig(
    width: u32,
    height: u32,
    parts: Vec<RigPart>,
    eyes: [Option<EyeAnchor>; 2],
    anchors: Anchors,
) -> Rig {
    Rig {
        id: next_rig_id(),
        width,
        height,
        parts,
        eyes,
        face_scale: anchors.face.width / 333.0,
        anchors,
        notes: Vec::new(),
        masks: Vec::new(),
    }
}

/// RigTests.swift's head: a face with a mouth, a neck under it, hair in front and behind, and a
/// top. `extra` goes on top of it all.
fn head_with(extra: Vec<RigPart>) -> Rig {
    let mut parts = vec![
        swinging(
            part("back hair", (60.0, 40.0, 180.0, 260.0), 0.55, Group::Head),
            Phys::Hair,
            false,
            3,
        ),
        part("neck", (130.0, 200.0, 40.0, 60.0), 0.95, Group::Body),
        part("topwear", (60.0, 250.0, 180.0, 150.0), 0.90, Group::Body),
        part("face", (100.0, 60.0, 100.0, 150.0), 1.00, Group::Head),
        part("mouth_open", (135.0, 160.0, 30.0, 12.0), 1.08, Group::Head),
        swinging(
            part("front hair", (80.0, 30.0, 140.0, 70.0), 1.28, Group::Head),
            Phys::Hair,
            true,
            3,
        ),
    ];
    parts.extend(extra);
    let anchors = Anchors {
        face: Rect::new(100.0, 60.0, 100.0, 150.0),
        face_center: Point::new(150.0, 135.0),
        neck_pivot: Point::new(150.0, 200.0 + 0.85 * 59.0),
        neck_top: 200.0,
        neck_bottom: 259.0,
        body_pivot: Point::new(150.0, 400.0),
        chest: Point::new(150.0, 320.0),
        chest_reach: Size::new(90.0, 75.0),
    };
    rig(300, 400, parts, [None, None], anchors)
}

fn head() -> Rig {
    head_with(Vec::new())
}

/// A face with two eyes, each a white, an iris, lashes and a closed eye.
fn eyes() -> Rig {
    let mut parts = vec![part("face", (50.0, 40.0, 100.0, 140.0), 1.0, Group::Head)];
    for (side, x) in [(Side::Left, 60.0), (Side::Right, 110.0)] {
        parts.push(sided(
            part("eyewhite", (x, 80.0, 30.0, 20.0), 1.06, Group::Head),
            side,
        ));
        parts.push(sided(
            part("irides", (x + 8.0, 78.0, 14.0, 24.0), 1.08, Group::Head),
            side,
        ));
        parts.push(sided(
            part("eyelash", (x - 2.0, 74.0, 34.0, 10.0), 1.12, Group::Head),
            side,
        ));
        parts.push(sided(
            part("eye_close", (x - 2.0, 92.0, 34.0, 6.0), 1.12, Group::Head),
            side,
        ));
    }
    let eye = |x: f64| EyeAnchor {
        white: Rect::new(x, 80.0, 30.0, 20.0),
        iris: Point::new(x + 15.0, 90.0),
        close_y: 96.0,
    };
    let anchors = Anchors {
        face: Rect::new(50.0, 40.0, 100.0, 140.0),
        face_center: Point::new(100.0, 110.0),
        neck_pivot: Point::new(100.0, 190.0),
        neck_top: 180.0,
        neck_bottom: 200.0,
        body_pivot: Point::new(100.0, 200.0),
        chest: Point::new(100.0, 200.0),
        chest_reach: Size::new(60.0, 40.0),
    };
    rig(
        200,
        200,
        parts,
        [Some(eye(60.0)), Some(eye(110.0))],
        anchors,
    )
}

fn find(rig: &Rig, slot: &str) -> PartId {
    PartId(
        rig.parts
            .iter()
            .position(|p| p.slot == slot)
            .unwrap_or_else(|| panic!("no {slot}")) as u32,
    )
}

fn find_sided(rig: &Rig, slot: &str, side: Side) -> PartId {
    PartId(
        rig.parts
            .iter()
            .position(|p| p.slot == slot && p.side == Some(side))
            .unwrap() as u32,
    )
}

/// Nothing on its own: no blink, sway, wandering look, swing or wind; after the pointer only.
fn still() -> Controls {
    Controls {
        switches: Switches {
            follow_pointer: true,
            idle: false,
            look_around: false,
            physics: false,
            blink: false,
            wind: false,
            auto_talk: false,
            gusts: false,
        },
        ..Controls::default()
    }
}

fn run(
    motion: &mut Motion,
    rig: &Rig,
    controls: &Controls,
    pointer: Option<[f32; 2]>,
    steps: usize,
) {
    for _ in 0..steps {
        motion.step(rig, controls, pointer, DT);
    }
}

#[test]
fn a_spring_settles_on_its_target() {
    let mut spring = Spring::default();
    // Stiff, and damped: there in two seconds, at 10 frames a second as at 240.
    for _ in 0..20 {
        spring.step(5.0, 70.0, 9.0, 0.1);
    }
    assert!((spring.x - 5.0).abs() < 0.01, "{}", spring.x);
    let mut fine = Spring::default();
    for _ in 0..480 {
        fine.step(5.0, 70.0, 9.0, 1.0 / 240.0);
    }
    assert!((fine.x - spring.x).abs() < 0.01, "{} {}", fine.x, spring.x);
}

#[test]
fn the_head_turns_more_in_front_of_the_face() {
    let rig = head();
    // Turned right, still: the front hair goes farther than the face, the back hair less far, the
    // body after it, less far again.
    let mut motion = Motion::new(&rig, 1);
    run(&mut motion, &rig, &still(), Some([1.0, 0.0]), 120);
    assert!((motion.pose.angle_x - 0.9).abs() < 0.01);
    let moved = |slot: &str| {
        let id = find(&rig, slot);
        let now = motion.deform(&rig, id, false);
        let rest = &rig.part(id).mesh.rest;
        now.iter().zip(rest).map(|(n, r)| n[0] - r[0]).sum::<f32>() / now.len() as f32
    };
    let (front, face, back, top) = (
        moved("front hair"),
        moved("face"),
        moved("back hair"),
        moved("topwear"),
    );
    assert!(face > 0.0);
    assert!(front > face, "{front} {face}");
    assert!(back < face, "{back} {face}");
    assert!(top > 0.0);
    assert!(top < face, "{top} {face}");
    assert!((motion.pose.body_turn - 0.9).abs() < 0.01);

    // The top turns at the shoulders, and hardly at all at the canvas's foot.
    let topwear = find(&rig, "topwear");
    let now = motion.deform(&rig, topwear, false);
    let rest = &rig.part(topwear).mesh.rest;
    let last = rest.len() - 1;
    assert!(now[0][0] - rest[0][0] > 2.0 * (now[last][0] - rest[last][0]).abs());

    // Without the body following, it does not turn.
    let mut alone = Motion::new(&rig, 1);
    let controls = Controls {
        body_follows: 0.0,
        ..still()
    };
    run(&mut alone, &rig, &controls, Some([1.0, 0.0]), 120);
    assert_eq!(alone.pose.body_turn, 0.0);
}

#[test]
fn looking_up_the_middle_of_the_face_goes_farther_than_its_outline() {
    let rig = head();
    let mut motion = Motion::new(&rig, 1);
    motion.pose = Pose {
        angle_y: 1.0,
        ..Pose::default()
    };
    let face = find(&rig, "face");
    let now = motion.deform(&rig, face, false);
    let rest = rig.part(face).mesh.rest.clone();
    let rise = |i: usize| rest[i][1] - now[i][1];
    // The vertex nearest the middle of the face, and one on its outline at the same height.
    let middle = [
        rig.anchors.face_center.x as f32,
        rig.anchors.face_center.y as f32,
    ];
    let distance = |p: [f32; 2]| ((p[0] - middle[0]).powi(2) + (p[1] - middle[1]).powi(2)).sqrt();
    let centre = (0..rest.len())
        .min_by(|&a, &b| distance(rest[a]).total_cmp(&distance(rest[b])))
        .unwrap();
    let edge = (0..rest.len())
        .filter(|&i| rest[i][1] == rest[centre][1])
        .min_by(|&a, &b| rest[a][0].total_cmp(&rest[b][0]))
        .unwrap();
    assert!(rise(centre) > 0.0);
    let width = rig.anchors.face.width as f32;
    assert!(
        rise(centre) > rise(edge) + 0.05 * width,
        "{} {}",
        rise(centre),
        rise(edge)
    );
    // The mouth goes with the middle of the face, farther than the chin's outline.
    let mouth = find(&rig, "mouth_open");
    let mouth_now = motion.deform(&rig, mouth, false);
    assert!(
        rig.part(mouth).mesh.rest[0][1] - mouth_now[0][1] > rise(rest.len() - 1) + 0.02 * width
    );
    // Down is the other way.
    motion.pose = Pose {
        angle_y: -1.0,
        ..Pose::default()
    };
    assert!(rest[centre][1] - motion.deform(&rig, face, false)[centre][1] < 0.0);
}

#[test]
fn the_hair_swings_after_the_head() {
    let rig = head();
    let hair = find(&rig, "front hair");
    let mut motion = Motion::new(&rig, 1);
    let swinging = Controls {
        switches: Switches {
            physics: true,
            ..still().switches
        },
        ..still()
    };
    // A sudden turn: the tips lag behind the head, then come to rest with it.
    motion.step(&rig, &swinging, Some([1.5, 0.0]), DT);
    let lag: Vec<f32> = motion.strands(hair).iter().map(|s| s.soft_lag).collect();
    assert_eq!(lag.len(), 3);
    assert!(lag.iter().all(|&l| l > 0.0), "{lag:?}");
    // And the hair is drawn swung: its tips behind where the head took them.
    let swung = motion.deform(&rig, hair, true);
    let unswung = motion.deform(&rig, hair, false);
    let tip = swung.len() - 1;
    assert!(swung[tip][0] > unswung[tip][0]);
    run(&mut motion, &rig, &swinging, Some([1.5, 0.0]), 60 * 20);
    assert!(motion
        .strands(hair)
        .iter()
        .all(|s| s.soft_lag.abs() < 0.05 && s.stiff_lag.abs() < 0.05));
}

#[test]
fn a_blink_closes_holds_and_opens() {
    assert_eq!(Blinker::shape(0.0), Some(1.0));
    assert!((Blinker::shape(0.04).unwrap() - 0.5).abs() < 1e-9);
    assert_eq!(Blinker::shape(0.2), Some(0.0));
    assert!((Blinker::shape(0.42 + 0.08).unwrap() - 0.5).abs() < 1e-9);
    assert_eq!(Blinker::shape(0.6), None);

    // On its own, the first one 1.8 s in: shut by the middle of it, open again after.
    let mut blinker = Blinker::default();
    let mut least: f64 = 1.0;
    for _ in 0..60 * 4 {
        blinker.step(1.0 / 60.0, 1.0, true, &mut || 0.5);
        if blinker.now < 1.8 {
            assert!((blinker.openness - 1.0).abs() < 1e-9);
        }
        least = least.min(blinker.openness);
    }
    assert!(least < 0.05, "{least}");
    assert!((blinker.openness - 1.0).abs() < 0.01);

    // Never on its own; once when asked.
    let mut still = Blinker::default();
    for _ in 0..60 * 4 {
        still.step(1.0 / 60.0, 1.0, false, &mut || 0.5);
    }
    assert!((still.openness - 1.0).abs() < 1e-9);
    still.blink_now();
    for _ in 0..15 {
        still.step(1.0 / 60.0, 1.0, false, &mut || 0.5);
    }
    assert!(still.openness < 0.2, "{}", still.openness);
}

#[test]
fn a_blinking_eye_fades_into_its_closed_one() {
    let rig = eyes();
    let white = find_sided(&rig, "eyewhite", Side::Left);
    let closed = find_sided(&rig, "eye_close", Side::Left);
    let mut motion = Motion::new(&rig, 1);
    let frame = motion.frame(&rig, false);
    assert_eq!((frame.alpha(white), frame.alpha(closed)), (1.0, 0.0));
    motion.blink_now();
    run(&mut motion, &rig, &still(), None, 12);
    let frame = motion.frame(&rig, false);
    assert_eq!((frame.alpha(white), frame.alpha(closed)), (0.0, 1.0));
}

#[test]
fn defaults_leave_every_windows_pose_at_its_default() {
    let rig = head();
    let mut motion = Motion::new(&rig, 3);
    let controls = Controls::default();
    for i in 0..600 {
        let pointer = if i % 200 < 100 {
            None
        } else {
            Some([0.4, -0.3])
        };
        motion.step(&rig, &controls, pointer, DT);
        let pose = motion.pose;
        let rest = Pose::default();
        assert_eq!(
            (
                pose.mouth,
                pose.smile,
                pose.brow,
                pose.wink_left,
                pose.wink_right,
                pose.twist
            ),
            (
                rest.mouth,
                rest.smile,
                rest.brow,
                rest.wink_left,
                rest.wink_right,
                rest.twist
            )
        );
        assert_eq!(
            (pose.jump_y, pose.squash, pose.grab),
            (rest.jump_y, rest.squash, rest.grab)
        );
    }
}

/// A mouth, a skirt and a brow of one eye, and the same three as layers the Windows port does not
/// move: at the defaults, none of the three is moved by anything the Mac does not move it by.
#[test]
fn defaults_move_the_mouth_skirt_and_brows_as_the_mac_does() {
    let with = |mouth: &str, skirt: &str, side: Option<Side>| {
        let mut brow = part("eyebrow", (110.0, 90.0, 30.0, 8.0), 1.14, Group::Head);
        brow.side = side;
        head_with(vec![
            part(mouth, (140.0, 170.0, 20.0, 10.0), 1.08, Group::Head),
            part(skirt, (80.0, 330.0, 140.0, 70.0), 0.88, Group::Body),
            brow,
        ])
    };
    let windows = with("mouth_open", "bottomwear", Some(Side::Left));
    let mac = with("facedetail", "legwear", None);
    let (mut a, mut b) = (Motion::new(&windows, 9), Motion::new(&mac, 9));
    let controls = Controls::default();
    for i in 0..400 {
        let pointer = if i < 200 { None } else { Some([-0.6, 0.2]) };
        a.step(&windows, &controls, pointer, DT);
        b.step(&mac, &controls, pointer, DT);
        if i % 40 == 0 {
            assert_eq!(a.frame(&windows, true), b.frame(&mac, true), "step {i}");
        }
    }
    // With the Windows port's gusts, the skirt's hem swings.
    let skirt = find(&windows, "bottomwear");
    let before = a.deform(&windows, skirt, true);
    let mut gusty = controls;
    gusty.switches.gusts = true;
    run(&mut a, &windows, &gusty, None, 30);
    run(&mut b, &mac, &gusty, None, 30);
    assert_ne!(
        a.deform(&windows, skirt, true),
        b.deform(&mac, find(&mac, "legwear"), true)
    );
    assert_ne!(before, a.deform(&windows, skirt, true));
}

#[test]
fn the_mouth_opens_with_the_talk() {
    let rig = head();
    let mouth = find(&rig, "mouth_open");
    let mut quiet = Motion::new(&rig, 1);
    let mut talking = quiet.clone();
    let talk = Controls {
        talk_level: 1.0,
        ..still()
    };
    // Fast to open.
    run(&mut quiet, &rig, &still(), None, 10);
    run(&mut talking, &rig, &talk, None, 10);
    assert!(talking.pose.mouth > 0.99, "{}", talking.pose.mouth);
    let (shut, open) = (
        quiet.deform(&rig, mouth, false),
        talking.deform(&rig, mouth, false),
    );
    // The lower lip down, the upper not, and the mouth a little narrower.
    let last = open.len() - 1;
    assert!(
        open[last][1] > shut[last][1] + 5.0,
        "{} {}",
        open[last][1],
        shut[last][1]
    );
    assert!((open[0][1] - shut[0][1]).abs() < 0.5);
    assert!(open[last][0] < shut[last][0]);
    // Nothing else moves with it.
    assert_eq!(
        quiet.deform(&rig, find(&rig, "face"), false),
        talking.deform(&rig, find(&rig, "face"), false)
    );
    // Slower to shut.
    run(&mut talking, &rig, &still(), None, 10);
    assert!(
        talking.pose.mouth > 0.1 && talking.pose.mouth < 0.3,
        "{}",
        talking.pose.mouth
    );
    // By hand, too.
    run(
        &mut talking,
        &rig,
        &Controls {
            mouth_open: 0.5,
            ..still()
        },
        None,
        60,
    );
    assert!((talking.pose.mouth - 0.5).abs() < 1e-3);
}

#[test]
fn a_wink_fades_only_its_own_eye() {
    let rig = eyes();
    let mut motion = Motion::new(&rig, 1);
    run(
        &mut motion,
        &rig,
        &Controls {
            wink_left: 1.0,
            ..still()
        },
        None,
        120,
    );
    let frame = motion.frame(&rig, false);
    for slot in ["eyewhite", "irides", "eyelash"] {
        assert_eq!(
            frame.alpha(find_sided(&rig, slot, Side::Left)),
            0.0,
            "{slot}"
        );
        assert_eq!(
            frame.alpha(find_sided(&rig, slot, Side::Right)),
            1.0,
            "{slot}"
        );
    }
    assert_eq!(frame.alpha(find_sided(&rig, "eye_close", Side::Left)), 1.0);
    assert_eq!(frame.alpha(find_sided(&rig, "eye_close", Side::Right)), 0.0);
    // The winked white is squashed onto the line it closes onto; the other is as it was.
    let (left, right) = (
        find_sided(&rig, "eyewhite", Side::Left),
        find_sided(&rig, "eyewhite", Side::Right),
    );
    let open = Motion::new(&rig, 1).frame(&rig, false);
    let height = |frame: &Frame, part| {
        let ys: Vec<f32> = frame.vertices(part).iter().map(|p| p[1]).collect();
        ys.iter().cloned().fold(f32::MIN, f32::max) - ys.iter().cloned().fold(f32::MAX, f32::min)
    };
    assert!(height(&frame, left) < 0.2 * height(&open, left));
    assert!((height(&frame, right) - height(&open, right)).abs() < 1e-3);
}

#[test]
fn a_jump_leaves_the_ground_and_comes_back() {
    let rig = head();
    let face = find(&rig, "face");
    let mut ground = Motion::new(&rig, 1);
    let mut jumping = ground.clone();
    jumping.jump();
    let (mut highest, mut lowest_squash) = (0.0f32, 1.0f32);
    let mut risen = false;
    for _ in 0..60 * 5 {
        ground.step(&rig, &still(), None, DT);
        jumping.step(&rig, &still(), None, DT);
        highest = highest.min(jumping.pose.jump_y);
        lowest_squash = lowest_squash.min(jumping.pose.squash);
        let (up, down) = (
            jumping.deform(&rig, face, false),
            ground.deform(&rig, face, false),
        );
        risen |= up[0][1] < down[0][1] - 10.0;
    }
    assert!(highest < -50.0, "{highest}");
    assert!(risen);
    // Down again, squashed as it landed, and still again.
    assert!(lowest_squash < 0.97, "{lowest_squash}");
    assert_eq!(jumping.pose.jump_y, 0.0);
    assert!((jumping.pose.squash - 1.0).abs() < 1e-3);
    assert_eq!(jumping.frame(&rig, false), ground.frame(&rig, false));
}

#[test]
fn a_shake_and_a_nod_happen_once() {
    let rig = head();
    let mut motion = Motion::new(&rig, 1);
    motion.shake();
    motion.nod_once();
    let (mut most_x, mut most_y) = (0.0f32, 0.0f32);
    for _ in 0..60 {
        motion.step(&rig, &still(), None, DT);
        most_x = most_x.max(motion.pose.angle_x.abs());
        most_y = most_y.max(motion.pose.angle_y.abs());
    }
    assert!(most_x > 0.2 && most_y > 0.2, "{most_x} {most_y}");
    run(&mut motion, &rig, &still(), None, 60 * 3);
    assert!(motion.pose.angle_x.abs() < 1e-3 && motion.pose.angle_y.abs() < 1e-3);
}

#[test]
fn a_grab_pulls_the_part_and_letting_go_releases_it() {
    let rig = head();
    let top = find(&rig, "topwear");
    let hair = find(&rig, "front hair");
    let mut free = Motion::new(&rig, 1);
    let mut held = free.clone();
    held.grab(Some(top), [30.0, -10.0]);
    run(&mut free, &rig, &still(), None, 60);
    run(&mut held, &rig, &still(), None, 60);
    // The grabbed part is pulled all the way, vertex by vertex; nothing else is.
    let (pulled, rest) = (held.frame(&rig, false), free.frame(&rig, false));
    for (p, r) in pulled.vertices(top).iter().zip(rest.vertices(top)) {
        assert!(
            (p[0] - r[0] - 30.0).abs() < 0.1 && (p[1] - r[1] + 10.0).abs() < 0.1,
            "{p:?} {r:?}"
        );
    }
    for i in (0..rig.parts.len()).filter(|&i| i != top.index()) {
        assert_eq!(
            pulled.vertices(PartId(i as u32)),
            rest.vertices(PartId(i as u32))
        );
    }

    // Hair is pulled more at its tips than at its root.
    held.grab(Some(hair), [0.0, 20.0]);
    run(&mut free, &rig, &still(), None, 60);
    run(&mut held, &rig, &still(), None, 60);
    let now = held.deform(&rig, hair, true);
    let before = free.deform(&rig, hair, true);
    let last = now.len() - 1;
    assert!(
        (now[0][1] - before[0][1] - 0.15 * 20.0).abs() < 0.1,
        "{} {}",
        now[0][1],
        before[0][1]
    );
    assert!((now[last][1] - before[last][1] - 20.0).abs() < 0.1);

    // Let go, it is where it was.
    held.grab(None, [0.0, 0.0]);
    held.step(&rig, &still(), None, DT);
    free.step(&rig, &still(), None, DT);
    assert_eq!(held.deform(&rig, top, false), free.deform(&rig, top, false));
    assert_eq!(
        held.deform(&rig, hair, false),
        free.deform(&rig, hair, false)
    );
    run(&mut held, &rig, &still(), None, 60);
    assert!(held.pose.grab[1].abs() < 0.1);
}

#[test]
fn the_same_seed_gives_the_same_frames() {
    let rig = head();
    let controls = Controls::default();
    let (mut a, mut b, mut c) = (
        Motion::new(&rig, 42),
        Motion::new(&rig, 42),
        Motion::new(&rig, 43),
    );
    let mut differs = false;
    for i in 0..600 {
        a.step(&rig, &controls, None, DT);
        b.step(&rig, &controls, None, DT);
        c.step(&rig, &controls, None, DT);
        if i % 50 == 0 {
            let frame = a.frame(&rig, true);
            assert_eq!(frame, b.frame(&rig, true), "step {i}");
            differs |= frame != c.frame(&rig, true);
        }
    }
    // Another seed looks somewhere else.
    assert!(differs);
}
