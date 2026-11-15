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

//! Drawing: the assets every backend uploads, the draw list the shared code decides, the CPU
//! reference, and every GPU backend on this machine held to it -- the backend contract of
//! docs/waifu2d.md. The rigs are made by hand, part by part, so that nothing here waits on
//! `Rig::build` or `Motion`.

use waifu2d::render::{self, Api, Assets, Backend, DrawList, Mask, Surface, MIN_ALPHA};
use waifu2d::rig::{next_rig_id, Anchors, Group, MaskDef, MaskId, RigMesh, RigPart, Role, Side};
use waifu2d::{Frame, PartId, Rect, RgbaImage, Rig};

// ---- making rigs -------------------------------------------------------------------------------

/// A grid of `cols` by `rows` cells over `rect` (x, y, width, height on the canvas), each vertex
/// moved by `warp`; uv 0 to 1 across it. The two triangles of a cell are wound opposite ways, so
/// that whatever draws them must not cull either.
fn grid(rect: [f32; 4], cols: usize, rows: usize, warp: impl Fn(f32, f32) -> [f32; 2]) -> RigMesh {
    let mut mesh = RigMesh::default();
    for j in 0..=rows {
        for i in 0..=cols {
            let (u, v) = (i as f32 / cols as f32, j as f32 / rows as f32);
            mesh.rest
                .push(warp(rect[0] + rect[2] * u, rect[1] + rect[3] * v));
            mesh.uv.push([u, v]);
        }
    }
    let at = |i: usize, j: usize| (j * (cols + 1) + i) as u32;
    for j in 0..rows {
        for i in 0..cols {
            let (a, b, c, d) = (at(i, j), at(i + 1, j), at(i, j + 1), at(i + 1, j + 1));
            mesh.indices.extend_from_slice(&[a, b, c, b, c, d]);
        }
    }
    mesh
}

fn flat(rect: [f32; 4]) -> RigMesh {
    grid(rect, 1, 1, |x, y| [x, y])
}

/// A picture `w` by `h`, texel by texel from `f(x, y)`, not premultiplied.
fn picture(w: usize, h: usize, f: impl Fn(usize, usize) -> [u8; 4]) -> RgbaImage {
    let mut rgba = Vec::with_capacity(w * h * 4);
    for y in 0..h {
        for x in 0..w {
            rgba.extend_from_slice(&f(x, y));
        }
    }
    RgbaImage::new(w, h, rgba)
}

/// An opaque disc filling a `w` by `h` picture, its edge soft over `soft` texels.
fn ellipse(w: usize, h: usize, soft: f32, colour: [u8; 3]) -> RgbaImage {
    picture(w, h, |x, y| {
        let dx = (x as f32 + 0.5) / w as f32 * 2.0 - 1.0;
        let dy = (y as f32 + 0.5) / h as f32 * 2.0 - 1.0;
        let r = (dx * dx + dy * dy).sqrt();
        let edge = soft / (w.min(h) as f32 / 2.0);
        let a = ((1.0 - r) / edge).clamp(0.0, 1.0);
        [colour[0], colour[1], colour[2], (a * 255.0).round() as u8]
    })
}

fn part(name: &str, role: Role, side: Option<Side>, pixels: RgbaImage, mesh: RigMesh) -> RigPart {
    let (mut lo, mut hi) = ([f32::MAX; 2], [f32::MIN; 2]);
    for p in &mesh.rest {
        lo = [lo[0].min(p[0]), lo[1].min(p[1])];
        hi = [hi[0].max(p[0]), hi[1].max(p[1])];
    }
    RigPart {
        name: name.into(),
        slot: name.into(),
        role,
        side,
        rect: Rect::new(
            lo[0] as f64,
            lo[1] as f64,
            (hi[0] - lo[0]) as f64,
            (hi[1] - lo[1]) as f64,
        ),
        pixels,
        opacity: 1.0,
        depth: 1.0,
        group: Group::Head,
        mesh,
        swing: None,
    }
}

fn rig(width: u32, height: u32, parts: Vec<RigPart>, masks: Vec<MaskDef>) -> Rig {
    Rig {
        id: next_rig_id(),
        width,
        height,
        parts,
        eyes: [None, None],
        anchors: Anchors::default(),
        face_scale: 1.0,
        notes: Vec::new(),
        masks,
        mouth: None,
    }
}

fn mask(id: u32, name: &str, parts: &[u32], clips: &[u32]) -> MaskDef {
    MaskDef {
        id: MaskId(id),
        name: name.into(),
        parts: parts.iter().map(|&p| PartId(p)).collect(),
        clips: clips.iter().map(|&p| PartId(p)).collect(),
    }
}

/// The rig at rest, each part at `alphas`.
fn at_rest(rig: &Rig, alphas: Vec<f32>) -> Frame {
    Frame::from_parts(
        rig.parts.iter().map(|p| p.mesh.rest.clone()).collect(),
        alphas,
    )
}

/// `list` drawn by a backend of `api`, offscreen, over `clear`.
fn draw(api: Api, assets: &Assets, list: &DrawList, clear: [f32; 4]) -> Vec<u8> {
    let [w, h] = list.target;
    let mut backend: Box<dyn Backend> = render::create(
        api,
        Surface::Offscreen {
            width: w,
            height: h,
        },
    )
    .expect("a backend");
    backend.resize(w, h).unwrap();
    backend.load(assets).unwrap();
    backend.draw(list, clear).unwrap();
    let mut out = vec![0; w as usize * h as usize * 4];
    backend.read_rgba(&mut out).unwrap();
    out
}

fn pixel(image: &[u8], width: u32, x: u32, y: u32) -> [u8; 4] {
    let at = ((y * width + x) * 4) as usize;
    [image[at], image[at + 1], image[at + 2], image[at + 3]]
}

// ---- assets ------------------------------------------------------------------------------------

#[test]
fn assets_premultiply_and_mip_every_part() {
    // Premultiplied as RigMetal.swift did: (c * a + 127) / 255.
    let a = picture(5, 3, |x, y| [(x * 10 + y) as u8, 0, 0, 255]);
    let b = picture(2, 1, |x, _| {
        if x == 0 {
            [200, 100, 50, 128]
        } else {
            [255, 255, 255, 0]
        }
    });
    // The wrong number of bytes, and no pixels at all.
    let c = RgbaImage::new(4, 4, vec![255; 12]);
    let d = RgbaImage::new(0, 0, Vec::new());
    let mut parts = Vec::new();
    for (i, pixels) in [a, b, c, d].into_iter().enumerate() {
        parts.push(part(
            &format!("p{i}"),
            Role::Other,
            None,
            pixels,
            flat([0.0, 0.0, 10.0, 10.0]),
        ));
    }
    let rig = rig(64, 32, parts, Vec::new());
    let assets = Assets::new(&rig);
    assert_eq!(assets.rig_id, rig.id);
    assert_eq!(assets.canvas_size, [64, 32]);
    assert_eq!(assets.parts.len(), 4);

    let sizes = |i: usize| -> Vec<(usize, usize)> {
        assets.parts[i]
            .mips
            .iter()
            .map(|m| (m.width, m.height))
            .collect()
    };
    assert_eq!(sizes(0), [(5, 3), (2, 1), (1, 1)]);
    assert_eq!(sizes(1), [(2, 1), (1, 1)]);
    assert_eq!(sizes(2), [(4, 4), (2, 2), (1, 1)]);
    assert_eq!(sizes(3), [(1, 1)]);
    for p in &assets.parts {
        assert!(!p.mips.is_empty());
        for m in &p.mips {
            assert_eq!(m.rgba.len(), m.width * m.height * 4);
        }
    }

    // 200 * 128 = 25600, (25600 + 127) / 255 = 100; 50 for 100, 25 for 50; a transparent texel
    // loses its colour.
    assert_eq!(assets.parts[1].mips[0].rgba, [100, 50, 25, 128, 0, 0, 0, 0]);
    // The 2x2 box, rounded: (100 + 0 + 1) / 2 = 50, (50 + 0 + 1) / 2 = 25, (25 + 1) / 2 = 13,
    // (128 + 1) / 2 = 64.
    assert_eq!(assets.parts[1].mips[1].rgba, [50, 25, 13, 64]);

    // 5 by 3 to 2 by 1: the first texel the 2 by 3 box of columns 0 and 1, (0+1+2+10+11+12) / 6
    // = 6; the last takes the odd column with its own, columns 2 to 4, a box of 9:
    // 3 * (20 + 30 + 40) + 3 * (0 + 1 + 2) = 279, (279 + 4) / 9 = 31. Then (6 + 31 + 1) / 2 = 19.
    let level1 = &assets.parts[0].mips[1].rgba;
    assert_eq!(level1[0], 6);
    assert_eq!(level1[4], 31);
    assert_eq!([level1[3], level1[7]], [255, 255]);
    assert_eq!(assets.parts[0].mips[2].rgba, [19, 0, 0, 255]);

    // The wrong length is transparent, the size it says; no pixels is one transparent texel.
    assert!(assets.parts[2]
        .mips
        .iter()
        .all(|m| m.rgba.iter().all(|&v| v == 0)));
    assert_eq!(assets.parts[3].mips[0].rgba, [0, 0, 0, 0]);

    // The grids as the parts have them.
    assert_eq!(assets.parts[0].uv, rig.parts[0].mesh.uv);
    assert_eq!(assets.parts[0].indices, rig.parts[0].mesh.indices);
}

// ---- the draw list -----------------------------------------------------------------------------

/// A body, two eyes -- white and iris each -- and a part too faint to draw. The masks' ids are
/// out of order and not from 0, as a rig may have them.
fn eyes_rig() -> Rig {
    let solid = |c: [u8; 4]| picture(4, 4, move |_, _| c);
    let parts = vec![
        part(
            "body",
            Role::Other,
            None,
            solid([200, 180, 160, 255]),
            flat([0.0, 0.0, 400.0, 300.0]),
        ),
        part(
            "eyewhite-l",
            Role::EyeWhite,
            Some(Side::Left),
            solid([255; 4]),
            flat([100.0, 100.0, 60.0, 30.0]),
        ),
        part(
            "irides-l",
            Role::Irides,
            Some(Side::Left),
            solid([0, 0, 255, 255]),
            flat([115.0, 90.0, 30.0, 50.0]),
        ),
        part(
            "eyewhite-r",
            Role::EyeWhite,
            Some(Side::Right),
            solid([255; 4]),
            flat([240.0, 100.0, 60.0, 30.0]),
        ),
        part(
            "irides-r",
            Role::Irides,
            Some(Side::Right),
            solid([0, 0, 255, 255]),
            flat([255.0, 90.0, 30.0, 50.0]),
        ),
        part(
            "faint",
            Role::Other,
            None,
            solid([255, 0, 0, 255]),
            flat([0.0, 0.0, 50.0, 50.0]),
        ),
    ];
    let masks = vec![
        mask(7, "eyewhite-left", &[1], &[2]),
        mask(3, "eyewhite-right", &[3], &[4]),
    ];
    rig(400, 300, parts, masks)
}

#[test]
fn the_draw_list_fits_the_canvas_and_decides_the_masks() {
    let rig = eyes_rig();
    let ids = |list: &DrawList| {
        list.draws
            .iter()
            .map(|d| (d.part.0, d.mask.map(|m| m.0)))
            .collect::<Vec<_>>()
    };

    // The canvas in the middle of the target: 400 by 300 in 800 by 800 is 800 by 600, 100 down.
    let list = render::draw_list(
        &rig,
        at_rest(&rig, vec![1.0, 1.0, 1.0, 1.0, 1.0, 0.003]),
        [800, 800],
    );
    assert_eq!(list.rig_id, rig.id);
    assert_eq!(list.canvas_size, [400, 300]);
    assert_eq!(list.target, [800, 800]);
    assert_eq!(list.viewport, Rect::new(0.0, 100.0, 800.0, 600.0));
    // Both eyes show: both masks, ids as the rig has them, each iris cut by its own; the faint
    // part left out.
    assert_eq!(
        list.masks,
        [
            Mask {
                id: MaskId(7),
                parts: vec![PartId(1)]
            },
            Mask {
                id: MaskId(3),
                parts: vec![PartId(3)]
            }
        ]
    );
    assert_eq!(
        ids(&list),
        [(0, None), (1, None), (2, Some(7)), (3, None), (4, Some(3))]
    );
    assert_eq!(list.frame.alphas.len(), 6);

    // Wider than the canvas: the canvas as tall as the target, in the middle across.
    let list = render::draw_list(&rig, at_rest(&rig, vec![1.0; 6]), [1000, 300]);
    assert_eq!(list.viewport, Rect::new(300.0, 0.0, 400.0, 300.0));
    assert_eq!(ids(&list).len(), 6);

    // The right iris does not show: no mask for it, though its white does. The left white is too
    // faint to draw, but its mask is made of it all the same, since its iris shows.
    let alphas = vec![1.0, MIN_ALPHA / 2.0, 0.5, 1.0, 0.0, 0.0];
    let list = render::draw_list(&rig, at_rest(&rig, alphas), [400, 300]);
    assert_eq!(
        list.masks,
        [Mask {
            id: MaskId(7),
            parts: vec![PartId(1)]
        }]
    );
    assert_eq!(ids(&list), [(0, None), (2, Some(7)), (3, None)]);
    assert_eq!(list.draws[1].alpha, 0.5);

    // Neither iris shows: no mask pass at all.
    let list = render::draw_list(
        &rig,
        at_rest(&rig, vec![1.0, 1.0, 0.0, 1.0, 0.001, 0.0]),
        [400, 300],
    );
    assert!(list.masks.is_empty());
    assert!(list.draws.iter().all(|d| d.mask.is_none()));

    // Every rule a list keeps: no mask twice, every draw's mask listed.
    for alphas in [
        vec![1.0; 6],
        vec![0.0, 0.0, 1.0, 0.0, 1.0, 0.0],
        vec![0.0; 6],
    ] {
        let list = render::draw_list(&rig, at_rest(&rig, alphas), [64, 48]);
        let mut seen: Vec<MaskId> = list.masks.iter().map(|m| m.id).collect();
        seen.sort();
        seen.dedup();
        assert_eq!(seen.len(), list.masks.len());
        assert!(list
            .draws
            .iter()
            .all(|d| d.mask.is_none_or(|id| seen.contains(&id))));
        assert!(list.draws.iter().all(|d| d.alpha >= MIN_ALPHA));
    }
}

// ---- the CPU reference -------------------------------------------------------------------------

#[test]
fn the_cpu_draws_a_textured_quad_where_it_belongs() {
    // A quad from (20, 30) to (60, 70) on a 100 by 100 canvas, its picture red at the top left,
    // green at the top right, blue at the bottom left, white at the bottom right.
    let quadrants = picture(8, 8, |x, y| match (x < 4, y < 4) {
        (true, true) => [255, 0, 0, 255],
        (false, true) => [0, 255, 0, 255],
        (true, false) => [0, 0, 255, 255],
        (false, false) => [255, 255, 255, 255],
    });
    let rig = rig(
        100,
        100,
        vec![part(
            "quad",
            Role::Other,
            None,
            quadrants,
            flat([20.0, 30.0, 40.0, 40.0]),
        )],
        vec![],
    );
    let assets = Assets::new(&rig);
    let list = render::draw_list(&rig, at_rest(&rig, vec![1.0]), [100, 100]);
    let image = draw(Api::Cpu, &assets, &list, [0.0; 4]);
    let at = |x, y| pixel(&image, 100, x, y);
    assert_eq!(at(5, 5), [0, 0, 0, 0]);
    assert_eq!(at(25, 35), [255, 0, 0, 255]);
    assert_eq!(at(55, 35), [0, 255, 0, 255]);
    assert_eq!(at(25, 65), [0, 0, 255, 255]);
    assert_eq!(at(55, 65), [255, 255, 255, 255]);
    // Its edges, by pixel centres: 20.5 is inside, 19.5 is not; 59.5 is, 60.5 not.
    assert_eq!(at(19, 50)[3], 0);
    assert_eq!(at(20, 50)[3], 255);
    assert_eq!(at(59, 50)[3], 255);
    assert_eq!(at(60, 50)[3], 0);
    assert_eq!(at(40, 29)[3], 0);
    assert_eq!(at(40, 30)[3], 255);
    assert_eq!(at(40, 69)[3], 255);
    assert_eq!(at(40, 70)[3], 0);

    // Twice the size, letterboxed: the canvas is 200 by 200 at (50, 0) in 300 by 200, and the
    // quad half faint over a clear colour.
    let list = render::draw_list(&rig, at_rest(&rig, vec![0.5]), [300, 200]);
    let image = draw(Api::Cpu, &assets, &list, [0.0, 0.0, 0.0, 1.0]);
    let at = |x, y| pixel(&image, 300, x, y);
    assert_eq!(at(10, 10), [0, 0, 0, 255]);
    // 255 * 0.5 = 127.5, rounded to the nearest: 128; over black, opaque.
    assert_eq!(at(50 + 50, 70), [128, 0, 0, 255]);
    assert_eq!(at(50 + 110, 130), [128, 128, 128, 255]);
    // The quad's left edge at 90, a quarter of a canvas pixel in, in its blue quarter.
    assert_eq!(at(89, 120), [0, 0, 0, 255]);
    assert_eq!(at(90, 120), [0, 0, 128, 255]);
}

#[test]
fn the_cpu_keeps_the_iris_inside_the_white() {
    // RigTests.swift's testMetalDrawsTheRigAndKeepsTheIrisInsideTheWhite, by hand: a face, a white
    // over it, and an iris taller than the white, which shows only where the white is.
    let face = picture(4, 4, |_, _| [250, 220, 200, 255]);
    let white = ellipse(64, 32, 2.0, [255, 255, 255]);
    let iris = picture(8, 8, |_, _| [40, 90, 200, 255]);
    let parts = vec![
        part(
            "face",
            Role::Face,
            None,
            face,
            flat([0.0, 0.0, 200.0, 200.0]),
        ),
        part(
            "eyewhite",
            Role::EyeWhite,
            Some(Side::Left),
            white,
            flat([40.0, 70.0, 80.0, 40.0]),
        ),
        part(
            "irides",
            Role::Irides,
            Some(Side::Left),
            iris,
            flat([65.0, 60.0, 30.0, 60.0]),
        ),
    ];
    let rig = rig(200, 200, parts, vec![mask(0, "eyewhite-left", &[1], &[2])]);
    let assets = Assets::new(&rig);
    let list = render::draw_list(&rig, at_rest(&rig, vec![1.0; 3]), [200, 200]);
    assert_eq!(list.draws[2].mask, Some(MaskId(0)));
    let image = draw(Api::Cpu, &assets, &list, [0.0; 4]);
    let at = |x, y| pixel(&image, 200, x, y);
    // The face where it is; the iris over the white, in its middle; the white beside the iris.
    assert_eq!(at(100, 160), [250, 220, 200, 255]);
    assert_eq!(at(80, 90), [40, 90, 200, 255]);
    assert_eq!(at(50, 90), [255, 255, 255, 255]);
    // Above and below the white, the iris's picture reaches, but it is cut: the face shows.
    assert_eq!(at(80, 63), [250, 220, 200, 255]);
    assert_eq!(at(80, 116), [250, 220, 200, 255]);
    // Uncut, it would not be.
    let mut uncut = list.clone();
    uncut.draws[2].mask = None;
    uncut.masks.clear();
    let image = draw(Api::Cpu, &assets, &uncut, [0.0; 4]);
    assert_eq!(pixel(&image, 200, 80, 63), [40, 90, 200, 255]);

    // The white too faint to draw still cuts the iris: its mask is the white's texels.
    let list = render::draw_list(&rig, at_rest(&rig, vec![1.0, 0.0, 1.0]), [200, 200]);
    let image = draw(Api::Cpu, &assets, &list, [0.0; 4]);
    assert_eq!(pixel(&image, 200, 80, 90), [40, 90, 200, 255]);
    assert_eq!(pixel(&image, 200, 50, 90), [250, 220, 200, 255]);
    assert_eq!(pixel(&image, 200, 80, 63), [250, 220, 200, 255]);
}

// ---- conformance -------------------------------------------------------------------------------

/// The fixture every backend is held to the reference on: overlapping semi-transparent parts, a
/// grid bent out of its rectangle, a picture drawn far smaller than it is, two masks with soft
/// edges, a part that reaches past the canvas, and triangles of both windings.
fn fixture() -> Rig {
    let backdrop = picture(64, 48, |x, y| [(x * 4) as u8, (y * 5) as u8, 128, 200]);
    let cloth = picture(256, 192, |x, y| {
        let check = ((x / 16 + y / 16) % 2) as u8;
        let a = 120 + ((x + y) % 136) as u8;
        [40 + check * 180, (x % 256) as u8, (y * 255 / 191) as u8, a]
    });
    // Fine stripes and dots: what a mip chain averages away.
    let fine = picture(512, 512, |x, y| {
        let stripe = if (x / 3 + y / 5) % 2 == 0 { 255 } else { 30 };
        let dot = if (x % 7 < 2) && (y % 9 < 3) { 0 } else { 255 };
        [
            stripe,
            dot,
            ((x ^ y) & 255) as u8,
            if (x + 2 * y) % 11 < 8 { 255 } else { 90 },
        ]
    });
    let spill = picture(32, 32, |x, y| [255, (x * 8) as u8, (y * 8) as u8, 160]);
    let iris = picture(48, 48, |x, y| {
        let r = (((x as f32 - 23.5).powi(2) + (y as f32 - 23.5).powi(2)).sqrt() * 10.0) as u32;
        [(r % 256) as u8, 60, 200, 255]
    });

    let bend = |x: f32, y: f32| {
        [
            x + 18.0 * ((y - 40.0) / 220.0 * 3.1).sin(),
            y + 12.0 * ((x - 50.0) / 300.0 * 4.3).cos(),
        ]
    };
    let parts = vec![
        part(
            "backdrop",
            Role::Other,
            None,
            backdrop,
            grid([0.0, 0.0, 400.0, 300.0], 3, 2, |x, y| [x, y]),
        ),
        part(
            "cloth",
            Role::Other,
            None,
            cloth,
            grid([50.0, 40.0, 300.0, 220.0], 6, 5, bend),
        ),
        part(
            "charm",
            Role::Other,
            None,
            fine,
            grid([300.0, 20.0, 40.0, 40.0], 2, 2, |x, y| [x + 0.1 * y, y]),
        ),
        part(
            "spill",
            Role::Other,
            None,
            spill,
            flat([-30.0, 230.0, 110.0, 100.0]),
        ),
        part(
            "eyewhite-l",
            Role::EyeWhite,
            Some(Side::Left),
            ellipse(96, 48, 6.0, [250, 250, 245]),
            flat([110.0, 110.0, 70.0, 36.0]),
        ),
        part(
            "irides-l",
            Role::Irides,
            Some(Side::Left),
            iris.clone(),
            grid([128.0, 98.0, 34.0, 58.0], 2, 3, |x, y| [x, y + 0.05 * x]),
        ),
        part(
            "eyewhite-r",
            Role::EyeWhite,
            Some(Side::Right),
            ellipse(96, 48, 10.0, [245, 250, 250]),
            flat([220.0, 108.0, 72.0, 40.0]),
        ),
        part(
            "irides-r",
            Role::Irides,
            Some(Side::Right),
            iris,
            grid([236.0, 96.0, 36.0, 60.0], 2, 3, |x, y| [x, y]),
        ),
        part(
            "veil",
            Role::Other,
            None,
            ellipse(128, 128, 40.0, [180, 40, 120]),
            grid([150.0, 60.0, 160.0, 200.0], 4, 4, bend),
        ),
    ];
    let masks = vec![
        mask(7, "eyewhite-left", &[4], &[5]),
        mask(3, "eyewhite-right", &[6], &[7]),
    ];
    rig(400, 300, parts, masks)
}

/// How close two images are: the share of channel values within 2/255, and the largest gap.
fn compare(a: &[u8], b: &[u8]) -> (f64, u8) {
    let close = a
        .iter()
        .zip(b)
        .filter(|(x, y)| x.abs_diff(**y) <= 2)
        .count();
    let worst = a
        .iter()
        .zip(b)
        .map(|(x, y)| x.abs_diff(*y))
        .max()
        .unwrap_or(0);
    (close as f64 / a.len() as f64, worst)
}

#[test]
fn every_backend_draws_what_the_reference_draws() {
    let rig = fixture();
    let assets = Assets::new(&rig);
    let alphas = vec![1.0, 0.85, 0.9, 0.7, 0.6, 1.0, 1.0, 0.8, 0.45];
    let clear = [0.05, 0.1, 0.15, 0.25];
    let apis: Vec<Api> = [Api::Metal, Api::D3D11, Api::Vulkan]
        .into_iter()
        .filter(|&a| render::is_available(a))
        .collect();
    if apis.is_empty() {
        eprintln!("skipped: no GPU backend is available here to hold to the reference");
    }
    let mut failures = Vec::new();
    for size in [[64, 48], [333, 500], [1024, 768]] {
        let list = render::draw_list(&rig, at_rest(&rig, alphas.clone()), size);
        assert_eq!(list.masks.len(), 2);
        let reference = draw(Api::Cpu, &assets, &list, clear);
        // The reference draws something everywhere the canvas is.
        let centre = pixel(&reference, size[0], size[0] / 2, size[1] / 2);
        assert!(centre[3] > 128, "{centre:?}");
        for &api in &apis {
            let image = draw(api, &assets, &list, clear);
            let (close, worst) = compare(&reference, &image);
            println!(
                "{api:?} at {}x{}: {:.4}% of channel values within 2/255, the largest gap {worst}/255",
                size[0],
                size[1],
                close * 100.0
            );
            if close < 0.995 || worst > 8 {
                failures.push(format!(
                    "{api:?} at {size:?}: {:.4}% within 2/255, {worst}/255 off",
                    close * 100.0
                ));
            }
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

#[test]
fn a_backend_refuses_a_list_it_cannot_draw() {
    let rig = eyes_rig();
    let assets = Assets::new(&rig);
    let mut cpu = render::create(
        Api::Cpu,
        Surface::Offscreen {
            width: 40,
            height: 30,
        },
    )
    .unwrap();
    let list = render::draw_list(&rig, at_rest(&rig, vec![1.0; 6]), [40, 30]);
    // Nothing loaded yet.
    assert!(cpu.draw(&list, [0.0; 4]).is_err());
    cpu.load(&assets).unwrap();
    cpu.draw(&list, [0.0; 4]).unwrap();
    // Another rig's frame; a frame for another size.
    let other = eyes_rig();
    assert!(cpu
        .draw(
            &render::draw_list(&other, at_rest(&other, vec![1.0; 6]), [40, 30]),
            [0.0; 4]
        )
        .is_err());
    assert!(cpu
        .draw(
            &render::draw_list(&rig, at_rest(&rig, vec![1.0; 6]), [41, 30]),
            [0.0; 4]
        )
        .is_err());
    // A buffer the wrong size.
    assert!(cpu.read_rgba(&mut [0; 7]).is_err());
    // The CPU draws offscreen only.
    let layer = std::ptr::NonNull::<u8>::dangling().cast();
    assert!(render::create(Api::Cpu, Surface::MetalLayer { layer }).is_err());
}

#[cfg(target_os = "macos")]
#[test]
fn metal_draws_into_a_layer_the_host_owns() {
    use objc2_metal::MTLPixelFormat;
    use objc2_quartz_core::CAMetalLayer;
    if !render::is_available(Api::Metal) {
        eprintln!("skipped: no Metal device");
        return;
    }
    let layer = CAMetalLayer::new();
    let pointer = std::ptr::NonNull::from(&*layer).cast();
    let mut metal = render::create(Api::Metal, Surface::MetalLayer { layer: pointer }).unwrap();
    // The backend sets the layer up for itself, and sizes its drawables.
    assert!(layer.device().is_some());
    assert_eq!(layer.pixelFormat(), MTLPixelFormat::BGRA8Unorm);
    metal.resize(120, 90).unwrap();
    let size = layer.drawableSize();
    assert_eq!([size.width, size.height], [120.0, 90.0]);

    let rig = eyes_rig();
    metal.load(&Assets::new(&rig)).unwrap();
    for _ in 0..3 {
        let list = render::draw_list(&rig, at_rest(&rig, vec![1.0; 6]), [120, 90]);
        metal.draw(&list, [0.0; 4]).unwrap();
    }
    // A layer is presented, not read back.
    assert!(matches!(
        metal.read_rgba(&mut vec![0; 120 * 90 * 4]),
        Err(waifu2d::Error::Unsupported(_))
    ));
    drop(metal);
    // The host's layer outlives the backend.
    assert_eq!(layer.drawableSize().width, 120.0);
}
