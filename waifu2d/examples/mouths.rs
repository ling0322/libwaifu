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

//! The mouth waifu2d makes, on a document's face, to look at:
//!
//!     cargo run --release -p waifu2d --example mouths -- drawing.psd out/
//!
//! writes out/mouths.png, the mouth close up in each viseme -- A E I O U @ MBP REST, left to right
//! and top to bottom -- out/faces.png, the face in each, and out/morph.png, the mouth on its way
//! from one shape to the next: a row a change, six steps along it.

use std::fs::File;
use std::io::BufWriter;
use std::path::Path;

use waifu2d::mouths::{self, Viseme};
use waifu2d::render::{self, Api, Assets, Surface};
use waifu2d::rig::Rig;
use waifu2d::{Motion, Rect};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let (psd, out) = (&args[1], Path::new(&args[2]));
    std::fs::create_dir_all(out).unwrap();
    let rig = mouths::with_drawn_mouths(Rig::read_psd(&std::fs::read(psd).unwrap()).unwrap());
    let mouth = mouths::drawn_mouth(&rig).expect("a document with a mouth");
    let m = rig.part(mouth).rect;
    let face = rig.anchors.face;

    let api = if render::is_available(Api::Metal) {
        Api::Metal
    } else {
        Api::Cpu
    };
    let (w, h) = (rig.width, rig.height);
    let mut backend = render::create(
        api,
        Surface::Offscreen {
            width: w,
            height: h,
        },
    )
    .unwrap();
    backend.resize(w, h).unwrap();
    backend.load(&Assets::new(&rig)).unwrap();
    let mut motion = Motion::new(&rig, 0);
    let mut draw = |weights: [f32; 8]| {
        motion.pose.visemes = weights;
        let frame = motion.frame(&rig, false);
        backend
            .draw(
                &render::draw_list(&rig, frame, [w, h]),
                [1.0, 1.0, 1.0, 1.0],
            )
            .unwrap();
        let mut pixels = vec![0u8; (w * h * 4) as usize];
        backend.read_rgba(&mut pixels).unwrap();
        pixels
    };

    // The mouth with room around it, three times over; the face, as it is.
    let close = Rect::new(
        m.mid_x() - 1.0 * m.width,
        m.mid_y() - 0.7 * m.width,
        2.0 * m.width,
        1.4 * m.width,
    );
    let mut closeups = Vec::new();
    let mut faces = Vec::new();
    for v in Viseme::ALL {
        let pixels = draw(v.alone());
        closeups.push(crop(&pixels, w, close, 3));
        faces.push(crop(&pixels, w, face, 1));
    }
    save(&sheet(&closeups, 4), &out.join("mouths.png"));
    save(&sheet(&faces, 4), &out.join("faces.png"));

    // Each change, six steps along it.
    let changes = [
        (Viseme::Rest, Viseme::A),
        (Viseme::A, Viseme::O),
        (Viseme::O, Viseme::I),
        (Viseme::I, Viseme::U),
        (Viseme::U, Viseme::Mbp),
        (Viseme::Mbp, Viseme::Rest),
    ];
    let mut steps = Vec::new();
    for (from, to) in changes {
        for k in 0..6 {
            let t = k as f32 / 5.0;
            let mut weights = [0.0; 8];
            weights[from as usize] += 1.0 - t;
            weights[to as usize] += t;
            steps.push(crop(&draw(weights), w, close, 2));
        }
    }
    save(&sheet(&steps, 6), &out.join("morph.png"));
    println!(
        "{api:?}: {} parts, mouth at {m:?}; wrote mouths.png, faces.png and morph.png",
        rig.parts.len()
    );
}

struct Picture {
    width: usize,
    height: usize,
    rgba: Vec<u8>,
}

/// `rect` of a `width`-wide picture, each pixel `scale` by `scale`.
fn crop(pixels: &[u8], width: u32, rect: Rect, scale: usize) -> Picture {
    let (x0, y0) = (rect.x.max(0.0) as usize, rect.y.max(0.0) as usize);
    let (cw, ch) = (rect.width as usize, rect.height as usize);
    let mut rgba = vec![255u8; cw * scale * ch * scale * 4];
    for y in 0..ch * scale {
        for x in 0..cw * scale {
            let (sx, sy) = (x0 + x / scale, y0 + y / scale);
            if sx < width as usize && sy * width as usize * 4 < pixels.len() {
                let s = (sy * width as usize + sx) * 4;
                let d = (y * cw * scale + x) * 4;
                rgba[d..d + 4].copy_from_slice(&pixels[s..s + 4]);
            }
        }
    }
    Picture {
        width: cw * scale,
        height: ch * scale,
        rgba,
    }
}

/// The pictures in rows of `across`, with a grey gutter between them.
fn sheet(pictures: &[Picture], across: usize) -> Picture {
    let gap = 6;
    let (pw, ph) = (pictures[0].width, pictures[0].height);
    let down = pictures.len().div_ceil(across);
    let (width, height) = (across * (pw + gap) + gap, down * (ph + gap) + gap);
    let mut rgba = vec![160u8; width * height * 4];
    for (i, p) in pictures.iter().enumerate() {
        let (ox, oy) = (
            gap + (i % across) * (pw + gap),
            gap + (i / across) * (ph + gap),
        );
        for y in 0..ph {
            let d = ((oy + y) * width + ox) * 4;
            rgba[d..d + pw * 4].copy_from_slice(&p.rgba[y * pw * 4..(y + 1) * pw * 4]);
        }
    }
    Picture {
        width,
        height,
        rgba,
    }
}

fn save(p: &Picture, path: &Path) {
    let mut encoder = png::Encoder::new(
        BufWriter::new(File::create(path).unwrap()),
        p.width as u32,
        p.height as u32,
    );
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    encoder
        .write_header()
        .unwrap()
        .write_image_data(&p.rgba)
        .unwrap();
}
