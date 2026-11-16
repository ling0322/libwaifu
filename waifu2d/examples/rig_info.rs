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

//! What a drawing's rig is made of, part by part, from a package (.waifu2d) or a PSD:
//!
//!     cargo run -p waifu2d --example rig_info -- drawing.waifu2d

use waifu2d::rig::Rig;

fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("usage: rig_info DRAWING.waifu2d|DRAWING.psd");
    let rig = Rig::read(&std::fs::read(&path).expect("read the drawing")).expect("rig it");
    println!(
        "{} x {}, face scale {:.3}",
        rig.width, rig.height, rig.face_scale
    );
    for (i, p) in rig.parts.iter().enumerate() {
        let r = p.rect;
        println!(
            "{i:3} {:<24} slot {:<14} {:?} {:?} {:?} depth {:.2} at ({:.0}, {:.0}) {:.0} x {:.0}",
            p.name, p.slot, p.role, p.side, p.group, p.depth, r.x, r.y, r.width, r.height
        );
    }
    println!(
        "eyes {:?}",
        rig.eyes
            .iter()
            .map(|e| e.map(|e| e.white))
            .collect::<Vec<_>>()
    );
    println!("anchors {:?}", rig.anchors);
    println!("notes {:?}", rig.notes);
}
