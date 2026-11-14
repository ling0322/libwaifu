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

//! What the tests share: a Photoshop document written by hand, as little of one as the reader
//! reads -- RigTests.swift's writer -- and the face the Swift tests drew.

#![allow(dead_code)]

/// A layer of a Photoshop document written by hand.
#[derive(Clone, Default)]
pub struct Layer {
    pub name: String,
    pub left: i64,
    pub top: i64,
    pub width: i64,
    pub height: i64,
    /// RGBA, not premultiplied.
    pub rgba: Vec<u8>,
    pub hidden: bool,
    pub pack_bits: bool,
    /// 1 opens a group, 3 closes one, 0 is a layer.
    pub divider: i64,
}

/// A Photoshop document written by hand, as little of one as the reader reads.
#[derive(Default)]
pub struct Writer {
    pub bytes: Vec<u8>,
}

impl Writer {
    pub fn u8(&mut self, v: i64) {
        self.bytes.push((v & 0xFF) as u8);
    }

    pub fn u16(&mut self, v: i64) {
        self.u8(v >> 8);
        self.u8(v);
    }

    pub fn u32(&mut self, v: i64) {
        self.u16(v >> 16);
        self.u16(v);
    }

    pub fn ascii(&mut self, s: &str) {
        self.bytes.extend_from_slice(s.as_bytes());
    }

    /// `layers` bottom first, as the format keeps them.
    pub fn document(width: i64, height: i64, layers: &[Layer]) -> Vec<u8> {
        let mut w = Writer::default();
        w.ascii("8BPS");
        w.u16(1);
        w.bytes.extend_from_slice(&[0; 6]);
        w.u16(3);
        w.u32(height);
        w.u32(width);
        w.u16(8);
        w.u16(3);
        w.u32(0);
        w.u32(0);

        let mut info = Writer::default();
        info.u16(layers.len() as i64);
        let mut data = Writer::default();
        for layer in layers {
            // Each channel compressed now, so that its length is known: alpha first.
            let mut channels: Vec<(i64, Vec<u8>)> = Vec::new();
            for (id, offset) in [(-1, 3), (0, 0), (1, 1), (2, 2)] {
                let mut c = Writer::default();
                let plane: Vec<u8> = layer.rgba.iter().skip(offset).step_by(4).copied().collect();
                if layer.pack_bits {
                    c.u16(1);
                    // Every row one literal run, or one repeat where it is all one byte.
                    let mut rows: Vec<Vec<u8>> = Vec::new();
                    for y in 0..layer.height as usize {
                        let w = layer.width as usize;
                        let row = &plane[y * w..(y + 1) * w];
                        if row.iter().all(|&b| b == row[0]) {
                            rows.push(vec![(1 - row.len() as i64) as i8 as u8, row[0]]);
                        } else {
                            let mut literal = vec![(row.len() - 1) as u8];
                            literal.extend_from_slice(row);
                            rows.push(literal);
                        }
                    }
                    for row in &rows {
                        c.u16(row.len() as i64);
                    }
                    for row in &rows {
                        c.bytes.extend_from_slice(row);
                    }
                } else {
                    c.u16(0);
                    c.bytes.extend_from_slice(&plane);
                }
                channels.push((id, c.bytes));
            }
            info.u32(layer.top);
            info.u32(layer.left);
            info.u32(layer.top + layer.height);
            info.u32(layer.left + layer.width);
            info.u16(channels.len() as i64);
            for (id, c) in &channels {
                info.u16(*id);
                info.u32(c.len() as i64);
            }
            info.ascii("8BIMnorm");
            info.u8(255);
            info.u8(0);
            info.u8(if layer.hidden { 2 } else { 0 });
            info.u8(0);

            let mut extra = Writer::default();
            extra.u32(0);
            extra.u32(0);
            // A short name in ASCII, padded to four; the whole of it in UTF-16 under "luni".
            let short: Vec<u8> = layer
                .name
                .bytes()
                .take(31)
                .map(|b| if b < 128 { b } else { b'?' })
                .collect();
            extra.u8(short.len() as i64);
            extra.bytes.extend_from_slice(&short);
            while (extra.bytes.len() - 8) % 4 != 0 {
                extra.u8(0);
            }
            let units: Vec<u16> = layer.name.encode_utf16().collect();
            extra.ascii("8BIMluni");
            extra.u32(4 + units.len() as i64 * 2);
            extra.u32(units.len() as i64);
            for u in units {
                extra.u16(i64::from(u));
            }
            if layer.divider != 0 {
                extra.ascii("8BIMlsct");
                extra.u32(4);
                extra.u32(layer.divider);
            }
            info.u32(extra.bytes.len() as i64);
            info.bytes.extend_from_slice(&extra.bytes);

            for (_, c) in &channels {
                data.bytes.extend_from_slice(c);
            }
        }
        let mut section = Writer::default();
        section.u32((info.bytes.len() + data.bytes.len()) as i64);
        section.bytes.extend_from_slice(&info.bytes);
        section.bytes.extend_from_slice(&data.bytes);
        section.u32(0);
        w.u32(section.bytes.len() as i64);
        w.bytes.extend_from_slice(&section.bytes);
        w.bytes
    }
}

/// A layer the size of `bounds` -- left, top, width, height -- `colour` where `inside` says,
/// clear elsewhere.
pub fn layer_where(
    name: &str,
    bounds: (i64, i64, i64, i64),
    colour: [u8; 4],
    inside: impl Fn(i64, i64) -> bool,
) -> Layer {
    let (left, top, width, height) = bounds;
    let mut rgba = Vec::new();
    for y in 0..height {
        for x in 0..width {
            rgba.extend_from_slice(if inside(x, y) { &colour } else { &[0, 0, 0, 0] });
        }
    }
    Layer {
        name: name.to_string(),
        left,
        top,
        width,
        height,
        rgba,
        ..Layer::default()
    }
}

/// A layer the size of `bounds`, `colour` all over.
pub fn layer(name: &str, bounds: (i64, i64, i64, i64), colour: [u8; 4]) -> Layer {
    layer_where(name, bounds, colour, |_, _| true)
}

/// A face 100 wide in the middle of a 200 by 200 canvas, and both eyes on it drawn as one layer
/// each: whites, irides, lashes.
pub fn face_layers() -> Vec<Layer> {
    let both = |x: i64, _| !(30..50).contains(&x);
    vec![
        layer("face", (50, 40, 100, 120), [250, 220, 200, 255]),
        layer_where("eyewhite", (65, 80, 80, 20), [255, 255, 255, 255], both),
        layer_where("irides", (70, 82, 70, 16), [40, 90, 200, 255], |x, _| {
            !(20..50).contains(&x)
        }),
        layer_where("eyelash", (63, 76, 84, 6), [60, 30, 30, 255], |x, _| {
            !(32..52).contains(&x)
        }),
        layer("front hair", (40, 20, 120, 30), [80, 160, 160, 255]),
    ]
}

/// The face as the bytes of a document.
pub fn face_psd() -> Vec<u8> {
    Writer::document(200, 200, &face_layers())
}

/// A figure 300 by 400 with as much on it as moves: hair in front and behind with ragged tips to
/// hang strands from, two eyes, a mouth, a neck and a top. What tests/parity.rs moves, as the Mac
/// app moved it.
pub fn figure_layers() -> Vec<Layer> {
    let eyes = |x: i64, _| !(30..50).contains(&x);
    vec![
        // Tips across the bottom of the back hair, and across the fringe.
        layer_where(
            "back hair",
            (60, 40, 180, 260),
            [60, 40, 40, 255],
            |x, y| y < 200 + 60 * ((x / 60) % 2) - (x % 60 - 30).abs(),
        ),
        layer("neck", (130, 200, 40, 60), [240, 210, 190, 255]),
        layer("topwear", (60, 250, 180, 150), [40, 60, 120, 255]),
        layer("face", (100, 60, 100, 150), [250, 220, 200, 255]),
        layer_where("eyewhite", (110, 100, 80, 20), [255, 255, 255, 255], eyes),
        layer_where("irides", (115, 102, 70, 16), [40, 90, 200, 255], |x, _| {
            !(20..50).contains(&x)
        }),
        layer_where("eyelash", (108, 96, 84, 6), [60, 30, 30, 255], |x, _| {
            !(32..52).contains(&x)
        }),
        layer("mouth", (135, 160, 30, 12), [150, 40, 50, 255]),
        layer_where(
            "front hair",
            (80, 30, 140, 70),
            [60, 40, 40, 255],
            |x, y| y < 40 + (14 - (x % 28 - 14).abs()) * 2,
        ),
    ]
}

/// The figure as the bytes of a document.
pub fn figure_psd() -> Vec<u8> {
    Writer::document(300, 400, &figure_layers())
}
