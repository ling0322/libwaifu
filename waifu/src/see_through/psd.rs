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

//! A Photoshop document: as little of the format as a stack of pixel layers needs.
//!
//! Version 1, eight bits a channel, RGB or grayscale, each with an alpha. Every layer is a rectangle
//! of pixels with a name and a transparency channel, at full opacity, blended normally; there are no
//! groups, masks or effects. Channels are compressed with PackBits, which is what Photoshop itself
//! writes, and the merged picture after the layers -- what a viewer that does not read layers shows
//! -- is the layers flattened in order, its alpha the document's first extra channel. The layer
//! count is written negative to say so: with a positive count a reader takes that channel for an
//! ordinary saved selection and shows the merged picture opaque.
//!
//! The layout is Adobe's "Photoshop File Formats Specification", sections Header, Color Mode Data,
//! Image Resources, Layer and Mask Information, and Image Data, all big-endian.

use super::image::Image;

/// One layer: its pixels, where its top left corner sits on the canvas, and its name.
///
/// `pixels` is the document's colour channels then an alpha, not premultiplied: RGBA for an RGB
/// document, gray and alpha for a grayscale one.
#[derive(Clone, Debug)]
pub struct Layer {
    pub name: String,
    pub left: i32,
    pub top: i32,
    pub pixels: Image<u8>,
}

/// What kind of document: the colour mode the header names, and how many channels a layer has.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    Rgb,
    Grayscale,
}

impl Mode {
    fn code(self) -> u16 {
        match self {
            Mode::Rgb => 3,
            Mode::Grayscale => 1,
        }
    }

    fn colour_channels(self) -> usize {
        match self {
            Mode::Rgb => 3,
            Mode::Grayscale => 1,
        }
    }

    /// A layer's channels: the colour ones, then the alpha.
    fn layer_channels(self) -> usize {
        self.colour_channels() + 1
    }
}

struct Writer(Vec<u8>);

impl Writer {
    fn u8(&mut self, v: u8) {
        self.0.push(v);
    }
    fn u16(&mut self, v: u16) {
        self.0.extend_from_slice(&v.to_be_bytes());
    }
    fn i16(&mut self, v: i16) {
        self.0.extend_from_slice(&v.to_be_bytes());
    }
    fn u32(&mut self, v: u32) {
        self.0.extend_from_slice(&v.to_be_bytes());
    }
    fn i32(&mut self, v: i32) {
        self.0.extend_from_slice(&v.to_be_bytes());
    }
    fn bytes(&mut self, v: &[u8]) {
        self.0.extend_from_slice(v);
    }
    /// A length to be filled in once what it measures is written.
    fn reserve(&mut self) -> usize {
        self.u32(0);
        self.0.len()
    }
    fn fill(&mut self, at: usize) {
        let length = (self.0.len() - at) as u32;
        self.0[at - 4..at].copy_from_slice(&length.to_be_bytes());
    }
}

/// One row in PackBits: runs of a repeated byte as `(1 - n, byte)`, everything else as
/// `(n - 1, bytes...)`, at most 128 either way.
fn packbits(row: &[u8], out: &mut Vec<u8>) {
    let mut i = 0;
    while i < row.len() {
        let mut run = 1;
        while i + run < row.len() && run < 128 && row[i + run] == row[i] {
            run += 1;
        }
        if run >= 2 {
            out.push((1i32 - run as i32) as i8 as u8);
            out.push(row[i]);
            i += run;
            continue;
        }
        let start = i;
        while i < row.len() && i - start < 128 {
            if i + 1 < row.len() && row[i + 1] == row[i] {
                break;
            }
            i += 1;
        }
        if i == start {
            i += 1;
        }
        out.push((i - start - 1) as u8);
        out.extend_from_slice(&row[start..i]);
    }
}

/// One channel of `width` by `height`, compressed: the compression code, every row's byte count,
/// then the rows. `rows_only` leaves off the code, which the merged picture writes once for all
/// of its channels.
fn rle_channel(plane: &[u8], width: usize, height: usize, rows_only: bool) -> Vec<u8> {
    let mut counts = Vec::with_capacity(height * 2);
    let mut data = Vec::new();
    for y in 0..height {
        let before = data.len();
        packbits(&plane[y * width..(y + 1) * width], &mut data);
        counts.extend_from_slice(&((data.len() - before) as u16).to_be_bytes());
    }
    let mut out = Vec::with_capacity(2 + counts.len() + data.len());
    if !rows_only {
        out.extend_from_slice(&1u16.to_be_bytes());
    }
    out.extend_from_slice(&counts);
    out.extend_from_slice(&data);
    out
}

fn plane(pixels: &Image<u8>, c: usize) -> Vec<u8> {
    pixels
        .data
        .iter()
        .skip(c)
        .step_by(pixels.channels)
        .copied()
        .collect()
}

/// The layers laid over each other in order on a `width` by `height` canvas, over transparent
/// black, as planes: colour, then alpha.
fn flatten(layers: &[Layer], width: usize, height: usize, mode: Mode) -> Vec<Vec<u8>> {
    let n = mode.colour_channels();
    let mut colour = vec![vec![0f32; width * height]; n];
    let mut alpha = vec![0f32; width * height];
    for layer in layers {
        let p = &layer.pixels;
        for y in 0..p.height {
            let cy = layer.top as i64 + y as i64;
            if cy < 0 || cy >= height as i64 {
                continue;
            }
            for x in 0..p.width {
                let cx = layer.left as i64 + x as i64;
                if cx < 0 || cx >= width as i64 {
                    continue;
                }
                let at = cy as usize * width + cx as usize;
                let a = p.at(x, y, n) as f32 / 255.0;
                let out_a = a + alpha[at] * (1.0 - a);
                for (c, plane) in colour.iter_mut().enumerate() {
                    let v = p.at(x, y, c) as f32;
                    plane[at] = match out_a > 0.0 {
                        true => (v * a + plane[at] * alpha[at] * (1.0 - a)) / out_a,
                        false => 0.0,
                    };
                }
                alpha[at] = out_a;
            }
        }
    }
    let mut planes: Vec<Vec<u8>> = colour
        .into_iter()
        .map(|p| {
            p.into_iter()
                .map(|v| v.round().clamp(0.0, 255.0) as u8)
                .collect()
        })
        .collect();
    planes.push(
        alpha
            .into_iter()
            .map(|a| (a * 255.0).round().clamp(0.0, 255.0) as u8)
            .collect(),
    );
    planes
}

/// A whole document: `layers` bottom first on a `width` by `height` canvas.
///
/// Every layer's pixels have [`Mode::layer_channels`] channels; a layer with any other number is a
/// caller's mistake, and panics.
pub fn write(layers: &[Layer], width: usize, height: usize, mode: Mode) -> Vec<u8> {
    let colour = mode.colour_channels();
    for layer in layers {
        assert_eq!(
            layer.pixels.channels,
            mode.layer_channels(),
            "layer {:?} is not the colour and alpha of a {mode:?} document",
            layer.name
        );
    }
    // The merged picture carries its alpha too, so a viewer that reads only it still sees what is
    // transparent.
    let merged_channels = mode.layer_channels();

    let mut w = Writer(Vec::new());
    w.bytes(b"8BPS");
    w.u16(1);
    w.bytes(&[0; 6]);
    w.u16(merged_channels as u16);
    w.u32(height as u32);
    w.u32(width as u32);
    w.u16(8);
    w.u16(mode.code());

    w.u32(0); // colour mode data
    w.u32(0); // image resources

    let layer_and_mask = w.reserve();
    let layer_info = w.reserve();
    // Negative: the merged picture's first extra channel is its transparency.
    w.i16(-(layers.len() as i16));

    // The channels of every layer, alpha first, compressed now so their lengths are known for the
    // records that come before them.
    let channel_data: Vec<Vec<(i16, Vec<u8>)>> = layers
        .iter()
        .map(|layer| {
            let p = &layer.pixels;
            let mut channels = vec![(-1, rle_channel(&plane(p, colour), p.width, p.height, false))];
            for c in 0..colour {
                channels.push((
                    c as i16,
                    rle_channel(&plane(p, c), p.width, p.height, false),
                ));
            }
            channels
        })
        .collect();

    for (layer, channels) in layers.iter().zip(&channel_data) {
        let p = &layer.pixels;
        w.i32(layer.top);
        w.i32(layer.left);
        w.i32(layer.top + p.height as i32);
        w.i32(layer.left + p.width as i32);
        w.u16(channels.len() as u16);
        for (id, data) in channels {
            w.i16(*id);
            w.u32(data.len() as u32);
        }
        w.bytes(b"8BIM");
        w.bytes(b"norm");
        w.u8(255); // opacity
        w.u8(0); // clipping: base
        w.u8(0); // flags: visible
        w.u8(0); // filler

        let extra = w.reserve();
        w.u32(0); // layer mask data
        w.u32(0); // blending ranges

        // The name, as a Pascal string padded to a multiple of four with its length byte.
        let name: Vec<u8> = layer.name.bytes().take(255).collect();
        w.u8(name.len() as u8);
        w.bytes(&name);
        for _ in 0..(4 - (name.len() + 1) % 4) % 4 {
            w.u8(0);
        }
        // The name again in UTF-16, which is what Photoshop shows for a name that is not ASCII.
        let units: Vec<u16> = layer.name.encode_utf16().collect();
        w.bytes(b"8BIMluni");
        w.u32((4 + units.len() * 2) as u32);
        w.u32(units.len() as u32);
        for unit in &units {
            w.u16(*unit);
        }
        w.fill(extra);
    }
    for channels in &channel_data {
        for (_, data) in channels {
            w.bytes(data);
        }
    }
    if (w.0.len() - layer_info) % 2 != 0 {
        w.u8(0);
    }
    w.fill(layer_info);
    w.u32(0); // global layer mask info
    w.fill(layer_and_mask);

    // The merged picture: one compression code, then every channel's row counts, then the rows.
    let planes = flatten(layers, width, height, mode);
    w.u16(1);
    let compressed: Vec<Vec<u8>> = planes
        .iter()
        .map(|p| rle_channel(p, width, height, true))
        .collect();
    for p in &compressed {
        w.bytes(&p[..height * 2]);
    }
    for p in &compressed {
        w.bytes(&p[height * 2..]);
    }
    w.0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unpack(mut data: &[u8], n: usize) -> Vec<u8> {
        let mut out = Vec::new();
        while out.len() < n {
            let head = data[0] as i8;
            if head >= 0 {
                let len = head as usize + 1;
                out.extend_from_slice(&data[1..1 + len]);
                data = &data[1 + len..];
            } else {
                out.extend(std::iter::repeat_n(data[1], (1 - head as isize) as usize));
                data = &data[2..];
            }
        }
        out
    }

    #[test]
    fn packbits_round_trips_runs_and_literals() {
        let row: Vec<u8> = [
            vec![7; 200],
            vec![1, 2, 3, 4],
            vec![9; 3],
            (0..=255).collect(),
        ]
        .concat();
        let mut packed = Vec::new();
        packbits(&row, &mut packed);
        assert_eq!(unpack(&packed, row.len()), row);
        assert!(packed.len() < row.len());
    }

    #[test]
    fn a_document_starts_with_its_header_and_names_its_layers() {
        let layer = Layer {
            name: "front hair".into(),
            left: 2,
            top: 3,
            pixels: Image::from_data(2, 2, 4, vec![255; 16]),
        };
        let psd = write(&[layer], 8, 8, Mode::Rgb);
        assert_eq!(&psd[..4], b"8BPS");
        assert_eq!(u16::from_be_bytes([psd[4], psd[5]]), 1);
        assert_eq!(u16::from_be_bytes([psd[12], psd[13]]), 4);
        assert_eq!(u32::from_be_bytes(psd[14..18].try_into().unwrap()), 8);
        assert!(psd.windows(10).any(|w| w == b"front hair"));
    }

    fn u32_at(psd: &[u8], at: usize) -> usize {
        u32::from_be_bytes(psd[at..at + 4].try_into().unwrap()) as usize
    }

    /// The layer count and the merged picture's planes, read back as a reader would.
    fn read_back(psd: &[u8]) -> (i16, Vec<Vec<u8>>) {
        let channels = u16::from_be_bytes([psd[12], psd[13]]) as usize;
        let height = u32_at(psd, 14);
        let width = u32_at(psd, 18);
        // Header, then the colour mode data and the image resources, both empty here.
        let mut at = 26;
        at += 4 + u32_at(psd, at);
        at += 4 + u32_at(psd, at);
        let count = i16::from_be_bytes([psd[at + 8], psd[at + 9]]);
        at += 4 + u32_at(psd, at);
        assert_eq!(u16::from_be_bytes([psd[at], psd[at + 1]]), 1, "PackBits");
        at += 2;
        let rows: Vec<usize> = (0..channels * height)
            .map(|i| u16::from_be_bytes([psd[at + 2 * i], psd[at + 2 * i + 1]]) as usize)
            .collect();
        at += 2 * channels * height;
        let mut planes = Vec::new();
        for c in 0..channels {
            let mut plane = Vec::new();
            for y in 0..height {
                let n = rows[c * height + y];
                plane.extend(unpack(&psd[at..at + n], width));
                at += n;
            }
            planes.push(plane);
        }
        (count, planes)
    }

    #[test]
    fn the_merged_picture_carries_its_transparency() {
        let layer = Layer {
            name: "face".into(),
            left: 1,
            top: 0,
            pixels: Image::from_data(2, 1, 4, vec![10, 20, 30, 255, 40, 50, 60, 255]),
        };
        let (count, planes) = read_back(&write(&[layer], 4, 1, Mode::Rgb));
        // Negative: the first extra channel is the merged picture's alpha, not a selection.
        assert_eq!(count, -1);
        assert_eq!(planes.len(), 4);
        assert_eq!(planes[3], vec![0, 255, 255, 0]);
        assert_eq!(planes[0], vec![0, 10, 40, 0]);
    }

    #[test]
    fn a_grayscale_layer_shows_what_is_under_its_transparent_pixels() {
        let under = Layer {
            name: "topwear".into(),
            left: 0,
            top: 0,
            pixels: Image::from_data(3, 1, 2, vec![200, 255, 200, 255, 200, 255]),
        };
        // The part above is transparent in its middle pixel, as a cut-out part's corners are.
        let over = Layer {
            name: "handwear".into(),
            left: 0,
            top: 0,
            pixels: Image::from_data(3, 1, 2, vec![10, 255, 10, 0, 10, 255]),
        };
        let (count, planes) = read_back(&write(&[under, over], 3, 1, Mode::Grayscale));
        assert_eq!(count, -2);
        assert_eq!(planes.len(), 2);
        assert_eq!(planes[0], vec![10, 200, 10]);
        assert_eq!(planes[1], vec![255, 255, 255]);
    }
}
