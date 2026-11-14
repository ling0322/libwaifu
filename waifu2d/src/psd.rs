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

//! A Photoshop document's layers, read: as much of the format as a drawing in layers needs. A
//! port of waifu-app/swiftui/Sources/Waifu/PSD.swift.
//!
//! Version 1 -- the .psd Photoshop saves; a .psb is version 2 -- eight bits a channel, RGB, or
//! grayscale read as RGB. Each layer is a rectangle of pixels with a name, an opacity and whether
//! it is shown; channel data raw or PackBits. Groups are read through: a layer in a hidden group
//! is hidden, and the group's own records are not layers. Masks, effects and blend modes are left
//! out. The layout is Adobe's "Photoshop File Formats Specification", big-endian throughout; the
//! Mac app's PSDWriter.swift writes the same format, which img2layers keeps.

use crate::{Error, RgbaImage};

/// The group a document keeps its layers' depth in, hidden, after them: img2layers' `depth`.
pub const DEPTH_GROUP: &str = "depth";

/// What a layer's depth is called in that group: `depth/topwear`. PSDWriter.swift's
/// `PSD.depthName`.
pub fn depth_name(name: &str) -> String {
    format!("{DEPTH_GROUP}/{name}")
}

/// One layer: where its top left corner is on the canvas, its pixels -- RGBA, not premultiplied,
/// as large as its rectangle -- and how it is shown.
#[derive(Clone, Debug, PartialEq)]
pub struct PsdLayer {
    pub name: String,
    pub left: i32,
    pub top: i32,
    pub image: RgbaImage,
    /// 0 to 1.
    pub opacity: f64,
    /// Shown: neither it nor a group it is in is hidden.
    pub visible: bool,
}

impl PsdLayer {
    pub fn width(&self) -> usize {
        self.image.width
    }

    pub fn height(&self) -> usize {
        self.image.height
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct PsdDocument {
    pub width: u32,
    pub height: u32,
    /// Bottom first, as they are drawn.
    pub layers: Vec<PsdLayer>,
}

impl PsdDocument {
    pub fn read(bytes: &[u8]) -> Result<PsdDocument, Error> {
        let mut reader = Reader {
            bytes,
            at: 0,
            gray: false,
        };
        reader.document()
    }

    /// Its layers, and apart from them the depth img2layers keeps in its hidden group: a hidden
    /// layer named `depth/<name>`, red its gray, 0 near to 255 far. Both in the document's order.
    pub fn split_depth(&self) -> (Vec<&PsdLayer>, Vec<&PsdLayer>) {
        self.layers.iter().partition(|layer| !is_depth(layer))
    }

    /// The depth of the layer named `name`, where the document keeps one.
    pub fn depth(&self, name: &str) -> Option<&PsdLayer> {
        let wanted = depth_name(name);
        self.layers
            .iter()
            .find(|layer| is_depth(layer) && layer.name == wanted)
    }
}

/// Whether a layer is one of the depth group's: hidden, and named for the layer it is the depth of.
fn is_depth(layer: &PsdLayer) -> bool {
    !layer.visible && layer.name.starts_with(&depth_name(""))
}

fn fail<T>(message: impl Into<String>) -> Result<T, Error> {
    Err(Error::Psd(message.into()))
}

/// Reads one document front to back.
struct Reader<'a> {
    bytes: &'a [u8],
    at: usize,
    /// Whether the document is grayscale rather than RGB.
    gray: bool,
}

/// A layer's record: everything but its pixels, which come after every record.
struct Record {
    name: String,
    top: i64,
    left: i64,
    bottom: i64,
    right: i64,
    /// Each channel's id and the length of its data.
    channels: Vec<(i64, usize)>,
    opacity: u8,
    hidden: bool,
    /// 1 or 2 opens a group -- going up the stack, its last record -- and 3 closes one.
    divider: u32,
}

impl Reader<'_> {
    fn need(&self, count: usize) -> Result<(), Error> {
        match self.at.checked_add(count) {
            Some(end) if end <= self.bytes.len() => Ok(()),
            _ => fail("the document ends where more of it was expected"),
        }
    }

    fn u8(&mut self) -> Result<u8, Error> {
        self.need(1)?;
        let v = self.bytes[self.at];
        self.at += 1;
        Ok(v)
    }

    fn u16(&mut self) -> Result<u16, Error> {
        self.need(2)?;
        let b = &self.bytes[self.at..self.at + 2];
        self.at += 2;
        Ok(u16::from_be_bytes([b[0], b[1]]))
    }

    fn i16(&mut self) -> Result<i16, Error> {
        Ok(self.u16()? as i16)
    }

    fn u32(&mut self) -> Result<u32, Error> {
        self.need(4)?;
        let b = &self.bytes[self.at..self.at + 4];
        self.at += 4;
        Ok(u32::from_be_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn i32(&mut self) -> Result<i32, Error> {
        Ok(self.u32()? as i32)
    }

    fn skip(&mut self, count: usize) -> Result<(), Error> {
        self.need(count)?;
        self.at += count;
        Ok(())
    }

    fn ascii(&mut self, count: usize) -> Result<String, Error> {
        self.need(count)?;
        let s = String::from_utf8_lossy(&self.bytes[self.at..self.at + count]).into_owned();
        self.at += count;
        Ok(s)
    }

    fn document(&mut self) -> Result<PsdDocument, Error> {
        if self.ascii(4)? != "8BPS" {
            return fail("this is not a Photoshop document");
        }
        match self.u16()? {
            1 => {}
            2 => return fail("this is a large document (.psb): save it as a .psd"),
            version => {
                return fail(format!(
                    "this is a Photoshop document of version {version}, which is not read here"
                ))
            }
        }
        self.skip(6)?;
        self.u16()?; // channels of the merged picture
        let height = self.u32()?;
        let width = self.u32()?;
        let depth = self.u16()?;
        let mode = self.u16()?;
        if depth != 8 {
            return fail(format!(
                "this document has {depth} bits a channel: save it with 8"
            ));
        }
        // Grayscale is img2layers' depth document's: its gray is read into red, green and blue.
        if mode != 3 && mode != 1 {
            return fail("this document is not RGB: convert it to RGB colour");
        }
        self.gray = mode == 1;

        let length = self.u32()? as usize;
        self.skip(length)?; // colour mode data
        let length = self.u32()? as usize;
        self.skip(length)?; // image resources
        let section = self.u32()? as usize;
        let end = self.at + section;
        if section == 0 {
            return Ok(PsdDocument {
                width,
                height,
                layers: vec![],
            });
        }
        let layers = self.layer_info()?;
        self.at = end;
        Ok(PsdDocument {
            width,
            height,
            layers,
        })
    }

    fn layer_info(&mut self) -> Result<Vec<PsdLayer>, Error> {
        let length = self.u32()?;
        if length == 0 {
            return Ok(vec![]);
        }
        let count = (self.i16()? as i32).abs();
        let mut records = Vec::new();
        for _ in 0..count {
            records.push(self.record()?);
        }

        let mut pixels = Vec::new();
        for record in &records {
            pixels.push(self.channels(record)?);
        }

        // Top first, so that a group's record comes before what is in it: a stack of whether each
        // open group is hidden.
        let mut layers = Vec::new();
        let mut groups: Vec<bool> = Vec::new();
        for (record, image) in records.into_iter().zip(pixels).rev() {
            match record.divider {
                1 | 2 => {
                    groups.push(record.hidden || groups.last() == Some(&true));
                    continue;
                }
                3 => {
                    groups.pop();
                    continue;
                }
                _ => {}
            }
            let Some(image) = image else { continue };
            layers.push(PsdLayer {
                name: record.name,
                left: record.left as i32,
                top: record.top as i32,
                image,
                opacity: f64::from(record.opacity) / 255.0,
                visible: !record.hidden && groups.last() != Some(&true),
            });
        }
        layers.reverse();
        Ok(layers)
    }

    fn record(&mut self) -> Result<Record, Error> {
        let top = i64::from(self.i32()?);
        let left = i64::from(self.i32()?);
        let bottom = i64::from(self.i32()?);
        let right = i64::from(self.i32()?);
        let mut channels = Vec::new();
        for _ in 0..self.u16()? {
            let id = i64::from(self.i16()?);
            let length = self.u32()? as usize;
            channels.push((id, length));
        }
        self.skip(8)?; // "8BIM" and the blend mode
        let opacity = self.u8()?;
        self.u8()?; // clipping
        let flags = self.u8()?;
        self.skip(1)?;
        let extra = self.u32()? as usize;
        let end = self.at + extra;
        let length = self.u32()? as usize;
        self.skip(length)?; // mask
        let length = self.u32()? as usize;
        self.skip(length)?; // blending ranges
        let short = self.u8()? as usize;
        let mut name = self.ascii(short)?;
        self.skip((4 - (short + 1) % 4) % 4)?;
        let mut divider = 0;
        // What else the layer says, by key: its name in full, and whether it opens or closes a
        // group.
        while self.at + 12 <= end {
            let signature = self.ascii(4)?;
            if signature != "8BIM" && signature != "8B64" {
                break;
            }
            let key = self.ascii(4)?;
            let size = self.u32()? as usize;
            let start = self.at;
            match key.as_str() {
                "luni" => {
                    let characters = self.u32()? as usize;
                    self.need(characters * 2)?;
                    let units: Vec<u16> = (0..characters)
                        .map(|i| {
                            let at = self.at + i * 2;
                            u16::from(self.bytes[at]) << 8 | u16::from(self.bytes[at + 1])
                        })
                        .collect();
                    name = String::from_utf16_lossy(&units);
                }
                "lsct" | "lsdk" => divider = self.u32()?,
                _ => {}
            }
            self.at = start + size;
        }
        self.at = end;
        Ok(Record {
            name,
            top,
            left,
            bottom,
            right,
            channels,
            opacity,
            hidden: flags & 2 != 0,
            divider,
        })
    }

    /// A layer's pixels, out of its channels: `None` for one with no area, a group's record among
    /// them.
    fn channels(&mut self, record: &Record) -> Result<Option<RgbaImage>, Error> {
        let width = record.right - record.left;
        let height = record.bottom - record.top;
        let area = (width.max(0) * height.max(0)) as usize;
        let mut rgba = if area > 0 {
            vec![255u8; area * 4]
        } else {
            vec![]
        };
        let last = if self.gray { 0 } else { 2 };
        for &(id, length) in &record.channels {
            let end = self.at + length;
            // Its pixels are its rectangle's: a mask's are not, and are not read.
            if area > 0 && (-1..=last).contains(&id) {
                let plane = self.plane(width as usize, height as usize)?;
                let single = [if id == -1 { 3 } else { id as usize }];
                let offsets: &[usize] = if id != -1 && self.gray {
                    &[0, 1, 2]
                } else {
                    &single
                };
                for i in 0..area {
                    for &offset in offsets {
                        rgba[i * 4 + offset] = plane[i];
                    }
                }
            }
            self.at = end;
        }
        Ok((area > 0).then(|| RgbaImage::new(width as usize, height as usize, rgba)))
    }

    /// One channel's `width` by `height` bytes: raw, or PackBits row by row.
    fn plane(&mut self, width: usize, height: usize) -> Result<Vec<u8>, Error> {
        match self.u16()? {
            0 => {
                self.need(width * height)?;
                let plane = self.bytes[self.at..self.at + width * height].to_vec();
                self.at += width * height;
                Ok(plane)
            }
            1 => {
                let mut counts = Vec::with_capacity(height);
                for _ in 0..height {
                    counts.push(self.u16()? as usize);
                }
                let mut plane = vec![0u8; width * height];
                for (row, &count) in counts.iter().enumerate() {
                    let end = self.at + count;
                    self.need(count)?;
                    let mut x = 0;
                    while self.at < end && x < width {
                        let n = i64::from(self.bytes[self.at] as i8);
                        self.at += 1;
                        if n >= 0 {
                            let run = ((n + 1) as usize).min(width - x).min(end - self.at);
                            plane[row * width + x..row * width + x + run]
                                .copy_from_slice(&self.bytes[self.at..self.at + run]);
                            self.at += (n + 1) as usize;
                            x += run;
                        } else if n != -128 {
                            if self.at >= end {
                                break;
                            }
                            let run = ((1 - n) as usize).min(width - x);
                            plane[row * width + x..row * width + x + run].fill(self.bytes[self.at]);
                            self.at += 1;
                            x += run;
                        }
                    }
                    self.at = end;
                }
                Ok(plane)
            }
            _ => fail("a layer is compressed with ZIP, which is not read here: save it without"),
        }
    }
}
