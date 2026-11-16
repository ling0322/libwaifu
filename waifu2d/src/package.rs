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

//! A package: the drawing in layers, as waifu2d keeps it -- a `.waifu2d` file.
//!
//! A zip of PNGs and an index, `index.yaml`: every layer's picture, where it is, how it is shown
//! and how far away it is, and the depth of each pixel of it as a picture of its own. It carries
//! what a Photoshop document from img2layers has and what one cannot say -- a layer's median
//! depth, which layers are the near and far halves of one part, where the picture is on the
//! canvas, what made it -- and any zip tool opens it. docs/waifu2d-package.md is the format;
//! "package" here is this file, not one of libwaifu's model packages.
//!
//! ```text
//! girl.waifu2d
//! ├── mimetype         application/vnd.libwaifu.waifu2d+zip: first, stored, no extra field
//! ├── index.yaml
//! ├── preview.png      the layers laid over each other, for a thumbnail
//! ├── source.png       the picture they were taken from, where there is one
//! ├── layers/000.png   RGBA, eight bits, not premultiplied, cut to what is drawn
//! └── depth/000.png    gray, its layer's size and place: 0 nearest, 255 farthest
//! ```

use std::collections::BTreeMap;
use std::io::{Cursor, Read, Write};

use crate::psd::PsdDocument;
use crate::yaml::{self, Node};
use crate::{Error, RgbaImage};

/// What a package's file name ends in, without the dot.
pub const EXTENSION: &str = "waifu2d";

/// The contents of `mimetype`, the package's first entry.
pub const MIMETYPE: &str = "application/vnd.libwaifu.waifu2d+zip";

/// `format:` in the index.
pub const FORMAT: &str = "waifu2d";

/// The version this writes and the newest it reads. It goes up only for a change a reader of the
/// last one would get wrong; a key it does not know, a reader leaves alone.
pub const VERSION: u32 = 1;

/// The widest and tallest canvas, and layer, a package may have.
pub const MAX_SIDE: u32 = 16384;

const INDEX: &str = "index.yaml";
const PREVIEW: &str = "preview.png";
const SOURCE: &str = "source.png";

/// The most one entry may unpack to, whatever the archive claims: a 16384 square of RGBA.
const MAX_ENTRY: u64 = 4 * (MAX_SIDE as u64) * (MAX_SIDE as u64);

/// What a layer is: a part as the model drew it, or one of the two halves a part was cut into by
/// its depth (docs/see_through.md, "Stratification") -- the near one in front, the far one under it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Kind {
    #[default]
    Part,
    Near,
    Far,
}

impl Kind {
    fn name(self) -> &'static str {
        match self {
            Kind::Part => "part",
            Kind::Near => "near",
            Kind::Far => "far",
        }
    }
}

/// Each pixel's depth, gray: 0 nearest, 255 farthest.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DepthMap {
    pub width: usize,
    pub height: usize,
    /// `width * height`, a row at a time.
    pub gray: Vec<u8>,
}

/// One layer: where its top left corner is on the canvas, its pixels -- RGBA, not premultiplied,
/// as large as its rectangle -- how it is shown, and how far away it is.
#[derive(Clone, Debug, PartialEq)]
pub struct Layer {
    pub name: String,
    pub left: i32,
    pub top: i32,
    pub image: RgbaImage,
    /// 0 to 1.
    pub opacity: f64,
    pub visible: bool,
    pub kind: Kind,
    /// A half's part: `topwear` for `topwear-0`.
    pub of: Option<String>,
    /// The median depth of what is drawn, 0 nearest to 1 farthest.
    pub depth: Option<f32>,
    /// The depth of each pixel, the layer's size; what is drawn is the layer's alpha.
    pub depth_map: Option<DepthMap>,
}

impl Layer {
    /// A part, shown, at full opacity, of no known depth.
    pub fn new(name: impl Into<String>, left: i32, top: i32, image: RgbaImage) -> Layer {
        Layer {
            name: name.into(),
            left,
            top,
            image,
            opacity: 1.0,
            visible: true,
            kind: Kind::Part,
            of: None,
            depth: None,
            depth_map: None,
        }
    }
}

/// Where the picture the layers were taken from is on the canvas, in canvas pixels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Area {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Package {
    pub width: u32,
    pub height: u32,
    /// Bottom first, as they are drawn.
    pub layers: Vec<Layer>,
    pub picture: Option<Area>,
    /// What made it, as its maker says: `name: img2layers`, `seed: 7`.
    pub generator: BTreeMap<String, String>,
    /// The picture the layers were taken from, as it was handed over.
    pub source: Option<RgbaImage>,
}

fn fail<T>(message: impl Into<String>) -> Result<T, Error> {
    Err(Error::Package(message.into()))
}

impl Package {
    /// An empty canvas.
    pub fn new(width: u32, height: u32) -> Package {
        Package {
            width,
            height,
            layers: Vec::new(),
            picture: None,
            generator: BTreeMap::new(),
            source: None,
        }
    }

    /// Whether `bytes` start as a package does: a zip whose first entry is a stored `mimetype`
    /// holding [`MIMETYPE`], with no extra field, so that its type is at offset 38.
    pub fn is_package(bytes: &[u8]) -> bool {
        let le16 = |at: usize| u16::from_le_bytes([bytes[at], bytes[at + 1]]);
        bytes.len() >= 38 + MIMETYPE.len()
            && bytes[..4] == *b"PK\x03\x04"
            && le16(8) == 0
            && usize::from(le16(26)) == "mimetype".len()
            && le16(28) == 0
            && &bytes[30..38] == b"mimetype"
            && bytes[38..].starts_with(MIMETYPE.as_bytes())
    }

    pub fn read(bytes: &[u8]) -> Result<Package, Error> {
        if !Package::is_package(bytes) {
            return fail(if bytes.starts_with(b"PK\x03\x04") {
                "a zip archive, but not a waifu2d package: its first entry is not its mimetype"
            } else {
                "not a waifu2d package"
            });
        }
        let mut archive = Archive::new(bytes)?;
        let index = archive.need(INDEX)?;
        let Ok(index) = String::from_utf8(index) else {
            return fail("index.yaml is not UTF-8");
        };
        let mut index = Index(yaml::parse(&index)?.into_map("index.yaml")?);

        let format = index.text("format")?;
        if format != FORMAT {
            return fail(format!(
                "index.yaml is of format {format:?}, not {FORMAT:?}"
            ));
        }
        let version: u32 = index.number("version")?;
        if version > VERSION {
            return fail(format!(
                "this package is version {version}, newer than this reads ({VERSION}): \
                update waifu2d"
            ));
        }
        let [width, height] = index.numbers::<u32, 2>("canvas")?;
        if !(1..=MAX_SIDE).contains(&width) || !(1..=MAX_SIDE).contains(&height) {
            return fail(format!(
                "a {width} by {height} canvas: each side is 1 to {MAX_SIDE}"
            ));
        }

        let picture = match index.has("picture") {
            true => {
                let [x, y, width, height] = index.numbers::<u32, 4>("picture")?;
                Some(Area {
                    x,
                    y,
                    width,
                    height,
                })
            }
            false => None,
        };
        // Only what is one value: a maker's nested notes are its own to read.
        let generator = match index.0.remove("generator") {
            Some(node) => node
                .into_map("generator")?
                .into_iter()
                .filter_map(|(key, value)| Some((key, value.as_str()?.to_string())))
                .collect(),
            None => BTreeMap::new(),
        };
        let source = match index.0.remove("source") {
            Some(name) => {
                let name = name.into_scalar("source")?;
                Some(rgba(&archive.need(&name)?, &name)?)
            }
            None => None,
        };

        let mut layers = Vec::new();
        for (i, node) in index
            .need("layers")?
            .into_seq("layers")?
            .into_iter()
            .enumerate()
        {
            let what = format!("layer {i}");
            let mut fields = Index(node.into_map(&what)?);
            let name = fields.text("name").map_err(|e| within(&what, e))?;
            let what = format!("layer {i} ({name})");
            layers.push(layer(name, &mut fields, &mut archive).map_err(|e| within(&what, e))?);
        }
        Ok(Package {
            width,
            height,
            layers,
            picture,
            generator,
            source,
        })
    }

    pub fn write(&self) -> Result<Vec<u8>, Error> {
        if !(1..=MAX_SIDE).contains(&self.width) || !(1..=MAX_SIDE).contains(&self.height) {
            return fail(format!(
                "a {} by {} canvas: each side is 1 to {MAX_SIDE}",
                self.width, self.height
            ));
        }
        let mut index = String::new();
        let mut files: Vec<(String, Vec<u8>)> = Vec::new();
        index.push_str(&format!("format: {}\n", quote(FORMAT)?));
        index.push_str(&format!("version: {VERSION}\n"));
        index.push_str(&format!("canvas: [{}, {}]\n", self.width, self.height));
        if let Some(a) = self.picture {
            index.push_str(&format!(
                "picture: [{}, {}, {}, {}]\n",
                a.x, a.y, a.width, a.height
            ));
        }
        index.push_str(&format!("preview: {}\n", quote(PREVIEW)?));
        files.push((PREVIEW.into(), encode_rgba(&self.preview())?));
        if let Some(source) = &self.source {
            index.push_str(&format!("source: {}\n", quote(SOURCE)?));
            files.push((SOURCE.into(), encode_rgba(source)?));
        }
        if !self.generator.is_empty() {
            index.push_str("generator:\n");
            for (key, value) in &self.generator {
                index.push_str(&format!("  {}: {}\n", quote(key)?, quote(value)?));
            }
        }
        index.push_str(if self.layers.is_empty() {
            "layers: []\n"
        } else {
            "layers:\n"
        });
        for (i, layer) in self.layers.iter().enumerate() {
            let what = format!("layer {i} ({})", layer.name);
            let image = &layer.image;
            if image.rgba.len() != image.width * image.height * 4 {
                return fail(format!(
                    "{what}: its pixels are not {} by {}",
                    image.width, image.height
                ));
            }
            if !(0.0..=1.0).contains(&layer.opacity) {
                return fail(format!("{what}: opacity {} is not 0 to 1", layer.opacity));
            }
            let file = format!("layers/{i:03}.png");
            index.push_str(&format!("  - name: {}\n", quote(&layer.name)?));
            index.push_str(&format!("    image: {}\n", quote(&file)?));
            index.push_str(&format!("    at: [{}, {}]\n", layer.left, layer.top));
            files.push((file, encode_rgba(image).map_err(|e| within(&what, e))?));
            if layer.opacity != 1.0 {
                index.push_str(&format!("    opacity: {}\n", layer.opacity));
            }
            if !layer.visible {
                index.push_str("    visible: false\n");
            }
            if layer.kind != Kind::Part {
                index.push_str(&format!("    kind: {}\n", layer.kind.name()));
            }
            if let Some(of) = &layer.of {
                index.push_str(&format!("    of: {}\n", quote(of)?));
            }
            if let Some(depth) = layer.depth {
                if !(0.0..=1.0).contains(&depth) {
                    return fail(format!("{what}: depth {depth} is not 0 to 1"));
                }
                index.push_str(&format!("    depth: {depth}\n"));
            }
            if let Some(map) = &layer.depth_map {
                if (map.width, map.height) != (image.width, image.height)
                    || map.gray.len() != map.width * map.height
                {
                    return fail(format!("{what}: its depth is not the layer's size"));
                }
                let file = format!("depth/{i:03}.png");
                index.push_str(&format!("    depth_map: {}\n", quote(&file)?));
                files.push((file, encode_gray(map)?));
            }
        }

        let package = |e: zip::result::ZipError| Error::Package(e.to_string());
        let mut zip = zip::ZipWriter::new(Cursor::new(Vec::new()));
        // Stored: what is in a package is PNG, deflated already. Dated 1980-01-01 so that the
        // same package is the same bytes.
        let options = zip::write::SimpleFileOptions::default()
            .compression_method(zip::CompressionMethod::Stored)
            .last_modified_time(zip::DateTime::DEFAULT);
        zip.start_file("mimetype", options).map_err(package)?;
        zip.write_all(MIMETYPE.as_bytes())?;
        zip.start_file(INDEX, options).map_err(package)?;
        zip.write_all(index.as_bytes())?;
        for (name, data) in files {
            zip.start_file(name, options).map_err(package)?;
            zip.write_all(&data)?;
        }
        Ok(zip.finish().map_err(package)?.into_inner())
    }

    /// The layers that are shown laid over each other on the canvas, each at its opacity: RGBA,
    /// not premultiplied, what `preview.png` holds.
    pub fn preview(&self) -> RgbaImage {
        let (w, h) = (self.width as usize, self.height as usize);
        let mut out = vec![0f32; w * h * 4];
        for layer in self.layers.iter().filter(|l| l.visible) {
            let image = &layer.image;
            for y in 0..image.height {
                let cy = i64::from(layer.top) + y as i64;
                if !(0..h as i64).contains(&cy) {
                    continue;
                }
                for x in 0..image.width {
                    let cx = i64::from(layer.left) + x as i64;
                    if !(0..w as i64).contains(&cx) {
                        continue;
                    }
                    let s = &image.rgba[(y * image.width + x) * 4..][..4];
                    let sa = f32::from(s[3]) / 255.0 * layer.opacity as f32;
                    if sa <= 0.0 {
                        continue;
                    }
                    let d = &mut out[(cy as usize * w + cx as usize) * 4..][..4];
                    let da = d[3] * (1.0 - sa);
                    let a = sa + da;
                    for c in 0..3 {
                        d[c] = (f32::from(s[c]) * sa + d[c] * da) / a;
                    }
                    d[3] = a;
                }
            }
        }
        let rgba = out
            .chunks(4)
            .flat_map(|p| [p[0], p[1], p[2], p[3] * 255.0])
            .map(|v| v.round().clamp(0.0, 255.0) as u8)
            .collect();
        RgbaImage::new(w, h, rgba)
    }

    /// Marks the halves a part was cut into, as img2layers and see_through_stratify name them:
    /// `<part>-0` near and `<part>-1` far, where the package has both and neither is marked yet.
    pub fn mark_halves(&mut self) {
        let names: Vec<String> = self.layers.iter().map(|l| l.name.clone()).collect();
        let has = |name: String| names.contains(&name);
        for layer in &mut self.layers {
            let Some((part, n)) = layer.name.rsplit_once('-') else {
                continue;
            };
            let kind = match n {
                "0" => Kind::Near,
                "1" => Kind::Far,
                _ => continue,
            };
            if layer.kind == Kind::Part && has(format!("{part}-0")) && has(format!("{part}-1")) {
                layer.of = Some(part.to_string());
                layer.kind = kind;
            }
        }
    }

    /// A Photoshop document img2layers wrote, as a package: its layers, each with its depth from
    /// the hidden group `depth` where the document keeps one and its median depth from that, and
    /// its halves marked ([`Package::mark_halves`]).
    pub fn from_psd(document: &PsdDocument) -> Package {
        let (shown, _) = document.split_depth();
        let layers = shown
            .iter()
            .map(|psd| {
                let mut layer = Layer::new(psd.name.clone(), psd.left, psd.top, psd.image.clone());
                layer.opacity = psd.opacity;
                layer.visible = psd.visible;
                let depth = document.depth(&psd.name).filter(|d| {
                    (d.left, d.top, d.width(), d.height())
                        == (psd.left, psd.top, psd.width(), psd.height())
                });
                if let Some(depth) = depth {
                    let map = DepthMap {
                        width: depth.width(),
                        height: depth.height(),
                        gray: depth.image.rgba.chunks(4).map(|p| p[0]).collect(),
                    };
                    layer.depth = map.median(&layer.image);
                    layer.depth_map = Some(map);
                }
                layer
            })
            .collect();
        let mut package = Package {
            layers,
            ..Package::new(document.width, document.height)
        };
        package.mark_halves();
        package
    }
}

impl DepthMap {
    /// The median depth of what `image` -- its layer's picture, the same size -- draws, 0 to 1;
    /// `None` where it draws nothing.
    pub fn median(&self, image: &RgbaImage) -> Option<f32> {
        let mut drawn: Vec<u8> = (self.gray.iter().zip(image.rgba.chunks(4)))
            .filter(|(_, p)| p[3] > 0)
            .map(|(g, _)| *g)
            .collect();
        if drawn.is_empty() {
            return None;
        }
        let mid = drawn.len() / 2;
        let (_, m, _) = drawn.select_nth_unstable(mid);
        Some(f32::from(*m) / 255.0)
    }
}

fn within(what: &str, error: Error) -> Error {
    Error::Package(format!("{what}: {error}"))
}

/// One layer's keys, once its name is out of them.
fn layer(name: String, fields: &mut Index, archive: &mut Archive) -> Result<Layer, Error> {
    let file = fields.text("image")?;
    let image = rgba(&archive.need(&file)?, &file)?;
    let [left, top] = fields.numbers::<i32, 2>("at")?;
    let mut layer = Layer::new(name, left, top, image);
    if fields.has("opacity") {
        layer.opacity = fields.number("opacity")?;
        if !(0.0..=1.0).contains(&layer.opacity) {
            return fail(format!("opacity {} is not 0 to 1", layer.opacity));
        }
    }
    if fields.has("visible") {
        layer.visible = match fields.text("visible")?.as_str() {
            "true" => true,
            "false" => false,
            other => return fail(format!("visible is {other:?}: true or false")),
        };
    }
    if fields.has("kind") {
        layer.kind = match fields.text("kind")?.as_str() {
            "part" => Kind::Part,
            "near" => Kind::Near,
            "far" => Kind::Far,
            other => return fail(format!("kind is {other:?}: part, near or far")),
        };
    }
    if fields.has("of") {
        layer.of = Some(fields.text("of")?);
    }
    if fields.has("depth") {
        let depth: f32 = fields.number("depth")?;
        if !(0.0..=1.0).contains(&depth) {
            return fail(format!("depth {depth} is not 0 to 1"));
        }
        layer.depth = Some(depth);
    }
    if fields.has("depth_map") {
        let file = fields.text("depth_map")?;
        let map = gray(&archive.need(&file)?, &file)?;
        if (map.width, map.height) != (layer.image.width, layer.image.height) {
            return fail(format!(
                "{file} is {} by {}, and its layer {} by {}",
                map.width, map.height, layer.image.width, layer.image.height
            ));
        }
        layer.depth_map = Some(map);
    }
    Ok(layer)
}

/// A mapping of the index, its keys taken out as they are read.
struct Index(BTreeMap<String, Node>);

impl Index {
    fn has(&self, key: &str) -> bool {
        self.0.contains_key(key)
    }

    fn need(&mut self, key: &str) -> Result<Node, Error> {
        self.0
            .remove(key)
            .ok_or_else(|| Error::Package(format!("{key} is missing")))
    }

    fn text(&mut self, key: &str) -> Result<String, Error> {
        self.need(key)?.into_scalar(key)
    }

    fn number<T: std::str::FromStr>(&mut self, key: &str) -> Result<T, Error> {
        let text = self.text(key)?;
        text.parse()
            .map_err(|_| Error::Package(format!("{key} is {text:?}, not a number")))
    }

    fn numbers<T: std::str::FromStr + Copy + Default, const N: usize>(
        &mut self,
        key: &str,
    ) -> Result<[T; N], Error> {
        let items = self.need(key)?.into_seq(key)?;
        if items.len() != N {
            return fail(format!("{key} is {} numbers, not {N}", items.len()));
        }
        let mut out = [T::default(); N];
        for (slot, item) in out.iter_mut().zip(items) {
            let text = item.into_scalar(key)?;
            *slot = text
                .parse()
                .map_err(|_| Error::Package(format!("{key} holds {text:?}, not a number")))?;
        }
        Ok(out)
    }
}

/// The zip, read entry by entry.
struct Archive<'a>(zip::ZipArchive<Cursor<&'a [u8]>>);

impl<'a> Archive<'a> {
    fn new(bytes: &'a [u8]) -> Result<Archive<'a>, Error> {
        zip::ZipArchive::new(Cursor::new(bytes))
            .map(Archive)
            .map_err(|e| Error::Package(format!("not a zip archive one can read: {e}")))
    }

    /// The bytes of the entry `name` names, which the index says is there.
    fn need(&mut self, name: &str) -> Result<Vec<u8>, Error> {
        let mut file = match self.0.by_name(name) {
            Ok(file) => file,
            Err(zip::result::ZipError::FileNotFound) => return fail(format!("{name} is missing")),
            Err(e) => return fail(format!("{name}: {e}")),
        };
        let size = file.size();
        if size > MAX_ENTRY {
            return fail(format!(
                "{name} unpacks to {size} bytes, more than a package holds"
            ));
        }
        // No more than it says it is, whatever its data turns out to be; the zip checks the CRC.
        let mut data = Vec::with_capacity(size as usize);
        (&mut file)
            .take(size + 1)
            .read_to_end(&mut data)
            .map_err(|e| Error::Package(format!("{name}: {e}")))?;
        if data.len() as u64 != size {
            return fail(format!(
                "{name} is damaged: it is not the size its archive says"
            ));
        }
        Ok(data)
    }
}

/// `text` as a double quoted scalar, which every reader reads the same whatever is in it.
fn quote(text: &str) -> Result<String, Error> {
    let mut out = String::with_capacity(text.len() + 2);
    out.push('"');
    for c in text.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\t' => out.push_str("\\t"),
            '\r' => out.push_str("\\r"),
            c if c.is_control() => {
                return fail(format!("{text:?} holds a control character a name cannot"))
            }
            c => out.push(c),
        }
    }
    out.push('"');
    Ok(out)
}

/// A PNG, decoded to eight bits a channel: what it is, how big, and its samples.
fn decode(bytes: &[u8], name: &str) -> Result<(png::ColorType, usize, usize, Vec<u8>), Error> {
    let bad = |e: png::DecodingError| Error::Package(format!("{name}: {e}"));
    let mut decoder = png::Decoder::new_with_limits(
        Cursor::new(bytes),
        png::Limits {
            bytes: MAX_ENTRY as usize,
        },
    );
    // Palettes and fewer than eight bits expanded, sixteen bits cut to their high byte.
    decoder.set_transformations(png::Transformations::normalize_to_color8());
    let mut reader = decoder.read_info().map_err(bad)?;
    let (width, height) = (reader.info().width, reader.info().height);
    if width > MAX_SIDE || height > MAX_SIDE {
        return fail(format!(
            "{name} is {width} by {height}: each side is at most {MAX_SIDE}"
        ));
    }
    let Some(size) = reader.output_buffer_size() else {
        return fail(format!("{name} is too large"));
    };
    let mut data = vec![0; size];
    let info = reader.next_frame(&mut data).map_err(bad)?;
    data.truncate(info.buffer_size());
    Ok((info.color_type, width as usize, height as usize, data))
}

fn rgba(bytes: &[u8], name: &str) -> Result<RgbaImage, Error> {
    let (color, width, height, data) = decode(bytes, name)?;
    let rgba = match color {
        png::ColorType::Rgba => data,
        png::ColorType::Rgb => data
            .chunks(3)
            .flat_map(|p| [p[0], p[1], p[2], 255])
            .collect(),
        png::ColorType::GrayscaleAlpha => data
            .chunks(2)
            .flat_map(|p| [p[0], p[0], p[0], p[1]])
            .collect(),
        png::ColorType::Grayscale => data.iter().flat_map(|&g| [g, g, g, 255]).collect(),
        png::ColorType::Indexed => return fail(format!("{name}: a palette not expanded")),
    };
    Ok(RgbaImage::new(width, height, rgba))
}

/// A depth map: its gray, or its red where it was saved in colour.
fn gray(bytes: &[u8], name: &str) -> Result<DepthMap, Error> {
    let (color, width, height, data) = decode(bytes, name)?;
    let step = match color {
        png::ColorType::Grayscale => 1,
        png::ColorType::GrayscaleAlpha => 2,
        png::ColorType::Rgb => 3,
        png::ColorType::Rgba => 4,
        png::ColorType::Indexed => return fail(format!("{name}: a palette not expanded")),
    };
    let gray = data.chunks(step).map(|p| p[0]).collect();
    Ok(DepthMap {
        width,
        height,
        gray,
    })
}

fn encode(
    width: usize,
    height: usize,
    color: png::ColorType,
    data: &[u8],
) -> Result<Vec<u8>, Error> {
    let bad = |e: png::EncodingError| Error::Package(e.to_string());
    let mut out = Vec::new();
    let mut encoder = png::Encoder::new(&mut out, width as u32, height as u32);
    encoder.set_color(color);
    encoder.set_depth(png::BitDepth::Eight);
    let mut writer = encoder.write_header().map_err(bad)?;
    writer.write_image_data(data).map_err(bad)?;
    writer.finish().map_err(bad)?;
    Ok(out)
}

fn encode_rgba(image: &RgbaImage) -> Result<Vec<u8>, Error> {
    if image.width == 0 || image.height == 0 {
        return fail("a picture with no pixels: PNG cannot hold one");
    }
    encode(image.width, image.height, png::ColorType::Rgba, &image.rgba)
}

fn encode_gray(map: &DepthMap) -> Result<Vec<u8>, Error> {
    encode(map.width, map.height, png::ColorType::Grayscale, &map.gray)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn image(width: usize, height: usize, seed: u8) -> RgbaImage {
        let rgba = (0..width * height * 4)
            .map(|i| (i as u8).wrapping_mul(31).wrapping_add(seed))
            .collect();
        RgbaImage::new(width, height, rgba)
    }

    fn package() -> Package {
        let mut package = Package::new(8, 6);
        package.picture = Some(Area {
            x: 0,
            y: 1,
            width: 8,
            height: 4,
        });
        package.generator.insert("name".into(), "img2layers".into());
        package.generator.insert("seed".into(), "7".into());
        package.source = Some(image(8, 6, 1));

        let mut hair = Layer::new("back hair", 1, -1, image(3, 2, 2));
        hair.depth = Some(0.83);
        hair.depth_map = Some(DepthMap {
            width: 3,
            height: 2,
            gray: vec![0, 50, 100, 150, 200, 255],
        });
        let mut near = Layer::new("topwear-0", 0, 2, image(4, 3, 3));
        near.kind = Kind::Near;
        near.of = Some("topwear".into());
        near.depth = Some(0.5);
        let mut fringe = Layer::new("前髪 \"2\": \\ #", 2, 0, image(2, 2, 4));
        fringe.opacity = 0.25;
        fringe.visible = false;
        package.layers = vec![hair, near, fringe];
        package
    }

    #[test]
    fn a_package_reads_back_as_it_was_written() {
        let package = package();
        let bytes = package.write().unwrap();
        assert!(Package::is_package(&bytes));
        assert_eq!(Package::read(&bytes).unwrap(), package);
        // Again the same bytes: the archive is dated, the PNGs deterministic.
        assert_eq!(package.write().unwrap(), bytes);
    }

    #[test]
    fn its_type_is_at_offset_38_and_its_files_are_numbered() {
        let bytes = package().write().unwrap();
        assert_eq!(&bytes[30..38], b"mimetype");
        assert_eq!(&bytes[38..38 + MIMETYPE.len()], MIMETYPE.as_bytes());
        let archive = zip::ZipArchive::new(Cursor::new(&bytes[..])).unwrap();
        let names: Vec<String> = archive
            .file_names()
            .map(|name| name.unwrap().into_owned())
            .collect();
        assert_eq!(
            names,
            [
                "mimetype",
                "index.yaml",
                "preview.png",
                "source.png",
                "layers/000.png",
                "depth/000.png",
                "layers/001.png",
                "layers/002.png",
            ]
        );
    }

    #[test]
    fn the_index_is_plain_yaml() {
        let bytes = package().write().unwrap();
        let mut archive = Archive::new(&bytes).unwrap();
        let index = String::from_utf8(archive.need(INDEX).unwrap()).unwrap();
        assert!(
            index.starts_with("format: \"waifu2d\"\nversion: 1\ncanvas: [8, 6]\n"),
            "{index}"
        );
        assert!(
            index.contains(
                "  - name: \"back hair\"\n    image: \"layers/000.png\"\n    at: [1, -1]\n"
            ),
            "{index}"
        );
        assert!(
            index.contains("    kind: near\n    of: \"topwear\"\n"),
            "{index}"
        );
    }

    /// An archive as another tool might write it: `mimetype` first and stored, then each entry,
    /// deflated where it says so.
    fn archive(entries: &[(&str, &[u8], bool)]) -> Vec<u8> {
        let stored = zip::write::SimpleFileOptions::default()
            .compression_method(zip::CompressionMethod::Stored);
        let deflated = stored.compression_method(zip::CompressionMethod::Deflated);
        let mut zip = zip::ZipWriter::new(Cursor::new(Vec::new()));
        zip.start_file("mimetype", stored).unwrap();
        zip.write_all(MIMETYPE.as_bytes()).unwrap();
        for (name, data, deflate) in entries {
            zip.start_file(*name, if *deflate { deflated } else { stored })
                .unwrap();
            zip.write_all(data).unwrap();
        }
        zip.finish().unwrap().into_inner()
    }

    fn error(bytes: &[u8]) -> String {
        Package::read(bytes).unwrap_err().to_string()
    }

    #[test]
    fn a_reader_takes_what_another_tool_wrote() {
        // Keys it does not know, deflated entries, a layer with only what it must have.
        let index = "# by hand\nformat: waifu2d\nversion: 1\ncanvas: [4, 4]\nfuture: [1, 2]\n\
            layers:\n  - name: face\n    image: a.png\n    at: [1, 1]\n    sparkle: yes\n";
        let png = encode_rgba(&image(2, 2, 9)).unwrap();
        let bytes = archive(&[
            ("index.yaml", index.as_bytes(), true),
            ("a.png", &png, true),
        ]);

        let package = Package::read(&bytes).unwrap();
        assert_eq!(package.layers, [Layer::new("face", 1, 1, image(2, 2, 9))]);
        assert_eq!((package.picture, package.source), (None, None));
    }

    #[test]
    fn what_is_not_a_package_says_why() {
        assert_eq!(error(b"8BPS"), "not a waifu2d package");

        let mut zip = zip::ZipWriter::new(Cursor::new(Vec::new()));
        zip.start_file("a", zip::write::SimpleFileOptions::default())
            .unwrap();
        let other = zip.finish().unwrap().into_inner();
        assert!(
            error(&other).contains("not a waifu2d package"),
            "{}",
            error(&other)
        );

        let newer = "format: waifu2d\nversion: 2\ncanvas: [4, 4]\nlayers: []\n";
        let newer = archive(&[("index.yaml", newer.as_bytes(), false)]);
        assert!(
            error(&newer).contains("version 2, newer than this reads (1)"),
            "{}",
            error(&newer)
        );

        let psd = "format: psd\nversion: 1\ncanvas: [4, 4]\nlayers: []\n";
        let psd = archive(&[("index.yaml", psd.as_bytes(), false)]);
        assert!(error(&psd).contains("of format \"psd\""), "{}", error(&psd));
    }

    #[test]
    fn a_missing_picture_names_its_layer() {
        let index = "format: waifu2d\nversion: 1\ncanvas: [4, 4]\nlayers:\n  - name: face\n    \
            image: nowhere.png\n    at: [0, 0]\n";
        let bytes = archive(&[("index.yaml", index.as_bytes(), false)]);
        assert_eq!(error(&bytes), "layer 0 (face): nowhere.png is missing");
    }

    #[test]
    fn a_damaged_entry_is_refused() {
        let mut bytes = package().write().unwrap();
        let text = b"format: \"waifu2d\"";
        let at = bytes.windows(text.len()).position(|w| w == text).unwrap();
        bytes[at] = b'F';
        assert!(Package::is_package(&bytes));
        let error = error(&bytes);
        assert!(
            error.contains("index.yaml") && error.contains("checksum"),
            "{error}"
        );
    }

    #[test]
    fn the_preview_lays_what_is_shown_over_each_other() {
        let mut package = Package::new(2, 1);
        let red = RgbaImage::new(2, 1, vec![255, 0, 0, 255, 255, 0, 0, 255]);
        let blue = RgbaImage::new(1, 1, vec![0, 0, 255, 255]);
        let mut half = Layer::new("half", 1, 0, blue.clone());
        half.opacity = 0.5;
        let mut hidden = Layer::new("hidden", 0, 0, blue);
        hidden.visible = false;
        package.layers = vec![Layer::new("red", 0, 0, red), half, hidden];
        assert_eq!(package.preview().rgba, [255, 0, 0, 255, 128, 0, 128, 255]);
    }
}
