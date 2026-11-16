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

//! A drawing in layers on disk, for the tools that write and read one: a package (`.waifu2d`,
//! `waifu2d::package`), which is what they write, or a Photoshop document, which they still
//! write when asked and read wherever one is handed to them.
//!
//! "Package" in this file is always `waifu2d::package::Package`, never one of libwaifu's model
//! packages: it is written out in full so that nobody reading a tool has to wonder.

use std::collections::BTreeMap;
use std::path::Path;

use waifu::see_through::{
    depth_layer_name, write_psd_with_hidden, Decomposition, DepthLayer, Image, PsdHiddenGroup,
    PsdLayer, PsdMode, Rgba, DEPTH_GROUP,
};
use waifu2d::package::{DepthMap, Kind};
use waifu2d::RgbaImage;

/// The two files a drawing is written as, told by the name it is written to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Format {
    /// `.waifu2d`.
    Package,
    /// `.psd`.
    Psd,
}

impl Format {
    /// The format `path` names by its extension; an error saying what it may be otherwise.
    pub fn of(path: &str) -> Result<Format, String> {
        let extension = Path::new(path)
            .extension()
            .and_then(|e| e.to_str())
            .map(str::to_ascii_lowercase);
        match extension.as_deref() {
            Some(waifu2d::package::EXTENSION) => Ok(Format::Package),
            Some("psd") => Ok(Format::Psd),
            _ => Err(format!(
                "{path}: write a drawing to a .{} package or a .psd",
                waifu2d::package::EXTENSION
            )),
        }
    }
}

fn to_rgba(image: &Rgba) -> RgbaImage {
    assert_eq!(image.channels, 4, "a layer is RGBA");
    RgbaImage::new(image.width, image.height, image.data.clone())
}

fn from_rgba(image: &RgbaImage) -> Rgba {
    Image::from_data(image.width, image.height, 4, image.rgba.clone())
}

fn depth_map(gray: &Image<u8>) -> DepthMap {
    assert_eq!(gray.channels, 1, "a depth is gray");
    DepthMap {
        width: gray.width,
        height: gray.height,
        gray: gray.data.clone(),
    }
}

/// What img2layers took `source` apart into, as a package: each part with its depth and its
/// median depth, where the picture is on the canvas, the picture itself, and `generator`.
pub fn decomposition_package(
    decomposition: &Decomposition,
    source: &Rgba,
    generator: BTreeMap<String, String>,
) -> waifu2d::package::Package {
    let side = decomposition.canvas as u32;
    let mut package = waifu2d::package::Package::new(side, side);
    let (x, y, width, height) = decomposition.picture;
    package.picture = Some(waifu2d::package::Area {
        x: x as u32,
        y: y as u32,
        width: width as u32,
        height: height as u32,
    });
    package.generator = generator;
    package.source = Some(to_rgba(source));
    package.layers = (decomposition.parts.iter())
        .map(|part| {
            let mut layer = waifu2d::package::Layer::new(
                part.tag.clone(),
                part.left as i32,
                part.top as i32,
                to_rgba(&part.pixels),
            );
            layer.depth = Some(part.depth_median);
            layer.depth_map = Some(depth_map(&part.depth));
            layer
        })
        .collect();
    package
}

/// Layers with their depth -- see_through_stratify's -- as a package on a `width` by `height`
/// canvas, each with its median depth and its halves marked.
pub fn depth_layers_package(
    layers: &[DepthLayer],
    width: u32,
    height: u32,
) -> waifu2d::package::Package {
    let mut package = waifu2d::package::Package::new(width, height);
    package.layers = (layers.iter())
        .map(|l| {
            let image = to_rgba(&l.pixels);
            let map = depth_map(&l.depth);
            let mut layer = waifu2d::package::Layer::new(l.name.clone(), l.left, l.top, image);
            layer.depth = map.median(&layer.image);
            layer.depth_map = Some(map);
            layer
        })
        .collect();
    package.mark_halves();
    package
}

/// A package's shown layers with their depth, for stratifying; an error naming a layer that has
/// none.
pub fn package_depth_layers(
    package: &waifu2d::package::Package,
) -> Result<Vec<DepthLayer>, String> {
    let mut out = Vec::new();
    for layer in package.layers.iter().filter(|l| l.visible) {
        if layer.kind != Kind::Part {
            return Err(format!("{} is stratified already", layer.name));
        }
        let Some(map) = &layer.depth_map else {
            return Err(format!("{} has no depth", layer.name));
        };
        out.push(DepthLayer {
            name: layer.name.clone(),
            left: layer.left,
            top: layer.top,
            pixels: from_rgba(&layer.image),
            depth: Image::from_data(map.width, map.height, 1, map.gray.clone()),
        });
    }
    Ok(out)
}

/// A package as the Photoshop document img2layers writes: its layers, each one's depth in the
/// hidden group `depth`. A document of that kind has neither opacity nor hidden layers: the
/// names of the layers that had either -- left out, or drawn at full opacity -- are returned
/// with it.
pub fn package_psd(package: &waifu2d::package::Package) -> (Vec<u8>, Vec<String>) {
    let mut changed = Vec::new();
    let mut layers = Vec::new();
    let mut depths = Vec::new();
    for layer in &package.layers {
        if !layer.visible || layer.opacity != 1.0 {
            changed.push(layer.name.clone());
        }
        if !layer.visible {
            continue;
        }
        layers.push(PsdLayer {
            name: layer.name.clone(),
            left: layer.left,
            top: layer.top,
            pixels: from_rgba(&layer.image),
        });
        if let Some(map) = &layer.depth_map {
            let mut pixels = Image::new(map.width, map.height, 4);
            for (p, &g) in map.gray.iter().enumerate() {
                pixels.data[p * 4..p * 4 + 3].fill(g);
                pixels.data[p * 4 + 3] = layer.image.rgba[p * 4 + 3];
            }
            depths.push(PsdLayer {
                name: depth_layer_name(&layer.name),
                left: layer.left,
                top: layer.top,
                pixels,
            });
        }
    }
    let group = (!depths.is_empty()).then_some(PsdHiddenGroup {
        name: DEPTH_GROUP,
        layers: &depths,
    });
    let bytes = write_psd_with_hidden(
        &layers,
        group,
        package.width as usize,
        package.height as usize,
        PsdMode::Rgb,
    );
    (bytes, changed)
}

/// The drawing at `path`, whichever of the two it is -- told by how it starts, not its name --
/// as a package.
pub fn read_drawing(path: &str) -> Result<waifu2d::package::Package, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{path}: {e}"))?;
    let read = if waifu2d::package::Package::is_package(&bytes) || bytes.starts_with(b"PK") {
        waifu2d::package::Package::read(&bytes)
    } else {
        waifu2d::psd::PsdDocument::read(&bytes).map(|d| waifu2d::package::Package::from_psd(&d))
    };
    read.map_err(|e| format!("{path}: {e}"))
}

/// Writes `package` to `path` in the format its name says. What a Photoshop document could not
/// keep of it is said on stderr.
pub fn write_drawing(package: &waifu2d::package::Package, path: &str) -> Result<(), String> {
    let bytes = match Format::of(path)? {
        Format::Package => package.write().map_err(|e| format!("{path}: {e}"))?,
        Format::Psd => {
            let (bytes, changed) = package_psd(package);
            if !changed.is_empty() {
                eprintln!(
                    "{path}: a Photoshop document from here keeps every layer shown at full \
                    opacity; {} were not",
                    changed.join(", ")
                );
            }
            bytes
        }
    };
    std::fs::write(path, bytes).map_err(|e| format!("{path}: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn package() -> waifu2d::package::Package {
        let mut package = waifu2d::package::Package::new(4, 4);
        let image = RgbaImage::new(2, 1, vec![10, 20, 30, 255, 40, 50, 60, 128]);
        let mut face = waifu2d::package::Layer::new("face", 1, 2, image.clone());
        face.depth = Some(40.0 / 255.0);
        face.depth_map = Some(DepthMap {
            width: 2,
            height: 1,
            gray: vec![40, 40],
        });
        package.layers = vec![face];
        package
    }

    #[test]
    fn a_format_is_named_by_its_extension() {
        assert_eq!(Format::of("out.waifu2d"), Ok(Format::Package));
        assert_eq!(Format::of("a/b/OUT.PSD"), Ok(Format::Psd));
        for bad in ["out.png", "out", "out.waifu2d.bak"] {
            assert!(
                Format::of(bad).unwrap_err().contains(".waifu2d package"),
                "{bad}"
            );
        }
    }

    #[test]
    fn a_package_survives_a_photoshop_document() {
        let package = package();
        let (bytes, changed) = package_psd(&package);
        assert!(changed.is_empty());
        let document = waifu2d::psd::PsdDocument::read(&bytes).unwrap();
        assert_eq!(waifu2d::package::Package::from_psd(&document), package);
    }

    #[test]
    fn what_a_photoshop_document_cannot_keep_is_named() {
        let mut package = package();
        let mut hidden = package.layers[0].clone();
        hidden.name = "hidden".into();
        hidden.visible = false;
        package.layers[0].opacity = 0.5;
        package.layers.push(hidden);
        let (bytes, changed) = package_psd(&package);
        assert_eq!(changed, ["face", "hidden"]);
        let document = waifu2d::psd::PsdDocument::read(&bytes).unwrap();
        let (shown, _) = document.split_depth();
        assert_eq!(shown.len(), 1);
    }

    #[test]
    fn stratifying_needs_every_layer_s_depth() {
        let mut package = package();
        assert_eq!(package_depth_layers(&package).unwrap().len(), 1);
        package.layers[0].depth_map = None;
        assert_eq!(
            package_depth_layers(&package).unwrap_err(),
            "face has no depth"
        );
    }
}
