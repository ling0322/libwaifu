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
//! `Package` in this file is always `waifu2d::package::Package`, never one of libwaifu's model
//! packages.

use std::collections::BTreeMap;
use std::path::Path;

use waifu::see_through::{
    depth_layer_name, write_psd_with_hidden, Decomposition, DepthLayer, Image, PsdHiddenGroup,
    PsdLayer, PsdMode, Rgba, DEPTH_GROUP,
};
use waifu2d::package::{Appearance, DepthMap, Kind, Layer, Package};
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
/// median depth, the picture itself, and `generator`.
pub fn decomposition_package(
    decomposition: &Decomposition,
    source: &Rgba,
    generator: BTreeMap<String, String>,
) -> Package {
    let side = decomposition.canvas as u32;
    let mut appearance = Appearance::new(side, side);
    appearance.generator = generator;
    appearance.source = Some(to_rgba(source));
    appearance.layers = (decomposition.parts.iter())
        .map(|part| {
            let mut layer = Layer::new(
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
    Package::new(appearance)
}

/// A layer with its depth, as stratifying takes and gives one, with its median depth.
fn depth_layer(l: &DepthLayer) -> Layer {
    let map = depth_map(&l.depth);
    let mut layer = Layer::new(l.name.clone(), l.left, l.top, to_rgba(&l.pixels));
    layer.depth = map.median(&layer.image);
    layer.depth_map = Some(map);
    layer
}

/// `package` stratified: `stratified`, what `waifu::see_through::stratify` made of `given`, with
/// its halves marked, and each part it split kept whole and hidden right under its near half --
/// its own place -- so that the package has the drawing both ways. Its name, its picture and what
/// made it are `package`'s, and what made it says it was stratified.
pub fn stratified_package(
    package: &Package,
    given: &[DepthLayer],
    stratified: &[DepthLayer],
) -> Package {
    let before = &package.appearance;
    let mut appearance = Appearance::new(before.width, before.height);
    appearance.layers = stratified.iter().map(depth_layer).collect();
    appearance.mark_halves();

    let mut layers = Vec::with_capacity(appearance.layers.len() + given.len());
    for layer in appearance.layers.drain(..) {
        if layer.kind == Kind::StratifiedNear {
            let part = layer.of.as_deref();
            if let Some(whole) = given.iter().find(|g| Some(g.name.as_str()) == part) {
                let mut whole = depth_layer(whole);
                whole.visible = false;
                layers.push(whole);
            }
        }
        layers.push(layer);
    }
    appearance.layers = layers;
    appearance.source = before.source.clone();
    appearance.generator = before.generator.clone();
    appearance
        .generator
        .insert("stratified_by".into(), "see_through_stratify".into());
    Package {
        name: package.name.clone(),
        appearance,
    }
}

/// Whether a layer that is shown has no depth map: what a separate depth document is for. A
/// hidden layer's depth, which stratifying never reads, does not count.
pub fn lacks_depth(appearance: &Appearance) -> bool {
    (appearance.layers.iter()).any(|l| l.visible && l.depth_map.is_none())
}

/// Gives each shown layer its depth from `document`: a depth document as img2layers once wrote
/// one beside its document, a layer for each shown layer in their order, the depth in its gray
/// (red, as it is read).
pub fn attach_depth_document(
    appearance: &mut Appearance,
    document: &waifu2d::psd::PsdDocument,
) -> Result<(), String> {
    let shown = appearance.layers.iter().filter(|l| l.visible).count();
    if document.layers.len() != shown {
        return Err(format!(
            "the depth document has {} layers, not one for each of the drawing's {shown}",
            document.layers.len()
        ));
    }
    let layers = appearance.layers.iter_mut().filter(|l| l.visible);
    for (layer, d) in layers.zip(&document.layers) {
        if (d.width(), d.height()) != (layer.image.width, layer.image.height) {
            return Err(format!("{}'s depth is not the same size as it", layer.name));
        }
        let map = DepthMap {
            width: d.width(),
            height: d.height(),
            gray: d.image.rgba.chunks(4).map(|p| p[0]).collect(),
        };
        layer.depth = map.median(&layer.image);
        layer.depth_map = Some(map);
    }
    Ok(())
}

/// The shown layers with their depth, for stratifying; an error naming a layer that has none, or
/// one that is a half already.
pub fn depth_layers(appearance: &Appearance) -> Result<Vec<DepthLayer>, String> {
    let mut out = Vec::new();
    for layer in appearance.layers.iter().filter(|l| l.visible) {
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

/// A package as the Photoshop document img2layers writes: its shown layers, each one's depth in
/// the hidden group `depth`. A document of that kind has neither opacity nor hidden layers: the
/// names of the layers that had either -- left out, or drawn at full opacity -- are returned
/// with it. A part a half names is left out and not named: the document is the stratified one,
/// as see_through_stratify has always written it.
pub fn package_psd(package: &Package) -> (Vec<u8>, Vec<String>) {
    let appearance = &package.appearance;
    let split: Vec<&str> = (appearance.layers.iter())
        .filter_map(|l| l.of.as_deref())
        .collect();
    let mut changed = Vec::new();
    let mut layers = Vec::new();
    let mut depths = Vec::new();
    for layer in &appearance.layers {
        let is_split = layer.kind == Kind::Part && split.contains(&layer.name.as_str());
        if !layer.visible && is_split {
            continue;
        }
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
        appearance.width as usize,
        appearance.height as usize,
        PsdMode::Rgb,
    );
    (bytes, changed)
}

/// The drawing at `path`, whichever of the two it is -- told by how it starts, not its name --
/// as a package.
pub fn read_drawing(path: &str) -> Result<Package, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{path}: {e}"))?;
    let read = if Package::is_package(&bytes) || bytes.starts_with(b"PK") {
        Package::read(&bytes)
    } else {
        waifu2d::psd::PsdDocument::read(&bytes).map(|d| Package::new(Appearance::from_psd(&d)))
    };
    read.map_err(|e| format!("{path}: {e}"))
}

/// Writes `package` to `path` in the format its name says. What a Photoshop document could not
/// keep of it is said on stderr.
pub fn write_drawing(package: &Package, path: &str) -> Result<(), String> {
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

    fn face() -> Layer {
        let image = RgbaImage::new(2, 1, vec![10, 20, 30, 255, 40, 50, 60, 128]);
        let mut face = Layer::new("face", 1, 2, image);
        face.depth = Some(40.0 / 255.0);
        face.depth_map = Some(DepthMap {
            width: 2,
            height: 1,
            gray: vec![40, 40],
        });
        face
    }

    fn package() -> Package {
        let mut appearance = Appearance::new(4, 4);
        appearance.layers = vec![face()];
        Package::new(appearance)
    }

    fn depth_layer_of(name: &str, left: i32, gray: u8) -> DepthLayer {
        DepthLayer {
            name: name.into(),
            left,
            top: 0,
            pixels: Image::from_data(1, 1, 4, vec![200, 0, 0, 255]),
            depth: Image::from_data(1, 1, 1, vec![gray]),
        }
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
        assert_eq!(Package::new(Appearance::from_psd(&document)), package);
    }

    #[test]
    fn what_a_photoshop_document_cannot_keep_is_named() {
        let mut package = package();
        let mut hidden = face();
        hidden.name = "hidden".into();
        hidden.visible = false;
        package.appearance.layers[0].opacity = 0.5;
        package.appearance.layers.push(hidden);
        let (bytes, changed) = package_psd(&package);
        assert_eq!(changed, ["face", "hidden"]);
        let document = waifu2d::psd::PsdDocument::read(&bytes).unwrap();
        let (shown, _) = document.split_depth();
        assert_eq!(shown.len(), 1);
    }

    #[test]
    fn a_stratified_package_has_the_drawing_both_ways() {
        let mut package = package();
        package.name = Some("ミク".into());
        package
            .appearance
            .generator
            .insert("name".into(), "img2layers".into());
        let given = [
            depth_layer_of("back", 0, 200),
            depth_layer_of("hair", 1, 100),
        ];
        // As stratify gives them: the far half down past "back", the near one in hair's place.
        let stratified = [
            depth_layer_of("hair-1", 1, 250),
            depth_layer_of("back", 0, 200),
            depth_layer_of("hair-0", 1, 90),
        ];
        let out = stratified_package(&package, &given, &stratified);

        let layers = &out.appearance.layers;
        let said: Vec<(&str, Kind, Option<&str>, bool)> = (layers.iter())
            .map(|l| (l.name.as_str(), l.kind, l.of.as_deref(), l.visible))
            .collect();
        assert_eq!(
            said,
            [
                ("hair-1", Kind::StratifiedFar, Some("hair"), true),
                ("back", Kind::Part, None, true),
                ("hair", Kind::Part, None, false),
                ("hair-0", Kind::StratifiedNear, Some("hair"), true),
            ]
        );
        // The split part as it was given, with its own depth.
        assert_eq!(layers[2].depth, Some(100.0 / 255.0));
        assert_eq!(out.name.as_deref(), Some("ミク"));
        assert_eq!(out.appearance.generator["name"], "img2layers");
        assert_eq!(
            out.appearance.generator["stratified_by"],
            "see_through_stratify"
        );

        // It cannot be stratified again; as a document it is the stratified drawing, the split
        // part left out without a word.
        assert_eq!(
            depth_layers(&out.appearance).unwrap_err(),
            "hair-1 is stratified already"
        );
        let (bytes, changed) = package_psd(&out);
        assert!(changed.is_empty(), "{changed:?}");
        let document = waifu2d::psd::PsdDocument::read(&bytes).unwrap();
        let names: Vec<&str> = (document.split_depth().0.iter())
            .map(|l| l.name.as_str())
            .collect();
        assert_eq!(names, ["hair-1", "back", "hair-0"]);
    }

    #[test]
    fn a_hidden_layer_s_depth_does_not_stand_for_a_shown_one_s() {
        // One shown layer without its depth and one hidden with: as many maps as shown layers,
        // and still the depth document is needed, and is used.
        let mut appearance = package().appearance;
        let mut hidden = face();
        hidden.name = "hidden".into();
        hidden.visible = false;
        appearance.layers[0].depth = None;
        appearance.layers[0].depth_map = None;
        appearance.layers.push(hidden);
        assert!(lacks_depth(&appearance));
        assert_eq!(depth_layers(&appearance).unwrap_err(), "face has no depth");

        let gray = RgbaImage::new(2, 1, vec![90, 90, 90, 255, 90, 90, 90, 255]);
        let document = waifu2d::psd::PsdDocument {
            width: 4,
            height: 4,
            layers: vec![waifu2d::psd::PsdLayer {
                name: "face".into(),
                left: 1,
                top: 2,
                image: gray,
                opacity: 1.0,
                visible: true,
            }],
        };
        attach_depth_document(&mut appearance, &document).unwrap();
        assert!(!lacks_depth(&appearance));
        assert_eq!(appearance.layers[0].depth, Some(90.0 / 255.0));
        let layers = depth_layers(&appearance).unwrap();
        assert_eq!(layers.len(), 1);
        assert_eq!(layers[0].depth.data, [90, 90]);
    }

    #[test]
    fn a_depth_document_must_be_one_layer_a_shown_layer() {
        let mut appearance = package().appearance;
        let document = waifu2d::psd::PsdDocument {
            width: 4,
            height: 4,
            layers: Vec::new(),
        };
        let error = attach_depth_document(&mut appearance, &document).unwrap_err();
        assert!(error.contains("has 0 layers"), "{error}");
    }

    #[test]
    fn stratifying_needs_every_layer_s_depth() {
        let mut appearance = package().appearance;
        assert_eq!(depth_layers(&appearance).unwrap().len(), 1);
        appearance.layers[0].depth_map = None;
        assert_eq!(depth_layers(&appearance).unwrap_err(), "face has no depth");
    }
}
