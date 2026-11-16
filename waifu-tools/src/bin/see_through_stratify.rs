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

//! Stratifies a picture img2layers took apart: its hair, hands, top and bottom each cut into a
//! near and a far layer by their own depth (see `waifu::see_through::stratify`), as upstream's
//! `heuristic_partseg.py seg_wdepth` does to a document it wrote.
//!
//! It reads a package (`.waifu2d`) or a Photoshop document, whichever it is handed, and writes
//! what the out file's name says. The depth is the drawing's own: each layer's depth map in a
//! package, the hidden group `depth` in a document. A document from before it kept one is read
//! with its depth document, `<id>.layers/depth.psd`, beside it. What is written keeps its depth,
//! and a package says which layers are whose halves.
//!
//! ```bash
//! cargo run --release -p waifu-tools --bin see_through_stratify -- \
//!     picture.waifu2d stratified.waifu2d [depth.psd]
//! ```

use waifu::see_through::stratify;
use waifu2d::package::DepthMap;
use waifu2d::psd::PsdDocument;
use waifu_tools::drawing::{
    depth_layers_package, package_depth_layers, read_drawing, write_drawing,
};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 2 {
        eprintln!(
            "usage: see_through_stratify <drawing.waifu2d|.psd> <out.waifu2d|.psd> [depth.psd]"
        );
        std::process::exit(2);
    }
    if let Err(error) = run(&args[0], &args[1], args.get(2).map(String::as_str)) {
        eprintln!("{error}");
        std::process::exit(1);
    }
}

fn run(drawing: &str, out: &str, depth: Option<&str>) -> Result<(), String> {
    waifu_tools::drawing::Format::of(out)?;
    let mut package = read_drawing(drawing)?;
    let shown = package.layers.iter().filter(|l| l.visible).count();
    let kept = package
        .layers
        .iter()
        .filter(|l| l.depth_map.is_some())
        .count();
    if kept < shown {
        // Each layer's depth from the depth document, in order.
        let Some(depth) = depth else {
            return Err(format!(
                "{drawing} keeps no depth of its own: name its depth document too"
            ));
        };
        let bytes = std::fs::read(depth).map_err(|e| format!("{depth}: {e}"))?;
        let document = PsdDocument::read(&bytes).map_err(|e| format!("{depth}: {e}"))?;
        if document.layers.len() != shown {
            return Err(format!(
                "the depth document is not the depth of the drawing's {shown} layers"
            ));
        }
        let layers = package.layers.iter_mut().filter(|l| l.visible);
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
    }

    let given = package_depth_layers(&package).map_err(|e| format!("{drawing}: {e}"))?;
    let before: Vec<String> = given.iter().map(|l| l.name.clone()).collect();
    let layers = stratify(given);
    eprintln!("before: {}", before.join(", "));
    eprintln!(
        "after:  {}",
        layers
            .iter()
            .map(|l| l.name.as_str())
            .collect::<Vec<_>>()
            .join(", ")
    );

    let mut stratified = depth_layers_package(&layers, package.width, package.height);
    stratified.picture = package.picture;
    stratified.source = package.source.take();
    stratified.generator = std::mem::take(&mut package.generator);
    stratified
        .generator
        .insert("stratified_by".into(), "see_through_stratify".into());
    write_drawing(&stratified, out)
}
