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
//! and a package says which layers are whose halves and keeps each part it split, hidden.
//!
//! ```bash
//! cargo run --release -p waifu-tools --bin see_through_stratify -- \
//!     picture.waifu2d stratified.waifu2d [depth.psd]
//! ```

use waifu::see_through::stratify;
use waifu2d::psd::PsdDocument;
use waifu_tools::drawing::{
    attach_depth_document, depth_layers, lacks_depth, read_drawing, stratified_package,
    write_drawing,
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
    if lacks_depth(&package.appearance) {
        let Some(depth) = depth else {
            return Err(format!(
                "{drawing} keeps no depth of its own: name its depth document too"
            ));
        };
        let bytes = std::fs::read(depth).map_err(|e| format!("{depth}: {e}"))?;
        let document = PsdDocument::read(&bytes).map_err(|e| format!("{depth}: {e}"))?;
        attach_depth_document(&mut package.appearance, &document)
            .map_err(|e| format!("{depth}: {e}"))?;
    }

    let given = depth_layers(&package.appearance).map_err(|e| format!("{drawing}: {e}"))?;
    let layers = stratify(given.clone());
    let names = |layers: &[waifu::see_through::DepthLayer]| {
        layers
            .iter()
            .map(|l| l.name.as_str())
            .collect::<Vec<_>>()
            .join(", ")
    };
    eprintln!("before: {}", names(&given));
    eprintln!("after:  {}", names(&layers));
    write_drawing(&stratified_package(&package, &given, &layers), out)
}
