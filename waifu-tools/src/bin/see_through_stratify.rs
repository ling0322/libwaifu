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
//! `heuristic_partseg.py seg_wdepth` does to a document it wrote. The depth is the document's own,
//! in its hidden group `depth`; a document from before it kept one is read with its depth document,
//! `<id>.layers/depth.psd`, beside it. What is written keeps its depth in the group too.
//!
//! ```bash
//! cargo run --release -p waifu-tools --bin see_through_stratify -- \
//!     picture.psd stratified.psd [depth.psd]
//! ```

use waifu::see_through::{
    depth_layer_name, read_psd, stratify, write_psd_with_hidden, DepthLayer, Image, PsdHiddenGroup,
    PsdLayer, PsdMode, DEPTH_GROUP,
};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 2 {
        eprintln!("usage: see_through_stratify <document.psd> <out.psd> [depth.psd]");
        std::process::exit(2);
    }
    if let Err(error) = run(&args[0], &args[1], args.get(2).map(String::as_str)) {
        eprintln!("{error}");
        std::process::exit(1);
    }
}

fn run(document: &str, out: &str, depth: Option<&str>) -> Result<(), String> {
    let read = |path: &str| {
        std::fs::read(path)
            .map_err(|e| format!("{path}: {e}"))
            .and_then(|bytes| read_psd(&bytes).map_err(|e| format!("{path}: {e}")))
    };
    let picture = read(document)?;
    if picture.mode != PsdMode::Rgb {
        return Err(format!("{document} is not an RGB document"));
    }
    let prefix = depth_layer_name("");
    let (depths, layers): (Vec<PsdLayer>, Vec<PsdLayer>) = picture
        .layers
        .into_iter()
        .partition(|layer| layer.name.starts_with(&prefix));

    // Each layer's depth, gray: its own from the group, or the depth document's in order.
    let grays: Vec<Image<u8>> = if depths.len() == layers.len() {
        depths
            .iter()
            .map(|d| {
                let n = d.pixels.width * d.pixels.height;
                Image::from_data(
                    d.pixels.width,
                    d.pixels.height,
                    1,
                    (0..n).map(|p| d.pixels.data[p * 4]).collect(),
                )
            })
            .collect()
    } else {
        let Some(depth) = depth else {
            return Err(format!(
                "{document} keeps no depth of its own: name its depth document too"
            ));
        };
        let depth = read(depth)?;
        if depth.mode != PsdMode::Grayscale || depth.layers.len() != layers.len() {
            return Err(format!(
                "the depth document is not the grayscale depth of the document's {} layers",
                layers.len()
            ));
        }
        depth
            .layers
            .iter()
            .map(|d| {
                let n = d.pixels.width * d.pixels.height;
                Image::from_data(
                    d.pixels.width,
                    d.pixels.height,
                    1,
                    (0..n).map(|p| d.pixels.data[p * 2]).collect(),
                )
            })
            .collect()
    };

    let mut given = Vec::with_capacity(layers.len());
    for (layer, gray) in layers.into_iter().zip(grays) {
        if (gray.width, gray.height) != (layer.pixels.width, layer.pixels.height) {
            return Err(format!("{}'s depth is not the same size as it", layer.name));
        }
        given.push(DepthLayer {
            name: layer.name,
            left: layer.left,
            top: layer.top,
            pixels: layer.pixels,
            depth: gray,
        });
    }
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

    let shown: Vec<PsdLayer> = layers
        .iter()
        .map(|l| PsdLayer {
            name: l.name.clone(),
            left: l.left,
            top: l.top,
            pixels: l.pixels.clone(),
        })
        .collect();
    let depths: Vec<PsdLayer> = layers
        .iter()
        .map(|l| {
            let n = l.depth.width * l.depth.height;
            let mut pixels = Image::new(l.depth.width, l.depth.height, 4);
            for p in 0..n {
                pixels.data[p * 4..p * 4 + 3].fill(l.depth.data[p]);
                pixels.data[p * 4 + 3] = l.pixels.data[p * 4 + 3];
            }
            PsdLayer {
                name: depth_layer_name(&l.name),
                left: l.left,
                top: l.top,
                pixels,
            }
        })
        .collect();
    let group = PsdHiddenGroup {
        name: DEPTH_GROUP,
        layers: &depths,
    };
    let bytes = write_psd_with_hidden(
        &shown,
        Some(group),
        picture.width,
        picture.height,
        PsdMode::Rgb,
    );
    std::fs::write(out, bytes).map_err(|e| format!("{out}: {e}"))
}
