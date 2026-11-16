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

//! Converts a drawing in layers between the two files it can be: a Photoshop document img2layers
//! wrote, and a package (`.waifu2d`, docs/waifu2d-package.md), whichever way round the names say.
//!
//! ```bash
//! cargo run --release -p waifu-tools --bin waifu2d_convert -- picture.psd picture.waifu2d
//! ```
//!
//! A document's depth, in its hidden group `depth`, becomes each layer's depth map and median
//! depth, and `<part>-0` and `<part>-1` its near and far halves. The other way, what a document
//! from here cannot keep -- a hidden layer, an opacity -- is said.

use waifu_tools::drawing::{read_drawing, write_drawing};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [from, to] = args.as_slice() else {
        eprintln!("usage: waifu2d_convert <in.psd|in.waifu2d> <out.waifu2d|out.psd>");
        std::process::exit(2);
    };
    let run = || -> Result<(), String> {
        let package = read_drawing(from)?;
        write_drawing(&package, to)?;
        let depths = package
            .appearance
            .layers
            .iter()
            .filter(|l| l.depth_map.is_some());
        eprintln!(
            "wrote {to}: {} layers on a {} by {} canvas, {} with their depth",
            package.appearance.layers.len(),
            package.appearance.width,
            package.appearance.height,
            depths.count()
        );
        Ok(())
    };
    if let Err(error) = run() {
        eprintln!("{error}");
        std::process::exit(1);
    }
}
