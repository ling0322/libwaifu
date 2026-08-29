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

//! Draws one picture and writes it out as a PPM, which is a header and the pixels.
//!
//! ```text
//! cargo run --release --example generate -- sdxl.waifupkg "an astronaut riding a horse on mars"
//! ```
//!
//! The `draw` command in the CLI is this with a screen around it: the same pipeline, reported on
//! as it goes and interruptible between steps.

use std::io::Write;

use waifu::{to_rgb8, Device, GenerationOptions, Sdxl, ZipFile};

fn main() -> Result<(), waifu::Error> {
    let mut arguments = std::env::args().skip(1);
    let (Some(package_path), Some(prompt)) = (arguments.next(), arguments.next()) else {
        eprintln!("usage: generate MODEL.waifupkg PROMPT");
        std::process::exit(1);
    };

    let package = ZipFile::open(&package_path)?;
    let model = Sdxl::from_package(Device::Cuda, &package)?;

    let options = GenerationOptions {
        width: 1024,
        height: 1024,
        num_steps: 30,
        guidance_scale: 5.0,
        negative_prompt: String::new(),
        seed: Some(7),
    };

    let image = model.generate(&prompt, &options)?;
    let pixels = to_rgb8(&image)?;

    // A PPM is what an image file looks like with nothing in the way: three numbers and the
    // pixels. The `draw` command writes a PNG, which is this plus two checksums.
    let mut out = std::fs::File::create("generate.ppm")?;
    write!(out, "P6\n{} {}\n255\n", options.width, options.height)?;
    out.write_all(&pixels)?;

    println!(
        "wrote generate.ppm, {} by {}",
        options.width, options.height
    );
    Ok(())
}
