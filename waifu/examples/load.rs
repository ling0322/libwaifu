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

//! Builds a whole model from its manifest and encodes a prompt with it, without drawing anything.
//!
//! ```text
//! cargo run --release --example load -- models/sdxl-base.yaml [cpu|cuda]
//! ```
//!
//! What `inspect` does not: the weights are not merely read but assembled into the model the
//! manifest says they are, and the tokenizer it names is read and asked for something. That is the
//! whole path a draw takes, short of the drawing -- so it is the cheapest thing that says a
//! converted model actually loads.

use waifu::{Anima, Device, Manifest, Residency, Sdxl};

fn main() -> Result<(), waifu::Error> {
    let arguments: Vec<String> = std::env::args().skip(1).collect();
    let Some(path) = arguments.first() else {
        eprintln!("usage: load <model.yaml> [cpu|cuda]");
        std::process::exit(2);
    };

    let device = match arguments.get(1).map(String::as_str) {
        Some("cuda") => Device::Cuda,
        _ => Device::Cpu,
    };

    let manifest = Manifest::open(path)?;
    let kind = manifest.section("model")?.get_str("type")?.to_string();
    println!("{}: a {kind}, on {device:?}", manifest.id());

    let started = std::time::Instant::now();
    let prompt = "1girl, solo, masterpiece";

    let ids = match kind.as_str() {
        Anima::MODEL_TYPE => {
            let model = Anima::from_manifest(device, Residency::Device, &manifest)?;
            println!("built in {:?}", started.elapsed());
            model.encode_prompt(prompt)?.shape()
        }
        _ => {
            let model = Sdxl::from_manifest(device, Residency::Device, &manifest)?;
            println!("built in {:?}", started.elapsed());
            model.encode_prompt(prompt)?.context.shape()
        }
    };

    println!("encoded {prompt:?} to a context of {ids:?}");
    Ok(())
}
