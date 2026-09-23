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

//! Prints what a model is: what its manifest says, which files it names, and the tensors they
//! hold.
//!
//! ```text
//! cargo run --release --example inspect -- models/sdxl-base.yaml
//! ```
//!
//! With `-n` it stops after the manifest and reads no weights, which is the quick way to see what
//! a model says about itself without waiting for several gigabytes to be read.

fn main() -> Result<(), waifu::Error> {
    let arguments: Vec<String> = std::env::args().skip(1).collect();
    let names_only = arguments.iter().any(|argument| argument == "-n");

    let Some(path) = arguments.iter().find(|argument| !argument.starts_with('-')) else {
        eprintln!("usage: inspect [-n] <model.yaml>");
        std::process::exit(2);
    };

    let manifest = waifu::Manifest::open(path)?;
    println!("model: {}", manifest.id());
    println!("weights: {:?}", manifest.weights());
    println!("files: {:?}", manifest.files());

    let model_type = manifest.section("model")?.get_str("type")?.to_string();
    println!("kind: {model_type}");

    let suggested = manifest.suggested();
    if let Some(prompt) = &suggested.prompt {
        println!("suggested prompt: {prompt:?}");
    }
    if !suggested.sizes.is_empty() {
        println!("suggested sizes: {:?}", suggested.sizes);
    }
    if let Some(steps) = suggested.steps {
        println!("suggested steps: {steps}");
    }
    if let Some(guidance) = suggested.guidance {
        println!("suggested guidance: {guidance}");
    }

    if names_only {
        return Ok(());
    }

    let start = std::time::Instant::now();
    let file = waifu::read_safetensors(&manifest.weight_paths()?)?;
    println!("{} tensors read in {:?}", file.len(), start.elapsed());

    let mut names: Vec<&str> = file.keys().map(String::as_str).collect();
    names.sort_unstable();
    for name in names {
        println!("  {name}");
    }

    Ok(())
}
