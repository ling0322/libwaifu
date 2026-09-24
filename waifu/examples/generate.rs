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
//! cargo run --release --example generate -- sdxl.yaml "an astronaut riding a horse on mars"
//! ```
//!
//! A third argument names the device -- cpu, cuda or metal. Left out, it takes the first
//! accelerator this build can reach.
//!
//! Any of the three kinds of model draws here. Which one this is comes out of the manifest, the
//! same way `load` asks, and it decides the numbers as much as the code: the distilled releases
//! -- Anima's turbo and Krea 2 Turbo -- want eight steps at no guidance and come out burnt at the
//! thirty and five SDXL likes. So the request starts at what this build believes about the kind of model, and then
//! takes whatever the model's own `suggested:` block says over that -- which is the order the
//! screen asks them in too.
//!
//! The `webui` command in the CLI is this with a screen around it: the same pipeline, reported on
//! as it goes and interruptible between steps.

use std::io::Write;

use waifu::{
    to_rgb8, Anima, Device, GenerationDefaults, GenerationOptions, Krea2, Manifest, Residency,
    Sdxl,
};

fn main() -> Result<(), waifu::Error> {
    let mut arguments: Vec<String> = std::env::args().skip(1).collect();

    // Taken out wherever it was put, since it is a flag and not one of the three positions.
    let residency = match arguments.iter().position(|a| a == "lowvram") {
        Some(at) => {
            arguments.remove(at);
            Residency::LowVram
        }
        None => Residency::Device,
    };

    let mut arguments = arguments.into_iter();
    let (Some(model_path), Some(prompt)) = (arguments.next(), arguments.next()) else {
        eprintln!("usage: generate MODEL.yaml PROMPT [cpu|cuda|metal] [lowvram]");
        std::process::exit(1);
    };

    let device = match arguments.next().as_deref() {
        Some("cpu") => Device::Cpu,
        Some("cuda") => Device::Cuda,
        Some("metal") => Device::Metal,
        None => {
            // At most one accelerator is ever built in, so this asks rather than assumes.
            if Device::Cuda.is_available() {
                Device::Cuda
            } else if Device::Metal.is_available() {
                Device::Metal
            } else {
                Device::Cpu
            }
        }
        Some(other) => {
            eprintln!("unknown device {other}: expected cpu, cuda or metal");
            std::process::exit(1);
        }
    };

    let manifest = Manifest::open(&model_path)?;
    let kind = manifest.section("model")?.get_str("type")?.to_string();
    eprintln!("drawing a {kind} on {device:?}, weights {residency:?}");

    // What the kind of model wants, and then what this one says over it. A model with no
    // `suggested:` block gets exactly its kind's defaults, which is what every model got before
    // there was anywhere to say otherwise.
    let defaults = match kind.as_str() {
        Anima::MODEL_TYPE => Anima::DEFAULTS,
        Krea2::MODEL_TYPE => Krea2::DEFAULTS,
        _ => GenerationDefaults::default(),
    };
    let suggested = manifest.suggested();
    let options = GenerationOptions {
        // The author's own, where the card gave one. A model that suggests nothing to steer away
        // from leaves this empty, which is what it always was here.
        negative_prompt: suggested.avoid.clone().unwrap_or_default(),
        seed: Some(7),
        ..suggested.over(defaults).options()
    };
    eprintln!(
        "{} by {}, {} steps, guidance {}",
        options.width, options.height, options.num_steps, options.guidance_scale
    );

    let image = match kind.as_str() {
        Anima::MODEL_TYPE => {
            Anima::from_manifest(device, residency, &manifest)?.generate(&prompt, &options)?
        }
        Krea2::MODEL_TYPE => {
            Krea2::from_manifest(device, residency, &manifest)?.generate(&prompt, &options)?
        }
        _ => Sdxl::from_manifest(device, residency, &manifest)?.generate(&prompt, &options)?,
    };
    let pixels = to_rgb8(&image)?;

    // A PPM is what an image file looks like with nothing in the way: three numbers and the
    // pixels. The `webui` command writes a PNG, which is this plus two checksums.
    let mut out = std::fs::File::create("generate.ppm")?;
    write!(out, "P6\n{} {}\n255\n", options.width, options.height)?;
    out.write_all(&pixels)?;

    println!(
        "wrote generate.ppm, {} by {}",
        options.width, options.height
    );
    Ok(())
}
