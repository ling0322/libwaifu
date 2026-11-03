// The MIT License (MIT)
//
// Copyright (c) 2026 Xiaoyang Chen
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software
// and associated documentation files (the "Software"), to deal in the Software without
// restriction, including without limitation the rights to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so.
//
// The above copyright notice and this permission notice shall be included in all copies or
// substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
// BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
// DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

//! Takes a picture apart into a layered Photoshop document, as the app's img2layers does.
//!
//! ```text
//! cargo run --release -p waifu-tools --bin img2layers -- models/see-through.yaml picture.png out.psd
//! ```
//!
//! The document carries each part's depth in a hidden group (`depth/<tag>`), which is what
//! `see_through_stratify` and the waifu2d page read. `<out>.layers/` gets a PNG of every part,
//! bottom first, for a look without Photoshop.
//!
//! `pairs` splits handwear, eyes and ears into a left and a right, as the rig wants; `seed=N`,
//! `steps=N`, `depth-steps=N` and `resolution=N` are the model's own unless named.

use std::ops::ControlFlow;
use std::path::PathBuf;

use waifu::see_through::{Image, Rgba, SeeThrough, SeeThroughOptions, SeeThroughProgress};
use waifu::{Device, Manifest};

fn main() -> Result<(), waifu::Error> {
    let mut arguments: Vec<String> = std::env::args().skip(1).collect();

    let device_weight_bytes = waifu_tools::device_weight_bytes(&mut arguments)?;

    let mut options = SeeThroughOptions {
        seed: 0,
        split_pairs: true,
        ..Default::default()
    };
    let number = |a: &str, key: &str| -> Option<i64> {
        a.strip_prefix(key).map(|value| {
            value.parse().unwrap_or_else(|_| {
                eprintln!("{a}: expected {key}N");
                std::process::exit(1);
            })
        })
    };
    arguments.retain(|a| {
        match a.as_str() {
            "pairs" => options.split_pairs = true,
            "no-pairs" => options.split_pairs = false,
            _ => {
                if let Some(v) = number(a, "seed=") {
                    options.seed = v as u64;
                } else if let Some(v) = number(a, "steps=") {
                    options.steps = Some(v as i32);
                } else if let Some(v) = number(a, "depth-steps=") {
                    options.depth_steps = Some(v as i32);
                } else if let Some(v) = number(a, "resolution=") {
                    options.resolution = Some(v as usize);
                } else {
                    return true;
                }
            }
        }
        false
    });

    let mut arguments = arguments.into_iter();
    let (Some(model_path), Some(picture_path), Some(out_path)) =
        (arguments.next(), arguments.next(), arguments.next())
    else {
        eprintln!(
            "usage: img2layers MODEL.yaml PICTURE.png OUT.psd [cpu|cuda|vulkan] \\\n\
            \x20   [pairs|no-pairs] [seed=N] [steps=N] [depth-steps=N] [resolution=N] \\\n\
            \x20   [lowvram|hybrid=GB]"
        );
        std::process::exit(1);
    };

    let device = match arguments.next().as_deref() {
        Some("cpu") => Device::Cpu,
        Some("cuda") => Device::Cuda,
        Some("vulkan") => Device::Vulkan,
        Some("metal") => Device::Metal,
        None => {
            if Device::Cuda.is_available() {
                Device::Cuda
            } else if Device::Vulkan.is_available() {
                Device::Vulkan
            } else if Device::Metal.is_available() {
                Device::Metal
            } else {
                Device::Cpu
            }
        }
        Some(other) => {
            eprintln!("unknown device {other}: expected cpu, cuda, vulkan or metal");
            std::process::exit(1);
        }
    };

    let manifest = Manifest::open(&model_path)?;
    eprintln!(
        "taking {picture_path} apart on {device:?}, weights kept on it: {}",
        waifu_tools::bytes_kept(device_weight_bytes)
    );

    let decoded = image::open(&picture_path)
        .map_err(|e| waifu::Error::Model(format!("{picture_path}: {e}")))?
        .to_rgba8();
    let (width, height) = decoded.dimensions();
    let picture: Rgba = Image::from_data(width as usize, height as usize, 4, decoded.into_vec());
    eprintln!("{width} by {height}, split_pairs: {}", options.split_pairs);

    let model = SeeThrough::from_manifest(device, device_weight_bytes, &manifest)?;
    let progress = &mut |what: SeeThroughProgress| {
        eprintln!("{what:?}");
        ControlFlow::Continue(())
    };
    let Some(decomposition) = model.decompose(&picture, &options, progress)? else {
        eprintln!("stopped");
        return Ok(());
    };

    std::fs::write(&out_path, decomposition.to_psd())?;

    let layers_dir = PathBuf::from(&out_path).with_extension("layers");
    std::fs::create_dir_all(&layers_dir)?;
    for (index, part) in decomposition.parts.iter().enumerate() {
        let name = format!(
            "{:02}-{}.png",
            decomposition.parts.len() - 1 - index,
            part.tag
        );
        image::save_buffer(
            layers_dir.join(name.replace('/', "_")),
            &part.pixels.data,
            part.pixels.width as u32,
            part.pixels.height as u32,
            image::ColorType::Rgba8,
        )
        .map_err(|e| waifu::Error::Model(format!("{}: {e}", layers_dir.display())))?;
    }

    let (px, py, pw, ph) = decomposition.picture;
    eprintln!(
        "wrote {out_path}: {} parts on a {} canvas, the picture at {px},{py} {pw} by {ph}; \
        a PNG a part in {}",
        decomposition.parts.len(),
        decomposition.canvas,
        layers_dir.display()
    );
    Ok(())
}
