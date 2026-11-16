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
//! numbered bottom first as the app numbers them (`00-<tag>.png` is the lowest), and a list of them
//! in `.img2layers`. A rerun removes the PNGs that list names and nothing else, so a PNG somebody
//! put there survives -- unless it has a name this run writes itself.
//!
//! Handwear, eyes and ears are split into a left and a right by default, as the rig wants -- unlike
//! the app, whose box starts off -- and `no-pairs` keeps them whole. `seed=N` is 0 unless named;
//! `steps=N`, `depth-steps=N` and `resolution=N` are the model's own unless named.

use std::ops::ControlFlow;
use std::path::{Path, PathBuf};

use waifu::see_through::{Image, Rgba, SeeThrough, SeeThroughOptions, SeeThroughProgress};
use waifu::{Device, Manifest};

const USAGE: &str = "usage: img2layers MODEL.yaml PICTURE.png OUT.psd [cpu|cuda|vulkan|metal] \\\n\
    \x20   [pairs (default)|no-pairs] [seed=N] [steps=N] [depth-steps=N] [resolution=N] \\\n\
    \x20   [lowvram|hybrid=GB]";

/// The largest square the tool draws at: a mistyped `resolution=` is an error, not an allocation
/// the size of the machine.
const MAX_RESOLUTION: usize = 4096;

/// What the command line asked for, once `lowvram` and `hybrid=` are out of it.
#[derive(Debug)]
struct Arguments {
    model_path: String,
    picture_path: String,
    out_path: String,
    /// `None` for the best one there is.
    device: Option<Device>,
    options: SeeThroughOptions,
}

/// Reads the command line, checking every number against what the model can take, so that a bad
/// one is said before a model is loaded.
fn parse_arguments(arguments: Vec<String>) -> Result<Arguments, String> {
    let mut options = SeeThroughOptions {
        seed: 0,
        split_pairs: true,
        ..Default::default()
    };
    let mut positional = Vec::new();
    for a in arguments {
        let bad = |key: &str, what: &str| format!("{a}: expected {key}N, N {what}");
        if a == "pairs" {
            options.split_pairs = true;
        } else if a == "no-pairs" {
            options.split_pairs = false;
        } else if let Some(v) = a.strip_prefix("seed=") {
            options.seed = v
                .parse()
                .map_err(|_| bad("seed=", "a whole number from 0"))?;
        } else if let Some(v) = a.strip_prefix("steps=") {
            options.steps = Some(steps(v).ok_or_else(|| bad("steps=", "at least 1"))?);
        } else if let Some(v) = a.strip_prefix("depth-steps=") {
            options.depth_steps = Some(steps(v).ok_or_else(|| bad("depth-steps=", "at least 1"))?);
        } else if let Some(v) = a.strip_prefix("resolution=") {
            let what = format!("a multiple of 64 from 64 to {MAX_RESOLUTION}");
            options.resolution = Some(resolution(v).ok_or_else(|| bad("resolution=", &what))?);
        } else {
            positional.push(a);
        }
    }

    let mut positional = positional.into_iter();
    let (Some(model_path), Some(picture_path), Some(out_path)) =
        (positional.next(), positional.next(), positional.next())
    else {
        return Err(USAGE.to_string());
    };
    let device = match positional.next().as_deref() {
        None => None,
        Some("cpu") => Some(Device::Cpu),
        Some("cuda") => Some(Device::Cuda),
        Some("vulkan") => Some(Device::Vulkan),
        Some("metal") => Some(Device::Metal),
        Some(other) => {
            return Err(format!(
                "unknown device {other}: expected cpu, cuda, vulkan or metal"
            ))
        }
    };
    let rest: Vec<String> = positional.collect();
    if !rest.is_empty() {
        return Err(format!("unexpected {}\n{USAGE}", rest.join(" ")));
    }
    Ok(Arguments {
        model_path,
        picture_path,
        out_path,
        device,
        options,
    })
}

fn steps(value: &str) -> Option<i32> {
    value.parse().ok().filter(|steps: &i32| *steps >= 1)
}

fn resolution(value: &str) -> Option<usize> {
    value
        .parse()
        .ok()
        .filter(|side: &usize| (64..=MAX_RESOLUTION).contains(side) && side.is_multiple_of(64))
}

/// The file a part is written to: its place counted from the bottom, as the app counts, and its
/// tag with any `/` made safe for a file name.
fn part_file_name(index: usize, tag: &str) -> String {
    format!("{index:02}-{}.png", tag.replace('/', "_"))
}

/// Whether `name` is one `part_file_name` could have written: a bare file name, so that a list
/// somebody edited cannot point the cleanup outside the directory.
fn is_part_file_name(name: &str) -> bool {
    let Some((index, tag)) = name
        .strip_suffix(".png")
        .and_then(|stem| stem.split_once('-'))
    else {
        return false;
    };
    index.len() >= 2
        && index.bytes().all(|b| b.is_ascii_digit())
        && !tag.is_empty()
        && !name.contains(['/', '\\'])
}

/// The file in a layers directory naming the PNGs the last run wrote there, one a line: the only
/// ones a later run removes.
const PART_LIST: &str = ".img2layers";

/// Makes `dir` if it is not there and points it at this run's parts: removes the PNGs the last run's
/// list names, then writes the list of `names` before any of them is written, so a run stopped
/// halfway still leaves every file it made on the list. A PNG the list does not name -- one somebody
/// put there, or one a run before the list existed wrote -- is left alone.
fn replace_part_list(dir: &Path, names: &[String]) -> std::io::Result<()> {
    std::fs::create_dir_all(dir)?;
    let list = dir.join(PART_LIST);
    match std::fs::read_to_string(&list) {
        Ok(last) => {
            for name in last.lines().filter(|name| is_part_file_name(name)) {
                match std::fs::remove_file(dir.join(name)) {
                    Err(e) if e.kind() != std::io::ErrorKind::NotFound => return Err(e),
                    _ => {}
                }
            }
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
        Err(e) => return Err(e),
    }
    let mut text = names.join("\n");
    text.push('\n');
    std::fs::write(list, text)
}

fn main() -> Result<(), waifu::Error> {
    let mut arguments: Vec<String> = std::env::args().skip(1).collect();
    let device_weight_bytes = waifu_tools::device_weight_bytes(&mut arguments)?;
    let Arguments {
        model_path,
        picture_path,
        out_path,
        device,
        options,
    } = parse_arguments(arguments).unwrap_or_else(|message| {
        eprintln!("{message}");
        std::process::exit(1);
    });
    let device = device.unwrap_or_else(|| {
        [Device::Cuda, Device::Vulkan, Device::Metal]
            .into_iter()
            .find(|device| device.is_available())
            .unwrap_or(Device::Cpu)
    });

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
    let names: Vec<String> = (decomposition.parts.iter().enumerate())
        .map(|(index, part)| part_file_name(index, &part.tag))
        .collect();
    replace_part_list(&layers_dir, &names)?;
    for (part, name) in decomposition.parts.iter().zip(&names) {
        image::save_buffer(
            layers_dir.join(name),
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

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(line: &str) -> Result<Arguments, String> {
        parse_arguments(line.split_whitespace().map(String::from).collect())
    }

    #[test]
    fn defaults_and_named_values() {
        let a = parse("m.yaml p.png o.psd").unwrap();
        assert_eq!(
            (a.model_path.as_str(), a.out_path.as_str()),
            ("m.yaml", "o.psd")
        );
        assert_eq!(a.device, None);
        assert!(a.options.split_pairs);
        assert_eq!(a.options.seed, 0);
        assert_eq!(a.options.steps, None);

        let a =
            parse("m.yaml p.png o.psd cuda no-pairs seed=7 steps=12 depth-steps=3 resolution=640")
                .unwrap();
        assert_eq!(a.device, Some(Device::Cuda));
        assert!(!a.options.split_pairs);
        assert_eq!(a.options.seed, 7);
        assert_eq!(a.options.steps, Some(12));
        assert_eq!(a.options.depth_steps, Some(3));
        assert_eq!(a.options.resolution, Some(640));
    }

    #[test]
    fn numbers_out_of_range_are_refused() {
        for bad in [
            "resolution=-64",
            "resolution=0",
            "resolution=100",
            "resolution=8192",
            "resolution=18446744073709551552",
            "steps=0",
            "steps=-1",
            "steps=4294967326",
            "depth-steps=0",
            "seed=-1",
            "seed=x",
        ] {
            let error = parse(&format!("m.yaml p.png o.psd {bad}")).unwrap_err();
            assert!(error.starts_with(bad), "{bad}: {error}");
        }
    }

    #[test]
    fn extra_and_missing_arguments_are_refused() {
        assert!(parse("m.yaml p.png o.psd cuda typo")
            .unwrap_err()
            .starts_with("unexpected typo"));
        assert!(parse("m.yaml p.png o.psd gpu")
            .unwrap_err()
            .starts_with("unknown device gpu"));
        assert!(parse("m.yaml p.png").unwrap_err().starts_with("usage:"));
    }

    #[test]
    fn parts_are_numbered_bottom_first() {
        assert_eq!(part_file_name(0, "back hair"), "00-back hair.png");
        assert_eq!(part_file_name(12, "eyes/left"), "12-eyes_left.png");
        assert_eq!(part_file_name(123, "face"), "123-face.png");
        for name in ["00-a.png", "12-eyes_left.png", "123-face.png"] {
            assert!(is_part_file_name(name), "{name}");
        }
        for name in [
            "notes.txt",
            "composite.png",
            "1-a.png",
            "00-.png",
            "ab-c.png",
            "00-a.psd",
            "00-../../x.png",
            "00-a\\b.png",
        ] {
            assert!(!is_part_file_name(name), "{name}");
        }
    }

    #[test]
    fn a_rerun_removes_only_the_listed_parts() {
        let dir = std::env::temp_dir().join(format!("img2layers-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let names = |list: &[&str]| list.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        let left = || {
            let mut left: Vec<String> = std::fs::read_dir(&dir)
                .unwrap()
                .map(|e| e.unwrap().file_name().into_string().unwrap())
                .collect();
            left.sort();
            left
        };

        // The first run, into a directory that is not there, makes it and removes nothing.
        replace_part_list(&dir, &names(&["00-body.png", "01-hair.png"])).unwrap();
        for name in [
            "00-body.png",
            "01-hair.png",
            "00-reference.png",
            "notes.txt",
        ] {
            std::fs::write(dir.join(name), b"x").unwrap();
        }
        // The second removes the first's two, one of which is gone already, and nothing else.
        std::fs::remove_file(dir.join("01-hair.png")).unwrap();
        replace_part_list(&dir, &names(&["00-face.png"])).unwrap();
        assert_eq!(left(), [".img2layers", "00-reference.png", "notes.txt"]);
        assert_eq!(
            std::fs::read_to_string(dir.join(PART_LIST)).unwrap(),
            "00-face.png\n"
        );

        // A list naming something outside the directory is not followed.
        std::fs::write(dir.join(PART_LIST), "00-../notes.txt\nnotes.txt\n").unwrap();
        replace_part_list(&dir, &[]).unwrap();
        assert_eq!(left(), [".img2layers", "00-reference.png", "notes.txt"]);
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
