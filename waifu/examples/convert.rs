// Copyright (c) 2026 Xiaoyang Chen
//
// Part of libwaifu's port of Seed-VC (https://github.com/Plachtaa/seed-vc), which is licensed
// under the GNU General Public License version 3 -- and so is this file, unlike the rest of
// libwaifu, which is MIT. It is compiled only with the `gpl` feature (CMake: -DENABLE_GPL=ON), and a
// build with it on is covered by the GPL as a whole. See LICENSE-GPL-3.0 at the top of the repository.
//
// This program is free software: you can redistribute it and/or modify it under the terms of the
// GNU General Public License, version 3, as published by the Free Software Foundation.
//
// This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
// without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See
// the GNU General Public License for more details.
//
// SPDX-License-Identifier: GPL-3.0-only

//! Say what one recording says in the voice of another, with Seed-VC v2.
//!
//! ```text
//! cargo run --release --example convert -- models/seed_vc.yaml source.wav voice.wav out.wav
//! cargo run --release --example convert -- models/seed_vc.yaml source.wav voice.wav out.wav \
//!     --style --seed 7
//! ```
//!
//! `--style` runs the AR as well, which re-says the source with the voice's accent and pacing
//! rather than only in its timbre. `--steps` sets the CFM's steps (thirty by default).

use std::ops::ControlFlow;
use std::time::Instant;

use waifu::seed_vc::{Conversion, SeedVc};
use waifu::{wav, Device, Manifest, Residency, SpeechProgress};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let arguments: Vec<String> = std::env::args().collect();
    let usage = "usage: convert <manifest.yaml> <source.wav> <voice.wav> <out.wav> \
                 [--style] [--seed N] [--steps N]";
    if arguments.len() < 5 {
        eprintln!("{usage}");
        std::process::exit(2);
    }
    let (manifest, source, voice, out) =
        (&arguments[1], &arguments[2], &arguments[3], &arguments[4]);

    let clock = Instant::now();
    let vc = SeedVc::from_manifest(Device::Cuda, Residency::Device, &Manifest::open(manifest)?)?;
    eprintln!("read the model in {:.1} s", clock.elapsed().as_secs_f64());

    let mut conversion: Conversion = vc.conversion();
    let mut rest = arguments[5..].iter();
    while let Some(flag) = rest.next() {
        match flag.as_str() {
            "--style" => conversion.convert_style = true,
            "--seed" => conversion.seed = Some(rest.next().ok_or(usage)?.parse()?),
            "--steps" => conversion.diffusion_steps = rest.next().ok_or(usage)?.parse()?,
            _ => {
                eprintln!("{usage}");
                std::process::exit(2);
            }
        }
    }

    let source = wav::read(&std::fs::read(source)?)?;
    let recording = wav::read(&std::fs::read(voice)?)?;

    let listening = Instant::now();
    let reference = vc.listen(&recording)?;
    eprintln!(
        "listened to {:.1} s of {} Hz in {:.1} s: {} tokens, {} mel frames",
        recording.seconds(),
        recording.rate,
        listening.elapsed().as_secs_f64(),
        reference.wide.len(),
        reference.frames
    );

    let converting = Instant::now();
    let mut last = 0;
    let mut report = |progress: SpeechProgress| {
        match progress {
            SpeechProgress::Saying { done, expected } if done >= last + 100 => {
                eprintln!("  token {done} of about {expected}");
                last = done;
            }
            SpeechProgress::Sounding => {
                eprintln!("  window at {:.1} s", converting.elapsed().as_secs_f64())
            }
            _ => {}
        }
        ControlFlow::Continue(())
    };

    let sound = vc
        .convert(&source, &reference, &conversion, &mut report)?
        .expect("nothing stops this run");
    eprintln!(
        "converted {:.2} s of audio into {:.2} s in {:.1} s",
        source.seconds(),
        sound.seconds(),
        converting.elapsed().as_secs_f64()
    );

    std::fs::write(out, wav::write(&sound))?;
    eprintln!("wrote {out}");

    Ok(())
}
