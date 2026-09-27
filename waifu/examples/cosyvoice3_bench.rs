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

//! Time CosyVoice3's stages, through nothing but its public API.
//!
//! ```text
//! cargo run --release --example cosyvoice3_bench -- models/cosyvoice3.yaml voice.wav [runs]
//! ```
//!
//! Listening to the recording, and then for each of three texts: the language model (from the
//! start of a reading to the first `Sounding`) and the flow with the vocoder (from there to the
//! sound), each the median of `runs` readings at a fixed seed after one to warm up. The seed makes
//! every run read the same tokens, so the flow and vocoder see the same work each time.

use std::ops::ControlFlow;
use std::time::Instant;

use waifu::cosyvoice3::CosyVoice3;
use waifu::{wav, Device, Manifest, Residency, SpeechOptions, SpeechProgress};

const TEXTS: [(&str, &str); 3] = [
    (
        "zh-short",
        "收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐。",
    ),
    (
        "zh-long",
        "八百标兵奔北坡，北坡炮兵并排跑，炮兵怕把标兵碰，标兵怕碰炮兵炮。\
         今天的天气非常好，我们一起去公园散步吧。春天来了，花都开了，小鸟在树上唱歌，\
         孩子们在草地上放风筝，一切都显得那么美好。",
    ),
    (
        "en",
        "The train leaves at 7 tonight, and the ticket costs 25 dollars. \
         Please arrive at the station a little early.",
    ),
];

fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    values[values.len() / 2]
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let arguments: Vec<String> = std::env::args().collect();
    if arguments.len() < 3 {
        eprintln!("usage: cosyvoice3_bench <manifest.yaml> <voice.wav> [runs]");
        std::process::exit(2);
    }
    let runs: usize = arguments
        .get(3)
        .map(|n| n.parse())
        .transpose()?
        .unwrap_or(5);

    let tts = CosyVoice3::from_manifest(
        Device::Cuda,
        Residency::Device,
        &Manifest::open(&arguments[1])?,
    )?;
    let recording = wav::read(&std::fs::read(&arguments[2])?)?;

    // Listening, once to warm up and then timed.
    let mut reference = tts.listen(&recording)?;
    let mut listening = Vec::new();
    for _ in 0..runs {
        let clock = Instant::now();
        reference = tts.listen(&recording)?;
        listening.push(clock.elapsed().as_secs_f64());
    }
    println!("listen: {:.3} s median of {runs}", median(listening));

    let options = SpeechOptions {
        seed: Some(11),
        ..CosyVoice3::DEFAULTS.options()
    };

    println!(
        "{:10} {:>8} {:>8} {:>8} {:>8} {:>7}",
        "text", "audio", "lm", "sound", "total", "rtf"
    );
    for (name, text) in TEXTS {
        let mut lm = Vec::new();
        let mut sound = Vec::new();
        let mut total = Vec::new();
        let mut seconds = 0.0;

        for run in 0..=runs {
            let clock = Instant::now();
            let mut sounding: Option<f64> = None;
            let said = tts
                .say(text, &reference, &options, &mut |progress| {
                    if progress == SpeechProgress::Sounding && sounding.is_none() {
                        sounding = Some(clock.elapsed().as_secs_f64());
                    }
                    ControlFlow::Continue(())
                })?
                .expect("nothing stops this reading");
            let elapsed = clock.elapsed().as_secs_f64();
            seconds = said.seconds();

            // The first run warms up: graphs, allocator, kernels chosen for these shapes.
            if run == 0 {
                continue;
            }
            let at = sounding.unwrap_or(elapsed);
            lm.push(at);
            sound.push(elapsed - at);
            total.push(elapsed);
        }

        let total = median(total);
        println!(
            "{name:10} {seconds:>7.2}s {:>7.3}s {:>7.3}s {total:>7.3}s {:>7.3}",
            median(lm),
            median(sound),
            total / seconds
        );
    }

    Ok(())
}
