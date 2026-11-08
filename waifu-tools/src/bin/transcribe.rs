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

//! Write down what is said in a recording, with Qwen3-ASR.
//!
//! ```text
//! cargo run --release -p waifu-tools --bin transcribe -- models/qwen3-asr.yaml speech.wav
//! cargo run --release -p waifu-tools --bin transcribe -- models/qwen3-asr.yaml speech.wav English
//! cargo run --release -p waifu-tools --bin transcribe -- models/qwen3-asr.yaml speech.wav "" \
//!     --context="Qwen, libwaifu" --device=cpu
//! ```
//!
//! The words go to standard output, one line, and the language and the times to standard error. A
//! language, where given, is one of those the manifest lists; "" or none lets the model tell.

use std::ops::ControlFlow;
use std::time::Instant;

use waifu::{
    wav, Device, Manifest, Qwen3Asr, Transcriber, TranscriptionOptions, TranscriptionProgress,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let arguments: Vec<String> = std::env::args().collect();
    let flag = |name: &str| {
        arguments
            .iter()
            .find_map(|argument| argument.strip_prefix(&format!("--{name}=")))
    };
    let positional: Vec<&String> = arguments
        .iter()
        .skip(1)
        .filter(|argument| !argument.starts_with("--"))
        .collect();
    if positional.len() < 2 {
        eprintln!(
            "usage: transcribe <manifest.yaml> <recording.wav> [language] [--context=<words>] \
             [--device=cuda|cpu|metal|vulkan]"
        );
        std::process::exit(2);
    }
    let device = match flag("device").unwrap_or("cuda") {
        "cpu" => Device::Cpu,
        "metal" => Device::Metal,
        "vulkan" => Device::Vulkan,
        _ => Device::Cuda,
    };

    let manifest = Manifest::open(positional[0])?;
    let recording = wav::read(&std::fs::read(positional[1])?)?;
    let options = TranscriptionOptions {
        language: positional
            .get(2)
            .map(|language| language.to_string())
            .filter(|language| !language.trim().is_empty()),
        context: flag("context").unwrap_or_default().to_string(),
    };

    let clock = Instant::now();
    let model = Qwen3Asr::from_manifest(device, u64::MAX, &manifest)?;
    eprintln!("read the model in {:.1} s", clock.elapsed().as_secs_f64());

    let started = Instant::now();
    let mut writing = None;
    let mut report = |progress: TranscriptionProgress| {
        if let TranscriptionProgress::Writing { .. } = progress {
            writing.get_or_insert_with(Instant::now);
        }
        ControlFlow::Continue(())
    };
    let transcript = model
        .transcribe(&recording, &options, &mut report)?
        .expect("nothing stops this run");

    let seconds = started.elapsed().as_secs_f64();
    eprintln!(
        "{:.1} s of {} Hz written down in {seconds:.2} s ({:.0}x real time){}; language {:?}",
        recording.seconds(),
        recording.rate,
        recording.seconds() / seconds.max(1e-9),
        match writing {
            Some(at) => format!(", heard in {:.2} s", (at - started).as_secs_f64()),
            None => String::new(),
        },
        transcript.language
    );
    println!("{}", transcript.text);

    Ok(())
}
