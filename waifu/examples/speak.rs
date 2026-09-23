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

//! Say a sentence in the voice of a recording, with IndexTTS-2.5.
//!
//! ```text
//! cargo run --release --example speak -- models/indextts25.yaml voice.wav "Hello there." out.wav
//! ```
//!
//! The whole pipeline from the command line, with nothing of the web page in the way: what each
//! stage took is printed as it finishes, which is what this is for.

use std::ops::ControlFlow;
use std::time::Instant;

use waifu::indextts::IndexTts;
use waifu::{wav, Device, Manifest, Residency, SpeechOptions, SpeechProgress};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let arguments: Vec<String> = std::env::args().collect();
    if arguments.len() < 5 {
        eprintln!("usage: speak <manifest.yaml> <voice.wav> <text> <out.wav> [seed]");
        std::process::exit(2);
    }
    let (manifest, voice, text, out) = (&arguments[1], &arguments[2], &arguments[3], &arguments[4]);
    let seed = arguments.get(5).map(|seed| seed.parse()).transpose()?;

    let clock = Instant::now();
    let tts = IndexTts::from_manifest(Device::Cuda, Residency::Device, &Manifest::open(manifest)?)?;
    eprintln!("read the model in {:.1} s", clock.elapsed().as_secs_f64());

    let recording = wav::read(&std::fs::read(voice)?)?;
    let clock = Instant::now();
    let reference = tts.listen(&recording)?;
    eprintln!(
        "listened to {:.1} s of {} Hz in {:.1} s: {} feature frames, {} mel frames",
        recording.seconds(),
        recording.rate,
        clock.elapsed().as_secs_f64(),
        reference.frames,
        reference.prompt_frames
    );

    let options = SpeechOptions {
        speed: 1.0,
        temperature: tts.settings().temperature,
        seed,
    };

    let clock = Instant::now();
    let mut last = 0;
    let sound = tts.say(text, &reference, &options, &mut |progress| {
        match progress {
            SpeechProgress::Saying { done, expected } if done >= last + 50 => {
                eprintln!("  token {done} of about {expected}");
                last = done;
            }
            SpeechProgress::Sounding => eprintln!(
                "  tokens done at {:.1} s, sounding",
                clock.elapsed().as_secs_f64()
            ),
            _ => {}
        }
        ControlFlow::Continue(())
    })?;

    let sound = sound.expect("nothing stops this run");
    eprintln!(
        "said {:.2} s of audio in {:.1} s",
        sound.seconds(),
        clock.elapsed().as_secs_f64()
    );

    std::fs::write(out, wav::write(&sound))?;
    eprintln!("wrote {out}");

    Ok(())
}
