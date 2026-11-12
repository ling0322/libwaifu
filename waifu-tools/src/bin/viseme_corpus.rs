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

//! Reads every sentence of a text file in every voice given, and writes what a viseme probe needs:
//! the audio, and S3Tokenizer's tokens of it, 25 a second.
//!
//! ```text
//! viseme_corpus <manifest.yaml> <sentences.txt> <out_dir> <voice.wav>...
//! ```
//!
//! `out_dir/corpus.jsonl` has one line a reading: `{"voice", "sentence", "text", "wav",
//! "tokens"}`. The tokens are the output heard again by [`CosyVoice3::listen`], not the language
//! model's own: they are what S3Tokenizer says of the sound, which is what the language model was
//! taught to say, and they line up with the audio by construction -- token `i` is the 40 ms from
//! `0.04 * i` -- with no prompt to cut off.

use std::io::Write;
use std::ops::ControlFlow;
use std::path::Path;

use waifu::cosyvoice3::CosyVoice3;
use waifu::{wav, Device, Manifest, SpeechOptions};

fn json_string(text: &str) -> String {
    let mut out = String::from("\"");
    for c in text.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let arguments: Vec<String> = std::env::args().collect();
    if arguments.len() < 5 {
        eprintln!("usage: viseme_corpus <manifest.yaml> <sentences.txt> <out_dir> <voice.wav>...");
        std::process::exit(2);
    }
    let manifest = Manifest::open(&arguments[1])?;
    let sentences: Vec<String> = std::fs::read_to_string(&arguments[2])?
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .map(str::to_string)
        .collect();
    let out_dir = Path::new(&arguments[3]);
    std::fs::create_dir_all(out_dir)?;
    let mut corpus = std::fs::File::create(out_dir.join("corpus.jsonl"))?;

    let tts = CosyVoice3::from_manifest(Device::Cuda, u64::MAX, &manifest)?;
    let mut report = |_| ControlFlow::Continue(());
    for (v, voice) in arguments[4..].iter().enumerate() {
        let reference = tts.listen(&wav::read(&std::fs::read(voice)?)?)?;
        for (s, text) in sentences.iter().enumerate() {
            let options = SpeechOptions {
                seed: Some((v * 1000 + s) as u64),
                ..CosyVoice3::DEFAULTS.options()
            };
            let sound = tts
                .say(text, &reference, &options, &mut report)?
                .expect("nothing stops this run");
            let name = format!("v{v}_s{s:03}.wav");
            std::fs::write(out_dir.join(&name), wav::write(&sound))?;
            let tokens = tts.listen(&sound)?.tokens;
            let tokens: Vec<String> = tokens.iter().map(i32::to_string).collect();
            writeln!(
                corpus,
                "{{\"voice\": {v}, \"sentence\": {s}, \"text\": {}, \"wav\": {}, \"tokens\": [{}]}}",
                json_string(text),
                json_string(&name),
                tokens.join(", ")
            )?;
            eprintln!("voice {v} sentence {s}: {:.1} s", sound.seconds());
        }
    }
    Ok(())
}
