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

//! The whole of IndexTTS-2.5, from a recording and a sentence to a waveform.
//!
//! Every model in it is checked against its reference by a test binary of its own. This checks
//! the joins, and the one thing none of those can: that what comes out is the sentence, in the
//! voice it was given.
//!
//! ```text
//! .venv/bin/python tools/indextts_package.py -output models/indextts25.safetensors
//! PYTHONPATH=~/.cache/libwaifu/pydeps .venv/bin/python tools/indextts_reference_voices.py
//! cargo test --release --manifest-path waifu/Cargo.toml --test indextts -- --ignored --nocapture
//! ```
//!
//! # Why this is one test
//!
//! The package is five gigabytes and the harness gives every test its own thread, so a fixture is
//! read once per test. What follows is one pass that asks several questions.
//!
//! # What "in the voice it was given" is measured with
//!
//! CAMPPlus's embedding of who is speaking -- the same model the pipeline conditions on, run over
//! what came out. Comparatively: a sentence said in speaker A's voice has to sound more like A
//! than like B, by a margin. A threshold alone would say little, because two unrelated speakers
//! are often only a few hundredths apart; measured with upstream's own CAMPPlus, the sentence below
//! sits at 0.85 against its own speaker and 0.57 against the other one, and two real recordings of
//! one speaker sit at 0.84.
//!
//! What this cannot measure is whether the words are right. That was checked while this was
//! written by running Whisper over the output -- it heard exactly the sentence, in English, in
//! Chinese, and across several segments -- but Whisper is not a thing this crate runs.

use std::ops::ControlFlow;
use std::path::PathBuf;

use waifu::flint::Tensor;
use waifu::indextts::{IndexTts, RATE};
use waifu::{wav, DType, Device, Manifest, Residency, SpeechOptions, SpeechProgress, Voice};

const SENTENCE: &str = "Hello, this is a test of the new voice.";

fn models_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../models")
}

fn recording(name: &str) -> waifu::Sound {
    let path = models_dir().join(name);
    let bytes = std::fs::read(&path).unwrap_or_else(|error| {
        panic!(
            "{}: {error} -- tools/indextts_reference_voices.py writes it",
            path.display()
        )
    });

    wav::read(&bytes).unwrap()
}

fn host(tensor: &Tensor) -> Vec<f32> {
    tensor
        .to_device(Device::Cpu)
        .unwrap()
        .cast(DType::Float)
        .unwrap()
        .to_vec_f32()
        .unwrap()
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm = |v: &[f32]| v.iter().map(|x| x * x).sum::<f32>().sqrt();

    dot / (norm(a) * norm(b))
}

fn options(seed: u64) -> SpeechOptions {
    SpeechOptions {
        speed: 1.0,
        temperature: 0.8,
        seed: Some(seed),
    }
}

fn carry_on(_: SpeechProgress) -> ControlFlow<()> {
    ControlFlow::Continue(())
}

#[test]
#[ignore]
fn says_a_sentence_in_the_voice_it_was_given() {
    let manifest = Manifest::open(models_dir().join("indextts25.yaml")).unwrap();
    let tts = IndexTts::from_manifest(Device::Cuda, Residency::Device, &manifest).unwrap();

    let (a, b) = (
        recording("indextts25-reference.wav"),
        recording("indextts25-other.wav"),
    );
    let (heard_a, heard_b) = (tts.listen(&a).unwrap(), tts.listen(&b).unwrap());

    // What comes out is a sound, of a length a short sentence takes to say.
    let said = tts
        .say(SENTENCE, &heard_a, &options(42), &mut carry_on)
        .unwrap()
        .expect("nothing stops this reading");

    assert_eq!(said.rate, RATE);
    let seconds = said.seconds();
    let loudness =
        (said.samples.iter().map(|s| s * s).sum::<f32>() / said.samples.len() as f32).sqrt();
    println!("said {seconds:.2} s at {} Hz, rms {loudness:.4}", said.rate);

    assert!(
        said.samples.iter().all(|s| s.is_finite()),
        "a sample is not a number"
    );
    assert!(
        (1.5..8.0).contains(&seconds),
        "{seconds:.2} s for eight words"
    );
    assert!(
        loudness > 0.01,
        "rms {loudness}: that is silence, not speech"
    );

    // In A's voice rather than B's.
    let speaker = host(&tts.listen(&said).unwrap().speaker);
    let (like_a, like_b) = (
        cosine(&speaker, &host(&heard_a.speaker)),
        cosine(&speaker, &host(&heard_b.speaker)),
    );
    println!("like the speaker it was given {like_a:.3}, like the other one {like_b:.3}");
    assert!(
        like_a > like_b + 0.1,
        "{like_a:.3} against {like_b:.3}: not recognisably the voice it was given"
    );

    // A seed says the same thing twice, to the sample -- which is what the page promises of one.
    let again = tts
        .say(SENTENCE, &heard_a, &options(42), &mut carry_on)
        .unwrap()
        .unwrap();
    assert_eq!(
        again.samples, said.samples,
        "the same seed read the sentence differently"
    );

    // A stop is a reading that comes back with nothing, not an error.
    let mut stop_at_the_first_token = |progress| match progress {
        SpeechProgress::Saying { .. } => ControlFlow::Break(()),
        _ => ControlFlow::Continue(()),
    };
    let stopped = tts
        .say(
            SENTENCE,
            &heard_a,
            &options(42),
            &mut stop_at_the_first_token,
        )
        .unwrap();
    assert!(
        stopped.is_none(),
        "a stopped reading came back with a sound"
    );

    // As a voice on the page: no recording is a sentence saying why, not a panic or a guess.
    let error = tts
        .speak(SENTENCE, None, &options(42), &mut carry_on)
        .unwrap_err()
        .to_string();
    assert!(error.contains("recording"), "{error}");
}
