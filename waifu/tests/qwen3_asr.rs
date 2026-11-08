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

//! Qwen3-ASR against the reference, stage by stage, on real recordings.
//!
//! Needs `models/qwen3-asr.yaml` (`tools/qwen3_asr_exporter.py`) and
//! `models/qwen3-asr_test.safetensors` (`tools/qwen3_asr_reference.py`, with the three recordings
//! docs/qwen3_asr.md names). Every input a stage is handed is one the reference really handed it:
//! the recording, the mel it heard, the tokens it said.
//!
//! One test, because reading the model is most of the time, and every question is asked of that
//! one reading.

use std::collections::HashMap;
use std::ops::ControlFlow;
use std::path::PathBuf;

use waifu::flint::Tensor;
use waifu::qwen3_asr::audio_encoder::Windows;
use waifu::qwen3_asr::decoder::argmax;
use waifu::{
    read_safetensors, DType, Device, Manifest, Qwen3Asr, Sound, Transcriber, TranscriptionOptions,
    TranscriptionProgress,
};

const DEVICE: Device = Device::Cuda;

/// What each recording says, as the reference wrote it.
const CASES: &[(&str, &str, &str)] = &[
    ("zh", "Chinese", "希望你以后能够做得比我还好哟。"),
    (
        "long",
        "Chinese",
        "在那之后，完全收购那家公司，因此保持管理层的一致性，利益与即将加入家族的资产保持一致。\
         这就是我们有时不买下全部的原因。",
    ),
    (
        "en",
        "English",
        "The train leaves at seven tonight, and the ticket costs twenty-five dollars.",
    ),
];

fn models_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../models")
}

fn fixture() -> HashMap<String, Tensor> {
    read_safetensors(&[models_dir().join("qwen3-asr_test.safetensors")]).unwrap()
}

fn floats(tensor: &Tensor) -> Vec<f32> {
    tensor
        .to_device(Device::Cpu)
        .unwrap()
        .cast(DType::Float)
        .unwrap()
        .to_vec_f32()
        .unwrap()
}

fn ints(tensor: &Tensor) -> Vec<i32> {
    tensor
        .to_vec_i64()
        .unwrap()
        .into_iter()
        .map(|x| x as i32)
        .collect()
}

/// The largest difference against the reference, as a share of the reference's largest value.
fn relative_error(got: &[f32], want: &[f32]) -> f32 {
    assert_eq!(got.len(), want.len());
    let scale = want.iter().fold(0f32, |m, x| m.max(x.abs())).max(1e-6);
    let worst = got
        .iter()
        .zip(want)
        .map(|(g, w)| (g - w).abs())
        .fold(0.0, f32::max);
    worst / scale
}

fn log_softmax(scores: &[f32]) -> Vec<f32> {
    let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let total: f64 = scores.iter().map(|s| (*s as f64 - max).exp()).sum();
    let log_total = total.ln() + max;
    scores
        .iter()
        .map(|s| (*s as f64 - log_total) as f32)
        .collect()
}

/// The largest difference between two rows of log probabilities over the tokens the reference
/// gives any real chance to.
fn logp_error(got: &[f32], want: &[f32]) -> f32 {
    got.iter()
        .zip(want)
        .filter(|(_, w)| **w > -10.0)
        .map(|(g, w)| (g - w).abs())
        .fold(0.0, f32::max)
}

#[test]
#[ignore]
fn hears_the_recordings_as_the_reference_does() {
    let manifest = Manifest::open(models_dir().join("qwen3-asr.yaml")).unwrap();
    let model = Qwen3Asr::from_manifest(DEVICE, u64::MAX, &manifest).unwrap();
    let cases = fixture();
    let vocab = model.config().text.vocab_size as usize;

    for (name, language, text) in CASES {
        let wave = floats(&cases[&format!("{name}.wave")]);

        // The features.
        let mel = model.mel(&wave).unwrap();
        let want_mel = &cases[&format!("{name}.mel")];
        assert_eq!(mel.shape()[1..], want_mel.shape()[..], "{name}: mel shape");
        let error = relative_error(&floats(&mel), &floats(want_mel));
        println!("{name}: mel {:?}, relative error {error:.2e}", mel.shape());
        assert!(error < 1e-3, "{name}: the mel is {error} off");

        // The encoder, on the reference's mel so that only the encoder is measured.
        let reference_mel = want_mel.unsqueeze(0).unwrap().to_device(DEVICE).unwrap();
        let audio = model.encode(&reference_mel).unwrap();
        let want_audio = &cases[&format!("{name}.audio")];
        assert_eq!(audio.shape(), want_audio.shape(), "{name}: encoder shape");
        let error = relative_error(&floats(&audio), &floats(want_audio));
        println!(
            "{name}: encoder {:?}, relative error {error:.2e}",
            audio.shape()
        );
        assert!(error < 2e-2, "{name}: the encoder is {error} off");

        // Both ways of keeping attention in its windows: the block diagonal mask CUDA's kernel
        // takes, and the windows split into batches every device runs. Each is the reference's,
        // and they are each other's to within what half precision leaves.
        let masked = floats(&model.encode_with(&reference_mel, Windows::Masked).unwrap());
        let split = floats(&model.encode_with(&reference_mel, Windows::Split).unwrap());
        let (masked_error, split_error) = (
            relative_error(&masked, &floats(want_audio)),
            relative_error(&split, &floats(want_audio)),
        );
        let apart = relative_error(&masked, &split);
        println!("{name}: masked {masked_error:.2e}, split {split_error:.2e}, apart {apart:.2e}");
        assert!(masked_error < 2e-2 && split_error < 2e-2, "{name}");
        assert!(apart < 1e-2, "{name}: masked and split are {apart} apart");

        // The prompt is the reference's, around as many placeholders as there are rows.
        let (head, tail) = model.prompt(&TranscriptionOptions::default()).unwrap();
        let rows = audio.shape()[0] as usize;
        let pad = model.config().audio_token_id;
        let mut prompt = head.clone();
        prompt.extend(std::iter::repeat_n(pad, rows));
        prompt.extend(&tail);
        assert_eq!(prompt, ints(&cases[&format!("{name}.prompt")]), "{name}");

        // The decoder, teacher-forced over what the reference said, from its encoder's rows.
        let said = ints(&cases[&format!("{name}.said")]);
        let want_logp = floats(&cases[&format!("{name}.logp")]);
        let reference_audio = want_audio.to_device(DEVICE).unwrap();
        let decoder = model.decoder();
        let mut reading = decoder.prefill(&head, &reference_audio, &tail).unwrap();
        let mut worst = 0f32;
        for (index, token) in said.iter().enumerate() {
            let scores = reading.scores().unwrap();
            let want = &want_logp[index * vocab..(index + 1) * vocab];
            worst = worst.max(logp_error(&log_softmax(&scores), want));
            assert_eq!(
                argmax(&scores) as i32,
                *token,
                "{name}: token {index} is not the one the reference chose"
            );
            if index + 1 < said.len() {
                reading = decoder.step(&reading, *token).unwrap();
            }
        }
        println!(
            "{name}: decoder over {} tokens, log probability error {worst:.3}",
            said.len()
        );
        assert!(worst < 0.5, "{name}: the log probabilities are {worst} off");

        // The whole of it, from the recording: the same tokens and the same words.
        let heard = model
            .hear(&wave, &TranscriptionOptions::default(), &mut |_| {
                ControlFlow::Continue(())
            })
            .unwrap()
            .unwrap();
        assert_eq!(heard.said, said[..said.len() - 1], "{name}: {}", heard.raw);

        let mut reports = Vec::new();
        let transcript = model
            .transcribe(
                &Sound::new(wave.clone(), 16000),
                &TranscriptionOptions::default(),
                &mut |progress| {
                    reports.push(progress);
                    ControlFlow::Continue(())
                },
            )
            .unwrap()
            .unwrap();
        println!("{name}: {} {:?}", transcript.language, transcript.text);
        assert_eq!(transcript.language, *language, "{name}");
        assert_eq!(transcript.text, *text, "{name}");
        assert_eq!(
            reports.first(),
            Some(&TranscriptionProgress::Listening { done: 0, total: 1 })
        );
        assert!(matches!(
            reports.last(),
            Some(TranscriptionProgress::Writing { done, .. }) if *done == said.len() as i32 - 1
        ));
    }

    // Told the language, it writes the words alone; and the name is taken in any case.
    let wave = floats(&cases["en.wave"]);
    let told = TranscriptionOptions {
        language: Some("english".to_string()),
        context: String::new(),
    };
    let transcript = model
        .transcribe(&Sound::new(wave.clone(), 16000), &told, &mut |_| {
            ControlFlow::Continue(())
        })
        .unwrap()
        .unwrap();
    assert_eq!(transcript.language, "English");
    assert_eq!(transcript.text, CASES[2].2);

    // A language it does not know is refused, saying which it does.
    let unknown = TranscriptionOptions {
        language: Some("Klingon".to_string()),
        context: String::new(),
    };
    let refused = model.transcribe(&Sound::new(wave.clone(), 16000), &unknown, &mut |_| {
        ControlFlow::Continue(())
    });
    assert!(refused.unwrap_err().to_string().contains("English"));

    // At another rate, through the resampler: the same words.
    let doubled: Vec<f32> = wave.iter().flat_map(|x| [*x, *x]).collect();
    let transcript = model
        .transcribe(
            &Sound::new(doubled, 32000),
            &TranscriptionOptions::default(),
            &mut |_| ControlFlow::Continue(()),
        )
        .unwrap()
        .unwrap();
    assert_eq!(transcript.text, CASES[2].2);

    // Stopped, it says nothing rather than half of it.
    let stopped = model
        .transcribe(
            &Sound::new(wave, 16000),
            &TranscriptionOptions::default(),
            &mut |progress| match progress {
                TranscriptionProgress::Writing { done: 3, .. } => ControlFlow::Break(()),
                _ => ControlFlow::Continue(()),
            },
        )
        .unwrap();
    assert_eq!(stopped, None);
}

#[test]
#[ignore]
fn the_engine_writes_down_a_recording() {
    use std::sync::mpsc;
    use std::time::Duration;
    use waifu::engine::{Ended, Engine, EngineOptions, Progress, Stage, TranscribeRequest};
    use waifu::runtime::DeviceOption;

    let engine = Engine::new(EngineOptions {
        device: DeviceOption::Cuda,
    })
    .unwrap();
    let wait = Duration::from_secs(120);

    let (send, loaded) = mpsc::channel();
    let manifest = models_dir().join("qwen3-asr.yaml");
    engine.load(
        manifest.to_str().unwrap(),
        Box::new(|_| {}),
        Box::new(move |ended| send.send(ended).unwrap()),
    );
    assert_eq!(loaded.recv_timeout(wait).unwrap(), Ended::Done(()));

    let (send_progress, progress) = mpsc::channel::<Progress>();
    let (send, written) = mpsc::channel();
    let wave = floats(&fixture()["zh.wave"]);
    engine
        .transcribe(
            TranscribeRequest {
                recording: Sound::new(wave, 16000),
                language: None,
                context: String::new(),
            },
            Box::new(move |report| send_progress.send(report.clone()).unwrap()),
            Box::new(move |ended| send.send(ended).unwrap()),
        )
        .unwrap();
    match written.recv_timeout(wait).unwrap() {
        Ended::Done(transcript) => {
            assert_eq!(transcript.text, CASES[0].2);
            assert_eq!(transcript.language, "Chinese");
        }
        other => panic!("{other:?}"),
    }

    // It reads the recording, then generates tokens, and the bar only moves forward.
    let reports: Vec<Progress> = progress.try_iter().collect();
    assert_eq!(
        reports.first().map(|r| r.stage),
        Some(Stage::SpeechToTextReadingRecording)
    );
    assert_eq!(
        reports.last().map(|r| r.stage),
        Some(Stage::SpeechToTextGeneratingTokens)
    );
    let fractions: Vec<f64> = reports.iter().filter_map(|r| r.fraction).collect();
    assert!(
        fractions.windows(2).all(|pair| pair[0] <= pair[1]),
        "{fractions:?}"
    );
    assert!(reports.iter().all(|report| !report.words.is_empty()));
}
