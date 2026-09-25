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

//! CosyVoice3 against upstream, stage by stage, on one real reading.
//!
//! Needs `models/cosyvoice3.yaml` (`tools/cosyvoice3_exporter.py`) and
//! `models/cosyvoice3_test.safetensors` (`tools/cosyvoice3_reference.py`). Every input a stage
//! is handed here is one upstream really handed it: the recording in `asset/`, the tokens its
//! language model said, the mel its flow drew.

use std::collections::HashMap;
use std::path::PathBuf;
use std::rc::Rc;

use std::ops::ControlFlow;
use waifu::cosyvoice3::flow::{self, Flow};
use waifu::cosyvoice3::hift::{Hift, Noise};
use waifu::cosyvoice3::lm::{self, Lm};
use waifu::cosyvoice3::{features, speech_tokenizer};
use waifu::flint::{Graph, Ir, ParamSource, RunContext, Tensor};
use waifu::indextts;
use waifu::indextts::features::resample;

use waifu::cosyvoice3::CosyVoice3;
use waifu::{
    read_safetensors, wav, DType, Device, Manifest, Residency, Sound, SpeechOptions,
    SpeechProgress, Weights,
};

/// What `tools/cosyvoice3_reference.py` read, and the transcript of its recording.
const TEXT: &str =
    "收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐。";
const PROMPT_TEXT: &str = "希望你以后能够做的比我还好呦。";
/// English with a number in it, for the other half of the frontend.
const ENGLISH: &str = "The train leaves at 7 tonight, and the ticket costs 25 dollars.";

const DEVICE: Device = Device::Cuda;

fn models_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../models")
}

fn manifest() -> Manifest {
    Manifest::open(models_dir().join("cosyvoice3.yaml")).unwrap()
}

fn weights(manifest: &Manifest) -> Rc<dyn ParamSource> {
    Rc::new(
        Weights::from_files(&manifest.weight_paths().unwrap(), DEVICE, Residency::Device).unwrap(),
    )
}

fn fixture() -> HashMap<String, Tensor> {
    read_safetensors(&[models_dir().join("cosyvoice3_test.safetensors")]).unwrap()
}

fn floats(cases: &HashMap<String, Tensor>, name: &str) -> Vec<f32> {
    cases[name]
        .cast(DType::Float)
        .unwrap()
        .to_vec_f32()
        .unwrap()
}

fn ints(cases: &HashMap<String, Tensor>, name: &str) -> Vec<i32> {
    cases[name]
        .to_vec_i64()
        .unwrap()
        .into_iter()
        .map(|x| x as i32)
        .collect()
}

/// The largest difference between two rows of log probabilities, over the tokens the reference
/// gives any real chance to; far down the tail both are noise below anything a sampler reaches.
fn logp_error(got: &[f32], want: &[f32]) -> f32 {
    got.iter()
        .zip(want)
        .filter(|(_, w)| **w > -12.0)
        .map(|(g, w)| (g - w).abs())
        .fold(0.0, f32::max)
}

fn argmax(row: &[f32]) -> usize {
    row.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap()
        .0
}

fn max_error(got: &[f32], want: &[f32]) -> f32 {
    assert_eq!(got.len(), want.len());
    got.iter()
        .zip(want)
        .map(|(g, w)| (g - w).abs())
        .fold(0.0, f32::max)
}

fn run(weights: &Rc<dyn ParamSource>, g: &Graph, inputs: &[(&str, &Tensor)]) -> Vec<Tensor> {
    let ir = Ir::compile(g);
    let mut context = RunContext::new(weights.as_ref());
    for (name, tensor) in inputs {
        context = context.input(name, tensor);
    }
    ir.run(&context)
        .unwrap()
        .into_iter()
        .map(|(_, tensor)| tensor)
        .collect()
}

fn upload(shape: &[i32], values: &[f32]) -> Tensor {
    Tensor::from_f32(shape, values)
        .unwrap()
        .to_device(DEVICE)
        .unwrap()
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

/// The recording as the frontend hears it: resampled, both spectrograms, the speech tokens and
/// the speaker.
#[test]
#[ignore]
fn the_frontend_hears_the_recording_as_upstream_does() {
    let manifest = manifest();
    let cases = fixture();
    let weights = weights(&manifest);

    let wave_24k = floats(&cases, "prompt_24k");
    let want_16k = floats(&cases, "prompt_16k");
    let got_16k = resample(&wave_24k, 24000, 16000);
    let error = max_error(&got_16k, &want_16k);
    println!("16 kHz resampling: max error {error:.2e}");
    assert!(error < 1e-4);

    // Whisper's mel, of upstream's own 16 kHz samples.
    let want_mel = floats(&cases, "whisper_mel");
    let (mel, frames) = features::whisper_mel(&want_16k).unwrap();
    let error = max_error(&mel, &want_mel);
    println!("whisper mel: {frames} frames, max error {error:.2e}");
    assert!(error < 1e-3);

    // The 24 kHz mel, frame-major as upstream holds it.
    let want_feat = floats(&cases, "prompt_feat");
    let (feat, feat_frames) = features::speech_mel(&wave_24k).unwrap();
    let transposed: Vec<f32> = (0..feat_frames)
        .flat_map(|frame| (0..80).map(move |band| (frame, band)))
        .map(|(frame, band)| feat[band * feat_frames + frame])
        .collect();
    let error = max_error(&transposed, &want_feat);
    println!("24 kHz mel: {feat_frames} frames, max error {error:.2e}");
    assert!(error < 1e-3);

    // The speech tokens, from upstream's own mel.
    let want_tokens = ints(&cases, "prompt_tokens");
    let out = speech_tokenizer::frames_out(frames as i32);
    let (cos, sin, shape) = speech_tokenizer::rotary(out);
    let g = Graph::new();
    let projected = speech_tokenizer::graph(
        &g.subgraph("cosyvoice3.speech_tokenizer"),
        g.input("mel"),
        g.input("cos"),
        g.input("sin"),
        DType::Float,
        DEVICE,
    )
    .unwrap();
    g.output("projected", projected);
    let outputs = run(
        &weights,
        &g,
        &[
            ("mel", &upload(&[1, 128, frames as i32], &want_mel)),
            ("cos", &upload(&shape, &cos)),
            ("sin", &upload(&shape, &sin)),
        ],
    );
    let tokens = speech_tokenizer::quantize(&host(&outputs[0]));
    let agree = tokens
        .iter()
        .zip(&want_tokens)
        .filter(|(a, b)| a == b)
        .count();
    println!("speech tokens: {agree} of {} agree", want_tokens.len());
    assert_eq!(tokens.len(), want_tokens.len());
    assert!(agree + 2 >= want_tokens.len());

    // CAMPPlus, through IndexTTS's port and the package's own copy of the weights.
    let (fbank, fbank_frames) = indextts::features::campplus(&want_16k);
    let g = Graph::new();
    let speaker = indextts::campplus::graph(
        &g.subgraph("cosyvoice3.campplus"),
        g.input("x"),
        &indextts::campplus::Config::indextts(),
        fbank_frames as i32,
        DType::Float,
        DEVICE,
    )
    .unwrap();
    g.output("speaker", speaker);
    let outputs = run(
        &weights,
        &g,
        &[("x", &upload(&[1, fbank_frames as i32, 80], &fbank))],
    );
    let error = max_error(&host(&outputs[0]), &floats(&cases, "speaker"));
    println!("speaker embedding: max error {error:.2e}");
    assert!(error < 2e-3);
}

/// Teacher-forced over the 231 tokens upstream said, through the prefill and then every step:
/// each row of log probabilities against upstream's, and the text as the tokenizer reads it.
#[test]
#[ignore]
fn the_language_model_scores_what_upstream_said_as_upstream_does() {
    let manifest = manifest();
    let cases = fixture();
    let config = lm::Config::cosyvoice3();
    let lm = Lm::build(
        config,
        "cosyvoice3.lm",
        &weights(&manifest),
        DType::Float,
        DEVICE,
    )
    .unwrap();

    let text = ints(&cases, "text_ids");
    let said = ints(&cases, "said");
    let want = floats(&cases, "lm_logp");
    let rows = config.speech_rows as usize;
    assert_eq!(want.len(), (said.len() + 1) * rows);

    let mut reading = lm.prefill(&text, &[]).unwrap();
    let mut worst = 0.0f32;
    let mut disagreements = 0;
    for (index, token) in said.iter().chain(std::iter::once(&-1)).enumerate() {
        let got = reading.log_probabilities().unwrap();
        let row = &want[index * rows..(index + 1) * rows];
        worst = worst.max(logp_error(&got, row));
        if argmax(&got) != argmax(row) {
            disagreements += 1;
        }
        if *token >= 0 {
            reading = lm.step(&reading, *token).unwrap();
        }
    }
    println!(
        "{} rows: worst log probability error {worst:.2e}, {disagreements} argmax disagreements",
        said.len() + 1
    );
    assert!(worst < 2e-2, "log probabilities drift by {worst}");
    assert!(
        disagreements <= 2,
        "{disagreements} rows pick a different likeliest token"
    );

    // The zero-shot prefix: transcript and prompt tokens, one row.
    let ids = ints(&cases, "zero_shot_ids");
    let prompt = ints(&cases, "flow_prompt_tokens");
    let reading = lm.prefill(&ids, &prompt).unwrap();
    let got = reading.log_probabilities().unwrap();
    let error = logp_error(&got, &floats(&cases, "zero_shot_logp"));
    println!("zero-shot prefix: error {error:.2e}");
    assert!(error < 2e-2);
}

/// `(C, T)` channel-major as `(T, C)` frame-major.
fn frame_major(values: &[f32], channels: usize) -> Vec<f32> {
    let frames = values.len() / channels;
    (0..frames)
        .flat_map(|t| (0..channels).map(move |c| c * frames + t))
        .map(|index| values[index])
        .collect()
}

/// The flow on the tokens upstream said: its condition, one call of the estimator against
/// upstream's first, and the whole ten-step mel.
#[test]
#[ignore]
fn the_flow_draws_the_mel_upstream_draws() {
    let manifest = manifest();
    let cases = fixture();
    let config = flow::Config::cosyvoice3();
    let flow = Flow::build(
        config,
        "cosyvoice3.flow",
        &weights(&manifest),
        DType::Float,
        DEVICE,
    )
    .unwrap();

    let prompt = ints(&cases, "flow_prompt_tokens");
    let said = ints(&cases, "said");
    let prompt_mel = floats(&cases, "flow_prompt_feat");
    let prompt_frames = (prompt_mel.len() / 80) as i32;
    assert_eq!(prompt_frames, 2 * prompt.len() as i32);

    let condition = flow
        .condition(&prompt, &said, &floats(&cases, "speaker"))
        .unwrap();
    let want_mu = frame_major(&floats(&cases, "flow_mu"), 80);
    let error = max_error(condition.mu(), &want_mu);
    println!("mu: {} frames, max error {error:.2e}", condition.frames());
    assert!(error < 1e-3);

    // The estimator's first call: the same noise in both halves, t = 0.
    let x = floats(&cases, "dit_x");
    let x: Vec<f32> = x
        .chunks(x.len() / 2)
        .flat_map(|half| frame_major(half, 80))
        .collect();
    let context = flow
        .context_tensor(&condition, &prompt_mel, prompt_frames)
        .unwrap();
    let rotary = flow.rotary_tables(condition.frames()).unwrap();
    let t = floats(&cases, "dit_t")[0];
    let velocity = flow.velocity(&x, &context, t, &rotary).unwrap();
    let want = floats(&cases, "dit_velocity");
    let want: Vec<f32> = want
        .chunks(want.len() / 2)
        .flat_map(|half| frame_major(half, 80))
        .collect();
    let error = max_error(&velocity, &want);
    let scale = want.iter().map(|v| v.abs()).fold(0.0, f32::max);
    println!("first velocity: max error {error:.2e} of {scale:.2}");
    assert!(error < 5e-3 * scale.max(1.0));

    let mel = flow
        .draw(&condition, &prompt_mel, prompt_frames, 10, 0.7)
        .unwrap();
    let want = frame_major(&floats(&cases, "mel"), 80);
    let error = max_error(&mel, &want);
    let mean: f32 = mel
        .iter()
        .zip(&want)
        .map(|(a, b)| (a - b).abs())
        .sum::<f32>()
        / mel.len() as f32;
    println!(
        "mel: {} frames, max error {error:.2e}, mean {mean:.2e}",
        mel.len() / 80
    );
    assert!(mean < 1e-2 && error < 0.2);
}

/// The vocoder on the mel upstream's flow drew, with the noise upstream's source was handed: the
/// pitch, the excitation, and the waveform.
#[test]
#[ignore]
fn the_vocoder_sounds_the_mel_as_upstream_does() {
    let manifest = manifest();
    let cases = fixture();
    let hift = Hift::build("cosyvoice3.hift", &weights(&manifest), DType::Float, DEVICE).unwrap();

    let mel = floats(&cases, "mel");
    let frames = mel.len() / 80;
    let noise = Noise {
        start: floats(&cases, "hift_rand_ini"),
        sine: floats(&cases, "hift_sine_noise"),
    };

    let f0 = hift.pitch(&mel, frames).unwrap();
    let want_f0 = floats(&cases, "f0");
    let error = max_error(&f0, &want_f0);
    println!("pitch: {frames} frames, max error {error:.2e} Hz");
    assert!(error < 0.5);

    // The excitation from upstream's own pitch, so that this measures the source alone.
    let excitation = hift.source(&want_f0, &noise).unwrap();
    let want_source = floats(&cases, "source");
    let error = max_error(&excitation, &want_source);
    println!(
        "source: {} samples, max error {error:.2e}",
        excitation.len()
    );
    assert!(error < 1e-3);

    // The filter over upstream's excitation, and then the whole of it from the mel.
    let want_wave = floats(&cases, "wave");
    let wave = hift.decode(&mel, frames, &want_source).unwrap();
    let error = max_error(&wave, &want_wave);
    println!("filter: {} samples, max error {error:.2e}", wave.len());
    assert!(error < 1e-2);

    let wave = hift.forward(&mel, frames, &noise).unwrap();
    let difference: f32 = wave
        .iter()
        .zip(&want_wave)
        .map(|(a, b)| (a - b) * (a - b))
        .sum();
    let signal: f32 = want_wave.iter().map(|b| b * b).sum();
    let snr = 10.0 * (signal / difference).log10();
    println!(
        "whole vocoder: {} samples, {snr:.1} dB against upstream",
        wave.len()
    );
    assert!(snr > 30.0);
}

/// The whole pipeline through `from_manifest`: what `listen` hears against upstream's frontend,
/// then the sentence said both ways -- as [`Voice::speak`] says it, with no transcript, and
/// zero-shot after one. Each is written to `models/` beside the fixture for a person, or Whisper,
/// to listen to.
#[test]
#[ignore]
fn says_the_sentence_in_the_voice_of_the_recording() {
    let cases = fixture();
    let tts = CosyVoice3::from_manifest(DEVICE, Residency::Device, &manifest()).unwrap();
    let recording = Sound::new(floats(&cases, "prompt_24k"), 24000);

    let heard = tts.listen(&recording).unwrap();
    assert_eq!(heard.tokens, ints(&cases, "flow_prompt_tokens"));
    let error = max_error(&heard.mel, &floats(&cases, "flow_prompt_feat"));
    assert!(error < 1e-3, "the prompt's mel is off by {error}");

    let options = SpeechOptions {
        speed: 1.0,
        temperature: 1.0,
        seed: Some(7),
    };
    let mut carry_on = |_: SpeechProgress| ControlFlow::Continue(());

    for (name, said) in [
        (
            "cosyvoice3-cross-lingual.wav",
            tts.say(TEXT, &heard, &options, &mut carry_on),
        ),
        (
            "cosyvoice3-zero-shot.wav",
            tts.say_after(TEXT, PROMPT_TEXT, &heard, &options, &mut carry_on),
        ),
        (
            "cosyvoice3-english.wav",
            tts.say(ENGLISH, &heard, &options, &mut carry_on),
        ),
    ] {
        let said = said.unwrap().expect("nothing stops this reading");
        assert_eq!(said.rate, 24000);
        let seconds = said.seconds();
        let rms =
            (said.samples.iter().map(|s| s * s).sum::<f32>() / said.samples.len() as f32).sqrt();
        println!("{name}: {seconds:.2} s, rms {rms:.4}");
        // Forty Chinese characters take upstream nine seconds and the English about four; well
        // outside that is a model that stopped early or never stopped.
        assert!((2.5..20.0).contains(&seconds));
        assert!(rms > 0.01);
        std::fs::write(models_dir().join(name), wav::write(&said)).unwrap();
    }

    // The same seed, the same reading.
    let again = tts
        .say(TEXT, &heard, &options, &mut carry_on)
        .unwrap()
        .unwrap();
    let first = tts
        .say(TEXT, &heard, &options, &mut carry_on)
        .unwrap()
        .unwrap();
    assert_eq!(again.samples, first.samples);
}

/// How long the language model's readings run, over eight seeds, against upstream's over eight
/// of its own. One draw says nothing about a sampler; the length of many is what a sampler that
/// struck out the wrong tokens, or stopped at the wrong ones, gets wrong first.
#[test]
#[ignore]
fn the_sampler_reads_about_as_long_as_upstream() {
    let manifest = manifest();
    let cases = fixture();
    let config = lm::Config::cosyvoice3();
    let lm = Lm::build(
        config,
        "cosyvoice3.lm",
        &weights(&manifest),
        DType::Float,
        DEVICE,
    )
    .unwrap();
    let text = ints(&cases, "text_ids");

    let lengths: Vec<usize> = (1..=8u64)
        .map(|seed| {
            let mut sampler = lm::Sampler::new(seed);
            lm.generate(
                &text,
                &[],
                text.len(),
                &lm::Sampling::cosyvoice3(),
                &mut sampler,
                &mut |_| ControlFlow::Continue(()),
            )
            .unwrap()
            .unwrap()
            .len()
        })
        .collect();
    let mean = lengths.iter().sum::<usize>() as f32 / lengths.len() as f32;
    println!("lengths {lengths:?}, mean {mean:.1}");

    // Upstream's `llm.inference` on the same ids, seeds 1 to 8 under `set_all_random_seed`:
    // [247, 208, 253, 244, 158, 205, 198, 182], mean 211.9 and deviation about 31. Eight draws
    // each put the two means within one deviation of each other by a wide margin.
    assert!((180.0..245.0).contains(&mean), "a mean of {mean} tokens");
}
