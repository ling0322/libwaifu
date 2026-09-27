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

//! Seed-VC v2 against upstream, stage by stage, on one real conversion.
//!
//! Needs `models/seed_vc.yaml` (`tools/seed_vc_exporter.py`) and `models/seed_vc_test.safetensors`
//! (`tools/seed_vc_reference.py`). Every input a stage is handed here is one upstream really
//! handed it: the recordings as upstream resampled them, the tokens its quantizers gave, the mel
//! its trajectory was at, the tokens its AR said.

use std::collections::HashMap;
use std::ops::ControlFlow;
use std::path::PathBuf;

use waifu::flint::Tensor;
use waifu::seed_vc::{astral, dit, Conversion, Reference, SeedVc};
use waifu::{read_safetensors, wav, DType, Device, Manifest, Residency, Sound};

const DEVICE: Device = Device::Cuda;

fn models_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../models")
}

fn model() -> SeedVc {
    let manifest = Manifest::open(models_dir().join("seed_vc.yaml")).unwrap();
    SeedVc::from_manifest(DEVICE, Residency::Device, &manifest).unwrap()
}

fn fixture() -> HashMap<String, Tensor> {
    read_safetensors(&[models_dir().join("seed_vc_test.safetensors")]).unwrap()
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

fn host(tensor: &Tensor) -> Vec<f32> {
    tensor
        .to_device(Device::Cpu)
        .unwrap()
        .cast(DType::Float)
        .unwrap()
        .to_vec_f32()
        .unwrap()
}

fn upload(shape: &[i32], values: &[f32]) -> Tensor {
    Tensor::from_f32(shape, values)
        .unwrap()
        .to_device(DEVICE)
        .unwrap()
}

fn max_error(got: &[f32], want: &[f32]) -> f32 {
    assert_eq!(got.len(), want.len());
    got.iter()
        .zip(want)
        .map(|(g, w)| (g - w).abs())
        .fold(0.0, f32::max)
}

/// The largest difference relative to the largest value upstream has, which is what makes an
/// error in a residual stream that runs into the hundreds comparable with one in a mel.
fn relative_error(got: &[f32], want: &[f32]) -> f32 {
    let scale = want.iter().map(|w| w.abs()).fold(0.0, f32::max);
    max_error(got, want) / scale
}

fn snr(got: &[f32], want: &[f32]) -> f32 {
    let noise: f64 = got
        .iter()
        .zip(want)
        .map(|(g, w)| f64::from(g - w).powi(2))
        .sum();
    let signal: f64 = want.iter().map(|w| f64::from(*w).powi(2)).sum();
    (10.0 * (signal / noise).log10()) as f32
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm = |v: &[f32]| v.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (norm(a) * norm(b))
}

fn agreement(got: &[i32], want: &[i32]) -> f32 {
    assert_eq!(got.len(), want.len());
    got.iter().zip(want).filter(|(g, w)| g == w).count() as f32 / want.len() as f32
}

/// The reference as upstream held it: its mel, its style and its tokens are upstream's own, so a
/// stage after them is measured alone.
fn upstream_reference(vc: &SeedVc, cases: &HashMap<String, Tensor>) -> Reference {
    let mel = floats(cases, "reference_mel");
    let frames = (mel.len() / 80) as i32;
    let wide = ints(cases, "wide_reference");
    Reference {
        prompt_condition: vc.regulate(&wide, frames).unwrap(),
        wide,
        narrow: ints(cases, "narrow_reference"),
        style: upload(&[1, 192], &floats(cases, "style")),
        mel,
        frames,
    }
}

/// HuBERT, both quantizers, CAMPPlus and the mel, on the samples upstream heard.
#[test]
#[ignore]
fn hears_the_recordings_as_upstream_does() {
    let vc = model();
    let cases = fixture();

    let heard = vc.hear(&floats(&cases, "source_16k")).unwrap();
    let error = relative_error(&host(&heard.hubert), &floats(&cases, "hubert"));
    println!(
        "hubert: {:?}, max error {error:.2e} of the largest value",
        heard.hubert.shape()
    );
    assert!(error < 5e-5);

    for (name, got, bits) in [
        ("wide", &heard.wide_projected, astral::WIDE_BITS),
        ("narrow", &heard.narrow_projected, astral::NARROW_BITS),
    ] {
        let want = floats(&cases, &format!("{name}_projected"));
        let error = max_error(got, &want);
        let tokens = agreement(
            &astral::tokens(got, bits),
            &ints(&cases, &format!("{name}_source")),
        );
        println!(
            "{name}: projection max error {error:.2e}, {:.1}% of tokens agree",
            tokens * 100.0
        );
        assert!(error < 5e-3);
        assert!(tokens > 0.99);
    }

    let reference = vc.hear(&floats(&cases, "reference_16k")).unwrap();
    let wide = agreement(&reference.wide, &ints(&cases, "wide_reference"));
    let narrow = agreement(&reference.narrow, &ints(&cases, "narrow_reference"));
    println!(
        "reference: {:.1}% wide, {:.1}% narrow agree",
        wide * 100.0,
        narrow * 100.0
    );
    assert!(wide > 0.99 && narrow > 0.99);

    let style = host(&vc.style(&floats(&cases, "reference_16k")).unwrap());
    let error = max_error(&style, &floats(&cases, "style"));
    println!("style: max error {error:.2e}");
    assert!(error < 1e-4);

    let (mel, frames) =
        waifu::indextts::features::reference_mel(&floats(&cases, "reference_22k")).unwrap();
    let error = max_error(&mel, &floats(&cases, "reference_mel"));
    println!("reference mel: {frames} frames, max error {error:.2e}");
    assert!(error < 1e-3);
}

/// The CFM's length regulator, on the reference's and the source's upstream tokens.
#[test]
#[ignore]
fn lays_tokens_onto_frames_as_upstream_does() {
    let vc = model();
    let cases = fixture();

    for (tokens, want) in [
        ("wide_reference", "prompt_condition"),
        ("wide_source", "condition"),
    ] {
        let want = floats(&cases, want);
        let frames = (want.len() / dit::CONDITION_DIM as usize) as i32;
        let got = host(&vc.regulate(&ints(&cases, tokens), frames).unwrap());
        let error = relative_error(&got, &want);
        println!("{tokens}: {frames} frames, max error {error:.2e} of the largest value");
        assert!(error < 5e-5);
    }
}

/// One call of the DiT from the middle of upstream's trajectory: the batch of three it guides with.
#[test]
#[ignore]
fn the_dit_points_where_upstream_points() {
    let vc = model();
    let cases = fixture();

    let x = floats(&cases, "dit_x");
    let frames = (x.len() / (3 * 80)) as i32;
    let denoiser = vc.denoiser(frames, 3).unwrap();
    let direction = denoiser
        .direction(
            &upload(&[3, 80, frames], &x),
            &upload(&[3, 80, frames], &floats(&cases, "dit_prompt_x")),
            &upload(&[3, frames, 512], &floats(&cases, "dit_cond")),
            &upload(&[3, 192], &floats(&cases, "dit_style")),
            &floats(&cases, "dit_t"),
        )
        .unwrap();

    let got = host(&direction);
    let want = floats(&cases, "dit_out");
    let row = got.len() / 3;
    for (index, name) in ["everything", "content only", "nothing"].iter().enumerate() {
        let range = index * row..(index + 1) * row;
        let error = relative_error(&got[range.clone()], &want[range]);
        println!("{name}: max error {error:.2e} of the largest value");
        assert!(error < 2e-5);
    }
}

/// Thirty guided steps from upstream's own noise, with upstream's condition, style and prompt.
#[test]
#[ignore]
fn the_cfm_draws_the_mel_upstream_draws() {
    let vc = model();
    let cases = fixture();
    let reference = upstream_reference(&vc, &cases);

    let mut condition = floats(&cases, "prompt_condition");
    condition.extend(floats(&cases, "condition"));
    let frames = (condition.len() / 512) as i32;
    let condition = upload(&[1, frames, 512], &condition);

    let guidance = vc.settings().guidance;
    let mel = vc
        .draw_mel(
            &floats(&cases, "cfm_noise"),
            &condition,
            &reference,
            30,
            guidance,
        )
        .unwrap();
    let want = floats(&cases, "cfm_mel");

    // Past the prompt, which is zero on both sides.
    let prompt = reference.frames as usize;
    let drawn = |values: &[f32]| -> Vec<f32> {
        (0..80)
            .flat_map(|channel| {
                values[channel * frames as usize + prompt..(channel + 1) * frames as usize].to_vec()
            })
            .collect()
    };
    let (got, want) = (drawn(&mel), drawn(&want));
    let error = max_error(&got, &want);
    let mean = got
        .iter()
        .zip(&want)
        .map(|(g, w)| (g - w).abs())
        .sum::<f32>()
        / got.len() as f32;
    println!(
        "mel: {} frames drawn, max error {error:.2e}, mean {mean:.2e}",
        got.len() / 80
    );
    assert!(mean < 2e-5);
    assert!(error < 1e-3);
}

/// Upstream's AR, teacher-forced over what it said: its scores at each of the first steps.
#[test]
#[ignore]
fn the_ar_scores_what_upstream_said_as_upstream_does() {
    let vc = model();
    let cases = fixture();
    let ar = vc.ar();

    let mut narrow = astral::reduce_durations(&ints(&cases, "narrow_reference"));
    narrow.extend(astral::reduce_durations(&ints(&cases, "narrow_source")));
    let condition = floats(&cases, "ar_condition");
    assert_eq!(
        narrow.len(),
        condition.len() / 768,
        "the reduced narrow tokens are upstream's"
    );

    let said = ints(&cases, "ar_said");
    let want = floats(&cases, "ar_logits");
    let vocab = ar.config().vocab_size as usize;
    let steps = want.len() / vocab;

    let mut reading = ar.prefill(&narrow, &ints(&cases, "ar_prompt")).unwrap();
    let mut worst = 0.0f32;
    let mut agree = 0;
    for step in 0..steps {
        let got = reading.logits().unwrap();
        let row = &want[step * vocab..(step + 1) * vocab];
        worst = worst.max(max_error(&got, row));
        let best = |values: &[f32]| {
            (0..vocab)
                .max_by(|a, b| values[*a].total_cmp(&values[*b]))
                .unwrap()
        };
        agree += usize::from(best(&got) == best(row));

        if step + 1 < steps {
            reading = ar.step(&reading, said[step]).unwrap();
        }
    }
    println!("ar: {steps} steps, max score error {worst:.2e}, {agree} of {steps} argmaxes agree");
    assert!(worst < 2e-4);
    assert_eq!(agree, steps);
}

/// BigVGAN on the last mel upstream sounded.
#[test]
#[ignore]
fn the_vocoder_sounds_the_mel_as_upstream_does() {
    let vc = model();
    let cases = fixture();

    let mel = floats(&cases, "vocoder_mel");
    let frames = (mel.len() / 80) as i32;
    let wave = host(
        &vc.vocoder()
            .forward(&upload(&[1, 80, frames], &mel))
            .unwrap(),
    );
    let snr = snr(&wave, &floats(&cases, "vocoder_wave"));
    println!(
        "vocoder: {} samples, {snr:.1} dB against upstream",
        wave.len()
    );
    assert!(snr > 60.0);
}

/// Both paths through `convert`, from the recordings on the disk. Written beside the fixture for
/// a person, or Whisper, to listen to; checked here by who CAMPPlus says is speaking.
#[test]
#[ignore]
fn converts_the_source_into_the_reference_voice() {
    let vc = model();
    let read = |name: &str| wav::read(&std::fs::read(models_dir().join(name)).unwrap()).unwrap();
    let source = read("seed_vc_test_source.wav");
    let reference = read("seed_vc_test_reference.wav");
    let voice = vc.listen(&reference).unwrap();

    let listen = |sound: &Sound| {
        let wave = waifu::indextts::features::resample(&sound.samples, sound.rate, 16000);
        host(&vc.style(&wave).unwrap())
    };
    let source_style = listen(&source);
    let reference_style = listen(&reference);
    println!(
        "source against reference: {:.2}",
        cosine(&source_style, &reference_style)
    );
    // What upstream's own conversions score, by the same measure, for comparison.
    for name in ["timbre_wave", "style_wave"] {
        let upstream = listen(&read(&format!("seed_vc_test_{name}.wav")));
        println!(
            "upstream {name}: like the reference {:.2}, like the source {:.2}",
            cosine(&upstream, &reference_style),
            cosine(&upstream, &source_style)
        );
    }

    for convert_style in [false, true] {
        let conversion = Conversion {
            convert_style,
            seed: Some(1234),
            ..vc.conversion()
        };
        let started = std::time::Instant::now();
        let out = vc
            .convert(&source, &voice, &conversion, &mut |_| {
                ControlFlow::Continue(())
            })
            .unwrap()
            .unwrap();
        let elapsed = started.elapsed().as_secs_f32();

        let name = if convert_style { "style" } else { "timbre" };
        std::fs::write(
            models_dir().join(format!("seed_vc_test_{name}_waifu.wav")),
            wav::write(&out),
        )
        .unwrap();

        let style = listen(&out);
        let like_reference = cosine(&style, &reference_style);
        let like_source = cosine(&style, &source_style);
        println!(
            "{name}: {:.2} s in {elapsed:.1} s; like the reference {like_reference:.2}, \
             like the source {like_source:.2}",
            out.seconds()
        );

        assert!(out.samples.iter().all(|x| x.is_finite()));
        assert!((out.seconds() - source.seconds()).abs() < 1.5 || convert_style);
        assert!(like_reference > like_source);
        assert!(like_reference > 0.5);
    }
}
