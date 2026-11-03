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

//! A CosyVoice3 package against another of the same release at another precision.
//!
//! ```text
//! cargo run --release -p waifu-tools --bin cosyvoice3_precision -- \
//!     models/cosyvoice3.yaml models/cosyvoice3-half.yaml voice.wav out
//! ```
//!
//! The first package is the reference: float32 throughout, which the reference tests check against
//! upstream. The second is read as its weights say -- the language model and the DiT in half where
//! it was exported with `-half` -- and each of those two is asked what the reference was asked, on
//! the same inputs:
//!
//! - the language model, teacher-forced over what the reference said: how far its log
//!   probabilities move, and how often its likeliest token is the reference's; then left to
//!   sample at the same seed, how long it says the same thing;
//! - the flow, drawing the mel of what the reference said, the voice the recording's;
//! - the whole reading, both mels through the reference's vocoder on one draw of its noise: how
//!   far apart the two waves are, and how near each is the recording's voice by CAMPPlus.
//!
//! `out.32.wav` and `out.16.wav` are the two readings of the first sentence.

use std::ops::ControlFlow;
use std::rc::Rc;
use std::time::Instant;

use waifu::cosyvoice3::flow::{self, Flow};
use waifu::cosyvoice3::hift::{self, Hift};
use waifu::cosyvoice3::lm::{self, Lm, Sampler, Sampling};
use waifu::cosyvoice3::CosyVoice3;
use waifu::flint::{functional as F, DType, Tensor};
use waifu::{wav, Device, Manifest, Sound, Tokenizer, Weights};

const DEVICE: Device = Device::Cuda;
const SEED: u64 = 1986;
const SYSTEM: &str = "You are a helpful assistant.<|endofprompt|>";
const SENTENCES: &[&str] = &[
    "收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。",
    "The train leaves at 7 tonight, and the ticket costs $25.",
];

/// One package's language model, flow and vocoder, each at the type its weights are stored in.
struct Models {
    lm: Lm,
    flow: Flow,
    hift: Hift,
}

/// One package's models, each at the type its weights are stored in -- read, and every weight
/// checked, by `CosyVoice3::precision`, as `from_manifest` reads it, so that a package the runtime
/// would refuse is refused here too rather than compared.
fn models(manifest: &Manifest) -> Result<Models, Box<dyn std::error::Error>> {
    let weights = Rc::new(Weights::from_files(
        &manifest.weight_paths()?,
        DEVICE,
        u64::MAX,
    )?);
    let precision = CosyVoice3::precision(&weights)?;
    eprintln!(
        "  language model {:?}, DiT {:?}",
        precision.lm, precision.flow
    );
    Ok(Models {
        lm: Lm::build(
            lm::Config::cosyvoice3(),
            "cosyvoice3.lm",
            &weights,
            precision.lm,
            DEVICE,
        )?,
        flow: Flow::build(
            flow::Config::cosyvoice3(),
            "cosyvoice3.flow",
            &weights,
            precision.flow,
            DEVICE,
        )?,
        hift: Hift::build("cosyvoice3.hift", &weights, DType::Float, DEVICE)?,
    })
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

/// `mel` `(1, frames, 80)` to samples, through `hift` on `noise`.
fn vocode(hift: &Hift, mel: &Tensor, noise: &Tensor) -> Vec<f32> {
    let mel = mel.transpose(1, 2).unwrap().contiguous().unwrap();
    host(&hift.forward(&mel, noise).unwrap())
}

/// The log-spectral distance between two waves, in dB: the root mean square over frequency of the
/// difference of their power spectra in decibels, averaged over frames of 1024 at a hop of 256.
/// Blind to phase, which is what makes it the measure here: HiFT's excitation is the integral of
/// its pitch, so a mel a hair apart drifts out of phase within a second and two waves that sound
/// the same are uncorrelated sample by sample.
fn log_spectral_distance(a: &[f32], b: &[f32]) -> f64 {
    const N: usize = 1024;
    const HOP: usize = 256;
    let window: Vec<f64> = (0..N)
        .map(|i| 0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / N as f64).cos())
        .collect();
    let cos: Vec<f64> = (0..N)
        .map(|i| (2.0 * std::f64::consts::PI * i as f64 / N as f64).cos())
        .collect();
    let sin: Vec<f64> = (0..N)
        .map(|i| (2.0 * std::f64::consts::PI * i as f64 / N as f64).sin())
        .collect();
    let power = |frame: &[f32]| -> Vec<f64> {
        (0..=N / 2)
            .map(|k| {
                let (mut re, mut im) = (0.0, 0.0);
                for (n, x) in frame.iter().enumerate() {
                    let at = (k * n) % N;
                    let v = *x as f64 * window[n];
                    re += v * cos[at];
                    im -= v * sin[at];
                }
                10.0 * (re * re + im * im + 1e-10).log10()
            })
            .collect()
    };
    let length = a.len().min(b.len());
    let mut total = 0.0;
    let mut frames = 0;
    let mut start = 0;
    while start + N <= length {
        let (pa, pb) = (power(&a[start..start + N]), power(&b[start..start + N]));
        let square: f64 = pa.iter().zip(&pb).map(|(x, y)| (x - y).powi(2)).sum();
        total += (square / pa.len() as f64).sqrt();
        frames += 1;
        start += HOP;
    }
    total / frames.max(1) as f64
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm = |v: &[f32]| v.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (norm(a) * norm(b))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let arguments: Vec<String> = std::env::args().collect();
    if arguments.len() < 5 {
        eprintln!("usage: cosyvoice3_precision <reference.yaml> <other.yaml> <voice.wav> <out>");
        std::process::exit(2);
    }
    let reference_manifest = Manifest::open(&arguments[1])?;
    let other_manifest = Manifest::open(&arguments[2])?;
    let recording = wav::read(&std::fs::read(&arguments[3])?)?;
    let out = &arguments[4];

    // The recording heard once, by the reference, whose frontend both packages hold in float32.
    let tts = CosyVoice3::from_manifest(DEVICE, u64::MAX, &reference_manifest)?;
    let voice = tts.listen(&recording)?;
    let tokenizer = Tokenizer::open(&reference_manifest)?;
    eprintln!("reference:");
    let a = models(&reference_manifest)?;
    eprintln!("other:");
    let b = models(&other_manifest)?;

    let sampling = Sampling::cosyvoice3();
    let mut quiet = |_| ControlFlow::Continue(());
    for (index, sentence) in SENTENCES.iter().enumerate() {
        println!("\n== sentence {index}: {sentence}");
        let ids = tokenizer.encode(&format!("{SYSTEM}{sentence}"))?;

        // What the reference says, and how long each language model takes to say it.
        let clock = Instant::now();
        let said =
            a.lm.generate(
                &ids,
                &[],
                ids.len(),
                &sampling,
                &mut Sampler::new(SEED),
                &mut quiet,
            )?
            .expect("nothing stops it");
        let reference_ms = clock.elapsed().as_secs_f64() * 1e3 / said.len() as f64;
        let clock = Instant::now();
        let own =
            b.lm.generate(
                &ids,
                &[],
                ids.len(),
                &sampling,
                &mut Sampler::new(SEED),
                &mut quiet,
            )?
            .expect("nothing stops it");
        let other_ms = clock.elapsed().as_secs_f64() * 1e3 / own.len() as f64;
        let same = said.iter().zip(&own).take_while(|(x, y)| x == y).count();
        println!(
            "language model, sampled at seed {SEED}: {} tokens and {} tokens, the first {same} the \
             same; {reference_ms:.1} ms and {other_ms:.1} ms a token",
            said.len(),
            own.len()
        );

        // Both teacher-forced over what the reference said.
        let mut ra = a.lm.prefill(&ids, &[])?;
        let mut rb = b.lm.prefill(&ids, &[])?;
        let (mut worst, mut worst_chosen, mut chosen_sum, mut agree) = (0f32, 0f32, 0f64, 0usize);
        for (step, token) in said.iter().enumerate() {
            let pa = ra.log_probabilities()?;
            let pb = rb.log_probabilities()?;
            for (x, y) in pa.iter().zip(&pb) {
                // Only where the reference gives the token a chance worth sampling.
                if *x > -20.0 {
                    worst = worst.max((x - y).abs());
                }
            }
            let at = *token as usize;
            let chosen = (pa[at] - pb[at]).abs();
            worst_chosen = worst_chosen.max(chosen);
            chosen_sum += chosen as f64;
            let argmax = |p: &[f32]| (0..p.len()).max_by(|&i, &j| p[i].total_cmp(&p[j])).unwrap();
            agree += (argmax(&pa) == argmax(&pb)) as usize;
            if step + 1 < said.len() {
                ra = a.lm.step(&ra, *token)?;
                rb = b.lm.step(&rb, *token)?;
            }
        }
        println!(
            "language model, teacher-forced over {} tokens: log probability within {worst:.4} \
             (above -20); of the token said, {worst_chosen:.4} at most and {:.5} on average; \
             the same likeliest token in {agree} of {}",
            said.len(),
            chosen_sum / said.len() as f64,
            said.len()
        );

        // The flow on what the reference said, in the recording's voice.
        let ca = a.flow.condition(&voice.tokens, &said, &voice.speaker)?;
        let cb = b.flow.condition(&voice.tokens, &said, &voice.speaker)?;
        let steps = 10;
        let clock = Instant::now();
        let ma = a.flow.draw(&ca, &voice.mel, steps, 0.7)?;
        let va = host(&ma);
        let flow_a = clock.elapsed().as_secs_f64() * 1e3;
        let clock = Instant::now();
        let mb = b.flow.draw(&cb, &voice.mel, steps, 0.7)?;
        let vb = host(&mb);
        let flow_b = clock.elapsed().as_secs_f64() * 1e3;
        let (max, mean) = va.iter().zip(&vb).fold((0f32, 0f64), |(m, s), (x, y)| {
            (m.max((x - y).abs()), s + (x - y).abs() as f64)
        });
        let spread = va.iter().fold(0f32, |m, x| m.max(x.abs()));
        println!(
            "flow, {} frames: mel within {max:.4} at most and {:.5} on average, of values up to \
             {spread:.2}; {flow_a:.1} ms and {flow_b:.1} ms",
            va.len() / 80,
            mean / va.len() as f64
        );

        // Both mels through the reference's vocoder, on one draw of its noise.
        let frames = ma.shape_at(1)?;
        F::manual_seed(DEVICE, SEED)?;
        let noise = F::rand(
            &[1, frames * hift::HOP as i32, hift::HARMONICS as i32],
            DType::Float,
            DEVICE,
        )?;
        let wa = vocode(&a.hift, &ma, &noise);
        let wb = vocode(&a.hift, &mb, &noise);
        // The yardstick: the reference's own mel again, on another draw of the vocoder's noise --
        // how far apart one mel sounds from itself, which the other's has to be measured against.
        F::manual_seed(DEVICE, SEED + 1)?;
        let other_noise = F::rand(
            &[1, frames * hift::HOP as i32, hift::HARMONICS as i32],
            DType::Float,
            DEVICE,
        )?;
        let wa_again = vocode(&a.hift, &ma, &other_noise);
        let between = log_spectral_distance(&wa, &wb);
        let itself = log_spectral_distance(&wa, &wa_again);
        let sa = Sound::new(wa, hift::RATE);
        let sb = Sound::new(wb, hift::RATE);
        let sa_again = Sound::new(wa_again, hift::RATE);
        let (ha, hb, ha_again) = (tts.listen(&sa)?, tts.listen(&sb)?, tts.listen(&sa_again)?);
        let speaker = host(&voice.speaker);
        let like_a = cosine(&host(&ha.speaker), &speaker);
        let like_b = cosine(&host(&hb.speaker), &speaker);
        // The waves heard again as 80-band mels, coarse enough that a harmonic a hair away is in
        // the same band: what the two readings sound like rather than where their partials are.
        let heard = |x: &waifu::cosyvoice3::Reference, y: &waifu::cosyvoice3::Reference| -> f64 {
            let (x, y) = (host(&x.mel), host(&y.mel));
            let n = x.len().min(y.len());
            x[..n]
                .iter()
                .zip(&y[..n])
                .map(|(p, q)| (p - q).abs() as f64)
                .sum::<f64>()
                / n as f64
        };
        println!(
            "the reading, {:.2} s: heard again as mels, the other's {:.4} from the reference's on \
             average, where the reference's mel on other noise is {:.4} from itself; by \
             log-spectral distance {between:.2} dB and {itself:.2} dB; the recording's voice by \
             CAMPPlus {like_a:.3} and {like_b:.3}",
            sa.seconds(),
            heard(&ha, &hb),
            heard(&ha, &ha_again),
        );
        if index == 0 {
            std::fs::write(format!("{out}.32.wav"), wav::write(&sa))?;
            std::fs::write(format!("{out}.16.wav"), wav::write(&sb))?;
        }
    }
    Ok(())
}
