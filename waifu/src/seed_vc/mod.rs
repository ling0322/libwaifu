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

//! Seed-VC v2, whole: somebody speaking and a recording of somebody else in; the first one's
//! words, in the second one's voice, out.
//!
//! ```text
//! let vc = SeedVc::from_manifest(Device::Cuda, Residency::Device, &manifest)?;
//! let voice = vc.listen(&reference)?;                            // once per voice
//! let sound = vc.convert(&source, &voice, &vc.conversion(), &mut report)?;
//! ```
//!
//! Upstream is `Plachtaa/seed-vc`'s `VoiceConversionWrapper.convert_voice_with_streaming`, which
//! is what `inference_v2.py` and the web demo both call. This is its order and its arithmetic in
//! the gaps; each model has a module of its own.
//!
//! # Listening, once per voice
//!
//! The reference is brought to 22.05 kHz and cut to twenty-five seconds, and from there to 16 kHz.
//!
//! | | reads | gives |
//! | --- | --- | --- |
//! | [`hubert`] | the 16 kHz audio | `(1, T, 1024)`, fifty frames a second |
//! | [`astral`], twice | HuBERT's features | wide and narrow tokens, one a frame |
//! | CAMPPlus | Kaldi energies of the 16 kHz audio | `style`, `(1, 192)` |
//! | the 22.05 kHz mel | the 22.05 kHz audio | the prompt, `(1, 80, M)` |
//! | the CFM's regulator | the wide tokens, onto `M` frames | `prompt_condition`, `(1, M, 512)` |
//!
//! # Converting
//!
//! The source is heard the same way -- tokens only -- and then one of two paths runs.
//!
//! **Timbre only** (the default, `convert_style: false`). The source's wide tokens are regulated
//! onto the source's own mel frame count, so the timing is the source's. The CFM draws the mel
//! after the reference's prompt, in windows of thirty seconds less the prompt, each overlapping
//! the one before by sixteen frames; BigVGAN sounds each window and the overlaps are crossfaded.
//!
//! **Style too** (`convert_style: true`). Both speakers' narrow tokens have their runs collapsed,
//! and [`ar::Ar`] continues the reference's wide tokens with the source's content -- in pieces of
//! at most 1500 narrow tokens, the reference's included. Each piece's tokens are regulated onto
//! as many frames as the source's frame rate gives them, and drawn and sounded as above.
//!
//! # Things upstream does in the gaps
//!
//! - **Long audio is heard in overlapping pieces.** HuBERT and ASTRAL see at most thirty seconds
//!   at a time; each piece after the first starts five seconds back and its first 250 tokens are
//!   dropped. [`SeedVc::content`] does the same, and since ASTRAL's normalization runs over time
//!   (see [`astral`]), where the pieces fall changes the tokens, as it does upstream.
//! - **The style path's pieces are crossfaded as if they overlapped.** They do not -- each is a
//!   different stretch of the source -- so upstream blends the last sixteen frames of one into the
//!   first sixteen of the next and loses a fraction of a second at every seam. Only a source long
//!   enough to need two pieces (well over a minute) meets it, and this does what upstream does.
//! - **Guidance is two rates.** See [`dit`].
//!
//! # What is not upstream's
//!
//! The random numbers -- the CFM's noise and the AR's draws are this crate's -- and resampling:
//! upstream reads files through `librosa.load` and resamples with soxr, this through
//! [`crate::indextts::features::resample`]'s windowed sinc. The two agree to well under a
//! quantization step of 16-bit audio; the tests hand every model the samples upstream held, so a
//! resampler cannot hide inside a model's error.

/// The AR that re-says the source in the reference's manner.
pub mod ar;
/// ASTRAL: HuBERT's features to wide and narrow tokens.
pub mod astral;
/// The CFM: its length regulator, its DiT and its sampler.
pub mod dit;
/// HuBERT-large, cut after its eighteenth layer.
pub mod hubert;

use std::ops::ControlFlow;
use std::rc::Rc;

use crate::flint::{
    functional as F, DType, Device, Graph, Ir, ParamSource, Residency, RunContext, Tensor, Weights,
};
use crate::indextts::bigvgan::{BigVgan, BigVganConfig};
use crate::indextts::{campplus, features};
use crate::{Error, Manifest, Result, Sound, SpeechProgress};

use self::ar::{Ar, Sampler, Sampling};
use self::dit::Guidance;

/// Where each model sits in the package. See `tools/seed_vc_exporter.py`.
const HUBERT: &str = "seed_vc.hubert";
const WIDE: &str = "seed_vc.wide";
const NARROW: &str = "seed_vc.narrow";
const CFM: &str = "seed_vc.cfm";
const CFM_REGULATOR: &str = "seed_vc.cfm_regulator";
const AR: &str = "seed_vc.ar";
const AR_REGULATOR: &str = "seed_vc.ar_regulator";
const CAMPPLUS: &str = "seed_vc.campplus";
const BIGVGAN: &str = "seed_vc.bigvgan";

/// The rate the mel, the CFM and the vocoder work at, and the rate of what comes out.
pub const RATE: u32 = 22050;
/// The rate HuBERT and CAMPPlus listen at.
pub const LISTENING_RATE: u32 = 16000;
/// Samples a mel frame.
pub const HOP: usize = 256;
/// Tokens a second, at 16 kHz and 320 samples a frame.
const TOKENS_A_SECOND: usize = 50;
/// The longest stretch HuBERT hears at once, and how far each piece after the first starts back.
const HEARING_SECONDS: usize = 30;
const HEARING_OVERLAP_SECONDS: usize = 5;
/// How many mel frames two windows of the CFM share, and are crossfaded over.
const OVERLAP_FRAMES: usize = 16;

/// What a conversion is steered by, as the manifest's `seed_vc:` section states it.
#[derive(Clone, Copy, Debug)]
pub struct Settings {
    pub diffusion_steps: i32,
    pub guidance: Guidance,
    pub sampling: Sampling,
    /// How much of the reference is listened to.
    pub reference_seconds: f32,
    /// How long one window of the CFM may be, prompt included.
    pub context_seconds: i32,
    /// How many narrow tokens one reading of the AR may take, the reference's included.
    pub ar_max_content_len: usize,
}

impl Settings {
    /// `inference_v2.py`'s defaults and the wrapper's limits, which the package also writes.
    pub fn seed_vc() -> Settings {
        Settings {
            diffusion_steps: 30,
            guidance: Guidance {
                intelligibility: 0.7,
                similarity: 0.7,
            },
            sampling: Sampling::seed_vc(),
            reference_seconds: 25.0,
            context_seconds: 30,
            ar_max_content_len: 1500,
        }
    }

    fn from_manifest(manifest: &Manifest) -> Result<Settings> {
        let mut settings = Settings::seed_vc();
        let Ok(section) = manifest.section(SeedVc::MODEL_TYPE) else {
            return Ok(settings);
        };

        let float = |key: &str, into: &mut f32| {
            if let Ok(value) = section.get::<f32>(key) {
                *into = value;
            }
        };
        float(
            "intelligibility_cfg_rate",
            &mut settings.guidance.intelligibility,
        );
        float("similarity_cfg_rate", &mut settings.guidance.similarity);
        float("top_p", &mut settings.sampling.top_p);
        float("temperature", &mut settings.sampling.temperature);
        float(
            "repetition_penalty",
            &mut settings.sampling.repetition_penalty,
        );
        float("reference_seconds", &mut settings.reference_seconds);

        if let Ok(value) = section.get::<i32>("diffusion_steps") {
            settings.diffusion_steps = value.max(1);
        }
        if let Ok(value) = section.get::<i32>("context_seconds") {
            settings.context_seconds = value.max(1);
        }
        if let Ok(value) = section.get::<i32>("ar_max_content_len") {
            settings.ar_max_content_len = value.max(1) as usize;
        }

        Ok(settings)
    }
}

/// How one conversion runs. [`SeedVc::conversion`] gives the package's defaults.
#[derive(Clone, Copy, Debug)]
pub struct Conversion {
    /// Re-say the source in the reference's accent and pacing with the AR, rather than only
    /// changing the voice.
    pub convert_style: bool,
    pub diffusion_steps: i32,
    pub guidance: Guidance,
    pub sampling: Sampling,
    /// Stretches what the style path draws: above one is slower.
    pub length_adjust: f32,
    /// Where the noise and the AR's draws come from; a fresh seed each time without one.
    pub seed: Option<u64>,
}

/// Everything a reference recording says about a voice, worked out once.
pub struct Reference {
    /// The reference's wide tokens, fifty a second.
    pub wide: Vec<i32>,
    /// Its narrow tokens, runs not collapsed.
    pub narrow: Vec<i32>,
    /// CAMPPlus's embedding, `(1, 192)`.
    pub style: Tensor,
    /// Its 22.05 kHz mel, band-major `(80, frames)`, which the CFM continues from.
    pub mel: Vec<f32>,
    pub frames: i32,
    /// The wide tokens regulated onto the mel's frames, `(1, frames, 512)`.
    pub prompt_condition: Tensor,
}

/// What HuBERT and both quantizers make of one piece of 16 kHz audio.
pub struct Heard {
    /// `(1, T, 1024)`, HuBERT's eighteenth layer.
    pub hubert: Tensor,
    /// `(T * 11)` and `(T * 5)`: BSQ's projections, whose signs are the codes.
    pub wide_projected: Vec<f32>,
    pub narrow_projected: Vec<f32>,
    pub wide: Vec<i32>,
    pub narrow: Vec<i32>,
}

/// The DiT compiled for one length and one batch, with the rotary rows it needs.
pub struct Denoiser<'a> {
    ir: Ir,
    frames: i32,
    batch: i32,
    cos: Tensor,
    sin: Tensor,
    owner: &'a SeedVc,
}

impl Denoiser<'_> {
    /// The direction at `x` `(batch, 80, frames)`, each row at its own time.
    pub fn direction(
        &self,
        x: &Tensor,
        prompt_x: &Tensor,
        cond: &Tensor,
        style: &Tensor,
        times: &[f32],
    ) -> Result<Tensor> {
        let config = dit::Config::seed_vc();
        let width = config.frequency_embedding_size;
        if times.len() != self.batch as usize {
            return Err(Error::model("one time a row"));
        }
        let embedded: Vec<f32> = times
            .iter()
            .flat_map(|t| dit::timestep_embedding(*t, width))
            .collect();
        let time = self.owner.upload(&[self.batch, width], &embedded)?;

        let context = RunContext::new(self.owner.weights.as_ref())
            .input("x", x)
            .input("prompt_x", prompt_x)
            .input("cond", cond)
            .input("style", style)
            .input("time", &time)
            .input("cos", &self.cos)
            .input("sin", &self.sin);

        Ok(self.ir.run(&context)?.remove(0).1)
    }

    pub fn frames(&self) -> i32 {
        self.frames
    }
}

/// Seed-VC v2, with every model it runs read and waiting.
pub struct SeedVc {
    settings: Settings,
    weights: Rc<dyn ParamSource>,
    ar: Ar,
    vocoder: BigVgan,
    /// The DiT's rotary table, `(8192, 32)` each, on the host.
    cos: Vec<f32>,
    sin: Vec<f32>,
    device: Device,
    dtype: DType,
}

impl SeedVc {
    pub const MODEL_TYPE: &'static str = "seed_vc";

    /// What the screen calls it.
    pub const NAME: &'static str = "Seed-VC v2";

    /// Rows of the DiT's rotary table.
    const ROTARY_ROWS: i32 = 8192;

    /// Read the whole model `manifest` describes, onto `device`.
    pub fn from_manifest(
        device: Device,
        residency: Residency,
        manifest: &Manifest,
    ) -> Result<SeedVc> {
        let model_type = manifest.section("model")?.get_str("type")?.to_string();
        if model_type != Self::MODEL_TYPE {
            return Err(Error::model(format!(
                "this manifest describes a model of kind {model_type:?}, which is not {:?}",
                Self::MODEL_TYPE
            )));
        }

        // Every model here runs in full precision, as upstream's are stored.
        let dtype = DType::Float;
        let weights: Rc<dyn ParamSource> = Rc::new(Weights::from_files(
            &manifest.weight_paths()?,
            device,
            residency,
        )?);

        let half = dit::Config::seed_vc().head_dim() / 2;
        let table = |which: &str| -> Result<Vec<f32>> {
            Ok(weights
                .load(&format!("{CFM}.{which}"), &[Self::ROTARY_ROWS, half])?
                .to_device(Device::Cpu)?
                .cast(DType::Float)?
                .to_vec_f32()?)
        };

        Ok(SeedVc {
            settings: Settings::from_manifest(manifest)?,
            ar: Ar::build(
                ar::Config::seed_vc(),
                AR,
                AR_REGULATOR,
                &weights,
                dtype,
                device,
            )?,
            vocoder: BigVgan::build(
                BigVganConfig::v2_22khz_80band_256x(),
                BIGVGAN,
                &weights,
                device,
                dtype,
            )?,
            cos: table("rotary_cos")?,
            sin: table("rotary_sin")?,
            weights,
            device,
            dtype,
        })
    }

    pub fn settings(&self) -> &Settings {
        &self.settings
    }

    pub fn device(&self) -> Device {
        self.device
    }

    pub fn ar(&self) -> &Ar {
        &self.ar
    }

    pub fn vocoder(&self) -> &BigVgan {
        &self.vocoder
    }

    /// The package's defaults, timbre only.
    pub fn conversion(&self) -> Conversion {
        Conversion {
            convert_style: false,
            diffusion_steps: self.settings.diffusion_steps,
            guidance: self.settings.guidance,
            sampling: self.settings.sampling,
            length_adjust: 1.0,
            seed: None,
        }
    }

    fn run(&self, g: &Graph, inputs: &[(&str, &Tensor)]) -> Result<Vec<Tensor>> {
        let ir = Ir::compile(g);
        let mut context = RunContext::new(self.weights.as_ref());
        for (name, tensor) in inputs {
            context = context.input(name, tensor);
        }

        Ok(ir
            .run(&context)?
            .into_iter()
            .map(|(_, tensor)| tensor)
            .collect())
    }

    fn upload(&self, shape: &[i32], values: &[f32]) -> Result<Tensor> {
        Ok(Tensor::from_f32(shape, values)?
            .to_device(self.device)?
            .cast(self.dtype)?)
    }

    fn host(tensor: &Tensor) -> Result<Vec<f32>> {
        Ok(tensor
            .to_device(Device::Cpu)?
            .cast(DType::Float)?
            .to_vec_f32()?)
    }

    /// Everything the voice in `recording` contributes, worked out once.
    pub fn listen(&self, recording: &Sound) -> Result<Reference> {
        let mut wave = features::resample(&recording.samples, recording.rate, RATE);
        wave.truncate((self.settings.reference_seconds * RATE as f32) as usize);
        let listening = features::resample(&wave, RATE, LISTENING_RATE);

        self.reference(&wave, &listening)
    }

    /// [`SeedVc::listen`] on the reference already at both rates -- `wave` at 22.05 kHz and
    /// `listening` at 16 kHz.
    pub fn reference(&self, wave: &[f32], listening: &[f32]) -> Result<Reference> {
        let (wide, narrow) = self.content(listening)?;
        let style = self.style(listening)?;

        let (mel, frames) = features::reference_mel(wave)?;
        let frames = frames as i32;
        if frames < 1 {
            return Err(Error::model(
                "the reference is too short to hear a voice in",
            ));
        }
        let prompt_condition = self.regulate(&wide, frames)?;

        Ok(Reference {
            wide,
            narrow,
            style,
            mel,
            frames,
            prompt_condition,
        })
    }

    /// HuBERT and both quantizers over one piece of 16 kHz audio, at most thirty seconds.
    pub fn hear(&self, listening: &[f32]) -> Result<Heard> {
        let config = hubert::Config::hubert_large();
        let frames = config.frames(listening.len());
        if frames < 1 {
            return Err(Error::model(
                "the recording is too short to hear -- a fraction of a second at least",
            ));
        }

        let normalized = hubert::normalize(listening);
        let input = self.upload(&[1, 1, listening.len() as i32], &normalized)?;

        let g = Graph::new();
        let hidden = hubert::graph(
            &g.subgraph(HUBERT),
            g.input("wave"),
            &config,
            frames,
            self.dtype,
            self.device,
        )?;
        let wide = astral::graph(
            &g.subgraph(WIDE),
            hidden,
            astral::WIDE_BITS,
            self.dtype,
            self.device,
        )?;
        let narrow = astral::graph(
            &g.subgraph(NARROW),
            hidden,
            astral::NARROW_BITS,
            self.dtype,
            self.device,
        )?;
        g.output("hubert", hidden);
        g.output("wide", wide);
        g.output("narrow", narrow);

        let mut out = self.run(&g, &[("wave", &input)])?;
        let narrow_projected = Self::host(&out.remove(2))?;
        let wide_projected = Self::host(&out.remove(1))?;

        Ok(Heard {
            hubert: out.remove(0),
            wide: astral::tokens(&wide_projected, astral::WIDE_BITS),
            narrow: astral::tokens(&narrow_projected, astral::NARROW_BITS),
            wide_projected,
            narrow_projected,
        })
    }

    /// The wide and narrow tokens of 16 kHz audio of any length, heard in overlapping pieces as
    /// upstream's `_process_content_features` hears it.
    pub fn content(&self, listening: &[f32]) -> Result<(Vec<i32>, Vec<i32>)> {
        let piece = HEARING_SECONDS * LISTENING_RATE as usize;
        if listening.len() <= piece {
            let heard = self.hear(listening)?;
            return Ok((heard.wide, heard.narrow));
        }

        let overlap = HEARING_OVERLAP_SECONDS * LISTENING_RATE as usize;
        let skipped = TOKENS_A_SECOND * HEARING_OVERLAP_SECONDS;
        let (mut wide, mut narrow) = (Vec::new(), Vec::new());
        let mut traversed = 0;
        let mut first = true;
        while traversed < listening.len() {
            let (from, to) = match first {
                true => (0, piece.min(listening.len())),
                false => (
                    traversed - overlap,
                    (traversed + piece - overlap).min(listening.len()),
                ),
            };
            let chunk = &listening[from..to];
            let heard = self.hear(chunk)?;
            let skip = if first { 0 } else { skipped };
            wide.extend(heard.wide.iter().skip(skip));
            narrow.extend(heard.narrow.iter().skip(skip));

            traversed = match first {
                true => piece,
                false => traversed + chunk.len() - overlap,
            };
            first = false;
        }

        Ok((wide, narrow))
    }

    /// CAMPPlus's embedding of who is speaking in 16 kHz audio.
    pub fn style(&self, listening: &[f32]) -> Result<Tensor> {
        let (values, frames) = features::campplus(listening);
        let frames = frames as i32;
        let input = self.upload(&[1, frames, 80], &values)?;

        let g = Graph::new();
        let embedding = campplus::graph(
            &g.subgraph(CAMPPLUS),
            g.input("x"),
            &campplus::Config::indextts(),
            frames,
            self.dtype,
            self.device,
        )?;
        g.output("style", embedding);

        Ok(self.run(&g, &[("x", &input)])?.remove(0))
    }

    /// Wide `tokens` laid onto `frames` mel frames by the CFM's regulator, `(1, frames, 512)`.
    pub fn regulate(&self, tokens: &[i32], frames: i32) -> Result<Tensor> {
        if tokens.is_empty() {
            return Err(Error::model("there are no tokens to lay onto frames"));
        }
        let selection =
            dit::nearest_selection(tokens.len() as i32, frames, self.device)?.cast(self.dtype)?;
        let ids: Vec<i64> = tokens.iter().map(|token| i64::from(*token)).collect();
        let ids = Tensor::from_i64(&[tokens.len() as i32], &ids)?.to_device(self.device)?;

        let g = Graph::new();
        let regulated = dit::regulator(
            &g.subgraph(CFM_REGULATOR),
            g.input("tokens"),
            g.input("selection"),
            self.dtype,
            self.device,
        )?;
        g.output("regulated", regulated);

        Ok(self
            .run(&g, &[("tokens", &ids), ("selection", &selection)])?
            .remove(0))
    }

    /// The DiT compiled for `frames` frames and `batch` rows.
    pub fn denoiser(&self, frames: i32, batch: i32) -> Result<Denoiser<'_>> {
        let config = dit::Config::seed_vc();
        let length = frames + 2;
        if length > Self::ROTARY_ROWS {
            return Err(Error::model(format!(
                "{frames} frames is more than the CFM's {} positions",
                Self::ROTARY_ROWS - 2
            )));
        }

        let g = Graph::new();
        let direction = dit::graph(
            &g.subgraph(CFM),
            g.input("x"),
            g.input("prompt_x"),
            g.input("cond"),
            g.input("style"),
            g.input("time"),
            &config,
            frames,
            g.input("cos"),
            g.input("sin"),
        );
        g.output("direction", direction);

        let half = (config.head_dim() / 2) as usize;
        let rows = length as usize * half;
        let shape = [length, half as i32];

        Ok(Denoiser {
            ir: Ir::compile(&g),
            frames,
            batch,
            cos: self.upload(&shape, &self.cos[..rows])?,
            sin: self.upload(&shape, &self.sin[..rows])?,
            owner: self,
        })
    }

    /// Draw the mel for `condition` `(1, frames, 512)` -- the reference's prompt condition and
    /// what follows it -- starting from `noise` `(80 * frames)`: `(80 * frames)`, the prompt's
    /// frames still on the front and zero.
    pub fn draw_mel(
        &self,
        noise: &[f32],
        condition: &Tensor,
        reference: &Reference,
        steps: i32,
        guidance: Guidance,
    ) -> Result<Vec<f32>> {
        let config = dit::Config::seed_vc();
        let channels = config.in_channels as usize;
        let frames = condition.shape_at(1)? as usize;
        let prompt = (reference.frames as usize).min(frames);

        let passes = dit::passes(guidance);
        let batch = passes.len();
        let denoiser = self.denoiser(frames as i32, batch as i32)?;

        // The prompt laid into the front of a zero mel, and the rows each pass is handed.
        let mut prompt_x = vec![0.0f32; channels * frames];
        let reference_frames = reference.frames as usize;
        for channel in 0..channels {
            prompt_x[channel * frames..channel * frames + prompt].copy_from_slice(
                &reference.mel[channel * reference_frames..channel * reference_frames + prompt],
            );
        }
        let condition_values = Self::host(condition)?;
        let style_values = Self::host(&reference.style)?;
        let width = condition_values.len() / frames;

        let mut prompts = Vec::with_capacity(batch * prompt_x.len());
        let mut conditions = Vec::with_capacity(batch * condition_values.len());
        let mut styles = Vec::with_capacity(batch * style_values.len());
        for (voice, content) in &passes {
            match voice {
                true => {
                    prompts.extend_from_slice(&prompt_x);
                    styles.extend_from_slice(&style_values);
                }
                false => {
                    prompts.extend(std::iter::repeat_n(0.0, prompt_x.len()));
                    styles.extend(std::iter::repeat_n(0.0, style_values.len()));
                }
            }
            match content {
                true => conditions.extend_from_slice(&condition_values),
                false => conditions.extend(std::iter::repeat_n(0.0, condition_values.len())),
            }
        }

        let n = batch as i32;
        let (frames_i, channels_i) = (frames as i32, channels as i32);
        let prompts = self.upload(&[n, channels_i, frames_i], &prompts)?;
        let conditions = self.upload(&[n, frames_i, width as i32], &conditions)?;
        let styles = self.upload(&[n, config.style_dim], &styles)?;
        let row = channels * frames;

        dit::solve_euler(noise, channels, frames, prompt, steps, |x, t| {
            let repeated: Vec<f32> = x.iter().copied().cycle().take(batch * row).collect();
            let x = self.upload(&[n, channels_i, frames_i], &repeated)?;
            let direction =
                denoiser.direction(&x, &prompts, &conditions, &styles, &vec![t; batch])?;
            Ok(dit::combine(guidance, &Self::host(&direction)?, row))
        })
    }

    /// Mel frames a 22.05 kHz wave of `samples` samples makes, as the mel function frames it.
    pub fn mel_frames(samples: usize) -> usize {
        let padded = samples + 2 * ((1024 - HOP) / 2);
        match padded >= 1024 {
            true => 1 + (padded - 1024) / HOP,
            false => 0,
        }
    }

    /// `source` said in the voice of `reference`.
    pub fn convert(
        &self,
        source: &Sound,
        reference: &Reference,
        conversion: &Conversion,
        report: &mut dyn FnMut(SpeechProgress) -> ControlFlow<()>,
    ) -> Result<Option<Sound>> {
        let wave = features::resample(&source.samples, source.rate, RATE);
        let listening = features::resample(&wave, RATE, LISTENING_RATE);
        self.convert_waves(&wave, &listening, reference, conversion, report)
    }

    /// [`SeedVc::convert`] on the source already at both rates.
    pub fn convert_waves(
        &self,
        wave: &[f32],
        listening: &[f32],
        reference: &Reference,
        conversion: &Conversion,
        report: &mut dyn FnMut(SpeechProgress) -> ControlFlow<()>,
    ) -> Result<Option<Sound>> {
        let seed = conversion.seed.unwrap_or_else(|| {
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|elapsed| elapsed.as_nanos() as u64)
                .unwrap_or(0)
        });
        F::manual_seed(self.device, seed)?;
        let mut sampler = Sampler::new(seed);

        if report(SpeechProgress::Reading).is_break() {
            return Ok(None);
        }

        let source_frames = Self::mel_frames(wave.len());
        let (source_wide, source_narrow) = self.content(listening)?;
        if source_frames == 0 || source_wide.is_empty() {
            return Err(Error::model("the source is too short to convert"));
        }

        let mut stitch = Stitch::new(OVERLAP_FRAMES * HOP);
        let window = (RATE as usize / HOP) * self.settings.context_seconds as usize;
        let prompt = reference.frames as usize;

        match conversion.convert_style {
            false => {
                let window = window
                    .checked_sub(prompt)
                    .filter(|window| *window > OVERLAP_FRAMES)
                    .ok_or_else(|| Error::model("the reference is longer than the CFM's window"))?;
                let condition = self.regulate(&source_wide, source_frames as i32)?;
                let condition = Self::host(&condition)?;
                let width = condition.len() / source_frames;

                let mut processed = 0;
                while processed < source_frames {
                    let end = (processed + window).min(source_frames);
                    let last = processed + window >= source_frames;
                    let chunk = &condition[processed * width..end * width];

                    let mel = self.window(chunk, width, reference, conversion)?;
                    let drawn = mel.len() / 80;
                    stitch.push(self.sound(&mel, drawn)?, last);

                    if report(SpeechProgress::Sounding).is_break() {
                        return Ok(None);
                    }
                    if last {
                        break;
                    }
                    processed += drawn - OVERLAP_FRAMES;
                }
            }
            true => {
                let target = astral::reduce_durations(&reference.narrow);
                let source = astral::reduce_durations(&source_narrow);
                let piece = self
                    .settings
                    .ar_max_content_len
                    .checked_sub(target.len())
                    .filter(|piece| *piece > 0)
                    .ok_or_else(|| {
                        Error::model("the reference says too much for the AR to read beside it")
                    })?;

                let expected = source.len() as i32;
                let mut done = 0;
                for (index, chunk) in source.chunks(piece).enumerate() {
                    let last = (index + 1) * piece >= source.len();
                    let mut narrow = target.clone();
                    narrow.extend_from_slice(chunk);

                    let said = self.ar.generate(
                        &narrow,
                        &reference.wide,
                        &conversion.sampling,
                        &mut sampler,
                        &mut |count| {
                            report(SpeechProgress::Saying {
                                done: done + count,
                                expected: expected.max(done + count),
                            })
                        },
                    )?;
                    let Some(said) = said else {
                        return Ok(None);
                    };
                    done += said.len() as i32;
                    if said.is_empty() {
                        continue;
                    }

                    // Upstream's arithmetic, in doubles and truncated: the source's frames per
                    // token, times the tokens said, times the stretch.
                    let frames = (source_frames as f64 / source_wide.len() as f64
                        * said.len() as f64
                        * f64::from(conversion.length_adjust))
                        as i32;
                    let condition = Self::host(&self.regulate(&said, frames.max(1))?)?;
                    let width = condition.len() / frames.max(1) as usize;

                    if report(SpeechProgress::Sounding).is_break() {
                        return Ok(None);
                    }
                    let mel = self.window(&condition, width, reference, conversion)?;
                    let drawn = mel.len() / 80;
                    stitch.push(self.sound(&mel, drawn)?, last);
                }
            }
        }

        Ok(Some(Sound::new(stitch.finish(), RATE)))
    }

    /// One window: `chunk` `(frames * width)` after the reference's prompt, drawn and cut back to
    /// its own frames, band-major `(80 * frames)`.
    fn window(
        &self,
        chunk: &[f32],
        width: usize,
        reference: &Reference,
        conversion: &Conversion,
    ) -> Result<Vec<f32>> {
        let prompt = reference.frames as usize;
        let own = chunk.len() / width;
        let frames = prompt + own;

        let mut joined = Self::host(&reference.prompt_condition)?;
        joined.extend_from_slice(chunk);
        let condition = self.upload(&[1, frames as i32, width as i32], &joined)?;

        let noise = F::randn(&[1, 80, frames as i32], self.device)?;
        let noise = Self::host(&noise)?;

        let mel = self.draw_mel(
            &noise,
            &condition,
            reference,
            conversion.diffusion_steps,
            conversion.guidance,
        )?;

        let mut out = Vec::with_capacity(80 * own);
        for channel in 0..80 {
            out.extend_from_slice(&mel[channel * frames + prompt..(channel + 1) * frames]);
        }
        Ok(out)
    }

    /// A band-major mel `(80 * frames)` to 22.05 kHz audio.
    fn sound(&self, mel: &[f32], frames: usize) -> Result<Vec<f32>> {
        let mel = self.upload(&[1, 80, frames as i32], mel)?;
        let wave = self.vocoder.forward(&mel)?;
        Ok(Self::host(&wave)?
            .into_iter()
            .map(|sample| sample.clamp(-1.0, 1.0))
            .collect())
    }
}

/// Windows of audio, each crossfaded into the last over `overlap` samples, as upstream's
/// `_stream_wave_chunks` does.
///
/// Every window but the last holds back its final `overlap` samples; the next window's first
/// `overlap` are faded in against them, `cos²` against `cos²`.
pub struct Stitch {
    out: Vec<f32>,
    held: Option<Vec<f32>>,
    overlap: usize,
}

impl Stitch {
    pub fn new(overlap: usize) -> Stitch {
        Stitch {
            out: Vec::new(),
            held: None,
            overlap,
        }
    }

    /// `numpy.linspace(from, to, count)`'s `cos²`.
    fn fade(from: f64, to: f64, count: usize) -> Vec<f64> {
        (0..count)
            .map(|index| {
                let t = match count {
                    1 => from,
                    _ => from + (to - from) * index as f64 / (count - 1) as f64,
                };
                t.cos().powi(2)
            })
            .collect()
    }

    fn crossfade(&self, held: &[f32], wave: &mut [f32]) {
        let half_pi = std::f64::consts::FRAC_PI_2;
        let fade_out = Self::fade(0.0, half_pi, self.overlap);
        let fade_in = Self::fade(half_pi, 0.0, self.overlap);
        let tail = &held[held.len().saturating_sub(self.overlap)..];

        for (index, sample) in wave.iter_mut().take(self.overlap).enumerate() {
            let faded = f64::from(*sample) * fade_in[index]
                + f64::from(tail.get(index).copied().unwrap_or(0.0)) * fade_out[index];
            *sample = faded as f32;
        }
    }

    pub fn push(&mut self, mut wave: Vec<f32>, last: bool) {
        let split = wave.len().saturating_sub(self.overlap);
        let held = match last {
            true => None,
            false => Some(wave.split_off(split)),
        };

        if let Some(previous) = self.held.take() {
            self.crossfade(&previous, &mut wave);
        }
        self.out.extend(wave);
        self.held = held;
    }

    /// Everything pushed, including what the last window held back if it was not marked last.
    pub fn finish(mut self) -> Vec<f32> {
        if let Some(held) = self.held.take() {
            self.out.extend(held);
        }
        self.out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_mel_frame_is_a_hop() {
        // Six seconds at 22.05 kHz: upstream's mel has 516 frames.
        assert_eq!(SeedVc::mel_frames(132_300), 516);
        assert_eq!(SeedVc::mel_frames(256), 1);
    }

    #[test]
    fn one_window_is_passed_through_whole() {
        let mut stitch = Stitch::new(4);
        stitch.push(vec![1.0; 10], true);
        assert_eq!(stitch.finish(), vec![1.0; 10]);
    }

    #[test]
    fn windows_are_crossfaded_over_the_overlap() {
        let mut stitch = Stitch::new(4);
        stitch.push(vec![1.0; 10], false);
        stitch.push(vec![0.0; 10], true);
        let out = stitch.finish();

        // Six of the first, then four faded from it into the second, then six of the second.
        assert_eq!(out.len(), 16);
        assert_eq!(&out[..6], &[1.0; 6]);
        assert!((out[6] - 1.0).abs() < 1e-6, "{out:?}");
        assert!(out[7] > out[8] && out[8] > out[9]);
        assert!(out[9].abs() < 1e-6);
        assert_eq!(&out[10..], &[0.0; 6]);
    }
}
