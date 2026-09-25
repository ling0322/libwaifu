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

//! Fun-CosyVoice3-0.5B-2512, whole: a recording and a sentence in, the sentence in that voice out.
//!
//! ```text
//! let tts = CosyVoice3::from_manifest(Device::Cuda, Residency::Device, &manifest)?;
//! let voice = tts.listen(&recording)?;                    // once per voice
//! let sound = tts.say("今天天气很好。", &voice, &options, &mut report)?;
//! ```
//!
//! Five models, run in the order upstream's `CosyVoice3` runs them; `docs/cosyvoice3.md` has the
//! whole of it. Each has a module:
//!
//! | | reads | gives |
//! | --- | --- | --- |
//! | [`speech_tokenizer`] | the recording's Whisper mel, 16 kHz | its speech tokens, 25 a second |
//! | [`crate::indextts::campplus`] | Kaldi energies of the 16 kHz recording | the speaker, `(192)` |
//! | [`features::speech_mel`] | the recording at 24 kHz | its mel, two frames a token |
//! | [`lm`] | the text, and in zero-shot the transcript and the prompt's tokens | speech tokens |
//! | [`flow`] | the prompt's and the sentence's tokens, the prompt's mel, the speaker | the mel |
//! | [`hift`] | the mel | 24 kHz audio |
//!
//! # Two ways to read, and which one [`Voice`] uses
//!
//! Upstream's **zero-shot** reading is handed the transcript of the recording: the language model
//! reads the transcript and the sentence as one text and continues the recording's own speech
//! tokens, so it carries on in the recording's voice, pace and manner. [`CosyVoice3::say_after`]
//! is that.
//!
//! [`Voice::speak`] is handed a recording and no transcript, which is upstream's **cross-lingual**
//! reading: the language model sees only the sentence, and the voice comes from the flow, which
//! still continues the recording's tokens and mel with its speaker. [`CosyVoice3::say`] is that.
//!
//! Both put the system prompt, `You are a helpful assistant.<|endofprompt|>`, in front of the
//! text -- the model asserts `<|endofprompt|>` is there. Upstream's own examples put it inside the
//! string handed to `inference_cross_lingual`, which then skips every text normalization because
//! the string holds a marker; this normalizes and splits the sentence first and puts the prompt
//! on each piece, which is the same ids for a text that needed no normalizing.
//!
//! # What is not upstream's
//!
//! - **The random numbers.** The sampler and HiFT's source noise draw from this crate's own
//!   generator, seeded by the reading. The flow's starting noise *is* upstream's: it is a fixed
//!   draw, and the package holds it.
//! - **A temperature.** Upstream samples at one; [`SpeechOptions::temperature`] divides the scores
//!   first, and the default is one.
//! - **Long recordings are cut to thirty seconds**, where upstream refuses them.

/// The two spectrograms a recording is read as.
pub mod features;
/// The flow: speech tokens in, the mel out.
pub mod flow;
/// Upstream's text normalization and sentence splitting.
pub mod frontend;
/// HiFT, the vocoder: the mel in, 24 kHz audio out.
pub mod hift;
/// The language model: text in, speech tokens out.
pub mod lm;
/// S3Tokenizer v3: a recording in, speech tokens out.
pub mod speech_tokenizer;

use std::cell::RefCell;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::ops::ControlFlow;
use std::rc::Rc;

use crate::flint::{DType, Device, Graph, Ir, ParamSource, Residency, RunContext, Tensor, Weights};
use crate::indextts::{campplus, features::resample};
use crate::{
    Error, Manifest, Result, Sound, SpeechDefaults, SpeechOptions, SpeechProgress, Tokenizer, Voice,
};

use self::flow::Flow;
use self::hift::{Hift, Noise};
use self::lm::{Lm, Sampler, Sampling};

const LM: &str = "cosyvoice3.lm";
const FLOW: &str = "cosyvoice3.flow";
const HIFT: &str = "cosyvoice3.hift";
const SPEECH_TOKENIZER: &str = "cosyvoice3.speech_tokenizer";
const CAMPPLUS: &str = "cosyvoice3.campplus";

/// The rate of what comes out.
pub const RATE: u32 = hift::RATE;

/// The rate the speech tokenizer and CAMPPlus listen at.
const LISTENING_RATE: u32 = 16000;

/// The longest recording the speech tokenizer reads, as upstream asserts.
const LONGEST_RECORDING: f32 = 30.0;

/// What a reading is steered by, as the manifest's `cosyvoice3:` section states it.
#[derive(Clone, Debug)]
pub struct Settings {
    pub sampling: Sampling,
    pub diffusion_steps: usize,
    pub cfg_rate: f32,
    pub system_prompt: String,
}

impl Settings {
    /// `cosyvoice3.yaml`'s own, which is also what the package writes.
    pub fn cosyvoice3() -> Settings {
        Settings {
            sampling: Sampling::cosyvoice3(),
            diffusion_steps: 10,
            cfg_rate: 0.7,
            system_prompt: "You are a helpful assistant.<|endofprompt|>".to_string(),
        }
    }

    fn from_manifest(manifest: &Manifest) -> Settings {
        let mut settings = Settings::cosyvoice3();
        let Ok(section) = manifest.section(CosyVoice3::MODEL_TYPE) else {
            return settings;
        };

        if let Ok(value) = section.get::<f32>("top_p") {
            settings.sampling.top_p = value;
        }
        if let Ok(value) = section.get::<i32>("top_k") {
            settings.sampling.top_k = value.max(1) as usize;
        }
        if let Ok(value) = section.get::<i32>("win_size") {
            settings.sampling.win_size = value.max(0) as usize;
        }
        if let Ok(value) = section.get::<f32>("tau_r") {
            settings.sampling.tau_r = value;
        }
        if let Ok(value) = section.get::<i32>("diffusion_steps") {
            settings.diffusion_steps = value.max(1) as usize;
        }
        if let Ok(value) = section.get::<f32>("cfg_rate") {
            settings.cfg_rate = value;
        }
        if let Ok(value) = section.get_str("system_prompt") {
            settings.system_prompt = value.to_string();
        }

        settings
    }
}

/// Everything a recording says about a voice, worked out once and kept for every sentence.
#[derive(Clone, Debug)]
pub struct Reference {
    /// The recording's speech tokens, cut to half its mel frames.
    pub tokens: Vec<i32>,
    /// Its 24 kHz mel, `(2 * tokens, 80)` frame-major.
    pub mel: Vec<f32>,
    /// CAMPPlus's embedding, `(192)`, not yet normalized.
    pub speaker: Vec<f32>,
}

/// Fun-CosyVoice3-0.5B, with every model it runs read and waiting.
pub struct CosyVoice3 {
    settings: Settings,
    weights: Rc<dyn ParamSource>,
    lm: Lm,
    flow: Flow,
    hift: Hift,
    tokenizer: Tokenizer,
    device: Device,
    dtype: DType,
    /// The last recording listened to, by fingerprint, and what was heard in it.
    heard: RefCell<Option<(u64, Reference)>>,
}

impl CosyVoice3 {
    pub const MODEL_TYPE: &'static str = "cosyvoice3";

    /// What the screen calls it.
    pub const NAME: &'static str = "Fun-CosyVoice3";

    /// Upstream samples with no temperature, which is one.
    pub const DEFAULTS: SpeechDefaults = SpeechDefaults {
        speed: 1.0,
        temperature: 1.0,
    };

    pub const NEEDS_A_RECORDING: &'static str = "CosyVoice3 speaks in the voice of a recording -- \
         drop one on the page, a few seconds of somebody speaking, and say the sentence again";

    /// Read the whole model `manifest` describes, onto `device`.
    pub fn from_manifest(
        device: Device,
        residency: Residency,
        manifest: &Manifest,
    ) -> Result<CosyVoice3> {
        let model_type = manifest.section("model")?.get_str("type")?.to_string();
        if model_type != Self::MODEL_TYPE {
            return Err(Error::model(format!(
                "this manifest describes a model of kind {model_type:?}, which is not {:?}",
                Self::MODEL_TYPE
            )));
        }

        // Upstream runs every model in float32 unless asked for fp16, and so does this.
        let dtype = DType::Float;
        let weights: Rc<dyn ParamSource> = Rc::new(Weights::from_files(
            &manifest.weight_paths()?,
            device,
            residency,
        )?);

        Ok(CosyVoice3 {
            settings: Settings::from_manifest(manifest),
            lm: Lm::build(lm::Config::cosyvoice3(), LM, &weights, dtype, device)?,
            flow: Flow::build(flow::Config::cosyvoice3(), FLOW, &weights, dtype, device)?,
            hift: Hift::build(HIFT, &weights, dtype, device)?,
            tokenizer: Tokenizer::open(manifest)?,
            weights,
            device,
            dtype,
            heard: RefCell::new(None),
        })
    }

    pub fn settings(&self) -> &Settings {
        &self.settings
    }

    fn run(&self, g: &Graph, inputs: &[(&str, &Tensor)]) -> Result<Vec<Tensor>> {
        let ir = Ir::compile(g);
        let mut context = RunContext::new(self.weights.as_ref());
        for (name, tensor) in inputs {
            context = context.input(name, tensor);
        }
        Ok(ir.run(&context)?.into_iter().map(|(_, t)| t).collect())
    }

    fn upload(&self, shape: &[i32], values: &[f32]) -> Result<Tensor> {
        Ok(Tensor::from_f32(shape, values)?.to_device(self.device)?)
    }

    fn host(tensor: &Tensor) -> Result<Vec<f32>> {
        Ok(tensor
            .to_device(Device::Cpu)?
            .cast(DType::Float)?
            .to_vec_f32()?)
    }

    /// Everything the voice in `recording` contributes, worked out once.
    pub fn listen(&self, recording: &Sound) -> Result<Reference> {
        let mut samples = recording.samples.clone();
        samples.truncate((LONGEST_RECORDING * recording.rate as f32) as usize);

        let listening = resample(&samples, recording.rate, LISTENING_RATE);
        let wave = resample(&samples, recording.rate, RATE);

        let tokens = self.speech_tokens(&listening)?;
        let (mel, frames) = features::speech_mel(&wave)?;

        // `frontend_zero_shot`: the mel is exactly two frames a token, whichever runs out first.
        let count = (frames / 2).min(tokens.len());
        if count == 0 {
            return Err(Error::model(
                "the recording is too short to hear a voice in -- a second or two at least",
            ));
        }
        let tokens = tokens[..count].to_vec();
        let kept = 2 * count;
        let mel: Vec<f32> = (0..kept)
            .flat_map(|frame| (0..80).map(move |band| band * frames + frame))
            .map(|index| mel[index])
            .collect();

        Ok(Reference {
            tokens,
            mel,
            speaker: self.speaker(&listening)?,
        })
    }

    /// S3Tokenizer's tokens of a 16 kHz recording.
    fn speech_tokens(&self, wave: &[f32]) -> Result<Vec<i32>> {
        let (mel, frames) = features::whisper_mel(wave)?;
        if frames < 4 {
            return Err(Error::model(
                "the recording is too short to hear a voice in",
            ));
        }
        let out = speech_tokenizer::frames_out(frames as i32);
        let (cos, sin, shape) = speech_tokenizer::rotary(out);

        let g = Graph::new();
        let projected = speech_tokenizer::graph(
            &g.subgraph(SPEECH_TOKENIZER),
            g.input("mel"),
            g.input("cos"),
            g.input("sin"),
            self.dtype,
            self.device,
        )?;
        g.output("projected", g.cast(projected, DType::Float));

        let outputs = self.run(
            &g,
            &[
                (
                    "mel",
                    &self.upload(&[1, speech_tokenizer::MELS, frames as i32], &mel)?,
                ),
                ("cos", &self.upload(&shape, &cos)?),
                ("sin", &self.upload(&shape, &sin)?),
            ],
        )?;
        Ok(speech_tokenizer::quantize(&Self::host(&outputs[0])?))
    }

    /// CAMPPlus's embedding of who is speaking, through IndexTTS-2.5's port of the same model.
    fn speaker(&self, wave: &[f32]) -> Result<Vec<f32>> {
        let (values, frames) = crate::indextts::features::campplus(wave);
        let frames = frames as i32;

        let g = Graph::new();
        let embedding = campplus::graph(
            &g.subgraph(CAMPPLUS),
            g.input("x"),
            &campplus::Config::indextts(),
            frames,
            self.dtype,
            self.device,
        )?;
        g.output("speaker", embedding);

        let outputs = self.run(&g, &[("x", &self.upload(&[1, frames, 80], &values)?)])?;
        Self::host(&outputs[0])
    }

    fn encode(&self, text: &str) -> Result<Vec<i32>> {
        self.tokenizer.encode(text)
    }

    /// `text` in the voice of `reference`, with no transcript of it: upstream's cross-lingual
    /// reading. See the module note.
    pub fn say(
        &self,
        text: &str,
        reference: &Reference,
        options: &SpeechOptions,
        report: &mut dyn FnMut(SpeechProgress) -> ControlFlow<()>,
    ) -> Result<Option<Sound>> {
        self.read(text, None, reference, options, report)
    }

    /// `text` continuing `reference`, whose words were `transcript`: upstream's zero-shot reading.
    pub fn say_after(
        &self,
        text: &str,
        transcript: &str,
        reference: &Reference,
        options: &SpeechOptions,
        report: &mut dyn FnMut(SpeechProgress) -> ControlFlow<()>,
    ) -> Result<Option<Sound>> {
        self.read(text, Some(transcript), reference, options, report)
    }

    fn read(
        &self,
        text: &str,
        transcript: Option<&str>,
        reference: &Reference,
        options: &SpeechOptions,
        report: &mut dyn FnMut(SpeechProgress) -> ControlFlow<()>,
    ) -> Result<Option<Sound>> {
        if report(SpeechProgress::Reading).is_break() {
            return Ok(None);
        }

        let seed = options.seed.unwrap_or(0);
        let mut sampler = Sampler::new(seed);
        let mut noise_source = Sampler::new(seed ^ 0x5eed_0f4e_15e0);
        let sampling = Sampling {
            temperature: options.temperature,
            ..self.settings.sampling
        };

        let count = |piece: &str| -> Result<usize> { Ok(self.encode(piece)?.len()) };
        let pieces = frontend::sentences(text, &count)?;
        if pieces.is_empty() {
            return Err(Error::model("there is nothing in the text to say"));
        }

        // What the language model reads, per piece: the ids, and how many of them are the
        // sentence's own -- which is what `inference` works its length limits out from.
        let system = &self.settings.system_prompt;
        let prefix: Vec<i32> = match transcript {
            Some(words) => self.encode(&format!("{system}{}", frontend::transcript(words)))?,
            None => Vec::new(),
        };
        let prompt: &[i32] = match transcript {
            Some(_) => &reference.tokens,
            None => &[],
        };
        let mut readings: Vec<(Vec<i32>, usize)> = Vec::with_capacity(pieces.len());
        for piece in &pieces {
            readings.push(match transcript {
                Some(_) => {
                    let own = self.encode(piece)?;
                    let length = own.len();
                    (prefix.iter().copied().chain(own).collect(), length)
                }
                None => {
                    let ids = self.encode(&format!("{system}{piece}"))?;
                    let length = ids.len();
                    (ids, length)
                }
            });
        }

        // About six speech tokens a text token, for the bar; a reading is free to run past it.
        let expected: i32 = readings.iter().map(|(_, own)| (*own as i32) * 6).sum();

        let mut said_all: Vec<Vec<i32>> = Vec::with_capacity(readings.len());
        let mut done = 0;
        for (ids, own) in &readings {
            let said =
                self.lm
                    .generate(ids, prompt, *own, &sampling, &mut sampler, &mut |count| {
                        report(SpeechProgress::Saying {
                            done: done + count,
                            expected: expected.max(done + count),
                        })
                    })?;
            let Some(said) = said else {
                return Ok(None);
            };
            done += said.len() as i32;
            said_all.push(said);
        }

        if report(SpeechProgress::Sounding).is_break() {
            return Ok(None);
        }

        let mut samples: Vec<f32> = Vec::new();
        for said in &said_all {
            if said.is_empty() {
                continue;
            }
            samples.extend(self.sound(said, reference, options.speed, &mut noise_source)?);
            if report(SpeechProgress::Sounding).is_break() {
                return Ok(None);
            }
        }

        Ok(Some(Sound::new(samples, RATE)))
    }

    /// One reading's tokens, all the way to audio: `token2wav` with `finalize=True`.
    pub fn sound(
        &self,
        said: &[i32],
        reference: &Reference,
        speed: f32,
        noise_source: &mut Sampler,
    ) -> Result<Vec<f32>> {
        let condition = self
            .flow
            .condition(&reference.tokens, said, &reference.speaker)?;
        let prompt_frames = (reference.mel.len() / 80) as i32;
        let mel = self.flow.draw(
            &condition,
            &reference.mel,
            prompt_frames,
            self.settings.diffusion_steps,
            self.settings.cfg_rate,
        )?;

        let frames = mel.len() / 80;
        let mut banded: Vec<f32> = (0..80)
            .flat_map(|band| (0..frames).map(move |frame| frame * 80 + band))
            .map(|index| mel[index])
            .collect();
        let mut frames = frames;
        if speed > 0.0 && (speed - 1.0).abs() > 1e-6 {
            (banded, frames) = stretch(&banded, frames, speed);
        }

        let noise = Noise::draw(frames * hift::HOP, &mut || noise_source.uniform() as f32);
        self.hift.forward(&banded, frames, &noise)
    }
}

/// `F.interpolate(mel, size=int(frames / speed), mode="linear")`, band by band.
fn stretch(mel: &[f32], frames: usize, speed: f32) -> (Vec<f32>, usize) {
    let out = ((frames as f32 / speed) as usize).max(1);
    let scale = frames as f64 / out as f64;
    let mut stretched = vec![0.0f32; 80 * out];
    for band in 0..80 {
        let row = &mel[band * frames..(band + 1) * frames];
        for (index, slot) in stretched[band * out..(band + 1) * out]
            .iter_mut()
            .enumerate()
        {
            let source = ((index as f64 + 0.5) * scale - 0.5).max(0.0);
            let left = (source.floor() as usize).min(frames - 1);
            let right = (left + 1).min(frames - 1);
            let weight = (source - left as f64) as f32;
            *slot = row[left] * (1.0 - weight) + row[right] * weight;
        }
    }
    (stretched, out)
}

impl Voice for CosyVoice3 {
    fn speak(
        &self,
        text: &str,
        like: Option<&Sound>,
        options: &SpeechOptions,
        report: &mut dyn FnMut(SpeechProgress) -> ControlFlow<()>,
    ) -> Result<Option<Sound>> {
        let Some(like) = like else {
            return Err(Error::model(Self::NEEDS_A_RECORDING));
        };
        if report(SpeechProgress::Reading).is_break() {
            return Ok(None);
        }

        let print = fingerprint(like);
        let known = matches!(&*self.heard.borrow(), Some((held, _)) if *held == print);
        if !known {
            let reference = self.listen(like)?;
            *self.heard.borrow_mut() = Some((print, reference));
        }

        let heard = self.heard.borrow();
        let (_, reference) = heard.as_ref().expect("a recording was just listened to");
        self.say(text, reference, options, report)
    }

    fn rate(&self) -> u32 {
        RATE
    }

    fn defaults(&self) -> SpeechDefaults {
        Self::DEFAULTS
    }

    fn name(&self) -> &'static str {
        Self::NAME
    }
}

/// Which recording this is: its rate and every sample's bits, hashed.
fn fingerprint(sound: &Sound) -> u64 {
    let mut hasher = DefaultHasher::new();
    sound.rate.hash(&mut hasher);
    for sample in &sound.samples {
        sample.to_bits().hash(&mut hasher);
    }
    hasher.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stretching_by_one_half_doubles_the_frames() {
        let mel: Vec<f32> = (0..80 * 4).map(|i| i as f32).collect();
        let (out, frames) = stretch(&mel, 4, 0.5);
        assert_eq!(frames, 8);
        assert_eq!(out.len(), 80 * 8);
        assert_eq!(out[0], 0.0);
    }

    #[test]
    fn stretching_leaves_a_steady_band_steady() {
        let mel = vec![2.5f32; 80 * 10];
        let (out, frames) = stretch(&mel, 10, 1.3);
        assert_eq!(frames, 7);
        assert!(out.iter().all(|x| (*x - 2.5).abs() < 1e-6));
    }
}
