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

//! IndexTTS-2.5, whole: a recording and a sentence in, the sentence in that voice out.
//!
//! ```text
//! let tts = IndexTts::from_manifest(Device::Cuda, Residency::Device, &manifest)?;
//! let voice = tts.listen(&recording)?;                  // once per voice
//! let sound = tts.say("今天天气很好。", &voice, &options, &mut report)?;
//! ```
//!
//! Seven graphs, run in the order `infer_v2_5.py` runs them. Each has a module and a document of
//! its own; this is only the order, the shapes between them, and the handful of things upstream
//! does in the gaps that no single module owns.
//!
//! # Listening, which happens once per voice
//!
//! The recording is brought to 22.05 kHz and cut to fifteen seconds -- upstream reads it with
//! `librosa.load`, whose default rate that is -- and from there to 16 kHz. Then:
//!
//! | | reads | gives |
//! | --- | --- | --- |
//! | [`crate::w2v_bert`] | the 16 kHz features, 160 wide | `features`, `(1, T, 1024)`, standardized |
//! | [`crate::campplus`] | Kaldi energies of the 16 kHz audio | `speaker`, `(1, 192)` |
//! | [`crate::indextts_emotion`] | `features` | `emotion`, `(1, 1280)` |
//! | the 22.05 kHz mel | the 22.05 kHz audio | `prompt_mel`, `(1, 80, M)` |
//! | S2Mel's length regulator | `features`, stretched onto `M` frames | `prompt_condition`, `(1, M, 512)` |
//!
//! The emotion comes from the *same* features the speaker's other conditioning does. Upstream
//! takes a separate emotion recording when it is given one and falls back to the speaker's, and
//! `merge_emovec` with the two the same is the speaker's own emotion vector, so a voice here
//! sounds the way its recording felt.
//!
//! # Saying, which happens once per sentence
//!
//! The text is normalized, prefixed with its language, cut into segments of at most 120 tokens,
//! and each segment goes through:
//!
//! 1. [`crate::indextts_gpt::Gpt::generate`] -- speaker, emotion and text in, semantic tokens out.
//! 2. [`crate::semantic_codec::decode`] -- the tokens back to 1024-wide features, two per token.
//! 3. The length regulator, stretching those onto `1.72` mel frames each -- 22 050 / 256 frames
//!    a second over the codec's fifty -- divided by the speed.
//! 4. [`crate::s2mel`], twenty-five Euler steps with guidance at 0.7, the reference mel held in
//!    front as the prompt and cut off the result afterwards.
//! 5. [`crate::bigvgan::BigVgan`] -- the mel to 22.05 kHz audio.
//!
//! Segments are joined with 200 milliseconds of silence between them, as upstream's
//! `interval_silence` does.
//!
//! # Things upstream does in the gaps
//!
//! - **Punctuation is replaced first.** Full-width commas and stops become ASCII and every bracket
//!   an apostrophe -- see [`clean_punctuation`], and why leaving it out made the model keep
//!   talking after a Chinese sentence had ended.
//! - **Ids 0 and 1 are taken out of the text.** `prepare_gpt_inputs` drops every start and stop
//!   token before putting one of each back, and those are ids 0 and 1 -- which are also ordinary
//!   tokens of the vocabulary. So a character that happens to encode to either is silently not
//!   read by upstream, and this does the same rather than reading something upstream never does.
//! - **No beam search.** Upstream samples inside a three-way beam search; [`Gpt::generate`] draws
//!   one sequence with the same temperature, top-k, top-p and repetition penalty.
//! - **The language is guessed from the script.** Upstream is told it. Kana makes a sentence
//!   Japanese, any other CJK character makes it Chinese, and anything else is English -- which is
//!   wrong for Spanish, and the reason [`IndexTts::say_in`] exists.
//! - **Everything runs in `float32`.** Upstream runs the GPT under autocast and S2Mel and the
//!   vocoder in full precision. Narrowing is a thing to measure before doing.

use std::cell::RefCell;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::ops::ControlFlow;
use std::rc::Rc;

use crate::bigvgan::{BigVgan, BigVganConfig};
use crate::flint::{
    functional as F, DType, Device, Graph, Ir, ParamSource, Residency, RunContext, Tensor,
};
use crate::indextts_gpt::{Gpt, Sampling};
use crate::indextts_normalize::{self, Language};
use crate::{
    campplus, indextts_emotion, indextts_features, indextts_gpt, s2mel, semantic_codec, w2v_bert,
    Error, Manifest, Result, Sound, SpeechDefaults, SpeechOptions, SpeechProgress, Tokenizer,
    Voice,
};

/// Where each of the six models sits in the package. See `tools/indextts_package.py`.
const GPT: &str = "indextts2.gpt";
const W2V_BERT: &str = "indextts2.w2v_bert";
const CODEC: &str = "indextts2.codec";
const S2MEL: &str = "indextts2.s2mel";
const CAMPPLUS: &str = "indextts2.campplus";
const BIGVGAN: &str = "indextts2.bigvgan";

/// The rate everything after the GPT works at, and the rate of what comes out.
pub const RATE: u32 = 22050;

/// The rate w2v-bert and CAMPPlus listen at.
const LISTENING_RATE: u32 = 16000;

/// How many rows the released sinusoid table has. See [`indextts_emotion::positional_encoding`]
/// for why the table is read out of the package rather than rebuilt.
const POSITION_ROWS: i32 = 5000;

/// What a reading is steered by, as the manifest's `indextts2:` section states it.
#[derive(Clone, Copy, Debug)]
pub struct Settings {
    pub temperature: f32,
    pub top_k: i32,
    pub top_p: f32,
    pub repetition_penalty: f32,
    pub max_mel_tokens: i32,
    pub diffusion_steps: i32,
    pub cfg_rate: f32,
    /// Mel frames per decoded codec frame: 22 050 / 256 over fifty.
    pub length_ratio: f32,
    pub max_text_tokens_per_segment: usize,
    pub reference_seconds: f32,
}

impl Settings {
    /// `infer_v2_5.py`'s own defaults, which is also what the package writes.
    pub fn indextts() -> Settings {
        Settings {
            temperature: 0.8,
            top_k: 30,
            top_p: 0.8,
            repetition_penalty: 10.0,
            max_mel_tokens: 1500,
            diffusion_steps: 25,
            cfg_rate: 0.7,
            length_ratio: 1.72,
            max_text_tokens_per_segment: 120,
            reference_seconds: 15.0,
        }
    }

    fn from_manifest(manifest: &Manifest) -> Result<Settings> {
        let mut settings = Settings::indextts();
        let Ok(section) = manifest.section(IndexTts::MODEL_TYPE) else {
            return Ok(settings);
        };

        let float = |key: &str, into: &mut f32| {
            if let Ok(value) = section.get::<f32>(key) {
                *into = value;
            }
        };
        float("temperature", &mut settings.temperature);
        float("top_p", &mut settings.top_p);
        float("repetition_penalty", &mut settings.repetition_penalty);
        float("cfg_rate", &mut settings.cfg_rate);
        float("length_ratio", &mut settings.length_ratio);
        float("reference_seconds", &mut settings.reference_seconds);

        let int = |key: &str, into: &mut i32| {
            if let Ok(value) = section.get::<i32>(key) {
                *into = value;
            }
        };
        int("top_k", &mut settings.top_k);
        int("max_mel_tokens", &mut settings.max_mel_tokens);
        int("diffusion_steps", &mut settings.diffusion_steps);

        if let Ok(value) = section.get::<i32>("max_text_tokens_per_segment") {
            settings.max_text_tokens_per_segment = value.max(1) as usize;
        }

        Ok(settings)
    }
}

/// Everything a recording says about a voice, worked out once and kept for every sentence.
pub struct Reference {
    /// w2v-bert's features, standardized: `(1, frames, 1024)`.
    pub features: Tensor,
    pub frames: i32,
    /// CAMPPlus's embedding, `(1, 192)`.
    pub speaker: Tensor,
    /// The emotion row, `(1, 1280)`.
    pub emotion: Tensor,
    /// The recording's own 22.05 kHz mel, `(1, 80, prompt_frames)`, which S2Mel continues from.
    pub prompt_mel: Tensor,
    pub prompt_frames: i32,
    /// `features` stretched onto the mel's frames by the length regulator, `(1, prompt_frames, 512)`.
    pub prompt_condition: Tensor,
}

/// IndexTTS-2.5, with every model it runs read and waiting.
pub struct IndexTts {
    settings: Settings,
    weights: Rc<dyn ParamSource>,
    gpt: Gpt,
    vocoder: BigVgan,
    tokenizer: Tokenizer,
    device: Device,
    dtype: DType,
    /// The last recording listened to, by fingerprint, and what was heard in it. The page sends
    /// the same recording with every sentence, and listening again would spend a second of the
    /// card on an answer that is already here.
    heard: RefCell<Option<(u64, Reference)>>,
}

impl IndexTts {
    pub const MODEL_TYPE: &'static str = "indextts2";

    /// What the screen calls it.
    pub const NAME: &'static str = "IndexTTS-2.5";

    /// What the speech tab's boxes start at before any weights are read: upstream's temperature,
    /// and speech at its own pace. Once the model is read, the temperature is the manifest's.
    pub const DEFAULTS: SpeechDefaults = SpeechDefaults {
        speed: 1.0,
        temperature: 0.8,
    };

    /// Why there is nothing to say without a recording. Upstream takes the speaker's audio as a
    /// required argument; there is no voice of its own for this to fall back on.
    pub const NEEDS_A_RECORDING: &'static str = "IndexTTS-2.5 speaks in the voice of a recording \
         -- drop one on the page, a few seconds of somebody speaking, and say the sentence again";

    /// Read the whole model `manifest` describes, onto `device`.
    pub fn from_manifest(
        device: Device,
        residency: Residency,
        manifest: &Manifest,
    ) -> Result<IndexTts> {
        let model_type = manifest.section("model")?.get_str("type")?.to_string();
        if model_type != Self::MODEL_TYPE {
            return Err(Error::model(format!(
                "this manifest describes a model of kind {model_type:?}, which is not {:?}",
                Self::MODEL_TYPE
            )));
        }

        // Every model here runs in full precision; see the module note.
        let dtype = DType::Float;
        let weights = residency.read(manifest.params()?, device)?;

        Ok(IndexTts {
            settings: Settings::from_manifest(manifest)?,
            gpt: Gpt::build(
                indextts_gpt::Config::indextts(),
                GPT,
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

    pub fn device(&self) -> Device {
        self.device
    }

    /// Compile `g`, read what it needs, run it with `inputs`, and hand back its outputs in the
    /// order they were declared.
    ///
    /// Most of the graphs here have a length built into them and are compiled once per recording
    /// or per sentence. That is cheap: under [`Residency::Device`] a weight is moved onto the card
    /// once, and a second graph asking for it gets the tensor that is already there.
    fn run(&self, g: &Graph, inputs: &[(&str, &Tensor)]) -> Result<Vec<Tensor>> {
        let ir = Ir::compile(g, self.weights.residency());
        let preloaded = ir.load(self.weights.as_ref())?;

        let mut context = RunContext::new(self.weights.as_ref()).preloaded(&preloaded);
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

    /// Everything the voice in `recording` contributes, worked out once.
    pub fn listen(&self, recording: &Sound) -> Result<Reference> {
        // Upstream reads the recording at 22.05 kHz and cuts it there, then resamples the cut.
        let mut wave = indextts_features::resample(&recording.samples, recording.rate, RATE);
        wave.truncate((self.settings.reference_seconds * RATE as f32) as usize);
        let listening = indextts_features::resample(&wave, RATE, LISTENING_RATE);

        let (features, frames) = self.features(&listening)?;
        let speaker = self.speaker(&listening)?;
        let emotion = self.emotion(&features, frames)?;

        let (mel, prompt_frames) = indextts_features::reference_mel(&wave)?;
        let prompt_frames = prompt_frames as i32;
        let prompt_mel = self.upload(&[1, 80, prompt_frames], &mel)?;
        let prompt_condition = self.regulate(&features, frames, prompt_frames)?;

        Ok(Reference {
            features,
            frames,
            speaker,
            emotion,
            prompt_mel,
            prompt_frames,
            prompt_condition,
        })
    }

    /// w2v-bert's `hidden_states[17]`, standardized by the release's own mean and deviation.
    fn features(&self, wave: &[f32]) -> Result<(Tensor, i32)> {
        let (values, pairs) = indextts_features::w2v_bert(wave);
        let frames = pairs as i32;
        if frames < 3 {
            return Err(Error::model(
                "the recording is too short to hear a voice in -- a fraction of a second at least",
            ));
        }

        let config = w2v_bert::Config::w2v_bert_2();
        let distances = w2v_bert::distance_indices(frames, &config, self.device)?;
        let input = self.upload(&[1, frames, config.feature_dim], &values)?;

        let g = Graph::new();
        let sub = g.subgraph(W2V_BERT);
        let hidden = w2v_bert::graph(
            &sub,
            g.input("x"),
            &config,
            frames,
            w2v_bert::Config::USED_LAYERS,
            g.input("distances"),
            self.dtype,
            self.device,
        )?;
        let stats = sub.subgraph("semantic");
        let mean = stats.load("mean", &[config.hidden_size]);
        let std = stats.load("std", &[config.hidden_size]);
        g.output("features", g.div(g.sub(hidden, mean), std));

        let mut out = self.run(&g, &[("x", &input), ("distances", &distances)])?;

        Ok((out.remove(0), frames))
    }

    /// CAMPPlus's embedding of who is speaking.
    fn speaker(&self, wave: &[f32]) -> Result<Tensor> {
        let (values, frames) = indextts_features::campplus(wave);
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
        g.output("speaker", embedding);

        Ok(self.run(&g, &[("x", &input)])?.remove(0))
    }

    /// The emotion row, from the same features, against the release's own position table.
    fn emotion(&self, features: &Tensor, frames: i32) -> Result<Tensor> {
        let config = indextts_emotion::Config::indextts();
        let rows = indextts_emotion::Config::subsampled(frames);

        let g = Graph::new();
        let sub = g.subgraph(GPT);
        let table = sub
            .subgraph("emo_conditioning_encoder")
            .subgraph("embed")
            .subgraph("pos_enc")
            .load("pe", &[1, POSITION_ROWS, config.encoder_dim]);
        let positions = g.contiguous(g.slice(table, 1, 0, rows));

        let emotion = indextts_emotion::graph(
            &sub,
            g.input("x"),
            &config,
            frames,
            positions,
            self.dtype,
            self.device,
        )?;
        g.output("emotion", emotion);

        Ok(self.run(&g, &[("x", features)])?.remove(0))
    }

    /// S2Mel's length regulator: `(1, from, 1024)` features stretched onto `to` mel frames.
    fn regulate(&self, features: &Tensor, from: i32, to: i32) -> Result<Tensor> {
        let selection = s2mel::nearest_selection(from, to, self.device)?.cast(self.dtype)?;

        let g = Graph::new();
        let regulated = s2mel::length_regulator(
            &g.subgraph(S2MEL).subgraph("length_regulator"),
            g.input("x"),
            g.input("selection"),
            &s2mel::RegulatorConfig::indextts(),
            to,
            self.dtype,
            self.device,
        )?;
        g.output("regulated", regulated);

        Ok(self
            .run(&g, &[("x", features), ("selection", &selection)])?
            .remove(0))
    }

    /// The semantic tokens back to the codec's features, two frames each.
    fn decode(&self, codes: &[i32]) -> Result<Tensor> {
        let tokens = codes.len() as i32;
        let ids: Vec<i64> = codes.iter().map(|code| i64::from(*code)).collect();
        let ids = Tensor::from_i64(&[tokens], &ids)?.to_device(self.device)?;

        let g = Graph::new();
        let features = semantic_codec::decode(
            &g.subgraph(CODEC),
            g.input("codes"),
            &semantic_codec::Config::indextts(),
            tokens,
            self.dtype,
            self.device,
        )?;
        g.output("features", features);

        Ok(self.run(&g, &[("codes", &ids)])?.remove(0))
    }

    /// Twenty-five steps of S2Mel over `condition`, continuing `reference`'s mel. `(1, 80, frames)`
    /// with the prompt's frames still on the front.
    fn draw_mel(&self, reference: &Reference, condition: &Tensor, frames: i32) -> Result<Tensor> {
        let config = s2mel::Config::indextts();

        let g = Graph::new();
        let direction = s2mel::graph(
            &g.subgraph(S2MEL),
            g.input("x"),
            g.input("prompt_x"),
            g.input("cond"),
            g.input("style"),
            g.input("time"),
            g.input("time"),
            &config,
            frames,
            g.input("cos"),
            g.input("sin"),
            self.dtype,
            self.device,
        )?;
        g.output("direction", direction);

        let ir = Ir::compile(&g, self.weights.residency());
        let preloaded = ir.load(self.weights.as_ref())?;

        let (cos, sin) = s2mel::rotary_tables(
            frames,
            config.hidden_dim / config.num_heads,
            10000.0,
            self.device,
        )?;
        let (cos, sin) = (cos.cast(self.dtype)?, sin.cast(self.dtype)?);

        // What the unguided pass is handed instead: no content, no speaker. The prompt it is
        // handed is `solve_euler`'s to zero.
        let empty_condition = self.upload(
            &[1, frames, config.content_dim],
            &vec![0.0; (frames * config.content_dim) as usize],
        )?;
        let empty_style = self.upload(
            &[1, config.style_dim],
            &vec![0.0; config.style_dim as usize],
        )?;

        let noise = F::randn(&[1, config.in_channels, frames], self.device)?.cast(self.dtype)?;

        s2mel::solve_euler(
            &noise,
            &reference.prompt_mel,
            reference.prompt_frames,
            self.settings.diffusion_steps,
            self.settings.cfg_rate,
            |x, prompt_x, t, guided| {
                let time =
                    s2mel::timestep_embedding(t, config.frequency_embedding_size, self.device)?
                        .cast(self.dtype)?;
                let (cond, style) = match guided {
                    true => (condition, &reference.speaker),
                    false => (&empty_condition, &empty_style),
                };

                let context = RunContext::new(self.weights.as_ref())
                    .preloaded(&preloaded)
                    .input("x", x)
                    .input("prompt_x", prompt_x)
                    .input("cond", cond)
                    .input("style", style)
                    .input("time", &time)
                    .input("cos", &cos)
                    .input("sin", &sin);

                Ok(ir.run(&context)?.remove(0).1)
            },
        )
    }

    /// `text` in the voice of `reference`, the language guessed from the text's script.
    pub fn say(
        &self,
        text: &str,
        reference: &Reference,
        options: &SpeechOptions,
        report: &mut dyn FnMut(SpeechProgress) -> ControlFlow<()>,
    ) -> Result<Option<Sound>> {
        self.say_in(text, guess_language(text), reference, options, report)
    }

    /// `text` in the voice of `reference`, read as `language`.
    pub fn say_in(
        &self,
        text: &str,
        language: Language,
        reference: &Reference,
        options: &SpeechOptions,
        report: &mut dyn FnMut(SpeechProgress) -> ControlFlow<()>,
    ) -> Result<Option<Sound>> {
        if let Some(seed) = options.seed {
            F::manual_seed(self.device, seed)?;
        }
        if report(SpeechProgress::Reading).is_break() {
            return Ok(None);
        }

        let segments = self.segments(text, language)?;
        let expected: i32 = segments
            .iter()
            .map(|ids| (ids.len() as i32 * 5).clamp(20, self.settings.max_mel_tokens))
            .sum();

        let sampling = Sampling {
            temperature: options.temperature,
            top_k: self.settings.top_k,
            top_p: self.settings.top_p,
            repetition_penalty: self.settings.repetition_penalty,
            max_tokens: self.settings.max_mel_tokens,
        };

        // Every segment's tokens first, so that the bar moves through one reading of the text and
        // the stop button is honoured before any of the slower work starts.
        let mut codes: Vec<Vec<i32>> = Vec::with_capacity(segments.len());
        let mut done = 0;
        for ids in &segments {
            let said = self.gpt.generate(
                &reference.speaker,
                &reference.emotion,
                ids,
                language_id(language),
                &sampling,
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
            codes.push(said);
        }

        if report(SpeechProgress::Sounding).is_break() {
            return Ok(None);
        }

        let silence = vec![0.0f32; (RATE as f32 * 0.2) as usize];
        let mut samples: Vec<f32> = Vec::new();
        for (index, said) in codes.iter().enumerate() {
            if said.is_empty() {
                continue;
            }

            let wave = self.sound(said, reference, options.speed)?;
            if index > 0 && !samples.is_empty() {
                samples.extend_from_slice(&silence);
            }
            samples.extend(wave);

            if report(SpeechProgress::Sounding).is_break() {
                return Ok(None);
            }
        }

        Ok(Some(Sound::new(samples, RATE)))
    }

    /// One segment's tokens, all the way to audio.
    fn sound(&self, codes: &[i32], reference: &Reference, speed: f32) -> Result<Vec<f32>> {
        let decoded = self.decode(codes)?;
        let decoded_frames = decoded.shape_at(1)?;

        let speed = if speed > 0.0 { speed } else { 1.0 };
        let target = ((decoded_frames as f32 * self.settings.length_ratio / speed) as i32).max(1);
        let condition = self.regulate(&decoded, decoded_frames, target)?;

        // The prompt's condition in front of the sentence's, which is what S2Mel continues.
        let frames = reference.prompt_frames + target;
        let joined = concatenate_frames(&reference.prompt_condition, &condition)?;
        let joined = self.upload(&[1, frames, joined.1], &joined.0)?;

        let mel = self.draw_mel(reference, &joined, frames)?;
        let mel = mel
            .slice(2, reference.prompt_frames, frames)?
            .contiguous()?;

        let wave = self.vocoder.forward(&mel)?;
        let samples = wave
            .to_device(Device::Cpu)?
            .cast(DType::Float)?
            .to_vec_f32()?;

        Ok(samples
            .into_iter()
            .map(|sample| sample.clamp(-1.0, 1.0))
            .collect())
    }

    /// `text`, normalized and cut into segments of token ids, each with its language prefix and
    /// without ids 0 and 1 -- see the module note.
    fn segments(&self, text: &str, language: Language) -> Result<Vec<Vec<i32>>> {
        let text = clean_punctuation(text);
        let normalized = match language {
            Language::Japanese => text,
            _ => indextts_normalize::normalize(&text, language),
        };
        let normalized = match language {
            Language::Spanish => normalized.to_uppercase(),
            _ => normalized.to_lowercase(),
        };

        let prefix = format!("<|{}|> ", language_tag(language));
        let prefix_length = self.tokenizer.encode(&prefix)?.len();
        let budget = self
            .settings
            .max_text_tokens_per_segment
            .saturating_sub(prefix_length)
            .max(1);

        let count = |piece: &str| -> Result<usize> { Ok(self.tokenizer.encode(piece)?.len()) };

        let mut segments = Vec::new();
        for segment in pack(&normalized, budget, &count)? {
            let ids: Vec<i32> = self
                .tokenizer
                .encode(&format!("{prefix}{segment}"))?
                .into_iter()
                .filter(|id| *id > 1)
                .collect();

            if !ids.is_empty() {
                segments.push(ids);
            }
        }

        if segments.is_empty() {
            return Err(Error::model("there is nothing in the text to say"));
        }

        Ok(segments)
    }
}

impl Voice for IndexTts {
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
        SpeechDefaults {
            temperature: self.settings.temperature,
            ..Self::DEFAULTS
        }
    }

    fn name(&self) -> &'static str {
        Self::NAME
    }
}

/// Which recording this is, for [`IndexTts`]'s memory of the last one: its rate and every
/// sample's bits, hashed. Not a content hash anybody should rely on across builds -- only a way to
/// tell this recording from the one before it.
fn fingerprint(sound: &Sound) -> u64 {
    let mut hasher = DefaultHasher::new();
    sound.rate.hash(&mut hasher);
    for sample in &sound.samples {
        sample.to_bits().hash(&mut hasher);
    }

    hasher.finish()
}

/// Upstream's `char_rep_map`, in its own order: full-width punctuation to the ASCII the GPT was
/// trained on, and every kind of bracket and quotation mark to an apostrophe.
///
/// **This is not optional.** Without it a Chinese sentence reaches the model ending in `。`, which
/// it has hardly seen, and it says something more after the sentence has ended -- Whisper heard
/// "今天天气很好，我们一起去公园散步吧" followed by "法尔" before this was here, and nothing after
/// it.
const PUNCTUATION: &[(&str, &str)] = &[
    ("：", ","),
    ("；", ","),
    (";", ","),
    ("，", ","),
    ("。", "."),
    ("！", "!"),
    ("？", "?"),
    ("\n", " "),
    ("·", "-"),
    ("、", ","),
    ("...", "…"),
    (",,,", "…"),
    ("，，，", "…"),
    ("……", "…"),
    ("“", "'"),
    ("”", "'"),
    ("\"", "'"),
    ("‘", "'"),
    ("’", "'"),
    ("（", "'"),
    ("）", "'"),
    ("(", "'"),
    (")", "'"),
    ("《", "'"),
    ("》", "'"),
    ("【", "'"),
    ("】", "'"),
    ("[", "'"),
    ("]", "'"),
    ("—", "-"),
    ("～", "-"),
    ("~", "-"),
    ("「", "'"),
    ("」", "'"),
    (":", ","),
];

/// [`PUNCTUATION`] applied the way upstream's regular expression applies it: at each position the
/// entries are tried in order and the first that matches wins.
///
/// Which has a consequence that looks like a mistake and is faithful: `，，，` is listed as an
/// ellipsis, but `，` is listed before it and matches first, so three full-width commas become
/// three commas and never an ellipsis. The same alternation in Python does exactly that.
pub fn clean_punctuation(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;

    'outer: while !rest.is_empty() {
        for (from, to) in PUNCTUATION {
            if let Some(after) = rest.strip_prefix(from) {
                out.push_str(to);
                rest = after;
                continue 'outer;
            }
        }

        let character = rest.chars().next().expect("rest is not empty");
        out.push(character);
        rest = &rest[character.len_utf8()..];
    }

    out
}

/// Two `(1, frames, width)` tensors end to end along the frames, on the host: `(values, width)`.
fn concatenate_frames(first: &Tensor, second: &Tensor) -> Result<(Vec<f32>, i32)> {
    let width = first.shape_at(2)?;
    if second.shape_at(2)? != width {
        return Err(Error::model(
            "two conditions of different widths cannot be joined",
        ));
    }

    let mut values = first
        .to_device(Device::Cpu)?
        .cast(DType::Float)?
        .to_vec_f32()?;
    values.extend(
        second
            .to_device(Device::Cpu)?
            .cast(DType::Float)?
            .to_vec_f32()?,
    );

    Ok((values, width))
}

/// `text` packed into pieces of at most `budget` tokens, breaking after punctuation where it can
/// and between characters where it must -- upstream's `split_text_by_tokens`.
fn pack(text: &str, budget: usize, count: &dyn Fn(&str) -> Result<usize>) -> Result<Vec<String>> {
    if count(text)? <= budget {
        return Ok(vec![text.to_string()]);
    }

    const BREAKS: &[char] = &[
        '，', '。', '！', '？', '、', '；', '：', ',', '.', '!', '?', ';', ':', '\n',
    ];

    // Pieces that end at a break, each small enough on its own.
    let mut chunks: Vec<String> = Vec::new();
    let mut current = String::new();
    for character in text.chars() {
        current.push(character);
        if BREAKS.contains(&character) {
            chunks.push(std::mem::take(&mut current));
        }
    }
    if !current.is_empty() {
        chunks.push(current);
    }

    let mut small: Vec<String> = Vec::new();
    for chunk in chunks {
        if count(&chunk)? <= budget {
            small.push(chunk);
            continue;
        }

        let mut current = String::new();
        for character in chunk.chars() {
            let mut next = current.clone();
            next.push(character);
            if !current.is_empty() && count(&next)? > budget {
                small.push(std::mem::replace(&mut current, character.to_string()));
            } else {
                current = next;
            }
        }
        if !current.is_empty() {
            small.push(current);
        }
    }

    // And packed back together as long as they fit.
    let mut segments: Vec<String> = Vec::new();
    let mut current = String::new();
    for chunk in small {
        let joined = format!("{current}{chunk}");
        if !current.is_empty() && count(&joined)? > budget {
            segments.push(std::mem::replace(&mut current, chunk));
        } else {
            current = joined;
        }
    }
    if !current.is_empty() {
        segments.push(current);
    }

    Ok(segments)
}

/// Kana is Japanese, any other CJK ideograph is Chinese, and everything else is English.
///
/// Upstream is told the language rather than guessing it, and this is a guess: it cannot tell
/// Spanish from English, which is what [`IndexTts::say_in`] is for.
pub fn guess_language(text: &str) -> Language {
    let kana = |c: char| matches!(c, '\u{3040}'..='\u{30ff}');
    let ideograph = |c: char| matches!(c, '\u{4e00}'..='\u{9fff}' | '\u{3400}'..='\u{4dbf}');

    if text.chars().any(kana) {
        Language::Japanese
    } else if text.chars().any(ideograph) {
        Language::Chinese
    } else {
        Language::English
    }
}

/// The tag the tokenizer's language token is spelled with.
fn language_tag(language: Language) -> &'static str {
    match language {
        Language::Chinese => "zh",
        Language::English => "en",
        Language::Spanish => "es",
        Language::Japanese => "ja",
    }
}

/// The row of `lang_embedding` a language reads: its index in Whisper's `LANGUAGES` table, which
/// is how `lang_to_token` numbers them.
pub fn language_id(language: Language) -> i32 {
    match language {
        Language::English => 0,
        Language::Chinese => 1,
        Language::Spanish => 3,
        Language::Japanese => 7,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A token counter that counts characters, which is enough to test the packing.
    fn characters(text: &str) -> Result<usize> {
        Ok(text.chars().count())
    }

    #[test]
    fn a_short_text_is_one_segment() {
        assert_eq!(pack("你好。", 10, &characters).unwrap(), vec!["你好。"]);
    }

    #[test]
    fn a_long_text_breaks_after_punctuation() {
        // Three sentences of four, against a budget of seven: no two fit together.
        let segments = pack("一二三。四五六。七八九。", 7, &characters).unwrap();
        assert_eq!(segments, vec!["一二三。", "四五六。", "七八九。"]);

        // Against a budget of eight, the first two do.
        let segments = pack("一二三。四五六。七八九。", 8, &characters).unwrap();
        assert_eq!(segments, vec!["一二三。四五六。", "七八九。"]);
    }

    #[test]
    fn a_run_with_no_punctuation_breaks_between_characters() {
        let segments = pack("一二三四五六七八九十", 4, &characters).unwrap();

        assert!(segments.iter().all(|segment| segment.chars().count() <= 4));
        assert_eq!(segments.concat(), "一二三四五六七八九十");
    }

    #[test]
    fn the_script_decides_the_language() {
        assert_eq!(guess_language("今天天气很好"), Language::Chinese);
        assert_eq!(guess_language("こんにちは、世界"), Language::Japanese);
        assert_eq!(guess_language("Hello there"), Language::English);
    }

    /// Against upstream's own `clean_pattern.sub(...)` over the same string: the table read out
    /// of `indextts/utils/front.py` with `ast` and the regular expression run in Python. The three
    /// full-width commas are the case to look at -- see [`clean_punctuation`].
    #[test]
    fn punctuation_is_replaced_the_way_upstream_replaces_it() {
        let probe = "你好，世界。真的吗？是的！“引号”（括号）《书名》【方括号】、顿号；分号：冒号…\
                     省略……两个...三点,,,三逗号，，，三全角—破折～波浪~\n换行·间隔「角」[方](圆)\
                     \"双\"'单'";
        let want = "你好,世界.真的吗?是的!'引号''括号''书名''方括号',顿号,分号,冒号…\
                    省略…两个…三点…三逗号,,,三全角-破折-波浪- 换行-间隔'角''方''圆'\
                    '双''单'";

        assert_eq!(clean_punctuation(probe), want);
    }

    #[test]
    fn the_language_ids_are_whispers() {
        assert_eq!(language_id(Language::English), 0);
        assert_eq!(language_id(Language::Chinese), 1);
        assert_eq!(language_id(Language::Spanish), 3);
        assert_eq!(language_id(Language::Japanese), 7);
    }
}
