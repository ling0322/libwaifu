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

//! Qwen3-ASR: a recording in, what was said in it and in which language out.
//!
//! [`Qwen/Qwen3-ASR-0.6B`](https://huggingface.co/Qwen/Qwen3-ASR-0.6B), under Apache 2.0, read the
//! way upstream's `qwen-asr` package reads it (see docs/qwen3_asr.md):
//!
//! 1. The recording is resampled to 16 kHz, and one longer than twenty minutes is cut into pieces
//!    at the quietest point near each twenty-minute mark, as `split_audio_into_chunks` does.
//! 2. Each piece is heard as Whisper's 128-band mel ([`features`]) and encoded into a row per
//!    80 ms ([`audio_encoder`]).
//! 3. The rows take the place of `<|audio_pad|>` in a chat prompt, and Qwen3-0.6B ([`decoder`])
//!    writes greedily after it: `language Chinese<asr_text>` and then the words.
//! 4. That is parsed as upstream's `parse_asr_output` parses it, a run of twenty repeats cut to
//!    one, and the pieces' words joined.

pub mod audio_encoder;
pub mod config;
pub mod decoder;
pub mod features;

use std::cell::Cell;
use std::ops::ControlFlow;
use std::rc::Rc;

use crate::error::Error;
use crate::flint::{functional as F, DType, Device, Graph, Ir, RunContext, Tensor, Weights};
use crate::indextts::features::resample;
use crate::tokenizer::Tokenizer;
use crate::transcription::{Transcriber, Transcript, TranscriptionOptions, TranscriptionProgress};
use crate::wav::Sound;
use crate::{Manifest, Result};

use audio_encoder::{Geometry, Windows};
use config::Config;
use decoder::{argmax, Decoder};
use features::Features;

/// The namespace every weight is under.
const NAMESPACE: &str = "qwen3_asr";

/// What separates the language from the words in what the model writes.
pub const ASR_TEXT: &str = "<asr_text>";

/// The shortest piece upstream reads: anything shorter is padded with silence to this.
const SHORTEST_SECONDS: f32 = 0.5;

/// Qwen3-ASR with its encoder and decoder read and waiting.
pub struct Qwen3Asr {
    config: Config,
    weights: Rc<Weights>,
    decoder: Decoder,
    features: Features,
    tokenizer: Tokenizer,
    device: Device,
    dtype: DType,
    /// How the encoder keeps attention inside its windows: under a block diagonal mask on CUDA,
    /// until its kernel refuses one -- a card before Ampere -- and split into batches from then on
    /// and on every other device. See [`audio_encoder::Windows`].
    windows: Cell<Windows>,
}

/// What one piece of a recording came to, before the pieces are joined.
#[derive(Clone, Debug, PartialEq)]
pub struct Heard {
    /// The ids the decoder wrote, without the end token.
    pub said: Vec<i32>,
    /// Those ids as text, special tokens dropped: `language Chinese<asr_text>...`.
    pub raw: String,
}

impl Qwen3Asr {
    pub const MODEL_TYPE: &'static str = "qwen3_asr";

    /// What the screen calls it.
    pub const NAME: &'static str = "Qwen3-ASR";

    /// Read the model `manifest` describes, onto `device`.
    pub fn from_manifest(
        device: Device,
        device_weight_bytes: u64,
        manifest: &Manifest,
    ) -> Result<Qwen3Asr> {
        let model_type = manifest.section("model")?.get_str("type")?.to_string();
        if model_type != Self::MODEL_TYPE {
            return Err(Error::model(format!(
                "this manifest describes a model of kind {model_type:?}, which is not {:?}",
                Self::MODEL_TYPE
            )));
        }
        let config = Config::from_manifest(manifest)?;
        let tied = manifest
            .section(Self::MODEL_TYPE)?
            .get_bool_or("tied_head", true)?;

        let dtype = F::default_float_type(device)?;
        let weights: Rc<Weights> = Rc::new(Weights::from_files(
            &manifest.weight_paths()?,
            device,
            device_weight_bytes,
        )?);

        Ok(Qwen3Asr {
            decoder: Decoder::build(
                config.text.clone(),
                NAMESPACE,
                tied,
                &weights,
                dtype,
                device,
            )?,
            features: Features::new(config.audio.mel_bins as usize, device)?,
            tokenizer: Tokenizer::open(manifest)?,
            config,
            weights,
            windows: Cell::new(match device {
                Device::Cuda => Windows::Masked,
                _ => Windows::Split,
            }),
            device,
            dtype,
        })
    }

    pub fn config(&self) -> &Config {
        &self.config
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    pub fn decoder(&self) -> &Decoder {
        &self.decoder
    }

    /// The features of a 16 kHz recording, `(1, mel_bins, frames)`, float, on the device.
    pub fn mel(&self, wave: &[f32]) -> Result<Tensor> {
        self.features.mel(wave)
    }

    /// The encoder over a `(1, mel_bins, frames)` mel: `(tokens, output_dim)` on the device.
    ///
    /// Under a block diagonal mask where the device has a kernel for one, and the windows split
    /// into batches where it has not: a mask the kernel refuses is tried once, and from then on
    /// this does without it.
    pub fn encode(&self, mel: &Tensor) -> Result<Tensor> {
        let windows = self.windows.get();
        match self.encode_with(mel, windows) {
            Err(crate::Error::Tensor(error))
                if windows == Windows::Masked && error.is_aborted() =>
            {
                crate::log::write(
                    crate::log::Level::Info,
                    &format!(
                        "no kernel takes the encoder's block diagonal mask here ({}); its \
                         windows are split from now on",
                        error.message()
                    ),
                );
                self.windows.set(Windows::Split);
                self.encode_with(mel, Windows::Split)
            }
            encoded => encoded,
        }
    }

    /// The encoder, keeping attention inside its windows the way `windows` says.
    pub fn encode_with(&self, mel: &Tensor, windows: Windows) -> Result<Tensor> {
        let frames = mel.shape_at(2)?;
        let geometry = Geometry::of(frames, &self.config.audio);
        let positions = audio_encoder::positions(geometry.per_chunk, self.config.audio.d_model)?
            .to_device(self.device)?
            .cast(self.dtype)?;

        let g = Graph::new();
        let out = audio_encoder::graph(
            &g.subgraph(NAMESPACE).subgraph("audio_tower"),
            g.input("mel"),
            g.input("positions"),
            &geometry,
            &self.config.audio,
            self.dtype,
            windows,
        );
        g.output("audio", out);

        let context = RunContext::new(self.weights.as_ref())
            .input("mel", mel)
            .input("positions", &positions);
        Ir::compile(&g)
            .run(&context)?
            .into_iter()
            .next()
            .map(|(_, tensor)| tensor)
            .ok_or_else(|| Error::model("the audio encoder produced nothing"))
    }

    /// The prompt on either side of `<|audio_pad|>`, as ids: upstream's chat template with the
    /// context as the system turn, and -- where a language is given -- `language X<asr_text>`
    /// after the assistant's turn opens, so that what is written is the words alone.
    pub fn prompt(&self, options: &TranscriptionOptions) -> Result<(Vec<i32>, Vec<i32>)> {
        let head = format!(
            "<|im_start|>system\n{}<|im_end|>\n<|im_start|>user\n<|audio_start|>",
            options.context
        );
        let mut tail = "<|audio_end|><|im_end|>\n<|im_start|>assistant\n".to_string();
        if let Some(language) = &options.language {
            tail.push_str(&format!("language {language}{ASR_TEXT}"));
        }
        Ok((self.tokenizer.encode(&head)?, self.tokenizer.encode(&tail)?))
    }

    /// The language a caller named, as the model names it -- upstream's `normalize_language_name`,
    /// first letter up and the rest down -- or why it is not one this knows.
    pub fn language(&self, asked: &str) -> Result<String> {
        let asked = asked.trim();
        let mut letters = asked.chars();
        let named = match letters.next() {
            Some(first) => first
                .to_uppercase()
                .chain(letters.flat_map(char::to_lowercase)),
            None => return Err(Error::model("the language is empty")),
        }
        .collect::<String>();
        match self.config.languages.contains(&named) {
            true => Ok(named),
            false => Err(Error::model(format!(
                "{} does not know {asked:?}: it knows {}",
                Self::NAME,
                self.config.languages.join(", ")
            ))),
        }
    }

    /// One piece of 16 kHz recording, heard and written out greedily -- or `None` where `report`
    /// asked to stop. `report` is handed the count of tokens so far.
    pub fn hear(
        &self,
        wave: &[f32],
        options: &TranscriptionOptions,
        report: &mut dyn FnMut(i32) -> ControlFlow<()>,
    ) -> Result<Option<Heard>> {
        let mel = self.mel(wave)?;
        let audio = self.encode(&mel)?;
        let (head, tail) = self.prompt(options)?;

        let mut reading = self.decoder.prefill(&head, &audio, &tail)?;
        let mut said = Vec::new();
        for _ in 0..self.config.max_new_tokens {
            let token = argmax(&reading.scores()?) as i32;
            if self.config.end_token_ids.contains(&token) {
                break;
            }
            said.push(token);
            if report(said.len() as i32).is_break() {
                return Ok(None);
            }
            reading = self.decoder.step(&reading, token)?;
        }

        let raw = self.tokenizer.decode(&said, true)?;
        Ok(Some(Heard { said, raw }))
    }
}

impl Transcriber for Qwen3Asr {
    fn transcribe(
        &self,
        recording: &Sound,
        options: &TranscriptionOptions,
        report: &mut dyn FnMut(TranscriptionProgress) -> ControlFlow<()>,
    ) -> Result<Option<Transcript>> {
        if recording.samples.is_empty() {
            return Err(Error::model("the recording is empty"));
        }
        let options = TranscriptionOptions {
            language: match options.language.as_deref().map(str::trim) {
                None | Some("") => None,
                Some(asked) => Some(self.language(asked)?),
            },
            context: options.context.clone(),
        };

        let rate = self.config.sample_rate;
        let wave = resample(&recording.samples, recording.rate, rate);
        let pieces = split(
            &wave,
            rate,
            self.config.longest_chunk_seconds,
            SHORTEST_SECONDS,
        );

        // About three tokens a second and the language tag, for the bar; it holds at the end.
        let expected = |piece: &[f32]| 4 + (3.0 * piece.len() as f32 / rate as f32).ceil() as i32;
        let total = pieces.len() as i32;
        let mut before = 0;
        let mut languages = Vec::new();
        let mut text = String::new();
        let all: i32 = pieces.iter().map(|piece| expected(piece)).sum();
        for (index, piece) in pieces.iter().enumerate() {
            if report(TranscriptionProgress::Listening {
                done: index as i32,
                total,
            })
            .is_break()
            {
                return Ok(None);
            }
            let heard = self.hear(piece, &options, &mut |done| {
                report(TranscriptionProgress::Writing {
                    done: before + done,
                    expected: all.max(before + done),
                })
            })?;
            let Some(heard) = heard else {
                return Ok(None);
            };
            before += heard.said.len() as i32;

            let (language, words) = parse(&heard.raw, options.language.as_deref());
            languages.push(language);
            text.push_str(&words);
        }

        Ok(Some(Transcript {
            text,
            language: merge_languages(&languages),
        }))
    }

    fn name(&self) -> &'static str {
        Self::NAME
    }

    fn languages(&self) -> Vec<String> {
        self.config.languages.clone()
    }
}

/// `split_audio_into_chunks`: `wave` in pieces of at most `longest` seconds, each cut at the
/// quietest tenth of a second within five seconds of where it would otherwise end, and any piece
/// shorter than `shortest` padded with silence. The pieces put back together are the recording.
pub fn split(wave: &[f32], rate: u32, longest: f32, shortest: f32) -> Vec<Vec<f32>> {
    let rate_f = rate as f32;
    let total = wave.len();
    let most = (longest * rate_f) as usize;
    let mut pieces = Vec::new();

    if total as f32 / rate_f > longest && most > 0 {
        let expand = (5.0 * rate_f) as usize;
        let window = ((0.1 * rate_f) as usize).max(4);
        let mut start = 0;
        while total - start > most {
            let cut = start + most;
            let left = start.max(cut.saturating_sub(expand));
            let right = total.min(cut + expand);
            let boundary = match right - left <= window {
                true => cut,
                false => {
                    // The window with the least energy, then its quietest sample.
                    let loud: Vec<f32> = wave[left..right].iter().map(|x| x.abs()).collect();
                    let mut sum: f32 = loud[..window].iter().sum();
                    let (mut best, mut at) = (sum, 0);
                    for index in 1..=loud.len() - window {
                        sum += loud[index + window - 1] - loud[index - 1];
                        if sum < best {
                            (best, at) = (sum, index);
                        }
                    }
                    let inner = loud[at..at + window]
                        .iter()
                        .enumerate()
                        .fold((0, f32::INFINITY), |quietest, (index, x)| {
                            match *x < quietest.1 {
                                true => (index, *x),
                                false => quietest,
                            }
                        })
                        .0;
                    left + at + inner
                }
            };
            let boundary = boundary.max(start + 1).min(total);
            pieces.push(wave[start..boundary].to_vec());
            start = boundary;
        }
        pieces.push(wave[start..].to_vec());
    } else {
        pieces.push(wave.to_vec());
    }

    let least = (shortest * rate_f) as usize;
    for piece in &mut pieces {
        if piece.len() < least {
            piece.resize(least, 0.0);
        }
    }
    pieces
}

/// `parse_asr_output`: what the model wrote as a language and the words. With a language it was
/// told, everything written is the words. `language None` is a recording with nothing said in it.
pub fn parse(raw: &str, told: Option<&str>) -> (String, String) {
    let written = fix_repetitions(raw.trim(), 20);
    if written.is_empty() {
        return (String::new(), String::new());
    }
    if let Some(language) = told {
        return (language.to_string(), written);
    }
    let Some((meta, words)) = written.split_once(ASR_TEXT) else {
        return (String::new(), written.trim().to_string());
    };
    if meta.to_lowercase().contains("language none") {
        return (String::new(), words.trim().to_string());
    }

    let mut language = String::new();
    for line in meta.lines().map(str::trim).filter(|line| !line.is_empty()) {
        if line.to_lowercase().starts_with("language ") {
            let named = line["language ".len()..].trim();
            let mut letters = named.chars();
            if let Some(first) = letters.next() {
                language = first
                    .to_uppercase()
                    .chain(letters.flat_map(char::to_lowercase))
                    .collect();
            }
            break;
        }
    }
    (language, words.trim().to_string())
}

/// `merge_languages`: the pieces' languages in order, empty ones and repeats in a row dropped.
pub fn merge_languages(languages: &[String]) -> String {
    let mut merged: Vec<&str> = Vec::new();
    for language in languages.iter().map(|language| language.trim()) {
        if !language.is_empty() && merged.last() != Some(&language) {
            merged.push(language);
        }
    }
    merged.join(",")
}

/// `detect_and_fix_repetitions`: a character said more than `threshold` times in a row is said
/// once, and so is a pattern of up to twenty characters repeated `threshold` times or more. On
/// characters, as upstream's is on a Python string.
pub fn fix_repetitions(text: &str, threshold: usize) -> String {
    let chars: Vec<char> = text.chars().collect();

    // Runs of one character.
    let mut once = Vec::with_capacity(chars.len());
    let mut index = 0;
    while index < chars.len() {
        let mut count = 1;
        while index + count < chars.len() && chars[index + count] == chars[index] {
            count += 1;
        }
        match count > threshold {
            true => once.push(chars[index]),
            false => once.extend_from_slice(&chars[index..index + count]),
        }
        index += count;
    }

    fix_patterns(&once, threshold, 20).into_iter().collect()
}

/// Upstream's `fix_pattern_repeats`, which keeps going after the first pattern it collapses.
fn fix_patterns(s: &[char], threshold: usize, longest: usize) -> Vec<char> {
    let n = s.len();
    if n < threshold * 2 {
        return s.to_vec();
    }

    let mut result = Vec::with_capacity(n);
    let mut index = 0;
    while index + threshold * 2 <= n {
        for k in 1..=longest {
            if index + k * threshold > n {
                break;
            }
            let pattern = &s[index..index + k];
            let repeats = (1..threshold).all(|rep| {
                let from = index + rep * k;
                s.get(from..from + k) == Some(pattern)
            });
            if repeats {
                let mut end = index + threshold * k;
                while end + k <= n && &s[end..end + k] == pattern {
                    end += k;
                }
                result.extend_from_slice(pattern);
                result.extend(fix_patterns(&s[end..], threshold, longest));
                return result;
            }
        }
        result.push(s[index]);
        index += 1;
    }
    result.extend_from_slice(&s[index..]);
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn what_the_model_writes_is_a_language_and_words() {
        assert_eq!(
            parse(
                "language Chinese<asr_text>希望你以后能够做得比我还好哟。",
                None
            ),
            (
                "Chinese".to_string(),
                "希望你以后能够做得比我还好哟。".to_string()
            )
        );
        assert_eq!(
            parse("language ENGLISH\n<asr_text> Hello. ", None),
            ("English".to_string(), "Hello.".to_string())
        );
        assert_eq!(
            parse("language None<asr_text>", None),
            (String::new(), String::new())
        );
        assert_eq!(
            parse("no tag at all", None),
            (String::new(), "no tag at all".to_string())
        );
        // Told the language, everything written is the words.
        assert_eq!(
            parse("Hello there.", Some("English")),
            ("English".to_string(), "Hello there.".to_string())
        );
        assert_eq!(parse("  ", None), (String::new(), String::new()));
    }

    #[test]
    fn languages_merge_in_order_without_repeats() {
        let named = |names: &[&str]| names.iter().map(|n| n.to_string()).collect::<Vec<_>>();
        assert_eq!(
            merge_languages(&named(&["Chinese", "English", "English", "", "Chinese"])),
            "Chinese,English,Chinese"
        );
        assert_eq!(merge_languages(&named(&["", ""])), "");
    }

    #[test]
    fn repetitions_are_cut_to_one() {
        // A character more than twenty times in a row.
        let many = format!("好{}的", "啊".repeat(30));
        assert_eq!(fix_repetitions(&many, 20), "好啊的");
        // Twenty times exactly is left.
        let twenty = "啊".repeat(20);
        assert_eq!(fix_repetitions(&twenty, 20), twenty);
        // A pattern repeated twenty times or more, with what follows it kept.
        let pattern = format!("开始{}结束", "哈哈呀".repeat(25));
        assert_eq!(fix_repetitions(&pattern, 20), "开始哈哈呀结束");
        // Ordinary text is untouched.
        let plain = "The train leaves at seven tonight, and the ticket costs twenty-five dollars.";
        assert_eq!(fix_repetitions(plain, 20), plain);
    }

    #[test]
    fn a_long_recording_is_cut_where_it_is_quiet() {
        let rate = 100;
        // Thirty seconds of noise with a silent sample at 9.5 s: cut there, within five seconds
        // of the ten-second mark.
        let mut wave: Vec<f32> = (0..3000).map(|i| 0.5 + 0.1 * ((i % 7) as f32)).collect();
        for sample in &mut wave[945..955] {
            *sample = 0.0;
        }
        let pieces = split(&wave, rate, 10.0, 0.5);
        assert!(pieces.len() >= 3, "{} pieces", pieces.len());
        assert!((945..955).contains(&pieces[0].len()), "{}", pieces[0].len());
        assert_eq!(pieces.iter().map(Vec::len).sum::<usize>(), wave.len());

        // Short enough is one piece, and a very short one is padded.
        assert_eq!(split(&wave[..500], rate, 10.0, 0.5).len(), 1);
        assert_eq!(split(&wave[..10], rate, 10.0, 0.5)[0].len(), 50);
    }
}
