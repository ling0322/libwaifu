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

//! What a Qwen3-ASR package says it is: `tools/qwen3_asr_exporter.py` writes the release's widths
//! into the manifest, so that a wider release is read by the same code.

use crate::error::{Error, Result};
use crate::mapping::Mapping;
use crate::Manifest;

/// The audio encoder's shape: `thinker_config.audio_config` of the release.
#[derive(Clone, Debug, PartialEq)]
pub struct AudioConfig {
    pub mel_bins: i32,
    pub d_model: i32,
    pub layers: i32,
    pub heads: i32,
    pub ffn_dim: i32,
    /// The channels of the three convolutions.
    pub downsample_hidden_size: i32,
    /// The decoder's width, which is what the last projection writes.
    pub output_dim: i32,
    /// Half a chunk, in mel frames: the convolutions read `2 * n_window` frames at a time.
    pub n_window: i32,
    /// How many mel frames' worth of tokens attend to one another: 800, eight seconds.
    pub n_window_infer: i32,
    /// Rows of the sinusoidal position table.
    pub max_source_positions: i32,
}

/// The decoder's shape: `thinker_config.text_config`, which is Qwen3's.
#[derive(Clone, Debug, PartialEq)]
pub struct TextConfig {
    pub hidden_size: i32,
    pub layers: i32,
    pub heads: i32,
    pub kv_heads: i32,
    pub head_dim: i32,
    pub mlp_size: i32,
    pub vocab_size: i32,
    pub rope_theta: f64,
    pub norm_eps: f32,
}

/// Everything else the package says: the markers, when to stop, and the languages it knows.
#[derive(Clone, Debug, PartialEq)]
pub struct Config {
    pub audio: AudioConfig,
    pub text: TextConfig,
    pub sample_rate: u32,
    pub audio_token_id: i32,
    pub audio_start_token_id: i32,
    pub audio_end_token_id: i32,
    pub end_token_ids: Vec<i32>,
    pub languages: Vec<String>,
    /// How long a piece of a recording may be before it is cut, at its quietest point near there.
    pub longest_chunk_seconds: f32,
    /// How many tokens one piece may write before it is stopped.
    pub max_new_tokens: i32,
}

fn list(section: &Mapping, key: &str) -> Result<Vec<String>> {
    Ok(section
        .get_str(key)?
        .split(',')
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .map(str::to_string)
        .collect())
}

impl AudioConfig {
    pub const SECTION: &'static str = "qwen3_asr_audio";

    pub fn from_section(section: &Mapping) -> Result<AudioConfig> {
        let config = AudioConfig {
            mel_bins: section.get("mel_bins")?,
            d_model: section.get("d_model")?,
            layers: section.get("layers")?,
            heads: section.get("heads")?,
            ffn_dim: section.get("ffn_dim")?,
            downsample_hidden_size: section.get("downsample_hidden_size")?,
            output_dim: section.get("output_dim")?,
            n_window: section.get("n_window")?,
            n_window_infer: section.get("n_window_infer")?,
            max_source_positions: section.get("max_source_positions")?,
        };
        if config.d_model % config.heads != 0 {
            return Err(Error::model(format!(
                "the audio encoder is {} wide, which {} heads do not divide",
                config.d_model, config.heads
            )));
        }
        if config.n_window <= 0 || config.n_window_infer % (2 * config.n_window) != 0 {
            return Err(Error::model(format!(
                "an attention window of {} frames is not a whole number of {}-frame chunks",
                config.n_window_infer,
                2 * config.n_window
            )));
        }
        Ok(config)
    }

    /// Frames a chunk the convolutions read.
    pub fn chunk(&self) -> i32 {
        2 * self.n_window
    }

    /// The mel bands after the three stride-two convolutions: 128 is 16.
    pub fn bands_after(&self) -> i32 {
        (0..3).fold(self.mel_bins, |bands, _| (bands - 1) / 2 + 1)
    }
}

impl TextConfig {
    pub const SECTION: &'static str = "qwen3_asr_text";

    pub fn from_section(section: &Mapping) -> Result<TextConfig> {
        let config = TextConfig {
            hidden_size: section.get("hidden_size")?,
            layers: section.get("layers")?,
            heads: section.get("heads")?,
            kv_heads: section.get("kv_heads")?,
            head_dim: section.get("head_dim")?,
            mlp_size: section.get("mlp_size")?,
            vocab_size: section.get("vocab_size")?,
            rope_theta: section.get("rope_theta")?,
            norm_eps: section.get("norm_eps")?,
        };
        if config.heads % config.kv_heads != 0 || config.head_dim % 2 != 0 {
            return Err(Error::model(format!(
                "{} query heads against {} for the keys, {} wide, is not a decoder this reads",
                config.heads, config.kv_heads, config.head_dim
            )));
        }
        Ok(config)
    }
}

impl Config {
    /// The languages a package's own section lists, which is all a description of it needs.
    pub fn languages_of(section: &Mapping) -> Result<Vec<String>> {
        list(section, "languages")
    }

    pub fn from_manifest(manifest: &Manifest) -> Result<Config> {
        let section = manifest.section(super::Qwen3Asr::MODEL_TYPE)?;
        let end_token_ids = list(section, "end_token_ids")?
            .iter()
            .map(|id| {
                id.parse::<i32>()
                    .map_err(|_| Error::model(format!("end_token_ids holds {id:?}")))
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Config {
            audio: AudioConfig::from_section(manifest.section(AudioConfig::SECTION)?)?,
            text: TextConfig::from_section(manifest.section(TextConfig::SECTION)?)?,
            sample_rate: section.get("sample_rate")?,
            audio_token_id: section.get("audio_token_id")?,
            audio_start_token_id: section.get("audio_start_token_id")?,
            audio_end_token_id: section.get("audio_end_token_id")?,
            end_token_ids,
            languages: list(section, "languages")?,
            longest_chunk_seconds: section.get("longest_chunk_seconds")?,
            max_new_tokens: section.get("max_new_tokens")?,
        })
    }
}

/// What three stride-two convolutions leave of `frames` mel frames read in chunks of `chunk`:
/// upstream's `_get_feat_extract_output_lengths`, which is thirteen tokens for every whole
/// hundred-frame chunk and what the convolutions leave of the rest.
pub fn tokens_of(frames: i32, chunk: i32) -> i32 {
    let after = |length: i32| (0..3).fold(length, |length, _| (length - 1) / 2 + 1);
    let rest = frames % chunk;
    let whole = (frames / chunk) * after(chunk);
    match rest {
        0 => whole,
        rest => whole + after(rest),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokens_follow_upstreams_lengths() {
        // `((leave - 1) // 2 + 1 - 1) // 2 + 1 ...` as upstream writes it, for every length up to
        // twenty seconds.
        let upstream = |frames: i32| {
            let leave = frames % 100;
            let feat = (leave - 1).div_euclid(2) + 1;
            ((feat - 1).div_euclid(2) + 1 - 1).div_euclid(2) + 1 + (frames / 100) * 13
        };
        for frames in 1..2000 {
            assert_eq!(tokens_of(frames, 100), upstream(frames), "{frames} frames");
        }
        assert_eq!(tokens_of(348, 100), 45);
        assert_eq!(tokens_of(1374, 100), 179);
    }
}
