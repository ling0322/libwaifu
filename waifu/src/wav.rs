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

//! Reading and writing WAV, which is the one audio format this crate handles itself.
//!
//! It is here because a speech model has to hand its waveform to something, and every other
//! format worth writing is a codec -- a dependency, several thousand lines of it, and a licence
//! to read. WAV is a header, a format chunk and the samples, and the whole of it fits in this
//! file.
//!
//! What is read is wider than what is written, which is the usual shape for a file format: one
//! encoder, because there is one right answer, and a decoder that takes what it is actually
//! handed. A recording someone drags onto the page has been through whatever wrote it -- a phone,
//! a recorder, ffmpeg -- and those write 8-, 16-, 24- and 32-bit integers and 32-bit floats, in
//! however many channels the microphone had.

use crate::{Error, Result};

/// What is written: signed 16-bit samples, which is what every player takes and what a waveform
/// out of a model is worth keeping at. A model works in floats and the ear does not: the noise
/// floor of sixteen bits is below the noise floor of anything that synthesized the sound.
const BITS: u16 = 16;

/// The two format tags this reads, as they are written in a format chunk.
const PCM: u16 = 1;
const FLOAT: u16 = 3;
/// The tag that says "the real one is in the extension", which is what anything writing more than
/// two channels, or more than sixteen bits, is supposed to use. The first two bytes of the
/// extension's GUID are one of the two above.
const EXTENSIBLE: u16 = 0xfffe;

/// A waveform: mono, between -1 and 1, at a rate.
///
/// Mono because everything here is one voice. A recording that arrives in stereo is mixed down on
/// the way in rather than carried as two channels that a speaker encoder would immediately have
/// to pick between.
///
/// Floats rather than the integers a file holds, because everything that does anything with a
/// waveform -- a filterbank, a resampling, a vocoder -- works in floats, and a conversion at the
/// edge is one conversion rather than one per operation.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Sound {
    pub samples: Vec<f32>,
    pub rate: u32,
}

impl Sound {
    pub fn new(samples: Vec<f32>, rate: u32) -> Sound {
        Sound { samples, rate }
    }

    /// How long it lasts. Zero for a rate of zero, which is not a sound but is a thing a file can
    /// claim to be.
    pub fn seconds(&self) -> f64 {
        match self.rate {
            0 => 0.0,
            rate => self.samples.len() as f64 / f64::from(rate),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }
}

/// Writes a sound as a WAV file: a RIFF header, a format chunk and the samples.
///
/// Every length in the header is the one that was actually written rather than one worked out in
/// advance, which is what keeps a file whose sample count nobody checked from being a file whose
/// header disagrees with it.
pub fn write(sound: &Sound) -> Vec<u8> {
    let channels: u16 = 1;
    let rate = sound.rate;
    let block = channels * BITS / 8;
    let data = sound.samples.len() * usize::from(block);

    let mut bytes = Vec::with_capacity(44 + data);
    bytes.extend_from_slice(b"RIFF");
    // Everything after this field: the eight bytes of "WAVE" and the format chunk's header, the
    // sixteen of the chunk itself, the eight of the data chunk's header, and the samples.
    bytes.extend_from_slice(&u32::try_from(36 + data).unwrap_or(u32::MAX).to_le_bytes());
    bytes.extend_from_slice(b"WAVE");

    bytes.extend_from_slice(b"fmt ");
    bytes.extend_from_slice(&16u32.to_le_bytes());
    bytes.extend_from_slice(&PCM.to_le_bytes());
    bytes.extend_from_slice(&channels.to_le_bytes());
    bytes.extend_from_slice(&rate.to_le_bytes());
    bytes.extend_from_slice(&(rate * u32::from(block)).to_le_bytes());
    bytes.extend_from_slice(&block.to_le_bytes());
    bytes.extend_from_slice(&BITS.to_le_bytes());

    bytes.extend_from_slice(b"data");
    bytes.extend_from_slice(&u32::try_from(data).unwrap_or(u32::MAX).to_le_bytes());
    for sample in &sound.samples {
        bytes.extend_from_slice(&quantize(*sample).to_le_bytes());
    }

    bytes
}

/// One sample as the sixteen bits it is written in.
///
/// Clamped before it is scaled, because a model that overshot by a hair is a model that would
/// otherwise wrap from the top of the range to the bottom of it -- which is a click, and a loud
/// one. Scaled by 32767 rather than 32768 so that the two ends of the range are symmetric and 1.0
/// is representable.
fn quantize(sample: f32) -> i16 {
    let held = sample.clamp(-1.0, 1.0);

    (held * f32::from(i16::MAX)).round() as i16
}

/// Reads a WAV file: the samples, mixed down to one channel, and the rate they are at.
///
/// What it takes is what things actually write. What it refuses, it refuses by name -- a file
/// that turns out to be an MP3 with the wrong extension on it is the commonest way this fails,
/// and "not a WAV file" is a more useful sentence than a parse error about a chunk.
pub fn read(bytes: &[u8]) -> Result<Sound> {
    if bytes.len() < 12 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err(Error::format(
            "this is not a WAV file: nothing here says RIFF and WAVE. Only WAV is read -- \
             anything else is a codec this program does not carry",
        ));
    }

    let mut format: Option<Format> = None;
    let mut at = 12;

    // The chunks, in whatever order they are in. `fmt ` comes before `data` in every file anyone
    // writes, and the standard does not actually say it has to, so this walks rather than assumes.
    while at + 8 <= bytes.len() {
        let name = &bytes[at..at + 4];
        let length =
            u32::from_le_bytes([bytes[at + 4], bytes[at + 5], bytes[at + 6], bytes[at + 7]])
                as usize;
        let from = at + 8;
        // A length that runs off the end is a truncated file. Taking what is there rather than
        // refusing: a recording that was cut short is still most of a recording.
        let to = from.saturating_add(length).min(bytes.len());

        match name {
            b"fmt " => format = Some(describe(&bytes[from..to])?),
            b"data" => {
                let Some(format) = format else {
                    return Err(Error::format(
                        "this WAV file has its samples before it says what they are",
                    ));
                };
                return Ok(Sound {
                    samples: decode(&bytes[from..to], &format)?,
                    rate: format.rate,
                });
            }
            _ => {}
        }

        // Chunks are padded to an even length, and the pad byte is not counted in the length.
        at = from + length + (length & 1);
    }

    Err(Error::format(
        "this WAV file has no samples in it: there is no data chunk",
    ))
}

/// What the format chunk says the samples are.
struct Format {
    tag: u16,
    channels: u16,
    rate: u32,
    bits: u16,
}

fn describe(chunk: &[u8]) -> Result<Format> {
    if chunk.len() < 16 {
        return Err(Error::format("this WAV file's format chunk is too short"));
    }

    let word = |at: usize| u16::from_le_bytes([chunk[at], chunk[at + 1]]);
    let long =
        |at: usize| u32::from_le_bytes([chunk[at], chunk[at + 1], chunk[at + 2], chunk[at + 3]]);

    let mut tag = word(0);
    // The extension says what it really is, which is how anything above two channels or sixteen
    // bits is supposed to be written. Its first two bytes are the tag the file would have carried
    // if it had fitted in one.
    if tag == EXTENSIBLE && chunk.len() >= 26 {
        tag = word(24);
    }

    let format = Format {
        tag,
        channels: word(2),
        rate: long(4),
        bits: word(14),
    };

    if format.channels == 0 {
        return Err(Error::format("this WAV file says it has no channels"));
    }
    if format.rate == 0 {
        return Err(Error::format("this WAV file says it has no sample rate"));
    }

    Ok(format)
}

/// The samples of the data chunk, as floats, mixed down to one channel.
fn decode(data: &[u8], format: &Format) -> Result<Vec<f32>> {
    let width = usize::from(format.bits / 8);
    let channels = usize::from(format.channels);
    if width == 0 {
        return Err(Error::format(
            "this WAV file says its samples are no bits wide",
        ));
    }

    // One sample of each channel, which is what a frame is and what gets mixed down to one number.
    let frame = width * channels;
    let frames = data.len() / frame;
    let mut samples = Vec::with_capacity(frames);

    for at in (0..frames * frame).step_by(frame) {
        let mut sum = 0.0;
        for channel in 0..channels {
            let from = at + channel * width;
            sum += one(&data[from..from + width], format)?;
        }
        // The mean rather than the sum: two channels of the same thing added together is that
        // thing at twice the amplitude, which clips.
        samples.push(sum / channels as f32);
    }

    Ok(samples)
}

/// One sample, from the bytes it is written in.
///
/// The scale is the one that makes the quietest and loudest representable values -1 and 1. Signed
/// integers are divided by the magnitude of their most negative value, which is the convention
/// every other decoder uses -- it makes full scale a hair under 1.0 rather than a hair over it.
fn one(bytes: &[u8], format: &Format) -> Result<f32> {
    match (format.tag, format.bits) {
        // Eight-bit WAV is unsigned, alone among the integer widths, with silence at 128.
        (PCM, 8) => Ok((f32::from(bytes[0]) - 128.0) / 128.0),
        (PCM, 16) => {
            let whole = i16::from_le_bytes([bytes[0], bytes[1]]);
            Ok(f32::from(whole) / 32768.0)
        }
        (PCM, 24) => {
            // Sign extended into the top byte of an i32, which is what shifting a left-aligned
            // value back down does arithmetically.
            let whole = i32::from_le_bytes([0, bytes[0], bytes[1], bytes[2]]) >> 8;
            Ok(whole as f32 / 8_388_608.0)
        }
        (PCM, 32) => {
            let whole = i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
            Ok(whole as f32 / 2_147_483_648.0)
        }
        // Already between -1 and 1, which is what the float formats are for.
        (FLOAT, 32) => Ok(f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])),
        (FLOAT, 64) => {
            let mut eight = [0u8; 8];
            eight.copy_from_slice(&bytes[0..8]);
            Ok(f64::from_le_bytes(eight) as f32)
        }
        (tag, bits) => Err(Error::format(format!(
            "this WAV file is in a shape that is not read here: format {tag}, {bits} bits a \
             sample. What is read is 8-, 16-, 24- and 32-bit integers and 32- and 64-bit floats"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A second of a quiet tone, which is a waveform with something in every part of the range.
    fn a_tone(rate: u32) -> Sound {
        let samples = (0..rate)
            .map(|at| {
                let seconds = f64::from(at) / f64::from(rate);
                (0.5 * (std::f64::consts::TAU * 220.0 * seconds).sin()) as f32
            })
            .collect();

        Sound::new(samples, rate)
    }

    #[test]
    fn what_is_written_is_what_comes_back() {
        // The round trip, which is the only property of this that anything depends on: a clip
        // this program wrote is a clip this program can read -- and so can everything else, which
        // is the part the header's arithmetic below is about.
        let sound = a_tone(8000);
        let read = read(&write(&sound)).expect("a WAV file this file wrote");

        assert_eq!(read.rate, 8000);
        assert_eq!(read.samples.len(), sound.samples.len());
        for (wrote, read) in sound.samples.iter().zip(&read.samples) {
            // Sixteen bits is one part in 32768, and a round trip through them cannot be nearer
            // than half of that.
            assert!(
                (wrote - read).abs() < 1.0 / 32768.0,
                "{wrote} came back {read}"
            );
        }
    }

    #[test]
    fn the_header_says_the_length_that_was_written() {
        // A header that disagrees with the file is the one way to write a WAV that some players
        // take and others refuse, and it is not a thing that shows up in a round trip through
        // this file's own reader.
        let bytes = write(&a_tone(16_000));
        let data = 16_000 * 2;

        assert_eq!(&bytes[0..4], b"RIFF");
        assert_eq!(&bytes[8..12], b"WAVE");
        assert_eq!(bytes.len(), 44 + data);
        assert_eq!(
            u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]) as usize,
            36 + data
        );
        assert_eq!(
            u32::from_le_bytes([bytes[40], bytes[41], bytes[42], bytes[43]]) as usize,
            data
        );
        // Mono, sixteen bits, and the byte rate and block alignment that follow from those two.
        assert_eq!(u16::from_le_bytes([bytes[20], bytes[21]]), PCM);
        assert_eq!(u16::from_le_bytes([bytes[22], bytes[23]]), 1);
        assert_eq!(
            u32::from_le_bytes([bytes[24], bytes[25], bytes[26], bytes[27]]),
            16_000
        );
        assert_eq!(
            u32::from_le_bytes([bytes[28], bytes[29], bytes[30], bytes[31]]),
            32_000
        );
        assert_eq!(u16::from_le_bytes([bytes[32], bytes[33]]), 2);
        assert_eq!(u16::from_le_bytes([bytes[34], bytes[35]]), BITS);
    }

    #[test]
    fn a_sample_past_the_end_of_the_range_is_held_rather_than_wrapped() {
        // A model that overshot by a hair would otherwise come out with the loudest possible
        // click in it, since the top of the range wraps to the bottom.
        assert_eq!(quantize(2.0), i16::MAX);
        assert_eq!(quantize(-2.0), -i16::MAX);
        assert_eq!(quantize(0.0), 0);
        assert!(quantize(f32::NAN) == 0 || quantize(f32::NAN) == i16::MAX);
    }

    #[test]
    fn a_file_that_is_not_one_says_so_rather_than_failing_to_parse() {
        // The commonest way this fails is an MP3 with the wrong name on it, and what somebody
        // needs to be told is which formats are read here.
        let refused = read(b"ID3\x04and the rest of it").expect_err("not a WAV file");
        assert!(refused.to_string().contains("not a WAV file"), "{refused}");

        assert!(read(&[]).is_err());
        assert!(read(b"RIFF").is_err());

        // A RIFF file that is not a WAV one at all: this is what an AVI starts with.
        let mut avi = b"RIFF\x00\x00\x00\x00AVI ".to_vec();
        avi.extend_from_slice(b"LIST");
        assert!(read(&avi).is_err());
    }

    #[test]
    fn a_wav_file_with_no_samples_in_it_is_refused_by_what_is_missing() {
        // A header and a format chunk and nothing after them, which is what a recording that was
        // opened and never written to leaves behind.
        let mut bytes = write(&Sound::new(Vec::new(), 16_000));
        bytes.truncate(36);
        bytes[4..8].copy_from_slice(&28u32.to_le_bytes());

        let refused = read(&bytes).expect_err("no samples");
        assert!(refused.to_string().contains("no data chunk"), "{refused}");
    }

    /// A file in one of the shapes this reads but does not write: the format chunk said whole,
    /// and the samples as the caller wrote them.
    fn a_file(tag: u16, bits: u16, channels: u16, rate: u32, data: &[u8]) -> Vec<u8> {
        let block = channels * bits / 8;
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"RIFF");
        bytes.extend_from_slice(&(36 + data.len() as u32).to_le_bytes());
        bytes.extend_from_slice(b"WAVE");
        bytes.extend_from_slice(b"fmt ");
        bytes.extend_from_slice(&16u32.to_le_bytes());
        bytes.extend_from_slice(&tag.to_le_bytes());
        bytes.extend_from_slice(&channels.to_le_bytes());
        bytes.extend_from_slice(&rate.to_le_bytes());
        bytes.extend_from_slice(&(rate * u32::from(block)).to_le_bytes());
        bytes.extend_from_slice(&block.to_le_bytes());
        bytes.extend_from_slice(&bits.to_le_bytes());
        bytes.extend_from_slice(b"data");
        bytes.extend_from_slice(&(data.len() as u32).to_le_bytes());
        bytes.extend_from_slice(data);

        bytes
    }

    #[test]
    fn the_widths_and_formats_things_actually_write_are_read() {
        // What is written here is one of these and what arrives is any of them: a recording has
        // been through whatever made it, and a phone, a recorder and ffmpeg do not agree.
        let half = |sound: Sound| {
            assert_eq!(sound.rate, 8000);
            assert_eq!(sound.samples.len(), 1);
            assert!((sound.samples[0] - 0.5).abs() < 0.01, "{:?}", sound.samples);
        };

        half(read(&a_file(PCM, 8, 1, 8000, &[192])).expect("eight bits, unsigned"));
        half(read(&a_file(PCM, 16, 1, 8000, &16_384i16.to_le_bytes())).expect("sixteen bits"));
        half(read(&a_file(PCM, 24, 1, 8000, &[0x00, 0x00, 0x40])).expect("twenty four bits"));
        half(read(&a_file(PCM, 32, 1, 8000, &1_073_741_824i32.to_le_bytes())).expect("thirty two"));
        half(read(&a_file(FLOAT, 32, 1, 8000, &0.5f32.to_le_bytes())).expect("a float"));
        half(read(&a_file(FLOAT, 64, 1, 8000, &0.5f64.to_le_bytes())).expect("a wide float"));
    }

    #[test]
    fn several_channels_come_back_as_one() {
        // Averaged rather than added: the same thing in two channels added together is that thing
        // at twice the amplitude, which clips.
        let mut data = Vec::new();
        for sample in [16_384i16, 0, -16_384, 0] {
            data.extend_from_slice(&sample.to_le_bytes());
        }

        let sound = read(&a_file(PCM, 16, 2, 44_100, &data)).expect("two channels");
        assert_eq!(sound.rate, 44_100);
        assert_eq!(sound.samples.len(), 2);
        assert!(
            (sound.samples[0] - 0.25).abs() < 0.01,
            "{:?}",
            sound.samples
        );
        assert!(
            (sound.samples[1] + 0.25).abs() < 0.01,
            "{:?}",
            sound.samples
        );
    }

    #[test]
    fn a_format_this_does_not_read_is_named_rather_than_guessed_at() {
        // Which is what an A-law file is, and what anything compressed inside a WAV container is.
        let alaw = a_file(6, 8, 1, 8000, &[0, 0, 0, 0]);
        let refused = read(&alaw).expect_err("A-law");
        assert!(refused.to_string().contains("format 6"), "{refused}");
    }

    #[test]
    fn the_extension_is_where_a_wide_file_says_what_it_is() {
        // Anything above two channels or sixteen bits is supposed to be written this way, and a
        // reader that stopped at the tag would call every one of them unreadable.
        let mut chunk = Vec::new();
        chunk.extend_from_slice(b"RIFF\x00\x00\x00\x00WAVEfmt ");
        chunk.extend_from_slice(&40u32.to_le_bytes());
        chunk.extend_from_slice(&EXTENSIBLE.to_le_bytes());
        chunk.extend_from_slice(&1u16.to_le_bytes());
        chunk.extend_from_slice(&48_000u32.to_le_bytes());
        chunk.extend_from_slice(&(48_000u32 * 4).to_le_bytes());
        chunk.extend_from_slice(&4u16.to_le_bytes());
        chunk.extend_from_slice(&32u16.to_le_bytes());
        // The extension: its own length, the valid bits, the channel mask, and the GUID whose
        // first two bytes are the tag this really is.
        chunk.extend_from_slice(&22u16.to_le_bytes());
        chunk.extend_from_slice(&32u16.to_le_bytes());
        chunk.extend_from_slice(&0u32.to_le_bytes());
        chunk.extend_from_slice(&FLOAT.to_le_bytes());
        chunk.extend_from_slice(&[0; 14]);
        chunk.extend_from_slice(b"data");
        chunk.extend_from_slice(&4u32.to_le_bytes());
        chunk.extend_from_slice(&0.25f32.to_le_bytes());

        let sound = read(&chunk).expect("an extensible float file");
        assert_eq!(sound.rate, 48_000);
        assert_eq!(sound.samples, vec![0.25]);
    }

    #[test]
    fn the_chunks_nobody_asked_for_are_walked_past() {
        // A recorder writes a LIST chunk of who made the file, and ffmpeg writes one saying so.
        // Both sit between the format chunk and the samples.
        let mut bytes = b"RIFF\x00\x00\x00\x00WAVE".to_vec();
        bytes.extend_from_slice(b"fmt ");
        bytes.extend_from_slice(&16u32.to_le_bytes());
        bytes.extend_from_slice(&PCM.to_le_bytes());
        bytes.extend_from_slice(&1u16.to_le_bytes());
        bytes.extend_from_slice(&8000u32.to_le_bytes());
        bytes.extend_from_slice(&16_000u32.to_le_bytes());
        bytes.extend_from_slice(&2u16.to_le_bytes());
        bytes.extend_from_slice(&16u16.to_le_bytes());
        // An odd length, so that the pad byte after it is part of what this has to walk past.
        bytes.extend_from_slice(b"LIST");
        bytes.extend_from_slice(&5u32.to_le_bytes());
        bytes.extend_from_slice(b"INFO\x00");
        bytes.push(0);
        bytes.extend_from_slice(b"data");
        bytes.extend_from_slice(&2u32.to_le_bytes());
        bytes.extend_from_slice(&16_384i16.to_le_bytes());

        let sound = read(&bytes).expect("a file with a LIST chunk in the middle");
        assert_eq!(sound.samples.len(), 1);
        assert!((sound.samples[0] - 0.5).abs() < 0.01);
    }

    #[test]
    fn a_file_that_was_cut_short_is_still_most_of_a_recording() {
        // A data chunk whose header claims more than is there, which is what a recording that was
        // interrupted leaves. What is there is worth having.
        let mut bytes = a_file(PCM, 16, 1, 8000, &[0; 8]);
        let length = bytes.len();
        bytes[length - 12..length - 8].copy_from_slice(&1024u32.to_le_bytes());

        let sound = read(&bytes).expect("what was written before it stopped");
        assert_eq!(sound.samples.len(), 4);
    }

    #[test]
    fn how_long_a_sound_is_does_not_divide_by_a_rate_of_nothing() {
        assert_eq!(Sound::new(vec![0.0; 16_000], 16_000).seconds(), 1.0);
        assert_eq!(Sound::new(vec![0.0; 16_000], 0).seconds(), 0.0);
        assert!(Sound::default().is_empty());
    }
}
