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

//! Text in, a waveform out: what the screen asks for when it is asked to speak.
//!
//! There is no speech model in this repository yet. What is here is the shape of the question --
//! [`Voice`] -- and one implementation of it, [`Tones`], which does not answer the question so
//! much as make a noise in the right places. The screen, the server and the worker thread are
//! all written against the trait, so the day a real model lands it implements [`Voice`] and
//! nothing above this file changes.
//!
//! That is the whole reason this file exists rather than waiting for the model. The half of a
//! speech feature that is not the model -- a page to type on, a reference recording to upload, a
//! clip that can be played and saved, a bar that moves, a run that can be stopped -- is written
//! once and is the same whichever model ends up behind it.
//!
//! # What a voice is asked for
//!
//! ```no_run
//! use std::ops::ControlFlow;
//! use waifu::{SpeechOptions, Tones, Voice};
//!
//! let voice = Tones::new();
//! let options = SpeechOptions { seed: Some(7), ..voice.defaults().options() };
//! let said = voice.speak("hello", None, &options, &mut |_| ControlFlow::Continue(()))?;
//! # Ok::<(), waifu::Error>(())
//! ```

use std::ops::ControlFlow;

use crate::wav::Sound;
use crate::Result;

/// What a run is asked for, beyond the words themselves.
///
/// Deliberately small, and deliberately not the union of every knob every speech model has. A
/// model that samples has a temperature; one that does not ignores it. What is here is what a
/// page can reasonably put in front of somebody without knowing which model is behind it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SpeechOptions {
    /// How fast to read it, as a multiple of the voice's own pace. Below one is slower.
    pub speed: f32,
    /// How much the model is allowed to vary from the likeliest thing to say next. Zero is the
    /// same reading every time; a model with nothing to sample ignores it.
    pub temperature: f32,
    /// Always filled in by the time a run starts, for the same reason a picture's is: a clip that
    /// came out well is asked about later, and by then nobody remembers the number.
    pub seed: Option<u64>,
}

impl Default for SpeechOptions {
    fn default() -> SpeechOptions {
        SpeechDefaults::default().options()
    }
}

/// What the boxes on the page start at, which is the voice's to say rather than the page's.
///
/// The same shape as [`GenerationDefaults`](crate::GenerationDefaults) and for the same reason: a
/// model that wants to be read at a different pace than the last one says so here, and the screen
/// takes it without knowing which model it came from.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SpeechDefaults {
    pub speed: f32,
    pub temperature: f32,
}

impl Default for SpeechDefaults {
    fn default() -> SpeechDefaults {
        SpeechDefaults {
            speed: 1.0,
            temperature: 0.8,
        }
    }
}

impl SpeechDefaults {
    /// These defaults as a run's options, with no seed in them yet.
    pub fn options(&self) -> SpeechOptions {
        SpeechOptions {
            speed: self.speed,
            temperature: self.temperature,
            seed: None,
        }
    }
}

/// How far along a run is, as the reporter given to [`Voice::speak`] is told.
///
/// Three stages, in the order a speech model goes through them, and not the same size as each
/// other: reading the text is fast, deciding what to say is most of it, and turning that into a
/// waveform is a fixed cost at the end.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SpeechProgress {
    /// Turning the text -- and the recording to sound like, where there is one -- into whatever
    /// the model takes. Once, before anything else.
    Reading,
    /// Deciding what to say, so far and out of how much.
    ///
    /// `expected` is an estimate and not a promise. A model that decides what to say one token at
    /// a time does not know how many there will be until it emits the one that ends the
    /// utterance, so what is here is worked out from the length of the text -- which means `done`
    /// can pass `expected` on a reading that runs long. Anything drawing a bar from these two has
    /// to hold it at the end rather than let it past.
    Saying { done: i32, expected: i32 },
    /// Turning what it decided into a waveform, which is the vocoder.
    Sounding,
}

/// Something that can be asked to say a sentence.
///
/// The one seam between the screen and whatever ends up speaking. Everything above it -- the
/// page, the routes, the worker thread, the clip on the disk -- is written against these five
/// methods and knows nothing else about the model.
///
/// `&self` rather than `&mut self`, like the picture models beside it: the weights do not change
/// as they are read, and the worker holds one of these for as long as a voice is loaded.
///
/// Not `Send`, for the same reason the picture models are not: a [`crate::flint::Tensor`] stays
/// on the thread that made it, and a voice with weights behind it is made of them. The bound this
/// trait used to carry cost nothing while the only voice was [`Tones`], two floats; the first
/// real one could not have implemented it. Nothing moves a voice between threads anyway -- it is
/// read, held and dropped on the worker's, exactly as a picture model is.
pub trait Voice {
    /// Says `text`, and hands back the waveform -- or `None`, where `report` asked it to stop.
    ///
    /// `like` is a recording to sound like, where the caller has one and the voice takes one. A
    /// voice that does not is free to ignore it, and says so through
    /// [`no_likeness_because`](Voice::no_likeness_because) so that the page need not offer what
    /// will be thrown away.
    fn speak(
        &self,
        text: &str,
        like: Option<&Sound>,
        options: &SpeechOptions,
        report: &mut dyn FnMut(SpeechProgress) -> ControlFlow<()>,
    ) -> Result<Option<Sound>>;

    /// The rate it speaks at, which is a property of the model rather than a setting.
    fn rate(&self) -> u32;

    /// What the boxes start at.
    fn defaults(&self) -> SpeechDefaults;

    /// What it is called on screen.
    fn name(&self) -> &'static str;

    /// Why it cannot be handed a recording to sound like, or `None` where it can.
    ///
    /// A sentence rather than a bool, for the same reason the picture models give one: a page
    /// that greys a box out without saying why is a page that cannot be asked.
    fn no_likeness_because(&self) -> Option<&'static str> {
        None
    }

    /// Why what comes out is not a voice, for the implementation where it is not.
    ///
    /// `None` for a real model, which is the point of it: this is how [`Tones`] tells the screen
    /// to say plainly that nothing here is speech yet, and how that sentence disappears by itself
    /// the day something is.
    fn not_a_voice_because(&self) -> Option<&'static str> {
        None
    }
}

/// What [`Tones`] says about itself, which is the whole of what it is for.
const NOT_A_VOICE: &str = "this is not a speech model. No voice has been published for libwaifu \
     yet, and what is speaking is a stand-in built into the binary: it reads the text as a run of \
     pitched tones, one to a syllable, so that everything around a voice -- the page, the \
     settings, the clip, the bar -- can be used and looked at before there is one. It is not \
     speech and is not meant to sound like any";

/// The rate the stand-in writes at.
///
/// 24 kHz because that is what the speech models this is standing in for emit, so a clip written
/// here and a clip written by the real thing are the same kind of file rather than two.
const RATE: u32 = 24_000;

/// How long one tone lasts at a speed of one, in seconds, and the gap after it.
const NOTE: f32 = 0.16;
const GAP: f32 = 0.04;

/// How long a pause is, for the punctuation that earns one.
const PAUSE: f32 = 0.22;

/// How many characters of a word get a tone of their own.
///
/// Roughly a syllable, which is roughly what an English word has one of every three letters. It
/// is not a syllabifier and is not trying to be -- what it buys is that a long word takes longer
/// to say than a short one, which is the part of speech this stand-in can honestly reproduce.
const PER_NOTE: usize = 3;

/// Where the tones sit when there is no recording to take a pitch from. Roughly a speaking voice.
const MIDDLE: f32 = 196.0;

/// The narrowest and widest a pitch taken off a recording is believed. Outside this is not a
/// voice: 70 Hz is below a bass and 400 is above a child, and an estimate out there is the
/// estimator having locked onto a harmonic or onto noise.
const LOWEST: f32 = 70.0;
const HIGHEST: f32 = 400.0;

/// The scale the tones are drawn from, as steps above the centre. A pentatonic one, because the
/// notes of it cannot be put in an order that sounds wrong -- which matters when the order is
/// coming from the letters of whatever somebody typed.
const SCALE: [i32; 5] = [0, 2, 4, 7, 9];

/// A stand-in for a speech model: text in, and a run of pitched tones out.
///
/// It exists so that the half of this feature that is not a model can be finished, used and
/// argued about before the model lands. What it does is honest and small: a tone to a syllable,
/// pitched from the letters, at a pace the settings change, centred on the pitch of the recording
/// it was handed if it was handed one.
///
/// It is not speech. Every screen that can reach it says so, out of
/// [`not_a_voice_because`](Voice::not_a_voice_because), and that sentence is how the warning
/// disappears by itself the day a real [`Voice`] is what is loaded.
pub struct Tones {
    defaults: SpeechDefaults,
}

impl Default for Tones {
    fn default() -> Tones {
        Tones::new()
    }
}

impl Tones {
    pub fn new() -> Tones {
        Tones {
            defaults: SpeechDefaults::default(),
        }
    }
}

impl Voice for Tones {
    fn rate(&self) -> u32 {
        RATE
    }

    fn defaults(&self) -> SpeechDefaults {
        self.defaults
    }

    fn name(&self) -> &'static str {
        "tones -- a stand-in, not a voice"
    }

    /// It takes one, and what it does with it is take its pitch. Which is a use of a recording
    /// and not a likeness of it, so the sentence above says what it is.
    fn no_likeness_because(&self) -> Option<&'static str> {
        None
    }

    fn not_a_voice_because(&self) -> Option<&'static str> {
        Some(NOT_A_VOICE)
    }

    fn speak(
        &self,
        text: &str,
        like: Option<&Sound>,
        options: &SpeechOptions,
        report: &mut dyn FnMut(SpeechProgress) -> ControlFlow<()>,
    ) -> Result<Option<Sound>> {
        if report(SpeechProgress::Reading).is_break() {
            return Ok(None);
        }

        // Where the tones sit: the pitch of the recording that was handed in, where one was and
        // where anything could be made of it, and a speaking pitch otherwise.
        let centre = like.and_then(pitch_of).unwrap_or(MIDDLE);
        let notes = read(text);
        let expected = notes.iter().filter(|note| !note.is_silent()).count() as i32;

        // A speed of zero is a clip of no length and a division by it; the page's box has a
        // minimum, and a request that did not come from the page does not.
        let speed = options.speed.clamp(0.25, 4.0);
        let spread = options.temperature.clamp(0.0, 2.0);

        let mut seed = Rolling::from(options.seed.unwrap_or(0));
        let mut samples = Vec::new();
        let mut done = 0;

        for note in &notes {
            let seconds = note.seconds() / speed;
            let length = (seconds * RATE as f32) as usize;

            match note {
                Note::Rest(_) => samples.resize(samples.len() + length, 0.0),
                Note::Sound { step, .. } => {
                    // The temperature moves the note off the one the letters chose, which is the
                    // only thing here that a seed can change. At zero it is the letters alone,
                    // which is what a temperature of zero means everywhere else.
                    let wandered = (seed.next_in(-2..=2) as f32 * spread).round() as i32;
                    let degree = step + wandered;
                    sing(&mut samples, pitch(centre, degree), length);

                    done += 1;
                    if report(SpeechProgress::Saying { done, expected }).is_break() {
                        return Ok(None);
                    }
                }
            }
        }

        if report(SpeechProgress::Sounding).is_break() {
            return Ok(None);
        }

        // Something rather than nothing, for text that had no letters in it at all. A clip of no
        // length is a player with nothing to play and a file some players refuse to open, which
        // reads as a bug rather than as "there was nothing to say".
        if samples.is_empty() {
            samples.resize((RATE / 10) as usize, 0.0);
        }

        Ok(Some(Sound::new(samples, RATE)))
    }
}

/// One thing to play: a tone at a scale degree, or a length of silence.
enum Note {
    Sound { step: i32, seconds: f32 },
    Rest(f32),
}

impl Note {
    fn seconds(&self) -> f32 {
        match self {
            Note::Sound { seconds, .. } => *seconds,
            Note::Rest(seconds) => *seconds,
        }
    }

    fn is_silent(&self) -> bool {
        matches!(self, Note::Rest(_))
    }
}

/// Turns text into the notes that stand in for saying it.
///
/// A tone every [`PER_NOTE`] characters of a word, pitched by what those characters are; a gap
/// between words; a longer one where the punctuation says a reader would take one. What this
/// reproduces of speech is its rhythm, which is the one part of it that can be had from the
/// letters alone.
fn read(text: &str) -> Vec<Note> {
    let mut notes = Vec::new();

    for word in text.split_whitespace() {
        let letters: Vec<char> = word.chars().filter(|c| c.is_alphanumeric()).collect();

        for chunk in letters.chunks(PER_NOTE) {
            // From the characters rather than from a counter, so that the same word is the same
            // little tune every time it appears -- which is what makes the stand-in sound like it
            // is reading something rather than playing scales.
            let mut of = Rolling::from(0);
            for letter in chunk {
                of.stir(u64::from(*letter as u32));
            }

            notes.push(Note::Sound {
                step: SCALE[of.next_below(SCALE.len())],
                seconds: NOTE,
            });
            notes.push(Note::Rest(GAP));
        }

        // The punctuation a reader would breathe at. The rest is left alone: an apostrophe or a
        // hyphen inside a word is not a pause, and treating it as one chops the word in half.
        if word.ends_with(['.', '!', '?', ',', ';', ':', '\u{2014}']) {
            notes.push(Note::Rest(PAUSE));
        }
    }

    notes
}

/// The frequency a scale degree lands on, in equal temperament: twelve steps to a doubling.
///
/// Held inside the range a voice occupies, so that a long word walking up the scale does not walk
/// off the top of one.
fn pitch(centre: f32, step: i32) -> f32 {
    let wanted = centre * 2f32.powf(step as f32 / 12.0);

    wanted.clamp(LOWEST, HIGHEST * 2.0)
}

/// Writes one tone of `length` samples onto the end of `samples`.
///
/// Three harmonics rather than one, because a sine is a test tone and anything with an overtone
/// in it is a sound. Under an envelope that rises and falls, because a tone that starts and stops
/// at full amplitude has a click at each end -- and a hundred of those in a row is what the ear
/// hears instead of the tones.
fn sing(samples: &mut Vec<f32>, hertz: f32, length: usize) {
    if length == 0 {
        return;
    }

    // Long enough to be heard as a shape rather than a click, short enough not to eat a tone this
    // short: an eighth at each end leaves three quarters of it at full amplitude.
    let edge = (length / 8).max(1);

    for at in 0..length {
        let seconds = at as f32 / RATE as f32;
        let turn = std::f32::consts::TAU * hertz * seconds;
        let wave = turn.sin() + 0.35 * (2.0 * turn).sin() + 0.15 * (3.0 * turn).sin();

        // Raised cosine at both ends, which is the shape that has no corner in it anywhere.
        let through = match (at < edge, at + edge >= length) {
            (true, _) => at as f32 / edge as f32,
            (_, true) => (length - at) as f32 / edge as f32,
            _ => 1.0,
        };
        let envelope = 0.5 - 0.5 * (std::f32::consts::PI * through.min(1.0)).cos();

        // A third of the range. The sum of the harmonics above is about 1.5 at its peak, and
        // three notes of it never overlap -- what this leaves is room rather than headroom.
        samples.push(0.33 * envelope * wave);
    }
}

/// The pitch of a recording, in hertz, where one can be made out.
///
/// Autocorrelation, which is the oldest way of asking this and the one that needs nothing but the
/// samples: a voiced sound repeats at its own period, so the lag that the waveform best matches
/// itself across is that period. Only the lags a voice could occupy are looked at, which is what
/// keeps it from answering with a harmonic or with the length of the window.
///
/// `None` where there is nothing to lock onto -- silence, noise, a recording of a door -- which
/// is a better answer than a number nothing measured.
pub fn pitch_of(sound: &Sound) -> Option<f32> {
    // Two seconds is plenty to find a pitch in and is a fixed cost whatever was dropped on the
    // page: this is quadratic in the window, and a three minute recording at 48 kHz is not.
    let most = (sound.rate as usize).saturating_mul(2);
    let samples = &sound.samples[..sound.samples.len().min(most)];

    let shortest = (sound.rate as f32 / HIGHEST) as usize;
    let longest = (sound.rate as f32 / LOWEST) as usize;
    if shortest == 0 || samples.len() < longest * 2 {
        return None;
    }

    // How much sound there is at all, to measure the match against. A recording of silence
    // correlates with itself perfectly at every lag, and the answer would be the shortest one.
    let energy: f32 = samples.iter().map(|sample| sample * sample).sum();
    if energy <= f32::EPSILON {
        return None;
    }

    // Walked from the shortest lag to the longest, and a longer one has to beat what is held by
    // a margin rather than merely tie it. That is the whole of the octave problem: a waveform
    // that repeats every T also repeats every 2T and every 3T, and correlates with itself just as
    // well at all three. Without the margin the answer is whichever of them the last bit of the
    // mantissa favoured, which is how a 330 Hz recording gets called 110.
    const CLEARLY_BETTER: f32 = 1.02;

    let mut best = (0usize, 0.0f32);
    for lag in shortest..=longest {
        let over = samples.len() - lag;
        let matched: f32 = (0..over).map(|at| samples[at] * samples[at + lag]).sum();
        // Against the length of what was compared rather than of the whole window, so that a long
        // lag is not penalised for having had less of the recording to work with.
        let scaled = matched / (over as f32).max(1.0);

        if scaled > best.1 * CLEARLY_BETTER {
            best = (lag, scaled);
        }
    }

    // A quarter of the mean energy is the line between "this repeats" and "this is noise that
    // happened to line up". Below it there is no pitch here, and saying so is the honest answer.
    let mean = energy / samples.len() as f32;
    match best.0 > 0 && best.1 > mean * 0.25 {
        true => Some(sound.rate as f32 / best.0 as f32),
        false => None,
    }
}

/// A small deterministic generator, for the one thing here that varies.
///
/// SplitMix64, which is eight lines and passes the tests that matter for choosing between five
/// notes. Nothing here is cryptographic and nothing here is sampled from a model; what a seed
/// buys is that the same text with the same number comes out the same clip, which is the whole
/// of what a seed is for.
struct Rolling(u64);

impl From<u64> for Rolling {
    fn from(seed: u64) -> Rolling {
        Rolling(seed)
    }
}

impl Rolling {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);

        z ^ (z >> 31)
    }

    /// Folds a number into the state, which is how a word is turned into its own tune.
    fn stir(&mut self, what: u64) {
        self.0 ^= what.wrapping_mul(0x9e37_79b9_7f4a_7c15);
        self.next();
    }

    fn next_below(&mut self, than: usize) -> usize {
        match than {
            0 => 0,
            than => (self.next() % than as u64) as usize,
        }
    }

    fn next_in(&mut self, range: std::ops::RangeInclusive<i64>) -> i64 {
        let width = (range.end() - range.start() + 1).max(1) as u64;

        range.start() + (self.next() % width) as i64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A run of the stand-in that nothing is watching, which is what most of these want.
    fn said(text: &str, options: SpeechOptions) -> Sound {
        Tones::new()
            .speak(text, None, &options, &mut |_| ControlFlow::Continue(()))
            .expect("the stand-in cannot fail")
            .expect("nothing stopped it")
    }

    fn at_a_seed(seed: u64) -> SpeechOptions {
        SpeechOptions {
            seed: Some(seed),
            ..SpeechOptions::default()
        }
    }

    #[test]
    fn the_stand_in_says_that_it_is_one() {
        // The one thing about it that everything above this file reads: a page that did not say
        // so would be a page claiming a voice it has not got.
        let tones = Tones::new();
        let why = tones.not_a_voice_because().expect("it is not a voice");
        assert!(why.contains("not a speech model"), "{why}");
        assert_eq!(tones.rate(), RATE);

        // And the sentence is what disappears by itself: a real model leaves the default alone.
        struct Real;
        impl Voice for Real {
            fn speak(
                &self,
                _: &str,
                _: Option<&Sound>,
                _: &SpeechOptions,
                _: &mut dyn FnMut(SpeechProgress) -> ControlFlow<()>,
            ) -> Result<Option<Sound>> {
                Ok(None)
            }
            fn rate(&self) -> u32 {
                24_000
            }
            fn defaults(&self) -> SpeechDefaults {
                SpeechDefaults::default()
            }
            fn name(&self) -> &'static str {
                "a model that does not exist yet"
            }
        }
        assert_eq!(Real.not_a_voice_because(), None);
        assert_eq!(Real.no_likeness_because(), None);
    }

    #[test]
    fn longer_text_takes_longer_to_say() {
        // The one thing about speech this stand-in reproduces honestly, and the thing every
        // screen above it is laid out around: a clip whose length has nothing to do with what was
        // typed would make the player, the bar and the gallery all untestable.
        let short = said("hello", at_a_seed(1));
        let long = said(
            "hello there, this is a much longer thing to read out",
            at_a_seed(1),
        );

        assert!(
            long.samples.len() > short.samples.len() * 4,
            "{} {}",
            long.samples.len(),
            short.samples.len()
        );
        assert_eq!(short.rate, RATE);
    }

    #[test]
    fn the_same_seed_says_it_the_same_way_twice() {
        // Which is the whole of what a seed is for, and the reason one is written beside the clip.
        assert_eq!(
            said("a sentence to read", at_a_seed(7)).samples,
            said("a sentence to read", at_a_seed(7)).samples
        );

        // And a different one does not -- at a temperature above zero, which is where the seed is
        // the only thing that can move.
        let other = said("a sentence to read", at_a_seed(8));
        assert_ne!(
            said("a sentence to read", at_a_seed(7)).samples,
            other.samples
        );
    }

    #[test]
    fn at_no_temperature_the_seed_changes_nothing() {
        // Which is what a temperature of zero means everywhere else: the likeliest reading, every
        // time, whatever number is beside it.
        let cold = |seed| {
            said(
                "the same every time",
                SpeechOptions {
                    temperature: 0.0,
                    seed: Some(seed),
                    ..SpeechOptions::default()
                },
            )
        };

        assert_eq!(cold(1).samples, cold(2).samples);
    }

    #[test]
    fn speed_is_how_long_it_takes_and_nothing_else() {
        let ordinary = said("a sentence to read out", at_a_seed(3));
        let quick = said(
            "a sentence to read out",
            SpeechOptions {
                speed: 2.0,
                ..at_a_seed(3)
            },
        );

        // Half the length, give or take the rounding of each note to a whole number of samples.
        let ratio = ordinary.samples.len() as f32 / quick.samples.len() as f32;
        assert!((ratio - 2.0).abs() < 0.05, "{ratio}");
    }

    #[test]
    fn a_speed_nothing_could_be_said_at_is_held_rather_than_divided_by() {
        // The page's box has a minimum and a request that did not come from the page does not.
        // What is on the other side of this is a division, and then a clip of no length.
        for speed in [0.0, -1.0, 1e9] {
            let sound = said(
                "hello",
                SpeechOptions {
                    speed,
                    ..at_a_seed(1)
                },
            );
            assert!(!sound.samples.is_empty(), "{speed}");
            assert!(
                sound.samples.iter().all(|sample| sample.is_finite()),
                "{speed}"
            );
        }
    }

    #[test]
    fn nothing_worth_saying_still_comes_back_as_a_clip() {
        // A clip of no length is a file some players refuse to open, which reads as a bug rather
        // than as "there was nothing in the box".
        for text in ["", "   ", "!!!", "\u{3000}"] {
            let sound = said(text, at_a_seed(1));
            assert!(!sound.samples.is_empty(), "{text:?} came back empty");
        }
    }

    #[test]
    fn nothing_it_writes_is_outside_what_a_file_holds() {
        // Everything downstream of this quantizes to sixteen bits, and a sample past the end of
        // the range is a click. The envelope and the harmonics are what have to add up.
        let sound = said(
            "The quick brown fox jumps over the lazy dog, again and again.",
            at_a_seed(5),
        );

        let loudest = sound
            .samples
            .iter()
            .fold(0.0f32, |most, sample| most.max(sample.abs()));
        assert!(loudest <= 1.0, "{loudest}");
        // And it is not silence either, which would pass the line above.
        assert!(loudest > 0.1, "{loudest}");
    }

    #[test]
    fn a_run_stops_where_it_is_asked_to() {
        // The button on the page. What comes back is nothing rather than half a clip: a stopped
        // run has not made anything worth keeping, and the gallery is what was finished.
        let stopped = Tones::new()
            .speak(
                "a good long sentence to be interrupted in the middle of",
                None,
                &at_a_seed(1),
                &mut |progress| match progress {
                    SpeechProgress::Saying { done, .. } if done > 2 => ControlFlow::Break(()),
                    _ => ControlFlow::Continue(()),
                },
            )
            .expect("stopping is not a failure");

        assert!(stopped.is_none());
    }

    #[test]
    fn what_it_is_doing_is_reported_in_the_order_it_does_it() {
        // The bar reads these, and one that arrived out of order would run backwards.
        let mut seen = Vec::new();
        said_watching("two words", &mut seen);

        assert_eq!(seen.first(), Some(&SpeechProgress::Reading));
        assert_eq!(seen.last(), Some(&SpeechProgress::Sounding));

        let counted: Vec<(i32, i32)> = seen
            .iter()
            .filter_map(|progress| match progress {
                SpeechProgress::Saying { done, expected } => Some((*done, *expected)),
                _ => None,
            })
            .collect();
        assert!(!counted.is_empty());
        // Rising by one, and the estimate the same throughout.
        for (at, (done, expected)) in counted.iter().enumerate() {
            assert_eq!(*done, at as i32 + 1);
            assert_eq!(*expected, counted[0].1);
        }
    }

    fn said_watching(text: &str, seen: &mut Vec<SpeechProgress>) {
        Tones::new()
            .speak(text, None, &at_a_seed(1), &mut |progress| {
                seen.push(progress);
                ControlFlow::Continue(())
            })
            .expect("the stand-in cannot fail");
    }

    #[test]
    fn punctuation_is_a_pause_and_an_apostrophe_is_not() {
        // A hyphen or an apostrophe inside a word is not a breath, and treating it as one chops
        // the word in half -- which is audible, and wrong in a way somebody would report.
        let notes = |text| read(text).iter().filter(|note| note.is_silent()).count();

        assert!(notes("one. two") > notes("one two"));
        assert_eq!(notes("dont"), notes("don't"));
    }

    #[test]
    fn the_same_word_is_the_same_little_tune_wherever_it_appears() {
        // From the characters rather than from a counter, which is what makes the stand-in sound
        // like it is reading something rather than playing scales.
        let steps = |text: &str| {
            read(text)
                .into_iter()
                .filter_map(|note| match note {
                    Note::Sound { step, .. } => Some(step),
                    Note::Rest(_) => None,
                })
                .collect::<Vec<_>>()
        };

        let twice = steps("hello hello");
        let (first, second) = twice.split_at(twice.len() / 2);
        assert_eq!(first, second);
        assert_ne!(steps("hello"), steps("goodbye"));
    }

    #[test]
    fn a_recording_is_where_the_pitch_comes_from() {
        // The one use the stand-in makes of an uploaded recording. Which is a use of it and not a
        // likeness of it -- the sentence the screen shows says so.
        let tone = |hertz: f32, rate: u32| {
            let samples = (0..rate)
                .map(|at| (std::f32::consts::TAU * hertz * at as f32 / rate as f32).sin())
                .collect();
            Sound::new(samples, rate)
        };

        let low = pitch_of(&tone(110.0, 16_000)).expect("a pitch");
        assert!((low - 110.0).abs() < 4.0, "{low}");

        let high = pitch_of(&tone(330.0, 44_100)).expect("a pitch");
        assert!((high - 330.0).abs() < 8.0, "{high}");

        // And it moves what comes out, which is the part the page's dropzone is for.
        let voice = Tones::new();
        let flat = |like| {
            voice
                .speak("hello there", like, &at_a_seed(1), &mut |_| {
                    ControlFlow::Continue(())
                })
                .expect("the stand-in cannot fail")
                .expect("nothing stopped it")
        };
        assert_ne!(flat(None).samples, flat(Some(&tone(110.0, 16_000))).samples);
    }

    #[test]
    fn a_recording_with_no_pitch_in_it_is_not_given_one() {
        // Silence, a recording too short to hold a period, and a rate of nothing. A number made
        // up for any of these is a number nothing measured.
        assert_eq!(pitch_of(&Sound::new(vec![0.0; 16_000], 16_000)), None);
        assert_eq!(pitch_of(&Sound::new(vec![0.5; 32], 16_000)), None);
        assert_eq!(pitch_of(&Sound::default()), None);
        assert_eq!(pitch_of(&Sound::new(vec![0.5; 16_000], 0)), None);
    }

    #[test]
    fn a_pitch_outside_a_voice_is_not_sung_outside_what_can_be_heard() {
        // A clamp rather than a trust: the pitch is an estimate, and a long word walks up the
        // scale from wherever it started.
        assert!(pitch(MIDDLE, 0) > LOWEST);
        assert!(pitch(MIDDLE, 60) <= HIGHEST * 2.0);
        assert!(pitch(MIDDLE, -60) >= LOWEST);
    }

    #[test]
    fn the_defaults_are_what_a_run_starts_from() {
        let defaults = SpeechDefaults::default();
        assert_eq!(defaults.options().speed, defaults.speed);
        assert_eq!(defaults.options().temperature, defaults.temperature);
        // Never a seed: one is put on by whoever posts the run, so that what comes out says which
        // number said it.
        assert_eq!(defaults.options().seed, None);
        assert_eq!(SpeechOptions::default(), defaults.options());
    }
}
