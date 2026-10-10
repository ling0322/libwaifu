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

//! How far along a fetch, a read or a run is: what a bar is drawn from, and the words on it.

use std::time::Instant;

use crate::{ConversionProgress, GenerationProgress, SpeechProgress};

/// How much of a run the parts that are not steps are worth, when a bar is drawn from them.
///
/// Weighed rather than counted. Reading the prompt costs about a step and turning the finished
/// latent into pixels costs several, so a bar drawn from the steps alone would fill up and then
/// sit there for the slowest part of the run.
const ENCODING: f64 = 1.0;
const DECODING: f64 = 3.0;

/// How long a voice's vocoder takes, as a multiple of everything before it, until a reading has
/// measured it.
///
/// A speech run cannot be weighed the way a picture is. The vocoder at the end is a pass over the
/// whole utterance, so it grows with the token count rather than costing a fixed few steps, and
/// how much of the run it is depends on the model and on the card: IndexTTS spends about twice as
/// long on it as on its tokens, CosyVoice3 about half as long. One is a guess for the first
/// reading; every reading after it uses what the last one took.
pub const VOCODER_RATIO: f64 = 1.0;

/// Where the speech bar waits while the vocoder runs longer than the last one did: short of the
/// end, which it reaches when there is a clip.
const HELD: f64 = 0.99;

/// And for a conversion, as shares of the whole, measured over 36 seconds on one card: re-saying
/// the source in the reference's manner, where that was asked for, is about two fifths of the
/// run; of the rest, the vocoder takes a little more than the steps it follows do.
const SAYING: f64 = 0.4;
const DRAWING: f64 = 0.45;

/// Which package of a model is being fetched, and how far into it.
#[derive(Clone)]
pub struct Fetch {
    pub model: String,
    /// Which hub it is coming from, once the fetch has settled that. Unknown for the moment
    /// before: it is decided by a probe, and a screen that guessed would be naming a hub while
    /// the probe is still out.
    pub hub: Option<&'static str>,
    pub file: String,
    pub done: u64,
    pub total: Option<u64>,
    pub part: usize,
    /// How many packages there are in all, or zero while that is still unknown.
    pub parts: usize,
}

/// A run in flight.
#[derive(Clone)]
pub struct Run {
    pub progress: GenerationProgress,
    /// How many steps this run was asked for, which the bar needs after the last one.
    pub steps: i32,
    pub started: Instant,
}

/// A reading in flight.
#[derive(Clone)]
pub struct Say {
    pub progress: SpeechProgress,
    /// How many tokens this reading is expected to take, which the bar needs after the last one
    /// and before the first.
    ///
    /// Unlike a picture's step count this is not known when the run is posted -- it is the
    /// model's estimate, and it arrives with the first token. Until then it is what the length of
    /// the text suggests, so that the bar has somewhere to start rather than jumping once the
    /// model has an opinion.
    pub expected: i32,
    pub started: Instant,
    /// When the vocoder started, which is when the tokens stopped saying how far along it is.
    pub sounding: Option<Instant>,
    /// How long the vocoder is expected to take, as a multiple of everything before it: what the
    /// last reading with this program took, or [`VOCODER_RATIO`] before there was one.
    pub ratio: f64,
}

impl Say {
    /// How long the vocoder took against everything before it, for a reading that has finished.
    pub fn measured_ratio(&self, finished: Instant) -> Option<f64> {
        let sounding = self.sounding?;
        let before = sounding.duration_since(self.started).as_secs_f64();
        let after = finished.saturating_duration_since(sounding).as_secs_f64();
        (before > 0.0).then(|| (after / before).clamp(0.05, 20.0))
    }
}

/// A conversion in flight: the latest of each count it reports, which between them say how far
/// along it is.
#[derive(Clone)]
pub struct Convert {
    /// Whether it re-says the source as well, which is the part [`ConversionProgress::Saying`]
    /// reports and the timbre-only path does not have.
    pub style: bool,
    pub said: (i32, i32),
    pub drawn: (i32, i32),
    pub sounded: (i32, i32),
    /// The last thing it said it was doing, for the words on the bar.
    pub progress: ConversionProgress,
    pub started: Instant,
}

impl Convert {
    pub fn new(style: bool) -> Convert {
        Convert {
            style,
            said: (0, 0),
            drawn: (0, 0),
            sounded: (0, 0),
            progress: ConversionProgress::Listening,
            started: Instant::now(),
        }
    }

    /// Takes in what the converter just reported.
    pub fn heard(&mut self, progress: ConversionProgress) {
        match progress {
            ConversionProgress::Listening => {}
            ConversionProgress::Saying { done, expected } => self.said = (done, expected),
            ConversionProgress::Drawing { done, total } => self.drawn = (done, total),
            ConversionProgress::Sounding { done, total } => self.sounded = (done, total),
        }
        self.progress = progress;
    }
}

/// What the worker is busy with, which is the only thing on the screen that moves on its own.
#[derive(Clone)]
pub enum Doing {
    /// Nothing: either no model has been asked for yet, or the last thing asked for has finished.
    Nothing,
    Fetching(Fetch),
    /// Reading a model off the disk and onto the device, which is minutes for a large one.
    Reading {
        model: String,
    },
    Drawing(Run),
    Speaking(Say),
    Converting(Convert),
}

impl Doing {
    /// Whether the worker is busy, which is what makes a second request wait its turn.
    pub fn is_busy(&self) -> bool {
        !matches!(self, Doing::Nothing)
    }

    /// How far along, between nothing and all of it, or `None` for work with no end in sight.
    ///
    /// A fetch knows how many bytes it is; reading a model does not know anything -- it is one
    /// call into the tensor library that returns when it returns -- and a bar drawn for it would
    /// be a bar making something up.
    pub fn fraction(&self) -> Option<f64> {
        match self {
            Doing::Nothing | Doing::Reading { .. } => None,
            Doing::Fetching(fetch) => fetch
                .total
                .filter(|total| *total > 0)
                .map(|total| fetch.done as f64 / total as f64),
            Doing::Drawing(run) => Some(drawn(run.progress, run.steps)),
            Doing::Speaking(say) => Some(spoken(say, Instant::now())),
            Doing::Converting(convert) => Some(converted(convert)),
        }
    }

    /// What it is busy with, in the words the bar carries.
    pub fn words(&self) -> String {
        match self {
            Doing::Nothing => String::new(),
            Doing::Fetching(fetch) if fetch.file.is_empty() => format!("fetching {}", fetch.model),
            Doing::Fetching(fetch) => match fetch.parts {
                0 => format!("fetching {} -- {}", fetch.model, fetch.file),
                parts => format!(
                    "fetching {} -- {} ({} of {parts})",
                    fetch.model, fetch.file, fetch.part
                ),
            },
            Doing::Reading { model } => format!("reading {model}"),
            Doing::Drawing(run) => match run.progress {
                GenerationProgress::Encoding => "reading the prompt".to_string(),
                GenerationProgress::Step { done, total } => format!("step {done} of {total}"),
                GenerationProgress::Decoding => "making the picture".to_string(),
            },
            Doing::Speaking(say) => match say.progress {
                SpeechProgress::Reading => "reading the text".to_string(),
                // "of about", because the number on the right is worked out from the length of
                // the text rather than known: a model that decides what to say one token at a
                // time does not know how many there will be until it stops. A bar that said "of"
                // and then went past it would be a bar that had been lying.
                SpeechProgress::Saying { done, expected } => {
                    format!("token {done} of about {expected}")
                }
                SpeechProgress::Sounding => "making the sound".to_string(),
            },
            Doing::Converting(convert) => match convert.progress {
                ConversionProgress::Listening => "listening to the recordings".to_string(),
                ConversionProgress::Saying { done, expected } => {
                    format!("token {done} of about {expected}")
                }
                ConversionProgress::Drawing { done, total } => format!("step {done} of {total}"),
                // The window being sounded is the one after those already done.
                ConversionProgress::Sounding { done, total } => match total {
                    1 => "making the sound".to_string(),
                    total => format!("making the sound, {} of {total}", (done + 1).min(total)),
                },
            },
        }
    }
}

/// How far along a run is, as a bar can show it.
fn drawn(progress: GenerationProgress, steps: i32) -> f64 {
    let all = |steps: i32| ENCODING + f64::from(steps.max(1)) + DECODING;
    match progress {
        GenerationProgress::Encoding => 0.0,
        GenerationProgress::Step { done, total } => (ENCODING + f64::from(done)) / all(total),
        GenerationProgress::Decoding => (ENCODING + f64::from(steps.max(1))) / all(steps),
    }
}

/// How far along a reading is, as a bar can show it.
///
/// The tokens fill the bar's first `1 / (1 + ratio)` and the vocoder the rest. The vocoder says
/// nothing while it runs, so its part is filled by the clock: the time everything before it took,
/// times the ratio, is how long it should take, and the bar is held short of the end if it takes
/// longer.
///
/// The token count is an estimate, so `done` can pass `expected` on a reading that runs long;
/// what that has to not do is run the bar off the end and back round, so it is held at the last
/// token instead -- which is where a run that is taking longer than expected actually is.
fn spoken(say: &Say, now: Instant) -> f64 {
    let tokens = 1.0 / (1.0 + say.ratio.max(0.0));
    let expected = f64::from(say.expected.max(1));

    match say.progress {
        SpeechProgress::Reading => 0.0,
        SpeechProgress::Saying { done, .. } => {
            tokens * f64::from(done.max(0)).min(expected) / expected
        }
        SpeechProgress::Sounding => {
            let Some(sounding) = say.sounding else {
                return tokens;
            };
            let before = sounding.duration_since(say.started).as_secs_f64();
            let after = now.saturating_duration_since(sounding).as_secs_f64();
            let expected = before * say.ratio;
            let sounded = match expected > 0.0 {
                true => (after / expected).min(1.0),
                false => 1.0,
            };
            (tokens + (1.0 - tokens) * sounded).min(HELD)
        }
    }
}

/// How far along a conversion is, as a bar can show it: each of its counts as a share of the run.
///
/// Every count is over the whole conversion rather than the window it is on, so the bar only moves
/// forward. A count with nothing in it yet is nothing done.
fn converted(convert: &Convert) -> f64 {
    let share = |(done, total): (i32, i32)| match total {
        total if total > 0 => (f64::from(done.max(0)) / f64::from(total)).min(1.0),
        _ => 0.0,
    };
    let sound = DRAWING * share(convert.drawn) + (1.0 - DRAWING) * share(convert.sounded);
    match convert.style {
        true => SAYING * share(convert.said) + (1.0 - SAYING) * sound,
        false => sound,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::time::Duration;

    #[test]
    fn the_bar_weighs_the_parts_that_are_not_steps() {
        // Reading the prompt costs about a step and the decode costs several, so a run that has
        // finished every step is not a run that has finished.
        let all_the_steps = drawn(GenerationProgress::Decoding, 30);
        assert!(all_the_steps < 1.0, "{all_the_steps}");
        assert!(all_the_steps > 0.8, "{all_the_steps}");

        assert_eq!(drawn(GenerationProgress::Encoding, 30), 0.0);
        assert!(
            drawn(
                GenerationProgress::Step {
                    done: 15,
                    total: 30
                },
                30
            ) > 0.4,
            "half the steps is about half way"
        );
    }

    #[test]
    fn the_conversion_bar_only_moves_forward_and_ends_full() {
        // The style path: two pieces, each re-said and then drawn and sounded, in that order.
        let mut convert = Convert::new(true);
        let mut last = converted(&convert);
        assert_eq!(last, 0.0);
        for progress in [
            ConversionProgress::Listening,
            ConversionProgress::Saying {
                done: 500,
                expected: 900,
            },
            ConversionProgress::Drawing {
                done: 15,
                total: 60,
            },
            ConversionProgress::Drawing {
                done: 30,
                total: 60,
            },
            // Said as the vocoder starts on a window and again as it finishes it.
            ConversionProgress::Sounding { done: 0, total: 2 },
            ConversionProgress::Sounding { done: 1, total: 2 },
            // The estimate was short: it grows with what is said, and the bar holds.
            ConversionProgress::Saying {
                done: 1000,
                expected: 1000,
            },
            ConversionProgress::Drawing {
                done: 60,
                total: 60,
            },
            ConversionProgress::Sounding { done: 2, total: 2 },
        ] {
            convert.heard(progress);
            let now = converted(&convert);
            assert!(now >= last, "{progress:?}: {now} after {last}");
            last = now;
        }
        assert!((last - 1.0).abs() < 1e-9, "{last}");

        // Timbre only has nothing to re-say, so drawing and sounding are the whole of it.
        let mut convert = Convert::new(false);
        convert.heard(ConversionProgress::Drawing {
            done: 30,
            total: 30,
        });
        assert!((converted(&convert) - DRAWING).abs() < 1e-9);
        let words = Doing::Converting(convert).words();
        assert_eq!(words, "step 30 of 30");
    }

    #[test]
    fn a_run_of_no_steps_does_not_divide_by_nothing() {
        // Not something the page can ask for -- the box has a minimum -- and the arithmetic has
        // to hold anyway, since what it would do instead is produce an infinity and paint with it.
        for progress in [
            GenerationProgress::Encoding,
            GenerationProgress::Step { done: 0, total: 0 },
            GenerationProgress::Decoding,
        ] {
            assert!(drawn(progress, 0).is_finite(), "{progress:?}");
        }
    }

    #[test]
    fn reading_a_model_does_not_claim_to_know_how_far_along_it_is() {
        // It is one call into the tensor library that returns when it returns. A fraction here
        // would be a number nothing measured.
        let reading = Doing::Reading {
            model: "sdxl:base".to_string(),
        };
        assert_eq!(reading.fraction(), None);
        assert!(reading.is_busy());
        assert!(reading.words().contains("sdxl:base"));
    }

    #[test]
    fn a_fetch_of_an_unknown_length_has_no_fraction_either() {
        // Which is what a server that did not say how long the file is leaves, and it is also
        // what the moment before the first byte looks like.
        let mut fetch = Fetch {
            model: "sdxl:base".to_string(),
            hub: None,
            file: "sdxl-base.waifupkg".to_string(),
            done: 512,
            total: None,
            part: 1,
            parts: 3,
        };
        assert_eq!(Doing::Fetching(fetch.clone()).fraction(), None);

        fetch.total = Some(1024);
        assert_eq!(Doing::Fetching(fetch.clone()).fraction(), Some(0.5));

        // And the words say which model, not only which file: the file names in a package are not
        // something anybody recognises.
        let words = Doing::Fetching(fetch).words();
        assert!(words.contains("sdxl:base"), "{words}");
        assert!(words.contains("1 of 3"), "{words}");
    }

    #[test]
    fn nothing_happening_is_not_busy() {
        assert!(!Doing::Nothing.is_busy());
        assert_eq!(Doing::Nothing.fraction(), None);
        assert_eq!(Doing::Nothing.words(), "");
    }

    /// A reading that started `before` seconds ago and is at `progress`, expecting the vocoder to
    /// take `ratio` times as long as everything before it.
    fn a_reading(progress: SpeechProgress, expected: i32, ratio: f64, before: u64) -> Say {
        let now = Instant::now();
        let started = now - Duration::from_secs(before);
        Say {
            progress,
            expected,
            started,
            sounding: matches!(progress, SpeechProgress::Sounding).then_some(now),
            ratio,
        }
    }

    #[test]
    fn the_speech_bar_gives_the_vocoder_the_share_it_last_took() {
        // IndexTTS spends about twice as long sounding as saying. A bar that weighed the vocoder
        // as a few tokens was at 97% with more than half the run still to go.
        let reading = |progress| a_reading(progress, 40, 2.0, 4);
        let at = |say: &Say| spoken(say, say.sounding.unwrap_or(say.started));

        assert_eq!(at(&reading(SpeechProgress::Reading)), 0.0);
        let half = at(&reading(SpeechProgress::Saying {
            done: 20,
            expected: 40,
        }));
        assert!((half - 1.0 / 6.0).abs() < 1e-9, "{half}");
        let sounding = at(&reading(SpeechProgress::Sounding));
        assert!((sounding - 1.0 / 3.0).abs() < 1e-9, "{sounding}");
    }

    #[test]
    fn the_speech_bar_moves_by_the_clock_while_the_vocoder_runs() {
        // Four seconds before the vocoder and a ratio of two: eight seconds of it expected.
        let say = a_reading(SpeechProgress::Sounding, 40, 2.0, 4);
        let sounding = say.sounding.unwrap();
        let after = |seconds: u64| spoken(&say, sounding + Duration::from_secs(seconds));

        assert!((after(4) - 2.0 / 3.0).abs() < 1e-9, "{}", after(4));
        assert!(after(2) < after(4) && after(4) < after(6));
        // And one that takes longer than the last waits short of the end rather than at it.
        assert_eq!(after(8), HELD);
        assert_eq!(after(60), HELD);
    }

    #[test]
    fn what_a_reading_took_is_what_the_next_one_expects() {
        let say = a_reading(SpeechProgress::Sounding, 40, 1.0, 4);
        let sounding = say.sounding.unwrap();
        let ratio = say
            .measured_ratio(sounding + Duration::from_secs(6))
            .unwrap();
        assert!((ratio - 1.5).abs() < 1e-6, "{ratio}");

        // A reading that never reached the vocoder measured nothing.
        let stopped = a_reading(SpeechProgress::Reading, 40, 1.0, 4);
        assert_eq!(stopped.measured_ratio(Instant::now()), None);
    }

    #[test]
    fn a_reading_that_runs_long_does_not_run_the_bar_off_the_end() {
        // The token count is an estimate, so a reading can pass it -- which is a bar that fills
        // up and starts again if nothing holds it.
        let past = a_reading(
            SpeechProgress::Saying {
                done: 400,
                expected: 40,
            },
            40,
            1.0,
            4,
        );
        let sounding = a_reading(SpeechProgress::Sounding, 40, 1.0, 4);
        assert_eq!(
            spoken(&past, past.started),
            spoken(&sounding, sounding.sounding.unwrap())
        );

        // And a reading with no estimate yet, or no time before the vocoder, does not divide by
        // one that is not there.
        for expected in [0, -1] {
            for progress in [
                SpeechProgress::Reading,
                SpeechProgress::Saying {
                    done: 0,
                    expected: 0,
                },
                SpeechProgress::Sounding,
            ] {
                for ratio in [0.0, 1.0] {
                    let say = a_reading(progress, expected, ratio, 0);
                    assert!(spoken(&say, Instant::now()).is_finite(), "{progress:?}");
                }
            }
        }
    }

    #[test]
    fn what_is_being_said_is_reported_without_claiming_to_know_how_long_it_will_be() {
        // "of about", because the number on the right is worked out from the text rather than
        // known. A bar that said "of" and then went past it would have been lying.
        let saying = Doing::Speaking(Say {
            progress: SpeechProgress::Saying {
                done: 3,
                expected: 40,
            },
            expected: 40,
            started: Instant::now(),
            sounding: None,
            ratio: VOCODER_RATIO,
        });

        assert!(saying.is_busy());
        let words = saying.words();
        assert!(words.contains("3 of about 40"), "{words}");
        assert!(saying.fraction().is_some());
    }
}
