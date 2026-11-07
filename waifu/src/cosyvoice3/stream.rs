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

//! When a streaming reading sounds what it has said, and which of its tokens the flow is handed:
//! upstream's `CosyVoice3Model.tts` with `stream=True`, and its `llm_job`.
//!
//! # Hops
//!
//! The flow is run each time the language model has said a *hop* more tokens than were last
//! sounded, and three more besides -- the look-ahead the flow's condition reads past the hop. A
//! hop is 25 tokens at first, and doubles each time to at most 100. The first of a sentence is
//! longer by however many tokens bring the recording's own up to a multiple of the hop: so every
//! hop starts where a block of the DiT's attention does, at a multiple of 25 tokens -- 50 frames
//! -- counted from the start of the recording.
//!
//! The hop is kept across the sentences of a reading, as upstream keeps `self.token_hop_len`:
//! a reading's second sentence starts at the hop its first left off at. Upstream keeps it across
//! readings as well, which this does not -- every reading starts at 25.
//!
//! # Silence
//!
//! CosyVoice3's language model says a few tokens that are the sound of nothing much -- silence and
//! breath. `llm_job` hands the flow no more than five of them in a row: the sixth and every one
//! after it, until something else is said, the language model goes on reading but the flow is
//! never handed. It is what every reading does, streaming or not.

/// The first hop, and what it grows from.
pub const FIRST_HOP: usize = 25;
/// The longest a hop grows to.
pub const LONGEST_HOP: usize = 4 * FIRST_HOP;
/// How much a hop grows by each time.
const GROWTH: usize = 2;
/// How many tokens past a hop the flow's condition reads: its look-ahead layer's.
pub const LOOKAHEAD: usize = 3;

/// `CosyVoice3Model.silent_tokens`.
const SILENT: [i32; 11] = [1, 2, 28, 29, 55, 248, 494, 2241, 2242, 2322, 2323];
/// The most of them in a row the flow is handed.
const MOST_SILENT: usize = 5;

/// What `llm_job` hands the flow of what the language model said: each token as it comes.
#[derive(Clone, Debug, Default)]
pub struct Silence {
    /// Silent tokens in a row, those held back included.
    run: usize,
}

impl Silence {
    /// Whether the flow is handed `token`, said next.
    pub fn keeps(&mut self, token: i32) -> bool {
        if SILENT.contains(&token) {
            self.run += 1;
            self.run <= MOST_SILENT
        } else {
            self.run = 0;
            true
        }
    }

    /// Everything in `said` the flow is handed, from the start of a sentence.
    pub fn kept(said: &[i32]) -> Vec<i32> {
        let mut silence = Silence::default();
        said.iter()
            .copied()
            .filter(|token| silence.keeps(*token))
            .collect()
    }
}

/// How long the next hop is: upstream's `self.token_hop_len`, kept for a whole reading.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Hops {
    hop: usize,
}

impl Default for Hops {
    fn default() -> Hops {
        Hops { hop: FIRST_HOP }
    }
}

impl Hops {
    pub fn hop(&self) -> usize {
        self.hop
    }

    fn grow(&mut self) {
        self.hop = (self.hop * GROWTH).min(LONGEST_HOP);
    }

    /// Where one sentence is, continuing a recording of `prompt` tokens.
    pub fn sentence(&self, prompt: usize) -> Sentence {
        Sentence {
            sounded: 0,
            pad: prompt.div_ceil(self.hop) * self.hop - prompt,
        }
    }
}

/// How far one sentence of a streaming reading has been sounded.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Sentence {
    /// The tokens before this, of those the flow is handed, are sounded: `token_offset`.
    sounded: usize,
    /// How much longer the first hop is: `prompt_token_pad`.
    pad: usize,
}

impl Sentence {
    pub fn sounded(&self) -> usize {
        self.sounded
    }

    fn this_hop(&self, hops: &Hops) -> usize {
        match self.sounded {
            0 => hops.hop + self.pad,
            _ => hops.hop,
        }
    }

    /// With `kept` tokens to hand the flow, how many of them it reads for the next hop, if one is
    /// due: the hop and its look-ahead.
    pub fn due(&self, hops: &Hops, kept: usize) -> Option<usize> {
        let reads = self.sounded + self.this_hop(hops) + LOOKAHEAD;
        (kept >= reads).then_some(reads)
    }

    /// The hop has been sounded: the next starts after it, and is longer.
    pub fn sounded_hop(&mut self, hops: &mut Hops) {
        self.sounded += self.this_hop(hops);
        hops.grow();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_flow_is_handed_no_more_than_five_silent_tokens_in_a_row() {
        let said = [7, 1, 2, 28, 29, 55, 248, 494, 9, 1, 1];
        assert_eq!(Silence::kept(&said), [7, 1, 2, 28, 29, 55, 9, 1, 1]);
        // Held back or not, a silent token still counts towards the run.
        assert_eq!(Silence::kept(&[1; 8]), [1; 5]);
        assert!(Silence::kept(&[]).is_empty());
    }

    #[test]
    fn hops_grow_from_twenty_five_to_a_hundred() {
        let mut hops = Hops::default();
        // 87 tokens of recording: thirteen more make 100, four hops of 25.
        let mut sentence = hops.sentence(87);
        assert_eq!(sentence.due(&hops, 40), None);
        assert_eq!(sentence.due(&hops, 41), Some(41));
        sentence.sounded_hop(&mut hops);
        assert_eq!((sentence.sounded(), hops.hop()), (38, 50));
        assert_eq!(sentence.due(&hops, 90), None);
        assert_eq!(sentence.due(&hops, 91), Some(91));
        sentence.sounded_hop(&mut hops);
        sentence.sounded_hop(&mut hops);
        sentence.sounded_hop(&mut hops);
        assert_eq!((sentence.sounded(), hops.hop()), (288, 100));

        // The next sentence starts at the hop the last left off at, padded to a hundred.
        let next = hops.sentence(87);
        assert_eq!(next.due(&hops, 115), None);
        assert_eq!(next.due(&hops, 116), Some(116));
    }

    #[test]
    fn every_hop_starts_where_a_block_of_attention_does() {
        // Blocks of 50 frames are 25 tokens counted from the start of the recording.
        for prompt in 1..400 {
            let mut hops = Hops::default();
            for _ in 0..3 {
                let mut sentence = hops.sentence(prompt);
                for _ in 0..6 {
                    sentence.sounded_hop(&mut hops);
                    assert_eq!((prompt + sentence.sounded()) % FIRST_HOP, 0, "{prompt}");
                }
            }
        }
    }
}
