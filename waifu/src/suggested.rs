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

//! What a model suggests being asked for: the `suggested:` block of a [`Manifest`].
//!
//! The rest of the manifest says what the model *is* -- how many layers, how wide, which packages
//! the weights are in -- and every key of it has to be there or the model cannot be built. This is
//! the other kind of thing: what its author says to start from. A turbo release is distilled for
//! eight steps at no guidance and gives a burnt, over-contrasted picture at thirty and five; a
//! fine tune trained on danbooru tags wants a list of tags and says so on its card, which nobody
//! who fetched the packages ever sees.
//!
//! ```yaml
//! suggested:
//!   prompt: masterpiece, best quality
//!   sizes:
//!     - [1024, 1024]
//!     - [832, 1216]
//!     - [1216, 832]
//!   steps: 8
//!   guidance: 1.0
//!   takes_guidance: false
//! ```
//!
//! Every field is optional and is [`None`] when the model does not say, because none of it is
//! needed to load a model -- and because "this model has no opinion" is a real answer, distinct
//! from any particular number. What fills the gaps is [`GenerationDefaults`], which is what this
//! build believed before a model could say anything at all.
//!
//! A manifest with no `suggested:` block reads as [`Suggestions::default`] rather than as an
//! error, and a key this version does not know is skipped rather than refused, so a newer model
//! keeps working here and an older one keeps working there.

use std::collections::BTreeMap;

use crate::generation::GenerationDefaults;
use crate::yaml::Node;

/// One size a model says it draws well at, in pixels.
pub type Size = (i32, i32);

/// What a model suggests being asked for.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Suggestions {
    /// The prompt the model card suggests, where the card had one. A starting point for whoever
    /// draws with the model, not a setting.
    pub prompt: Option<String>,
    /// What it suggests steering away from -- the negative prompt, under the name the screen
    /// gives it rather than the one the API does. Cards publish one as often as they publish a
    /// prompt, and a fine tune's is not generic: NoobAI names the tags it was trained to treat as
    /// spoiled, which are not the ones another model would name. [`None`] leaves the box empty,
    /// which is not the same as a model suggesting the empty string -- the model has an opinion
    /// about that too.
    pub avoid: Option<String>,
    /// The sizes it was trained to draw at, widest-recommended first: SDXL and its fine tunes are
    /// trained on a set of aspect ratios rather than on one square, and a card that lists them is
    /// listing several. Empty when the model does not say, which is the same answer as [`None`]
    /// and is spelled the way a list spells it.
    pub sizes: Vec<Size>,
    /// How many denoising steps it is distilled for, and how hard to push away from the
    /// unprompted answer. A turbo release and an aesthetic one want very different numbers here,
    /// and this is how each says which.
    pub steps: Option<i32>,
    pub guidance: Option<f32>,
    /// Whether asking this model for guidance means anything at all.
    ///
    /// The rest of this block is advice and this one is not: a model distilled to draw in a few
    /// steps was trained to answer as though it had already been guided, and there is no second
    /// answer left to push away from. Krea 2 Turbo is the one here. Its reference pipeline takes
    /// no negative prompt and has no guidance to configure -- not a dial set to zero, but no dial
    /// -- and handing it either is asking for a picture nobody promised.
    ///
    /// Which is why it is one field and not two. Guidance is the second pass through the
    /// denoiser and the negative prompt is what that pass is given; a model without the one has
    /// no use for the other, and a screen that offers a box the model will discard is a screen
    /// that lies about what it did.
    ///
    /// [`None`] like everything else here, and for the same reason: a package written before
    /// there was a key for this says nothing, and nothing is not the same answer as yes. What
    /// fills the gap is what the screen already believed about the kind of model, which for the
    /// only distilled release published here is the right answer without the key.
    pub takes_guidance: Option<bool>,
}

impl Suggestions {
    /// What each field is called in the manifest.
    pub const PROMPT: &'static str = "prompt";
    pub const AVOID: &'static str = "avoid";
    pub const SIZES: &'static str = "sizes";
    pub const STEPS: &'static str = "steps";
    pub const GUIDANCE: &'static str = "guidance";
    pub const TAKES_GUIDANCE: &'static str = "takes_guidance";

    /// The suggestions a `suggested:` block holds.
    ///
    /// A field that does not parse as what it is read as is [`None`] rather than an error. This is
    /// advice: a model whose card says its guidance is "about 5" should still load and draw, with
    /// the box starting where it always did, rather than fail to open over a line nothing has to
    /// read. Keys this version does not recognise are left where they are, for the same reason.
    pub fn from_fields(fields: &BTreeMap<String, Node>) -> Suggestions {
        // Written but empty is the same as not written: an empty prompt box is what an absent
        // suggestion already gives.
        let text = |key: &str| {
            fields
                .get(key)
                .and_then(Node::as_str)
                .map(str::trim)
                .filter(|value| !value.is_empty())
        };
        Suggestions {
            prompt: text(Suggestions::PROMPT).map(str::to_string),
            avoid: text(Suggestions::AVOID).map(str::to_string),
            sizes: fields.get(Suggestions::SIZES).map(sizes).unwrap_or_default(),
            steps: text(Suggestions::STEPS).and_then(|value| value.parse().ok()),
            guidance: text(Suggestions::GUIDANCE).and_then(|value| value.parse().ok()),
            // Absent, or written as a word this cannot read, is a model that has not answered --
            // which is not the same as one that answered yes, and is left to whoever asked to
            // fill in. A line nothing could read is the one case where guessing would take a
            // control off the screen that somebody then goes looking for.
            takes_guidance: text(Suggestions::TAKES_GUIDANCE).and_then(truth),
        }
    }

    /// The size to start at, which is the first the model lists.
    pub fn size(&self) -> Option<Size> {
        self.sizes.first().copied()
    }

    /// `defaults` with whatever this model had an opinion about replaced by its opinion.
    ///
    /// The two are asked in that order on purpose: `defaults` is what this build believes about
    /// the kind of model, and a model that says nothing gets exactly that, which is what every
    /// model got before there was anywhere to say it.
    pub fn over(&self, defaults: GenerationDefaults) -> GenerationDefaults {
        let (width, height) = self.size().unwrap_or((defaults.width, defaults.height));

        GenerationDefaults {
            width,
            height,
            num_steps: self.steps.unwrap_or(defaults.num_steps),
            guidance_scale: self.guidance.unwrap_or(defaults.guidance_scale),
        }
    }
}

/// One yes or no, however the manifest spelled it.
///
/// YAML has more ways of writing a boolean than Rust's `parse` knows, and the exporter that
/// writes these is Python, where the word is `False`. None where it is none of them, which the
/// caller reads as the model not having said.
fn truth(text: &str) -> Option<bool> {
    match text.to_ascii_lowercase().as_str() {
        "true" | "yes" | "on" | "1" => Some(true),
        "false" | "no" | "off" | "0" => Some(false),
        _ => None,
    }
}

/// Every `[width, height]` of a `sizes:` list.
///
/// An item that is not a pair of numbers is dropped rather than refused, and a `sizes:` that is
/// not a list at all reads as no sizes: this is advice, and the cost of ignoring a malformed
/// piece of it is a box that starts where it used to.
fn sizes(node: &Node) -> Vec<Size> {
    let Some(items) = node.as_seq() else {
        return Vec::new();
    };

    items
        .iter()
        .filter_map(|item| match item.as_seq() {
            Some([width, height]) => {
                let width = width.as_str()?.trim().parse().ok()?;
                let height = height.as_str()?.trim().parse().ok()?;
                Some((width, height))
            }
            _ => None,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::yaml;

    /// The `suggested:` block of a manifest holding `text`, as the reader hands it over.
    fn fields(text: &str) -> BTreeMap<String, Node> {
        yaml::parse(text)
            .unwrap()
            .into_map("suggested")
            .unwrap()
            .remove("suggested")
            .unwrap()
            .into_map("suggested")
            .unwrap()
    }

    #[test]
    fn reads_everything_a_model_suggests() {
        let suggested = Suggestions::from_fields(&fields(
            "suggested:\n  \
             prompt: masterpiece, best quality,\n  \
             avoid: worst quality, bad hands\n  \
             sizes:\n    \
             - [1024, 1024]\n    \
             - [832, 1216]\n    \
             - [1216, 832]\n  \
             steps: 8\n  \
             guidance: 1.0\n  \
             takes_guidance: false\n",
        ));

        assert_eq!(
            suggested.prompt.as_deref(),
            Some("masterpiece, best quality,")
        );
        assert_eq!(suggested.avoid.as_deref(), Some("worst quality, bad hands"));
        assert_eq!(
            suggested.sizes,
            vec![(1024, 1024), (832, 1216), (1216, 832)]
        );
        assert_eq!(suggested.size(), Some((1024, 1024)));
        assert_eq!(suggested.steps, Some(8));
        assert_eq!(suggested.guidance, Some(1.0));
        assert_eq!(suggested.takes_guidance, Some(false));
    }

    #[test]
    fn a_model_says_in_so_many_words_when_it_takes_no_guidance() {
        // Silence is not an answer either way: it is left for whatever the screen already knew
        // about the kind of model, which is the only thing an older package can be read against.
        assert_eq!(Suggestions::default().takes_guidance, None);
        assert_eq!(
            Suggestions::from_fields(&fields("suggested:\n  steps: 8\n")).takes_guidance,
            None
        );

        // And the spellings a hand written manifest reaches for, in either case. The exporter is
        // Python and writes `False`, which is not a word Rust's own `parse` knows.
        for no in ["false", "False", "FALSE", "no", "off", "0"] {
            let said =
                Suggestions::from_fields(&fields(&format!("suggested:\n  takes_guidance: {no}\n")));
            assert_eq!(said.takes_guidance, Some(false), "{no} should read as no");
        }
        for yes in ["true", "True", "yes", "on", "1"] {
            let said = Suggestions::from_fields(&fields(&format!(
                "suggested:\n  takes_guidance: {yes}\n"
            )));
            assert_eq!(said.takes_guidance, Some(true), "{yes} should read as yes");
        }

        // And the spelling the exporter actually writes, which is quoted: `_quote` in
        // `model_writer.py` puts quotes round anything YAML would otherwise read as a boolean of
        // its own, so the published manifest says `takes_guidance: "false"` and never the bare
        // word. This is the line that would have caught reading it as the four characters of a
        // quoted string.
        let published =
            Suggestions::from_fields(&fields("suggested:\n  takes_guidance: \"false\"\n"));
        assert_eq!(published.takes_guidance, Some(false));

        // A word neither way is no answer, and no answer leaves the dial where it was. Taking a
        // control off the screen over a line nothing could read is the one direction that leaves
        // somebody hunting for it.
        let vague = Suggestions::from_fields(&fields("suggested:\n  takes_guidance: sometimes\n"));
        assert_eq!(vague.takes_guidance, None);
    }

    #[test]
    fn the_sizes_may_be_written_on_one_line() {
        // A flow sequence and a block one are the same list, which is the point of accepting both.
        let inline =
            Suggestions::from_fields(&fields("suggested:\n  sizes: [[1024, 1024], [832, 1216]]\n"));
        assert_eq!(inline.sizes, vec![(1024, 1024), (832, 1216)]);
    }

    #[test]
    fn a_model_with_nothing_to_say_suggests_nothing() {
        let nothing = Suggestions::default();
        assert_eq!(nothing.takes_guidance, None);
        assert_eq!(nothing.prompt, None);
        assert_eq!(nothing.steps, None);
        assert_eq!(nothing.guidance, None);
        assert!(nothing.sizes.is_empty());
        assert_eq!(nothing.size(), None);

        // Written but empty, which a hand edited manifest is one backspace away from.
        let empty = Suggestions::from_fields(&fields("suggested:\n  prompt: \"\"\n  steps: \"\"\n"));
        assert_eq!(empty, Suggestions::default());
    }

    #[test]
    fn a_field_that_does_not_parse_is_no_suggestion_rather_than_a_failure() {
        // Advice, not a setting: a model whose card says "about 8" still loads and draws.
        let vague = Suggestions::from_fields(&fields(
            "suggested:\n  \
             prompt: 1girl\n  \
             steps: about 8\n  \
             guidance: five\n  \
             sizes:\n    \
             - [1024, 1024]\n    \
             - [832]\n    \
             - [wide, 1216]\n",
        ));

        assert_eq!(vague.prompt.as_deref(), Some("1girl"));
        assert_eq!(vague.steps, None);
        assert_eq!(vague.guidance, None);
        // The pair that reads is kept and the two that do not are dropped.
        assert_eq!(vague.sizes, vec![(1024, 1024)]);
    }

    #[test]
    fn keys_this_version_does_not_know_are_stepped_over() {
        let suggested =
            Suggestions::from_fields(&fields("suggested:\n  sampler: euler_a\n  prompt: 1girl\n"));
        assert_eq!(suggested.prompt.as_deref(), Some("1girl"));
    }

    #[test]
    fn what_a_model_says_wins_and_what_it_does_not_say_is_left_alone() {
        let defaults = GenerationDefaults {
            width: 1024,
            height: 1024,
            num_steps: 30,
            guidance_scale: 5.0,
        };

        // A model with nothing to say gets exactly what this build believed.
        assert_eq!(Suggestions::default().over(defaults), defaults);

        // One that speaks only about the steps changes only the steps.
        let turbo = Suggestions {
            steps: Some(8),
            guidance: Some(1.0),
            ..Suggestions::default()
        };
        let applied = turbo.over(defaults);
        assert_eq!(applied.num_steps, 8);
        assert_eq!(applied.guidance_scale, 1.0);
        assert_eq!(applied.width, 1024);
        assert_eq!(applied.height, 1024);

        // The first size it lists is the one the boxes start at.
        let portrait = Suggestions {
            sizes: vec![(832, 1216), (1024, 1024)],
            ..Suggestions::default()
        };
        let applied = portrait.over(defaults);
        assert_eq!((applied.width, applied.height), (832, 1216));
        assert_eq!(applied.num_steps, 30);
    }
}
