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

//! What a screen should start a model's boxes at: the kind of model's own defaults, with what its
//! manifest's `suggested:` block says laid over them.
//!
//! None of it is needed to run a model -- a request says every number itself -- so it is the
//! model manager's to say, not the engine's. What the engine says of a model is what a request
//! may hold: `waifu_describe_json`.

use std::collections::BTreeMap;
use std::path::Path;

use serde_json::{json, Value};

use crate::suggested::Suggestions;
use crate::yaml::{self, Node};

/// The stand-in voice, which is libwaifu's own and has no manifest.
pub const TONES: &str = "tones";

/// The sizes offered for a model that does not name its own, which are the ones SDXL was trained
/// at. Every one of them is a multiple of the 32 pixels the U-Net's own halvings need.
pub const SIZES: &[(i32, i32)] = &[
    (512, 512),
    (640, 640),
    (768, 768),
    (896, 896),
    (1024, 1024),
    (832, 1216),
    (1216, 832),
    (768, 1344),
    (1344, 768),
];

/// How many sizes a model has to name before its list is used instead of the one above.
///
/// One size is not a choice and two is barely one. A model that names fewer has not replaced the
/// list, it has emptied it, and the built-in one is the better thing to offer.
pub const ENOUGH_SIZES: usize = 3;

/// The numbers a picture model's boxes start at.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Picture {
    pub width: i32,
    pub height: i32,
    pub steps: i32,
    pub guidance: f32,
}

/// What a kind of model starts at before its manifest says anything, and whether guidance is a
/// thing to offer it at all -- false for Krea 2, whose only release here is distilled and has no
/// second pass. Anything that is not one of the other kinds is SDXL, which is what the engine
/// builds it as too.
fn kind_of_picture(kind: &str) -> (Picture, bool) {
    let picture = |steps, guidance| Picture {
        width: 1024,
        height: 1024,
        steps,
        guidance,
    };
    match kind {
        "anima" => (picture(8, 1.0), true),
        "krea2" => (picture(8, 1.0), false),
        // Sampled without guidance by default, and able to take it: the reference's
        // `true_cfg_scale` is a second pass on a negative prompt.
        "qwen_image" => (picture(40, 1.0), true),
        _ => (picture(30, 5.0), true),
    }
}

/// Whether to offer guidance and a negative prompt, for a package suggesting `suggested`, of a kind
/// that offers them where `kind_does`.
///
/// What the package says, where it says it. Where it does not, a card that asks for a guidance of
/// one has said it all the same: at one there is no unconditional pass, so a dial above it is a
/// setting the card advises against and a negative prompt is text nothing reads. Anima Turbo is
/// the one published like that -- "at CFG 1 and 8-12 steps", and no `takes_guidance` key. Silence
/// with any other number leaves the kind's answer alone.
fn guided_by(suggested: &Suggestions, kind_does: bool) -> bool {
    suggested
        .takes_guidance
        .unwrap_or(kind_does && suggested.guidance != Some(1.0))
}

/// A float as a box can show it: four places, which is finer than any setting steps.
fn showable(value: f32) -> f64 {
    (f64::from(value) * 10_000.0).round() / 10_000.0
}

/// The top level of the manifest at `path`, where it is one a model could be built from: a
/// `weights` list with something in it and a `config` block. None for anything else, which is
/// described as what a model with nothing to say is.
fn manifest(path: &Path) -> Option<BTreeMap<String, Node>> {
    let text = std::fs::read_to_string(path).ok()?;
    let top = yaml::parse(&text).ok()?.into_map("the manifest").ok()?;
    let weights = top.get("weights")?.as_seq()?;
    if weights.is_empty() || !matches!(top.get("config"), Some(Node::Map(_))) {
        return None;
    }
    Some(top)
}

/// `config: model: type:`, the kind of model a manifest is.
fn kind(top: &BTreeMap<String, Node>) -> Option<String> {
    let Some(Node::Map(config)) = top.get("config") else {
        return None;
    };
    let Some(Node::Map(model)) = config.get("model") else {
        return None;
    };
    model.get("type")?.as_str().map(str::to_string)
}

/// The `suggested:` block, or nothing suggested.
fn suggestions(top: &BTreeMap<String, Node>) -> Suggestions {
    match top.get("suggested") {
        Some(Node::Map(fields)) => Suggestions::from_fields(fields),
        _ => Suggestions::default(),
    }
}

/// What the boxes start at for the model `asked` names -- the path of a manifest, or "tones" --
/// as `waifu_modelmanager_defaults_json` hands it out:
///
/// - a picture model: `width`, `height`, `steps`, `guidance`, `takes_guidance`, `sizes`, `prompt`
///   and `avoid`;
/// - a voice: `speed` and `temperature`, and a `conversion` holding `steps` for one that also
///   converts.
///
/// None for anything that is not a manifest on the disk.
pub fn defaults_json(asked: &str) -> Option<Value> {
    if asked == TONES {
        return Some(json!({ "speed": 1.0, "temperature": 0.8 }));
    }
    if !Path::new(asked).is_file() {
        return None;
    }

    let top = manifest(Path::new(asked));
    let kind = top.as_ref().and_then(kind).unwrap_or_default();
    match kind.as_str() {
        "indextts2" => return Some(json!({ "speed": 1.0, "temperature": 0.8 })),
        "cosyvoice3" => {
            return Some(json!({
                "speed": 1.0,
                "temperature": 1.0,
                "conversion": { "steps": 10 },
            }))
        }
        _ => {}
    }

    let suggested = top.as_ref().map(suggestions).unwrap_or_default();
    let (picture, kind_does) = kind_of_picture(&kind);
    let picture = suggested.over(picture);
    let sizes = match suggested.sizes.len() >= ENOUGH_SIZES {
        true => suggested.sizes.clone(),
        false => SIZES.to_vec(),
    };
    Some(json!({
        "width": picture.width,
        "height": picture.height,
        "steps": picture.steps,
        "guidance": showable(picture.guidance),
        "takes_guidance": guided_by(&suggested, kind_does),
        "sizes": sizes.iter().map(|(width, height)| json!([width, height])).collect::<Vec<_>>(),
        "prompt": suggested.prompt,
        "avoid": suggested.avoid,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_card_that_asks_for_a_guidance_of_one_takes_none() {
        let turbo = Suggestions {
            guidance: Some(1.0),
            ..Suggestions::default()
        };
        assert!(!guided_by(&turbo, true), "Anima Turbo's card says CFG 1 and nothing else");

        // What the package says outranks what it suggests: an undistilled model that likes one.
        let says_so = Suggestions {
            guidance: Some(1.0),
            takes_guidance: Some(true),
            ..Suggestions::default()
        };
        assert!(guided_by(&says_so, true));

        // Silence with another number, or with none, is the kind's answer.
        let anima = Suggestions {
            guidance: Some(4.0),
            ..Suggestions::default()
        };
        assert!(guided_by(&anima, true));
        assert!(guided_by(&Suggestions::default(), true));
        assert!(!guided_by(&Suggestions::default(), false));
    }

    /// A manifest of `kind` with `suggested` under it, in a file of its own.
    fn a_manifest(stem: &str, kind: &str, suggested: &str) -> String {
        let path = std::env::temp_dir().join(format!("waifu-defaults-{}-{stem}.yaml", std::process::id()));
        let text = format!("weights:\n  - w.safetensors\nconfig:\n  model:\n    type: {kind}\n{suggested}");
        std::fs::write(&path, text).unwrap();
        path.to_str().unwrap().to_string()
    }

    #[test]
    fn a_turbo_release_starts_at_its_own_numbers_and_offers_no_guidance() {
        let turbo = defaults_json(&a_manifest("turbo", "anima", "suggested:\n  steps: 8\n  guidance: 1\n")).unwrap();
        assert_eq!(turbo["steps"], 8);
        assert_eq!(turbo["guidance"], 1.0);
        assert_eq!(turbo["takes_guidance"], false);

        let sdxl = defaults_json(&a_manifest("sdxl", "sdxl", "")).unwrap();
        assert_eq!(sdxl["steps"], 30);
        assert_eq!(sdxl["guidance"], 5.0);
        assert_eq!(sdxl["takes_guidance"], true);
        assert_eq!(sdxl["sizes"].as_array().unwrap().len(), SIZES.len());

        let krea2 = defaults_json(&a_manifest("krea2", "krea2", "")).unwrap();
        assert_eq!(krea2["takes_guidance"], false);
    }

    #[test]
    fn a_voice_starts_at_its_kind_s_numbers() {
        assert_eq!(defaults_json(TONES).unwrap()["temperature"], 0.8);
        let cosy = defaults_json(&a_manifest("cosy", "cosyvoice3", "")).unwrap();
        assert_eq!(cosy["temperature"], 1.0);
        assert_eq!(cosy["conversion"]["steps"], 10);
        assert!(defaults_json("/no/such/model.yaml").is_none());
    }
}
