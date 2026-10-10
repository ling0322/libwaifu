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

//! What a model is, said without reading its weights: what a window offers before a run.

use std::path::{Path, PathBuf};

use crate::cosyvoice3::{self, CosyVoice3};
use crate::hub;
use crate::indextts::{self, IndexTts};
use crate::{
    Anima, ConversionDefaults, GenerationDefaults, Krea2, Manifest, QwenImage, Sdxl,
    SpeechDefaults, Style, Suggestions, Tones, Voice,
};

/// The model the page is set to draw with, as the screen describes it.
///
/// Chosen is not loaded. Picking one is a click and reading one is minutes, so what is here is
/// what could be known when it was picked -- guessed from the name, for a model that may not
/// even be on the disk yet -- and it is described again out of the package the first time a run
/// needs the weights. Everything on the screen that is about the model reads this: the boxes it
/// fills in, and whether image to image is offered at all.
#[derive(Clone)]
pub struct Chosen {
    /// What to ask for when the time comes to read it: the catalogue name, or the path that was
    /// given. Not for showing -- a path does not fit on the screen -- but it is what a run is
    /// posted with, and what the worker hands to the hub.
    pub name: String,
    /// What it is called on screen: the catalogue's full name, or the file name for a path.
    pub full_name: String,
    /// Whether every package of it is already fetched. A model that is not is still a model that
    /// can be chosen; the download happens at the first run, where the bar can say so.
    pub on_disk: bool,
    /// Whether its weights are read and on the device. False until the first run needs them.
    pub in_memory: bool,
    /// What the boxes start at. Guessed from the name until the package is read, and read out of
    /// the package after that -- a distilled release answers both of these differently.
    pub defaults: GenerationDefaults,
    /// The sizes to offer: the model's own where its manifest names enough of them.
    pub sizes: Vec<(i32, i32)>,
    /// What both sides have to be a multiple of, which is what the kind's halvings need: a size
    /// typed in is rounded to this, and one that is not is refused by the run.
    pub alignment: i32,
    /// What walks the noise back, named. One kind of model has one of these here, so it is
    /// something the screen reports rather than something it offers -- but a picture that came
    /// out unlike another tool's is a question about this first, and a screen that does not say
    /// is a screen that cannot be asked.
    pub sampler: &'static str,
    /// What the package's own card suggests be typed in the boxes. Known once it has been read,
    /// which is why the boxes take it whenever it arrives rather than only when a page opens.
    pub suggested_prompt: Option<String>,
    pub suggested_avoid: Option<String>,
    /// Why this one cannot be handed a picture to start from, or None where it can. A page that
    /// offers what the model will refuse is a refusal someone finds out about after waiting, and
    /// one that is simply greyed out is a screen that cannot be asked why.
    pub no_picture_because: Option<String>,
    /// Whether to offer guidance and a negative prompt at all. False for a distilled release,
    /// which has no second pass for either to reach. Guessed from the name like the rest of this
    /// and answered by the package once it has been read.
    pub takes_guidance: bool,
}

/// The voice the page is set to speak with, as the screen describes it.
///
/// What [`Chosen`] is for a picture model, and for the same reasons: it is what could be known
/// before any weights were read, and it is described again out of the voice itself once one is
/// loaded. There is one of these even before anything has been asked for, because the boxes on
/// the speech tab are a voice's numbers and have to start somewhere.
#[derive(Clone)]
pub struct Spoken {
    /// What to ask for when the time comes to read it. It travels with a run the way a model's
    /// name does, so that a run is of the voice that was chosen when the button was pressed.
    pub name: String,
    /// What it is called on screen.
    pub full_name: String,
    /// Whether the package is already here, which is the difference between a first reading that
    /// takes seconds and one that starts with a download of several gigabytes.
    pub on_disk: bool,
    /// Whether it is read and on the device. The stand-in, `tones`, holds nothing, so for it this
    /// is true from the first run rather than after a fetch.
    pub in_memory: bool,
    /// What the boxes start at, which is the voice's to say.
    pub defaults: SpeechDefaults,
    /// The rate it writes at, which is a property of the voice and not a setting. On the screen
    /// because a clip that came out at the wrong pitch is a question about this first.
    pub rate: u32,
    /// Why it cannot be handed a recording to sound like, or None where it can.
    pub no_likeness_because: Option<String>,
    /// The ways of saying something it was taught, for the page to list. Empty where it takes no
    /// style, which is where the page offers none.
    pub styles: &'static [Style],
    /// Why what comes out is not speech, for as long as that is true.
    ///
    /// The one sentence this whole tab is built around saying. It comes out of the voice itself
    /// rather than being written on the page, so the day a real model is what is loaded the
    /// warning goes away on its own rather than by somebody remembering to delete it.
    pub not_a_voice_because: Option<String>,
}

/// The converter the page is set to convert with, as the screen describes it: what [`Spoken`] is
/// for a voice.
#[derive(Clone)]
pub struct ChosenConverter {
    /// What to ask for when the time comes to read it, as a voice's name is.
    pub name: String,
    pub full_name: String,
    pub on_disk: bool,
    pub in_memory: bool,
    /// What the boxes start at.
    pub defaults: ConversionDefaults,
    /// The rate it writes at.
    pub rate: u32,
    /// Why it cannot convert the style as well, or None where it can -- which is where the page
    /// offers the box.
    pub no_style_because: Option<String>,
}

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

/// Why Anima cannot be handed a picture to start from. True of the kind rather than of any one
/// package, which is what lets it be said before a package has been looked at.
pub const ANIMA_DRAWS_FROM_NO_PICTURE: &str = "Anima cannot start from a picture yet: its package \
     carries an image encoder, but the layer that reads one is not written";

/// And why Krea 2 cannot either, which is the same reason: it draws through the same Qwen-Image
/// autoencoder, and the half of it that reads a picture has no layer to run it.
pub const KREA2_DRAWS_FROM_NO_PICTURE: &str = "Krea 2 cannot start from a picture yet: its package \
     carries an image encoder, but the layer that reads one is not written";

/// And why Qwen-Image 2.1 cannot, for a different reason: its model edits pictures by reading
/// them as well as starting from them, and neither half of that is written here.
pub const QWEN_IMAGE_DRAWS_FROM_NO_PICTURE: &str = "Qwen-Image 2.1 cannot start from a picture yet: \
     it edits by reading the picture through its text encoder too, and that is not written";

/// What the built-in stand-in voice is called where a name is asked for.
///
/// A constant rather than a catalogue entry, because it is not published anywhere and cannot be
/// fetched: it is in the binary. Only ever had by asking for it with `-m tones` -- it makes a
/// noise where the syllables are, which is for checking the page and not for listening to.
pub const TONES: &str = "tones";

/// How many sizes a model has to name before its list is used instead of the one above.
///
/// One size is not a choice and two is barely one. A model that names fewer has not replaced the
/// list, it has emptied it, and the built-in one is the better thing to offer.
pub const ENOUGH_SIZES: usize = 3;

/// The voice `asked` names, as the screen describes it, without reading any of it.
///
/// What [`look_at`] is for a picture model. [`TONES`] is [`Tones`], built into the binary, which
/// can be constructed and asked about itself for nothing. Anything else is a package -- a manifest
/// on the disk, or a published name -- and what can be said of one before minutes of reading is
/// what its kind says: IndexTTS-2.5's rate and starting values, whether it is on the disk, and
/// that it is not in memory. The read itself is [`read_voice`], at the first run that wants it,
/// and it says so if the package turns out to be something else.
pub fn look_at_voice(asked: &str) -> Spoken {
    if asked == TONES {
        return describe_voice(asked, &Tones::new(), false);
    }

    let path = on_disk(asked);
    // The manifest says what it is where there is one; a published voice not yet fetched is
    // known by its family, so the page offers its own settings before the download.
    let kind = path
        .as_deref()
        .and_then(voice_kind)
        .or_else(|| published_voice_kind(asked));
    let (kind_name, defaults, rate, styles) = match kind {
        Some(VoiceKind::CosyVoice3) => (
            CosyVoice3::NAME,
            CosyVoice3::DEFAULTS,
            cosyvoice3::RATE,
            CosyVoice3::STYLES,
        ),
        _ => (IndexTts::NAME, IndexTts::DEFAULTS, indextts::RATE, &[][..]),
    };

    Spoken {
        name: asked.to_string(),
        // The catalogue's name for it where it has one, and the kind's where it was named by path.
        full_name: hub::full_name(asked).unwrap_or(kind_name).to_string(),
        on_disk: path.is_some(),
        in_memory: false,
        defaults,
        rate,
        no_likeness_because: None,
        styles,
        not_a_voice_because: None,
    }
}

/// Whether `asked` is a voice: the stand-in, a published voice, or a manifest on the disk whose
/// `model.type` is one of the speech models. What `-m` alone is taken to mean text2speech by.
pub fn is_a_voice(asked: &str) -> bool {
    asked == TONES
        || hub::is_voice(asked)
        || on_disk(asked).as_deref().and_then(voice_kind).is_some()
}

/// The speech models a package can be.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum VoiceKind {
    IndexTts,
    CosyVoice3,
}

/// Which speech model a published voice is, by the family its versioned name starts with.
pub fn published_voice_kind(asked: &str) -> Option<VoiceKind> {
    match hub::published_name(asked)?.split(':').next()? {
        "indextts" => Some(VoiceKind::IndexTts),
        "cosyvoice" => Some(VoiceKind::CosyVoice3),
        _ => None,
    }
}

/// Which speech model the manifest at `path` describes, by its `model.type`; `None` where it
/// cannot be read or is not a speech model this reads.
pub fn voice_kind(path: &std::path::Path) -> Option<VoiceKind> {
    let manifest = Manifest::open(path).ok()?;
    match manifest.section("model").ok()?.get_str("type").ok()? {
        IndexTts::MODEL_TYPE => Some(VoiceKind::IndexTts),
        CosyVoice3::MODEL_TYPE => Some(VoiceKind::CosyVoice3),
        _ => None,
    }
}

/// What a voice says about itself, in the words the screen shows, under the name it was asked for.
pub fn describe_voice(name: &str, voice: &dyn Voice, in_memory: bool) -> Spoken {
    Spoken {
        name: name.to_string(),
        // The catalogue's name first, so the button does not change its wording when it is read.
        full_name: hub::full_name(name).unwrap_or(voice.name()).to_string(),
        // Built in, or just read off the disk: either way there is nothing left to fetch.
        on_disk: true,
        in_memory,
        defaults: voice.defaults(),
        rate: voice.rate(),
        no_likeness_because: voice.no_likeness_because().map(str::to_string),
        styles: voice.styles(),
        not_a_voice_because: voice.not_a_voice_because().map(str::to_string),
    }
}

/// The converter `asked` names, as the screen describes it, without reading any of it: what
/// [`look_at_voice`] is for a voice.
pub fn look_at_converter(asked: &str) -> ChosenConverter {
    // CosyVoice3 is the one converter there is, so there is no kind to tell apart yet.
    ChosenConverter {
        name: asked.to_string(),
        full_name: hub::full_name(asked).unwrap_or(CosyVoice3::NAME).to_string(),
        on_disk: on_disk(asked).is_some(),
        in_memory: false,
        defaults: CosyVoice3::CONVERSION_DEFAULTS,
        rate: cosyvoice3::RATE,
        no_style_because: Some(CosyVoice3::NO_STYLE_CONVERSION.to_string()),
    }
}

/// Whether `asked` turns one recording into another: a published converter, or a manifest on the
/// disk whose `model.type` is one. What `-m` alone is taken to mean speech2speech by, after
/// [`is_a_voice`] -- CosyVoice3 is both, and alone it reads.
pub fn is_a_converter(asked: &str) -> bool {
    hub::is_conversion(asked) || on_disk(asked).as_deref().and_then(converter_kind).is_some()
}

/// The models a package can convert voices with.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ConverterKind {
    CosyVoice3,
}

/// Which converter the manifest at `path` describes, by its `model.type`.
pub fn converter_kind(path: &Path) -> Option<ConverterKind> {
    let manifest = Manifest::open(path).ok()?;
    match manifest.section("model").ok()?.get_str("type").ok()? {
        CosyVoice3::MODEL_TYPE => Some(ConverterKind::CosyVoice3),
        _ => None,
    }
}

/// What a kind of model implies for the screen before any of its weights are read: what to ask
/// it for, what walks the noise back, why it cannot start from a picture, and whether guidance is
/// a thing to offer at all.
///
/// One table for the three kinds, read twice -- once from a name that has not been fetched yet,
/// and once from the `type` of a manifest that is already on the disk.
///
/// The last of the four is false for Krea 2 for the same reason its defaults are eight steps at a
/// guidance of one: the only release of it this exports is the distilled one, and the distilled
/// one has no second pass. A package's own `suggested:` overrides this, so the day there is an
/// undistilled one it says `takes_guidance: "true"` and gets the dial back.
pub fn about(kind: &str) -> (GenerationDefaults, &'static str, Option<&'static str>, bool) {
    match kind {
        Anima::MODEL_TYPE => (
            Anima::DEFAULTS,
            "Flow match Euler",
            Some(ANIMA_DRAWS_FROM_NO_PICTURE),
            true,
        ),
        Krea2::MODEL_TYPE => (
            Krea2::DEFAULTS,
            "Flow match Euler",
            Some(KREA2_DRAWS_FROM_NO_PICTURE),
            false,
        ),
        // Sampled without guidance by default, and able to take it: the reference's
        // `true_cfg_scale` is a second pass on a negative prompt, which is this runtime's.
        QwenImage::MODEL_TYPE => (
            QwenImage::DEFAULTS,
            "Flow match Euler",
            Some(QWEN_IMAGE_DRAWS_FROM_NO_PICTURE),
            true,
        ),
        // Everything else is SDXL, which is what `Model::from_manifest` decides too.
        _ => (GenerationDefaults::default(), "Euler", None, true),
    }
}

/// What both sides of a picture of `kind` have to be a multiple of: the VAE's eight times the
/// patch for the transformers, and times the U-Net's two halvings for SDXL. The pipelines check
/// the same out of the package's config; this is the number for the screen, before one is read.
pub fn alignment(kind: &str) -> i32 {
    match kind {
        Anima::MODEL_TYPE | Krea2::MODEL_TYPE | QwenImage::MODEL_TYPE => 16,
        _ => 32,
    }
}

/// What can be said about a model that has not been read, which is what a choice is made from.
///
/// Guessed from the name where there is nothing else to go on, and read out of the manifest where
/// the package is already here. Neither is the last word -- the moment a run reads the package,
/// [`load`] describes it again out of what is actually in there, wrong guesses included.
pub fn look_at(asked: &str) -> Chosen {
    // The kinds are told apart by the catalogue's own naming. A package somebody exported
    // themselves and named by path is taken as SDXL until the manifest below says otherwise,
    // because the one thing this decides -- whether image to image is offered -- is a thing SDXL
    // packages do.
    let guessed = match asked {
        name if name.starts_with("anima") => Anima::MODEL_TYPE,
        name if name.starts_with("krea") => Krea2::MODEL_TYPE,
        name if name.starts_with("qwen") => QwenImage::MODEL_TYPE,
        _ => "",
    };
    let (defaults, sampler, no_picture, takes_guidance) = about(guessed);

    let mut chosen = Chosen {
        name: asked.to_string(),
        // The catalogue's name for it, or the file name where a path was given: the whole path
        // does not fit on screen and its tail is the part that identifies it.
        full_name: hub::full_name(asked)
            .map(str::to_string)
            .unwrap_or_else(|| match asked.rsplit_once('/') {
                Some((_, file)) if !file.is_empty() => file.to_string(),
                _ => asked.to_string(),
            }),
        on_disk: hub::is_cached(asked),
        in_memory: false,
        defaults,
        sizes: SIZES.to_vec(),
        alignment: alignment(guessed),
        sampler,
        // Not known until the package is read: what a card suggests be typed is written in it.
        suggested_prompt: None,
        suggested_avoid: None,
        no_picture_because: no_picture.map(str::to_string),
        takes_guidance,
    };

    // And where the package is already here, what it says about itself. The manifest is a few
    // kilobytes of text beside gigabytes of weights, and reading it is what keeps the numbers in
    // the boxes from changing under somebody later: a model described twice, once from its name
    // and once from its package, is a model whose settings move while they are being used.
    //
    // The weights are still not touched. Whether this one can start from a picture is the guess
    // above until a run opens the package.
    let Some(path) = on_disk(asked) else {
        return chosen;
    };
    let Ok(manifest) = Manifest::open(path) else {
        return chosen;
    };

    // What kind it is, which the manifest states and a name only hints at. It matters most for a
    // package named by its path, which is every package somebody exported themselves: `krea2` is
    // not in the catalogue at all, so this is the only place its kind is ever read.
    if let Ok(kind) = manifest
        .section(Sdxl::MODEL_SECTION)
        .and_then(|section| section.get_str("type").map(str::to_string))
    {
        let (defaults, sampler, no_picture, takes_guidance) = about(&kind);
        chosen.defaults = defaults;
        chosen.sampler = sampler;
        chosen.no_picture_because = no_picture.map(str::to_string);
        chosen.takes_guidance = takes_guidance;
        chosen.alignment = alignment(&kind);
    }

    let suggested = manifest.suggested();
    chosen.defaults = suggested.over(chosen.defaults);
    chosen.suggested_prompt = suggested.prompt.clone();
    chosen.suggested_avoid = suggested.avoid.clone();
    chosen.takes_guidance = guided_by(suggested, chosen.takes_guidance);
    if suggested.sizes.len() >= ENOUGH_SIZES {
        chosen.sizes = suggested.sizes.clone();
    }

    chosen
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

/// The manifest of a model that can be read without fetching anything: one already in the cache,
/// or one that was named by the path of its manifest in the first place.
pub fn on_disk(asked: &str) -> Option<PathBuf> {
    hub::cached_manifest(asked).or_else(|| {
        let path = PathBuf::from(asked);
        path.is_file().then_some(path)
    })
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
}
