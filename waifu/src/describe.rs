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

//! What a model is, said without reading its weights: what a request for it may hold.
//!
//! What a screen should start the boxes at -- a kind's defaults, and what a manifest's
//! `suggested:` block says -- is not here: running a model needs none of it, and it is the model
//! manager's to say (`waifu_modelmanager_defaults_json`).

use std::path::{Path, PathBuf};

use crate::cosyvoice3::{self, CosyVoice3};
use crate::indextts::{self, IndexTts};
use crate::{Anima, Krea2, Manifest, QwenImage, Sdxl, Style, Tones, Voice};

/// The model the page is set to draw with, as the screen describes it.
///
/// Chosen is not loaded. Picking one is a click and reading one is minutes, so what is here is
/// what could be known when it was picked -- read from its manifest, whose weights may not even
/// be on the disk yet -- and it is described again out of the package the first time a run needs
/// the weights.
#[derive(Clone)]
pub struct Chosen {
    /// What to ask for when the time comes to read it: the path of its manifest. Not for
    /// showing -- a path does not fit on the screen -- but it is what a load is asked with.
    pub name: String,
    /// What it is called on screen: the manifest's file name.
    pub full_name: String,
    /// Whether every file its manifest names is beside it. A model that is not is still a model that
    /// can be chosen; the download happens at the first run, where the bar can say so.
    pub on_disk: bool,
    /// Whether its weights are read and on the device. False until the first run needs them.
    pub in_memory: bool,
    /// What both sides have to be a multiple of, which is what the kind's halvings need: a size
    /// typed in is rounded to this, and one that is not is refused by the run.
    pub alignment: i32,
    /// What walks the noise back, named. One kind of model has one of these here, so it is
    /// something the screen reports rather than something it offers -- but a picture that came
    /// out unlike another tool's is a question about this first, and a screen that does not say
    /// is a screen that cannot be asked.
    pub sampler: &'static str,
    /// Why this one cannot be handed a picture to start from, or None where it can. A page that
    /// offers what the model will refuse is a refusal someone finds out about after waiting, and
    /// one that is simply greyed out is a screen that cannot be asked why.
    pub no_picture_because: Option<String>,
}

/// The voice the page is set to speak with, as the screen describes it.
///
/// What [`Chosen`] is for a picture model, and for the same reasons: it is what could be known
/// before any weights were read, and it is described again out of the voice itself once one is
/// loaded.
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
    /// The rate it writes at.
    pub rate: u32,
    /// Why it cannot convert the style as well, or None where it can -- which is where the page
    /// offers the box.
    pub no_style_because: Option<String>,
}

/// Why Anima cannot be handed a picture to start from. True of the kind rather than of any one
/// package, which is what lets it be said before a package has been looked at.
pub const ANIMA_DRAWS_FROM_NO_PICTURE: &str = "Anima cannot start from a picture yet: its package \
     carries an image encoder, but the layer that reads one is not written";

/// And why Krea 2 cannot either, which is the same reason: it draws through the same Qwen-Image
/// autoencoder, and the half of it that reads a picture has no layer to run it.
pub const KREA2_DRAWS_FROM_NO_PICTURE: &str =
    "Krea 2 cannot start from a picture yet: its package \
     carries an image encoder, but the layer that reads one is not written";

/// And why Qwen-Image 2.1 cannot, for a different reason: its model edits pictures by reading
/// them as well as starting from them, and neither half of that is written here.
pub const QWEN_IMAGE_DRAWS_FROM_NO_PICTURE: &str =
    "Qwen-Image 2.1 cannot start from a picture yet: \
     it edits by reading the picture through its text encoder too, and that is not written";

/// What the built-in stand-in voice is called where a name is asked for.
///
/// A constant rather than a catalogue entry, because it is not published anywhere and cannot be
/// fetched: it is in the binary. Only ever had by asking for it with `-m tones` -- it makes a
/// noise where the syllables are, which is for checking the page and not for listening to.
pub const TONES: &str = "tones";

/// The voice `asked` names, as the screen describes it, without reading any of it.
///
/// What [`look_at`] is for a picture model. [`TONES`] is [`Tones`], built into the binary, which
/// can be constructed and asked about itself for nothing. Anything else is the path of a manifest,
/// and what can be said of one before minutes of reading is what its kind says: IndexTTS-2.5's
/// rate, whether its files are on the disk, and that it is not in memory.
pub fn look_at_voice(asked: &str) -> Spoken {
    if asked == TONES {
        return describe_voice(asked, &Tones::new(), false);
    }

    let kind = manifest_of(asked).as_deref().and_then(voice_kind);
    let (kind_name, rate, styles) = match kind {
        Some(VoiceKind::CosyVoice3) => (CosyVoice3::NAME, cosyvoice3::RATE, CosyVoice3::STYLES),
        _ => (IndexTts::NAME, indextts::RATE, &[][..]),
    };

    Spoken {
        name: asked.to_string(),
        // The kind's name: what the catalogue calls a published one is the model manager's.
        full_name: kind_name.to_string(),
        on_disk: on_disk(asked).is_some(),
        in_memory: false,
        rate,
        no_likeness_because: None,
        styles,
        not_a_voice_because: None,
    }
}

/// Whether `asked` is a voice: the stand-in, or a manifest on the disk whose `model.type` is one
/// of the speech models.
pub fn is_a_voice(asked: &str) -> bool {
    asked == TONES || manifest_of(asked).as_deref().and_then(voice_kind).is_some()
}

/// The speech models a package can be.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum VoiceKind {
    IndexTts,
    CosyVoice3,
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
        full_name: voice.name().to_string(),
        // Built in, or just read off the disk: either way there is nothing left to fetch.
        on_disk: true,
        in_memory,
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
        full_name: CosyVoice3::NAME.to_string(),
        on_disk: on_disk(asked).is_some(),
        in_memory: false,
        rate: cosyvoice3::RATE,
        no_style_because: Some(CosyVoice3::NO_STYLE_CONVERSION.to_string()),
    }
}

/// Whether `asked` turns one recording into another: a manifest on the disk whose `model.type` is
/// one. CosyVoice3 is both this and a voice.
pub fn is_a_converter(asked: &str) -> bool {
    manifest_of(asked)
        .as_deref()
        .and_then(converter_kind)
        .is_some()
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

/// What a kind of model implies before any of its weights are read: what walks the noise back,
/// and why it cannot start from a picture. One table for the kinds, read from the `type` a
/// manifest states.
pub fn about(kind: &str) -> (&'static str, Option<&'static str>) {
    match kind {
        Anima::MODEL_TYPE => ("Flow match Euler", Some(ANIMA_DRAWS_FROM_NO_PICTURE)),
        Krea2::MODEL_TYPE => ("Flow match Euler", Some(KREA2_DRAWS_FROM_NO_PICTURE)),
        QwenImage::MODEL_TYPE => ("Flow match Euler", Some(QWEN_IMAGE_DRAWS_FROM_NO_PICTURE)),
        // Everything else is SDXL, which is what `Model::from_manifest` decides too.
        _ => ("Euler", None),
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

/// What can be said about a model that has not been read, which is what a choice is made from:
/// what the manifest at `asked` says, and none of its weights. Not the last word -- the moment a
/// run reads the package, the engine describes it again out of what is actually in there.
///
/// The manifest need not have its packages beside it: one of the bundled ones, or one fetched on
/// its own, says what the model is before gigabytes of it are downloaded. A manifest that cannot
/// be read describes an SDXL model, because the one thing this decides -- whether image to image
/// is offered -- is a thing SDXL packages do.
pub fn look_at(asked: &str) -> Chosen {
    let (sampler, no_picture) = about("");

    let mut chosen = Chosen {
        name: asked.to_string(),
        // The file name: the whole path does not fit on screen and its tail is the part that
        // identifies it. What the catalogue calls a published one is the model manager's.
        full_name: shown_name(asked),
        on_disk: on_disk(asked).is_some(),
        in_memory: false,
        alignment: alignment(""),
        sampler,
        no_picture_because: no_picture.map(str::to_string),
    };

    // What the manifest says about itself. It is a few kilobytes of text beside gigabytes of
    // weights, and the weights are not touched: whether this one can start from a picture is what
    // its kind says until a run opens the package.
    let Some(path) = manifest_of(asked) else {
        return chosen;
    };
    let Ok(manifest) = Manifest::open(path) else {
        return chosen;
    };

    // What kind it is, which the manifest states.
    if let Ok(kind) = manifest
        .section(Sdxl::MODEL_SECTION)
        .and_then(|section| section.get_str("type").map(str::to_string))
    {
        let (sampler, no_picture) = about(&kind);
        chosen.sampler = sampler;
        chosen.no_picture_because = no_picture.map(str::to_string);
        chosen.alignment = alignment(&kind);
    }

    chosen
}

/// The manifest at `asked`, where there is one.
pub fn manifest_of(asked: &str) -> Option<PathBuf> {
    let path = PathBuf::from(asked);
    path.is_file().then_some(path)
}

/// The manifest at `asked`, where every file it names is beside it: a model that can be read
/// without fetching anything.
pub fn on_disk(asked: &str) -> Option<PathBuf> {
    bytes_on_disk(asked).map(|_| PathBuf::from(asked))
}

/// How many bytes the files the manifest at `asked` names are, where all of them are here: what
/// choosing a device it fits on weighs.
pub fn bytes_on_disk(asked: &str) -> Option<u64> {
    let manifest = Manifest::open(manifest_of(asked)?).ok()?;
    let mut bytes = 0;
    for name in manifest.files() {
        bytes += std::fs::metadata(manifest.file(&name).ok()?).ok()?.len();
    }
    Some(bytes)
}

/// The name a screen shows for `asked`: the file name of a path, since the whole of one does not
/// fit, and the name itself otherwise.
pub fn shown_name(asked: &str) -> String {
    match Path::new(asked).file_name().and_then(|name| name.to_str()) {
        Some(file) if !file.is_empty() => file.to_string(),
        _ => asked.to_string(),
    }
}

/// Why `asked` is not a model this can read.
pub fn no_manifest_at(asked: &str) -> String {
    format!(
        "there is no manifest at \"{asked}\": a model is named by the path of its manifest, which \
         waifu_modelmanager_manifest_path hands back for a published one"
    )
}
