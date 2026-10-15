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

//! Fetching a model, or every model's manifest, as one job that reports how far along it is: what
//! the C API runs on a thread of its own, and what the engine's load starts with.

use std::path::{Path, PathBuf};

use crate::hub;

/// Which package of a model is being fetched, and how far into it.
#[derive(Clone, Debug, PartialEq)]
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

impl Fetch {
    /// A fetch of `model` that has not heard anything yet.
    pub fn of(model: impl Into<String>) -> Fetch {
        Fetch {
            model: model.into(),
            hub: None,
            file: String::new(),
            done: 0,
            total: None,
            part: 0,
            parts: 0,
        }
    }

    /// Takes in what the hub said.
    pub fn heard(&mut self, progress: hub::Progress) {
        match progress {
            hub::Progress::From { hub } => self.hub = Some(hub),
            hub::Progress::Fetching {
                file,
                done,
                total,
                part,
                parts,
            } => {
                self.file = file.to_string();
                (self.done, self.total, self.part, self.parts) = (done, total, part, parts);
            }
            hub::Progress::Fetched {
                file,
                bytes,
                part,
                parts,
            } => {
                self.file = file.to_string();
                (self.done, self.total, self.part, self.parts) = (bytes, Some(bytes), part, parts);
            }
        }
    }

    /// How far into the file, between nothing and all of it, or `None` where the server did not
    /// say how long it is -- which is also what the moment before the first byte looks like.
    pub fn fraction(&self) -> Option<f64> {
        self.total
            .filter(|total| *total > 0)
            .map(|total| self.done as f64 / total as f64)
    }

    /// What it is busy with, in the words a bar carries. They say which model, not only which
    /// file -- the file names in a package are not something anybody recognises -- and which hub,
    /// once that is settled.
    pub fn words(&self) -> String {
        if self.file.is_empty() {
            return format!("fetching {}", self.model);
        }
        let from = self.hub.map(|hub| format!(" from {hub}")).unwrap_or_default();
        match self.parts {
            0 => format!("fetching {}{from} -- {}", self.model, self.file),
            parts => format!(
                "fetching {}{from} -- {} ({} of {parts})",
                self.model, self.file, self.part
            ),
        }
    }
}

/// Why a fetch did not come back with what it was asked for.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Failed {
    /// Neither a published name nor a manifest that exists.
    UnknownModel(String),
    /// The `stop` it was given said to give up.
    Stopped,
    /// The download failed.
    Fetch(String),
}

impl Failed {
    fn from_hub(error: hub::Error) -> Failed {
        match hub::stopped(&error) {
            true => Failed::Stopped,
            false => Failed::Fetch(error.to_string()),
        }
    }
}

/// The name a screen shows for `asked`: the file name of a path, since the whole of one does not
/// fit, and the name itself otherwise.
pub fn shown_name(asked: &str) -> String {
    match asked.rsplit_once('/') {
        Some((_, file)) if !file.is_empty() => file.to_string(),
        _ => asked.to_string(),
    }
}

/// Fetches the published model `asked` names, if it is not here already, and hands back its
/// manifest; a manifest's own path is handed back as it is. On the caller's thread.
pub fn fetch(
    asked: &str,
    on_progress: &mut dyn FnMut(&Fetch),
    stop: &dyn Fn() -> bool,
) -> Result<PathBuf, Failed> {
    if hub::full_name(asked).is_none() && !Path::new(asked).is_file() {
        return Err(Failed::UnknownModel(format!(
            "there is no model called \"{asked}\": neither a published name nor a manifest that \
             exists"
        )));
    }

    let mut fetch = Fetch::of(shown_name(asked));
    on_progress(&fetch);
    hub::resolve_reporting(
        asked,
        &mut |progress| {
            fetch.heard(progress);
            on_progress(&fetch);
        },
        stop,
    )
    .map_err(Failed::from_hub)
}

/// Fetches the manifest of every published model not yet here, and none of their packages: see
/// [`hub::fetch_manifests`]. Each is reported as a part of the whole, under the model name
/// "model metadata".
pub fn fetch_manifests(
    on_progress: &mut dyn FnMut(&Fetch),
    stop: &dyn Fn() -> bool,
) -> Result<(), Failed> {
    let mut fetch = Fetch::of("model metadata");
    hub::fetch_manifests(
        &mut |progress| {
            fetch.heard(progress);
            on_progress(&fetch);
        },
        stop,
    )
    .map_err(Failed::from_hub)
}

/// The published models, aliases only, as JSON: what `waifu_modelmanager_catalog_json` hands out.
pub fn catalog_json() -> serde_json::Value {
    let entry = |listed: hub::Listed, kind: &str| {
        serde_json::json!({
            "name": listed.name,
            "full_name": listed.full_name,
            "kind": kind,
            "cached": listed.cached,
            // Whether its manifest is here, with or without its packages: what it suggests can
            // be read from it. See waifu_modelmanager_fetch_manifests_async.
            "manifest_here": hub::local_manifest(listed.name).is_some(),
            "bytes_on_disk": listed.bytes,
            // Where its packages are, once any of them is: the model's own directory, which
            // removing it deletes.
            "directory": hub::local_directory(listed.name).map(|directory| directory.to_string_lossy().into_owned()),
            "explicit": listed.explicit,
        })
    };
    let models: Vec<serde_json::Value> = hub::listed()
        .into_iter()
        .map(|listed| entry(listed, "image"))
        .chain(hub::listed_voices().into_iter().map(|listed| entry(listed, "speech")))
        .chain(hub::listed_conversions().into_iter().map(|listed| entry(listed, "conversion")))
        .collect();
    serde_json::json!({ "models": models })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_fetch_of_an_unknown_length_has_no_fraction() {
        // Which is what a server that did not say how long the file is leaves, and it is also
        // what the moment before the first byte looks like.
        let mut fetch = Fetch {
            file: "sdxl-base.waifupkg".to_string(),
            done: 512,
            part: 1,
            parts: 3,
            ..Fetch::of("sdxl:base")
        };
        assert_eq!(fetch.fraction(), None);

        fetch.total = Some(1024);
        assert_eq!(fetch.fraction(), Some(0.5));

        let words = fetch.words();
        assert!(words.contains("sdxl:base"), "{words}");
        assert!(words.contains("1 of 3"), "{words}");
    }

    #[test]
    fn a_name_that_is_neither_published_nor_a_file_is_refused_before_anything_is_fetched() {
        let mut heard = 0;
        let fetched = fetch("no-such:model", &mut |_| heard += 1, &|| false);
        assert!(matches!(fetched, Err(Failed::UnknownModel(_))), "{fetched:?}");
        assert_eq!(heard, 0);
    }

    #[test]
    fn what_is_published_comes_back_as_json() {
        let catalog = catalog_json();
        let models = catalog["models"].as_array().unwrap();
        assert!(models.iter().any(|model| model["name"] == "sdxl:base" && model["kind"] == "image"));
        assert!(models.iter().any(|model| model["kind"] == "speech"));
    }

    #[test]
    fn a_path_shows_as_its_file_name() {
        assert_eq!(shown_name("/models/sdxl-base.yaml"), "sdxl-base.yaml");
        assert_eq!(shown_name("sdxl:base"), "sdxl:base");
    }
}
