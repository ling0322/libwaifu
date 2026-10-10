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

//! The models that draw, behind one type: which one a manifest is, read and run.

use std::ops::ControlFlow;
use std::path::Path;

use crate::describe::{
    ANIMA_DRAWS_FROM_NO_PICTURE, KREA2_DRAWS_FROM_NO_PICTURE, QWEN_IMAGE_DRAWS_FROM_NO_PICTURE,
};
use crate::flint::Tensor;
use crate::{
    Anima, Device, GenerationDefaults, GenerationOptions, GenerationProgress, Krea2, Manifest,
    QwenImage, Residency, Sdxl,
};

/// One of the models that draw, read and on its device.
pub enum ImageModel {
    Sdxl(Sdxl),
    Anima(Anima),
    Krea2(Krea2),
    QwenImage(QwenImage),
}

impl ImageModel {
    /// What kind of model a manifest describes, without reading the model.
    ///
    /// One line of a file that is a couple of kilobytes. The alternative is to try each kind in
    /// turn and keep the one that does not complain, which reads gigabytes to answer a question
    /// the manifest states.
    ///
    /// Either model's `MODEL_SECTION` would do -- the block that says what a model is, is the
    /// same block whatever it turns out to be, which is the point of it.
    fn kind(manifest: &Manifest) -> crate::Result<String> {
        Ok(manifest
            .section(Sdxl::MODEL_SECTION)?
            .get_str("type")?
            .to_string())
    }

    pub fn from_manifest(
        model_path: &Path,
        device: Device,
        residency: Residency,
    ) -> crate::Result<ImageModel> {
        let manifest = Manifest::open(model_path)?;

        match Self::kind(&manifest)?.as_str() {
            Anima::MODEL_TYPE => Ok(ImageModel::Anima(Anima::from_manifest(
                device,
                residency,
                &manifest,
            )?)),
            Krea2::MODEL_TYPE => Ok(ImageModel::Krea2(Krea2::from_manifest(
                device,
                residency,
                &manifest,
            )?)),
            QwenImage::MODEL_TYPE => Ok(ImageModel::QwenImage(QwenImage::from_manifest(
                device,
                residency,
                &manifest,
            )?)),
            // Everything else goes to SDXL, which says what it makes of it. A model of a kind
            // nobody here has heard of is its complaint to make rather than this one's, since it
            // is the one that knows what it can read.
            _ => Ok(ImageModel::Sdxl(Sdxl::from_manifest(
                device,
                residency,
                &manifest,
            )?)),
        }
    }

    /// What this build believes about the kind of model, which is what the manifest's own
    /// suggestions are laid over. A distilled release wants eight steps at no guidance and comes
    /// out burnt at the thirty and five SDXL likes.
    pub fn defaults(&self) -> GenerationDefaults {
        match self {
            ImageModel::Sdxl(_) => GenerationDefaults::default(),
            ImageModel::Anima(_) => Anima::DEFAULTS,
            ImageModel::Krea2(_) => Krea2::DEFAULTS,
            ImageModel::QwenImage(_) => QwenImage::DEFAULTS,
        }
    }

    /// What walks the noise back, as the model's own reference implementation names it.
    pub fn sampler(&self) -> &'static str {
        match self {
            ImageModel::Sdxl(_) => "Euler",
            ImageModel::Anima(_) | ImageModel::Krea2(_) | ImageModel::QwenImage(_) => "Flow match Euler",
        }
    }

    /// Why it cannot be handed a picture to start from, or None where it can.
    ///
    /// A sentence rather than a bool, because both answers of no are somebody's next move: one
    /// is fetched away by taking the model down and bringing it back, and the other is not.
    pub fn no_picture_because(&self) -> Option<&'static str> {
        match self {
            // Not a property of SDXL but of the copy on this disk: the encoder's weights are in
            // packages exported since image to image landed, and in none written before it.
            ImageModel::Sdxl(model) => (!model.draws_from_a_picture()).then_some(
                "this copy was packaged before image to image existed, so it carries no VAE \
                 encoder. Deleting it below and fetching it again brings one down",
            ),
            // The weights for one are in the package; the layer that reads them is not written.
            // See docs/anima.md and docs/krea2.md -- it is the same autoencoder and the same
            // missing half.
            ImageModel::Anima(_) => Some(ANIMA_DRAWS_FROM_NO_PICTURE),
            ImageModel::Krea2(_) => Some(KREA2_DRAWS_FROM_NO_PICTURE),
            ImageModel::QwenImage(_) => Some(QWEN_IMAGE_DRAWS_FROM_NO_PICTURE),
        }
    }

    pub fn generate_reporting(
        &self,
        prompt: &str,
        options: &GenerationOptions,
        report: &mut dyn FnMut(GenerationProgress) -> ControlFlow<()>,
    ) -> crate::Result<Option<Tensor>> {
        match self {
            ImageModel::Sdxl(model) => model.generate_reporting(prompt, options, report),
            ImageModel::Anima(model) => model.generate_reporting(prompt, options, report),
            ImageModel::Krea2(model) => model.generate_reporting(prompt, options, report),
            ImageModel::QwenImage(model) => model.generate_reporting(prompt, options, report),
        }
    }

    pub fn generate_from_image_reporting(
        &self,
        image: &Tensor,
        prompt: &str,
        options: &GenerationOptions,
        report: &mut dyn FnMut(GenerationProgress) -> ControlFlow<()>,
    ) -> crate::Result<Option<Tensor>> {
        match self {
            ImageModel::Sdxl(model) => {
                model.generate_from_image_reporting(image, prompt, options, report)
            }
            // The same sentence the door turns a run away with, for the caller that is not the
            // door: one wording for one refusal.
            ImageModel::Anima(_) | ImageModel::Krea2(_) | ImageModel::QwenImage(_) => Err(crate::Error::model(
                self.no_picture_because()
                    .unwrap_or("this model cannot start from a picture"),
            )),
        }
    }
}
