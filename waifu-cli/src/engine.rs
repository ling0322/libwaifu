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

//! libwaifu, as the drawing screen asks it: what a model is, and an image generation pipeline on
//! a device that reads it and draws with it.
//!
//! The types are the screen's own -- the boxes it fills in, the progress it draws a bar from --
//! and what they are filled from is what libwaifu says through its C API.

use std::ffi::c_void;
use std::path::Path;
use std::ptr;
use std::sync::mpsc::{channel, Sender};

use crate::c::{
    self, WaifuImageGenerationRequest, WaifuImageGenerationResult, WaifuJob, WaifuModel,
    WaifuModelProperty, WaifuPipeline, WaifuProgress, WaifuResult, WaifuStatusCode,
};
use crate::runtime::Runtime;

/// The sizes offered for a model whose manifest does not name its own: the ones SDXL was trained
/// at, which is also what libwaifu offers for one.
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

/// What a run is asked for.
#[derive(Clone, Debug, PartialEq)]
pub struct GenerationOptions {
    pub width: i32,
    pub height: i32,
    pub num_steps: i32,
    pub guidance_scale: f32,
    pub negative_prompt: String,
    /// None picks one; the screen always names one, so that the picture can say which.
    pub seed: Option<u64>,
}

impl Default for GenerationOptions {
    fn default() -> GenerationOptions {
        GenerationOptions {
            width: 1024,
            height: 1024,
            num_steps: 30,
            guidance_scale: 5.0,
            negative_prompt: String::new(),
            seed: None,
        }
    }
}

/// What the boxes start at for a model.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GenerationDefaults {
    pub width: i32,
    pub height: i32,
    pub num_steps: i32,
    pub guidance_scale: f32,
}

impl Default for GenerationDefaults {
    fn default() -> GenerationDefaults {
        let options = GenerationOptions::default();
        GenerationDefaults {
            width: options.width,
            height: options.height,
            num_steps: options.num_steps,
            guidance_scale: options.guidance_scale,
        }
    }
}

/// How far along a run is.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GenerationProgress {
    Encoding,
    Step { done: i32, total: i32 },
    Decoding,
}

/// The model the screen is set to draw with, as libwaifu describes it from its manifest before
/// any of its weights are read.
#[derive(Clone, Debug, PartialEq)]
pub struct Chosen {
    /// Whether it draws: a voice does not, and the terminal draws pictures.
    pub draws: bool,
    pub defaults: GenerationDefaults,
    pub sizes: Vec<(i32, i32)>,
    pub takes_guidance: bool,
    pub suggested_prompt: Option<String>,
    pub suggested_avoid: Option<String>,
    /// How many bytes its files are, where they are all on the disk.
    pub bytes: Option<u64>,
}

impl Default for Chosen {
    fn default() -> Chosen {
        Chosen {
            draws: true,
            defaults: GenerationDefaults::default(),
            sizes: SIZES.to_vec(),
            takes_guidance: true,
            suggested_prompt: None,
            suggested_avoid: None,
            bytes: None,
        }
    }
}

/// A model libwaifu has opened: its manifest, read, and what can be asked of it.
struct Model(*mut WaifuModel);

impl Model {
    /// The manifest at `manifest`, or why libwaifu cannot read it.
    fn open(manifest: &Path) -> Result<Model, String> {
        let asked = c::c_string(&manifest.to_string_lossy());
        let model = unsafe { c::waifu_model_open(asked.as_ptr()) };
        match model.is_null() {
            true => Err(c::waifu_error()),
            false => Ok(Model(model)),
        }
    }

    fn int(&self, property: WaifuModelProperty) -> Option<i64> {
        let mut value = 0;
        unsafe { c::waifu_model_get_int_property(self.0, property, &mut value) }.then_some(value)
    }
}

impl Drop for Model {
    fn drop(&mut self) {
        unsafe { c::waifu_model_free(self.0) };
    }
}

/// What libwaifu says the model at `manifest` can be asked for, and what libwaifu_modelmanager
/// says its boxes start at, as one -- or why either says nothing.
pub fn look_at(manifest: &Path) -> Result<Chosen, String> {
    let model = Model::open(manifest)?;
    let asked = c::c_string(&manifest.to_string_lossy());
    let defaults = c::manager_json(unsafe { c::waifu_modelmanager_defaults_json(asked.as_ptr()) })
        .ok_or_else(c::manager_error)?;
    Ok(chosen_from(
        &defaults,
        model.int(c::WAIFU_MODEL_SUPPORTS_IMAGE_GENERATION) == Some(1),
        model
            .int(c::WAIFU_MODEL_SIZE_ON_DISK)
            .map(|bytes| bytes.max(0) as u64),
    ))
}

/// A model's description: libwaifu_modelmanager's defaults, as `waifu_modelmanager_defaults_json`
/// writes them, with whether libwaifu says it draws and how large its files are.
pub fn chosen_from(said: &serde_json::Value, draws: bool, bytes: Option<u64>) -> Chosen {
    let defaults = GenerationDefaults::default();
    let sizes: Vec<(i32, i32)> = said["sizes"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|size| Some((size[0].as_i64()? as i32, size[1].as_i64()? as i32)))
        .collect();
    Chosen {
        draws,
        defaults: GenerationDefaults {
            width: said["width"]
                .as_i64()
                .map_or(defaults.width, |width| width as i32),
            height: said["height"]
                .as_i64()
                .map_or(defaults.height, |height| height as i32),
            num_steps: said["steps"]
                .as_i64()
                .map_or(defaults.num_steps, |steps| steps as i32),
            guidance_scale: said["guidance"]
                .as_f64()
                .map_or(defaults.guidance_scale, |guidance| guidance as f32),
        },
        sizes: match sizes.is_empty() {
            true => SIZES.to_vec(),
            false => sizes,
        },
        takes_guidance: said["takes_guidance"].as_bool().unwrap_or(true),
        suggested_prompt: said["prompt"].as_str().map(str::to_string),
        suggested_avoid: said["avoid"].as_str().map(str::to_string),
        bytes,
    }
}

/// What an engine's job ends with: a picture as rows of RGB, or nothing.
#[derive(Debug, PartialEq)]
pub enum Ended<T> {
    Done(T),
    Cancelled,
    Failed(String),
}

/// What a job's callbacks say, down the channel the job was started with.
#[derive(Debug, PartialEq)]
pub enum Heard {
    Progress(GenerationProgress),
    Loaded(Ended<()>),
    Drawn(Ended<(u32, u32, Vec<u8>)>),
}

/// libwaifu's image generation pipeline on one device: a thread of its own in the library,
/// holding at most one model.
pub struct Engine {
    handle: *mut WaifuPipeline,
}

// The library's pipeline is called from any thread; what it holds stays on its own.
unsafe impl Send for Engine {}

impl Engine {
    /// Starts a pipeline on `runtime`, or says why it cannot.
    pub fn new(runtime: Runtime) -> Result<Engine, String> {
        let handle =
            c::waifu_pipeline_new(c::WAIFU_PIPELINE_IMAGE_GENERATION, runtime.waifu_device());
        match handle.is_null() {
            true => Err(c::waifu_error()),
            false => Ok(Engine { handle }),
        }
    }

    /// Reads the model whose manifest is at `manifest`, and waits for the end: how it went.
    pub fn load(&self, manifest: &Path) -> Ended<()> {
        let model = match Model::open(manifest) {
            Ok(model) => model,
            Err(error) => return Ended::Failed(error),
        };
        let (sender, heard) = channel();
        let sender = Box::new(sender);
        // The call copies what it needs: the model may go as soon as it returns.
        let job = unsafe {
            c::waifu_pipeline_load_model_async(
                self.handle,
                model.0,
                &*sender as *const Sender<Heard> as *mut c_void,
                None,
                Some(on_loaded),
            )
        };
        if job == 0 {
            return Ended::Failed(c::waifu_error());
        }
        // Exactly one end comes for a job with an id; the sender lives until it has.
        loop {
            match heard.recv() {
                Ok(Heard::Loaded(ended)) => return ended,
                Ok(_) => continue,
                Err(_) => return Ended::Failed("the engine went away".to_string()),
            }
        }
    }

    /// Starts drawing from noise; what it says comes down `sender`, ending with [`Heard::Drawn`].
    /// The job, to cancel it by.
    pub fn draw(
        &self,
        prompt: &str,
        options: &GenerationOptions,
        sender: &Sender<Heard>,
    ) -> Result<WaifuJob, String> {
        let prompt = c::c_string(prompt);
        let negative = c::c_string(&options.negative_prompt);
        let request = WaifuImageGenerationRequest {
            struct_size: std::mem::size_of::<WaifuImageGenerationRequest>() as u32,
            prompt: prompt.as_ptr(),
            negative: negative.as_ptr(),
            width: options.width.max(0) as u32,
            height: options.height.max(0) as u32,
            steps: options.num_steps.max(0) as u32,
            guidance: options.guidance_scale,
            seed: options.seed.unwrap_or_default(),
            // The terminal hands a run no pictures: what it draws is drawn from a prompt. The
            // strength and the scale are read only beside one, so they are the values a caller
            // with nothing to say about them passes.
            start_image: ptr::null(),
            start_image_strength: 0.8,
            control_image: ptr::null(),
            control_image_scale: 1.0,
            prompt_images: ptr::null(),
            prompt_image_count: 0,
        };
        let job = unsafe {
            c::waifu_pipeline_run_image_generation_async(
                self.handle,
                &request,
                sender as *const Sender<Heard> as *mut c_void,
                Some(on_progress),
                Some(on_drawn),
            )
        };
        match job {
            0 => Err(c::waifu_error()),
            job => Ok(job),
        }
    }

    /// Stops a job after the step it is on.
    pub fn cancel(&self, job: WaifuJob) {
        unsafe { c::waifu_pipeline_cancel(self.handle, job) };
    }
}

impl Drop for Engine {
    /// Stops what runs, ends what waits, drops the model and joins the library's thread: no
    /// callback of this pipeline's runs after this.
    fn drop(&mut self) {
        unsafe { c::waifu_pipeline_free(self.handle) };
    }
}

fn status_of(code: WaifuStatusCode, message: Option<String>) -> Ended<()> {
    match code {
        c::WAIFU_OK => Ended::Done(()),
        c::WAIFU_CANCELLED => Ended::Cancelled,
        _ => Ended::Failed(message.unwrap_or_else(|| "it failed".to_string())),
    }
}

unsafe extern "C" fn on_loaded(user_data: *mut c_void, event: *const WaifuResult) {
    let (sender, status) = (&*(user_data as *const Sender<Heard>), &(*event).status);
    let _ = sender.send(Heard::Loaded(status_of(
        status.code,
        c::text(status.message),
    )));
}

unsafe extern "C" fn on_progress(user_data: *mut c_void, event: *const WaifuProgress) {
    let (sender, event) = (&*(user_data as *const Sender<Heard>), &*event);
    let progress = match event.stage {
        c::WAIFU_STAGE_IMAGE_GENERATION_ENCODING_PROMPT => GenerationProgress::Encoding,
        c::WAIFU_STAGE_IMAGE_GENERATION_DENOISING => GenerationProgress::Step {
            done: event.done as i32,
            total: event.total as i32,
        },
        c::WAIFU_STAGE_IMAGE_GENERATION_DECODING_IMAGE => GenerationProgress::Decoding,
        // Loading, and a stage a newer library has that this tool does not know.
        _ => return,
    };
    let _ = sender.send(Heard::Progress(progress));
}

unsafe extern "C" fn on_drawn(user_data: *mut c_void, event: *const WaifuImageGenerationResult) {
    let (sender, event) = (&*(user_data as *const Sender<Heard>), &*event);
    let ended = match (
        status_of(event.status.code, c::text(event.status.message)),
        event.image.as_ref(),
    ) {
        (Ended::Done(()), Some(image)) => {
            let rgb = std::slice::from_raw_parts(image.rgb, image.len).to_vec();
            Ended::Done((image.width, image.height, rgb))
        }
        (Ended::Done(()), None) => Ended::Failed("the run ended with no picture".to_string()),
        (Ended::Cancelled, _) => Ended::Cancelled,
        (Ended::Failed(message), _) => Ended::Failed(message),
    };
    let _ = sender.send(Heard::Drawn(ended));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_description_fills_the_boxes_and_falls_back_where_it_says_nothing() {
        let said = serde_json::json!({
            "width": 832, "height": 1216, "steps": 8, "guidance": 1.0,
            "takes_guidance": false, "sizes": [[832, 1216], [1024, 1024], [1216, 832]],
            "prompt": "masterpiece", "avoid": null,
        });
        let chosen = chosen_from(&said, true, Some(6_938_000_000));
        assert!(chosen.draws);
        assert_eq!(chosen.defaults.num_steps, 8);
        assert_eq!(chosen.sizes.len(), 3);
        assert!(!chosen.takes_guidance);
        assert_eq!(chosen.suggested_prompt.as_deref(), Some("masterpiece"));
        assert_eq!(chosen.bytes, Some(6_938_000_000));

        let voice = chosen_from(&serde_json::json!({}), false, None);
        assert!(!voice.draws);
        assert_eq!(voice.sizes, SIZES.to_vec());
    }

    #[test]
    fn a_manifest_that_is_not_there_is_said_why() {
        let error = look_at(Path::new("/no/such/model.yaml")).unwrap_err();
        assert!(error.contains("/no/such/model.yaml"), "{error}");
    }
}
