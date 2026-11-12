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

use std::path::Path;
use std::ptr;

use crate::c::{
    self, WaifuImage, WaifuModel, WaifuModelAttribute, WaifuPipeline, WaifuRequest,
    WaifuRequestAttribute, WaifuStatusCode, WaifuTask, WaifuTaskAttribute,
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

    fn int(&self, attribute: WaifuModelAttribute) -> Option<i64> {
        let mut value = 0;
        unsafe { c::waifu_model_get_int_attribute(self.0, attribute, &mut value) }.then_some(value)
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

/// What a task ends with: a picture as rows of RGB, or nothing.
#[derive(Debug, PartialEq)]
pub enum Ended<T> {
    Done(T),
    Cancelled,
    Failed(String),
}

/// What a drawing says, read from its task.
#[derive(Debug, PartialEq)]
pub enum Heard {
    Progress(GenerationProgress),
    Drawn(Ended<(u32, u32, Vec<u8>)>),
}

/// A task of libwaifu's, freed when this is dropped -- which, before it has ended, cancels it.
struct Task(*mut WaifuTask);

// A task may be read, waited on and cancelled from any thread; this is used from one at a time.
unsafe impl Send for Task {}

impl Task {
    /// The task a call returned, or what the refusal said.
    fn started(task: *mut WaifuTask) -> Result<Task, String> {
        match task.is_null() {
            true => Err(c::waifu_error()),
            false => Ok(Task(task)),
        }
    }

    fn int(&self, attribute: WaifuTaskAttribute) -> Option<i64> {
        let mut value = 0;
        unsafe { c::waifu_task_get_int_attribute(self.0, attribute, &mut value) }.then_some(value)
    }

    /// How it ended, once it has.
    fn ended(&self) -> Ended<()> {
        let message = || {
            unsafe {
                c::text(c::waifu_task_get_string_attribute(
                    self.0,
                    c::WAIFU_TASK_STATUS_MESSAGE,
                ))
            }
            .unwrap_or_else(|| "it failed".to_string())
        };
        match self
            .int(c::WAIFU_TASK_STATUS)
            .map(|code| code as WaifuStatusCode)
        {
            Some(c::WAIFU_OK) => Ended::Done(()),
            Some(c::WAIFU_CANCELLED) => Ended::Cancelled,
            Some(_) => Ended::Failed(message()),
            None => Ended::Failed("the task has not ended".to_string()),
        }
    }
}

impl Drop for Task {
    fn drop(&mut self) {
        unsafe { c::waifu_task_free(self.0) };
    }
}

/// A picture being drawn: read what it says with [`next`](Drawing::next), the last of it
/// [`Heard::Drawn`]. Dropped before then, it is stopped.
pub struct Drawing(Task);

impl Drawing {
    /// The next thing it says, waiting up to `timeout_ms` for it; None where nothing came.
    pub fn next(&self, timeout_ms: u32) -> Option<Heard> {
        let task = &self.0;
        loop {
            match unsafe { c::waifu_task_next_event(task.0, timeout_ms) } {
                c::WAIFU_EVENT_PROGRESS => {
                    let progress = match task.int(c::WAIFU_TASK_PROGRESS_STAGE)? as i32 {
                        c::WAIFU_STAGE_IMAGE_GENERATION_ENCODING_PROMPT => {
                            GenerationProgress::Encoding
                        }
                        c::WAIFU_STAGE_IMAGE_GENERATION_DENOISING => GenerationProgress::Step {
                            done: task.int(c::WAIFU_TASK_PROGRESS_DONE).unwrap_or(0) as i32,
                            total: task.int(c::WAIFU_TASK_PROGRESS_TOTAL).unwrap_or(0) as i32,
                        },
                        c::WAIFU_STAGE_IMAGE_GENERATION_DECODING_IMAGE => {
                            GenerationProgress::Decoding
                        }
                        // Loading, and a stage a newer library has that this tool does not know.
                        _ => continue,
                    };
                    return Some(Heard::Progress(progress));
                }
                c::WAIFU_EVENT_END => return Some(Heard::Drawn(self.picture())),
                // Nothing in time.
                c::WAIFU_EVENT_NONE => return None,
                // A type this tool never asks for.
                _ => continue,
            }
        }
    }

    fn picture(&self) -> Ended<(u32, u32, Vec<u8>)> {
        let task = &self.0;
        match task.ended() {
            Ended::Done(()) => {}
            Ended::Cancelled => return Ended::Cancelled,
            Ended::Failed(message) => return Ended::Failed(message),
        }
        let mut image = WaifuImage {
            width: 0,
            height: 0,
            rgb: ptr::null(),
            len: 0,
        };
        let found = unsafe {
            c::waifu_task_get_image_attribute(
                task.0,
                c::WAIFU_TASK_IMAGE_GENERATION_IMAGE,
                &mut image,
            )
        };
        match found {
            true => {
                let rgb = unsafe { std::slice::from_raw_parts(image.rgb, image.len) }.to_vec();
                Ended::Done((image.width, image.height, rgb))
            }
            false => Ended::Failed("the run ended with no picture".to_string()),
        }
    }

    /// Stops it after the step it is on.
    pub fn cancel(&self) {
        unsafe { c::waifu_task_cancel(self.0 .0) };
    }
}

/// A request of libwaifu's, freed when this is dropped.
struct Request(*mut WaifuRequest);

impl Drop for Request {
    fn drop(&mut self) {
        unsafe { c::waifu_request_free(self.0) };
    }
}

impl Request {
    fn checked(&self, set: bool) -> Result<(), String> {
        match set {
            true => Ok(()),
            false => Err(c::waifu_error()),
        }
    }

    fn int(&self, attribute: WaifuRequestAttribute, value: i64) -> Result<(), String> {
        self.checked(unsafe { c::waifu_request_set_int_attribute(self.0, attribute, value) })
    }

    fn float(&self, attribute: WaifuRequestAttribute, value: f32) -> Result<(), String> {
        self.checked(unsafe {
            c::waifu_request_set_float_attribute(self.0, attribute, value as f64)
        })
    }

    fn string(&self, attribute: WaifuRequestAttribute, value: &str) -> Result<(), String> {
        let value = c::c_string(value);
        self.checked(unsafe {
            c::waifu_request_set_string_attribute(self.0, attribute, value.as_ptr())
        })
    }
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
        // The call copies what it needs: the model may go as soon as it returns.
        let task =
            match Task::started(unsafe { c::waifu_pipeline_load_model(self.handle, model.0) }) {
                Ok(task) => task,
                Err(error) => return Ended::Failed(error),
            };
        unsafe { c::waifu_task_wait(task.0, c::WAIFU_WAIT_FOREVER) };
        task.ended()
    }

    /// Starts drawing from noise.
    pub fn draw(&self, prompt: &str, options: &GenerationOptions) -> Result<Drawing, String> {
        let request = Request(c::waifu_request_new(c::WAIFU_PIPELINE_IMAGE_GENERATION));
        if request.0.is_null() {
            return Err(c::waifu_error());
        }
        // The terminal hands a run no pictures: what it draws is drawn from a prompt.
        request.string(c::WAIFU_REQUEST_IMAGE_GENERATION_PROMPT, prompt)?;
        if !options.negative_prompt.is_empty() {
            request.string(
                c::WAIFU_REQUEST_IMAGE_GENERATION_NEGATIVE,
                &options.negative_prompt,
            )?;
        }
        request.int(
            c::WAIFU_REQUEST_IMAGE_GENERATION_WIDTH,
            options.width as i64,
        )?;
        request.int(
            c::WAIFU_REQUEST_IMAGE_GENERATION_HEIGHT,
            options.height as i64,
        )?;
        request.int(
            c::WAIFU_REQUEST_IMAGE_GENERATION_STEPS,
            options.num_steps as i64,
        )?;
        request.float(
            c::WAIFU_REQUEST_IMAGE_GENERATION_GUIDANCE,
            options.guidance_scale,
        )?;
        request.int(
            c::WAIFU_REQUEST_IMAGE_GENERATION_SEED,
            options.seed.unwrap_or_default() as i64,
        )?;
        let task = unsafe { c::waifu_pipeline_run(self.handle, request.0) };
        Ok(Drawing(Task::started(task)?))
    }
}

impl Drop for Engine {
    /// Stops what runs, ends what waits, drops the model and joins the library's thread.
    fn drop(&mut self) {
        unsafe { c::waifu_pipeline_free(self.handle) };
    }
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
