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

//! libwaifu, as the drawing screen asks it: what a model is, and an engine on a device that reads
//! it and draws with it.
//!
//! The types are the screen's own -- the boxes it fills in, the progress it draws a bar from --
//! and what they are filled from is what libwaifu says through its C API.

use std::ffi::c_void;
use std::path::Path;
use std::ptr;
use std::sync::mpsc::{channel, Sender};

use crate::c::{
    self, WaifuCompletionEvent, WaifuDrawRequest, WaifuEngine, WaifuImage,
    WaifuImageCompletionEvent, WaifuImageInput, WaifuJob, WaifuProgressEvent, WaifuStage,
    WaifuStatusCode,
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
    /// With a picture to start from: how far the run walks away from it, 0 to 1.
    pub strength: f32,
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
            strength: 0.8,
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
    /// Why it cannot be handed a picture to start from, or None where it can.
    pub no_picture_because: Option<String>,
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
            no_picture_because: None,
            bytes: None,
        }
    }
}

/// What libwaifu says of the manifest at `manifest`, or why it says nothing.
/// What libwaifu says the model can be asked for, and what libwaifu_modelmanager says its boxes
/// start at, as one.
pub fn look_at(manifest: &Path) -> Result<Chosen, String> {
    let asked = c::c_string(&manifest.to_string_lossy());
    let mut said = c::waifu_json(unsafe { c::waifu_describe_json(asked.as_ptr()) }).ok_or_else(c::waifu_error)?;
    let defaults = c::manager_json(unsafe { c::waifu_modelmanager_defaults_json(asked.as_ptr()) })
        .ok_or_else(c::manager_error)?;
    // The two halves of one description. A `conversion` is in both, and its defaults go inside.
    if let (Some(said), Some(defaults)) = (said.as_object_mut(), defaults.as_object()) {
        for (key, value) in defaults {
            match (said.get_mut(key), value) {
                (Some(serde_json::Value::Object(inner)), serde_json::Value::Object(more)) => {
                    inner.extend(more.clone())
                }
                _ => {
                    said.insert(key.clone(), value.clone());
                }
            }
        }
    }
    Ok(chosen_from(&said))
}

/// A description, as `waifu_describe_json` and `waifu_modelmanager_defaults_json` write one
/// between them.
pub fn chosen_from(said: &serde_json::Value) -> Chosen {
    let defaults = GenerationDefaults::default();
    let sizes: Vec<(i32, i32)> = said["sizes"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|size| Some((size[0].as_i64()? as i32, size[1].as_i64()? as i32)))
        .collect();
    Chosen {
        draws: said["kind"] == "image",
        defaults: GenerationDefaults {
            width: said["width"].as_i64().map_or(defaults.width, |width| width as i32),
            height: said["height"].as_i64().map_or(defaults.height, |height| height as i32),
            num_steps: said["steps"].as_i64().map_or(defaults.num_steps, |steps| steps as i32),
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
        no_picture_because: said["why_not"]["start_from"].as_str().map(str::to_string),
        bytes: said["bytes"].as_u64(),
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

/// libwaifu's engine on one device: a thread of its own in the library, holding at most one model.
pub struct Engine {
    handle: *mut WaifuEngine,
}

// The library's engine is called from any thread; what it holds stays on its own.
unsafe impl Send for Engine {}

impl Engine {
    /// Starts an engine on `runtime`, or says why it cannot.
    pub fn new(runtime: Runtime) -> Result<Engine, String> {
        let handle = c::waifu_engine_new(runtime.waifu_device());
        match handle.is_null() {
            true => Err(c::waifu_error()),
            false => Ok(Engine { handle }),
        }
    }

    /// Reads the model whose manifest is at `manifest`, and waits for the end: how it went.
    pub fn load(&self, manifest: &Path) -> Ended<()> {
        let (sender, heard) = channel();
        let sender = Box::new(sender);
        let model = c::c_string(&manifest.to_string_lossy());
        let job = unsafe {
            c::waifu_engine_load_async(
                self.handle,
                model.as_ptr(),
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

    /// Starts drawing; what it says comes down `sender`, ending with [`Heard::Drawn`]. `from` is
    /// the picture to start from, as rows of RGB, at any size. The job, to cancel it by.
    pub fn draw(
        &self,
        prompt: &str,
        options: &GenerationOptions,
        from: Option<(u32, u32, &[u8])>,
        sender: &Sender<Heard>,
    ) -> Result<WaifuJob, String> {
        let prompt = c::c_string(prompt);
        let negative = c::c_string(&options.negative_prompt);
        let key = c::c_string("start_from");
        let image = from.map(|(width, height, rgb)| WaifuImage {
            width,
            height,
            rgb: rgb.as_ptr(),
            len: rgb.len(),
        });
        let input = image.as_ref().map(|image| WaifuImageInput {
            key: key.as_ptr(),
            image,
        });
        let request = WaifuDrawRequest {
            struct_size: std::mem::size_of::<WaifuDrawRequest>() as u32,
            prompt: prompt.as_ptr(),
            negative: negative.as_ptr(),
            images: input.as_ref().map_or(ptr::null(), |input| input as *const WaifuImageInput),
            image_count: input.is_some() as usize,
            width: options.width.max(0) as u32,
            height: options.height.max(0) as u32,
            steps: options.num_steps.max(0) as u32,
            guidance: options.guidance_scale,
            seed: options.seed.unwrap_or_default(),
            strength: options.strength,
            control_scale: 1.0,
        };
        let job = unsafe {
            c::waifu_engine_draw_async(
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
        unsafe { c::waifu_engine_cancel(self.handle, job) };
    }
}

impl Drop for Engine {
    /// Stops what runs, ends what waits, drops the model and joins the library's thread: no
    /// callback of this engine's runs after this.
    fn drop(&mut self) {
        unsafe { c::waifu_engine_free(self.handle) };
    }
}

fn status_of(code: WaifuStatusCode, message: Option<String>) -> Ended<()> {
    match code {
        WaifuStatusCode::WAIFU_OK => Ended::Done(()),
        WaifuStatusCode::WAIFU_CANCELLED => Ended::Cancelled,
        _ => Ended::Failed(message.unwrap_or_else(|| "it failed".to_string())),
    }
}

unsafe extern "C" fn on_loaded(user_data: *mut c_void, event: *const WaifuCompletionEvent) {
    let (sender, status) = (&*(user_data as *const Sender<Heard>), &(*event).status);
    let _ = sender.send(Heard::Loaded(status_of(status.code, c::text(status.message))));
}

unsafe extern "C" fn on_progress(user_data: *mut c_void, event: *const WaifuProgressEvent) {
    let (sender, event) = (&*(user_data as *const Sender<Heard>), &*event);
    let progress = match event.stage {
        WaifuStage::WAIFU_STAGE_ENCODING => GenerationProgress::Encoding,
        WaifuStage::WAIFU_STAGE_DRAWING => GenerationProgress::Step {
            done: event.done as i32,
            total: event.total as i32,
        },
        WaifuStage::WAIFU_STAGE_DECODING => GenerationProgress::Decoding,
        _ => return,
    };
    let _ = sender.send(Heard::Progress(progress));
}

unsafe extern "C" fn on_drawn(user_data: *mut c_void, event: *const WaifuImageCompletionEvent) {
    let (sender, event) = (&*(user_data as *const Sender<Heard>), &*event);
    let ended = match (status_of(event.status.code, c::text(event.status.message)), event.image.as_ref()) {
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
            "kind": "image", "width": 832, "height": 1216, "steps": 8, "guidance": 1.0,
            "takes_guidance": false, "sizes": [[832, 1216], [1024, 1024], [1216, 832]],
            "prompt": "masterpiece", "avoid": null,
            "why_not": { "start_from": "Anima cannot start from a picture yet" },
            "bytes": 6938000000u64,
        });
        let chosen = chosen_from(&said);
        assert!(chosen.draws);
        assert_eq!(chosen.defaults.num_steps, 8);
        assert_eq!(chosen.sizes.len(), 3);
        assert!(!chosen.takes_guidance);
        assert_eq!(chosen.suggested_prompt.as_deref(), Some("masterpiece"));
        assert!(chosen.no_picture_because.unwrap().contains("Anima"));
        assert_eq!(chosen.bytes, Some(6_938_000_000));

        let voice = chosen_from(&serde_json::json!({ "kind": "speech" }));
        assert!(!voice.draws);
        assert_eq!(voice.sizes, SIZES.to_vec());
    }

    #[test]
    fn a_manifest_that_is_not_there_is_said_why() {
        let error = look_at(Path::new("/no/such/model.yaml")).unwrap_err();
        assert!(error.contains("/no/such/model.yaml"), "{error}");
    }
}
