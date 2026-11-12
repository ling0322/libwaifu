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

//! The C API, called the way C calls it: through the `extern "C"` functions, setting a request a
//! attribute at a time and reading a task's events and attributes back.

use super::*;

use std::ffi::CStr;
use std::sync::Mutex;

/// Long enough for anything the stand-in voice does; a test that waits this long has failed.
const WAIT_MS: u32 = 20_000;

/// What a test reads of one task: its events, as they came.
#[derive(Debug, PartialEq)]
enum Heard {
    Progress(WaifuStage),
    /// A part of a streaming reading: its samples and its rate.
    Part(Vec<f32>, u32),
}

unsafe fn text(pointer: *const c_char) -> Option<String> {
    (!pointer.is_null()).then(|| CStr::from_ptr(pointer).to_string_lossy().into_owned())
}

fn last_error() -> String {
    unsafe { text(waifu_last_error()) }.unwrap_or_default()
}

/// The last refusal, as its code and its sentence.
fn refused() -> (WaifuStatusCode, String) {
    (waifu_last_error_code(), last_error())
}

/// A pipeline of `type` on the processor.
fn a_pipeline(r#type: WaifuPipelineType) -> *mut WaifuPipeline {
    let pipeline = waifu_pipeline_new(r#type, WAIFU_DEVICE_CPU);
    assert!(!pipeline.is_null(), "{}", last_error());
    pipeline
}

fn a_model(path: &str) -> *mut WaifuModel {
    let path = CString::new(path).unwrap();
    let model = unsafe { waifu_model_open(path.as_ptr()) };
    assert!(!model.is_null(), "{}", last_error());
    model
}

fn a_request(r#type: WaifuPipelineType) -> *mut WaifuRequest {
    let request = waifu_request_new(r#type);
    assert!(!request.is_null(), "{}", last_error());
    request
}

unsafe fn set_string(request: *mut WaifuRequest, attribute: WaifuRequestAttribute, value: &CStr) {
    assert!(
        waifu_request_set_string_attribute(request, attribute, value.as_ptr()),
        "{}",
        last_error()
    );
}

unsafe fn set_int(request: *mut WaifuRequest, attribute: WaifuRequestAttribute, value: i64) {
    assert!(
        waifu_request_set_int_attribute(request, attribute, value),
        "{}",
        last_error()
    );
}

/// A speech request for `text`, with the seed and temperature the tests read with.
unsafe fn a_speech_request(r#type: WaifuPipelineType, text: &CStr) -> *mut WaifuRequest {
    let request = a_request(r#type);
    let (text_attribute, seed, temperature) = match r#type {
        WAIFU_PIPELINE_TEXT_TO_SPEECH => (
            WAIFU_REQUEST_TEXT_TO_SPEECH_TEXT,
            WAIFU_REQUEST_TEXT_TO_SPEECH_SEED,
            WAIFU_REQUEST_TEXT_TO_SPEECH_TEMPERATURE,
        ),
        _ => (
            WAIFU_REQUEST_STREAMING_TEXT_TO_SPEECH_TEXT,
            WAIFU_REQUEST_STREAMING_TEXT_TO_SPEECH_SEED,
            WAIFU_REQUEST_STREAMING_TEXT_TO_SPEECH_TEMPERATURE,
        ),
    };
    set_string(request, text_attribute, text);
    set_int(request, seed, 7);
    assert!(waifu_request_set_float_attribute(request, temperature, 0.8));
    request
}

/// An image request for `prompt` at the smallest size and one step.
unsafe fn an_image_request(prompt: &CStr) -> *mut WaifuRequest {
    let request = a_request(WAIFU_PIPELINE_IMAGE_GENERATION);
    set_string(request, WAIFU_REQUEST_IMAGE_GENERATION_PROMPT, prompt);
    set_int(request, WAIFU_REQUEST_IMAGE_GENERATION_WIDTH, 64);
    set_int(request, WAIFU_REQUEST_IMAGE_GENERATION_HEIGHT, 64);
    set_int(request, WAIFU_REQUEST_IMAGE_GENERATION_STEPS, 1);
    request
}

/// Runs `request` on `pipeline`, and frees the request: the task.
unsafe fn run(pipeline: *mut WaifuPipeline, request: *mut WaifuRequest) -> *mut WaifuTask {
    let task = waifu_pipeline_run(pipeline, request);
    waifu_request_free(request);
    task
}

/// Every event of `task` up to its end, in order.
unsafe fn events(task: *mut WaifuTask) -> Vec<Heard> {
    let mut heard = Vec::new();
    loop {
        match waifu_task_next_event(task, WAIT_MS) {
            WAIFU_EVENT_PROGRESS => {
                let mut stage = 0;
                assert!(waifu_task_get_int_attribute(
                    task,
                    WAIFU_TASK_PROGRESS_STAGE,
                    &mut stage
                ));
                heard.push(Heard::Progress(stage as WaifuStage));
            }
            WAIFU_EVENT_AUDIO_PART => {
                let mut part = WaifuAudio {
                    samples: ptr::null(),
                    count: 0,
                    sample_rate: 0,
                };
                assert!(waifu_task_get_audio_attribute(
                    task,
                    WAIFU_TASK_STREAMING_TEXT_TO_SPEECH_AUDIO_PART,
                    &mut part
                ));
                let samples = std::slice::from_raw_parts(part.samples, part.count).to_vec();
                heard.push(Heard::Part(samples, part.sample_rate));
            }
            WAIFU_EVENT_END => return heard,
            other => panic!("event {other} before the end"),
        }
    }
}

/// How `task` ended: its status and its message.
unsafe fn ended(task: *mut WaifuTask) -> (WaifuStatusCode, Option<String>) {
    assert!(waifu_task_wait(task, WAIT_MS), "the task did not end");
    let mut code = -1;
    assert!(waifu_task_get_int_attribute(
        task,
        WAIFU_TASK_STATUS,
        &mut code
    ));
    (
        code as WaifuStatusCode,
        text(waifu_task_get_string_attribute(
            task,
            WAIFU_TASK_STATUS_MESSAGE,
        )),
    )
}

/// The audio a task made under `attribute`.
unsafe fn audio(task: *mut WaifuTask, attribute: WaifuTaskAttribute) -> Option<(usize, u32)> {
    let mut audio = WaifuAudio {
        samples: ptr::null(),
        count: 0,
        sample_rate: 0,
    };
    waifu_task_get_audio_attribute(task, attribute, &mut audio)
        .then_some((audio.count, audio.sample_rate))
}

/// `pipeline` with the stand-in voice loaded.
unsafe fn with_tones(pipeline: *mut WaifuPipeline) {
    let tones = a_model("tones");
    let task = waifu_pipeline_load_model(pipeline, tones);
    // The model is the caller's to free as soon as the call returns.
    waifu_model_free(tones);
    assert!(!task.is_null(), "{}", last_error());
    assert_eq!(ended(task), (WAIFU_OK, None));
    waifu_task_free(task);
}

unsafe fn int_attribute(model: *const WaifuModel, attribute: WaifuModelAttribute) -> Option<i64> {
    let mut value = -7;
    waifu_model_get_int_attribute(model, attribute, &mut value).then_some(value)
}

unsafe fn string_attribute(
    model: *const WaifuModel,
    attribute: WaifuModelAttribute,
) -> Option<String> {
    text(waifu_model_get_string_attribute(model, attribute))
}

#[test]
fn the_library_says_what_it_is() {
    assert_eq!(waifu_abi_version(), 6 << 16 | 1);
    assert_eq!(waifu_abi_version() >> 16, WAIFU_ABI_VERSION_MAJOR);
    assert_eq!(waifu_abi_version() & 0xffff, WAIFU_ABI_VERSION_MINOR);
    assert_eq!(waifu_abi_version(), WAIFU_ABI_VERSION);
    let version = unsafe { text(waifu_version()) }.unwrap();
    assert_eq!(version, env!("CARGO_PKG_VERSION"));
    // The same pointer every time: static.
    assert_eq!(waifu_version(), waifu_version());
    assert_eq!(waifu_revision(), waifu_revision());
}

#[test]
fn the_last_error_is_empty_until_a_call_is_refused_and_only_a_refusal_sets_it() {
    // On a thread of its own, which no other test has refused anything on.
    std::thread::spawn(|| unsafe {
        assert_eq!(refused(), (WAIFU_OK, String::new()));

        assert!(waifu_model_open(c"no-such-model:v9".as_ptr()).is_null());
        let (code, message) = refused();
        assert_eq!(code, WAIFU_ERR_UNKNOWN_MODEL);
        assert!(message.contains("no-such-model:v9"), "{message}");

        // A query that finds nothing is not refused, and leaves the last error as it was.
        let mut value = 0;
        assert!(!waifu_model_get_int_attribute(ptr::null(), 1, &mut value));
        assert!(!waifu_device_get_int_attribute(
            WAIFU_DEVICE_CPU,
            WAIFU_DEVICE_MEMORY_BUDGET,
            &mut value
        ));
        assert!(!waifu_device_is_available(99));
        assert!(!waifu_task_get_int_attribute(
            ptr::null(),
            WAIFU_TASK_STATUS,
            &mut value
        ));
        assert_eq!(waifu_task_next_event(ptr::null_mut(), 0), WAIFU_EVENT_NONE);
        assert_eq!(refused(), (code, message));

        assert!(waifu_model_open(ptr::null()).is_null());
        assert_eq!(waifu_last_error_code(), WAIFU_ERR_INVALID_ARGUMENT);
    })
    .join()
    .unwrap();
}

#[test]
fn a_model_is_asked_about_one_attribute_at_a_time() {
    unsafe {
        let sdxl = a_model(&crate::engine::tests::a_manifest("c-api-sdxl", "sdxl"));
        assert_eq!(
            int_attribute(sdxl, WAIFU_MODEL_SUPPORTS_IMAGE_GENERATION),
            Some(1)
        );
        assert_eq!(
            string_attribute(sdxl, WAIFU_MODEL_SUPPORTS_IMAGE_GENERATION),
            None
        );
        assert_eq!(
            int_attribute(sdxl, WAIFU_MODEL_IMAGE_GENERATION_WIDTH_HEIGHT_MULTIPLE_OF),
            Some(32)
        );
        assert!(string_attribute(sdxl, WAIFU_MODEL_IMAGE_GENERATION_SAMPLER).is_some());
        assert_eq!(
            int_attribute(sdxl, WAIFU_MODEL_SUPPORTS_IMAGE_GENERATION_FROM_START_IMAGE),
            Some(1)
        );
        // The weights the manifest names are not there.
        assert_eq!(int_attribute(sdxl, WAIFU_MODEL_SIZE_ON_DISK), None);
        assert_eq!(int_attribute(sdxl, WAIFU_MODEL_SAMPLE_RATE), None);

        // Every SUPPORTS_ it does not have is 0, with a reason, never NULL -- a type's details too.
        for attribute in [
            WAIFU_MODEL_SUPPORTS_TEXT_TO_SPEECH,
            WAIFU_MODEL_SUPPORTS_STREAMING_TEXT_TO_SPEECH,
            WAIFU_MODEL_SUPPORTS_VOICE_CONVERSION,
            WAIFU_MODEL_SUPPORTS_LAYER_DECOMPOSITION,
            WAIFU_MODEL_SUPPORTS_SPEECH_TO_TEXT,
            WAIFU_MODEL_SUPPORTS_IMAGE_GENERATION_WITH_CONTROL_IMAGE,
            WAIFU_MODEL_SUPPORTS_IMAGE_GENERATION_WITH_PROMPT_IMAGES,
            WAIFU_MODEL_SUPPORTS_TEXT_TO_SPEECH_LIKE_REFERENCE,
            WAIFU_MODEL_SUPPORTS_TEXT_TO_SPEECH_WITHOUT_REFERENCE,
            WAIFU_MODEL_SUPPORTS_TEXT_TO_SPEECH_WITHOUT_REFERENCE_TRANSCRIPT,
            WAIFU_MODEL_SUPPORTS_TEXT_TO_SPEECH_WITH_STYLE,
            WAIFU_MODEL_SUPPORTS_VOICE_CONVERSION_OF_STYLE,
        ] {
            assert_eq!(int_attribute(sdxl, attribute), Some(0), "{attribute}");
            let why = string_attribute(sdxl, attribute).expect("a reason");
            assert!(!why.is_empty(), "{attribute}");
        }
        // Another type's constraints are not there at all.
        assert_eq!(
            int_attribute(sdxl, WAIFU_MODEL_LAYER_DECOMPOSITION_RESOLUTION_MULTIPLE_OF),
            None
        );
        // A number this library does not know is not there, and *value is left alone.
        let mut value = 42;
        assert!(!waifu_model_get_int_attribute(sdxl, 777, &mut value));
        assert_eq!(value, 42);
        assert!(waifu_model_get_string_attribute(sdxl, 777).is_null());
        waifu_model_free(sdxl);

        let tones = a_model("tones");
        assert_eq!(
            int_attribute(tones, WAIFU_MODEL_SUPPORTS_TEXT_TO_SPEECH),
            Some(1)
        );
        assert_eq!(
            int_attribute(tones, WAIFU_MODEL_SUPPORTS_STREAMING_TEXT_TO_SPEECH),
            Some(1)
        );
        assert!(int_attribute(tones, WAIFU_MODEL_SAMPLE_RATE).unwrap() > 0);
        assert_eq!(
            int_attribute(tones, WAIFU_MODEL_SUPPORTS_IMAGE_GENERATION),
            Some(0)
        );
        waifu_model_free(tones);
        waifu_model_free(ptr::null_mut());
    }
}

#[test]
fn a_pipeline_is_made_for_one_type_on_one_device() {
    unsafe {
        let pipeline = a_pipeline(WAIFU_PIPELINE_TEXT_TO_SPEECH);
        assert_eq!(
            waifu_pipeline_get_type(pipeline),
            WAIFU_PIPELINE_TEXT_TO_SPEECH
        );
        assert_eq!(waifu_pipeline_get_device(pipeline), WAIFU_DEVICE_CPU);
        waifu_pipeline_free(pipeline);
        waifu_pipeline_free(ptr::null_mut());

        // A type or a device this library does not know is refused.
        assert!(waifu_pipeline_new(0, WAIFU_DEVICE_CPU).is_null());
        assert_eq!(waifu_last_error_code(), WAIFU_ERR_INVALID_ARGUMENT);
        assert!(waifu_pipeline_new(WAIFU_PIPELINE_TEXT_TO_SPEECH, 4).is_null());
        assert_eq!(waifu_last_error_code(), WAIFU_ERR_INVALID_ARGUMENT);

        // A device this build or machine cannot use.
        for device in [WAIFU_DEVICE_METAL, WAIFU_DEVICE_CUDA, WAIFU_DEVICE_VULKAN] {
            if !waifu_device_is_available(device) {
                assert!(waifu_pipeline_new(WAIFU_PIPELINE_TEXT_TO_SPEECH, device).is_null());
                assert_eq!(waifu_last_error_code(), WAIFU_ERR_DEVICE_UNAVAILABLE);
            }
        }
        assert!(waifu_device_is_available(WAIFU_DEVICE_CPU));
    }
}

#[test]
fn a_request_takes_its_own_type_s_attributes_each_of_its_type_and_range() {
    unsafe {
        assert!(waifu_request_new(0).is_null());
        assert_eq!(waifu_last_error_code(), WAIFU_ERR_INVALID_ARGUMENT);

        let request = a_request(WAIFU_PIPELINE_IMAGE_GENERATION);
        assert_eq!(
            waifu_request_get_type(request),
            WAIFU_PIPELINE_IMAGE_GENERATION
        );
        assert_eq!(waifu_request_get_type(ptr::null()), 0);

        // A number this library does not know, and another type's attribute.
        assert!(!waifu_request_set_int_attribute(request, 777, 1));
        assert_eq!(waifu_last_error_code(), WAIFU_ERR_UNKNOWN_ATTRIBUTE);
        assert!(!waifu_request_set_string_attribute(
            request,
            WAIFU_REQUEST_TEXT_TO_SPEECH_TEXT,
            c"hello".as_ptr()
        ));
        let (code, message) = refused();
        assert_eq!(code, WAIFU_ERR_UNKNOWN_ATTRIBUTE);
        assert!(message.contains("TEXT_TO_SPEECH_TEXT"), "{message}");
        assert!(message.contains("image generation"), "{message}");
        assert!(!waifu_request_unset_attribute(request, 777));
        assert_eq!(waifu_last_error_code(), WAIFU_ERR_UNKNOWN_ATTRIBUTE);

        // A value of another type says which setter it takes.
        assert!(!waifu_request_set_float_attribute(
            request,
            WAIFU_REQUEST_IMAGE_GENERATION_STEPS,
            20.0
        ));
        let (code, message) = refused();
        assert_eq!(code, WAIFU_ERR_INVALID_ARGUMENT);
        assert!(
            message.contains("waifu_request_set_int_attribute"),
            "{message}"
        );
        assert!(!waifu_request_set_int_attribute(
            request,
            WAIFU_REQUEST_IMAGE_GENERATION_PROMPT,
            1
        ));
        assert_eq!(waifu_last_error_code(), WAIFU_ERR_INVALID_ARGUMENT);

        // Out of range, and not UTF-8.
        for (attribute, value) in [
            (WAIFU_REQUEST_IMAGE_GENERATION_WIDTH, 32),
            (WAIFU_REQUEST_IMAGE_GENERATION_WIDTH, 4096),
            (WAIFU_REQUEST_IMAGE_GENERATION_STEPS, 0),
        ] {
            assert!(!waifu_request_set_int_attribute(request, attribute, value));
            assert_eq!(waifu_last_error_code(), WAIFU_ERR_INVALID_ARGUMENT);
        }
        for value in [-0.5, 1.5, f64::NAN] {
            assert!(!waifu_request_set_float_attribute(
                request,
                WAIFU_REQUEST_IMAGE_GENERATION_START_IMAGE_STRENGTH,
                value
            ));
        }

        // A float is checked as the f32 the engine is handed: a double too large for one would be
        // infinity, and one too small for a positive minimum would be 0.
        for value in [1e100, f64::MAX, f64::INFINITY, -1e100] {
            assert!(!waifu_request_set_float_attribute(
                request,
                WAIFU_REQUEST_IMAGE_GENERATION_GUIDANCE,
                value
            ));
            assert_eq!(waifu_last_error_code(), WAIFU_ERR_INVALID_ARGUMENT);
        }
        assert!(waifu_request_set_float_attribute(
            request,
            WAIFU_REQUEST_IMAGE_GENERATION_GUIDANCE,
            f32::MAX as f64
        ));
        let speech = a_request(WAIFU_PIPELINE_TEXT_TO_SPEECH);
        for value in [1e-100, 0.0, -1.0] {
            assert!(!waifu_request_set_float_attribute(
                speech,
                WAIFU_REQUEST_TEXT_TO_SPEECH_SPEED,
                value
            ));
            assert_eq!(waifu_last_error_code(), WAIFU_ERR_INVALID_ARGUMENT);
        }
        // What is kept is the f32, and what the engine is handed is finite and above 0.
        set_string(speech, WAIFU_REQUEST_TEXT_TO_SPEECH_TEXT, c"hello");
        assert!(waifu_request_set_float_attribute(
            speech,
            WAIFU_REQUEST_TEXT_TO_SPEECH_SPEED,
            1e-40
        ));
        let speed = (*speech).speak().unwrap().speed;
        assert!(speed > 0.0 && speed.is_finite(), "{speed}");
        assert!(waifu_request_set_float_attribute(
            speech,
            WAIFU_REQUEST_TEXT_TO_SPEECH_SPEED,
            1.1
        ));
        assert_eq!((*speech).speak().unwrap().speed, 1.1f32);
        waifu_request_free(speech);
        assert!(!waifu_request_set_string_attribute(
            request,
            WAIFU_REQUEST_IMAGE_GENERATION_PROMPT,
            c"\xff".as_ptr()
        ));
        assert!(last_error().contains("UTF-8"), "{}", last_error());

        // A seed is any 64 bits; NULL unsets a string; a NULL request is refused.
        set_int(request, WAIFU_REQUEST_IMAGE_GENERATION_SEED, -1);
        assert!(waifu_request_set_string_attribute(
            request,
            WAIFU_REQUEST_IMAGE_GENERATION_NEGATIVE,
            ptr::null()
        ));
        assert!(!waifu_request_set_int_attribute(
            ptr::null_mut(),
            WAIFU_REQUEST_IMAGE_GENERATION_STEPS,
            20
        ));

        // Booleans are 0 or 1.
        let layers = a_request(WAIFU_PIPELINE_LAYER_DECOMPOSITION);
        assert!(!waifu_request_set_int_attribute(
            layers,
            WAIFU_REQUEST_LAYER_DECOMPOSITION_STRATIFY_LAYERS,
            2
        ));
        set_int(layers, WAIFU_REQUEST_LAYER_DECOMPOSITION_STRATIFY_LAYERS, 1);
        waifu_request_free(layers);
        waifu_request_free(request);
        waifu_request_free(ptr::null_mut());
    }
}

#[test]
fn an_unset_attribute_is_the_library_s_default() {
    unsafe {
        let request = &mut *a_request(WAIFU_PIPELINE_IMAGE_GENERATION);
        set_string(request, WAIFU_REQUEST_IMAGE_GENERATION_PROMPT, c"a cat");
        let draw = request.draw().unwrap();
        assert_eq!((draw.width, draw.height, draw.steps), (1024, 1024, 20));
        assert_eq!((draw.guidance, draw.seed), (5.0, 0));
        assert_eq!((draw.strength, draw.control_scale), (0.8, 1.0));
        assert!(draw.negative.is_empty() && draw.images.is_empty());

        // Set, then unset: the default again.
        set_int(request, WAIFU_REQUEST_IMAGE_GENERATION_STEPS, 8);
        assert_eq!(request.draw().unwrap().steps, 8);
        assert!(waifu_request_unset_attribute(
            request,
            WAIFU_REQUEST_IMAGE_GENERATION_STEPS
        ));
        assert_eq!(request.draw().unwrap().steps, 20);
        waifu_request_free(request);

        let speech = &mut *a_request(WAIFU_PIPELINE_TEXT_TO_SPEECH);
        set_string(speech, WAIFU_REQUEST_TEXT_TO_SPEECH_TEXT, c"hello");
        let speak = speech.speak().unwrap();
        assert_eq!((speak.speed, speak.temperature, speak.seed), (1.0, 1.0, 0));
        assert!(speak.like.is_none() && speak.style.is_none());
        assert!(speak.reference_transcript.is_none());
        waifu_request_free(speech);

        let layers = &mut *a_request(WAIFU_PIPELINE_LAYER_DECOMPOSITION);
        let pixels = [0u8; 4 * 4 * 4];
        let image = WaifuRgbaImage {
            width: 4,
            height: 4,
            rgba: pixels.as_ptr(),
            len: pixels.len(),
        };
        assert!(waifu_request_set_rgba_image_attribute(
            layers,
            WAIFU_REQUEST_LAYER_DECOMPOSITION_IMAGE,
            &image
        ));
        // 0 is the engine's "the model's own".
        let decompose = layers.layers().unwrap();
        assert_eq!(
            (decompose.steps, decompose.depth_steps, decompose.resolution),
            (0, 0, 0)
        );
        assert!(!decompose.split_pairs && !decompose.stratify);
        waifu_request_free(layers);
    }
}

#[test]
fn a_model_another_type_s_and_a_request_another_type_s_are_refused_at_the_call() {
    unsafe {
        let pipeline = a_pipeline(WAIFU_PIPELINE_TEXT_TO_SPEECH);

        let sdxl = a_model(&crate::engine::tests::a_manifest("c-api-sdxl-tts", "sdxl"));
        let task = waifu_pipeline_load_model(pipeline, sdxl);
        waifu_model_free(sdxl);
        assert!(task.is_null());
        let (code, message) = refused();
        assert_eq!(code, WAIFU_ERR_INVALID_ARGUMENT);
        assert!(message.contains("text to speech"), "{message}");

        let task = run(pipeline, an_image_request(c"a cat"));
        assert!(task.is_null());
        let (code, message) = refused();
        assert_eq!(code, WAIFU_ERR_INVALID_ARGUMENT);
        assert!(message.contains("image generation"), "{message}");

        assert!(waifu_pipeline_load_model(pipeline, ptr::null()).is_null());
        assert!(waifu_pipeline_run(pipeline, ptr::null()).is_null());
        assert!(waifu_pipeline_run(ptr::null_mut(), ptr::null()).is_null());
        waifu_pipeline_free(pipeline);
    }
}

#[test]
fn an_image_request_that_cannot_be_run_is_refused_at_the_call() {
    unsafe {
        let pipeline = a_pipeline(WAIFU_PIPELINE_IMAGE_GENERATION);

        // A placeholder no image answers.
        assert!(run(pipeline, an_image_request(c"the girl in <|girl|>")).is_null());
        assert!(last_error().contains("<|girl|>"), "{}", last_error());

        // No prompt at all.
        let request = a_request(WAIFU_PIPELINE_IMAGE_GENERATION);
        assert!(run(pipeline, request).is_null());
        assert!(
            last_error().contains("IMAGE_GENERATION_PROMPT"),
            "{}",
            last_error()
        );

        // A prompt image under the key the engine keeps for the start image, and one given twice.
        let pixels = [0u8; 4 * 4 * 3];
        let image = WaifuImage {
            width: 4,
            height: 4,
            rgb: pixels.as_ptr(),
            len: pixels.len(),
        };
        let request = an_image_request(c"a cat like <|cat|>");
        assert!(!waifu_request_add_prompt_image(
            request,
            c"start_from".as_ptr(),
            &image
        ));
        assert!(
            last_error().contains("IMAGE_GENERATION_START_IMAGE"),
            "{}",
            last_error()
        );
        assert_eq!(waifu_last_error_code(), WAIFU_ERR_INVALID_ARGUMENT);
        assert!(waifu_request_add_prompt_image(
            request,
            c"cat".as_ptr(),
            &image
        ));
        assert!(!waifu_request_add_prompt_image(
            request,
            c"cat".as_ptr(),
            &image
        ));
        assert!(last_error().contains("\"cat\""), "{}", last_error());
        // Unset, the prompt names an image there is not.
        assert!(waifu_request_unset_attribute(
            request,
            WAIFU_REQUEST_IMAGE_GENERATION_PROMPT_IMAGES
        ));
        assert!(run(pipeline, request).is_null());
        assert!(last_error().contains("<|cat|>"), "{}", last_error());

        // A run with nothing loaded ends as not loaded, and not at the call: no image comes.
        let task = run(pipeline, an_image_request(c"a cat"));
        assert!(!task.is_null(), "{}", last_error());
        assert_eq!(ended(task).0, WAIFU_ERR_NOT_LOADED);
        let mut picture = WaifuImage {
            width: 0,
            height: 0,
            rgb: ptr::null(),
            len: 0,
        };
        assert!(!waifu_task_get_image_attribute(
            task,
            WAIFU_TASK_IMAGE_GENERATION_IMAGE,
            &mut picture
        ));
        waifu_task_free(task);
        waifu_pipeline_free(pipeline);
    }
}

#[test]
fn a_voice_is_loaded_and_speaks_through_the_c_api() {
    unsafe {
        let pipeline = a_pipeline(WAIFU_PIPELINE_TEXT_TO_SPEECH);
        with_tones(pipeline);

        let task = run(
            pipeline,
            a_speech_request(WAIFU_PIPELINE_TEXT_TO_SPEECH, c"hello there"),
        );
        assert!(!task.is_null(), "{}", last_error());

        // Progress before the end, in text to speech's stages, and the end with the samples. Which
        // stages are read is not fixed: progress not yet read is the latest, and the stand-in voice
        // can be through reading the text before this reads anything. At least the last is there.
        let heard = events(task);
        assert!(!heard.is_empty(), "no progress before the end");
        assert!(
            heard
                .iter()
                .all(|event| matches!(event, Heard::Progress(stage) if (200..300).contains(stage))),
            "{heard:?}"
        );
        // END for every call after.
        assert_eq!(waifu_task_next_event(task, 0), WAIFU_EVENT_END);
        assert_eq!(ended(task), (WAIFU_OK, None));
        let (count, rate) = audio(task, WAIFU_TASK_TEXT_TO_SPEECH_AUDIO).expect("audio");
        assert!(count > 0 && rate > 0);
        // Another type's result is not there, and nor is a number this library does not know.
        assert_eq!(audio(task, WAIFU_TASK_VOICE_CONVERSION_AUDIO), None);
        let mut value = 0;
        assert!(!waifu_task_get_int_attribute(task, 777, &mut value));
        // A progress the last event said is still there to read.
        assert!(waifu_task_get_int_attribute(
            task,
            WAIFU_TASK_PROGRESS_STAGE,
            &mut value
        ));
        let mut seconds = -1.0;
        assert!(waifu_task_get_float_attribute(
            task,
            WAIFU_TASK_PROGRESS_SECONDS,
            &mut seconds
        ));
        assert!(seconds >= 0.0);
        waifu_task_free(task);

        // An empty text is refused at the call.
        assert!(run(
            pipeline,
            a_speech_request(WAIFU_PIPELINE_TEXT_TO_SPEECH, c" ")
        )
        .is_null());
        assert_eq!(waifu_last_error_code(), WAIFU_ERR_INVALID_ARGUMENT);

        // Unloaded, a run is not loaded at the end.
        let unload = waifu_pipeline_unload_model(pipeline);
        assert_eq!(ended(unload), (WAIFU_OK, None));
        waifu_task_free(unload);
        let task = run(
            pipeline,
            a_speech_request(WAIFU_PIPELINE_TEXT_TO_SPEECH, c"hello there"),
        );
        let (code, message) = ended(task);
        assert_eq!(code, WAIFU_ERR_NOT_LOADED);
        assert!(message.is_some());
        assert_eq!(audio(task, WAIFU_TASK_TEXT_TO_SPEECH_AUDIO), None);
        waifu_task_free(task);

        waifu_pipeline_free(pipeline);
    }
}

#[test]
fn a_request_is_copied_at_the_call_and_can_be_run_again() {
    unsafe {
        let pipeline = a_pipeline(WAIFU_PIPELINE_TEXT_TO_SPEECH);
        with_tones(pipeline);
        let request = a_speech_request(WAIFU_PIPELINE_TEXT_TO_SPEECH, c"hello there");
        let first = waifu_pipeline_run(pipeline, request);
        // Changed after the call: the first run is not.
        set_string(
            request,
            WAIFU_REQUEST_TEXT_TO_SPEECH_TEXT,
            c"hello there, and a good deal more than that",
        );
        let second = waifu_pipeline_run(pipeline, request);
        waifu_request_free(request);
        assert_eq!(ended(first).0, WAIFU_OK);
        assert_eq!(ended(second).0, WAIFU_OK);
        let short = audio(first, WAIFU_TASK_TEXT_TO_SPEECH_AUDIO).unwrap().0;
        let long = audio(second, WAIFU_TASK_TEXT_TO_SPEECH_AUDIO).unwrap().0;
        assert!(long > short, "{long} {short}");
        waifu_task_free(first);
        waifu_task_free(second);
        waifu_pipeline_free(pipeline);
    }
}

#[test]
fn a_task_is_cancelled_and_waits_with_a_timeout() {
    unsafe {
        let pipeline = a_pipeline(WAIFU_PIPELINE_TEXT_TO_SPEECH);
        with_tones(pipeline);
        // Three in the queue. The stand-in voice is quick enough that the last may be over by the
        // time it is cancelled: the engine's own tests say when a cancel lands.
        let tasks: Vec<_> = (0..3)
            .map(|_| {
                run(
                    pipeline,
                    a_speech_request(WAIFU_PIPELINE_TEXT_TO_SPEECH, c"hello there"),
                )
            })
            .collect();
        waifu_task_cancel(tasks[2]);
        let (code, message) = ended(tasks[2]);
        assert!(code == WAIFU_CANCELLED || code == WAIFU_OK, "{code}");
        assert_eq!(message, None);
        assert_eq!(waifu_task_next_event(tasks[2], 0), WAIFU_EVENT_END);
        // Cancelling an ended task does nothing.
        waifu_task_cancel(tasks[2]);
        for task in &tasks[..2] {
            assert_eq!(ended(*task).0, WAIFU_OK);
        }
        for task in tasks {
            waifu_task_free(task);
        }
        waifu_task_free(ptr::null_mut());
        waifu_task_cancel(ptr::null_mut());

        // A wait that times out says so; the task is still there to wait on.
        let unload = waifu_pipeline_unload_model(pipeline);
        let mut waits = 0;
        while !waifu_task_wait(unload, 0) {
            waits += 1;
            assert!(waits < 1_000_000);
        }
        waifu_task_free(unload);
        waifu_pipeline_free(pipeline);
    }
}

#[test]
fn freeing_a_pipeline_ends_every_task_and_the_tasks_outlive_it() {
    unsafe {
        let pipeline = a_pipeline(WAIFU_PIPELINE_TEXT_TO_SPEECH);
        let tones = a_model("tones");
        let load = waifu_pipeline_load_model(pipeline, tones);
        waifu_model_free(tones);
        let runs: Vec<_> = (0..3)
            .map(|_| {
                run(
                    pipeline,
                    a_speech_request(WAIFU_PIPELINE_TEXT_TO_SPEECH, c"hello there"),
                )
            })
            .collect();
        // A task freed before it ends is cancelled, and let go of when it does.
        waifu_task_free(runs[2]);
        waifu_pipeline_free(pipeline);

        // Every task has ended by the time the free returned, whatever it came to.
        assert!(waifu_task_wait(load, 0));
        for task in &runs[..2] {
            assert!(waifu_task_wait(*task, 0));
            let (code, _) = ended(*task);
            assert!(code == WAIFU_OK || code == WAIFU_CANCELLED, "{code}");
            // Cancelling one whose pipeline is gone does nothing.
            waifu_task_cancel(*task);
        }
        waifu_task_free(load);
        waifu_task_free(runs[0]);
        waifu_task_free(runs[1]);
    }
}

#[test]
fn a_streaming_reading_hands_over_its_parts_before_the_whole() {
    unsafe {
        let pipeline = a_pipeline(WAIFU_PIPELINE_STREAMING_TEXT_TO_SPEECH);
        with_tones(pipeline);

        // Every part before the end, and the whole the parts end to end.
        let task = run(
            pipeline,
            a_speech_request(WAIFU_PIPELINE_STREAMING_TEXT_TO_SPEECH, c"hello there"),
        );
        assert!(!task.is_null(), "{}", last_error());
        let heard = events(task);
        let parts: Vec<_> = heard
            .iter()
            .filter_map(|event| match event {
                Heard::Part(samples, rate) => Some((samples, *rate)),
                Heard::Progress(_) => None,
            })
            .collect();
        assert!(!parts.is_empty());
        assert!(
            heard.iter().all(|event| match event {
                Heard::Progress(stage) => (600..700).contains(stage),
                Heard::Part(..) => true,
            }),
            "{heard:?}"
        );
        assert_eq!(ended(task).0, WAIFU_OK);
        let count: usize = parts.iter().map(|(samples, _)| samples.len()).sum();
        let (whole, rate) = audio(task, WAIFU_TASK_STREAMING_TEXT_TO_SPEECH_AUDIO).unwrap();
        assert_eq!(whole, count);
        assert!(parts.iter().all(|(_, part_rate)| *part_rate == rate));
        // The last part is still there to read after the end, and it is the last one heard.
        assert_eq!(
            audio(task, WAIFU_TASK_STREAMING_TEXT_TO_SPEECH_AUDIO_PART).map(|(count, _)| count),
            Some(parts.last().unwrap().0.len())
        );
        waifu_task_free(task);

        // A task only waited on still has the whole at the end.
        let task = run(
            pipeline,
            a_speech_request(WAIFU_PIPELINE_STREAMING_TEXT_TO_SPEECH, c"hello there"),
        );
        assert_eq!(ended(task).0, WAIFU_OK);
        assert_eq!(
            audio(task, WAIFU_TASK_STREAMING_TEXT_TO_SPEECH_AUDIO).map(|(count, _)| count),
            Some(count)
        );
        // No part is read until its event is.
        assert_eq!(
            audio(task, WAIFU_TASK_STREAMING_TEXT_TO_SPEECH_AUDIO_PART),
            None
        );
        waifu_task_free(task);

        // Text to speech's request is another type's.
        assert!(run(
            pipeline,
            a_speech_request(WAIFU_PIPELINE_TEXT_TO_SPEECH, c"hello")
        )
        .is_null());
        waifu_pipeline_free(pipeline);
    }
}

#[test]
fn a_transcription_is_refused_at_the_call_or_ends_not_loaded() {
    unsafe {
        let pipeline = a_pipeline(WAIFU_PIPELINE_SPEECH_TO_TEXT);

        // No samples: refused at the call.
        let empty = WaifuAudio {
            samples: ptr::null(),
            count: 0,
            sample_rate: 16000,
        };
        let request = a_request(WAIFU_PIPELINE_SPEECH_TO_TEXT);
        assert!(waifu_request_set_audio_attribute(
            request,
            WAIFU_REQUEST_SPEECH_TO_TEXT_AUDIO,
            &empty
        ));
        assert!(waifu_pipeline_run(pipeline, request).is_null());
        assert!(last_error().contains("empty"), "{}", last_error());

        // No recording at all.
        assert!(waifu_request_set_audio_attribute(
            request,
            WAIFU_REQUEST_SPEECH_TO_TEXT_AUDIO,
            ptr::null()
        ));
        assert!(waifu_pipeline_run(pipeline, request).is_null());
        assert!(
            last_error().contains("SPEECH_TO_TEXT_AUDIO"),
            "{}",
            last_error()
        );

        // A rate of 0 is refused by the setter.
        let silent = WaifuAudio {
            sample_rate: 0,
            ..empty
        };
        assert!(!waifu_request_set_audio_attribute(
            request,
            WAIFU_REQUEST_SPEECH_TO_TEXT_AUDIO,
            &silent
        ));

        // The stand-in voice is not a transcriber.
        let tones = a_model("tones");
        assert!(waifu_pipeline_load_model(pipeline, tones).is_null());
        waifu_model_free(tones);

        // Nothing loaded: said at the end, with no text.
        let second = vec![0.0f32; 16000];
        let audio = WaifuAudio {
            samples: second.as_ptr(),
            count: second.len(),
            sample_rate: 16000,
        };
        assert!(waifu_request_set_audio_attribute(
            request,
            WAIFU_REQUEST_SPEECH_TO_TEXT_AUDIO,
            &audio
        ));
        set_string(request, WAIFU_REQUEST_SPEECH_TO_TEXT_LANGUAGE, c"English");
        set_string(request, WAIFU_REQUEST_SPEECH_TO_TEXT_CONTEXT, c"");
        let task = run(pipeline, request);
        assert!(!task.is_null(), "{}", last_error());
        assert_eq!(ended(task).0, WAIFU_ERR_NOT_LOADED);
        assert!(waifu_task_get_string_attribute(task, WAIFU_TASK_SPEECH_TO_TEXT_TEXT).is_null());
        waifu_task_free(task);
        waifu_pipeline_free(pipeline);
    }
}

#[test]
fn a_layer_decomposition_is_refused_at_the_call_or_ends_not_loaded() {
    unsafe {
        let pipeline = a_pipeline(WAIFU_PIPELINE_LAYER_DECOMPOSITION);

        // Pixels that are not the picture's size.
        let pixels = [0u8; 4 * 4 * 4];
        let short = WaifuRgbaImage {
            width: 5,
            height: 4,
            rgba: pixels.as_ptr(),
            len: pixels.len(),
        };
        let request = a_request(WAIFU_PIPELINE_LAYER_DECOMPOSITION);
        assert!(waifu_request_set_rgba_image_attribute(
            request,
            WAIFU_REQUEST_LAYER_DECOMPOSITION_IMAGE,
            &short
        ));
        set_int(
            request,
            WAIFU_REQUEST_LAYER_DECOMPOSITION_STRATIFY_LAYERS,
            1,
        );
        set_int(request, WAIFU_REQUEST_LAYER_DECOMPOSITION_SEED, 7);
        assert!(waifu_pipeline_run(pipeline, request).is_null());
        assert_eq!(waifu_last_error_code(), WAIFU_ERR_INVALID_ARGUMENT);

        let image = WaifuRgbaImage { width: 4, ..short };
        assert!(waifu_request_set_rgba_image_attribute(
            request,
            WAIFU_REQUEST_LAYER_DECOMPOSITION_IMAGE,
            &image
        ));
        let task = run(pipeline, request);
        assert!(!task.is_null(), "{}", last_error());
        let (code, message) = ended(task);
        assert_eq!(code, WAIFU_ERR_NOT_LOADED);
        assert!(message.is_some());
        // No layers, and an index is never read where there are none.
        let mut value = 0;
        assert!(!waifu_task_get_int_attribute(
            task,
            WAIFU_TASK_LAYER_DECOMPOSITION_LAYER_COUNT,
            &mut value
        ));
        assert!(!waifu_task_get_int_attribute_at(
            task,
            WAIFU_TASK_LAYER_DECOMPOSITION_LAYER_ID,
            0,
            &mut value
        ));
        // An indexed attribute asked without an index, and the other way round, is not there.
        assert!(!waifu_task_get_int_attribute(
            task,
            WAIFU_TASK_LAYER_DECOMPOSITION_LAYER_ID,
            &mut value
        ));
        assert!(!waifu_task_get_int_attribute_at(
            task,
            WAIFU_TASK_STATUS,
            0,
            &mut value
        ));
        waifu_task_free(task);
        waifu_pipeline_free(pipeline);
    }
}

unsafe extern "C" fn on_log(user_data: *mut c_void, log: *const WaifuLog) {
    let lines = &*(user_data as *const Mutex<Vec<(WaifuLogLevel, String)>>);
    let log = &*log;
    lines
        .lock()
        .unwrap()
        .push((log.level, text(log.message).unwrap_or_default()));
}

#[test]
fn a_log_callback_hears_the_library_s_lines() {
    let lines: Box<Mutex<Vec<(WaifuLogLevel, String)>>> = Box::default();
    waifu_set_log_callback(&*lines as *const _ as *mut c_void, Some(on_log));
    crate::log::write(crate::log::Level::Warning, "a line for the app");
    waifu_set_log_callback(ptr::null_mut(), None);

    let lines = lines.lock().unwrap();
    assert!(
        lines
            .iter()
            .any(|(level, line)| *level == WAIFU_LOG_WARNING && line == "a line for the app"),
        "{lines:?}"
    );
    // A level this library does not know changes nothing.
    waifu_set_log_level(99);
}

#[test]
fn a_memory_limit_is_for_a_device_that_has_one() {
    // The processor's allocator holds nothing back, so there is nothing for a limit to limit.
    assert!(!waifu_device_set_memory_limit(WAIFU_DEVICE_CPU, 1 << 30));
    assert_eq!(waifu_last_error_code(), WAIFU_ERR_INVALID_ARGUMENT);
    assert!(!waifu_device_set_memory_limit(77, 1 << 30));
    assert_eq!(waifu_last_error_code(), WAIFU_ERR_INVALID_ARGUMENT);

    // Every device with memory of its own has one, where the device is here: set, read back, and
    // lifted again so that no other test runs under it.
    for device in [WAIFU_DEVICE_METAL, WAIFU_DEVICE_CUDA, WAIFU_DEVICE_VULKAN] {
        let mut value = 0;
        if !waifu_device_is_available(device) {
            assert!(!waifu_device_set_memory_limit(device, 1 << 40));
            assert_eq!(waifu_last_error_code(), WAIFU_ERR_DEVICE_UNAVAILABLE);
            assert!(unsafe {
                !waifu_device_get_int_attribute(device, WAIFU_DEVICE_MEMORY_BUDGET, &mut value)
            });
            continue;
        }
        assert!(waifu_device_set_memory_limit(device, 1 << 40));
        unsafe {
            assert!(waifu_device_get_int_attribute(
                device,
                WAIFU_DEVICE_MEMORY_LIMIT,
                &mut value
            ));
            assert_eq!(value, 1 << 40);
            assert!(waifu_device_get_int_attribute(
                device,
                WAIFU_DEVICE_MEMORY_BUDGET,
                &mut value
            ));
            let budget = value;
            assert!(waifu_device_get_int_attribute(
                device,
                WAIFU_DEVICE_MEMORY_AVAILABLE,
                &mut value
            ));
            assert!((0..=budget).contains(&value));
        }
        assert!(waifu_device_set_memory_limit(device, 0));
        assert!(unsafe {
            !waifu_device_get_int_attribute(device, WAIFU_DEVICE_MEMORY_LIMIT, &mut value)
        });
    }
}

/// The int array `attribute` of `task`, or None.
unsafe fn ints(task: *const WaifuTask, attribute: WaifuTaskAttribute) -> Option<Vec<i32>> {
    let mut count = usize::MAX;
    let values = waifu_task_get_int_array_attribute(task, attribute, &mut count);
    match values.is_null() {
        true => {
            assert_eq!(
                count,
                usize::MAX,
                "*count is touched only where there is an array"
            );
            None
        }
        false => Some(std::slice::from_raw_parts(values, count).to_vec()),
    }
}

#[test]
fn a_voice_without_visemes_says_so_and_ends_a_reading_that_asks_for_them() {
    unsafe {
        let tones = a_model("tones");
        assert_eq!(
            int_attribute(tones, WAIFU_MODEL_SUPPORTS_TEXT_TO_SPEECH_VISEMES),
            Some(0)
        );
        let why = string_attribute(tones, WAIFU_MODEL_SUPPORTS_TEXT_TO_SPEECH_VISEMES);
        assert!(why.is_some_and(|why| !why.is_empty()));
        waifu_model_free(tones);

        for (r#type, attribute, visemes) in [
            (
                WAIFU_PIPELINE_TEXT_TO_SPEECH,
                WAIFU_REQUEST_TEXT_TO_SPEECH_VISEMES,
                WAIFU_TASK_TEXT_TO_SPEECH_VISEMES,
            ),
            (
                WAIFU_PIPELINE_STREAMING_TEXT_TO_SPEECH,
                WAIFU_REQUEST_STREAMING_TEXT_TO_SPEECH_VISEMES,
                WAIFU_TASK_STREAMING_TEXT_TO_SPEECH_VISEMES,
            ),
        ] {
            let pipeline = a_pipeline(r#type);
            with_tones(pipeline);

            // A boolean: 0 and 1, nothing else.
            let request = a_speech_request(r#type, c"hello there");
            assert!(!waifu_request_set_int_attribute(request, attribute, 2));
            assert_eq!(waifu_last_error_code(), WAIFU_ERR_INVALID_ARGUMENT);
            assert!(waifu_request_set_int_attribute(request, attribute, 1));
            let task = run(pipeline, request);
            assert!(!task.is_null(), "{}", last_error());
            let (code, message) = ended(task);
            assert_eq!(code, WAIFU_ERR_MODEL);
            assert!(message.unwrap().contains("mouth shape"));
            assert_eq!(ints(task, visemes), None);
            waifu_task_free(task);

            // Not asked for: the reading is as it was, with no visemes.
            let task = run(pipeline, a_speech_request(r#type, c"hello there"));
            assert_eq!(ended(task).0, WAIFU_OK);
            assert_eq!(ints(task, visemes), None);
            let mut ms = 0;
            assert!(!waifu_task_get_int_attribute(
                task,
                WAIFU_TASK_TEXT_TO_SPEECH_VISEME_FRAME_MS,
                &mut ms
            ));
            // An attribute that is not an array is not read as one.
            assert_eq!(ints(task, WAIFU_TASK_STATUS), None);
            waifu_task_free(task);

            waifu_pipeline_free(pipeline);
        }
        // Another pipeline type's attribute is refused by the setter.
        let request = a_request(WAIFU_PIPELINE_TEXT_TO_SPEECH);
        assert!(!waifu_request_set_int_attribute(
            request,
            WAIFU_REQUEST_STREAMING_TEXT_TO_SPEECH_VISEMES,
            1
        ));
        assert_eq!(waifu_last_error_code(), WAIFU_ERR_UNKNOWN_ATTRIBUTE);
        waifu_request_free(request);
    }
}

#[test]
fn a_streaming_reading_hands_over_the_visemes_its_parts_settle() {
    use crate::speech::Viseme;
    unsafe {
        let pipeline = a_pipeline(WAIFU_PIPELINE_STREAMING_TEXT_TO_SPEECH);
        let canceller = (*pipeline).engine.canceller();
        // What the engine does for a reading with visemes, without a voice that says them: the
        // visemes of a part, then the part, and the whole at the end.
        let started = task::Started::new();
        let (mut said, mut part) = (started.on_visemes(), started.on_part());
        let on_audio = started.on_audio();
        let task = started.task(0, canceller, Some(WAIFU_PIPELINE_STREAMING_TEXT_TO_SPEECH));
        let attribute = WAIFU_TASK_STREAMING_TEXT_TO_SPEECH_VISEMES;

        said(40, &[Viseme::A, Viseme::Mbp]);
        part(&Sound::new(vec![0.0; 1920], 24000));
        said(40, &[Viseme::Rest]);
        part(&Sound::new(vec![0.0; 960], 24000));
        // Nothing is there until a part's event is read.
        assert_eq!(ints(task, attribute), None);

        assert_eq!(waifu_task_next_event(task, 0), WAIFU_EVENT_AUDIO_PART);
        let mut count = 0;
        let first = waifu_task_get_int_array_attribute(task, attribute, &mut count);
        assert_eq!(std::slice::from_raw_parts(first, count), [0, 6]);
        let mut ms = 0;
        assert!(waifu_task_get_int_attribute(
            task,
            WAIFU_TASK_STREAMING_TEXT_TO_SPEECH_VISEME_FRAME_MS,
            &mut ms
        ));
        assert_eq!(ms, 40);
        // Not the other pipeline type's.
        assert!(!waifu_task_get_int_attribute(
            task,
            WAIFU_TASK_TEXT_TO_SPEECH_VISEME_FRAME_MS,
            &mut ms
        ));

        assert_eq!(waifu_task_next_event(task, 0), WAIFU_EVENT_AUDIO_PART);
        assert_eq!(ints(task, attribute), Some(vec![0, 6, 7]));
        // What the first part's read handed out is still there, and still what it was.
        assert_eq!(std::slice::from_raw_parts(first, 2), [0, 6]);

        said(40, &[Viseme::I]);
        on_audio(engine::Ended::Done(Sound::new(vec![0.0; 3000], 24000)));
        assert_eq!(waifu_task_next_event(task, 0), WAIFU_EVENT_END);
        // Every frame, the last after the last part, once the reading is over.
        assert_eq!(ints(task, attribute), Some(vec![0, 6, 7, 2]));
        assert_eq!(std::slice::from_raw_parts(first, 2), [0, 6]);
        waifu_task_free(task);
        waifu_pipeline_free(pipeline);
    }
}
