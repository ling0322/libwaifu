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

//! The C API, called the way C calls it: through the `extern "C"` functions, with callbacks that
//! are `extern "C"` functions and a `user_data` pointer -- here, to a channel the test waits on.

use super::*;

use std::ffi::CStr;
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::Mutex;
use std::time::Duration;

const WAIT: Duration = Duration::from_secs(20);

/// What a test hears of one job, sent from the callbacks.
#[derive(Debug, PartialEq)]
enum Heard {
    Progress(WaifuStage),
    Complete(WaifuStatusCode, Option<String>),
    Audio(WaifuStatusCode, usize, u32),
    /// A part of a streaming reading: its samples and its rate.
    Part(Vec<f32>, u32),
    Image(WaifuStatusCode, Option<(u32, u32, usize)>),
}

unsafe fn text(pointer: *const c_char) -> Option<String> {
    (!pointer.is_null()).then(|| CStr::from_ptr(pointer).to_string_lossy().into_owned())
}

unsafe fn sender<'a>(user_data: *mut c_void) -> &'a Sender<Heard> {
    &*(user_data as *const Sender<Heard>)
}

unsafe extern "C" fn on_progress(user_data: *mut c_void, progress: *const WaifuProgress) {
    let _ = sender(user_data).send(Heard::Progress((*progress).stage));
}

unsafe extern "C" fn on_complete(user_data: *mut c_void, result: *const WaifuResult) {
    let status = &(*result).status;
    let _ = sender(user_data).send(Heard::Complete(status.code, text(status.message)));
}

unsafe extern "C" fn on_audio(user_data: *mut c_void, result: *const WaifuAudioResult) {
    let result = &*result;
    let (count, rate) = result
        .audio
        .as_ref()
        .map_or((0, 0), |audio| (audio.count, audio.sample_rate));
    let _ = sender(user_data).send(Heard::Audio(result.status.code, count, rate));
}

unsafe extern "C" fn on_part(user_data: *mut c_void, part: *const WaifuAudio) {
    let part = &*part;
    let samples = std::slice::from_raw_parts(part.samples, part.count).to_vec();
    let _ = sender(user_data).send(Heard::Part(samples, part.sample_rate));
}

unsafe extern "C" fn on_image(user_data: *mut c_void, result: *const WaifuImageGenerationResult) {
    let result = &*result;
    let image = result
        .image
        .as_ref()
        .map(|image| (image.width, image.height, image.len));
    let _ = sender(user_data).send(Heard::Image(result.status.code, image));
}

unsafe extern "C" fn on_layers(
    user_data: *mut c_void,
    result: *const WaifuLayerDecompositionResult,
) {
    let status = &(*result).status;
    let _ = sender(user_data).send(Heard::Complete(status.code, text(status.message)));
}

unsafe extern "C" fn on_transcript(user_data: *mut c_void, result: *const WaifuSpeechToTextResult) {
    let result = &*result;
    // Text and language are both there or both NULL.
    assert_eq!(result.text.is_null(), result.language.is_null());
    let _ = sender(user_data).send(Heard::Complete(result.status.code, text(result.text)));
}

/// A channel whose sending end is never dropped, behind the pointer the callbacks get. The
/// engine's thread is still inside `Sender::send` when a test hears the end -- the message is
/// readable before `send` has notified the receiver -- so a test that returns there would free
/// the sender, and with it the channel, under a thread that is still writing to it. The next
/// channel is handed the same block, and the half-finished send lands on its waker instead.
fn a_channel() -> (&'static Sender<Heard>, Receiver<Heard>) {
    let (send, receive) = mpsc::channel();
    (Box::leak(Box::new(send)), receive)
}

fn user_data(sender: &Sender<Heard>) -> *mut c_void {
    sender as *const Sender<Heard> as *mut c_void
}

/// The first end a job comes to, skipping its progress.
fn the_end(heard: &Receiver<Heard>) -> Heard {
    loop {
        match heard.recv_timeout(WAIT).expect("an end") {
            Heard::Progress(..) => continue,
            end => return end,
        }
    }
}

fn last_error() -> String {
    unsafe { text(waifu_last_error()) }.unwrap_or_default()
}

/// The last refusal, as its code and its sentence.
fn refused() -> (WaifuStatusCode, String) {
    (waifu_last_error_code(), last_error())
}

fn a_speech_request(text: &CStr) -> WaifuTextToSpeechRequest {
    WaifuTextToSpeechRequest {
        struct_size: std::mem::size_of::<WaifuTextToSpeechRequest>() as u32,
        text: text.as_ptr(),
        reference: ptr::null(),
        speed: 1.0,
        temperature: 0.8,
        style: ptr::null(),
        seed: 7,
        reference_transcript: ptr::null(),
    }
}

fn an_image_request(prompt: &CStr) -> WaifuImageGenerationRequest {
    WaifuImageGenerationRequest {
        struct_size: std::mem::size_of::<WaifuImageGenerationRequest>() as u32,
        prompt: prompt.as_ptr(),
        negative: ptr::null(),
        width: 64,
        height: 64,
        steps: 1,
        guidance: 5.0,
        seed: 7,
        start_image: ptr::null(),
        start_image_strength: 0.8,
        control_image: ptr::null(),
        control_image_scale: 1.0,
        prompt_images: ptr::null(),
        prompt_image_count: 0,
    }
}

/// A pipeline for `task` on the processor.
fn a_pipeline(task: WaifuPipelineType) -> *mut WaifuPipeline {
    let pipeline = waifu_pipeline_new(task, WAIFU_DEVICE_CPU);
    assert!(!pipeline.is_null(), "{}", last_error());
    pipeline
}

fn a_model(path: &str) -> *mut WaifuModel {
    let path = CString::new(path).unwrap();
    let model = unsafe { waifu_model_open(path.as_ptr()) };
    assert!(!model.is_null(), "{}", last_error());
    model
}

/// `pipeline` with the stand-in voice loaded.
unsafe fn with_tones(pipeline: *mut WaifuPipeline) {
    let tones = a_model("tones");
    let (send, heard) = a_channel();
    let job =
        waifu_pipeline_load_model_async(pipeline, tones, user_data(&send), None, Some(on_complete));
    // The model is the caller's to free as soon as the call returns.
    waifu_model_free(tones);
    assert_ne!(job, 0, "{}", last_error());
    assert_eq!(the_end(&heard), Heard::Complete(WAIFU_OK, None));
}

unsafe fn int_property(model: *const WaifuModel, property: WaifuModelProperty) -> Option<i64> {
    let mut value = -7;
    waifu_model_get_int_property(model, property, &mut value).then_some(value)
}

unsafe fn string_property(
    model: *const WaifuModel,
    property: WaifuModelProperty,
) -> Option<String> {
    text(waifu_model_get_string_property(model, property))
}

#[test]
fn the_library_says_what_it_is() {
    assert_eq!(waifu_abi_version(), 5);
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
        assert!(!waifu_model_get_int_property(ptr::null(), 1, &mut value));
        assert!(!waifu_device_get_int_property(
            WAIFU_DEVICE_CPU,
            WAIFU_DEVICE_MEMORY_BUDGET,
            &mut value
        ));
        assert!(!waifu_device_is_available(99));
        assert_eq!(refused(), (code, message));

        assert!(waifu_model_open(ptr::null()).is_null());
        assert_eq!(waifu_last_error_code(), WAIFU_ERR_INVALID_ARGUMENT);
    })
    .join()
    .unwrap();
}

#[test]
fn a_model_is_asked_about_one_property_at_a_time() {
    unsafe {
        let sdxl = a_model(&crate::engine::tests::a_manifest("c-api-sdxl", "sdxl"));
        assert_eq!(
            int_property(sdxl, WAIFU_MODEL_SUPPORTS_IMAGE_GENERATION),
            Some(1)
        );
        assert_eq!(
            string_property(sdxl, WAIFU_MODEL_SUPPORTS_IMAGE_GENERATION),
            None
        );
        assert_eq!(
            int_property(sdxl, WAIFU_MODEL_IMAGE_GENERATION_WIDTH_HEIGHT_MULTIPLE_OF),
            Some(32)
        );
        assert!(string_property(sdxl, WAIFU_MODEL_IMAGE_GENERATION_SAMPLER).is_some());
        assert_eq!(
            int_property(sdxl, WAIFU_MODEL_SUPPORTS_IMAGE_GENERATION_FROM_START_IMAGE),
            Some(1)
        );
        // The weights the manifest names are not there.
        assert_eq!(int_property(sdxl, WAIFU_MODEL_SIZE_ON_DISK), None);
        assert_eq!(int_property(sdxl, WAIFU_MODEL_SAMPLE_RATE), None);

        // Every SUPPORTS_ it does not have is 0, with a reason, never NULL -- a task's details too.
        for property in [
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
            assert_eq!(int_property(sdxl, property), Some(0), "{property}");
            let why = string_property(sdxl, property).expect("a reason");
            assert!(!why.is_empty(), "{property}");
        }
        // Another task's constraints are not there at all.
        assert_eq!(
            int_property(sdxl, WAIFU_MODEL_LAYER_DECOMPOSITION_RESOLUTION_MULTIPLE_OF),
            None
        );
        // A number this library does not know is not there, and *value is left alone.
        let mut value = 42;
        assert!(!waifu_model_get_int_property(sdxl, 777, &mut value));
        assert_eq!(value, 42);
        assert!(waifu_model_get_string_property(sdxl, 777).is_null());
        waifu_model_free(sdxl);

        let tones = a_model("tones");
        assert_eq!(
            int_property(tones, WAIFU_MODEL_SUPPORTS_TEXT_TO_SPEECH),
            Some(1)
        );
        assert_eq!(
            int_property(tones, WAIFU_MODEL_SUPPORTS_STREAMING_TEXT_TO_SPEECH),
            Some(1)
        );
        assert!(int_property(tones, WAIFU_MODEL_SAMPLE_RATE).unwrap() > 0);
        assert_eq!(
            int_property(tones, WAIFU_MODEL_SUPPORTS_IMAGE_GENERATION),
            Some(0)
        );
        waifu_model_free(tones);
        waifu_model_free(ptr::null_mut());
    }
}

#[test]
fn a_pipeline_is_made_for_one_task_on_one_device() {
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
fn a_model_another_task_s_and_another_task_s_run_are_refused_at_the_call() {
    unsafe {
        let pipeline = a_pipeline(WAIFU_PIPELINE_TEXT_TO_SPEECH);
        let (send, heard) = a_channel();

        let sdxl = a_model(&crate::engine::tests::a_manifest("c-api-sdxl-tts", "sdxl"));
        let job = waifu_pipeline_load_model_async(
            pipeline,
            sdxl,
            user_data(&send),
            None,
            Some(on_complete),
        );
        waifu_model_free(sdxl);
        assert_eq!(job, 0);
        let (code, message) = refused();
        assert_eq!(code, WAIFU_ERR_INVALID_ARGUMENT);
        assert!(message.contains("text to speech"), "{message}");

        let request = an_image_request(c"a cat");
        let job = waifu_pipeline_run_image_generation_async(
            pipeline,
            &request,
            user_data(&send),
            None,
            Some(on_image),
        );
        assert_eq!(job, 0);
        let (code, message) = refused();
        assert_eq!(code, WAIFU_ERR_INVALID_ARGUMENT);
        assert!(message.contains("image generation"), "{message}");

        assert_eq!(
            waifu_pipeline_load_model_async(
                pipeline,
                ptr::null(),
                user_data(&send),
                None,
                Some(on_complete)
            ),
            0
        );

        waifu_pipeline_free(pipeline);
        // Nothing was ever called for any of them.
        assert!(heard.recv_timeout(Duration::from_millis(100)).is_err());
    }
}

#[test]
fn an_image_request_that_cannot_be_run_is_refused_at_the_call() {
    unsafe {
        let pipeline = a_pipeline(WAIFU_PIPELINE_IMAGE_GENERATION);
        let (send, heard) = a_channel();

        // A placeholder no image answers.
        let request = an_image_request(c"the girl in <|girl|>");
        let job = waifu_pipeline_run_image_generation_async(
            pipeline,
            &request,
            user_data(&send),
            Some(on_progress),
            Some(on_image),
        );
        assert_eq!(job, 0);
        assert!(last_error().contains("<|girl|>"), "{}", last_error());

        // A struct the caller says is smaller than this library's: compiled against another header.
        let short = WaifuImageGenerationRequest {
            struct_size: 8,
            ..an_image_request(c"a cat")
        };
        assert_eq!(
            waifu_pipeline_run_image_generation_async(
                pipeline,
                &short,
                user_data(&send),
                None,
                Some(on_image)
            ),
            0
        );
        assert!(last_error().contains("struct_size"), "{}", last_error());

        // A prompt image under the key the engine keeps for the start image.
        let pixels = [0u8; 4 * 4 * 3];
        let image = WaifuImage {
            width: 4,
            height: 4,
            rgb: pixels.as_ptr(),
            len: pixels.len(),
        };
        let prompt_image = WaifuPromptImage {
            key: c"start_from".as_ptr(),
            image: &image,
        };
        let keyed = WaifuImageGenerationRequest {
            prompt_images: &prompt_image,
            prompt_image_count: 1,
            ..an_image_request(c"a cat")
        };
        assert_eq!(
            waifu_pipeline_run_image_generation_async(
                pipeline,
                &keyed,
                user_data(&send),
                None,
                Some(on_image)
            ),
            0
        );
        assert!(last_error().contains("start_image"), "{}", last_error());
        assert_eq!(waifu_last_error_code(), WAIFU_ERR_INVALID_ARGUMENT);

        // A run with nothing loaded ends as not loaded, and not at the call.
        let request = an_image_request(c"a cat");
        assert_ne!(
            waifu_pipeline_run_image_generation_async(
                pipeline,
                &request,
                user_data(&send),
                None,
                Some(on_image)
            ),
            0
        );
        assert_eq!(the_end(&heard), Heard::Image(WAIFU_ERR_NOT_LOADED, None));

        waifu_pipeline_free(pipeline);
    }
}

#[test]
fn a_voice_is_loaded_and_speaks_through_the_c_api() {
    unsafe {
        let pipeline = a_pipeline(WAIFU_PIPELINE_TEXT_TO_SPEECH);
        with_tones(pipeline);

        let (send, heard) = a_channel();
        let request = a_speech_request(c"hello there");
        let job = waifu_pipeline_run_text_to_speech_async(
            pipeline,
            &request,
            user_data(&send),
            Some(on_progress),
            Some(on_audio),
        );
        assert_ne!(job, 0);

        // Progress before the end, in text to speech's stages, and the end with the samples.
        let first = heard.recv_timeout(WAIT).unwrap();
        assert_eq!(
            first,
            Heard::Progress(WAIFU_STAGE_TEXT_TO_SPEECH_READING_TEXT)
        );
        let mut stages = Vec::new();
        let end = loop {
            match heard.recv_timeout(WAIT).expect("an end") {
                Heard::Progress(stage) => stages.push(stage),
                end => break end,
            }
        };
        assert!(
            stages.iter().all(|stage| (200..300).contains(stage)),
            "{stages:?}"
        );
        match end {
            Heard::Audio(WAIFU_OK, count, rate) => {
                assert!(count > 0);
                assert!(rate > 0);
            }
            other => panic!("{other:?}"),
        }

        // An empty text is refused at the call.
        let empty = a_speech_request(c" ");
        assert_eq!(
            waifu_pipeline_run_text_to_speech_async(
                pipeline,
                &empty,
                user_data(&send),
                None,
                Some(on_audio)
            ),
            0
        );
        assert_eq!(waifu_last_error_code(), WAIFU_ERR_INVALID_ARGUMENT);

        // Unloaded, a run is not loaded at the end.
        let (send, heard) = a_channel();
        assert_ne!(
            waifu_pipeline_unload_model_async(pipeline, user_data(&send), Some(on_complete)),
            0
        );
        assert_eq!(the_end(&heard), Heard::Complete(WAIFU_OK, None));
        assert_ne!(
            waifu_pipeline_run_text_to_speech_async(
                pipeline,
                &request,
                user_data(&send),
                None,
                Some(on_audio)
            ),
            0
        );
        assert_eq!(the_end(&heard), Heard::Audio(WAIFU_ERR_NOT_LOADED, 0, 0));

        waifu_pipeline_free(pipeline);
    }
}

#[test]
fn freeing_a_pipeline_ends_every_job_it_had() {
    unsafe {
        let pipeline = a_pipeline(WAIFU_PIPELINE_TEXT_TO_SPEECH);
        let tones = a_model("tones");
        let (send, heard) = a_channel();
        waifu_pipeline_load_model_async(pipeline, tones, user_data(&send), None, Some(on_complete));
        waifu_model_free(tones);
        let request = a_speech_request(c"hello there");
        for _ in 0..3 {
            waifu_pipeline_run_text_to_speech_async(
                pipeline,
                &request,
                user_data(&send),
                None,
                Some(on_audio),
            );
        }
        waifu_pipeline_free(pipeline);

        // Four jobs, four ends, whatever each came to, all before the free returned.
        let ends = heard
            .try_iter()
            .filter(|heard| !matches!(heard, Heard::Progress(..)))
            .count();
        assert_eq!(ends, 4);
    }
}

/// WaifuTextToSpeechRequest as ABI 4 wrote it, before reference_transcript: what a caller built
/// against that header hands over, and all it hands over.
#[repr(C)]
struct Abi4TextToSpeechRequest {
    struct_size: u32,
    text: *const c_char,
    reference: *const WaifuAudio,
    speed: f32,
    temperature: f32,
    style: *const c_char,
    seed: u64,
}

#[test]
fn a_speech_request_reads_its_reference_transcript_and_refuses_an_older_size() {
    unsafe {
        let pipeline = a_pipeline(WAIFU_PIPELINE_TEXT_TO_SPEECH);
        with_tones(pipeline);
        let run = |request: *const WaifuTextToSpeechRequest| -> WaifuJob {
            let (send, _) = a_channel();
            waifu_pipeline_run_text_to_speech_async(
                pipeline,
                request,
                user_data(&send),
                None,
                Some(on_audio),
            )
        };

        // Bytes that are not UTF-8 in the field: it is read, and refused at the call.
        let request = WaifuTextToSpeechRequest {
            reference_transcript: c"\xff".as_ptr(),
            ..a_speech_request(c"hello there")
        };
        assert_eq!(run(&request), 0);
        assert!(last_error().contains("UTF-8"), "{}", last_error());

        // ABI 4's struct, smaller than this one, is refused at the call -- on its struct_size
        // alone, without reading past what the caller handed over.
        let older = Abi4TextToSpeechRequest {
            struct_size: std::mem::size_of::<Abi4TextToSpeechRequest>() as u32,
            text: c"hello there".as_ptr(),
            reference: ptr::null(),
            speed: 1.0,
            temperature: 0.8,
            style: ptr::null(),
            seed: 7,
        };
        assert!(
            std::mem::size_of::<Abi4TextToSpeechRequest>()
                < std::mem::size_of::<WaifuTextToSpeechRequest>()
        );
        let older = Box::new(older);
        assert_eq!(
            run(&*older as *const _ as *const WaifuTextToSpeechRequest),
            0
        );
        assert!(last_error().contains("struct_size"), "{}", last_error());

        // A NULL request is refused too, before anything is read.
        assert_eq!(run(ptr::null()), 0);
        assert!(last_error().contains("NULL"), "{}", last_error());
        waifu_pipeline_free(pipeline);
    }
}

#[test]
fn a_streaming_reading_hands_over_its_parts_before_the_whole() {
    unsafe {
        let pipeline = a_pipeline(WAIFU_PIPELINE_STREAMING_TEXT_TO_SPEECH);
        with_tones(pipeline);

        // Every part before the end, and the end the parts end to end.
        let (send, heard) = a_channel();
        let request = WaifuStreamingTextToSpeechRequest {
            struct_size: std::mem::size_of::<WaifuStreamingTextToSpeechRequest>() as u32,
            text: c"hello there".as_ptr(),
            reference: ptr::null(),
            temperature: 0.8,
            style: ptr::null(),
            seed: 7,
            reference_transcript: ptr::null(),
        };
        let job = waifu_pipeline_run_streaming_text_to_speech_async(
            pipeline,
            &request,
            user_data(&send),
            Some(on_progress),
            Some(on_part),
            Some(on_audio),
        );
        assert_ne!(job, 0);
        let mut parts = Vec::new();
        let mut stages = Vec::new();
        let end = loop {
            match heard.recv_timeout(WAIT).expect("an end") {
                Heard::Progress(stage) => stages.push(stage),
                Heard::Part(samples, rate) => parts.push((samples, rate)),
                end => break end,
            }
        };
        assert!(!parts.is_empty());
        assert!(
            stages.iter().all(|stage| (600..700).contains(stage)),
            "{stages:?}"
        );
        let count: usize = parts.iter().map(|(samples, _)| samples.len()).sum();
        match end {
            Heard::Audio(WAIFU_OK, whole, rate) => {
                assert_eq!(whole, count);
                assert!(parts.iter().all(|(_, part_rate)| *part_rate == rate));
            }
            other => panic!("{other:?}"),
        }

        // No part callback is no parts, and still the whole at the end.
        let (send, heard) = a_channel();
        waifu_pipeline_run_streaming_text_to_speech_async(
            pipeline,
            &request,
            user_data(&send),
            None,
            None,
            Some(on_audio),
        );
        assert!(matches!(
            the_end(&heard),
            Heard::Audio(WAIFU_OK, count, _) if count > 0
        ));

        // A request that is not one is refused at the call.
        assert_eq!(
            waifu_pipeline_run_streaming_text_to_speech_async(
                pipeline,
                ptr::null(),
                user_data(&send),
                None,
                Some(on_part),
                Some(on_audio)
            ),
            0
        );
        // Text to speech's run is another task's.
        assert_eq!(
            waifu_pipeline_run_text_to_speech_async(
                pipeline,
                &a_speech_request(c"hello"),
                user_data(&send),
                None,
                Some(on_audio)
            ),
            0
        );
        waifu_pipeline_free(pipeline);
    }
}

#[test]
fn a_transcription_is_refused_at_the_call_or_ends_not_loaded() {
    unsafe {
        let pipeline = a_pipeline(WAIFU_PIPELINE_SPEECH_TO_TEXT);
        let (send, heard) = a_channel();

        // No samples: refused at the call.
        let empty = WaifuAudio {
            samples: ptr::null(),
            count: 0,
            sample_rate: 16000,
        };
        let request = WaifuSpeechToTextRequest {
            struct_size: std::mem::size_of::<WaifuSpeechToTextRequest>() as u32,
            audio: &empty,
            language: ptr::null(),
            context: ptr::null(),
        };
        let job = waifu_pipeline_run_speech_to_text_async(
            pipeline,
            &request,
            user_data(&send),
            Some(on_progress),
            Some(on_transcript),
        );
        assert_eq!(job, 0);
        assert!(last_error().contains("empty"), "{}", last_error());

        // No recording at all.
        let null = WaifuSpeechToTextRequest {
            audio: ptr::null(),
            ..request
        };
        assert_eq!(
            waifu_pipeline_run_speech_to_text_async(pipeline, &null, user_data(&send), None, None),
            0
        );
        assert!(last_error().contains("audio"), "{}", last_error());
        assert!(heard.recv_timeout(Duration::from_millis(100)).is_err());

        // The stand-in voice is not a transcriber.
        let tones = a_model("tones");
        assert_eq!(
            waifu_pipeline_load_model_async(pipeline, tones, user_data(&send), None, None),
            0
        );
        waifu_model_free(tones);

        // Nothing loaded: said at the end, with no text.
        let second = vec![0.0f32; 16000];
        let audio = WaifuAudio {
            samples: second.as_ptr(),
            count: second.len(),
            sample_rate: 16000,
        };
        let request = WaifuSpeechToTextRequest {
            audio: &audio,
            language: c"English".as_ptr(),
            context: c"".as_ptr(),
            ..request
        };
        assert_ne!(
            waifu_pipeline_run_speech_to_text_async(
                pipeline,
                &request,
                user_data(&send),
                None,
                Some(on_transcript)
            ),
            0
        );
        assert_eq!(the_end(&heard), Heard::Complete(WAIFU_ERR_NOT_LOADED, None));

        waifu_pipeline_free(pipeline);
    }
}

#[test]
fn a_layer_decomposition_is_refused_at_the_call_or_ends_not_loaded() {
    unsafe {
        let pipeline = a_pipeline(WAIFU_PIPELINE_LAYER_DECOMPOSITION);
        let (send, heard) = a_channel();

        // Pixels that are not the picture's size.
        let pixels = [0u8; 4 * 4 * 4];
        let short = WaifuRgbaImage {
            width: 5,
            height: 4,
            rgba: pixels.as_ptr(),
            len: pixels.len(),
        };
        let request = WaifuLayerDecompositionRequest {
            struct_size: std::mem::size_of::<WaifuLayerDecompositionRequest>() as u32,
            image: &short,
            steps: 0,
            depth_steps: 0,
            resolution: 0,
            split_pairs: false,
            stratify_layers: true,
            seed: 7,
        };
        assert_eq!(
            waifu_pipeline_run_layer_decomposition_async(
                pipeline,
                &request,
                user_data(&send),
                None,
                Some(on_layers)
            ),
            0
        );
        assert_eq!(waifu_last_error_code(), WAIFU_ERR_INVALID_ARGUMENT);

        let image = WaifuRgbaImage { width: 4, ..short };
        let request = WaifuLayerDecompositionRequest {
            image: &image,
            ..request
        };
        assert_ne!(
            waifu_pipeline_run_layer_decomposition_async(
                pipeline,
                &request,
                user_data(&send),
                None,
                Some(on_layers)
            ),
            0
        );
        assert!(matches!(
            the_end(&heard),
            Heard::Complete(WAIFU_ERR_NOT_LOADED, Some(_))
        ));
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
                !waifu_device_get_int_property(device, WAIFU_DEVICE_MEMORY_BUDGET, &mut value)
            });
            continue;
        }
        assert!(waifu_device_set_memory_limit(device, 1 << 40));
        unsafe {
            assert!(waifu_device_get_int_property(
                device,
                WAIFU_DEVICE_MEMORY_LIMIT,
                &mut value
            ));
            assert_eq!(value, 1 << 40);
            assert!(waifu_device_get_int_property(
                device,
                WAIFU_DEVICE_MEMORY_BUDGET,
                &mut value
            ));
            let budget = value;
            assert!(waifu_device_get_int_property(
                device,
                WAIFU_DEVICE_MEMORY_AVAILABLE,
                &mut value
            ));
            assert!((0..=budget).contains(&value));
        }
        assert!(waifu_device_set_memory_limit(device, 0));
        assert!(unsafe {
            !waifu_device_get_int_property(device, WAIFU_DEVICE_MEMORY_LIMIT, &mut value)
        });
    }
}
