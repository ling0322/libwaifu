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

//! The binding over a built libwaifu, the one LIBWAIFU_PATH names, on the processor and the
//! stand-in voice: nothing to download.

use std::sync::{Arc, Mutex};

use waifu::{
    Capability, DecomposedImage, Device, ErrorCode, Event, ImageGeneration, Layer, LayerType,
    Model, Pipeline, RgbaImage, SpeechToText, Stage, StreamingTextToSpeech,
    StreamingTextToSpeechRequest, Support, TextToSpeech, TextToSpeechRequest,
};

fn ready() {
    waifu::init().expect("LIBWAIFU_PATH names a built libwaifu");
}

fn a_voice() -> Pipeline<TextToSpeech> {
    ready();
    let pipeline = Pipeline::<TextToSpeech>::new(Device::Cpu).unwrap();
    let tones = Model::open("tones").unwrap();
    waifu::block_on(pipeline.load_model(&tones).unwrap()).unwrap();
    pipeline
}

#[test]
fn the_library_is_found_once_and_says_what_it_is() {
    ready();
    assert!(waifu::is_initialized());
    let path = waifu::library_path().unwrap();
    assert!(path.is_absolute(), "{}", path.display());

    // Again, or by the path it was loaded from: nothing happens.
    waifu::init().unwrap();
    waifu::init_from_path(path).unwrap();
    // Another file: one process has one libwaifu.
    let another = std::env::current_exe().unwrap();
    assert_eq!(
        waifu::init_from_path(&another).unwrap_err().code,
        ErrorCode::Library
    );

    assert_eq!(waifu::ABI_VERSION, 5);
    assert!(!waifu::version().unwrap().is_empty());
    let _ = waifu::revision().unwrap();
}

#[test]
fn a_model_says_what_it_supports_and_why_not() {
    ready();
    let tones = Model::open("tones").unwrap();
    assert_eq!(tones.supports(Capability::TextToSpeech), Support::Yes);
    assert_eq!(
        tones.supports(Capability::StreamingTextToSpeech),
        Support::Yes
    );
    match tones.supports(Capability::ImageGeneration) {
        Support::No(why) => assert!(!why.is_empty()),
        other => panic!("{other:?}"),
    }
    assert!(tones.sample_rate().unwrap() > 0);
    assert_eq!(tones.image_generation_sampler(), None);
    assert!(tones.speech_to_text_languages().is_empty());

    let missing = Model::open("/no/such/model.yaml").unwrap_err();
    assert_eq!(missing.code, ErrorCode::UnknownModel);
    assert!(missing.message.contains("/no/such/model.yaml"), "{missing}");
}

#[test]
fn a_voice_speaks_awaited_or_walked() {
    let pipeline = a_voice();
    assert_eq!(pipeline.device(), Device::Cpu);
    let request = TextToSpeechRequest {
        seed: 7,
        ..TextToSpeechRequest::new("hello there")
    };

    // Awaited: the end alone.
    let audio = waifu::block_on(pipeline.run(&request).unwrap()).unwrap();
    assert!(!audio.samples.is_empty());
    assert!(audio.sample_rate > 0);

    // Walked: progress in text to speech's stages, then the end, then nothing.
    let mut job = pipeline.run(&request).unwrap();
    let (stages, end) = waifu::block_on(async {
        let mut stages = Vec::new();
        loop {
            match job.next().await.expect("an end") {
                Event::Progress(progress) => stages.push(progress.stage),
                Event::Part(never) => match never {},
                Event::Done(done) => break (stages, done),
            }
        }
    });
    assert_eq!(end.unwrap().samples.len(), audio.samples.len());
    assert!(!stages.is_empty());
    assert!(stages.iter().all(|stage| matches!(
        stage,
        Stage::TextToSpeechReadingText
            | Stage::TextToSpeechGeneratingTokens
            | Stage::TextToSpeechSynthesizingAudio
    )));
    assert!(waifu::block_on(job.next()).is_none());

    // Refused at the call: nothing to say.
    let refused = pipeline.run(&TextToSpeechRequest::new(" ")).unwrap_err();
    assert_eq!(refused.code, ErrorCode::InvalidArgument);
    // A NUL is refused before the library sees it.
    let refused = pipeline.run(&TextToSpeechRequest::new("a\0b")).unwrap_err();
    assert_eq!(refused.code, ErrorCode::InvalidArgument);
}

#[test]
fn a_streaming_reading_hands_over_its_parts_before_the_whole() {
    ready();
    let pipeline = Pipeline::<StreamingTextToSpeech>::new(Device::Cpu).unwrap();
    let tones = Model::open("tones").unwrap();
    waifu::block_on(pipeline.load_model(&tones).unwrap()).unwrap();
    drop(tones);

    let mut job = pipeline
        .run(&StreamingTextToSpeechRequest::new("hello there"))
        .unwrap();
    let (parts, whole) = waifu::block_on(async {
        let mut parts = Vec::new();
        loop {
            match job.next().await.expect("an end") {
                Event::Progress(_) => {}
                Event::Part(part) => parts.push(part),
                Event::Done(done) => break (parts, done.unwrap()),
            }
        }
    });
    assert!(!parts.is_empty());
    let count: usize = parts.iter().map(|part| part.samples.len()).sum();
    assert_eq!(count, whole.samples.len());
}

#[test]
fn a_model_another_task_s_is_refused_at_the_call() {
    ready();
    let pipeline = Pipeline::<SpeechToText>::new(Device::Cpu).unwrap();
    let tones = Model::open("tones").unwrap();
    let refused = pipeline.load_model(&tones).unwrap_err();
    assert_eq!(refused.code, ErrorCode::InvalidArgument);
    assert!(refused.message.contains("speech to text"), "{refused}");
}

#[test]
fn a_job_dropped_or_cancelled_is_cancelled_and_the_pipeline_still_works() {
    let pipeline = a_voice();
    let request = TextToSpeechRequest::new("hello there, and a little more to say");
    // Queued behind one another, the second dropped and the third cancelled before they run.
    let first = pipeline.run(&request).unwrap();
    drop(pipeline.run(&request).unwrap());
    let third = pipeline.run(&request).unwrap();
    third.cancel();
    assert!(waifu::block_on(first).is_ok());
    match waifu::block_on(third) {
        Err(error) => assert_eq!(error.code, ErrorCode::Cancelled),
        // It may have finished before the cancel reached it.
        Ok(audio) => assert!(!audio.samples.is_empty()),
    }

    // A clone is the same pipeline; dropped with work in it, every job still ends.
    let clone = pipeline.clone();
    let waiting = clone.run(&request).unwrap();
    drop(pipeline);
    drop(clone);
    assert!(waifu::block_on(waiting).is_ok());
}

#[test]
fn the_processor_takes_no_memory_limit() {
    ready();
    assert!(Device::Cpu.is_available().unwrap());
    assert_eq!(Device::Cpu.memory_budget().unwrap(), None);
    assert_eq!(
        Device::Cpu.set_memory_limit(1 << 30).unwrap_err().code,
        ErrorCode::InvalidArgument
    );
}

#[test]
fn a_log_callback_hears_the_library() {
    ready();
    let heard: Arc<Mutex<Vec<String>>> = Arc::default();
    let into = Arc::clone(&heard);
    waifu::log::set_callback(move |event| into.lock().unwrap().push(event.message.to_string()))
        .unwrap();

    // A manifest whose weights are not there: the load says what it reads, and then fails.
    let directory = std::env::temp_dir().join(format!("waifu-binding-{}", std::process::id()));
    std::fs::create_dir_all(&directory).unwrap();
    let manifest = directory.join("missing.yaml");
    std::fs::write(
        &manifest,
        "weights:\n  - missing.safetensors\nconfig:\n  model:\n    type: sdxl\n",
    )
    .unwrap();
    let model = Model::open(&manifest).unwrap();
    assert_eq!(model.supports(Capability::ImageGeneration), Support::Yes);
    assert_eq!(model.size_on_disk(), None);
    let pipeline = Pipeline::<ImageGeneration>::new(Device::Cpu).unwrap();
    let failed = waifu::block_on(pipeline.load_model(&model).unwrap()).unwrap_err();
    assert_eq!(failed.code, ErrorCode::Model);

    waifu::log::clear_callback().unwrap();
    let heard = heard.lock().unwrap();
    assert!(
        heard.iter().any(|line| line.contains("reading")),
        "{heard:?}"
    );
}

fn a_layer(layer_type: LayerType, id: u32, depth: f32, derived_from_id: Option<u32>) -> Layer {
    Layer {
        layer_type,
        id,
        name: String::new(),
        left: 0,
        top: 0,
        image: RgbaImage {
            width: 0,
            height: 0,
            rgba: Vec::new(),
        },
        depth,
        derived_from_id,
    }
}

#[test]
fn the_two_stacks_are_the_c_api_s() {
    // Back hair, a top, the back hair's halves -- the near one at the part's depth -- and a depth
    // layer for each.
    let decomposed = DecomposedImage {
        canvas: 64,
        layers: vec![
            a_layer(LayerType::Part, 1, 0.95, None),
            a_layer(LayerType::Part, 2, 0.6, None),
            a_layer(LayerType::StratifiedNear, 3, 0.95, Some(1)),
            a_layer(LayerType::StratifiedFar, 4, 0.95, Some(1)),
            a_layer(LayerType::Depth, 5, 0.95, Some(1)),
            a_layer(LayerType::Depth, 6, 0.6, Some(2)),
            a_layer(LayerType::Depth, 7, 0.95, Some(3)),
            a_layer(LayerType::Depth, 8, 0.95, Some(4)),
        ],
    };
    assert_eq!(decomposed.stack(), [0, 1]);
    // The split part is not in it, nor any depth layer -- the top's names it and does not split
    // it; at one depth, the far half under the near one.
    assert_eq!(decomposed.stratified_stack(), [3, 2, 1]);
}
