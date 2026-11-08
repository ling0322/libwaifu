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

//! Before `init`: a process of its own, which nothing initializes.

use waifu::{Device, ErrorCode, Model, Pipeline, TextToSpeech};

#[test]
fn everything_that_reaches_the_library_is_an_error_before_init() {
    assert!(!waifu::is_initialized());
    assert_eq!(waifu::library_path(), None);
    assert_eq!(waifu::version().unwrap_err().code, ErrorCode::Library);
    assert_eq!(waifu::revision().unwrap_err().code, ErrorCode::Library);
    assert_eq!(
        Device::Cpu.is_available().unwrap_err().code,
        ErrorCode::Library
    );
    assert_eq!(Model::open("tones").unwrap_err().code, ErrorCode::Library);
    let refused = Pipeline::<TextToSpeech>::new(Device::Cpu).unwrap_err();
    assert_eq!(refused.code, ErrorCode::Library);
    assert!(refused.message.contains("init"), "{refused}");
    assert_eq!(
        waifu::log::set_level(waifu::LogLevel::Info)
            .unwrap_err()
            .code,
        ErrorCode::Library
    );

    // A file that is not libwaifu is not loaded, and leaves nothing loaded.
    let not_the_library = std::env::current_exe().unwrap();
    let error = waifu::init_from_path(&not_the_library).unwrap_err();
    assert_eq!(error.code, ErrorCode::Library);
    assert!(!waifu::is_initialized());
}
