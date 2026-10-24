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

// A voice as the text2speech tab reads it, from what the library says without reading any of it:
// no weights, no engine.

import Foundation
import XCTest

@testable import Waifu

final class VoiceTests: XCTestCase {
    /// A manifest of `type` with no weights beside it, which is enough to be described.
    private func manifest(_ stem: String, type: String) throws -> String {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("waifu-voice-\(ProcessInfo.processInfo.processIdentifier)-\(stem)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let path = directory.appendingPathComponent("\(stem).yaml")
        try "weights:\n  - \(stem).safetensors\nconfig:\n  model:\n    type: \(type)\n"
            .write(to: path, atomically: true, encoding: .utf8)
        return path.path
    }

    func testTheStandInSpeaksWithoutARecording() throws {
        let tones = try ModelInfo.describe("tones", device: "cpu")
        XCTAssertEqual(tones.kind, "speech")
        let voice = try XCTUnwrap(tones.voice)
        XCTAssertEqual(voice.needsARecording, false)
        XCTAssertNotNil(voice.notAVoiceBecause)
    }

    func testARealVoiceAsksForARecordingFirst() throws {
        let indextts = try ModelInfo.describe(manifest("indextts25", type: "indextts2"), device: "cpu")
        XCTAssertEqual(try XCTUnwrap(indextts.voice).needsARecording, true)
        XCTAssertNil(indextts.voice?.notAVoiceBecause)

        let cosyvoice = try ModelInfo.describe(manifest("cosyvoice3", type: "cosyvoice3"), device: "cpu")
        let voice = try XCTUnwrap(cosyvoice.voice)
        XCTAssertEqual(voice.needsARecording, true)
        XCTAssertFalse((voice.styles ?? []).isEmpty)
        // It converts as well, with the same model.
        XCTAssertNotNil(cosyvoice.converter)
    }
}
