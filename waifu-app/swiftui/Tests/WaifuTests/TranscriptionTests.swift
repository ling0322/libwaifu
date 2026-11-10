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

// What speech2text does with a recording before any model hears it: one too long is refused, not
// cut short.

import Foundation
import XCTest

@testable import Waifu

final class TranscriptionTests: XCTestCase {
    /// A folder of this test's own, empty.
    private func folder(_ name: String) throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("waifu-transcription-\(ProcessInfo.processInfo.processIdentifier)-\(name)")
        try? FileManager.default.removeItem(at: directory)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        return directory
    }

    /// Three seconds of a tone at 16 kHz, as a WAV file.
    private func threeSeconds() throws -> URL {
        let samples = (0..<48_000).map { Float(sin(Double($0) * 0.1) * 0.5) }
        let url = try folder("tone").appendingPathComponent("tone.wav")
        try Media.wav(samples, rate: 16_000).write(to: url)
        return url
    }

    func testARecordingLongerThanTheLimitIsRefusedNotCutShort() throws {
        let url = try threeSeconds()
        XCTAssertThrowsError(try Media.clip(from: url, seconds: 2, whole: true)) { error in
            let said = error.localizedDescription
            XCTAssertTrue(said.contains("tone.wav is 0:03 long"), said)
            XCTAssertTrue(said.contains("up to 0:02"), said)
        }
        // Within the limit, all of it.
        XCTAssertEqual(try Media.clip(from: url, seconds: 5, whole: true).samples.count, 48_000)
        // A recording to sound like is the first of it, as it was: a sentence is enough.
        XCTAssertEqual(try Media.clip(from: url, seconds: 2).samples.count, 32_000)
    }
}
