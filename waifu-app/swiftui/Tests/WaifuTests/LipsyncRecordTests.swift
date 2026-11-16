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

// A reading's mouth shapes as they are kept beside its WAV, and read back to be said again.

import Foundation
import WaifuKit
import XCTest

@testable import Waifu

final class LipsyncRecordTests: XCTestCase {
    private let clip = AudioClip(samples: [0, 0.5, -0.5], rate: 24000)

    func testAReadingsShapesComeBackAsTheyWereSaid() throws {
        let spoken = Spoken(
            clip: clip, visemes: Visemes(values: [7, 0, 6, 7], frameMilliseconds: 40),
            loudness: Loudness(values: [-100, -42.5, -12.25], frameMilliseconds: 10))
        let record = try XCTUnwrap(LipsyncRecord(spoken, audio: "a.wav"))
        let again = try XCTUnwrap(LipsyncRecord.read(try record.json()))
        XCTAssertEqual(again, record)
        let said = again.spoken(clip)
        XCTAssertEqual(said.visemes, spoken.visemes)
        XCTAssertEqual(said.loudness, spoken.loudness)
    }

    func testTheFileSaysWhatItsNumbersAre() throws {
        let spoken = Spoken(clip: clip, visemes: Visemes(values: [5], frameMilliseconds: 40), loudness: nil)
        let json = try XCTUnwrap(LipsyncRecord(spoken, audio: "a.wav")).json()
        let object = try XCTUnwrap(JSONSerialization.jsonObject(with: json) as? [String: Any])
        XCTAssertEqual(object["version"] as? Int, 1)
        XCTAssertEqual(object["audio"] as? String, "a.wav")
        XCTAssertNil(object["loudness"])
        let visemes = try XCTUnwrap(object["visemes"] as? [String: Any])
        XCTAssertEqual(visemes["frame_ms"] as? Int, 40)
        XCTAssertEqual(visemes["names"] as? [String], ["A", "E", "I", "O", "U", "@", "MBP", "REST"])
        XCTAssertEqual(visemes["values"] as? [Int], [5])
    }

    func testLoudnessIsKeptToAHundredthOfADecibel() throws {
        let spoken = Spoken(
            clip: clip, visemes: nil, loudness: Loudness(values: [-23.456789], frameMilliseconds: 10))
        let json = String(decoding: try XCTUnwrap(LipsyncRecord(spoken, audio: "a.wav")).json(), as: UTF8.self)
        XCTAssertTrue(json.contains("-23.46"), json)
        XCTAssertFalse(json.contains("-23.456"), json)
    }

    /// What a reading made before this run is said from: the whole of the WAV it was kept as.
    func testAKeptWavIsReadWhole() throws {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("waifu-lipsync-\(ProcessInfo.processInfo.processIdentifier).wav")
        defer { try? FileManager.default.removeItem(at: url) }
        let samples = (0..<48000).map { Float(sin(Double($0) / 10)) * 0.5 }
        try Media.wav(samples, rate: 24000).write(to: url)
        let clip = try Media.clip(from: url, seconds: .infinity)
        XCTAssertEqual(clip.rate, 24000)
        XCTAssertEqual(clip.samples.count, samples.count)
    }

    func testAReadingWithNeitherKeepsNoFile() {
        XCTAssertNil(LipsyncRecord(Spoken(clip: clip, visemes: nil, loudness: nil), audio: "a.wav"))
    }

    func testARecordOfAnotherVersionIsNotRead() {
        XCTAssertNil(LipsyncRecord.read(Data(#"{"version": 2, "audio": "a.wav"}"#.utf8)))
    }

    func testSavedElsewhereItNamesTheNewFile() throws {
        let spoken = Spoken(clip: clip, visemes: Visemes(values: [1], frameMilliseconds: 40), loudness: nil)
        let record = try XCTUnwrap(LipsyncRecord(spoken, audio: "a.wav"))
        XCTAssertEqual(record.renamed("b.wav").audio, "b.wav")
        XCTAssertEqual(record.renamed("b.wav").visemes, record.visemes)
    }
}
