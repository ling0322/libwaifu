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

// A picture's record as the app writes it, against docs/image-metadata.md: no model, no engine.

import Foundation
import WaifuKit
import XCTest

@testable import Waifu

final class ImageRecordTests: XCTestCase {
    private func picture(strength: Double? = nil, startFrom: String? = nil) -> Made {
        var made = Made.picture(
            seed: 15_603_977_047_213_861_476, model: "anima:turbo:v1.1",
            prompt: "masterpiece, best quality, 2girl", negative: "", width: 1216, height: 832,
            steps: 10, guidance: 1.0, strength: strength)
        made.seconds = 15.531
        made.device = "metal"
        made.startFrom = startFrom
        return made
    }

    private let output = URL(fileURLWithPath: "/Users/alice/Pictures/waifu/waifu-20261011-152219.png")
    // 2026-10-11T07:22:19.199Z
    private let finished = 1_791_703_339_199.0

    private func written(_ made: Made, library: LibraryVersion?) throws -> [String: Any] {
        let json = try ImageRecord(made: made, output: output, finished: finished, library: library).json()
        return try XCTUnwrap(JSONSerialization.jsonObject(with: json) as? [String: Any])
    }

    func testATxt2imgRecordIsTheDocumentsLayout() throws {
        let library = LibraryVersion(version: "0.1.0", revision: String(repeating: "3c", count: 20))
        let record = try written(picture(), library: library)

        XCTAssertEqual(Set(record.keys), ["environment", "request", "result"])
        let libwaifu = try XCTUnwrap((record["environment"] as? [String: Any])?["libwaifu"] as? [String: Any])
        XCTAssertEqual(libwaifu["version"] as? String, "0.1.0")
        XCTAssertEqual(libwaifu["revision"] as? String, library.revision)

        let request = try XCTUnwrap(record["request"] as? [String: Any])
        XCTAssertEqual(request["task"] as? String, "txt2img")
        let parameters = try XCTUnwrap(request["parameters"] as? [String: Any])
        XCTAssertEqual(
            Set(parameters.keys),
            ["model", "device", "prompt", "negative", "width", "height", "steps", "guidance", "seed", "inputs"])
        XCTAssertEqual(parameters["model"] as? String, "anima:turbo:v1.1")
        XCTAssertEqual(parameters["device"] as? String, "metal")
        XCTAssertEqual(parameters["negative"] as? String, "")
        XCTAssertEqual(parameters["width"] as? Int, 1216)
        XCTAssertEqual(parameters["steps"] as? Int, 10)
        XCTAssertEqual(parameters["guidance"] as? Double, 1.0)
        // A string: a u64 is past what a JSON number keeps.
        XCTAssertEqual(parameters["seed"] as? String, "15603977047213861476")
        XCTAssertEqual((parameters["inputs"] as? [String: Any])?.isEmpty, true)

        let result = try XCTUnwrap(record["result"] as? [String: Any])
        XCTAssertEqual(result["output"] as? String, output.path)
        XCTAssertEqual(result["created"] as? String, "2026-10-11T07:22:19.199Z")
        XCTAssertEqual(result["generation_seconds"] as? Double, 15.53)
    }

    func testAnImg2imgRecordHasItsStrengthAndInput() throws {
        let record = try written(picture(strength: 0.6, startFrom: "/Users/alice/Pictures/ref-0412.jpg"), library: nil)
        let request = try XCTUnwrap(record["request"] as? [String: Any])
        XCTAssertEqual(request["task"] as? String, "img2img")
        let parameters = try XCTUnwrap(request["parameters"] as? [String: Any])
        XCTAssertEqual(parameters["strength"] as? Double, 0.6)
        XCTAssertEqual(
            (parameters["inputs"] as? [String: String])?["start_from"], "/Users/alice/Pictures/ref-0412.jpg")
    }

    func testPicturesThePromptNamesAreItsInputsByName() throws {
        var made = picture()
        made.prompt = "put a small red wizard hat on <|cat|>"
        made.promptPictures = ["cat": "/Users/alice/Pictures/cat.png"]
        let record = try written(made, library: nil)
        let request = try XCTUnwrap(record["request"] as? [String: Any])
        // It starts from noise: the picture is read, not started from.
        XCTAssertEqual(request["task"] as? String, "txt2img")
        let parameters = try XCTUnwrap(request["parameters"] as? [String: Any])
        XCTAssertEqual(parameters["inputs"] as? [String: String], ["cat": "/Users/alice/Pictures/cat.png"])
        XCTAssertNil(parameters["strength"])

        let json = try ImageRecord(made: made, output: output, finished: finished, library: nil).json()
        let back = try XCTUnwrap(try JSONDecoder().decode(ImageRecord.self, from: json).made)
        XCTAssertEqual(back.promptPictures, ["cat": "/Users/alice/Pictures/cat.png"])
        XCTAssertNil(back.startFrom)
    }

    @MainActor
    func testTheNamesAPromptWritesAreReadTheWayTheEngineReadsThem() {
        XCTAssertEqual(
            Session.placeholders(in: "<|b|> beside <|a|>, and <|b|> again"), ["b", "a"])
        XCTAssertEqual(Session.placeholders(in: "no pictures at all"), [])
        XCTAssertEqual(Session.placeholders(in: "an open <|end"), [])
    }

    @MainActor
    func testPicturesAreNamedImage0OnTheFirstNumberFree() {
        XCTAssertEqual(Session.nextName(after: []), "image_0")
        XCTAssertEqual(Session.nextName(after: ["image_0", "image_1"]), "image_2")
        // image_0 taken away: the next picture is image_0 again.
        XCTAssertEqual(Session.nextName(after: ["image_1", "image_2"]), "image_0")
    }

    @MainActor
    func testAPictureIsReadAtAMegapixelOfItsOwnShape() {
        // `QwenImage::picture_size`'s own cases.
        XCTAssertEqual(Session.readSize(width: 512, height: 512).0, 1024)
        XCTAssertTrue(Session.readSize(width: 4000, height: 3000) == (1184, 896))
        XCTAssertTrue(Session.readSize(width: 1024, height: 704) == (1248, 864))
        XCTAssertTrue(Session.readSize(width: 1920, height: 1080) == (1376, 768))
    }

    func testARevisionNobodyKnowsIsNullNotMissing() throws {
        let made = picture()
        let json = try ImageRecord(
            made: made, output: output, finished: finished,
            library: LibraryVersion(version: "0.1.0", revision: nil)
        ).json()
        let text = try XCTUnwrap(String(data: json, encoding: .utf8))
        XCTAssertTrue(text.contains("\"revision\" : null"), text)
        // Paths as they are, not with every slash escaped.
        XCTAssertTrue(text.contains("\"/Users/alice/Pictures/waifu/"), text)
        // In the order the document lists them.
        let order = ["\"environment\"", "\"request\"", "\"result\""].compactMap { text.range(of: $0)?.lowerBound }
        XCTAssertEqual(order, order.sorted())
        XCTAssertEqual(order.count, 3)
    }

    func testARecordReadsBackAsWhatMadeIt() throws {
        let made = picture(strength: 0.6, startFrom: "/tmp/in.png")
        let json = try ImageRecord(made: made, output: output, finished: finished, library: nil).json()
        let record = try JSONDecoder().decode(ImageRecord.self, from: json)
        let back = try XCTUnwrap(record.made)
        XCTAssertEqual(back.seed, made.seed)
        XCTAssertEqual(back.prompt, made.prompt)
        XCTAssertEqual(back.width, 1216)
        XCTAssertEqual(back.strength, 0.6)
        XCTAssertEqual(back.device, "metal")
        XCTAssertEqual(back.startFrom, "/tmp/in.png")
        XCTAssertEqual(back.parameters, made.parameters)
        XCTAssertEqual(record.finished, finished)
        // A record from before these is not taken for one.
        let old = #"{"id":"x","kind":"image","created":1,"finished":2,"made":{"seed":1,"seconds":1,"model":"m","parameters":"p"}}"#
        XCTAssertNil(try? JSONDecoder().decode(ImageRecord.self, from: Data(old.utf8)))
    }
}
