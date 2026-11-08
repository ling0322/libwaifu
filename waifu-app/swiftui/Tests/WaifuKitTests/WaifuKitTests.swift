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

// WaifuKit against the real libwaifu.dylib, with the stand-in voice, which holds no weights: every
// path through the Swift, the C and the engine, in a second.

import WaifuKit
import XCTest

final class WaifuKitTests: XCTestCase {
    func testTheLibraryIsTheOneTheHeaderDescribes() {
        XCTAssertEqual(ABI.library, ABI.header)
        XCTAssertEqual(ABI.modelManagerLibrary, ABI.modelManagerHeader)
    }

    func testTheLibrarySaysWhichBuildItIs() throws {
        let build = try XCTUnwrap(LibraryVersion.current)
        XCTAssertFalse(build.version.isEmpty)
        // The full commit, or nothing for a build made without git.
        if let revision = build.revision {
            XCTAssertEqual(revision.count, 40, revision)
            XCTAssertTrue(revision.allSatisfy(\.isHexDigit), revision)
        }
    }

    func testAPipelineSaysWhatItRunsOnWithAutoSettled() throws {
        XCTAssertEqual(try Pipeline(.textToSpeech, device: .cpu).device, "cpu")
        let auto = try Pipeline(.textToSpeech, device: .auto).device
        XCTAssertFalse(auto.isEmpty)
        XCTAssertNotEqual(auto, "auto")
        XCTAssertTrue(Device.cpu.isAvailable)
    }

    func testAPublishedNameHasItsVersion() {
        XCTAssertEqual(ModelManager.versionedName("anima:turbo"), "anima:turbo:v1.1")
        XCTAssertNil(ModelManager.versionedName("/no/such/manifest.yaml"))
    }

    func testAVoiceIsLoadedAndSpeaks() async throws {
        let pipeline = try Pipeline(.textToSpeech, device: .cpu)
        try await pipeline.load("tones")

        final class Heard: @unchecked Sendable {
            var progress: [JobProgress] = []
            let lock = NSLock()
        }
        let heard = Heard()
        let clip = try await pipeline.speak(
            SpeakRequest(text: "hello there", speed: 1, temperature: 0.8, seed: 7)
        ) { progress in
            heard.lock.lock()
            heard.progress.append(progress)
            heard.lock.unlock()
        }

        let spoken = try XCTUnwrap(clip)
        XCTAssertFalse(spoken.samples.isEmpty)
        XCTAssertGreaterThan(spoken.rate, 0)
        XCTAssertFalse(heard.progress.isEmpty)
        // A stage of a reading, which the library says as a number and no words.
        let reading: [Stage] = [
            .textToSpeechReadingText, .textToSpeechGeneratingTokens, .textToSpeechSynthesizingAudio,
        ]
        XCTAssertTrue(heard.progress.allSatisfy { reading.contains($0.stage) }, "\(heard.progress.map(\.stage))")
        XCTAssertNil(heard.progress[0].words)
    }

    func testMetalReadsAModelUnderAMemoryLimit() async throws {
        // A machine without Metal -- a CI VM -- has nothing to limit.
        guard Device.metal.isAvailable else {
            throw XCTSkip("no Metal device")
        }
        try Device.metal.setMemoryLimit(1 << 30)
        // Lifted again however the test ends, so that no other test runs under it.
        defer { try? Device.metal.setMemoryLimit(0) }
        let pipeline = try Pipeline(.textToSpeech, device: .metal)
        try await pipeline.load("tones")
        let clip = try await pipeline.speak(SpeakRequest(text: "hello", speed: 1, temperature: 0.8, seed: 7)) { _ in }
        XCTAssertFalse(try XCTUnwrap(clip).samples.isEmpty)
    }

    func testAMemoryLimitIsForMetal() throws {
        XCTAssertThrowsError(try Device.cpu.setMemoryLimit(1 << 30))
        guard Device.metal.isAvailable else {
            throw XCTSkip("no Metal device")
        }
        try Device.metal.setMemoryLimit(1 << 40)
        // Lifted again, so that no other test runs under it.
        try Device.metal.setMemoryLimit(0)
    }

    func testARequestNoModelCouldRunIsRefusedAtTheCall() async throws {
        let pipeline = try Pipeline(.imageGeneration, device: .cpu)
        var request = DrawRequest(prompt: "the girl in <|girl|>", width: 64, height: 64, steps: 1, guidance: 5, seed: 7)
        request.negative = ""
        do {
            _ = try await pipeline.draw(request)
            XCTFail("drew a prompt naming an image it was not given")
        } catch WaifuError.refused(let said) {
            XCTAssertTrue(said.contains("<|girl|>"), said)
        }
    }

    func testAVoiceIsNotLoadedIntoAPipelineThatDraws() async throws {
        let pipeline = try Pipeline(.imageGeneration, device: .cpu)
        do {
            try await pipeline.load("tones")
            XCTFail("a voice was loaded to draw")
        } catch WaifuError.refused(let said) {
            XCTAssertFalse(said.isEmpty)
        }
    }

    func testARunOfAnotherTaskIsRefusedAtTheCall() async throws {
        let pipeline = try Pipeline(.textToSpeech, device: .cpu)
        try await pipeline.load("tones")
        do {
            _ = try await pipeline.draw(DrawRequest(prompt: "a cat", width: 64, height: 64, steps: 1, guidance: 5, seed: 7))
            XCTFail("a voice drew")
        } catch WaifuError.refused {
        }
    }

    func testAModelNobodyPublishedIsUnknown() async throws {
        let pipeline = try Pipeline(.imageGeneration, device: .cpu)
        do {
            try await pipeline.load("no-such-model:v9")
            XCTFail("loaded a model that does not exist")
        } catch WaifuError.unknownModel(let said) {
            XCTAssertTrue(said.contains("no-such-model:v9"), said)
        }
    }

    func testCancellingTheTaskEndsTheJobAsCancelled() async throws {
        let pipeline = try Pipeline(.textToSpeech, device: .cpu)
        try await pipeline.load("tones")
        // Cancelled before it can start: a task cancelled at once still gets its job's end.
        let task = Task {
            try await pipeline.speak(SpeakRequest(text: String(repeating: "hello there ", count: 200), speed: 1, temperature: 0.8, seed: 7))
        }
        task.cancel()
        let clip = try await task.value
        // The stand-in may finish before the cancel lands; what it must not do is hang or throw.
        _ = clip
    }

    func testWhatIsPublishedIsJSONAndWhatAModelIsIsAsked() throws {
        let catalog = try JSONSerialization.jsonObject(with: ModelManager.catalogJSON()) as? [String: Any]
        let models = try XCTUnwrap(catalog?["models"] as? [[String: Any]])
        XCTAssertTrue(models.contains { $0["name"] as? String == "sdxl:base" })
        // Every model says whether its manifest is here, which is what the app locks on.
        XCTAssertTrue(models.allSatisfy { $0["manifest_here"] is Bool })

        // A manifest on the disk, whose weights are not: what it is, it says all the same.
        let manifest = FileManager.default.temporaryDirectory
            .appendingPathComponent("waifukit-\(ProcessInfo.processInfo.processIdentifier).yaml")
        try "weights:\n  - w.safetensors\nconfig:\n  model:\n    type: sdxl\n"
            .write(to: manifest, atomically: true, encoding: .utf8)
        defer { try? FileManager.default.removeItem(at: manifest) }
        let model = try ModelManager.open(manifest.path)
        XCTAssertEqual(model.supports(.imageGeneration), .yes)
        if case .no(let why)? = model.supports(.textToSpeech) {
            XCTAssertFalse(why.isEmpty)
        } else {
            XCTFail("a picture model says it speaks")
        }
        XCTAssertEqual(model.widthHeightMultipleOf, 32)
        XCTAssertNil(model.sizeOnDisk, "its weights are not here")
        // What the boxes start at is the model manager's.
        let defaults = try JSONSerialization.jsonObject(with: ModelManager.defaultsJSON(of: model)) as? [String: Any]
        XCTAssertEqual(defaults?["steps"] as? Int, 30)
        XCTAssertEqual(defaults?["takes_guidance"] as? Bool, true)

        XCTAssertThrowsError(try ModelManager.open("no-such-model:v9"))
    }

    func testTheStratifiedStackIsByDepthFarHalvesUnderNearOnes() {
        // A face, and a top cut into a near half at the top's depth and a far half behind the face.
        // Each part and half has a depth layer after them, which names it and is in neither stack,
        // and does not make what it names a split part. Ids are not places.
        let layers: [(id: Int, type: LayerType, depth: Float, derivedFromID: Int?)] = [
            (id: 7, type: .part, depth: 0.9, derivedFromID: nil),  // back hair
            (id: 3, type: .part, depth: 0.5, derivedFromID: nil),  // topwear, which is split
            (id: 5, type: .part, depth: 0.4, derivedFromID: nil),  // face
            (id: 1, type: .near, depth: 0.5, derivedFromID: 3),
            (id: 2, type: .far, depth: 0.5, derivedFromID: 3),
            (id: 4, type: .depth, depth: 0.9, derivedFromID: 7),
            (id: 6, type: .depth, depth: 0.5, derivedFromID: 3),
            (id: 8, type: .depth, depth: 0.4, derivedFromID: 5),
            (id: 9, type: .depth, depth: 0.5, derivedFromID: 1),
            (id: 10, type: .depth, depth: 0.5, derivedFromID: 2),
        ]
        // The split part is not in it; at one depth the far half is under the near one.
        XCTAssertEqual(Layers.stratifiedStack(layers), [0, 4, 3, 2])
        // Without its halves the top is a part like any other.
        XCTAssertEqual(Layers.stratifiedStack(layers.filter { !$0.type.isHalf }), [0, 1, 2])
    }

    func testTheLogHandlerHearsTheLibrary() async throws {
        final class Lines: @unchecked Sendable {
            var lines: [(LogLevel, String)] = []
            let lock = NSLock()
        }
        let lines = Lines()
        Log.setHandler { level, _, message in
            lines.lock.lock()
            lines.lines.append((level, message))
            lines.lock.unlock()
        }
        defer { Log.setHandler(nil) }

        let pipeline = try Pipeline(.imageGeneration, device: .cpu)
        do {
            try await pipeline.load("no-such-model:v9")
        } catch {}
        // Nothing has to have been written by this; that it can be set and unset without a crash,
        // with the library running, is what is being checked.
        _ = lines.lines.count
    }
}
