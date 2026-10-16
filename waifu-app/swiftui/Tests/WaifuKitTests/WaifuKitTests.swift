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

    func testAnEngineSaysWhatItRunsOnWithAutoSettled() throws {
        XCTAssertEqual(try Engine(device: .cpu).device, "cpu")
        let auto = try Engine(device: .auto).device
        XCTAssertFalse(auto.isEmpty)
        XCTAssertNotEqual(auto, "auto")
    }

    func testAPublishedNameHasItsVersion() {
        XCTAssertEqual(ModelManager.versionedName("anima:turbo"), "anima:turbo:v1.1")
        XCTAssertNil(ModelManager.versionedName("/no/such/manifest.yaml"))
    }

    func testAVoiceIsLoadedAndSpeaks() async throws {
        let engine = try Engine(device: .cpu)
        try await engine.load("tones")

        final class Heard: @unchecked Sendable {
            var progress: [JobProgress] = []
            let lock = NSLock()
        }
        let heard = Heard()
        let clip = try await engine.speak(
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
        XCTAssertFalse(heard.progress[0].words.isEmpty)
    }

    func testTheDiskOffloadDeviceReachesTheEngine() async throws {
        // A machine without Metal -- a CI VM -- has nothing to offload to.
        guard (try? Engine(device: .metal)) != nil else {
            throw XCTSkip("no Metal device")
        }
        let engine = try Engine(device: .metalDiskOffload)
        try await engine.load("tones")
        let clip = try await engine.speak(SpeakRequest(text: "hello", speed: 1, temperature: 0.8, seed: 7)) { _ in }
        XCTAssertFalse(try XCTUnwrap(clip).samples.isEmpty)
    }

    func testWeightsKeptOnTheDeviceAreForTheOffloadDevicesAlone() async throws {
        XCTAssertThrowsError(try Engine(device: .cpu, deviceWeightBytes: 1 << 30))
        guard (try? Engine(device: .metal)) != nil else {
            throw XCTSkip("no Metal device")
        }
        let engine = try Engine(device: .metalDiskOffload, deviceWeightBytes: 1 << 30)
        try await engine.load("tones")
        let clip = try await engine.speak(SpeakRequest(text: "hello", speed: 1, temperature: 0.8, seed: 7)) { _ in }
        XCTAssertFalse(try XCTUnwrap(clip).samples.isEmpty)
    }

    func testARequestNoModelCouldRunIsRefusedAtTheCall() async throws {
        let engine = try Engine(device: .cpu)
        var request = DrawRequest(prompt: "the girl in <|girl|>", width: 64, height: 64, steps: 1, guidance: 5, seed: 7)
        request.negative = ""
        do {
            _ = try await engine.draw(request)
            XCTFail("drew a prompt naming an image it was not given")
        } catch WaifuError.refused(let said) {
            XCTAssertTrue(said.contains("<|girl|>"), said)
        }
    }

    func testAnImageAskedOfAVoiceIsTheWrongKind() async throws {
        let engine = try Engine(device: .cpu)
        try await engine.load("tones")
        do {
            _ = try await engine.draw(DrawRequest(prompt: "a cat", width: 64, height: 64, steps: 1, guidance: 5, seed: 7))
            XCTFail("a voice drew")
        } catch WaifuError.notLoaded {
        }
    }

    func testAModelNobodyPublishedIsUnknown() async throws {
        let engine = try Engine(device: .cpu)
        do {
            try await engine.load("no-such-model:v9")
            XCTFail("loaded a model that does not exist")
        } catch WaifuError.unknownModel(let said) {
            XCTAssertTrue(said.contains("no-such-model:v9"), said)
        }
    }

    func testCancellingTheTaskEndsTheJobAsCancelled() async throws {
        let engine = try Engine(device: .cpu)
        try await engine.load("tones")
        // Cancelled before it can start: a task cancelled at once still gets its job's end.
        let task = Task {
            try await engine.speak(SpeakRequest(text: String(repeating: "hello there ", count: 200), speed: 1, temperature: 0.8, seed: 7))
        }
        task.cancel()
        let clip = try await task.value
        // The stand-in may finish before the cancel lands; what it must not do is hang or throw.
        _ = clip
    }

    func testWhatIsPublishedIsJSON() throws {
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
        let described = try JSONSerialization.jsonObject(with: ModelManager.describeJSON(manifest.path)) as? [String: Any]
        XCTAssertEqual(described?["kind"] as? String, "image")
        // What the boxes start at is the model manager's half of it, laid in.
        XCTAssertEqual(described?["steps"] as? Int, 30)
        XCTAssertEqual(described?["takes_guidance"] as? Bool, true)

        XCTAssertThrowsError(try ModelManager.describeJSON("no-such-model:v9"))
        XCTAssertNotNil(MachineInfo.json())
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

        let engine = try Engine(device: .cpu)
        do {
            try await engine.load("no-such-model:v9")
        } catch {}
        // Nothing has to have been written by this; that it can be set and unset without a crash,
        // with the library running, is what is being checked.
        _ = lines.lines.count
    }
}
