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

// A reading's mouth shapes: the viseme model attached to a voice, and the shapes it says, one for
// every frame of the audio, whole and streamed. With the stand-in voice, which takes no viseme
// model, the refusals; with a real one, where LIBWAIFU_VISEME_VOICE names it, the shapes.

import AVFoundation
import WaifuKit
import XCTest

final class VisemeTests: XCTestCase {
    func testAVisemeLooksUpItsFrame() {
        let v = Visemes(values: [0, 2, 7], frameMilliseconds: 40)
        XCTAssertEqual(v.at(0), 0)
        XCTAssertEqual(v.at(0.039), 0)
        XCTAssertEqual(v.at(0.04), 2)
        XCTAssertEqual(v.at(0.119), 7)
        XCTAssertNil(v.at(0.12))
        XCTAssertNil(v.at(-0.01))
    }

    /// Any voice says how loud it is, with no model attached: a frame each 10 ms of its audio.
    func testAReadingSaysHowLoudItIs() async throws {
        let pipeline = try Pipeline(.textToSpeech, device: .cpu)
        try await pipeline.load("tones")
        let read = try await pipeline.speakReading(
            SpeakRequest(text: "hello", speed: 1, temperature: 0.8, seed: 1, loudness: true))
        let reading = try XCTUnwrap(read)
        let loudness = try XCTUnwrap(reading.loudness)
        XCTAssertNil(reading.visemes)
        XCTAssertEqual(loudness.frameMilliseconds, 10)
        let ms = Double(reading.audio.samples.count) / Double(reading.audio.rate) * 1000
        XCTAssertEqual(loudness.values.count, Int((ms / 10).rounded(.up)))
        XCTAssertTrue(loudness.values.allSatisfy { $0 >= -100 && $0 <= 0 })
        XCTAssertNotNil(loudness.at(0))
        XCTAssertNil(loudness.at(ms / 1000 + 0.01))

        // Not asked for, not there.
        let plain = try await pipeline.speakReading(SpeakRequest(text: "hello", speed: 1, temperature: 0.8, seed: 1))
        XCTAssertNil(try XCTUnwrap(plain).loudness)
    }

    func testAVoiceThatTakesNoVisemeModelSaysSo() throws {
        let tones = try Model(manifest: "tones")
        XCTAssertFalse(tones.does(.attachedTextToSpeechVisemes))
    }

    func testTheVisemeModelIsNotAttachedToAVoiceThatTakesNone() async throws {
        let pipeline = try Pipeline(.textToSpeech, device: .cpu)
        try await pipeline.load("tones")
        // Asked for without one attached, the reading fails rather than saying none.
        do {
            _ = try await pipeline.speakReading(
                SpeakRequest(text: "hello", speed: 1, temperature: 0.8, seed: 1, visemes: true))
            XCTFail("a reading asked for visemes with no viseme model attached")
        } catch is CancellationError {
            XCTFail("cancelled")
        } catch {}
        XCTAssertNil(pipeline.attached(as: .textToSpeechVisemes))
    }

    /// LIBWAIFU_VISEME_VOICE -- `cosyvoice` -- and LIBWAIFU_VISEME_REFERENCE, a recording of
    /// somebody speaking, with LIBWAIFU_VISEME_TRANSCRIPT, what it says. On Metal; slow.
    func testAReadingSaysItsMouthShapes() async throws {
        let env = ProcessInfo.processInfo.environment
        guard let voice = env["LIBWAIFU_VISEME_VOICE"], let referencePath = env["LIBWAIFU_VISEME_REFERENCE"],
            let transcript = env["LIBWAIFU_VISEME_TRANSCRIPT"]
        else {
            throw XCTSkip("set LIBWAIFU_VISEME_VOICE, LIBWAIFU_VISEME_REFERENCE and LIBWAIFU_VISEME_TRANSCRIPT")
        }
        let reference = try clip(URL(fileURLWithPath: referencePath))
        let text = env["LIBWAIFU_VISEME_TEXT"] ?? "こんにちは、今日はいい天気ですね。"
        let request = SpeakRequest(
            text: text, reference: reference, referenceTranscript: transcript, speed: 1, temperature: 0.8,
            seed: 3, visemes: true)

        // Whole.
        let whole = try Pipeline(.textToSpeech, device: .metal)
        try await whole.load(voice)
        try await whole.attach("cosyvoice3-visemes:v1", as: .textToSpeechVisemes)
        XCTAssertEqual(whole.attached(as: .textToSpeechVisemes), "cosyvoice3-visemes:v1")
        let read = try await whole.speakReading(request)
        let reading = try XCTUnwrap(read)
        let visemes = try XCTUnwrap(reading.visemes)
        check(visemes, against: reading.audio)
        print("whole:", visemes.values.count, "frames of", visemes.frameMilliseconds, "ms:", visemes.values)
        // With LIBWAIFU_VISEME_DUMP, a directory: the reading's samples and shapes, to study.
        if let dump = env["LIBWAIFU_VISEME_DUMP"] {
            let out = URL(fileURLWithPath: dump)
            let samples = reading.audio.samples.withUnsafeBufferPointer { Data(buffer: $0) }
            try samples.write(to: out.appendingPathComponent("audio.f32"))
            let said: [String: Any] = [
                "rate": reading.audio.rate, "frame_ms": visemes.frameMilliseconds,
                "visemes": visemes.values.map(Int.init),
            ]
            try JSONSerialization.data(withJSONObject: said).write(to: out.appendingPathComponent("visemes.json"))
        }

        // A load lets what was attached go.
        try await whole.load(voice)
        XCTAssertNil(whole.attached(as: .textToSpeechVisemes))

        // Streamed: the shapes settled after each part never run past the audio handed over.
        let streaming = try Pipeline(.streamingTextToSpeech, device: .metal)
        try await streaming.load(voice)
        try await streaming.attach("cosyvoice3-visemes:v1", as: .textToSpeechVisemes)
        final class Heard: @unchecked Sendable {
            var samples = 0
            var rate = 0
            var settled: [Visemes] = []
            var ahead = false
            let lock = NSLock()
        }
        let heard = Heard()
        let audio = try await streaming.speakStreaming(
            request,
            heard: { part in
                heard.lock.lock()
                heard.samples += part.samples.count
                heard.rate = part.rate
                heard.lock.unlock()
            },
            settled: { so in
                heard.lock.lock()
                let seconds = Double(heard.samples) / Double(max(heard.rate, 1))
                if Double(so.values.count * so.frameMilliseconds) > (seconds * 1000).rounded(.up) + Double(so.frameMilliseconds) {
                    heard.ahead = true
                }
                heard.settled.append(so)
                heard.lock.unlock()
            })
        let all = try XCTUnwrap(audio)
        let (ahead, settled) = heard.lock.withLock { (heard.ahead, heard.settled) }
        XCTAssertFalse(ahead, "shapes settled past the audio handed over")
        XCTAssertGreaterThan(settled.count, 1)
        let last = try XCTUnwrap(settled.last)
        check(last, against: all)
        // Each part's shapes are the ones before and more.
        for (a, b) in zip(settled, settled.dropFirst()) {
            XCTAssertEqual(Array(b.values.prefix(a.values.count)), a.values)
        }
        print("streamed:", last.values.count, "frames in", settled.count, "settlings")
    }

    /// A shape a frame of `audio`, each one of the eight.
    private func check(_ visemes: Visemes, against audio: AudioClip, file: StaticString = #filePath, line: UInt = #line) {
        let ms = Double(audio.samples.count) / Double(audio.rate) * 1000
        let frames = Int((ms / Double(visemes.frameMilliseconds)).rounded(.up))
        XCTAssertEqual(visemes.frameMilliseconds, 40, file: file, line: line)
        XCTAssertEqual(visemes.values.count, frames, accuracy: 1, file: file, line: line)
        XCTAssertTrue(visemes.values.allSatisfy { (0...7).contains($0) }, file: file, line: line)
        // A reading opens its mouth: not every frame at rest.
        XCTAssertTrue(visemes.values.contains { $0 != 7 }, file: file, line: line)
    }

    private func clip(_ url: URL) throws -> AudioClip {
        let file = try AVAudioFile(forReading: url)
        let format = try XCTUnwrap(
            AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: file.fileFormat.sampleRate, channels: 1, interleaved: false))
        let buffer = try XCTUnwrap(AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(file.length)))
        let converter = try XCTUnwrap(AVAudioConverter(from: file.processingFormat, to: format))
        let source = try XCTUnwrap(AVAudioPCMBuffer(pcmFormat: file.processingFormat, frameCapacity: AVAudioFrameCount(file.length)))
        try file.read(into: source)
        try converter.convert(to: buffer, from: source)
        let samples = Array(UnsafeBufferPointer(start: buffer.floatChannelData![0], count: Int(buffer.frameLength)))
        return AudioClip(samples: samples, rate: Int(format.sampleRate))
    }
}
