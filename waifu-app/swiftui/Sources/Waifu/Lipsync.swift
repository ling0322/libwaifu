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

// The reading being heard, for the drawing on the text-to-speech page to say: how loud it is --
// libwaifu's loudness, or its sound to hear that from -- the mouth shapes its voice said, and a
// clock for where in it the sound is. Whatever plays a reading -- the stream as it is said, or a
// clip from the output -- hands them over; the drawing takes what is new each frame it draws and
// hands it to libwaifu2d, which opens the mouth as the sound is loud, in the shape the voice said.

import Foundation
import os
import WaifuKit

/// A reading kept to be played again with the drawing's mouth moving: its sound, the shapes its
/// voice said, if it said them, and how loud it is.
struct Spoken: Sendable {
    var clip: AudioClip
    var visemes: Visemes?
    var loudness: Loudness?
}

/// A reading's mouth shapes and loudness, kept beside its WAV as `<name>.visemes.json` so that the
/// drawing says it again when it is played later, from the output or wherever it was saved:
///
///     {
///       "version": 1,
///       "audio": "waifu-20261115-101500.wav",
///       "visemes": { "frame_ms": 40, "names": ["A", "E", "I", "O", "U", "@", "MBP", "REST"],
///                    "values": [7, 7, 0, 6, ...] },
///       "loudness": { "frame_ms": 10, "unit": "dBFS", "values": [-100, -42.5, ...] }
///     }
///
/// `visemes.values` are `WaifuViseme`s, frame i the milliseconds [i, i + 1) * frame_ms of the
/// audio, and `names` spells them in that order. `loudness` is libwaifu's, ABI 6.2, to 0.01 dB.
/// Either is left out where the reading had none.
struct LipsyncRecord: Codable, Equatable {
    /// What the file beside a reading's WAV ends in, for `<name>.wav`: `<name>.visemes.json`.
    static let suffix = ".visemes.json"
    /// `WaifuViseme`, 0 to 7.
    static let visemeNames = ["A", "E", "I", "O", "U", "@", "MBP", "REST"]

    struct VisemeTrack: Codable, Equatable {
        var frameMs: Int
        var names: [String]
        var values: [Int32]

        enum CodingKeys: String, CodingKey {
            case frameMs = "frame_ms"
            case names, values
        }
    }

    struct LoudnessTrack: Codable, Equatable {
        var frameMs: Int
        var unit: String
        var values: [Float]

        enum CodingKeys: String, CodingKey {
            case frameMs = "frame_ms"
            case unit, values
        }
    }

    var version = 1
    /// The WAV's file name, beside this file.
    var audio: String
    var visemes: VisemeTrack?
    var loudness: LoudnessTrack?

    /// The record of `spoken`, the reading in `audio`; nil where it has neither shapes nor loudness.
    init?(_ spoken: Spoken, audio: String) {
        guard spoken.visemes != nil || spoken.loudness != nil else { return nil }
        self.audio = audio
        visemes = spoken.visemes.map {
            VisemeTrack(frameMs: $0.frameMilliseconds, names: Self.visemeNames, values: $0.values)
        }
        // To the hundredth of a dB: as finely as a mouth can follow, and a tenth of the digits.
        loudness = spoken.loudness.map {
            LoudnessTrack(
                frameMs: $0.frameMilliseconds, unit: "dBFS",
                values: $0.values.map { ($0 * 100).rounded() / 100 })
        }
    }

    /// The same reading's record beside a WAV of another name.
    func renamed(_ audio: String) -> LipsyncRecord {
        var record = self
        record.audio = audio
        return record
    }

    /// The shapes and loudness again, for `clip`, the reading in `audio`.
    func spoken(_ clip: AudioClip) -> Spoken {
        Spoken(
            clip: clip,
            visemes: visemes.map { Visemes(values: $0.values, frameMilliseconds: $0.frameMs) },
            loudness: loudness.map { Loudness(values: $0.values, frameMilliseconds: $0.frameMs) })
    }

    func json() throws -> Data {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        return try encoder.encode(self)
    }

    static func read(_ data: Data) -> LipsyncRecord? {
        guard let record = try? JSONDecoder().decode(LipsyncRecord.self, from: data), record.version == 1
        else { return nil }
        return record
    }
}

final class Lipsync: @unchecked Sendable {
    private struct Playing {
        var id: Int
        var rate: Int
        /// The sound, to hear how loud it is from, where libwaifu has not said.
        var samples: [Float]
        var visemes: Visemes?
        /// How many times the visemes have been handed over: each time all of them so far.
        var visemesSeen = 0
        var loudness: Loudness?
        var loudnessSeen = 0
        /// Seconds into the reading the sound is, or nil while none of it is being heard.
        var clock: @Sendable () -> Double?
    }

    private let state = OSAllocatedUnfairLock<(playing: Playing?, next: Int)>(initialState: (nil, 0))

    /// From now on, a reading at `rate` -- its `visemes` and `loudness` so far, or, with no
    /// loudness, its `samples` to hear it from -- heard at `clock`: until `stop` with what this
    /// returns, or another reading.
    @discardableResult
    func play(
        rate: Int, samples: [Float] = [], visemes: Visemes? = nil, loudness: Loudness? = nil,
        clock: @escaping @Sendable () -> Double?
    ) -> Int {
        state.withLock { s in
            s.next += 1
            s.playing = Playing(
                id: s.next, rate: rate, samples: loudness == nil ? samples : [], visemes: visemes,
                visemesSeen: visemes == nil ? 0 : 1, loudness: loudness, loudnessSeen: loudness == nil ? 0 : 1,
                clock: clock)
            return s.next
        }
    }

    /// The loudness settled so far of a streaming reading: the one `id` names, if it is still
    /// playing.
    func loud(_ loudness: Loudness, for id: Int) {
        state.withLock { s in
            guard s.playing?.id == id else { return }
            s.playing?.loudness = loudness
            s.playing?.loudnessSeen += 1
        }
    }

    /// The shapes settled so far of a streaming reading: the one `id` names, if it is still playing.
    func settled(_ visemes: Visemes, for id: Int) {
        state.withLock { s in
            guard s.playing?.id == id else { return }
            s.playing?.visemes = visemes
            s.playing?.visemesSeen += 1
        }
    }

    /// The reading `id` names no longer plays; another that has started since is left playing.
    func stop(_ id: Int) {
        state.withLock { s in
            if s.playing?.id == id { s.playing = nil }
        }
    }

    /// What a drawing has not been handed yet of the reading playing.
    struct News {
        var id: Int
        var rate: Int
        var samples: ArraySlice<Float>
        var visemes: Visemes?
        var visemesSeen: Int
        var loudness: Loudness?
        var loudnessSeen: Int
        var seconds: Double?
    }

    /// What a drawing has had of the reading `id` names.
    struct Had {
        var id: Int
        var samples = 0
        var visemes = 0
        var loudness = 0
    }

    /// The reading playing, with its samples after the ones `had`, and its visemes and loudness
    /// where they are newer -- or nil where none is playing.
    func news(had: Had?) -> News? {
        let found = state.withLock { s -> (Playing, ArraySlice<Float>, Visemes?, Loudness?)? in
            guard let p = s.playing else { return nil }
            let same = p.id == had?.id
            let from = same ? min(had?.samples ?? 0, p.samples.count) : 0
            let newerVisemes = !same || p.visemesSeen != had?.visemes
            let newerLoudness = !same || p.loudnessSeen != had?.loudness
            return (p, p.samples[from...], newerVisemes ? p.visemes : nil, newerLoudness ? p.loudness : nil)
        }
        guard let (playing, samples, visemes, loudness) = found else { return nil }
        return News(
            id: playing.id, rate: playing.rate, samples: samples, visemes: visemes, visemesSeen: playing.visemesSeen,
            loudness: loudness, loudnessSeen: playing.loudnessSeen, seconds: playing.clock())
    }
}
