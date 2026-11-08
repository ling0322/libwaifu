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

import Foundation
import WaifuKit
import XCTest

@testable import Waifu

final class StreamPlayerTests: XCTestCase {
    /// `seconds` of quiet at 24 kHz: a part as short as a reading's could be.
    private func part(_ seconds: Double) -> AudioClip {
        AudioClip(samples: [Float](repeating: 0, count: Int(seconds * 24000)), rate: 24000)
    }

    /// Waits up to `seconds` for `condition`.
    private func eventually(_ seconds: Double, _ condition: () -> Bool) -> Bool {
        let deadline = Date().addingTimeInterval(seconds)
        while Date() < deadline {
            if condition() { return true }
            Thread.sleep(forTimeInterval: 0.02)
        }
        return condition()
    }

    func testPartsAreQueuedAndTheHardwareLetGoOnceTheyHavePlayed() throws {
        let player = StreamPlayer()
        player.play(part(0.05))
        // A Mac with no audio output -- a CI machine -- has no engine to start.
        guard player.heard == 1 else { throw XCTSkip("no audio output here") }
        player.play(part(0.05))
        player.play(part(0.05))
        XCTAssertEqual(player.heard, 3)
        XCTAssertTrue(player.playing)

        // Finished, it plays what it has and then stops on its own.
        player.finish()
        XCTAssertTrue(eventually(3) { !player.playing })
        // And takes nothing more.
        player.play(part(0.05))
        XCTAssertEqual(player.heard, 3)
    }

    func testAStopSilencesItForGood() throws {
        let player = StreamPlayer()
        player.play(part(1))
        guard player.heard == 1 else { throw XCTSkip("no audio output here") }
        player.stop()
        XCTAssertFalse(player.playing)
        // A part the engine hands over after the Stop is dropped.
        player.play(part(0.05))
        XCTAssertEqual(player.heard, 1)
        XCTAssertFalse(player.playing)
        // And a finish after it has nothing to wait for.
        player.finish()
        XCTAssertFalse(player.playing)
    }

    func testAFinishBeforeAnyPartHoldsNothing() {
        let player = StreamPlayer()
        player.finish()
        XCTAssertFalse(player.playing)
        player.play(part(0.05))
        XCTAssertEqual(player.heard, 0)
    }
}
