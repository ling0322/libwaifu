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

import XCTest

@testable import Waifu

final class JitterBufferTests: XCTestCase {
    /// A loud constant: where a cut is a click, it is the step from it to silence.
    private func loud(_ count: Int) -> [Float] {
        [Float](repeating: 0.5, count: count)
    }

    /// The largest step from one sample to the next, across `samples`.
    private func largestStep(_ samples: [Float]) -> Float {
        zip(samples, samples.dropFirst()).map { abs($1 - $0) }.max() ?? 0
    }

    func testItWaitsThePrerollBeforePlaying() {
        var buffer = JitterBuffer(preroll: 100, fade: 10)
        // Nothing held: silence, and no wait counted.
        XCTAssertEqual(buffer.pull(50), [Float](repeating: 0, count: 50))
        // However much is held -- more than the preroll here, as a reading's first part is -- it
        // waits the preroll of the hardware's time, and starts from nothing on a fade in.
        buffer.push(loud(1000))
        let out = buffer.pull(150)
        XCTAssertEqual(Array(out[..<100]), [Float](repeating: 0, count: 100), "plays before the wait")
        XCTAssertGreaterThan(out[100], 0)
        XCTAssertLessThan(out[100], 0.5)
        XCTAssertEqual(out[149], 0.5)
        XCTAssertLessThanOrEqual(largestStep(out), 0.5 / 10 + 1e-6)
    }

    func testRunningDryFadesOutRatherThanCutting() {
        var buffer = JitterBuffer(preroll: 20, fade: 10)
        buffer.push(loud(100))
        // Pulled past what it holds, with more to come, 32 at a time -- fades that cross a pull --
        // it fades out and is silent, never cut.
        var played: [Float] = []
        for _ in 0..<6 { played += buffer.pull(32) }
        XCTAssertEqual(buffer.underruns, 1)
        XCTAssertEqual(played.filter { $0 != 0 }.count, 100)
        XCTAssertEqual(played.suffix(40), ArraySlice([Float](repeating: 0, count: 40)))
        XCTAssertLessThanOrEqual(largestStep(played), 0.5 / 10 + 1e-6, "a click")

        // More comes: it waits the preroll again, then comes in on a fade.
        buffer.push(loud(100))
        var again: [Float] = []
        for _ in 0..<3 { again += buffer.pull(32) }
        XCTAssertEqual(Array(again[..<20]), [Float](repeating: 0, count: 20))
        XCTAssertEqual(again[40], 0.5)
        XCTAssertLessThanOrEqual(largestStep(again), 0.5 / 10 + 1e-6, "a click")
    }

    func testFinishedItPlaysWhatItHoldsWithoutThePreroll() {
        var buffer = JitterBuffer(preroll: 1000, fade: 10)
        buffer.push(loud(50))
        XCTAssertEqual(buffer.pull(32), [Float](repeating: 0, count: 32))
        buffer.finish()
        // All of it, held back no longer, and then silence: played out, not run dry.
        var played: [Float] = []
        for _ in 0..<3 { played += buffer.pull(32) }
        XCTAssertEqual(played.filter { $0 != 0 }.count, 50)
        XCTAssertTrue(buffer.drained)
        XCTAssertEqual(buffer.underruns, 0)
        // Nothing is taken after it.
        buffer.push(loud(10))
        XCTAssertEqual(buffer.held, 0)
    }

    func testAStopFadesOutAndIsSilentForGood() {
        var buffer = JitterBuffer(preroll: 50, fade: 10)
        buffer.push(loud(500))
        _ = buffer.pull(64)
        _ = buffer.pull(64)
        buffer.stop()
        let out = buffer.pull(64)
        // At most a fade of what was held, falling, then silence.
        XCTAssertGreaterThan(out[0], 0)
        XCTAssertEqual(out[10...], ArraySlice([Float](repeating: 0, count: 54)))
        XCTAssertLessThanOrEqual(largestStep([0.5] + out), 0.5 / 10 + 1e-6, "a click")
        XCTAssertTrue(buffer.drained)
        buffer.push(loud(100))
        XCTAssertEqual(buffer.pull(64), [Float](repeating: 0, count: 64))
    }

    func testAStopBeforeItHasStartedPlaysNothing() {
        var buffer = JitterBuffer(preroll: 100, fade: 10)
        buffer.push(loud(60))
        buffer.stop()
        XCTAssertEqual(buffer.pull(64), [Float](repeating: 0, count: 64))
        XCTAssertTrue(buffer.drained)
    }

    func testAReadingThatComesInTimeIsPlayedWhole() {
        // Parts that always come before the one ahead has played: every sample out, as it was.
        var buffer = JitterBuffer(preroll: 64, fade: 8)
        // Off zero from its first sample, so that where it starts is where the first sample is heard.
        let signal = (0..<1000).map { Float(sin(Double($0) * 0.05 + 1)) }
        var played: [Float] = []
        var next = 0
        while played.count < 1200 {
            if next < signal.count {
                buffer.push(Array(signal[next..<min(next + 100, signal.count)]))
                next += 100
                if next >= signal.count { buffer.finish() }
            }
            played += buffer.pull(50)
        }
        XCTAssertEqual(buffer.underruns, 0)
        let start = played.firstIndex { $0 != 0 } ?? 0
        // After the fade in, sample for sample.
        XCTAssertEqual(Array(played[(start + 8)..<(start + 1000)]), Array(signal[8..<1000]))
    }
}
