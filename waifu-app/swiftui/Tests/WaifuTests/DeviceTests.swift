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

// What the Device card starts at on a Mac that has never been told anything.

import XCTest

@testable import Waifu

@MainActor
final class DeviceTests: XCTestCase {
    func testTheVramLimitStartsAtAThirdOfTheMacRoundedUpToAGigabyte() {
        let gib: UInt64 = 1 << 30
        // A Mac sold as 8, 16, 48 or 128 GB has that many GiB: 8.6, 17.2, 51.5 and 137.4 GB.
        XCTAssertEqual(Session.defaultVramLimitGB(memory: 8 * gib), 3)
        XCTAssertEqual(Session.defaultVramLimitGB(memory: 16 * gib), 6)
        XCTAssertEqual(Session.defaultVramLimitGB(memory: 48 * gib), 18)
        XCTAssertEqual(Session.defaultVramLimitGB(memory: 128 * gib), 46)
        // A whole gigabyte is not rounded past.
        XCTAssertEqual(Session.defaultVramLimitGB(memory: 30_000_000_000), 10)
    }
}
