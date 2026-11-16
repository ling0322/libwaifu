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

// The words under a reading's player, measured as SwiftUI lays them out: their own height, a most,
// and less where the window has less.

import AppKit
import SwiftUI
import XCTest

@testable import Waifu

@MainActor
final class OfferHeightTests: XCTestCase {
    /// The words as a reading's player has them: as tall as they are, up to `most`, then scrolling.
    private func words(_ text: String, most: CGFloat = 160) -> some View {
        OfferHeight(height: most) {
            ViewThatFits(in: .vertical) {
                Text(text).frame(maxWidth: .infinity, alignment: .leading)
                ScrollView { Text(text).frame(maxWidth: .infinity, alignment: .leading) }
            }
        }
    }

    private func height(_ view: some View, width: CGFloat = 400, offered: CGFloat) -> CGFloat {
        NSHostingController(rootView: view).sizeThatFits(in: CGSize(width: width, height: offered)).height
    }

    private let long = String(repeating: "金鱼并不是鱼，这一点我当然知道。你不许借机嘲笑大飞鱼。", count: 40)

    /// A line of words, with all the room in the world, is a line high: nothing empty under it.
    func testShortWordsKeepToTheirOwnHeight() {
        let line = height(Text("金鱼并不是鱼。"), offered: 1000)
        XCTAssertEqual(height(words("金鱼并不是鱼。"), offered: 1000), line, accuracy: 1)
        XCTAssertLessThan(line, 40)
    }

    /// Long words, with the room, take the most and no more.
    func testLongWordsTakeTheMost() {
        XCTAssertEqual(height(words(long), offered: 1000), 160, accuracy: 1)
    }

    /// Long words in a short window give way to it, rather than push what is under them off.
    func testLongWordsGiveWayToAShortWindow() {
        XCTAssertLessThanOrEqual(height(words(long), offered: 60), 60)
    }
}
