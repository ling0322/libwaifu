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

// Photoshop documents read back: what the reader makes of a document written by hand.

import Foundation
import WaifuKit
import XCTest

@testable import Waifu

/// A Photoshop document written by hand, as little of one as the reader reads.
private struct PSDWriter {
    struct Layer {
        var name: String
        var left = 0, top = 0, width = 0, height = 0
        /// RGBA, not premultiplied.
        var rgba: [UInt8] = []
        var hidden = false
        var packBits = false
        /// 1 opens a group, 3 closes one, 0 is a layer.
        var divider = 0
    }

    var bytes: [UInt8] = []

    mutating func u8(_ v: Int) { bytes.append(UInt8(v & 0xFF)) }
    mutating func u16(_ v: Int) { u8(v >> 8); u8(v) }
    mutating func u32(_ v: Int) { u16(v >> 16); u16(v) }
    mutating func ascii(_ s: String) { bytes.append(contentsOf: Array(s.utf8)) }

    /// `layers` bottom first, as the format keeps them.
    static func document(width: Int, height: Int, layers: [Layer]) -> Data {
        var w = PSDWriter()
        w.ascii("8BPS"); w.u16(1); w.bytes += [0, 0, 0, 0, 0, 0]
        w.u16(3); w.u32(height); w.u32(width); w.u16(8); w.u16(3)
        w.u32(0); w.u32(0)

        var info = PSDWriter()
        info.u16(layers.count)
        var data = PSDWriter()
        for layer in layers {
            // Each channel compressed now, so that its length is known: alpha first.
            var channels: [(Int, [UInt8])] = []
            for (id, offset) in [(-1, 3), (0, 0), (1, 1), (2, 2)] {
                var c = PSDWriter()
                let plane = stride(from: offset, to: layer.rgba.count, by: 4).map { layer.rgba[$0] }
                if layer.packBits {
                    c.u16(1)
                    // Every row one literal run, or one repeat where it is all one byte.
                    var rows: [[UInt8]] = []
                    for y in 0..<layer.height {
                        let row = Array(plane[y * layer.width..<(y + 1) * layer.width])
                        rows.append(Set(row).count == 1 ? [UInt8(bitPattern: Int8(1 - row.count)), row[0]] : [UInt8(row.count - 1)] + row)
                    }
                    for row in rows { c.u16(row.count) }
                    for row in rows { c.bytes += row }
                } else {
                    c.u16(0)
                    c.bytes += plane
                }
                channels.append((id, c.bytes))
            }
            info.u32(layer.top); info.u32(layer.left); info.u32(layer.top + layer.height); info.u32(layer.left + layer.width)
            info.u16(channels.count)
            for (id, c) in channels { info.u16(id); info.u32(c.count) }
            info.ascii("8BIMnorm"); info.u8(255); info.u8(0); info.u8(layer.hidden ? 2 : 0); info.u8(0)

            var extra = PSDWriter()
            extra.u32(0); extra.u32(0)
            // A short name in ASCII, padded to four; the whole of it in UTF-16 under "luni".
            let short = Array(layer.name.utf8.prefix(31)).map { $0 < 128 ? $0 : UInt8(ascii: "?") }
            extra.u8(short.count); extra.bytes += short
            while (extra.bytes.count - 8) % 4 != 0 { extra.u8(0) }
            let units = Array(layer.name.utf16)
            extra.ascii("8BIMluni"); extra.u32(4 + units.count * 2); extra.u32(units.count)
            for u in units { extra.u16(Int(u)) }
            if layer.divider != 0 { extra.ascii("8BIMlsct"); extra.u32(4); extra.u32(layer.divider) }
            info.u32(extra.bytes.count); info.bytes += extra.bytes

            for (_, c) in channels { data.bytes += c }
        }
        var section = PSDWriter()
        section.u32(info.bytes.count + data.bytes.count)
        section.bytes += info.bytes + data.bytes
        section.u32(0)
        w.u32(section.bytes.count)
        w.bytes += section.bytes
        return Data(w.bytes)
    }
}

/// A layer the size of `box`, `colour` all over.
private func layer(_ name: String, _ box: (Int, Int, Int, Int), colour: [UInt8], packBits: Bool = false) -> PSDWriter.Layer {
    let (left, top, width, height) = box
    let rgba = (0..<(width * height)).flatMap { _ in colour }
    return PSDWriter.Layer(name: name, left: left, top: top, width: width, height: height, rgba: rgba, packBits: packBits)
}

final class PSDTests: XCTestCase {
    func testADocumentIsReadLayerByLayer() throws {
        var hidden = layer("hidden one", (0, 0, 2, 2), colour: [9, 9, 9, 255])
        hidden.hidden = true
        let data = PSDWriter.document(
            width: 8, height: 6,
            layers: [
                layer("back", (0, 0, 8, 6), colour: [10, 20, 30, 255]),
                layer("前髪", (1, 2, 3, 2), colour: [200, 100, 50, 128], packBits: true),
                hidden,
                // A group, closed then opened going up, hidden: what is in it is hidden too.
                PSDWriter.Layer(name: "</Layer group>", divider: 3),
                layer("in a hidden group", (2, 2, 2, 2), colour: [1, 2, 3, 255]),
                { var g = PSDWriter.Layer(name: "group", divider: 1); g.hidden = true; return g }(),
            ])
        let document = try PSDDocument.read(data)
        XCTAssertEqual(document.width, 8)
        XCTAssertEqual(document.height, 6)
        XCTAssertEqual(document.layers.map(\.name), ["back", "前髪", "hidden one", "in a hidden group"])
        XCTAssertEqual(document.layers.map(\.visible), [true, true, false, false])

        let fringe = document.layers[1]
        XCTAssertEqual([fringe.left, fringe.top, fringe.width, fringe.height], [1, 2, 3, 2])
        XCTAssertEqual(Array(fringe.image.rgba.prefix(4)), [200, 100, 50, 128], "PackBits, alpha its own channel")
        XCTAssertEqual(Array(document.layers[0].image.rgba.suffix(4)), [10, 20, 30, 255])
    }

    func testWhatIsNotAPhotoshopDocumentSaysSo() {
        XCTAssertThrowsError(try PSDDocument.read(Data("not a psd".utf8)))
        var big = [UInt8](PSDWriter.document(width: 1, height: 1, layers: []))
        big[5] = 2
        XCTAssertThrowsError(try PSDDocument.read(Data(big))) { error in
            XCTAssertTrue(error.localizedDescription.contains(".psb"), error.localizedDescription)
        }
    }
}
