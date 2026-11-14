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

// The waifu2d page without a window: a Photoshop document handed to libwaifu2d, rigged, moved and
// drawn. How the rig is made and how it moves are libwaifu2d's to test (waifu2d/tests/); these are
// the page's way to it, through WaifuKit.

import AppKit
import Foundation
import WaifuKit
import XCTest

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

/// A layer the size of `box` on a `width` by `height` canvas, `colour` where `inside` says, clear
/// elsewhere.
private func layer(
    _ name: String, _ box: (Int, Int, Int, Int), colour: [UInt8], packBits: Bool = false,
    inside: (Int, Int) -> Bool = { _, _ in true }
) -> PSDWriter.Layer {
    let (left, top, width, height) = box
    var rgba: [UInt8] = []
    for y in 0..<height {
        for x in 0..<width { rgba += inside(x, y) ? colour : [0, 0, 0, 0] }
    }
    return PSDWriter.Layer(name: name, left: left, top: top, width: width, height: height, rgba: rgba, packBits: packBits)
}

final class RigTests: XCTestCase {
    /// A face 100 wide in the middle of a 200 by 200 canvas, and both eyes on it drawn as one
    /// layer each: whites, irides, lashes.
    private func face() -> Data {
        let both = { (x: Int, _: Int) in x < 30 || x >= 50 }
        return PSDWriter.document(
            width: 200, height: 200,
            layers: [
                layer("face", (50, 40, 100, 120), colour: [250, 220, 200, 255]),
                layer("eyewhite", (65, 80, 80, 20), colour: [255, 255, 255, 255], inside: both),
                layer("irides", (70, 82, 70, 16), colour: [40, 90, 200, 255], inside: { x, _ in x < 20 || x >= 50 }),
                layer("eyelash", (63, 76, 84, 6), colour: [60, 30, 30, 255], inside: { x, _ in x < 32 || x >= 52 }),
                layer("front hair", (40, 20, 120, 30), colour: [80, 160, 160, 255]),
            ])
    }

    func testTheLibraryIsTheOneThisWasWrittenFor() {
        XCTAssertTrue(Waifu2D.isCompatible, "libwaifu2d's ABI is \(Waifu2D.abiVersion)")
    }

    func testBothEyesOfOneLayerEachAreFoundAndGivenAClosedEye() throws {
        let rig = try Waifu2D.Rig(psd: face())
        XCTAssertEqual(rig.width, 200)
        XCTAssertEqual(rig.height, 200)
        XCTAssertEqual(rig.notes, [])
        XCTAssertEqual(rig.eyeCount, 2)
        // Each eye's white, iris and lashes, cut at the middle of the face, a closed eye each, the
        // face and the hair.
        XCTAssertEqual(rig.partCount, 10)
    }

    func testAPictureWithoutEyesSaysWhy() throws {
        let rig = try Waifu2D.Rig(
            psd: PSDWriter.document(width: 10, height: 10, layers: [layer("body", (0, 0, 10, 10), colour: [1, 1, 1, 255])]))
        XCTAssertEqual(rig.eyeCount, 0)
        XCTAssertEqual(rig.notes.count, 1)
    }

    func testWhatIsNotADocumentIsRefused() {
        XCTAssertThrowsError(try Waifu2D.Rig(psd: Data("not a document".utf8))) { error in
            XCTAssertFalse((error as? Waifu2D.Failure)?.message.isEmpty ?? true)
        }
    }

    /// A stage that moves on nothing of its own: what is drawn is the rig at rest, and a blink.
    private func still(_ rig: Waifu2D.Rig) throws -> Waifu2D.Stage {
        let stage = try Waifu2D.Stage(rig: rig, seed: 1)
        for s in [Waifu2D.Switch.followPointer, .idle, .wind, .lookAround, .physics, .blink] { stage.set(s, false) }
        return stage
    }

    func testMetalDrawsTheRigAndKeepsTheIrisInsideTheWhite() throws {
        guard Waifu2D.API.metal.isAvailable else { throw XCTSkip("no Metal device") }
        let rig = try Waifu2D.Rig(psd: face())
        let stage = try still(rig)
        let renderer = try Waifu2D.Renderer(width: 200, height: 200, api: .metal)
        try renderer.draw(stage)
        let pixels = try renderer.rgba().rgba
        let at = { (x: Int, y: Int) in Array(pixels[(y * 200 + x) * 4..<(y * 200 + x) * 4 + 4]) }
        // Nothing drawn in the corner; the face where it is; the iris over the white.
        XCTAssertEqual(at(5, 5), [0, 0, 0, 0])
        XCTAssertEqual(at(100, 140), [250, 220, 200, 255])
        XCTAssertEqual(at(75, 90), [40, 90, 200, 255])
        // The iris layer reaches past the white's bottom only where the white is: shut, there is no
        // iris over the face below the line the eye closes onto.
        stage.act(.blink)
        for _ in 0..<12 { stage.step(by: 1.0 / 60) }
        try renderer.draw(stage)
        let blinking = try renderer.rgba().rgba
        XCTAssertNotEqual(Array(blinking[(97 * 200 + 75) * 4..<(97 * 200 + 75) * 4 + 3]), [40, 90, 200])
    }

    func testTheCPUDrawsWhatMetalDraws() throws {
        guard Waifu2D.API.metal.isAvailable else { throw XCTSkip("no Metal device") }
        let rig = try Waifu2D.Rig(psd: face())
        let stage = try still(rig)
        let drawn = try [Waifu2D.API.cpu, .metal].map { api -> [UInt8] in
            let renderer = try Waifu2D.Renderer(width: 200, height: 200, api: api)
            try renderer.draw(stage)
            return try renderer.rgba().rgba
        }
        let far = zip(drawn[0], drawn[1]).filter { abs(Int($0) - Int($1)) > 2 }.count
        XCTAssertLessThanOrEqual(Double(far) / Double(drawn[0].count), 0.005)
    }

    /// An img2layers document, where one is named by LIBWAIFU_RIG_PSD: what the page would make of
    /// it. With LIBWAIFU_RIG_FRAMES, a directory, the rig at rest in rig.png there, to look at.
    func testARealDocument() throws {
        guard let path = ProcessInfo.processInfo.environment["LIBWAIFU_RIG_PSD"] else {
            throw XCTSkip("set LIBWAIFU_RIG_PSD to an img2layers document")
        }
        let rig = try Waifu2D.Rig(contentsOf: URL(fileURLWithPath: path))
        print("\(rig.width) x \(rig.height), \(rig.partCount) parts, \(rig.eyeCount) eyes; notes:", rig.notes)
        XCTAssertEqual(rig.eyeCount, 2)

        guard let frames = ProcessInfo.processInfo.environment["LIBWAIFU_RIG_FRAMES"] else { return }
        let renderer = try Waifu2D.Renderer(width: rig.width, height: rig.height)
        try renderer.draw(try still(rig), clear: (1, 1, 1, 1))
        let image = try XCTUnwrap(cgImage(try renderer.rgba()))
        let png = try XCTUnwrap(NSBitmapImageRep(cgImage: image).representation(using: .png, properties: [:]))
        try png.write(to: URL(fileURLWithPath: frames).appendingPathComponent("rig.png"))
    }

    /// A premultiplied picture the library drew, as a CGImage.
    private func cgImage(_ image: RGBAImage) -> CGImage? {
        guard let provider = CGDataProvider(data: Data(image.rgba) as CFData) else { return nil }
        return CGImage(
            width: image.width, height: image.height, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: image.width * 4,
            space: CGColorSpace(name: CGColorSpace.sRGB)!,
            bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue), provider: provider, decode: nil,
            shouldInterpolate: false, intent: .defaultIntent)
    }
}
