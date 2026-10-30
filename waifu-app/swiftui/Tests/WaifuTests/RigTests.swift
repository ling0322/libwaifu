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

// The live2d page without a window: a Photoshop document read, the rig made of it, and the blink.

import AppKit
import Foundation
import SwiftUI
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

    func testLayersAreKnownByTheirNames() {
        XCTAssertEqual(Rig.role(of: "eyewhite").0, .eyewhite)
        XCTAssertEqual(Rig.role(of: "白目").0, .eyewhite)
        XCTAssertEqual(Rig.role(of: "Eyelash_2").0, .eyelash)
        XCTAssertEqual(Rig.role(of: "eyelash のコピー").0, .eyelash)
        XCTAssertEqual(Rig.role(of: "front hair").0, .other)
        // See-through's sides are the character's: its right eye is on the picture's left.
        XCTAssertTrue(Rig.role(of: "irides-r") == (.irides, .left))
        XCTAssertTrue(Rig.role(of: "eyebrow-l") == (.eyebrow, .right))
        XCTAssertTrue(Rig.role(of: "右白目") == (.eyewhite, .left))
        XCTAssertTrue(Rig.role(of: "face") == (.face, nil))
    }

    /// A face 100 wide in the middle of a 200 by 200 canvas, and both eyes on it drawn as one
    /// layer each: whites, irides, lashes.
    private func face() throws -> PSDDocument {
        let both = { (x: Int, _: Int) in x < 30 || x >= 50 }
        return try PSDDocument.read(
            PSDWriter.document(
                width: 200, height: 200,
                layers: [
                    layer("face", (50, 40, 100, 120), colour: [250, 220, 200, 255]),
                    layer("eyewhite", (65, 80, 80, 20), colour: [255, 255, 255, 255], inside: both),
                    layer("irides", (70, 82, 70, 16), colour: [40, 90, 200, 255], inside: { x, _ in x < 20 || x >= 50 }),
                    layer("eyelash", (63, 76, 84, 6), colour: [60, 30, 30, 255], inside: { x, _ in x < 32 || x >= 52 }),
                    layer("front hair", (40, 20, 120, 30), colour: [80, 160, 160, 255]),
                ]))
    }

    func testBothEyesOfOneLayerEachAreFoundAndGivenAClosedEye() throws {
        let rig = Rig.build(try face())
        XCTAssertEqual(rig.notes, [])
        let left = try XCTUnwrap(rig.eyes[.left])
        let right = try XCTUnwrap(rig.eyes[.right])
        // Cut at the middle of the face, x = 100: each white is its own 30 by 20.
        XCTAssertEqual(left.white, CGRect(x: 65, y: 80, width: 30, height: 20))
        XCTAssertEqual(right.white, CGRect(x: 115, y: 80, width: 30, height: 20))
        XCTAssertEqual(left.closeY, 80 + 0.62 * 20, accuracy: 1e-9)
        XCTAssertEqual(left.iris.y, 90, accuracy: 1e-9)

        // A closed eye each, over the lashes and under the hair, 1.1 times the white and lash-coloured.
        let names = rig.parts.map { "\($0.role)\($0.side.map { "-\($0)" } ?? "")" }
        XCTAssertEqual(
            names,
            ["face", "eyewhite-left", "eyewhite-right", "irides-left", "irides-right", "eyelash-left",
             "eyelash-right", "eyeClose-left", "eyeClose-right", "other"])
        let closed = try XCTUnwrap(rig.parts.first { $0.role == .eyeClose && $0.side == .left })
        XCTAssertEqual(closed.rect.width, 33, accuracy: 1)
        XCTAssertEqual(closed.rect.midX, left.white.midX, accuracy: 1)
        XCTAssertEqual(rig.faceScale, 100 / 333, accuracy: 1e-9)
    }

    func testAClosedEyeIsPaintedTheColourOfTheLashes() throws {
        let eye = try XCTUnwrap(Rig.closedEye(.left, width: 88, tint: (0.5, 0.25, 0.25)))
        XCTAssertEqual(eye.width, 88)
        XCTAssertEqual(eye.height, 42)
        let colour = try XCTUnwrap(Pixels.meanColour(eye))
        // Around the tint: each pixel as light as it was against the drawing's mean.
        XCTAssertEqual(colour.0 / colour.1, 2, accuracy: 0.1)
        XCTAssertEqual(colour.1, colour.2, accuracy: 0.01)
    }

    func testAPictureWithoutEyesSaysWhy() throws {
        let rig = Rig.build(
            try PSDDocument.read(PSDWriter.document(width: 10, height: 10, layers: [layer("body", (0, 0, 10, 10), colour: [1, 1, 1, 255])])))
        XCTAssertTrue(rig.eyes.isEmpty)
        XCTAssertEqual(rig.notes.count, 1)
    }

    /// A face and a top over it, the top's left half near and its right half far, with their depth
    /// in the document's own hidden group, as img2layers writes it.
    func testADocumentKeepingItsDepthIsStratifiedOnTheWayToTheRig() throws {
        let fill = { (w: Int, h: Int, pixel: (Int, Int) -> [UInt8]) in
            RGBAImage(width: w, height: h, rgba: (0..<(w * h)).flatMap { pixel($0 % w, $0 / w) })
        }
        let face = fill(40, 20) { _, _ in [250, 220, 200, 255] }
        let top = fill(20, 10) { _, _ in [90, 90, 90, 255] }
        let faceDepth = fill(40, 20) { _, _ in [120, 120, 120, 255] }
        let topDepth = fill(20, 10) { x, _ in x < 10 ? [40, 40, 40, 255] : [200, 200, 200, 255] }
        let document = try PSDDocument.read(
            PSD.write(
                [PSD.Layer(name: "face", left: 0, top: 0, image: face), PSD.Layer(name: "topwear", left: 10, top: 5, image: top)],
                width: 40, height: 20,
                depths: [
                    PSD.Layer(name: PSD.depthName(of: "face"), left: 0, top: 0, image: faceDepth),
                    PSD.Layer(name: PSD.depthName(of: "topwear"), left: 10, top: 5, image: topDepth),
                ]))
        let depths = RigStage.depths(of: document, at: URL(fileURLWithPath: "/nowhere/x.psd"))
        XCTAssertEqual(Set(depths.keys), ["face", "topwear"])

        // As it is: the face and the top, and not the depth, which is hidden.
        XCTAssertEqual(try RigStage.rig(document, depths: depths, stratify: false).parts.map(\.name), ["face", "topwear"])
        // Stratified: the top's far half under the face, its near half where the top was.
        XCTAssertEqual(
            try RigStage.rig(document, depths: depths, stratify: true).parts.map(\.name), ["topwear-1", "face", "topwear-0"])
        // Without its depth it is left as it is.
        XCTAssertEqual(try RigStage.rig(document, depths: [:], stratify: true).parts.map(\.name), ["face", "topwear"])
    }

    func testABlinkClosesHoldsAndOpens() {
        XCTAssertEqual(Blinker.shape(0), 1)
        XCTAssertEqual(try XCTUnwrap(Blinker.shape(0.04)), 0.5, accuracy: 1e-9)
        XCTAssertEqual(Blinker.shape(0.2), 0)
        XCTAssertEqual(try XCTUnwrap(Blinker.shape(0.42 + 0.08)), 0.5, accuracy: 1e-9)
        XCTAssertNil(Blinker.shape(0.6))

        // On its own, the first one 1.8 s in: shut by the middle of it, open again after.
        var blinker = Blinker()
        var least = 1.0
        for _ in 0..<(60 * 4) {
            blinker.step(by: 1.0 / 60, open: 1, automatic: true, random: { 0.5 })
            if blinker.now < 1.8 { XCTAssertEqual(blinker.openness, 1, accuracy: 1e-9) }
            least = min(least, blinker.openness)
        }
        XCTAssertLessThan(least, 0.05)
        XCTAssertEqual(blinker.openness, 1, accuracy: 0.01)

        // Never on its own; once when asked.
        var still = Blinker()
        for _ in 0..<(60 * 4) { still.step(by: 1.0 / 60, open: 1, automatic: false) }
        XCTAssertEqual(still.openness, 1, accuracy: 1e-9)
        still.blinkNow()
        for _ in 0..<15 { still.step(by: 1.0 / 60, open: 1, automatic: false) }
        XCTAssertLessThan(still.openness, 0.2)
    }

    /// An img2layers document, where one is named by LIBWAIFU_RIG_PSD: what the page would make of
    /// it. With LIBWAIFU_RIG_FRAMES, a directory, the eyes as they close -- open, half, a third,
    /// a sixth, shut -- one under the other in blink.png there, to look at.
    @MainActor
    func testARealDocument() throws {
        guard let path = ProcessInfo.processInfo.environment["LIBWAIFU_RIG_PSD"] else {
            throw XCTSkip("set LIBWAIFU_RIG_PSD to an img2layers document")
        }
        let document = try PSDDocument.read(Data(contentsOf: URL(fileURLWithPath: path)))
        let rig = Rig.build(document)
        print("\(document.width) x \(document.height):", document.layers.map(\.name))
        print("parts:", rig.parts.map { "\($0.name) \($0.role) \($0.side.map { "\($0)" } ?? "-") \($0.rect)" })
        print("eyes:", rig.eyes, "notes:", rig.notes)
        XCTAssertEqual(rig.eyes.count, 2)

        guard let frames = ProcessInfo.processInfo.environment["LIBWAIFU_RIG_FRAMES"] else { return }
        let eyes = rig.eyes.values.map(\.white).reduce(CGRect.null) { $0.union($1) }.insetBy(dx: -60, dy: -50)
        let side = CGSize(width: rig.width, height: rig.height)
        let crops = try [1.0, 0.5, 0.3, 0.15, 0].map { v -> CGImage in
            let renderer = ImageRenderer(
                content: Canvas { context, size in RigView.draw(rig, in: &context, size: size, openness: v) }
                    .frame(width: side.width, height: side.height)
                    .background(Color.white))
            renderer.scale = 1
            let frame = try XCTUnwrap(renderer.cgImage)
            return try XCTUnwrap(frame.cropping(to: eyes.integral))
        }
        let (w, h) = (crops[0].width, crops[0].height)
        let sheet = try XCTUnwrap(
            CGContext(
                data: nil, width: w, height: h * crops.count, bitsPerComponent: 8, bytesPerRow: 0,
                space: CGColorSpace(name: CGColorSpace.sRGB)!, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue))
        for (i, crop) in crops.enumerated() {
            sheet.draw(crop, in: CGRect(x: 0, y: h * (crops.count - 1 - i), width: w, height: h))
        }
        let png = try XCTUnwrap(NSBitmapImageRep(cgImage: try XCTUnwrap(sheet.makeImage())).representation(using: .png, properties: [:]))
        try png.write(to: URL(fileURLWithPath: frames).appendingPathComponent("blink.png"))
    }
}
