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

// The waifu2d page without a window: a Photoshop document read, the rig made of it, and the blink.

import AppKit
import Foundation
import Metal
import simd
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

// Reading a document is PSDTests'; these are the rig made of one.
final class RigTests: XCTestCase {
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

    func testSlotsAreKnownByTheirNames() {
        XCTAssertEqual(Rig.slot(of: "前髪2").slot, "front hair")
        XCTAssertEqual(Rig.slot(of: "front hair-0").slot, "front hair")
        XCTAssertEqual(Rig.slot(of: "mouth").slot, "mouth_open")
        XCTAssertEqual(Rig.slot(of: "mouth_c").slot, "mouth_close")
        XCTAssertEqual(Rig.slot(of: "eyelash_c").slot, "eye_close")
        XCTAssertEqual(Rig.slot(of: "Skirt").slot, "bottomwear")
        XCTAssertEqual(Rig.slot(of: "earwear-l").slot, "earwear")
        XCTAssertEqual(Rig.slot(of: "earwear-l").side, .right)
        XCTAssertEqual(Rig.slot(of: "something").slot, "something")
        XCTAssertEqual(Rig.slots["front hair"]?.depth, 1.28)
    }

    func testAGridCoversItsLayer() {
        // 42-pixel cells at a 768 canvas: a 100 by 50 layer is 2 by 2 of them at least.
        let mesh = RigMesh.grid(over: CGRect(x: 10, y: 20, width: 100, height: 50), canvas: 768)
        XCTAssertEqual(mesh.rest.count, 3 * 3)
        XCTAssertEqual(mesh.indices.count, 2 * 2 * 6)
        XCTAssertEqual(mesh.rest.first, SIMD2(10, 20))
        XCTAssertEqual(mesh.rest.last, SIMD2(110, 70))
        XCTAssertEqual(mesh.uv.last, SIMD2(1, 1))
        // What swings is cut finer.
        let fine = RigMesh.grid(over: CGRect(x: 0, y: 0, width: 300, height: 300), canvas: 768, swings: true)
        XCTAssertEqual(fine.rest.count, 11 * 11)
    }

    func testHairHangsInAStrandAtEachTip() {
        // Three locks 60 wide, 100, 160 and 100 long: a strand at the bottom of each.
        let (w, h) = (180, 170)
        var rgba = [UInt8](repeating: 0, count: w * h * 4)
        for x in 0..<w {
            let long = [100, 160, 100][x / 60]
            for y in 0..<long { rgba[(y * w + x) * 4 + 3] = 255 }
        }
        let strands = Rig.strands(RGBAImage(width: w, height: h, rgba: rgba), want: 3, minSep: 30)
        XCTAssertEqual(strands.count, 3)
        XCTAssertTrue(strands.contains { abs($0.x - 90) <= 2 && $0.tipY == 159 && $0.rootY == 0 })
        XCTAssertEqual(strands.map(\.x), strands.map(\.x).sorted())

        let mesh = RigMesh.grid(over: CGRect(x: 0, y: 0, width: w, height: h), canvas: 768, swings: true)
        let swing = Rig.swing(.hair, front: true, strands: strands, mesh: mesh)
        XCTAssertEqual(swing.weights.count, mesh.rest.count * 3)
        for i in mesh.rest.indices {
            XCTAssertEqual(swing.weights[i * 3..<(i + 1) * 3].reduce(0, +), 1, accuracy: 1e-4)
        }
        XCTAssertEqual(swing.along.first, 0)
        XCTAssertEqual(swing.along.last, 1)
    }

    func testASpringSettlesOnItsTarget() {
        var spring = Spring()
        // Stiff, and damped: there in two seconds, at 10 frames a second as at 240.
        for _ in 0..<20 { spring.step(to: 5, k: 70, damping: 9, dt: 0.1) }
        XCTAssertEqual(spring.x, 5, accuracy: 0.01)
        var fine = Spring()
        for _ in 0..<480 { fine.step(to: 5, k: 70, damping: 9, dt: 1.0 / 240) }
        XCTAssertEqual(fine.x, spring.x, accuracy: 0.01)
    }

    /// A face with a mouth, a neck under it, hair in front and behind.
    private func head() throws -> Rig {
        Rig.build(
            try PSDDocument.read(
                PSDWriter.document(
                    width: 300, height: 400,
                    layers: [
                        layer("back hair", (60, 40, 180, 260), colour: [60, 40, 40, 255]),
                        layer("neck", (130, 200, 40, 60), colour: [240, 210, 190, 255]),
                        layer("topwear", (60, 250, 180, 150), colour: [40, 60, 120, 255]),
                        layer("face", (100, 60, 100, 150), colour: [250, 220, 200, 255]),
                        layer("mouth", (135, 160, 30, 12), colour: [150, 40, 50, 255]),
                        layer("front hair", (80, 30, 140, 70), colour: [60, 40, 40, 255]),
                    ])))
    }

    func testTheHeadTurnsMoreInFrontOfTheFace() throws {
        let rig = try head()
        XCTAssertEqual(rig.anchors.neckPivot.y, 200 + 0.85 * 59, accuracy: 1)
        XCTAssertNotNil(rig.parts.first { $0.slot == "front hair" }?.swing)

        // Turned right, still: the front hair goes farther than the face, the back hair less far,
        // the body after it, less far again.
        var motion = RigMotion(rig)
        let still = RigSwitches(blink: false, idle: false, lookAround: false, physics: false, followMouse: true)
        for _ in 0..<120 { motion.step(by: 1.0 / 60, rig: rig, switches: still, mouse: SIMD2(1, 0), open: 1, ease: 0.3) }
        XCTAssertEqual(motion.pose.angleX, 0.9, accuracy: 0.01)
        let moved = { (slot: String) -> Float in
            let i = rig.parts.firstIndex { $0.slot == slot }!
            let now = motion.deform(rig, part: i, physics: false)
            return zip(now, rig.parts[i].mesh.rest).map { $0.x - $1.x }.reduce(0, +) / Float(now.count)
        }
        let (front, face, back, top) = (moved("front hair"), moved("face"), moved("back hair"), moved("topwear"))
        XCTAssertGreaterThan(face, 0)
        XCTAssertGreaterThan(front, face)
        XCTAssertLessThan(back, face)
        XCTAssertGreaterThan(top, 0)
        XCTAssertLessThan(top, face)
        XCTAssertEqual(motion.pose.bodyTurn, 0.9, accuracy: 0.01)

        // The top turns at the shoulders, and hardly at all at the canvas's foot.
        let topwear = try XCTUnwrap(rig.parts.firstIndex { $0.slot == "topwear" })
        let now = motion.deform(rig, part: topwear, physics: false)
        let rest = rig.parts[topwear].mesh.rest
        XCTAssertGreaterThan(now[0].x - rest[0].x, 2 * abs(now[rest.count - 1].x - rest[rest.count - 1].x))

        // Without the body following, it does not turn.
        var alone = RigMotion(rig)
        for _ in 0..<120 {
            alone.step(by: 1.0 / 60, rig: rig, switches: still, mouse: SIMD2(1, 0), open: 1, ease: 0.3, follow: 0)
        }
        XCTAssertEqual(alone.pose.bodyTurn, 0)
    }

    func testLookingUpTheMiddleOfTheFaceGoesFartherThanItsOutline() throws {
        let rig = try head()
        var motion = RigMotion(rig)
        motion.pose = RigPose(angleY: 1)
        let face = try XCTUnwrap(rig.parts.firstIndex { $0.slot == "face" })
        let now = motion.deform(rig, part: face, physics: false)
        let rest = rig.parts[face].mesh.rest
        let rise = { (i: Int) in rest[i].y - now[i].y }
        // The vertex nearest the middle of the face, and one on its outline at the same height.
        let middle = SIMD2<Float>(Float(rig.anchors.faceCenter.x), Float(rig.anchors.faceCenter.y))
        let centre = try XCTUnwrap(rest.indices.min { simd_distance(rest[$0], middle) < simd_distance(rest[$1], middle) })
        let edge = try XCTUnwrap(rest.indices.filter { rest[$0].y == rest[centre].y }.min { rest[$0].x < rest[$1].x })
        XCTAssertGreaterThan(rise(centre), 0)
        let width = Float(rig.anchors.face.width)
        XCTAssertGreaterThan(rise(centre), rise(edge) + 0.05 * width)
        // The mouth goes with the middle of the face, farther than the chin's outline.
        let mouth = try XCTUnwrap(rig.parts.firstIndex { $0.slot == "mouth_open" })
        let mouthNow = motion.deform(rig, part: mouth, physics: false)
        XCTAssertGreaterThan(rig.parts[mouth].mesh.rest[0].y - mouthNow[0].y, rise(rest.count - 1) + 0.02 * width)
        // Down is the other way.
        motion.pose = RigPose(angleY: -1)
        XCTAssertLessThan(rest[centre].y - motion.deform(rig, part: face, physics: false)[centre].y, 0)
    }

    func testTheHairSwingsAfterTheHead() throws {
        let rig = try head()
        let i = try XCTUnwrap(rig.parts.firstIndex { $0.slot == "front hair" })
        var motion = RigMotion(rig)
        let swinging = RigSwitches(blink: false, idle: false, lookAround: false, physics: true, followMouse: true)
        // A sudden turn: the tips lag behind the head, then come to rest with it.
        motion.step(by: 1.0 / 60, rig: rig, switches: swinging, mouse: SIMD2(1.5, 0), open: 1, ease: 0.3)
        let lag = motion.strands[i].map(\.softLag)
        XCTAssertTrue(lag.allSatisfy { $0 > 0 }, "\(lag)")
        for _ in 0..<(60 * 20) { motion.step(by: 1.0 / 60, rig: rig, switches: swinging, mouse: SIMD2(1.5, 0), open: 1, ease: 0.3) }
        XCTAssertTrue(motion.strands[i].allSatisfy { abs($0.softLag) < 0.05 && abs($0.stiffLag) < 0.05 })
    }

    @MainActor
    func testMetalDrawsTheRigAndKeepsTheIrisInsideTheWhite() throws {
        guard let renderer = RigRenderer(pixelFormat: .rgba8Unorm) else { throw XCTSkip("no Metal device") }
        let rig = Rig.build(try face())
        renderer.load(rig)
        // At rest: the eyes open, the closed ones not shown.
        let image = try XCTUnwrap(renderer.image(RigMotion(rig).frame(rig, physics: false), width: 200, height: 200))
        let pixels = try rgba(image)
        let at = { (x: Int, y: Int) in Array(pixels[(y * 200 + x) * 4..<(y * 200 + x) * 4 + 4]) }
        // Nothing drawn in the corner; the face where it is; the iris over the white.
        XCTAssertEqual(at(5, 5), [0, 0, 0, 0])
        XCTAssertEqual(at(100, 140), [250, 220, 200, 255])
        XCTAssertEqual(at(75, 90), [40, 90, 200, 255])
        // The iris layer reaches past the white's bottom only where the white is: shut, there is no
        // iris over the face below the line the eye closes onto.
        var motion = RigMotion(rig)
        motion.blinker.blinkNow()
        let still = RigSwitches(blink: false, idle: false, lookAround: false, physics: false, followMouse: false)
        for _ in 0..<12 { motion.step(by: 1.0 / 60, rig: rig, switches: still, mouse: nil, open: 1, ease: 0.3) }
        let blinking = try rgba(try XCTUnwrap(renderer.image(motion.frame(rig, physics: false), width: 200, height: 200)))
        XCTAssertNotEqual(Array(blinking[(97 * 200 + 75) * 4..<(97 * 200 + 75) * 4 + 3]), [40, 90, 200])
    }

    private func rgba(_ image: CGImage) throws -> [UInt8] {
        var bytes = [UInt8](repeating: 0, count: image.width * image.height * 4)
        let context = try XCTUnwrap(
            CGContext(
                data: &bytes, width: image.width, height: image.height, bitsPerComponent: 8, bytesPerRow: image.width * 4,
                space: CGColorSpace(name: CGColorSpace.sRGB)!, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue))
        context.draw(image, in: CGRect(x: 0, y: 0, width: image.width, height: image.height))
        return bytes
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
        let renderer = try XCTUnwrap(RigRenderer(pixelFormat: .rgba8Unorm))
        renderer.load(rig)
        let white = MTLClearColor(red: 1, green: 1, blue: 1, alpha: 1)
        let draw = { (pose: RigPose) throws -> CGImage in
            var motion = RigMotion(rig)
            motion.pose = pose
            return try XCTUnwrap(renderer.image(motion.frame(rig, physics: false), width: rig.width, height: rig.height, background: white))
        }
        let blink = try [1.0, 0.5, 0.3, 0.15, 0].map { v -> CGImage in
            try XCTUnwrap(try draw(RigPose(openness: Float(v))).cropping(to: eyes.integral))
        }
        try sheet(blink, down: true).write(to: URL(fileURLWithPath: frames).appendingPathComponent("blink.png"))
        // The head turned left, still, right, and up, down.
        let turns = try [
            RigPose(angleX: -1, body: -0.25, bodyTurn: -1), RigPose(), RigPose(angleX: 1, body: 0.25, bodyTurn: 1),
            RigPose(angleY: 1), RigPose(angleY: -1),
        ].map(draw)
        try sheet(turns, down: false).write(to: URL(fileURLWithPath: frames).appendingPathComponent("turn.png"))
    }

    /// `images`, all one size, one under the other or side by side, as a PNG.
    private func sheet(_ images: [CGImage], down: Bool) throws -> Data {
        let (w, h) = (images[0].width, images[0].height)
        let sheet = try XCTUnwrap(
            CGContext(
                data: nil, width: down ? w : w * images.count, height: down ? h * images.count : h, bitsPerComponent: 8,
                bytesPerRow: 0, space: CGColorSpace(name: CGColorSpace.sRGB)!,
                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue))
        for (i, image) in images.enumerated() {
            let at = down ? CGRect(x: 0, y: h * (images.count - 1 - i), width: w, height: h) : CGRect(x: w * i, y: 0, width: w, height: h)
            sheet.draw(image, in: at)
        }
        return try XCTUnwrap(NSBitmapImageRep(cgImage: try XCTUnwrap(sheet.makeImage())).representation(using: .png, properties: [:]))
    }
}
