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

// img2layers as the app does it around the engine: what See-through is described as, the picture
// handed to it with its transparency, and what is kept of the layers that come back. No weights,
// no engine.

import AppKit
import Foundation
import WaifuKit
import XCTest

@testable import Waifu

final class LayersTests: XCTestCase {
    private func manifest(_ stem: String, config: String) throws -> String {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("waifu-layers-\(ProcessInfo.processInfo.processIdentifier)-\(stem)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let path = directory.appendingPathComponent("\(stem).yaml")
        try "weights:\n  - \(stem).safetensors\nconfig:\n  model:\n    type: see_through\n\(config)"
            .write(to: path, atomically: true, encoding: .utf8)
        return path.path
    }

    func testSeeThroughIsALayersModelWithItsOwnNumbers() throws {
        let path = try manifest(
            "see-through", config: "  see_through:\n    layers_steps: 30\n    depth_steps: 4\n    layers_resolution: 1280\n")
        let info = try ModelInfo.describe(path, device: "cpu")
        XCTAssertEqual(info.kind, "layers")
        XCTAssertNil(info.model, "it is not a picture model: no prompt, no size")
        let layerer = try XCTUnwrap(info.layerer)
        XCTAssertEqual(layerer.steps, 30)
        XCTAssertEqual(layerer.depthSteps, 4)
        XCTAssertEqual(layerer.resolution, 1280)
        XCTAssertFalse(info.onDisk)
    }

    /// One pixel each of opaque, half and clear, drawn and read back.
    private func picture(_ rgba: [UInt8], width: Int, height: Int) throws -> NSImage {
        let png = try XCTUnwrap(Media.png(RGBAImage(width: width, height: height, rgba: rgba)))
        return try XCTUnwrap(NSImage(data: png))
    }

    func testAPictureKeepsItsTransparencyAndIsNotPremultiplied() throws {
        let pixels: [UInt8] = [200, 100, 50, 255, 200, 100, 50, 128, 0, 0, 0, 0]
        let rgba = try XCTUnwrap(Media.rgba(picture(pixels, width: 3, height: 1)))
        XCTAssertEqual(rgba.width, 3)
        XCTAssertEqual(rgba.rgba.count, 12)
        for (got, want) in zip(rgba.rgba, pixels) {
            // Premultiplied and back loses up to a level where alpha is low.
            XCTAssertLessThanOrEqual(abs(Int(got) - Int(want)), 2, "\(rgba.rgba) against \(pixels)")
        }
    }

    func testTheLayersArePutBackTogetherWhereTheyWere() throws {
        // A red square at the top left of a 4 x 4 canvas, two by two, one in from each side.
        let red = RGBAImage(width: 2, height: 2, rgba: Array(repeating: [255, 0, 0, 255], count: 4).flatMap { $0 })
        let layer = Layer(name: "face", left: 1, top: 0, image: red, depth: 0.5)
        let png = try XCTUnwrap(Media.composite([layer], canvas: 4))
        let back = try XCTUnwrap(Media.rgba(XCTUnwrap(NSImage(data: png))))
        XCTAssertEqual(back.width, 4)
        let alpha = { (x: Int, y: Int) in back.rgba[(y * 4 + x) * 4 + 3] }
        // Counted down from the top, as the canvas is: not upside down.
        XCTAssertEqual(alpha(1, 0), 255)
        XCTAssertEqual(alpha(2, 1), 255)
        XCTAssertEqual(alpha(0, 0), 0)
        XCTAssertEqual(alpha(1, 3), 0)
    }

    func testWhatIsKeptOfTheLayersNamesAFileForEach() throws {
        let one = RGBAImage(width: 1, height: 1, rgba: [1, 2, 3, 255])
        let taken = Layers(
            canvas: 8,
            layers: [
                Layer(name: "back hair", left: 0, top: 0, image: one, depth: 0.9),
                Layer(name: "face", left: 3, top: 2, image: one, depth: 0.4),
            ],
            psd: Data([1]), depthPSD: Data([2]))
        let (kept, beside) = Session.files(of: taken)
        XCTAssertEqual(kept.map(\.name), ["back hair", "face"])
        XCTAssertEqual(kept[1].left, 3)
        XCTAssertEqual(kept[1].top, 2)
        for layer in kept { XCTAssertNotNil(beside[layer.file], layer.file) }
        XCTAssertEqual(beside[Session.depthDocument], Data([2]))
        XCTAssertNotNil(beside[Session.composite])

        // And read back as a record, the way the strip lists it.
        var made = Made(seed: 7, seconds: 1, model: "see-through:v0.0.2", parameters: "")
        made.layers = kept
        made.canvas = 8
        let record = Record(id: "x", kind: "layers", created: 0, finished: 1, made: made)
        let again = try JSONDecoder().decode(Record.self, from: JSONEncoder().encode(record))
        XCTAssertEqual(again.made.layers, kept)
        XCTAssertEqual(again.made.canvas, 8)
    }

    private func kept(_ name: String, _ depth: Float, _ file: String) -> KeptLayer {
        KeptLayer(name: name, left: 0, top: 0, width: 1, height: 1, depth: depth, file: file)
    }

    func testLayersAreLaidFarthestFirstAndTiesKeepTheirOrder() {
        let order = Session.ordered([kept("a", 0.2, "0"), kept("b", 0.9, "1"), kept("c", 0.2, "2"), kept("d", 0.5, "3")])
        XCTAssertEqual(order.map(\.name), ["b", "d", "a", "c"])
    }

    /// A picture taken apart as it is kept -- its document, its depth document, its record -- with
    /// one layer renamed and another given a depth that puts it on top: both documents are written
    /// again in the new order under the new names, every pixel as it was.
    func testChangedLayersAreWrittenAgainPixelForPixel() throws {
        let folder = FileManager.default.temporaryDirectory
            .appendingPathComponent("waifu-rewrite-\(ProcessInfo.processInfo.processIdentifier)")
        let besides = folder.appendingPathComponent("x.layers")
        try FileManager.default.createDirectory(at: besides, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: folder) }

        let red = RGBAImage(width: 2, height: 1, rgba: [255, 0, 0, 255, 200, 0, 0, 100])
        let blue = RGBAImage(width: 1, height: 2, rgba: [0, 0, 255, 255, 0, 0, 250, 30])
        let near = RGBAImage(width: 1, height: 1, rgba: [70, 70, 70, 255])
        let far = RGBAImage(width: 1, height: 1, rgba: [180, 180, 180, 255])
        let document = folder.appendingPathComponent("x.psd")
        // With its depth in the hidden group, as img2layers writes it: red the gray.
        let gray = { (image: RGBAImage) in
            RGBAImage(width: image.width, height: image.height, rgba: image.rgba.enumerated().map { $0.offset % 4 == 3 ? $0.element : image.rgba[$0.offset / 4 * 4] })
        }
        try PSD.write(
            [PSD.Layer(name: "back hair", left: 0, top: 0, image: red), PSD.Layer(name: "face", left: 3, top: 1, image: blue)],
            width: 4, height: 4,
            depths: [
                PSD.Layer(name: PSD.depthName(of: "back hair"), left: 0, top: 0, image: gray(far)),
                PSD.Layer(name: PSD.depthName(of: "face"), left: 3, top: 1, image: gray(near)),
            ]
        ).write(to: document)
        try PSD.write(
            [PSD.Layer(name: "back hair", left: 0, top: 0, image: far), PSD.Layer(name: "face", left: 3, top: 1, image: near)],
            width: 4, height: 4, grayscale: true
        ).write(to: besides.appendingPathComponent(Session.depthDocument))

        let old = [kept("back hair", 0.8, "00.png"), kept("face", 0.4, "01.png")]
        var changed = old
        changed[0].name = "hair"
        changed[0].depth = 0.1
        let new = Session.ordered(changed)
        XCTAssertEqual(new.map(\.name), ["face", "hair"])
        var made = Made(seed: 1, seconds: 0, model: "see-through", parameters: "")
        made.layers = new
        let recordFile = folder.appendingPathComponent("x.json")
        let (psd, composite) = try Session.rewritten(
            document: document, besides: besides,
            record: Record(id: "x", kind: "layers", created: 0, finished: 0, made: made), recordFile: recordFile,
            from: old, to: new)

        let again = try PSDDocument.read(psd)
        let (shown, depths) = again.splitDepth()
        XCTAssertEqual(shown.map(\.name), ["face", "hair"])
        XCTAssertEqual(shown[0].image, blue)
        XCTAssertEqual(shown[1].image, red)
        XCTAssertEqual([shown[0].left, shown[0].top], [3, 1])
        // The depth in the hidden group, in the new order under the new names, every pixel as it was.
        XCTAssertEqual(depths.map(\.name), ["depth/face", "depth/hair"])
        XCTAssertEqual(depths.map(\.visible), [false, false])
        XCTAssertEqual(depths[0].image, gray(near))
        XCTAssertEqual(again.depth(of: "hair")?.image, gray(far))
        XCTAssertEqual(try PSDDocument.read(Data(contentsOf: document)).splitDepth().layers.map(\.name), ["face", "hair"])

        // The depth document in the same order, its gray read into red, green and blue.
        let depth = try PSDDocument.read(Data(contentsOf: besides.appendingPathComponent(Session.depthDocument)))
        XCTAssertEqual(depth.layers.map(\.name), ["face", "hair"])
        XCTAssertEqual(depth.layers[0].image, near)
        XCTAssertEqual(depth.layers[1].image, far)

        XCTAssertNotNil(composite)
        XCTAssertTrue(FileManager.default.fileExists(atPath: besides.appendingPathComponent(Session.composite).path))
        let record = try JSONDecoder().decode(Record.self, from: Data(contentsOf: recordFile))
        XCTAssertEqual(record.made.layers?.map(\.name), ["face", "hair"])
    }
}
