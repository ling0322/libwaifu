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

// A Photoshop document written: as little of the format as a stack of pixel layers needs. The
// library hands over a picture taken apart as pixels; the documents are this app's to write. A
// port of waifu/src/see_through/psd.rs, which wrote them before, byte for byte.
//
// Version 1, eight bits a channel, RGB or grayscale, each with an alpha. Every layer is a rectangle
// of pixels with a name and a transparency channel, at full opacity, blended normally; the only
// group is the hidden one a document keeps its layers' depth in. Channels are compressed with
// PackBits, which is what Photoshop itself writes, and the merged picture after the layers -- what
// a viewer that does not read layers shows -- is the layers flattened in order, its alpha the
// document's first extra channel. The layer count is written negative to say so: with a positive
// count a reader takes that channel for an ordinary saved selection and shows the merged picture
// opaque.
//
// The layout is Adobe's "Photoshop File Formats Specification", sections Header, Color Mode Data,
// Image Resources, Layer and Mask Information, and Image Data, all big-endian. PSD.swift reads it.

import Foundation
import WaifuKit

enum PSD {
    struct Layer: Sendable {
        var name: String
        var left: Int
        var top: Int
        /// RGBA, not premultiplied. In a grayscale document, red is the gray.
        var image: RGBAImage

        init(name: String, left: Int, top: Int, image: RGBAImage) {
            self.name = name
            self.left = left
            self.top = top
            self.image = image
        }
    }

    /// The group a document keeps its layers' depth in, hidden, after them: img2layers' `depth`.
    static let depthGroup = "depth"

    /// What a layer's depth is called in that group: `depth/topwear`.
    static func depthName(of name: String) -> String { "\(depthGroup)/\(name)" }

    /// The name a group's end is written under, as Photoshop writes it.
    private static let groupEnd = "</Layer group>"

    /// A document of `layers`, bottom first, `width` by `height`: RGB, or grayscale as the depth
    /// document is. `depths`, in an RGB one, are kept after them in the hidden group `depth`; the
    /// merged picture is of `layers` alone. Throws for a layer whose pixels are not its size.
    static func write(
        _ layers: [Layer], width: Int, height: Int, grayscale: Bool = false, depths: [Layer] = []
    ) throws -> Data {
        guard width >= 0, height >= 0 else { throw PSDError(message: "a document of \(width) by \(height)") }
        if grayscale && !depths.isEmpty { throw PSDError(message: "depth is kept in an RGB document only") }
        for layer in layers + depths {
            let image = layer.image
            guard image.width >= 0, image.height >= 0, image.rgba.count == image.width * image.height * 4 else {
                throw PSDError(message: "layer \(layer.name)'s pixels are not \(image.width) by \(image.height) of RGBA")
            }
        }
        // The colour channels of a layer's RGBA: red, green and blue, or red alone as the gray.
        let colour = grayscale ? 1 : 3

        // Every record, bottom first: the layers, then -- for a group -- its end, what is in it,
        // and the group itself, which a document has after what it holds.
        struct Entry {
            var name: String
            var layer: Layer?
            var hidden: Bool
            /// `lsct`: 0 for a layer, 2 for a closed group, 3 for a group's end.
            var divider: UInt32
        }
        var entries = layers.map { Entry(name: $0.name, layer: $0, hidden: false, divider: 0) }
        if !depths.isEmpty {
            entries.append(Entry(name: groupEnd, layer: nil, hidden: true, divider: 3))
            entries += depths.map { Entry(name: $0.name, layer: $0, hidden: true, divider: 0) }
            entries.append(Entry(name: depthGroup, layer: nil, hidden: true, divider: 2))
        }

        var w = Writer()
        w.ascii("8BPS")
        w.u16(1)
        w.bytes(Array(repeating: 0, count: 6))
        // The merged picture carries its alpha too, so a viewer that reads only it still sees
        // what is transparent.
        w.u16(UInt16(colour + 1))
        w.u32(UInt32(height))
        w.u32(UInt32(width))
        w.u16(8)
        w.u16(grayscale ? 1 : 3)

        w.u32(0)  // colour mode data
        w.u32(0)  // image resources

        let layerAndMask = w.reserve()
        let layerInfo = w.reserve()
        // Negative: the merged picture's first extra channel is its transparency.
        w.i16(-Int16(entries.count))

        // The channels of every layer, alpha first, compressed now so their lengths are known for
        // the records that come before them. A group's records have no pixels: a compression code
        // each.
        let channelData: [[(id: Int16, data: [UInt8])]] = entries.map { entry in
            guard let layer = entry.layer else {
                return (-1..<Int16(colour)).map { (id: $0, data: [0, 0]) }
            }
            let image = layer.image
            var channels = [(id: Int16(-1), data: rle(plane(image, 3), image.width, image.height, rowsOnly: false))]
            for c in 0..<colour {
                channels.append((id: Int16(c), data: rle(plane(image, c), image.width, image.height, rowsOnly: false)))
            }
            return channels
        }

        for (entry, channels) in zip(entries, channelData) {
            if let layer = entry.layer {
                w.i32(Int32(clamping: layer.top))
                w.i32(Int32(clamping: layer.left))
                w.i32(Int32(clamping: layer.top + layer.image.height))
                w.i32(Int32(clamping: layer.left + layer.image.width))
            } else {
                for _ in 0..<4 { w.i32(0) }
            }
            w.u16(UInt16(channels.count))
            for channel in channels {
                w.i16(channel.id)
                w.u32(UInt32(channel.data.count))
            }
            w.ascii("8BIM")
            w.ascii("norm")
            w.u8(255)  // opacity
            w.u8(0)  // clipping: base
            // Flags: 2 is hidden; a group's records also say their pixels mean nothing (8 | 16).
            var flags: UInt8 = entry.hidden ? 2 : 0
            if entry.divider != 0 { flags |= 0x18 }
            w.u8(flags)
            w.u8(0)  // filler

            let extra = w.reserve()
            w.u32(0)  // layer mask data
            w.u32(0)  // blending ranges

            // The name, as a Pascal string padded to a multiple of four with its length byte.
            let name = Array(Array(entry.name.utf8).prefix(255))
            w.u8(UInt8(name.count))
            w.bytes(name)
            for _ in 0..<((4 - (name.count + 1) % 4) % 4) { w.u8(0) }
            // The name again in UTF-16, which is what Photoshop shows for a name that is not ASCII.
            let units = Array(entry.name.utf16)
            w.ascii("8BIMluni")
            w.u32(UInt32(4 + units.count * 2))
            w.u32(UInt32(units.count))
            for unit in units { w.u16(unit) }
            if entry.divider != 0 {
                w.ascii("8BIMlsct")
                w.u32(4)
                w.u32(entry.divider)
            }
            w.fill(extra)
        }
        for channels in channelData {
            for channel in channels { w.bytes(channel.data) }
        }
        if (w.count - layerInfo) % 2 != 0 { w.u8(0) }
        w.fill(layerInfo)
        w.u32(0)  // global layer mask info
        w.fill(layerAndMask)

        // The merged picture: one compression code, then every channel's row counts, then the rows.
        let planes = flatten(layers, width: width, height: height, colour: colour)
        w.u16(1)
        let compressed = planes.map { rle($0, width, height, rowsOnly: true) }
        for p in compressed { w.bytes(Array(p[..<(height * 2)])) }
        for p in compressed { w.bytes(Array(p[(height * 2)...])) }
        return Data(w.out)
    }

    /// One channel of an RGBA picture, as a plane of its own.
    private static func plane(_ image: RGBAImage, _ channel: Int) -> [UInt8] {
        var plane = [UInt8](repeating: 0, count: image.width * image.height)
        for p in plane.indices { plane[p] = image.rgba[p * 4 + channel] }
        return plane
    }

    /// One row in PackBits: runs of a repeated byte as `(1 - n, byte)`, everything else as
    /// `(n - 1, bytes...)`, at most 128 either way.
    static func packBits(_ bytes: [UInt8], from start: Int, count: Int, into out: inout [UInt8]) {
        let end = start + count
        var i = start
        while i < end {
            var run = 1
            while i + run < end && run < 128 && bytes[i + run] == bytes[i] { run += 1 }
            if run >= 2 {
                out.append(UInt8(bitPattern: Int8(1 - run)))
                out.append(bytes[i])
                i += run
                continue
            }
            let literal = i
            while i < end && i - literal < 128 {
                if i + 1 < end && bytes[i + 1] == bytes[i] { break }
                i += 1
            }
            if i == literal { i += 1 }
            out.append(UInt8(i - literal - 1))
            out.append(contentsOf: bytes[literal..<i])
        }
    }

    /// One channel of `width` by `height`, compressed: the compression code, every row's byte
    /// count, then the rows. `rowsOnly` leaves off the code, which the merged picture writes once
    /// for all of its channels.
    private static func rle(_ plane: [UInt8], _ width: Int, _ height: Int, rowsOnly: Bool) -> [UInt8] {
        var counts: [UInt8] = []
        counts.reserveCapacity(height * 2)
        var data: [UInt8] = []
        for y in 0..<height {
            let before = data.count
            packBits(plane, from: y * width, count: width, into: &data)
            let length = UInt16(data.count - before)
            counts.append(UInt8(length >> 8))
            counts.append(UInt8(length & 0xFF))
        }
        return (rowsOnly ? [] : [0, 1]) + counts + data
    }

    /// The layers laid over each other in order on a `width` by `height` canvas, over transparent
    /// black, as planes: colour, then alpha.
    private static func flatten(_ layers: [Layer], width: Int, height: Int, colour n: Int) -> [[UInt8]] {
        var colour = [[Float]](repeating: [Float](repeating: 0, count: width * height), count: n)
        var alpha = [Float](repeating: 0, count: width * height)
        for layer in layers {
            let image = layer.image
            for y in 0..<image.height {
                let cy = layer.top + y
                if cy < 0 || cy >= height { continue }
                for x in 0..<image.width {
                    let cx = layer.left + x
                    if cx < 0 || cx >= width { continue }
                    let at = cy * width + cx
                    let pixel = (y * image.width + x) * 4
                    let a = Float(image.rgba[pixel + 3]) / 255
                    let outA = a + alpha[at] * (1 - a)
                    for c in 0..<n {
                        let v = Float(image.rgba[pixel + c])
                        colour[c][at] = outA > 0 ? (v * a + colour[c][at] * alpha[at] * (1 - a)) / outA : 0
                    }
                    alpha[at] = outA
                }
            }
        }
        let byte = { (v: Float) in UInt8(min(max(v.rounded(), 0), 255)) }
        return colour.map { $0.map(byte) } + [alpha.map { byte($0 * 255) }]
    }

    /// Big-endian bytes, and lengths filled in once what they measure is written.
    private struct Writer {
        var out: [UInt8] = []

        var count: Int { out.count }

        mutating func u8(_ v: UInt8) { out.append(v) }
        mutating func u16(_ v: UInt16) { out += [UInt8(v >> 8), UInt8(v & 0xFF)] }
        mutating func i16(_ v: Int16) { u16(UInt16(bitPattern: v)) }
        mutating func u32(_ v: UInt32) {
            out += [UInt8(v >> 24), UInt8((v >> 16) & 0xFF), UInt8((v >> 8) & 0xFF), UInt8(v & 0xFF)]
        }
        mutating func i32(_ v: Int32) { u32(UInt32(bitPattern: v)) }
        mutating func bytes(_ v: [UInt8]) { out += v }
        mutating func ascii(_ s: String) { out += Array(s.utf8) }

        /// A length to be filled in once what it measures is written: where it ends.
        mutating func reserve() -> Int {
            u32(0)
            return out.count
        }

        mutating func fill(_ at: Int) {
            let length = UInt32(out.count - at)
            out[at - 4] = UInt8(length >> 24)
            out[at - 3] = UInt8((length >> 16) & 0xFF)
            out[at - 2] = UInt8((length >> 8) & 0xFF)
            out[at - 1] = UInt8(length & 0xFF)
        }
    }
}
