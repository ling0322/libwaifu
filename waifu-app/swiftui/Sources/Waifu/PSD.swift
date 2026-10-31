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

// A Photoshop document's layers, read: as much of the format as a drawing in layers needs.
//
// Version 1 -- the .psd Photoshop saves; a .psb is version 2 -- eight bits a channel, RGB, or
// grayscale read as RGB. Each
// layer is a rectangle of pixels with a name, an opacity and whether it is shown; channel data
// raw or PackBits. Groups are read through: a layer in a hidden group is hidden, and the group's
// own records are not layers. Masks, effects and blend modes are left out. The layout is Adobe's
// "Photoshop File Formats Specification", big-endian throughout; waifu/src/see_through/psd.rs
// writes the same format, which img2layers hands out.

import Foundation
import WaifuKit

struct PSDError: LocalizedError {
    var message: String
    var errorDescription: String? { message }
}

/// One layer: where its top left corner is on the canvas, its pixels -- RGBA, not premultiplied,
/// as large as its rectangle -- and how it is shown.
struct PSDLayer {
    var name: String
    var left: Int
    var top: Int
    var image: RGBAImage
    /// 0 to 1.
    var opacity: Double
    /// Shown: neither it nor a group it is in is hidden.
    var visible: Bool

    var width: Int { image.width }
    var height: Int { image.height }
}

struct PSDDocument {
    var width: Int
    var height: Int
    /// Bottom first, as they are drawn.
    var layers: [PSDLayer]

    static func read(_ data: Data) throws -> PSDDocument {
        var reader = PSDReader(bytes: [UInt8](data))
        return try reader.document()
    }

    /// Its layers, and apart from them the depth img2layers keeps in its hidden group: a hidden
    /// layer named `depth/<name>`, red its gray, 0 near to 255 far. Both in the document's order.
    func splitDepth() -> (layers: [PSDLayer], depths: [PSDLayer]) {
        let prefix = PSD.depthName(of: "")
        let isDepth = { (layer: PSDLayer) in !layer.visible && layer.name.hasPrefix(prefix) }
        return (layers.filter { !isDepth($0) }, layers.filter(isDepth))
    }

    /// The depth of the layer named `name`, where the document keeps one.
    func depth(of name: String) -> PSDLayer? {
        splitDepth().depths.first { $0.name == PSD.depthName(of: name) }
    }
}

/// Reads one document front to back.
private struct PSDReader {
    var bytes: [UInt8]
    var at = 0
    /// Whether the document is grayscale rather than RGB.
    var gray = false

    mutating func need(_ count: Int) throws {
        guard count >= 0, at + count <= bytes.count else {
            throw PSDError(message: "the document ends where more of it was expected")
        }
    }

    mutating func u8() throws -> Int {
        try need(1)
        defer { at += 1 }
        return Int(bytes[at])
    }

    mutating func u16() throws -> Int {
        try need(2)
        defer { at += 2 }
        return Int(bytes[at]) << 8 | Int(bytes[at + 1])
    }

    mutating func i16() throws -> Int { Int(Int16(bitPattern: UInt16(try u16()))) }

    mutating func u32() throws -> Int {
        try need(4)
        defer { at += 4 }
        return Int(bytes[at]) << 24 | Int(bytes[at + 1]) << 16 | Int(bytes[at + 2]) << 8 | Int(bytes[at + 3])
    }

    mutating func i32() throws -> Int { Int(Int32(bitPattern: UInt32(try u32()))) }

    mutating func skip(_ count: Int) throws {
        try need(count)
        at += count
    }

    mutating func ascii(_ count: Int) throws -> String {
        try need(count)
        defer { at += count }
        return String(decoding: bytes[at..<at + count], as: UTF8.self)
    }

    mutating func document() throws -> PSDDocument {
        guard try ascii(4) == "8BPS" else { throw PSDError(message: "this is not a Photoshop document") }
        switch try u16() {
        case 1: break
        case 2: throw PSDError(message: "this is a large document (.psb): save it as a .psd")
        case let version: throw PSDError(message: "this is a Photoshop document of version \(version), which is not read here")
        }
        try skip(6)
        _ = try u16()  // channels of the merged picture
        let height = try u32()
        let width = try u32()
        let depth = try u16()
        let mode = try u16()
        guard depth == 8 else { throw PSDError(message: "this document has \(depth) bits a channel: save it with 8") }
        // Grayscale is img2layers' depth document's: its gray is read into red, green and blue.
        guard mode == 3 || mode == 1 else { throw PSDError(message: "this document is not RGB: convert it to RGB colour") }
        gray = mode == 1

        try skip(try u32())  // colour mode data
        try skip(try u32())  // image resources
        let section = try u32()
        let end = at + section
        guard section > 0 else { return PSDDocument(width: width, height: height, layers: []) }
        let layers = try layerInfo()
        at = end
        return PSDDocument(width: width, height: height, layers: layers)
    }

    /// A layer's record: everything but its pixels, which come after every record.
    struct Record {
        var name: String
        var top, left, bottom, right: Int
        var channels: [(id: Int, length: Int)]
        var opacity: Int
        var hidden: Bool
        /// 1 or 2 opens a group -- going up the stack, its last record -- and 3 closes one.
        var divider: Int
    }

    mutating func layerInfo() throws -> [PSDLayer] {
        let length = try u32()
        guard length > 0 else { return [] }
        let count = abs(try i16())
        var records: [Record] = []
        for _ in 0..<count { records.append(try record()) }

        var pixels: [RGBAImage?] = []
        for record in records { pixels.append(try channels(of: record)) }

        // Top first, so that a group's record comes before what is in it: a stack of whether each
        // open group is hidden.
        var layers: [PSDLayer] = []
        var groups: [Bool] = []
        for (record, image) in zip(records, pixels).reversed() {
            switch record.divider {
            case 1, 2:
                groups.append(record.hidden || groups.last == true)
                continue
            case 3:
                _ = groups.popLast()
                continue
            default:
                break
            }
            guard let image else { continue }
            layers.append(
                PSDLayer(
                    name: record.name, left: record.left, top: record.top, image: image,
                    opacity: Double(record.opacity) / 255, visible: !record.hidden && groups.last != true))
        }
        return layers.reversed()
    }

    mutating func record() throws -> Record {
        let top = try i32(), left = try i32(), bottom = try i32(), right = try i32()
        var channels: [(id: Int, length: Int)] = []
        for _ in 0..<(try u16()) { channels.append((try i16(), try u32())) }
        try skip(8)  // "8BIM" and the blend mode
        let opacity = try u8()
        _ = try u8()  // clipping
        let flags = try u8()
        try skip(1)
        let extra = try u32()
        let end = at + extra
        try skip(try u32())  // mask
        try skip(try u32())  // blending ranges
        let short = try u8()
        var name = try ascii(short)
        try skip((4 - (short + 1) % 4) % 4)
        var divider = 0
        // What else the layer says, by key: its name in full, and whether it opens or closes a group.
        while at + 12 <= end {
            let signature = try ascii(4)
            guard signature == "8BIM" || signature == "8B64" else { break }
            let key = try ascii(4)
            let size = try u32()
            let start = at
            switch key {
            case "luni":
                let characters = try u32()
                try need(characters * 2)
                var units: [UInt16] = []
                for i in 0..<characters { units.append(UInt16(bytes[at + i * 2]) << 8 | UInt16(bytes[at + i * 2 + 1])) }
                name = String(decoding: units, as: UTF16.self)
            case "lsct", "lsdk":
                divider = try u32()
            default:
                break
            }
            at = start + size
        }
        at = end
        return Record(
            name: name, top: top, left: left, bottom: bottom, right: right, channels: channels,
            opacity: opacity, hidden: flags & 2 != 0, divider: divider)
    }

    /// A layer's pixels, out of its channels: nil for one with no area, a group's record among them.
    mutating func channels(of record: Record) throws -> RGBAImage? {
        let width = record.right - record.left
        let height = record.bottom - record.top
        let area = max(0, width) * max(0, height)
        var rgba = area > 0 ? [UInt8](repeating: 255, count: area * 4) : []
        for channel in record.channels {
            let end = at + channel.length
            // Its pixels are its rectangle's: a mask's are not, and are not read.
            if area > 0, (-1...(gray ? 0 : 2)).contains(channel.id) {
                let plane = try self.plane(width: width, height: height, length: channel.length)
                let offsets = channel.id == -1 ? [3] : gray ? [0, 1, 2] : [channel.id]
                for i in 0..<area {
                    for offset in offsets { rgba[i * 4 + offset] = plane[i] }
                }
            }
            at = end
        }
        return area > 0 ? RGBAImage(width: width, height: height, rgba: rgba) : nil
    }

    /// One channel's `width` by `height` bytes: raw, or PackBits row by row.
    mutating func plane(width: Int, height: Int, length: Int) throws -> [UInt8] {
        let compression = try u16()
        switch compression {
        case 0:
            try need(width * height)
            defer { at += width * height }
            return Array(bytes[at..<at + width * height])
        case 1:
            var counts: [Int] = []
            for _ in 0..<height { counts.append(try u16()) }
            var plane = [UInt8](repeating: 0, count: width * height)
            for row in 0..<height {
                let end = at + counts[row]
                try need(counts[row])
                var x = 0
                while at < end && x < width {
                    let n = Int(Int8(bitPattern: bytes[at]))
                    at += 1
                    if n >= 0 {
                        let run = min(n + 1, width - x, end - at)
                        for i in 0..<run { plane[row * width + x + i] = bytes[at + i] }
                        at += n + 1
                        x += run
                    } else if n != -128 {
                        guard at < end else { break }
                        let run = min(1 - n, width - x)
                        for i in 0..<run { plane[row * width + x + i] = bytes[at] }
                        at += 1
                        x += run
                    }
                }
                at = end
            }
            return plane
        default:
            throw PSDError(message: "a layer is compressed with ZIP, which is not read here: save it without")
        }
    }
}
