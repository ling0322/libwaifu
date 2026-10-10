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

// Turning what was dropped into what the engine takes, and what it hands back into files.
//
// The engine takes pixels and samples, not files: AppKit and AVFoundation decode whatever this Mac
// can open, so a HEIC or an m4a is as good as a PNG or a WAV.

import AVFoundation
import AppKit
import UniformTypeIdentifiers
import WaifuKit

/// Why something dropped could not be used, in words to show under it.
struct MediaError: LocalizedError {
    var message: String
    var errorDescription: String? { message }
}

enum Media {
    /// How much of a recording to sound like is kept: a sentence says how a voice sounds.
    static let recordingSeconds = 30.0
    /// And of a recording to convert, which is the whole of what is said again.
    static let sourceSeconds = 300.0

    /// A picture's bytes as they are kept: PNG and JPEG as they are, anything else NSImage can
    /// read as a PNG.
    static func picture(at url: URL) throws -> Data {
        let bytes = try Data(contentsOf: url)
        if bytes.starts(with: [0x89, 0x50, 0x4E, 0x47]) || bytes.starts(with: [0xFF, 0xD8, 0xFF]) {
            return bytes
        }
        guard let png = NSImage(data: bytes).flatMap(png) else {
            throw MediaError(message: "\(url.lastPathComponent) is not a picture this Mac can read")
        }
        return png
    }

    static func png(_ image: NSImage) -> Data? {
        guard let tiff = image.tiffRepresentation, let bitmap = NSBitmapImageRep(data: tiff) else {
            return nil
        }
        return bitmap.representation(using: .png, properties: [:])
    }

    /// A picture as the engine takes it: 8-bit RGB, row by row, drawn at its pixel size -- not
    /// its point size, which on a Retina picture is half.
    static func rgb(_ image: NSImage) -> RGBImage? {
        guard let cg = image.cgImage(forProposedRect: nil, context: nil, hints: nil) else { return nil }
        let width = cg.width
        let height = cg.height
        guard width > 0, height > 0 else { return nil }
        var rgba = [UInt8](repeating: 0, count: width * height * 4)
        let drawn = rgba.withUnsafeMutableBytes { buffer -> Bool in
            guard
                let context = CGContext(
                    data: buffer.baseAddress, width: width, height: height, bitsPerComponent: 8,
                    bytesPerRow: width * 4, space: CGColorSpace(name: CGColorSpace.sRGB)!,
                    bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)
            else { return false }
            context.draw(cg, in: CGRect(x: 0, y: 0, width: width, height: height))
            return true
        }
        guard drawn else { return nil }
        var rgb = [UInt8](repeating: 0, count: width * height * 3)
        for at in 0..<(width * height) {
            rgb[at * 3] = rgba[at * 4]
            rgb[at * 3 + 1] = rgba[at * 4 + 1]
            rgb[at * 3 + 2] = rgba[at * 4 + 2]
        }
        return RGBImage(width: width, height: height, rgb: rgb)
    }

    /// What the engine drew, as the bytes of a PNG file.
    static func png(_ image: RGBImage) -> Data? {
        guard image.width > 0, image.height > 0, image.rgb.count == image.width * image.height * 3,
            let bitmap = NSBitmapImageRep(
                bitmapDataPlanes: nil, pixelsWide: image.width, pixelsHigh: image.height,
                bitsPerSample: 8, samplesPerPixel: 3, hasAlpha: false, isPlanar: false,
                colorSpaceName: .deviceRGB, bytesPerRow: image.width * 3, bitsPerPixel: 24),
            let pixels = bitmap.bitmapData
        else { return nil }
        image.rgb.withUnsafeBytes { pixels.update(from: $0.bindMemory(to: UInt8.self).baseAddress!, count: $0.count) }
        return bitmap.representation(using: .png, properties: [:])
    }

    /// Any recording AVFoundation can play, as mono samples of at most `seconds` of it.
    static func clip(from url: URL, seconds: Double) throws -> AudioClip {
        let file: AVAudioFile
        do {
            file = try AVAudioFile(forReading: url)
        } catch {
            throw MediaError(
                message: "\(url.lastPathComponent) could not be read: anything this Mac can play "
                    + "will work -- wav, mp3, m4a, aac, flac, aiff")
        }

        let format = file.processingFormat
        let rate = format.sampleRate
        let frames = AVAudioFrameCount(min(Double(file.length), rate * seconds))
        guard frames > 0, let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frames)
        else {
            throw MediaError(message: "\(url.lastPathComponent) has no sound in it")
        }
        try file.read(into: buffer, frameCount: frames)
        guard let channels = buffer.floatChannelData else {
            throw MediaError(message: "\(url.lastPathComponent) could not be decoded to samples")
        }

        // The mean of the channels rather than the sum: the same sound in two channels added
        // together is that sound at twice the amplitude, which clips.
        let count = Int(format.channelCount)
        let length = Int(buffer.frameLength)
        var mono = [Float](repeating: 0, count: length)
        for channel in 0..<count {
            let samples = channels[channel]
            for at in 0..<length { mono[at] += samples[at] }
        }
        if count > 1 {
            for at in 0..<length { mono[at] /= Float(count) }
        }

        return AudioClip(samples: mono, rate: Int(rate.rounded()))
    }

    /// Samples and a rate, as the bytes of a 16-bit mono WAV file.
    static func wav(_ samples: [Float], rate: Int) -> Data {
        var bytes = Data(capacity: 44 + samples.count * 2)
        func put<T: FixedWidthInteger>(_ value: T) {
            withUnsafeBytes(of: value.littleEndian) { bytes.append(contentsOf: $0) }
        }
        let size = UInt32(samples.count * 2)

        bytes.append(contentsOf: Array("RIFF".utf8))
        put(36 + size)
        bytes.append(contentsOf: Array("WAVEfmt ".utf8))
        put(UInt32(16))
        put(UInt16(1))
        put(UInt16(1))
        put(UInt32(rate))
        put(UInt32(rate * 2))
        put(UInt16(2))
        put(UInt16(16))
        bytes.append(contentsOf: Array("data".utf8))
        put(size)
        for sample in samples {
            // Held inside the range before it is scaled: the top of it wraps to the bottom,
            // which is a click, and a loud one.
            put(Int16((max(-1, min(1, sample)) * 32767).rounded()))
        }
        return bytes
    }

    /// What a thing is called when it is saved: when it was made, to the second.
    static func fileName(created milliseconds: Double, kind: String) -> String {
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyyMMdd-HHmmss"
        let stamp = formatter.string(from: Date(timeIntervalSince1970: milliseconds / 1000))
        return "waifu-\(stamp).\(kind == "image" ? "png" : "wav")"
    }

    /// Asks where to save `bytes`, and saves them there.
    @MainActor
    static func save(_ bytes: Data, suggesting name: String) {
        let panel = NSSavePanel()
        panel.nameFieldStringValue = name
        panel.allowedContentTypes = [name.hasSuffix(".png") ? .png : .wav]
        panel.canCreateDirectories = true
        guard panel.runModal() == .OK, let url = panel.url else { return }
        do {
            try bytes.write(to: url)
        } catch {
            NSAlert(error: error).runModal()
        }
    }

    /// Asks for a file of one of `types`.
    @MainActor
    static func choose(_ types: [UTType]) -> URL? {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = types
        panel.allowsMultipleSelection = false
        return panel.runModal() == .OK ? panel.url : nil
    }
}
