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

// A drawing in layers made to blink, the way Anime2.5DRig (https://github.com/852wa/Anime2.5DRig,
// MIT, Copyright (c) 2026 hakoniwa) does it: the names its layers are known by, and its numbers.
//
// A blink is not a lid. The open eye -- its white, iris and lashes -- is squashed toward a
// line across it, and when it is a quarter open it fades into a closed eye drawn apart: the
// document's own, or a stock one (ClosedEyes) fitted to the eye and tinted to its lashes. The
// iris stays inside the white as the white is squashed. Which eye is which is the picture's: an
// eye layer drawn as one is cut in two at the middle of the face.

import CoreGraphics
import Foundation
import WaifuKit

/// Which eye, as the picture has them: `.left` is the character's right.
enum Side: CaseIterable {
    case left, right
}

/// What a layer is, for the rig: the parts of an eye, a brow, the face, and everything else.
enum Role: Equatable {
    case eyewhite, irides, eyelash, eyeClose, eyebrow, face, other
}

struct RigPart {
    var name: String
    var role: Role
    var side: Side?
    /// Where it is on the canvas.
    var rect: CGRect
    var image: CGImage
    var opacity: Double
}

/// An eye at rest: the box of its white, the middle of its iris, and the line it closes onto.
struct EyeAnchor: Equatable {
    var white: CGRect
    var iris: CGPoint
    var closeY: CGFloat
}

struct Rig {
    var width: Int
    var height: Int
    /// Bottom first, the hidden ones left out.
    var parts: [RigPart]
    var eyes: [Side: EyeAnchor]
    /// How wide the face is over 333, which a brow's dip is scaled by.
    var faceScale: CGFloat
    /// What it could not do, to say under the picture.
    var notes: [String]

    /// The white of `side`'s eye, which its iris is kept inside.
    func white(_ side: Side) -> RigPart? {
        parts.last { $0.role == .eyewhite && $0.side == side }
    }
}

// MARK: - names

extension Rig {
    /// The names each role is known by once a name is folded (see `fold`): Anime2.5DRig's, with
    /// See-through's among them.
    static let aliases: [(Role, [String])] = [
        (.eyewhite, ["eyewhite", "eyewhites", "sclera", "白目"]),
        (.irides, ["irides", "iris", "irises", "pupil", "pupils", "瞳", "黒目", "虹彩"]),
        (.eyelash, ["eyelash", "eyelashes", "lash", "lashes", "eyeline", "まつ毛", "まつげ", "睫毛", "アイライン"]),
        (.eyeClose, ["eyeclose", "eyeclosed", "closedeye", "closedeyes", "eyelashc", "閉じ目", "目閉じ", "閉眼"]),
        (.eyebrow, ["eyebrow", "eyebrows", "brow", "brows", "眉", "眉毛"]),
        (.face, ["face", "顔"]),
    ]

    /// A layer's name as the aliases are written: compatibility-normalized, lower case, without a
    /// copy's suffix or the marks between words.
    static func fold(_ name: String) -> String {
        var folded = name.precomposedStringWithCompatibilityMapping.lowercased()
            .trimmingCharacters(in: .whitespaces)
        for suffix in ["のコピー"] where folded.hasSuffix(suffix) { folded.removeLast(suffix.count) }
        if let copy = folded.range(of: #"\s+copy(\s+\d+)?$"#, options: .regularExpression) {
            folded.removeSubrange(copy)
        }
        return folded.filter { !" _-・.".contains($0) }
    }

    /// What a layer is and which eye it is of, by its name. The side is the character's in the
    /// name -- See-through's "-r", "右" -- and the picture's here: the character's right is on the
    /// picture's left.
    static func role(of name: String) -> (Role, Side?) {
        var name = name.precomposedStringWithCompatibilityMapping.lowercased()
            .trimmingCharacters(in: .whitespaces)
        var side: Side?
        let marks: [(String, Side)] = [
            ("-r", .left), ("_r", .left), (" r", .left), (".r", .left), (" right", .left), ("右", .left),
            ("-l", .right), ("_l", .right), (" l", .right), (".l", .right), (" left", .right), ("左", .right),
        ]
        for (mark, which) in marks {
            if name.hasSuffix(mark) {
                name.removeLast(mark.count)
                side = which
                break
            }
            if mark.count == 1, name.hasPrefix(mark) {
                name.removeFirst()
                side = which
                break
            }
        }
        let folded = fold(name)
        // A number on the end is another piece of the same thing: "前髪2", "eyelash_2".
        let bare = folded.replacingOccurrences(of: #"\d+$"#, with: "", options: .regularExpression)
        for (role, names) in aliases where names.contains(folded) || names.contains(bare) {
            return (role, side)
        }
        return (.other, side)
    }
}

// MARK: - pixels

/// What a layer's pixels say about it, in its own coordinates.
enum Pixels {
    /// The box of what is drawn, alpha above `floor`; nil for nothing.
    static func box(_ image: RGBAImage, floor: UInt8 = 16) -> CGRect? {
        var (x0, y0, x1, y1) = (Int.max, Int.max, -1, -1)
        for y in 0..<image.height {
            for x in 0..<image.width where image.rgba[(y * image.width + x) * 4 + 3] > floor {
                x0 = min(x0, x)
                x1 = max(x1, x)
                y0 = min(y0, y)
                y1 = max(y1, y)
            }
        }
        guard x1 >= 0 else { return nil }
        return CGRect(x: x0, y: y0, width: x1 - x0 + 1, height: y1 - y0 + 1)
    }

    /// Clears every speck: each piece of what is drawn above `floor`, its pixels touching side to
    /// side, smaller than `smallest`. A model's layers are scattered with them, faint and alone,
    /// all over the canvas -- enough to make every layer's box the whole of it.
    static func denoise(_ image: inout RGBAImage, floor: UInt8 = 16, smallest: Int = 40) {
        let (w, h) = (image.width, image.height)
        var seen = [Bool](repeating: false, count: w * h)
        image.rgba.withUnsafeMutableBufferPointer { px in
            var piece: [Int] = []
            for start in 0..<(w * h) where !seen[start] && px[start * 4 + 3] > floor {
                piece.removeAll(keepingCapacity: true)
                piece.append(start)
                seen[start] = true
                var next = 0
                while next < piece.count {
                    let p = piece[next]
                    next += 1
                    let (x, y) = (p % w, p / w)
                    for (q, inside) in [(p - 1, x > 0), (p + 1, x < w - 1), (p - w, y > 0), (p + w, y < h - 1)]
                    where inside && !seen[q] && px[q * 4 + 3] > floor {
                        seen[q] = true
                        piece.append(q)
                    }
                }
                if piece.count < smallest {
                    for p in piece { px[p * 4 + 3] = 0 }
                }
            }
        }
    }

    /// The middle of what is drawn above `floor`, each pixel weighed by its alpha.
    static func centroid(_ image: RGBAImage, floor: UInt8 = 16) -> CGPoint? {
        var (sx, sy, sw) = (0.0, 0.0, 0.0)
        for y in 0..<image.height {
            for x in 0..<image.width {
                let alpha = image.rgba[(y * image.width + x) * 4 + 3]
                guard alpha > floor else { continue }
                let a = Double(alpha)
                sx += a * (Double(x) + 0.5)
                sy += a * (Double(y) + 0.5)
                sw += a
            }
        }
        return sw > 0 ? CGPoint(x: sx / sw, y: sy / sw) : nil
    }

    /// The colour of what is drawn, weighed by alpha: 0 to 1 each.
    static func meanColour(_ image: RGBAImage) -> (Double, Double, Double)? {
        var (r, g, b, w) = (0.0, 0.0, 0.0, 0.0)
        for i in stride(from: 0, to: image.rgba.count, by: 4) {
            let a = Double(image.rgba[i + 3])
            r += a * Double(image.rgba[i])
            g += a * Double(image.rgba[i + 1])
            b += a * Double(image.rgba[i + 2])
            w += a
        }
        return w > 0 ? (r / w / 255, g / w / 255, b / w / 255) : nil
    }

    /// The part of it inside `box`, which is in its own pixels.
    static func crop(_ image: RGBAImage, _ box: CGRect) -> RGBAImage {
        let (x0, y0, w, h) = (Int(box.minX), Int(box.minY), Int(box.width), Int(box.height))
        var rgba = [UInt8](repeating: 0, count: w * h * 4)
        for y in 0..<h {
            let from = ((y0 + y) * image.width + x0) * 4
            rgba.replaceSubrange(y * w * 4..<(y + 1) * w * 4, with: image.rgba[from..<from + w * 4])
        }
        return RGBAImage(width: w, height: h, rgba: rgba)
    }

    /// A CGImage of it, its alpha not premultiplied.
    static func cgImage(_ image: RGBAImage) -> CGImage? {
        guard image.width > 0, image.height > 0,
            let provider = CGDataProvider(data: Data(image.rgba) as CFData)
        else { return nil }
        return CGImage(
            width: image.width, height: image.height, bitsPerComponent: 8, bitsPerPixel: 32,
            bytesPerRow: image.width * 4, space: CGColorSpace(name: CGColorSpace.sRGB)!,
            bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.last.rawValue), provider: provider,
            decode: nil, shouldInterpolate: true, intent: .defaultIntent)
    }
}

// MARK: - building

extension Rig {
    /// What blinks in `document`, and how. Anything it does not know is drawn as it is.
    static func build(_ document: PSDDocument) -> Rig {
        // Each shown layer, trimmed to what is drawn.
        struct Piece {
            var name: String
            var role: Role
            var side: Side?
            var left: Int
            var top: Int
            var image: RGBAImage
            var opacity: Double
        }
        // Cleared of specks first, as Anime2.5DRig does, and trimmed to what is left above 16.
        var pieces: [Piece] = []
        for layer in document.layers where layer.visible {
            var image = layer.image
            Pixels.denoise(&image)
            guard let box = Pixels.box(image) else { continue }
            let (role, side) = Self.role(of: layer.name)
            pieces.append(
                Piece(
                    name: layer.name, role: role, side: side, left: layer.left + Int(box.minX),
                    top: layer.top + Int(box.minY), image: Pixels.crop(image, box), opacity: layer.opacity))
        }

        // The middle of the face, which an eye drawn as one layer is cut at.
        let face = pieces.last { $0.role == .face }
        let middle: Double = {
            if let face, let c = Pixels.centroid(face.image) { return Double(face.left) + c.x }
            return Double(document.width) / 2
        }()
        var cut: [Piece] = []
        for piece in pieces {
            let eyeish: Set<Role> = [.eyewhite, .irides, .eyelash, .eyeClose, .eyebrow]
            guard eyeish.contains(piece.role), piece.side == nil else {
                cut.append(piece)
                continue
            }
            let at = min(max(0, Int(middle.rounded()) - piece.left), piece.image.width)
            for (side, from, to) in [(Side.left, 0, at), (Side.right, at, piece.image.width)] where to > from {
                let half = Pixels.crop(piece.image, CGRect(x: from, y: 0, width: to - from, height: piece.image.height))
                guard let box = Pixels.box(half, floor: 16) else { continue }
                var one = piece
                one.side = side
                one.left = piece.left + from + Int(box.minX)
                one.top = piece.top + Int(box.minY)
                one.image = Pixels.crop(half, box)
                cut.append(one)
            }
        }
        pieces = cut

        // Each eye's anchor: its white's box, its iris's middle, and where it closes -- its own
        // closed eye's middle, or 62 % of the way down its white.
        var eyes: [Side: EyeAnchor] = [:]
        var notes: [String] = []
        for side in Side.allCases {
            guard let white = pieces.last(where: { $0.role == .eyewhite && $0.side == side }),
                let box = Pixels.box(white.image)
            else { continue }
            let rect = box.offsetBy(dx: CGFloat(white.left), dy: CGFloat(white.top))
            let iris = pieces.last { $0.role == .irides && $0.side == side }
            let middle = iris.flatMap { iris in
                Pixels.centroid(iris.image).map { CGPoint(x: $0.x + CGFloat(iris.left), y: $0.y + CGFloat(iris.top)) }
            } ?? CGPoint(x: rect.midX, y: rect.midY)
            let closed = pieces.last { $0.role == .eyeClose && $0.side == side }
            let closeY = closed.flatMap { closed in Pixels.centroid(closed.image).map { $0.y + CGFloat(closed.top) } }
                ?? rect.minY + 0.62 * rect.height
            eyes[side] = EyeAnchor(white: rect, iris: middle, closeY: closeY)
        }
        if eyes.isEmpty {
            notes.append("No eye white layer (eyewhite, 白目): the eyes cannot blink.")
        }

        // A closed eye for each eye without one: the stock drawing, 1.1 times as wide as the
        // white, its middle on the white's, 55 % of it above the line it closes onto, in the
        // colour of that eye's lashes (or brow).
        var made: [Piece] = []
        for side in Side.allCases {
            guard let eye = eyes[side], !pieces.contains(where: { $0.role == .eyeClose && $0.side == side }) else {
                continue
            }
            let lash = pieces.last { $0.role == .eyelash && $0.side == side }
                ?? pieces.last { $0.role == .eyebrow && $0.side == side }
            let tint = lash.flatMap { Pixels.meanColour($0.image) } ?? (0.2, 0.15, 0.15)
            guard let drawing = closedEye(side, width: eye.white.width * 1.1, tint: tint) else { continue }
            made.append(
                Piece(
                    name: "eye_close (made)", role: .eyeClose, side: side,
                    left: Int((eye.white.midX - CGFloat(drawing.width) / 2).rounded()),
                    top: Int((eye.closeY - 0.55 * CGFloat(drawing.height)).rounded()), image: drawing, opacity: 1))
        }
        if !made.isEmpty {
            // Over the lashes, or the irides, or the face: where a closed eye would be drawn.
            let at = (pieces.lastIndex { $0.role == .eyelash } ?? pieces.lastIndex { $0.role == .irides }
                ?? pieces.lastIndex { $0.role == .face }).map { $0 + 1 } ?? pieces.count
            pieces.insert(contentsOf: made, at: at)
        }

        let parts = pieces.compactMap { piece -> RigPart? in
            guard let image = Pixels.cgImage(piece.image) else { return nil }
            return RigPart(
                name: piece.name, role: piece.role, side: piece.side,
                rect: CGRect(x: piece.left, y: piece.top, width: piece.image.width, height: piece.image.height),
                image: image, opacity: piece.opacity)
        }
        let faceWidth = face.flatMap { Pixels.box($0.image)?.width } ?? CGFloat(document.width) / 3
        return Rig(
            width: document.width, height: document.height, parts: parts, eyes: eyes,
            faceScale: faceWidth / 333, notes: notes)
    }

    /// The stock closed eye for `side`, scaled to `width` and painted `tint`, each pixel as light
    /// as it was against the drawing's mean, at most 2.2 times.
    static func closedEye(_ side: Side, width: CGFloat, tint: (Double, Double, Double)) -> RGBAImage? {
        let drawing = side == .left ? ClosedEyes.left : ClosedEyes.right
        guard let source = drawing.rgba, width >= 1 else { return nil }
        let w = max(1, Int(width.rounded()))
        let h = max(1, Int((CGFloat(drawing.height) * width / CGFloat(drawing.width)).rounded()))
        let luma = { (i: Int) in (0.299 * Double(source[i]) + 0.587 * Double(source[i + 1]) + 0.114 * Double(source[i + 2])) / 255 }
        var (sum, weight) = (0.0, 0.0)
        for i in stride(from: 0, to: source.count, by: 4) {
            let a = Double(source[i + 3])
            sum += a * luma(i)
            weight += a
        }
        let mean = weight > 0 ? max(sum / weight, 1e-3) : 1
        // Bilinear, in premultiplied alpha so that the edge does not darken, then back.
        var out = [UInt8](repeating: 0, count: w * h * 4)
        let sample = { (x: Double, y: Double, c: Int) -> Double in
            let fx = min(max(x, 0), Double(drawing.width - 1)), fy = min(max(y, 0), Double(drawing.height - 1))
            let x0 = Int(fx), y0 = Int(fy), x1 = min(x0 + 1, drawing.width - 1), y1 = min(y0 + 1, drawing.height - 1)
            let tx = fx - Double(x0), ty = fy - Double(y0)
            let at = { (x: Int, y: Int) -> Double in
                let i = (y * drawing.width + x) * 4
                let a = Double(source[i + 3]) / 255
                return c == 3 ? a : a * luma(i)
            }
            return (at(x0, y0) * (1 - tx) + at(x1, y0) * tx) * (1 - ty) + (at(x0, y1) * (1 - tx) + at(x1, y1) * tx) * ty
        }
        for y in 0..<h {
            for x in 0..<w {
                let sx = (Double(x) + 0.5) * Double(drawing.width) / Double(w) - 0.5
                let sy = (Double(y) + 0.5) * Double(drawing.height) / Double(h) - 0.5
                let a = sample(sx, sy, 3)
                guard a > 0 else { continue }
                let shade = min(2.2, sample(sx, sy, 0) / a / mean)
                let i = (y * w + x) * 4
                out[i] = UInt8(min(255, (tint.0 * shade * 255).rounded()))
                out[i + 1] = UInt8(min(255, (tint.1 * shade * 255).rounded()))
                out[i + 2] = UInt8(min(255, (tint.2 * shade * 255).rounded()))
                out[i + 3] = UInt8(min(255, (a * 255).rounded()))
            }
        }
        return RGBAImage(width: w, height: h, rgba: out)
    }
}

// MARK: - blinking

/// How open the eyes are from moment to moment: Anime2.5DRig's blink. Closed in 80 ms, held
/// 340 ms, open in 160 ms; the next one 1.6 to 5.4 s after this one began, or -- one time in six
/// or so -- right after it; the first 1.8 s in. Every change is eased by a low-pass of 71 ms, and an
/// eye just opened gives its iris a small bounce.
struct Blinker {
    /// Seconds since it began, at its last step.
    private(set) var now = 0.0
    private var nextBlink = 1.8
    /// Seconds into the blink going on, or nil.
    private var blinking: Double?
    private var bounced = false
    private var bounceTime: Double?
    /// How open the eyes are, 0 shut to 1 open, eased.
    private(set) var openness = 1.0
    /// The iris's scale across and down, from its bounce.
    private(set) var bounce = CGSize(width: 1, height: 1)

    /// Blinks once, now, whatever the clock says.
    mutating func blinkNow() {
        if blinking == nil {
            blinking = 0
            bounced = false
        }
    }

    /// Where the blink is `seconds` into it: how open the eyes are, or nil once it is over.
    static func shape(_ seconds: Double) -> Double? {
        let hold = 0.34
        switch seconds {
        case ..<0.08: return 1 - seconds / 0.08
        case ..<(0.08 + hold): return 0
        case ..<(0.24 + hold): return (seconds - 0.08 - hold) / 0.16
        default: return nil
        }
    }

    /// Moves on by `dt` seconds. `open` is as open as the eyes may be; `automatic` blinks on its own.
    mutating func step(by dt: Double, open: Double, automatic: Bool, random: () -> Double = { Double.random(in: 0..<1) }) {
        let dt = max(0, min(dt, 0.25))
        now += dt
        if automatic, blinking == nil, now > nextBlink {
            blinking = 0
            bounced = false
            nextBlink = now + 1.6 + random() * 3.8
            if random() < 0.18 { nextBlink = now + 0.28 }
        }
        var target = open
        if var t = blinking {
            t += dt
            if let v = Self.shape(t) {
                blinking = t
                target = min(target, v)
                if t >= 0.08 + 0.34, v > 0.12, !bounced {
                    bounced = true
                    bounceTime = 0
                }
            } else {
                blinking = nil
            }
        }
        openness += (target - openness) * (1 - exp(-dt * 14))

        bounce = CGSize(width: 1, height: 1)
        if var t = bounceTime {
            t += dt
            let p = t / 0.52
            if p >= 1 {
                bounceTime = nil
            } else {
                bounceTime = t
                let damp = exp(-2.2 * p)
                let scale = 1 + 0.18 * sin(4 * .pi * p) * damp
                let squash = 0.10 * sin(4 * .pi * p + .pi / 2) * damp
                bounce = CGSize(width: scale * (1 + squash), height: scale * (1 - squash))
            }
        }
    }
}

/// Smoothstep, clamped: 0 below 0, 1 above 1.
func smooth(_ t: Double) -> Double {
    let t = min(max(t, 0), 1)
    return t * t * (3 - 2 * t)
}
