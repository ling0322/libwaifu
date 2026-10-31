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

// A drawing in layers rigged to move, the way Anime2.5DRig (https://github.com/852wa/Anime2.5DRig,
// MIT, Copyright (c) 2026 hakoniwa) rigs one: the names its layers are known by, where each sits
// in depth and whether it is the head's or the body's, the anchors it moves about, the strands its
// hair swings in, and the closed eyes it is given where it has none. Its numbers are
// Anime2.5DRig's (lib/rigger.js); how it moves is RigMotion's.
//
// A blink is not a lid. The open eye -- its white, iris and lashes -- is squashed toward a
// line across it, and when it is a quarter open it fades into a closed eye drawn apart: the
// document's own, or a stock one (ClosedParts) fitted to the eye and tinted to its lashes. The
// iris stays inside the white as the white is squashed. Which eye is which is the picture's: an
// eye layer drawn as one is cut in two at the middle of the face.

import CoreGraphics
import Foundation
import WaifuKit

/// Which eye, as the picture has them: `.left` is the character's right.
enum Side: CaseIterable {
    case left, right
}

/// What a layer is, for the parts of the rig that treat it as its own: an eye's, a brow, the face,
/// and everything else.
enum Role: Equatable {
    case eyewhite, irides, eyelash, eyeClose, eyebrow, face, other
}

/// Whether a layer moves with the head or with the body.
enum RigGroup {
    case head, body
}

/// How a layer swings: hair, or a thing hung from something -- an earring, a tail.
enum Phys {
    case hair, sway
}

/// What a kind of layer is to the rig: Anime2.5DRig's `SLOTS`. Depth is 1 at the face and more
/// in front of it: how far a layer moves as the head turns.
struct Slot {
    var depth: Float
    /// Nil for one that is the head's or the body's by where it is.
    var group: RigGroup?
    var phys: Phys?
}

/// One strand a layer of hair swings in: where it hangs, its root and its tip, on the canvas.
struct Strand: Equatable {
    var x: Float
    var rootY: Float
    var tipY: Float
}

/// How a layer of hair or a hanging thing swings: its strands, each vertex's weight towards each
/// one, and how far down it each vertex is, 0 at the root to 1 at the tip.
struct Swing {
    var kind: Phys
    /// Front hair, side hair and ahoge swing as front hair does.
    var front: Bool
    var strands: [Strand]
    /// `strands.count` a vertex, vertex by vertex.
    var weights: [Float]
    var along: [Float]
}

struct RigPart {
    var name: String
    /// Its slot's name -- "front hair", "topwear" -- or the name as it is folded where it has none.
    var slot: String
    var role: Role
    var side: Side?
    /// Where it is on the canvas.
    var rect: CGRect
    /// RGBA, not premultiplied, the size of `rect`.
    var pixels: RGBAImage
    var opacity: Double
    var depth: Float
    var group: RigGroup
    /// The grid it is drawn on, moved vertex by vertex as it moves.
    var mesh: RigMesh
    var swing: Swing?
}

/// A layer's grid: its vertices at rest, on the canvas, where each one reads the layer's picture,
/// and the triangles between them. Anime2.5DRig's: cells of 42 pixels at a 768 canvas -- 30 for
/// what swings -- as wide as the canvas is over 768 but never under 0.6 of that, at least two a side.
struct RigMesh {
    var rest: [SIMD2<Float>]
    var uv: [SIMD2<Float>]
    var indices: [UInt32]

    static func grid(over rect: CGRect, canvas: Int, swings: Bool = false) -> RigMesh {
        let cell = (swings ? 30 : 42) * max(0.6, Double(canvas) / 768)
        let nx = max(2, Int((rect.width / cell).rounded()))
        let ny = max(2, Int((rect.height / cell).rounded()))
        var rest: [SIMD2<Float>] = []
        var uv: [SIMD2<Float>] = []
        for j in 0...ny {
            for i in 0...nx {
                let (u, v) = (Float(i) / Float(nx), Float(j) / Float(ny))
                uv.append(SIMD2(u, v))
                rest.append(SIMD2(Float(rect.minX) + u * Float(rect.width), Float(rect.minY) + v * Float(rect.height)))
            }
        }
        var indices: [UInt32] = []
        for j in 0..<ny {
            for i in 0..<nx {
                let a = UInt32(j * (nx + 1) + i)
                let b = a + 1
                let c = a + UInt32(nx + 1)
                let d = c + 1
                indices += [a, b, c, b, d, c]
            }
        }
        return RigMesh(rest: rest, uv: uv, indices: indices)
    }
}

/// An eye at rest: the box of its white, the middle of its iris, and the line it closes onto.
struct EyeAnchor: Equatable {
    var white: CGRect
    var iris: CGPoint
    var closeY: CGFloat
}

/// Where the rig moves about: the face, the neck it turns on, the body's foot and the chest.
struct Anchors {
    var face: CGRect
    var faceCenter: CGPoint
    /// What the head turns about: 85 % of the way down the neck.
    var neckPivot: CGPoint
    var neckTop: CGFloat
    var neckBottom: CGFloat
    /// What the body leans about: under the neck, at the canvas's foot.
    var bodyPivot: CGPoint
    /// The middle of the chest and how far it reaches across and down.
    var chest: CGPoint
    var chestReach: CGSize
}

struct Rig {
    var width: Int
    var height: Int
    /// Bottom first, the hidden ones left out.
    var parts: [RigPart]
    var eyes: [Side: EyeAnchor]
    var anchors: Anchors
    /// How wide the face is over 333: what every distance the rig moves by is scaled by.
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
    /// Anime2.5DRig's slots, by name.
    static let slots: [String: Slot] = [
        "wings": Slot(depth: 0.48, group: .body),
        "tail": Slot(depth: 0.50, group: .body, phys: .sway),
        "back hair": Slot(depth: 0.55, group: .head, phys: .hair),
        "footwear": Slot(depth: 0.83, group: .body),
        "legwear": Slot(depth: 0.84, group: .body),
        "bottomwear": Slot(depth: 0.88, group: .body),
        "neck": Slot(depth: 0.95, group: .body),
        "topwear": Slot(depth: 0.90, group: .body),
        "neckwear": Slot(depth: 0.98, group: .body),
        "handwear": Slot(depth: 0.86, group: .body),
        "objects": Slot(depth: 1.00, group: nil),
        "earwear": Slot(depth: 0.97, group: .head, phys: .sway),
        "ears": Slot(depth: 0.96, group: .head),
        "face": Slot(depth: 1.00, group: .head),
        "facedetail": Slot(depth: 1.02, group: .head),
        "headwear": Slot(depth: 1.20, group: .head),
        "mouth_close": Slot(depth: 1.08, group: .head),
        "mouth_open": Slot(depth: 1.08, group: .head),
        "nose": Slot(depth: 1.15, group: .head),
        "eyewhite": Slot(depth: 1.06, group: .head),
        "eyebrow": Slot(depth: 1.14, group: .head),
        "irides": Slot(depth: 1.08, group: .head),
        "eyelash": Slot(depth: 1.12, group: .head),
        "eye_close": Slot(depth: 1.12, group: .head),
        "eye_close2": Slot(depth: 1.12, group: .head),
        "eyewear": Slot(depth: 1.18, group: .head),
        "side hair": Slot(depth: 1.22, group: .head, phys: .hair),
        "front hair": Slot(depth: 1.28, group: .head, phys: .hair),
        "ahoge": Slot(depth: 1.30, group: .head, phys: .hair),
    ]

    /// The names each slot is known by once folded (see `fold`): Anime2.5DRig's, with See-through's
    /// among them.
    static let aliases: [String: String] = {
        let groups: [(String, String)] = [
            ("front hair", "fronthair hairfront bangs fringe 前髪 まえがみ"),
            ("back hair", "backhair hairback 後ろ髪 後髪 うしろがみ"),
            ("side hair", "sidehair hairside sidelock sidelocks 横髪 サイド髪 もみあげ"),
            ("ahoge", "ahoge アホ毛 あほ毛"),
            ("hair", "hair 髪 髪の毛 かみ"),
            ("face", "face 顔 かお 輪郭 facebase"),
            ("facedetail", "facedetail blush cheek cheeks 頬 頬染め チーク"),
            ("eyewhite", "eyewhite eyewhites sclera 白目"),
            ("irides", "irides iris irises pupil pupils 瞳 黒目 虹彩"),
            ("eyelash", "eyelash eyelashes lash lashes eyeline まつ毛 まつげ 睫毛 アイライン"),
            ("eyebrow", "eyebrow eyebrows brow brows 眉 眉毛 まゆ まゆげ"),
            ("eye_close", "eyeclose eyeclosed eyesclosed closedeye closedeyes eyelashc 閉じ目 目閉じ 閉眼"),
            ("mouth_open", "mouthopen openmouth mouth 口 開き口 口開き 開口"),
            ("mouth_close", "mouthclose mouthclosed closedmouth mouthc 閉じ口 口閉じ"),
            ("nose", "nose 鼻 はな"),
            ("ears", "ears ear 耳"),
            ("earwear", "earwear earring earrings イヤリング ピアス 耳飾り"),
            ("neck", "neck 首"),
            ("neckwear", "neckwear necktie tie necklace choker ネクタイ ネックレス チョーカー 首飾り"),
            ("topwear", "topwear top tops shirt body torso clothes 服 上着 体 胴体 上半身 トップス"),
            ("bottomwear", "bottomwear bottom bottoms skirt pants スカート ズボン ボトムス 下半身"),
            ("handwear", "handwear hand hands arm arms gloves 手 腕 手袋"),
            ("legwear", "legwear leg legs socks tights 脚 足 靴下 ソックス タイツ"),
            ("footwear", "footwear shoes shoe boots 靴 ブーツ"),
            ("headwear", "headwear hat cap hairpin hairband 帽子 カチューシャ 髪飾り ヘアピン"),
            ("eyewear", "eyewear glasses 眼鏡 メガネ めがね"),
            ("tail", "tail しっぽ 尻尾 シッポ"),
            ("wings", "wings wing 羽 翼 羽根"),
            ("objects", "objects object 小物 持ち物"),
        ]
        var map: [String: String] = [:]
        for (slot, names) in groups {
            for name in names.split(separator: " ") { map[fold(String(name))] = slot }
        }
        return map
    }()

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

    /// A layer's slot and which eye it is of, by its name: "前髪2" is a piece of `front hair`, "eye_close2"
    /// the long closed eye, See-through's "mouth" an open one. The side is the character's in the
    /// name -- See-through's "-r", "右" -- and the picture's here: the character's right is on the
    /// picture's left. A name no slot has comes back folded, its own.
    static func slot(of name: String) -> (slot: String, side: Side?) {
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
        if ["eyeclose2", "閉じ目2"].contains(folded) { return ("eye_close2", side) }
        if let slot = aliases[folded] { return (slot, side) }
        // A number on the end is another piece of the same thing: "前髪2", "eyelash_2".
        let bare = folded.replacingOccurrences(of: #"\d+$"#, with: "", options: .regularExpression)
        if let slot = aliases[bare] { return (slot, side) }
        return (folded, side)
    }

    /// What a layer is, for the eyes, brows and face, and which eye.
    static func role(of name: String) -> (Role, Side?) {
        let (slot, side) = slot(of: name)
        return (role(ofSlot: slot), side)
    }

    static func role(ofSlot slot: String) -> Role {
        switch slot {
        case "eyewhite": .eyewhite
        case "irides": .irides
        case "eyelash": .eyelash
        case "eye_close", "eye_close2": .eyeClose
        case "eyebrow": .eyebrow
        case "face": .face
        default: .other
        }
    }

    /// Whether a layer's name has a number on its end: one strand group among several.
    static func isNumbered(_ name: String) -> Bool {
        fold(name).range(of: #"\d+$"#, options: .regularExpression) != nil
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

// MARK: - strands

extension Rig {
    /// Where a layer of hair hangs in strands: Anime2.5DRig's `detectStrands`. The bottom edge of
    /// what is drawn, column by column, smoothed over 41 columns, has a peak at each strand's tip:
    /// up to `want` of the most prominent, at least `minSep` apart, and more where the layer is wide
    /// enough that a column is that far from every one. Each strand runs from the top of its
    /// column to the bottom, in the layer's own pixels.
    static func strands(_ image: RGBAImage, want: Int, minSep: Int) -> [Strand] {
        let (w, h) = (image.width, image.height)
        var top = [Int](repeating: -1, count: w)
        var bottom = [Int](repeating: 0, count: w)
        for x in 0..<w {
            for y in 0..<h where image.rgba[(y * w + x) * 4 + 3] > 16 {
                if top[x] < 0 { top[x] = y }
                bottom[x] = y
            }
        }
        let painted = (0..<w).filter { top[$0] >= 0 }
        guard let minX = painted.first, let maxX = painted.last else { return [] }

        // Box-smoothed over 41 columns, always divided by 41, beyond the edges 0.
        var smoothed = [Double](repeating: 0, count: w)
        for x in 0..<w {
            var sum = 0.0
            for k in -20...20 where x + k >= 0 && x + k < w { sum += Double(bottom[x + k]) }
            smoothed[x] = sum / 41
        }
        var picked: [Int] = []
        for peak in peaks(smoothed, minSep: minSep, minProminence: 10) where picked.count < want {
            picked.append(peak)
        }
        // More, at whichever of 40 columns across it is farthest from every strand, while that is
        // at least minSep away.
        let inset = min(30, (maxX - minX) / 4)
        let candidates = (0..<40).map { k in
            let x = minX + inset + Int((Double(maxX - minX - 2 * inset) * Double(k) / 39).rounded())
            return painted.min { abs($0 - x) < abs($1 - x) }!
        }
        for _ in 0..<50 where picked.count < want {
            let distance = { (x: Int) in picked.map { abs($0 - x) }.min() ?? Int.max }
            guard let far = candidates.max(by: { distance($0) < distance($1) }),
                distance(far) >= max(1, minSep)
            else { break }
            picked.append(far)
        }
        let columns = Set(picked.map { x in painted.min { abs($0 - x) < abs($1 - x) }! }).sorted()
        return columns.map { Strand(x: Float($0), rootY: Float(top[$0]), tipY: Float(bottom[$0])) }
    }

    /// The peaks of `a`, most prominent first, none within `minSep` of one kept: a rise to a flat
    /// top counts at the top's middle, unless it rises again after. A peak's prominence is how far
    /// it stands over the higher of the lowest points on either side before something higher.
    static func peaks(_ a: [Double], minSep: Int, minProminence: Double) -> [Int] {
        var found: [(Int, Double)] = []
        var i = 1
        while i < a.count - 1 {
            guard a[i] > a[i - 1] else {
                i += 1
                continue
            }
            var j = i
            while j + 1 < a.count && a[j + 1] == a[i] { j += 1 }
            if j + 1 < a.count && a[j + 1] < a[i] {
                let p = (i + j) / 2
                var (left, right) = (a[p], a[p])
                var k = p - 1
                while k >= 0 && a[k] <= a[p] { left = min(left, a[k]); k -= 1 }
                k = p + 1
                while k < a.count && a[k] <= a[p] { right = min(right, a[k]); k += 1 }
                let prominence = a[p] - max(left, right)
                if prominence >= minProminence { found.append((p, prominence)) }
            }
            i = j + 1
        }
        var kept: [Int] = []
        for (p, _) in found.sorted(by: { $0.1 > $1.1 }) where kept.allSatisfy({ abs($0 - p) >= minSep }) {
            kept.append(p)
        }
        return kept
    }

    /// Each vertex's weight towards each strand -- a Gaussian over 0.6 of the strands' median
    /// spacing, summing to 1 -- and how far down the weighted strand it is.
    static func swing(_ kind: Phys, front: Bool, strands: [Strand], mesh: RigMesh) -> Swing {
        let gaps = zip(strands.dropFirst(), strands).map { $0.x - $1.x }.sorted()
        let spacing = gaps.isEmpty ? 120 : gaps[gaps.count / 2]
        let sigma = max(1, spacing * 0.6)
        var weights: [Float] = []
        var along: [Float] = []
        for p in mesh.rest {
            var w = strands.map { exp(-pow(($0.x - p.x) / sigma, 2)) }
            let sum = w.reduce(0, +)
            if sum < 1e-6 {
                w = strands.indices.map { $0 == 0 ? 1 : 0 }
            } else {
                w = w.map { $0 / sum }
            }
            let root = zip(w, strands).map { $0 * $1.rootY }.reduce(0, +)
            let tip = zip(w, strands).map { $0 * $1.tipY }.reduce(0, +)
            weights += w
            along.append(min(max((p.y - root) / max(1, tip - root), 0), 1))
        }
        return Swing(kind: kind, front: front, strands: strands, weights: weights, along: along)
    }
}

// MARK: - building

extension Rig {
    /// What moves in `document`, and how. Anything it does not know is drawn as it is, with the
    /// head or the body by where it is.
    static func build(_ document: PSDDocument) -> Rig {
        // Each shown layer, trimmed to what is drawn.
        struct Piece {
            var name: String
            var slot: String
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
            let (slot, side) = Self.slot(of: layer.name)
            pieces.append(
                Piece(
                    name: layer.name, slot: slot, side: side, left: layer.left + Int(box.minX),
                    top: layer.top + Int(box.minY), image: Pixels.crop(image, box), opacity: layer.opacity))
        }
        // A plain "hair" is the front hair above the face and the back hair under it.
        if let faceAt = pieces.lastIndex(where: { $0.slot == "face" }) {
            for i in pieces.indices where pieces[i].slot == "hair" {
                pieces[i].slot = i > faceAt ? "front hair" : "back hair"
            }
        }

        // The face: its box and middle, or where a face would be.
        let face = pieces.last { $0.slot == "face" }
        let faceBox: CGRect = face.flatMap { f in Pixels.box(f.image, floor: 8)?.offsetBy(dx: CGFloat(f.left), dy: CGFloat(f.top)) }
            ?? CGRect(x: 0.35 * Double(document.width), y: 0.1 * Double(document.height),
                      width: 0.3 * Double(document.width), height: 0.4 * Double(document.height))
        let faceCenter: CGPoint = face.flatMap { f in Pixels.centroid(f.image).map { CGPoint(x: $0.x + CGFloat(f.left), y: $0.y + CGFloat(f.top)) } }
            ?? CGPoint(x: Double(document.width) / 2, y: 0.3 * Double(document.height))

        // Eyes drawn as one layer each, cut at the middle of the face.
        var cut: [Piece] = []
        for piece in pieces {
            let eyeish: Set<String> = ["eyewhite", "irides", "eyelash", "eye_close", "eye_close2", "eyebrow"]
            guard eyeish.contains(piece.slot), piece.side == nil else {
                cut.append(piece)
                continue
            }
            let at = min(max(0, Int(faceCenter.x.rounded()) - piece.left), piece.image.width)
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
            guard let white = pieces.last(where: { $0.slot == "eyewhite" && $0.side == side }),
                let box = Pixels.box(white.image)
            else { continue }
            let rect = box.offsetBy(dx: CGFloat(white.left), dy: CGFloat(white.top))
            let iris = pieces.last { $0.slot == "irides" && $0.side == side }
            let middle = iris.flatMap { iris in
                Pixels.centroid(iris.image).map { CGPoint(x: $0.x + CGFloat(iris.left), y: $0.y + CGFloat(iris.top)) }
            } ?? CGPoint(x: rect.midX, y: rect.midY)
            let closed = pieces.last { $0.slot == "eye_close" && $0.side == side }
            let closeY = closed.flatMap { closed in Pixels.centroid(closed.image).map { $0.y + CGFloat(closed.top) } }
                ?? rect.minY + 0.62 * rect.height
            eyes[side] = EyeAnchor(white: rect, iris: middle, closeY: closeY)
        }
        if eyes.isEmpty {
            notes.append("No eye white layer (eyewhite, 白目): the eyes cannot blink.")
        }

        // The neck the head turns on: 85 % down it, or under the face where there is none.
        let neck = pieces.last { $0.slot == "neck" }
        let neckBox = neck.flatMap { n in Pixels.box(n.image, floor: 8)?.offsetBy(dx: CGFloat(n.left), dy: CGFloat(n.top)) }
        let neckCenterX = neck.flatMap { n in Pixels.centroid(n.image).map { $0.x + CGFloat(n.left) } }
        let neckPivot = neckBox.map { CGPoint(x: neckCenterX ?? $0.midX, y: $0.minY + 0.85 * $0.height) }
            ?? CGPoint(x: faceCenter.x, y: faceBox.maxY + 20)
        let (neckTop, neckBottom) = neckBox.map { ($0.minY, $0.maxY) } ?? (faceBox.maxY, faceBox.maxY + 60)

        // A closed eye for each eye without one: the stock drawing, 1.1 times as wide as the
        // white, its middle on the white's, 55 % of it above the line it closes onto, in the
        // colour of that eye's lashes (or brow).
        var made: [Piece] = []
        for side in Side.allCases {
            guard let eye = eyes[side], !pieces.contains(where: { $0.slot == "eye_close" && $0.side == side }) else {
                continue
            }
            let lash = pieces.last { $0.slot == "eyelash" && $0.side == side }
                ?? pieces.last { $0.slot == "eyebrow" && $0.side == side }
            let tint = lash.flatMap { Pixels.meanColour($0.image) } ?? (0.2, 0.15, 0.15)
            guard let drawing = closedEye(side, width: eye.white.width * 1.1, tint: tint) else { continue }
            made.append(
                Piece(
                    name: "eye_close (made)", slot: "eye_close", side: side,
                    left: Int((eye.white.midX - CGFloat(drawing.width) / 2).rounded()),
                    top: Int((eye.closeY - 0.55 * CGFloat(drawing.height)).rounded()), image: drawing, opacity: 1))
        }
        if !made.isEmpty {
            // Over the lashes, or the irides, or the face: where a closed eye would be drawn.
            let at = (pieces.lastIndex { $0.slot == "eyelash" } ?? pieces.lastIndex { $0.slot == "irides" }
                ?? pieces.lastIndex { $0.slot == "face" }).map { $0 + 1 } ?? pieces.count
            pieces.insert(contentsOf: made, at: at)
        }

        let parts = pieces.map { piece -> RigPart in
            let rect = CGRect(x: piece.left, y: piece.top, width: piece.image.width, height: piece.image.height)
            let slot = slots[piece.slot]
            // One no slot places is the head's above the bottom of the face and the body's under it.
            let group = slot?.group ?? {
                let y = Pixels.centroid(piece.image).map { $0.y + CGFloat(piece.top) } ?? rect.midY
                return y < faceBox.maxY ? .head : .body
            }()
            let mesh = RigMesh.grid(over: rect, canvas: document.width, swings: slot?.phys != nil)
            var swing: Swing?
            if let phys = slot?.phys {
                let width = piece.image.width
                let want: Int
                let minSep: Int
                switch phys {
                case .hair:
                    want = isNumbered(piece.name) ? min(max(Int((Double(width) / 110).rounded()), 2), 6) : 6
                    minSep = max(30, Int((Double(width) / (Double(want) * 1.6)).rounded()))
                case .sway:
                    want = min(max(Int((Double(width) / 140).rounded()), 1), 4)
                    minSep = max(20, Int((Double(width) / (Double(want) * 1.6)).rounded()))
                }
                let found = strands(piece.image, want: want, minSep: minSep).map {
                    Strand(x: $0.x + Float(piece.left), rootY: $0.rootY + Float(piece.top), tipY: $0.tipY + Float(piece.top))
                }
                if !found.isEmpty {
                    let front = ["front hair", "side hair", "ahoge"].contains(piece.slot)
                    swing = Self.swing(phys, front: front, strands: found, mesh: mesh)
                }
            }
            return RigPart(
                name: piece.name, slot: piece.slot, role: role(ofSlot: piece.slot), side: piece.side, rect: rect,
                pixels: piece.image, opacity: piece.opacity, depth: slot?.depth ?? 1, group: group, mesh: mesh,
                swing: swing)
        }

        let faceHeight = faceBox.height
        let anchors = Anchors(
            face: faceBox, faceCenter: faceCenter, neckPivot: neckPivot, neckTop: neckTop, neckBottom: neckBottom,
            bodyPivot: CGPoint(x: neckPivot.x, y: CGFloat(document.height)),
            chest: CGPoint(x: neckPivot.x, y: neckBottom + faceHeight * 0.60),
            chestReach: CGSize(width: max(1, faceBox.width * 0.60), height: max(1, faceHeight * 0.45)))
        return Rig(
            width: document.width, height: document.height, parts: parts, eyes: eyes, anchors: anchors,
            faceScale: (face.flatMap { Pixels.box($0.image)?.width } ?? CGFloat(document.width) / 3) / 333, notes: notes)
    }

    /// The stock closed eye for `side`, scaled to `width` and painted `tint`.
    static func closedEye(_ side: Side, width: CGFloat, tint: (Double, Double, Double)) -> RGBAImage? {
        stock(side == .left ? ClosedParts.left : ClosedParts.right, width: width, tint: tint)
    }

    /// A stock drawing's ink scaled to `width` and painted `tint`, each pixel as light as it was
    /// against the ink's mean, at most 2.2 times; the skin it is edged in left out.
    static func stock(_ drawing: ClosedParts.Drawing, width: CGFloat, tint: (Double, Double, Double)) -> RGBAImage? {
        guard let source = drawing.rgba, width >= 1 else { return nil }
        let w = max(1, Int(width.rounded()))
        let h = max(1, Int((CGFloat(drawing.height) * width / CGFloat(drawing.width)).rounded()))
        let luma = { (i: Int) in (0.299 * Double(source[i]) + 0.587 * Double(source[i + 1]) + 0.114 * Double(source[i + 2])) / 255 }
        // How much of a pixel is ink: the drawings are edged in skin -- a line's soft edge, a lower
        // lip's highlight -- which, tinted, would be a red blot on the face. Skin, at a luma of
        // 0.85 or more, is left out; the line, under 0.5, is kept whole.
        let ink = { (luma: Double) in smooth((0.85 - luma) / 0.35) }
        var (sum, weight) = (0.0, 0.0)
        for i in stride(from: 0, to: source.count, by: 4) {
            let a = Double(source[i + 3]) * ink(luma(i))
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
                let covered = sample(sx, sy, 3)
                guard covered > 0 else { continue }
                let lightness = sample(sx, sy, 0) / covered
                let a = covered * ink(lightness)
                guard a > 0 else { continue }
                let shade = min(2.2, lightness / mean)
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
