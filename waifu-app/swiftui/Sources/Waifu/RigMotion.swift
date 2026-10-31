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

// How a rig moves, frame by frame, as Anime2.5DRig moves one (lib/app.js, MIT, Copyright (c) 2026
// hakoniwa): what it is told to do -- where it looks, how open its eyes are, how it
// breathes -- and every vertex of every layer moved by that.
//
// The head turns about the neck and every layer of it moves by how far in front of the face it
// is, which is what makes it look turned rather than slid: front hair more than the face, back hair
// less. The body follows the head a little and leans about its foot. Hair swings in strands, each a
// stiff spring at the root and a soft one at the tip, after the head; an earring or a tail sways.

import CoreGraphics
import Foundation
import simd

/// What the rig is told to do, this frame. Angles and the eyes' look are -1 to 1.
struct RigPose {
    var angleX: Float = 0
    var angleY: Float = 0
    var angleZ: Float = 0
    var body: Float = 0
    var eyeX: Float = 0
    var eyeY: Float = 0
    var openness: Float = 1
    /// The iris's scale across and down, from its bounce after a blink.
    var bounce = SIMD2<Float>(1, 1)
    /// 0 to 1: the chest's breath, and the head's, which lags it.
    var breath: Float = 0
    var breathHead: Float = 0
    /// How soon the open eye gives way to the closed one, 0 to 1.
    var ease: Float = 0.3
    /// How far the head turns up and down, 1 as the rig turns it.
    var nod: Float = 1
    /// How far the upper body has turned after the head, -1 to 1 as the head's turn is.
    var bodyTurn: Float = 0
}

/// What moves on its own.
struct RigSwitches: Equatable {
    var blink = true
    /// Sways a little, all the time.
    var idle = true
    /// Looks somewhere else now and then.
    var lookAround = true
    var physics = true
    var followMouse = true
}

/// A spring after a target: Anime2.5DRig's, semi-implicit Euler in steps of at most 1/120 s, so that
/// it is the same at 10 frames a second as at 240.
struct Spring {
    var x: Float = 0
    var v: Float = 0

    mutating func step(to target: Float, k: Float, damping: Float, dt: Float) {
        let count = max(1, Int((dt * 120).rounded(.up)))
        let h = dt / Float(count)
        for _ in 0..<count {
            v += (-k * (x - target) - damping * v) * h
            x += v * h
        }
    }
}

/// The moving rig's state from one frame to the next.
struct RigMotion {
    /// A strand's two springs, and how far each lags its target.
    struct StrandSprings {
        var stiff = Spring()
        var soft = Spring()
        var phase: Float
        var stiffLag: Float = 0
        var softLag: Float = 0
    }

    private(set) var time: Float = 0
    /// Where the pose is, eased toward where it is told to be.
    var pose = RigPose()
    var blinker = Blinker()
    /// Each part's strands' springs, part by part: empty for one that does not swing.
    private(set) var strands: [[StrandSprings]]
    private var bust = Spring()
    private(set) var bustLag: Float = 0

    private var look = RigPose()
    private var nextLook: Float = 0

    init(_ rig: Rig) {
        strands = rig.parts.enumerated().map { index, part in
            (part.swing?.strands ?? []).indices.map { StrandSprings(phase: Float($0) * 1.37 + Float(index)) }
        }
    }

    /// Moves on by `dt` seconds. `mouse` is where the pointer is over the canvas, -1 to 1 across it
    /// and down it, or nil where it is not over it; `open` is as open as the eyes may be.
    mutating func step(
        by dt: Double, rig: Rig, switches: RigSwitches, mouse: SIMD2<Float>?, open: Double, ease: Double, nod: Double = 1, follow: Double = 1,
        random: () -> Float = { Float.random(in: 0..<1) }
    ) {
        let dt = Float(max(0, min(dt, 0.05)))
        time += dt
        let t = time
        var target = RigPose()

        // Where it is told to look: after the pointer, or wherever it looks next.
        if switches.followMouse, let m = mouse {
            target.angleX = clamp(m.x * 0.9)
            target.angleY = clamp(-m.y * 0.7)
            target.eyeX = clamp(m.x * 1.2)
            target.eyeY = clamp(-m.y * 0.8)
        }
        if switches.idle {
            target.angleX += 0.13 * sin(t * 0.42) + 0.05 * sin(t * 1.13)
            target.angleY += 0.08 * sin(t * 0.31 + 1.7)
            target.angleZ += 0.07 * sin(t * 0.23 + 0.5)
            target.body += 0.10 * sin(t * 0.19 + 2.1)
        }
        // Anime2.5DRig looks around over the pointer too; here it does not while it follows one,
        // which it would only wander from.
        if switches.lookAround, !(switches.followMouse && mouse != nil) {
            if t > nextLook {
                nextLook = t + 1.4 + random() * 2.6
                look.angleX = (random() * 2 - 1) * 0.55
                look.angleY = (random() * 2 - 1) * 0.40
                look.angleZ = (random() * 2 - 1) * 0.35
                look.body = (random() * 2 - 1) * 0.30
                look.eyeX = (random() * 2 - 1) * 0.60
                look.eyeY = (random() * 2 - 1) * 0.35
            }
            target.angleX += look.angleX
            target.angleY += look.angleY
            target.angleZ += look.angleZ
            target.body += look.body
            target.eyeX += look.eyeX
            target.eyeY += look.eyeY
        }

        // The body leans the way the head turns, as Anime2.5DRig's does after a tracked face.
        target.body += clamp(target.angleX) * 0.25 * Float(follow)

        blinker.step(by: Double(dt), open: open, automatic: switches.blink)

        // Eased, 71 ms; the body after the head, over 170.
        let k = 1 - exp(-dt * 14)
        pose.angleX += (clamp(target.angleX) - pose.angleX) * k
        pose.angleY += (clamp(target.angleY) - pose.angleY) * k
        pose.angleZ += (clamp(target.angleZ) - pose.angleZ) * k
        pose.body += (clamp(target.body) - pose.body) * (1 - exp(-dt * 6))
        // How far the body has turned after the head: the head's turn, eased the body's way.
        pose.bodyTurn += (clamp(target.angleX) * Float(follow) - pose.bodyTurn) * (1 - exp(-dt * 6))
        pose.eyeX += (clamp(target.eyeX) - pose.eyeX) * k
        pose.eyeY += (clamp(target.eyeY) - pose.eyeY) * k
        pose.openness = Float(blinker.openness)
        pose.bounce = SIMD2(Float(blinker.bounce.width), Float(blinker.bounce.height))
        pose.breath = 0.5 + 0.5 * sin(t * 2 * .pi / 3.4)
        pose.breathHead = 0.5 + 0.5 * sin(t * 2 * .pi / 3.4 - 0.6)
        pose.ease = Float(ease)
        pose.nod = Float(nod)

        stepSprings(rig, dt: dt, wind: switches.idle)
    }

    /// Each strand's springs after where the head or the body has gone, and the wind; and the bust's.
    private mutating func stepSprings(_ rig: Rig, dt: Float, wind: Bool) {
        let a = rig.anchors
        let fs = Float(rig.faceScale)
        let headDX = (pose.angleX * 14 + pose.angleZ * 0.07 * Float(a.neckPivot.y - a.faceCenter.y)) * fs
        let bodyDX = pose.body * 0.028 * Float(a.bodyPivot.y - a.neckPivot.y) * 0.35 + pose.angleX * 14 * 0.16 * fs
        for (index, part) in rig.parts.enumerated() where !strands[index].isEmpty {
            let base = part.group == .head ? headDX : bodyDX
            let windScale: Float = part.swing?.kind == .sway && part.group != .head ? 2.5 : 1
            for s in strands[index].indices {
                let phase = strands[index][s].phase
                let w = wind ? (1.8 * sin(time * 0.8 + phase) + 1.0 * sin(time * 1.9 + phase * 2.3)) * windScale : 0
                let target = base + w * fs
                strands[index][s].stiff.step(to: target, k: 70, damping: 9, dt: dt)
                strands[index][s].stiffLag = -(strands[index][s].stiff.x - target) * 2.2
                strands[index][s].soft.step(to: target, k: 16, damping: 1.3, dt: dt)
                strands[index][s].softLag = -(strands[index][s].soft.x - target) * 3.0
            }
        }
        let bustTarget = (pose.breath * 3 - pose.angleY * 6 + pose.body * 4) * fs
        bust.step(to: bustTarget, k: 140, damping: 4.2, dt: dt)
        bustLag = -(bust.x - bustTarget) * 3
    }

    /// Every part where it is now, and how much of it shows.
    func frame(_ rig: Rig, physics: Bool) -> RigFrame {
        RigFrame(
            positions: rig.parts.indices.map { deform(rig, part: $0, physics: physics) },
            alphas: rig.parts.indices.map { Self.alpha(rig, part: $0, pose: pose) })
    }

    /// How much of a part shows: an open eye's parts fading into its closed eye as it shuts.
    static func alpha(_ rig: Rig, part index: Int, pose: RigPose) -> Float {
        let part = rig.parts[index]
        let opacity = Float(part.opacity)
        // The long closed eye, for a long blink, is not blinked with.
        if part.slot == "eye_close2" { return 0 }
        if let side = part.side, rig.eyes[side] != nil {
            let opened = Float(smooth(Double((pose.openness - (0.10 + pose.ease * 0.45)) / 0.15)))
            switch part.role {
            case .eyewhite, .irides, .eyelash: return opacity * opened
            case .eyeClose: return opacity * (1 - opened)
            default: break
            }
        }
        return opacity
    }

    /// Where each vertex of part `index` is now, on the canvas: Anime2.5DRig's deform, in its order --
    /// the eyes, the head turned, the breath, the chest, the strands, and the
    /// body's lean last.
    func deform(_ rig: Rig, part index: Int, physics: Bool) -> [SIMD2<Float>] {
        let part = rig.parts[index]
        let e = pose
        let a = rig.anchors
        let fs = Float(rig.faceScale)
        let np = SIMD2<Float>(Float(a.neckPivot.x), Float(a.neckPivot.y))
        let bp = SIMD2<Float>(Float(a.bodyPivot.x), Float(a.bodyPivot.y))
        let faceCenterY = Float(a.faceCenter.y)
        // The face's surface, which the head turns up and down on: an ellipse over the face's box
        // about its middle, as high in the middle as 0.3 of its width and nothing at its edge.
        let surface = (
            center: SIMD2<Float>(Float(a.faceCenter.x), faceCenterY),
            radius: SIMD2<Float>(max(1, Float(a.face.width) / 2), max(1, Float(a.face.height) / 2)),
            height: Float(a.face.width) * 0.3, width: Float(a.face.width)
        )
        let az = e.angleZ * 0.07
        let (cz, sz) = (cos(az), sin(az))
        let ab = e.body * 0.028
        let (cb, sb) = (cos(ab), sin(ab))
        let eye = part.side.flatMap { rig.eyes[$0] }
        let eyeRole = eye != nil ? part.role : .other
        let v = e.openness
        let chest = SIMD2<Float>(Float(a.chest.x), Float(a.chest.y))
        let reach = SIMD2<Float>(Float(a.chestReach.width), Float(a.chestReach.height))
        let neckTop = Float(a.neckTop)
        let neckBottom = Float(a.neckBottom)
        let isTop = part.slot == "topwear"

        // How the strands swing it.
        let swing = physics ? part.swing : nil
        let springs = strands[index]
        let (ampK, softK, power): (Float, Float, Float) =
            switch (swing?.kind, swing?.front) {
            case (.sway?, _): (1 * 1.4, 0.8, 1.2)
            case (.hair?, true?): (2, 0.4, 1.8)
            default: (2, 2, 2.1)
            }

        var out = part.mesh.rest
        for i in out.indices {
            var (x, y) = (out[i].x, out[i].y)

            // An eye: the white and lashes squashed toward the line it closes onto, the iris
            // bounced, looking, and round until the eye is nearly shut.
            if let eye {
                let closeY = Float(eye.closeY)
                switch eyeRole {
                case .irides:
                    let (cx, cy) = (Float(eye.iris.x), Float(eye.iris.y))
                    x = cx + (x - cx) * e.bounce.x
                    y = cy + (y - cy) * e.bounce.y
                    // Anime2.5DRig moves it down for a look up; here it goes where it looks.
                    x += e.eyeX * 11 * fs
                    y -= e.eyeY * 6 * fs
                    let shut = Float(smooth(Double((0.32 - v) / 0.32)))
                    y = closeY + (y - closeY) * (1 - 0.80 * shut)
                case .eyewhite, .eyelash:
                    y = closeY + (y - closeY) * (1 - 0.85 * (1 - v))
                case .eyeClose:
                    y -= v * 3
                default: break
                }
            }
            if part.role == .eyebrow { y += (1 - v) * 3.5 * fs }

            // The head turned about the neck: all of the head, a little of the body, the neck more
            // the higher up it.
            var hw: Float = part.group == .head ? 1 : 0.16
            if part.slot == "neck" { hw = 0.55 * Float(smooth(Double((neckBottom - y) / max(1, neckBottom - neckTop)))) }
            if hw > 0 {
                let (rx, ry) = (x - np.x, y - np.y)
                x += (rx * cz - ry * sz - rx) * hw
                y += (rx * sz + ry * cz - ry) * hw
                let d = part.depth
                x += hw * fs * (e.angleX * (14 + 40 * (d - 1)) + e.angleX * (np.y - y) * 0.028)
                if part.group == .head {
                    // Up and down, the head as a ball: every vertex as near as the face's surface is
                    // there, and its layer in front of that, turned about the face's middle. The
                    // middle of the face is nearest, so the eyes, nose and mouth go farther than its
                    // outline, and on an arc; the head is shortened; and a layer goes as far as it is
                    // in front -- the same as Anime2.5DRig's 30 a unit of depth, which slides the
                    // head up and down rather than turning it.
                    let pitch = e.angleY * 0.35 * e.nod
                    let (u, w) = ((x - surface.center.x) / surface.radius.x, (y - surface.center.y) / surface.radius.y)
                    let z = surface.height * max(0, 1 - u * u - w * w).squareRoot() + (d - 1) * 0.3 * surface.width
                    y = surface.center.y + (y - surface.center.y) * cos(pitch) - sin(pitch) * z - e.angleY * e.nod * 9 * fs
                } else {
                    y += hw * fs * (-e.angleY * (9 + 30 * (d - 1)) - e.angleY * (d - 1) * (y - faceCenterY) * 0.05)
                }
            }
            // The upper body turned after the head: the more the higher up, nothing at the canvas's
            // foot, and each layer as far as it is in front -- the tie past the top, the arms behind it.
            if part.group == .body {
                let up = Float(smooth(Double((bp.y - y) / max(1, bp.y - np.y))))
                x += e.bodyTurn * fs * (6 + 40 * (part.depth - 0.9)) * up
            }

            // The breath: the body rises with it, the head after; the shoulders rise and the top widens.
            y -= (part.group == .body ? e.breath * 2.0 : e.breathHead * 1.6) * fs
            if isTop {
                if y < chest.y { y -= e.breath * 2.2 * fs * Float(smooth(Double((chest.y - y) / (reach.y * 2)))) }
                x = np.x + (x - np.x) * (1 + e.breath * 0.003)
                let gx = (x - chest.x) / reach.x
                let gy = (y - (chest.y + 70 * fs)) / reach.y
                y += bustLag * 2.5 * exp(-gx * gx - gy * gy)
            }

            // The strands: a stiff root and a soft tip, more of the soft the farther down.
            if let swing, !springs.isEmpty {
                let along = swing.along[i]
                let u = swing.front ? min(1, along * 1.6) : along
                let amp = pow(u, power) * ampK
                let softMix = min(1, pow(u, 1.2) * softK)
                var dx: Float = 0
                let n = springs.count
                for s in 0..<n {
                    let w = swing.weights[i * n + s]
                    if w < 0.001 { continue }
                    dx += w * (springs[s].stiffLag * (1 - softMix) + springs[s].softLag * softMix)
                }
                x += dx * amp
                y += abs(dx) * amp * 0.12
            }
            out[i] = SIMD2(x, y)
        }

        // The body's lean about its foot, which everything leans with.
        if abs(ab) > 1e-4 {
            for i in out.indices {
                let r = out[i] - bp
                out[i] = bp + SIMD2(r.x * cb - r.y * sb, r.x * sb + r.y * cb)
            }
        }
        return out
    }
}

/// One frame of a rig: each part's vertices where they are, and how much of it shows.
struct RigFrame {
    var positions: [[SIMD2<Float>]]
    var alphas: [Float]
}

private func clamp(_ v: Float) -> Float { min(max(v, -1), 1) }
