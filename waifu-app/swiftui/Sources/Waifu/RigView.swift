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

// A rig drawn as it blinks, every frame: the open eye squashed toward its line and faded into the
// closed one, the iris kept inside the white, the brow dipping. Anime2.5DRig's renderer is WebGL
// over a mesh; a blink only ever moves a layer by a squash in y, so here it is a transform.

import SwiftUI

/// The clock the drawing is moved on: kept between frames, and never observed -- the timeline
/// redraws it.
final class RigClock {
    var blinker = Blinker()
    var last: Date?
    var blinks = 0
}

struct RigView: View {
    var rig: Rig
    var stage: RigStage

    @State private var clock = RigClock()

    var body: some View {
        TimelineView(.animation) { timeline in
            Canvas { context, size in
                let now = timeline.date
                let dt = clock.last.map { now.timeIntervalSince($0) } ?? 0
                clock.last = now
                if stage.blinks != clock.blinks {
                    clock.blinks = stage.blinks
                    clock.blinker.blinkNow()
                }
                clock.blinker.step(by: dt, open: stage.open, automatic: stage.automatic)
                Self.draw(
                    rig, in: &context, size: size, openness: clock.blinker.openness, bounce: clock.blinker.bounce,
                    ease: stage.ease)
            }
        }
        .background(Checkerboard())
    }

    /// One frame of `rig`, its eyes `v` open, fitted into `size`.
    static func draw(
        _ rig: Rig, in context: inout GraphicsContext, size: CGSize, openness v: Double, bounce: CGSize = CGSize(width: 1, height: 1),
        ease: Double = 0.3
    ) {
        // The canvas fitted into the room there is, in the middle of it.
        let scale = min(size.width / CGFloat(rig.width), size.height / CGFloat(rig.height))
        context.translateBy(
            x: (size.width - CGFloat(rig.width) * scale) / 2, y: (size.height - CGFloat(rig.height) * scale) / 2)
        context.scaleBy(x: scale, y: scale)
        context.clip(to: Path(CGRect(x: 0, y: 0, width: rig.width, height: rig.height)))

        // Where the open eye gives way to the closed one: a quarter open, by default, over 0.15.
        let threshold = 0.10 + ease * 0.45
        let opened = smooth((v - threshold) / 0.15)

        for part in rig.parts {
            let image = Image(decorative: part.image, scale: 1)
            var layer = context
            layer.opacity = part.opacity
            guard let side = part.side, let eye = rig.eyes[side] else {
                if part.role == .eyebrow { layer.translateBy(x: 0, y: (1 - v) * 3.5 * rig.faceScale) }
                layer.draw(image, in: part.rect)
                continue
            }
            switch part.role {
            case .eyewhite, .eyelash:
                layer.opacity *= opened
                layer.concatenate(Self.squash(eye.closeY, by: 1 - 0.85 * (1 - v)))
                layer.draw(image, in: part.rect)
            case .irides:
                // Round until the eye is nearly shut, then squashed too; and inside the white,
                // which is squashed as the eye closes, whether the white is shown or not.
                let squash = Self.squash(eye.closeY, by: 1 - 0.80 * smooth((0.32 - v) / 0.32))
                let bounced = CGAffineTransform(translationX: eye.iris.x, y: eye.iris.y)
                    .scaledBy(x: bounce.width, y: bounce.height)
                    .translatedBy(x: -eye.iris.x, y: -eye.iris.y)
                let white = rig.white(side)
                let whiteSquash = Self.squash(eye.closeY, by: 1 - 0.85 * (1 - v))
                // The layer's own opacity is the group's, as it is put down.
                layer.drawLayer { inside in
                    var inside = inside
                    // Destination-in reaches only as far as the white's own rectangle: what is
                    // outside it, squashed, is cut away first.
                    if let white { inside.clip(to: Path(white.rect.applying(whiteSquash))) }
                    var iris = inside
                    iris.opacity = opened
                    iris.concatenate(bounced.concatenating(squash))
                    iris.draw(image, in: part.rect)
                    if let white {
                        var mask = inside
                        mask.blendMode = .destinationIn
                        mask.concatenate(whiteSquash)
                        mask.draw(Image(decorative: white.image, scale: 1), in: white.rect)
                    }
                }
            case .eyeClose:
                layer.opacity *= 1 - opened
                layer.translateBy(x: 0, y: -v * 3)
                layer.draw(image, in: part.rect)
            case .eyebrow:
                layer.translateBy(x: 0, y: (1 - v) * 3.5 * rig.faceScale)
                layer.draw(image, in: part.rect)
            default:
                layer.draw(image, in: part.rect)
            }
        }
    }

    /// Every y pulled toward `line` to `factor` of how far it was: y' = line + (y - line) * factor.
    static func squash(_ line: CGFloat, by factor: Double) -> CGAffineTransform {
        CGAffineTransform(a: 1, b: 0, c: 0, d: factor, tx: 0, ty: line * (1 - factor))
    }
}
