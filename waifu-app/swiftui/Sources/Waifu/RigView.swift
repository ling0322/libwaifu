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

// A rig moving, every frame the screen draws: RigMotion moves it, RigRenderer draws it, after the
// pointer where it is over the drawing.

import MetalKit
import SwiftUI

struct RigView: NSViewRepresentable {
    var rig: Rig
    var stage: RigStage

    func makeCoordinator() -> Driver { Driver(rig: rig, stage: stage) }

    func makeNSView(context: Context) -> PointerView {
        let view = PointerView()
        view.device = context.coordinator.renderer?.device
        view.colorPixelFormat = .bgra8Unorm
        view.clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 0)
        view.layer?.isOpaque = false
        view.preferredFramesPerSecond = 60
        view.delegate = context.coordinator
        view.canvas = CGSize(width: rig.width, height: rig.height)
        context.coordinator.view = view
        return view
    }

    func updateNSView(_ view: PointerView, context: Context) {
        context.coordinator.stage = stage
    }

    /// Moves the rig on and draws it, each frame.
    final class Driver: NSObject, MTKViewDelegate {
        let rig: Rig
        var stage: RigStage
        let renderer: RigRenderer?
        weak var view: PointerView?
        private var motion: RigMotion
        private var last: CFTimeInterval?
        private var blinks: Int

        init(rig: Rig, stage: RigStage) {
            self.rig = rig
            self.stage = stage
            renderer = RigRenderer()
            renderer?.load(rig)
            motion = RigMotion(rig)
            blinks = MainActor.assumeIsolated { stage.blinks }
        }

        func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {}

        func draw(in view: MTKView) {
            MainActor.assumeIsolated { frame(view) }
        }

        @MainActor private func frame(_ view: MTKView) {
            let now = CACurrentMediaTime()
            let dt = last.map { now - $0 } ?? 0
            last = now
            if stage.blinks != blinks {
                blinks = stage.blinks
                motion.blinker.blinkNow()
            }
            motion.step(
                by: dt, rig: rig, switches: stage.switches, mouse: (view as? PointerView)?.pointer, open: stage.open,
                ease: stage.ease, nod: stage.nod, follow: stage.follow)
            guard let renderer, let drawable = view.currentDrawable, let commands = renderer.queue.makeCommandBuffer() else {
                return
            }
            renderer.encode(
                motion.frame(rig, physics: stage.switches.physics), to: drawable.texture, in: commands, clear: view.clearColor)
            commands.present(drawable)
            commands.commit()
        }
    }
}

/// An MTKView that knows where the pointer is over the drawing in it: -1 to 1 across it and down
/// it, as far as 1.5 beyond it, or nil when it is out of the view.
final class PointerView: MTKView {
    var canvas = CGSize(width: 1, height: 1)
    private(set) var pointer: SIMD2<Float>?

    override func updateTrackingAreas() {
        super.updateTrackingAreas()
        for area in trackingAreas { removeTrackingArea(area) }
        addTrackingArea(
            NSTrackingArea(
                rect: bounds, options: [.mouseMoved, .mouseEnteredAndExited, .activeInKeyWindow, .inVisibleRect],
                owner: self))
    }

    override func mouseMoved(with event: NSEvent) {
        let p = convert(event.locationInWindow, from: nil)
        let fitted = RigRenderer.fit(canvas: canvas, into: bounds.size)
        guard fitted.width > 0, fitted.height > 0 else { return }
        let x = (p.x - fitted.minX) / fitted.width * 2 - 1
        // Down the drawing: the view's y is up.
        let y = (1 - (p.y - fitted.minY) / fitted.height) * 2 - 1
        pointer = SIMD2(Float(min(max(x, -1.5), 1.5)), Float(min(max(y, -1.5), 1.5)))
    }

    override func mouseEntered(with event: NSEvent) { mouseMoved(with: event) }

    override func mouseExited(with event: NSEvent) { pointer = nil }
}
