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

// A rig moving, every frame the screen draws. libwaifu2d moves it and draws it into this view's
// CAMetalLayer; the view only tells it what the settings say, where the pointer is, and how much
// time has gone by.

import QuartzCore
import SwiftUI
import WaifuKit

struct RigView: NSViewRepresentable {
    var rig: Waifu2D.Rig
    var stage: RigStage
    /// Where the mouth's shapes come from, if anywhere but the page's own settings: a reading
    /// being heard.
    var lipsync: Lipsync? = nil

    func makeNSView(context: Context) -> RigCanvas {
        let view = RigCanvas(rig: rig, settings: stage)
        view.lipsync = lipsync
        return view
    }

    func updateNSView(_ view: RigCanvas, context: Context) {
        view.settings = stage
        view.lipsync = lipsync
    }

    static func dismantleNSView(_ view: RigCanvas, coordinator: ()) {
        view.stop()
    }
}

/// A view backed by a CAMetalLayer the library draws into, a frame each time the display shows
/// one.
@MainActor
final class RigCanvas: NSView {
    var settings: RigStage
    var lipsync: Lipsync?
    private let stage: Waifu2D.Stage?
    private var renderer: Waifu2D.Renderer?
    private var link: CADisplayLink?
    private var last: CFTimeInterval?
    private var blinks: Int
    private let metalLayer = CAMetalLayer()

    init(rig: Waifu2D.Rig, settings: RigStage) {
        self.settings = settings
        stage = try? Waifu2D.Stage(rig: rig)
        blinks = settings.blinks
        super.init(frame: .zero)
        wantsLayer = true
        metalLayer.isOpaque = false
        metalLayer.framebufferOnly = true
        renderer = try? Waifu2D.Renderer(layer: metalLayer)
    }

    @available(*, unavailable)
    required init?(coder: NSCoder) { fatalError("not from a nib") }

    override func makeBackingLayer() -> CALayer { metalLayer }

    override func viewDidMoveToWindow() {
        super.viewDidMoveToWindow()
        link?.invalidate()
        link = nil
        guard window != nil else { return }
        let link = displayLink(target: self, selector: #selector(frame(_:)))
        link.add(to: .main, forMode: .common)
        self.link = link
    }

    func stop() {
        link?.invalidate()
        link = nil
    }

    override func updateTrackingAreas() {
        super.updateTrackingAreas()
        for area in trackingAreas { removeTrackingArea(area) }
        addTrackingArea(
            NSTrackingArea(
                rect: bounds, options: [.mouseMoved, .mouseEnteredAndExited, .activeInKeyWindow, .inVisibleRect],
                owner: self))
    }

    /// The pointer as the library takes it: from the top left, y down.
    override func mouseMoved(with event: NSEvent) {
        let p = convert(event.locationInWindow, from: nil)
        stage?.pointerMoved(to: CGPoint(x: p.x, y: bounds.height - p.y))
    }

    override func mouseEntered(with event: NSEvent) { mouseMoved(with: event) }

    override func mouseExited(with event: NSEvent) { stage?.pointerLeft() }

    @objc private func frame(_ link: CADisplayLink) {
        guard let stage, let renderer else { return }
        let now = link.timestamp
        let dt = last.map { now - $0 } ?? 0
        last = now
        talked = settings.talking ? talked + dt : 0
        apply(settings, to: stage)
        stage.setViewSize(bounds.size)
        stage.step(by: dt)

        let scale = window?.backingScaleFactor ?? 1
        metalLayer.contentsScale = scale
        let (w, h) = (Int((bounds.width * scale).rounded()), Int((bounds.height * scale).rounded()))
        guard w > 0, h > 0, (try? renderer.resize(width: w, height: h)) != nil else { return }
        try? renderer.draw(stage)
    }

    /// The page's settings, as the stage has them: the wind with the sway, as the Mac always had it.
    private func apply(_ settings: RigStage, to stage: Waifu2D.Stage) {
        let s = settings.switches
        stage.set(.followPointer, s.followMouse)
        stage.set(.idle, s.idle)
        stage.set(.wind, s.idle)
        stage.set(.lookAround, s.lookAround)
        stage.set(.physics, s.physics)
        stage.set(.blink, s.blink)
        try? stage.set(.eyesOpen, settings.open)
        try? stage.set(.closesAt, settings.ease)
        try? stage.set(.nod, settings.nod)
        try? stage.set(.bodyFollows, settings.follow)
        if settings.blinks != blinks {
            blinks = settings.blinks
            stage.act(.blink)
        }
        if let lipsync {
            say(lipsync, on: stage)
        } else if settings.talking {
            stage.setViseme(Self.spoken(at: talked))
        } else {
            stage.setViseme(settings.viseme == .rest ? nil : settings.viseme)
        }
    }

    /// Seconds the mouth has talked for, while it talks.
    private var talked = 0.0

    /// The reading the stage has been handed, and how much of it.
    private var saying: Lipsync.Had?

    /// What is new of the reading being heard, handed to the stage, and where the sound is; at
    /// rest between readings.
    private func say(_ lipsync: Lipsync, on stage: Waifu2D.Stage) {
        guard let news = lipsync.news(had: saying) else {
            if saying != nil {
                stage.silence()
                saying = nil
            }
            return
        }
        if news.id != saying?.id {
            stage.speak(rate: news.rate)
            saying = Lipsync.Had(id: news.id)
        }
        if !news.samples.isEmpty {
            stage.speechSamples(news.samples)
            saying?.samples += news.samples.count
        }
        if let visemes = news.visemes {
            stage.speechVisemes(visemes.values, frameMilliseconds: visemes.frameMilliseconds)
            saying?.visemes = news.visemesSeen
        }
        if let loudness = news.loudness {
            stage.speechLoudness(loudness.values, frameMilliseconds: loudness.frameMilliseconds)
            saying?.loudness = news.loudnessSeen
        }
        stage.speechAt(news.seconds)
    }

    /// "こんにちは、わたしはワイフです。まま、パパ。" as text to speech would shape it: a viseme
    /// and how many 40 ms frames it is held, then a breath.
    private static let sentence: [(Waifu2D.Viseme, Int)] = [
        (.o, 3), (.rest, 1), (.i, 2), (.i, 2), (.a, 4), (.rest, 4),
        (.a, 3), (.a, 2), (.i, 2), (.a, 2), (.a, 3), (.i, 2), (.u, 2), (.schwa, 1), (.e, 2), (.u, 3), (.rest, 6),
        (.mbp, 2), (.a, 3), (.mbp, 2), (.a, 4), (.rest, 3), (.mbp, 2), (.a, 3), (.mbp, 2), (.a, 4), (.rest, 12),
    ]

    private static let sentenceFrames = sentence.reduce(0) { $0 + $1.1 }

    /// The sentence's viseme `seconds` into saying it, over and over.
    private static func spoken(at seconds: Double) -> Waifu2D.Viseme {
        var frame = Int(seconds / 0.04) % sentenceFrames
        for (v, n) in sentence {
            if frame < n { return v }
            frame -= n
        }
        return .rest
    }
}
