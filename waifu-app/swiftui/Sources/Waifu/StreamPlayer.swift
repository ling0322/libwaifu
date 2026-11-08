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

// A streaming reading played as it is said: each part the engine hands over goes to the end of
// one player's queue, so the parts follow one another with no gap but the engine's own lateness.

import AVFoundation
import WaifuKit

/// One streaming reading's playback. Parts are queued from the engine's thread, in the order they
/// are handed over -- not through the main actor, whose tasks keep no order. Once stopped it is
/// silent for good: a part the engine hands over after a Stop is dropped. Once finished it plays
/// what is queued and then lets the audio hardware go.
///
/// Its state is kept on one serial queue rather than behind a lock: the player node calls a
/// buffer's completion as it stops, and a completion that waited on a lock the stop held would
/// never return.
final class StreamPlayer: @unchecked Sendable {
    private let queue = DispatchQueue(label: "waifu.stream-player")
    private var engine: AVAudioEngine?
    private var node: AVAudioPlayerNode?
    private var format: AVAudioFormat?
    private var stopped = false
    private var finished = false
    private var parts = 0
    /// Buffers scheduled and not yet played out, of the engine `generation` names: a completion
    /// from one torn down before it arrived counts for nothing.
    private var pending = 0
    private var generation = 0

    /// How many parts have been queued: none means nothing of the reading was heard.
    var heard: Int {
        queue.sync { parts }
    }

    /// Whether the audio engine is running: from the first part to a Stop, or to the last part
    /// played out after `finish`.
    var playing: Bool {
        queue.sync { engine != nil }
    }

    /// Queues `part` after the ones before it, starting playback with the first.
    func play(_ part: AudioClip) {
        queue.sync {
            guard !stopped, !finished, !part.samples.isEmpty, part.rate > 0 else { return }
            // Every part of a reading is at its voice's rate; one at another starts a player for it.
            if format?.sampleRate != Double(part.rate) {
                guard start(rate: part.rate) else { return }
            }
            guard let node, let format,
                let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(part.samples.count)),
                let channel = buffer.floatChannelData?[0]
            else { return }
            buffer.frameLength = AVAudioFrameCount(part.samples.count)
            part.samples.withUnsafeBufferPointer { samples in
                channel.update(from: samples.baseAddress!, count: samples.count)
            }
            pending += 1
            let scheduledIn = generation
            node.scheduleBuffer(buffer) { [weak self] in
                guard let self else { return }
                self.queue.async { self.playedOne(of: scheduledIn) }
            }
            if !node.isPlaying { node.play() }
            parts += 1
        }
    }

    /// No more parts are coming: what is queued plays out, and then the audio hardware is let go.
    func finish() {
        queue.sync {
            finished = true
            if pending == 0 { tearDown() }
        }
    }

    /// Silences it at once, what is queued with it, and every part after.
    func stop() {
        queue.sync {
            stopped = true
            tearDown()
        }
    }

    private func playedOne(of scheduledIn: Int) {
        guard scheduledIn == generation else { return }
        pending = max(0, pending - 1)
        if finished && pending == 0 { tearDown() }
    }

    private func start(rate: Int) -> Bool {
        tearDown()
        guard let format = AVAudioFormat(standardFormatWithSampleRate: Double(rate), channels: 1) else {
            return false
        }
        let engine = AVAudioEngine()
        let node = AVAudioPlayerNode()
        engine.attach(node)
        engine.connect(node, to: engine.mainMixerNode, format: format)
        do {
            try engine.start()
        } catch {
            return false
        }
        self.engine = engine
        self.node = node
        self.format = format
        return true
    }

    private func tearDown() {
        node?.stop()
        engine?.stop()
        engine = nil
        node = nil
        format = nil
        pending = 0
        generation += 1
    }
}
