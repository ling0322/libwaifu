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

// A streaming reading played as it is said. The engine's parts go into a JitterBuffer, and the
// audio hardware pulls from it a few milliseconds at a time through an AVAudioSourceNode -- pulled
// rather than pushed, so that what it is handed where the buffer runs dry is a fade and not the
// middle of a waveform. See JitterBuffer for why it holds a quarter of a second back first.

import AVFoundation
import os
import WaifuKit

/// One streaming reading's playback. Parts are added from the engine's thread, in the order they
/// are handed over -- not through the main actor, whose tasks keep no order. Once stopped it fades
/// out and is silent for good: a part the engine hands over after a Stop is dropped. Once finished
/// it plays what it holds and then lets the audio hardware go.
final class StreamPlayer: @unchecked Sendable {
    /// What is held back before playing starts, and how long a fade is.
    static let prerollSeconds = 0.25
    static let fadeSeconds = 0.005

    /// The samples, between the engine's thread and the audio thread, which takes this lock for as
    /// long as it takes to copy a few milliseconds out.
    private struct Shared {
        var buffer: JitterBuffer
        /// Whether the audio thread has said, once, that everything has been played.
        var said = false
    }
    private let shared = OSAllocatedUnfairLock<Shared?>(initialState: nil)

    /// The rest of its state, on one serial queue: the audio engine is started and stopped here,
    /// never on the audio thread.
    private let queue = DispatchQueue(label: "waifu.stream-player")
    private var engine: AVAudioEngine?
    private var rate = 0
    private var stopped = false
    private var finished = false
    private var parts = 0
    /// Which engine the audio thread's word that it has played everything is about.
    private var generation = 0
    /// How many times the reading before the one playing, if any, ran dry.
    private var ranDry = 0
    /// Who is waiting for it to have played out.
    private var waiting: [CheckedContinuation<Void, Never>] = []

    /// How many parts have been added: none means nothing of the reading was heard.
    var heard: Int {
        queue.sync { parts }
    }

    /// Whether the audio engine is running: from the first part until what is held has been played
    /// after `finish`, or faded out after `stop`.
    var playing: Bool {
        queue.sync { engine != nil }
    }

    /// How many times the reading ran dry -- a part came after the one before had played out -- and
    /// faded out to wait for it: none where the preroll covered every part's lateness.
    var underruns: Int {
        queue.sync { shared.withLock { $0?.buffer.underruns } ?? ranDry }
    }

    /// Adds `part` after the ones before it, starting the audio engine with the first.
    func play(_ part: AudioClip) {
        queue.sync {
            guard !stopped, !finished, !part.samples.isEmpty, part.rate > 0 else { return }
            // Every part of a reading is at its voice's rate; one at another starts afresh.
            if engine == nil || part.rate != rate {
                guard start(rate: part.rate) else { return }
            }
            shared.withLock { $0?.buffer.push(part.samples) }
            parts += 1
        }
    }

    /// No more parts: what is held plays out, held back no longer, and the audio hardware is let go.
    func finish() {
        queue.sync {
            finished = true
            shared.withLock { $0?.buffer.finish() }
        }
    }

    /// Silences it, after a fade of a few milliseconds, and every part after.
    func stop() {
        queue.sync {
            stopped = true
            shared.withLock { $0?.buffer.stop() }
            // The fade is the audio thread's to play; where it has stopped pulling -- the device
            // went away -- the engine is let go anyway, so that a Stop always ends it.
            let ours = generation
            queue.asyncAfter(deadline: .now() + 0.1) { self.playedOut(ours) }
        }
    }

    /// Returns once what was handed over has been played, after `finish`, or faded out, after
    /// `stop`, and the audio hardware let go: at once where nothing was ever played.
    func playedOut() async {
        await withCheckedContinuation { (waiter: CheckedContinuation<Void, Never>) in
            queue.async {
                if self.engine == nil { waiter.resume() } else { self.waiting.append(waiter) }
            }
        }
    }

    private func start(rate: Int) -> Bool {
        tearDown()
        guard let format = AVAudioFormat(standardFormatWithSampleRate: Double(rate), channels: 1) else {
            return false
        }
        shared.withLock {
            $0 = Shared(
                buffer: JitterBuffer(
                    preroll: Int(Double(rate) * Self.prerollSeconds),
                    fade: Int(Double(rate) * Self.fadeSeconds)))
        }
        let ours = generation
        let source = AVAudioSourceNode(format: format) { [weak self] _, _, frameCount, audioBufferList in
            guard let self else { return noErr }
            let buffers = UnsafeMutableAudioBufferListPointer(audioBufferList)
            guard let data = buffers.first?.mData else { return noErr }
            let out = UnsafeMutableBufferPointer(
                start: data.assumingMemoryBound(to: Float.self), count: Int(frameCount))
            let done = self.shared.withLock { shared -> Bool in
                guard shared != nil else {
                    for i in 0..<out.count { out[i] = 0 }
                    return false
                }
                shared!.buffer.pull(out.count, into: out)
                guard shared!.buffer.drained, !shared!.said else { return false }
                shared!.said = true
                return true
            }
            if done { self.queue.async { self.playedOut(ours) } }
            return noErr
        }
        let engine = AVAudioEngine()
        engine.attach(source)
        engine.connect(source, to: engine.mainMixerNode, format: format)
        do {
            try engine.start()
        } catch {
            shared.withLock { $0 = nil }
            return false
        }
        self.engine = engine
        self.rate = rate
        return true
    }

    /// The audio thread has played everything there is of the engine `which` names.
    private func playedOut(_ which: Int) {
        guard which == generation, engine != nil else { return }
        tearDown()
        let waiters = waiting
        waiting = []
        for waiter in waiters { waiter.resume() }
    }

    private func tearDown() {
        engine?.stop()
        engine = nil
        ranDry = shared.withLock { shared -> Int in
            let ran = shared?.buffer.underruns ?? 0
            shared = nil
            return ran
        }
        generation += 1
    }
}
