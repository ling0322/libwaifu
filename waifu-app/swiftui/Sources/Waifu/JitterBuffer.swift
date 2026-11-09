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

// The samples of a streaming reading between the engine, which hands them over a part at a time
// and not evenly, and the audio hardware, which takes them a few milliseconds at a time and
// evenly: what StreamPlayer's source node pulls from.
//
// A reading's first part is short -- CosyVoice3's first hop is 0.88 s of sound -- and the second
// takes longer to make than the first takes to play: on an M5 Pro it came 0.18 s after the first
// had played out, every reading. Played as it came, the sound stopped mid-waveform there and
// started again mid-waveform, two clicks. So this waits `preroll` samples of the hardware's time
// once it has something, before it plays -- a delay, not an amount held, for the parts come whole
// and the first alone is more than any amount worth holding -- and the gap fits in that. Where it
// runs dry all the same -- a slower Mac -- it fades out over the last few milliseconds it has, and
// once more comes, waits `preroll` again and fades in: a pause, not a click.
//
// No AVAudio here, so that it is tested as it is.

import Foundation

struct JitterBuffer {
    /// Samples of the hardware's time waited, once something is held, before playing starts, or
    /// starts again after running dry.
    let preroll: Int
    /// Samples a fade in or out takes.
    let fade: Int

    private var samples: [Float] = []
    private var read = 0
    private var playing = false
    /// Samples waited, while held back with something to play.
    private var waited = 0
    private var finished = false
    private var stopping = false
    /// Samples into the fade in, from 0 to `fade`; `fade` once it is done.
    private var fadedIn = 0
    /// The fade out under way, which once begun runs to silence across as many pulls as it takes:
    /// the samples left of it, and how many it began with.
    private var fadeOutLeft: Int?
    private var fadeOutLength = 0
    /// How many times it ran dry while there was more to come.
    private(set) var underruns = 0

    init(preroll: Int, fade: Int) {
        self.preroll = max(0, preroll)
        self.fade = max(1, fade)
        fadedIn = self.fade
    }

    /// Samples held and not yet played.
    var held: Int { samples.count - read }

    /// Whether everything there will be has been played: finished and played out, or stopped.
    var drained: Bool { (finished || stopping) && held == 0 }

    /// The next part's samples, after those held. Nothing after a stop or a finish.
    mutating func push(_ part: [Float]) {
        guard !stopping, !finished else { return }
        // What has been played goes before the array grows, here rather than on the audio thread.
        if read > 0 && read >= samples.count / 2 {
            samples.removeFirst(read)
            read = 0
        }
        samples.append(contentsOf: part)
    }

    /// No more parts: what is held plays out, held back no longer.
    mutating func finish() {
        finished = true
    }

    /// Silence, after a fade out over what little is held.
    mutating func stop() {
        stopping = true
    }

    /// The next `count` samples into `out`, silence where there is nothing to play.
    mutating func pull(_ count: Int, into out: UnsafeMutableBufferPointer<Float>) {
        var at = 0
        func silence() {
            while at < count {
                out[at] = 0
                at += 1
            }
        }

        while at < count {
            if !playing {
                // Stopped before it started: nothing has been heard, so nothing is.
                if stopping {
                    read = samples.count
                    return silence()
                }
                // Held back until it has waited `preroll` with something to play, or holds all
                // there will be: one sample of silence, of the wait if there is something held.
                if held == 0 || (waited < preroll && !finished) {
                    if held > 0 { waited += 1 }
                    out[at] = 0
                    at += 1
                    continue
                }
                playing = true
                waited = 0
                fadedIn = 0
            }
            if fadeOutLeft == nil {
                if stopping {
                    // Out over what is held, at most a fade's worth of it.
                    beginFadeOut(min(held, fade))
                } else if !finished && held <= fade {
                    // About to run dry with more to come: the last of what is held fades out
                    // now, while there is still some to fade.
                    beginFadeOut(held)
                }
            }
            if let left = fadeOutLeft {
                if left == 0 {
                    fadeOutLeft = nil
                    playing = false
                    if stopping {
                        read = samples.count
                        return silence()
                    }
                    underruns += 1
                    continue
                }
                out[at] = samples[read] * gainIn() * Float(left) / Float(fadeOutLength + 1)
                fadeOutLeft = left - 1
            } else {
                // Played out, with no more to come; the end of a reading is quiet already.
                if held == 0 {
                    playing = false
                    continue
                }
                out[at] = samples[read] * gainIn()
            }
            if fadedIn < fade { fadedIn += 1 }
            read += 1
            at += 1
        }
    }

    private mutating func beginFadeOut(_ length: Int) {
        fadeOutLength = length
        fadeOutLeft = length
    }

    /// The fade in's gain on the next sample.
    private func gainIn() -> Float {
        fadedIn >= fade ? 1 : Float(fadedIn + 1) / Float(fade + 1)
    }

    /// The same, into an array: for tests.
    mutating func pull(_ count: Int) -> [Float] {
        var out = [Float](repeating: 0, count: count)
        out.withUnsafeMutableBufferPointer { pull(count, into: $0) }
        return out
    }
}
