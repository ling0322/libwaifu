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

// The pipeline, and getting a model into it: done by a job, before it runs, when the model it is
// of is not the one loaded for its task.
//
// One pipeline at a time, made for the task of the job that runs, on the device the Model card
// chose -- a new one only when the task changes, or the device, or the most of the GPU's memory the
// model may hold, which is filled as a model is read. A pipeline does one task, so CosyVoice3, which
// reads and converts, is read again for a conversion after a reading: two models are never on one
// card. Its progress -- the download, then the read -- is the job's bar until the job has steps of
// its own. What the library writes to its log is kept here too, for when that line is not enough.

import Foundation
import Observation
import WaifuKit

/// Where a model is read to: a device, as the Model card names it, and the most of the GPU's
/// memory it may hold there -- 0 for no limit but the Mac's. Only Metal has one: it fills what the
/// limit leaves with the model's weights, the smallest first, and reads the rest off the disk as
/// they are used, moving weights back off the GPU when a job needs the room.
struct Placement: Equatable {
    var device: String
    var memoryLimitBytes: UInt64 = 0
}

@MainActor @Observable
final class Launcher {
    enum State {
        case idle
        /// Fetching or reading `model`. `line` is how far along; `fraction` where it can say.
        case loading(model: String, line: String, fraction: Double?)
        /// `model` is on `placement`, in a pipeline for `type`.
        case ready(model: String, type: PipelineType, placement: Placement)
        case failed(String)
    }

    private(set) var state: State = .idle

    /// What the library has written, newest last, cut to the last few hundred lines.
    private(set) var log: [String] = []

    private var pipeline: Pipeline?
    private var pipelinePlacement: Placement?

    init() {
        // Before anything else asks the library anything: the first device query writes what
        // hardware it found, and that is worth having in the log.
        Log.setHandler { [weak self] level, source, message in
            let line = "\(Launcher.name(level)) \(source)] \(message)"
            Task { @MainActor in self?.heard(line) }
        }
    }

    private nonisolated static func name(_ level: LogLevel) -> String {
        switch level {
        case .debug: "DEBUG"
        case .info: "INFO"
        case .warning: "WARNING"
        case .error: "ERROR"
        case .fatal: "FATAL"
        }
    }

    private func heard(_ line: String) {
        log.append(line)
        if log.count > 500 { log.removeFirst(log.count - 500) }
    }

    var isLoading: Bool {
        if case .loading = state { true } else { false }
    }

    /// What a pipeline for a job of `kind` -- "image", "speech", "conversion", "layers" -- is made
    /// for.
    static func pipelineType(for kind: String) -> PipelineType {
        switch kind {
        case "speech": .textToSpeech
        case "conversion": .voiceConversion
        case "layers": .layerDecomposition
        default: .imageGeneration
        }
    }

    /// Whether `model` on `placement`, for `type`, is what is loaded now.
    func has(_ model: String, for type: PipelineType, on placement: Placement) -> Bool {
        if case .ready(model, type, placement) = state { true } else { false }
    }

    /// What is loaded now, for what, and where.
    var loaded: (model: String, type: PipelineType, placement: Placement)? {
        if case .ready(let model, let type, let placement) = state { (model, type, placement) } else { nil }
    }

    /// The pipeline with `model` on `placement`, for `type`: the one there is where it is already
    /// loaded, and otherwise the one it has just been fetched and read into.
    func ensure(_ model: String, for type: PipelineType, on placement: Placement) async throws -> Pipeline {
        if has(model, for: type, on: placement), let pipeline { return pipeline }
        // The one there goes before the next is read, and says so: two models are never on one
        // card, and memory the old one still holds is memory the new one is read into. A pipeline
        // for another task, or another placement, is freed instead, which drops its model.
        if case .ready(let old, type, placement) = state, let pipeline {
            state = .loading(model: model, line: "unloading \(old)", fraction: nil)
            do {
                try await pipeline.unload()
            } catch {
                state = .failed(error.localizedDescription)
                throw error
            }
            state = .idle
        }
        return try await load(model, for: type, on: placement)
    }

    /// Loads `model` onto `placement` for `type`, and hands back the pipeline it is in. Cancelling
    /// the task that awaits this stops the download between files; the read cannot be stopped.
    func load(_ model: String, for type: PipelineType, on placement: Placement) async throws -> Pipeline {
        let pipeline: Pipeline
        do {
            pipeline = try self.pipeline(for: type, on: placement)
        } catch {
            state = .failed(error.localizedDescription)
            throw error
        }

        state = .loading(model: model, line: "starting \(model)", fraction: nil)
        do {
            try await pipeline.load(model) { progress in
                Task { @MainActor in
                    guard case .loading(let model, _, _) = self.state else { return }
                    self.state = .loading(model: model, line: progress.line(reading: model), fraction: progress.fraction)
                }
            }
            state = .ready(model: model, type: type, placement: placement)
            return pipeline
        } catch is CancellationError {
            state = .idle
            throw CancellationError()
        } catch {
            state = .failed(error.localizedDescription)
            throw error
        }
    }

    /// A pipeline for `type` on `placement`: the one there is where it was made for both, and a
    /// new one otherwise -- which reads the model again, since what stays on the GPU is decided as
    /// it is.
    private func pipeline(for type: PipelineType, on placement: Placement) throws -> Pipeline {
        if let pipeline, pipeline.type == type, pipelinePlacement == placement { return pipeline }
        // Freed first, and its model with it: two pipelines would be two models on one card.
        pipeline = nil
        pipelinePlacement = nil
        state = .idle

        // The limit before the pipeline, which fills what it leaves with weights as it reads the
        // model. The process's rather than the pipeline's, so lifted again for any other device --
        // where there is Metal to lift it on.
        // Not thrown on, since auto may be a Mac without Metal.
        try? Device.metal.setMemoryLimit(placement.memoryLimitBytes)

        let made = try Pipeline(type, device: Launcher.device(placement.device))
        pipeline = made
        pipelinePlacement = placement
        return made
    }

    private static func device(_ name: String) -> Device {
        switch name {
        case "metal": .metal
        case "cpu": .cpu
        case "cuda": .cuda
        case "vulkan": .vulkan
        default: .auto
        }
    }

    /// Where what is made is kept: the app's own folder in Application Support -- not inside the app,
    /// which is signed, often not writable, and replaced whole by the next version.
    static var defaultOutput: URL {
        let support = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
        return support.appendingPathComponent("libwaifu/output")
    }
}
