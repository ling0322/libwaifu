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

// The engine, and getting a model into it: done by a job, before it runs, when the model it is of
// is not the one loaded.
//
// One engine for as long as the app runs, on the device the Model card chose -- a new one only when
// the device changes, or how much of the model metal_disk_offload keeps on the GPU, which is decided
// when a model is read. Its progress -- the download, then the read -- is the job's bar until the
// job has steps of its own. What the library writes to its log is kept here too, for when that
// line is not enough.

import Foundation
import Observation
import WaifuKit

/// Where a model is read to: a device, as the Model card names it, and the most of the GPU's
/// memory it may hold there -- 0 for no limit but the Mac's. Only metal_disk_offload has one: it
/// fills what the limit leaves with the model's weights, the smallest first, and reads the rest
/// off the disk as they are used, moving weights back off the GPU when a job needs the room.
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
        /// `model` is on `placement`.
        case ready(model: String, placement: Placement)
        case failed(String)
    }

    private(set) var state: State = .idle

    /// What the library has written, newest last, cut to the last few hundred lines.
    private(set) var log: [String] = []

    private var engine: Engine?
    private var enginePlacement: Placement?

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

    /// Whether `model` on `placement` is what is loaded now.
    func has(_ model: String, on placement: Placement) -> Bool {
        if case .ready(model, placement) = state { true } else { false }
    }

    /// What is loaded now, and where.
    var loaded: (model: String, placement: Placement)? {
        if case .ready(let model, let placement) = state { (model, placement) } else { nil }
    }

    /// The engine with `model` on `placement`: the one there is where it is already loaded, and
    /// otherwise the one it has just been fetched and read into.
    func ensure(_ model: String, on placement: Placement) async throws -> Engine {
        if has(model, on: placement), let engine { return engine }
        // The one there goes before the next is read, and says so: two models are never on one
        // card, and memory the old one still holds is memory the new one is read into.
        if case .ready(let old, placement) = state, let engine {
            state = .loading(model: model, line: "unloading \(old)", fraction: nil)
            do {
                try await engine.unload()
            } catch {
                state = .failed(error.localizedDescription)
                throw error
            }
            state = .idle
        }
        return try await load(model, on: placement)
    }

    /// Loads `model` onto `placement`, and hands back the engine it is on. Cancelling the task
    /// that awaits this stops the download between files; the read cannot be stopped.
    func load(_ model: String, on placement: Placement) async throws -> Engine {
        let engine: Engine
        do {
            engine = try self.engine(on: placement)
        } catch {
            state = .failed(error.localizedDescription)
            throw error
        }

        state = .loading(model: model, line: "starting \(model)", fraction: nil)
        do {
            try await engine.load(model) { progress in
                Task { @MainActor in
                    guard case .loading(let model, _, _) = self.state else { return }
                    self.state = .loading(model: model, line: progress.words, fraction: progress.fraction)
                }
            }
            state = .ready(model: model, placement: placement)
            return engine
        } catch is CancellationError {
            state = .idle
            throw CancellationError()
        } catch {
            state = .failed(error.localizedDescription)
            throw error
        }
    }

    /// An engine for `placement`: the one there is where it was made for it, and a new one
    /// otherwise -- which reads the model again, since what stays on the GPU is decided as it is.
    private func engine(on placement: Placement) throws -> Engine {
        if let engine, enginePlacement == placement { return engine }
        // Freed first: two engines would be two models on one card.
        engine = nil

        // The limit before the engine, which fills what it leaves with weights as it reads the
        // model. The process's rather than the engine's, so lifted again for any other device --
        // where there is Metal to lift it on.
        let offload = placement.device == "metal_disk_offload"
        if offload {
            try Engine.setMemoryLimit(placement.memoryLimitBytes, on: .metal)
        } else {
            try? Engine.setMemoryLimit(0, on: .metal)
        }

        let made = try Engine(device: Launcher.device(placement.device))
        engine = made
        enginePlacement = placement
        return made
    }

    private static func device(_ name: String) -> Device {
        switch name {
        case "metal": .metal
        case "metal_disk_offload": .metalDiskOffload
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
