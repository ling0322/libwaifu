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

// libwaifu's C API, as Swift: async functions where it has `_async` ones, errors that throw, and
// Swift's own types for what goes in and comes out. See docs/ffi.md for what is promised.
//
// Each `_async` call is a continuation. What it needs when a callback arrives -- the continuation,
// and the progress closure -- is a `Pending`, passed to C as `user_data` with one retain that
// `on_complete` takes back. A call refused at the door calls nothing, so it takes it back itself.
// Cancelling the task cancels the job.

import CWaifu
import Foundation

// MARK: - errors

/// Why a call or a job failed: the C API's status codes, with the library's message.
public enum WaifuError: Error, LocalizedError, Equatable {
    /// Refused at the call: a request no model could run, a string that is not UTF-8.
    case refused(String)
    case unknownModel(String)
    case fetch(String)
    case model(String)
    case notLoaded(String)
    /// A bug in the library.
    case internalError(String)

    public var errorDescription: String? {
        switch self {
        case .refused(let said), .unknownModel(let said), .fetch(let said), .model(let said),
            .notLoaded(let said), .internalError(let said):
            said
        }
    }

    init(code: WaifuStatusCode, message: String) {
        switch code {
        case WAIFU_ERR_INVALID_ARGUMENT: self = .refused(message)
        case WAIFU_ERR_UNKNOWN_MODEL: self = .unknownModel(message)
        case WAIFU_ERR_FETCH: self = .fetch(message)
        case WAIFU_ERR_MODEL: self = .model(message)
        case WAIFU_ERR_NOT_LOADED: self = .notLoaded(message)
        default: self = .internalError(message)
        }
    }
}

/// This thread's last message from the library.
func lastError() -> String {
    guard let said = waifu_last_error() else { return "" }
    return String(cString: said)
}

/// The library's ABI, and the one this was compiled against: a mismatch is a libwaifu.dylib from
/// another build than waifu.h.
public enum ABI {
    public static var library: UInt32 { waifu_abi_version() }
    public static var header: UInt32 { UInt32(WAIFU_ABI_VERSION) }
}

// MARK: - what goes in and comes out

/// A picture as rows of RGB, three bytes a pixel, top to bottom.
public struct RGBImage: Sendable, Equatable {
    public var width: Int
    public var height: Int
    public var rgb: [UInt8]

    public init(width: Int, height: Int, rgb: [UInt8]) {
        self.width = width
        self.height = height
        self.rgb = rgb
    }
}

/// Mono samples, -1 to 1, and their rate.
public struct AudioClip: Sendable, Equatable {
    public var samples: [Float]
    public var rate: Int

    public init(samples: [Float], rate: Int) {
        self.samples = samples
        self.rate = rate
    }

    public var seconds: Double { rate > 0 ? Double(samples.count) / Double(rate) : 0 }
}

public enum Device: Sendable {
    case auto, cpu, metal, cuda, cudaCPUOffload, vulkan

    var c: WaifuDevice {
        switch self {
        case .auto: WAIFU_DEVICE_AUTO
        case .cpu: WAIFU_DEVICE_CPU
        case .metal: WAIFU_DEVICE_METAL
        case .cuda: WAIFU_DEVICE_CUDA
        case .cudaCPUOffload: WAIFU_DEVICE_CUDA_CPU_OFFLOAD
        case .vulkan: WAIFU_DEVICE_VULKAN
        }
    }
}

public struct DrawRequest: Sendable {
    /// Words, and <|key|> where an image under that key is to be read.
    public var prompt: String
    public var negative: String = ""
    /// By key: a name the prompt writes as <|key|>, or "start_from" and "control".
    public var images: [(key: String, image: RGBImage)] = []
    public var width: Int
    public var height: Int
    public var steps: Int
    public var guidance: Double
    public var seed: UInt64
    /// With "start_from": how far it walks away from it.
    public var strength: Double = 0.8
    /// With "control": how hard it holds to it.
    public var controlScale: Double = 1.0

    public init(prompt: String, width: Int, height: Int, steps: Int, guidance: Double, seed: UInt64) {
        self.prompt = prompt
        self.width = width
        self.height = height
        self.steps = steps
        self.guidance = guidance
        self.seed = seed
    }
}

public struct SpeakRequest: Sendable {
    public var text: String
    public var like: AudioClip?
    public var speed: Double
    public var temperature: Double
    public var style: String?
    public var seed: UInt64

    public init(text: String, like: AudioClip? = nil, speed: Double, temperature: Double, style: String? = nil, seed: UInt64) {
        self.text = text
        self.like = like
        self.speed = speed
        self.temperature = temperature
        self.style = style
        self.seed = seed
    }
}

public struct VoiceConversionRequest: Sendable {
    public var source: AudioClip
    public var reference: AudioClip
    public var steps: Int
    public var convertStyle: Bool
    public var seed: UInt64

    public init(source: AudioClip, reference: AudioClip, steps: Int, convertStyle: Bool, seed: UInt64) {
        self.source = source
        self.reference = reference
        self.steps = steps
        self.convertStyle = convertStyle
        self.seed = seed
    }
}

// MARK: - progress

public enum Stage: Sendable {
    case fetching, reading, encoding, drawing, decoding, listening, saying, sounding

    init(_ stage: WaifuStage) {
        switch stage {
        case WAIFU_STAGE_FETCHING: self = .fetching
        case WAIFU_STAGE_READING: self = .reading
        case WAIFU_STAGE_ENCODING: self = .encoding
        case WAIFU_STAGE_DRAWING: self = .drawing
        case WAIFU_STAGE_DECODING: self = .decoding
        case WAIFU_STAGE_LISTENING: self = .listening
        case WAIFU_STAGE_SAYING: self = .saying
        default: self = .sounding
        }
    }
}

/// How far along a job is.
public struct JobProgress: Sendable {
    public var stage: Stage
    /// Of the whole job, 0 to 1, or nil where nothing can say -- reading weights.
    public var fraction: Double?
    public var done: UInt64
    public var total: UInt64
    public var part: Int
    public var parts: Int
    public var file: String?
    /// "step 3 of 8", for a status line.
    public var words: String
    public var seconds: Double

    init(_ event: WaifuProgressEvent) {
        stage = Stage(event.stage)
        fraction = event.fraction < 0 ? nil : event.fraction
        done = event.done
        total = event.total
        part = Int(event.part)
        parts = Int(event.parts)
        file = event.file.map { String(cString: $0) }
        words = event.words.map { String(cString: $0) } ?? ""
        seconds = event.seconds
    }
}

public typealias ProgressHandler = @Sendable (JobProgress) -> Void

// MARK: - the plumbing every async call shares

/// What a callback needs when it arrives: the progress closure, and how to end the call.
final class Pending: @unchecked Sendable {
    let progress: ProgressHandler?
    let finish: (WaifuStatusCode, String?, Any?) -> Void

    init(progress: ProgressHandler?, finish: @escaping (WaifuStatusCode, String?, Any?) -> Void) {
        self.progress = progress
        self.finish = finish
    }
}

/// The job a call was given, for a cancel that may arrive before it was.
final class Ticket: @unchecked Sendable {
    private let lock = NSLock()
    private var job: WaifuJob = 0
    private var cancelled = false
    private var cancel: ((WaifuJob) -> Void)?

    func given(_ job: WaifuJob, cancel: @escaping (WaifuJob) -> Void) {
        lock.lock()
        self.job = job
        self.cancel = cancel
        let already = cancelled
        lock.unlock()
        if already { cancel(job) }
    }

    func cancelNow() {
        lock.lock()
        cancelled = true
        let job = job
        let cancel = cancel
        lock.unlock()
        if job != 0 { cancel?(job) }
    }
}

private let onProgress: WaifuProgressCallback = { userData, event in
    guard let userData, let event else { return }
    let pending = Unmanaged<Pending>.fromOpaque(userData).takeUnretainedValue()
    pending.progress?(JobProgress(event.pointee))
}

private func message(_ status: WaifuStatus) -> String? {
    status.message.map { String(cString: $0) }
}

private let onComplete: WaifuCompleteCallback = { userData, event in
    guard let userData, let event else { return }
    let pending = Unmanaged<Pending>.fromOpaque(userData).takeRetainedValue()
    pending.finish(event.pointee.status.code, message(event.pointee.status), ())
}

private let onImage: WaifuImageCompleteCallback = { userData, event in
    guard let userData, let event else { return }
    let pending = Unmanaged<Pending>.fromOpaque(userData).takeRetainedValue()
    // Copied now: the pixels are the library's only until this returns.
    let image = event.pointee.image.map { image -> RGBImage in
        let pixels = image.pointee
        let rgb = pixels.rgb.map { Array(UnsafeBufferPointer(start: $0, count: Int(pixels.len))) } ?? []
        return RGBImage(width: Int(pixels.width), height: Int(pixels.height), rgb: rgb)
    }
    pending.finish(event.pointee.status.code, message(event.pointee.status), image)
}

private let onAudio: WaifuAudioCompleteCallback = { userData, event in
    guard let userData, let event else { return }
    let pending = Unmanaged<Pending>.fromOpaque(userData).takeRetainedValue()
    let clip = event.pointee.audio.map { audio -> AudioClip in
        let sound = audio.pointee
        let samples = sound.samples.map { Array(UnsafeBufferPointer(start: $0, count: Int(sound.count))) } ?? []
        return AudioClip(samples: samples, rate: Int(sound.rate))
    }
    pending.finish(event.pointee.status.code, message(event.pointee.status), clip)
}

/// Memory for one call's strings and buffers, freed when the call has returned: the library
/// copies what it keeps.
final class Scratch {
    private var frees: [() -> Void] = []

    deinit { frees.forEach { $0() } }

    func string(_ text: String) -> UnsafePointer<CChar> {
        let copy = strdup(text)!
        frees.append { free(copy) }
        return UnsafePointer(copy)
    }

    func optionalString(_ text: String?) -> UnsafePointer<CChar>? {
        text.map(string)
    }

    func array<T>(_ values: [T]) -> UnsafePointer<T>? {
        guard !values.isEmpty else { return nil }
        let copy = UnsafeMutablePointer<T>.allocate(capacity: values.count)
        copy.initialize(from: values, count: values.count)
        frees.append {
            copy.deinitialize(count: values.count)
            copy.deallocate()
        }
        return UnsafePointer(copy)
    }

    func one<T>(_ value: T) -> UnsafePointer<T> {
        array([value])!
    }

    func image(_ image: RGBImage) -> UnsafePointer<WaifuImage> {
        one(WaifuImage(width: UInt32(image.width), height: UInt32(image.height), rgb: array(image.rgb), len: image.rgb.count))
    }

    func audio(_ clip: AudioClip) -> UnsafePointer<WaifuAudio> {
        one(WaifuAudio(samples: array(clip.samples), count: clip.samples.count, rate: UInt32(clip.rate)))
    }
}

/// Runs one `_async` call as an async function: its result, nil where it was cancelled, or what
/// it failed of. `submit` makes the C call with the `user_data` it is handed and returns its job.
func call<T>(
    progress: ProgressHandler?,
    cancel: @escaping (WaifuJob) -> Void,
    submit: (UnsafeMutableRawPointer) -> WaifuJob
) async throws -> T? {
    let ticket = Ticket()
    return try await withTaskCancellationHandler {
        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<T?, Error>) in
            let pending = Pending(progress: progress) { code, message, value in
                switch code {
                case WAIFU_OK: continuation.resume(returning: value as? T)
                case WAIFU_CANCELLED: continuation.resume(returning: nil)
                default: continuation.resume(throwing: WaifuError(code: code, message: message ?? ""))
                }
            }
            let userData = Unmanaged.passRetained(pending).toOpaque()
            let job = submit(userData)
            if job == 0 {
                // Refused at the door: nothing will ever be called, so the retain is this side's
                // to take back.
                Unmanaged<Pending>.fromOpaque(userData).release()
                continuation.resume(throwing: WaifuError.refused(lastError()))
                return
            }
            ticket.given(job, cancel: cancel)
        }
    } onCancel: {
        ticket.cancelNow()
    }
}

// MARK: - the engine

/// One thread in the library that holds a model and runs jobs on it, one at a time, in the order
/// asked. Freeing it stops what runs, cancels what waits, and drops the model.
public final class Engine: @unchecked Sendable {
    private let handle: OpaquePointer

    public init(device: Device) throws {
        guard let handle = waifu_engine_new(device.c) else {
            throw WaifuError.refused(lastError())
        }
        self.handle = handle
    }

    deinit {
        waifu_engine_free(handle)
    }

    private var cancel: (WaifuJob) -> Void {
        { [handle] job in waifu_engine_cancel(handle, job) }
    }

    /// Fetches `model` if it has to, then reads it onto the device, dropping the one before.
    /// Throws CancellationError where it was cancelled.
    public func load(_ model: String, progress: ProgressHandler? = nil) async throws {
        let loaded: Void? = try await call(progress: progress, cancel: cancel) { userData in
            model.withCString { model in
                waifu_engine_load_async(handle, model, userData, onProgress, onComplete)
            }
        }
        if loaded == nil { throw CancellationError() }
    }

    public func unload() async throws {
        let _: Void? = try await call(progress: nil, cancel: cancel) { userData in
            waifu_engine_unload_async(handle, userData, onComplete)
        }
    }

    /// The image, or nil where it was cancelled.
    public func draw(_ request: DrawRequest, progress: ProgressHandler? = nil) async throws -> RGBImage? {
        try await call(progress: progress, cancel: cancel) { userData in
            let scratch = Scratch()
            let images = request.images.map { input in
                WaifuImageInput(key: scratch.string(input.key), image: scratch.image(input.image))
            }
            var c = WaifuDrawRequest(
                struct_size: UInt32(MemoryLayout<WaifuDrawRequest>.size),
                prompt: scratch.string(request.prompt),
                negative: scratch.string(request.negative),
                images: scratch.array(images),
                image_count: images.count,
                width: UInt32(clamping: request.width),
                height: UInt32(clamping: request.height),
                steps: UInt32(clamping: request.steps),
                guidance: Float(request.guidance),
                seed: request.seed,
                strength: Float(request.strength),
                control_scale: Float(request.controlScale)
            )
            return withExtendedLifetime(scratch) {
                waifu_engine_draw_async(handle, &c, userData, onProgress, onImage)
            }
        }
    }

    /// The reading, or nil where it was cancelled.
    public func speak(_ request: SpeakRequest, progress: ProgressHandler? = nil) async throws -> AudioClip? {
        try await call(progress: progress, cancel: cancel) { userData in
            let scratch = Scratch()
            var c = WaifuSpeakRequest(
                struct_size: UInt32(MemoryLayout<WaifuSpeakRequest>.size),
                text: scratch.string(request.text),
                like: request.like.map(scratch.audio),
                speed: Float(request.speed),
                temperature: Float(request.temperature),
                style: scratch.optionalString(request.style),
                seed: request.seed
            )
            return withExtendedLifetime(scratch) {
                waifu_engine_speak_async(handle, &c, userData, onProgress, onAudio)
            }
        }
    }

    /// The source said again in the reference's voice, or nil where it was cancelled.
    public func voiceConversion(_ request: VoiceConversionRequest, progress: ProgressHandler? = nil) async throws -> AudioClip? {
        try await call(progress: progress, cancel: cancel) { userData in
            let scratch = Scratch()
            var c = WaifuVoiceConversionRequest(
                struct_size: UInt32(MemoryLayout<WaifuVoiceConversionRequest>.size),
                source: scratch.audio(request.source),
                reference: scratch.audio(request.reference),
                steps: UInt32(clamping: request.steps),
                convert_style: request.convertStyle,
                seed: request.seed
            )
            return withExtendedLifetime(scratch) {
                waifu_engine_voice_conversion_async(handle, &c, userData, onProgress, onAudio)
            }
        }
    }
}

// MARK: - the model manager

/// What is published, what is here, and fetching it. None of it needs an engine.
public enum ModelManager {
    private static func json(_ said: UnsafeMutablePointer<CChar>?) throws -> Data {
        guard let said else { throw WaifuError.refused(lastError()) }
        defer { waifu_string_free(said) }
        return Data(String(cString: said).utf8)
    }

    /// The published models, as the library's JSON: `{"models": [...]}`.
    public static func catalogJSON() throws -> Data {
        try json(waifu_modelmanager_catalog_json())
    }

    /// What a model is without reading it, as the library's JSON. Throws for a name that is
    /// neither published nor a manifest on the disk.
    public static func describeJSON(_ model: String) throws -> Data {
        try json(model.withCString { waifu_modelmanager_describe_json($0) })
    }

    public static var directory: String? {
        guard let said = waifu_modelmanager_directory() else { return nil }
        defer { waifu_string_free(said) }
        return String(cString: said)
    }

    /// Downloads go to `path` from now on; nil for the default.
    public static func setDirectory(_ path: String?) throws {
        let code = path.map { $0.withCString { waifu_modelmanager_set_directory($0) } }
            ?? waifu_modelmanager_set_directory(nil)
        if code != WAIFU_OK { throw WaifuError(code: code, message: lastError()) }
    }

    /// Reads published models' manifests out of `path` where the download directory has none: the
    /// ones an app was built with, so that what a model suggests is known before it is fetched.
    /// nil for none.
    public static func setBundledManifests(_ path: String?) throws {
        let code = path.map { $0.withCString { waifu_modelmanager_set_bundled_manifests($0) } }
            ?? waifu_modelmanager_set_bundled_manifests(nil)
        if code != WAIFU_OK { throw WaifuError(code: code, message: lastError()) }
    }

    public static func remove(_ name: String) throws {
        let code = name.withCString { waifu_modelmanager_remove($0) }
        if code != WAIFU_OK { throw WaifuError(code: code, message: lastError()) }
    }

    /// Fetches a published model without loading it. Throws CancellationError where it was
    /// cancelled.
    public static func fetch(_ model: String, progress: ProgressHandler? = nil) async throws {
        final class Handle: @unchecked Sendable {
            var fetch: OpaquePointer?
        }
        let handle = Handle()
        let fetched: Void? = try await call(
            progress: progress,
            cancel: { _ in waifu_modelmanager_fetch_cancel(handle.fetch) },
            submit: { userData in
                handle.fetch = model.withCString {
                    waifu_modelmanager_fetch_async($0, userData, onProgress, onComplete)
                }
                return handle.fetch == nil ? 0 : 1
            })
        waifu_modelmanager_fetch_free(handle.fetch)
        if fetched == nil { throw CancellationError() }
    }

    /// Fetches the manifest of every published model not here yet, and none of their weights:
    /// what each model suggests, so that it can be read the moment the model is chosen. Finishes
    /// at once when they are all here. Throws CancellationError where it was cancelled.
    public static func fetchManifests(progress: ProgressHandler? = nil) async throws {
        final class Handle: @unchecked Sendable {
            var fetch: OpaquePointer?
        }
        let handle = Handle()
        let fetched: Void? = try await call(
            progress: progress,
            cancel: { _ in waifu_modelmanager_fetch_cancel(handle.fetch) },
            submit: { userData in
                handle.fetch = waifu_modelmanager_fetch_manifests_async(userData, onProgress, onComplete)
                return handle.fetch == nil ? 0 : 1
            })
        waifu_modelmanager_fetch_free(handle.fetch)
        if fetched == nil { throw CancellationError() }
    }
}

// MARK: - the machine

public enum MachineInfo {
    /// The processor, the memory, the card and the accelerators this build can use, as JSON.
    public static func json() -> Data? {
        guard let said = waifu_machine_json() else { return nil }
        defer { waifu_string_free(said) }
        return Data(String(cString: said).utf8)
    }
}

// MARK: - the log

public enum LogLevel: Int, Sendable, Comparable {
    case debug = 0, info, warning, error, fatal

    public static func < (a: LogLevel, b: LogLevel) -> Bool { a.rawValue < b.rawValue }
}

public typealias LogHandler = @Sendable (LogLevel, _ source: String, _ message: String) -> Void

public enum Log {
    final class Box: @unchecked Sendable {
        let handler: LogHandler
        init(_ handler: @escaping LogHandler) { self.handler = handler }
    }

    private static let lock = NSLock()
    nonisolated(unsafe) private static var current: Unmanaged<Box>?

    private static let callback: WaifuLogCallback = { userData, event in
        guard let userData, let event else { return }
        let box = Unmanaged<Box>.fromOpaque(userData).takeUnretainedValue()
        let level = LogLevel(rawValue: Int(event.pointee.level.rawValue)) ?? .info
        box.handler(
            level,
            event.pointee.source.map { String(cString: $0) } ?? "",
            event.pointee.message.map { String(cString: $0) } ?? "")
    }

    /// Every line the library would print goes to `handler` from now on, on whichever thread
    /// wrote it; nil puts them back on the console. Set it before anything else is asked of the
    /// library, to hear what hardware it found.
    public static func setHandler(_ handler: LogHandler?) {
        lock.lock()
        defer { lock.unlock() }
        let previous = current
        if let handler {
            let box = Unmanaged.passRetained(Box(handler))
            waifu_set_log_callback(box.toOpaque(), callback)
            current = box
        } else {
            waifu_set_log_callback(nil, nil)
            current = nil
        }
        // After the library has stopped calling it.
        previous?.release()
    }

    public static func setLevel(_ level: LogLevel) {
        waifu_set_log_level(WaifuLogLevel(rawValue: UInt32(level.rawValue)))
    }
}
