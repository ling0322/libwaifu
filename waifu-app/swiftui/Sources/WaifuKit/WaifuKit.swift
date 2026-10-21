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

    /// The same, from libwaifu_modelmanager, which has codes of its own.
    init(code: WaifuModelManagerStatusCode, message: String) {
        switch code {
        case WAIFU_MODELMANAGER_ERR_INVALID_ARGUMENT: self = .refused(message)
        case WAIFU_MODELMANAGER_ERR_UNKNOWN_MODEL: self = .unknownModel(message)
        case WAIFU_MODELMANAGER_ERR_FETCH: self = .fetch(message)
        default: self = .internalError(message)
        }
    }
}

/// This thread's last message from libwaifu.
func lastError() -> String {
    guard let said = waifu_last_error() else { return "" }
    return String(cString: said)
}

/// This thread's last message from libwaifu_modelmanager, which keeps its own.
func modelManagerLastError() -> String {
    guard let said = waifu_modelmanager_last_error() else { return "" }
    return String(cString: said)
}

/// The libraries' ABIs, and the ones this was compiled against: a mismatch is a library from
/// another build than its header.
public enum ABI {
    public static var library: UInt32 { waifu_abi_version() }
    public static var header: UInt32 { UInt32(WAIFU_ABI_VERSION) }
    public static var modelManagerLibrary: UInt32 { waifu_modelmanager_abi_version() }
    public static var modelManagerHeader: UInt32 { UInt32(WAIFU_MODELMANAGER_ABI_VERSION) }
}

/// The build of libwaifu that is loaded: what a picture's record says drew it.
public struct LibraryVersion: Decodable, Sendable, Equatable {
    /// The crate's version: "0.1.0".
    public var version: String
    /// The full commit it was built from; nil for a build made without git.
    public var revision: String?

    public init(version: String, revision: String?) {
        self.version = version
        self.revision = revision
    }

    public static let current: LibraryVersion? = {
        guard let said = waifu_version_json() else { return nil }
        defer { waifu_string_free(said) }
        return try? JSONDecoder().decode(LibraryVersion.self, from: Data(String(cString: said).utf8))
    }()
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
    case auto, cpu, metal, metalDiskOffload, cuda, cudaCPUOffload, vulkan

    var c: WaifuDevice {
        switch self {
        case .auto: WAIFU_DEVICE_AUTO
        case .cpu: WAIFU_DEVICE_CPU
        case .metal: WAIFU_DEVICE_METAL
        case .metalDiskOffload: WAIFU_DEVICE_METAL_DISK_OFFLOAD
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

    /// A download's progress, as libwaifu_modelmanager says it.
    init(fetching event: WaifuModelManagerProgressEvent) {
        stage = .fetching
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

/// libwaifu_modelmanager's progress and ending, which are its own types, ended as a call is.
private let onFetchProgress: WaifuModelManagerProgressCallback = { userData, event in
    guard let userData, let event else { return }
    let pending = Unmanaged<Pending>.fromOpaque(userData).takeUnretainedValue()
    pending.progress?(JobProgress(fetching: event.pointee))
}

private let onFetchComplete: WaifuModelManagerCompleteCallback = { userData, event in
    guard let userData, let event else { return }
    let pending = Unmanaged<Pending>.fromOpaque(userData).takeRetainedValue()
    let status = event.pointee.status
    let said = status.message.map { String(cString: $0) }
    switch status.code {
    case WAIFU_MODELMANAGER_OK: pending.finish(WAIFU_OK, nil, ())
    case WAIFU_MODELMANAGER_CANCELLED: pending.finish(WAIFU_CANCELLED, nil, nil)
    case WAIFU_MODELMANAGER_ERR_UNKNOWN_MODEL: pending.finish(WAIFU_ERR_UNKNOWN_MODEL, said, nil)
    case WAIFU_MODELMANAGER_ERR_FETCH: pending.finish(WAIFU_ERR_FETCH, said, nil)
    case WAIFU_MODELMANAGER_ERR_INVALID_ARGUMENT: pending.finish(WAIFU_ERR_INVALID_ARGUMENT, said, nil)
    default: pending.finish(WAIFU_ERR_INTERNAL, said, nil)
    }
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
    refusal: () -> String = lastError,
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
                continuation.resume(throwing: WaifuError.refused(refusal()))
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

    /// Limits the memory of `device` -- the tensors of whatever runs on it, the weights kept there,
    /// and what it keeps for reuse -- to `bytes`; 0 for no limit but the device's. For the whole
    /// process rather than an engine, so set before an engine reads a model. Only Metal has one.
    public static func setMemoryLimit(_ bytes: UInt64, on device: Device) throws {
        guard waifu_set_memory_limit(device.c, bytes) else {
            throw WaifuError.refused(lastError())
        }
    }

    /// An engine on `device`. On `.metalDiskOffload` and `.cudaCPUOffload`, up to
    /// `deviceWeightBytes` of each model's weights -- the smallest first -- stay on the device and
    /// only the rest are read as they are used, no more than the device has free: `.max` fills
    /// what the memory limit leaves. Any other device refuses more than 0.
    public init(device: Device, deviceWeightBytes: UInt64 = 0) throws {
        var options = WaifuEngineOptions(
            struct_size: UInt32(MemoryLayout<WaifuEngineOptions>.size),
            device: device.c,
            device_weight_bytes: deviceWeightBytes)
        guard let handle = waifu_engine_new_with_options(&options) else {
            throw WaifuError.refused(lastError())
        }
        self.handle = handle
    }

    deinit {
        waifu_engine_free(handle)
    }

    /// What it runs on, as waifu_devices_json names it: "metal", "cpu", "metal_disk_offload". Never
    /// "auto", which the library settles when the engine starts.
    public var device: String {
        guard let said = waifu_engine_device(handle) else { return "" }
        defer { waifu_string_free(said) }
        return String(cString: said)
    }

    private var cancel: (WaifuJob) -> Void {
        { [handle] job in waifu_engine_cancel(handle, job) }
    }

    /// Fetches `model` if it has to -- a published name, through libwaifu_modelmanager -- then
    /// reads it onto the device, dropping the one before. `model` is a published name, the path
    /// of a manifest, or "tones". Throws CancellationError where it was cancelled.
    public func load(_ model: String, progress: ProgressHandler? = nil) async throws {
        let manifest = try await ModelManager.fetchedManifest(model, progress: progress)
        let loaded: Void? = try await call(progress: progress, cancel: cancel) { userData in
            manifest.withCString { manifest in
                waifu_engine_load_async(handle, manifest, userData, onProgress, onComplete)
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

/// What is published, what is here, and fetching it: libwaifu_modelmanager's. None of it needs an
/// engine; describeJSON asks libwaifu what the manifest the model manager found says.
public enum ModelManager {
    /// The published models, as the library's JSON: `{"models": [...]}`.
    public static func catalogJSON() throws -> Data {
        guard let said = waifu_modelmanager_catalog_json() else {
            throw WaifuError.refused(modelManagerLastError())
        }
        defer { waifu_modelmanager_string_free(said) }
        return Data(String(cString: said).utf8)
    }

    /// What a model is without reading it, as the library's JSON: read by libwaifu out of the
    /// manifest libwaifu_modelmanager finds for it, whose weights need not be here. Throws for a
    /// name with no manifest here -- fetch it, or every manifest, first -- and for one that is
    /// neither published nor a manifest on the disk.
    public static func describeJSON(_ model: String) throws -> Data {
        let manifest = model == Self.tones ? model : try manifestPath(model)
        guard let said = manifest.withCString({ waifu_describe_json($0) }) else {
            throw WaifuError.refused(lastError())
        }
        defer { waifu_string_free(said) }
        let described = try withDefaults(Data(String(cString: said).utf8), manifest)
        return try named(described, model)
    }

    /// What libwaifu says a model can be asked for, with what libwaifu_modelmanager says its
    /// boxes start at laid in: the two halves of one description. A `conversion` is in both, and
    /// its defaults go inside it.
    static func withDefaults(_ described: Data, _ manifest: String) throws -> Data {
        guard let said = manifest.withCString({ waifu_modelmanager_defaults_json($0) }) else {
            throw WaifuError.refused(modelManagerLastError())
        }
        defer { waifu_modelmanager_string_free(said) }
        guard var whole = try JSONSerialization.jsonObject(with: described) as? [String: Any],
              let defaults = try JSONSerialization.jsonObject(with: Data(String(cString: said).utf8))
                as? [String: Any]
        else { return described }
        for (key, value) in defaults {
            if var inner = whole[key] as? [String: Any], let more = value as? [String: Any] {
                inner.merge(more) { _, new in new }
                whole[key] = inner
            } else {
                whole[key] = value
            }
        }
        return try JSONSerialization.data(withJSONObject: whole)
    }

    /// libwaifu calls a model by its manifest's path and file name; one asked for by its published
    /// name is called that, and what the catalogue calls it, here too.
    static func named(_ described: Data, _ model: String) throws -> Data {
        guard let listed = try? listed(model),
              var said = try JSONSerialization.jsonObject(with: described) as? [String: Any]
        else { return described }
        said["name"] = model
        said["full_name"] = listed.full_name
        if var conversion = said["conversion"] as? [String: Any] {
            conversion["name"] = model
            conversion["full_name"] = listed.full_name
            said["conversion"] = conversion
        }
        return try JSONSerialization.data(withJSONObject: said)
    }

    /// The stand-in voice, which is in libwaifu itself and has no manifest.
    static let tones = "tones"

    /// The manifest of a published model here -- fetched, or bundled -- or a manifest's own path,
    /// handed back as it is: what libwaifu is given. Throws where there is none here.
    public static func manifestPath(_ model: String) throws -> String {
        guard let said = model.withCString({ waifu_modelmanager_manifest_path($0) }) else {
            throw WaifuError.unknownModel(modelManagerLastError())
        }
        defer { waifu_modelmanager_string_free(said) }
        return String(cString: said)
    }

    /// The manifest of `model` with its weights beside it, fetching it first where it is a
    /// published model that is not all here. What a load is handed.
    static func fetchedManifest(_ model: String, progress: ProgressHandler?) async throws -> String {
        if model == tones || FileManager.default.fileExists(atPath: model) { return model }
        if try !isCached(model) { try await fetch(model, progress: progress) }
        return try manifestPath(model)
    }

    /// A published model, as the catalogue lists it.
    struct Listed: Decodable {
        var name: String
        var versioned_name: String?
        var full_name: String
        var cached: Bool
    }

    /// What the catalogue says of `model`. Throws for a name it does not list.
    static func listed(_ model: String) throws -> Listed {
        struct Catalog: Decodable {
            var models: [Listed]
        }
        let catalog = try JSONDecoder().decode(Catalog.self, from: catalogJSON())
        guard let listed = catalog.models.first(where: { $0.name == model }) else {
            throw WaifuError.unknownModel("there is no model called \"\(model)\"")
        }
        return listed
    }

    /// The catalogue's name for `model` with its version, the alias followed: "anima:turbo" is
    /// "anima:turbo:v1.1". Nil for what the catalogue does not list, a manifest's path included.
    public static func versionedName(_ model: String) -> String? {
        try? listed(model).versioned_name
    }

    /// Whether every package of a published model is here, as the catalogue says.
    static func isCached(_ model: String) throws -> Bool {
        try listed(model).cached
    }

    public static var directory: String? {
        guard let said = waifu_modelmanager_directory() else { return nil }
        defer { waifu_modelmanager_string_free(said) }
        return String(cString: said)
    }

    /// Downloads go to `path` from now on; nil for the default.
    public static func setDirectory(_ path: String?) throws {
        let code = path.map { $0.withCString { waifu_modelmanager_set_directory($0) } }
            ?? waifu_modelmanager_set_directory(nil)
        if code != WAIFU_MODELMANAGER_OK { throw WaifuError(code: code, message: modelManagerLastError()) }
    }

    /// Reads published models' manifests out of `path` where the download directory has none: the
    /// ones an app was built with, so that what a model suggests is known before it is fetched.
    /// nil for none.
    public static func setBundledManifests(_ path: String?) throws {
        let code = path.map { $0.withCString { waifu_modelmanager_set_bundled_manifests($0) } }
            ?? waifu_modelmanager_set_bundled_manifests(nil)
        if code != WAIFU_MODELMANAGER_OK { throw WaifuError(code: code, message: modelManagerLastError()) }
    }

    public static func remove(_ name: String) throws {
        let code = name.withCString { waifu_modelmanager_remove($0) }
        if code != WAIFU_MODELMANAGER_OK { throw WaifuError(code: code, message: modelManagerLastError()) }
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
            refusal: modelManagerLastError,
            submit: { userData in
                handle.fetch = model.withCString {
                    waifu_modelmanager_fetch_async($0, userData, onFetchProgress, onFetchComplete)
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
            refusal: modelManagerLastError,
            submit: { userData in
                handle.fetch = waifu_modelmanager_fetch_manifests_async(userData, onFetchProgress, onFetchComplete)
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

    /// The same for libwaifu_modelmanager, whose event is its own type.
    private static let managerCallback: WaifuModelManagerLogCallback = { userData, event in
        guard let userData, let event else { return }
        let box = Unmanaged<Box>.fromOpaque(userData).takeUnretainedValue()
        let level = LogLevel(rawValue: Int(event.pointee.level.rawValue)) ?? .info
        box.handler(
            level,
            event.pointee.source.map { String(cString: $0) } ?? "",
            event.pointee.message.map { String(cString: $0) } ?? "")
    }

    /// Every line either library would print goes to `handler` from now on, on whichever thread
    /// wrote it; nil puts them back on the console. Set it before anything else is asked of the
    /// libraries, to hear what hardware libwaifu found.
    public static func setHandler(_ handler: LogHandler?) {
        lock.lock()
        defer { lock.unlock() }
        let previous = current
        if let handler {
            let box = Unmanaged.passRetained(Box(handler))
            waifu_set_log_callback(box.toOpaque(), callback)
            waifu_modelmanager_set_log_callback(box.toOpaque(), managerCallback)
            current = box
        } else {
            waifu_set_log_callback(nil, nil)
            waifu_modelmanager_set_log_callback(nil, nil)
            current = nil
        }
        // After the library has stopped calling it.
        previous?.release()
    }

    public static func setLevel(_ level: LogLevel) {
        waifu_set_log_level(WaifuLogLevel(rawValue: UInt32(level.rawValue)))
        waifu_modelmanager_set_log_level(WaifuModelManagerLogLevel(rawValue: UInt32(level.rawValue)))
    }
}
