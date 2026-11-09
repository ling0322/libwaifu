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
// Swift's own types for what goes in and comes out. See docs/ffi-abi4.md for what libwaifu
// promises, and docs/ffi.md for libwaifu_modelmanager.
//
// Each `_async` call is a continuation. What it needs when a callback arrives -- the continuation,
// and the progress closure -- is a `Pending`, passed to C as `user_data` with one retain that
// `on_complete` takes back. A call refused at the door calls nothing, so it takes it back itself.
// Cancelling the task cancels the job.
//
// Every enum of waifu.h is an int32_t and a list of #defines, which a newer library may add to:
// each is a Swift enum here, with a way out for a number this does not know.

import CWaifu
import Foundation

// MARK: - errors

/// Why a call or a job failed: the C API's status codes, with the library's message.
public enum WaifuError: Error, LocalizedError, Equatable {
    /// Refused at the call: a request no model could run, a string that is not UTF-8, a model the
    /// pipeline's task is not supported by.
    case refused(String)
    case unknownModel(String)
    /// A download failed: libwaifu_modelmanager's.
    case fetch(String)
    case model(String)
    case notLoaded(String)
    /// A pipeline asked for on a device this build or machine cannot use.
    case deviceUnavailable(String)
    /// A bug in the library.
    case internalError(String)

    public var errorDescription: String? {
        switch self {
        case .refused(let said), .unknownModel(let said), .fetch(let said), .model(let said),
            .notLoaded(let said), .deviceUnavailable(let said), .internalError(let said):
            said
        }
    }

    init(code: WaifuStatusCode, message: String) {
        switch code {
        case WAIFU_ERR_INVALID_ARGUMENT: self = .refused(message)
        case WAIFU_ERR_UNKNOWN_MODEL: self = .unknownModel(message)
        case WAIFU_ERR_MODEL: self = .model(message)
        case WAIFU_ERR_NOT_LOADED: self = .notLoaded(message)
        case WAIFU_ERR_DEVICE_UNAVAILABLE: self = .deviceUnavailable(message)
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

/// This thread's last refused call's sentence, from libwaifu.
func lastError() -> String {
    guard let said = waifu_last_error() else { return "" }
    return String(cString: said)
}

/// This thread's last refused call in libwaifu, as the error it was: its code and its sentence,
/// which are set together.
func libraryRefusal() -> WaifuError {
    let code = waifu_last_error_code()
    let said = lastError()
    return code == WAIFU_OK ? .refused(said) : WaifuError(code: code, message: said)
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
public struct LibraryVersion: Sendable, Equatable {
    /// The crate's version: "0.1.0".
    public var version: String
    /// The full commit it was built from; nil for a build made without git.
    public var revision: String?

    public init(version: String, revision: String?) {
        self.version = version
        self.revision = revision
    }

    public static let current: LibraryVersion? = {
        guard let version = waifu_version() else { return nil }
        let revision: UnsafePointer<CChar>? = waifu_revision()
        return LibraryVersion(version: String(cString: version), revision: revision.map { String(cString: $0) })
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

/// A picture as rows of RGBA, four bytes a pixel, top to bottom, not premultiplied: what
/// img2layers takes and gives.
public struct RGBAImage: Sendable, Equatable {
    public var width: Int
    public var height: Int
    public var rgba: [UInt8]

    public init(width: Int, height: Int, rgba: [UInt8]) {
        self.width = width
        self.height = height
        self.rgba = rgba
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

public enum Device: Sendable, Equatable {
    case auto, cpu, metal, cuda, vulkan

    var c: WaifuDevice {
        switch self {
        case .auto: WAIFU_DEVICE_AUTO
        case .cpu: WAIFU_DEVICE_CPU
        case .metal: WAIFU_DEVICE_METAL
        case .cuda: WAIFU_DEVICE_CUDA
        case .vulkan: WAIFU_DEVICE_VULKAN
        }
    }

    /// What a device is called, as the Model card and a picture's record name it: "metal", "cpu".
    static func name(_ device: WaifuDevice) -> String {
        switch device {
        case WAIFU_DEVICE_AUTO: "auto"
        case WAIFU_DEVICE_CPU: "cpu"
        case WAIFU_DEVICE_METAL: "metal"
        case WAIFU_DEVICE_CUDA: "cuda"
        case WAIFU_DEVICE_VULKAN: "vulkan"
        default: "device \(device)"
        }
    }

    /// Whether this build, on this machine, can run anything on it.
    public var isAvailable: Bool { waifu_device_is_available(c) }

    /// Limits the memory of this device -- the tensors of whatever runs on it, the weights kept
    /// there, and what it keeps for reuse -- to `bytes`; 0 for no limit but the device's. For the
    /// whole process, shared by every pipeline on it, so set before a model is loaded. The
    /// processor has none, and is refused.
    public func setMemoryLimit(_ bytes: UInt64) throws {
        guard waifu_device_set_memory_limit(c, bytes) else { throw libraryRefusal() }
    }
}

/// What a pipeline is made for: one task, and only that one.
public enum PipelineType: Sendable, Equatable {
    case imageGeneration, textToSpeech, voiceConversion, layerDecomposition, speechToText,
        streamingTextToSpeech

    var c: WaifuPipelineType {
        switch self {
        case .imageGeneration: WAIFU_PIPELINE_IMAGE_GENERATION
        case .textToSpeech: WAIFU_PIPELINE_TEXT_TO_SPEECH
        case .voiceConversion: WAIFU_PIPELINE_VOICE_CONVERSION
        case .layerDecomposition: WAIFU_PIPELINE_LAYER_DECOMPOSITION
        case .speechToText: WAIFU_PIPELINE_SPEECH_TO_TEXT
        case .streamingTextToSpeech: WAIFU_PIPELINE_STREAMING_TEXT_TO_SPEECH
        }
    }

    /// What a model has to support to be loaded into it.
    public var capability: Capability {
        switch self {
        case .imageGeneration: .imageGeneration
        case .textToSpeech: .textToSpeech
        case .voiceConversion: .voiceConversion
        case .layerDecomposition: .layerDecomposition
        case .speechToText: .speechToText
        case .streamingTextToSpeech: .streamingTextToSpeech
        }
    }
}

public struct DrawRequest: Sendable {
    /// Words, and <|key|> where an image under that key in `promptImages` is to be read.
    public var prompt: String
    public var negative: String = ""
    /// The pictures the prompt names, each under the key it writes as <|key|>.
    public var promptImages: [(key: String, image: RGBImage)] = []
    /// The picture to start from: img2img.
    public var startImage: RGBImage?
    /// With a start image: how far it walks away from it.
    public var strength: Double = 0.8
    /// A ControlNet's condition.
    public var controlImage: RGBImage?
    /// With a control image: how hard it holds to it.
    public var controlScale: Double = 1.0
    public var width: Int
    public var height: Int
    public var steps: Int
    public var guidance: Double
    public var seed: UInt64

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
    /// A recording to sound like.
    public var reference: AudioClip?
    public var speed: Double
    public var temperature: Double
    /// Any instruction; nil or "" for the voice's own way.
    public var style: String?
    public var seed: UInt64

    public init(
        text: String, reference: AudioClip? = nil, speed: Double, temperature: Double, style: String? = nil,
        seed: UInt64
    ) {
        self.text = text
        self.reference = reference
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

public struct LayersRequest: Sendable {
    /// The illustration: any size, transparent where it likes.
    public var image: RGBAImage
    /// 0 for the model's own, as are the two below.
    public var steps: Int = 0
    public var depthSteps: Int = 0
    /// The side of the square the layers are drawn at, a multiple of 64.
    public var resolution: Int = 0
    /// Hands, eyes, eyebrows and ears as a -l and a -r layer each.
    public var splitPairs: Bool = false
    /// Hair, hands, top and bottom also cut into a near and a far half by their own depth, the
    /// halves after every part.
    public var stratifyLayers: Bool = false
    public var seed: UInt64

    public init(image: RGBAImage, seed: UInt64) {
        self.image = image
        self.seed = seed
    }
}

/// What a layer of a picture taken apart is.
public enum LayerType: Sendable, Equatable {
    /// A part as the model drew it.
    case part
    /// The near half of a stratified part.
    case near
    /// The far half of a stratified part.
    case far
    /// The depth of another layer, pixel by pixel: the same size and place, red, green and blue
    /// the depth -- 0 nearest, 255 farthest -- under that layer's alpha.
    case depth
    /// A kind a newer library sends that this one does not know: in neither stack.
    case other(Int32)

    init(_ type: WaifuLayerType) {
        switch type {
        case WAIFU_LAYER_PART: self = .part
        case WAIFU_LAYER_STRATIFIED_NEAR: self = .near
        case WAIFU_LAYER_STRATIFIED_FAR: self = .far
        case WAIFU_LAYER_DEPTH: self = .depth
        default: self = .other(type)
        }
    }

    /// A near or a far half.
    public var isHalf: Bool { self == .near || self == .far }
}

/// One layer of a picture taken apart.
public struct Layer: Sendable, Equatable {
    public var type: LayerType
    /// Unique among the layers of one picture taken apart, from 1; it means nothing else.
    public var id: Int
    /// What it is: "front hair", "irides", "topwear"; a half is "topwear-0" or "topwear-1", and a
    /// depth layer "topwear_depth".
    public var name: String
    /// Where its top left corner is on the canvas.
    public var left: Int
    public var top: Int
    /// Only as large as what is drawn.
    public var image: RGBAImage
    /// 0 near to 1 far. A near half has its part's, and a depth layer its layer's.
    public var depth: Float
    /// The id of the layer this one is made from: a half's part, a depth layer's layer; nil for a
    /// part.
    public var derivedFromID: Int?

    public init(
        type: LayerType = .part, id: Int = 0, name: String, left: Int, top: Int, image: RGBAImage,
        depth: Float, derivedFromID: Int? = nil
    ) {
        self.type = type
        self.id = id
        self.name = name
        self.left = left
        self.top = top
        self.image = image
        self.depth = depth
        self.derivedFromID = derivedFromID
    }
}

/// A picture taken apart: its layers on a square canvas, in the one order the library gives them
/// -- every part, bottom first, which is the stack unstratified; then, where they were asked for,
/// each split part's halves, near before far; then a depth layer for each of those, in the same
/// order. A layer names another by its id, never by where it is.
public struct Layers: Sendable, Equatable {
    public var canvas: Int
    public var layers: [Layer]

    public init(canvas: Int, layers: [Layer]) {
        self.canvas = canvas
        self.layers = layers
    }

    /// The parts, bottom first: the stack unstratified.
    public var parts: [Layer] { layers.filter { $0.type == .part } }

    /// The layer whose id is `id`.
    public func layer(id: Int) -> Layer? { layers.first { $0.id == id } }

    /// The depth of `layer`, pixel by pixel, where the library gave one.
    public func depth(of layer: Layer) -> Layer? {
        layers.first { $0.type == .depth && $0.derivedFromID == layer.id }
    }

    /// The stratified stack, bottom first, as indices into `layers`.
    public var stratifiedStack: [Int] {
        Layers.stratifiedStack(
            layers.map { (id: $0.id, type: $0.type, depth: $0.depth, derivedFromID: $0.derivedFromID) })
    }

    /// The stratified stack of `layers` -- in the library's order -- bottom first, as indices into
    /// it: every part that was not split and every half, from the largest depth to the smallest;
    /// at one depth, a far half under a near one; then by place in `layers` (docs/ffi-abi4.md,
    /// "Stratification"). A split part is one a half names by its id; depth layers are in neither
    /// stack.
    public static func stratifiedStack(
        _ layers: [(id: Int, type: LayerType, depth: Float, derivedFromID: Int?)]
    ) -> [Int] {
        let split = Set(layers.filter { $0.type.isHalf }.compactMap { $0.derivedFromID })
        let lower = { (type: LayerType) -> Int in type == .far ? 0 : 1 }
        let stacked = layers.indices.filter { index in
            let layer = layers[index]
            return layer.type.isHalf || (layer.type == .part && !split.contains(layer.id))
        }
        return stacked.sorted { a, b in
            let (x, y) = (layers[a], layers[b])
            if x.depth != y.depth { return x.depth > y.depth }
            if lower(x.type) != lower(y.type) { return lower(x.type) < lower(y.type) }
            return a < b
        }
    }
}

// MARK: - progress

/// What a job is doing: one of libwaifu's stages, one for each thing a job does, or the model
/// manager's download. The words for each are the app's.
public enum Stage: Sendable, Equatable {
    /// Downloading the model: libwaifu_modelmanager's, which words it itself.
    case fetching
    case loadingWeights

    case imageGenerationEncodingPrompt
    /// Steps.
    case imageGenerationDenoising
    case imageGenerationDecodingImage

    case textToSpeechReadingText
    /// Tokens; the total is an estimate, and can be passed.
    case textToSpeechGeneratingTokens
    case textToSpeechSynthesizingAudio

    case voiceConversionReadingRecordings
    /// Tokens; the total is an estimate.
    case voiceConversionGeneratingTokens
    /// Steps.
    case voiceConversionDenoising
    /// Vocoder windows.
    case voiceConversionSynthesizingAudio

    case layerDecompositionEncodingImage
    /// Steps.
    case layerDecompositionDenoisingLayers
    /// Layers.
    case layerDecompositionDecodingLayers
    case layerDecompositionEncodingHead
    /// Steps.
    case layerDecompositionDenoisingHeadLayers
    /// Layers.
    case layerDecompositionDecodingHeadLayers
    /// Layers.
    case layerDecompositionEncodingForDepth
    /// Steps.
    case layerDecompositionDenoisingDepth
    /// Layers.
    case layerDecompositionDecodingDepth
    case layerDecompositionCuttingIntoParts

    /// Pieces the recording is cut into.
    case speechToTextReadingRecording
    /// Tokens; the total is an estimate, and can be passed.
    case speechToTextGeneratingTokens

    case streamingTextToSpeechReadingText
    /// Tokens; the total is an estimate.
    case streamingTextToSpeechGeneratingTokens
    /// Parts handed over.
    case streamingTextToSpeechSynthesizingAudio

    /// A stage a newer library says that this one does not know.
    case other(Int32)

    init(_ stage: WaifuStage) {
        switch stage {
        case WAIFU_STAGE_LOADING_WEIGHTS: self = .loadingWeights
        case WAIFU_STAGE_IMAGE_GENERATION_ENCODING_PROMPT: self = .imageGenerationEncodingPrompt
        case WAIFU_STAGE_IMAGE_GENERATION_DENOISING: self = .imageGenerationDenoising
        case WAIFU_STAGE_IMAGE_GENERATION_DECODING_IMAGE: self = .imageGenerationDecodingImage
        case WAIFU_STAGE_TEXT_TO_SPEECH_READING_TEXT: self = .textToSpeechReadingText
        case WAIFU_STAGE_TEXT_TO_SPEECH_GENERATING_TOKENS: self = .textToSpeechGeneratingTokens
        case WAIFU_STAGE_TEXT_TO_SPEECH_SYNTHESIZING_AUDIO: self = .textToSpeechSynthesizingAudio
        case WAIFU_STAGE_VOICE_CONVERSION_READING_RECORDINGS: self = .voiceConversionReadingRecordings
        case WAIFU_STAGE_VOICE_CONVERSION_GENERATING_TOKENS: self = .voiceConversionGeneratingTokens
        case WAIFU_STAGE_VOICE_CONVERSION_DENOISING: self = .voiceConversionDenoising
        case WAIFU_STAGE_VOICE_CONVERSION_SYNTHESIZING_AUDIO: self = .voiceConversionSynthesizingAudio
        case WAIFU_STAGE_LAYER_DECOMPOSITION_ENCODING_IMAGE: self = .layerDecompositionEncodingImage
        case WAIFU_STAGE_LAYER_DECOMPOSITION_DENOISING_LAYERS: self = .layerDecompositionDenoisingLayers
        case WAIFU_STAGE_LAYER_DECOMPOSITION_DECODING_LAYERS: self = .layerDecompositionDecodingLayers
        case WAIFU_STAGE_LAYER_DECOMPOSITION_ENCODING_HEAD: self = .layerDecompositionEncodingHead
        case WAIFU_STAGE_LAYER_DECOMPOSITION_DENOISING_HEAD_LAYERS: self = .layerDecompositionDenoisingHeadLayers
        case WAIFU_STAGE_LAYER_DECOMPOSITION_DECODING_HEAD_LAYERS: self = .layerDecompositionDecodingHeadLayers
        case WAIFU_STAGE_LAYER_DECOMPOSITION_ENCODING_FOR_DEPTH: self = .layerDecompositionEncodingForDepth
        case WAIFU_STAGE_LAYER_DECOMPOSITION_DENOISING_DEPTH: self = .layerDecompositionDenoisingDepth
        case WAIFU_STAGE_LAYER_DECOMPOSITION_DECODING_DEPTH: self = .layerDecompositionDecodingDepth
        case WAIFU_STAGE_LAYER_DECOMPOSITION_CUTTING_INTO_PARTS: self = .layerDecompositionCuttingIntoParts
        case WAIFU_STAGE_SPEECH_TO_TEXT_READING_RECORDING: self = .speechToTextReadingRecording
        case WAIFU_STAGE_SPEECH_TO_TEXT_GENERATING_TOKENS: self = .speechToTextGeneratingTokens
        case WAIFU_STAGE_STREAMING_TEXT_TO_SPEECH_READING_TEXT: self = .streamingTextToSpeechReadingText
        case WAIFU_STAGE_STREAMING_TEXT_TO_SPEECH_GENERATING_TOKENS: self = .streamingTextToSpeechGeneratingTokens
        case WAIFU_STAGE_STREAMING_TEXT_TO_SPEECH_SYNTHESIZING_AUDIO: self = .streamingTextToSpeechSynthesizingAudio
        default: self = .other(stage)
        }
    }
}

/// How far along a job is. libwaifu says what it is doing as a stage, for the app to put in words;
/// a download comes with libwaifu_modelmanager's own.
public struct JobProgress: Sendable {
    public var stage: Stage
    /// Of the whole job, 0 to 1, or nil where nothing can say -- reading weights.
    public var fraction: Double?
    /// Of the stage: what the stage counts, done already, of how many; 0 and 0 for one that
    /// counts nothing.
    public var done: UInt64
    public var total: UInt64
    public var seconds: Double
    /// A download's: which file of how many, and the file. 0, 0 and nil for libwaifu's stages.
    public var part: Int = 0
    public var parts: Int = 0
    public var file: String?
    /// A download's status line, as libwaifu_modelmanager words it: "fetching sdxl:base from
    /// Hugging Face -- ... (1 of 3)". Nil for libwaifu's stages, which have no words of their own.
    public var words: String?

    init(_ progress: WaifuProgress) {
        stage = Stage(progress.stage)
        fraction = progress.fraction < 0 ? nil : progress.fraction
        done = progress.done
        total = progress.total
        seconds = progress.seconds
    }

    /// A download's progress, as libwaifu_modelmanager says it.
    init(fetching event: WaifuModelManagerProgressEvent) {
        stage = .fetching
        fraction = event.fraction < 0 ? nil : event.fraction
        done = event.done
        total = event.total
        seconds = event.seconds
        part = Int(event.part)
        parts = Int(event.parts)
        file = event.file.map { String(cString: $0) }
        words = event.words.map { String(cString: $0) }
    }
}

public typealias ProgressHandler = @Sendable (JobProgress) -> Void

// MARK: - the plumbing every async call shares

/// How a job ended: with what it made, cancelled, or failed.
enum Outcome {
    case done(Any?)
    case cancelled
    case failed(WaifuError)

    init(_ status: WaifuStatus, _ value: Any?) {
        switch status.code {
        case WAIFU_OK: self = .done(value)
        case WAIFU_CANCELLED: self = .cancelled
        default:
            self = .failed(WaifuError(code: status.code, message: status.message.map { String(cString: $0) } ?? ""))
        }
    }
}

/// What a callback needs when it arrives: the progress closure, and how to end the call.
final class Pending: @unchecked Sendable {
    let progress: ProgressHandler?
    let finish: (Outcome) -> Void

    init(progress: ProgressHandler?, finish: @escaping (Outcome) -> Void) {
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

private let onProgress: WaifuProgressCallback = { userData, progress in
    guard let userData, let progress else { return }
    let pending = Unmanaged<Pending>.fromOpaque(userData).takeUnretainedValue()
    pending.progress?(JobProgress(progress.pointee))
}

private let onComplete: WaifuCallback = { userData, result in
    guard let userData, let result else { return }
    let pending = Unmanaged<Pending>.fromOpaque(userData).takeRetainedValue()
    pending.finish(Outcome(result.pointee.status, ()))
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
    let said = status.message.map { String(cString: $0) } ?? ""
    switch status.code {
    case WAIFU_MODELMANAGER_OK: pending.finish(.done(()))
    case WAIFU_MODELMANAGER_CANCELLED: pending.finish(.cancelled)
    default: pending.finish(.failed(WaifuError(code: status.code, message: said)))
    }
}

private let onImage: WaifuImageGenerationCallback = { userData, result in
    guard let userData, let result else { return }
    let pending = Unmanaged<Pending>.fromOpaque(userData).takeRetainedValue()
    // Copied now: the pixels are the library's only until this returns.
    let image = result.pointee.image.map { image -> RGBImage in
        let pixels = image.pointee
        let rgb = pixels.rgb.map { Array(UnsafeBufferPointer(start: $0, count: pixels.len)) } ?? []
        return RGBImage(width: Int(pixels.width), height: Int(pixels.height), rgb: rgb)
    }
    pending.finish(Outcome(result.pointee.status, image))
}

private let onAudio: WaifuAudioCallback = { userData, result in
    guard let userData, let result else { return }
    let pending = Unmanaged<Pending>.fromOpaque(userData).takeRetainedValue()
    let clip = result.pointee.audio.map { audio -> AudioClip in
        let sound = audio.pointee
        let samples = sound.samples.map { Array(UnsafeBufferPointer(start: $0, count: sound.count)) } ?? []
        return AudioClip(samples: samples, rate: Int(sound.sample_rate))
    }
    pending.finish(Outcome(result.pointee.status, clip))
}

private func rgba(_ image: WaifuRgbaImage) -> RGBAImage {
    let pixels = image.rgba.map { Array(UnsafeBufferPointer(start: $0, count: image.len)) } ?? []
    return RGBAImage(width: Int(image.width), height: Int(image.height), rgba: pixels)
}

private let onLayers: WaifuLayerDecompositionCallback = { userData, result in
    guard let userData, let result else { return }
    let pending = Unmanaged<Pending>.fromOpaque(userData).takeRetainedValue()
    let said = result.pointee
    // Copied now, every layer: they are the library's only until this returns.
    var layers: Layers?
    if said.status.code == WAIFU_OK {
        let all = said.layers.map { Array(UnsafeBufferPointer(start: $0, count: said.layer_count)) } ?? []
        layers = Layers(
            canvas: Int(said.canvas),
            layers: all.map { layer in
                Layer(
                    type: LayerType(layer.type), id: Int(layer.id),
                    name: layer.name.map { String(cString: $0) } ?? "", left: Int(layer.left),
                    top: Int(layer.top), image: rgba(layer.image), depth: layer.depth,
                    derivedFromID: layer.derived_from_id == 0 ? nil : Int(layer.derived_from_id))
            })
    }
    pending.finish(Outcome(said.status, layers))
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

    func image(_ image: RGBAImage) -> UnsafePointer<WaifuRgbaImage> {
        one(WaifuRgbaImage(width: UInt32(image.width), height: UInt32(image.height), rgba: array(image.rgba), len: image.rgba.count))
    }

    func audio(_ clip: AudioClip) -> UnsafePointer<WaifuAudio> {
        one(WaifuAudio(samples: array(clip.samples), count: clip.samples.count, sample_rate: UInt32(clip.rate)))
    }
}

/// Runs one `_async` call as an async function: its result, nil where it was cancelled, or what
/// it failed of. `submit` makes the C call with the `user_data` it is handed and returns its job;
/// `refused` says why a job of 0 was.
func call<T>(
    progress: ProgressHandler?,
    cancel: @escaping (WaifuJob) -> Void,
    refused: () -> WaifuError = libraryRefusal,
    submit: (UnsafeMutableRawPointer) -> WaifuJob
) async throws -> T? {
    let ticket = Ticket()
    return try await withTaskCancellationHandler {
        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<T?, Error>) in
            let pending = Pending(progress: progress) { outcome in
                switch outcome {
                case .done(let value): continuation.resume(returning: value as? T)
                case .cancelled: continuation.resume(returning: nil)
                case .failed(let error): continuation.resume(throwing: error)
                }
            }
            let userData = Unmanaged.passRetained(pending).toOpaque()
            let job = submit(userData)
            if job == 0 {
                // Refused at the door: nothing will ever be called, so the retain is this side's
                // to take back.
                Unmanaged<Pending>.fromOpaque(userData).release()
                continuation.resume(throwing: refused())
                return
            }
            ticket.given(job, cancel: cancel)
        }
    } onCancel: {
        ticket.cancelNow()
    }
}

// MARK: - models

/// What a model may be asked whether it does, as WAIFU_MODEL_SUPPORTS_ numbers it: a task, or
/// something within one.
public enum Capability: Sendable {
    case imageGeneration, textToSpeech, voiceConversion, layerDecomposition, speechToText,
        streamingTextToSpeech
    /// img2img.
    case imageGenerationFromStartImage
    /// A ControlNet.
    case imageGenerationWithControlImage
    /// Images the prompt names as <|key|>.
    case imageGenerationWithPromptImages
    /// In the voice of a recording.
    case textToSpeechLikeReference
    /// With no recording at all: CosyVoice3 does not.
    case textToSpeechWithoutReference
    /// Following an instruction such as "speak happily".
    case textToSpeechWithStyle
    /// The manner as well as the voice.
    case voiceConversionOfStyle

    var c: WaifuModelProperty {
        switch self {
        case .imageGeneration: WAIFU_MODEL_SUPPORTS_IMAGE_GENERATION
        case .textToSpeech: WAIFU_MODEL_SUPPORTS_TEXT_TO_SPEECH
        case .voiceConversion: WAIFU_MODEL_SUPPORTS_VOICE_CONVERSION
        case .layerDecomposition: WAIFU_MODEL_SUPPORTS_LAYER_DECOMPOSITION
        case .speechToText: WAIFU_MODEL_SUPPORTS_SPEECH_TO_TEXT
        case .streamingTextToSpeech: WAIFU_MODEL_SUPPORTS_STREAMING_TEXT_TO_SPEECH
        case .imageGenerationFromStartImage: WAIFU_MODEL_SUPPORTS_IMAGE_GENERATION_FROM_START_IMAGE
        case .imageGenerationWithControlImage: WAIFU_MODEL_SUPPORTS_IMAGE_GENERATION_WITH_CONTROL_IMAGE
        case .imageGenerationWithPromptImages: WAIFU_MODEL_SUPPORTS_IMAGE_GENERATION_WITH_PROMPT_IMAGES
        case .textToSpeechLikeReference: WAIFU_MODEL_SUPPORTS_TEXT_TO_SPEECH_LIKE_REFERENCE
        case .textToSpeechWithoutReference: WAIFU_MODEL_SUPPORTS_TEXT_TO_SPEECH_WITHOUT_REFERENCE
        case .textToSpeechWithStyle: WAIFU_MODEL_SUPPORTS_TEXT_TO_SPEECH_WITH_STYLE
        case .voiceConversionOfStyle: WAIFU_MODEL_SUPPORTS_VOICE_CONVERSION_OF_STYLE
        }
    }
}

/// Whether a model does something, and where it does not, why, in words to show.
public enum Support: Sendable, Equatable {
    case yes
    case no(because: String)
}

/// A model's manifest, read -- a few kilobytes, none of the weights -- and what can be asked of
/// it: what a pipeline is loaded with.
public final class Model: @unchecked Sendable {
    fileprivate let handle: OpaquePointer
    /// What it was opened from: the path of a manifest, or "tones".
    public let manifest: String

    /// Throws `unknownModel` where there is no manifest at the path, `model` where there is one
    /// that cannot be read.
    public init(manifest: String) throws {
        guard let handle = manifest.withCString({ waifu_model_open($0) }) else { throw libraryRefusal() }
        self.handle = handle
        self.manifest = manifest
    }

    deinit {
        waifu_model_free(handle)
    }

    private func int(_ property: WaifuModelProperty) -> Int64? {
        var value: Int64 = 0
        return waifu_model_get_int_property(handle, property, &value) ? value : nil
    }

    private func string(_ property: WaifuModelProperty) -> String? {
        let said: UnsafePointer<CChar>? = waifu_model_get_string_property(handle, property)
        return said.map { String(cString: $0) }
    }

    /// Whether it does `capability`, and why not where it does not; nil where the library does
    /// not say, as one that does not know the question does not.
    public func supports(_ capability: Capability) -> Support? {
        guard let does = int(capability.c) else { return nil }
        return does != 0 ? .yes : .no(because: string(capability.c) ?? "")
    }

    public func does(_ capability: Capability) -> Bool { supports(capability) == .yes }

    /// Why it does not do `capability`, where it says it does not.
    public func whyNot(_ capability: Capability) -> String? {
        if case .no(let why)? = supports(capability) { why } else { nil }
    }

    /// Bytes of its files; nil until every one of them is here.
    public var sizeOnDisk: UInt64? { int(WAIFU_MODEL_SIZE_ON_DISK).map { UInt64(max(0, $0)) } }
    /// The rate of what it says or converts.
    public var sampleRate: Int? { int(WAIFU_MODEL_SAMPLE_RATE).map { Int($0) } }
    /// What both sides of a picture it draws are a multiple of: 16, or 32 for SDXL and Qwen-Image.
    public var widthHeightMultipleOf: Int? {
        int(WAIFU_MODEL_IMAGE_GENERATION_WIDTH_HEIGHT_MULTIPLE_OF).map { Int($0) }
    }
    /// How it samples: "Euler".
    public var sampler: String? { string(WAIFU_MODEL_IMAGE_GENERATION_SAMPLER) }
    /// How many images a prompt can name.
    public var maxPromptImages: Int? { int(WAIFU_MODEL_IMAGE_GENERATION_MAX_PROMPT_IMAGES).map { Int($0) } }
    /// What the side a picture is taken apart at is a multiple of: 64.
    public var resolutionMultipleOf: Int? {
        int(WAIFU_MODEL_LAYER_DECOMPOSITION_RESOLUTION_MULTIPLE_OF).map { Int($0) }
    }
    /// The languages a recording to write down may be said to be in.
    public var speechToTextLanguages: [String]? {
        string(WAIFU_MODEL_SPEECH_TO_TEXT_LANGUAGES).map { $0.split(separator: ",").map(String.init) }
    }
}

// MARK: - the pipeline

/// One thread in the library, made for one task, that holds at most one model and runs jobs on
/// it, one at a time, in the order asked. Freeing it stops what runs, cancels what waits, and
/// drops the model.
public final class Pipeline: @unchecked Sendable {
    private let handle: OpaquePointer
    public let type: PipelineType

    /// A pipeline for `type` on `device`. On `.metal` the model's smallest weights fill what the
    /// memory limit leaves -- see Device.setMemoryLimit -- and the rest are read off the disk as
    /// they are used.
    public init(_ type: PipelineType, device: Device) throws {
        guard let handle = waifu_pipeline_new(type.c, device.c) else { throw libraryRefusal() }
        self.handle = handle
        self.type = type
    }

    deinit {
        waifu_pipeline_free(handle)
    }

    /// What it runs on: "metal", "cpu", "cuda". Never "auto", which the library settles when the
    /// pipeline starts.
    public var device: String { Device.name(waifu_pipeline_get_device(handle)) }

    private var cancel: (WaifuJob) -> Void {
        { [handle] job in waifu_pipeline_cancel(handle, job) }
    }

    /// Fetches `model` if it has to -- a published name, through libwaifu_modelmanager -- then
    /// reads it onto the device, dropping the one before. `model` is a published name, the path
    /// of a manifest, or "tones". Throws `refused`, with the model's own reason, for one that does
    /// not do this pipeline's task, and CancellationError where it was cancelled.
    public func load(_ model: String, progress: ProgressHandler? = nil) async throws {
        let manifest = try await ModelManager.fetchedManifest(model, progress: progress)
        let opened = try Model(manifest: manifest)
        if case .no(let why)? = opened.supports(type.capability) { throw WaifuError.refused(why) }
        let loaded: Void? = try await call(progress: progress, cancel: cancel) { userData in
            // The call copies the manifest: the model may go as soon as it has returned.
            withExtendedLifetime(opened) {
                waifu_pipeline_load_model_async(handle, opened.handle, userData, onProgress, onComplete)
            }
        }
        if loaded == nil { throw CancellationError() }
    }

    public func unload() async throws {
        let _: Void? = try await call(progress: nil, cancel: cancel) { userData in
            waifu_pipeline_unload_model_async(handle, userData, onComplete)
        }
    }

    /// The image, or nil where it was cancelled.
    public func draw(_ request: DrawRequest, progress: ProgressHandler? = nil) async throws -> RGBImage? {
        try await call(progress: progress, cancel: cancel) { userData in
            let scratch = Scratch()
            let named = request.promptImages.map { input in
                WaifuPromptImage(key: scratch.string(input.key), image: scratch.image(input.image))
            }
            let start: UnsafePointer<WaifuImage>? = request.startImage.map { scratch.image($0) }
            let control: UnsafePointer<WaifuImage>? = request.controlImage.map { scratch.image($0) }
            var c = WaifuImageGenerationRequest(
                struct_size: UInt32(MemoryLayout<WaifuImageGenerationRequest>.size),
                prompt: scratch.string(request.prompt),
                negative: scratch.string(request.negative),
                width: UInt32(clamping: request.width),
                height: UInt32(clamping: request.height),
                steps: UInt32(clamping: request.steps),
                guidance: Float(request.guidance),
                seed: request.seed,
                start_image: start,
                start_image_strength: Float(request.strength),
                control_image: control,
                control_image_scale: Float(request.controlScale),
                prompt_images: scratch.array(named),
                prompt_image_count: named.count
            )
            return withExtendedLifetime(scratch) {
                waifu_pipeline_run_image_generation_async(handle, &c, userData, onProgress, onImage)
            }
        }
    }

    /// The reading, or nil where it was cancelled.
    public func speak(_ request: SpeakRequest, progress: ProgressHandler? = nil) async throws -> AudioClip? {
        try await call(progress: progress, cancel: cancel) { userData in
            let scratch = Scratch()
            let reference: UnsafePointer<WaifuAudio>? = request.reference.map { scratch.audio($0) }
            var c = WaifuTextToSpeechRequest(
                struct_size: UInt32(MemoryLayout<WaifuTextToSpeechRequest>.size),
                text: scratch.string(request.text),
                reference: reference,
                speed: Float(request.speed),
                temperature: Float(request.temperature),
                style: scratch.optionalString(request.style),
                seed: request.seed,
                transcript: nil
            )
            return withExtendedLifetime(scratch) {
                waifu_pipeline_run_text_to_speech_async(handle, &c, userData, onProgress, onAudio)
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
                waifu_pipeline_run_voice_conversion_async(handle, &c, userData, onProgress, onAudio)
            }
        }
    }

    /// The illustration taken apart into its layers, with See-through loaded, or nil where it was
    /// cancelled.
    public func layers(_ request: LayersRequest, progress: ProgressHandler? = nil) async throws -> Layers? {
        try await call(progress: progress, cancel: cancel) { userData in
            let scratch = Scratch()
            var c = WaifuLayerDecompositionRequest(
                struct_size: UInt32(MemoryLayout<WaifuLayerDecompositionRequest>.size),
                image: scratch.image(request.image),
                steps: UInt32(clamping: request.steps),
                depth_steps: UInt32(clamping: request.depthSteps),
                resolution: UInt32(clamping: request.resolution),
                split_pairs: request.splitPairs,
                stratify_layers: request.stratifyLayers,
                seed: request.seed
            )
            return withExtendedLifetime(scratch) {
                waifu_pipeline_run_layer_decomposition_async(handle, &c, userData, onProgress, onLayers)
            }
        }
    }
}

// MARK: - the model manager

/// What is published, what is here, and fetching it: libwaifu_modelmanager's. None of it needs a
/// pipeline.
public enum ModelManager {
    /// The published models, as the library's JSON: `{"models": [...]}`.
    public static func catalogJSON() throws -> Data {
        guard let said = waifu_modelmanager_catalog_json() else {
            throw WaifuError.refused(modelManagerLastError())
        }
        defer { waifu_modelmanager_string_free(said) }
        return Data(String(cString: said).utf8)
    }

    /// `model` opened to be asked what it is, without reading it: the manifest libwaifu_modelmanager
    /// finds for it, whose weights need not be here. Throws for a name with no manifest here --
    /// fetch it, or every manifest, first -- and for one that is neither published nor a manifest
    /// on the disk.
    public static func open(_ model: String) throws -> Model {
        try Model(manifest: model == tones ? model : manifestPath(model))
    }

    /// What libwaifu_modelmanager says a model's boxes start at, as its JSON: a picture model's
    /// `width`, `height`, `steps`, `guidance`, `takes_guidance`, `sizes`, `prompt` and `avoid`; a
    /// voice's `speed`, `temperature` and `conversion.steps`; a layer model's `steps`,
    /// `depth_steps` and `resolution`.
    public static func defaultsJSON(of model: Model) throws -> Data {
        guard let said = model.manifest.withCString({ waifu_modelmanager_defaults_json($0) }) else {
            throw WaifuError.refused(modelManagerLastError())
        }
        defer { waifu_modelmanager_string_free(said) }
        return Data(String(cString: said).utf8)
    }

    /// What the catalogue calls a published model: "Stable Diffusion XL". Nil for what it does
    /// not list, a manifest's path included.
    public static func fullName(_ model: String) -> String? {
        try? listed(model).full_name
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
            refused: { .refused(modelManagerLastError()) },
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
            refused: { .refused(modelManagerLastError()) },
            submit: { userData in
                handle.fetch = waifu_modelmanager_fetch_manifests_async(userData, onFetchProgress, onFetchComplete)
                return handle.fetch == nil ? 0 : 1
            })
        waifu_modelmanager_fetch_free(handle.fetch)
        if fetched == nil { throw CancellationError() }
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

    private static let callback: WaifuLogCallback = { userData, log in
        guard let userData, let log else { return }
        let box = Unmanaged<Box>.fromOpaque(userData).takeUnretainedValue()
        // A level a newer library writes that this does not know is said as information.
        let level = LogLevel(rawValue: Int(log.pointee.level)) ?? .info
        box.handler(
            level,
            log.pointee.source.map { String(cString: $0) } ?? "",
            log.pointee.message.map { String(cString: $0) } ?? "")
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
        waifu_set_log_level(WaifuLogLevel(level.rawValue))
        waifu_modelmanager_set_log_level(WaifuModelManagerLogLevel(rawValue: UInt32(level.rawValue)))
    }
}
