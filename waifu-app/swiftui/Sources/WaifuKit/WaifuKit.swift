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

// libwaifu's C API, as Swift: async functions for what it runs, errors that throw, and Swift's own
// types for what goes in and comes out. See docs/ffi-abi6.md for what libwaifu promises, and
// docs/ffi.md for libwaifu_modelmanager.
//
// libwaifu calls nothing of ours: a load, an unload or a run hands back a WaifuTask, which is read.
// Each is a continuation, and a `Thread` of its own that blocks in waifu_task_next_event, hands
// each event -- progress, a streaming reading's parts -- to its closure in order, copies the
// result out at the end, frees the task, and resumes the continuation. Cancelling the Swift task
// cancels the library's, under the lock the reader frees it under (`TaskHandle`).
//
// libwaifu_modelmanager still calls back: each of its `_async` calls is a continuation too, and
// what it needs when a callback arrives -- the continuation, and the progress closure -- is a
// `Pending`, passed to C as `user_data` with one retain that `on_complete` takes back. A call
// refused at the door calls nothing, so it takes it back itself.
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
        // A request attribute the library does not know, or another task's: refused at the call.
        case WAIFU_ERR_UNKNOWN_ATTRIBUTE: self = .refused(message)
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
    /// libwaifu's, as (major << 16) | minor.
    public static var library: UInt32 { waifu_abi_version() }
    /// The same, from the header's two parts: WAIFU_ABI_VERSION is an expression of them, which
    /// Swift may not import as a constant.
    public static var header: UInt32 {
        (UInt32(WAIFU_ABI_VERSION_MAJOR) << 16) | UInt32(WAIFU_ABI_VERSION_MINOR)
    }
    /// Whether the loaded libwaifu runs what this was built against: the same major, and a minor
    /// no older.
    public static var libraryIsCompatible: Bool {
        library >> 16 == header >> 16 && library & 0xffff >= header & 0xffff
    }
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
    /// What `reference` says, word for word; nil or "" where it is not known. CosyVoice3 reads on
    /// from it, as upstream's zero-shot does, and takes it or a style, not both; a voice that
    /// reads none ignores it.
    public var referenceTranscript: String?
    public var speed: Double
    public var temperature: Double
    /// Any instruction; nil or "" for the voice's own way.
    public var style: String?
    public var seed: UInt64

    public init(
        text: String, reference: AudioClip? = nil, referenceTranscript: String? = nil, speed: Double,
        temperature: Double, style: String? = nil, seed: UInt64
    ) {
        self.text = text
        self.reference = reference
        self.referenceTranscript = referenceTranscript
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

/// A recording to write down: speech2text.
public struct TranscribeRequest: Sendable {
    /// Any rate, any length: the library resamples it and cuts it into pieces.
    public var recording: AudioClip
    /// One of the model's `speechToTextLanguages` -- "English", "Chinese" -- or nil for the model
    /// to tell.
    public var language: String?
    /// Words that help it hear: names, terms, the subject. Nil for none.
    public var context: String?

    public init(recording: AudioClip, language: String? = nil, context: String? = nil) {
        self.recording = recording
        self.language = language
        self.context = context
    }
}

/// What was said in a recording, and in which language.
public struct Transcript: Sendable, Equatable {
    /// "" for nothing said.
    public var text: String
    /// As the model names it -- "Chinese" -- or several, comma separated, for a recording that
    /// changes; "" where none was heard.
    public var language: String

    public init(text: String, language: String) {
        self.text = text
        self.language = language
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
    /// at one depth, a far half under a near one; then by place in `layers` (docs/ffi-abi5.md,
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

    /// The progress the task's last PROGRESS event said; nil where it has none.
    init?(task: OpaquePointer) {
        guard let stage = TaskRead.int(task, WAIFU_TASK_PROGRESS_STAGE),
            let done = TaskRead.int(task, WAIFU_TASK_PROGRESS_DONE),
            let total = TaskRead.int(task, WAIFU_TASK_PROGRESS_TOTAL)
        else { return nil }
        self.stage = Stage(WaifuStage(truncatingIfNeeded: stage))
        // Absent where nothing can say.
        let fraction = TaskRead.float(task, WAIFU_TASK_PROGRESS_FRACTION)
        self.fraction = fraction.flatMap { $0 < 0 ? nil : $0 }
        self.done = UInt64(clamping: done)
        self.total = UInt64(clamping: total)
        seconds = TaskRead.float(task, WAIFU_TASK_PROGRESS_SECONDS) ?? 0
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

/// A part of a streaming reading, in order, on the task's reader thread as soon as it is made.
public typealias AudioPartHandler = @Sendable (AudioClip) -> Void

/// A pointer the library handed over, moved to the thread that reads it: a task may be read,
/// waited on and cancelled from any thread.
struct TaskPointer: @unchecked Sendable {
    let raw: OpaquePointer
}

/// The library's task, until its reader has read it to its end and freed it. Cancelling and
/// freeing are both done holding the lock: a task must not be freed while another call is on it.
/// A cancel that comes before the task does is kept, and done when it comes.
final class TaskHandle: @unchecked Sendable {
    private let lock = NSLock()
    private var task: OpaquePointer?
    private var cancelled = false

    func hold(_ task: OpaquePointer) {
        lock.lock()
        defer { lock.unlock() }
        self.task = task
        if cancelled { waifu_task_cancel(task) }
    }

    func cancel() {
        lock.lock()
        defer { lock.unlock() }
        cancelled = true
        if let task { waifu_task_cancel(task) }
    }

    /// Once, after its end has been read: every pointer it handed out is then gone.
    func free() {
        lock.lock()
        defer { lock.unlock() }
        if let task { waifu_task_free(task) }
        task = nil
    }
}

/// What a task's attributes say, copied out: everything a task hands out is the library's only
/// until it is freed.
enum TaskRead {
    static func int(_ task: OpaquePointer, _ attribute: WaifuTaskAttribute) -> Int64? {
        var value: Int64 = 0
        return waifu_task_get_int_attribute(task, attribute, &value) ? value : nil
    }

    static func float(_ task: OpaquePointer, _ attribute: WaifuTaskAttribute) -> Double? {
        var value: Double = 0
        return waifu_task_get_float_attribute(task, attribute, &value) ? value : nil
    }

    static func string(_ task: OpaquePointer, _ attribute: WaifuTaskAttribute) -> String? {
        let said: UnsafePointer<CChar>? = waifu_task_get_string_attribute(task, attribute)
        return said.map { String(cString: $0) }
    }

    static func image(_ task: OpaquePointer, _ attribute: WaifuTaskAttribute) -> RGBImage? {
        var image = WaifuImage()
        guard waifu_task_get_image_attribute(task, attribute, &image) else { return nil }
        let rgb = image.rgb.map { Array(UnsafeBufferPointer(start: $0, count: image.len)) } ?? []
        return RGBImage(width: Int(image.width), height: Int(image.height), rgb: rgb)
    }

    static func audio(_ task: OpaquePointer, _ attribute: WaifuTaskAttribute) -> AudioClip? {
        var sound = WaifuAudio()
        guard waifu_task_get_audio_attribute(task, attribute, &sound) else { return nil }
        return clip(sound)
    }

    static func clip(_ sound: WaifuAudio) -> AudioClip {
        let samples = sound.samples.map { Array(UnsafeBufferPointer(start: $0, count: sound.count)) } ?? []
        return AudioClip(samples: samples, rate: Int(sound.sample_rate))
    }

    static func rgba(_ image: WaifuRgbaImage) -> RGBAImage {
        let pixels = image.rgba.map { Array(UnsafeBufferPointer(start: $0, count: image.len)) } ?? []
        return RGBAImage(width: Int(image.width), height: Int(image.height), rgba: pixels)
    }

    /// Layer `index` of a layer decomposition; nil past the count, or where an attribute is not there.
    static func layer(_ task: OpaquePointer, at index: Int) -> Layer? {
        func indexed(_ attribute: WaifuTaskAttribute) -> Int64? {
            var value: Int64 = 0
            return waifu_task_get_int_attribute_at(task, attribute, index, &value) ? value : nil
        }
        var depth: Double = 0
        var image = WaifuRgbaImage()
        guard waifu_task_get_float_attribute_at(task, WAIFU_TASK_LAYER_DECOMPOSITION_LAYER_DEPTH, index, &depth),
            waifu_task_get_rgba_image_attribute_at(task, WAIFU_TASK_LAYER_DECOMPOSITION_LAYER_IMAGE, index, &image),
            let type = indexed(WAIFU_TASK_LAYER_DECOMPOSITION_LAYER_TYPE),
            let id = indexed(WAIFU_TASK_LAYER_DECOMPOSITION_LAYER_ID),
            let left = indexed(WAIFU_TASK_LAYER_DECOMPOSITION_LAYER_LEFT),
            let top = indexed(WAIFU_TASK_LAYER_DECOMPOSITION_LAYER_TOP),
            let derived = indexed(WAIFU_TASK_LAYER_DECOMPOSITION_LAYER_DERIVED_FROM_ID)
        else { return nil }
        let name: UnsafePointer<CChar>? =
            waifu_task_get_string_attribute_at(task, WAIFU_TASK_LAYER_DECOMPOSITION_LAYER_NAME, index)
        return Layer(
            type: LayerType(WaifuLayerType(truncatingIfNeeded: type)), id: Int(id),
            name: name.map { String(cString: $0) } ?? "", left: Int(left), top: Int(top),
            image: rgba(image), depth: Float(depth), derivedFromID: derived == 0 ? nil : Int(derived))
    }

    static func layers(_ task: OpaquePointer) throws -> Layers {
        guard let canvas = int(task, WAIFU_TASK_LAYER_DECOMPOSITION_CANVAS),
            let count = int(task, WAIFU_TASK_LAYER_DECOMPOSITION_LAYER_COUNT)
        else { throw missing("layers") }
        var layers: [Layer] = []
        layers.reserveCapacity(Int(max(0, count)))
        for index in 0..<Int(max(0, count)) {
            guard let found = layer(task, at: index) else { throw missing("layer") }
            layers.append(found)
        }
        return Layers(canvas: Int(canvas), layers: layers)
    }

    static func transcript(_ task: OpaquePointer) throws -> Transcript {
        guard let text = string(task, WAIFU_TASK_SPEECH_TO_TEXT_TEXT) else { throw missing("transcript") }
        return Transcript(text: text, language: string(task, WAIFU_TASK_SPEECH_TO_TEXT_LANGUAGE) ?? "")
    }

    static func missing(_ what: String) -> WaifuError {
        .internalError("the library said the job was done, and handed over no \(what)")
    }

    /// How an ended task ended: what `output` copies out of it for WAIFU_OK, nil where it was
    /// cancelled, or the error it failed of.
    static func ending<T>(_ task: OpaquePointer, _ output: (OpaquePointer) throws -> T) throws -> T? {
        guard let code = int(task, WAIFU_TASK_STATUS) else {
            throw WaifuError.internalError("the library said the job had ended, and had no status for it")
        }
        switch WaifuStatusCode(truncatingIfNeeded: code) {
        case WAIFU_OK: return try output(task)
        case WAIFU_CANCELLED: return nil
        case let failed: throw WaifuError(code: failed, message: string(task, WAIFU_TASK_STATUS_MESSAGE) ?? "")
        }
    }
}

/// Reads `task` to its end, on the thread that calls it: each progress and part handed over in
/// the order they happened, then the result copied out and the task freed.
private func readToTheEnd<T>(
    _ task: OpaquePointer,
    handle: TaskHandle,
    progress: ProgressHandler?,
    heard: AudioPartHandler?,
    output: (OpaquePointer) throws -> T
) -> Result<T?, Error> {
    while true {
        switch waifu_task_next_event(task, UInt32.max) {
        case WAIFU_EVENT_PROGRESS:
            if let progress, let said = JobProgress(task: task) { progress(said) }
        case WAIFU_EVENT_AUDIO_PART:
            if let heard, let part = TaskRead.audio(task, WAIFU_TASK_STREAMING_TEXT_TO_SPEECH_AUDIO_PART) {
                heard(part)
            }
        case WAIFU_EVENT_END:
            // Copied out before it is freed, which takes every pointer it handed out with it.
            let ended: Result<T?, Error>
            do {
                ended = .success(try TaskRead.ending(task, output))
            } catch {
                ended = .failure(error)
            }
            handle.free()
            return ended
        default:
            // NONE never comes without a timeout; anything else is a type this did not ask for.
            continue
        }
    }
}

/// Runs one load, unload or run as an async function: its result, nil where it was cancelled, or
/// what it failed of. `start` makes the C call -- on this thread, so that a refusal's last error is
/// this thread's -- and returns its task, or throws why there was none. `output` copies the result
/// out of a task that ended well, on the task's reader thread.
func runTask<T: Sendable>(
    progress: ProgressHandler?,
    heard: AudioPartHandler? = nil,
    output: @escaping @Sendable (OpaquePointer) throws -> T,
    start: () throws -> OpaquePointer
) async throws -> T? {
    let handle = TaskHandle()
    return try await withTaskCancellationHandler {
        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<T?, Error>) in
            let task: OpaquePointer
            do {
                task = try start()
            } catch {
                continuation.resume(throwing: error)
                return
            }
            handle.hold(task)
            let pointer = TaskPointer(raw: task)
            // A thread of its own, blocked in the library until the task ends: not one of the
            // cooperative pool's.
            let reader = Thread {
                continuation.resume(
                    with: readToTheEnd(
                        pointer.raw, handle: handle, progress: progress, heard: heard, output: output))
            }
            reader.name = "waifu-task"
            reader.start()
        }
    } onCancel: {
        handle.cancel()
    }
}

/// A request of libwaifu's, set an attribute at a time and freed when this goes. Each setter copies
/// what it is handed, and throws why where the library refuses it.
final class RawRequest {
    let handle: OpaquePointer

    init(_ type: PipelineType) throws {
        guard let handle = waifu_request_new(type.c) else { throw libraryRefusal() }
        self.handle = handle
    }

    deinit {
        waifu_request_free(handle)
    }

    private func checked(_ set: Bool) throws {
        if !set { throw libraryRefusal() }
    }

    func set(_ attribute: WaifuRequestAttribute, int value: Int64) throws {
        try checked(waifu_request_set_int_attribute(handle, attribute, value))
    }

    /// A count or a size, as the ABI 5 structs' uint32_t had it.
    func set(_ attribute: WaifuRequestAttribute, count value: Int) throws {
        try set(attribute, int: Int64(UInt32(clamping: value)))
    }

    /// Read as a uint64 by the library.
    func set(_ attribute: WaifuRequestAttribute, seed value: UInt64) throws {
        try set(attribute, int: Int64(bitPattern: value))
    }

    func set(_ attribute: WaifuRequestAttribute, flag value: Bool) throws {
        try set(attribute, int: value ? 1 : 0)
    }

    func set(_ attribute: WaifuRequestAttribute, float value: Double) throws {
        try checked(waifu_request_set_float_attribute(handle, attribute, value))
    }

    func set(_ attribute: WaifuRequestAttribute, string value: String) throws {
        try checked(value.withCString { waifu_request_set_string_attribute(handle, attribute, $0) })
    }

    func set(_ attribute: WaifuRequestAttribute, image: RGBImage) throws {
        try checked(
            image.rgb.withUnsafeBufferPointer { pixels in
                var c = WaifuImage(
                    width: UInt32(image.width), height: UInt32(image.height), rgb: pixels.baseAddress,
                    len: pixels.count)
                return waifu_request_set_image_attribute(handle, attribute, &c)
            })
    }

    func set(_ attribute: WaifuRequestAttribute, rgba image: RGBAImage) throws {
        try checked(
            image.rgba.withUnsafeBufferPointer { pixels in
                var c = WaifuRgbaImage(
                    width: UInt32(image.width), height: UInt32(image.height), rgba: pixels.baseAddress,
                    len: pixels.count)
                return waifu_request_set_rgba_image_attribute(handle, attribute, &c)
            })
    }

    func set(_ attribute: WaifuRequestAttribute, audio clip: AudioClip) throws {
        try checked(
            clip.samples.withUnsafeBufferPointer { samples in
                var c = WaifuAudio(samples: samples.baseAddress, count: samples.count, sample_rate: UInt32(clip.rate))
                return waifu_request_set_audio_attribute(handle, attribute, &c)
            })
    }

    /// IMAGE_GENERATION_PROMPT_IMAGES: one more image the prompt names as <|key|>.
    func addPromptImage(key: String, image: RGBImage) throws {
        try checked(
            key.withCString { key in
                image.rgb.withUnsafeBufferPointer { pixels in
                    var c = WaifuImage(
                        width: UInt32(image.width), height: UInt32(image.height), rgb: pixels.baseAddress,
                        len: pixels.count)
                    return waifu_request_add_prompt_image(handle, key, &c)
                }
            })
    }
}

// libwaifu_modelmanager's calls, which still call back.

/// How a fetch ended: done, cancelled, or failed.
enum Outcome {
    case done(Any?)
    case cancelled
    case failed(WaifuError)
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

/// How to cancel a call, for a cancel that may arrive before the call was made.
final class Ticket: @unchecked Sendable {
    private let lock = NSLock()
    private var cancelled = false
    private var cancel: (() -> Void)?

    func given(cancel: @escaping () -> Void) {
        lock.lock()
        self.cancel = cancel
        let already = cancelled
        lock.unlock()
        if already { cancel() }
    }

    func cancelNow() {
        lock.lock()
        cancelled = true
        let cancel = cancel
        lock.unlock()
        cancel?()
    }
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

/// Runs one of libwaifu_modelmanager's `_async` calls as an async function: its result, nil where
/// it was cancelled, or what it failed of. `submit` makes the C call with the `user_data` it is
/// handed and says whether it was made; `refused` says why it was not.
func call<T>(
    progress: ProgressHandler?,
    cancel: @escaping () -> Void,
    refused: () -> WaifuError,
    submit: (UnsafeMutableRawPointer) -> Bool
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
            if !submit(userData) {
                // Refused at the door: nothing will ever be called, so the retain is this side's
                // to take back.
                Unmanaged<Pending>.fromOpaque(userData).release()
                continuation.resume(throwing: refused())
                return
            }
            ticket.given(cancel: cancel)
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
    /// A recording without its transcript: CosyVoice3 does not, unless a style is given.
    case textToSpeechWithoutReferenceTranscript
    /// Following an instruction such as "speak happily".
    case textToSpeechWithStyle
    /// The manner as well as the voice.
    case voiceConversionOfStyle

    var c: WaifuModelAttribute {
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
        case .textToSpeechWithoutReferenceTranscript:
            WAIFU_MODEL_SUPPORTS_TEXT_TO_SPEECH_WITHOUT_REFERENCE_TRANSCRIPT
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

    private func int(_ attribute: WaifuModelAttribute) -> Int64? {
        var value: Int64 = 0
        return waifu_model_get_int_attribute(handle, attribute, &value) ? value : nil
    }

    private func string(_ attribute: WaifuModelAttribute) -> String? {
        let said: UnsafePointer<CChar>? = waifu_model_get_string_attribute(handle, attribute)
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

    /// The task a load, an unload or a run returned, or why there was none, as this thread's last
    /// error says it: read straight after the call, on its thread.
    private func started(_ task: OpaquePointer?) throws -> OpaquePointer {
        guard let task else { throw libraryRefusal() }
        return task
    }

    /// Runs `request` -- built here, and freed as soon as the run has copied it -- to its end.
    private func run<T: Sendable>(
        progress: ProgressHandler?,
        heard: AudioPartHandler? = nil,
        output: @escaping @Sendable (OpaquePointer) throws -> T,
        request: () throws -> RawRequest
    ) async throws -> T? {
        try await runTask(progress: progress, heard: heard, output: output) {
            let raw = try request()
            return try withExtendedLifetime(raw) { try started(waifu_pipeline_run(handle, raw.handle)) }
        }
    }

    /// Fetches `model` if it has to -- a published name, through libwaifu_modelmanager -- then
    /// reads it onto the device, dropping the one before. `model` is a published name, the path
    /// of a manifest, or "tones". Throws `refused`, with the model's own reason, for one that does
    /// not do this pipeline's task, and CancellationError where it was cancelled.
    public func load(_ model: String, progress: ProgressHandler? = nil) async throws {
        let manifest = try await ModelManager.fetchedManifest(model, progress: progress)
        let opened = try Model(manifest: manifest)
        if case .no(let why)? = opened.supports(type.capability) { throw WaifuError.refused(why) }
        let loaded: Void? = try await runTask(progress: progress, output: { _ in () }) {
            // The call copies the manifest: the model may go as soon as it has returned.
            try withExtendedLifetime(opened) { try started(waifu_pipeline_load_model(handle, opened.handle)) }
        }
        if loaded == nil { throw CancellationError() }
    }

    public func unload() async throws {
        let _: Void? = try await runTask(progress: nil, output: { _ in () }) {
            try started(waifu_pipeline_unload_model(handle))
        }
    }

    /// The image, or nil where it was cancelled.
    public func draw(_ request: DrawRequest, progress: ProgressHandler? = nil) async throws -> RGBImage? {
        try await run(
            progress: progress,
            output: { task in
                guard let image = TaskRead.image(task, WAIFU_TASK_IMAGE_GENERATION_IMAGE) else {
                    throw TaskRead.missing("image")
                }
                return image
            }
        ) {
            let c = try RawRequest(.imageGeneration)
            try c.set(WAIFU_REQUEST_IMAGE_GENERATION_PROMPT, string: request.prompt)
            try c.set(WAIFU_REQUEST_IMAGE_GENERATION_NEGATIVE, string: request.negative)
            try c.set(WAIFU_REQUEST_IMAGE_GENERATION_WIDTH, count: request.width)
            try c.set(WAIFU_REQUEST_IMAGE_GENERATION_HEIGHT, count: request.height)
            try c.set(WAIFU_REQUEST_IMAGE_GENERATION_STEPS, count: request.steps)
            try c.set(WAIFU_REQUEST_IMAGE_GENERATION_GUIDANCE, float: request.guidance)
            try c.set(WAIFU_REQUEST_IMAGE_GENERATION_SEED, seed: request.seed)
            // A strength or a scale means something only with its image.
            if let start = request.startImage {
                try c.set(WAIFU_REQUEST_IMAGE_GENERATION_START_IMAGE, image: start)
                try c.set(WAIFU_REQUEST_IMAGE_GENERATION_START_IMAGE_STRENGTH, float: request.strength)
            }
            if let control = request.controlImage {
                try c.set(WAIFU_REQUEST_IMAGE_GENERATION_CONTROL_IMAGE, image: control)
                try c.set(WAIFU_REQUEST_IMAGE_GENERATION_CONTROL_IMAGE_SCALE, float: request.controlScale)
            }
            for named in request.promptImages {
                try c.addPromptImage(key: named.key, image: named.image)
            }
            return c
        }
    }

    /// The reading, or nil where it was cancelled.
    public func speak(_ request: SpeakRequest, progress: ProgressHandler? = nil) async throws -> AudioClip? {
        try await run(
            progress: progress,
            output: { task in
                guard let audio = TaskRead.audio(task, WAIFU_TASK_TEXT_TO_SPEECH_AUDIO) else {
                    throw TaskRead.missing("audio")
                }
                return audio
            }
        ) {
            let c = try RawRequest(.textToSpeech)
            try c.set(WAIFU_REQUEST_TEXT_TO_SPEECH_TEXT, string: request.text)
            if let reference = request.reference {
                try c.set(WAIFU_REQUEST_TEXT_TO_SPEECH_REFERENCE, audio: reference)
            }
            if let transcript = request.referenceTranscript {
                try c.set(WAIFU_REQUEST_TEXT_TO_SPEECH_REFERENCE_TRANSCRIPT, string: transcript)
            }
            try c.set(WAIFU_REQUEST_TEXT_TO_SPEECH_SPEED, float: request.speed)
            try c.set(WAIFU_REQUEST_TEXT_TO_SPEECH_TEMPERATURE, float: request.temperature)
            if let style = request.style {
                try c.set(WAIFU_REQUEST_TEXT_TO_SPEECH_STYLE, string: style)
            }
            try c.set(WAIFU_REQUEST_TEXT_TO_SPEECH_SEED, seed: request.seed)
            return c
        }
    }

    /// The reading, with each part of it handed to `heard` as soon as it is made -- CosyVoice3's a
    /// hop of tokens at a time -- while the rest is still being said, and the whole of it, the parts
    /// end to end, at the end; nil where it was cancelled, after the part it was on. For a pipeline
    /// made for `.streamingTextToSpeech`. A streaming reading is at the speed it is said, so the
    /// request's speed is not read; it is a reading of its own, not the samples `speak` makes of the
    /// same request. The parts come in order, on one thread, and all of them before this returns.
    public func speakStreaming(
        _ request: SpeakRequest, progress: ProgressHandler? = nil, heard: @escaping AudioPartHandler
    ) async throws -> AudioClip? {
        try await run(
            progress: progress,
            heard: heard,
            output: { task in
                guard let audio = TaskRead.audio(task, WAIFU_TASK_STREAMING_TEXT_TO_SPEECH_AUDIO) else {
                    throw TaskRead.missing("audio")
                }
                return audio
            }
        ) {
            let c = try RawRequest(.streamingTextToSpeech)
            try c.set(WAIFU_REQUEST_STREAMING_TEXT_TO_SPEECH_TEXT, string: request.text)
            if let reference = request.reference {
                try c.set(WAIFU_REQUEST_STREAMING_TEXT_TO_SPEECH_REFERENCE, audio: reference)
            }
            if let transcript = request.referenceTranscript {
                try c.set(WAIFU_REQUEST_STREAMING_TEXT_TO_SPEECH_REFERENCE_TRANSCRIPT, string: transcript)
            }
            try c.set(WAIFU_REQUEST_STREAMING_TEXT_TO_SPEECH_TEMPERATURE, float: request.temperature)
            if let style = request.style {
                try c.set(WAIFU_REQUEST_STREAMING_TEXT_TO_SPEECH_STYLE, string: style)
            }
            try c.set(WAIFU_REQUEST_STREAMING_TEXT_TO_SPEECH_SEED, seed: request.seed)
            return c
        }
    }

    /// The source said again in the reference's voice, or nil where it was cancelled.
    public func voiceConversion(_ request: VoiceConversionRequest, progress: ProgressHandler? = nil) async throws -> AudioClip? {
        try await run(
            progress: progress,
            output: { task in
                guard let audio = TaskRead.audio(task, WAIFU_TASK_VOICE_CONVERSION_AUDIO) else {
                    throw TaskRead.missing("audio")
                }
                return audio
            }
        ) {
            let c = try RawRequest(.voiceConversion)
            try c.set(WAIFU_REQUEST_VOICE_CONVERSION_SOURCE, audio: request.source)
            try c.set(WAIFU_REQUEST_VOICE_CONVERSION_REFERENCE, audio: request.reference)
            try c.set(WAIFU_REQUEST_VOICE_CONVERSION_STEPS, count: request.steps)
            try c.set(WAIFU_REQUEST_VOICE_CONVERSION_CONVERT_STYLE, flag: request.convertStyle)
            try c.set(WAIFU_REQUEST_VOICE_CONVERSION_SEED, seed: request.seed)
            return c
        }
    }

    /// What was said in the recording, with a transcriber loaded -- Qwen3-ASR -- or nil where it
    /// was cancelled.
    public func transcribe(_ request: TranscribeRequest, progress: ProgressHandler? = nil) async throws -> Transcript? {
        try await run(progress: progress, output: { task in try TaskRead.transcript(task) }) {
            let c = try RawRequest(.speechToText)
            try c.set(WAIFU_REQUEST_SPEECH_TO_TEXT_AUDIO, audio: request.recording)
            // Blank is none: the model tells the language, and hears with no help.
            let given = { (text: String?) -> String? in
                guard let text = text?.trimmingCharacters(in: .whitespacesAndNewlines), !text.isEmpty else { return nil }
                return text
            }
            if let language = given(request.language) {
                try c.set(WAIFU_REQUEST_SPEECH_TO_TEXT_LANGUAGE, string: language)
            }
            if let context = given(request.context) {
                try c.set(WAIFU_REQUEST_SPEECH_TO_TEXT_CONTEXT, string: context)
            }
            return c
        }
    }

    /// The illustration taken apart into its layers, with See-through loaded, or nil where it was
    /// cancelled.
    public func layers(_ request: LayersRequest, progress: ProgressHandler? = nil) async throws -> Layers? {
        try await run(progress: progress, output: { task in try TaskRead.layers(task) }) {
            let c = try RawRequest(.layerDecomposition)
            try c.set(WAIFU_REQUEST_LAYER_DECOMPOSITION_IMAGE, rgba: request.image)
            // 0 is the model's own, which an unset attribute is: the library refuses a 0.
            if request.steps > 0 { try c.set(WAIFU_REQUEST_LAYER_DECOMPOSITION_STEPS, count: request.steps) }
            if request.depthSteps > 0 {
                try c.set(WAIFU_REQUEST_LAYER_DECOMPOSITION_DEPTH_STEPS, count: request.depthSteps)
            }
            if request.resolution > 0 {
                try c.set(WAIFU_REQUEST_LAYER_DECOMPOSITION_RESOLUTION, count: request.resolution)
            }
            try c.set(WAIFU_REQUEST_LAYER_DECOMPOSITION_SPLIT_PAIRS, flag: request.splitPairs)
            try c.set(WAIFU_REQUEST_LAYER_DECOMPOSITION_STRATIFY_LAYERS, flag: request.stratifyLayers)
            try c.set(WAIFU_REQUEST_LAYER_DECOMPOSITION_SEED, seed: request.seed)
            return c
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
            cancel: { waifu_modelmanager_fetch_cancel(handle.fetch) },
            refused: { .refused(modelManagerLastError()) },
            submit: { userData in
                handle.fetch = model.withCString {
                    waifu_modelmanager_fetch_async($0, userData, onFetchProgress, onFetchComplete)
                }
                return handle.fetch != nil
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
            cancel: { waifu_modelmanager_fetch_cancel(handle.fetch) },
            refused: { .refused(modelManagerLastError()) },
            submit: { userData in
                handle.fetch = waifu_modelmanager_fetch_manifests_async(userData, onFetchProgress, onFetchComplete)
                return handle.fetch != nil
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
