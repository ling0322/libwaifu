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

// What the library says about models -- what each can do, asked of libwaifu one property at a
// time, and what its boxes start at, as libwaifu_modelmanager's JSON has it -- and what this app
// keeps of the jobs it ran.

import Foundation
import WaifuKit

private let decoder: JSONDecoder = {
    let decoder = JSONDecoder()
    decoder.keyDecodingStrategy = .convertFromSnakeCase
    return decoder
}()

// MARK: - a model

/// A model that draws.
struct Chosen {
    var name: String
    var fullName: String?
    var onDisk: Bool
    var width: Int
    var height: Int
    var steps: Int
    var guidance: Double
    var takesGuidance: Bool?
    var sampler: String?
    var sizes: [[Int]]?
    /// What both sides have to be a multiple of: 16, or 32 for SDXL.
    var alignment: Int?
    var prompt: String?
    var avoid: String?
    /// How many images a prompt can name; 0 for words only.
    var promptImages: Int?
    /// Whether its prompt reads pictures, named in it as <|name|>.
    var readsPictures: Bool { (promptImages ?? 0) > 0 }
    /// Whether it starts from a picture: img2img.
    var drawsFromAPicture: Bool
    /// Where it does not, the sentence to show in its place.
    var noPictureBecause: String?

    var guided: Bool { takesGuidance ?? true }
}

struct Voice {
    struct Style: Hashable {
        var label: String
        var instruction: String
    }

    var name: String
    var fullName: String?
    var onDisk: Bool
    var speed: Double
    var temperature: Double
    var rate: Int
    var takesARecording: Bool?
    var noLikenessBecause: String?
    /// Whether it has no voice of its own and says nothing without a recording to sound like.
    var needsARecording: Bool?
    /// Whether it follows an instruction for how to say it.
    var followsStyle: Bool
    /// Whether it can be played as it is said, a part at a time: a model that streams, on a device
    /// the load then checks.
    var streams: Bool = false
    /// What the style picker offers, above a box for any instruction: CosyVoice3's examples, for a
    /// voice that follows a style.
    var styles: [Style]? { followsStyle ? Voice.examples : nil }

    /// CosyVoice3's own examples of an instruction -- upstream's `instruct_list`
    /// (`cosyvoice/utils/common.py`), word for word, as `CosyVoice3::STYLES` keeps them: the manner
    /// first, the dialects after. Examples, not the set: any instruction may be written instead.
    static let examples: [Style] = [
        Style(label: "Happy", instruction: "请非常开心地说一句话。"),
        Style(label: "Sad", instruction: "请非常伤心地说一句话。"),
        Style(label: "Angry", instruction: "请非常生气地说一句话。"),
        Style(label: "Fast", instruction: "请用尽可能快地语速说一句话。"),
        Style(label: "Slow", instruction: "请用尽可能慢地语速说一句话。"),
        Style(label: "Loud", instruction: "Please say a sentence as loudly as possible."),
        Style(label: "Soft", instruction: "Please say a sentence in a very soft voice."),
        Style(label: "Peppa Pig", instruction: "我想体验一下小猪佩奇风格，可以吗？"),
        Style(label: "Robot", instruction: "你可以尝试用机器人的方式解答吗？"),
        Style(label: "Cantonese", instruction: "请用广东话表达。"),
        Style(label: "Northeastern", instruction: "请用东北话表达。"),
        Style(label: "Gansu", instruction: "请用甘肃话表达。"),
        Style(label: "Guizhou", instruction: "请用贵州话表达。"),
        Style(label: "Henan", instruction: "请用河南话表达。"),
        Style(label: "Hubei", instruction: "请用湖北话表达。"),
        Style(label: "Hunan", instruction: "请用湖南话表达。"),
        Style(label: "Jiangxi", instruction: "请用江西话表达。"),
        Style(label: "Minnan", instruction: "请用闽南话表达。"),
        Style(label: "Ningxia", instruction: "请用宁夏话表达。"),
        Style(label: "Shanxi", instruction: "请用山西话表达。"),
        Style(label: "Shaanxi", instruction: "请用陕西话表达。"),
        Style(label: "Shandong", instruction: "请用山东话表达。"),
        Style(label: "Shanghainese", instruction: "请用上海话表达。"),
        Style(label: "Sichuan", instruction: "请用四川话表达。"),
        Style(label: "Tianjin", instruction: "请用天津话表达。"),
        Style(label: "Yunnan", instruction: "请用云南话表达。"),
    ]
}

struct Converter {
    var name: String
    var fullName: String?
    var onDisk: Bool
    var steps: Int
    var rate: Int
    var convertsStyle: Bool?
    var noStyleBecause: String?
}

/// A model that takes a picture apart into layers: See-through.
struct Layerer {
    var name: String
    var fullName: String?
    var onDisk: Bool
    var steps: Int
    var depthSteps: Int
    /// The side of the square it draws the layers at.
    var resolution: Int
}

/// A model that writes down what is said in a recording: Qwen3-ASR.
struct Transcriber {
    var name: String
    var fullName: String?
    var onDisk: Bool
    /// The languages a recording may be said to be in, as the model names them; empty where it
    /// cannot be told one, and then it always finds out for itself.
    var languages: [String]
    /// What the box starts at: no language, which the model tells.
    var language: String
}

/// What libwaifu_modelmanager says a model's boxes start at: whichever of these its kind has.
private struct Defaults: Decodable {
    struct Conversion: Decodable {
        var steps: Int?
    }

    var width: Int?
    var height: Int?
    var steps: Int?
    var guidance: Double?
    var takesGuidance: Bool?
    var sizes: [[Int]]?
    var prompt: String?
    var avoid: String?
    var speed: Double?
    var temperature: Double?
    var conversion: Conversion?
    var depthSteps: Int?
    var resolution: Int?
    var language: String?
}

/// The model that is loaded, and what it can be asked for.
struct ModelInfo {
    /// What it was loaded as: a published name, or a manifest's path.
    var name: String
    /// "image", "speech", "conversion", "layers" or "transcription": what it is first.
    var kind: String
    /// Everything it does: CosyVoice3 reads and converts.
    var kinds: [String]
    /// What it runs on, as the launch screen chose it.
    var device: String
    /// How much its files are, where they are all on this Mac; nil before it is downloaded.
    var bytes: UInt64?
    var model: Chosen?
    var voice: Voice?
    var converter: Converter?
    var layerer: Layerer?
    var transcriber: Transcriber?

    /// What the library says of `name`, for a model about to be, or just, loaded: what it can do,
    /// asked of its manifest, and what its boxes start at.
    static func describe(_ name: String, device: String) throws -> ModelInfo {
        let opened = try ModelManager.open(name)
        let defaults = try decoder.decode(Defaults.self, from: ModelManager.defaultsJSON(of: opened))
        // What the catalogue calls a published model; a manifest of one's own, its file's name.
        let fullName = ModelManager.fullName(name) ?? (name as NSString).lastPathComponent
        let bytes = opened.sizeOnDisk
        let onDisk = bytes != nil

        // What it is first, as the tabs are kinds: CosyVoice3 reads, and converts as well.
        var kinds: [String] = []
        if opened.does(.layerDecomposition) { kinds.append("layers") }
        if opened.does(.speechToText) { kinds.append("transcription") }
        if opened.does(.textToSpeech) { kinds.append("speech") }
        if opened.does(.voiceConversion) { kinds.append("conversion") }
        if opened.does(.imageGeneration) { kinds.append("image") }
        guard let kind = kinds.first else {
            throw WaifuError.refused("\(fullName) does nothing this app asks of a model")
        }
        var info = ModelInfo(name: name, kind: kind, kinds: kinds, device: device, bytes: bytes)
        let rate = opened.sampleRate ?? 0
        let converter = Converter(
            name: name, fullName: fullName, onDisk: onDisk,
            steps: defaults.conversion?.steps ?? defaults.steps ?? 10, rate: rate,
            convertsStyle: opened.supports(.voiceConversionOfStyle).map { $0 == .yes },
            noStyleBecause: opened.whyNot(.voiceConversionOfStyle))
        switch kind {
        case "image":
            info.model = Chosen(
                name: name, fullName: fullName, onDisk: onDisk, width: defaults.width ?? 1024,
                height: defaults.height ?? 1024, steps: defaults.steps ?? 20, guidance: defaults.guidance ?? 7,
                takesGuidance: defaults.takesGuidance, sampler: opened.sampler, sizes: defaults.sizes,
                alignment: opened.widthHeightMultipleOf, prompt: defaults.prompt, avoid: defaults.avoid,
                promptImages: opened.does(.imageGenerationWithPromptImages) ? opened.maxPromptImages ?? 0 : 0,
                drawsFromAPicture: opened.does(.imageGenerationFromStartImage),
                noPictureBecause: opened.whyNot(.imageGenerationFromStartImage))
        case "transcription":
            info.transcriber = Transcriber(
                name: name, fullName: fullName, onDisk: onDisk, languages: opened.speechToTextLanguages ?? [],
                language: defaults.language ?? "")
        case "layers":
            info.layerer = Layerer(
                name: name, fullName: fullName, onDisk: onDisk, steps: defaults.steps ?? 30,
                depthSteps: defaults.depthSteps ?? 4, resolution: defaults.resolution ?? 1280)
        case "speech":
            info.voice = Voice(
                name: name, fullName: fullName, onDisk: onDisk, speed: defaults.speed ?? 1,
                temperature: defaults.temperature ?? 0.8, rate: rate,
                takesARecording: opened.supports(.textToSpeechLikeReference).map { $0 == .yes },
                noLikenessBecause: opened.whyNot(.textToSpeechLikeReference),
                needsARecording: opened.supports(.textToSpeechWithoutReference).map { $0 != .yes },
                followsStyle: opened.does(.textToSpeechWithStyle),
                streams: opened.does(.streamingTextToSpeech))
            if kinds.contains("conversion") { info.converter = converter }
        default:
            info.converter = converter
        }
        return info
    }

    var fullName: String {
        model?.fullName ?? voice?.fullName ?? converter?.fullName ?? layerer?.fullName
            ?? transcriber?.fullName ?? name
    }

    /// Whether it is already fetched: one that is not is fetched by the first job that needs it.
    var onDisk: Bool {
        model?.onDisk ?? voice?.onDisk ?? converter?.onDisk ?? layerer?.onDisk ?? transcriber?.onDisk
            ?? false
    }
}

// MARK: - the catalogue

struct CatalogEntry: Decodable, Identifiable, Hashable {
    var name: String
    var fullName: String
    var kind: String
    var cached: Bool
    /// Whether its manifest is here, with or without its weights: what it suggests can be read.
    var manifestHere: Bool?
    var bytesOnDisk: UInt64
    var explicit: Bool

    var id: String { "\(kind):\(name)" }

    /// The models listed before the rest, in this order: what to try first.
    static let first = ["anima:turbo"]

    /// Everything published, or nothing where the library could not say. The library lists by
    /// name; the ones in `first` are moved to the top, and the rest keep its order.
    static func all() -> [CatalogEntry] {
        struct Catalog: Decodable { var models: [CatalogEntry] }
        guard let data = try? ModelManager.catalogJSON(),
            let models = try? decoder.decode(Catalog.self, from: data).models
        else { return [] }
        let rank = { (entry: CatalogEntry) in first.firstIndex(of: entry.name) ?? first.count }
        return models.enumerated()
            .sorted { (rank($0.element), $0.offset) < (rank($1.element), $1.offset) }
            .map(\.element)
    }
}

// MARK: - what was made

/// What a finished job was made with: a picture's settings, a reading's or a conversion's. Kept
/// beside what it made, so that it can be read back, reused, and written under it.
struct Made: Codable {
    var seed: UInt64
    var seconds: Double
    var model: String
    var parameters: String
    // A picture.
    var prompt: String?
    var negative: String?
    var width: Int?
    var height: Int?
    var steps: Int?
    var guidance: Double?
    var strength: Double?
    /// What it was drawn on, as a picture's record spells it: "metal", never "auto".
    var device: String?
    /// The picture it started from, as an absolute path.
    var startFrom: String?
    /// The pictures its prompt named, by the name it wrote as <|name|>, as absolute paths.
    var promptPictures: [String: String]?
    // A reading.
    var text: String?
    var speed: Double?
    var temperature: Double?
    var style: String?
    var fromARecording: Bool?
    /// Whether it was played as it was said, a part at a time: a reading of its own.
    var streamed: Bool?
    // A conversion.
    var convertStyle: Bool?
    // A transcription, which is held in memory and never kept: the language it heard. The words are
    // `text`.
    var heardLanguage: String?
    // A sound.
    var length: Double?
    var rate: Int?
    // A picture taken apart: the picture it was, and the steps, which are `steps`, of both models.
    var depthSteps: Int?
    var resolution: Int?
    var splitPairs: Bool?
    /// The side of the square canvas the layers are on.
    var canvas: Int?
    /// Bottom first, each in a PNG of its own in the job's folder.
    var layers: [KeptLayer]?
    /// Hair, hands, top and bottom each cut into a near and a far half by their own depth, in the
    /// order the library gave them, their pixels in a document of their own in the job's folder:
    /// what Split by depth lays among the layers. Nil for a picture taken apart before they were
    /// kept.
    var halves: [KeptHalf]?

    /// A picture's settings, with the lines that are copied and shown under it.
    static func picture(
        seed: UInt64, model: String, prompt: String, negative: String, width: Int, height: Int,
        steps: Int, guidance: Double, strength: Double?
    ) -> Made {
        var lines = [prompt]
        if !negative.isBlank { lines.append("Negative prompt: \(negative)") }
        var settings = "Steps: \(steps), CFG scale: \(guidance), Seed: \(seed), Size: \(width)x\(height), Model: \(model)"
        if let strength { settings += ", Denoising strength: \(strength)" }
        lines.append(settings)
        return Made(
            seed: seed, seconds: 0, model: model, parameters: lines.joined(separator: "\n"),
            prompt: prompt, negative: negative, width: width, height: height, steps: steps,
            guidance: guidance, strength: strength)
    }
}

/// One layer of a picture taken apart, as it is kept: where on the canvas, how far away, and the
/// file of its pixels, named in the job's own folder.
struct KeptLayer: Codable, Hashable {
    var name: String
    var left: Int
    var top: Int
    var width: Int
    var height: Int
    var depth: Float
    var file: String
    /// What the model named it and how far away it said it was, kept the first time either is
    /// changed: what Reset puts back. Nil while they are the model's.
    var modelName: String?
    var modelDepth: Float?
    /// The library's id for it, which its halves name it by; nil in a record from before ids.
    var id: Int?

    /// Whether its name or depth is somebody's rather than the model's.
    var edited: Bool { modelName != nil || modelDepth != nil }
}

/// A half of a layer of a picture taken apart, as it is kept: where on the canvas, how far away,
/// and the layer it was cut from. Its pixels are in the job's document of halves, in the order the
/// record lists them.
struct KeptHalf: Codable, Hashable {
    /// As the library named it: "topwear-0", "topwear-1".
    var name: String
    var left: Int
    var top: Int
    var width: Int
    var height: Int
    /// Its own depth, for a far half; a near one is at its layer's, whatever that is now.
    var depth: Float
    /// Whether it is the near half (-0) rather than the far (-1).
    var near: Bool
    /// The library's id for it.
    var id: Int
    /// The id of the layer it was cut from, which neither a name nor a depth changes.
    var of: Int
}

/// A job this window asked for, or one read back out of the output folder.
struct Job: Identifiable {
    enum Status { case queued, running, done, failed, cancelled }

    var id: String
    /// "image", "speech", "conversion", "layers" or "transcription".
    var kind: String
    var status: Status
    /// Milliseconds since 1970, as file names are made from.
    var created: Double
    var finished: Double?
    var error: String?
    var made: Made?

    var isGoing: Bool { status == .queued || status == .running }
    var isDone: Bool { status == .done && made != nil }
}

/// How a picture was made, as docs/image-metadata.md lays it out: kept beside the PNG of the same
/// name, saying which build drew it and what it was asked to draw.
struct ImageRecord: Codable {
    struct Environment: Codable {
        var libwaifu: Library
    }

    struct Library: Codable {
        var version: String
        /// Written as null, not left out, for a build made without git.
        var revision: String?

        func encode(to encoder: Encoder) throws {
            var container = encoder.container(keyedBy: CodingKeys.self)
            try container.encode(version, forKey: .version)
            try container.encode(revision, forKey: .revision)
        }
    }

    struct Request: Codable {
        /// "txt2img" or "img2img".
        var task: String
        var parameters: Parameters
    }

    struct Parameters: Codable {
        var model: String
        var device: String
        var prompt: String
        var negative: String
        var width: Int
        var height: Int
        var steps: Int
        var guidance: Double
        /// A u64 in decimal: past what a JSON number keeps in JavaScript.
        var seed: String
        /// img2img only.
        var strength: Double?
        var inputs: [String: String]

        init(made: Made) {
            model = made.model
            device = made.device ?? ""
            prompt = made.prompt ?? ""
            negative = made.negative ?? ""
            width = made.width ?? 0
            height = made.height ?? 0
            steps = made.steps ?? 0
            guidance = made.guidance ?? 0
            seed = String(made.seed)
            strength = made.strength
            inputs = made.promptPictures ?? [:]
            if let startFrom = made.startFrom { inputs["start_from"] = startFrom }
        }
    }

    struct Result: Codable {
        /// The absolute path the PNG was written to.
        var output: String
        /// When drawing finished: UTC, ISO 8601 with milliseconds.
        var created: String
        var generationSeconds: Double

        enum CodingKeys: String, CodingKey {
            case output, created
            case generationSeconds = "generation_seconds"
        }
    }

    var environment: Environment
    var request: Request
    var result: Result

    private static func stamp() -> ISO8601DateFormatter {
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        formatter.timeZone = TimeZone(identifier: "UTC")
        return formatter
    }

    /// The record of a picture written to `output`, finished at `finished` milliseconds since 1970.
    init(made: Made, output: URL, finished: Double, library: LibraryVersion?) {
        environment = Environment(
            libwaifu: Library(version: library?.version ?? "unknown", revision: library?.revision))
        request = Request(task: made.startFrom == nil ? "txt2img" : "img2img", parameters: Parameters(made: made))
        result = Result(
            output: output.path,
            created: Self.stamp().string(from: Date(timeIntervalSince1970: finished / 1000)),
            generationSeconds: (made.seconds * 100).rounded() / 100)
    }

    /// When drawing finished, in milliseconds since 1970.
    var finished: Double? {
        Self.stamp().date(from: result.created).map { $0.timeIntervalSince1970 * 1000 }
    }

    /// What the app keeps of it, to list, show and reuse.
    var made: Made? {
        let parameters = request.parameters
        guard let seed = UInt64(parameters.seed) else { return nil }
        var made = Made.picture(
            seed: seed, model: parameters.model, prompt: parameters.prompt, negative: parameters.negative,
            width: parameters.width, height: parameters.height, steps: parameters.steps,
            guidance: parameters.guidance, strength: parameters.strength)
        made.seconds = result.generationSeconds
        made.device = parameters.device
        made.startFrom = parameters.inputs["start_from"]
        // Every other input is a picture the prompt named: "control" is kept for a ControlNet's,
        // which nothing here draws with.
        let named = parameters.inputs.filter { $0.key != "start_from" && $0.key != "control" }
        made.promptPictures = named.isEmpty ? nil : named
        return made
    }

    /// The JSON written beside the picture: keys sorted, which puts environment, request and
    /// result in the order the format lists them, and paths unescaped.
    func json() throws -> Data {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        return try encoder.encode(self)
    }
}

/// What is kept on the disk for a finished reading, conversion or picture taken apart -- a picture has an
/// `ImageRecord` -- and for a picture made before that: its id, kind and when, and what made it,
/// beside a file of the same id holding what it made.
struct Record: Codable {
    var id: String
    var kind: String
    var created: Double
    var finished: Double
    var made: Made
}
