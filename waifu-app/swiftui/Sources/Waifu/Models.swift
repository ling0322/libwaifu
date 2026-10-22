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

// What the library says about models and the machine, as the JSON WaifuKit hands over, and what
// this app keeps of the jobs it ran.

import Foundation
import WaifuKit

private let decoder: JSONDecoder = {
    let decoder = JSONDecoder()
    decoder.keyDecodingStrategy = .convertFromSnakeCase
    return decoder
}()

// MARK: - a model

/// A model that draws, as `waifu_describe_json` says it.
struct Chosen: Decodable {
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
    /// The kept keys it takes images under: "start_from", "control".
    var imageKeys: [String]?
    /// For what it cannot take, the sentence to show in its place.
    var whyNot: [String: String]?

    var guided: Bool { takesGuidance ?? true }
    var drawsFromAPicture: Bool { imageKeys?.contains("start_from") ?? false }
    var noPictureBecause: String? { drawsFromAPicture ? nil : whyNot?["start_from"] }
}

struct Voice: Decodable {
    struct Style: Decodable, Hashable {
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
    var styles: [Style]?
    var notAVoiceBecause: String?
}

struct Converter: Decodable {
    var name: String
    var fullName: String?
    var onDisk: Bool
    var steps: Int
    var rate: Int
    var convertsStyle: Bool?
    var noStyleBecause: String?
}

/// The model that is loaded, and what it can be asked for.
struct ModelInfo {
    /// What it was loaded as: a published name, or a manifest's path.
    var name: String
    /// "image", "speech" or "conversion": what it is first.
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

    /// What the library says of `name`, for a model about to be, or just, loaded.
    static func describe(_ name: String, device: String) throws -> ModelInfo {
        let data = try ModelManager.describeJSON(name)
        struct Kinds: Decodable {
            var kind: String
            var kinds: [String]?
            var bytes: UInt64?
        }
        struct Both: Decodable {
            var conversion: Converter?
        }
        let kinds = try decoder.decode(Kinds.self, from: data)
        var info = ModelInfo(
            name: name, kind: kinds.kind, kinds: kinds.kinds ?? [kinds.kind], device: device,
            bytes: kinds.bytes)
        switch kinds.kind {
        case "image":
            info.model = try decoder.decode(Chosen.self, from: data)
        case "speech":
            info.voice = try decoder.decode(Voice.self, from: data)
            info.converter = try decoder.decode(Both.self, from: data).conversion
        default:
            info.converter = try decoder.decode(Converter.self, from: data)
        }
        return info
    }

    var fullName: String {
        model?.fullName ?? voice?.fullName ?? converter?.fullName ?? name
    }

    /// Whether it is already fetched: one that is not is fetched by the first job that needs it.
    var onDisk: Bool {
        model?.onDisk ?? voice?.onDisk ?? converter?.onDisk ?? false
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

// MARK: - the machine

struct Machine: Decodable {
    struct Memory: Decodable {
        var total: Double?
        var used: Double?
    }

    struct GPU: Decodable {
        var name: String
        var unified: Bool?
    }

    struct VRAM: Decodable {
        var used: Double?
        var total: Double?
        var ours: Double?
    }

    var cpu: String?
    var cores: Int?
    var threads: Int?
    var memory: Memory?
    var gpu: GPU?
    var vram: VRAM?
    var whyNoVram: String?
    var accelerators: [String]?

    /// As the library measures it now. Runs sysctl and vm_stat: off the main thread.
    static func now() -> Machine? {
        guard let data = MachineInfo.json() else { return nil }
        return try? decoder.decode(Machine.self, from: data)
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
    // A conversion.
    var convertStyle: Bool?
    // A sound.
    var length: Double?
    var rate: Int?

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

/// A job this window asked for, or one read back out of the output folder.
struct Job: Identifiable {
    enum Status { case queued, running, done, failed, cancelled }

    var id: String
    /// "image", "speech" or "conversion".
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

/// What is kept on the disk for a finished reading or conversion -- a picture has an
/// `ImageRecord` -- and for a picture made before that: its id, kind and when, and what made it,
/// beside a file of the same id holding what it made.
struct Record: Codable {
    var id: String
    var kind: String
    var created: Double
    var finished: Double
    var made: Made
}
