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

/// A model that draws, as `waifu_modelmanager_describe_json` says it.
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
    var model: Chosen?
    var voice: Voice?
    var converter: Converter?

    /// What the library says of `name`, for a model about to be, or just, loaded.
    static func describe(_ name: String, device: String) throws -> ModelInfo {
        let data = try ModelManager.describeJSON(name)
        struct Kinds: Decodable {
            var kind: String
            var kinds: [String]?
        }
        struct Both: Decodable {
            var conversion: Converter?
        }
        let kinds = try decoder.decode(Kinds.self, from: data)
        var info = ModelInfo(
            name: name, kind: kinds.kind, kinds: kinds.kinds ?? [kinds.kind], device: device)
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

/// What is kept on the disk for a finished job: its id, kind and when, and what made it, beside a
/// file of the same id holding what it made.
struct Record: Codable {
    var id: String
    var kind: String
    var created: Double
    var finished: Double
    var made: Made
}
