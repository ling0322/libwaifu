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

// The live2d page: a drawing in layers, as a Photoshop document, made to blink. Nothing here runs
// a model -- the engine is not asked for anything -- so it is a page of its own beside the tasks,
// as the Model Manager is.

import SwiftUI
import UniformTypeIdentifiers
import WaifuKit

/// The document on the page and how it blinks.
@MainActor @Observable
final class RigStage {
    private(set) var file: URL?
    private(set) var rig: Rig?
    private(set) var reading = false
    /// Why the last document could not be read.
    private(set) var complaint: String?

    /// The document as read, and each of its layers' depth by name, where it keeps it -- in its
    /// hidden group, or in the depth document beside it.
    private var document: PSDDocument?
    private var depths: [String: PSDLayer] = [:]
    /// Whether hair, hands, top and bottom are cut into a near and a far layer by their depth
    /// before the rig is made: the back of a collar behind the neck and its flaps in front.
    var stratify = false {
        didSet { if stratify != oldValue { remake() } }
    }
    /// Whether every layer of the document has its depth, which stratifying needs.
    var canStratify: Bool {
        guard let document else { return false }
        let layers = document.splitDepth().layers
        return !layers.isEmpty && layers.allSatisfy { depths[$0.name] != nil }
    }

    /// Whether it blinks on its own.
    var automatic = true
    /// As open as the eyes may be, 0 shut to 1 open.
    var open = 1.0
    /// How soon the open eye gives way to the closed one: Anime2.5DRig's "closes easily", 0 to 1.
    var ease = 0.3
    /// How many blinks have been asked for: the view blinks once each time it goes up.
    private(set) var blinks = 0

    func blinkNow() { blinks += 1 }

    /// Reads the document at `url`, and its depth, and makes a rig of it, off the main thread.
    func open(_ url: URL) {
        reading = true
        complaint = nil
        let stratify = stratify
        Task {
            let made = await Task.detached { () -> Result<(PSDDocument, [String: PSDLayer], Rig), Error> in
                Result {
                    let document = try PSDDocument.read(Data(contentsOf: url))
                    let depths = Self.depths(of: document, at: url)
                    return (document, depths, try Self.rig(document, depths: depths, stratify: stratify))
                }
            }.value
            reading = false
            switch made {
            case .success(let (document, depths, rig)):
                self.document = document
                self.depths = depths
                self.rig = rig
                file = url
            case .failure(let error):
                complaint = "\(url.lastPathComponent): \(error.localizedDescription)"
            }
        }
    }

    /// The rig made again from the document as read: stratified now, or not.
    private func remake() {
        guard let document, let file else { return }
        let (depths, stratify) = (depths, stratify)
        reading = true
        Task {
            let made = await Task.detached { Result { try Self.rig(document, depths: depths, stratify: stratify) } }.value
            reading = false
            switch made {
            case .success(let rig): self.rig = rig
            case .failure(let error): complaint = "\(file.lastPathComponent): \(error.localizedDescription)"
            }
        }
    }

    /// `document`'s rig: its layers as they are, or stratified first by their depth.
    nonisolated static func rig(_ document: PSDDocument, depths: [String: PSDLayer], stratify: Bool) throws -> Rig {
        let layers = document.splitDepth().layers
        guard stratify, !layers.isEmpty, layers.allSatisfy({ depths[$0.name] != nil }) else {
            return Rig.build(document)
        }
        let cut = try Stratify.layers(
            layers.map { layer in
                Stratify.Layer(
                    name: layer.name, left: layer.left, top: layer.top, image: layer.image,
                    depth: depths[layer.name]!.image)
            })
        return Rig.build(
            PSDDocument(
                width: document.width, height: document.height,
                layers: cut.map { PSDLayer(name: $0.name, left: $0.left, top: $0.top, image: $0.image, opacity: 1, visible: true) }))
    }

    /// Each layer's depth by name: the document's own, in its hidden group, or else the depth
    /// document img2layers keeps beside it -- `<id>.layers/depth.psd` in its folder, or one saved
    /// as `<name>-depth.psd` -- where its layers are the document's, in order.
    nonisolated static func depths(of document: PSDDocument, at url: URL) -> [String: PSDLayer] {
        let (layers, own) = document.splitDepth()
        if !own.isEmpty {
            var depths: [String: PSDLayer] = [:]
            for layer in layers {
                if let depth = own.first(where: { $0.name == PSD.depthName(of: layer.name) }) { depths[layer.name] = depth }
            }
            return depths
        }
        let stem = url.deletingPathExtension()
        let beside = [
            stem.appendingPathExtension("layers").appendingPathComponent("depth.psd"),
            stem.deletingLastPathComponent().appendingPathComponent(stem.lastPathComponent + "-depth.psd"),
        ]
        for candidate in beside {
            guard let depth = try? PSDDocument.read(Data(contentsOf: candidate)),
                depth.layers.map(\.name) == layers.map(\.name)
            else { continue }
            // A name twice over is a document's own doing: the first is the one kept.
            return Dictionary(zip(layers.map(\.name), depth.layers), uniquingKeysWith: { first, _ in first })
        }
        return [:]
    }
}

/// The middle of the page: the document to open, and the drawing blinking under it.
struct RigPage: View {
    var stage: RigStage

    var body: some View {
        VStack(spacing: 0) {
            FormGroup {
                PSDDrop(stage: stage)
            } header: {
                Header("Drawing in layers (PSD)")
            } footer: {
                Footnote("Layers named as See-through names them blink: eyewhite, irides, eyelash. An img2layers document works as it is.")
            }
            .padding([.horizontal, .bottom], 16)
            .padding(.top, Theme.top(16, from26: 2))
            .labeledContentStyle(RowStyle())

            Divider()

            VStack(spacing: 10) {
                if let complaint = stage.complaint {
                    Label(complaint, systemImage: "exclamationmark.triangle.fill")
                        .foregroundStyle(.red)
                        .textSelection(.enabled)
                        .frame(maxWidth: .infinity, alignment: .leading)
                }
                if let rig = stage.rig {
                    ForEach(rig.notes, id: \.self) { note in
                        Label(note, systemImage: "info.circle")
                            .foregroundStyle(.secondary)
                            .frame(maxWidth: .infinity, alignment: .leading)
                    }
                    RigView(rig: rig, stage: stage)
                        .id(stage.file)
                        .frame(maxWidth: .infinity, maxHeight: .infinity)
                } else {
                    Text(stage.reading ? "Reading it…" : "Nothing to blink yet.")
                        .foregroundStyle(.secondary)
                        .frame(maxWidth: .infinity, maxHeight: .infinity)
                }
            }
            .padding(12)
        }
    }
}

/// The document to open: a box of one height, a prompt's, as every other box over the middle is.
struct PSDDrop: View {
    var stage: RigStage
    @State private var over = false

    var body: some View {
        DropBox(over: over, height: PromptField.height) {
            if let file = stage.file, let rig = stage.rig {
                VStack(spacing: 4) {
                    Label(file.lastPathComponent, systemImage: "square.3.layers.3d")
                    Text("\(rig.parts.count) layers, \(rig.width) × \(rig.height); \(rig.eyes.count) of 2 eyes blink. Drop or choose another.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                .padding(.horizontal, 10)
            } else {
                Label(stage.reading ? "Reading it…" : "Drop or choose a .psd", systemImage: "doc.badge.plus")
                    .foregroundStyle(.secondary)
            }
        }
        .onTapGesture {
            if let url = Media.choose([Media.psd]) { stage.open(url) }
        }
        .onDrop(of: [.fileURL], isTargeted: $over) { providers in
            guard let provider = providers.first else { return false }
            _ = provider.loadObject(ofClass: URL.self) { url, _ in
                if let url { Task { @MainActor in stage.open(url) } }
            }
            return true
        }
    }
}

/// The right column of the page: how it blinks.
struct RigSettings: View {
    @Bindable var stage: RigStage

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 12) {
                FormGroup {
                    LabeledContent("On its own") {
                        Toggle("On its own", isOn: $stage.automatic)
                            .labelsHidden()
                            .toggleStyle(.switch)
                    }
                    .help("Blink every 1.6 to 5.4 seconds, now and then twice")
                    LabeledContent("Now") {
                        Button("Blink Now") { stage.blinkNow() }
                            .frame(maxWidth: .infinity, alignment: .trailing)
                    }
                    SliderRow(
                        label: "Eyes open", value: $stage.open, range: 0...1, step: 0.05, decimals: 2,
                        explanation: "As open as the eyes are when they are not blinking: 1 is as drawn, 0 shut.")
                    SliderRow(
                        label: "Closes at", value: $stage.ease, range: 0...1, step: 0.05, decimals: 2,
                        explanation: "How soon the open eye gives way to the closed one. Higher shows the closed eye sooner.")
                } header: {
                    Header("Blink")
                } footer: {
                    Footnote("Closed in 80 ms, held 340 ms, open in 160 ms, as Anime2.5DRig blinks.")
                }
                .disabled(stage.rig == nil)

                FormGroup {
                    LabeledContent("Split by depth") {
                        Toggle("Split by depth", isOn: $stage.stratify)
                            .labelsHidden()
                            .toggleStyle(.switch)
                    }
                    .help("Hair, hands, top and bottom each as a near and a far layer, by their own depth")
                } header: {
                    Header("Layers")
                } footer: {
                    Footnote(
                        stage.canStratify || stage.rig == nil
                            ? "Hair, hands, top and bottom each cut into a near and a far layer by their own depth, as "
                                + "the See-through paper does: the back of a collar behind the neck, its flaps in front."
                            : "This document keeps no depth to split by. img2layers writes it into every document, and "
                                + "beside the ones it wrote before.")
                }
                .disabled(!stage.canStratify)
            }
            .padding([.horizontal, .bottom], 12)
            .padding(.top, Theme.top(12, from26: 1))
        }
        .environment(\.rowPadding, 5)
        .labeledContentStyle(SettingRowStyle())
        .textFieldStyle(.plain)
    }
}
