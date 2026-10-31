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

// The waifu2d page: a drawing in layers, as a Photoshop document, made to move. Nothing here runs
// a model -- the engine is not asked for anything -- so it is a page of its own beside the tasks,
// as the Model Manager is.

import SwiftUI
import UniformTypeIdentifiers

/// The document on the page and how it moves.
@MainActor @Observable
final class RigStage {
    private(set) var file: URL?
    private(set) var rig: Rig?
    private(set) var reading = false
    /// Why the last document could not be read.
    private(set) var complaint: String?

    /// What moves on its own, and after the pointer.
    var switches = RigSwitches()
    /// As open as the eyes may be, 0 shut to 1 open.
    var open = 1.0
    /// How soon the open eye gives way to the closed one: Anime2.5DRig's "closes easily", 0 to 1.
    var ease = 0.3
    /// How far the head turns up and down: 1 as the rig turns it, 0 not at all.
    var nod = 1.0
    /// How far the body turns and leans after the head: 1 as the rig turns it, 0 not at all.
    var follow = 1.0
    /// How many blinks have been asked for: the view blinks once each time it goes up.
    private(set) var blinks = 0

    func blinkNow() { blinks += 1 }

    /// Reads the document at `url` and makes a rig of it, off the main thread.
    func open(_ url: URL) {
        reading = true
        complaint = nil
        Task {
            let made = await Task.detached { Result { Rig.build(try PSDDocument.read(Data(contentsOf: url))) } }.value
            reading = false
            switch made {
            case .success(let rig):
                self.rig = rig
                file = url
            case .failure(let error):
                complaint = "\(url.lastPathComponent): \(error.localizedDescription)"
            }
        }
    }
}

/// The middle of the page: the document to open, and the drawing moving under it.
struct RigPage: View {
    var stage: RigStage

    var body: some View {
        VStack(spacing: 0) {
            FormGroup {
                PSDDrop(stage: stage)
            } header: {
                Header("Drawing in layers (PSD)")
            } footer: {
                Footnote(
                    "Layers named as See-through or Anime2.5DRig names them move: the eyes blink, the head turns, "
                        + "the hair swings. An img2layers document works as it is.")
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
                        .background(Checkerboard())
                        .frame(maxWidth: .infinity, maxHeight: .infinity)
                } else {
                    Text(stage.reading ? "Reading it…" : "Nothing to move yet.")
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

/// The right column of the page: how it moves.
struct RigSettings: View {
    @Bindable var stage: RigStage

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 12) {
                FormGroup {
                    switchRow("Follow the pointer", \.followMouse, help: "Turn the head and look toward the pointer over the drawing")
                    switchRow("Sway", \.idle, help: "Sway a little all the time, and let the hair drift")
                    switchRow(
                        "Look around", \.lookAround,
                        help: "Look somewhere else every 1.4 to 4 seconds, while not following the pointer")
                    switchRow("Hair swings", \.physics, help: "Swing the hair in strands, and sway earrings and tails")
                    SliderRow(
                        label: "Up and down", value: $stage.nod, range: 0...2, step: 0.1, decimals: 1,
                        explanation: "How far the head turns up and down: 1 as the rig turns it, 0 not at all.")
                    SliderRow(
                        label: "Body follows", value: $stage.follow, range: 0...2, step: 0.1, decimals: 1,
                        explanation: "How far the body turns and leans after the head as it turns left and right: 0 not at all.")
                } header: {
                    Header("Motion")
                } footer: {
                    Footnote(
                        "The head turns about the neck, each layer by how far in front of the face it is; the hair "
                            + "follows on two springs a strand, as Anime2.5DRig moves it.")
                }
                .disabled(stage.rig == nil)

                FormGroup {
                    switchRow("On its own", \.blink, help: "Blink every 1.6 to 5.4 seconds, now and then twice")
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
            }
            .padding([.horizontal, .bottom], 12)
            .padding(.top, Theme.top(12, from26: 1))
        }
        .environment(\.rowPadding, 5)
        .labeledContentStyle(SettingRowStyle())
        .textFieldStyle(.plain)
    }

    private func switchRow(_ label: String, _ key: WritableKeyPath<RigSwitches, Bool>, help: String) -> some View {
        LabeledContent(label) {
            Toggle(label, isOn: $stage.switches[dynamicMember: key])
                .labelsHidden()
                .toggleStyle(.switch)
        }
        .help(help)
    }
}
