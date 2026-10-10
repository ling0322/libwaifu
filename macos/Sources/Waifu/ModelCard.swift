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

// The Model card at the top of the right column: which model the task on show runs on, and on
// which device. Nothing is loaded from here -- the first Generate, Speak or Convert fetches the
// model and reads it, and the bar says how that is going.

import SwiftUI

struct ModelCard: View {
    @Bindable var session: Session

    @State private var published: [CatalogEntry] = []
    @State private var showLog = false

    private var kind: String { session.tab.kind }
    private var launcher: Launcher { session.launcher }

    /// The tag of the last item, which opens a panel rather than being chosen.
    private static let manifest = "\u{0}manifest"

    /// What is chosen for this tab's kind, as it is asked for.
    private var current: String { session.chosen[kind] ?? "" }

    /// The published models of this tab's kind.
    private var listed: [CatalogEntry] { published.filter { $0.kind == kind } }

    private var selection: Binding<String> {
        Binding(
            get: { current },
            set: { picked in
                if picked == ModelCard.manifest {
                    if let url = Media.choose([.yaml]) { session.choose(url.path, for: kind) }
                } else if !picked.isEmpty {
                    session.choose(picked, for: kind)
                }
            })
    }

    var body: some View {
        FormGroup {
            LabeledContent("Model") {
                // A pop-up like Device's: the published models of this tab's kind, what is
                // downloaded first, and a manifest of one's own at the end.
                Picker("Model", selection: selection) {
                    if current.isEmpty { Text("None").tag("") }
                    if !current.isEmpty, !listed.contains(where: { $0.name == current }) {
                        Text((current as NSString).lastPathComponent).tag(current)
                    }
                    let downloaded = listed.filter(\.cached)
                    if !downloaded.isEmpty {
                        Section("Downloaded") {
                            ForEach(downloaded) { Text($0.name).tag($0.name) }
                        }
                    }
                    let rest = listed.filter { !$0.cached }
                    if !rest.isEmpty {
                        Section("Not downloaded") {
                            ForEach(rest) { Text($0.name).tag($0.name) }
                        }
                    }
                    Divider()
                    Text("Choose a Manifest…").tag(ModelCard.manifest)
                }
                .labelsHidden()
                .help(current)
            }
            // Held while a job runs, like the settings under the card: the job is of this model,
            // on this device, kept in this folder.
            .disabled(session.canStop)
            LabeledContent("Device") {
                Picker("Device", selection: $session.device) {
                    Text("auto").tag("auto")
                    Text("metal").tag("metal")
                    Text("cpu").tag("cpu")
                }
                .labelsHidden()
                .fixedSize()
            }
            .disabled(session.canStop)
        } header: {
            HStack {
                Header("Model")
                Spacer()
                Button {
                    showLog = true
                } label: {
                    Image(systemName: "text.alignleft")
                }
                .buttonStyle(.borderless)
                .help("What the library has written")
            }
        } footer: {
            footnote
        }
        .task {
            published = await Task.detached { CatalogEntry.all() }.value
        }
        .sheet(isPresented: $showLog) {
            LogView(lines: launcher.log).frame(width: 760, height: 480)
        }
    }

    /// Where the chosen model is: not fetched yet, fetched, or loaded -- or why it is not a model.
    @ViewBuilder private var footnote: some View {
        if let why = session.notDescribed[kind] {
            Text(why).font(.caption).foregroundStyle(.red).fixedSize(horizontal: false, vertical: true)
        } else if let model = session.model {
            if launcher.has(model.name, on: session.device) {
                Footnote("Loaded on \(session.device).")
            } else if model.onDisk {
                Footnote("Downloaded. Read onto the device by the first job.")
            } else {
                Footnote("Not downloaded yet: the first job fetches it.")
            }
        } else {
            Footnote("Choose a model for \(session.tab.rawValue).")
        }
    }

}

/// What the library has written, for when the last line is not enough.
struct LogView: View {
    var lines: [String]
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        VStack(alignment: .trailing) {
            ScrollView {
                Text(lines.joined(separator: "\n"))
                    .font(.system(.caption, design: .monospaced))
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(8)
            }
            .background(Theme.field)
            Button("Close") { dismiss() }.keyboardShortcut(.cancelAction)
        }
        .padding()
    }
}
