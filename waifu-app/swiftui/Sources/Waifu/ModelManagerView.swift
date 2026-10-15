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

// The Model Manager page: what is published, what of it is on this Mac and how much room it
// takes, and fetching or deleting it -- what `waifu models` does in the terminal.

import AppKit
import Observation
import SwiftUI
import WaifuKit

/// The catalogue, and the downloads under way.
@MainActor @Observable
final class ModelStore {
    private(set) var entries: [CatalogEntry] = []
    private(set) var directory: String?
    /// How far each download has got, by model name.
    private(set) var fetching: [String: JobProgress?] = [:]
    var complaint: String?

    private var tasks: [String: Task<Void, Never>] = [:]

    func refresh() async {
        entries = await Task.detached { CatalogEntry.all() }.value
        directory = ModelManager.directory
    }

    /// One row per model: a model that both reads and converts is listed once, under what it is
    /// first.
    var models: [CatalogEntry] {
        var seen = Set<String>()
        return entries.filter { seen.insert($0.name).inserted }
    }

    func download(_ name: String) {
        guard fetching[name] == nil else { return }
        complaint = nil
        fetching[name] = .some(nil)
        tasks[name] = Task {
            do {
                try await ModelManager.fetch(name) { progress in
                    Task { @MainActor in
                        if self.fetching[name] != nil { self.fetching[name] = progress }
                    }
                }
            } catch is CancellationError {
            } catch {
                complaint = "\(name): \(error.localizedDescription)"
            }
            fetching[name] = nil
            tasks[name] = nil
            await refresh()
        }
    }

    /// Stops a download between files; what was fetched stays, and the next download goes on
    /// from it.
    func cancel(_ name: String) {
        tasks[name]?.cancel()
    }

    func delete(_ name: String) async {
        do {
            try ModelManager.remove(name)
        } catch {
            complaint = "\(name): \(error.localizedDescription)"
        }
        await refresh()
    }

    func move(to path: String?) async {
        do {
            try ModelManager.setDirectory(path)
        } catch {
            complaint = error.localizedDescription
        }
        await refresh()
    }
}

struct ModelManagerView: View {
    var session: Session
    @State private var store = ModelStore()
    @State private var deleting: CatalogEntry?

    private static let kinds: [(kind: String, title: String)] = [
        ("image", "Pictures"), ("speech", "Voices"), ("conversion", "Voice conversion"),
    ]

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                folder
                if let complaint = store.complaint {
                    Label(complaint, systemImage: "exclamationmark.triangle.fill")
                        .font(.caption).foregroundStyle(.red).textSelection(.enabled)
                }
                ForEach(ModelManagerView.kinds, id: \.kind) { kind in
                    let models = store.models.filter { $0.kind == kind.kind }
                    if !models.isEmpty {
                        FormGroup {
                            ForEach(models) { row($0) }
                        } header: {
                            Header(kind.title)
                        }
                    }
                }
            }
            .padding(16)
            .frame(maxWidth: 760)
            .frame(maxWidth: .infinity)
        }
        .labeledContentStyle(RowStyle())
        .background(Theme.window)
        .task { await store.refresh() }
        .confirmationDialog(
            "Delete \(deleting?.name ?? "")?", isPresented: .constant(deleting != nil), presenting: deleting
        ) { entry in
            Button("Delete", role: .destructive) {
                deleting = nil
                Task { await store.delete(entry.name) }
            }
            Button("Cancel", role: .cancel) { deleting = nil }
        } message: { entry in
            Text("\(bytes(entry.bytesOnDisk)) on the disk. It is downloaded again the next time it is used.")
        }
    }

    private var folder: some View {
        FormGroup {
            LabeledContent("Kept in") {
                HStack(spacing: 6) {
                    Text(store.directory ?? "the default folder")
                        .lineLimit(1).truncationMode(.middle)
                        .foregroundStyle(.secondary)
                        .textSelection(.enabled)
                    Button("Show in Finder") {
                        if let directory = store.directory {
                            NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: directory)])
                        }
                    }
                    .controlSize(.small)
                    .disabled(store.directory == nil)
                    Button("Change…") {
                        if let url = chooseFolder() { Task { await store.move(to: url.path) } }
                    }
                    .controlSize(.small)
                }
            }
        } header: {
            Header("Model Manager")
        } footer: {
            Footnote("Changing the folder does not move what is in the old one: a model is fetched into the new one when it is next used.")
        }
    }

    @ViewBuilder private func row(_ entry: CatalogEntry) -> some View {
        HStack(spacing: 12) {
            VStack(alignment: .leading, spacing: 2) {
                HStack(spacing: 6) {
                    Text(entry.name).fontWeight(.medium)
                    if session.launcher.has(entry.name, on: session.device) {
                        Text("loaded").font(.caption2).padding(.horizontal, 5).padding(.vertical, 1)
                            .background(Color.green.opacity(0.25), in: Capsule())
                    }
                }
                Text(entry.fullName).font(.caption).foregroundStyle(.secondary).lineLimit(1)
            }
            Spacer(minLength: 8)
            if let progress = store.fetching[entry.name] {
                fetchingStatus(progress)
                Button { store.cancel(entry.name) } label: { rowLabel("Cancel") }.controlSize(.small)
            } else if entry.cached {
                Text(bytes(entry.bytesOnDisk)).font(.caption).foregroundStyle(.secondary).monospacedDigit()
                // A trash can rather than a word: quiet in a list where most rows have one, and
                // what it does is asked again before it is done.
                Button(role: .destructive) {
                    deleting = entry
                } label: {
                    // As wide as a Download button, so the buttons line up down the list.
                    Image(systemName: "trash").frame(width: 64)
                }
                .controlSize(.small)
                .help("Delete \(entry.name)")
                .disabled(session.canStop)
            } else {
                if entry.bytesOnDisk > 0 {
                    Text("\(bytes(entry.bytesOnDisk)) of it here")
                        .font(.caption).foregroundStyle(.secondary).monospacedDigit()
                }
                Button { store.download(entry.name) } label: { rowLabel("Download") }.controlSize(.small)
            }
        }
    }

    /// A row's button title, at one width for all of them, so the buttons line up down the list.
    private func rowLabel(_ title: String) -> some View {
        Text(title).frame(width: 64)
    }

    private func fetchingStatus(_ progress: JobProgress?) -> some View {
        VStack(alignment: .trailing, spacing: 3) {
            if let fraction = progress?.fraction {
                ProgressView(value: fraction).frame(width: 140)
            } else {
                ProgressView().progressViewStyle(.linear).frame(width: 140)
            }
            Text(progress?.words ?? "starting")
                .font(.caption2).foregroundStyle(.secondary).lineLimit(1).monospacedDigit()
                .frame(maxWidth: 220, alignment: .trailing)
        }
    }

    private func bytes(_ count: UInt64) -> String {
        ByteCountFormatter.string(fromByteCount: Int64(count), countStyle: .file)
    }

    private func chooseFolder() -> URL? {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.canCreateDirectories = true
        return panel.runModal() == .OK ? panel.url : nil
    }
}
