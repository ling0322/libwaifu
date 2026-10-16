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

// What came of it: the bar, the newest picture or clip where it can be looked at or listened to,
// and everything else in a strip under it.

import AVFoundation
import SwiftUI

struct OutputPane: View {
    @Bindable var session: Session
    @State private var deleting: Job?
    @State private var deletingAll = false

    private var pictures: Bool { session.tab.kind == "image" }

    var body: some View {
        VStack(spacing: 10) {
            ProgressBar(bar: session.bar)
            if let note = session.note {
                Label(note.said, systemImage: note.bad ? "exclamationmark.triangle.fill" : "info.circle")
                    .foregroundStyle(note.bad ? .red : .secondary)
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }

            Group {
                if let job = session.shown {
                    if pictures { PictureFrame(session: session, job: job) } else { ClipFrame(session: session, job: job) }
                } else {
                    Text(nothing).foregroundStyle(.secondary)
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)

            if let job = session.shown { actions(job) }
            strip
        }
        .padding(12)
        .confirmationDialog(
            "Delete this \(pictures ? "picture" : "clip")?", isPresented: .constant(deleting != nil),
            presenting: deleting
        ) { job in
            Button("Delete", role: .destructive) {
                session.forget(job)
                deleting = nil
            }
            Button("Cancel", role: .cancel) { deleting = nil }
        } message: { _ in
            Text("Nothing else keeps a copy.")
        }
    }

    private var nothing: String {
        switch session.tab.kind {
        case "speech": "Nothing said yet."
        case "conversion": "Nothing converted yet."
        default: "Nothing drawn yet."
        }
    }

    private func actions(_ job: Job) -> some View {
        HStack {
            Button("Save…") { session.save(job) }
                .keyboardShortcut("s", modifiers: .command)
                .disabled(session.outputs[job.id] == nil)
            Button("Reuse these settings") { session.reuse(job) }
            Spacer()
            Button("Delete", role: .destructive) { deleting = job }
            Button("Delete All…", role: .destructive) { deletingAll = true }
                .confirmationDialog(
                    "Delete all \(session.done.count) \(pictures ? "pictures" : "clips")?",
                    isPresented: $deletingAll
                ) {
                    Button("Delete All", role: .destructive) { session.forgetAll() }
                    Button("Cancel", role: .cancel) {}
                } message: {
                    Text("Nothing else keeps a copy.")
                }
        }
    }

    @ViewBuilder private var strip: some View {
        let done = session.done
        if !done.isEmpty {
            ScrollView(.horizontal) {
                LazyHStack(spacing: 6) {
                    ForEach(done) { job in
                        let on = job.id == session.shown?.id
                        Button {
                            if pictures { session.showing = job.id } else { session.playing = job.id }
                        } label: {
                            if pictures { Thumbnail(session: session, job: job) } else { ClipChip(job: job) }
                        }
                        .buttonStyle(.plain)
                        .overlay(
                            RoundedRectangle(cornerRadius: 5)
                                .stroke(on ? Color.accentColor : .clear, lineWidth: 2))
                        .help(Media.fileName(created: job.finished ?? job.created, kind: job.kind)
                            + (job.made.map { " -- seed \($0.seed)" } ?? ""))
                    }
                }
                .padding(2)
            }
            .frame(height: pictures ? 62 : 56)
        }
    }
}

/// The bar, which is the only thing in the window that moves on its own.
struct ProgressBar: View {
    var bar: BarState

    var body: some View {
        if bar.busy {
            VStack(alignment: .leading, spacing: 4) {
                if let fraction = bar.fraction {
                    ProgressView(value: fraction)
                } else {
                    // Reading a model is one call that returns when it returns: there is no
                    // fraction to show, so it says so by moving.
                    ProgressView().progressViewStyle(.linear)
                }
                Text(words).font(.caption).foregroundStyle(.secondary).monospacedDigit()
            }
        }
    }

    private var words: String {
        if bar.stopping { return "stopping after this step…" }
        return [
            bar.doing.isEmpty ? nil : bar.doing,
            bar.fraction.map { "\(Int(($0 * 100).rounded()))%" },
            bar.seconds.map { String(format: "%.1fs", $0) },
        ].compactMap { $0 }.joined(separator: "   ")
    }
}

struct PictureFrame: View {
    var session: Session
    var job: Job

    @Environment(\.openWindow) private var openWindow
    /// Whether the mouse is over the picture, which then says that a click opens it.
    @State private var hovering = false

    var body: some View {
        Group {
            if let image = session.images[job.id] {
                Image(nsImage: image)
                    .resizable()
                    .scaledToFit()
                    .draggable(Image(nsImage: image))
                    .overlay { OpensOnClick(shown: hovering) }
                    .overlay(PointingHand())
                    .onHover { over in
                        withAnimation(.easeOut(duration: 0.15)) { hovering = over }
                    }
                    .onTapGesture { openWindow(id: "preview", value: job.id) }
                    .contextMenu {
                        Button("Copy") {
                            NSPasteboard.general.clearContents()
                            NSPasteboard.general.writeObjects([image])
                        }
                        Button("Copy parameters") {
                            NSPasteboard.general.clearContents()
                            NSPasteboard.general.setString(job.made?.parameters ?? "", forType: .string)
                        }
                    }
                    .help(["Click to open it in a window of its own", job.made?.parameters]
                        .compactMap { $0 }.joined(separator: "\n\n"))
            } else {
                ProgressView()
            }
        }
        .task(id: job.id) { await session.load(job) }
    }
}

/// Laid over a picture the mouse is on: the picture dimmed a little, and in its corner the sign
/// for opening it larger. It takes no clicks; the picture under it does.
struct OpensOnClick: View {
    var shown: Bool

    var body: some View {
        ZStack(alignment: .topTrailing) {
            Color.black.opacity(shown ? 0.12 : 0)
            Image(systemName: "arrow.up.left.and.arrow.down.right")
                .font(.system(size: 13, weight: .semibold))
                .foregroundStyle(.white)
                .padding(8)
                .background(.black.opacity(0.5), in: Circle())
                .padding(10)
                .opacity(shown ? 1 : 0)
                .scaleEffect(shown ? 1 : 0.85)
        }
        .allowsHitTesting(false)
    }
}

/// The window a picture opens in when it is clicked, opening at one size whatever the picture: the
/// picture fitted into it, and the record kept beside it, where there is one, in a column on its
/// right that the toolbar shows and hides. Esc closes it; one deleted while it is open leaves the
/// window saying so.
struct PreviewWindow: View {
    var session: Session
    var id: String

    @Environment(\.dismiss) private var dismiss
    @State private var record: String?
    @AppStorage("preview.showsRecord") private var showsRecord = true

    /// What it opens at: small enough for a 13-inch screen, title bar and all.
    static let size = CGSize(width: 1000, height: 700)

    init(session: Session, id: String) {
        self.session = session
        self.id = id
        // Read before the first frame, or the column comes in after it and the picture jumps.
        _record = State(initialValue: session.recordText(id))
    }

    private var job: Job? { session.jobs.first { $0.id == id } }

    var body: some View {
        HStack(spacing: 0) {
            ZStack {
                if let image = session.images[id] {
                    Image(nsImage: image)
                        .resizable()
                        .interpolation(.high)
                        .scaledToFit()
                        .draggable(Image(nsImage: image))
                        .contextMenu {
                            Button("Copy") {
                                NSPasteboard.general.clearContents()
                                NSPasteboard.general.writeObjects([image])
                            }
                        }
                } else if job == nil {
                    Text("This picture has been deleted.").foregroundStyle(.secondary).padding(40)
                } else {
                    ProgressView().padding(40)
                }
                Button("Close") { dismiss() }
                    .keyboardShortcut(.cancelAction)
                    .hidden()
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            if let record, showsRecord {
                Divider()
                RecordColumn(text: record).frame(width: 300)
            }
        }
        .frame(minWidth: 400, minHeight: 300)
        .toolbar {
            if record != nil {
                Button {
                    showsRecord.toggle()
                } label: {
                    Label(showsRecord ? "Hide Record" : "Show Record", systemImage: "sidebar.right")
                }
                .help(showsRecord ? "Hide how it was made" : "Show how it was made")
            }
        }
        .navigationTitle(job.map { Media.fileName(created: $0.finished ?? $0.created, kind: $0.kind) } ?? "Preview")
        // For a window the big frame did not open -- one restored with the app -- whose picture
        // has not been read.
        .task(id: id) { if let job { await session.load(job) } }
    }
}

/// A picture's record as it is on the disk, to read and copy from.
struct RecordColumn: View {
    var text: String

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack {
                Text("Record").font(.headline)
                Spacer()
                Button {
                    NSPasteboard.general.clearContents()
                    NSPasteboard.general.setString(text, forType: .string)
                } label: {
                    Image(systemName: "doc.on.doc")
                }
                .buttonStyle(.borderless)
                .help("Copy the record")
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 8)
            Divider()
            ScrollView {
                Text(text)
                    .font(.system(size: 13, design: .monospaced))
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .topLeading)
                    .padding(12)
            }
        }
        .background(.background)
    }
}

/// The pointing hand over what it is laid on, which says that a click does something. A cursor
/// rect, not a push on hover: nothing is left to pop when the view goes while the mouse is on it.
struct PointingHand: NSViewRepresentable {
    final class Area: NSView {
        override func resetCursorRects() { addCursorRect(bounds, cursor: .pointingHand) }
        override func hitTest(_ point: NSPoint) -> NSView? { nil }
    }

    func makeNSView(context: Context) -> Area { Area() }
    func updateNSView(_ view: Area, context: Context) { view.window?.invalidateCursorRects(for: view) }
}

struct Thumbnail: View {
    var session: Session
    var job: Job

    var body: some View {
        Group {
            if let image = session.images[job.id] {
                Image(nsImage: image).resizable().scaledToFill()
            } else {
                Color.secondary.opacity(0.15)
            }
        }
        .frame(width: 56, height: 56)
        .clipShape(RoundedRectangle(cornerRadius: 5))
        .task(id: job.id) { await session.load(job) }
    }
}

struct ClipFrame: View {
    var session: Session
    var job: Job

    var body: some View {
        VStack(spacing: 16) {
            if let bytes = session.outputs[job.id] {
                Player(bytes: bytes).id(job.id)
            } else {
                ProgressView()
            }
            Text(said(job)).font(.title3).multilineTextAlignment(.center).textSelection(.enabled)
            if let parameters = job.made?.parameters {
                Text(parameters).font(.caption).foregroundStyle(.secondary).textSelection(.enabled)
            }
        }
        .frame(maxWidth: 560)
        .task(id: job.id) { await session.load(job) }
    }
}

/// What a clip is called in the strip: its first words, or for a conversion, how it was made.
func said(_ job: Job) -> String {
    let made = job.made
    if job.kind == "conversion" {
        return "\(made?.convertStyle == true ? "Voice and style" : "Voice only"), \(made?.steps ?? 0) steps"
    }
    return made?.text ?? ""
}

struct ClipChip: View {
    var job: Job

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(said(job)).lineLimit(1)
            Text(clock(job.made?.length ?? 0)).font(.caption).foregroundStyle(.secondary)
        }
        .padding(8)
        .frame(width: 160, alignment: .leading)
        .background(Theme.card, in: RoundedRectangle(cornerRadius: 6))
    }
}

/// A length of sound as a player says it: `1:05`.
func clock(_ seconds: Double) -> String {
    guard seconds.isFinite, seconds > 0 else { return "0:00" }
    let whole = Int(seconds)
    return String(format: "%d:%02d", whole / 60, whole % 60)
}

/// A play button, a bar that can be dragged, and the time. Never plays by itself: a new clip
/// finishing, or a click on an old one in the strip, is not a reason to make a sound.
struct Player: View {
    var bytes: Data

    @State private var player: AVAudioPlayer?
    @State private var playing = false
    @State private var at = 0.0
    @State private var broken = false

    var body: some View {
        HStack(spacing: 10) {
            Button {
                toggle()
            } label: {
                Image(systemName: playing ? "pause.fill" : "play.fill").frame(width: 16)
            }
            .disabled(broken)

            Slider(
                value: Binding(get: { at }, set: { at = $0; player?.currentTime = $0 }),
                in: 0...max(player?.duration ?? 0, 0.01))

            Text(broken ? "cannot play this" : "\(clock(at)) / \(clock(player?.duration ?? 0))")
                .font(.caption).monospacedDigit().foregroundStyle(.secondary)
        }
        .task(id: bytes) {
            player?.stop()
            playing = false
            at = 0
            player = try? AVAudioPlayer(data: bytes)
            broken = player == nil
        }
        .task(id: playing) {
            // The bar follows the sound while it plays.
            while playing, !Task.isCancelled {
                at = player?.currentTime ?? 0
                if player?.isPlaying != true {
                    playing = false
                    at = player?.duration ?? 0
                }
                try? await Task.sleep(for: .milliseconds(33))
            }
        }
        .onDisappear { player?.stop() }
    }

    private func toggle() {
        guard let player else { return }
        if player.isPlaying {
            player.pause()
            playing = false
        } else {
            if player.currentTime >= player.duration - 0.01 { player.currentTime = 0 }
            player.play()
            playing = true
        }
    }
}
