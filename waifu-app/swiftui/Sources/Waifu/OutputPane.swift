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

    private var pictures: Bool { session.showsPictures }
    private var layers: Bool { session.tab.kind == "layers" }

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
                    if layers {
                        LayersFrame(session: session, job: job).id(job.id)
                    } else if pictures {
                        PictureFrame(session: session, job: job)
                    } else {
                        ClipFrame(session: session, job: job)
                    }
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
        case "layers": "Nothing split into layers yet."
        default: "Nothing drawn yet."
        }
    }

    private func actions(_ job: Job) -> some View {
        HStack {
            Button(layers ? "Save PSD…" : "Save…") { session.save(job) }
                .keyboardShortcut("s", modifiers: .command)
                .disabled(session.outputs[job.id] == nil)
            if layers {
                Button("Save Depth PSD…") { session.saveDepth(job) }
                    .help("The same document with each layer's depth in place of its colour")
                Button("Show in Finder") { session.reveal(job) }
                Button("Open in waifu2d") { session.openInRig(job) }
                    .help("Make these layers move")
            }
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
            .frame(height: pictures ? Thumbnail.side + 6 : 56)
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

/// img2layers: the layers stacked on their canvas, on a checkerboard so that what is transparent
/// shows, and beside them their list, nearest first as an editor lists them: an eye that hides one,
/// and a click that shows it alone.
struct LayersFrame: View {
    var session: Session
    var job: Job

    /// Each layer's picture, by its file, once read.
    @State private var read: [String: NSImage] = [:]
    @State private var hidden: Set<String> = []
    /// The one layer on show by itself, where one was clicked.
    @State private var alone: String?
    /// The layers as Split by depth saves them, and their pictures, while it is on.
    @State private var split: (layers: [KeptLayer], images: [String: NSImage])?

    /// Whether the layers shown are split by depth: changing them is for the model's own.
    private var splitting: Bool { session.form.splitByDepth && split != nil }
    private var layers: [KeptLayer] { splitting ? split!.layers : job.made?.layers ?? [] }
    private var pictures: [String: NSImage] { splitting ? split!.images : read }
    private var canvas: CGFloat { CGFloat(max(1, job.made?.canvas ?? 1280)) }

    var body: some View {
        HStack(spacing: 12) {
            stack.frame(maxWidth: .infinity, maxHeight: .infinity)
            list.frame(width: 240)
        }
        .task(id: job.id) {
            await session.load(job)
            read = await session.layerImages(job)
        }
        .task(id: "\(job.id) \(session.form.splitByDepth)") {
            alone = nil
            hidden = []
            split = nil
            guard session.form.splitByDepth else { return }
            await session.load(job)
            split = await session.splitLayers(job)
        }
    }

    /// An eye closed hides a layer whatever else: the one shown alone too, so that its eye says
    /// what the canvas shows.
    private func shows(_ layer: KeptLayer) -> Bool {
        if hidden.contains(layer.file) { return false }
        if let alone { return layer.file == alone }
        return true
    }

    /// Bottom first, each where it is on the canvas, the canvas fitted into the room there is.
    private var stack: some View {
        GeometryReader { space in
            let scale = min(space.size.width, space.size.height) / canvas
            ZStack(alignment: .topLeading) {
                Checkerboard()
                ForEach(layers, id: \.file) { layer in
                    if shows(layer), let image = pictures[layer.file] {
                        Image(nsImage: image)
                            .resizable()
                            .interpolation(.high)
                            .frame(width: CGFloat(layer.width) * scale, height: CGFloat(layer.height) * scale)
                            .offset(x: CGFloat(layer.left) * scale, y: CGFloat(layer.top) * scale)
                    }
                }
            }
            .frame(width: canvas * scale, height: canvas * scale, alignment: .topLeading)
            .clipped()
            .frame(width: space.size.width, height: space.size.height)
        }
        .overlay {
            if (pictures.isEmpty && !layers.isEmpty) || (session.form.splitByDepth && split == nil) { ProgressView() }
        }
        .help(job.made?.parameters ?? "")
    }

    private var list: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text("\(layers.count) layers").font(.headline)
                Spacer()
                if alone != nil || !hidden.isEmpty {
                    Button("Show All") {
                        alone = nil
                        hidden = []
                    }
                    .buttonStyle(.link)
                }
                if !splitting, layers.contains(where: \.edited) {
                    Button("Reset") { session.resetLayers(job) }
                        .buttonStyle(.link)
                        .help("Put back every name and depth the model gave, and its order")
                }
            }
            if splitting {
                Text("Split by depth, as it is saved. Turn it off to change names and depths.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            ScrollView {
                LazyVStack(spacing: 2) {
                    ForEach(layers.reversed(), id: \.file) { layer in
                        LayerRow(
                            layer: layer, image: pictures[layer.file], alone: alone == layer.file,
                            hidden: hidden.contains(layer.file), editable: !splitting,
                            toggleHidden: {
                                if hidden.contains(layer.file) { hidden.remove(layer.file) } else { hidden.insert(layer.file) }
                            },
                            toggleAlone: {
                                // Shown alone is shown: a layer whose eye was closed opens it,
                                // rather than leaving the canvas empty.
                                if alone != layer.file { hidden.remove(layer.file) }
                                alone = alone == layer.file ? nil : layer.file
                            },
                            edit: { name, depth in session.editLayer(job, file: layer.file, name: name, depth: depth) })
                    }
                }
            }
        }
        .padding(8)
        .background(Theme.card, in: RoundedRectangle(cornerRadius: 8, style: .continuous))
    }
}

/// One layer in the list: its eye, its picture, its name and depth, and a pencil that opens a
/// small window to change them in. A click anywhere else on it shows it alone.
struct LayerRow: View {
    var layer: KeptLayer
    var image: NSImage?
    var alone: Bool
    var hidden: Bool
    /// Whether its name and depth can be changed: not while the list is split by depth.
    var editable = true
    var toggleHidden: () -> Void
    var toggleAlone: () -> Void
    var edit: (_ name: String?, _ depth: Float?) -> Void

    @State private var editing = false

    var body: some View {
        HStack(spacing: 8) {
            Button(action: toggleHidden) {
                Image(systemName: hidden ? "eye.slash" : "eye").frame(width: 18)
            }
            .buttonStyle(.borderless)
            .foregroundStyle(Color.secondary.opacity(hidden ? 0.5 : 1))
            .help(hidden ? "Show this layer" : "Hide this layer")

            ZStack {
                Checkerboard()
                if let image {
                    Image(nsImage: image).resizable().scaledToFit()
                }
            }
            .frame(width: 32, height: 32)
            .clipShape(RoundedRectangle(cornerRadius: 4))

            VStack(alignment: .leading, spacing: 1) {
                Text(layer.name).lineLimit(1)
                HStack(spacing: 4) {
                    Text(String(format: "depth %.2f", layer.depth))
                        .font(.caption).foregroundStyle(.secondary).monospacedDigit()
                    if layer.edited {
                        Text("edited").font(.caption2).foregroundStyle(.orange)
                    }
                }
            }
            .opacity(hidden ? 0.5 : 1)
            Spacer(minLength: 0)

            if editable {
                Button {
                    editing = true
                } label: {
                    // Heavier than the system draws it: at its own weight it is a hairline beside the eye.
                    Image(systemName: "pencil")
                        .font(.system(size: 14, weight: .semibold))
                        .frame(width: 18)
                }
                .buttonStyle(.borderless)
                .foregroundStyle(Color.primary.opacity(0.75))
                .help("Change its name or depth")
                .popover(isPresented: $editing, arrowEdge: .leading) {
                    LayerEditor(layer: layer) { name, depth in
                        edit(name, depth)
                        editing = false
                    } cancel: {
                        editing = false
                    }
                }
            }
        }
        .padding(.horizontal, 6)
        .padding(.vertical, 4)
        .background(alone ? Color.accentColor.opacity(0.25) : .clear, in: RoundedRectangle(cornerRadius: 5))
        .contentShape(Rectangle())
        .onTapGesture(perform: toggleAlone)
        .help(alone ? "Click to show every layer again" : "Click to show this layer alone")
        .contextMenu {
            if editable { Button("Edit…") { editing = true } }
            Button(alone ? "Show Every Layer" : "Show Alone", action: toggleAlone)
            if let image {
                Button("Copy") {
                    NSPasteboard.general.clearContents()
                    NSPasteboard.general.writeObjects([image])
                }
            }
        }
    }
}

/// The small window a layer is changed in: its name and its depth, kept on Save and dropped on
/// Cancel. What the model said is shown where either has been changed.
struct LayerEditor: View {
    var layer: KeptLayer
    var save: (_ name: String, _ depth: Float) -> Void
    var cancel: () -> Void

    @State private var name: String
    /// As typed: a number field keeps what is typed only once it is left, which a click on Save
    /// does not do.
    @State private var typed: String

    init(layer: KeptLayer, save: @escaping (String, Float) -> Void, cancel: @escaping () -> Void) {
        self.layer = layer
        self.save = save
        self.cancel = cancel
        _name = State(initialValue: layer.name)
        _typed = State(initialValue: String(format: "%.2f", layer.depth))
    }

    /// The depth typed, where it is a number from 0 to 1.
    private var depth: Float? {
        Float(typed.trimmingCharacters(in: .whitespaces)).flatMap { (0...1).contains($0) ? $0 : nil }
    }

    private var valid: Bool { !name.isBlank && depth != nil }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Edit Layer").font(.headline)
            Grid(alignment: .leading, horizontalSpacing: 10, verticalSpacing: 8) {
                GridRow {
                    Text("Name").gridColumnAlignment(.trailing)
                    TextField("Name", text: $name)
                        .textFieldStyle(.roundedBorder)
                        .frame(width: 180)
                }
                if let model = layer.modelName {
                    GridRow {
                        Color.clear.frame(width: 1, height: 1)
                        Text("The model called it \(model).").font(.caption).foregroundStyle(.secondary)
                    }
                }
                GridRow {
                    Text("Depth")
                    TextField("Depth", text: $typed)
                        .textFieldStyle(.roundedBorder)
                        .monospacedDigit()
                        .frame(width: 70)
                }
                GridRow {
                    Color.clear.frame(width: 1, height: 1)
                    Text(
                        "0 is nearest, 1 farthest: a layer is drawn over every one farther than it."
                            + (layer.modelDepth.map { String(format: " The model said %.2f.", $0) } ?? "")
                    )
                    .font(.caption)
                    .foregroundStyle(depth != nil ? Color.secondary : Color.red)
                    .frame(width: 180, alignment: .leading)
                    .fixedSize(horizontal: false, vertical: true)
                }
            }
            HStack {
                Spacer()
                Button("Cancel", action: cancel)
                    .keyboardShortcut(.cancelAction)
                Button("Save") { if let depth { save(name, depth) } }
                    .keyboardShortcut(.defaultAction)
                    .disabled(!valid)
            }
        }
        .padding(16)
    }
}

/// Grey squares in two shades, which is how an editor shows what is transparent.
struct Checkerboard: View {
    var side: CGFloat = 8

    var body: some View {
        Canvas { context, size in
            context.fill(Path(CGRect(origin: .zero, size: size)), with: .color(Color(white: 0.30)))
            var light = Path()
            for row in 0..<Int((size.height / side).rounded(.up)) {
                for column in 0..<Int((size.width / side).rounded(.up)) where (row + column) % 2 == 0 {
                    light.addRect(CGRect(x: CGFloat(column) * side, y: CGFloat(row) * side, width: side, height: side))
                }
            }
            context.fill(light, with: .color(Color(white: 0.36)))
        }
    }
}

struct Thumbnail: View {
    var session: Session
    var job: Job

    /// The side of a picture's thumbnail, and of a picture the prompt names, which is shown the
    /// same way.
    static let side: CGFloat = 48

    var body: some View {
        Group {
            if let image = session.images[job.id] {
                Image(nsImage: image).resizable().scaledToFill()
            } else {
                Color.secondary.opacity(0.15)
            }
        }
        .frame(width: Self.side, height: Self.side)
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
