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

// Everything a job is asked for that is not the prompt, as a grouped form in the manner of System
// Settings: a label on the left and its control on the right, a heading over each group and what
// needs saying about it under it.

import SwiftUI
import UniformTypeIdentifiers

struct SettingsColumn: View {
    /// Which half of a job's settings: what it is of and the button that makes it, which sit over
    /// the preview in the middle, or everything else, which has the column on the right.
    enum Part { case prompts, settings }

    @Bindable var session: Session
    var part: Part

    var body: some View {
        Group {
            switch part {
            case .prompts:
                VStack(alignment: .leading, spacing: 16) {
                    what
                    // Closer to the prompts than the groups are to each other: it belongs to them.
                    GoBar(session: session)
                        .padding(.top, -6)
                }
                .padding(16)
            case .settings:
                ScrollView {
                    VStack(alignment: .leading, spacing: 12) {
                        ModelCard(session: session)
                        DeviceCard(session: session)
                        if session.model != nil {
                            Group {
                                switch session.tab {
                                case .txt2img, .img2img: pictureSettings
                                case .text2speech: speechSettings
                                case .speech2speech: conversionSettings
                                }
                            }
                            // Held while a job runs: what it was asked with is what it is
                            // making, and a box that moves under it says otherwise.
                            .disabled(session.canStop)
                        }
                    }
                    .padding(12)
                }
                // Tighter rows than the prompts' cards: this column is a list of settings.
                .environment(\.rowPadding, 5)
                .labeledContentStyle(SettingRowStyle())
            }
        }
        .labeledContentStyle(RowStyle())
        .textFieldStyle(.plain)
    }

    // MARK: - what to make

    /// What the run is of, first: the prompts, the text to read out, or the recording to convert.
    /// What the model and the device are is in the title bar and the Model card, and not said
    /// again here.
    @ViewBuilder private var what: some View {
        switch session.tab {
        case .txt2img, .img2img:
            if let chosen = session.model?.model {
                // Side by side: what to draw, and what to draw it away from.
                HStack(alignment: .top, spacing: 12) {
                    FormGroup {
                        PromptField(
                            text: $session.form.prompt,
                            placeholder: "What to draw: a list of tags, the most important first",
                            lines: 4...10)
                    } header: {
                        Header("Prompt")
                    }

                    // The negative prompt, which is what the second pass is given: not there at all
                    // for a model that runs one pass, which has nowhere to put it.
                    if chosen.guided {
                        FormGroup {
                            PromptField(
                                text: $session.form.negative, placeholder: "What to keep out: worst quality, bad hands",
                                lines: 4...10)
                        } header: {
                            Header("Away from")
                        }
                    }
                }
                // Made again when "Away from" comes or goes: a text field that grows to fit its
                // lines keeps wrapping them at the width it had, and the prompt's is not the same.
                .id(chosen.guided)

                if session.tab == .txt2img, chosen.readsPictures {
                    PromptPictures(session: session)
                        .padding(.top, -6)
                }
            }
        case .text2speech:
            FormGroup {
                PromptField(
                    text: $session.form.text,
                    placeholder: "The text to read out. Punctuation is where it pauses",
                    lines: 5...14)
            } header: {
                Header("What to say")
            } footer: {
                if let why = session.model?.voice?.notAVoiceBecause {
                    Label(why + ".", systemImage: "exclamationmark.triangle.fill")
                        .font(.caption).foregroundStyle(.orange)
                }
            }
        case .speech2speech:
            FormGroup {
                RecordingDrop(session: session, which: .source)
            } header: {
                Header("Recording to convert")
            }
        }
    }

    // MARK: - a picture

    @ViewBuilder private var pictureSettings: some View {
        if let chosen = session.model?.model {
            if session.tab == .img2img {
                FormGroup {
                    if chosen.noPictureBecause == nil { PictureDrop(session: session) }
                } header: {
                    Header("Picture to draw from")
                } footer: {
                    if let why = chosen.noPictureBecause { Footnote("This model cannot start from a picture: \(why).") }
                }
            }

            FormGroup {
                SizeRows(form: $session.form, sizes: chosen.sizes ?? [], alignment: max(1, chosen.alignment ?? 16))
            } header: {
                Header("Size")
            }

            FormGroup {
                SliderRow(
                    label: "Steps", value: intBinding(\.steps), range: 1...150, slider: 1...80,
                    step: 1, decimals: 0)
                if chosen.guided {
                    SliderRow(label: "CFG scale", value: $session.form.guidance, range: 1...30, step: 0.1, decimals: 1)
                }
                if session.tab == .img2img, chosen.noPictureBecause == nil {
                    SliderRow(
                        label: "Strength", value: $session.form.strength, range: 0...1,
                        step: 0.05, decimals: 2)
                }
                SeedRow(seed: $session.form.seed)
            } header: {
                Header("Sampling")
            }
        }
    }

    // MARK: - a reading

    @ViewBuilder private var speechSettings: some View {
        if let voice = session.model?.voice {
            FormGroup {
                if voice.noLikenessBecause == nil {
                    RecordingDrop(session: session, which: .recording)
                }
            } header: {
                Header("Recording to sound like")
            } footer: {
                if let why = voice.noLikenessBecause {
                    Footnote("This voice cannot be handed a recording: \(why).")
                }
            }

            if let styles = voice.styles, !styles.isEmpty {
                FormGroup {
                    StylePicker(styles: styles, style: $session.form.style)
                    LabeledContent("Instruction") {
                        TextField("Instruction", text: $session.form.style, prompt: Text("none"))
                            .labelsHidden()
                            .multilineTextAlignment(.trailing)
                            .frame(maxWidth: .infinity)
                            .fieldBox()
                    }
                } header: {
                    Header("Style")
                }
            }

            FormGroup {
                SliderRow(label: "Speed", value: $session.form.speed, range: 0.25...4, step: 0.05, decimals: 2)
                SliderRow(label: "Temperature", value: $session.form.temperature, range: 0...2, step: 0.05, decimals: 2)
                SeedRow(seed: $session.form.seed)
            } header: {
                Header("Reading")
            }
        }
    }

    // MARK: - a conversion

    @ViewBuilder private var conversionSettings: some View {
        if let converter = session.model?.converter {
            FormGroup {
                RecordingDrop(session: session, which: .recording)
            } header: {
                Header("Voice to convert it to")
            }
            FormGroup {
                SliderRow(label: "Steps", value: intBinding(\.conversionSteps), range: 1...100, step: 1, decimals: 0)
                if converter.convertsStyle == true {
                    LabeledContent("Convert the style too") {
                        Toggle("Convert the style too", isOn: $session.form.convertStyle)
                            .labelsHidden()
                            .toggleStyle(.switch)
                    }
                    .help("Off, only whose voice it is changes. On, the accent and pacing change as well.")
                }
                SeedRow(seed: $session.form.seed)
            } header: {
                Header("Conversion")
            }
        }
    }

    private func intBinding(_ path: WritableKeyPath<JobForm, Int>) -> Binding<Double> {
        Binding(
            get: { Double(session.form[keyPath: path]) },
            set: { session.form[keyPath: path] = Int($0.rounded()) })
    }
}

/// A group's heading.
struct Header: View {
    var title: String
    init(_ title: String) { self.title = title }

    var body: some View {
        Text(title).font(.headline)
    }
}

// MARK: - the groups

/// A group of rows: a heading, the rows on a rounded fill with a line between each, and a note
/// under it. What the system's grouped form draws, drawn here because its fill is a wash over the
/// background rather than a colour of its own, and on a dark one the groups all but vanish.
///
/// Spelt like `Section` -- the rows, then `header:` and `footer:` -- so that it reads like one.
/// A group with no rows draws no fill, only its header and footer.
struct FormGroup<Content: View, Header: View, Footer: View>: View {
    @ViewBuilder var content: Content
    @ViewBuilder var header: Header
    @ViewBuilder var footer: Footer

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            header.padding(.leading, 2)
            _VariadicView.Tree(Rows()) { content }
            footer.padding(.horizontal, 2)
        }
    }
}

extension FormGroup where Footer == EmptyView {
    init(@ViewBuilder content: () -> Content, @ViewBuilder header: () -> Header) {
        self.init(content: content, header: header, footer: { EmptyView() })
    }
}

extension FormGroup where Header == EmptyView {
    init(@ViewBuilder content: () -> Content, @ViewBuilder footer: () -> Footer) {
        self.init(content: content, header: { EmptyView() }, footer: footer)
    }
}

/// How much room a row of a group has above and below it.
private struct RowPaddingKey: EnvironmentKey {
    static let defaultValue: CGFloat = 8
}

extension EnvironmentValues {
    var rowPadding: CGFloat {
        get { self[RowPaddingKey.self] }
        set { self[RowPaddingKey.self] = newValue }
    }
}

/// The rows of a group, each padded the way a form pads one and with a line under every one but
/// the last.
private struct Rows: _VariadicView_UnaryViewRoot {
    @Environment(\.rowPadding) private var rowPadding

    func body(children: _VariadicView.Children) -> some View {
        if !children.isEmpty {
            VStack(spacing: 0) {
                ForEach(children) { child in
                    child
                        .padding(.horizontal, 12)
                        .padding(.vertical, rowPadding)
                        .frame(maxWidth: .infinity, minHeight: 18 + 2 * rowPadding, alignment: .leading)
                    if child.id != children.last?.id {
                        Divider().padding(.leading, 12)
                    }
                }
            }
            .background(Theme.card, in: RoundedRectangle(cornerRadius: 8, style: .continuous))
            .overlay(
                RoundedRectangle(cornerRadius: 8, style: .continuous)
                    .strokeBorder(Color.white.opacity(0.07)))
        }
    }
}

extension View {
    /// A box around a small field, in the grey of a pop-up button, so that a number that can be
    /// typed over looks like a control rather than like a label. As tall and as round as the
    /// pop-up beside it: 24 points and 6 round from macOS 26, 20 and 5 before.
    func fieldBox() -> some View {
        let new = if #available(macOS 26, *) { true } else { false }
        return padding(.horizontal, 6)
            .frame(height: new ? 24 : 20)
            .background(Theme.control, in: RoundedRectangle(cornerRadius: new ? 6 : 5, style: .continuous))
    }

    /// A pop-up as wide as the control column, as every other control of a row is. Before macOS 26
    /// a pop-up fills the room it is given; from 26 it is as wide as what it says unless asked.
    @ViewBuilder func fillsRow() -> some View {
        if #available(macOS 26, *) { buttonSizing(.flexible) } else { self }
    }
}

/// A row's label on the left and what it holds on the right, as a form lays one out.
struct RowStyle: LabeledContentStyle {
    func makeBody(configuration: Configuration) -> some View {
        HStack(spacing: 12) {
            configuration.label
            Spacer(minLength: 8)
            configuration.content
        }
    }
}

/// How wide the labels of the settings column are, so that what they label starts in a line.
private let labelWidth: CGFloat = 86

/// A row of the settings column: the label in a column of its own width, and the control filling
/// the rest, so that every control of the column starts and ends where the others do. A label too
/// long for its column pushes its control over rather than being cut.
struct SettingRowStyle: LabeledContentStyle {
    func makeBody(configuration: Configuration) -> some View {
        HStack(spacing: 8) {
            configuration.label
                .lineLimit(1)
                .frame(minWidth: labelWidth, alignment: .leading)
                .fixedSize()
            configuration.content
                .frame(maxWidth: .infinity, alignment: .trailing)
        }
    }
}

/// A control column cut in two equal halves with a narrow gap between: the width and the height
/// either side of the "×".
struct Halves<Left: View, Gap: View, Right: View>: View {
    @ViewBuilder var left: Left
    @ViewBuilder var gap: Gap
    @ViewBuilder var right: Right

    var body: some View {
        HStack(spacing: 6) {
            left.frame(maxWidth: .infinity)
            gap.frame(width: 10)
            right.frame(maxWidth: .infinity)
        }
    }
}

// MARK: - the rows

/// Several lines of text that fill a row of the form: the field the system draws, with the hint
/// in it until something is typed.
struct PromptField: View {
    @Binding var text: String
    var placeholder: String
    var lines: ClosedRange<Int>

    var body: some View {
        TextField("", text: $text, prompt: Text(placeholder), axis: .vertical)
            .labelsHidden()
            .lineLimit(lines)
    }
}

/// The grey words under a group.
struct Footnote: View {
    var said: String
    init(_ said: String) { self.said = said }

    var body: some View {
        Text(said).font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
    }
}

/// A label, a slider and the number beside it, on one line: two ways of saying one value. The
/// number has a narrow box of one width at the end of the row, and the slider all the rest of the
/// control column, so that the sliders of a group start and end in a line.
struct SliderRow: View {
    var label: String
    @Binding var value: Double
    var range: ClosedRange<Double>
    /// Narrower than what the box takes, where the useful part of the range is.
    var slider: ClosedRange<Double>?
    var step: Double
    var decimals: Int
    /// What the value means, where the label cannot say it: behind a ? beside the label.
    var explanation: String? = nil

    @State private var explaining = false

    var body: some View {
        let sliding = slider ?? range
        LabeledContent {
            HStack(spacing: 8) {
                // Rounded here rather than given a `step`, which draws a tick for every one of them.
                Slider(
                    value: Binding(
                        get: { min(max(value, sliding.lowerBound), sliding.upperBound) },
                        set: { value = ($0 / step).rounded() * step }),
                    in: sliding
                ) {
                    Text(label)
                }
                // The row says what it is: without this the slider keeps room for its own label,
                // and its track starts short of the boxes above it.
                .labelsHidden()
                .frame(maxWidth: .infinity)
                TextField(
                    label,
                    value: Binding(get: { value }, set: { value = min(max($0, range.lowerBound), range.upperBound) }),
                    format: .number.precision(.fractionLength(decimals))
                )
                .labelsHidden()
                .multilineTextAlignment(.trailing)
                .frame(width: 36)
                .fieldBox()
            }
        } label: {
            HStack(spacing: 4) {
                Text(label)
                if let explanation {
                    Button {
                        explaining.toggle()
                    } label: {
                        Image(systemName: "questionmark.circle")
                    }
                    .buttonStyle(.borderless)
                    .help("What \(label) means")
                    .popover(isPresented: $explaining, arrowEdge: .bottom) {
                        // As a paragraph: the label it hangs from is one line, and would make
                        // this one too.
                        Text(explanation)
                            .lineLimit(nil)
                            .multilineTextAlignment(.leading)
                            .fixedSize(horizontal: false, vertical: true)
                            .frame(width: 280, alignment: .leading)
                            .padding(14)
                    }
                }
            }
        }
    }
}

/// The size: the model's own list, and the two boxes for somebody who knows they are leaving it.
struct SizeRows: View {
    @Binding var form: JobForm
    var sizes: [[Int]]
    /// What both sides have to be a multiple of, for the model that is loaded.
    var alignment: Int

    var body: some View {
        LabeledContent("Preset") {
            Picker("Preset", selection: preset) {
                if !isPreset { Text("Custom").tag("custom") }
                ForEach(sizes, id: \.self) { size in
                    Text(verbatim: "\(size[0]) × \(size[1])").tag("\(size[0])x\(size[1])")
                }
            }
            .labelsHidden()
            .fillsRow()
        }
        LabeledContent("W × H") {
            Halves {
                TextField("Width", value: side(\.width), format: .number.grouping(.never))
                    .labelsHidden().multilineTextAlignment(.trailing).frame(maxWidth: .infinity).fieldBox()
            } gap: {
                Text("×").foregroundStyle(.secondary)
            } right: {
                TextField("Height", value: side(\.height), format: .number.grouping(.never))
                    .labelsHidden().multilineTextAlignment(.trailing).frame(maxWidth: .infinity).fieldBox()
            }
        }
    }

    private var isPreset: Bool { sizes.contains { $0 == [form.width, form.height] } }

    private var preset: Binding<String> {
        Binding(
            get: { isPreset ? "\(form.width)x\(form.height)" : "custom" },
            set: { tag in
                let parts = tag.split(separator: "x").compactMap { Int($0) }
                if parts.count == 2 { (form.width, form.height) = (parts[0], parts[1]) }
            })
    }

    /// A side, held to what the model will take: a multiple of its alignment from 64 to 2048.
    private func side(_ path: WritableKeyPath<JobForm, Int>) -> Binding<Int> {
        Binding(
            get: { form[keyPath: path] },
            set: { form[keyPath: path] = min(2048 / alignment * alignment, max(64, $0 / alignment * alignment)) })
    }
}

/// The seed, and the die that rolls one: inside the seed's box, so that the box fills the control
/// column as every other one does.
struct SeedRow: View {
    @Binding var seed: String

    var body: some View {
        LabeledContent("Seed") {
            HStack(spacing: 6) {
                Button {
                    seed = String(Int.random(in: 1...65535))
                } label: {
                    Image(systemName: "dice")
                }
                .buttonStyle(.borderless)
                .help("A random seed")
                TextField("Seed", text: $seed)
                    .labelsHidden()
                    .multilineTextAlignment(.trailing)
                    .frame(maxWidth: .infinity)
            }
            .fieldBox()
        }
    }
}

/// The list of styles a voice was taught, which writes into the instruction box below it.
struct StylePicker: View {
    var styles: [Voice.Style]
    @Binding var style: String

    var body: some View {
        let typed = style.trimmingCharacters(in: .whitespaces)
        let chosen = styles.first { $0.instruction == typed }
        LabeledContent("Say it") {
            Picker(
                "Say it",
                selection: Binding(
                    get: { typed.isEmpty ? "" : chosen?.instruction ?? "custom" },
                    set: { if $0 != "custom" { style = $0 } })
            ) {
                Text("The voice's own way").tag("")
                if !typed.isEmpty && chosen == nil { Text("As written below").tag("custom") }
                ForEach(styles, id: \.self) { one in Text(one.label).tag(one.instruction) }
            }
            .labelsHidden()
            .fillsRow()
        }
    }
}

// MARK: - what a job starts from

/// A dashed box to drop a file on or click, which is what both drops below are drawn in.
struct DropBox<Content: View>: View {
    var over: Bool
    var height: CGFloat
    @ViewBuilder var content: Content

    var body: some View {
        ZStack {
            RoundedRectangle(cornerRadius: 8)
                .fill(over ? Color.accentColor.opacity(0.12) : Color.primary.opacity(0.03))
            RoundedRectangle(cornerRadius: 8)
                .strokeBorder(style: StrokeStyle(lineWidth: 1, dash: [5, 4]))
                .foregroundStyle(over ? Color.accentColor : Color.secondary.opacity(0.5))
            content
        }
        .frame(maxWidth: .infinity, minHeight: height, maxHeight: height)
        .contentShape(Rectangle())
    }
}

/// img2img: the picture a run starts from instead of noise.
struct PictureDrop: View {
    var session: Session
    @State private var over = false

    var body: some View {
        DropBox(over: over, height: session.picture == nil ? 90 : 180) {
            if let image = session.picture?.image {
                Image(nsImage: image).resizable().scaledToFit().padding(6)
            } else {
                Label("Drop or choose a picture", systemImage: "photo.badge.plus")
                    .foregroundStyle(.secondary)
            }
        }
        .onTapGesture {
            if let url = Media.choose([.image]) { session.holdPicture(at: url) }
        }
        .onDrop(of: [.fileURL, .image], isTargeted: $over) { providers in
            take(providers)
        }

        if session.picture != nil {
            Button("Remove the picture", role: .destructive) { session.letGo(.picture) }
        }
    }

    private func take(_ providers: [NSItemProvider]) -> Bool {
        guard let provider = providers.first else { return false }
        takePicture(
            from: provider, file: { session.holdPicture(at: $0) }, bytes: { session.holdPicture($0) })
        return true
    }
}

/// Reads a dropped picture: a file where it is one, which a record can name, or else its bytes
/// as a PNG.
private func takePicture(
    from provider: NSItemProvider, file: @escaping @MainActor (URL) -> Void,
    bytes: @escaping @MainActor (Data) -> Void
) {
    if provider.canLoadObject(ofClass: URL.self) {
        _ = provider.loadObject(ofClass: URL.self) { url, _ in
            if let url { Task { @MainActor in file(url) } }
        }
        return
    }
    provider.loadDataRepresentation(forTypeIdentifier: UTType.image.identifier) { data, _ in
        guard let data, let image = NSImage(data: data), let png = Media.png(image) else { return }
        Task { @MainActor in bytes(png) }
    }
}

/// The pictures the prompt names, under it: each with its name in a tooltip, a click that writes
/// that name into the prompt again, and a cross that takes it away. And what keeps
/// the prompt and the pictures from going together, where something does.
struct PromptPictures: View {
    var session: Session

    var body: some View {
        if !session.named.isEmpty || session.namedTrouble != nil {
            VStack(alignment: .leading, spacing: 6) {
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 8) {
                        ForEach(session.named) { picture in
                            chip(picture)
                        }
                    }
                }
                if let trouble = session.namedTrouble {
                    Label(trouble + ".", systemImage: "exclamationmark.triangle.fill")
                        .font(.caption).foregroundStyle(.orange)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
        }
    }

    /// The picture alone: its name is in the prompt, and in the tooltip.
    private func chip(_ picture: NamedPicture) -> some View {
        Group {
            if let image = picture.held.image {
                Image(nsImage: image).resizable().scaledToFill()
            } else {
                Color.primary.opacity(0.05)
            }
        }
        .frame(width: Thumbnail.side, height: Thumbnail.side)
        .clipShape(RoundedRectangle(cornerRadius: 5))
        .overlay(alignment: .topTrailing) {
            Button {
                session.letGoNamed(picture.name)
            } label: {
                Image(systemName: "xmark.circle.fill")
                    .symbolRenderingMode(.palette)
                    .foregroundStyle(.white, .black.opacity(0.6))
            }
            .buttonStyle(.borderless)
            // Inside the corner rather than over it: the row scrolls, and a scroll view cuts off
            // whatever sticks out of it, so a cross over the edge would need room the row then
            // has above it and not below.
            .padding(3)
            .help("Remove the picture")
        }
        .onTapGesture { session.writeName(picture.name) }
        .help("<|\(picture.name)|>: click to write it into the prompt again")
    }
}

/// A recording to hold: to sound like, to convert, or the voice to convert it to.
struct RecordingDrop: View {
    var session: Session
    var which: Session.Which
    @State private var over = false
    @State private var reading = false

    private var held: Held? { which == .source ? session.source : session.recording }

    var body: some View {
        if let held {
            HStack {
                Player(bytes: held.bytes)
                Button {
                    session.letGo(which)
                } label: {
                    Image(systemName: "xmark.circle.fill").foregroundStyle(.secondary)
                }
                .buttonStyle(.borderless)
                .help("Remove the recording")
            }
        }
        DropBox(over: over, height: 56) {
            Label(
                reading
                    ? "Reading it…"
                    : held == nil
                        ? "Drop or choose a recording"
                        : "Drop or choose another",
                systemImage: "waveform.badge.plus"
            )
            .foregroundStyle(.secondary)
        }
        .onTapGesture {
            if let url = Media.choose([.audio]) { hold(url) }
        }
        .onDrop(of: [.fileURL], isTargeted: $over) { providers in
            guard let provider = providers.first else { return false }
            _ = provider.loadObject(ofClass: URL.self) { url, _ in
                if let url { Task { @MainActor in hold(url) } }
            }
            return true
        }
    }

    private func hold(_ url: URL) {
        reading = true
        Task {
            await session.holdAudio(which, at: url)
            reading = false
        }
    }
}
