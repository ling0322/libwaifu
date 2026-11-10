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

// The window once the program is answering: the page's layout in native controls. The tabs down
// the side; what to make across the top; its settings on the left and what came of it on the
// right.

import SwiftUI

/// The colours the window is drawn in: fixed neutral greys rather than the system's, which in dark
/// mode are tinted by the desktop picture -- a red wallpaper made the whole window red -- and
/// rather than the see-through sidebar, which shows the desktop itself.
enum Theme {
    static let window = Color(white: 0.155)
    static let sidebar = Color(white: 0.125)
    static let card = Color(white: 0.22)
    static let field = Color(white: 0.11)
    /// What a pop-up button is filled with in dark mode, for the fields that sit beside one: on a
    /// card, from macOS 26, a darker grey than the one it had before.
    static var control: Color {
        if #available(macOS 26, *) { Color(white: 0.305) } else { Color(white: 0.416) }
    }

    /// How far under its toolbar a column's first heading starts, given what it would be: from
    /// macOS 26 the sidebar runs up to the top of the window and its "Tasks" sits just under the
    /// window's buttons, while the columns beside it start under their toolbars -- so they start
    /// all but at once, which puts their first headings on its line. Before 26 every column starts
    /// under the toolbar, and `regular` lines them up already.
    static func top(_ regular: CGFloat, from26: CGFloat) -> CGFloat {
        if #available(macOS 26, *) { from26 } else { regular }
    }
}

struct MainView: View {
    @Bindable var session: Session
    @State private var showSettings = true

    var body: some View {
        NavigationSplitView {
            Sidebar(session: session)
                .navigationSplitViewColumnWidth(min: 200, ideal: 300, max: 420)
        } detail: {
            if session.showingModels {
                ModelManagerView(session: session)
            } else if session.showingRig {
                rigPage
            } else {
                task
            }
        }
        .toolbarFill()
        .navigationTitle(session.showingModels ? "Model Manager" : session.showingRig ? "waifu2d" : session.tab.rawValue)
        .navigationSubtitle(
            session.showingModels ? "" : session.showingRig ? session.rig.file?.lastPathComponent ?? "no drawing" : subtitle)
        .toolbar {
            if !session.showingModels {
                ToolbarItem(placement: .primaryAction) {
                    Button {
                        showSettings.toggle()
                    } label: {
                        Label("Settings", systemImage: "sidebar.right")
                    }
                    .help("Show or hide the settings")
                }
            }
        }
    }

    /// The waifu2d page: the drawing blinking in the middle, and how it blinks on the right.
    private var rigPage: some View {
        RigPage(stage: session.rig)
            .frame(minWidth: 460, maxWidth: .infinity, maxHeight: .infinity)
            .background(Theme.window)
            .inspector(isPresented: $showSettings) {
                RigSettings(stage: session.rig)
                    .frame(maxHeight: .infinity)
                    .background(Theme.sidebar)
                    .inspectorColumnWidth(min: 300, ideal: 300, max: 420)
            }
    }

    /// A task: what to make, the button that makes it, and what came of it in the middle, and
    /// everything else it is asked for on the right.
    private var task: some View {
            VStack(spacing: 0) {
                if session.model != nil {
                    SettingsColumn(session: session, part: .prompts)
                    Divider()
                    OutputPane(session: session)
                } else {
                    NoModel(session: session)
                }
            }
            .frame(minWidth: 460, maxWidth: .infinity, maxHeight: .infinity)
            .background(Theme.window)
            // And everything else it is asked for, in the inspector: a column the system sizes and
            // lets be dragged, which an `HSplitView` did not do reliably.
            .inspector(isPresented: $showSettings) {
                SettingsColumn(session: session, part: .settings)
                    .frame(maxHeight: .infinity)
                    .background(Theme.sidebar)
                    .inspectorColumnWidth(min: 300, ideal: 300, max: 420)
            }
    }

    private var subtitle: String {
        guard let model = session.model else { return "no model chosen" }
        let name =
            model.model?.fullName ?? model.voice?.fullName ?? model.converter?.fullName
            ?? model.layerer?.fullName ?? model.transcriber?.fullName ?? model.model?.name
            ?? model.voice?.name ?? model.converter?.name ?? model.layerer?.name
            ?? model.transcriber?.name ?? ""
        return [name, session.device].compactMap { $0 }.filter { !$0.isEmpty }
            .joined(separator: " on ")
    }
}

/// The middle of a task whose model cannot be described yet: every task has one, so this is the
/// moment its manifest is being fetched, or why it could not be.
struct NoModel: View {
    var session: Session

    var body: some View {
        VStack(spacing: 10) {
            if session.fetchingManifests, let name = session.chosenHere {
                ProgressView()
                Text("Reading what \(name) is…").foregroundStyle(.secondary)
            } else {
                Image(systemName: "exclamationmark.triangle").font(.system(size: 36)).foregroundStyle(.secondary)
                Text(session.notDescribed[session.tab.kind] ?? "No model for \(session.tab.rawValue)")
                    .multilineTextAlignment(.center)
                    .foregroundStyle(.secondary)
                    .textSelection(.enabled)
            }
        }
        .padding(24)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}

// MARK: - the side

/// What the sidebar picks between: a task, the waifu2d page, or the Model Manager page.
enum Page: Hashable {
    case task(Tab)
    case waifu2d
    case models
}

struct Sidebar: View {
    @Bindable var session: Session

    private var page: Binding<Page?> {
        Binding(
            get: { session.showingModels ? .models : session.showingRig ? .waifu2d : .task(session.tab) },
            set: { picked in
                switch picked {
                case .task(let tab):
                    session.tab = tab
                    session.showingModels = false
                    session.showingRig = false
                case .waifu2d:
                    session.showingRig = true
                    session.showingModels = false
                case .models:
                    session.showingModels = true
                    session.showingRig = false
                case nil:
                    break
                }
            })
    }

    var body: some View {
        List(selection: page) {
            Section("Tasks") {
                ForEach(session.tabs) { tab in
                    Label {
                        Text(tab.rawValue)
                    } icon: {
                        SettingsIcon(symbol: tab.symbol, color: tab.color)
                    }
                    .tag(Page.task(tab))
                }
                Label {
                    Text("waifu2d")
                } icon: {
                    Waifu2dIcon()
                }
                .tag(Page.waifu2d)
            }
            Section("Models") {
                Label {
                    Text("Model Manager")
                } icon: {
                    SettingsIcon(symbol: "shippingbox", color: .gray)
                }
                .tag(Page.models)
            }
        }
        .scrollContentBackground(.hidden)
        .background(Theme.sidebar)
    }
}

extension View {
    /// The toolbar in the window's grey before macOS 26. From 26 the sidebar is a panel that runs
    /// up to the top of the window with the window's buttons in it, as Mail's does, and a toolbar
    /// of its own colour would cut the top off it; nor is it the system's, which the desktop
    /// picture tints: there it is not drawn, and the columns' own greys show through.
    @ViewBuilder fileprivate func toolbarFill() -> some View {
        if #available(macOS 26, *) {
            toolbarBackgroundVisibility(.hidden, for: .windowToolbar)
        } else {
            toolbarBackground(Theme.window, for: .windowToolbar)
                .toolbarBackground(.visible, for: .windowToolbar)
        }
    }
}

extension Tab {
    var color: Color {
        switch self {
        case .text2img: .blue
        case .img2img: .purple
        case .img2layers: .teal
        case .text2speech: .green
        case .speech2speech: .orange
        case .speech2text: .indigo
        }
    }
}

/// A symbol on a small coloured square, the way System Settings draws the rows down its side.
struct SettingsIcon: View {
    var symbol: String
    var color: Color

    var body: some View {
        Image(systemName: symbol)
            .font(.system(size: 11, weight: .semibold))
            .foregroundStyle(.white)
            .frame(width: 20, height: 20)
            .background(color.gradient, in: RoundedRectangle(cornerRadius: 5, style: .continuous))
    }
}

/// waifu2d's square: a girl drawn here, because SF Symbols has nobody like her.
struct Waifu2dIcon: View {
    var body: some View {
        WaifuGlyph()
            .fill(.white)
            .frame(width: 16, height: 16)
            .frame(width: 20, height: 20)
            .background(Color.pink.gradient, in: RoundedRectangle(cornerRadius: 5, style: .continuous))
    }
}

/// A chibi girl, her twin tails tied high and hanging to her feet, big eyes and a smile, filled: at
/// this size her features close up in outline. The path is the Windows app's (Theme/Icons.xaml, IconWaifu) in its 24x24 box, scaled to
/// fit whatever frame it is given.
struct WaifuGlyph: Shape {
    static let data =
        "M3.4,21.6 C1.4,18.8 0.4,15 0.6,11.4 C0.8,7.8 1.8,5 3.6,3.6 C3.8,2.4 4.6,1.4 5.7,1.4 C6.6,1.4 7.2,2 7.3,2.8 C8.6,0.9 10.2,0.3 12,0.3 C13.8,0.3 15.4,0.9 16.7,2.8 C16.8,2 17.4,1.4 18.3,1.4 C19.4,1.4 20.2,2.4 20.4,3.6 C22.2,5 23.2,7.8 23.4,11.4 C23.6,15 22.6,18.8 20.6,21.6 C20.6,19 20.4,16 19.8,13.4 C19.4,11.6 18.8,10.2 18,9.4 C18.4,11.6 18.2,13.8 17.4,15.6 C17,15.2 16.6,14.9 16.2,14.8 C16.4,13 16.4,11.4 16.2,9.8 C14.2,9.6 12.8,8.4 12,6.8 C11.2,8.4 9.8,9.6 7.8,9.8 C7.6,11.4 7.6,13 7.8,14.8 C7.4,14.9 7,15.2 6.6,15.6 C5.8,13.8 5.6,11.6 6,9.4 C5.2,10.2 4.6,11.6 4.2,13.4 C3.6,16 3.4,19 3.4,21.6 Z " +
        "M9.8,11.05 C10.41,11.05 10.9,11.79 10.9,12.7 C10.9,13.61 10.41,14.35 9.8,14.35 C9.19,14.35 8.7,13.61 8.7,12.7 C8.7,11.79 9.19,11.05 9.8,11.05 Z " +
        "M14.2,11.05 C14.81,11.05 15.3,11.79 15.3,12.7 C15.3,13.61 14.81,14.35 14.2,14.35 C13.59,14.35 13.1,13.61 13.1,12.7 C13.1,11.79 13.59,11.05 14.2,11.05 Z " +
        "M10.4,15.3 C11.2,16.1 12.8,16.1 13.6,15.3 C13.2,16.8 12.6,17.2 12,17.2 C11.4,17.2 10.8,16.8 10.4,15.3 Z"

    /// The path in its 24x24 box. It has only absolute moves, cubics and closes.
    static let drawing: Path = {
        var path = Path()
        var command: Character = "M"
        var numbers: [CGFloat] = []
        for token in data.split(whereSeparator: { $0 == " " || $0 == "," }) {
            var number = token
            if let first = token.first, first.isLetter {
                command = first
                number = token.dropFirst()
                if command == "Z" {
                    path.closeSubpath()
                }
            }
            if let value = Double(number) {
                numbers.append(CGFloat(value))
            }
            if command == "M", numbers.count == 2 {
                path.move(to: CGPoint(x: numbers[0], y: numbers[1]))
                numbers.removeAll()
            } else if command == "C", numbers.count == 6 {
                path.addCurve(
                    to: CGPoint(x: numbers[4], y: numbers[5]),
                    control1: CGPoint(x: numbers[0], y: numbers[1]),
                    control2: CGPoint(x: numbers[2], y: numbers[3]))
                numbers.removeAll()
            }
        }
        return path
    }()

    func path(in rect: CGRect) -> Path {
        let bounds = Self.drawing.boundingRect
        let scale = min(rect.width / bounds.width, rect.height / bounds.height)
        let fit = CGAffineTransform(translationX: rect.midX, y: rect.midY)
            .scaledBy(x: scale, y: scale)
            .translatedBy(x: -bounds.midX, y: -bounds.midY)
        return Self.drawing.applying(fit)
    }
}

// MARK: - the middle

/// The one button: it makes something, and while that is being made it stops it.
struct GoBar: View {
    var session: Session

    var body: some View {
        if session.canStop {
            Button {
                session.stop()
            } label: {
                Text("Cancel").frame(maxWidth: .infinity)
            }
            .buttonStyle(BigButton(fill: .red))
            .keyboardShortcut(".", modifiers: .command)
            .help("Stop after the step it is on, or take the waiting job out of line (⌘.)")
        } else {
            HStack(spacing: 8) {
                // A picture for the prompt to read, for a model that reads any. In the weight of
                // an ordinary button: it is not what the bar is for.
                if session.tab == .text2img, session.model?.model?.readsPictures == true {
                    Button {
                        session.chooseNamed()
                    } label: {
                        Label("Add Picture", systemImage: "photo.badge.plus")
                            .font(.system(size: 13, weight: .regular))
                            .padding(.horizontal, 14)
                            .fixedSize()
                    }
                    .buttonStyle(BigButton(fill: Theme.control))
                    .disabled(!session.takesAnotherPicture)
                    .help("Add a picture for the prompt to read")
                }

                Button {
                    session.go()
                } label: {
                    Text(goWord).frame(maxWidth: .infinity)
                }
                .buttonStyle(BigButton(fill: .accentColor))
                .keyboardShortcut(.return, modifiers: .command)
                .disabled(!session.canGo)
                .help("⌘↩")
            }
        }
    }

    private var goWord: String {
        switch session.tab {
        case .text2img, .img2img: "Generate"
        case .img2layers: "Split into Layers"
        case .text2speech: "Speak"
        case .speech2speech: "Convert"
        case .speech2text: "Transcribe"
        }
    }
}

/// A button taller than the system will draw one: the two the whole column is for. A push button
/// on macOS keeps its height whatever control size it is given, so this one is drawn here.
struct BigButton: ButtonStyle {
    var fill: Color
    @Environment(\.isEnabled) private var isEnabled

    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .font(.system(size: 13, weight: .semibold))
            .foregroundStyle(.white)
            .frame(height: 32)
            .background(fill, in: RoundedRectangle(cornerRadius: 8, style: .continuous))
            .overlay(
                RoundedRectangle(cornerRadius: 8, style: .continuous)
                    .fill(Color.black.opacity(configuration.isPressed ? 0.2 : 0)))
            .opacity(isEnabled ? 1 : 0.45)
            .contentShape(Rectangle())
    }
}
