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

// The window once the program is answering: the page's layout in native controls. The tabs and
// the machine down the side; what to make across the top; its settings on the left and what came
// of it on the right.

import SwiftUI

/// The colours the window is drawn in: fixed neutral greys rather than the system's, which in dark
/// mode are tinted by the desktop picture -- a red wallpaper made the whole window red -- and
/// rather than the see-through sidebar, which shows the desktop itself.
enum Theme {
    static let window = Color(white: 0.155)
    static let sidebar = Color(white: 0.125)
    static let card = Color(white: 0.22)
    static let field = Color(white: 0.11)
    /// What a pop-up button is filled with in dark mode, for the fields that sit beside one.
    static let control = Color(white: 0.416)
}

struct MainView: View {
    @Bindable var session: Session
    @State private var showSettings = true

    var body: some View {
        NavigationSplitView {
            Sidebar(session: session)
                .navigationSplitViewColumnWidth(min: 200, ideal: 230, max: 300)
        } detail: {
            if session.showingModels {
                ModelManagerView(session: session)
            } else {
                task
            }
        }
        .toolbarBackground(Theme.window, for: .windowToolbar)
        .toolbarBackground(.visible, for: .windowToolbar)
        .navigationTitle(session.showingModels ? "Model Manager" : session.tab.rawValue)
        .navigationSubtitle(session.showingModels ? "" : subtitle)
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

    /// A task: what to make, the button that makes it, and what came of it in the middle, and
    /// everything else it is asked for on the right.
    private var task: some View {
            VStack(spacing: 0) {
                if session.model != nil {
                    SettingsColumn(session: session, part: .prompts)
                    Divider()
                    OutputPane(session: session)
                } else {
                    NoModel(tab: session.tab) { showSettings = true }
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
                    .inspectorColumnWidth(min: 300, ideal: 340, max: 420)
            }
    }

    private var subtitle: String {
        guard let model = session.model else { return "no model chosen" }
        let name =
            model.model?.fullName ?? model.voice?.fullName ?? model.converter?.fullName
            ?? model.model?.name ?? model.voice?.name ?? model.converter?.name ?? ""
        return [name, session.device].compactMap { $0 }.filter { !$0.isEmpty }
            .joined(separator: " on ")
    }
}

/// The middle of a task no model has been chosen for: where to choose one.
struct NoModel: View {
    var tab: Tab
    var showCard: () -> Void

    var body: some View {
        VStack(spacing: 10) {
            Image(systemName: "shippingbox").font(.system(size: 36)).foregroundStyle(.secondary)
            Text("No model for \(tab.rawValue)").font(.title3)
            Button("Choose one in the Model card on the right", action: showCard)
                .buttonStyle(.link)
        }
        .padding(24)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}

// MARK: - the side

/// What the sidebar picks between: a task, or the Model Manager page.
enum Page: Hashable {
    case task(Tab)
    case models
}

struct Sidebar: View {
    @Bindable var session: Session

    private var page: Binding<Page?> {
        Binding(
            get: { session.showingModels ? .models : .task(session.tab) },
            set: { picked in
                switch picked {
                case .task(let tab):
                    session.tab = tab
                    session.showingModels = false
                case .models:
                    session.showingModels = true
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
        .safeAreaInset(edge: .bottom) {
            if let machine = session.machine {
                MachineView(machine: machine)
                    .padding(12)
            }
        }
    }
}

extension Tab {
    var color: Color {
        switch self {
        case .txt2img: .blue
        case .img2img: .purple
        case .text2speech: .pink
        case .speech2speech: .orange
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

/// What this is running on: the processor, how much memory is left, the card. What is *left* is
/// what answers why a run that worked yesterday now aborts.
struct MachineView: View {
    var machine: Machine

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("This Mac").font(.headline)

            if let cpu = machine.cpu {
                part("CPU") {
                    Text(cpu).lineLimit(1).help(cpu)
                    let counted = [machine.cores.map { "\($0) cores" }, machine.threads.map { "\($0) threads" }]
                        .compactMap { $0 }
                    if !counted.isEmpty { Text(counted.joined(separator: " · ")).foregroundStyle(.secondary) }
                }
            }
            if let total = machine.memory?.total {
                part("Memory") { Meter(used: machine.memory?.used, total: total) }
            }
            if let gpu = machine.gpu {
                part("GPU") {
                    Text(gpu.name).lineLimit(1).help(gpu.name)
                    if gpu.unified == true {
                        Text("memory is shared with the CPU").foregroundStyle(.secondary)
                    } else if let vram = machine.vram, let total = vram.total {
                        Meter(used: vram.used, total: total)
                    } else if let why = machine.whyNoVram {
                        Text(why).foregroundStyle(.secondary)
                    }
                }
            }
            part("Accelerators") {
                let accelerators = machine.accelerators ?? []
                Text(accelerators.isEmpty ? "none -- runs go to the processor" : accelerators.joined(separator: " · "))
                    .foregroundStyle(.secondary)
            }
        }
        .font(.caption)
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private func part<Content: View>(_ what: String, @ViewBuilder _ content: () -> Content) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(what.uppercased()).font(.caption2).foregroundStyle(.tertiary)
            content()
        }
    }
}

struct Meter: View {
    var used: Double?
    var total: Double

    var body: some View {
        if let used, total > 0 {
            VStack(alignment: .leading, spacing: 3) {
                Text("\(room(used)) of \(room(total))").foregroundStyle(.secondary)
                ProgressView(value: min(1, max(0, used / total)))
                    .controlSize(.mini)
                    .tint(used / total > 0.9 ? .red : .accentColor)
            }
        } else {
            Text(room(total)).foregroundStyle(.secondary)
        }
    }
}

/// How much room something takes, in the unit that says it in two or three digits.
func room(_ bytes: Double) -> String {
    if bytes >= 1e9 { return String(format: "%.1f GB", bytes / 1e9) }
    if bytes >= 1e6 { return String(format: "%.0f MB", bytes / 1e6) }
    return String(format: "%.0f kB", bytes / 1e3)
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

    private var goWord: String {
        switch session.tab {
        case .txt2img, .img2img: "Generate"
        case .text2speech: "Speak"
        case .speech2speech: "Convert"
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
