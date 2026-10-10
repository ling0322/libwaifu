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

import AppKit
import SwiftUI

@main
struct WaifuApp: App {
    @NSApplicationDelegateAdaptor private var delegate: AppDelegate
    @State private var session = Session(folder: WaifuApp.output)

    var body: some Scene {
        Window("libwaifu", id: "main") {
            MainView(session: session)
                .onAppear(perform: startFromTheCommandLine)        }
        .defaultSize(width: 1280, height: 860)
        .commands {
            CommandGroup(replacing: .newItem) {
                Button("Show Output Folder") {
                    let folder = session.folder
                    try? FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
                    NSWorkspace.shared.activateFileViewerSelecting([folder])
                }
                .keyboardShortcut("o", modifiers: [.command, .shift])
            }
        }
    }

    /// Where what is made is kept: the app's own folder, not one to choose. `-output <folder>` on
    /// the command line puts it somewhere else for that run, and is not kept.
    private static var output: URL {
        let given = UserDefaults.standard.volatileDomain(forName: UserDefaults.argumentDomain)["output"] as? String
        guard let given else { return Launcher.defaultOutput }
        return URL(fileURLWithPath: (given as NSString).expandingTildeInPath)
    }

    /// `Waifu -m sdxl:base [-device metal]` chooses that model, and shows its task, as the window
    /// opens -- the way `waifu webui -m` does. The first job loads it.
    private func startFromTheCommandLine() {
        let arguments = Array(CommandLine.arguments.dropFirst())
        func value(_ names: String...) -> String? {
            guard let at = arguments.firstIndex(where: names.contains), at + 1 < arguments.count else {
                return nil
            }
            return arguments[at + 1]
        }

        if let device = value("-device", "--device") { session.device = device }
        if let model = value("-m", "--m") { session.choose(model) }
    }
}

final class AppDelegate: NSObject, NSApplicationDelegate {
    func applicationDidFinishLaunching(_ notification: Notification) {
        // Run as `swift run`, there is no bundle to say this is an app with a Dock icon and
        // windows rather than a tool in a terminal.
        NSApp.setActivationPolicy(.regular)
        // Dark whatever the system is set to: a picture is judged against a dark surround, and
        // the save and open panels and the alerts follow the app rather than the system.
        NSApp.appearance = NSAppearance(named: .darkAqua)
        NSApp.activate(ignoringOtherApps: true)
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool { true }
}
