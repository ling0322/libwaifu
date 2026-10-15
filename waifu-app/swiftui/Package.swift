// swift-tools-version: 5.10

// The macOS app: native controls over libwaifu and libwaifu_modelmanager, which it links. WaifuKit
// is the Swift face of the C APIs in waifu and waifu-modelmanager (docs/ffi.md); CWaifu is their
// headers, as a module Swift can import.
//
// Both dylibs are CMake's to build, into build/ at the top of the repository: this links them from
// there, and finds them there at run time while the app runs unbundled. Inside libwaifu-gui.app
// they are in Contents/Frameworks, which the second rpath is for.

import Foundation
import PackageDescription

// This file is waifu-app/swiftui/Package.swift: the repository is three levels up from it.
let repository = URL(fileURLWithPath: #filePath)
    .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
let build = repository.appendingPathComponent("build").path

let package = Package(
    name: "Waifu",
    platforms: [.macOS(.v14)],
    targets: [
        .target(
            name: "CWaifu",
            path: "Sources/CWaifu",
            linkerSettings: [
                .linkedLibrary("waifu"),
                .linkedLibrary("waifu_modelmanager"),
                .unsafeFlags(["-L", build]),
            ]
        ),
        .target(
            name: "WaifuKit",
            dependencies: ["CWaifu"],
            path: "Sources/WaifuKit"
        ),
        .executableTarget(
            name: "Waifu",
            dependencies: ["WaifuKit"],
            path: "Sources/Waifu",
            linkerSettings: [
                .unsafeFlags([
                    "-Xlinker", "-rpath", "-Xlinker", "@executable_path/../Frameworks",
                    "-Xlinker", "-rpath", "-Xlinker", build,
                ])
            ]
        ),
        .testTarget(
            name: "WaifuKitTests",
            dependencies: ["WaifuKit"],
            path: "Tests/WaifuKitTests",
            linkerSettings: [
                .unsafeFlags(["-Xlinker", "-rpath", "-Xlinker", build])
            ]
        ),
    ]
)
