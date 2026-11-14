// swift-tools-version: 5.10

// The macOS app: native controls over libwaifu, libwaifu_modelmanager and libwaifu2d, which it
// links. WaifuKit is the Swift face of the C APIs in waifu (docs/ffi-abi6.md), waifu-modelmanager
// (docs/ffi.md) and waifu2d (docs/waifu2d.md); CWaifu is their headers, as a module Swift can import.
//
// The dylibs are CMake's to build, into build/ at the top of the repository: this links them from
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
                .linkedLibrary("waifu2d"),
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
        .testTarget(
            name: "WaifuTests",
            dependencies: ["Waifu"],
            path: "Tests/WaifuTests",
            linkerSettings: [
                .unsafeFlags(["-Xlinker", "-rpath", "-Xlinker", build])
            ]
        ),
    ]
)
