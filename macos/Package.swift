// swift-tools-version: 5.10

// The macOS app: native controls over libwaifu, which it links. WaifuKit is the Swift face of the
// C API in waifu-ffi (docs/ffi.md); CWaifu is that API's header, as a module Swift can import.
//
// libwaifu.dylib is CMake's to build, into build/ at the top of the repository: this links it
// from there, and finds it there at run time while the app runs unbundled. Inside Waifu.app it is
// in Contents/Frameworks, which the second rpath is for.

import Foundation
import PackageDescription

let repository = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
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
