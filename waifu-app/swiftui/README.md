# libwaifu for macOS

A native SwiftUI app on libwaifu. It needs macOS 14.

The app does none of the drawing itself, and runs no `waifu` program: it calls the library in
process, through `libwaifu.dylib` and `libwaifu_modelmanager.dylib` and their C APIs
(`docs/ffi.md`), wrapped for Swift in `WaifuKit` (`Sources/WaifuKit`).

## Build

```bash
# the libraries, from the top of the repository: build/libwaifu.dylib,
# build/libwaifu_modelmanager.dylib and their headers in build/include, for macOS 14 and later
# (docs/metal.md)
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DWITH_METAL=ON
cmake --build build -j$(sysctl -n hw.ncpu)

# the app, with both libraries in its Frameworks, every published model's manifest in its
# Resources (fetched now, a few KB each), and its icon compiled in
waifu-app/swiftui/build-app.sh
open waifu-app/swiftui/build/libwaifu-gui.app

# the disk image, waifu-app/swiftui/build/libwaifu-gui.dmg: the app beside an Applications link, at 128 pixels
waifu-app/swiftui/package-dmg.sh
```

`package-dmg.sh` lays the window out with `dmg_layout.py`, which writes the image's `.DS_Store`
with the `ds_store` package; the script installs it the first time into `waifu-app/swiftui/build/dmg-venv`.
The release workflow runs both scripts and attaches it to the release as
`libwaifu-gui-<version>-macos-arm64.dmg`.

Built as above, the app is signed for this machine only (`codesign --sign -`). To hand it out, sign
both with a Developer ID and have Apple notarize the image:

```bash
# once: an app-specific password from appleid.apple.com, kept in the keychain
xcrun notarytool store-credentials waifu-notary --apple-id <apple id> --team-id <team id>

export WAIFU_SIGN_IDENTITY="Developer ID Application: <name> (<team id>)"
waifu-app/swiftui/build-app.sh
WAIFU_NOTARY_PROFILE=waifu-notary waifu-app/swiftui/package-dmg.sh
```

The release workflow does the same from the secrets `MACOS_CERT_P12` (the certificate and its key,
exported as .p12, in base64), `MACOS_CERT_PASSWORD`, `MACOS_SIGN_IDENTITY`, and an App Store
Connect API key: `ASC_KEY_P8` (in base64), `ASC_KEY_ID` and `ASC_ISSUER_ID`.

While working on it, `swift run --package-path waifu-app/swiftui Waifu` runs it without a bundle, against
`build/libwaifu.dylib` and `build/libwaifu_modelmanager.dylib`; `swift test --package-path waifu-app/swiftui`
checks WaifuKit against them.

The icon is `Resources/AppIcon.icon`, the Icon Composer format, which `build-app.sh` compiles with
`actool` (Xcode 26 or later). Its one layer is `Assets/picture.png`, 1024 pixels square.

## Use

The sidebar lists the tasks -- txt2img for now -- and the Model Manager.

On a task, the Model card at the top of the right column chooses the model and the device. Nothing
is loaded from there: **Generate** fetches the model if it has to and reads it onto the device,
unloading the one before, with the download and the read on the bar. The settings are held while a
job runs.

What is made is kept in the app's own folder, `~/Library/Application Support/libwaifu/output`, which
**File > Show Output Folder** (⇧⌘O) opens in Finder.

The Model Manager lists every published model, with how much of it is on this Mac: a download with
its progress, and a delete.

From a terminal, `Waifu -m sdxl:base -device metal` chooses that model as the window opens.

⌘↩ generates, ⌘. cancels, ⌘S saves what is on show.
