# libwaifu for macOS

A native SwiftUI app on libwaifu. It needs macOS 14.

The app does none of the drawing itself, and runs no `waifu` program: it calls the library in
process, through `libwaifu.dylib` and its C API (`docs/ffi.md`), wrapped for Swift in `WaifuKit`
(`Sources/WaifuKit`).

## Build

```bash
# the library, from the top of the repository: build/libwaifu.dylib and build/include/waifu.h
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DWITH_OPENMP=ON -DWITH_MLX=ON
cmake --build build -j$(sysctl -n hw.ncpu)

# the app, with build/libwaifu.dylib in its Frameworks, every published model's manifest in its
# Resources (fetched now, a few KB each), and its icon compiled in
macos/build-app.sh
open macos/build/Waifu.app
```

While working on it, `swift run --package-path macos Waifu` runs it without a bundle, against
`build/libwaifu.dylib`; `swift test --package-path macos` checks WaifuKit against it.

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
