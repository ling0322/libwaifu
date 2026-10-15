# libwaifu-gui, on Windows

The same screen as `waifu-app/swiftui/`, in WPF. Laid out from the design spec
(`libwaifu-gui-design.pdf`, sections 2 to 13) and from the screenshots in it: one window, three
columns, a fixed dark grey that follows neither the system theme nor the wallpaper.

It draws: the window calls `waifu.dll` through the C API in `docs/ffi.md`, in the app's own
process, and a model that is on the disk is read onto the card by the first job that needs it.
With no library beside it the same window runs on a stand-in engine instead, says so in its
title, and draws nothing.

## Build

This machine has Visual Studio 2026 with the C++ workload and no .NET SDK, so the project is
.NET Framework 4.8 and builds with MSBuild alone -- there is nothing to restore and nothing to
install. It is built as x64 because the library is: an AnyCPU program would be loaded as a
32-bit one and could not open it at all.

```powershell
& "C:\Program Files\Microsoft Visual Studio\18\Community\MSBuild\Current\Bin\MSBuild.exe" `
    waifu-app\wpf\Libwaifu.Gui\Libwaifu.Gui.csproj -p:Configuration=Release
waifu-app\wpf\Libwaifu.Gui\bin\Release\libwaifu-gui.exe
```

The library itself comes from this repository's own build:

```powershell
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DWITH_OPENMP=ON -DWITH_CUDA=ON `
    -DCMAKE_CUDA_ARCHITECTURES=86-real
cmake --build build --target waifu-modelmanager  # build\waifu.dll, build\waifu_modelmanager.dll
```

The program looks for `waifu.dll` beside itself, then under `%LIBWAIFU_LIB_DIR%`, then in this
repository's `build\`, and for `waifu_modelmanager.dll` beside whichever it found. Copying both
next to the exe is what an installed copy would look like.

`-m <model>` chooses a model and turns to the page of the task it is for, `-device <name>` picks
the device, and `-output <folder>` changes where results are written for that run only.

## What is in it

- **The window.** Sidebar, task, settings; both dividers drag, both side columns can be put away,
  and the title bar is the dark one (`DWMWA_USE_IMMERSIVE_DARK_MODE`, applied once the window is
  on screen -- asking before it is shown is accepted and does nothing). Per-monitor v2 DPI, so a
  size in the code is a point on the mac.
- **All four task pages.** txt2img, img2img, text2speech, speech2speech. The sidebar lists what
  `Tasks.Tabs` holds, which is `txt2img` alone, the same switch the mac has; the other three are
  reachable with Ctrl+Shift+1 to Ctrl+Shift+4 while they are not listed.
- **The Model card.** Grouped popup (Downloaded / Not downloaded / Choose a Manifest...), the
  device list Windows offers (`auto`, `cuda`, `cuda (offload)`, `vulkan`, `cpu`), and the footnote
  that says whether a model is chosen, downloaded or read onto a device -- and whether it is big
  enough to leave the card with nothing to draw into. Choosing one reads what it says about
  itself and fills the form in; it downloads nothing and loads nothing.
- **The settings**, each group as section 9 describes it: sizes taken down to the model's
  alignment, a slider and a box that are one knob, guidance and strength shown only for models
  that have them, and a seed that is text because "-1" is not a number anyone meant.
- **Running.** One job at a time and one waiting. The bar is indeterminate while a model is
  fetched and read and determinate while it draws; the button turns red and the settings go dead
  for both. Cancel stops after the step it is on, and says "stopped where it was".
- **What came out.** The picture or the clip, Save... / Reuse these settings / Delete, and the
  strip of everything ever made in that category, newest first. Results are written to
  `%APPDATA%\libwaifu\output` as `<id>.png|wav` plus `<id>.json`, with the key names the mac
  writes, so one machine's output folder opens on the other.
- **Model Manager.** The library's own folder, Show in Explorer, Change..., and a row per model
  with its size and the one button for it. Downloads run through the library, show its progress,
  and can be cancelled between files; deleting one asks first and then deletes the package.
- **This PC.** Processor, cores and threads, memory with a bar that turns red past ninety percent,
  the card with how much of its memory is in use, and the accelerators this build can reach. Read
  every two seconds on a thread of its own.
- **Shortcuts.** Ctrl+Enter generates, Ctrl+. and Esc cancel, Ctrl+S saves,
  Ctrl+Shift+O opens the output folder.

## The library behind it

`Engine/IEngine.cs` is the seam: `Begin(JobRequest)`, `Cancel()`, a `Progress` event carrying the
library's own phases, and a `Finished` event. Two things implement it -- `NativeEngine` over the C
API, and `MockEngine`, which waits out a fetch, a read and a count of steps and then writes a
picture made from the seed. The window is given whichever one `Library.Available` found, and
nothing above that line knows which it got.

Around the engine, everything the library can answer comes from it: `Engine/NativeModels.cs`
reads `waifu_modelmanager_catalog_json` and `_directory` from `waifu_modelmanager.dll`, and
`waifu_describe_json` and `waifu_machine_json` from `waifu.dll`, and runs downloads through
`waifu_modelmanager_fetch_async`. `Engine/Models.cs` is the one door
the view models knock on, and it is what decides between the library and the stand-in.

Three rules shape the interop, and they are the library's rather than this program's:

- **Every callback arrives on a thread of the library's.** Nothing touches the screen from one: a
  callback copies what it was handed -- strings, pixels, samples, all valid only until it returns
  -- and hands the copy to the dispatcher.
- **A callback must not call back into the library** (cancel and the free functions aside), so the
  next step of a job is always taken on the UI thread after the hand-off, never inside the
  completion that asked for it.
- **A delegate the library holds has to outlive the call.** Every one of them is a field; a
  collected delegate is a crash in somebody else's stack frame.

`Engine/Output.cs` writes what a job made. The engine hands over pixels and samples and keeps
nothing: the PNG, the WAV and the JSON beside them are this program's, and both engines go
through the same two calls, so a picture from either is filed the same way and the mac can read
it.

### What it was tested against

`anima:turbo`, already downloaded, on an RTX 3060 Laptop (6 GB): read onto the card, drawn at
512x512 in ten steps in six seconds, written to `%APPDATA%\libwaifu\output` with its JSON, and
shown in the main area and the strip. The same prompt at 1024x1024 ends with the library's own
`Aborted: out of memory (code 0x0102)` in the notice line, which is the window reporting a real
failure correctly rather than a bug in the wiring -- 5.5 GB of weights leave too little of a 6 GB
card for the picture. That is what the Model card's footnote now warns about before the button is
pressed, and what the `cuda (offload)` device is there for.

## Not done yet

- **The voice pages have never been run against a model**, because none is downloaded here. The
  calls are written the same way the picture ones are; what has been exercised end to end is
  txt2img.
- **Audio is read as WAV only.** .NET Framework has no decoder of its own and the ones Windows has
  are COM interfaces, so a dropped recording that is not a WAV says so in the box rather than
  failing further in. What this program writes is WAV, so what it made can be handed back to it.
  The thirty-second and five-minute limits are applied to what it does read.
- A picture cannot be dragged out of the main display (section 7 asks for it); Copy and Copy
  parameters are on its menu.
- There is no menu bar; the shortcuts above do what the File menu does on the mac.
- `waifu_machine_json` answers null for the processor and the memory on Windows today, so those
  lines are read off Windows here and the library's answer is laid over them -- which is where
  the video memory and the accelerator list come from.
- No manifests are shipped beside the program yet. `Library.cs` hands over a `manifests\` folder
  if it finds one; `cargo run -p waifu-tools --bin fetch_manifests -- <dir>` is what fills it.
