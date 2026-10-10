#!/bin/sh
# Builds Waifu.app: the SwiftUI app, with libwaifu.dylib in its Frameworks, where the app's rpath
# looks for it. The executable is called libwaifu, as the app is.
#
#   macos/build-app.sh [path/to/libwaifu.dylib]
#
# The library defaults to build/libwaifu.dylib, which the CMake build at the top of the repository
# makes (its waifu-ffi target). The app is written to macos/build/Waifu.app.

set -eu

here=$(cd "$(dirname "$0")" && pwd)
library=${1:-"$here/../build/libwaifu.dylib"}
if [ ! -f "$library" ]; then
    echo "no libwaifu.dylib at $library: build it with cmake first, or name it" >&2
    exit 1
fi

swift build --package-path "$here" -c release
binary=$(swift build --package-path "$here" -c release --show-bin-path)/Waifu

# Every published model's manifest, a few KB each and none of their weights, fetched now so that the
# app knows what each model suggests the moment it is chosen, with nothing to download first.
manifests="$here/build/manifests"
rm -rf "$manifests"
LIBWAIFU_LIB_DIR="$(cd "$here/.." && pwd)/build" cargo run --quiet --release \
    --manifest-path "$here/../waifu/Cargo.toml" --features hub --example fetch_manifests -- "$manifests"

app="$here/build/Waifu.app"
rm -rf "$app"
mkdir -p "$app/Contents/MacOS" "$app/Contents/Frameworks" "$app/Contents/Resources"
cp "$binary" "$app/Contents/MacOS/libwaifu"
cp "$library" "$app/Contents/Frameworks/libwaifu.dylib"
cp -R "$manifests" "$app/Contents/Resources/manifests"
# The icon, in the format macOS 26 draws as its own rather than on a grey plate: an Icon Composer
# .icon, compiled into Assets.car, with an AppIcon.icns beside it for older systems.
xcrun actool "$here/Resources/AppIcon.icon" --compile "$app/Contents/Resources" \
    --platform macosx --minimum-deployment-target 14.0 --app-icon AppIcon \
    --output-partial-info-plist "$here/build/icon-partial.plist" >/dev/null

cat > "$app/Contents/Info.plist" <<'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleExecutable</key>
    <string>libwaifu</string>
    <key>CFBundleIconFile</key>
    <string>AppIcon</string>
    <key>CFBundleIconName</key>
    <string>AppIcon</string>
    <key>CFBundleIdentifier</key>
    <string>io.github.ling0322.libwaifu</string>
    <key>CFBundleName</key>
    <string>libwaifu</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleShortVersionString</key>
    <string>0.1</string>
    <key>LSMinimumSystemVersion</key>
    <string>14.0</string>
    <key>NSHighResolutionCapable</key>
    <true/>
</dict>
</plist>
EOF

# Signed for this machine only, which is what lets it run here without a developer identity.
codesign --force --deep --sign - "$app"
echo "$app"
