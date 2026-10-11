#!/bin/sh
# Packs Waifu.app as Waifu.dmg: one window, in icon view at 128 pixels, with the app beside an
# Applications link to drag it onto.
#
#   macos/package-dmg.sh [path/to/Waifu.app]
#
# The app defaults to macos/build/Waifu.app, which build-app.sh makes. The image is written beside
# it, to macos/build/Waifu.dmg.
#
# The window's layout is a .DS_Store that dmg_layout.py writes with the ds_store package, installed
# the first time into a venv under macos/build, so nothing here needs Finder or a desktop session.

set -eu

here=$(cd "$(dirname "$0")" && pwd)
app=${1:-"$here/build/Waifu.app"}
if [ ! -d "$app" ]; then
    echo "no Waifu.app at $app: run macos/build-app.sh first, or name it" >&2
    exit 1
fi
codesign --verify --deep "$app"

venv="$here/build/dmg-venv"
if [ ! -x "$venv/bin/python" ]; then
    python3 -m venv "$venv"
    "$venv/bin/pip" install --quiet --disable-pip-version-check ds_store==1.3.1 mac_alias==2.2.2
fi

stage="$here/build/dmg-stage"
rm -rf "$stage"
mkdir -p "$stage"
ditto "$app" "$stage/Waifu.app"
ln -s /Applications "$stage/Applications"
"$venv/bin/python" -I "$here/dmg_layout.py" "$stage"

dmg="$here/build/Waifu.dmg"
rm -f "$dmg"
hdiutil create -volname Waifu -srcfolder "$stage" -fs HFS+ -format UDZO -imagekey zlib-level=9 \
    "$dmg" >/dev/null
rm -rf "$stage"
hdiutil verify -quiet "$dmg"
echo "$dmg"
