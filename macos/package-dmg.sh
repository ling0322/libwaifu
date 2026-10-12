#!/bin/sh
# Packs libwaifu-gui.app as libwaifu-gui.dmg: one window, in icon view at 128 pixels, with the app
# beside an Applications link to drag it onto.
#
#   macos/package-dmg.sh [path/to/libwaifu-gui.app]
#
# The app defaults to macos/build/libwaifu-gui.app, which build-app.sh makes. The image is written
# beside it, to macos/build/libwaifu-gui.dmg.
#
# The window's layout is a .DS_Store that dmg_layout.py writes with the ds_store package, installed
# the first time into a venv under macos/build, so nothing here needs Finder or a desktop session.
#
# To hand it out: WAIFU_SIGN_IDENTITY, the Developer ID build-app.sh signed the app with, signs the
# image too, and WAIFU_NOTARY_PROFILE, a profile `xcrun notarytool store-credentials` saved to the
# keychain, then has Apple notarize it and staples the ticket to it, so that it opens offline.

set -eu

here=$(cd "$(dirname "$0")" && pwd)
app=${1:-"$here/build/libwaifu-gui.app"}
if [ ! -d "$app" ]; then
    echo "no libwaifu-gui.app at $app: run macos/build-app.sh first, or name it" >&2
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
ditto "$app" "$stage/libwaifu-gui.app"
ln -s /Applications "$stage/Applications"
"$venv/bin/python" -I "$here/dmg_layout.py" "$stage"

dmg="$here/build/libwaifu-gui.dmg"
rm -f "$dmg"
hdiutil create -volname libwaifu-gui -srcfolder "$stage" -fs HFS+ -format UDZO \
    -imagekey zlib-level=9 "$dmg" >/dev/null
rm -rf "$stage"
hdiutil verify -quiet "$dmg"

if [ -n "${WAIFU_SIGN_IDENTITY:-}" ]; then
    codesign --force --timestamp --sign "$WAIFU_SIGN_IDENTITY" "$dmg"
fi
if [ -n "${WAIFU_NOTARY_PROFILE:-}" ]; then
    if [ -z "${WAIFU_SIGN_IDENTITY:-}" ]; then
        echo "WAIFU_NOTARY_PROFILE needs WAIFU_SIGN_IDENTITY: Apple notarizes only Developer ID" >&2
        exit 1
    fi
    # Minutes as a rule; an account's first submissions can take an hour or two.
    xcrun notarytool submit "$dmg" --keychain-profile "$WAIFU_NOTARY_PROFILE" --wait
    xcrun stapler staple "$dmg"
    spctl --assess --type open --context context:primary-signature "$dmg"
fi
echo "$dmg"
