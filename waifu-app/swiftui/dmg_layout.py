# Writes the .DS_Store that makes Finder open libwaifu-gui.dmg as one window of two large icons: the
# app on the left, the Applications link on the right, to drag the one onto the other.
#
#   python3 dmg_layout.py <folder the image is made from>
#
# Written directly rather than by asking Finder through AppleScript, which a CI runner has no
# session for and which a desktop asks permission for the first time.

import sys

from ds_store import DSStore

ICON_SIZE = 128
WINDOW = (560, 380)

with DSStore.open(sys.argv[1] + "/.DS_Store", "w+") as store:
    store["."]["bwsp"] = {
        "ShowStatusBar": False,
        "ShowToolbar": False,
        "ShowTabView": False,
        "ShowPathbar": False,
        "ShowSidebar": False,
        "WindowBounds": "{{200, 120}, {%d, %d}}" % WINDOW,
    }
    store["."]["icvp"] = {
        "viewOptionsVersion": 1,
        "iconSize": float(ICON_SIZE),
        "textSize": 14.0,
        "arrangeBy": "none",
        "gridSpacing": 100.0,
        "gridOffsetX": 0.0,
        "gridOffsetY": 0.0,
        "labelOnBottom": True,
        "showItemInfo": False,
        "showIconPreview": True,
        "backgroundType": 0,
        "backgroundColorRed": 1.0,
        "backgroundColorGreen": 1.0,
        "backgroundColorBlue": 1.0,
    }
    store["."]["vSrn"] = ("long", 1)
    # An icon's position is its centre, in the window's own coordinates.
    store["libwaifu-gui.app"]["Iloc"] = (WINDOW[0] // 4, 170)
    store["Applications"]["Iloc"] = (WINDOW[0] * 3 // 4, 170)
