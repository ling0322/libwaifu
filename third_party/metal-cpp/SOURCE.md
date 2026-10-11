Apple's metal-cpp, the C++ interface to Metal that flint's Metal backend is written against.

- From: https://github.com/apple/metal-cpp
- Tag: `release/metal-cpp_macOS14_iOS17` (commit 807c2cf)
- Licence: Apache-2.0, in `LICENSE.txt`

Only `Metal/` and the `Foundation/` it includes are kept; `MetalFX/`, `QuartzCore/` and
`SingleHeader/` are not used.

It is the macOS 14 release on purpose, which is the deployment target: these headers declare no API
newer than macOS 14, so calling one is a compile error rather than a crash on an older Mac. Moving
to a newer release is moving the deployment target with it.
