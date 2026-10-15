// The MIT License (MIT)
//
// Copyright (c) 2026 Xiaoyang Chen
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software
// and associated documentation files (the "Software"), to deal in the Software without
// restriction, including without limitation the rights to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all copies or
// substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
// BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
// DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

//! Links what only a final link can be given.
//!
//! The waifu crate's build script names flint's static libraries, and those reach these links on
//! their own. Its link arguments do not: cargo keeps a `rustc-link-arg` to the package that said
//! it, so whatever flint's link flags carry as one is said again here, for these binaries.

use std::path::PathBuf;

fn main() {
    let manifest = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let lib_dir = std::env::var("LIBWAIFU_LIB_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| manifest.join("../build"));
    let flags_path = lib_dir.join("flint_link_flags.txt");
    let flags = std::fs::read_to_string(&flags_path).unwrap_or_else(|error| {
        panic!(
            "cannot read {}: {error}\nBuild with CMake first, or point LIBWAIFU_LIB_DIR at a \
             directory that has one.",
            flags_path.display()
        )
    });
    println!("cargo:rerun-if-env-changed=LIBWAIFU_LIB_DIR");
    println!("cargo:rerun-if-changed={}", flags_path.display());

    for line in flags
        .lines()
        .filter(|line| line.starts_with("cargo:rustc-link-arg="))
    {
        println!("{line}");
    }
}
