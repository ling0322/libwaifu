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

//! Names the library for where it is loaded from, and writes waifu_modelmanager.h.
//!
//! Nothing native is linked: the model manager is Rust all the way down, so this builds without
//! CMake having built anything first.

use std::path::{Path, PathBuf};

fn main() {
    let manifest = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());

    // Found next to whatever loads it -- an app's Frameworks directory, a test's own -- rather than
    // at the path cargo happened to write it to.
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
        println!(
            "cargo:rustc-cdylib-link-arg=-Wl,-install_name,@rpath/libwaifu_modelmanager.dylib"
        );
    }

    write_header(&manifest);
}

/// waifu_modelmanager.h, from the `extern "C"` functions and `#[repr(C)]` types in src/c_api.rs and
/// src/c_api/support.rs, with their doc comments. Written beside the source, where a change to the
/// ABI shows up in a diff.
fn write_header(manifest: &Path) {
    println!("cargo:rerun-if-changed=src/c_api.rs");
    println!("cargo:rerun-if-changed=src/c_api/support.rs");
    println!("cargo:rerun-if-changed=cbindgen.toml");

    let config = cbindgen::Config::from_file(manifest.join("cbindgen.toml"))
        .expect("waifu-modelmanager/cbindgen.toml");
    match cbindgen::Builder::new()
        .with_config(config)
        .with_src(manifest.join("src/c_api.rs"))
        .with_src(manifest.join("src/c_api/support.rs"))
        .generate()
    {
        Ok(bindings) => {
            bindings.write_to_file(manifest.join("include/waifu_modelmanager.h"));
        }
        // Said rather than failed on: a header that cannot be regenerated leaves the last one,
        // and the library under it still builds.
        Err(error) => println!("cargo:warning=waifu_modelmanager.h was not regenerated: {error}"),
    }
}
