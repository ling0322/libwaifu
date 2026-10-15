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

//! Links what only a final link can be given, and stamps the commit the tool was built from.
//!
//! The waifu crate's build script names flint's static libraries, and those reach this link on
//! their own. Its link arguments do not: cargo keeps a `rustc-link-arg` to the package that said
//! it, so whatever flint's link flags carry as one is said again here, for this binary's link.

use std::path::{Path, PathBuf};

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

    stamp_revision();
}

/// Put the commit this was built from into the binary, for the screen to show.
///
/// A screenshot of a run is worth little without it: two builds of the same version can differ by
/// anything, and the first thing to ask about a picture that came out wrong is which code made it.
/// Missing on purpose is a failure: a copy of the source with no git around it still builds, and
/// says "unknown" where the hash would be rather than refusing.
fn stamp_revision() {
    let manifest = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let git = manifest.join("../.git");

    // Rebuilt when the checkout moves. HEAD covers a checkout, and the file HEAD points at covers
    // a commit made on the branch already checked out.
    if git.is_dir() {
        println!("cargo:rerun-if-changed={}", git.join("HEAD").display());
        if let Some(reference) = head_reference(&git) {
            println!("cargo:rerun-if-changed={}", git.join(reference).display());
        }
    }

    let revision = std::process::Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .current_dir(&manifest)
        .output()
        .ok()
        .filter(|out| out.status.success())
        .and_then(|out| String::from_utf8(out.stdout).ok())
        .map(|text| text.trim().to_string())
        .filter(|text| !text.is_empty())
        .unwrap_or_else(|| "unknown".to_string());

    println!("cargo:rustc-env=WAIFU_REVISION={revision}");
}

/// The ref file HEAD names, relative to the git directory, when it names one at all. A detached
/// HEAD holds a commit instead, and then there is no second file to watch.
fn head_reference(git: &Path) -> Option<String> {
    let head = std::fs::read_to_string(git.join("HEAD")).ok()?;
    let reference = head.trim().strip_prefix("ref: ")?;
    Some(reference.to_string())
}
