// Points the linker at the static library that the CMake build produces. CMake is what drives
// this crate's build -- the top-level CMakeLists.txt's `waifu-cli` target runs `cargo build` after
// the `flint` archive is fresh, so a plain `cmake --build build` builds both. This script never
// invokes CMake itself; it only reads the link flags CMake already wrote out. That keeps a lone
// `cargo build` working too (useful when iterating on Rust code only), as long as the native
// archive is already up to date. Override the location with LIBWAIFU_LIB_DIR when building
// somewhere other than the in-tree `build` directory.
//
// libflint.a carries no record of what it still needs -- libunwind, the CUDA runtime, OpenMP and
// the C++ runtime are all resolved by whoever links it -- and that set depends on the CMake
// options the archive was built with. So CMake writes the whole list out as cargo directives and
// this script echoes them. They are `rustc-link-lib` and `rustc-link-search` rather than raw link
// args on purpose: those two propagate to whatever binary or cdylib ends up linking this crate,
// which is why nothing downstream needs a build script of its own.
use std::path::{Path, PathBuf};

fn main() {
    let lib_dir = std::env::var("LIBWAIFU_LIB_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            let manifest = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
            manifest.join("../build")
        });

    let flags_path = lib_dir.join("flint_link_flags.txt");
    let flags = std::fs::read_to_string(&flags_path).unwrap_or_else(|e| {
        panic!(
            "cannot read {}: {e}\n\
             Build with CMake first (cmake -S . -B build && cmake --build build), or point \
             LIBWAIFU_LIB_DIR at a directory that has one.",
            flags_path.display()
        )
    });

    println!("cargo:rerun-if-env-changed=LIBWAIFU_LIB_DIR");
    println!("cargo:rerun-if-changed={}", flags_path.display());

    // Without this, a rebuilt archive with the same link flags leaves cargo believing the last
    // link is still good, and a `cargo test` after a C++ change quietly tests the old library.
    println!(
        "cargo:rerun-if-changed={}",
        lib_dir.join("libflint.a").display()
    );
    for line in flags.lines().filter(|l| !l.trim().is_empty()) {
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
