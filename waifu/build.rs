// Points the linker at the static library that the CMake build produces, and writes waifu.h. CMake
// is what drives this crate's build -- the top-level CMakeLists.txt's `waifu-engine` target runs
// `cargo build` after the `flint` archive is fresh, so a plain `cmake --build build` builds both. This script never
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

    // Found next to whatever loads it -- an app's Frameworks directory, the command line's own --
    // rather than at the path cargo happened to write it to.
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
        println!("cargo:rustc-cdylib-link-arg=-Wl,-install_name,@rpath/libwaifu.dylib");
    }

    let manifest = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    write_header(&manifest);
}

/// waifu.h, from the `extern "C"` functions and `#[repr(C)]` types in src/c_api.rs and
/// src/c_api/support.rs, with their doc comments. Written beside the source, where a change to the
/// ABI shows up in a diff.
fn write_header(manifest: &Path) {
    println!("cargo:rerun-if-changed=src/c_api.rs");
    println!("cargo:rerun-if-changed=src/c_api/support.rs");
    println!("cargo:rerun-if-changed=cbindgen.toml");

    let config =
        cbindgen::Config::from_file(manifest.join("cbindgen.toml")).expect("waifu/cbindgen.toml");
    match cbindgen::Builder::new()
        .with_config(config)
        .with_src(manifest.join("src/c_api.rs"))
        .with_src(manifest.join("src/c_api/support.rs"))
        .generate()
    {
        Ok(bindings) => {
            bindings.write_to_file(manifest.join("include/waifu.h"));
        }
        // Said rather than failed on: a header that cannot be regenerated leaves the last one,
        // and the library under it still builds.
        Err(error) => println!("cargo:warning=waifu.h was not regenerated: {error}"),
    }
}
