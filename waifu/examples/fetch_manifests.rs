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

//! Fetches the manifest of every published model, and none of their weights, into a directory laid
//! out as the model cache is -- for an app to be built with, so that what each model suggests can
//! be shown before anything is fetched (see `hub::set_bundled_manifests`).
//!
//! ```bash
//! cargo run --release --manifest-path waifu/Cargo.toml --features hub \
//!     --example fetch_manifests -- macos/build/manifests
//! ```

use std::path::PathBuf;

fn main() {
    let Some(directory) = std::env::args().nth(1).map(PathBuf::from) else {
        eprintln!("usage: fetch_manifests <directory>");
        std::process::exit(2);
    };
    if let Err(error) = std::fs::create_dir_all(&directory) {
        eprintln!("{}: {error}", directory.display());
        std::process::exit(1);
    }
    // The cache is where fetch_manifests writes; pointed here, it writes nowhere else.
    std::env::set_var("WAIFU_CACHE", &directory);

    let fetched = waifu::hub::fetch_manifests(
        &mut |progress| {
            if let waifu::hub::Progress::Fetched { file, part, parts, .. } = progress {
                eprintln!("{part}/{parts} {file}");
            }
        },
        &|| false,
    );
    if let Err(error) = fetched {
        eprintln!("could not fetch the manifests: {error}");
        std::process::exit(1);
    }
}
