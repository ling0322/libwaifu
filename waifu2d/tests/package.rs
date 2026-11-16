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

//! A package rigged: the same drawing as a package and as a Photoshop document is the same rig,
//! and a document img2layers wrote becomes a package with its depth and its halves said outright.

use waifu2d::package::{Appearance, Kind, Package};
use waifu2d::psd::{depth_name, PsdDocument};
use waifu2d::Rig;

mod common;

use common::{face_psd, figure_psd, layer, Layer, Writer};

/// `rig` without its id, which is new every time one is built.
fn same(mut rig: Rig) -> Rig {
    rig.id = 0;
    rig
}

/// `document`, as a package, written and read back.
fn round_trip(document: &PsdDocument) -> Package {
    let bytes = Package::new(Appearance::from_psd(document))
        .write()
        .expect("the package is written");
    Package::read(&bytes).expect("the package reads")
}

#[test]
fn a_package_is_the_rig_its_document_is() {
    for bytes in [face_psd(), figure_psd()] {
        let document = PsdDocument::read(&bytes).expect("a Photoshop document");
        let package = round_trip(&document);
        assert!(same(Rig::from_package(&package)) == same(Rig::build(&document)));
        // And told apart from a document by how it starts.
        let written = package.write().unwrap();
        assert!(same(Rig::read(&written).unwrap()) == same(Rig::read(&bytes).unwrap()));
    }
}

#[test]
fn a_document_brings_its_depth_and_its_halves() {
    let mut depth = layer(&depth_name("topwear-0"), (2, 2, 4, 4), [40, 40, 40, 255]);
    depth.hidden = true;
    let bytes = Writer::document(
        8,
        8,
        &[
            layer("face", (0, 0, 8, 8), [250, 220, 200, 255]),
            layer("topwear-1", (0, 2, 8, 6), [10, 10, 90, 255]),
            layer("topwear-0", (2, 2, 4, 4), [10, 10, 200, 255]),
            // Not a half: there is no "ribbon-1".
            layer("ribbon-0", (0, 0, 2, 2), [200, 0, 0, 255]),
            Layer {
                name: "</Layer group>".into(),
                divider: 3,
                hidden: true,
                ..Layer::default()
            },
            depth,
            Layer {
                name: "depth".into(),
                divider: 2,
                hidden: true,
                ..Layer::default()
            },
        ],
    );
    let document = PsdDocument::read(&bytes).unwrap();
    let package = round_trip(&document);

    let names: Vec<&str> = package
        .appearance
        .layers
        .iter()
        .map(|l| l.name.as_str())
        .collect();
    assert_eq!(names, ["face", "topwear-1", "topwear-0", "ribbon-0"]);
    let kinds: Vec<(Kind, Option<&str>)> = (package.appearance.layers.iter())
        .map(|l| (l.kind, l.of.as_deref()))
        .collect();
    assert_eq!(
        kinds,
        [
            (Kind::Part, None),
            (Kind::StratifiedFar, Some("topwear")),
            (Kind::StratifiedNear, Some("topwear")),
            (Kind::Part, None),
        ]
    );
    let near = &package.appearance.layers[2];
    assert_eq!(near.depth, Some(40.0 / 255.0));
    let map = near.depth_map.as_ref().expect("its depth map");
    assert_eq!((map.width, map.height), (4, 4));
    assert!(map.gray.iter().all(|&g| g == 40));
    assert!(package.appearance.layers[0].depth_map.is_none());
}
