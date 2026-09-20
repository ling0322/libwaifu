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

//! Tests for `<model-id>.yaml`: the file that says what a model is and which files it is made of.
//!
//! [`WHOLE`] is written the way `tools/model_writer.py` writes one, down to the comment at the
//! top and which values it quotes. What these tests say is that what the exporter emits is what
//! this crate reads -- the two halves of the format are in different languages and the only thing
//! holding them together is that they agree here.

use waifu::Manifest;

/// A manifest in the shape the exporter writes, with something of each kind in it.
const WHOLE: &str = r#"# What this model is, and which files it is made of.
#
# The safetensors beside this file hold the weights and nothing else; everything
# that says what they are is here.

weights:
  - sdxl-base-00001-of-00002.safetensors
  - sdxl-base-00002-of-00002.safetensors

tokenizers:
  tokenizer: sdxl-base.tokenizer.json

config:
  model:
    type: sdxl
  sdxl:
    latent_channels: 4
    vae_scaling_factor: 0.13025
    unet_block_out_channels: 320,640,1280
    latents_mean: -0.7571,-0.7089
    text_norm_eps: 1e-05
    text_hidden_act: quick_gelu
    clip_skip: 2

suggested:
  prompt: masterpiece, best quality
  sizes:
    - [1024, 1024]
    - [832, 1216]
    - [1216, 832]
  steps: 8
  guidance: 1.0
"#;

/// A directory of this test's own, emptied first so a failed run does not haunt the next one.
fn temp_dir(what: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("waifu-manifest-{what}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// A whole manifest whose one suggestion is `line`, written as it stands.
fn suggesting(line: &str) -> String {
    format!("weights:\n  - m.safetensors\nconfig:\n  model:\n    type: sdxl\nsuggested:\n  {line}\n")
}

#[test]
fn reads_the_three_blocks_a_manifest_merged() {
    let manifest = Manifest::parse(WHOLE).unwrap();

    // The weights, in the order they are to be read.
    assert_eq!(
        manifest.weights(),
        [
            "sdxl-base-00001-of-00002.safetensors",
            "sdxl-base-00002-of-00002.safetensors"
        ]
    );

    // And every file the model is made of, which is those plus each tokenizer's own.
    assert_eq!(
        manifest.files(),
        [
            "sdxl-base-00001-of-00002.safetensors",
            "sdxl-base-00002-of-00002.safetensors",
            "sdxl-base.tokenizer.json"
        ]
    );

    // A tokenizer is one line, because a tokenizer is one file.
    assert_eq!(
        manifest.tokenizer("tokenizer").unwrap(),
        "sdxl-base.tokenizer.json"
    );
    assert!(manifest.tokenizer("qwen3").is_err());

    // What used to be model.ini, asked for by the names it always had.
    assert_eq!(
        manifest.section("model").unwrap().get_str("type").unwrap(),
        "sdxl"
    );

    let sdxl = manifest.section("sdxl").unwrap();
    assert_eq!(sdxl.get::<i32>("latent_channels").unwrap(), 4);
    assert_eq!(sdxl.get::<f32>("vae_scaling_factor").unwrap(), 0.13025);
    assert_eq!(sdxl.get::<f32>("text_norm_eps").unwrap(), 1e-5);
    assert_eq!(sdxl.get_str("text_hidden_act").unwrap(), "quick_gelu");

    // A list stays the text it was written as, because what it means is the model's to say.
    assert_eq!(
        sdxl.get_str("unet_block_out_channels").unwrap(),
        "320,640,1280"
    );
    assert_eq!(sdxl.get_str("latents_mean").unwrap(), "-0.7571,-0.7089");

    // A key a model only writes down when it departs from the usual, and one it did write.
    assert_eq!(sdxl.get_or("force_upcast", 1).unwrap(), 1);
    assert_eq!(sdxl.get::<i32>("clip_skip").unwrap(), 2);

    // What used to be metadata.json, which has grown from one field to several.
    let suggested = manifest.suggested();
    assert_eq!(
        suggested.prompt.as_deref(),
        Some("masterpiece, best quality")
    );
    assert_eq!(suggested.sizes, vec![(1024, 1024), (832, 1216), (1216, 832)]);
    assert_eq!(suggested.size(), Some((1024, 1024)));
    assert_eq!(suggested.steps, Some(8));
    assert_eq!(suggested.guidance, Some(1.0));

    assert!(manifest.has_section("sdxl"));
    assert!(!manifest.has_section("anima"));
    assert!(manifest.section("anima").is_err());
}

#[test]
fn a_model_may_tokenize_more_than_one_way() {
    // Anima's shape: two of them, neither called `tokenizer`, because there is nothing to make one
    // of the two the one a reader should assume.
    let manifest = Manifest::parse(
        "weights:\n  - m.safetensors\n\
         tokenizers:\n  qwen3: m.qwen3.tokenizer.json\n  t5: m.t5.tokenizer.json\n\
         config:\n  model:\n    type: anima\n",
    )
    .unwrap();

    assert_eq!(manifest.tokenizer("qwen3").unwrap(), "m.qwen3.tokenizer.json");
    assert_eq!(manifest.tokenizer("t5").unwrap(), "m.t5.tokenizer.json");

    // Asking for the one it does not have says which it does have, since that is the thing worth
    // knowing when a pipeline and a model disagree about the name.
    let error = manifest.tokenizer("tokenizer").unwrap_err().to_string();
    assert!(error.contains("qwen3"), "{error}");
    assert!(error.contains("t5"), "{error}");

    assert_eq!(
        manifest.files(),
        ["m.safetensors", "m.qwen3.tokenizer.json", "m.t5.tokenizer.json"]
    );
}

#[test]
fn a_model_that_tokenizes_nothing_names_no_tokenizers() {
    // Not an error here: whether a model needs one is the model's to say, and this file is read
    // before anything knows which kind it is.
    let manifest =
        Manifest::parse("weights:\n  - m.safetensors\nconfig:\n  model:\n    type: sdxl\n")
            .unwrap();

    assert_eq!(manifest.files(), ["m.safetensors"]);
    assert!(manifest.tokenizer("tokenizer").is_err());
}

#[test]
fn a_model_with_no_advice_to_give_has_no_suggested_block() {
    let manifest =
        Manifest::parse("weights:\n  - m.safetensors\nconfig:\n  model:\n    type: sdxl\n").unwrap();

    let suggested = manifest.suggested();
    assert_eq!(suggested.prompt, None);
    assert_eq!(suggested.steps, None);
    assert_eq!(suggested.guidance, None);
    assert!(suggested.sizes.is_empty());
}

#[test]
fn a_block_this_build_has_not_heard_of_is_stepped_over() {
    // A manifest from a newer writer still names the weights and still says what the model is.
    let manifest = Manifest::parse(
        "weights:\n  - m.safetensors\n\
         config:\n  model:\n    type: sdxl\n\
         suggested:\n  prompt: 1girl\n\
         provenance:\n  exported_by: a later version\n",
    )
    .unwrap();

    assert_eq!(manifest.weights(), ["m.safetensors"]);
    assert_eq!(
        manifest.section("model").unwrap().get_str("type").unwrap(),
        "sdxl"
    );
    assert_eq!(manifest.suggested().prompt.as_deref(), Some("1girl"));
}

#[test]
fn scalars_mean_what_they_mean_everywhere_else() {
    let prompt = |line: &str| {
        Manifest::parse(&suggesting(line))
            .unwrap()
            .suggested()
            .prompt
            .clone()
    };

    // A space and then a `#` begins a comment, the way it does everywhere else; a `#` with no
    // space before it is part of the value. A prompt that really wants the first has to be
    // quoted, and the exporter quotes it.
    assert_eq!(
        prompt("prompt: a plain value  # and a comment").as_deref(),
        Some("a plain value")
    );
    assert_eq!(
        prompt("prompt: tag#1 keeps its hash").as_deref(),
        Some("tag#1 keeps its hash")
    );
    assert_eq!(
        prompt(r#"prompt: "tag #1, quoted""#).as_deref(),
        Some("tag #1, quoted")
    );

    // Quoted, with the escapes each kind of quote has and no others.
    assert_eq!(
        prompt(r#"prompt: "a: colon, and a \"quote\"""#).as_deref(),
        Some(r#"a: colon, and a "quote""#)
    );
    assert_eq!(prompt("prompt: 'it''s here'").as_deref(), Some("it's here"));

    // Anything wider than ASCII arrives as itself: the exporter writes utf-8 rather than escapes.
    assert_eq!(prompt("prompt: 猫, 🐱").as_deref(), Some("猫, 🐱"));
}

#[test]
fn a_list_may_be_written_on_one_line_or_on_several() {
    let block = Manifest::parse(&suggesting("sizes:\n    - [1024, 1024]\n    - [832, 1216]"))
        .unwrap()
        .suggested()
        .sizes
        .clone();
    let inline = Manifest::parse(&suggesting("sizes: [[1024, 1024], [832, 1216]]"))
        .unwrap()
        .suggested()
        .sizes
        .clone();

    assert_eq!(block, vec![(1024, 1024), (832, 1216)]);
    assert_eq!(block, inline);
}

#[test]
fn a_manifest_says_whether_the_model_draws_explicit_pictures() {
    let body = "weights:\n  - m.safetensors\nconfig:\n  model:\n    type: sdxl\n";

    // Said, either way.
    assert!(Manifest::parse(&format!("{body}explicit: true\n"))
        .unwrap()
        .explicit());
    assert!(!Manifest::parse(&format!("{body}explicit: false\n"))
        .unwrap()
        .explicit());

    // And not said at all, which is every manifest written before there was anywhere to say it.
    // It reads as no rather than as an error: an old package still names its weights.
    assert!(!Manifest::parse(body).unwrap().explicit());
    assert!(!Manifest::parse(WHOLE).unwrap().explicit());
}

#[test]
fn a_file_that_is_not_a_manifest_is_an_error_rather_than_a_guess() {
    let cases: &[(&str, &str)] = &[
        ("", "nothing in this file"),
        // Two of the three blocks have to be there: a model with no weights and a model that does
        // not say what it is are both unusable, and quietly so.
        ("weights:\n  - m.safetensors\n", "config"),
        ("config:\n  model:\n    type: sdxl\n", "weights"),
        (
            "weights:\nconfig:\n  model:\n    type: sdxl\n",
            "given nothing",
        ),
        // The shapes the reader refuses rather than half-understands.
        (
            "weights:\n\t- m.safetensors\nconfig:\n  model:\n    type: sdxl\n",
            "tab",
        ),
        (
            "weights:\n  - m.safetensors\nconfig:\n  model:\n    type: sdxl\n    type: anima\n",
            "written twice",
        ),
        (
            "weights:\n  - m.safetensors\nconfig:\n  model:\n    type: \"unclosed\n",
            "quote",
        ),
        ("  weights:\n", "starts indented"),
        (
            "weights:\n  - m.safetensors\nconfig:\n  - a list\n",
            "config is not a block of keys",
        ),
        (
            "weights:\n  - m.safetensors\nconfig:\n  model:\n    type: sdxl\n  - a list\n",
            "list item where a key was expected",
        ),
        (
            "weights:\n  - [1, 2\nconfig:\n  model:\n    type: sdxl\n",
            "not closed",
        ),
    ];

    for (text, expected) in cases {
        let error = Manifest::parse(text).map(|_| ()).unwrap_err().to_string();
        assert!(
            error.contains(expected),
            "{text:?} was answered with {error:?}, which does not mention {expected:?}"
        );
    }
}

#[test]
fn a_manifest_may_only_name_a_neighbour() {
    // The list decides which files get opened, and it arrives over the network. It may name a
    // file beside the manifest and nothing else -- not a path, not a parent.
    let dir = temp_dir("neighbour");
    let path = dir.join("model.yaml");

    for name in [
        "''",
        "'.'",
        "'..'",
        "../model.waifupkg",
        "sub/model.waifupkg",
        "/etc/passwd",
    ] {
        std::fs::write(
            &path,
            format!("weights:\n  - {name}\nconfig:\n  model:\n    type: sdxl\n"),
        )
        .unwrap();

        let manifest = Manifest::open(&path).unwrap();
        assert!(
            manifest.weight_paths().is_err(),
            "{name:?} was accepted as a neighbour"
        );
    }

    // A plain file name beside it is what the list really holds, and it resolves to that file.
    std::fs::write(
        &path,
        "weights:\n  - other.safetensors\nconfig:\n  model:\n    type: sdxl\n",
    )
    .unwrap();
    let manifest = Manifest::open(&path).unwrap();
    assert_eq!(
        manifest.weight_paths().unwrap(),
        vec![dir.join("other.safetensors")]
    );

    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn the_weights_are_read_from_beside_the_manifest_and_as_one() {
    // What a model is, end to end: a manifest naming two files, read into one namespace, with
    // neither the model nor this test having to know which tensor was written to which.
    let dir = temp_dir("weights");

    std::fs::write(dir.join("m-00001-of-00002.safetensors"), tensor("a", 1.0)).unwrap();
    std::fs::write(dir.join("m-00002-of-00002.safetensors"), tensor("b", 3.0)).unwrap();

    let path = dir.join("m.yaml");
    std::fs::write(
        &path,
        "weights:\n  \
         - m-00001-of-00002.safetensors\n  \
         - m-00002-of-00002.safetensors\n\
         config:\n  model:\n    type: sdxl\n",
    )
    .unwrap();

    let manifest = Manifest::open(&path).unwrap();
    assert_eq!(manifest.id(), "m");

    let file = manifest.params().unwrap();
    assert_eq!(file.names(), vec!["a", "b"]);
    assert_eq!(
        file.get("a", &[2]).unwrap().to_vec_f32().unwrap(),
        vec![1.0, 2.0]
    );
    assert_eq!(
        file.get("b", &[2]).unwrap().to_vec_f32().unwrap(),
        vec![3.0, 4.0]
    );

    std::fs::remove_dir_all(&dir).unwrap();
}

/// A safetensors file holding one tensor called `name`, of `first` and the number after it.
fn tensor(name: &str, first: f32) -> Vec<u8> {
    let mut data = Vec::new();
    data.extend_from_slice(&first.to_le_bytes());
    data.extend_from_slice(&(first + 1.0).to_le_bytes());

    let header =
        format!("{{{name:?}:{{\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[0,8]}}}}");
    let mut out = (header.len() as u64).to_le_bytes().to_vec();
    out.extend_from_slice(header.as_bytes());
    out.extend_from_slice(&data);
    out
}

#[test]
fn a_manifest_that_is_not_there_says_which_file_it_looked_for() {
    let missing = std::env::temp_dir().join("waifu-no-such-manifest.yaml");
    let _ = std::fs::remove_file(&missing);

    let error = Manifest::open(&missing).map(|_| ()).unwrap_err().to_string();
    assert!(error.contains("waifu-no-such-manifest.yaml"), "{error}");
}
