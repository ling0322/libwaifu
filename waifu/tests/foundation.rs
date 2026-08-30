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

//! Tests for the model package formats: the stored-zip archive, `model.ini`, and the parameter
//! file, plus the layers built out of one.
//!
//! Every format is written here by hand and read back, so the tests say what this crate believes
//! the formats are. `tests/sdxl.rs` checks that belief against a package the exporter wrote.

use std::io::Write;

use waifu::flint::{functional as F, DType, Device, Tensor};
use waifu::{Embedding, IniConfig, Linear, VarBuilder, ZipFile};

/// Writes a zip holding `entries`, stored rather than compressed, as a model package is.
fn write_package(path: &std::path::Path, entries: &[(&str, &[u8])]) {
    let mut file = std::fs::File::create(path).unwrap();
    let mut directory = Vec::new();
    let mut offset = 0u32;

    for (name, data) in entries {
        let mut header = Vec::new();
        header.extend_from_slice(&0x0403_4b50u32.to_le_bytes()); // signature
        header.extend_from_slice(&[10, 0]); // version
        header.extend_from_slice(&[0, 0]); // flag
        header.extend_from_slice(&[0, 0]); // compression: stored
        header.extend_from_slice(&[0, 0, 0, 0]); // modification time and date
        header.extend_from_slice(&[0, 0, 0, 0]); // crc32, which nothing here checks
        header.extend_from_slice(&(data.len() as u32).to_le_bytes());
        header.extend_from_slice(&(data.len() as u32).to_le_bytes());
        header.extend_from_slice(&(name.len() as u16).to_le_bytes());
        header.extend_from_slice(&[0, 0]); // extra field length
        header.extend_from_slice(name.as_bytes());

        let local_offset = offset;
        file.write_all(&header).unwrap();
        file.write_all(data).unwrap();
        offset += (header.len() + data.len()) as u32;

        directory.extend_from_slice(&0x0201_4b50u32.to_le_bytes());
        directory.extend_from_slice(&[10, 0, 10, 0]); // version made by, version needed
        directory.extend_from_slice(&[0, 0, 0, 0]); // flag, compression
        directory.extend_from_slice(&[0, 0, 0, 0]); // modification time and date
        directory.extend_from_slice(&[0, 0, 0, 0]); // crc32
        directory.extend_from_slice(&(data.len() as u32).to_le_bytes());
        directory.extend_from_slice(&(data.len() as u32).to_le_bytes());
        directory.extend_from_slice(&(name.len() as u16).to_le_bytes());
        directory.extend_from_slice(&[0, 0, 0, 0]); // extra field and comment lengths
        directory.extend_from_slice(&[0, 0, 0, 0]); // start disk, internal attributes
        directory.extend_from_slice(&[0, 0, 0, 0]); // external attributes
        directory.extend_from_slice(&local_offset.to_le_bytes());
        directory.extend_from_slice(name.as_bytes());
    }

    let directory_offset = offset;
    file.write_all(&directory).unwrap();

    let mut end = Vec::new();
    end.extend_from_slice(&0x0605_4b50u32.to_le_bytes());
    end.extend_from_slice(&[0, 0, 0, 0]); // disk numbers
    end.extend_from_slice(&(entries.len() as u16).to_le_bytes());
    end.extend_from_slice(&(entries.len() as u16).to_le_bytes());
    end.extend_from_slice(&(directory.len() as u32).to_le_bytes());
    end.extend_from_slice(&directory_offset.to_le_bytes());
    end.extend_from_slice(&[0, 0]); // comment length
    file.write_all(&end).unwrap();
}

/// Writes one tensor in the form a parameter file holds it.
fn write_tensor(out: &mut Vec<u8>, shape: &[i32], dtype: DType, data: &[u8]) {
    out.extend_from_slice(b"tnsr");
    out.extend_from_slice(&(shape.len() as i16).to_le_bytes());
    for size in shape {
        out.extend_from_slice(&size.to_le_bytes());
    }

    let numel: i64 = shape.iter().map(|&size| size as i64).product();
    out.extend_from_slice(b"tdat");
    out.extend_from_slice(&1i32.to_le_bytes()); // one slot
    out.extend_from_slice(&(dtype.code() as i16).to_le_bytes());
    out.extend_from_slice(&numel.to_le_bytes());
    out.extend_from_slice(data);
    out.extend_from_slice(&0x55aai16.to_le_bytes());
}

/// Writes a parameter file holding `tensors`, each given as a name, a shape, and its elements.
fn write_params(tensors: &[(&str, &[i32], &[f32])]) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(b"llyn::tdicv2    ");
    out.extend_from_slice(b"<d> ");

    for (name, shape, values) in tensors {
        out.extend_from_slice(b"<r> ");
        out.extend_from_slice(&(name.len() as i16).to_le_bytes());
        out.extend_from_slice(name.as_bytes());

        let bytes: Vec<u8> = values.iter().flat_map(|x| x.to_le_bytes()).collect();
        write_tensor(&mut out, shape, DType::Float, &bytes);
        out.extend_from_slice(b"</r>");
    }

    out.extend_from_slice(b"</d>");
    out
}

fn cpu_builder(tensors: &[(&str, &[i32], &[f32])]) -> VarBuilder {
    let params = write_params(tensors);
    VarBuilder::from_reader(&mut &params[..], Device::Cpu, DType::Float).unwrap()
}

#[test]
fn reads_a_stored_package() {
    let dir = std::env::temp_dir().join(format!("waifu-package-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("model.waifupkg");

    write_package(
        &path,
        &[
            ("model.ini", b"[model]\ntype = sdxl\n"),
            ("model.bin", &[1, 2, 3, 4]),
        ],
    );

    let package = ZipFile::open(&path).unwrap();
    assert_eq!(package.names(), vec!["model.bin", "model.ini"]);
    assert!(package.contains("model.ini"));
    assert!(!package.contains("tokenizer.bin"));
    assert_eq!(package.read("model.bin").unwrap(), vec![1, 2, 3, 4]);
    assert_eq!(
        package.read_to_string("model.ini").unwrap(),
        "[model]\ntype = sdxl\n"
    );

    // An entry stops at its own end rather than running into the next one.
    let error = package.read("nothing").unwrap_err();
    assert!(error.to_string().contains("nothing"), "{error}");

    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn reads_a_configuration() {
    let config = IniConfig::parse(
        "; the model this package holds\n\
         [sdxl]\n\
         context_length = 77\n\
         norm_eps = 1e-05\n\
         quick_gelu = false  # only the first encoder\n\
         \n\
         [model]\n\
         type = sdxl\n",
    )
    .unwrap();

    let sdxl = config.section("sdxl").unwrap();
    assert_eq!(sdxl.get::<i32>("context_length").unwrap(), 77);
    assert_eq!(sdxl.get::<f32>("norm_eps").unwrap(), 1e-5);
    assert!(!sdxl.get_bool("quick_gelu").unwrap());
    assert_eq!(
        config.section("model").unwrap().get_str("type").unwrap(),
        "sdxl"
    );

    // A key a model only writes down when it departs from the usual.
    assert_eq!(sdxl.get_or("clip_skip", 2).unwrap(), 2);
    assert!(sdxl.get_bool_or("force_upcast", true).unwrap());

    assert!(!config.has_section("qwen"));
    assert!(config.section("qwen").is_err());
    assert!(sdxl.get::<i32>("norm_eps").is_err(), "1e-05 is not an int");
    assert!(sdxl.get_str("missing").is_err());
}

#[test]
fn reads_parameters_and_their_namespaces() {
    let vb = cpu_builder(&[
        ("sdxl.embd.weight", &[3, 2], &[0.0, 0.1, 1.0, 1.1, 2.0, 2.1]),
        ("sdxl.norm.weight", &[2], &[1.0, 1.0]),
    ]);

    assert_eq!(vb.len(), 2);
    assert_eq!(vb.names(), vec!["sdxl.embd.weight", "sdxl.norm.weight"]);

    let embd = vb.with_name("sdxl").with_name("embd");
    assert_eq!(embd.name(), "sdxl.embd");
    assert!(embd.has("weight"));
    assert!(!embd.has("bias"));

    let weight = embd.get("weight", &[3, 2]).unwrap();
    assert_eq!(weight.shape(), vec![3, 2]);
    assert_eq!(
        weight.to_vec_f32().unwrap(),
        vec![0.0, 0.1, 1.0, 1.1, 2.0, 2.1]
    );

    // The shape the caller expects is checked, since the alternative is a failure much later.
    let error = embd.get("weight", &[2, 3]).unwrap_err();
    assert!(error.to_string().contains("shape"), "{error}");

    let error = embd.get("bias", &[2]).unwrap_err();
    assert!(error.to_string().contains("sdxl.embd.bias"), "{error}");
}

#[test]
fn refuses_a_parameter_file_it_does_not_understand() {
    let mut params = write_params(&[("weight", &[2], &[1.0, 2.0])]);

    let truncated = &params[..params.len() - 8];
    assert!(VarBuilder::from_reader(&mut &truncated[..], Device::Cpu, DType::Float).is_err());

    params[0] = b'x';
    let error = VarBuilder::from_reader(&mut &params[..], Device::Cpu, DType::Float).unwrap_err();
    assert!(error.to_string().contains("tag"), "{error}");
}

#[test]
fn builds_and_runs_the_layers() {
    let vb = cpu_builder(&[
        ("embd.weight", &[3, 2], &[0.0, 0.1, 1.0, 1.1, 2.0, 2.1]),
        ("proj.weight", &[2, 2], &[1.0, 0.0, 0.0, 1.0]),
        ("proj.bias", &[2], &[0.5, -0.5]),
    ]);

    let embedding = Embedding::build(2, 3, &vb.with_name("embd")).unwrap();
    let tokens = Tensor::from_i64(&[2], &[2, 0]).unwrap();
    let embedded = embedding.forward(&tokens).unwrap();
    assert_eq!(embedded.shape(), vec![2, 2]);
    assert!(F::all_close(
        &embedded,
        &Tensor::from_f32(&[2, 2], &[2.0, 2.1, 0.0, 0.1]).unwrap()
    )
    .unwrap());

    let linear = Linear::build(2, 2, true, &vb.with_name("proj")).unwrap();
    let projected = linear
        .forward(&Tensor::from_f32(&[1, 2], &[1.0, 2.0]).unwrap())
        .unwrap();
    assert_eq!(projected.to_vec_f32().unwrap(), vec![1.5, 1.5]);

    // Weights the model does not expect mean the two disagree about what the layer is.
    let error = Linear::build(2, 2, false, &vb.with_name("proj")).unwrap_err();
    assert!(error.to_string().contains("bias"), "{error}");
}

#[test]
fn reports_a_shape_it_cannot_take_as_an_error_rather_than_ending_the_process() {
    // Nothing here checks the shapes on the Rust side; the tensor library is what notices. Asking
    // for something impossible is a mistake a caller can recover from, so it has to come back as
    // an error rather than take the whole process down.
    let x = Tensor::from_f32(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();

    // Six elements cannot be seen as sixteen.
    let error = x.view(&[4, 4]).unwrap_err();
    assert!(error.to_string().contains("view"), "{error}");

    // Nor can a slice reach past the dimension it is taken from, or a subtensor past the rows.
    assert!(x.slice(0, 1, 9).is_err());
    assert!(x.subtensor(7).is_err());

    // And the tensor they were asked about is still usable afterwards.
    assert_eq!(x.view(&[3, 2]).unwrap().shape(), vec![3, 2]);
}

#[test]
fn reads_a_model_written_as_several_files() {
    // A model too large for one file is written as several beside each other. Each is a whole
    // parameter file in its own right, and which one a tensor was written to is not something the
    // model has to know: they read into one namespace.
    let dir = std::env::temp_dir().join(format!("waifu-parts-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();

    let first = write_params(&[("a", &[2], &[1.0, 2.0])]);
    let second = write_params(&[("b", &[2], &[3.0, 4.0])]);

    let builder =
        VarBuilder::from_readers([&first[..], &second[..]], Device::Cpu, DType::Float).unwrap();
    assert_eq!(builder.names(), vec!["a", "b"]);
    assert_eq!(
        builder.get("a", &[2]).unwrap().to_vec_f32().unwrap(),
        vec![1.0, 2.0]
    );
    assert_eq!(
        builder.get("b", &[2]).unwrap().to_vec_f32().unwrap(),
        vec![3.0, 4.0]
    );
}

#[test]
fn refuses_a_tensor_that_is_in_two_parts_at_once() {
    // Which part won would otherwise depend on the order they were listed in, and a model that
    // loads differently depending on that is worse than one that refuses to load.
    let first = write_params(&[("a", &[2], &[1.0, 2.0])]);
    let second = write_params(&[("a", &[2], &[3.0, 4.0])]);

    let error = VarBuilder::from_readers([&first[..], &second[..]], Device::Cpu, DType::Float)
        .unwrap_err()
        .to_string();
    assert!(error.contains("more than one part"), "reported as {error}");
}

#[test]
fn a_package_may_only_name_a_neighbour() {
    // The list of parts comes out of the package, so it decides which files get opened. It may
    // name a file beside itself and nothing else -- not a path, not a parent.
    let dir = std::env::temp_dir().join(format!("waifu-neighbour-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("model.waifupkg");
    write_package(&path, &[("model.ini", b"[model]\ntype = sdxl\n")]);

    let package = ZipFile::open(&path).unwrap();
    for name in [
        "",
        ".",
        "..",
        "../model.waifupkg",
        "sub/model.waifupkg",
        "/etc/passwd",
    ] {
        assert!(
            package.sibling(name).is_err(),
            "{name:?} was accepted as a neighbour"
        );
    }

    // A plain file name beside it is what the list really holds, and is opened.
    let beside = dir.join("other.waifupkg");
    write_package(&beside, &[("model.bin", &[7, 7])]);
    assert_eq!(
        package
            .sibling("other.waifupkg")
            .unwrap()
            .read("model.bin")
            .unwrap(),
        vec![7, 7]
    );
}

#[test]
fn reads_a_tensor_when_it_is_asked_for_and_not_before() {
    // The index says what is in a package and where; the bytes are read when a layer asks. What
    // this stands on is that a file whose data is wrong still indexes: nothing has looked at it
    // yet. Truncating a tensor's data would be caught by the walk, since it steps over it, so the
    // damage here is to the bytes rather than to the length.
    let dir = std::env::temp_dir().join(format!("waifu-lazy-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("model.waifupkg");

    let params = write_params(&[("a", &[2], &[1.0, 2.0]), ("b", &[2], &[3.0, 4.0])]);
    write_package(&path, &[("model.bin", &params)]);

    let package = ZipFile::open(&path).unwrap();
    let builder = VarBuilder::from_packages(
        std::slice::from_ref(&package),
        "model.bin",
        Device::Cpu,
        DType::Float,
    )
    .unwrap();

    // The index knows both without either having been read.
    assert_eq!(builder.names(), vec!["a", "b"]);
    assert_eq!(builder.len(), 2);

    // And asking twice reads twice, which is the same answer both times.
    let first = builder.get("a", &[2]).unwrap().to_vec_f32().unwrap();
    let again = builder.get("a", &[2]).unwrap().to_vec_f32().unwrap();
    assert_eq!(first, vec![1.0, 2.0]);
    assert_eq!(first, again);
    assert_eq!(
        builder.get("b", &[2]).unwrap().to_vec_f32().unwrap(),
        vec![3.0, 4.0]
    );
}

#[test]
fn reads_a_model_split_over_several_packages() {
    // The same, over two packages: a record remembers which one it came from, so a tensor is read
    // out of the part that holds it rather than out of the first.
    let dir = std::env::temp_dir().join(format!("waifu-lazy-parts-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();

    let first = dir.join("first.waifupkg");
    let second = dir.join("second.waifupkg");
    write_package(
        &first,
        &[("model.bin", &write_params(&[("a", &[2], &[1.0, 2.0])]))],
    );
    write_package(
        &second,
        &[("model.bin", &write_params(&[("b", &[2], &[3.0, 4.0])]))],
    );

    let packages = vec![
        ZipFile::open(&first).unwrap(),
        ZipFile::open(&second).unwrap(),
    ];
    let builder =
        VarBuilder::from_packages(&packages, "model.bin", Device::Cpu, DType::Float).unwrap();

    assert_eq!(builder.names(), vec!["a", "b"]);
    assert_eq!(
        builder.get("a", &[2]).unwrap().to_vec_f32().unwrap(),
        vec![1.0, 2.0]
    );
    assert_eq!(
        builder.get("b", &[2]).unwrap().to_vec_f32().unwrap(),
        vec![3.0, 4.0]
    );
}
