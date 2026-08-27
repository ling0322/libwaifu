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
//! the formats are. `tests/model_package.rs` checks that belief against a real package.

use std::io::Write;

use waifu::flint::{functional as F, DType, Device, Tensor};
use waifu::{Embedding, IniConfig, Linear, Nvfp4Linear, RmsNorm, VarBuilder, ZipFile};

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

fn cuda_half_builder(tensors: &[(&str, &[i32], &[f32])]) -> VarBuilder {
    let params = write_params(tensors);
    VarBuilder::from_reader(&mut &params[..], Device::Cuda, DType::Float16).unwrap()
}

/// Values that look like weights rather than a pattern, without pulling in a random number
/// generator: quantization error depends on the spread within each block of 16.
fn spread(count: usize, seed: u32) -> Vec<f32> {
    let mut state = seed | 1;
    (0..count)
        .map(|_| {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            ((state >> 8) as f32 / (1 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}

/// Root mean square of the difference, over the root mean square of the reference.
fn relative_rmse(x: &Tensor, reference: &Tensor) -> f32 {
    let a = x.to_vec_f32().unwrap();
    let b = reference.to_vec_f32().unwrap();
    assert_eq!(a.len(), b.len());

    let error: f64 = a
        .iter()
        .zip(&b)
        .map(|(x, y)| (*x as f64 - *y as f64).powi(2))
        .sum();
    let scale: f64 = b.iter().map(|y| (*y as f64).powi(2)).sum();

    (error / scale).sqrt() as f32
}

#[test]
fn reads_a_stored_package() {
    let dir = std::env::temp_dir().join(format!("waifu-package-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("model.llmpkg");

    write_package(
        &path,
        &[
            ("model.ini", b"[model]\ntype = llama\n"),
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
        "[model]\ntype = llama\n"
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
         [llama]\n\
         hidden_size = 3072\n\
         norm_eps = 1e-05\n\
         qkv_proj_bias = false  # off for llama\n\
         \n\
         [model]\n\
         type = llama\n",
    )
    .unwrap();

    let llama = config.section("llama").unwrap();
    assert_eq!(llama.get::<i32>("hidden_size").unwrap(), 3072);
    assert_eq!(llama.get::<f32>("norm_eps").unwrap(), 1e-5);
    assert!(!llama.get_bool("qkv_proj_bias").unwrap());
    assert_eq!(
        config.section("model").unwrap().get_str("type").unwrap(),
        "llama"
    );

    // A key a model only writes down when it departs from the usual.
    assert_eq!(llama.get_or("num_layers", 28).unwrap(), 28);
    assert!(llama.get_bool_or("tie_embeddings", true).unwrap());

    assert!(!config.has_section("qwen"));
    assert!(config.section("qwen").is_err());
    assert!(llama.get::<i32>("norm_eps").is_err(), "1e-05 is not an int");
    assert!(llama.get_str("missing").is_err());
}

#[test]
fn reads_parameters_and_their_namespaces() {
    let vb = cpu_builder(&[
        (
            "llama.embd.weight",
            &[3, 2],
            &[0.0, 0.1, 1.0, 1.1, 2.0, 2.1],
        ),
        ("llama.norm.weight", &[2], &[1.0, 1.0]),
    ]);

    assert_eq!(vb.len(), 2);
    assert_eq!(vb.names(), vec!["llama.embd.weight", "llama.norm.weight"]);

    let embd = vb.with_name("llama").with_name("embd");
    assert_eq!(embd.name(), "llama.embd");
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
    assert!(error.to_string().contains("llama.embd.bias"), "{error}");
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
        ("norm.weight", &[2], &[2.0, 2.0]),
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

    // The root mean square of a row of ones is one, so the weight is what is left.
    let norm = RmsNorm::build(2, 1e-5, &vb.with_name("norm")).unwrap();
    let normed = norm
        .forward(&Tensor::from_f32(&[1, 2], &[1.0, 1.0]).unwrap())
        .unwrap();
    assert!(F::all_close(&normed, &Tensor::from_f32(&[1, 2], &[2.0, 2.0]).unwrap()).unwrap());

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
fn refuses_a_layer_the_nvfp4_kernel_cannot_take() {
    // Both of these are settled before anything reaches a device, so they hold whether or not
    // this machine has the tensor cores.
    let vb = cpu_builder(&[("proj.weight", &[8, 24], &[0.0; 192])]);

    let error = Nvfp4Linear::build(24, 8, false, &vb.with_name("proj")).unwrap_err();
    assert!(error.to_string().contains("multiple of 32"), "{error}");

    let error = Nvfp4Linear::build(32, 12, false, &vb.with_name("proj")).unwrap_err();
    assert!(error.to_string().contains("multiple of 8"), "{error}");
}

#[test]
fn refuses_a_weight_that_is_not_on_the_device_instead_of_ending_the_process() {
    if !Nvfp4Linear::is_available() {
        return;
    }

    // A host side weight is the mistake most worth naming, so the boundary catches it before the
    // kernels report it as an internal condition.
    let vb = cpu_builder(&[("proj.weight", &[8, 32], &[0.0; 256])]);
    let error = Nvfp4Linear::build(32, 8, false, &vb.with_name("proj")).unwrap_err();
    assert!(error.to_string().contains("CUDA"), "{error}");
}

#[test]
fn builds_and_runs_an_nvfp4_linear() {
    if !Nvfp4Linear::is_available() {
        return;
    }

    const IN_DIM: i32 = 64;
    const OUT_DIM: i32 = 16;
    const ROWS: i32 = 6;

    let weight = spread((IN_DIM * OUT_DIM) as usize, 7);
    let bias = spread(OUT_DIM as usize, 11);
    let vb = cuda_half_builder(&[
        ("proj.weight", &[OUT_DIM, IN_DIM], &weight),
        ("proj.bias", &[OUT_DIM], &bias),
    ]);

    let layer = Nvfp4Linear::build(IN_DIM, OUT_DIM, true, &vb.with_name("proj")).unwrap();

    let input_values = spread((ROWS * IN_DIM) as usize, 13);
    let input = Tensor::from_f32(&[ROWS, IN_DIM], &input_values)
        .unwrap()
        .to_device(Device::Cuda)
        .unwrap()
        .cast(DType::Float16)
        .unwrap();

    let out = layer.forward(&input).unwrap();
    assert_eq!(out.shape(), vec![ROWS, OUT_DIM]);

    // The weight as the multiply sees it, put through the float16 path, is the reference: what is
    // left between the two is the activation's own quantization and the accumulation order.
    let dequantized = layer.dequantized_weight().unwrap();
    let reference = F::add(
        &F::matmul(&input, &dequantized.transpose(0, 1).unwrap()).unwrap(),
        &vb.with_name("proj").get("bias", &[OUT_DIM]).unwrap(),
    )
    .unwrap();

    let out_cpu = out
        .to_device(Device::Cpu)
        .unwrap()
        .cast(DType::Float)
        .unwrap();
    let reference_cpu = reference
        .to_device(Device::Cpu)
        .unwrap()
        .cast(DType::Float)
        .unwrap();
    let rmse = relative_rmse(&out_cpu, &reference_cpu);
    assert!(
        rmse < 0.2,
        "nvfp4 linear drifted from its own dequantized weight by {rmse}"
    );

    // Leading batch axes survive the multiply.
    let batched = Tensor::from_f32(&[2, 3, IN_DIM], &spread((6 * IN_DIM) as usize, 17))
        .unwrap()
        .to_device(Device::Cuda)
        .unwrap()
        .cast(DType::Float16)
        .unwrap();
    assert_eq!(
        layer.forward(&batched).unwrap().shape(),
        vec![2, 3, OUT_DIM]
    );

    // A 1-D input is not a linear layer's input, whatever the precision.
    let error = layer
        .forward(&Tensor::from_f32(&[IN_DIM], &spread(IN_DIM as usize, 19)).unwrap())
        .unwrap_err();
    assert!(error.to_string().contains("2-D"), "{error}");
}

#[test]
fn reports_a_wrongly_shaped_input_instead_of_ending_the_process() {
    let vb = cpu_builder(&[("embd.weight", &[3, 2], &[0.0, 0.1, 1.0, 1.1, 2.0, 2.1])]);
    let embedding = Embedding::build(2, 3, &vb.with_name("embd")).unwrap();

    // Packed token ids are 1-D. The layer says so itself, rather than leaving the tensor library
    // to report a condition the caller never wrote.
    let tokens = Tensor::from_i64(&[1, 2], &[2, 0]).unwrap();
    let error = embedding.forward(&tokens).unwrap_err();
    assert!(error.to_string().contains("2-D"), "{error}");
}
