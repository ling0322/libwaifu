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

//! Tests for the parameters of a model: the safetensors they are read out of, and the layers
//! built on top of them. What a model *is* -- its manifest -- is next door in `tests/manifest.rs`.
//!
//! The file is written here by hand rather than with the crate that reads it, so that what these
//! tests say is what safetensors is and not what one library agrees with itself about.

use waifu::flint::{check_parameters, resident, Graph, Ir, ParamSource, Pinned, RunContext};
use waifu::flint::{functional as F, DType, Device, MemorySnapshot, Tensor};
use waifu::{Embedding, Linear, ParamFile};

/// A safetensors file holding `tensors`, each given as a name, a shape, and its elements.
///
/// Eight bytes of header length, that many bytes of JSON saying where each tensor is, and then
/// the tensors one after another. Written out by hand, which is the whole format.
fn write_params(tensors: &[(&str, &[i32], &[f32])]) -> Vec<u8> {
    let mut entries = Vec::new();
    let mut data = Vec::new();

    // Sorted by name, which is what a writer does and what keeps the header stable.
    let mut sorted: Vec<_> = tensors.iter().collect();
    sorted.sort_by_key(|(name, _, _)| *name);

    for (name, shape, values) in sorted {
        let begin = data.len();
        for value in *values {
            data.extend_from_slice(&value.to_le_bytes());
        }

        let dimensions: Vec<String> = shape.iter().map(|size| size.to_string()).collect();
        entries.push(format!(
            "{name:?}:{{\"dtype\":\"F32\",\"shape\":[{}],\"data_offsets\":[{begin},{}]}}",
            dimensions.join(","),
            data.len()
        ));
    }

    let header = format!("{{{}}}", entries.join(","));
    let mut out = (header.len() as u64).to_le_bytes().to_vec();
    out.extend_from_slice(header.as_bytes());
    out.extend_from_slice(&data);
    out
}

fn param_file(tensors: &[(&str, &[i32], &[f32])]) -> ParamFile {
    ParamFile::parse(&write_params(tensors)).unwrap()
}

/// Every tensor of several files at once, which is what a model written as several is read as.
///
/// Counted as well as named after the process, because these tests run beside each other and two
/// of them writing one directory would have each reading what the other wrote.
fn param_files(each: &[Vec<u8>]) -> waifu::Result<ParamFile> {
    static WRITTEN: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    let which = WRITTEN.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

    let dir = std::env::temp_dir().join(format!("waifu-parts-{}-{which}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    let mut paths = Vec::new();
    for (index, bytes) in each.iter().enumerate() {
        let path = dir.join(format!("part-{index}.safetensors"));
        std::fs::write(&path, bytes).unwrap();
        paths.push(path);
    }

    let read = ParamFile::open(&paths);
    let _ = std::fs::remove_dir_all(&dir);
    read
}

#[test]
fn reads_parameters_by_their_whole_names() {
    let file = param_file(&[
        ("sdxl.embd.weight", &[3, 2], &[0.0, 0.1, 1.0, 1.1, 2.0, 2.1]),
        ("sdxl.norm.weight", &[2], &[1.0, 1.0]),
    ]);

    assert_eq!(file.len(), 2);
    assert_eq!(file.names(), vec!["sdxl.embd.weight", "sdxl.norm.weight"]);

    // A file knows whole names and nothing else. Which namespace is being read is the caller's,
    // and for a model it is the graph's: `Graph::subgraph` is what makes a whole name out of the
    // short one a layer asks for.
    assert!(file.has("sdxl.embd.weight"));
    assert!(!file.has("sdxl.embd.bias"));

    let weight = file.get("sdxl.embd.weight", &[3, 2]).unwrap();
    assert_eq!(weight.shape(), vec![3, 2]);
    assert_eq!(
        weight.to_vec_f32().unwrap(),
        vec![0.0, 0.1, 1.0, 1.1, 2.0, 2.1]
    );

    // The shape the caller expects is checked, since the alternative is a failure much later.
    let error = file.get("sdxl.embd.weight", &[2, 3]).unwrap_err();
    assert!(error.to_string().contains("shape"), "{error}");

    let error = file.get("sdxl.embd.bias", &[2]).unwrap_err();
    assert!(error.to_string().contains("sdxl.embd.bias"), "{error}");
}

#[test]
fn refuses_a_parameter_file_it_does_not_understand() {
    let params = write_params(&[("weight", &[2], &[1.0, 2.0])]);

    // A file that stops in the middle of a tensor's data is refused rather than read short: the
    // header says how long each tensor is, and the bytes have to be there.
    let truncated = &params[..params.len() - 4];
    assert!(ParamFile::parse(truncated).is_err());

    // A header that is not JSON at all, and one that claims a length the file does not have.
    assert!(ParamFile::parse(b"not a safetensors file").is_err());
    assert!(ParamFile::parse(&[]).is_err());

    let mut lying = params.clone();
    lying[0] = 0xff;
    assert!(ParamFile::parse(&lying).is_err());
}

/// A pass written out of layers, read out of a package, and run.
///
/// The whole of what building a model is now: the layers say which weights they want and what
/// shape each is, `resident` reads exactly those out of the file, and the compiled pass computes
/// with them. Nothing between the file and the answer is checked anywhere else.
#[test]
fn writes_reads_and_runs_a_pass_of_layers() {
    let file = param_file(&[
        ("embd.weight", &[3, 2], &[0.0, 0.1, 1.0, 1.1, 2.0, 2.1]),
        ("proj.weight", &[2, 2], &[1.0, 0.0, 0.0, 1.0]),
        ("proj.bias", &[2], &[0.5, -0.5]),
    ]);

    // Both layers are written into narrowings of one graph, so their weights are named the way a
    // model's are.
    let g = Graph::new();
    let tokens = g.input("tokens");
    let embedded = Embedding::graph(&g.subgraph("embd"), tokens, 2, 3, DType::Float);
    g.output("embedded", embedded);
    g.output(
        "projected",
        Linear::graph(&g.subgraph("proj"), embedded, 2, 2, true),
    );

    assert_eq!(
        g.parameters(),
        vec!["embd.weight", "proj.weight", "proj.bias"]
    );

    // And exactly those come out of the package, once each however many nodes read them.
    let weights = resident(&file, Device::Cpu).unwrap();
    assert_eq!(weights.len(), 3);

    let ids = Tensor::from_i64(&[2], &[2, 0]).unwrap();
    let context = RunContext::new(&weights).input("tokens", &ids);
    let outputs = Ir::compile(&g).run(&context).unwrap();

    let named = |name: &str| {
        outputs
            .iter()
            .find(|(other, _)| other == name)
            .map(|(_, tensor)| tensor.clone())
            .unwrap()
    };

    let embedded = named("embedded");
    assert_eq!(embedded.shape(), vec![2, 2]);
    assert!(F::all_close(
        &embedded,
        &Tensor::from_f32(&[2, 2], &[2.0, 2.1, 0.0, 0.1]).unwrap()
    )
    .unwrap());

    // An identity with a bias, so the projection is the embedding plus the bias.
    assert!(F::all_close(
        &named("projected"),
        &Tensor::from_f32(&[2, 2], &[2.5, 1.6, 0.5, -0.4]).unwrap()
    )
    .unwrap());
}

/// A weight two layers read is read once.
///
/// What tied weights look like, and the reason `resident` walks the graph with an `entry` rather
/// than reading every `load` node it meets.
#[test]
fn a_weight_read_twice_is_read_once() {
    let file = param_file(&[("shared.weight", &[2, 2], &[1.0, 0.0, 0.0, 1.0])]);

    let g = Graph::new();
    let x = g.input("x");
    let once = Linear::graph(&g.subgraph("shared"), x, 2, 2, false);
    let twice = Linear::graph(&g.subgraph("shared"), once, 2, 2, false);
    g.output("y", twice);

    // Two nodes name it, and it is one weight.
    assert_eq!(g.parameters(), vec!["shared.weight"]);
    assert_eq!(resident(&file, Device::Cpu).unwrap().len(), 1);
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
    let first = write_params(&[("a", &[2], &[1.0, 2.0])]);
    let second = write_params(&[("b", &[2], &[3.0, 4.0])]);

    let file = param_files(&[first, second]).unwrap();
    assert_eq!(file.names(), vec!["a", "b"]);
    assert_eq!(
        file.get("a", &[2]).unwrap().to_vec_f32().unwrap(),
        vec![1.0, 2.0]
    );
    assert_eq!(
        file.get("b", &[2]).unwrap().to_vec_f32().unwrap(),
        vec![3.0, 4.0]
    );
}

#[test]
fn refuses_a_tensor_that_is_in_two_parts_at_once() {
    // Which part won would otherwise depend on the order they were listed in, and a model that
    // loads differently depending on that is worse than one that refuses to load.
    let first = write_params(&[("a", &[2], &[1.0, 2.0])]);
    let second = write_params(&[("a", &[2], &[3.0, 4.0])]);

    let error = param_files(&[first, second]).unwrap_err().to_string();
    assert!(error.contains("more than one file"), "reported as {error}");
}

#[test]
fn a_tensor_is_handed_over_rather_than_read_again() {
    // The parameters are read in one pass, when the file is opened, and what comes out of it
    // afterwards is what was in it. A tensor is handed over as another handle on the storage the
    // file holds, so asking for the same one twice is the same answer twice and costs nothing the
    // second time.
    let file = param_file(&[("a", &[2], &[1.0, 2.0]), ("b", &[2], &[3.0, 4.0])]);

    assert_eq!(file.names(), vec!["a", "b"]);
    assert_eq!(file.len(), 2);

    let first = file.get("a", &[2]).unwrap().to_vec_f32().unwrap();
    let again = file.get("a", &[2]).unwrap().to_vec_f32().unwrap();
    assert_eq!(first, vec![1.0, 2.0]);
    assert_eq!(first, again);
    assert_eq!(
        file.get("b", &[2]).unwrap().to_vec_f32().unwrap(),
        vec![3.0, 4.0]
    );
}

fn to_host(tensor: &Tensor) -> Vec<f32> {
    tensor.to_device(Device::Cpu).unwrap().to_vec_f32().unwrap()
}

/// The same pass, once with the weights on the card and once with them page-locked on the host.
///
/// What a low-vram run promises is the picture, not a cheaper picture: the same instructions read
/// the same weights, and all that changed is where a weight was sitting when the `load` asked for
/// it. So the two are compared for equality rather than to a tolerance.
#[test]
#[ignore = "needs a CUDA device"]
fn a_low_vram_run_computes_what_a_resident_one_does() {
    let tensors: &[(&str, &[i32], &[f32])] = &[("proj.weight", &[2, 2], &[0.25, -1.5, 2.0, 0.75])];

    let graph = Graph::new();
    let x = graph.input("x");
    let weight = graph.load("proj.weight", &[2, 2]);
    graph.output("y", graph.add(x, graph.matmul(x, weight)));
    let ir = Ir::compile(&graph);

    let x = Tensor::from_f32(&[2, 2], &[1.0, -2.0, 0.5, 3.0])
        .unwrap()
        .to_device(Device::Cuda)
        .unwrap();

    let on_the_card = resident(&param_file(tensors), Device::Cuda).unwrap();
    let kept = ir
        .run(&RunContext::new(&on_the_card).input("x", &x))
        .unwrap();

    let over_the_bus = Pinned::read(param_file(tensors), Device::Cuda).unwrap();
    assert_eq!(over_the_bus.len(), 1);
    let streamed = ir
        .run(&RunContext::new(&over_the_bus).input("x", &x))
        .unwrap();

    assert_eq!(to_host(&kept[0].1), to_host(&streamed[0].1));

    // Again off the same source, because a low-vram run is the one that reads its weights over and
    // over: twenty sampler steps are twenty copies of the model, and every one of them has to be
    // the same weight.
    let again = ir
        .run(&RunContext::new(&over_the_bus).input("x", &x))
        .unwrap();
    assert_eq!(to_host(&streamed[0].1), to_host(&again[0].1));
}

/// What low-vram is actually for: the card never holds the model.
#[test]
#[ignore = "needs a CUDA device"]
fn a_low_vram_source_leaves_the_weights_off_the_card() {
    // Large enough that there is no arguing with the numbers: 256 KB of weight against an
    // allocator that rounds.
    let values = vec![0.5f32; 256 * 256];
    let tensors: &[(&str, &[i32], &[f32])] = &[("big.weight", &[256, 256], &values)];
    let bytes = (values.len() * std::mem::size_of::<f32>()) as i64;

    let graph = Graph::new();
    let x = graph.input("x");
    let weight = graph.load("big.weight", &[256, 256]);
    graph.output("y", graph.matmul(x, weight));

    let allocated = || MemorySnapshot::capture(Device::Cuda).unwrap().allocated;

    let before = allocated();
    let pinned = Pinned::read(param_file(tensors), Device::Cuda).unwrap();
    assert_eq!(
        allocated(),
        before,
        "reading a package for a low-vram run puts nothing on the card"
    );

    // And neither does checking the model against it, which is the whole reason `check` is a
    // question of its own rather than a `load` whose answer is dropped.
    check_parameters(&graph, &pinned).unwrap();
    assert_eq!(allocated(), before, "checking a model must not move it");

    // The load is where the bytes cross, and they land on the card.
    let loaded = pinned.load("big.weight", &[256, 256]).unwrap();
    assert_eq!(loaded.device(), Device::Cuda);
    assert!(allocated() - before >= bytes, "the weight is on the card");

    // And they are gone again once nothing holds them, which is what the `free` instruction after
    // each `load` does inside a run.
    drop(loaded);
    assert_eq!(allocated(), before);

    // Against which: reading the same package the resident way does put it there, and keeps it.
    let kept = resident(&param_file(tensors), Device::Cuda).unwrap();
    assert!(allocated() - before >= bytes);
    drop(kept);
}
