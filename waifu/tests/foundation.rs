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

use std::collections::HashMap;
use std::rc::Rc;

use waifu::flint::{check_parameters, Graph, Ir, ParamSource, Residency, RunContext, Weights};
use waifu::flint::{functional as F, DType, Device, MemorySnapshot, Tensor};
use waifu::{parse_safetensors, read_safetensors, Embedding, Linear, WeightFormat};

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

/// What a file holding `tensors` reads back as, exactly as stored.
fn stored(tensors: &[(&str, &[i32], &[f32])]) -> HashMap<String, Tensor> {
    parse_safetensors(&write_params(tensors)).unwrap()
}

/// A source read out of a file holding `tensors`, the way a model reads its package.
fn source(
    tensors: &[(&str, &[i32], &[f32])],
    device: Device,
    residency: Residency,
) -> Rc<dyn ParamSource> {
    Rc::new(Weights::from_bytes(&write_params(tensors), device, residency).unwrap())
}

/// The same, for a file whose tensors are not all one element type: each is given the name
/// safetensors knows its type by and the bytes it is stored as.
///
/// Which is what a quantized weight needs -- `F8_E4M3` elements and an `F32` scale in one file --
/// and hand written for the same reason the rest of this is: so that what is under test is the
/// format rather than one library agreeing with itself.
fn write_typed(tensors: &[(&str, &str, &[i32], Vec<u8>)]) -> Vec<u8> {
    let mut entries = Vec::new();
    let mut data = Vec::new();

    let mut sorted: Vec<_> = tensors.iter().collect();
    sorted.sort_by_key(|(name, _, _, _)| *name);

    for (name, dtype, shape, bytes) in sorted {
        let begin = data.len();
        data.extend_from_slice(bytes);

        let dimensions: Vec<String> = shape.iter().map(|size| size.to_string()).collect();
        entries.push(format!(
            "{name:?}:{{\"dtype\":{dtype:?},\"shape\":[{}],\"data_offsets\":[{begin},{}]}}",
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

/// The `f32` elements of a tensor, as a file holds them.
fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// Every tensor of several files at once, which is what a model written as several is read as.
///
/// Counted as well as named after the process, because these tests run beside each other and two
/// of them writing one directory would have each reading what the other wrote.
fn param_files(each: &[Vec<u8>]) -> waifu::Result<HashMap<String, Tensor>> {
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

    let read = read_safetensors(&paths);
    let _ = std::fs::remove_dir_all(&dir);
    read
}

#[test]
fn reads_parameters_by_their_whole_names() {
    let file = stored(&[
        ("sdxl.embd.weight", &[3, 2], &[0.0, 0.1, 1.0, 1.1, 2.0, 2.1]),
        ("sdxl.norm.weight", &[2], &[1.0, 1.0]),
    ]);

    assert_eq!(file.len(), 2);
    assert_eq!(names(&file), vec!["sdxl.embd.weight", "sdxl.norm.weight"]);

    // A file knows whole names and nothing else. Which namespace is being read is the caller's,
    // and for a model it is the graph's: `Graph::subgraph` is what makes a whole name out of the
    // short one a layer asks for.
    assert!(file.contains_key("sdxl.embd.weight"));
    assert!(!file.contains_key("sdxl.embd.bias"));

    let weight = file.load("sdxl.embd.weight", &[3, 2]).unwrap();
    assert_eq!(weight.shape(), vec![3, 2]);
    assert_eq!(
        weight.to_vec_f32().unwrap(),
        vec![0.0, 0.1, 1.0, 1.1, 2.0, 2.1]
    );

    // The shape the caller expects is checked, since the alternative is a failure much later.
    let error = file.load("sdxl.embd.weight", &[2, 3]).unwrap_err();
    assert!(error.to_string().contains("shape"), "{error}");

    let error = file.load("sdxl.embd.bias", &[2]).unwrap_err();
    assert!(error.to_string().contains("sdxl.embd.bias"), "{error}");
}

#[test]
fn refuses_a_parameter_file_it_does_not_understand() {
    let params = write_params(&[("weight", &[2], &[1.0, 2.0])]);

    // A file that stops in the middle of a tensor's data is refused rather than read short: the
    // header says how long each tensor is, and the bytes have to be there.
    let truncated = &params[..params.len() - 4];
    assert!(parse_safetensors(truncated).is_err());

    // A header that is not JSON at all, and one that claims a length the file does not have.
    assert!(parse_safetensors(b"not a safetensors file").is_err());
    assert!(parse_safetensors(&[]).is_err());

    let mut lying = params.clone();
    lying[0] = 0xff;
    assert!(parse_safetensors(&lying).is_err());
}

/// A pass written out of layers, read out of a package, and run.
///
/// The whole of what building a model is now: the layers say which weights they want and what
/// shape each is, the package holds them under those names, and the compiled pass computes
/// with them. Nothing between the file and the answer is checked anywhere else.
#[test]
fn writes_reads_and_runs_a_pass_of_layers() {
    let file = stored(&[
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
    assert_eq!(file.len(), 3);

    let ids = Tensor::from_i64(&[2], &[2, 0]).unwrap();
    let context = RunContext::new(&file).input("tokens", &ids);
    let ir = Ir::compile(&g);
    let outputs = ir.run(&context).unwrap();

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

/// A projection whose weight the package stored quantized is built out of two loads and one node.
///
/// No device runs this one: FP8 is the card's for now, and what a machine without one can still be
/// held to is the shape of the pass. Which is most of what storing a weight quantized changes --
/// two tensors in the file become two loads in the graph and three operands at the multiply -- and
/// it is the half that has nothing to do with a kernel.
#[test]
fn builds_a_quantized_projection_out_of_two_loads() {
    let written = |format| {
        let g = Graph::with_weights(format);
        let x = g.input("x");
        g.output("y", Linear::graph(&g.subgraph("proj"), x, 4, 2, true));

        (g.parameters(), format!("{g}"))
    };

    // The scale is a weight like any other, so it is named among them and read like them.
    let (asked, pass) = written(WeightFormat::Fp8);
    assert_eq!(asked, vec!["proj.weight", "proj.weight.scale", "proj.bias"]);
    assert!(pass.contains("fp8_matmul("), "{pass}");

    // And no transpose before it: an FP8 multiply reads the weight in the (out, in) a package
    // stores, where the float one has to be handed the other way round.
    assert!(!pass.contains("transpose("), "{pass}");

    let (asked, pass) = written(WeightFormat::Float);
    assert_eq!(asked, vec!["proj.weight", "proj.bias"]);
    assert!(pass.contains("transpose("), "{pass}");
    assert!(!pass.contains("fp8_matmul("), "{pass}");
}

/// And on the only device that has the kernels, it computes what it should.
///
/// The whole path: two tensors in the file, across the bus as `<fp8e4m3>` -- something no weight
/// has ever done here, since FP8 used to be made on the device it was used on -- and into the
/// CUTLASS mixed input multiply. Every code and every scale is a value E4M3 and float16 both hold
/// exactly, so what comes back is compared to the arithmetic rather than to a tolerance.
///
/// What must not differ is the scales, which stay `<float>`: reading a model narrows a per-channel
/// vector to what the device computes in, and a scale is the one such vector that would be wrong
/// to narrow.
#[test]
#[ignore = "needs a CUDA device"]
fn multiplies_by_a_quantized_weight_on_the_card() {
    const ONE: u8 = 0x38;

    // Eight rows and sixteen columns, which is what the kernel can read: the row count has to be
    // a multiple of 8 and k a multiple of 16. Every element is a one, so a row is worth its scale.
    let scales: Vec<f32> = (1..=8).map(|r| r as f32 * 0.5).collect();
    let bytes = write_typed(&[
        ("proj.weight", "F8_E4M3", &[8, 16], vec![ONE; 8 * 16]),
        ("proj.weight.scale", "F32", &[8], f32_bytes(&scales)),
    ]);

    let g = Graph::with_weights(WeightFormat::Fp8);
    let x = g.input("x");
    g.output("y", Linear::graph(&g.subgraph("proj"), x, 16, 8, false));

    let weights = Weights::from_bytes(&bytes, Device::Cuda, Residency::Device).unwrap();
    assert_eq!(
        weights.load("proj.weight", &[8, 16]).unwrap().dtype(),
        DType::Fp8E4M3,
        "the elements are read as themselves"
    );
    assert_eq!(
        weights.load("proj.weight.scale", &[8]).unwrap().dtype(),
        DType::Float,
        "a scale is not narrowed to what the device computes in"
    );

    let x = Tensor::from_f32(&[1, 16], &[1.0; 16])
        .unwrap()
        .to_device(Device::Cuda)
        .unwrap()
        .cast(DType::Float16)
        .unwrap();
    let ir = Ir::compile(&g);
    let outputs = ir.run(&RunContext::new(&weights).input("x", &x)).unwrap();

    // Sixteen ones against a row worth its scale, and every one of these is exact in float16.
    let expected: Vec<f32> = scales.iter().map(|scale| scale * 16.0).collect();
    let y = outputs[0].1.cast(DType::Float).unwrap();
    assert_eq!(to_host(&y), expected);
}

/// What the graph calls the scales and what the file reader looks for are one name.
#[test]
fn the_scale_is_named_the_same_by_both_halves() {
    assert_eq!(
        Linear::WEIGHT_SCALE,
        format!("{}{}", Linear::WEIGHT, waifu::flint::CHANNEL_SCALE_SUFFIX)
    );
}

/// Quantized elements with no scales beside them are refused where they are read.
///
/// The one thing a file cannot say for itself: safetensors has no way to tie two tensors together,
/// so the pairing is a name, and a name is only a convention until something checks it.
#[test]
fn refuses_quantized_elements_with_no_scales_beside_them() {
    let error = parse_safetensors(&write_typed(&[(
        "proj.weight",
        "F8_E4M3",
        &[2, 4],
        vec![0x38; 8],
    )]))
    .unwrap_err();
    assert!(error.to_string().contains("proj.weight.scale"), "{error}");

    // One per row, and a float: a scale of the wrong shape would be read as the wrong rows'.
    let error = parse_safetensors(&write_typed(&[
        ("proj.weight", "F8_E4M3", &[2, 4], vec![0x38; 8]),
        ("proj.weight.scale", "F32", &[4], f32_bytes(&[1.0; 4])),
    ]))
    .unwrap_err();
    assert!(error.to_string().contains("one per row"), "{error}");

    let error = parse_safetensors(&write_typed(&[
        ("proj.weight", "F8_E4M3", &[2, 4], vec![0x38; 8]),
        ("proj.weight.scale", "F16", &[2], vec![0; 4]),
    ]))
    .unwrap_err();
    assert!(error.to_string().contains("one per row"), "{error}");

    // A tensor whose name happens to end in the suffix is nobody's scale and stays an ordinary
    // weight: the suffix is this library's convention, not a word reserved in the format.
    let file = stored(&[("gain.scale", &[2], &[1.0, 2.0])]);
    assert!(file.contains_key("gain.scale"));
}

/// A weight two layers read is read once.
///
/// What tied weights look like: the graph holds one `load` for a weight however many layers name
/// it, so a pass asks its source for it once.
#[test]
fn a_weight_read_twice_is_read_once() {
    let file = stored(&[("shared.weight", &[2, 2], &[1.0, 0.0, 0.0, 1.0])]);

    let g = Graph::new();
    let x = g.input("x");
    let once = Linear::graph(&g.subgraph("shared"), x, 2, 2, false);
    let twice = Linear::graph(&g.subgraph("shared"), once, 2, 2, false);
    g.output("y", twice);

    // Two nodes name it, and it is one weight.
    assert_eq!(g.parameters(), vec!["shared.weight"]);
    assert_eq!(file.len(), 1);

    // And it is loaded once: the graph holds one `load` for it however many nodes name it, so a
    // pass asks the source for it once -- which, from a `Pinned` source, is one trip over the bus.
    let ir = Ir::compile(&g);
    let loads = ir
        .insts()
        .iter()
        .filter(|inst| {
            matches!(
                inst,
                waifu::flint::Inst::Run {
                    op: waifu::flint::Op::Load { .. },
                    ..
                }
            )
        })
        .count();

    assert_eq!(loads, 1);
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
    assert_eq!(names(&file), vec!["a", "b"]);
    assert_eq!(
        file.load("a", &[2]).unwrap().to_vec_f32().unwrap(),
        vec![1.0, 2.0]
    );
    assert_eq!(
        file.load("b", &[2]).unwrap().to_vec_f32().unwrap(),
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
fn a_source_answers_for_a_shape_without_handing_the_weight_over() {
    // A model that works out its own architecture by looking for weights -- Anima's autoencoder
    // counts its stages this way -- asks this hundreds of times, and from a low-vram source
    // handing the weight over would be a copy across the bus each time.
    let weights = source(&[("a", &[2, 3], &[0.0; 6])], Device::Cpu, Residency::Device);

    assert_eq!(weights.shape_of("a"), Some(vec![2, 3]));
    assert_eq!(weights.shape_of("missing"), None);
    assert!(weights.has("a"));
    assert!(!weights.has("missing"));
}

/// The names a stored map holds, sorted, which is the order a reader of a failure wants.
fn names(tensors: &HashMap<String, Tensor>) -> Vec<&str> {
    let mut names: Vec<&str> = tensors.keys().map(String::as_str).collect();
    names.sort_unstable();
    names
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

    let x = Tensor::from_f32(&[2, 2], &[1.0, -2.0, 0.5, 3.0])
        .unwrap()
        .to_device(Device::Cuda)
        .unwrap();

    // One graph, one IR, and two sources. What the two modes differ by is what a load is answered
    // out of -- a tensor already on the card, or one page-locked on the host and copied across --
    // and nothing the IR says.
    let ir = Ir::compile(&graph);

    let on_the_card = source(tensors, Device::Cuda, Residency::Device);
    let kept = ir
        .run(&RunContext::new(on_the_card.as_ref()).input("x", &x))
        .unwrap();

    let over_the_bus = source(tensors, Device::Cuda, Residency::LowVram);
    assert!(over_the_bus.has("proj.weight"));

    let streamed = ir
        .run(&RunContext::new(over_the_bus.as_ref()).input("x", &x))
        .unwrap();
    assert_eq!(to_host(&kept[0].1), to_host(&streamed[0].1));

    // Again off the same source, because a low-vram run is the one that copies its weights over
    // and over: twenty sampler steps are twenty copies of the model, and every one of them has to
    // be the same weight.
    let again = ir
        .run(&RunContext::new(over_the_bus.as_ref()).input("x", &x))
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

    // Read the way a model reads it, through `Residency`. The whole package is page-locked on the
    // host now, and none of it is on the card.
    let before = allocated();
    let pinned = source(tensors, Device::Cuda, Residency::LowVram);
    assert_eq!(
        allocated(),
        before,
        "reading a package for a low-vram run puts nothing on the card"
    );

    // And neither does checking the model against it, which is the whole reason `check` is a
    // question of its own rather than a load whose answer is dropped.
    check_parameters(&graph, pinned.as_ref()).unwrap();
    assert_eq!(allocated(), before, "checking a model must not move it");

    let ir = Ir::compile(&graph);
    assert_eq!(allocated(), before, "compiling a model moves nothing");

    // The pass is where the bytes cross. They land on the card, and they are gone again by the
    // time it ends: the `free` the compiler put after the weight's last reader is the card taking
    // them back.
    let x = Tensor::zeros(&[256, 256], DType::Float, Device::Cuda).unwrap();
    let out = ir
        .run(&RunContext::new(pinned.as_ref()).input("x", &x))
        .unwrap();
    drop(out);
    assert_eq!(allocated() - before, bytes_of(&x), "only the input is left");

    // Against which: the same package read to keep its weights does put the model on the card as
    // it is read, and holds it there.
    let kept = source(tensors, Device::Cuda, Residency::Device);

    assert!(allocated() - before >= bytes);
    drop(kept);
}

/// Asking a device to hand its spare memory back is a question every device answers, including
/// the one that has nothing to answer it with.
///
/// The webui calls this wherever it takes a model down, with whatever device the run was going to
/// -- so a session on the CPU asks it too, and an error there would be a red line on the page
/// about a card that is not in the machine.
#[test]
fn a_device_that_holds_nothing_back_is_still_asked_to_hand_it_over() {
    MemorySnapshot::release_unused(Device::Cpu).unwrap();
}

/// Letting go of a tensor is not letting go of the card, and what is.
#[test]
#[ignore = "needs a CUDA device"]
fn handing_the_memory_back_is_a_second_thing_after_dropping_it() {
    // Half a gigabyte, against a number that everything else running on the machine also moves:
    // the free memory of a card is shared, so what is asserted below is the size of this test's
    // own tensor with room to spare rather than an exact figure.
    const BYTES: i64 = 512 * 1024 * 1024;
    let free = || MemorySnapshot::capture(Device::Cuda).unwrap().free;

    // From a pool holding nothing, so that the first measurement is not of what an earlier test
    // left in it.
    MemorySnapshot::release_unused(Device::Cuda).unwrap();
    let before = free();

    let big = Tensor::zeros(&[8192, 16384], DType::Float, Device::Cuda).unwrap();
    assert_eq!(bytes_of(&big), BYTES);
    drop(big);

    // The tensor is gone and the card is still full. This is the allocator doing what it was told
    // at startup -- keep what is freed, so the next run of this size asks it rather than the
    // driver -- and it is why a model dropped on a stop is not yet a card anybody else can have.
    assert!(
        before - free() >= BYTES / 2,
        "a dropped tensor leaves its bytes in the pool"
    );

    // And this is the call that ends it.
    MemorySnapshot::release_unused(Device::Cuda).unwrap();
    assert!(
        free() >= before - BYTES / 2,
        "handing them back is what the rest of the machine can see"
    );
}

/// How much of the card one tensor is holding, which the input above is and the weights are not.
fn bytes_of(tensor: &Tensor) -> i64 {
    tensor.nbytes().unwrap()
}
