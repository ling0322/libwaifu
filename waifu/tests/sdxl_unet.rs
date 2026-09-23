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

//! One U-Net step against what diffusers makes of the same latent, timestep and prompt.
//!
//! This is the model: 2.6 billion of the package's 3.5 billion parameters, and the only part a
//! sampler calls more than once. Everything upstream of it is checked by its own test, so a
//! disagreement here is a disagreement about the U-Net.

use std::cell::OnceCell;
use std::collections::HashMap;
use std::path::PathBuf;
use std::rc::Rc;

use waifu::flint::{functional as F, ParamSource, Tensor, Weights};
use waifu::{
    read_safetensors, DType, Device, Manifest, Residency, Unet, UnetCondition, UnetConfig,
    WeightFormat,
};

fn models_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../models")
}

/// The whole package on the device, read once for this whole test binary.
///
/// The package is read whole rather than a model's part of it, so reading it per test would read
/// seven gigabytes as many times as there are tests here.
fn weights() -> Rc<dyn ParamSource> {
    thread_local! {
        static WEIGHTS: OnceCell<Rc<dyn ParamSource>> = const { OnceCell::new() };
    }

    WEIGHTS.with(|cell| {
        Rc::clone(cell.get_or_init(|| {
            let manifest = Manifest::open(models_dir().join("sdxl-base.yaml")).unwrap();
            Rc::new(
                Weights::from_files(
                    &manifest.weight_paths().unwrap(),
                    Device::Cuda,
                    Residency::Device,
                )
                .unwrap(),
            )
        }))
    })
}

fn cases() -> HashMap<String, Tensor> {
    read_safetensors(&[models_dir().join("sdxl-base_test.safetensors")]).unwrap()
}

fn config() -> UnetConfig {
    UnetConfig {
        latent_channels: 4,
        block_out_channels: vec![320, 640, 1280],
        layers_per_block: 2,
        transformer_layers_per_block: vec![0, 2, 10],
        num_heads: vec![5, 10, 20],
        norm_num_groups: 32,
        cross_attention_dim: 2048,
        addition_time_embed_dim: 256,
        projection_class_embeddings_input_dim: 2816,
        weight_format: WeightFormat::Float
    }
}

fn to_cuda(tensor: &Tensor) -> Tensor {
    tensor
        .to_device(Device::Cuda)
        .unwrap()
        .cast(DType::Float16)
        .unwrap()
}

fn relative_rmse(actual: &Tensor, reference: &Tensor) -> f32 {
    let a = actual
        .to_device(Device::Cpu)
        .unwrap()
        .cast(DType::Float)
        .unwrap()
        .to_vec_f32()
        .unwrap();
    let b = reference
        .to_device(Device::Cpu)
        .unwrap()
        .cast(DType::Float)
        .unwrap()
        .to_vec_f32()
        .unwrap();
    assert_eq!(a.len(), b.len(), "the shapes do not match");

    let error: f64 = a
        .iter()
        .zip(&b)
        .map(|(x, y)| (*x as f64 - *y as f64).powi(2))
        .sum();
    let scale: f64 = b.iter().map(|y| (*y as f64).powi(2)).sum();
    (error / scale).sqrt() as f32
}

/// Everything one step needs, read from the reference package.
struct Inputs {
    latent: Tensor,
    context: Tensor,
    pooled: Tensor,
    time_ids: [f32; 6],
    timestep: f32
}

fn inputs(cases: &HashMap<String, Tensor>) -> Inputs {
    let hidden = cases["test_case.hidden"].clone();
    let hidden2 = cases["test_case.hidden2"].clone();

    let time_ids: Vec<f32> = cases["test_case.time_ids"].to_vec_f32().unwrap();
    let timestep = cases["test_case.timestep"].to_vec_i64().unwrap()[0] as f32;

    Inputs {
        latent: to_cuda(&cases["test_case.latent"]),
        // The two encoders are conditioned on side by side, 768 and 1280 making the 2048 the
        // cross attention reads.
        context: to_cuda(&F::cat(&hidden, &hidden2, -1).unwrap()),
        pooled: to_cuda(&cases["test_case.pooled2"]),
        time_ids: time_ids.try_into().unwrap(),
        timestep
    }
}

fn condition(inputs: &Inputs) -> UnetCondition<'_> {
    UnetCondition {
        context: &inputs.context,
        pooled: &inputs.pooled,
        time_ids: inputs.time_ids
    }
}

#[test]
#[ignore = "needs the sdxl package"]
fn one_step_matches_the_reference() {
    let cases = cases();
    let inputs = inputs(&cases);
    let unet = Unet::build(config(), "sdxl.unet", &weights()).unwrap();

    let noise = unet
        .forward(&inputs.latent, inputs.timestep, &condition(&inputs))
        .unwrap();

    // A U-Net answers in the shape it was asked in: this much noise, per latent channel.
    assert_eq!(noise.shape(), inputs.latent.shape());

    let rmse = relative_rmse(&noise, &cases["test_case.noise"]);
    println!("unet rmse = {rmse}");
    assert!(rmse < 2e-2, "one step drifted by {rmse}");
}

#[test]
#[ignore = "needs the sdxl package"]
fn the_timestep_changes_the_answer() {
    // Half of what a U-Net does is know how noisy its input is. A model that ignored the timestep
    // would still pass a single comparison if the reference happened to be taken at that step, so
    // this says the answer moves when the step does -- and by much more than half precision does.
    let cases = cases();
    let inputs = inputs(&cases);
    let unet = Unet::build(config(), "sdxl.unet", &weights()).unwrap();

    let early = unet
        .forward(&inputs.latent, 999.0, &condition(&inputs))
        .unwrap();
    let late = unet
        .forward(&inputs.latent, 1.0, &condition(&inputs))
        .unwrap();

    let difference = relative_rmse(&early, &late);
    println!("timestep difference = {difference}");
    assert!(
        difference > 0.1,
        "the timestep moved the answer by only {difference}"
    );
}

#[test]
#[ignore = "needs the sdxl package"]
fn the_prompt_changes_the_answer() {
    // The other half is the prompt, which arrives by a different road entirely: cross attention
    // rather than an addition. An unconditional step is what classifier free guidance subtracts,
    // so this is a shape the sampler will really ask for.
    let cases = cases();
    let inputs = inputs(&cases);
    let unet = Unet::build(config(), "sdxl.unet", &weights()).unwrap();

    let prompted = unet
        .forward(&inputs.latent, inputs.timestep, &condition(&inputs))
        .unwrap();

    let empty = Tensor::zeros(&inputs.context.shape(), DType::Float16, Device::Cuda).unwrap();
    let unprompted = unet
        .forward(
            &inputs.latent,
            inputs.timestep,
            &UnetCondition {
                context: &empty,
                pooled: &inputs.pooled,
                time_ids: inputs.time_ids
            },
        )
        .unwrap();

    // A few percent, which is small next to what the timestep does but two orders of magnitude
    // above the 5e-4 this model agrees with itself to. A U-Net that dropped the conditioning on
    // the floor would land at zero here and still pass every other test in this file.
    let difference = relative_rmse(&prompted, &unprompted);
    println!("prompt difference = {difference}");
    assert!(
        difference > 0.01,
        "the prompt moved the answer by only {difference}"
    );
}

#[test]
#[ignore = "needs the sdxl package"]
fn gives_the_same_answer_twice() {
    let cases = cases();
    let inputs = inputs(&cases);
    let unet = Unet::build(config(), "sdxl.unet", &weights()).unwrap();

    let first = unet
        .forward(&inputs.latent, inputs.timestep, &condition(&inputs))
        .unwrap();
    let second = unet
        .forward(&inputs.latent, inputs.timestep, &condition(&inputs))
        .unwrap();

    assert_eq!(relative_rmse(&first, &second), 0.0);
}

#[test]
#[ignore = "needs the sdxl package"]
fn works_at_another_latent_size() {
    // Nothing in a U-Net is tied to one resolution, and every size is a different number of
    // positions for the transformer blocks to attend over.
    let cases = cases();
    let inputs = inputs(&cases);
    let unet = Unet::build(config(), "sdxl.unet", &weights()).unwrap();

    for (height, width) in [(16, 16), (16, 24)] {
        let latent = inputs
            .latent
            .slice(2, 0, height)
            .unwrap()
            .slice(3, 0, width)
            .unwrap()
            .contiguous()
            .unwrap();

        let noise = unet
            .forward(&latent, inputs.timestep, &condition(&inputs))
            .unwrap();
        assert_eq!(noise.shape(), vec![1, 4, height, width]);

        let values = noise
            .to_device(Device::Cpu)
            .unwrap()
            .cast(DType::Float)
            .unwrap()
            .to_vec_f32()
            .unwrap();
        assert!(
            values.iter().all(|x| x.is_finite()),
            "a {height} by {width} latent gave a NaN"
        );
    }
}

#[test]
#[ignore = "needs the sdxl package"]
fn refuses_a_latent_it_cannot_read() {
    let cases = cases();
    let inputs = inputs(&cases);
    let unet = Unet::build(config(), "sdxl.unet", &weights()).unwrap();

    let three_d = inputs.latent.squeeze(0).unwrap();
    assert!(unet
        .forward(&three_d, inputs.timestep, &condition(&inputs))
        .is_err());

    // Three levels means halving twice, so a size that is not a multiple of four would come back
    // from the way up smaller than it went in.
    let odd = inputs.latent.slice(2, 0, 30).unwrap().contiguous().unwrap();
    assert!(unet
        .forward(&odd, inputs.timestep, &condition(&inputs))
        .is_err());
}

/// The pass can be read before it is run, which is the whole of what writing the model down
/// rather than building it bought.
///
/// The eager U-Net could be asked what layers it held; it could not be asked what it does, in
/// order, or which weight each step reads. This can.
#[test]
#[ignore = "needs the sdxl package"]
fn the_pass_can_be_read_before_it_is_run() {
    let unet = Unet::build(config(), "sdxl.unet", &weights()).unwrap();
    let listing = unet.ir().to_string();

    // Every weight is asked for by the whole name and the shape the layer expects, which is what
    // makes a package that does not match say so instead of aborting somewhere in a kernel.
    assert!(
        listing.contains(r#"load("sdxl.unet.down1.resnet0.conv1.weight", [640, 320, 3, 3])"#),
        "{listing}"
    );

    // Cross attention fuses the keys and values but not the query, so its one weight is twice the
    // width of the level and as deep as the two text encoders side by side.
    assert!(
        listing
            .contains(r#"load("sdxl.unet.mid.attn0.block0.attn2.kv_proj.weight", [2560, 2048])"#),
        "{listing}"
    );

    // Nothing in it was written for one latent size. The transformers fold the height and width
    // together to read the image as a sequence, and that is the only arithmetic on a size here.
    assert!(listing.contains("dims(%"), "{listing}");

    // Nothing in it that nothing reads: one free apiece for everything but the one output.
    let (runs, frees): (Vec<&str>, Vec<&str>) = listing
        .lines()
        .map(str::trim)
        .filter(|line| line.starts_with('%') || line.starts_with("free "))
        .partition(|line| line.starts_with('%'));
    assert_eq!(frees.len(), runs.len() - 1);
}

/// Every weight the package holds under the U-Net is one the pass reads.
///
/// The tripwire the layer walk used to be: a block that quietly stopped writing part of itself
/// down would leave weights behind, and nothing else here would notice -- the answer would just
/// be wrong by however much that block was worth.
#[test]
#[ignore = "needs the sdxl package"]
fn the_pass_reads_every_weight_the_package_holds() {
    let unet = Unet::build(config(), "sdxl.unet", &weights()).unwrap();
    let listing = unet.ir().to_string();

    let manifest = Manifest::open(models_dir().join("sdxl-base.yaml")).unwrap();
    let file = read_safetensors(&manifest.weight_paths().unwrap()).unwrap();

    let mut held: Vec<&str> = file
        .keys()
        .map(String::as_str)
        .filter(|name| name.starts_with("sdxl.unet."))
        .collect();
    held.sort_unstable();
    assert!(!held.is_empty());

    let unread: Vec<&&str> = held
        .iter()
        .filter(|name| !listing.contains(&format!("load({name:?}")))
        .collect();
    assert!(unread.is_empty(), "the pass never reads {unread:?}");
}
