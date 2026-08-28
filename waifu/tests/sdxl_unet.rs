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

use std::path::PathBuf;

use waifu::flint::{functional as F, Tensor};
use waifu::{DType, Device, Unet, UnetCondition, UnetConfig, VarBuilder, ZipFile};

fn models_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../models")
}

fn weights() -> VarBuilder {
    let package = ZipFile::open(models_dir().join("sdxl-base.llmpkg")).unwrap();
    VarBuilder::from_reader(
        &mut package.open_entry("model.bin").unwrap(),
        Device::Cuda,
        DType::Float16,
    )
    .unwrap()
}

fn cases() -> VarBuilder {
    let package = ZipFile::open(models_dir().join("sdxl-base_test.llmpkg")).unwrap();
    VarBuilder::from_reader(
        &mut package.open_entry("test_case.bin").unwrap(),
        Device::Cpu,
        DType::Float,
    )
    .unwrap()
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
    timestep: f32,
}

fn inputs(cases: &VarBuilder) -> Inputs {
    let hidden = cases.get_unchecked("test_case.hidden").unwrap();
    let hidden2 = cases.get_unchecked("test_case.hidden2").unwrap();

    let time_ids: Vec<f32> = cases
        .get_unchecked("test_case.time_ids")
        .unwrap()
        .to_vec_f32()
        .unwrap();
    let timestep = cases
        .get_unchecked("test_case.timestep")
        .unwrap()
        .to_vec_i64()
        .unwrap()[0] as f32;

    Inputs {
        latent: to_cuda(&cases.get_unchecked("test_case.latent").unwrap()),
        // The two encoders are conditioned on side by side, 768 and 1280 making the 2048 the
        // cross attention reads.
        context: to_cuda(&F::cat(&hidden, &hidden2, -1).unwrap()),
        pooled: to_cuda(&cases.get_unchecked("test_case.pooled2").unwrap()),
        time_ids: time_ids.try_into().unwrap(),
        timestep,
    }
}

fn condition(inputs: &Inputs) -> UnetCondition<'_> {
    UnetCondition {
        context: &inputs.context,
        pooled: &inputs.pooled,
        time_ids: inputs.time_ids,
    }
}

#[test]
#[ignore = "needs the sdxl package"]
fn one_step_matches_the_reference() {
    let cases = cases();
    let inputs = inputs(&cases);
    let unet = Unet::build(config(), &weights().with_name("sdxl.unet")).unwrap();

    let noise = unet
        .forward(&inputs.latent, inputs.timestep, &condition(&inputs))
        .unwrap();

    // A U-Net answers in the shape it was asked in: this much noise, per latent channel.
    assert_eq!(noise.shape(), inputs.latent.shape());

    let rmse = relative_rmse(&noise, &cases.get_unchecked("test_case.noise").unwrap());
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
    let unet = Unet::build(config(), &weights().with_name("sdxl.unet")).unwrap();

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
    let unet = Unet::build(config(), &weights().with_name("sdxl.unet")).unwrap();

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
                time_ids: inputs.time_ids,
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
    let unet = Unet::build(config(), &weights().with_name("sdxl.unet")).unwrap();

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
    let unet = Unet::build(config(), &weights().with_name("sdxl.unet")).unwrap();

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
    let unet = Unet::build(config(), &weights().with_name("sdxl.unet")).unwrap();

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
