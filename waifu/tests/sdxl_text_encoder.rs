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

//! Both of SDXL's text encoders against what huggingface produces for the same prompt.
//!
//! The reference is what conditions everything downstream: the second to last hidden state of
//! each encoder, and the pooled vector the second one projects out of its end-of-text position.
//! Getting these wrong is not something an image would show clearly, so they are compared as
//! numbers rather than looked at.

use std::path::PathBuf;

use waifu::flint::{functional as F, Tensor};
use waifu::{ClipTextConfig, ClipTextEncoder, DType, Device, VarBuilder, ZipFile};

fn models_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../models")
}

fn device() -> Device {
    Device::Cuda
}

fn weights() -> VarBuilder {
    let package = ZipFile::open(models_dir().join("sdxl-base.waifupkg")).unwrap();
    VarBuilder::from_reader(
        &mut package.open_entry("model.bin").unwrap(),
        device(),
        DType::Float16,
    )
    .unwrap()
}

fn cases() -> VarBuilder {
    let package = ZipFile::open(models_dir().join("sdxl-base_test.waifupkg")).unwrap();
    VarBuilder::from_reader(
        &mut package.open_entry("test_case.bin").unwrap(),
        Device::Cpu,
        DType::Float,
    )
    .unwrap()
}

fn to_cpu_f32(x: &Tensor) -> Tensor {
    x.to_device(Device::Cpu)
        .unwrap()
        .cast(DType::Float)
        .unwrap()
}

/// The root mean square of the difference over the root mean square of the reference, which is
/// what says whether two tensors are the same answer rather than whether any one element is.
fn relative_rmse(actual: &Tensor, reference: &Tensor) -> f32 {
    let a = actual
        .to_device(Device::Cpu)
        .unwrap()
        .cast(DType::Float)
        .unwrap()
        .to_vec_f32()
        .unwrap();
    let b = reference.to_vec_f32().unwrap();
    assert_eq!(a.len(), b.len(), "the shapes do not match");

    let error: f64 = a
        .iter()
        .zip(&b)
        .map(|(x, y)| (*x as f64 - *y as f64).powi(2))
        .sum();
    let scale: f64 = b.iter().map(|y| (*y as f64).powi(2)).sum();
    (error / scale).sqrt() as f32
}

fn config_l() -> ClipTextConfig {
    ClipTextConfig {
        hidden_size: 768,
        intermediate_size: 3072,
        num_layers: 12,
        num_heads: 12,
        context_length: 77,
        vocab_size: 49408,
        quick_gelu: true,
        norm_eps: 1e-5,
        eot_token_id: 49407,
    }
}

fn config_big_g() -> ClipTextConfig {
    ClipTextConfig {
        hidden_size: 1280,
        intermediate_size: 5120,
        num_layers: 32,
        num_heads: 20,
        context_length: 77,
        vocab_size: 49408,
        quick_gelu: false,
        norm_eps: 1e-5,
        eot_token_id: 49407,
    }
}

fn input_ids(cases: &VarBuilder, name: &str) -> Tensor {
    cases
        .get_unchecked(name)
        .unwrap()
        .view(&[77])
        .unwrap()
        .to_device(device())
        .unwrap()
}

#[test]
#[ignore = "needs the sdxl package"]
fn the_first_encoder_matches_the_reference() {
    let cases = cases();
    let encoder =
        ClipTextEncoder::build(config_l(), &weights().with_name("sdxl.text_encoder")).unwrap();

    let out = encoder
        .forward(&input_ids(&cases, "test_case.input_ids"))
        .unwrap();
    assert_eq!(out.hidden.shape(), vec![1, 77, 768]);

    let reference = cases.get_unchecked("test_case.hidden").unwrap();
    let rmse = relative_rmse(&out.hidden, &reference);
    println!("encoder-1 hidden rmse = {rmse}");
    assert!(rmse < 2e-3, "the hidden state drifted by {rmse}");
}

#[test]
#[ignore = "needs the sdxl package"]
fn the_second_encoder_matches_the_reference() {
    let cases = cases();
    let encoder =
        ClipTextEncoder::build(config_big_g(), &weights().with_name("sdxl.text_encoder2")).unwrap();

    let out = encoder
        .forward(&input_ids(&cases, "test_case.input_ids2"))
        .unwrap();
    assert_eq!(out.hidden.shape(), vec![1, 77, 1280]);
    assert_eq!(out.pooled.shape(), vec![1, 1280]);

    let hidden = relative_rmse(
        &out.hidden,
        &cases.get_unchecked("test_case.hidden2").unwrap(),
    );
    println!("encoder-2 hidden rmse = {hidden}");
    assert!(hidden < 1e-2, "the hidden state drifted by {hidden}");

    // The pooled vector is what SDXL adds to its timestep embedding, and it comes out of one
    // position of one layer, so an error anywhere upstream lands here concentrated.
    let pooled = relative_rmse(
        &out.pooled,
        &cases.get_unchecked("test_case.pooled2").unwrap(),
    );
    println!("encoder-2 pooled rmse = {pooled}");
    assert!(pooled < 5e-3, "the pooled vector drifted by {pooled}");
}

#[test]
#[ignore = "needs the sdxl package"]
fn what_the_unet_is_conditioned_on_is_the_two_side_by_side() {
    // SDXL concatenates the two hidden states along their width, which is where the U-Net's 2048
    // wide cross attention comes from.
    let cases = cases();
    let vb = weights();

    let first = ClipTextEncoder::build(config_l(), &vb.with_name("sdxl.text_encoder"))
        .unwrap()
        .forward(&input_ids(&cases, "test_case.input_ids"))
        .unwrap();
    let second = ClipTextEncoder::build(config_big_g(), &vb.with_name("sdxl.text_encoder2"))
        .unwrap()
        .forward(&input_ids(&cases, "test_case.input_ids2"))
        .unwrap();

    let context = F::cat(&first.hidden, &second.hidden, -1).unwrap();
    assert_eq!(context.shape(), vec![1, 77, 2048]);
}

#[test]
#[ignore = "needs the sdxl package"]
fn gives_the_same_answer_twice() {
    // Nothing here draws from the generator, so the same prompt has to come back identical rather
    // than merely close: a difference would mean something is reading uninitialized memory.
    let cases = cases();
    let encoder =
        ClipTextEncoder::build(config_l(), &weights().with_name("sdxl.text_encoder")).unwrap();

    let ids = input_ids(&cases, "test_case.input_ids");
    let first = encoder.forward(&ids).unwrap();
    let second = encoder.forward(&ids).unwrap();

    assert_eq!(
        relative_rmse(&first.hidden, &to_cpu_f32(&second.hidden)),
        0.0
    );
}

#[test]
#[ignore = "needs the sdxl package"]
fn the_activation_is_what_tells_the_two_encoders_apart() {
    // The first encoder uses the sigmoid approximation and the second the ordinary GELU, which is
    // the sort of difference that produces a plausible image rather than an obviously broken one.
    // Building it wrong on purpose says the comparison above would have caught it.
    let cases = cases();
    let mut wrong = config_l();
    wrong.quick_gelu = false;

    let vb = weights();
    let ids = input_ids(&cases, "test_case.input_ids");
    let reference = cases.get_unchecked("test_case.hidden").unwrap();

    let right = ClipTextEncoder::build(config_l(), &vb.with_name("sdxl.text_encoder"))
        .unwrap()
        .forward(&ids)
        .unwrap();
    let wrong = ClipTextEncoder::build(wrong, &vb.with_name("sdxl.text_encoder"))
        .unwrap()
        .forward(&ids)
        .unwrap();

    let right = relative_rmse(&right.hidden, &reference);
    let wrong = relative_rmse(&wrong.hidden, &reference);
    println!("right activation {right}, wrong activation {wrong}");

    // Two orders of magnitude apart, which is what says the comparison above is measuring the
    // model rather than the precision it runs in.
    assert!(
        wrong > 50.0 * right,
        "the wrong activation drifts by {wrong} against {right}, which is not far enough apart"
    );
}

#[test]
#[ignore = "needs the sdxl package"]
fn refuses_what_it_cannot_encode() {
    let encoder =
        ClipTextEncoder::build(config_l(), &weights().with_name("sdxl.text_encoder")).unwrap();

    // Token ids are one dimensional, and there is no position embedding past the context length.
    let two_d = Tensor::from_i64(&[2, 3], &[0; 6])
        .unwrap()
        .to_device(device())
        .unwrap();
    assert!(encoder.forward(&two_d).is_err());

    let too_long = Tensor::from_i64(&[78], &[49407; 78])
        .unwrap()
        .to_device(device())
        .unwrap();
    assert!(encoder.forward(&too_long).is_err());

    // And the pooled vector has nowhere to come from without an end-of-text marker.
    let no_marker = Tensor::from_i64(&[4], &[1, 2, 3, 4])
        .unwrap()
        .to_device(device())
        .unwrap();
    assert!(encoder.forward(&no_marker).is_err());
}
