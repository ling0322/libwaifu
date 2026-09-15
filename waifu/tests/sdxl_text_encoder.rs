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

use std::cell::OnceCell;
use std::path::PathBuf;
use std::rc::Rc;

use waifu::flint::{functional as F, resident, ParamSource, Tensor};
use waifu::{
    ClipTextConfig, ClipTextEncoder, DType, Device, Manifest, ParamFile
};

fn models_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../models")
}

fn device() -> Device {
    Device::Cuda
}

/// The whole package on the device, read once for this whole test binary.
///
/// `resident` reads the file rather than a model's part of it, so reading it per test would read
/// seven gigabytes as many times as there are tests here.
fn weights() -> Rc<dyn ParamSource> {
    thread_local! {
        static WEIGHTS: OnceCell<Rc<dyn ParamSource>> = const { OnceCell::new() };
    }

    WEIGHTS.with(|cell| {
        Rc::clone(cell.get_or_init(|| {
            let manifest = Manifest::open(models_dir().join("sdxl-base.yaml")).unwrap();
            let file = params(&manifest);

            let weights: Rc<dyn ParamSource> = Rc::new(resident(&file, device()).unwrap());
            weights
        }))
    })
}

/// The parameters of the model, out of whichever files its manifest names.
fn params(manifest: &Manifest) -> ParamFile {
    manifest.params().unwrap()
}

fn cases() -> ParamFile {
    ParamFile::open(&[models_dir().join("sdxl-base_test.safetensors")]).unwrap()
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
        eot_token_id: 49407
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
        eot_token_id: 49407
    }
}

fn input_ids(cases: &ParamFile, name: &str) -> Tensor {
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
        ClipTextEncoder::build(config_l(), "sdxl.text_encoder", &weights(), DType::Float16)
            .unwrap();

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
    let encoder = ClipTextEncoder::build(
        config_big_g(),
        "sdxl.text_encoder2",
        &weights(),
        DType::Float16,
    )
    .unwrap();

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
    let file = weights();

    let first = ClipTextEncoder::build(config_l(), "sdxl.text_encoder", &file, DType::Float16)
        .unwrap()
        .forward(&input_ids(&cases, "test_case.input_ids"))
        .unwrap();
    let second =
        ClipTextEncoder::build(config_big_g(), "sdxl.text_encoder2", &file, DType::Float16)
            .unwrap()
            .forward(&input_ids(&cases, "test_case.input_ids2"))
            .unwrap();

    let context = F::cat(&first.hidden, &second.hidden, -1).unwrap();
    assert_eq!(context.shape(), vec![1, 77, 2048]);
}

#[test]
#[ignore = "needs the sdxl package"]
fn the_pass_can_be_read_before_it_is_run() {
    // What an eager encoder could not be asked. The instructions are the pass, in the order it
    // runs, and every weight is named as the package names it.
    let encoder =
        ClipTextEncoder::build(config_l(), "sdxl.text_encoder", &weights(), DType::Float16)
            .unwrap();
    let listing = encoder.ir().to_string();

    // Every weight is asked for by the whole name and the shape the layer expects, which is what
    // makes a package that does not match say so instead of aborting somewhere in a kernel.
    assert!(
        listing.contains(r#"load("sdxl.text_encoder.block0.attn.qkv_proj.weight", [2304, 768])"#),
        "{listing}"
    );

    // And is let go of at the instruction that read it last. The token table is 145 MB of the
    // 246 this encoder is, and it is finished with four instructions in.
    assert!(
        listing.contains(concat!(
            "  %2 = load(\"sdxl.text_encoder.token_embd.weight\", [49408, 768])\n",
            "  %3 = lookup(%2, %0)\n",
            "  free %2\n",
        )),
        "{listing}"
    );

    // Nothing in it was written for one prompt length: a reshape says which value it takes its
    // length from, and the position table is read for as many rows as there are ids.
    assert!(listing.contains("dim(%0, 0)"), "{listing}");
    assert!(listing.contains("dim(%"), "{listing}");
    assert!(!listing.contains(", 77,"), "{listing}");

    // Nothing in it that nothing reads: one free apiece for everything but the two outputs.
    let (runs, frees): (Vec<&str>, Vec<&str>) = listing
        .lines()
        .map(str::trim)
        .filter(|line| line.starts_with('%') || line.starts_with("free "))
        .partition(|line| line.starts_with('%'));
    assert_eq!(frees.len(), runs.len() - 2);
}

#[test]
#[ignore = "needs the sdxl package"]
fn one_compiled_pass_encodes_whatever_length_it_is_given() {
    // The pass names where it takes its lengths from rather than having been written for one, so
    // the same instructions run at both of these. Nothing is compiled a second time.
    let encoder =
        ClipTextEncoder::build(config_l(), "sdxl.text_encoder", &weights(), DType::Float16)
            .unwrap();

    for length in [16, 77] {
        let mut ids = vec![49406i64; length as usize];
        ids[length as usize - 1] = 49407;
        let ids = Tensor::from_i64(&[length], &ids)
            .unwrap()
            .to_device(device())
            .unwrap();

        let out = encoder.forward(&ids).unwrap();
        assert_eq!(out.hidden.shape(), vec![1, length, 768]);
        assert_eq!(out.pooled.shape(), vec![1, 768]);
    }
}

#[test]
#[ignore = "needs the sdxl package"]
fn gives_the_same_answer_twice() {
    // Nothing here draws from the generator, so the same prompt has to come back identical rather
    // than merely close: a difference would mean something is reading uninitialized memory.
    let cases = cases();
    let encoder =
        ClipTextEncoder::build(config_l(), "sdxl.text_encoder", &weights(), DType::Float16)
            .unwrap();

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

    let file = weights();
    let ids = input_ids(&cases, "test_case.input_ids");
    let reference = cases.get_unchecked("test_case.hidden").unwrap();

    let right = ClipTextEncoder::build(config_l(), "sdxl.text_encoder", &file, DType::Float16)
        .unwrap()
        .forward(&ids)
        .unwrap();
    let wrong = ClipTextEncoder::build(wrong, "sdxl.text_encoder", &file, DType::Float16)
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
        ClipTextEncoder::build(config_l(), "sdxl.text_encoder", &weights(), DType::Float16)
            .unwrap();

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
