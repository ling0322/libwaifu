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

//! The VAE decoder against what huggingface makes of the same latent.
//!
//! A decoder is the one part of this whose output can be looked at, which is exactly why it is
//! compared as numbers instead: an image that is plausible but wrong is the failure mode, and the
//! eye is no good at telling one from the other.

use std::path::PathBuf;

use waifu::flint::Tensor;
use waifu::{DType, Device, VaeConfig, VaeDecoder, VarBuilder, ZipFile};

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

fn config() -> VaeConfig {
    VaeConfig {
        latent_channels: 4,
        block_out_channels: vec![128, 256, 512, 512],
        layers_per_block: 2,
        norm_num_groups: 32,
        norm_eps: 1e-6,
        scaling_factor: 0.13025,
    }
}

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

fn latent(cases: &VarBuilder) -> Tensor {
    cases
        .get_unchecked("test_case.latent")
        .unwrap()
        .to_device(Device::Cuda)
        .unwrap()
        .cast(DType::Float16)
        .unwrap()
}

#[test]
#[ignore = "needs the sdxl package"]
fn decodes_a_latent_the_way_the_reference_does() {
    let cases = cases();
    let decoder = VaeDecoder::build(config(), &weights().with_name("sdxl.vae")).unwrap();

    let image = decoder.forward(&latent(&cases)).unwrap();

    // Eight times larger on each axis, and three channels rather than four.
    assert_eq!(image.shape(), vec![1, 3, 256, 256]);

    let rmse = relative_rmse(&image, &cases.get_unchecked("test_case.decoded").unwrap());
    println!("vae decode rmse = {rmse}");
    assert!(rmse < 2e-2, "the decoded image drifted by {rmse}");
}

#[test]
#[ignore = "needs the sdxl package"]
fn produces_something_an_image_could_be_made_of() {
    // A decoder ends in roughly [-1, 1], which is what the conversion to pixels assumes. A result
    // that is finite but far outside that range would still pass a loose comparison while being
    // useless, so it is worth saying separately.
    let cases = cases();
    let decoder = VaeDecoder::build(config(), &weights().with_name("sdxl.vae")).unwrap();

    let image = decoder.forward(&latent(&cases)).unwrap();
    let values = image
        .to_device(Device::Cpu)
        .unwrap()
        .cast(DType::Float)
        .unwrap()
        .to_vec_f32()
        .unwrap();

    assert!(
        values.iter().all(|x| x.is_finite()),
        "the decoder produced a NaN or an infinity"
    );

    let extreme = values.iter().filter(|x| x.abs() > 1.5).count();
    assert!(
        extreme * 100 < values.len(),
        "{extreme} of {} pixels are far outside [-1, 1]",
        values.len()
    );
}

#[test]
#[ignore = "needs the sdxl package"]
fn gives_the_same_image_twice() {
    let cases = cases();
    let decoder = VaeDecoder::build(config(), &weights().with_name("sdxl.vae")).unwrap();

    let latent = latent(&cases);
    let first = decoder.forward(&latent).unwrap();
    let second = decoder.forward(&latent).unwrap();

    let cpu_second = second
        .to_device(Device::Cpu)
        .unwrap()
        .cast(DType::Float)
        .unwrap();
    assert_eq!(relative_rmse(&first, &cpu_second), 0.0);
}

#[test]
#[ignore = "needs the sdxl package"]
fn refuses_a_latent_it_cannot_read() {
    let decoder = VaeDecoder::build(config(), &weights().with_name("sdxl.vae")).unwrap();

    // A latent is four dimensional and four channels deep; anything else is a caller's mistake
    // rather than something to guess at.
    let three_d = waifu::flint::functional::rand(&[4, 8, 8], DType::Float16, Device::Cuda).unwrap();
    assert!(decoder.forward(&three_d).is_err());

    let wrong_channels =
        waifu::flint::functional::rand(&[1, 3, 8, 8], DType::Float16, Device::Cuda).unwrap();
    assert!(decoder.forward(&wrong_channels).is_err());
}

#[test]
#[ignore = "needs the sdxl package"]
fn decodes_a_latent_of_another_size() {
    // Nothing in a convolutional decoder is tied to one resolution, and SDXL is used at several,
    // including ones that are not square. A different size also gives the mid-block attention a
    // different number of positions to attend over, which is the part most likely to have a size
    // baked into it.
    //
    // These are corners of the reference latent rather than random numbers: this decoder is the
    // one that famously overflows in half precision, and feeding it something no encoder would
    // ever produce tests that overflow instead of testing the shapes.
    let cases = cases();
    let reference = latent(&cases);
    let decoder = VaeDecoder::build(config(), &weights().with_name("sdxl.vae")).unwrap();

    for (height, width) in [(8, 8), (16, 16), (16, 24), (32, 8)] {
        let corner = reference
            .slice(2, 0, height)
            .unwrap()
            .slice(3, 0, width)
            .unwrap()
            .contiguous()
            .unwrap();

        let image = decoder.forward(&corner).unwrap();
        assert_eq!(image.shape(), vec![1, 3, height * 8, width * 8]);

        let values = image
            .to_device(Device::Cpu)
            .unwrap()
            .cast(DType::Float)
            .unwrap()
            .to_vec_f32()
            .unwrap();
        assert!(
            values.iter().all(|x| x.is_finite()),
            "a {height} by {width} latent decoded to a NaN"
        );
    }
}
