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

//! The vocoder against NVIDIA's BigVGAN, which is the only thing worth checking it against.
//!
//! Every number at the bottom of this file came out of the reference implementation itself --
//! `tools/bigvgan_reference.py` downloads it, builds a small one, and prints what it produced.
//! A second implementation written from the same paper would agree with a bug as readily as with
//! the model, so none of these expectations was worked out here.
//!
//! The model is a small one: 8 mel bands, two upsamplings by two, 24 samples out. What it is not
//! is a *simplified* one -- it has both kinds of convolution, four AMP blocks, dilations, and
//! seventeen anti-aliased snakes, which is every distinct thing the 112 M parameter release does.
//! Being small is what keeps it in the fast suite, where a vocoder that stopped matching is
//! noticed in the same minute it stopped.
//!
//! Neither side reads a weight from the other. Both fill every parameter from its own name, in
//! float64 rounded once -- see [`fill`] -- so the two models are equal bit for bit without a file
//! passing between them, and this needs no package.

use std::collections::HashMap;
use std::path::PathBuf;
use std::rc::Rc;

use waifu::audio::{downsample1d, kaiser_sinc_filter, upsample1d};
use waifu::indextts::bigvgan::{BigVgan, BigVganConfig};
use waifu::flint::{DType, Device, Graph, Ir, ParamSource, Residency, RunContext, Tensor, Weights};
use waifu::read_safetensors;

const CPU: Device = Device::Cpu;
const F32: DType = DType::Float;

/// How far apart two float32 implementations of the same arithmetic are allowed to be. The
/// convolutions here are a GEMM where the reference's are a kernel, so the two sum in different
/// orders; nothing else about them differs.
const TOLERANCE: f32 = 2e-4;

/// And on a card, where the float type is half rather than float32. An order of magnitude looser,
/// which is what sixteen bits of mantissa through a hundred-odd layers costs: the worst of the 24
/// samples is 3.4e-3 away from BigVGAN's, against a waveform that peaks at 0.75.
const CUDA_TOLERANCE: f32 = 4e-3;

/// The scales `tools/bigvgan_reference.py` generated each waveform at.
const WEIGHT_SCALE: f64 = 0.25;
const LOUD_SCALE: f64 = 0.35;

/// FNV-1a over a parameter's name. The Python side computes the same thing the same way.
fn hash32(name: &str) -> u32 {
    let mut value: u32 = 0x811c_9dc5;
    for byte in name.bytes() {
        value = (value ^ u32::from(byte)).wrapping_mul(0x0100_0193);
    }

    value
}

/// `count` numbers that belong to this parameter and no other.
///
/// Filling from the name rather than from a generator is what lets the two implementations agree
/// without a file between them: there is no order to walk the model in, so there is no order for
/// the two to disagree about. The arithmetic is float64 in both languages and rounded once, at
/// the end, so the fills are equal bit for bit rather than close.
fn fill(name: &str, count: usize, scale: f64) -> Vec<f32> {
    let seed = f64::from(hash32(name) % 10007) * 0.001;

    (0..count)
        .map(|i| ((i as f64 * 0.7371 + seed).sin() * scale) as f32)
        .collect()
}

fn tensor(shape: &[i32], values: &[f32]) -> Tensor {
    Tensor::from_f32(shape, values).unwrap()
}

/// Every weight the reference model has, filled from its name. The list is the reference's own --
/// see `PARAMETERS` -- so a graph that asks for something else does not find it.
fn weights(scale: f64) -> Rc<dyn ParamSource> {
    weights_on(CPU, F32, scale)
}

/// The same, where the model is going to run. A weight has to be on the device already: nothing
/// between a [`ParamSource`] and a graph moves one.
fn weights_on(device: Device, dtype: DType, scale: f64) -> Rc<dyn ParamSource> {
    let held: HashMap<String, Tensor> = PARAMETERS
        .iter()
        .map(|(name, shape)| {
            let count = shape.iter().product::<i32>() as usize;
            let made = tensor(shape, &fill(name, count, scale));

            (
                name.to_string(),
                made.to_device(device).unwrap().cast(dtype).unwrap(),
            )
        })
        .collect();

    Rc::new(held)
}

/// The model `tools/bigvgan_reference.py` builds, which is the released one with every count cut
/// down and nothing left out.
fn config(tanh: bool) -> BigVganConfig {
    BigVganConfig {
        mel_channels: 8,
        upsample_rates: vec![2, 2],
        upsample_kernel_sizes: vec![4, 4],
        upsample_initial_channel: 16,
        resblock_kernel_sizes: vec![3, 5],
        resblock_dilation_sizes: vec![vec![1, 3], vec![1, 3]],
        snake_beta: true,
        snake_logscale: true,
        use_tanh_at_final: tanh,
        use_bias_at_final: false,
    }
}

/// The mel both sides run, six frames of eight bands.
fn mel() -> Tensor {
    tensor(&[1, 8, 6], &fill("mel", 8 * 6, 1.0))
}

fn close(a: &[f32], b: &[f32], tolerance: f32) {
    assert_eq!(
        a.len(),
        b.len(),
        "lengths differ: {} vs {}",
        a.len(),
        b.len()
    );

    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        assert!(
            (x - y).abs() <= tolerance * (1.0 + y.abs()),
            "element {i}: {x} is not {y} (tolerance {tolerance})"
        );
    }
}

/// The signal the two halves of an anti-aliased activation are checked on, `(2, 3, 9)`.
fn signal() -> Tensor {
    tensor(&[2, 3, 9], &fill("signal", 2 * 3 * 9, 1.0))
}

/// Compile `g` with whatever `write` named and run it over one input.
fn run(
    build: impl FnOnce(&Graph, waifu::flint::Value) -> waifu::flint::Value,
    x: &Tensor,
) -> (Vec<i32>, Vec<f32>) {
    let g = Graph::new();
    let out = build(&g, g.input("x"));
    g.output("out", out);

    let empty: HashMap<String, Tensor> = HashMap::new();
    let ir = Ir::compile(&g);
    let outputs = ir
        .run(&RunContext::new(&empty).input("x", x))
        .unwrap();
    let tensor = outputs[0].1.to_device(CPU).unwrap();

    (tensor.shape(), tensor.to_vec_f32().unwrap())
}

// ---------------------------------------------------------------------------------------------
// The filter, and the two halves of an anti-aliased activation
// ---------------------------------------------------------------------------------------------

#[test]
fn both_sides_make_the_same_weights() {
    // Everything below stands on this: the two models are only running the same arithmetic if
    // they were filled with the same numbers, and nothing checks that but this. `SIGNAL` is what
    // the Python side got out of `fill("signal", 54, 1.0)`.
    close(&fill("signal", 54, 1.0), &SIGNAL, 0.0);
}

#[test]
fn the_filter_is_the_one_the_reference_designs() {
    let filter = kaiser_sinc_filter(0.5 / 2.0, 0.6 / 2.0, 12).unwrap();

    assert_eq!(filter.shape(), vec![1, 1, 12]);
    close(&filter.to_vec_f32().unwrap(), &KAISER_12, 1e-6);

    // The taps sum to one, which is what leaves a constant signal constant.
    let total: f32 = filter.to_vec_f32().unwrap().iter().sum();
    assert!((total - 1.0).abs() < 1e-6, "the taps sum to {total}");
}

#[test]
fn upsampling_is_the_reference_upsampling() {
    let filter = kaiser_sinc_filter(0.5 / 2.0, 0.6 / 2.0, 12).unwrap();
    let signal = signal();

    let (shape, values) = run(
        |g, x| {
            let taps = g.constant(filter.clone());

            upsample1d(g, x, taps, 3, 2, 12, F32, CPU).unwrap()
        },
        &signal,
    );

    // Exactly twice as long: the taper a transposed convolution leaves on both ends is in the
    // padding, and the padding is what was cut off.
    assert_eq!(shape, vec![2, 3, 18]);
    close(&values, &UPSAMPLED, TOLERANCE);
}

#[test]
fn downsampling_is_the_reference_downsampling() {
    let filter = kaiser_sinc_filter(0.5 / 2.0, 0.6 / 2.0, 12).unwrap();
    let signal = signal();

    let (shape, values) = run(
        |g, x| {
            let taps = g.constant(filter.clone());

            downsample1d(g, x, taps, 3, 2, 12, F32, CPU).unwrap()
        },
        &signal,
    );

    assert_eq!(shape, vec![2, 3, 5]);
    close(&values, &DOWNSAMPLED, TOLERANCE);
}

#[test]
fn an_anti_aliased_snake_is_the_two_of_them_around_a_snake() {
    let filter = kaiser_sinc_filter(0.5 / 2.0, 0.6 / 2.0, 12).unwrap();
    let signal = signal();

    // What the model does inside an AMP block, written out here rather than reached through one:
    // up by two, the snake, down by two.
    let (shape, values) = run(
        |g, x| {
            let taps = g.constant(filter.clone());
            let alpha = g.exp(g.constant(tensor(&[3], &fill("act.alpha", 3, WEIGHT_SCALE))));
            let beta = g.exp(g.constant(tensor(&[3], &fill("act.beta", 3, WEIGHT_SCALE))));

            let raised = upsample1d(g, x, taps, 3, 2, 12, F32, CPU).unwrap();
            let curled = waifu::audio::snake(g, raised, alpha, Some(beta), 1e-9, F32, CPU).unwrap();

            downsample1d(g, curled, taps, 3, 2, 12, F32, CPU).unwrap()
        },
        &signal,
    );

    assert_eq!(shape, vec![2, 3, 9]);
    close(&values, &ACTIVATED, TOLERANCE);
}

// ---------------------------------------------------------------------------------------------
// The whole vocoder
// ---------------------------------------------------------------------------------------------

#[test]
fn a_mel_becomes_the_waveform_bigvgan_makes_of_it() {
    let vocoder = BigVgan::build(config(false), "", &weights(WEIGHT_SCALE), CPU, F32).unwrap();
    let waveform = vocoder.forward(&mel()).unwrap();

    // One frame is one hop of samples, and the hop is what the upsamplings multiply to.
    assert_eq!(waveform.shape(), vec![1, 1, 6 * 4]);
    assert_eq!(vocoder.config().upsample_factor(), 4);

    close(&waveform.to_vec_f32().unwrap(), &WAVEFORM, TOLERANCE);
}

#[test]
fn a_loud_one_is_clamped_where_the_reference_clamps_it() {
    let vocoder = BigVgan::build(config(false), "", &weights(LOUD_SCALE), CPU, F32).unwrap();
    let waveform = vocoder.forward(&mel()).unwrap();
    let values = waveform.to_vec_f32().unwrap();

    // The weights here are large enough that most of the output is outside [-1, 1] before the
    // last step, which is the only reason this says anything about the clamp.
    let bounded = values.iter().filter(|v| v.abs() >= 0.999).count();
    assert!(bounded > 10, "only {bounded} samples reached the bound");

    close(&values, &WAVEFORM_CLAMPED, TOLERANCE);
}

#[test]
fn the_other_ending_is_a_tanh() {
    let vocoder = BigVgan::build(config(true), "", &weights(LOUD_SCALE), CPU, F32).unwrap();
    let waveform = vocoder.forward(&mel()).unwrap();

    close(&waveform.to_vec_f32().unwrap(), &WAVEFORM_TANH, TOLERANCE);
}

#[test]
fn every_weight_it_reads_is_one_the_checkpoint_holds() {
    // `BigVgan::build` checks every load against the source, and the source here holds the
    // reference's parameters and nothing else -- so building at all is the assertion. What is
    // left is the other direction: a weight in the checkpoint that the graph never reads is a
    // layer that was not written.
    let read = Reading::default();
    let source: Rc<dyn ParamSource> = Rc::new(read.clone());

    let vocoder = BigVgan::build(config(false), "", &source, CPU, F32).unwrap();
    vocoder.forward(&mel()).unwrap();

    let mut unread: Vec<&str> = PARAMETERS
        .iter()
        .map(|(name, _)| *name)
        .filter(|name| !read.names().iter().any(|asked| asked == name))
        .collect();
    unread.sort_unstable();

    assert!(unread.is_empty(), "never read: {unread:?}");
}

#[test]
fn a_mel_of_the_wrong_shape_is_refused() {
    let vocoder = BigVgan::build(config(false), "", &weights(WEIGHT_SCALE), CPU, F32).unwrap();

    let flat = tensor(&[8, 6], &fill("mel", 8 * 6, 1.0));
    assert!(vocoder
        .forward(&flat)
        .unwrap_err()
        .to_string()
        .contains("2-D tensor"));

    let bands = tensor(&[1, 7, 6], &fill("mel", 7 * 6, 1.0));
    assert!(vocoder
        .forward(&bands)
        .unwrap_err()
        .to_string()
        .contains("7 bands"));
}

#[test]
fn a_configuration_that_cannot_be_built_says_so() {
    let refused = |change: fn(&mut BigVganConfig)| {
        let mut broken = config(false);
        change(&mut broken);

        BigVgan::build(broken, "", &weights(WEIGHT_SCALE), CPU, F32)
            .expect_err("that configuration should not build")
            .to_string()
    };

    assert!(refused(|c| c.upsample_rates.clear()).contains("at least one upsampling"));
    assert!(refused(|c| c.upsample_kernel_sizes.push(4)).contains("not a pair each"));
    assert!(
        refused(|c| c.resblock_dilation_sizes.pop().map(drop).unwrap()).contains("not a pair each")
    );

    // A kernel that is not a whole number of strides away from its rate would leave a sample per
    // stage unaccounted for, which is the one thing a caller cannot check for themselves.
    assert!(refused(|c| c.upsample_kernel_sizes[1] = 5).contains("even difference"));
    assert!(refused(|c| c.upsample_initial_channel = 2).contains("do not survive"));
}

#[test]
#[ignore = "needs a CUDA device"]
fn it_runs_on_a_card() {
    // The whole point of a model composed out of `conv2d` and `matmul` is that it runs wherever
    // those do, so this is the same small model as `a_mel_becomes_the_waveform_bigvgan_makes_of_it`
    // against the same expectation, on the GPU and in whatever float type that backend works in.
    let dtype = waifu::flint::functional::default_float_type(Device::Cuda).unwrap();
    let weights = weights_on(Device::Cuda, dtype, WEIGHT_SCALE);

    let vocoder = BigVgan::build(config(false), "", &weights, Device::Cuda, dtype).unwrap();
    let waveform = vocoder.forward(&mel()).unwrap();

    assert_eq!(waveform.shape(), vec![1, 1, 24]);

    // Against the same numbers the processor is checked on: this is BigVGAN's float32 answer, and
    // the distance to it is what half costs rather than anything about the card.
    let host = waveform.to_device(CPU).unwrap().cast(F32).unwrap();
    close(&host.to_vec_f32().unwrap(), &WAVEFORM, CUDA_TOLERANCE);
}

// ---------------------------------------------------------------------------------------------
// The released 112 M parameter vocoder, which needs the checkpoint exported first
// ---------------------------------------------------------------------------------------------

/// Where `tools/bigvgan_exporter.py` was told to put the two files these read.
fn models_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../models")
}

fn exported(name: &str) -> HashMap<String, Tensor> {
    read_safetensors(&[models_dir().join(name)]).unwrap_or_else(|error| {
        panic!(
            "{name}: {error}\nExport it first:\n    .venv/bin/python tools/bigvgan_exporter.py \
             -output models/bigvgan-22khz-80band.safetensors \
             -test_output models/bigvgan-22khz-80band_test.safetensors"
        )
    })
}

#[test]
#[ignore = "needs the exported BigVGAN checkpoint in models/"]
fn the_released_vocoder_is_the_released_vocoder() {
    let name = "bigvgan-22khz-80band.safetensors";
    let weights: Rc<dyn ParamSource> = Rc::new(
        Weights::from_files(&[models_dir().join(name)], CPU, Residency::Device)
            .unwrap_or_else(|error| panic!("{name}: {error}\nExport it first; see `exported`.")),
    );
    let vocoder = BigVgan::build(
        BigVganConfig::v2_22khz_80band_256x(),
        "",
        &weights,
        CPU,
        F32,
    )
    .unwrap();

    // A quarter of a second of chirp, analysed by the reference's own mel spectrogram -- so what
    // is being compared is the whole model over a signal it could plausibly be asked for, and not
    // a shape test dressed up as one.
    let bundle = exported("bigvgan-22khz-80band_test.safetensors");
    let mel = bundle["mel"].clone();
    let expected = bundle["waveform"].clone();

    let waveform = vocoder.forward(&mel).unwrap();
    assert_eq!(waveform.shape(), expected.shape());
    assert_eq!(waveform.shape()[2], mel.shape()[2] * 256);

    // Measured rather than guessed: the worst sample of the 5376 differs by 1.2e-5, against a
    // waveform that peaks at 0.58. This is 1e-4 for headroom, which is still a part in six
    // thousand of full scale -- inaudible, and orders below what any structural mistake costs.
    // The two disagree at all because a hundred and sixteen convolutions of float32 accumulation
    // are summed in a different order here than in the reference's kernels.
    close(
        &waveform.to_vec_f32().unwrap(),
        &expected.to_vec_f32().unwrap(),
        1e-4,
    );
}

/// A [`ParamSource`] that writes down what it was asked for.
///
/// It holds the reference's parameters, the way [`weights`] does, and remembers every name that
/// was looked up -- which is how `every_weight_it_reads_is_one_the_checkpoint_holds` can ask the
/// question the other way round.
#[derive(Clone, Default)]
struct Reading {
    asked: Rc<std::cell::RefCell<Vec<String>>>,
}

impl Reading {
    fn names(&self) -> Vec<String> {
        self.asked.borrow().clone()
    }

    fn held(&self, name: &str) -> Option<Tensor> {
        PARAMETERS
            .iter()
            .find(|(held, _)| *held == name)
            .map(|(held, shape)| {
                let count = shape.iter().product::<i32>() as usize;

                tensor(shape, &fill(held, count, WEIGHT_SCALE))
            })
    }
}

impl ParamSource for Reading {
    fn load(&self, name: &str, shape: &[i32]) -> waifu::Result<Tensor> {
        self.asked.borrow_mut().push(name.to_string());
        self.check(name, shape)?;

        Ok(self.held(name).expect("checked just above"))
    }

    fn shape_of(&self, name: &str) -> Option<Vec<i32>> {
        PARAMETERS
            .iter()
            .find(|(held, _)| *held == name)
            .map(|(_, shape)| shape.to_vec())
    }

    fn check(&self, name: &str, shape: &[i32]) -> waifu::Result<()> {
        match self.shape_of(name) {
            Some(held) if held == shape => Ok(()),
            Some(held) => panic!("{name} is {held:?} here, not {shape:?}"),
            None => panic!("no {name} in the reference checkpoint"),
        }
    }
}

// ---------------------------------------------------------------------------------------------
// What BigVGAN produced. Generated by tools/bigvgan_reference.py -- do not edit by hand.
// ---------------------------------------------------------------------------------------------


const KAISER_12: [f32; 12] = [
    2.0289647e-03, 9.389466e-03, -2.5543459e-02, -5.7657383e-02,
    1.2857258e-01, 4.432098e-01, 4.432098e-01, 1.2857258e-01,
    -5.7657383e-02, -2.5543459e-02, 9.389466e-03, 2.0289647e-03,
];

const SIGNAL: [f32; 54] = [
    7.080818e-01, 4.965753e-02, -6.345468e-01, -9.89321e-01,
    -8.30481e-01, -2.4048997e-01, 4.743534e-01, 9.429323e-01,
    9.2198014e-01, 4.223744e-01, -2.965105e-01, -8.614595e-01,
    -9.7917473e-01, -5.8854336e-01, 1.07635155e-01, 7.47934e-01,
    9.9993676e-01, 7.3281413e-01, 8.524499e-02, -6.065798e-01,
    -9.834936e-01, -8.498187e-01, -2.749534e-01, 4.4265622e-01,
    9.3045723e-01, 9.3520373e-01, 4.5443147e-01, -2.6226258e-01,
    -8.4280086e-01, -9.8579216e-01, -6.170013e-01, 7.2110824e-02,
    7.2378606e-01, 9.997018e-01, 7.566141e-01, 1.2072401e-01,
    -5.77841e-01, -9.7641504e-01, -8.680752e-01, -3.09067e-01,
    4.103959e-01, 9.1679835e-01, 9.4723743e-01, 4.859104e-01,
    -2.2768104e-01, -8.2307e-01, -9.911554e-01, -6.446743e-01,
    3.6494743e-02, 6.9871724e-01, 9.9819493e-01, 7.794515e-01,
    1.5604943e-01, -5.48367e-01,
];

const UPSAMPLED: [f32; 108] = [
    7.5879467e-01, 6.0047394e-01, 2.4739107e-01, -1.4033717e-01,
    -4.8451814e-01, -7.6785225e-01, -9.4853705e-01, -1.0021918e+00,
    -9.212953e-01, -7.1623516e-01, -4.1575548e-01, -5.8439177e-02,
    3.0562723e-01, 6.3172334e-01, 8.777231e-01, 9.7034055e-01,
    9.4551843e-01, 9.11158e-01, 4.8116338e-01, 2.9741633e-01,
    -9.313387e-02, -4.7749236e-01, -7.578636e-01, -9.4277716e-01,
    -1.002729e+00, -9.2799354e-01, -7.290468e-01, -4.3143445e-01,
    -7.687398e-02, 2.891074e-01, 6.1520857e-01, 8.621846e-01,
    1.0000613e+00, 9.561131e-01, 7.9819405e-01, 7.022949e-01,
    1.4495282e-01, -4.1850273e-02, -4.2232022e-01, -7.5651515e-01,
    -9.389427e-01, -1.0029235e+00, -9.348436e-01, -7.4081665e-01,
    -4.480405e-01, -9.410868e-02, 2.7136654e-01, 6.0145664e-01,
    8.498914e-01, 9.87679e-01, 1.0006468e+00, 8.2548344e-01,
    5.5369323e-01, 4.0793055e-01, -2.0890504e-01, -3.760218e-01,
    -7.0009106e-01, -9.4343585e-01, -1.0057102e+00, -9.409689e-01,
    -7.531454e-01, -4.6344888e-01, -1.1248729e-01, 2.5467438e-01,
    5.8656955e-01, 8.405814e-01, 9.8110396e-01, 9.929283e-01,
    8.794085e-01, 5.9435517e-01, 2.4178298e-01, 6.390274e-02,
    -5.373298e-01, -6.644145e-01, -8.9262927e-01, -1.0154978e+00,
    -9.5003724e-01, -7.644557e-01, -4.7975543e-01, -1.2965846e-01,
    2.3676066e-01, 5.7245207e-01, 8.303605e-01, 9.7736937e-01,
    9.928718e-01, 8.7729335e-01, 6.5110636e-01, 2.9086706e-01,
    -9.9563174e-02, -2.8790495e-01, -8.0033714e-01, -8.7191796e-01,
    -9.7649413e-01, -9.6392775e-01, -7.7870184e-01, -4.9487373e-01,
    -1.479575e-01, 2.1991731e-01, 5.571841e-01, 8.2053643e-01,
    9.73059e-01, 9.951673e-01, 8.8376236e-01, 6.548521e-01,
    3.4353516e-01, -4.8032746e-02, -4.28788e-01, -6.046616e-01,
];

const DOWNSAMPLED: [f32; 30] = [
    3.7142363e-01, -8.584607e-01, -5.6486243e-01, 7.168907e-01,
    9.598439e-01, 4.3029014e-02, -9.7433496e-01, -2.4881296e-01,
    8.8038415e-01, 7.6637584e-01, -2.9060417e-01, -9.7158843e-01,
    9.7528346e-02, 9.3669516e-01, 4.796051e-01, -5.888577e-01,
    -8.5055584e-01, 4.3199608e-01, 8.7896794e-01, 1.344449e-01,
    -8.154206e-01, -6.259721e-01, 7.138703e-01, 7.142307e-01,
    -2.2708336e-01, -9.427099e-01, -3.251793e-01, 9.0883434e-01,
    4.6253923e-01, -5.6096536e-01,
];

const ACTIVATED: [f32; 54] = [
    1.1150995e+00, 1.0848617e-01, -2.6840258e-01, -2.6314855e-01,
    -2.6148373e-01, -1.6168751e-01, 7.1890527e-01, 1.6112809e+00,
    1.610511e+00, 5.973812e-01, -1.0992784e-01, -9.934649e-02,
    -7.541284e-02, -1.573503e-01, 1.6070412e-01, 1.3987252e+00,
    1.8904898e+00, 1.4191186e+00, 5.566905e-02, -9.0596616e-02,
    3.5830546e-02, -1.12566585e-02, -1.3562989e-01, 7.386976e-01,
    1.9108884e+00, 1.8432451e+00, 8.432098e-01, -1.8738969e-01,
    -2.566383e-01, -2.6632628e-01, -2.6196116e-01, 1.0366882e-01,
    1.1906433e+00, 1.7598763e+00, 1.2191045e+00, 1.8175738e-01,
    -1.2040269e-01, -8.557282e-02, -9.207361e-02, -1.5322237e-01,
    6.559179e-01, 1.7602649e+00, 1.8364005e+00, 7.916288e-01,
    -1.5275078e-01, 2.3991785e-03, 3.5564434e-02, -9.7918294e-02,
    7.659131e-02, 1.3282768e+00, 2.0513058e+00, 1.529944e+00,
    2.330084e-01, -1.5169369e-01,
];

const PARAMETERS: [(&str, &[i32]); 73] = [
    ("conv_pre.bias", &[16]),
    ("conv_pre.weight", &[16, 8, 7]),
    ("ups.0.0.bias", &[8]),
    ("ups.0.0.weight", &[16, 8, 4]),
    ("ups.1.0.bias", &[4]),
    ("ups.1.0.weight", &[8, 4, 4]),
    ("resblocks.0.convs1.0.bias", &[8]),
    ("resblocks.0.convs1.0.weight", &[8, 8, 3]),
    ("resblocks.0.convs1.1.bias", &[8]),
    ("resblocks.0.convs1.1.weight", &[8, 8, 3]),
    ("resblocks.0.convs2.0.bias", &[8]),
    ("resblocks.0.convs2.0.weight", &[8, 8, 3]),
    ("resblocks.0.convs2.1.bias", &[8]),
    ("resblocks.0.convs2.1.weight", &[8, 8, 3]),
    ("resblocks.0.activations.0.act.alpha", &[8]),
    ("resblocks.0.activations.0.act.beta", &[8]),
    ("resblocks.0.activations.1.act.alpha", &[8]),
    ("resblocks.0.activations.1.act.beta", &[8]),
    ("resblocks.0.activations.2.act.alpha", &[8]),
    ("resblocks.0.activations.2.act.beta", &[8]),
    ("resblocks.0.activations.3.act.alpha", &[8]),
    ("resblocks.0.activations.3.act.beta", &[8]),
    ("resblocks.1.convs1.0.bias", &[8]),
    ("resblocks.1.convs1.0.weight", &[8, 8, 5]),
    ("resblocks.1.convs1.1.bias", &[8]),
    ("resblocks.1.convs1.1.weight", &[8, 8, 5]),
    ("resblocks.1.convs2.0.bias", &[8]),
    ("resblocks.1.convs2.0.weight", &[8, 8, 5]),
    ("resblocks.1.convs2.1.bias", &[8]),
    ("resblocks.1.convs2.1.weight", &[8, 8, 5]),
    ("resblocks.1.activations.0.act.alpha", &[8]),
    ("resblocks.1.activations.0.act.beta", &[8]),
    ("resblocks.1.activations.1.act.alpha", &[8]),
    ("resblocks.1.activations.1.act.beta", &[8]),
    ("resblocks.1.activations.2.act.alpha", &[8]),
    ("resblocks.1.activations.2.act.beta", &[8]),
    ("resblocks.1.activations.3.act.alpha", &[8]),
    ("resblocks.1.activations.3.act.beta", &[8]),
    ("resblocks.2.convs1.0.bias", &[4]),
    ("resblocks.2.convs1.0.weight", &[4, 4, 3]),
    ("resblocks.2.convs1.1.bias", &[4]),
    ("resblocks.2.convs1.1.weight", &[4, 4, 3]),
    ("resblocks.2.convs2.0.bias", &[4]),
    ("resblocks.2.convs2.0.weight", &[4, 4, 3]),
    ("resblocks.2.convs2.1.bias", &[4]),
    ("resblocks.2.convs2.1.weight", &[4, 4, 3]),
    ("resblocks.2.activations.0.act.alpha", &[4]),
    ("resblocks.2.activations.0.act.beta", &[4]),
    ("resblocks.2.activations.1.act.alpha", &[4]),
    ("resblocks.2.activations.1.act.beta", &[4]),
    ("resblocks.2.activations.2.act.alpha", &[4]),
    ("resblocks.2.activations.2.act.beta", &[4]),
    ("resblocks.2.activations.3.act.alpha", &[4]),
    ("resblocks.2.activations.3.act.beta", &[4]),
    ("resblocks.3.convs1.0.bias", &[4]),
    ("resblocks.3.convs1.0.weight", &[4, 4, 5]),
    ("resblocks.3.convs1.1.bias", &[4]),
    ("resblocks.3.convs1.1.weight", &[4, 4, 5]),
    ("resblocks.3.convs2.0.bias", &[4]),
    ("resblocks.3.convs2.0.weight", &[4, 4, 5]),
    ("resblocks.3.convs2.1.bias", &[4]),
    ("resblocks.3.convs2.1.weight", &[4, 4, 5]),
    ("resblocks.3.activations.0.act.alpha", &[4]),
    ("resblocks.3.activations.0.act.beta", &[4]),
    ("resblocks.3.activations.1.act.alpha", &[4]),
    ("resblocks.3.activations.1.act.beta", &[4]),
    ("resblocks.3.activations.2.act.alpha", &[4]),
    ("resblocks.3.activations.2.act.beta", &[4]),
    ("resblocks.3.activations.3.act.alpha", &[4]),
    ("resblocks.3.activations.3.act.beta", &[4]),
    ("activation_post.act.alpha", &[4]),
    ("activation_post.act.beta", &[4]),
    ("conv_post.weight", &[1, 4, 7]),
];

const WAVEFORM: [f32; 24] = [
    5.260239e-01, 6.043837e-01, 4.086704e-01, 1.1123174e-01,
    -8.452295e-02, -1.707218e-01, -7.8501254e-02, 2.523513e-01,
    5.1139235e-01, 7.094442e-01, 7.2350764e-01, 6.4065343e-01,
    4.3932718e-01, 1.9496334e-01, -3.1578213e-02, -6.1400194e-02,
    8.7036565e-03, 9.0777494e-02, 9.9419035e-02, 1.3862309e-01,
    2.9710016e-01, 6.770912e-01, 7.5320965e-01, 4.8737505e-01,
];

// WAVEFORM: peak 0.7532, 0 samples on the bound
// 4448 parameters

const WAVEFORM_CLAMPED: [f32; 24] = [
    -9.265512e-01, -1.0e+00, -1.0e+00, -1.0e+00,
    -1.0e+00, -1.0e+00, -1.0e+00, -1.0e+00,
    8.3868474e-01, 1.0e+00, 1.0e+00, 1.0e+00,
    1.0e+00, -1.9330478e-01, -1.0e+00, -1.0e+00,
    -1.0e+00, -1.0e+00, -1.0e+00, 1.0e+00,
    1.0e+00, 1.0e+00, 1.0e+00, 1.0e+00,
];

// WAVEFORM_CLAMPED: peak 1.0000, 21 samples on the bound
// 4448 parameters

const WAVEFORM_TANH: [f32; 24] = [
    -7.289819e-01, -9.9985117e-01, -9.9999976e-01, -1.0e+00,
    -1.0e+00, -9.999995e-01, -9.9997425e-01, -9.812165e-01,
    6.8511176e-01, 9.8627704e-01, 9.961089e-01, 9.949795e-01,
    9.8381215e-01, -1.9093251e-01, -9.977982e-01, -9.99995e-01,
    -9.999996e-01, -9.9999493e-01, -9.975054e-01, 9.6627825e-01,
    9.9999976e-01, 1.0e+00, 1.0e+00, 1.0e+00,
];

// WAVEFORM_TANH: peak 1.0000, 13 samples on the bound
// 4448 parameters

