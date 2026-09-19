//! The audio primitives on the card, against the same primitives on the host.
//!
//! `tests/audio.rs` is what says these functions are correct; it checks each one against its own
//! definition, on the CPU. What this file is for is the claim the module is built on -- that a
//! 1-D convolution written as a 2-D one, and a Fourier transform written as a convolution, run
//! wherever `conv2d` and `matmul` run, with no kernel of their own on any backend. That claim is
//! either true on a second backend or it is not true at all.
//!
//! So each test here writes one graph, runs it twice, and compares. The tolerances are loose
//! because the CUDA operators work in half precision and the host ones do not: what is being
//! tested is that the composition holds, not that sixteen bits equal thirty-two.
//!
//! They need a CUDA device and a build configured with `WITH_CUDA`, so they are `#[ignore]`d and
//! run with `cargo test --test audio_cuda -- --ignored`. On a machine without one the library
//! ends the process rather than reporting an error, which is why they do not detect and skip.

use std::collections::HashMap;

use waifu::audio::{
    conv1d, conv_transpose1d, depthwise_conv1d, hann_window, istft, istft_basis, pad1d, snake,
    stft, stft_basis, window_envelope, Padding,
};
use waifu::flint::{functional as F, DType, Device, Graph, Ir, RunContext, Tensor, Value};

/// The float type the CUDA operators work in, which the inputs have to be in already.
fn cuda_float() -> DType {
    F::default_float_type(Device::Cuda).unwrap()
}

fn run(g: &Graph, out: Value, inputs: &[(&str, &Tensor)]) -> (Vec<i32>, Vec<f32>) {
    g.output("out", out);

    let weights: HashMap<String, Tensor> = HashMap::new();
    let mut context = RunContext::new(&weights);
    for (name, tensor) in inputs {
        context = context.input(name, tensor);
    }

    let outputs = Ir::compile(g).run(&context).unwrap();
    let tensor = outputs[0]
        .1
        .to_device(Device::Cpu)
        .unwrap()
        .cast(DType::Float)
        .unwrap();

    (tensor.shape(), tensor.to_vec_f32().unwrap())
}

/// The same tensor on the host and on the card, the second in whatever float the card works in.
fn pair(shape: &[i32], values: &[f32]) -> (Tensor, Tensor) {
    let host = Tensor::from_f32(shape, values).unwrap();
    let device = host
        .cast(cuda_float())
        .unwrap()
        .to_device(Device::Cuda)
        .unwrap();

    (host, device)
}

/// Small numbers, so that half precision has bits left over for the accumulation.
fn ramp(n: usize, seed: f32) -> Vec<f32> {
    (0..n)
        .map(|i| (i as f32 * 0.37 + seed).sin() * 0.5)
        .collect()
}

fn close(got: &[f32], want: &[f32], tolerance: f32, what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: lengths differ");

    for (i, (a, b)) in got.iter().zip(want).enumerate() {
        assert!(
            (a - b).abs() <= tolerance * (1.0 + b.abs()),
            "{what} element {i}: {a} is not {b}"
        );
    }
}

#[test]
#[ignore = "needs a CUDA device"]
fn conv1d_on_the_card_is_conv1d_on_the_host() {
    // (in, out, length, kernel, stride, padding, dilation, groups).
    let cases = [
        (
            3usize, 4usize, 12usize, 3usize, 1usize, 1usize, 1usize, 1usize,
        ),
        (2, 6, 16, 5, 2, 2, 1, 1),
        (4, 4, 20, 3, 1, 4, 4, 1),
        // No grouped case here. `flint`'s CUDA conv2d is CUTLASS's and it throws on any group
        // count above one, while the cuDNN path that would handle it is built into the benchmark
        // rather than the runtime. `tests/audio.rs` covers grouped and depthwise on the
        // processor; see the note on `conv1d` for what the card would need.
    ];

    for (index, &(inc, outc, length, kernel, stride, padding, dilation, groups)) in
        cases.iter().enumerate()
    {
        let batch = 2;
        let x = ramp(batch * inc * length, index as f32);
        let w = ramp(outc * (inc / groups) * kernel, index as f32 + 2.0);
        let bias = ramp(outc, index as f32 + 4.0);

        let (x_host, x_cuda) = pair(&[batch as i32, inc as i32, length as i32], &x);
        let (w_host, w_cuda) = pair(&[outc as i32, (inc / groups) as i32, kernel as i32], &w);
        let (bias_host, bias_cuda) = pair(&[outc as i32], &bias);

        let build = |dtype, device| {
            let g = Graph::new();
            let out = conv1d(
                &g,
                g.input("x"),
                g.input("w"),
                Some(g.input("bias")),
                stride as i32,
                padding as i32,
                dilation as i32,
                groups as i32,
                dtype,
                device,
            )
            .unwrap();
            (g, out)
        };

        let (g, out) = build(DType::Float, Device::Cpu);
        let (host_shape, host) = run(
            &g,
            out,
            &[("x", &x_host), ("w", &w_host), ("bias", &bias_host)],
        );

        let (g, out) = build(cuda_float(), Device::Cuda);
        let (cuda_shape, cuda) = run(
            &g,
            out,
            &[("x", &x_cuda), ("w", &w_cuda), ("bias", &bias_cuda)],
        );

        assert_eq!(
            host_shape, cuda_shape,
            "case {index} disagreed on the shape"
        );
        close(&cuda, &host, 2e-2, &format!("conv1d case {index}"));
    }
}

#[test]
#[ignore = "needs a CUDA device"]
fn conv_transpose1d_on_the_card_is_conv_transpose1d_on_the_host() {
    // The shapes a vocoder upsamples with, plus a kernel that is not a whole number of strides.
    let cases = [
        (2usize, 3usize, 6usize, 4usize, 2usize, 0usize),
        (4, 2, 8, 16, 8, 4),
        (3, 3, 7, 5, 2, 1),
    ];

    for (index, &(inc, outc, length, kernel, stride, padding)) in cases.iter().enumerate() {
        let batch = 2;
        let x = ramp(batch * inc * length, index as f32);
        let w = ramp(inc * outc * kernel, index as f32 + 6.0);

        let (x_host, x_cuda) = pair(&[batch as i32, inc as i32, length as i32], &x);
        let (w_host, w_cuda) = pair(&[inc as i32, outc as i32, kernel as i32], &w);

        let build = |dtype, device| {
            let g = Graph::new();
            let out = conv_transpose1d(
                &g,
                g.input("x"),
                g.input("w"),
                None,
                inc as i32,
                outc as i32,
                kernel as i32,
                stride as i32,
                padding as i32,
                dtype,
                device,
            )
            .unwrap();
            (g, out)
        };

        let (g, out) = build(DType::Float, Device::Cpu);
        let (host_shape, host) = run(&g, out, &[("x", &x_host), ("w", &w_host)]);

        let (g, out) = build(cuda_float(), Device::Cuda);
        let (cuda_shape, cuda) = run(&g, out, &[("x", &x_cuda), ("w", &w_cuda)]);

        assert_eq!(
            host_shape, cuda_shape,
            "case {index} disagreed on the shape"
        );
        close(
            &cuda,
            &host,
            2e-2,
            &format!("conv_transpose1d case {index}"),
        );
    }
}

/// A reflection is a permutation, so it is exact in any float wide enough to hold the samples.
#[test]
#[ignore = "needs a CUDA device"]
fn reflect_padding_on_the_card_mirrors_the_way_it_does_on_the_host() {
    let values = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let (_, cuda) = pair(&[1, 1, 6], &values);

    let g = Graph::new();
    let out = pad1d(
        &g,
        g.input("x"),
        2,
        3,
        Padding::Reflect,
        cuda_float(),
        Device::Cuda,
    )
    .unwrap();
    let (shape, got) = run(&g, out, &[("x", &cuda)]);

    assert_eq!(shape, vec![1, 1, 11]);
    assert_eq!(
        got,
        vec![3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 5.0, 4.0, 3.0]
    );
}

#[test]
#[ignore = "needs a CUDA device"]
fn snake_on_the_card_is_snake_on_the_host() {
    let (channels, length) = (4usize, 16usize);
    let x = ramp(channels * length, 3.0);
    let alpha: Vec<f32> = (0..channels).map(|c| 0.5 + c as f32 * 0.3).collect();
    let beta: Vec<f32> = (0..channels).map(|c| 1.0 + c as f32 * 0.2).collect();

    let (x_host, x_cuda) = pair(&[1, channels as i32, length as i32], &x);
    let (a_host, a_cuda) = pair(&[channels as i32], &alpha);
    let (b_host, b_cuda) = pair(&[channels as i32], &beta);

    let build = |dtype, device| {
        let g = Graph::new();
        let out = snake(
            &g,
            g.input("x"),
            g.input("alpha"),
            Some(g.input("beta")),
            1e-6,
            dtype,
            device,
        )
        .unwrap();
        (g, out)
    };

    let (g, out) = build(DType::Float, Device::Cpu);
    let (_, host) = run(
        &g,
        out,
        &[("x", &x_host), ("alpha", &a_host), ("beta", &b_host)],
    );

    let (g, out) = build(cuda_float(), Device::Cuda);
    let (_, cuda) = run(
        &g,
        out,
        &[("x", &x_cuda), ("alpha", &a_cuda), ("beta", &b_cuda)],
    );

    close(&cuda, &host, 2e-2, "snake");
}

/// The transform and its inverse, on the card, back to the signal they started as.
///
/// A looser tolerance than the host test uses: a 64-point transform in half precision sums 64
/// products per bin and then sums them back, and that is where the bits went. What it is here to
/// catch is a composition that does not survive the move -- a view that needed a contiguous
/// tensor, a broadcast that lined up differently -- not the last digit.
#[test]
#[ignore = "needs a CUDA device"]
fn an_istft_of_an_stft_on_the_card_is_still_the_signal() {
    let (n_fft, hop, length) = (64usize, 16usize, 256usize);
    let window = hann_window(n_fft);
    let signal = ramp(length, 8.0);
    let frames = (length - n_fft) / hop + 1;

    let (_, x) = pair(&[1, 1, length as i32], &signal);

    let basis = stft_basis(n_fft, &window)
        .unwrap()
        .cast(cuda_float())
        .unwrap()
        .to_device(Device::Cuda)
        .unwrap();
    let inverse = istft_basis(n_fft, &window)
        .unwrap()
        .cast(cuda_float())
        .unwrap()
        .to_device(Device::Cuda)
        .unwrap();
    let envelope = window_envelope(&window, hop, frames, 1e-6)
        .unwrap()
        .cast(cuda_float())
        .unwrap()
        .to_device(Device::Cuda)
        .unwrap();

    let g = Graph::new();
    let (real, imaginary) = stft(
        &g,
        g.input("x"),
        g.constant(basis),
        n_fft as i32,
        hop as i32,
        None,
        cuda_float(),
        Device::Cuda,
    )
    .unwrap();
    let back = istft(
        &g,
        real,
        imaginary,
        g.constant(inverse),
        g.constant(envelope),
        n_fft as i32,
        hop as i32,
        false,
        cuda_float(),
        Device::Cuda,
    )
    .unwrap();

    let (shape, got) = run(&g, back, &[("x", &x)]);
    assert_eq!(shape, vec![1, 1, ((frames - 1) * hop + n_fft) as i32]);

    let covered = n_fft..(frames - 1) * hop;
    close(
        &got[covered.clone()],
        &signal[covered],
        5e-2,
        "istft of stft",
    );
}

/// The one that matters: this is the shape `conv1d` refuses on a card, computed on the card.
///
/// A conformer's convolution module is depthwise, and so is w2v-bert-2.0's, so without this
/// IndexTTS-2.5 has no GPU path at all. The host side is `conv1d` with one group per channel --
/// the thing CUTLASS will not do -- which is exactly the comparison worth making.
#[test]
#[ignore = "needs a CUDA device"]
fn a_depthwise_convolution_runs_on_the_card_where_a_grouped_one_cannot() {
    // (channels, length, kernel, padding, dilation), including the 31 tap conformer kernel.
    let cases = [
        (8usize, 24usize, 3usize, 1usize, 1usize),
        (16, 48, 31, 15, 1),
        (6, 32, 3, 4, 4),
    ];

    for (index, &(channels, length, kernel, padding, dilation)) in cases.iter().enumerate() {
        let batch = 2;
        let x = ramp(batch * channels * length, index as f32);
        let w = ramp(channels * kernel, index as f32 + 3.0);

        let (x_host, x_cuda) = pair(&[batch as i32, channels as i32, length as i32], &x);
        let (w_host, w_cuda) = pair(&[channels as i32, 1, kernel as i32], &w);

        // On the host, by the grouped convolution this is meant to stand in for.
        let g = Graph::new();
        let out = conv1d(
            &g,
            g.input("x"),
            g.input("w"),
            None,
            1,
            padding as i32,
            dilation as i32,
            channels as i32,
            DType::Float,
            Device::Cpu,
        )
        .unwrap();
        let (host_shape, host) = run(&g, out, &[("x", &x_host), ("w", &w_host)]);

        // On the card, by the composition.
        let g = Graph::new();
        let out = depthwise_conv1d(
            &g,
            g.input("x"),
            g.input("w"),
            None,
            channels as i32,
            kernel as i32,
            padding as i32,
            dilation as i32,
            cuda_float(),
            Device::Cuda,
        )
        .unwrap();
        let (cuda_shape, cuda) = run(&g, out, &[("x", &x_cuda), ("w", &w_cuda)]);

        assert_eq!(
            host_shape, cuda_shape,
            "case {index} disagreed on the shape"
        );
        close(&cuda, &host, 2e-2, &format!("depthwise case {index}"));
    }
}
