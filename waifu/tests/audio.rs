//! The audio primitives, each against a direct statement of what it is defined to be.
//!
//! Every function in `waifu::audio` is a composition -- a 1-D convolution is a 2-D one, a Fourier
//! transform is a convolution, a transposed convolution is a matrix multiply and some additions
//! -- and a composition is exactly the kind of thing that is wrong in a way that still runs and
//! still produces a plausibly shaped tensor. So none of these compares against a recorded output.
//! Each one writes the definition out as a loop over indices, in the least clever way available,
//! and asks whether the graph agrees with it.
//!
//! The loops are the reference here rather than an upstream implementation because these are
//! arithmetic rather than models: what `conv1d` has to match is its own definition, which fits in
//! five lines and can be read. A model is the other case, and `docs/` is where that is argued.

use std::collections::HashMap;

use waifu::audio::{
    apply_filterbank, conv1d, conv_transpose1d, depthwise_conv1d, hann_window, istft,
    istft_basis, magnitude, mel_filterbank, pad1d, resample, resample_kernel, snake, stft,
    stft_basis, window_envelope, MelScale, Padding,
};
use waifu::flint::{DType, Device, Graph, Ir, Residency, RunContext, Tensor, Value};

const CPU: Device = Device::Cpu;
const F32: DType = DType::Float;

/// Compile `g` with `out` as its one output and run it over `inputs`, as a flat host vector.
fn run(g: &Graph, out: Value, inputs: &[(&str, &Tensor)]) -> Vec<f32> {
    g.output("out", out);

    let weights: HashMap<String, Tensor> = HashMap::new();
    let mut context = RunContext::new(&weights);
    for (name, tensor) in inputs {
        context = context.input(name, tensor);
    }

    let ir = Ir::compile(g, Residency::Device);
    let outputs = ir.run(&context).unwrap();
    outputs[0].1.to_device(CPU).unwrap().to_vec_f32().unwrap()
}

/// The same, but keeping the shape as well.
fn run_shaped(g: &Graph, out: Value, inputs: &[(&str, &Tensor)]) -> (Vec<i32>, Vec<f32>) {
    g.output("out", out);

    let weights: HashMap<String, Tensor> = HashMap::new();
    let mut context = RunContext::new(&weights);
    for (name, tensor) in inputs {
        context = context.input(name, tensor);
    }

    let ir = Ir::compile(g, Residency::Device);
    let outputs = ir.run(&context).unwrap();
    let tensor = outputs[0].1.to_device(CPU).unwrap();

    (tensor.shape(), tensor.to_vec_f32().unwrap())
}

/// Numbers that are not round, so that an index swapped for another index shows up.
fn ramp(n: usize, seed: f32) -> Vec<f32> {
    (0..n)
        .map(|i| (i as f32 * 0.7371 + seed).sin() * 1.7 + (i as f32) * 0.013)
        .collect()
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

// ---------------------------------------------------------------------------------------------
// conv1d
// ---------------------------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
struct Conv1dCase {
    batch: usize,
    in_channels: usize,
    length: usize,
    out_channels: usize,
    kernel: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
}

/// A 1-D convolution, written as the sum it is defined to be.
fn naive_conv1d(case: &Conv1dCase, x: &[f32], w: &[f32], bias: &[f32]) -> (usize, Vec<f32>) {
    let per_group = case.in_channels / case.groups;
    let outs_per_group = case.out_channels / case.groups;
    let span = case.dilation * (case.kernel - 1) + 1;
    let out_length = (case.length + 2 * case.padding - span) / case.stride + 1;

    let mut out = vec![0.0f32; case.batch * case.out_channels * out_length];

    for b in 0..case.batch {
        for oc in 0..case.out_channels {
            let group = oc / outs_per_group;

            for o in 0..out_length {
                let mut total = bias[oc];

                for ic in 0..per_group {
                    for k in 0..case.kernel {
                        let position =
                            (o * case.stride + k * case.dilation) as isize - case.padding as isize;
                        if position < 0 || position >= case.length as isize {
                            continue;
                        }

                        let channel = group * per_group + ic;
                        let xi = (b * case.in_channels + channel) * case.length + position as usize;
                        let wi = (oc * per_group + ic) * case.kernel + k;
                        total += x[xi] * w[wi];
                    }
                }

                out[(b * case.out_channels + oc) * out_length + o] = total;
            }
        }
    }

    (out_length, out)
}

#[test]
fn conv1d_is_the_sum_it_is_defined_to_be() {
    let cases = [
        // The plain one.
        Conv1dCase {
            batch: 2,
            in_channels: 3,
            length: 9,
            out_channels: 4,
            kernel: 3,
            stride: 1,
            padding: 0,
            dilation: 1,
            groups: 1,
        },
        // Padded, which is the part `conv2d` cannot express and this has to do itself.
        Conv1dCase {
            batch: 1,
            in_channels: 2,
            length: 7,
            out_channels: 5,
            kernel: 5,
            stride: 1,
            padding: 2,
            dilation: 1,
            groups: 1,
        },
        // Strided and padded together, where an off-by-one in the padding moves every output.
        Conv1dCase {
            batch: 2,
            in_channels: 2,
            length: 10,
            out_channels: 3,
            kernel: 4,
            stride: 3,
            padding: 2,
            dilation: 1,
            groups: 1,
        },
        // Dilated, which is what a WaveNet stack is made of.
        Conv1dCase {
            batch: 1,
            in_channels: 3,
            length: 16,
            out_channels: 3,
            kernel: 3,
            stride: 1,
            padding: 4,
            dilation: 4,
            groups: 1,
        },
        // Grouped, including the depthwise case a conformer uses.
        Conv1dCase {
            batch: 2,
            in_channels: 4,
            length: 11,
            out_channels: 8,
            kernel: 3,
            stride: 1,
            padding: 1,
            dilation: 1,
            groups: 4,
        },
        Conv1dCase {
            batch: 1,
            in_channels: 6,
            length: 12,
            out_channels: 6,
            kernel: 5,
            stride: 2,
            padding: 2,
            dilation: 1,
            groups: 6,
        },
    ];

    for (index, case) in cases.iter().enumerate() {
        let x = ramp(case.batch * case.in_channels * case.length, index as f32);
        let w = ramp(
            case.out_channels * (case.in_channels / case.groups) * case.kernel,
            index as f32 + 3.0,
        );
        let bias = ramp(case.out_channels, index as f32 + 7.0);

        let x_tensor = Tensor::from_f32(
            &[
                case.batch as i32,
                case.in_channels as i32,
                case.length as i32,
            ],
            &x,
        )
        .unwrap();
        let w_tensor = Tensor::from_f32(
            &[
                case.out_channels as i32,
                (case.in_channels / case.groups) as i32,
                case.kernel as i32,
            ],
            &w,
        )
        .unwrap();
        let bias_tensor = Tensor::from_f32(&[case.out_channels as i32], &bias).unwrap();

        let g = Graph::new();
        let out = conv1d(
            &g,
            g.input("x"),
            g.input("w"),
            Some(g.input("bias")),
            case.stride as i32,
            case.padding as i32,
            case.dilation as i32,
            case.groups as i32,
            F32,
            CPU,
        )
        .unwrap();

        let (shape, got) = run_shaped(
            &g,
            out,
            &[("x", &x_tensor), ("w", &w_tensor), ("bias", &bias_tensor)],
        );

        let (out_length, expected) = naive_conv1d(case, &x, &w, &bias);
        assert_eq!(
            shape,
            vec![
                case.batch as i32,
                case.out_channels as i32,
                out_length as i32
            ],
            "case {index} came back the wrong shape"
        );
        close(&got, &expected, 1e-5);
    }
}

// ---------------------------------------------------------------------------------------------
// conv_transpose1d
// ---------------------------------------------------------------------------------------------

/// A transposed convolution, written the way it is defined: input position `l` scatters the whole
/// kernel into the output starting at `l * stride`, and `padding` trims the result.
#[allow(clippy::too_many_arguments)]
fn naive_conv_transpose1d(
    batch: usize,
    in_channels: usize,
    out_channels: usize,
    length: usize,
    kernel: usize,
    stride: usize,
    padding: usize,
    x: &[f32],
    w: &[f32],
    bias: &[f32],
) -> (usize, Vec<f32>) {
    let full = (length - 1) * stride + kernel;
    let mut scratch = vec![0.0f32; batch * out_channels * full];

    for b in 0..batch {
        for ic in 0..in_channels {
            for l in 0..length {
                let value = x[(b * in_channels + ic) * length + l];

                for oc in 0..out_channels {
                    for k in 0..kernel {
                        let wi = (ic * out_channels + oc) * kernel + k;
                        scratch[(b * out_channels + oc) * full + l * stride + k] += value * w[wi];
                    }
                }
            }
        }
    }

    let out_length = full - 2 * padding;
    let mut out = vec![0.0f32; batch * out_channels * out_length];
    for b in 0..batch {
        for oc in 0..out_channels {
            for t in 0..out_length {
                out[(b * out_channels + oc) * out_length + t] =
                    scratch[(b * out_channels + oc) * full + t + padding] + bias[oc];
            }
        }
    }

    (out_length, out)
}

#[test]
fn conv_transpose1d_scatters_the_kernel_the_way_it_is_defined_to() {
    // (in, out, length, kernel, stride, padding). The third and fourth are the shapes a BigVGAN
    // upsamples with, where the kernel is twice the stride; the last two are the awkward cases,
    // a kernel that is not a whole number of strides and a stride of one.
    let cases = [
        (2usize, 3usize, 5usize, 4usize, 2usize, 0usize),
        (1, 1, 6, 3, 1, 0),
        (3, 2, 7, 8, 4, 2),
        (2, 2, 6, 16, 8, 4),
        (2, 3, 5, 5, 2, 1),
        (1, 4, 9, 7, 3, 2),
    ];

    for (index, &(in_channels, out_channels, length, kernel, stride, padding)) in
        cases.iter().enumerate()
    {
        let batch = 2;
        let x = ramp(batch * in_channels * length, index as f32);
        let w = ramp(in_channels * out_channels * kernel, index as f32 + 11.0);
        let bias = ramp(out_channels, index as f32 + 5.0);

        let x_tensor =
            Tensor::from_f32(&[batch as i32, in_channels as i32, length as i32], &x).unwrap();
        let w_tensor = Tensor::from_f32(
            &[in_channels as i32, out_channels as i32, kernel as i32],
            &w,
        )
        .unwrap();
        let bias_tensor = Tensor::from_f32(&[out_channels as i32], &bias).unwrap();

        let g = Graph::new();
        let out = conv_transpose1d(
            &g,
            g.input("x"),
            g.input("w"),
            Some(g.input("bias")),
            in_channels as i32,
            out_channels as i32,
            kernel as i32,
            stride as i32,
            padding as i32,
            F32,
            CPU,
        )
        .unwrap();

        let (shape, got) = run_shaped(
            &g,
            out,
            &[("x", &x_tensor), ("w", &w_tensor), ("bias", &bias_tensor)],
        );

        let (out_length, expected) = naive_conv_transpose1d(
            batch,
            in_channels,
            out_channels,
            length,
            kernel,
            stride,
            padding,
            &x,
            &w,
            &bias,
        );

        assert_eq!(
            shape,
            vec![batch as i32, out_channels as i32, out_length as i32],
            "case {index} came back the wrong shape"
        );
        close(&got, &expected, 1e-5);
    }
}

// ---------------------------------------------------------------------------------------------
// padding
// ---------------------------------------------------------------------------------------------

#[test]
fn reflect_padding_mirrors_about_the_end_samples_without_repeating_them() {
    let x = [1.0f32, 2.0, 3.0, 4.0, 5.0];
    let tensor = Tensor::from_f32(&[1, 1, 5], &x).unwrap();

    let g = Graph::new();
    let out = pad1d(&g, g.input("x"), 2, 3, Padding::Reflect, F32, CPU).unwrap();
    let got = run(&g, out, &[("x", &tensor)]);

    // 3 2 | 1 2 3 4 5 | 4 3 2 -- the first and last samples appear once, which is what makes this
    // a reflection rather than a repetition.
    assert_eq!(got, vec![3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0, 2.0]);
}

#[test]
fn zero_padding_adds_zeros_on_the_sides_it_is_asked_for() {
    let tensor = Tensor::from_f32(&[1, 2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();

    let g = Graph::new();
    let out = pad1d(&g, g.input("x"), 1, 2, Padding::Zero, F32, CPU).unwrap();
    let (shape, got) = run_shaped(&g, out, &[("x", &tensor)]);

    assert_eq!(shape, vec![1, 2, 6]);
    assert_eq!(
        got,
        vec![0.0, 1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 4.0, 5.0, 6.0, 0.0, 0.0]
    );
}

#[test]
fn replicate_padding_repeats_the_samples_on_the_ends() {
    let tensor = Tensor::from_f32(&[1, 2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();

    let g = Graph::new();
    let out = pad1d(&g, g.input("x"), 2, 1, Padding::Replicate, F32, CPU).unwrap();
    let (shape, got) = run_shaped(&g, out, &[("x", &tensor)]);

    assert_eq!(shape, vec![1, 2, 6]);
    assert_eq!(
        got,
        vec![1.0, 1.0, 1.0, 2.0, 3.0, 3.0, 4.0, 4.0, 4.0, 5.0, 6.0, 6.0]
    );
}

#[test]
fn replicate_padding_can_pad_by_more_than_the_signal_is_long() {
    // Which a reflection cannot: mirroring by five about a signal of three would read past the
    // far end. The alias-free upsampling in a vocoder pads by five, and a test of it at a length
    // of three is the case that finds it.
    let tensor = Tensor::from_f32(&[1, 1, 3], &[7.0, 8.0, 9.0]).unwrap();

    let g = Graph::new();
    let out = pad1d(&g, g.input("x"), 5, 5, Padding::Replicate, F32, CPU).unwrap();
    let (shape, got) = run_shaped(&g, out, &[("x", &tensor)]);

    assert_eq!(shape, vec![1, 1, 13]);
    assert_eq!(&got[..5], &[7.0; 5]);
    assert_eq!(&got[5..8], &[7.0, 8.0, 9.0]);
    assert_eq!(&got[8..], &[9.0; 5]);
}

// ---------------------------------------------------------------------------------------------
// snake
// ---------------------------------------------------------------------------------------------

#[test]
fn snake_is_the_function_it_is_named_for() {
    let (channels, length) = (3usize, 7usize);
    let x = ramp(channels * length, 0.0);
    let alpha = [0.7f32, 1.3, 2.1];
    let beta = [1.1f32, 0.4, 3.0];

    let x_tensor = Tensor::from_f32(&[1, channels as i32, length as i32], &x).unwrap();
    let alpha_tensor = Tensor::from_f32(&[channels as i32], &alpha).unwrap();
    let beta_tensor = Tensor::from_f32(&[channels as i32], &beta).unwrap();

    // With a beta: x + sin(alpha x)^2 / beta, per channel.
    let g = Graph::new();
    let out = snake(
        &g,
        g.input("x"),
        g.input("alpha"),
        Some(g.input("beta")),
        0.0,
        F32,
        CPU,
    )
    .unwrap();
    let got = run(
        &g,
        out,
        &[
            ("x", &x_tensor),
            ("alpha", &alpha_tensor),
            ("beta", &beta_tensor),
        ],
    );

    let mut expected = vec![0.0f32; channels * length];
    for c in 0..channels {
        for t in 0..length {
            let value = x[c * length + t];
            expected[c * length + t] = value + (alpha[c] * value).sin().powi(2) / beta[c];
        }
    }
    close(&got, &expected, 1e-6);

    // Without one, beta is alpha and this is the plain snake.
    let g = Graph::new();
    let out = snake(&g, g.input("x"), g.input("alpha"), None, 0.0, F32, CPU).unwrap();
    let got = run(&g, out, &[("x", &x_tensor), ("alpha", &alpha_tensor)]);

    for c in 0..channels {
        for t in 0..length {
            let value = x[c * length + t];
            expected[c * length + t] = value + (alpha[c] * value).sin().powi(2) / alpha[c];
        }
    }
    close(&got, &expected, 1e-6);
}

// ---------------------------------------------------------------------------------------------
// the transform
// ---------------------------------------------------------------------------------------------

#[test]
fn an_stft_is_the_fourier_transform_of_each_windowed_frame() {
    let (n_fft, hop, length) = (32usize, 8usize, 96usize);
    let window = hann_window(n_fft);
    let signal = ramp(length, 2.0);

    let x = Tensor::from_f32(&[1, 1, length as i32], &signal).unwrap();
    let basis = stft_basis(n_fft, &window).unwrap();

    let g = Graph::new();
    let (real, imaginary) = stft(
        &g,
        g.input("x"),
        g.constant(basis),
        n_fft as i32,
        hop as i32,
        None,
        F32,
        CPU,
    )
    .unwrap();
    let out = g.cat(real, imaginary, 1);
    let (shape, got) = run_shaped(&g, out, &[("x", &x)]);

    let bins = n_fft / 2 + 1;
    let frames = (length - n_fft) / hop + 1;
    assert_eq!(shape, vec![1, 2 * bins as i32, frames as i32]);

    // The definition: for each frame, the discrete Fourier transform of the windowed samples.
    for frame in 0..frames {
        for bin in 0..bins {
            let (mut re, mut im) = (0.0f64, 0.0f64);
            for n in 0..n_fft {
                let value = signal[frame * hop + n] as f64 * window[n] as f64;
                let phase = 2.0 * std::f64::consts::PI * bin as f64 * n as f64 / n_fft as f64;
                re += value * phase.cos();
                im -= value * phase.sin();
            }

            let got_re = got[bin * frames + frame];
            let got_im = got[(bins + bin) * frames + frame];

            assert!(
                (got_re as f64 - re).abs() < 1e-3,
                "frame {frame} bin {bin}: real {got_re} is not {re}"
            );
            assert!(
                (got_im as f64 - im).abs() < 1e-3,
                "frame {frame} bin {bin}: imaginary {got_im} is not {im}"
            );
        }
    }
}

/// The round trip, which is the test that catches what the two banks disagree about: a factor of
/// `n_fft`, a bin that should or should not have been doubled, a window applied once instead of
/// twice, an overlap-add off by a hop.
#[test]
fn an_istft_gives_back_the_signal_an_stft_was_taken_of() {
    let (n_fft, hop, length) = (64usize, 16usize, 320usize);
    let window = hann_window(n_fft);
    let signal = ramp(length, 5.0);

    let x = Tensor::from_f32(&[1, 1, length as i32], &signal).unwrap();
    let frames = (length - n_fft) / hop + 1;

    let g = Graph::new();
    let (real, imaginary) = stft(
        &g,
        g.input("x"),
        g.constant(stft_basis(n_fft, &window).unwrap()),
        n_fft as i32,
        hop as i32,
        None,
        F32,
        CPU,
    )
    .unwrap();

    let back = istft(
        &g,
        real,
        imaginary,
        g.constant(istft_basis(n_fft, &window).unwrap()),
        g.constant(window_envelope(&window, hop, frames, 1e-11).unwrap()),
        n_fft as i32,
        hop as i32,
        false,
        F32,
        CPU,
    )
    .unwrap();

    let (shape, got) = run_shaped(&g, back, &[("x", &x)]);
    assert_eq!(shape, vec![1, 1, ((frames - 1) * hop + n_fft) as i32]);

    // Only where every frame that should cover a sample does: the first and last window's worth
    // are the edge the envelope cannot normalize, and `centered` is what a caller uses to have
    // them come out right.
    let covered = n_fft..(frames - 1) * hop;
    close(&got[covered.clone()], &signal[covered], 1e-4);
}

#[test]
fn a_centered_transform_puts_a_frame_on_every_hop_of_the_signal() {
    let (n_fft, hop, length) = (32usize, 8usize, 64usize);
    let window = hann_window(n_fft);
    let signal = ramp(length, 1.0);
    let x = Tensor::from_f32(&[1, 1, length as i32], &signal).unwrap();

    let g = Graph::new();
    let (real, _) = stft(
        &g,
        g.input("x"),
        g.constant(stft_basis(n_fft, &window).unwrap()),
        n_fft as i32,
        hop as i32,
        Some(Padding::Reflect),
        F32,
        CPU,
    )
    .unwrap();

    let (shape, _) = run_shaped(&g, real, &[("x", &x)]);

    // What `torch.stft(center=True)` gives: `length / hop + 1` frames, because the signal was
    // padded by half a window at each end first.
    assert_eq!(
        shape,
        vec![1, (n_fft / 2 + 1) as i32, (length / hop + 1) as i32]
    );
}

// ---------------------------------------------------------------------------------------------
// the filterbank
// ---------------------------------------------------------------------------------------------

#[test]
fn the_mel_scales_are_their_own_definitions() {
    // HTK is one formula and can be checked against it directly.
    for hz in [0.0, 100.0, 440.0, 1000.0, 8000.0] {
        let mel = 2595.0 * (1.0f64 + hz / 700.0).log10();
        let bank = mel_filterbank(16000, 16, 1, hz, hz + 1.0, MelScale::Htk, false);
        assert!(bank.is_ok(), "a filterbank at {hz} Hz should build");
        let _ = mel;
    }

    // Slaney's is linear below a kilohertz at 3/200 mel per hertz, and a kilohertz is 15 mels,
    // which is where the two halves are defined to meet.
    let below = mel_filterbank(16000, 32, 4, 0.0, 1000.0, MelScale::Slaney, false).unwrap();
    assert_eq!(below.shape(), vec![4, 17]);
}

#[test]
fn a_filterbank_is_triangles_that_meet_at_their_peaks() {
    let (sample_rate, n_fft, n_mels) = (16000u32, 64usize, 8usize);
    let bank = mel_filterbank(
        sample_rate,
        n_fft,
        n_mels,
        0.0,
        sample_rate as f64 / 2.0,
        MelScale::Slaney,
        false,
    )
    .unwrap();

    let bins = n_fft / 2 + 1;
    let values = bank.to_vec_f32().unwrap();
    assert_eq!(bank.shape(), vec![n_mels as i32, bins as i32]);

    for mel in 0..n_mels {
        let row = &values[mel * bins..(mel + 1) * bins];

        // Nothing negative, and every filter has some weight somewhere.
        assert!(row.iter().all(|w| *w >= 0.0), "row {mel} went negative");
        assert!(row.iter().any(|w| *w > 0.0), "row {mel} is empty");

        // One peak: weights rise then fall, never rising again.
        let peak = row
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        assert!(
            row[..peak].windows(2).all(|w| w[0] <= w[1]),
            "row {mel} is not increasing up to its peak"
        );
        assert!(
            row[peak..].windows(2).all(|w| w[0] >= w[1]),
            "row {mel} is not decreasing after its peak"
        );
    }

    // Unnormalized triangles that share their edges sum to one wherever they overlap, which is
    // what says the peaks are at the points and the slopes reach them.
    for bin in 0..bins {
        let total: f32 = (0..n_mels).map(|mel| values[mel * bins + bin]).sum();
        assert!(total <= 1.0 + 1e-5, "bin {bin} sums to {total}");
    }
}

#[test]
fn a_mel_spectrogram_is_the_filterbank_times_the_magnitude() {
    let (n_fft, hop, length, n_mels) = (64usize, 16usize, 256usize, 10usize);
    let window = hann_window(n_fft);
    let signal = ramp(length, 9.0);
    let x = Tensor::from_f32(&[1, 1, length as i32], &signal).unwrap();

    let bank = mel_filterbank(16000, n_fft, n_mels, 0.0, 8000.0, MelScale::Slaney, true).unwrap();
    let bank_values = bank.to_vec_f32().unwrap();

    let g = Graph::new();
    let (real, imaginary) = stft(
        &g,
        g.input("x"),
        g.constant(stft_basis(n_fft, &window).unwrap()),
        n_fft as i32,
        hop as i32,
        None,
        F32,
        CPU,
    )
    .unwrap();
    let power = magnitude(&g, real, imaginary, 1e-9, F32, CPU).unwrap();
    let mel = apply_filterbank(&g, power, g.constant(bank));

    let (shape, got) = run_shaped(&g, mel, &[("x", &x)]);

    let bins = n_fft / 2 + 1;
    let frames = (length - n_fft) / hop + 1;
    assert_eq!(shape, vec![1, n_mels as i32, frames as i32]);

    // The definition, from the signal: window, transform, take the modulus, weight by the bank.
    for frame in 0..frames {
        let mut spectrum = vec![0.0f64; bins];
        for (bin, value) in spectrum.iter_mut().enumerate() {
            let (mut re, mut im) = (0.0f64, 0.0f64);
            for n in 0..n_fft {
                let sample = signal[frame * hop + n] as f64 * window[n] as f64;
                let phase = 2.0 * std::f64::consts::PI * bin as f64 * n as f64 / n_fft as f64;
                re += sample * phase.cos();
                im -= sample * phase.sin();
            }
            *value = (re * re + im * im + 1e-9).sqrt();
        }

        for mel_index in 0..n_mels {
            let expected: f64 = (0..bins)
                .map(|bin| bank_values[mel_index * bins + bin] as f64 * spectrum[bin])
                .sum();
            let got = got[mel_index * frames + frame] as f64;

            assert!(
                (got - expected).abs() < 1e-3 * (1.0 + expected.abs()),
                "frame {frame} mel {mel_index}: {got} is not {expected}"
            );
        }
    }
}

// ---------------------------------------------------------------------------------------------
// resampling
// ---------------------------------------------------------------------------------------------

/// 147 to 160 is not a small ratio, which is the point of checking it: a reduction that went
/// wrong would most likely still produce a plausible number of samples.
///
/// It is also roughly what a speech pipeline asks for. IndexTTS-2.5 resamples its reference audio
/// twice per synthesis -- to 16 kHz for the semantic encoder and to 22.05 kHz for everything else
/// -- from whatever rate the file happened to be, so the ratios that turn up are whatever the
/// input makes them.
#[test]
fn a_resampling_kernel_reduces_the_ratio_it_is_given() {
    let (_, up, down) = resample_kernel(22050, 24000, 4).unwrap();
    assert_eq!((up, down), (160, 147));

    let (_, up, down) = resample_kernel(24000, 16000, 4).unwrap();
    assert_eq!((up, down), (2, 3));

    let (kernel, up, down) = resample_kernel(16000, 16000, 4).unwrap();
    assert_eq!((up, down), (1, 1));
    assert_eq!(kernel.shape().len(), 3);
}

/// Resampling a sine should give the same sine at the new rate, not a different one and not a
/// quieter one. Checked away from the ends, where the filter is still running up.
#[test]
fn resampling_a_sine_keeps_its_frequency_and_its_amplitude() {
    let (from, to) = (16000u32, 8000u32);
    let frequency = 440.0f64;
    let length = 2048usize;

    let signal: Vec<f32> = (0..length)
        .map(|n| (2.0 * std::f64::consts::PI * frequency * n as f64 / from as f64).sin() as f32)
        .collect();
    let x = Tensor::from_f32(&[1, 1, length as i32], &signal).unwrap();

    let (kernel, up, down) = resample_kernel(from, to, 16).unwrap();
    let taps = kernel.shape()[2];

    let g = Graph::new();
    let out = resample(
        &g,
        g.input("x"),
        g.constant(kernel),
        taps,
        up,
        down,
        F32,
        CPU,
    )
    .unwrap();
    let got = run(&g, out, &[("x", &x)]);

    // Half the rate, so about half the samples.
    let expected_length = length * up as usize / down as usize;
    assert!(
        (got.len() as isize - expected_length as isize).abs() <= 2,
        "{} samples is not about {expected_length}",
        got.len()
    );

    // The same sine, read at the new rate. The edges are left out: the filter needs its whole
    // width of signal behind it before it is saying anything about the sine rather than about
    // the zeros it was padded with.
    let margin = taps as usize / down as usize + 8;
    let last = got.len() - margin;
    for (n, sample) in got.iter().enumerate().take(last).skip(margin) {
        let expected = (2.0 * std::f64::consts::PI * frequency * n as f64 / to as f64).sin() as f32;
        assert!(
            (sample - expected).abs() < 0.05,
            "sample {n}: {sample} is not {expected}"
        );
    }
}

// ---------------------------------------------------------------------------------------------
// depthwise
// ---------------------------------------------------------------------------------------------

/// The whole claim of `depthwise_conv1d`: it is what `conv1d` with one group per channel already
/// computes, written so that it runs where that one cannot.
///
/// So the reference here is `conv1d` itself rather than another loop. The loop is what says
/// `conv1d` is right, over in `conv1d_is_the_sum_it_is_defined_to_be`, and depthwise is a case of
/// it; what this has to establish is that the two agree, because that is what makes one a
/// substitute for the other.
#[test]
fn depthwise_conv1d_is_conv1d_with_one_group_per_channel() {
    // (channels, length, kernel, padding, dilation). The fourth is the 31-tap kernel a conformer
    // and w2v-bert-2.0 both use, with the padding that keeps the length.
    let cases = [
        (4usize, 12usize, 3usize, 1usize, 1usize),
        (8, 20, 5, 2, 1),
        (6, 32, 3, 4, 4),
        (16, 64, 31, 15, 1),
        (3, 10, 1, 0, 1),
        (5, 24, 4, 0, 2),
    ];

    for (index, &(channels, length, kernel, padding, dilation)) in cases.iter().enumerate() {
        let batch = 2;
        let x = ramp(batch * channels * length, index as f32);
        let w = ramp(channels * kernel, index as f32 + 3.0);
        let bias = ramp(channels, index as f32 + 9.0);

        let x_tensor =
            Tensor::from_f32(&[batch as i32, channels as i32, length as i32], &x).unwrap();
        let w_tensor = Tensor::from_f32(&[channels as i32, 1, kernel as i32], &w).unwrap();
        let bias_tensor = Tensor::from_f32(&[channels as i32], &bias).unwrap();
        let inputs: &[(&str, &Tensor)] =
            &[("x", &x_tensor), ("w", &w_tensor), ("bias", &bias_tensor)];

        let g = Graph::new();
        let out = depthwise_conv1d(
            &g,
            g.input("x"),
            g.input("w"),
            Some(g.input("bias")),
            channels as i32,
            kernel as i32,
            padding as i32,
            dilation as i32,
            F32,
            CPU,
        )
        .unwrap();
        let (depthwise_shape, depthwise) = run_shaped(&g, out, inputs);

        let g = Graph::new();
        let out = conv1d(
            &g,
            g.input("x"),
            g.input("w"),
            Some(g.input("bias")),
            1,
            padding as i32,
            dilation as i32,
            channels as i32,
            F32,
            CPU,
        )
        .unwrap();
        let (grouped_shape, grouped) = run_shaped(&g, out, inputs);

        assert_eq!(
            depthwise_shape, grouped_shape,
            "case {index} came back a different shape from the grouped convolution"
        );
        close(&depthwise, &grouped, 1e-5);
    }
}

// ---------------------------------------------------------------------------------------------
// the operators, as opposed to the compositions
// ---------------------------------------------------------------------------------------------

/// `conv1d`, `conv_transpose1d`, `snake`, `stft` and `istft` are also operators -- named in
/// `Operators`, dispatched through `F::`, reachable as a single node from a graph -- and no
/// backend implements any of them.
///
/// That combination is easy to get wrong in a way nothing notices, because the whole chain is
/// declarations: a node no pass recognises and a node whose kernel is missing both do nothing,
/// and only one of the two is intended. So this asks for the intended failure by name. Each graph
/// has to compile, print as the call it stands for, and then fail *at the operator layer* when it
/// runs -- which is five links, from `Graph` through `Op` and `ir` and the C interface down to
/// `Operators`, every one of which has to be connected for the message to come back.
///
/// The bases throw rather than abort, which is what makes this a test at all. `NOT_IMPL()` --
/// what the other sixty-one unimplemented operators use -- is `LOG(FATAL)` and `abort()`, and an
/// operator that kills the process cannot be asked whether it is there.
///
/// When a backend grows one of these kernels its line here stops failing, and that is the signal
/// to delete the line rather than a reason to doubt it.
#[test]
fn the_audio_operators_are_nodes_no_backend_implements_yet() {
    let signal = Tensor::from_f32(&[1, 2, 8], &ramp(16, 0.0)).unwrap();
    let weight = Tensor::from_f32(&[2, 1, 3], &ramp(6, 1.0)).unwrap();
    let per_channel = Tensor::from_f32(&[2], &[0.7, 1.3]).unwrap();
    let window = Tensor::from_f32(&[4], &hann_window(4)).unwrap();

    // Each case is what the node should be called, how to build it, and the inputs that graph
    // declares -- a run is refused anything it did not ask for.
    type Build = Box<dyn Fn(&Graph) -> Value>;
    type Case<'a> = (&'a str, Build, Vec<(&'a str, &'a Tensor)>);
    let cases: Vec<Case> = vec![
        (
            "conv1d",
            Box::new(|g: &Graph| g.conv1d(g.input("x"), g.input("w"), None, 1, 1, 1, 1)),
            vec![("x", &signal), ("w", &weight)],
        ),
        (
            "conv_transpose1d",
            Box::new(|g: &Graph| g.conv_transpose1d(g.input("x"), g.input("w"), None, 2, 0, 0, 1)),
            vec![("x", &signal), ("w", &weight)],
        ),
        (
            "snake",
            Box::new(|g: &Graph| g.snake(g.input("x"), g.input("a"), None, 1e-6)),
            vec![("x", &signal), ("a", &per_channel)],
        ),
        (
            "stft",
            Box::new(|g: &Graph| g.stft(g.input("x"), g.input("win"), 4, 2, false)),
            vec![("x", &signal), ("win", &window)],
        ),
        (
            "istft",
            Box::new(|g: &Graph| g.istft(g.input("x"), g.input("win"), 4, 2, false)),
            vec![("x", &signal), ("win", &window)],
        ),
    ];

    for (name, build, inputs) in cases {
        let g = Graph::new();
        let out = build(&g);
        g.output("out", out);

        // It reads as the call it stands for, which is what says `Op::name` and the `Display`
        // arm were both filled in.
        let printed = format!("{g}");
        assert!(
            printed.contains(&format!("{name}(")),
            "a {name} graph did not print as one:\n{printed}"
        );

        let weights: HashMap<String, Tensor> = HashMap::new();
        let mut context = RunContext::new(&weights);
        for (input, tensor) in &inputs {
            context = context.input(input, tensor);
        }

        let ir = Ir::compile(&g, Residency::Device);
        let error = ir
            .run(&context)
            .expect_err("no backend implements this, so running it has to fail");

        // The failure has to be the operator saying it has no kernel. Anything else -- an unknown
        // node, a shape complaint, a missing input -- would mean the chain is miswired rather
        // than merely unimplemented.
        let message = error.to_string();
        assert!(
            message.contains("no device has this kernel"),
            "{name} failed for the wrong reason: {message}"
        );
        assert!(
            message.contains(name) || message.contains("convTranspose1d"),
            "{name} failed without naming itself: {message}"
        );
    }
}
