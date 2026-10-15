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

//! What a speech model asks for that a diffusion model never did, beyond the operators: a window,
//! a mel filterbank, the filters a resampling convolves by, and the alias-free resampling around a
//! BigVGAN's activation.
//!
//! # The operators are flint's
//!
//! A convolution along one axis and its transpose, a padding by zeros, a mirror or a repeated
//! edge, a short time Fourier transform and its inverse, a snake activation and a clamp are each a
//! single [`Graph`] node -- [`Graph::conv1d`], [`Graph::conv_transpose1d`], [`Graph::pad1d`],
//! [`Graph::stft`], [`Graph::istft`], [`Graph::snake`], [`Graph::clamp`] -- which the CPU, CUDA
//! and Vulkan run as kernels of their own and Metal as a composition of the operators it has.
//! What is left here is what a model *designs* rather than computes: the numbers a filter is made
//! of, worked out on the host once, and the few lines that put two operators together the way one
//! reference implementation did.
//!
//! # Shapes
//!
//! Everything here is `(N, C, L)` -- batch, channel, and the axis that is time. A waveform is
//! `C = 1`; a spectrogram is one channel per frequency and one position per frame. That is what
//! the reference implementations use, and keeping it means a weight can be read from a package in
//! the layout its authors stored it in.

use crate::flint::{Bound, DType, Device, Extent, Graph, PadMode, Tensor, Value};
use crate::Result;

/// A periodic Hann window of `n` points, which is what every speech model here analyses with.
///
/// Periodic rather than symmetric -- `n` in the denominator rather than `n - 1` -- because that
/// is the one whose squares overlap to a constant, and therefore the one an inverse transform can
/// undo. It is also what `torch.hann_window` gives by default.
pub fn hann_window(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let phase = 2.0 * std::f64::consts::PI * i as f64 / n as f64;
            (0.5 - 0.5 * phase.cos()) as f32
        })
        .collect()
}

/// Which mel scale a filterbank is laid out on.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MelScale {
    /// `2595 * log10(1 + hz / 700)`, the one HTK uses and the one a model that says `htk=True`
    /// was trained with.
    Htk,
    /// Linear below 1 kHz and logarithmic above it, which is what Slaney's auditory toolbox does
    /// and what `librosa.filters.mel` gives by default. Every speech model here uses this one.
    Slaney,
}

impl MelScale {
    fn to_mel(self, hz: f64) -> f64 {
        match self {
            MelScale::Htk => 2595.0 * (1.0 + hz / 700.0).log10(),
            MelScale::Slaney => {
                let linear = 200.0 / 3.0;
                if hz < 1000.0 {
                    hz / linear
                } else {
                    15.0 + (hz / 1000.0).ln() / ((6.4f64).ln() / 27.0)
                }
            }
        }
    }

    fn to_hz(self, mel: f64) -> f64 {
        match self {
            MelScale::Htk => 700.0 * (10f64.powf(mel / 2595.0) - 1.0),
            MelScale::Slaney => {
                let linear = 200.0 / 3.0;
                if mel < 15.0 {
                    mel * linear
                } else {
                    1000.0 * ((mel - 15.0) * ((6.4f64).ln() / 27.0)).exp()
                }
            }
        }
    }
}

/// The triangular filterbank that turns a magnitude spectrum into a mel one: `(n_mels, n_fft / 2
/// + 1)`, to be multiplied by a `(.., n_fft / 2 + 1)` spectrum.
///
/// Area-normalized, the way `librosa.filters.mel` normalizes by default: each triangle is scaled
/// by `2 / (hz[i + 2] - hz[i])` so that a flat spectrum comes out flat rather than tilted by how
/// wide the filters got. A model exported from a pipeline that passed `norm=None` wants
/// `normalize` false.
pub fn mel_filterbank(
    sample_rate: u32,
    n_fft: usize,
    n_mels: usize,
    f_min: f64,
    f_max: f64,
    scale: MelScale,
    normalize: bool,
) -> Result<Tensor> {
    let bins = n_fft / 2 + 1;

    // The centre frequency of each FFT bin, and the n_mels + 2 points that give n_mels triangles:
    // every triangle rises from one point, peaks at the next and falls to the one after.
    let bin_hz: Vec<f64> = (0..bins)
        .map(|bin| bin as f64 * sample_rate as f64 / n_fft as f64)
        .collect();

    let (mel_min, mel_max) = (scale.to_mel(f_min), scale.to_mel(f_max));
    let points: Vec<f64> = (0..n_mels + 2)
        .map(|i| scale.to_hz(mel_min + (mel_max - mel_min) * i as f64 / (n_mels + 1) as f64))
        .collect();

    let mut values = vec![0.0f32; n_mels * bins];
    for mel in 0..n_mels {
        let (lower, centre, upper) = (points[mel], points[mel + 1], points[mel + 2]);

        for (bin, &hz) in bin_hz.iter().enumerate() {
            let rising = (hz - lower) / (centre - lower);
            let falling = (upper - hz) / (upper - centre);
            let weight = rising.min(falling).max(0.0);

            let weight = match normalize {
                true => weight * 2.0 / (upper - lower),
                false => weight,
            };

            values[mel * bins + bin] = weight as f32;
        }
    }

    Ok(Tensor::from_f32(&[n_mels as i32, bins as i32], &values)?)
}

/// `value` as something the graph can add to a tensor: a constant of one element, which a binary
/// operation broadcasts over every dimension of its left operand.
#[track_caller]
fn scalar(g: &Graph, value: f32, dtype: DType, device: Device) -> Result<Value> {
    Ok(g.constant(
        Tensor::from_f32(&[1], &[value])?
            .to_device(device)?
            .cast(dtype)?,
    ))
}

/// The real and the imaginary part of what [`Graph::stft`] gives at `n_fft`, each `(N, n_fft / 2
/// + 1, frames)`: the first half of its channels and the second.
#[track_caller]
pub fn halves(g: &Graph, spectrum: Value, n_fft: i32) -> (Value, Value) {
    let bins = n_fft / 2 + 1;
    (g.slice(spectrum, 1, 0, bins), g.slice(spectrum, 1, bins, 2 * bins))
}

/// `sqrt(real^2 + imaginary^2)`, the magnitude spectrum a mel filterbank is applied to.
///
/// The epsilon is inside the square root. A magnitude of exactly zero is a perfectly ordinary
/// thing for a spectrum to contain and its derivative there is infinite, which is a NaN in
/// anything that differentiates and a surprise in anything that does not.
#[track_caller]
pub fn magnitude(
    g: &Graph,
    real: Value,
    imaginary: Value,
    eps: f32,
    dtype: DType,
    device: Device,
) -> Result<Value> {
    let power = g.add(g.square(real), g.square(imaginary));
    Ok(g.sqrt(g.add(power, scalar(g, eps, dtype, device)?)))
}

/// A mel spectrogram: [`mel_filterbank`] applied to a `(N, bins, frames)` magnitude spectrum,
/// giving `(N, n_mels, frames)`.
///
/// The filterbank is `(n_mels, bins)` and the spectrum has its bins on the channel axis, so the
/// multiply is against the transpose of the spectrum and the result is transposed back. Both
/// transposes are views.
#[track_caller]
pub fn apply_filterbank(g: &Graph, magnitude: Value, filterbank: Value) -> Value {
    let framewise = g.contiguous(g.transpose(magnitude, 1, 2));
    let mel = g.matmul(framewise, g.transpose(filterbank, 0, 1));

    g.contiguous(g.transpose(mel, 1, 2))
}

/// The windowed sinc a resampling convolves by, as a `(1, 1, taps)` kernel and the stride to read
/// it with.
///
/// Resampling from `from` to `to` is three things: stuff `p - 1` zeros between every pair of
/// samples, low-pass at whichever of the two rates is lower, and keep one sample in `q`, where
/// `p / q` is the ratio in lowest terms. The low-pass is this, a sinc at the lower cutoff times a
/// Hann window `taps` wide, scaled by `p` to put back the amplitude the zeros took out.
pub fn resample_kernel(from: u32, to: u32, half_width: usize) -> Result<(Tensor, i32, i32)> {
    let divisor = gcd(from, to);
    let (up, down) = ((to / divisor) as usize, (from / divisor) as usize);

    // The cutoff is the lower of the two rates, expressed against the upsampled one.
    let cutoff = 1.0 / up.max(down) as f64;
    let taps = 2 * half_width * up.max(down) + 1;

    let mut values = vec![0.0f32; taps];
    for (i, value) in values.iter_mut().enumerate() {
        let t = i as f64 - (taps - 1) as f64 / 2.0;

        let sinc = match t == 0.0 {
            true => cutoff,
            false => (std::f64::consts::PI * cutoff * t).sin() / (std::f64::consts::PI * t),
        };

        let phase = 2.0 * std::f64::consts::PI * i as f64 / (taps - 1) as f64;
        let window = 0.5 - 0.5 * phase.cos();

        *value = (sinc * window * up as f64) as f32;
    }

    Ok((
        Tensor::from_f32(&[1, 1, taps as i32], &values)?,
        up as i32,
        down as i32,
    ))
}

/// Resample `x` `(N, 1, L)` from one rate to another, by [`resample_kernel`].
///
/// The zero stuffing is a transposed convolution by a kernel of one tap and a stride of `up`,
/// which is what puts a sample every `up` positions and nothing in between; the decimation is the
/// stride of the low-pass that follows.
#[track_caller]
#[allow(clippy::too_many_arguments)]
pub fn resample(
    g: &Graph,
    x: Value,
    kernel: Value,
    taps: i32,
    up: i32,
    down: i32,
    dtype: DType,
    device: Device,
) -> Result<Value> {
    let stuffed = match up {
        1 => x,
        _ => {
            let one = g.constant(
                Tensor::from_f32(&[1, 1, 1], &[1.0])?
                    .to_device(device)?
                    .cast(dtype)?,
            );
            g.conv_transpose1d(x, one, None, up, 0, 0, 1)
        }
    };

    Ok(g.conv1d(stuffed, kernel, None, down, taps / 2, 1, 1))
}

/// The windowed sinc the alias-free resampling in a BigVGAN's activation convolves by, as a
/// `(1, 1, kernel)` tensor.
///
/// `cutoff` is in cycles per sample, so half of the rate being resampled to; `half_width` is how
/// wide the transition band either side of it is, and it is what picks the Kaiser window's shape
/// parameter. A BigVGAN asks for `0.5 / ratio` and `0.6 / ratio` at twelve taps, and nothing else.
///
/// # This is not [`resample_kernel`]
///
/// The two do the same job and are not the same filter. [`resample_kernel`] converts between two
/// sample rates -- 24 kHz audio to 22.05 kHz -- and is a Hann-windowed sinc at whatever ratio
/// those two are in. This one is the fixed doubling and halving either side of an activation,
/// where the window is a Kaiser one whose beta is derived from the transition width, and it is
/// what `alias_free_activation/torch/filter.py` computes. Using either in the other's place
/// changes the answer, so both are here.
///
/// The taps are normalized to sum to one, which is what keeps a constant input a constant output.
pub fn kaiser_sinc_filter(cutoff: f64, half_width: f64, kernel: usize) -> Result<Tensor> {
    assert!(kernel >= 1, "a kernel must be positive");
    assert!(
        (0.0..=0.5).contains(&cutoff),
        "a cutoff is at most half the rate"
    );

    let half = kernel / 2;

    // The Kaiser design formula: the stopband attenuation a transition this wide buys, and the
    // window shape that reaches it. Both lines are Oppenheim and Schafer's, and are what
    // `kaiser_sinc_filter1d` uses.
    let attenuation = 2.285 * (half as f64 - 1.0) * std::f64::consts::PI * 4.0 * half_width + 7.95;
    let beta = match attenuation {
        a if a > 50.0 => 0.1102 * (a - 8.7),
        a if a >= 21.0 => 0.5842 * (a - 21.0).powf(0.4) + 0.07886 * (a - 21.0),
        _ => 0.0,
    };

    let mut values = vec![0.0f32; kernel];
    if cutoff > 0.0 {
        let mut taps = vec![0.0f64; kernel];
        for (i, tap) in taps.iter_mut().enumerate() {
            // An even kernel has no centre tap, so the positions sit half way between samples;
            // an odd one is centred on `half`. `torch.kaiser_window(periodic = false)` is the
            // window, which spans its whole length rather than wrapping around.
            let time = match kernel % 2 {
                0 => i as f64 - half as f64 + 0.5,
                _ => i as f64 - half as f64,
            };

            // The window's argument runs from -1 to 1 across the whole kernel. A kernel of one
            // has nowhere to run, and is the one tap in the middle.
            let window = match kernel {
                1 => 1.0,
                _ => {
                    let at = 2.0 * (i as f64) / (kernel as f64 - 1.0) - 1.0;

                    bessel_i0(beta * (1.0 - at * at).max(0.0).sqrt()) / bessel_i0(beta)
                }
            };

            *tap = 2.0 * cutoff * window * sinc(2.0 * cutoff * time);
        }

        let total: f64 = taps.iter().sum();
        for (value, tap) in values.iter_mut().zip(&taps) {
            *value = (tap / total) as f32;
        }
    }

    Ok(Tensor::from_f32(&[1, 1, kernel as i32], &values)?)
}

/// `sin(pi x) / (pi x)`, and one at zero.
fn sinc(x: f64) -> f64 {
    match x == 0.0 {
        true => 1.0,
        false => (std::f64::consts::PI * x).sin() / (std::f64::consts::PI * x),
    }
}

/// The modified Bessel function of the first kind, order zero, which is what shapes a Kaiser
/// window. Its series converges in a dozen terms at the beta a twelve-tap filter asks for.
fn bessel_i0(x: f64) -> f64 {
    let quarter_square = x * x / 4.0;

    let mut term = 1.0;
    let mut total = 1.0;
    for k in 1..64 {
        term *= quarter_square / (k as f64 * k as f64);
        total += term;

        if term < 1e-18 * total {
            break;
        }
    }

    total
}

/// `x` `(N, C, L)` resampled up by `ratio`, the way the activation in a BigVGAN is.
///
/// `filter` is `(1, 1, kernel)` from [`kaiser_sinc_filter`] at `0.5 / ratio` and `0.6 / ratio`,
/// and what comes back is `(N, C, L * ratio)` -- exactly `ratio` times as long, which is what the
/// padding and the trimming here are for.
///
/// # One filter, every channel
///
/// The reference convolves with `groups = C` and the same taps in every group, which is a
/// depthwise convolution whose kernel does not vary by channel. That is not a grouped convolution
/// at all: reading `(N, C, L)` as `(N * C, 1, L)` makes each channel its own item in a batch, and
/// one ungrouped single-channel convolution does all of them at once -- with one filter as its
/// whole weight, rather than `C` copies of it.
///
/// # Why it pads before it upsamples
///
/// A transposed convolution's first and last output positions see only part of the kernel, which
/// puts a taper on both ends of the signal that is not in the signal. Padding by
/// `kernel / ratio - 1` samples of the edge value and cutting the result back to length moves
/// that taper into the padding, where it is thrown away. This is what
/// `alias_free_activation/torch/resample.py` does, down to which side the odd sample goes on.
#[track_caller]
pub fn upsample1d(
    g: &Graph,
    x: Value,
    filter: Value,
    channels: i32,
    ratio: i32,
    kernel: i32,
) -> Value {
    assert!(ratio >= 1, "a ratio must be positive");
    assert!(
        kernel >= ratio,
        "a filter shorter than the ratio has nothing to interpolate with"
    );

    let pad = kernel / ratio - 1;
    let left = pad * ratio + (kernel - ratio) / 2;
    let right = pad * ratio + (kernel - ratio + 1) / 2;

    let padded = g.pad1d(x, pad, pad, PadMode::Replicate);
    let folded = fold_channels(g, padded);

    // The `ratio` is the amplitude the zeros a transposed convolution interleaves took out, the
    // same factor `resample_kernel` folds into its taps. Here the filter is shared with
    // `downsample1d`, which must not have it, so it is applied to the signal instead.
    let up = g.conv_transpose1d(folded, filter, None, ratio, 0, 0, 1);
    let up = g.mul_scalar(up, ratio as f32);

    let trimmed = g.slice(
        up,
        2,
        left,
        match right {
            0 => Extent::from(Bound::End),
            _ => Extent::At(-right),
        },
    );

    unfold_channels(g, trimmed, x, channels)
}

/// `x` `(N, C, L)` resampled down by `ratio`, which is the other half of what an alias-free
/// activation does. See [`upsample1d`], whose filter this shares.
///
/// What comes back is `(N, C, ceil(L / ratio))`, and for the even `L` an upsampling produced that
/// is `L / ratio`.
#[track_caller]
pub fn downsample1d(
    g: &Graph,
    x: Value,
    filter: Value,
    channels: i32,
    ratio: i32,
    kernel: i32,
) -> Value {
    assert!(ratio >= 1, "a ratio must be positive");

    // An even kernel is centred between two samples, so it takes one less on the left than on
    // the right; an odd one is centred on a sample and takes the same either side.
    let left = kernel / 2 - i32::from(kernel % 2 == 0);
    let padded = g.pad1d(x, left, kernel / 2, PadMode::Replicate);

    let folded = fold_channels(g, padded);
    let down = g.conv1d(folded, filter, None, ratio, 0, 1, 1);

    unfold_channels(g, down, x, channels)
}

/// `(N, C, L)` read as `(N * C, 1, L)`, so that a convolution by one channel's worth of taps
/// covers every channel at once.
fn fold_channels(g: &Graph, x: Value) -> Value {
    let x = g.contiguous(x);

    g.view(x, [Extent::prod(x, 0, 2), Extent::At(1), Extent::of(x, 2)])
}

/// The other way, taking the batch from `like` -- the tensor this started as, which is the only
/// thing that still knows how many items there were.
fn unfold_channels(g: &Graph, x: Value, like: Value, channels: i32) -> Value {
    let x = g.contiguous(x);

    g.view(
        x,
        [Extent::of(like, 0), Extent::At(channels), Extent::of(x, 2)],
    )
}

fn gcd(a: u32, b: u32) -> u32 {
    match b {
        0 => a,
        _ => gcd(b, a % b),
    }
}
