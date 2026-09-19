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

//! What a speech model asks for that a diffusion model never did: convolution along one axis, a
//! short time Fourier transform, a mel filterbank, and the activation a vocoder is built from.
//!
//! # None of this is a new kernel
//!
//! The obvious way to read that list is as six operators to write three times over, once for the
//! CPU, once for CUDA and once for Metal. It is not what happened here, because none of the six
//! is irreducible:
//!
//! - a 1-D convolution is a 2-D one whose image is one pixel tall, and [`conv1d`] is that view
//!   plus the padding `conv2d` cannot express, since its padding is square and this one's is not;
//! - a transposed convolution is a matrix multiply per input position and then an overlap-add of
//!   the columns that fall on each other, which [`conv_transpose1d`] writes as a fixed number of
//!   shifted additions -- `ceil(kernel / stride)` of them, which is two for every vocoder here;
//! - a short time Fourier transform is a convolution by a bank of windowed sinusoids, so
//!   [`stft`] is [`conv1d`] against a constant this module computes on the host, and [`istft`] is
//!   [`conv_transpose1d`] against the matching one, divided by the window's own overlap;
//! - a mel filterbank is a matrix multiply;
//! - resampling is a convolution by a windowed sinc, between a zero-stuffing and a decimation
//!   that are themselves a transposed convolution and a stride;
//! - and a snake is arithmetic.
//!
//! So this module is composition, not kernels, and it buys three things by being so. It runs
//! wherever `conv2d` and `matmul` already run, which is every backend, on the day it is written
//! rather than three ports later. It is correct wherever they are correct, and they are tested.
//! And a fused kernel can replace any one of these later without a caller noticing, because what
//! a caller sees is the function, not the seven nodes behind it.
//!
//! What it costs is real and worth saying plainly: the sinusoid bank makes an STFT an O(N^2)
//! matrix multiply per frame where a radix-2 transform would be O(N log N). At the 1024-point
//! window these models use that is about a hundredfold more arithmetic -- but it is arithmetic
//! shaped like a GEMM, which is the one shape this library is fastest at, and a ten second
//! utterance is a couple of GFLOP. It is the vocoder that costs, not the transform in front of
//! it. If that ever stops being true, [`stft`] is the function to replace.
//!
//! # Shapes
//!
//! Everything here is `(N, C, L)` -- batch, channel, and the axis that is time. A waveform is
//! `C = 1`; a spectrogram is one channel per frequency and one position per frame. That is what
//! the reference implementations use, and keeping it means a weight can be read from a package in
//! the layout its authors stored it in.

use crate::flint::{DType, Device, Extent, Graph, Tensor, Value};
use crate::Result;

/// How [`pad1d`] fills what it adds.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Padding {
    /// Zeros, which is what a convolution means by padding.
    Zero,
    /// The signal mirrored about its own first and last samples, which is what `torch.stft`
    /// means by `center=True` and therefore what every mel spectrogram in a speech model was
    /// computed with. Mirroring rather than zeroing keeps the edge frames from seeing a step
    /// that is not in the audio.
    ///
    /// A reversal is a matrix multiply by the anti-diagonal here -- see [`reversal`] -- because
    /// there is no operator that reads a dimension backwards and this needs no new one.
    Reflect,
}

/// `left` and `right` more positions on the time axis of `x` `(N, C, L)`.
///
/// `dtype` and `device` are the ones `x` will have when the graph runs; a graph holds no tensors,
/// so it cannot work them out for itself, and the model that writes this call is the thing that
/// knows both.
#[track_caller]
pub fn pad1d(
    g: &Graph,
    x: Value,
    left: i32,
    right: i32,
    mode: Padding,
    dtype: DType,
    device: Device,
) -> Result<Value> {
    if left == 0 && right == 0 {
        return Ok(x);
    }

    assert!(left >= 0 && right >= 0, "a padding cannot be negative");

    Ok(match mode {
        Padding::Zero => {
            let mut out = x;
            if left > 0 {
                let zeros = g.zeros(
                    [Extent::of(x, 0), Extent::of(x, 1), Extent::At(left)],
                    dtype,
                    device,
                );
                out = g.cat(zeros, out, 2);
            }
            if right > 0 {
                let zeros = g.zeros(
                    [Extent::of(x, 0), Extent::of(x, 1), Extent::At(right)],
                    dtype,
                    device,
                );
                out = g.cat(out, zeros, 2);
            }
            out
        }

        // `x[1 ..= left]` reversed on the front and `x[L - 1 - right .. L - 1]` reversed on the
        // back: the mirror is about the first and last samples, which are therefore not repeated.
        // Both slices are written with constant bounds -- negative ones count from the back --
        // so neither needs the graph to know how long `x` is.
        Padding::Reflect => {
            let mut out = x;
            if left > 0 {
                let edge = g.slice(x, 2, 1, left + 1);
                let flip = g.constant(reversal(left)?.to_device(device)?.cast(dtype)?);
                out = g.cat(g.matmul(edge, flip), out, 2);
            }
            if right > 0 {
                let edge = g.slice(x, 2, -(right + 1), -1);
                let flip = g.constant(reversal(right)?.to_device(device)?.cast(dtype)?);
                out = g.cat(out, g.matmul(edge, flip), 2);
            }
            out
        }
    })
}

/// The `n` by `n` anti-diagonal, which reverses the last axis of whatever is multiplied by it.
///
/// `x @ reversal(n)` is `x` backwards. It exists because reading a dimension in reverse is not
/// an operator here and does not need to be: a reversal is a permutation, a permutation is a
/// matrix, and a matrix multiply is the operator this library is built on.
pub fn reversal(n: i32) -> Result<Tensor> {
    let n = n as usize;
    let mut values = vec![0.0f32; n * n];
    for row in 0..n {
        values[row * n + (n - 1 - row)] = 1.0;
    }

    Ok(Tensor::from_f32(&[n as i32, n as i32], &values)?)
}

/// A convolution along the time axis of `x` `(N, C, L)` by `weight` `(K, C / groups, R)`, with an
/// optional per-channel `bias` `(K)`.
///
/// What comes back is `(N, K, Lout)`, `Lout = (L + 2 * padding - dilation * (R - 1) - 1) / stride
/// + 1`, which is what every reference implementation of this means.
///
/// # Why this is `conv2d`
///
/// A 1-D convolution is the 2-D one over an image one row tall, and the only thing that does not
/// survive the reshape is the padding: `conv2d` pads both axes by the same amount, and padding
/// the axis of length one would make it `2 * padding + 1` long and the output the wrong shape. So
/// the padding is done here, explicitly, and `conv2d` is asked for none. Stride and dilation need
/// no such care -- a stride over one row leaves one row, and a dilation of a kernel one tall
/// dilates nothing.
///
/// # A general group count runs on the processor only
///
/// Not a limit of this function: `flint`'s CUDA `conv2d` is CUTLASS's, `conv2d_cutlass.cu` throws
/// on any group count above one, and `cuda/conv2d.cc` calls it unconditionally. The cuDNN path
/// beside it does set a group count, but it is built into the benchmark rather than into the
/// runtime, so reaching it is a change to `conv2d.cc` and not a build flag.
///
/// In practice this has turned out not to block anything. Nothing in a speech model here wants a
/// *general* grouped convolution -- what the conformers want is the depthwise case, where the
/// group count is the channel count, and that one is a composition rather than a kernel:
/// [`depthwise_conv1d`] computes it out of a slice and a multiply and runs on a card today.
///
/// So use [`depthwise_conv1d`] when the groups are the channels, which is every grouped
/// convolution in IndexTTS-2.5. Reach for a group count here only for something in between --
/// more than one channel per group -- and expect it on the processor only.
#[track_caller]
#[allow(clippy::too_many_arguments)]
pub fn conv1d(
    g: &Graph,
    x: Value,
    weight: Value,
    bias: Option<Value>,
    stride: i32,
    padding: i32,
    dilation: i32,
    groups: i32,
    dtype: DType,
    device: Device,
) -> Result<Value> {
    let x = pad1d(g, x, padding, padding, Padding::Zero, dtype, device)?;

    let x = g.unsqueeze(x, 2);
    let weight = g.unsqueeze(weight, 2);
    let y = g.conv2d(x, weight, bias, stride, 0, dilation, groups);

    Ok(g.squeeze(y, 2))
}

/// A depthwise convolution along the time axis of `x` `(N, C, L)`: one kernel per channel and no
/// mixing between them. `weight` is `(C, 1, R)`, the shape torch stores a
/// `Conv1d(C, C, R, groups=C)` in, and `bias` is `(C)`.
///
/// What comes back is `(N, C, L + 2 * padding - dilation * (R - 1))`. Stride one only, which is
/// what both callers that need this use; see below.
///
/// # Why this exists beside [`conv1d`]
///
/// [`conv1d`] passes a group count straight through to `conv2d`, and `flint`'s CUDA `conv2d`
/// refuses any group count above one. That would leave a conformer -- and so IndexTTS-2.5's
/// emotion conditioning encoder and the w2v-bert-2.0 in front of it -- unable to run on a card.
///
/// It does not, because those convolutions are not merely grouped, they are *depthwise*: the
/// group count equals both the input and the output channel count, so every group is one channel
/// wide and there is no channel multiplier. A general grouped convolution is not a composition of
/// the operators here. A depthwise one is:
///
/// ```text
/// out[n, c, t] = sum over k of  weight[c, k] * x[n, c, t + k * dilation - padding]
/// ```
///
/// Read that as a sum over `k` rather than over channels and it is `R` shifted copies of `x`,
/// each scaled by one number per channel and added up -- the same shape of trick
/// [`conv_transpose1d`] uses. A shift is a slice, a per-channel scale is a broadcast multiply,
/// and both of those run everywhere.
///
/// # What it costs
///
/// The arithmetic is exactly the convolution's, `R * C * L` multiply-adds either way. What is
/// worse is the traffic: `R` elementwise passes over the whole tensor rather than one fused
/// kernel that reads it once, so roughly `R`-fold the memory bandwidth, and `3 * R` nodes in the
/// graph rather than one. For a conformer that is a kernel of 31 over a tensor the attention and
/// the feed-forward beside it dominate anyway.
///
/// So this is the honest version of "it runs today", not a claim that it runs as fast as a real
/// depthwise kernel would. If a conformer ever turns out to be bandwidth-bound here, this is the
/// function to replace, and replacing it changes nothing above it.
///
/// # Stride
///
/// One. A stride would need every `s`-th position of a slice, and a strided view is not something
/// the graph can express; the depthwise convolution in a conformer and the one in w2v-bert-2.0
/// are both stride one, so nothing here needs it.
#[track_caller]
#[allow(clippy::too_many_arguments)]
pub fn depthwise_conv1d(
    g: &Graph,
    x: Value,
    weight: Value,
    bias: Option<Value>,
    channels: i32,
    kernel: i32,
    padding: i32,
    dilation: i32,
    dtype: DType,
    device: Device,
) -> Result<Value> {
    assert!(kernel >= 1, "a kernel must be positive");
    assert!(dilation >= 1, "a dilation must be positive");

    let x = pad1d(g, x, padding, padding, Padding::Zero, dtype, device)?;

    // Everything below works in `(N, L, C)`, because a binary operation broadcasts its right
    // operand over the *leading* dimensions of its left: a per-channel scale is a `(C)` vector,
    // so the channel has to be the last axis for it to line up.
    let x = g.contiguous(g.transpose(x, 1, 2));

    // `(C, 1, R)` -> `(R, C)`, so that tap `k` is one `subtensor` away.
    let taps = g.contiguous(g.transpose(
        g.view(weight, [Extent::At(channels), Extent::At(kernel)]),
        0,
        1,
    ));

    let mut total: Option<Value> = None;
    for k in 0..kernel {
        // Tap `k` reads from `k * dilation` and stops the same distance from the far end that the
        // last tap does, which is what makes every slice the same length. Both bounds are
        // constants -- a negative one counts from the back -- so the graph never needs to know
        // how long the signal is.
        let trim = dilation * (kernel - 1 - k);
        let shifted = g.slice(
            x,
            1,
            k * dilation,
            match trim {
                0 => Extent::from(crate::flint::Bound::End),
                _ => Extent::At(-trim),
            },
        );

        let scaled = g.mul(shifted, g.subtensor(taps, k));
        total = Some(match total {
            None => scaled,
            Some(sofar) => g.add(sofar, scaled),
        });
    }

    let out = total.expect("a kernel of at least one tap has at least one term");
    let out = match bias {
        // The channel is already last here, so a bias costs nothing but the add.
        Some(bias) => g.add(out, bias),
        None => out,
    };

    Ok(g.contiguous(g.transpose(out, 1, 2)))
}

/// A transposed convolution along the time axis of `x` `(N, C, L)` by `weight` `(C, K, R)`, with
/// an optional per-channel `bias` `(K)`. This is what a vocoder upsamples with.
///
/// What comes back is `(N, K, (L - 1) * stride - 2 * padding + R)`, which is what
/// `torch.nn.ConvTranspose1d` produces for the same arguments with no output padding.
///
/// # Why this is a matrix multiply and some additions
///
/// Read the operation the way it is defined rather than as a convolution: input position `l`
/// contributes `weight[:, :, k]` to output position `l * stride + k`, for every `k`. The
/// contribution is a matrix multiply -- `(N, L, C) @ (C, K * R)` gives every one of them at once
/// -- and what is left is adding up the ones that landed on the same output position.
///
/// Which ones those are is fixed: positions `l * stride + k` and `l' * stride + k'` collide only
/// when `k` and `k'` are `stride` apart, so cutting the kernel into `ceil(R / stride)` pieces of
/// `stride` each makes every piece collision-free within itself and the whole thing a sum of
/// `ceil(R / stride)` shifted copies. For the vocoders here the kernel is twice the stride, so
/// that is two additions.
///
/// # Groups
///
/// One group only, which is why there is no argument for it. The upsampling in a vocoder and the
/// inverse transform in [`istft`] are the two callers this was written for and neither groups, so
/// rather than write a general case with nothing to test it against there is no general case. A
/// grouped transposed convolution is the same derivation with a block-diagonal weight.
#[track_caller]
#[allow(clippy::too_many_arguments)]
pub fn conv_transpose1d(
    g: &Graph,
    x: Value,
    weight: Value,
    bias: Option<Value>,
    in_channels: i32,
    out_channels: i32,
    kernel: i32,
    stride: i32,
    padding: i32,
    dtype: DType,
    device: Device,
) -> Result<Value> {
    assert!(stride >= 1, "a stride must be positive");
    assert!(kernel >= 1, "a kernel must be positive");

    // The kernel is rounded up to a whole number of strides so that it cuts evenly into pieces.
    // The zeros this pads it with contribute nothing, and what they add to the far end of the
    // output is trimmed off below.
    let pieces = (kernel + stride - 1) / stride;
    let padded = pieces * stride;

    let weight = pad1d(g, weight, 0, padded - kernel, Padding::Zero, dtype, device)?;

    // (N, L, C) @ (C, K * R) -> (N, L, K * R): every contribution of every input position, before
    // anything has been said about where they land.
    let flat = g.view(
        weight,
        [Extent::At(in_channels), Extent::At(out_channels * padded)],
    );
    let columns = g.matmul(g.contiguous(g.transpose(x, 1, 2)), flat);

    // Cut the kernel axis into `pieces` of `stride`. Within a piece no two contributions collide,
    // so a piece flattens straight onto the time axis; between pieces the collision is a shift of
    // a whole stride, which is the padding below.
    let split = g.view(
        columns,
        [
            Extent::of(x, 0),
            Extent::of(x, 2),
            Extent::At(out_channels),
            Extent::At(pieces),
            Extent::At(stride),
        ],
    );

    let mut total: Option<Value> = None;
    for piece in 0..pieces {
        // (N, L, K, 1, stride) -> (N, K, L, 1, stride) -> (N, K, L * stride).
        let part = g.contiguous(g.transpose(g.slice(split, 3, piece, piece + 1), 1, 2));
        let part = g.view(
            part,
            [
                Extent::of(x, 0),
                Extent::At(out_channels),
                Extent::prod(part, 2, 5),
            ],
        );

        let part = pad1d(
            g,
            part,
            piece * stride,
            (pieces - 1 - piece) * stride,
            Padding::Zero,
            dtype,
            device,
        )?;

        total = Some(match total {
            None => part,
            Some(sofar) => g.add(sofar, part),
        });
    }

    let full = total.expect("a kernel of at least one stride cuts into at least one piece");

    // Every piece is `(L + pieces - 1) * stride` long, which is what a transposed convolution by
    // the *padded* kernel produces. The real one is shorter by what the padding added, and
    // `padding` itself takes that many more off each end.
    let trim = padded - kernel + padding;
    let full = match (padding, trim) {
        (0, 0) => full,
        _ => g.slice(
            full,
            2,
            padding,
            match trim {
                0 => Extent::from(crate::flint::Bound::End),
                _ => Extent::At(-trim),
            },
        ),
    };

    // Contiguous either way: `full` is a slice, which is a view, and a view cannot be moved off
    // the device or reshaped by whatever reads this next.
    Ok(g.contiguous(match bias {
        None => full,
        // A bias is per output channel, and a binary operation broadcasts its right operand over
        // the *leading* dimensions of its left. So the channel has to be last for the add, which
        // is two transposes and no copy.
        Some(bias) => g.transpose(g.add(g.transpose(full, 1, 2), bias), 1, 2),
    }))
}

/// The activation a BigVGAN is built from: `x + sin(alpha * x)^2 / alpha`.
///
/// `alpha` and `beta` are `(C)`, one per channel of `x` `(N, C, L)`. With `beta` given this is
/// the "snakebeta" variant, `x + sin(alpha * x)^2 / beta`, which is what the vocoders in these
/// models use; the two are the same function when `beta == alpha`.
///
/// Both are taken already exponentiated. A BigVGAN trained with `alpha_logscale` stores the
/// logarithm and raises it where it is read, and that belongs to whoever reads the weight, not
/// here -- this is the arithmetic, not the storage.
///
/// The channel is moved to the last axis and back because a binary operation broadcasts its right
/// operand over the leading dimensions of its left, so a per-channel vector has to sit against
/// the axis it names. Neither transpose moves any data.
///
/// # What it costs
///
/// Of everything in this module this is the one most likely to want a kernel of its own. It is
/// elementwise, so it is bound by memory rather than arithmetic, and written this way it is about
/// six passes over the whole tensor plus a copy to make the result contiguous, where a fused
/// kernel would read it once and write it once.
///
/// That matters because of where it runs. A BigVGAN applies it after every convolution in every
/// AMP block -- eighteen times per upsampling stage, on the tensors that have already been
/// upsampled -- so it is the vocoder's bandwidth, not its arithmetic, that this spends. The
/// convolutions beside it go through `conv2d` and a tuned GEMM; this does not.
#[track_caller]
pub fn snake(
    g: &Graph,
    x: Value,
    alpha: Value,
    beta: Option<Value>,
    eps: f32,
    dtype: DType,
    device: Device,
) -> Result<Value> {
    let t = g.transpose(x, 1, 2);

    let scaled = g.mul(t, alpha);
    let squared = g.square(g.sin(scaled));

    // The epsilon is added to the divisor rather than guarding the division, which is what the
    // reference does and is there for the channel whose parameter has decayed to zero.
    let divisor = g.add(beta.unwrap_or(alpha), scalar(g, eps, dtype, device)?);
    let out = g.add(t, g.div(squared, divisor));

    // Contiguous, not just transposed back: what comes out of here is a tensor a caller may do
    // anything with, including move it off the device, and a view cannot be moved.
    Ok(g.contiguous(g.transpose(out, 1, 2)))
}

/// `eps` as something the graph can add to a tensor: a constant of one element, which a binary
/// operation broadcasts over every dimension of its left operand.
#[track_caller]
fn scalar(g: &Graph, value: f32, dtype: DType, device: Device) -> Result<Value> {
    Ok(g.constant(
        Tensor::from_f32(&[1], &[value])?
            .to_device(device)?
            .cast(dtype)?,
    ))
}

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

/// The bank of windowed sinusoids that [`stft`] convolves by: `(2 * (n_fft / 2 + 1), 1, n_fft)`.
///
/// The first `n_fft / 2 + 1` channels are the real part of the transform and the rest are the
/// imaginary part, in that order, because a tensor here holds real numbers and a spectrum is the
/// two halves of one laid end to end. Only half the spectrum is kept: the input is real, so the
/// other half is its conjugate and says nothing new.
pub fn stft_basis(n_fft: usize, window: &[f32]) -> Result<Tensor> {
    assert_eq!(window.len(), n_fft, "a window is as long as the transform");

    let bins = n_fft / 2 + 1;
    let mut values = vec![0.0f32; 2 * bins * n_fft];

    for bin in 0..bins {
        for n in 0..n_fft {
            let phase = 2.0 * std::f64::consts::PI * bin as f64 * n as f64 / n_fft as f64;
            let w = window[n] as f64;
            values[bin * n_fft + n] = (w * phase.cos()) as f32;
            values[(bins + bin) * n_fft + n] = (-w * phase.sin()) as f32;
        }
    }

    Ok(Tensor::from_f32(
        &[2 * bins as i32, 1, n_fft as i32],
        &values,
    )?)
}

/// The bank [`istft`] transposes by, the inverse of [`stft_basis`]: `(2 * (n_fft / 2 + 1), 1,
/// n_fft)`.
///
/// Two things distinguish it from the forward bank beyond the sign. It carries the `1 / n_fft` an
/// inverse transform divides by, and it doubles every bin except the two -- the constant one and
/// the Nyquist one -- that have no conjugate partner, which is what recovers the half of the
/// spectrum the forward transform did not keep. It is windowed again, which is what an overlap-add
/// resynthesis does; [`window_envelope`] is what divides that second window back out.
pub fn istft_basis(n_fft: usize, window: &[f32]) -> Result<Tensor> {
    assert_eq!(window.len(), n_fft, "a window is as long as the transform");

    let bins = n_fft / 2 + 1;
    let mut values = vec![0.0f32; 2 * bins * n_fft];

    for bin in 0..bins {
        // The bins that are their own conjugate appear once in a full spectrum; every other bin
        // appears twice, and this bank sees only one of the two.
        let fold = if bin == 0 || (n_fft.is_multiple_of(2) && bin == n_fft / 2) {
            1.0
        } else {
            2.0
        };
        let scale = fold / n_fft as f64;

        for n in 0..n_fft {
            let phase = 2.0 * std::f64::consts::PI * bin as f64 * n as f64 / n_fft as f64;
            let w = window[n] as f64;
            values[bin * n_fft + n] = (scale * phase.cos() * w) as f32;
            values[(bins + bin) * n_fft + n] = (-scale * phase.sin() * w) as f32;
        }
    }

    Ok(Tensor::from_f32(
        &[2 * bins as i32, 1, n_fft as i32],
        &values,
    )?)
}

/// What the analysis and synthesis windows together multiply each sample by, once every frame
/// that covers it has been added up: `sum_t window[n - t * hop]^2`, `(1, 1, length)`.
///
/// Dividing by this is what makes an [`istft`] of an [`stft`] the signal it started as. It is a
/// constant of the window and the hop rather than of the audio, but its length is the audio's, so
/// it is built for a known number of frames.
pub fn window_envelope(window: &[f32], hop: usize, frames: usize, eps: f32) -> Result<Tensor> {
    let n_fft = window.len();
    let length = (frames - 1) * hop + n_fft;
    let mut values = vec![0.0f32; length];

    for frame in 0..frames {
        for n in 0..n_fft {
            values[frame * hop + n] += window[n] * window[n];
        }
    }

    // A position no frame covered would divide by zero. With a Hann window and a hop of a quarter
    // of it that is only the very ends, which is exactly where `center` padding is trimmed off.
    for value in &mut values {
        if *value < eps {
            *value = eps;
        }
    }

    Ok(Tensor::from_f32(&[1, 1, length as i32], &values)?)
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

/// The short time Fourier transform of `x` `(N, 1, L)`, as a real and an imaginary part, each
/// `(N, n_fft / 2 + 1, frames)`.
///
/// `basis` is [`stft_basis`] as a constant, which is where the window went. `centered` pads the
/// signal by half a window at both ends so that frame `t` is centred on sample `t * hop` rather
/// than starting there, which is what `torch.stft` does by default and what every mel
/// spectrogram in these models was computed with; [`Padding::Reflect`] is likewise its default
/// and the reason this takes a mode at all.
#[track_caller]
#[allow(clippy::too_many_arguments)]
pub fn stft(
    g: &Graph,
    x: Value,
    basis: Value,
    n_fft: i32,
    hop: i32,
    centered: Option<Padding>,
    dtype: DType,
    device: Device,
) -> Result<(Value, Value)> {
    let x = match centered {
        None => x,
        Some(mode) => pad1d(g, x, n_fft / 2, n_fft / 2, mode, dtype, device)?,
    };

    // One channel in, two per bin out, and a stride that is the hop: a frame is what the kernel
    // covers and the hop is how far it moves, which is the definition of the transform.
    let spectrum = conv1d(g, x, basis, None, hop, 0, 1, 1, dtype, device)?;

    let bins = n_fft / 2 + 1;
    let real = g.slice(spectrum, 1, 0, bins);
    let imaginary = g.slice(spectrum, 1, bins, 2 * bins);

    Ok((real, imaginary))
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

/// The inverse of [`stft`]: `(N, 1, (frames - 1) * hop + n_fft)`, or that trimmed by half a
/// window at each end when the transform was `centered`.
///
/// `basis` is [`istft_basis`] and `envelope` is [`window_envelope`], both as constants. The
/// transposed convolution is the overlap-add; the division by the envelope is what undoes the
/// two windows the signal has been multiplied by on the way through.
#[track_caller]
#[allow(clippy::too_many_arguments)]
pub fn istft(
    g: &Graph,
    real: Value,
    imaginary: Value,
    basis: Value,
    envelope: Value,
    n_fft: i32,
    hop: i32,
    centered: bool,
    dtype: DType,
    device: Device,
) -> Result<Value> {
    let bins = n_fft / 2 + 1;
    let spectrum = g.cat(real, imaginary, 1);

    let wave = conv_transpose1d(
        g,
        spectrum,
        basis,
        None,
        2 * bins,
        1,
        n_fft,
        hop,
        0,
        dtype,
        device,
    )?;
    let wave = g.div(wave, envelope);

    Ok(match centered {
        false => wave,
        true => g.slice(wave, 2, n_fft / 2, -(n_fft / 2)),
    })
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
            conv_transpose1d(g, x, one, None, 1, 1, 1, up, 0, dtype, device)?
        }
    };

    conv1d(
        g,
        stuffed,
        kernel,
        None,
        down,
        taps / 2,
        1,
        1,
        dtype,
        device,
    )
}

fn gcd(a: u32, b: u32) -> u32 {
    match b {
        0 => a,
        _ => gcd(b, a % b),
    }
}
