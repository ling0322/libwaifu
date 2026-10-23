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

#pragma once

#include <stdint.h>

#include <memory>

#include "lutil/span.h"
#include "flint/device.h"
#include "flint/dtype.h"
#include "flint/memory.h"
#include "flint/operators.h"
#include "flint/tensor.h"
#include "flint/tensor_view.h"

namespace fl {

/// @brief What the operators themselves need of allocation, and the one operation they share that
/// is made of others.
///
/// What every operation's result is -- its shape, its type, where it lives -- is decided by the
/// caller that allocates it: the binding, for everything that comes in through the C API, and
/// `flint/test_functional.h` for the C++ tests, which are written against results rather than
/// against output views. The operators in `flint/operators.h` compute and nothing else.
///
/// Its own namespace rather than `fl`'s, because the test helpers in the same namespace name
/// `exp`, `log`, `sqrt` and the rest, which would otherwise hide the C library's functions of the
/// same names from every unqualified call in flint.
namespace F {

// --- Allocation -----------------------------------------------------------------------------

/// @brief Uninitialized storage for `numel` elements of `dtype` on `device` -- Device::kCudaHost
/// being page-locked host memory the CUDA device reads directly. At least one element is
/// allocated, so that a tensor with a dimension of zero still has storage to point at. `hint` says
/// what it is for, which Metal and CUDA tell apart.
std::unique_ptr<TensorData> allocate(
    Device device,
    int64_t numel,
    DType dtype,
    AllocHint hint = AllocHint::kActivation);

/// @brief An uninitialized contiguous tensor of `shape` and `dtype` on `device`. Device::kCudaHost
/// is page-locked host memory the CUDA device reads directly.
Tensor empty(Device device, lut::Span<const int> shape, DType dtype);

/// @brief An uninitialized contiguous tensor of `input`'s shape, type and device.
Tensor emptyLike(const TensorView &input);

// --- Attention ------------------------------------------------------------------------------

/// @brief Attention as matmul, softmax and matmul, a block of queries at a time, written into
/// `out`. What Operators::attention does on a device with no kernel of its own.
void composedAttention(
    Operators *op,
    TensorView q,
    TensorView k,
    TensorView v,
    bool causal,
    TensorView out);

// --- Signal ---------------------------------------------------------------------------------
//
// What the operators of the same names do on a device with no kernel of their own, out of
// operators every device has. Each writes `out`, which is contiguous, and reads its input as
// Operators::convTranspose1d and the rest describe it.

/// A transposed 1-D convolution as, per group, one `matmul` giving every input position's
/// contribution and ceil(R / stride) shifted additions landing them: positions `l * stride + r`
/// and `l' * stride + r'` only collide when `r` and `r'` are a whole stride apart, so cutting the
/// kernel into stride-long pieces makes each piece collision-free within itself.
void composedConvTranspose1d(
    Operators *op,
    TensorView input,
    TensorView weight,
    TensorView bias,
    int stride,
    int padding,
    int outputPadding,
    int groups,
    TensorView out);

/// What one group of a transposed 1-D convolution of `input` (N, C, L) by `weight` (C, K / groups,
/// R) contributes, before anything has been said about where it lands: (N * L, K / groups * R),
/// row `n * L + l` and column `k * R + r` being what input position `l` of item `n` adds to output
/// channel `k` of the group at position `l * stride + r - padding`. One `copy` and one `matmul`.
///
/// A device with a kernel of its own for `convTranspose1d` runs this and then gathers every
/// output position from the columns that land on it, in one pass: col2im.
Tensor transposedConvolutionColumns(Operators *op, TensorView input, TensorView weight, int groups, int group);

/// What `convTranspose1d` checks of its arguments on every device, and the output length.
int checkConvTranspose1d(
    TensorView input,
    TensorView weight,
    TensorView bias,
    int stride,
    int padding,
    int outputPadding,
    int groups,
    TensorView out);

/// A snake as `mul`, `sin`, `square`, `divTensor` and `add`, with the channel parameters
/// broadcast across the length by an expanded view.
void composedSnake(
    Operators *op,
    TensorView input,
    TensorView alpha,
    TensorView beta,
    float eps,
    TensorView out);

/// A padding as a `copy` of the middle and one of each edge: zeros by `fill`, a repeated sample
/// by an expanded view of it, and a mirror by a `matmul` against the anti-diagonal -- reading a
/// dimension backwards is a permutation, and a permutation is a matrix.
void composedPad1d(
    Operators *op,
    TensorView input,
    int left,
    int right,
    PadMode mode,
    TensorView out);

/// A clamp as the `max` and then the `min` reduction of each element beside its bound, so that a
/// value inside the interval comes out exactly itself; either is skipped when its bound is
/// infinite, and a NaN the reductions dropped is put back.
void composedClamp(Operators *op, TensorView input, float lo, float hi, TensorView out);

/// A short time Fourier transform as a reflection padding and a `conv1d` against the bank of
/// windowed sinusoids, which is built on the host from `window`.
void composedStft(
    Operators *op,
    TensorView input,
    TensorView window,
    int nFft,
    int hop,
    bool centered,
    TensorView out);

/// Its inverse as a `convTranspose1d` against the inverse bank -- the overlap-add -- divided by the
/// window's squared overlap, both built on the host from `window`.
void composedIstft(
    Operators *op,
    TensorView spectrum,
    TensorView window,
    int nFft,
    int hop,
    bool centered,
    TensorView out);

}  // namespace F
}  // namespace fl
