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

#include "flint/functional.h"

namespace fl {

/// @brief Tensors in, a tensor out: the C++ tests' way of calling the operators.
///
/// Every function here works out its result's shape and type from the inputs, allocates it on
/// their device, and hands everything to the operator as views, the result last. These are the
/// same rules the binding applies before it calls through the C API -- a copy kept for the tests,
/// which are far easier to write against results than against output views -- and nothing in the
/// library itself calls them.
///
/// `op` is the operators the caller chose, which are the inputs' device's.
namespace F {

/// @brief A tensor of `shape` and `dtype` on `device`, every element zero.
Tensor zeros(Device device, lut::Span<const int> shape, DType dtype);

/// @brief A tensor of `shape` and `dtype` on `device` filled from [0, 1).
Tensor rand(Device device, lut::Span<const int> shape, DType dtype);

/// @brief A float tensor of `shape` on `device` filled from the standard normal distribution.
Tensor randNormal(Device device, lut::Span<const int> shape);

/// @brief `begin`, `begin + step`, ... up to and not including `end`, as <int64> on `device`.
Tensor arangeLong(Device device, LongType begin, LongType end, LongType step);

/// @brief The (n, n) causal mask in `device`'s default float type: zero on and below the
/// diagonal, minus infinity above it.
Tensor causalMask(Device device, int n);

// --- Moving and reshaping -------------------------------------------------------------------

/// @brief `input` on `device`: `input` itself if it is already there, otherwise a contiguous copy.
Tensor toDevice(Operators *op, const Tensor &input, Device device);

/// @brief `input` with its elements converted to `dtype`: `input` itself if it is already that.
Tensor cast(Operators *op, const Tensor &input, DType dtype);

/// @brief `input` if it is contiguous, otherwise a contiguous copy of it.
Tensor contiguous(Operators *op, const Tensor &input);

/// @brief `A` and `B` joined along `dim`, the one dimension they may disagree on.
Tensor cat(Operators *op, TensorView A, TensorView B, int dim);

// --- Elementwise ----------------------------------------------------------------------------

Tensor add(Operators *op, TensorView input, TensorView other);
Tensor sub(Operators *op, TensorView input, TensorView other);
Tensor mul(Operators *op, TensorView input, TensorView other);
Tensor divTensor(Operators *op, TensorView input, TensorView other);
Tensor eq(Operators *op, TensorView input, TensorView other);
Tensor mul(Operators *op, TensorView input, float other);
Tensor div(Operators *op, TensorView input, float other);
Tensor mod(Operators *op, TensorView input, LongType other);

Tensor square(Operators *op, TensorView input);
Tensor neg(Operators *op, TensorView input);
Tensor abs(Operators *op, TensorView input);
Tensor exp(Operators *op, TensorView input);
Tensor log(Operators *op, TensorView input);
Tensor round(Operators *op, TensorView input);
Tensor sqrt(Operators *op, TensorView input);
Tensor rsqrt(Operators *op, TensorView input);
Tensor sigmoid(Operators *op, TensorView input);
Tensor tanh(Operators *op, TensorView input);
Tensor relu(Operators *op, TensorView input);
Tensor gelu(Operators *op, TensorView input);
Tensor silu(Operators *op, TensorView input);
Tensor sin(Operators *op, TensorView input);
Tensor cos(Operators *op, TensorView input);
Tensor quickGelu(Operators *op, TensorView input);
Tensor swiglu(Operators *op, TensorView input);
Tensor geglu(Operators *op, TensorView input);
Tensor snake(Operators *op, TensorView input, TensorView alpha, TensorView beta, float eps);

// --- Reductions -----------------------------------------------------------------------------

Tensor sum(Operators *op, TensorView input, int dim);
Tensor cumsum(Operators *op, TensorView input, int dim);
Tensor max(Operators *op, TensorView input);
Tensor min(Operators *op, TensorView input);
Tensor softmax(Operators *op, TensorView input);

// --- Normalization and resampling -----------------------------------------------------------

Tensor rmsNorm(Operators *op, TensorView input, TensorView weight, float eps);
Tensor layerNorm(Operators *op, TensorView input, TensorView weight, TensorView bias, float eps);
Tensor groupNorm(
    Operators *op,
    TensorView input,
    TensorView weight,
    TensorView bias,
    int groups,
    float eps);
Tensor upsampleNearest2d(Operators *op, TensorView input, int scale);
Tensor upsampleNearest1d(Operators *op, TensorView input, int size);

// --- Products and convolutions --------------------------------------------------------------

Tensor matmul(Operators *op, TensorView A, TensorView B);
Tensor conv2d(
    Operators *op,
    TensorView input,
    TensorView weight,
    TensorView bias,
    int stride,
    int padding,
    int dilation,
    int groups);
Tensor conv1d(
    Operators *op,
    TensorView input,
    TensorView weight,
    TensorView bias,
    int stride,
    int padding,
    int dilation,
    int groups);
Tensor convTranspose1d(
    Operators *op,
    TensorView input,
    TensorView weight,
    TensorView bias,
    int stride,
    int padding,
    int outputPadding,
    int groups);
Tensor stft(Operators *op, TensorView input, TensorView window, int nFft, int hop, bool centered);
Tensor istft(
    Operators *op,
    TensorView spectrum,
    TensorView window,
    int nFft,
    int hop,
    bool centered);
Tensor lookup(Operators *op, TensorView table, TensorView indices);

// --- Attention and sampling -----------------------------------------------------------------

Tensor attention(Operators *op, TensorView q, TensorView k, TensorView v, bool causal);

Tensor pagedAttention(
    Operators *op,
    TensorView q,
    TensorView keyCache,
    TensorView valueCache,
    TensorView blockTable,
    TensorView cuSeqlensQ,
    TensorView seqlensK,
    int maxQLen,
    int maxKLen,
    bool causal);
Tensor gatedDeltaNetPrefill(
    Operators *op,
    TensorView q,
    TensorView k,
    TensorView v,
    TensorView g,
    TensorView beta,
    TensorView cuSeqlens,
    TensorView stateSlots,
    TensorView state);
Tensor sample(
    Operators *op,
    TensorView logits,
    TensorView temperatures,
    TensorView topKs,
    TensorView topPs);

}  // namespace F
}  // namespace fl
