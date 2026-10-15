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

#include "flint/operators.h"
#include "flint/tensor_view.h"

namespace fl {
namespace op {
namespace cuda {

/// @brief col2im for a transposed 1-D convolution: one thread per output element of one group,
///        gathering the columns that land on it and adding the bias. See flint/cpu/signal.h,
///        whose results these match.
/// @param columns <half|float>(N * L, K * R), contiguous.
/// @param bias <half|float>(K), or an empty tensor.
/// @param out <half|float>(N, K, Lout) with a contiguous last dimension.
void overlapAdd1d(
    const TensorView &columns,
    const TensorView &bias,
    int length,
    int stride,
    int padding,
    int kernel,
    const TensorView &out);

/// @brief `x + sin(alpha * x)^2 / (beta + eps)` per channel of contiguous `input` (N, C, L).
void snake(
    const TensorView &input,
    const TensorView &alpha,
    const TensorView &beta,
    float eps,
    const TensorView &out);

/// @brief Operators::pad1d of contiguous `input` into contiguous `out`.
void pad1d(const TensorView &input, int left, int right, PadMode mode, const TensorView &out);

/// @brief `min(max(x, lo), hi)` of contiguous `input` into `out`.
void clamp(const TensorView &input, float lo, float hi, const TensorView &out);

}  // namespace cuda
}  // namespace op
}  // namespace fl
