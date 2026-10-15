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
namespace cpu {

/// @brief The second half of a transposed 1-D convolution, col2im: every output position of one
/// group gathered from the columns that land on it, plus the bias, in one pass.
/// @param columns <float|half>(N * L, K * R), contiguous, from F::transposedConvolutionColumns.
/// @param bias <float|half>(K), the group's share of the bias, or an empty tensor for none.
/// @param out <float|half>(N, K, Lout), the group's channels of the result. Its last dimension is
///        contiguous; the others may have any stride.
void overlapAdd1d(
    const TensorView &columns,
    const TensorView &bias,
    int length,
    int stride,
    int padding,
    int kernel,
    const TensorView &out);

/// @brief `x + sin(alpha * x)^2 / (beta + eps)` per channel of `input` (N, C, L), contiguous, in
/// one pass. An empty `beta` is `alpha`.
void snake(
    const TensorView &input,
    const TensorView &alpha,
    const TensorView &beta,
    float eps,
    const TensorView &out);

/// @brief `input` (..., L), contiguous, padded on its last dimension into `out` (..., left + L +
/// right) as Operators::pad1d describes.
void pad1d(const TensorView &input, int left, int right, PadMode mode, const TensorView &out);

/// @brief `min(max(x, lo), hi)` of contiguous `input`, into `out` of its shape.
void clamp(const TensorView &input, float lo, float hi, const TensorView &out);

}  // namespace cpu
}  // namespace op
}  // namespace fl
