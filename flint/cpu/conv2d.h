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

#include "flint/tensor.h"

namespace fl {
namespace op {
namespace cpu {

/// @brief A 2-D convolution, as an im2col and the GEMM that is already here.
/// @param input <float>(N, C, H, W), contiguous.
/// @param weight <float>(K, C / groups, R, S), contiguous and of the same type.
/// @param bias <float>(K), or an empty tensor for no bias.
/// @return <float>(N, K, outH, outW), where outH is
///         (H + 2 * padding - dilation * (R - 1) - 1) / stride + 1, and outW likewise.
Tensor conv2d(
    const Tensor &input,
    const Tensor &weight,
    const Tensor &bias,
    int stride,
    int padding,
    int dilation,
    int groups);

}  // namespace cpu
}  // namespace op
}  // namespace fl
