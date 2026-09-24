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

/// @brief A 1-D convolution, as the 2-D one over an image one row tall.
///
/// Shares `conv.cc`'s im2col and GEMM with @ref conv2d -- the only thing a flat image needs of
/// its own is that the padding applies to the length and not to the height, which is why the
/// problem carries the two separately.
///
/// Any group count that divides both channel counts, the depthwise case included.
/// @param input <float|half>(N, C, L), contiguous.
/// @param weight <float|half>(K, C / groups, R), contiguous, and of the input's type or half to
///        its float.
/// @param bias <float|half>(K), or an empty tensor for no bias.
/// @return <float|half>(N, K, Lout), where Lout is
///         (L + 2 * padding - dilation * (R - 1) - 1) / stride + 1.
Tensor conv1d(
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
