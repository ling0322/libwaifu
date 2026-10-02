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
#include "flint/tensor_view.h"

namespace fl {

/// E4M3's largest finite magnitude. A row scaled by rowAmax / 448 has its largest element land
/// exactly on it, so the format's whole range is used and nothing saturates.
constexpr float kFp8E4M3Max = 448.0f;

namespace op {
namespace cuda {

/// @brief Quantize a half tensor to E4M3, one scale per row.
///
/// The scale is `rowAmax / 448` -- E4M3's largest finite magnitude -- so the largest element of
/// each row lands exactly on the top of the format's range. A row that is all zero gets a zero
/// scale and quantizes to zeros rather than to NaN.
///
/// One scale per row rather than one per tensor, because a projection's output channels do not
/// share a magnitude -- a single outlier channel would otherwise push every other channel down
/// into E4M3's subnormals. It is also the coarsest scaling a multiply can undo for free: a scale
/// that is constant down a column of the result is one pass over the result.
/// @param x <half>(rows, k), contiguous, k a multiple of 16.
/// @param data <fp8e4m3>(rows, k), contiguous: the codes. Row `r` of `x` is
///             `data[r] * channelScale[r]`.
/// @param channelScale <float>(rows), contiguous.
void quantizeFp8(TensorView x, TensorView data, TensorView channelScale);

/// @brief Inverse of quantizeFp8, which is what a caller wants to see the quantization error on
///        its own.
/// @param out <half>(rows, k), contiguous.
void dequantFp8ToHalf(TensorView data, TensorView channelScale, TensorView out);

}  // namespace cuda
}  // namespace op
}  // namespace fl
