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

#include "flint/fp8.h"
#include "flint/tensor.h"

namespace fl {
namespace op {
namespace cuda {

/// @brief Quantize a half tensor to E4M3, one scale per row.
///
/// The scale is `rowAmax / 448` -- E4M3's largest finite magnitude -- so the largest element of
/// each row lands exactly on the top of the format's range. A row that is all zero gets a zero
/// scale and quantizes to zeros rather than to NaN.
/// @param x <half>(rows, k), contiguous, k a multiple of 16.
Fp8Operand quantizeFp8(const Tensor &x);

/// @brief Inverse of quantizeFp8, which is what a caller wants to see the quantization error on
///        its own.
/// @return <half>(rows, k).
Tensor dequantFp8ToHalf(const Fp8Operand &operand);

}  // namespace cuda
}  // namespace op
}  // namespace fl
