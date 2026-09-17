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
namespace cpu {

/// @brief Quantize a tensor to E4M3, one scale per row.
///
/// The same arithmetic the CUDA quantizer does, and the same bytes out: the two conversions were
/// checked against each other over every float there is.
/// @param x <float> or <float16>(rows, k), contiguous.
Fp8Operand quantizeFp8(const Tensor &x);

/// @brief Inverse of quantizeFp8, which is what a caller wants to see the quantization error on
///        its own.
/// @return <float>(rows, k). Float rather than half because float is what this device computes
///         in; a model on the CPU widens its weights on the way into the arithmetic anyway.
Tensor dequantFp8ToFloat(const Fp8Operand &operand);

/// @brief D = A * transpose(B) in float32, with B read as E4M3 and widened while it is packed.
///
/// Unlike the CUDA path this buys no speed -- the CPU has no FP8 arithmetic to reach for, and the
/// multiply was float32 either way -- so what it is for is the weight being one byte an element
/// in memory rather than two.
/// @param A <float>(..., k), contiguous. Leading batch axes are folded into the row count.
/// @param B the weight, one output channel per row, quantized once at load.
/// @return <float>(..., B.rows).
Tensor gemmFp8(const Tensor &A, const Fp8Operand &B);

}  // namespace cpu
}  // namespace op
}  // namespace fl
