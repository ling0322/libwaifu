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
namespace cuda {

/// @brief An operand in the form the SM120 block scaled tensor cores read: E2M1 elements, one
///        E4M3 scale per 16 of them in the interleaved atom layout the MMA expects, and a
///        per-tensor scale that brings the block scales into the range E4M3 can hold.
struct Nvfp4Operand {
  /// <fp4x2>(rows, k / 2), row major, the even element of a pair in the low nibble.
  Tensor data;
  /// <uint8>(numScaleByte), E4M3, in the SM1xx atom layout rather than row major.
  Tensor blockScale;
  /// <float>(1), on the device: amax / (6 * 448). The block scales are relative to it.
  Tensor globalScale;
  /// Logical shape of the operand. `data` only carries the packed one.
  int rows;
  int k;
};

/// @brief The prologue. Quantizes a half operand to NVFP4: one pass for the tensor wide maximum
///        that fixes the global scale, a second for the elements, the block scales, and the
///        padding the atom layout carries beyond the operand's own extent.
/// @param x <half>(rows, k), contiguous, k a multiple of 32.
Nvfp4Operand quantizeNvfp4(const Tensor &x);

/// @brief The scalars an epilogue has to multiply by: globalScaleA * globalScaleB, and a zero
///        beta next to it. Both global scales come off the device, so the product is formed there
///        rather than costing a synchronization, and cuBLASLt's device pointer mode wants beta on
///        the device as well.
/// @return <float>(2), on the device: {alpha, 0}.
Tensor nvfp4Alpha(const Nvfp4Operand &A, const Nvfp4Operand &B);

/// @brief Inverse of quantizeNvfp4, which is what a caller wants to see the quantization error on
///        its own. Reads the interleaved scale layout, so it also pins that layout down in tests.
/// @return <half>(rows, k).
Tensor dequantNvfp4ToHalf(const Nvfp4Operand &operand);

}  // namespace cuda
}  // namespace op
}  // namespace fl
