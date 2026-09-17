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

/// E4M3's largest finite magnitude. A row scaled by rowAmax / 448 has its largest element land
/// exactly on it, so the format's whole range is used and nothing saturates.
constexpr float kFp8E4M3Max = 448.0f;

/// @brief A weight held in E4M3 with one scale per output channel: row `r` of `data` means
///        `data[r] * channelScale[r]`.
///
/// One scale per row rather than one per tensor, because a projection's output channels do not
/// share a magnitude -- a single outlier channel would otherwise push every other channel down
/// into E4M3's subnormals. It is also the coarsest scaling a multiply can undo for free: a scale
/// that is constant down a column of the result is one pass over the result, where a finer one
/// would have to be applied inside the mainloop.
///
/// Not `op::cuda`'s, though `op::cuda` is the only namespace that can multiply by one today. The
/// bytes mean what they mean without a device -- a package stores them, an exporter writes them on
/// a processor -- so what holds them together lives here rather than beside the kernels that
/// happen to read them.
struct Fp8Operand {
  /// <fp8e4m3>(rows, k), row major.
  Tensor data;
  /// <float>(rows). Row `r` of the original tensor is `data[r] * channelScale[r]`.
  Tensor channelScale;
  /// Shape of `data`, repeated because a GEMM wants it as two ints.
  int rows;
  int k;
};

/// @brief Rebuild an operand from the two tensors a quantizer produced, checking what a caller
///        could have got wrong.
///
/// Reached from outside the library, where what arrives is a caller's mistake rather than a
/// broken invariant of ours, so these throw InvalidArg rather than CHECK. What is checked here is
/// the format alone: whatever a device's own kernel needs beyond it -- an alignment, a row count
/// that divides -- is that kernel's to ask for.
Fp8Operand makeFp8Operand(const Tensor &data, const Tensor &channelScale);

}  // namespace fl
