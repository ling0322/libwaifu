// The MIT License (MIT)
//
// Copyright (c) 2024 Xiaoyang Chen
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

#include "flint/tensor_view.h"

namespace fl {
namespace op {
namespace cpu {

/// The chunk length the prefill factorises the sequence into. It is also the size of the triangular
/// system each chunk solves, so both backends have to agree on it for their results to line up.
constexpr int kGatedDeltaNetChunkSize = 64;

/// Gated DeltaNet linear attention over a packed (varlen) batch, into `o` <float>(T, VH, D). See
/// `Operators::gatedDeltaNetPrefill`.
void gatedDeltaNetPrefill(
    const TensorView &q,
    const TensorView &k,
    const TensorView &v,
    const TensorView &g,
    const TensorView &beta,
    const TensorView &cuSeqlens,
    const TensorView &stateSlots,
    const TensorView &state,
    const TensorView &o);

}  // namespace cpu
}  // namespace op
}  // namespace fl
