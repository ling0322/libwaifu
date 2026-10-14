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
namespace op {
namespace cuda {

/// @brief Return true if this build and this GPU can run the mixed precision kernel. It is an
///        sm_80 kernel, so this is true on anything from Ampere on.
bool isFp8GemmAvailable();

/// @brief D = A * transpose(B) * scale in half, with A read as half and B as E4M3 upcast in
///        registers, and one scale for the whole of B.
///
/// The activation stays in half: what the tensor cores run is an HMMA either way, so quantizing
/// it would buy nothing here. What the narrow weight buys is the traffic between global memory
/// and the tensor cores, which is what a projection is bound by.
/// @param A <half>(..., k), contiguous. Leading batch axes are folded into the row count.
/// @param B <fp8e4m3>(n, k), contiguous, one output channel per row, as a package stores it.
/// @param scale <float> with one element, on the device. It is read there by the kernel, so the
///              launch does not wait for it.
/// @param D <half>(..., n), contiguous. n has to be a multiple of 8, which is how wide the
///          epilogue writes, and k a multiple of 16, which is how wide the mainloop reads B.
void gemmFp8(TensorView A, TensorView B, TensorView scale, TensorView D);

}  // namespace cuda
}  // namespace op
}  // namespace fl
