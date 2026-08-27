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

#include "flint/cuda/nvfp4.h"
#include "flint/tensor.h"

namespace fl {
namespace op {
namespace cuda {

/// @brief Return true if this build and this GPU can run the block scaled kernel. The tensor core
///        instruction it needs is arch conditional (sm_120a), so a plain sm_120 build fails here.
bool isNvfp4GemmAvailable();

/// @brief D = A * transpose(B) in half, with A quantized on the way in. This is the whole path:
///        the prologue turns the half activation into NVFP4, the block scaled tensor cores
///        multiply, and the epilogue puts the two global scales back. The weight arrives already
///        quantized because, unlike an activation, it does not change between calls.
/// @param A <half>(..., k), contiguous. Leading batch axes are folded into the row count.
/// @return <half>(..., B.rows). B.rows has to be a multiple of 8, which is how wide the epilogue
///         writes; the row count is free, and k was fixed at a multiple of 32 by the prologue.
Tensor gemmNvfp4(const Tensor &A, const Nvfp4Operand &B);

/// @brief The same, for an activation that is already NVFP4. What the epilogue multiplies by is
///        globalScaleA * globalScaleB, and both of those live on the device, so that scalar is
///        formed there rather than costing a synchronization to read back.
/// @return <half>(A.rows, B.rows).
Tensor gemmNvfp4(const Nvfp4Operand &A, const Nvfp4Operand &B);

}  // namespace cuda
}  // namespace op
}  // namespace fl
