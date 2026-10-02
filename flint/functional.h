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

#include <stdint.h>

#include <memory>

#include "lutil/span.h"
#include "flint/device.h"
#include "flint/dtype.h"
#include "flint/tensor.h"
#include "flint/tensor_view.h"

namespace fl {

class Operators;

/// @brief What the operators themselves need of allocation, and the one operation they share that
/// is made of others.
///
/// What every operation's result is -- its shape, its type, where it lives -- is decided by the
/// caller that allocates it: the binding, for everything that comes in through the C API, and
/// `flint/test_functional.h` for the C++ tests, which are written against results rather than
/// against output views. The operators in `flint/operators.h` compute and nothing else.
///
/// Its own namespace rather than `fl`'s, because the test helpers in the same namespace name
/// `exp`, `log`, `sqrt` and the rest, which would otherwise hide the C library's functions of the
/// same names from every unqualified call in flint.
namespace F {

// --- Allocation -----------------------------------------------------------------------------

/// @brief Uninitialized storage for `numel` elements of `dtype` on `device` -- Device::kCudaHost
/// being page-locked host memory the CUDA device reads directly. At least one element is
/// allocated, so that a tensor with a dimension of zero still has storage to point at.
std::unique_ptr<TensorData> allocate(Device device, int64_t numel, DType dtype);

/// @brief An uninitialized contiguous tensor of `shape` and `dtype` on `device`. Device::kCudaHost
/// is page-locked host memory the CUDA device reads directly.
Tensor empty(Device device, lut::Span<const int> shape, DType dtype);

/// @brief An uninitialized contiguous tensor of `input`'s shape, type and device.
Tensor emptyLike(const TensorView &input);

// --- Attention ------------------------------------------------------------------------------

/// @brief Attention as matmul, softmax and matmul, a block of queries at a time, written into
/// `out`. What Operators::attention does on a device with no kernel of its own.
void composedAttention(
    Operators *op,
    TensorView q,
    TensorView k,
    TensorView v,
    bool causal,
    TensorView out);

}  // namespace F
}  // namespace fl
