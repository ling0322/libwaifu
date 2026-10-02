// The MIT License (MIT)
//
// Copyright (c) 2023 Xiaoyang Chen
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

enum class MapReduceType {
  // Sum of exp(x). FP16_FP32 means the input type if fp16, intermediate and output type is fp32.
  SUM_EXP,

  // Sum of x^2.
  SUM_SQUARE,

  // Sum of x.
  SUM,

  // Get maximun number in list.
  MAX,

  // Get minimum number in list.
  MIN,

  // Return true if all elements are true
  ALL
};

/// Reduce the last dimension of `A` into `C`, contiguous, holding one element per row of `A` in
/// whatever shape; the reduction's output type is C's.
void reduceLastDim(const TensorView &A, MapReduceType reduceType, const TensorView &C);

/// Reduce every element of contiguous `A` into a new <outType>(1) temporary.
Tensor reduceAll(const TensorView &A, DType outType, MapReduceType reduceType);

}  // namespace cuda
}  // namespace op
}  // namespace fl
