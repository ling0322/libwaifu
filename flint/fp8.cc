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

#include "flint/fp8.h"

#include <limits>

#include "lutil/error.h"

namespace fl {

Fp8Operand makeFp8Operand(const Tensor &data, const Tensor &channelScale) {
  if (data.getDType() != DType::kFp8E4M3 || data.getDim() != 2) {
    throw lut::InvalidArgError("fp8 operand: data is not <fp8e4m3>(rows, k)");
  }
  if (channelScale.getDType() != DType::kFloat || channelScale.getDim() != 1) {
    throw lut::InvalidArgError("fp8 operand: channel scale is not <float>(rows)");
  }
  if (channelScale.getShape(0) != data.getShape(0)) {
    throw lut::InvalidArgError("fp8 operand: one scale per row is not what was handed over");
  }
  // A weight and its scales on different devices would reach a kernel as one pointer it may touch
  // and one it may not, and nothing would say so.
  if (data.getDevice().getType() != channelScale.getDevice().getType()) {
    throw lut::InvalidArgError("fp8 operand: the data and the scales are on different devices");
  }
  if (!data.isContiguous() || !channelScale.isContiguous()) {
    throw lut::InvalidArgError("fp8 operand: not contiguous");
  }
  // The kernels index the operand in int, which is two gigabytes of E4M3 and more than any weight
  // here is. A caller reaching past it would otherwise get a wrapped index rather than an error.
  if (data.getNumEl() >= std::numeric_limits<int32_t>::max()) {
    throw lut::InvalidArgError("fp8 operand: more elements than the kernels can index");
  }

  Fp8Operand operand;
  operand.data = data;
  operand.channelScale = channelScale;
  operand.rows = data.getShape(0);
  operand.k = data.getShape(1);

  return operand;
}

}  // namespace fl
