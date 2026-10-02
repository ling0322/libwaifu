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

#include "flint/cpu/tensor.h"

#include <limits>

#include "flint/cpu/common.h"

namespace fl {
namespace op {
namespace cpu {

template<typename T>
void fillZeroKernel(const TensorView &tensor) {
  CHECK(tensor.isContiguous());

  T *data = getDataPtrCpu<T>(tensor);
  int64_t numel = tensor.getNumEl();
  for (int64_t i = 0; i < numel; ++i) {
    data[i] = T(0);
  }
}

void fillZero(const TensorView &tensor) {
  if (tensor.getDType() == DType::kFloat) {
    fillZeroKernel<float>(tensor);
  }
#if LUT_CPU_ARCH == LUT_AARCH64
  else if (tensor.getDType() == DType::kFloat16) {
    fillZeroKernel<Float16>(tensor);
  }
#endif
  else {
    NOT_IMPL();
  }
}

template<typename T>
void causalMaskKernel(const TensorView &mask) {
  CHECK(mask.getDim() == 2 && mask.getShape(0) == mask.getShape(1) && mask.isContiguous());
  int length = mask.getShape(0);

  T *data = getDataPtrCpu<T>(mask);
  for (int i = 0; i < length; ++i) {
    T *row = data + static_cast<int64_t>(i) * length;
    for (int j = 0; j <= i; ++j) {
      row[j] = 0.0f;
    }
    for (int j = i + 1; j < length; ++j) {
      row[j] = -std::numeric_limits<float>::infinity();
    }
  }
}

void causalMask(const TensorView &out) {
  if (out.getDType() == DType::kFloat) return causalMaskKernel<float>(out);
#if LUT_CPU_ARCH == LUT_AARCH64
  if (out.getDType() == DType::kFloat16) return causalMaskKernel<Float16>(out);
#endif

  NOT_IMPL();
}

}  // namespace cpu
}  // namespace op
}  // namespace fl
