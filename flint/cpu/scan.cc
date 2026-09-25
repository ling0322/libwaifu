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

#include "flint/cpu/scan.h"

#include "flint/cpu/common.h"
#include "flint/cpu/tensor.h"

namespace fl {
namespace op {
namespace cpu {

Tensor cumsumLastDim(const Tensor &A) {
  CHECK(A.getDType() == DType::kFloat);
  CHECK(A.isContiguous());
  CHECK(A.getDim() >= 1);

  Tensor C = tensor(A.getShape(), DType::kFloat);
  int64_t length = A.getShape(-1);
  if (length == 0) return C;
  int64_t rows = A.getNumEl() / length;

  const float *a = getDataPtrCpu<float>(A);
  float *c = getDataPtrCpu<float>(C);

#pragma omp parallel for schedule(static)
  for (int64_t row = 0; row < rows; ++row) {
    const float *in = a + row * length;
    float *out = c + row * length;

    double running = 0.0;
    for (int64_t i = 0; i < length; ++i) {
      running += in[i];
      out[i] = static_cast<float>(running);
    }
  }

  return C;
}

}  // namespace cpu
}  // namespace op
}  // namespace fl
