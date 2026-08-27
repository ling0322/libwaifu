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

#include "flint/cpu/glu.h"

#include <cmath>

#include <math.h>

#include "lutil/thread_pool.h"
#include "flint/cpu/accessor.h"
#include "flint/cpu/tensor.h"

namespace fl {
namespace op {
namespace cpu {

/// Which activation the gate half goes through. Either way the first half of the last dimension
/// is the gate and the second is the value.
enum class GateOp { SILU, GELU };

template<GateOp OP>
inline float applyGate(float x) {
  if constexpr (OP == GateOp::SILU) {
    return x / (1.0f + expf(-x));
  } else {
    // The exact GELU, so that this matches torch.nn.GELU() rather than its tanh approximation.
    return x * 0.5f * (1.0f + erff(x * 0.70710678118654752f));
  }
}

template<typename T, GateOp OP>
Tensor gatedLinearKernel(const Tensor &A) {
  std::vector<int> shapeC = A.getShape();
  shapeC.back() /= 2;
  Tensor C = tensor(shapeC, DType::getType<T>());

  TensorList<const T, 1> vA = TensorList<const T, 1>::fromTensor(A);
  TensorList<T, 1> vC = TensorList<T, 1>::fromTensor(C);
  CHECK(vA.getLength() == vC.getLength());

  int numRows = vA.getLength();
#pragma omp parallel for schedule(dynamic, 1)
  for (int j = 0; j < numRows; ++j) {
    TensorAccessor<const T, 1> a = vA.getTensor(j);
    TensorAccessor<T, 1> c = vC.getTensor(j);

    int n = c.getShape(0);
    for (int i = 0; i < n; ++i) {
      float gate = applyGate<OP>(static_cast<float>(a[i]));
      c[i] = static_cast<T>(gate * static_cast<float>(a[i + n]));
    }
  }

  return C;
}

template<GateOp OP>
Tensor gatedLinear(const Tensor &A) {
  CHECK(A.getShape(-1) % 2 == 0);

  if (A.getDType() == DType::kFloat) return gatedLinearKernel<float, OP>(A);
#if LUT_CPU_ARCH == LUT_AARCH64
  if (A.getDType() == DType::kFloat16) return gatedLinearKernel<Float16, OP>(A);
#endif

  NOT_IMPL();
}

Tensor swiglu(const Tensor &A) {
  return gatedLinear<GateOp::SILU>(A);
}

Tensor geglu(const Tensor &A) {
  return gatedLinear<GateOp::GELU>(A);
}

}  // namespace cpu
}  // namespace op
}  // namespace fl
