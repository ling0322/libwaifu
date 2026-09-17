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

// Half of what docs/fp8.md describes, on the processor. Nothing here is fast for being narrow:
// x64 and aarch64 have no FP8 arithmetic, so the weight is widened to float32 on its way into the
// micro-kernel and the multiply is the float32 one it would have been anyway. What the format
// buys on this device is the weight taking one byte an element in memory instead of two.

#include "flint/cpu/fp8.h"

#include <cmath>
#include <limits>

#include "lutil/error.h"
#include "flint/cpu/accessor.h"
#include "flint/cpu/common.h"
#include "flint/cpu/kernel/interface.h"
#include "flint/cpu/kernel/util.h"
#include "flint/cpu/matmul.h"
#include "flint/cpu/tensor.h"

namespace fl {
namespace op {
namespace cpu {
namespace {

/// One row's largest magnitude, reading the row in whichever type it is stored in.
template<typename T>
float rowAmax(const T *row, int k);

template<>
float rowAmax<float>(const float *row, int k) {
  float amax = 0.0f;
  for (int i = 0; i < k; ++i) amax = std::fmax(amax, std::fabs(row[i]));
  return amax;
}

template<>
float rowAmax<Float16>(const Float16 *row, int k) {
  const kernel::Float16 *x = reinterpret_cast<const kernel::Float16 *>(row);
  float amax = 0.0f;
  for (int i = 0; i < k; ++i) amax = std::fmax(amax, std::fabs(kernel::cvt_h2s(x[i])));
  return amax;
}

template<typename T>
float widen(T v);

template<>
float widen<float>(float v) {
  return v;
}

template<>
float widen<Float16>(Float16 v) {
  return kernel::cvt_h2s(*reinterpret_cast<const kernel::Float16 *>(&v));
}

/// One row per iteration, so the scale and the elements it scales are found in one pass over the
/// row rather than two over the tensor. The row is read twice, which is what keeps it to one
/// pass; a weight is quantized once at load, so nothing pays that twice.
template<typename T>
void quantizeRows(const Tensor &x, Tensor &data, Tensor &channelScale) {
  const T *src = getDataPtrCpu<T>(x);
  kernel::Fp8E4M3 *dst = reinterpret_cast<kernel::Fp8E4M3 *>(getDataPtrCpu<Fp8E4M3>(data));
  float *scale = getDataPtrCpu<float>(channelScale);

  int rows = x.getShape(0);
  int k = x.getShape(1);

#pragma omp parallel for schedule(dynamic, 1)
  for (int r = 0; r < rows; ++r) {
    const T *row = src + static_cast<int64_t>(r) * k;
    kernel::Fp8E4M3 *out = dst + static_cast<int64_t>(r) * k;

    float amax = rowAmax<T>(row, k);

    // An all zero row has no scale to speak of. Zero is the one that makes the round trip exact
    // and leaves nothing to divide by later.
    scale[r] = amax / kFp8E4M3Max;
    float rcpScale = amax > 0.0f ? kFp8E4M3Max / amax : 0.0f;

    for (int i = 0; i < k; ++i) {
      out[i] = kernel::convertFloatToFp8(widen<T>(row[i]) * rcpScale);
    }
  }
}

/// C[m][n] *= channelScale[n]. The scale is constant down a column of the result, so applying it
/// afterwards is exactly what applying it to the weight would have given -- and it is a pass over
/// C, which is M*N against the multiply's M*N*K.
void applyChannelScale(Tensor &C, const Tensor &channelScale) {
  float *c = getDataPtrCpu<float>(C);
  const float *scale = getDataPtrCpu<float>(channelScale);

  int m = C.getShape(0);
  int n = C.getShape(1);

#pragma omp parallel for schedule(dynamic, 1)
  for (int i = 0; i < m; ++i) {
    float *row = c + static_cast<int64_t>(i) * n;
    for (int j = 0; j < n; ++j) row[j] *= scale[j];
  }
}

Tensor gemmFp8_2d(const Tensor &A, const Fp8Operand &B) {
  Tensor C = op::cpu::zeros({A.getShape(0), B.rows}, DType::kFloat);

  // The weight is stored one output channel per row, so as the right hand operand of A * B it is
  // its own transpose: (k, n) read down its columns.
  Tensor Bt = B.data.transpose(0, 1);
  GEMMArgs gemmArgs = generateGemmArgs(A, Bt, C);

  kernel::gemmFp8WeightFloat(
      gemmArgs.transA,
      gemmArgs.transB,
      gemmArgs.M,
      gemmArgs.N,
      gemmArgs.K,
      getDataPtrCpu<float>(A),
      gemmArgs.lda,
      reinterpret_cast<const kernel::Fp8E4M3 *>(getDataPtrCpu<Fp8E4M3>(B.data)),
      gemmArgs.ldb,
      getDataPtrCpu<float>(C),
      gemmArgs.ldc,
      kernel::Mode::OMP);

  applyChannelScale(C, B.channelScale);

  return C;
}

}  // namespace

Fp8Operand quantizeFp8(const Tensor &x) {
  CHECK(x.getDevice().getType() == Device::kCpu);
  CHECK(x.getDim() == 2);
  CHECK(x.isContiguous());
  CHECK(x.getNumEl() < std::numeric_limits<int32_t>::max());

  Fp8Operand operand;
  operand.rows = x.getShape(0);
  operand.k = x.getShape(1);
  operand.data = op::cpu::tensor({operand.rows, operand.k}, DType::kFp8E4M3);
  operand.channelScale = op::cpu::tensor({operand.rows}, DType::kFloat);

  if (x.getDType() == DType::kFloat) {
    quantizeRows<float>(x, operand.data, operand.channelScale);
  } else if (x.getDType() == DType::kFloat16) {
    quantizeRows<Float16>(x, operand.data, operand.channelScale);
  } else {
    NOT_IMPL();
  }

  return operand;
}

Tensor dequantFp8ToFloat(const Fp8Operand &operand) {
  CHECK(operand.data.getDevice().getType() == Device::kCpu);

  Tensor x = op::cpu::tensor({operand.rows, operand.k}, DType::kFloat);

  const kernel::Fp8E4M3 *src =
      reinterpret_cast<const kernel::Fp8E4M3 *>(getDataPtrCpu<Fp8E4M3>(operand.data));
  const float *scale = getDataPtrCpu<float>(operand.channelScale);
  float *dst = getDataPtrCpu<float>(x);

  int k = operand.k;
#pragma omp parallel for schedule(dynamic, 1)
  for (int r = 0; r < operand.rows; ++r) {
    const kernel::Fp8E4M3 *row = src + static_cast<int64_t>(r) * k;
    float *out = dst + static_cast<int64_t>(r) * k;
    float rowScale = scale[r];

    for (int i = 0; i < k; ++i) out[i] = kernel::convertFp8ToFloat(row[i]) * rowScale;
  }

  return x;
}

Tensor gemmFp8(const Tensor &A, const Fp8Operand &B) {
  CHECK(A.getDevice().getType() == Device::kCpu);
  CHECK(B.data.getDevice().getType() == Device::kCpu);
  CHECK(A.getDType() == DType::kFloat);
  CHECK(A.isContiguous());
  CHECK(A.getDim() >= 2);
  CHECK(A.getShape(-1) == B.k);

  if (A.getDim() == 2) return gemmFp8_2d(A, B);

  std::vector<int> shape = A.getShape();
  Tensor C = gemmFp8_2d(A.view({-1, A.getShape(-1)}), B);

  shape.back() = B.rows;
  return C.view(shape);
}

}  // namespace cpu
}  // namespace op
}  // namespace fl
