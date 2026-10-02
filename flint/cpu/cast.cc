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

#include "flint/cpu/cast.h"

#include <math.h>
#include <string.h>

#include <algorithm>

#include "lutil/half.h"
#include "flint/cpu/common.h"
#include "flint/cpu/kernel/interface.h"
#include "flint/functional.h"

namespace fl {
namespace op {
namespace cpu {

namespace {

void checkCastPair(const TensorView &A, const TensorView &C, const char *what) {
  CHECK(A.isContiguous() && C.isContiguous()) << "unable to cast " << what << " not contiguous";
  C.throwIfInvalidShape(A.getShape(), "cast");
}

}  // namespace

void cast(const TensorView &A, const TensorView &C) {
  DType from = A.getDType();
  DType to = C.getDType();
  if (from == DType::kFloat16 && to == DType::kFloat) {
    castFp16ToFp32(A, C);
  } else if (from == DType::kFloat && to == DType::kFloat16) {
    castFp32ToFp16(A, C);
  } else if (from == DType::kFloat && to == DType::kLong) {
    castFp32ToLong(A, C);
  } else if (from == DType::kFloat16 && to == DType::kLong) {
    // Every half is exactly a float, so going through one changes nothing.
    Tensor wide = F::empty(Device::getCpu(), A.getShape(), DType::kFloat);
    castFp16ToFp32(A, wide);
    castFp32ToLong(wide, C);
  } else {
    NOT_IMPL();
  }
}

void castFp16ToFp32(const TensorView &A, const TensorView &C) {
  checkCastPair(A, C, "a half tensor to float,");
  kernel::convertHalfToFloat(
      A.getNumEl(),
      reinterpret_cast<const kernel::Float16 *>(getDataPtrCpu<Float16>(A)),
      getDataPtrCpu<float>(C),
      kernel::Mode::OMP,
      kernel::CpuMathBackend::DEFAULT);
}

void castFp32ToFp16(const TensorView &A, const TensorView &C) {
  checkCastPair(A, C, "a float tensor to half,");
  kernel::convertFloatToHalf(
      A.getNumEl(),
      getDataPtrCpu<float>(A),
      reinterpret_cast<kernel::Float16 *>(getDataPtrCpu<Float16>(C)),
      kernel::Mode::OMP,
      kernel::CpuMathBackend::DEFAULT);
}

void castFp32ToLong(const TensorView &A, const TensorView &C) {
  checkCastPair(A, C, "a float tensor to int64,");
  const float *src = getDataPtrCpu<float>(A);
  LongType *dest = getDataPtrCpu<LongType>(C);

  int64_t numel = A.getNumEl();
#pragma omp parallel for schedule(static)
  for (int64_t i = 0; i < numel; ++i) {
    dest[i] = static_cast<LongType>(src[i]);
  }
}

}  // namespace cpu
}  // namespace op
}  // namespace fl
