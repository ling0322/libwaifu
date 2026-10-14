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

#include "lutil/error.h"
#include "lutil/log.h"
#include "flint/metal_native/common.h"
#include "flint/metal_native/ops.h"

namespace fl {
namespace op {
namespace metal_native {

namespace {

// In step with Fp8MatmulParams in kernels/fp8.metal.
struct Fp8MatmulParams {
  uint32_t M;
  uint32_t N;
  uint32_t K;
  uint32_t vecA;
  uint32_t vecB;
};

// The tile of the output each threadgroup takes, and its threads: kernels/gemm.h's.
constexpr int64_t kTile = 64;
constexpr uint32_t kGemmThreads = 128;

// And of kernels/nax/fp8.metal, whose threadgroups are as many threads: activation rows by weight
// rows.
constexpr int64_t kNaxTileM = 64;
constexpr int64_t kNaxTileN = 128;

uint32_t ceilDiv(int64_t a, int64_t b) {
  return static_cast<uint32_t>((a + b - 1) / b);
}

}  // namespace

void fp8Matmul(
    const TensorView &a,
    const TensorView &data,
    const TensorView &scale,
    const TensorView &out) {
  // The C API has checked the operands: a <float16>(..., k) and out <float16>(..., rows), both
  // contiguous, the weight <fp8e4m3>(rows, k) and the scale <float>, contiguous too.
  CHECK(a.getDType() == DType::kFloat16 && out.getDType() == DType::kFloat16);
  CHECK(data.getDType() == DType::kFp8E4M3 && scale.getDType() == DType::kFloat);
  CHECK(a.isContiguous() && out.isContiguous() && data.isContiguous() && scale.isContiguous());

  // A product over nothing is zero everywhere, and the rows of `a` cannot be counted from its
  // elements then: there are none, however many rows it has. So k == 0 is answered before the
  // division by k, from `out`, which has every row.
  int64_t k = data.getShape(1);
  int64_t rows = data.getShape(0);
  if (k == 0) {
    if (out.getNumEl() > 0) fill(out, 0.0f);
    return;
  }
  int64_t m = a.getNumEl() / k;
  if (m == 0 || rows == 0) return;

  // Rows that start on a multiple of four elements are read four at a time.
  auto fours = [](int64_t x) { return x % 4 == 0; };
  Fp8MatmulParams params{};
  params.M = static_cast<uint32_t>(m);
  params.N = static_cast<uint32_t>(rows);
  params.K = static_cast<uint32_t>(k);
  params.vecA = fours(k) && fours(a.getInternalOffset());
  params.vecB = fours(k) && fours(data.getInternalOffset());

  // On the neural accelerators where there are any, with the same parameters.
  if (getContext(a)->hasNax()) {
    getContext(a)->dispatchGroups(
        "nax_fp8_matmul",
        {getArg(out), getArg(a), getArg(data), getArg(scale)},
        &params,
        sizeof(params),
        ceilDiv(rows, kNaxTileN),
        ceilDiv(m, kNaxTileM),
        1,
        kGemmThreads);
    return;
  }

  getContext(a)->dispatchGroups(
      "fp8_matmul",
      {getArg(out), getArg(a), getArg(data), getArg(scale)},
      &params,
      sizeof(params),
      ceilDiv(rows, kTile),
      ceilDiv(m, kTile),
      1,
      kGemmThreads);
}

}  // namespace metal_native
}  // namespace op
}  // namespace fl
