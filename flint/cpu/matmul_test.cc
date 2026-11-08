// The MIT License (MIT)
//
// Copyright (c) 2026 Xiaoyang Chen
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
// of the Software, and to permit persons to whom the Software is furnished to do
// so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <algorithm>
#include <cmath>
#include <vector>

#include "catch2/catch_amalgamated.hpp"
#include "flint/test_functional.h"
#include "flint/cpu/common.h"
#include "flint/operators.h"
#include "flint/tensor.h"

namespace fl {

namespace {

/// The CPU operators, which is what the calls in this file are asked of.
Operators *cpuOps() {
  return getOperators(Device::kCpu);
}

}  // namespace

namespace op {
namespace cpu {

namespace {

Tensor RefMatMulFp32(const Tensor &A, const Tensor &B) {
  CATCH_REQUIRE(A.getDType() == B.getDType());
  CATCH_REQUIRE(A.getDim() == 2);
  CATCH_REQUIRE(B.getDim() == 2);
  CATCH_REQUIRE(A.getShape(1) == B.getShape(0));
  CATCH_REQUIRE(A.getDType() == DType::kFloat);

  Tensor C = fl::test::zeros(Device::getCpu(), {A.getShape(0), B.getShape(1)}, DType::kFloat);
  float *dataC = getDataPtrCpu<float>(C);
  const float *dataA = getDataPtrCpu<float>(A), *dataB = getDataPtrCpu<float>(B);
  int stride0A = A.getStride(0);
  int stride1A = A.getStride(1);
  int stride0B = B.getStride(0);
  int stride1B = B.getStride(1);
  int ldc = C.getStride(0);

  for (int m = 0; m < A.getShape(0); ++m) {
    for (int n = 0; n < B.getShape(1); ++n) {
      for (int k = 0; k < A.getShape(1); ++k) {
        float va = dataA[stride0A * m + k * stride1A];
        float vb = dataB[stride0B * k + n * stride1B];
        dataC[ldc * m + n] += va * vb;
      }
    }
  }

  return C;
}

void testGEMM(int m, int k, int n, bool transa, bool transb) {
  Tensor A = transa ? fl::test::rand(Device::getCpu(), {k, m}, DType::kFloat) : fl::test::rand(Device::getCpu(), {m, k}, DType::kFloat);
  Tensor B = transb ? fl::test::rand(Device::getCpu(), {n, k}, DType::kFloat) : fl::test::rand(Device::getCpu(), {k, n}, DType::kFloat);

  if (transa) A = A.transpose(0, 1);
  if (transb) B = B.transpose(0, 1);

  Tensor C = fl::test::matmul(cpuOps(), A, B);
  Tensor C_ref = RefMatMulFp32(A, B);

  CATCH_REQUIRE(cpuOps()->allClose(C, C_ref));
}

void testGEMV(int M, int N, bool TransA) {
  Tensor A = TransA ? fl::test::rand(Device::getCpu(), {N, M}, DType::kFloat) : fl::test::rand(Device::getCpu(), {M, N}, DType::kFloat);
  Tensor x = fl::test::rand(Device::getCpu(), {N, 1}, DType::kFloat);

  if (TransA) A = A.transpose(0, 1);

  Tensor C = fl::test::matmul(cpuOps(), A, x);
  Tensor C_ref = RefMatMulFp32(A, x);

  CATCH_REQUIRE(cpuOps()->allClose(C, C_ref));
}

int gemmTestShapes[][3] = {
    {50, 50, 1},
    {1, 1, 1},
    {2, 2, 2},
    {50, 50, 1},
    {513, 2, 513},
    {200, 1, 300},
    {1, 200, 300},
    {200, 300, 1},
    {16, 16, 5000},
    {16, 5000, 16},
    {16, 512, 16},
    {16, 1024, 16},
    {5000, 16, 16},
    {0, 0, 0}};

int gemvTestShapes[][2] = {{2, 8}, {50, 10}, {1, 1}, {1024, 3}, {0, 0}};

}  // namespace

// The reference SGEMM/SGEMV path is only wired up for x86_64.
#if defined(LUT_CPU_ARCH) && LUT_CPU_ARCH == LUT_AMD64

CATCH_TEST_CASE("float32 GEMM BVT", "[core][nn][gemm]") {
  int (*pshape)[3];

  for (pshape = &gemmTestShapes[0]; **pshape != 0; ++pshape) {
    int m = (*pshape)[0];
    int k = (*pshape)[1];
    int n = (*pshape)[2];

    testGEMM(m, k, n, false, false);
    testGEMM(m, k, n, true, false);
    testGEMM(m, k, n, false, true);
  }
}

CATCH_TEST_CASE("float32 GEMV BVT", "[core][nn][gemv]") {
  int (*pshape)[2];

  for (pshape = &gemvTestShapes[0]; **pshape != 0; ++pshape) {
    int m = (*pshape)[0];
    int n = (*pshape)[1];

    testGEMV(m, n, false);
    testGEMV(m, n, true);
  }
}

#endif

/// A half matmul against the float32 one of the same, widened, operands. x64 has no half
/// arithmetic, so this is what says its half matmul is the float32 one rounded once: the 2-D GEMM,
/// an activation of higher rank against a weight, the GEMV shapes, and two batched operands.
CATCH_TEST_CASE("float16 matmul agrees with float32", "[core][nn][gemm]") {
  // batchA and batchB are the leading dimensions of each; B is held (n, k) and transposed when
  // transB is set, which is how a linear layer's weight is stored.
  auto check = [](std::vector<int> batchA, std::vector<int> batchB, int m, int k, int n,
                  bool transB) {
    std::vector<int> shapeA = batchA, shapeB = batchB;
    shapeA.insert(shapeA.end(), {m, k});
    if (transB) {
      shapeB.insert(shapeB.end(), {n, k});
    } else {
      shapeB.insert(shapeB.end(), {k, n});
    }

    // cast asks for a contiguous tensor, so B is transposed once it has been cast.
    Tensor hA = fl::test::rand(Device::getCpu(), shapeA, DType::kFloat);
    Tensor hB = fl::test::rand(Device::getCpu(), shapeB, DType::kFloat);
    hA = fl::test::cast(cpuOps(), hA, DType::kFloat16);
    hB = fl::test::cast(cpuOps(), hB, DType::kFloat16);
    Tensor A = fl::test::cast(cpuOps(), hA, DType::kFloat);
    Tensor B = fl::test::cast(cpuOps(), hB, DType::kFloat);
    if (transB) {
      hB = hB.transpose(-1, -2);
      B = B.transpose(-1, -2);
    }

    Tensor C = fl::test::cast(cpuOps(), fl::test::matmul(cpuOps(), hA, hB), DType::kFloat);
    Tensor C_ref = fl::test::matmul(cpuOps(), A, B);

    CATCH_REQUIRE(cpuOps()->allClose(C, C_ref, 1e-2, 1e-2));
  };

  for (bool transB : {false, true}) {
    CATCH_INFO("transB " << transB);
    check({}, {}, 50, 70, 30, transB);
    check({}, {}, 1, 300, 200, transB);
    check({}, {}, 200, 300, 1, transB);
    check({2, 17}, {}, 5, 64, 48, transB);
    check({2, 3}, {2, 3}, 17, 64, 48, transB);
  }
}

namespace {

/// Values of a float tensor on the CPU, in order.
std::vector<float> values(const Tensor &x) {
  Tensor c = fl::test::contiguous(cpuOps(), x);
  const float *data = c.getInternalData()->getData<float>(c.getInternalOffset());
  return std::vector<float>(data, data + c.getNumEl());
}

bool allFinite(const Tensor &x) {
  std::vector<float> got = values(x);
  return std::all_of(got.begin(), got.end(), [](float y) { return std::isfinite(y); });
}

/// `shape` at random, with every element in row 0 (`column` false) or column 0 (`column` true) of
/// each matrix set to `big`.
Tensor loud(std::vector<int> shape, bool column, float big) {
  Tensor x = fl::test::rand(Device::getCpu(), shape, DType::kFloat);
  int rows = shape[shape.size() - 2], cols = shape.back();
  int64_t matrices = x.getNumEl() / (static_cast<int64_t>(rows) * cols);
  float *data = x.getInternalData()->getData<float>(x.getInternalOffset());
  for (int64_t m = 0; m < matrices; ++m) {
    for (int r = 0; r < rows; ++r) {
      for (int c = 0; c < cols; ++c) {
        if ((column ? c : r) == 0) data[(m * rows + r) * cols + c] = big;
      }
    }
  }
  return x;
}

}  // namespace

CATCH_TEST_CASE("float16 matmul kept in float", "[core][nn][gemm]") {
  CATCH_REQUIRE(cpuOps()->matmulWidensHalf());

  // Half times half into float, as the composed attention takes its scores: the sum as it was,
  // past what half holds. A's first column is 300 and B's first row 1500, so every product is
  // 450,000 and more; the rest is random. K is 75, not a whole number of any kernel's steps. B as
  // it is, and along K -- the transpose of an (N, K), as the keys are -- each as one matrix, as a
  // batch of (2, 3) the way attention hands them over, and as a single row and a single column,
  // which are GEMVs.
  auto check = [](std::vector<int> batch, int m, int n, bool alongK) {
    std::vector<int> shapeA = batch, shapeB = batch;
    shapeA.insert(shapeA.end(), {m, 75});
    if (alongK) {
      shapeB.insert(shapeB.end(), {n, 75});
    } else {
      shapeB.insert(shapeB.end(), {75, n});
    }
    Tensor ha = fl::test::cast(cpuOps(), loud(shapeA, true, 300.0f), DType::kFloat16);
    Tensor hb = fl::test::cast(cpuOps(), loud(shapeB, alongK, 1500.0f), DType::kFloat16);
    if (alongK) hb = hb.transpose(-2, -1);

    // The same operands, rounded to half, multiplied in float.
    Tensor reference = fl::test::matmul(
        cpuOps(),
        fl::test::cast(cpuOps(), ha, DType::kFloat),
        fl::test::cast(cpuOps(), fl::test::contiguous(cpuOps(), hb), DType::kFloat));

    std::vector<int> shape = ha.getShape();
    shape.back() = hb.getShape(-1);
    Tensor x = fl::test::zeros(Device::getCpu(), shape, DType::kFloat);
    cpuOps()->matmul(ha, hb, x);
    CATCH_REQUIRE(allFinite(x));
    CATCH_REQUIRE(cpuOps()->allClose(x, reference, 1e-4f, 1e-4f));
  };

  for (bool alongK : {false, true}) {
    CATCH_INFO("B along K " << alongK);
    check({}, 70, 33, alongK);
    check({2, 3}, 70, 33, alongK);
    check({}, 1, 33, alongK);
    check({}, 70, 1, alongK);
  }
}

CATCH_TEST_CASE("float16 attention keeps its scores in float", "[core][nn][gemm]") {
  // CosyVoice3's DiT: a channel or two every query and key carry large, so that every score is
  // past what half holds -- 97,000 there -- while the softmax, which a constant added to a row does
  // not move, is decided by the rest. Here the last two channels are 300 in the queries and 1500 in
  // the keys, both exact in half: scores of 112,500 with a head of 64. The CPU has no attention
  // kernel, so this is the composition; in half its scores were infinities and their softmax NaN.
  // The answer is checked against the same composition in float, on the operands rounded to half.
  auto sunk = [](int length, int headDim, float big) {
    Tensor x = fl::test::rand(Device::getCpu(), {1, 2, length, headDim}, DType::kFloat);
    float *data = x.getInternalData()->getData<float>(x.getInternalOffset());
    for (int row = 0; row < 2 * length; ++row) {
      data[row * headDim + headDim - 1] = big;
      data[row * headDim + headDim - 2] = big;
    }
    return fl::test::cast(cpuOps(), x, DType::kFloat16);
  };
  auto widened = [](const Tensor &x) { return fl::test::cast(cpuOps(), x, DType::kFloat); };

  // A head of 512 as a VAE's, whose keys carry 4500 -- exact in half too -- for scores past half
  // there as well.
  for (int headDim : {64, 512}) {
    float big = headDim == 512 ? 4500.0f : 1500.0f;
    for (MaskType mask : {MaskType::kNone, MaskType::kCausal}) {
      CATCH_INFO("headDim " << headDim << ", mask " << static_cast<int>(mask));
      Tensor q = sunk(40, headDim, 300.0f);
      Tensor k = sunk(40, headDim, big);
      Tensor v = fl::test::rand(Device::getCpu(), {1, 2, 40, headDim}, DType::kFloat);
      v = fl::test::cast(cpuOps(), v, DType::kFloat16);
      Tensor x = widened(fl::test::attention(cpuOps(), q, k, v, mask));
      Tensor reference = fl::test::attention(cpuOps(), widened(q), widened(k), widened(v), mask);
      CATCH_REQUIRE(allFinite(x));
      CATCH_REQUIRE(cpuOps()->allClose(x, reference, 1e-2f, 1e-2f));
    }
  }
}

}  // namespace cpu
}  // namespace op
}  // namespace fl
