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

#include <cmath>
#include <vector>

#include "catch2/catch_amalgamated.hpp"
#include "lutil/span.h"
#include "flint/cuda/gemm_nvfp4_cutlass.h"
#include "flint/cuda/nvfp4.h"
#include "flint/device.h"
#include "flint/functional.h"
#include "flint/operators.h"
#include "flint/tensor.h"

namespace fl {
namespace {

Tensor toCudaHalf(const Tensor &a) {
  return F::cast(F::to(Device::getCuda(), a), DType::kFloat16);
}

Tensor toCpuFloat(const Tensor &a) {
  return F::to(Device::getCpu(), F::cast(a, DType::kFloat));
}

/// allClose compares magnitudes, and a NaN is not larger than anything, so it slips through the
/// maximum. Saturation and an all zero operand are exactly the cases that would produce one.
bool allFinite(const Tensor &x) {
  Tensor a = toCpuFloat(x);
  const float *data = a.getInternalData()->getData<float>(a.getInternalOffset());
  for (int64_t i = 0; i < a.getNumEl(); ++i) {
    if (!std::isfinite(data[i])) return false;
  }
  return true;
}

/// allClose measures the largest difference against the mean magnitude, which says nothing useful
/// when one row of the result is thousands of times larger than the rest: a single half ULP on
/// that row swamps the mean. This weighs every element by its own size instead.
double relativeRmse(const Tensor &x, const Tensor &reference) {
  Tensor a = toCpuFloat(x);
  Tensor b = toCpuFloat(reference);
  const float *pa = a.getInternalData()->getData<float>(a.getInternalOffset());
  const float *pb = b.getInternalData()->getData<float>(b.getInternalOffset());

  double squaredError = 0.0;
  double squaredReference = 0.0;
  for (int64_t i = 0; i < a.getNumEl(); ++i) {
    double diff = double(pa[i]) - double(pb[i]);
    squaredError += diff * diff;
    squaredReference += double(pb[i]) * double(pb[i]);
  }

  return std::sqrt(squaredError / squaredReference);
}

/// Data every step of the quantization holds exactly: each block is the full set of E2M1 codes
/// times a scale E4M3 holds exactly, and the largest block reaches 6 * 448, which is what makes
/// the global scale a power of two. Anything the prologue gets wrong -- nibble order, block size,
/// the interleaved scale offset, the scale arithmetic -- turns an exact round trip into a wrong
/// one rather than into slightly more error.
std::vector<float> exactNvfp4Data(const std::vector<float> &blockScales, float tensorScale) {
  const float codes[16] = {
      0.0f,
      0.5f,
      1.0f,
      1.5f,
      2.0f,
      3.0f,
      4.0f,
      6.0f,
      -0.5f,
      -1.0f,
      -1.5f,
      -2.0f,
      -3.0f,
      -4.0f,
      -6.0f,
      -0.5f};

  std::vector<float> data;
  for (float scale : blockScales) {
    for (float code : codes) {
      data.push_back(code * scale * tensorScale);
    }
  }
  return data;
}

bool skipUnavailable() {
  if (!isOperatorsAvailable(Device::kCuda)) return true;
  return !op::cuda::isNvfp4GemmAvailable();
}

/// The reference dequantizes both operands and multiplies them in half, so the quantization error
/// sits on both sides and what is compared is the mainloop and the epilogue.
bool gemmMatchesReference(const Tensor &a, const Tensor &w) {
  op::cuda::Nvfp4Operand qa = op::cuda::quantizeNvfp4(toCudaHalf(a));
  op::cuda::Nvfp4Operand qw = op::cuda::quantizeNvfp4(toCudaHalf(w));

  Tensor expected = F::matmul(
      op::cuda::dequantNvfp4ToHalf(qa),
      op::cuda::dequantNvfp4ToHalf(qw).transpose(0, 1));
  Tensor actual = op::cuda::gemmNvfp4(qa, qw);

  if (actual.getShape() != std::vector<int>{a.getShape(0), w.getShape(0)}) return false;
  if (!allFinite(actual)) return false;

  return F::allClose(toCpuFloat(actual), toCpuFloat(expected), 1e-2f);
}

}  // namespace

CATCH_TEST_CASE("test nvfp4 prologue round trip", "[fl][op][cuda][cutlass][nvfp4]") {
  if (skipUnavailable()) CATCH_SKIP("no sm_120 cuda device available");

  std::vector<float> blockScales = {448.0f, 1.0f, 0.25f, 8.0f, 2.0f, 0.5f, 64.0f, 4.0f};
  std::vector<float> data = exactNvfp4Data(blockScales, 1.0f);
  Tensor w = Tensor::create<float>({4, 32}, lut::makeConstSpan(data));

  op::cuda::Nvfp4Operand q = op::cuda::quantizeNvfp4(toCudaHalf(w));

  // The scale array is padded out to the atom, which covers 128 rows and 4 scale blocks, so 4
  // rows of 2 blocks still take 128 * 4 bytes.
  CATCH_REQUIRE(q.data.getShape() == std::vector<int>{4, 16});
  CATCH_REQUIRE(q.blockScale.getShape() == std::vector<int>{512});
  CATCH_REQUIRE(q.globalScale.getShape() == std::vector<int>{1});

  Tensor x = op::cuda::dequantNvfp4ToHalf(q);
  CATCH_REQUIRE(x.getShape() == std::vector<int>{4, 32});
  CATCH_REQUIRE(F::allClose(toCpuFloat(x), w, 1e-6f, 1e-6f));
}

CATCH_TEST_CASE("test nvfp4 global scale", "[fl][op][cuda][cutlass][nvfp4]") {
  if (skipUnavailable()) CATCH_SKIP("no sm_120 cuda device available");

  // The same data scaled by 16. E4M3 tops out at 448, so a tensor whose blocks reach beyond that
  // only stays representable because the global scale divides it back down first.
  std::vector<float> blockScales = {448.0f, 1.0f, 0.25f, 8.0f, 2.0f, 0.5f, 64.0f, 4.0f};
  std::vector<float> data = exactNvfp4Data(blockScales, 16.0f);
  Tensor w = Tensor::create<float>({4, 32}, lut::makeConstSpan(data));

  op::cuda::Nvfp4Operand q = op::cuda::quantizeNvfp4(toCudaHalf(w));

  // amax / (6 * 448) with an amax of 6 * 448 * 16.
  Tensor globalScale = Tensor::create<float>({1}, {16.0f});
  CATCH_REQUIRE(F::allClose(toCpuFloat(q.globalScale), globalScale, 1e-6f, 1e-6f));

  Tensor x = op::cuda::dequantNvfp4ToHalf(q);
  CATCH_REQUIRE(F::allClose(toCpuFloat(x), w, 1e-6f, 1e-6f));
}

CATCH_TEST_CASE("test nvfp4 prologue scale array padding", "[fl][op][cuda][cutlass][nvfp4]") {
  if (skipUnavailable()) CATCH_SKIP("no sm_120 cuda device available");

  // Rows round up to 128 and scale blocks to 4, whatever the operand's own extent is.
  struct Case {
    int rows;
    int k;
    int scaleByte;
  };
  std::vector<Case> cases = {{1, 32, 128 * 4}, {128, 64, 128 * 4}, {130, 96, 256 * 8},
                             {129, 256, 256 * 16}, {256, 32, 256 * 4}};

  for (const Case &c : cases) {
    CATCH_INFO("rows = " << c.rows << ", k = " << c.k);
    op::cuda::Nvfp4Operand q = op::cuda::quantizeNvfp4(toCudaHalf(F::randn({c.rows, c.k})));

    CATCH_REQUIRE(q.data.getShape() == std::vector<int>{c.rows, c.k / 2});
    CATCH_REQUIRE(q.blockScale.getShape() == std::vector<int>{c.scaleByte});
    CATCH_REQUIRE(op::cuda::dequantNvfp4ToHalf(q).getShape() == std::vector<int>{c.rows, c.k});
  }
}

CATCH_TEST_CASE("test gemmNvfp4 (shapes)", "[fl][op][cuda][cutlass][nvfp4]") {
  if (skipUnavailable()) CATCH_SKIP("no sm_120 cuda device available");

  auto runCase = [](int m, int n, int k) {
    CATCH_INFO("m = " << m << ", n = " << n << ", k = " << k);
    return gemmMatchesReference(F::randn({m, k}), F::randn({n, k}));
  };

  // One whole tile, and the tile shape is 128x128x128.
  CATCH_REQUIRE(runCase(128, 128, 128));
  // A single row is the decode step, and the case the 128 row tile pads the most.
  CATCH_REQUIRE(runCase(1, 128, 128));
  CATCH_REQUIRE(runCase(1, 5120, 3072));
  // The smallest operand the prologue accepts: k of 32 is two scale blocks, and the scale atom is
  // four blocks wide, so even the scale array is mostly padding.
  CATCH_REQUIRE(runCase(1, 8, 32));
  CATCH_REQUIRE(runCase(2, 16, 32));
  // Residues on each axis separately, then both at once.
  CATCH_REQUIRE(runCase(17, 128, 128));
  CATCH_REQUIRE(runCase(128, 264, 128));
  CATCH_REQUIRE(runCase(129, 264, 96));
  // k that leaves a partial 128 deep tile, and k below one tile.
  CATCH_REQUIRE(runCase(64, 64, 160));
  CATCH_REQUIRE(runCase(3, 8, 64));
  // More than one tile on both axes.
  CATCH_REQUIRE(runCase(300, 200, 256));
  // A thin operand against a long k, which is the lm_head shape in miniature.
  CATCH_REQUIRE(runCase(1, 8, 4096));
}

CATCH_TEST_CASE("test gemmNvfp4 (half activation)", "[fl][op][cuda][cutlass][nvfp4]") {
  if (skipUnavailable()) CATCH_SKIP("no sm_120 cuda device available");

  // What a projection calls: a half activation in, a weight quantized once, half back out. The
  // activation is quantized by the prologue inside the call.
  Tensor w = F::randn({64, 128});
  op::cuda::Nvfp4Operand qw = op::cuda::quantizeNvfp4(toCudaHalf(w));

  Tensor a2 = toCudaHalf(F::randn({6, 128}));
  Tensor expected = op::cuda::gemmNvfp4(op::cuda::quantizeNvfp4(a2), qw);
  Tensor actual = op::cuda::gemmNvfp4(a2, qw);
  CATCH_REQUIRE(actual.getShape() == std::vector<int>{6, 64});
  CATCH_REQUIRE(F::allClose(toCpuFloat(actual), toCpuFloat(expected), 1e-6f, 1e-6f));

  // Leading batch axes fold into the row count and come back on the result.
  Tensor a3 = toCudaHalf(F::randn({2, 3, 128}));
  Tensor out3 = op::cuda::gemmNvfp4(a3, qw);
  CATCH_REQUIRE(out3.getShape() == std::vector<int>{2, 3, 64});
  CATCH_REQUIRE(F::allClose(
      toCpuFloat(out3.view({-1, 64})),
      toCpuFloat(op::cuda::gemmNvfp4(op::cuda::quantizeNvfp4(a3.view({-1, 128})), qw)),
      1e-6f,
      1e-6f));

  Tensor a4 = toCudaHalf(F::randn({2, 3, 5, 128}));
  CATCH_REQUIRE(op::cuda::gemmNvfp4(a4, qw).getShape() == std::vector<int>{2, 3, 5, 64});
}

CATCH_TEST_CASE("test gemmNvfp4 (reused weight)", "[fl][op][cuda][cutlass][nvfp4]") {
  if (skipUnavailable()) CATCH_SKIP("no sm_120 cuda device available");

  // A weight is quantized once at load and multiplied for the rest of the process, so the operand
  // has to survive being used again, and by a different row count than the first time.
  Tensor w = F::randn({128, 256});
  op::cuda::Nvfp4Operand qw = op::cuda::quantizeNvfp4(toCudaHalf(w));
  Tensor reference = op::cuda::dequantNvfp4ToHalf(qw).transpose(0, 1);

  for (int m : {1, 7, 64}) {
    CATCH_INFO("m = " << m);
    Tensor a = toCudaHalf(F::randn({m, 256}));
    op::cuda::Nvfp4Operand qa = op::cuda::quantizeNvfp4(a);

    Tensor expected = F::matmul(op::cuda::dequantNvfp4ToHalf(qa), reference);
    CATCH_REQUIRE(F::allClose(
        toCpuFloat(op::cuda::gemmNvfp4(qa, qw)),
        toCpuFloat(expected),
        1e-2f));
  }
}

CATCH_TEST_CASE("test gemmNvfp4 (zero operand)", "[fl][op][cuda][cutlass][nvfp4]") {
  if (skipUnavailable()) CATCH_SKIP("no sm_120 cuda device available");

  // An all zero operand drives the tensor wide maximum to zero, and every scale with it. Both the
  // global scale and the per block one are divisors in the prologue, so this is the case that
  // turns into NaN if either guard is missing.
  Tensor zeros = F::zeros({64, 128}, DType::kFloat);
  Tensor w = F::randn({64, 128});

  op::cuda::Nvfp4Operand qZero = op::cuda::quantizeNvfp4(toCudaHalf(zeros));
  CATCH_REQUIRE(allFinite(op::cuda::dequantNvfp4ToHalf(qZero)));
  CATCH_REQUIRE(F::allClose(toCpuFloat(op::cuda::dequantNvfp4ToHalf(qZero)), zeros, 1e-6f, 1e-6f));

  op::cuda::Nvfp4Operand qw = op::cuda::quantizeNvfp4(toCudaHalf(w));
  Tensor out = op::cuda::gemmNvfp4(qZero, qw);
  CATCH_REQUIRE(allFinite(out));
  CATCH_REQUIRE(F::allClose(toCpuFloat(out), F::zeros({64, 64}, DType::kFloat), 1e-6f, 1e-6f));

  // And the other way round, where it is the weight that carries no signal.
  Tensor outW = op::cuda::gemmNvfp4(qw, qZero);
  CATCH_REQUIRE(allFinite(outW));
  CATCH_REQUIRE(F::allClose(toCpuFloat(outW), F::zeros({64, 64}, DType::kFloat), 1e-6f, 1e-6f));
}

CATCH_TEST_CASE("test gemmNvfp4 (dynamic range)", "[fl][op][cuda][cutlass][nvfp4]") {
  if (skipUnavailable()) CATCH_SKIP("no sm_120 cuda device available");

  // One outlier sets the tensor wide maximum and drags the global scale up with it, which pushes
  // every ordinary block towards the bottom of E4M3 and the smallest ones under it. Nothing here
  // may saturate to an infinity or divide its way to a NaN.
  std::vector<float> data(64 * 128, 0.0f);
  for (size_t i = 0; i < data.size(); ++i) {
    data[i] = (i % 3 == 0) ? 1.0e-4f : 1.0f;
  }
  data[0] = 4096.0f;

  Tensor a = Tensor::create<float>({64, 128}, lut::makeConstSpan(data));
  Tensor w = F::randn({64, 128});

  op::cuda::Nvfp4Operand qa = op::cuda::quantizeNvfp4(toCudaHalf(a));
  op::cuda::Nvfp4Operand qw = op::cuda::quantizeNvfp4(toCudaHalf(w));

  CATCH_REQUIRE(allFinite(op::cuda::dequantNvfp4ToHalf(qa)));

  Tensor out = op::cuda::gemmNvfp4(qa, qw);
  CATCH_REQUIRE(allFinite(out));

  Tensor expected = F::matmul(
      op::cuda::dequantNvfp4ToHalf(qa),
      op::cuda::dequantNvfp4ToHalf(qw).transpose(0, 1));
  CATCH_REQUIRE(relativeRmse(out, expected) < 1e-2);
}

}  // namespace fl
