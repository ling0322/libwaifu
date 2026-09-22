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
#include "lutil/error.h"
#include "lutil/span.h"
#include "flint/cuda/fp8.h"
#include "flint/fp8.h"
#include "flint/cuda/gemm_fp8_cutlass.h"
#include "flint/device.h"
#include "flint/operators.h"
#include "flint/tensor.h"

namespace fl {

namespace {

/// The CUDA operators, which the calls here that run on CUDA are asked of.
Operators *cudaOps() {
  return getOperators(Device::kCuda);
}

/// The CPU operators, which the calls here that run on CPU are asked of.
Operators *cpuOps() {
  return getOperators(Device::kCpu);
}

Tensor toCudaHalf(const Tensor &a) {
  return cudaOps()->cast(cudaOps()->toDevice(Device::getCuda(), a), DType::kFloat16);
}

Tensor toCpuFloat(const Tensor &a) {
  return cudaOps()->toDevice(Device::getCpu(), cudaOps()->cast(a, DType::kFloat));
}

/// allClose compares magnitudes, and a NaN is not larger than anything, so it slips through the
/// maximum. An all zero row is exactly the case that would produce one.
bool allFinite(const Tensor &x) {
  Tensor a = toCpuFloat(x);
  const float *data = a.getInternalData()->getData<float>(a.getInternalOffset());
  for (int64_t i = 0; i < a.getNumEl(); ++i) {
    if (!std::isfinite(data[i])) return false;
  }
  return true;
}

/// allClose measures the largest difference against the mean magnitude, which says nothing useful
/// when one row of the result is thousands of times larger than the rest. This weighs every
/// element by its own size instead.
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

/// Thirty-two magnitudes E4M3 holds exactly, the largest of them 448. A row of these has a scale
/// of exactly one, so quantizing it is a round trip rather than an approximation, and anything
/// the quantizer gets wrong -- the scale arithmetic, the element order, the byte packing -- turns
/// it into a wrong answer rather than into slightly more error.
std::vector<float> exactE4m3Row(float rowScale) {
  const float codes[32] = {0.0f,    448.0f,  -448.0f, 1.0f,   -1.0f,  1.5f,   2.0f,   3.0f,
                           4.0f,    6.0f,    8.0f,    12.0f,  16.0f,  24.0f,  32.0f,  48.0f,
                           64.0f,   96.0f,   128.0f,  192.0f, 256.0f, 384.0f, 416.0f, 320.0f,
                           -256.0f, -0.5f,   0.5f,    0.25f,  0.125f, -96.0f, 224.0f, -3.0f};

  std::vector<float> row;
  for (float code : codes) {
    row.push_back(code * rowScale);
  }
  return row;
}

bool skipUnavailable() {
  if (!isOperatorsAvailable(Device::kCuda)) return true;
  return !op::cuda::isFp8GemmAvailable();
}

/// The reference dequantizes the weight and multiplies in half, so the quantization error sits on
/// the weight in both and what is compared is the mainloop and the epilogue.
bool gemmMatchesReference(const Tensor &a, const Tensor &w) {
  Tensor ha = toCudaHalf(a);
  Fp8Operand qw = op::cuda::quantizeFp8(toCudaHalf(w));

  Tensor expected = cudaOps()->matmul(ha, op::cuda::dequantFp8ToHalf(qw).transpose(0, 1));
  Tensor actual = op::cuda::gemmFp8(ha, qw);

  if (actual.getShape() != std::vector<int>{a.getShape(0), w.getShape(0)}) return false;
  if (!allFinite(actual)) return false;

  return relativeRmse(actual, expected) < 5e-3;
}

}  // namespace

CATCH_TEST_CASE("test fp8 quantizer round trip", "[fl][op][cuda][cutlass][fp8]") {
  if (skipUnavailable()) CATCH_SKIP("no cuda device available");

  // Four rows, each a different power of two away from the next, so the per channel scale is the
  // only thing that can tell them apart.
  std::vector<float> data;
  for (float rowScale : {1.0f, 8.0f, 0.03125f, 64.0f}) {
    std::vector<float> row = exactE4m3Row(rowScale);
    data.insert(data.end(), row.begin(), row.end());
  }
  Tensor w = Tensor::create<float>({4, 32}, lut::makeConstSpan(data));

  Fp8Operand q = op::cuda::quantizeFp8(toCudaHalf(w));
  CATCH_REQUIRE(q.data.getShape() == std::vector<int>{4, 32});
  CATCH_REQUIRE(q.data.getDType() == DType::kFp8E4M3);
  CATCH_REQUIRE(q.channelScale.getShape() == std::vector<int>{4});

  // rowAmax / 448, and every row's amax is 448 times the row's own scale.
  Tensor scale = Tensor::create<float>({4}, {1.0f, 8.0f, 0.03125f, 64.0f});
  CATCH_REQUIRE(cpuOps()->allClose(toCpuFloat(q.channelScale), scale, 1e-6f, 1e-6f));

  Tensor x = op::cuda::dequantFp8ToHalf(q);
  CATCH_REQUIRE(x.getShape() == std::vector<int>{4, 32});
  CATCH_REQUIRE(cpuOps()->allClose(toCpuFloat(x), w, 1e-6f, 1e-6f));
}

CATCH_TEST_CASE("test fp8 quantizer shapes", "[fl][op][cuda][cutlass][fp8]") {
  if (skipUnavailable()) CATCH_SKIP("no cuda device available");

  struct Case {
    int rows;
    int k;
  };
  // k of 16 is the shortest row the quantizer takes, and 3072 is longer than one block of threads
  // reaches in a single step.
  std::vector<Case> cases = {{1, 16}, {7, 32}, {128, 64}, {130, 96}, {3, 3072}};

  for (const Case &c : cases) {
    CATCH_INFO("rows = " << c.rows << ", k = " << c.k);
    Tensor w = cpuOps()->randNormal({c.rows, c.k});
    Fp8Operand q = op::cuda::quantizeFp8(toCudaHalf(w));

    CATCH_REQUIRE(q.data.getShape() == std::vector<int>{c.rows, c.k});
    CATCH_REQUIRE(q.channelScale.getShape() == std::vector<int>{c.rows});

    Tensor x = op::cuda::dequantFp8ToHalf(q);
    CATCH_REQUIRE(x.getShape() == std::vector<int>{c.rows, c.k});
    CATCH_REQUIRE(allFinite(x));

    // Three mantissa bits is a step of 1/16 at the top of a binade, so half of that is the worst
    // a rounded element can be off by relative to the row's largest.
    CATCH_REQUIRE(relativeRmse(x, toCudaHalf(w)) < 4e-2);
  }
}

CATCH_TEST_CASE("test fp8 quantizer zero row", "[fl][op][cuda][cutlass][fp8]") {
  if (skipUnavailable()) CATCH_SKIP("no cuda device available");

  // A row with nothing in it drives its own scale to zero, and the scale is a divisor on the way
  // in. This is the case that turns into NaN if the guard is missing.
  std::vector<float> data(2 * 32, 0.0f);
  for (int i = 32; i < 64; ++i) data[i] = 1.0f;

  Tensor w = Tensor::create<float>({2, 32}, lut::makeConstSpan(data));
  Fp8Operand q = op::cuda::quantizeFp8(toCudaHalf(w));

  Tensor x = op::cuda::dequantFp8ToHalf(q);
  CATCH_REQUIRE(allFinite(x));
  CATCH_REQUIRE(cpuOps()->allClose(toCpuFloat(x), w, 1e-6f, 1e-6f));

  // And a GEMM against it comes back as zeros rather than as NaN.
  Tensor a = toCudaHalf(cpuOps()->randNormal({8, 32}));
  Fp8Operand qZero = op::cuda::quantizeFp8(toCudaHalf(cpuOps()->zeros({8, 32}, DType::kFloat)));
  Tensor out = op::cuda::gemmFp8(a, qZero);
  CATCH_REQUIRE(allFinite(out));
  CATCH_REQUIRE(
      cpuOps()->allClose(toCpuFloat(out), cpuOps()->zeros({8, 8}, DType::kFloat), 1e-6f, 1e-6f));
}

CATCH_TEST_CASE("test fp8 quantizer dynamic range", "[fl][op][cuda][cutlass][fp8]") {
  if (skipUnavailable()) CATCH_SKIP("no cuda device available");

  // One channel thousands of times larger than the rest is the case a per tensor scale cannot
  // hold: with one scale for everything the small channels would quantize to zero. Each row gets
  // its own, so each row keeps its own signal.
  std::vector<float> data(4 * 64);
  for (int r = 0; r < 4; ++r) {
    float rowScale = r == 0 ? 4096.0f : 1.0e-3f;
    for (int i = 0; i < 64; ++i) {
      data[r * 64 + i] = rowScale * ((i % 5) - 2) / 2.0f;
    }
  }

  Tensor w = Tensor::create<float>({4, 64}, lut::makeConstSpan(data));
  Fp8Operand q = op::cuda::quantizeFp8(toCudaHalf(w));
  Tensor x = op::cuda::dequantFp8ToHalf(q);

  CATCH_REQUIRE(allFinite(x));
  for (int r = 0; r < 4; ++r) {
    CATCH_INFO("row = " << r);
    CATCH_REQUIRE(relativeRmse(x.slice(0, {r, r + 1}), toCudaHalf(w).slice(0, {r, r + 1})) < 4e-2);
  }
}

CATCH_TEST_CASE("test gemmFp8 (shapes)", "[fl][op][cuda][cutlass][fp8]") {
  if (skipUnavailable()) CATCH_SKIP("no cuda device available");

  auto runCase = [](int m, int n, int k) {
    CATCH_INFO("m = " << m << ", n = " << n << ", k = " << k);
    return gemmMatchesReference(cpuOps()->randNormal({m, k}), cpuOps()->randNormal({n, k}));
  };

  // One whole tile, and the tile shape is 128x128x64.
  CATCH_REQUIRE(runCase(128, 128, 64));
  // A single row is the decode step, and the case the 128 row tile pads the most.
  CATCH_REQUIRE(runCase(1, 128, 64));
  CATCH_REQUIRE(runCase(1, 5120, 3072));
  // The smallest operand the kernel takes: eight output channels and a k of sixteen.
  CATCH_REQUIRE(runCase(1, 8, 16));
  CATCH_REQUIRE(runCase(2, 16, 16));
  // Residues on each axis separately, then both at once.
  CATCH_REQUIRE(runCase(17, 128, 64));
  CATCH_REQUIRE(runCase(128, 264, 64));
  CATCH_REQUIRE(runCase(129, 264, 96));
  // k that leaves a partial 64 deep tile, and k below one tile.
  CATCH_REQUIRE(runCase(64, 64, 80));
  CATCH_REQUIRE(runCase(3, 8, 32));
  // More than one tile on both axes.
  CATCH_REQUIRE(runCase(300, 200, 256));
  // A thin operand against a long k, which is the lm_head shape in miniature.
  CATCH_REQUIRE(runCase(1, 8, 4096));
  // Either side of the row count that picks the other tile.
  CATCH_REQUIRE(runCase(512, 64, 64));
  CATCH_REQUIRE(runCase(513, 64, 64));
}

CATCH_TEST_CASE("test gemmFp8 (batch axes)", "[fl][op][cuda][cutlass][fp8]") {
  if (skipUnavailable()) CATCH_SKIP("no cuda device available");

  Fp8Operand qw = op::cuda::quantizeFp8(toCudaHalf(cpuOps()->randNormal({64, 128})));

  Tensor a3 = toCudaHalf(cpuOps()->randNormal({2, 3, 128}));
  Tensor out3 = op::cuda::gemmFp8(a3, qw);
  CATCH_REQUIRE(out3.getShape() == std::vector<int>{2, 3, 64});
  CATCH_REQUIRE(cpuOps()->allClose(
      toCpuFloat(out3.view({-1, 64})),
      toCpuFloat(op::cuda::gemmFp8(a3.view({-1, 128}), qw)),
      1e-6f,
      1e-6f));

  Tensor a4 = toCudaHalf(cpuOps()->randNormal({2, 3, 5, 128}));
  CATCH_REQUIRE(op::cuda::gemmFp8(a4, qw).getShape() == std::vector<int>{2, 3, 5, 64});
}

CATCH_TEST_CASE("test gemmFp8 (reused weight)", "[fl][op][cuda][cutlass][fp8]") {
  if (skipUnavailable()) CATCH_SKIP("no cuda device available");

  // A weight is quantized once at load and multiplied for the rest of the process, so the operand
  // has to survive being used again, and by a different row count than the first time.
  Fp8Operand qw = op::cuda::quantizeFp8(toCudaHalf(cpuOps()->randNormal({128, 256})));
  Tensor reference = op::cuda::dequantFp8ToHalf(qw).transpose(0, 1);

  for (int m : {1, 7, 64}) {
    CATCH_INFO("m = " << m);
    Tensor a = toCudaHalf(cpuOps()->randNormal({m, 256}));
    CATCH_REQUIRE(relativeRmse(op::cuda::gemmFp8(a, qw), cudaOps()->matmul(a, reference)) < 5e-3);
  }
}

CATCH_TEST_CASE("test gemmFp8 (channel scale per column)", "[fl][op][cuda][cutlass][fp8]") {
  if (skipUnavailable()) CATCH_SKIP("no cuda device available");

  // A weight whose channels differ by four orders of magnitude, against an activation of ones:
  // each column of the result is then that channel's own row sum, so a scale applied to the wrong
  // axis -- or to the tile rather than to the channel -- cannot look right by accident.
  std::vector<float> data(16 * 16);
  for (int r = 0; r < 16; ++r) {
    for (int i = 0; i < 16; ++i) {
      data[r * 16 + i] = std::pow(10.0f, float(r % 5) - 2.0f);
    }
  }

  Tensor w = Tensor::create<float>({16, 16}, lut::makeConstSpan(data));
  Fp8Operand qw = op::cuda::quantizeFp8(toCudaHalf(w));

  std::vector<float> ones(4 * 16, 1.0f);
  Tensor a = toCudaHalf(Tensor::create<float>({4, 16}, lut::makeConstSpan(ones)));
  Tensor out = op::cuda::gemmFp8(a, qw);

  std::vector<float> expected(4 * 16);
  for (int m = 0; m < 4; ++m) {
    for (int r = 0; r < 16; ++r) {
      expected[m * 16 + r] = 16.0f * std::pow(10.0f, float(r % 5) - 2.0f);
    }
  }

  // Element by element rather than through relativeRmse, which weighs by magnitude and so would
  // not notice the channels four orders of magnitude below the largest being wrong. Each row here
  // is a constant, so quantizing it is exact -- every element is the row's own maximum -- and the
  // only error left is the half the result is written in.
  CATCH_REQUIRE(allFinite(out));
  Tensor actual = toCpuFloat(out);
  const float *got = actual.getInternalData()->getData<float>(actual.getInternalOffset());
  for (int i = 0; i < 4 * 16; ++i) {
    CATCH_INFO("element " << i << ": " << got[i] << " against " << expected[i]);
    CATCH_REQUIRE(std::fabs(got[i] - expected[i]) <= std::fabs(expected[i]) * 1e-3f);
  }
}

CATCH_TEST_CASE("test makeFp8Operand rejects what it cannot use", "[fl][op][cuda][cutlass][fp8]") {
  if (skipUnavailable()) CATCH_SKIP("no cuda device available");

  Fp8Operand q = op::cuda::quantizeFp8(toCudaHalf(cpuOps()->randNormal({8, 32})));

  CATCH_REQUIRE_NOTHROW(makeFp8Operand(q.data, q.channelScale));
  // The data where the scale belongs.
  CATCH_REQUIRE_THROWS_AS(makeFp8Operand(q.channelScale, q.channelScale), lut::Error);
  // One scale for a weight that has eight channels.
  CATCH_REQUIRE_THROWS_AS(
      makeFp8Operand(q.data, q.channelScale.slice(0, {0, 1})),
      lut::Error);
  // A host side weight, which would otherwise reach the kernel as an address it may not touch.
  CATCH_REQUIRE_THROWS_AS(
      makeFp8Operand(q.data, cudaOps()->toDevice(Device::getCpu(), q.channelScale)),
      lut::Error);
}

}  // namespace fl
