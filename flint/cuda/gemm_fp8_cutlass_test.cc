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
#include <random>
#include <vector>

#include "catch2/catch_amalgamated.hpp"
#include "lutil/error.h"
#include "lutil/span.h"
#include "flint/test_functional.h"
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
  return F::cast(cudaOps(), F::toDevice(cudaOps(), a, Device::getCuda()), DType::kFloat16);
}

Tensor toCpuFloat(const Tensor &a) {
  return F::toDevice(cudaOps(), F::cast(cudaOps(), a, DType::kFloat), Device::getCpu());
}

/// allClose compares magnitudes, and a NaN is not larger than anything, so it slips through the
/// maximum. A scale of zero is exactly the case that would produce one.
bool allFinite(const Tensor &x) {
  Tensor a = toCpuFloat(x);
  const float *data = a.getInternalData()->getData<float>(a.getInternalOffset());
  for (int64_t i = 0; i < a.getNumEl(); ++i) {
    if (!std::isfinite(data[i])) return false;
  }
  return true;
}

/// allClose measures the largest difference against the mean magnitude, which says nothing useful
/// when one element of the result is thousands of times larger than the rest. This weighs every
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

/// The value an E4M3 code stands for, worked out here rather than by anything the kernel shares.
float decodeE4M3(uint8_t code) {
  int sign = code >> 7;
  int exponent = (code >> 3) & 0xf;
  int mantissa = code & 0x7;
  float magnitude = exponent == 0 ? ldexpf(static_cast<float>(mantissa), -9)
                                  : ldexpf(1.0f + mantissa / 8.0f, exponent - 7);
  return sign ? -magnitude : magnitude;
}

/// A weight's E4M3 codes on the card, and what they stand for on the host.
struct Fp8Weight {
  Tensor codes;
  Tensor values;  // <float>(n, k) on the CPU, unscaled
};

Fp8Weight makeWeight(int n, int k, const std::vector<uint8_t> &bytes) {
  Tensor codes = F::empty(Device::getCpu(), {n, k}, DType::kFp8E4M3);
  Tensor values = F::empty(Device::getCpu(), {n, k}, DType::kFloat);
  Fp8E4M3 *code = codes.getInternalData()->getData<Fp8E4M3>(0);
  float *value = values.getInternalData()->getData<float>(0);
  for (size_t i = 0; i < bytes.size(); ++i) {
    code[i].v = bytes[i];
    value[i] = decodeE4M3(bytes[i]);
  }
  return {F::toDevice(cudaOps(), codes, Device::getCuda()), values};
}

/// Codes from all of E4M3 but its two NaNs, which a package never holds.
Fp8Weight randomWeight(int n, int k) {
  static std::mt19937 rng(7);
  std::vector<uint8_t> bytes(static_cast<size_t>(n) * k);
  for (uint8_t &x : bytes) {
    x = static_cast<uint8_t>(rng() % 256);
    if ((x & 0x7f) == 0x7f) x = 0x3c;
  }
  return makeWeight(n, k, bytes);
}

/// <float>(1) on the device.
Tensor cudaScale(float value) {
  return F::toDevice(cudaOps(), Tensor::create<float>({1}, {value}), Device::getCuda());
}

/// What `A` (..., k) times a weight of `rows` rows is written into.
Tensor gemmResult(const Tensor &A, int rows) {
  std::vector<int> shape = A.getShape();
  shape.back() = rows;
  return F::empty(Device::getCuda(), shape, DType::kFloat16);
}

Tensor gemm(const Tensor &A, const Fp8Weight &weight, const Tensor &scale) {
  Tensor D = gemmResult(A, weight.codes.getShape(0));
  op::cuda::gemmFp8(A, weight.codes, scale, D);
  return D;
}

/// The weight as a half GEMM sees it: the codes' values times the scale, transposed.
Tensor dequantized(const Fp8Weight &weight, float scale) {
  return toCudaHalf(F::mul(cpuOps(), weight.values, scale)).transpose(0, 1);
}

bool skipUnavailable() {
  if (!isOperatorsAvailable(Device::kCuda)) return true;
  return !op::cuda::isFp8GemmAvailable();
}

// Not a power of two, so a scale that went missing or was applied twice cannot come out right.
constexpr float kScale = 0.0137f;

/// The reference multiplies in half by the weight the codes stand for, so what is compared is the
/// mainloop and the epilogue.
bool gemmMatchesReference(int m, int n, int k) {
  Tensor a = toCudaHalf(F::randNormal(Device::getCpu(), {m, k}));
  Fp8Weight w = randomWeight(n, k);

  Tensor expected = F::matmul(cudaOps(), a, dequantized(w, kScale));
  Tensor actual = gemm(a, w, cudaScale(kScale));

  if (actual.getShape() != std::vector<int>{m, n}) return false;
  if (!allFinite(actual)) return false;

  return relativeRmse(actual, expected) < 5e-3;
}

}  // namespace

CATCH_TEST_CASE("test gemmFp8 (shapes)", "[fl][op][cuda][cutlass][fp8]") {
  if (skipUnavailable()) CATCH_SKIP("no cuda device available");

  auto runCase = [](int m, int n, int k) {
    CATCH_INFO("m = " << m << ", n = " << n << ", k = " << k);
    return gemmMatchesReference(m, n, k);
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

  Fp8Weight w = randomWeight(64, 128);
  Tensor scale = cudaScale(0.5f);

  Tensor a3 = toCudaHalf(F::randNormal(Device::getCpu(), {2, 3, 128}));
  Tensor out3 = gemm(a3, w, scale);
  CATCH_REQUIRE(out3.getShape() == std::vector<int>{2, 3, 64});
  CATCH_REQUIRE(cpuOps()->allClose(
      toCpuFloat(out3.view({-1, 64})),
      toCpuFloat(gemm(a3.view({-1, 128}), w, scale)),
      1e-6f,
      1e-6f));

  Tensor a4 = toCudaHalf(F::randNormal(Device::getCpu(), {2, 3, 5, 128}));
  CATCH_REQUIRE(gemm(a4, w, scale).getShape() == std::vector<int>{2, 3, 5, 64});
}

CATCH_TEST_CASE("test gemmFp8 (reused weight)", "[fl][op][cuda][cutlass][fp8]") {
  if (skipUnavailable()) CATCH_SKIP("no cuda device available");

  // A weight is read once at load and multiplied for the rest of the process, so the operand has
  // to survive being used again, and by a different row count than the first time.
  Fp8Weight w = randomWeight(128, 256);
  Tensor scale = cudaScale(kScale);
  Tensor reference = dequantized(w, kScale);

  for (int m : {1, 7, 64}) {
    CATCH_INFO("m = " << m);
    Tensor a = toCudaHalf(F::randNormal(Device::getCpu(), {m, 256}));
    CATCH_REQUIRE(relativeRmse(gemm(a, w, scale), F::matmul(cudaOps(), a, reference)) < 5e-3);
  }
}

CATCH_TEST_CASE("test gemmFp8 (scale applied)", "[fl][op][cuda][cutlass][fp8]") {
  if (skipUnavailable()) CATCH_SKIP("no cuda device available");

  // The same thirty-two codes in every row against an activation of ones, so each column of the
  // result is the row's code sum times the scale, and the only error left is the half it is
  // written in. Two scales four orders of magnitude apart over the same codes: an epilogue that
  // ignored the scale would get at most one of them right, and a scale of zero has to give zeros
  // rather than NaN.
  const uint8_t row[32] = {0x00, 0x7e, 0xfe, 0x38, 0xb8, 0x3c, 0x40, 0x44, 0x48, 0x4c, 0x50,
                           0x54, 0x58, 0x5c, 0x60, 0x64, 0x68, 0x6c, 0x70, 0x74, 0x78, 0x7c,
                           0x7d, 0x7a, 0xf8, 0xb0, 0x30, 0x28, 0x20, 0xec, 0x76, 0xc4};
  std::vector<uint8_t> bytes;
  for (int r = 0; r < 8; ++r) bytes.insert(bytes.end(), row, row + 32);
  Fp8Weight w = makeWeight(8, 32, bytes);

  float rowSum = 0.0f;
  for (uint8_t code : row) rowSum += decodeE4M3(code);
  CATCH_REQUIRE(rowSum != 0.0f);

  std::vector<float> ones(4 * 32, 1.0f);
  Tensor a = toCudaHalf(Tensor::create<float>({4, 32}, lut::makeConstSpan(ones)));

  for (float scale : {0.001953125f, 3.0f, 0.0f}) {
    CATCH_INFO("scale = " << scale);
    Tensor out = gemm(a, w, cudaScale(scale));
    CATCH_REQUIRE(allFinite(out));

    Tensor actual = toCpuFloat(out);
    const float *got = actual.getInternalData()->getData<float>(actual.getInternalOffset());
    float expected = rowSum * scale;
    for (int i = 0; i < 4 * 8; ++i) {
      CATCH_INFO("element " << i << ": " << got[i] << " against " << expected);
      CATCH_REQUIRE(std::fabs(got[i] - expected) <= std::fabs(expected) * 1e-3f);
    }
  }
}

}  // namespace fl
