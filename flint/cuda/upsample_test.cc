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

// The nearest-neighbour upsample a diffusion U-Net and its VAE decoder run before the convolution
// that follows them.

#include <algorithm>
#include <cmath>
#include <vector>

#include "catch2/catch_amalgamated.hpp"
#include "lutil/span.h"
#include "flint/device.h"
#include "flint/functional.h"
#include "flint/operators.h"
#include "flint/tensor.h"

namespace fl {
namespace {

Tensor cudaHalf(std::initializer_list<int> shape, const std::vector<float> &values) {
  return F::cast(
      F::toDevice(Device::getCuda(), Tensor::create<float>(shape, lut::makeConstSpan(values))),
      DType::kFloat16);
}

Tensor toCpuFloat(const Tensor &x) {
  return F::toDevice(Device::getCpu(), F::cast(x, DType::kFloat));
}

Tensor cpuFloat(std::initializer_list<int> shape, const std::vector<float> &values) {
  return Tensor::create<float>(shape, lut::makeConstSpan(values));
}

/// The same values left in float and moved to the device as they stand, which is what the
/// autoencoder hands this operator.
Tensor cudaFloat(std::initializer_list<int> shape, const std::vector<float> &values) {
  return F::toDevice(Device::getCuda(), Tensor::create<float>(shape, lut::makeConstSpan(values)));
}

/// An upsample only ever copies a pixel, so in float its result is exact. `F::allClose` compares
/// with a strict `<` and so cannot be asked for that; these compare the elements themselves.
bool equalsOnHost(const Tensor &device, const std::vector<float> &expected) {
  Tensor host = F::toDevice(Device::getCpu(), device);
  const float *p = host.getInternalData()->getData<float>(host.getInternalOffset());
  return std::equal(p, p + host.getNumEl(), expected.begin());
}

}  // namespace

CATCH_TEST_CASE("test upsampleNearest2d", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  constexpr int kBatch = 2;
  constexpr int kChannel = 3;
  constexpr int kHeight = 3;
  constexpr int kWidth = 4;
  constexpr int kScale = 2;

  // Distinct whole numbers, which half holds exactly, so what this compares is the arrangement
  // and nothing else: an upsample copies values, and any misplaced one shows up as a difference
  // rather than as rounding.
  std::vector<float> x;
  for (int i = 0; i < kBatch * kChannel * kHeight * kWidth; ++i) x.push_back(float(i));

  int outH = kHeight * kScale;
  int outW = kWidth * kScale;
  std::vector<float> expected(size_t(kBatch) * kChannel * outH * outW);
  for (int p = 0; p < kBatch * kChannel; ++p) {
    for (int y = 0; y < outH; ++y) {
      for (int x2 = 0; x2 < outW; ++x2) {
        expected[(size_t(p) * outH + y) * outW + x2] =
            x[(size_t(p) * kHeight + y / kScale) * kWidth + x2 / kScale];
      }
    }
  }

  Tensor out = F::upsampleNearest2d(cudaHalf({kBatch, kChannel, kHeight, kWidth}, x), kScale);
  CATCH_REQUIRE(out.getShape() == std::vector<int>{kBatch, kChannel, outH, outW});
  CATCH_REQUIRE(F::allClose(
      toCpuFloat(out),
      cpuFloat({kBatch, kChannel, outH, outW}, expected),
      1e-6f,
      1e-6f));

  // A scale of one hands the tensor back unchanged, and a scale of three is not only a power of
  // two away from what a U-Net asks for.
  Tensor same = F::upsampleNearest2d(cudaHalf({1, 1, 2, 2}, {1.0f, 2.0f, 3.0f, 4.0f}), 1);
  CATCH_REQUIRE(same.getShape() == std::vector<int>{1, 1, 2, 2});
  CATCH_REQUIRE(F::allClose(
      toCpuFloat(same),
      cpuFloat({1, 1, 2, 2}, {1.0f, 2.0f, 3.0f, 4.0f}),
      1e-6f,
      1e-6f));

  Tensor thrice = F::upsampleNearest2d(cudaHalf({1, 1, 1, 2}, {5.0f, 6.0f}), 3);
  CATCH_REQUIRE(thrice.getShape() == std::vector<int>{1, 1, 3, 6});
  CATCH_REQUIRE(F::allClose(
      toCpuFloat(thrice),
      cpuFloat({1, 1, 3, 6}, {5, 5, 5, 6, 6, 6, 5, 5, 5, 6, 6, 6, 5, 5, 5, 6, 6, 6}),
      1e-6f,
      1e-6f));
}

CATCH_TEST_CASE("test upsampleNearest2d (float)", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  // SDXL's autoencoder runs in float32, and this operator sits between two of its up blocks. It
  // only ever copies a pixel, so the float result has to be exact and the shape has to be the
  // same one the half arm produces.
  constexpr int kHeight = 3;
  constexpr int kWidth = 4;
  constexpr int kScale = 2;

  std::vector<float> x;
  for (int i = 0; i < kHeight * kWidth; ++i) x.push_back(float(i) + 0.5f);

  int outH = kHeight * kScale;
  int outW = kWidth * kScale;
  std::vector<float> expected(size_t(outH) * outW);
  for (int y = 0; y < outH; ++y) {
    for (int x2 = 0; x2 < outW; ++x2) {
      expected[size_t(y) * outW + x2] = x[size_t(y / kScale) * kWidth + x2 / kScale];
    }
  }

  Tensor out = F::upsampleNearest2d(cudaFloat({1, 1, kHeight, kWidth}, x), kScale);
  CATCH_REQUIRE(out.getDType() == DType::kFloat);
  CATCH_REQUIRE(out.getShape() == std::vector<int>{1, 1, outH, outW});
  CATCH_REQUIRE(equalsOnHost(out, expected));

  // A value half cannot hold at all, which is the whole reason this arm exists.
  Tensor huge = F::upsampleNearest2d(cudaFloat({1, 1, 1, 2}, {1e20f, -3e5f}), 2);
  CATCH_REQUIRE(equalsOnHost(
      huge,
      {1e20f, 1e20f, -3e5f, -3e5f, 1e20f, 1e20f, -3e5f, -3e5f}));
}

}  // namespace fl
