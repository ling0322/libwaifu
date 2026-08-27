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

// The three normalizations norm.cu implements: rmsNorm over the last dimension, layerNorm over
// the same dimension but subtracting the mean first, and groupNorm over a group of channels
// together with the space it covers. Each is checked against the same arithmetic written out
// plainly on the host, except rmsNorm which has a CPU operator to compare against.

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

/// Values that vary without a pattern the operator could accidentally satisfy, and without
/// pulling in a random number generator.
std::vector<float> spread(int count, uint32_t seed) {
  std::vector<float> values;
  uint32_t state = seed | 1;
  for (int i = 0; i < count; ++i) {
    state = state * 1664525u + 1013904223u;
    values.push_back(float(state >> 8) / float(1 << 24) * 4.0f - 2.0f);
  }
  return values;
}

Tensor cudaHalf(std::initializer_list<int> shape, const std::vector<float> &values) {
  return F::cast(
      F::to(Device::getCuda(), Tensor::create<float>(shape, lut::makeConstSpan(values))),
      DType::kFloat16);
}

Tensor toCpuFloat(const Tensor &x) {
  return F::to(Device::getCpu(), F::cast(x, DType::kFloat));
}

/// The counterpart of toCpuFloat, for a tensor that already holds its values.
Tensor toCudaHalf(const Tensor &x) {
  return F::cast(F::to(Device::getCuda(), x), DType::kFloat16);
}

Tensor cpuFloat(std::initializer_list<int> shape, const std::vector<float> &values) {
  return Tensor::create<float>(shape, lut::makeConstSpan(values));
}

}  // namespace

CATCH_TEST_CASE("test rmsNorm", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  for (int lastDim : {10, 11}) {
    Tensor a = F::rand({2, 5, lastDim}, DType::kFloat);
    Tensor w = F::rand({lastDim}, DType::kFloat);
    Tensor x = F::rmsNorm(toCudaHalf(a), toCudaHalf(w), 1e-5);
    CATCH_REQUIRE(F::allClose(toCpuFloat(x), F::rmsNorm(a, w, 1e-5), 5e-3));
  }

  // strided input.
  Tensor a = F::rand({2, 3, 11}, DType::kFloat);
  Tensor w = F::rand({11}, DType::kFloat);
  Tensor x = F::rmsNorm(toCudaHalf(a).transpose(0, 1), toCudaHalf(w), 1e-5);
  CATCH_REQUIRE(F::allClose(toCpuFloat(x), F::rmsNorm(a.transpose(0, 1), w, 1e-5), 5e-3));
}

CATCH_TEST_CASE("test rmsNorm (packed 2D batch)", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  // A packed batch is [tokens, hidden]; the operator adds a leading axis and strips it again, so
  // the result must come back 2D.
  for (int hidden : {8, 11, 512}) {
    Tensor a = F::rand({4, hidden}, DType::kFloat);
    Tensor w = F::rand({hidden}, DType::kFloat);
    Tensor x = F::rmsNorm(toCudaHalf(a), toCudaHalf(w), 1e-5);

    CATCH_INFO("hidden = " << hidden);
    CATCH_REQUIRE(x.getShape() == std::vector<int>{4, hidden});
    CATCH_REQUIRE(F::allClose(toCpuFloat(x), F::rmsNorm(a, w, 1e-5), 5e-3));
  }

  // one token, the decode-step shape.
  Tensor one = F::rand({1, 64}, DType::kFloat);
  Tensor oneW = F::rand({64}, DType::kFloat);
  CATCH_REQUIRE(F::allClose(
      toCpuFloat(F::rmsNorm(toCudaHalf(one), toCudaHalf(oneW), 1e-5)),
      F::rmsNorm(one, oneW, 1e-5),
      5e-3));
}

CATCH_TEST_CASE("test rmsNorm (hidden sizes)", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  // One 256-thread block reduces a whole row, so widths on both sides of the block size take a
  // different number of loop iterations, and odd widths disable the half2 path.
  for (int hidden : {1, 2, 3, 255, 256, 257, 512, 2048, 4096}) {
    Tensor a = F::rand({2, 3, hidden}, DType::kFloat);
    Tensor w = F::rand({hidden}, DType::kFloat);

    CATCH_INFO("hidden = " << hidden);
    CATCH_REQUIRE(F::allClose(
        toCpuFloat(F::rmsNorm(toCudaHalf(a), toCudaHalf(w), 1e-5)),
        F::rmsNorm(a, w, 1e-5),
        1e-2));
  }
}

CATCH_TEST_CASE("test rmsNorm (strided weight)", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  // A non-contiguous weight is enough on its own to force the strided kernel, even when the
  // input is contiguous.
  Tensor a = F::rand({2, 3, 6}, DType::kFloat);
  Tensor wSource = F::rand({6, 4}, DType::kFloat);
  Tensor w = wSource.transpose(0, 1).subtensor(0);
  Tensor wDevice = toCudaHalf(wSource).transpose(0, 1).subtensor(0);
  CATCH_REQUIRE(!wDevice.isContiguous());

  Tensor x = F::rmsNorm(toCudaHalf(a), wDevice, 1e-5);
  CATCH_REQUIRE(F::allClose(toCpuFloat(x), F::rmsNorm(a, F::contiguous(w), 1e-5), 5e-3));
}

CATCH_TEST_CASE("test rmsNorm (eps dominates a zero row)", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  // An all-zero row has zero mean square, so eps is the only thing keeping the reciprocal square
  // root finite. The output stays zero rather than becoming NaN.
  Tensor a = F::zeros({2, 8}, DType::kFloat);
  Tensor w = F::rand({8}, DType::kFloat);

  Tensor x = toCpuFloat(F::rmsNorm(toCudaHalf(a), toCudaHalf(w), 1e-5));
  const float *data = x.getInternalData()->getData<float>(x.getInternalOffset());
  for (int i = 0; i < 16; ++i) {
    CATCH_INFO("i = " << i);
    CATCH_REQUIRE(!std::isnan(data[i]));
    CATCH_REQUIRE(data[i] == 0.0f);
  }
}

CATCH_TEST_CASE("test layerNorm", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  constexpr int kRow = 5;
  constexpr int kHidden = 48;
  constexpr float kEps = 1e-5f;

  std::vector<float> x = spread(kRow * kHidden, 3);
  std::vector<float> weight = spread(kHidden, 5);
  std::vector<float> bias = spread(kHidden, 7);

  std::vector<float> expected(x.size());
  for (int row = 0; row < kRow; ++row) {
    double mean = 0.0;
    for (int i = 0; i < kHidden; ++i) mean += x[row * kHidden + i];
    mean /= kHidden;

    double variance = 0.0;
    for (int i = 0; i < kHidden; ++i) {
      double d = x[row * kHidden + i] - mean;
      variance += d * d;
    }
    variance /= kHidden;

    for (int i = 0; i < kHidden; ++i) {
      double normed = (x[row * kHidden + i] - mean) / std::sqrt(variance + kEps);
      expected[row * kHidden + i] = float(normed * weight[i] + bias[i]);
    }
  }

  Tensor out = F::layerNorm(
      cudaHalf({kRow, kHidden}, x),
      cudaHalf({kHidden}, weight),
      cudaHalf({kHidden}, bias),
      kEps);

  CATCH_REQUIRE(out.getShape() == std::vector<int>{kRow, kHidden});
  CATCH_REQUIRE(F::allClose(toCpuFloat(out), cpuFloat({kRow, kHidden}, expected), 2e-2f));
}

CATCH_TEST_CASE("test layerNorm (no weight, no bias)", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  // Without either, what comes out has a mean of zero and a variance of one, which is worth
  // checking directly rather than against another implementation of the same formula.
  constexpr int kHidden = 64;
  std::vector<float> x = spread(kHidden, 11);

  Tensor out = F::layerNorm(cudaHalf({1, kHidden}, x), Tensor(), Tensor(), 1e-5f);
  Tensor cpu = toCpuFloat(out);
  const float *data = cpu.getInternalData()->getData<float>(cpu.getInternalOffset());
  double mean = 0.0;
  double square = 0.0;
  for (int i = 0; i < kHidden; ++i) {
    mean += data[i];
    square += double(data[i]) * data[i];
  }

  CATCH_REQUIRE(std::abs(mean / kHidden) < 1e-2);
  CATCH_REQUIRE(std::abs(square / kHidden - 1.0) < 2e-2);
}

CATCH_TEST_CASE("test groupNorm", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  constexpr int kBatch = 2;
  constexpr int kChannel = 8;
  constexpr int kHeight = 3;
  constexpr int kWidth = 5;
  constexpr int kGroups = 4;
  constexpr float kEps = 1e-5f;
  constexpr int kSpatial = kHeight * kWidth;
  constexpr int kChannelPerGroup = kChannel / kGroups;

  std::vector<float> x = spread(kBatch * kChannel * kSpatial, 13);
  std::vector<float> weight = spread(kChannel, 17);
  std::vector<float> bias = spread(kChannel, 19);

  std::vector<float> expected(x.size());
  for (int n = 0; n < kBatch; ++n) {
    for (int g = 0; g < kGroups; ++g) {
      int base = (n * kChannel + g * kChannelPerGroup) * kSpatial;
      int count = kChannelPerGroup * kSpatial;

      double mean = 0.0;
      for (int i = 0; i < count; ++i) mean += x[base + i];
      mean /= count;

      double variance = 0.0;
      for (int i = 0; i < count; ++i) {
        double d = x[base + i] - mean;
        variance += d * d;
      }
      variance /= count;

      for (int i = 0; i < count; ++i) {
        int channel = g * kChannelPerGroup + i / kSpatial;
        double normed = (x[base + i] - mean) / std::sqrt(variance + kEps);
        expected[base + i] = float(normed * weight[channel] + bias[channel]);
      }
    }
  }

  Tensor out = F::groupNorm(
      cudaHalf({kBatch, kChannel, kHeight, kWidth}, x),
      cudaHalf({kChannel}, weight),
      cudaHalf({kChannel}, bias),
      kGroups,
      kEps);

  CATCH_REQUIRE(out.getShape() == std::vector<int>{kBatch, kChannel, kHeight, kWidth});
  CATCH_REQUIRE(F::allClose(
      toCpuFloat(out),
      cpuFloat({kBatch, kChannel, kHeight, kWidth}, expected),
      2e-2f));
}

CATCH_TEST_CASE("test groupNorm (one group, and one per channel)", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  // A single group normalizes everything but the batch together; a group per channel normalizes
  // each channel on its own. Both are the same kernel, and both are shapes a model asks for.
  std::vector<float> x = spread(1 * 4 * 2 * 2, 23);
  Tensor input = cudaHalf({1, 4, 2, 2}, x);

  Tensor one = F::groupNorm(input, Tensor(), Tensor(), 1, 1e-5f);
  Tensor each = F::groupNorm(input, Tensor(), Tensor(), 4, 1e-5f);
  CATCH_REQUIRE(one.getShape() == std::vector<int>{1, 4, 2, 2});
  CATCH_REQUIRE(each.getShape() == std::vector<int>{1, 4, 2, 2});

  // Channels that were normalized on their own each have a zero mean of their own.
  Tensor cpu = toCpuFloat(each);
  const float *data = cpu.getInternalData()->getData<float>(cpu.getInternalOffset());
  for (int c = 0; c < 4; ++c) {
    double mean = 0.0;
    for (int i = 0; i < 4; ++i) mean += data[c * 4 + i];
    CATCH_REQUIRE(std::abs(mean / 4) < 2e-2);
  }

  // A channel count that does not divide is a caller's mistake, not a reason to stop.
  CATCH_REQUIRE_THROWS(F::groupNorm(input, Tensor(), Tensor(), 3, 1e-5f));
}

}  // namespace fl
