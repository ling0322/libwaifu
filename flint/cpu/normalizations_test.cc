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

// The two normalizations a diffusion model needs on the CPU, and the upsample between its blocks.
// Each is checked against the same arithmetic written out plainly here, rather than against the
// CUDA kernel: the two are meant to agree, so comparing them would hide a misreading they share.

#include <cmath>
#include <vector>

#include "catch2/catch_amalgamated.hpp"
#include "lutil/span.h"
#include "flint/cpu/upsample.h"
#include "flint/functional.h"
#include "flint/tensor.h"

namespace fl {
namespace op {
namespace cpu {
namespace {

/// Values that vary without a pattern a normalization could accidentally satisfy, and without
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

Tensor of(std::initializer_list<int> shape, const std::vector<float> &values) {
  return Tensor::create<float>(shape, lut::makeConstSpan(values));
}

}  // namespace

CATCH_TEST_CASE("test layerNorm", "[core][nn][operators]") {
  constexpr int kRows = 3;
  constexpr int kWidth = 10;
  constexpr float kEps = 1e-5f;

  std::vector<float> x = spread(kRows * kWidth, 11);
  std::vector<float> weight = spread(kWidth, 13);
  std::vector<float> bias = spread(kWidth, 17);

  std::vector<float> expected(x.size());
  for (int row = 0; row < kRows; ++row) {
    double sum = 0.0;
    for (int i = 0; i < kWidth; ++i) sum += x[row * kWidth + i];
    double mean = sum / kWidth;

    double variance = 0.0;
    for (int i = 0; i < kWidth; ++i) {
      double d = x[row * kWidth + i] - mean;
      variance += d * d;
    }
    variance /= kWidth;

    double invStd = 1.0 / std::sqrt(variance + kEps);
    for (int i = 0; i < kWidth; ++i) {
      expected[row * kWidth + i] =
          float((x[row * kWidth + i] - mean) * invStd * weight[i] + bias[i]);
    }
  }

  CATCH_REQUIRE(F::allClose(
      F::layerNorm(of({kRows, kWidth}, x), of({kWidth}, weight), of({kWidth}, bias), kEps),
      of({kRows, kWidth}, expected),
      1e-4f));

  // A weight and a bias are both optional, and an absent one is not a tensor of ones or zeros to
  // multiply and add -- it is skipped, which is what makes this worth saying separately.
  std::vector<float> bare(x.size());
  for (int row = 0; row < kRows; ++row) {
    double sum = 0.0;
    for (int i = 0; i < kWidth; ++i) sum += x[row * kWidth + i];
    double mean = sum / kWidth;
    double variance = 0.0;
    for (int i = 0; i < kWidth; ++i) {
      double d = x[row * kWidth + i] - mean;
      variance += d * d;
    }
    double invStd = 1.0 / std::sqrt(variance / kWidth + kEps);
    for (int i = 0; i < kWidth; ++i) {
      bare[row * kWidth + i] = float((x[row * kWidth + i] - mean) * invStd);
    }
  }

  CATCH_REQUIRE(F::allClose(
      F::layerNorm(of({kRows, kWidth}, x), Tensor(), Tensor(), kEps),
      of({kRows, kWidth}, bare),
      1e-4f));
}

CATCH_TEST_CASE("test groupNorm", "[core][nn][operators]") {
  constexpr int kBatch = 2;
  constexpr int kChannels = 8;
  constexpr int kHeight = 3;
  constexpr int kWidth = 5;
  constexpr int kGroups = 4;
  constexpr float kEps = 1e-5f;
  constexpr int kSpatial = kHeight * kWidth;
  constexpr int kPerGroup = kChannels / kGroups;

  std::vector<float> x = spread(kBatch * kChannels * kSpatial, 23);
  std::vector<float> weight = spread(kChannels, 29);
  std::vector<float> bias = spread(kChannels, 31);

  std::vector<float> expected(x.size());
  for (int n = 0; n < kBatch; ++n) {
    for (int g = 0; g < kGroups; ++g) {
      double sum = 0.0;
      double sumSquare = 0.0;
      for (int c = 0; c < kPerGroup; ++c) {
        for (int p = 0; p < kSpatial; ++p) {
          double value = x[(size_t(n) * kChannels + g * kPerGroup + c) * kSpatial + p];
          sum += value;
          sumSquare += value * value;
        }
      }

      int count = kPerGroup * kSpatial;
      double mean = sum / count;
      double invStd = 1.0 / std::sqrt(sumSquare / count - mean * mean + kEps);

      for (int c = 0; c < kPerGroup; ++c) {
        int channel = g * kPerGroup + c;
        for (int p = 0; p < kSpatial; ++p) {
          size_t index = (size_t(n) * kChannels + channel) * kSpatial + p;
          expected[index] = float((x[index] - mean) * invStd * weight[channel] + bias[channel]);
        }
      }
    }
  }

  CATCH_REQUIRE(F::allClose(
      F::groupNorm(
          of({kBatch, kChannels, kHeight, kWidth}, x),
          of({kChannels}, weight),
          of({kChannels}, bias),
          kGroups,
          kEps),
      of({kBatch, kChannels, kHeight, kWidth}, expected),
      1e-4f));

  // One group is a normalization over the whole image, and one group per channel is a
  // normalization of each channel on its own. Both are ends the arithmetic has to reach.
  CATCH_REQUIRE_NOTHROW(F::groupNorm(
      of({kBatch, kChannels, kHeight, kWidth}, x), Tensor(), Tensor(), 1, kEps));
  CATCH_REQUIRE_NOTHROW(F::groupNorm(
      of({kBatch, kChannels, kHeight, kWidth}, x), Tensor(), Tensor(), kChannels, kEps));

  // Channels that do not divide into the groups is a caller's mistake rather than something to
  // round off.
  CATCH_REQUIRE_THROWS(F::groupNorm(
      of({kBatch, kChannels, kHeight, kWidth}, x), Tensor(), Tensor(), 3, kEps));
}

CATCH_TEST_CASE("test upsampleNearest2d", "[core][nn][operators]") {
  constexpr int kChannels = 2;
  constexpr int kHeight = 3;
  constexpr int kWidth = 4;
  constexpr int kScale = 2;

  // Distinct whole numbers, so that a misplaced pixel shows as a difference rather than as
  // rounding: an upsample copies values and never computes one.
  std::vector<float> x;
  for (int i = 0; i < kChannels * kHeight * kWidth; ++i) x.push_back(float(i));

  int outH = kHeight * kScale;
  int outW = kWidth * kScale;
  std::vector<float> expected(size_t(kChannels) * outH * outW);
  for (int c = 0; c < kChannels; ++c) {
    for (int y = 0; y < outH; ++y) {
      for (int x2 = 0; x2 < outW; ++x2) {
        expected[(size_t(c) * outH + y) * outW + x2] =
            x[(size_t(c) * kHeight + y / kScale) * kWidth + x2 / kScale];
      }
    }
  }

  Tensor out = F::upsampleNearest2d(of({1, kChannels, kHeight, kWidth}, x), kScale);
  CATCH_REQUIRE(out.getShape() == std::vector<int>{1, kChannels, outH, outW});
  CATCH_REQUIRE(F::allClose(out, of({1, kChannels, outH, outW}, expected), 1e-6f, 1e-6f));

  // A scale of one hands the image back as it was, and a scale of three is not a power of two.
  Tensor same = F::upsampleNearest2d(of({1, 1, 2, 2}, {1.0f, 2.0f, 3.0f, 4.0f}), 1);
  CATCH_REQUIRE(F::allClose(same, of({1, 1, 2, 2}, {1.0f, 2.0f, 3.0f, 4.0f}), 1e-6f, 1e-6f));

  Tensor thrice = F::upsampleNearest2d(of({1, 1, 1, 2}, {5.0f, 6.0f}), 3);
  CATCH_REQUIRE(thrice.getShape() == std::vector<int>{1, 1, 3, 6});
  CATCH_REQUIRE(F::allClose(
      thrice,
      of({1, 1, 3, 6}, {5, 5, 5, 6, 6, 6, 5, 5, 5, 6, 6, 6, 5, 5, 5, 6, 6, 6}),
      1e-6f,
      1e-6f));
}

}  // namespace cpu
}  // namespace op
}  // namespace fl
