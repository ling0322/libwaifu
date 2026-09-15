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

#include <stdint.h>

#include <cmath>
#include <vector>

#include "catch2/catch_amalgamated.hpp"
#include "flint/cuda/rand.h"
#include "flint/device.h"
#include "flint/functional.h"
#include "flint/operators.h"

namespace fl {
namespace {

Tensor toCpu(const Tensor &a) {
  return F::toDevice(Device::getCpu(), F::cast(a, DType::kFloat));
}

std::vector<float> values(const Tensor &a) {
  Tensor host = toCpu(a);
  const float *data = host.getInternalData()->getData<float>(host.getInternalOffset());

  return std::vector<float>(data, data + host.getNumEl());
}

}  // namespace

// The vectors Random123 publishes for philox4x32 at ten rounds. They pin the bijection itself,
// which is the half of this generator that is not ours to choose: an implementation that gets
// these right is Philox, and one that gets them wrong is some other generator that would still
// have passed every distribution test below.
CATCH_TEST_CASE("test Philox4x32-10 against the published vectors", "[op][cuda]") {
  struct Vector {
    uint32_t ctr[4];
    uint32_t key[2];
    uint32_t expected[4];
  };

  const Vector vectors[] = {
      {{0x00000000, 0x00000000, 0x00000000, 0x00000000},
       {0x00000000, 0x00000000},
       {0x6627e8d5, 0xe169c58d, 0xbc57ac4c, 0x9b00dbd8}},
      {{0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff},
       {0xffffffff, 0xffffffff},
       {0x408f276d, 0x41c83b0e, 0xa20bc7c6, 0x6d5451fd}},
      {{0x243f6a88, 0x85a308d3, 0x13198a2e, 0x03707344},
       {0xa4093822, 0x299f31d0},
       {0xd16cfe09, 0x94fdcceb, 0x5001e420, 0x24126ea1}},
  };

  for (const Vector &v : vectors) {
    uint32_t out[4];
    op::cuda::philox4x32_10ForTest(v.ctr, v.key, out);

    for (int i = 0; i < 4; ++i) {
      CATCH_REQUIRE(out[i] == v.expected[i]);
    }
  }
}

CATCH_TEST_CASE("test CUDA randNormal", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  Tensor x = toCpu(getOperators(Device::kCuda)->randNormal({4096}));

  const float *data = x.getInternalData()->getData<float>(x.getInternalOffset());
  double sum = 0.0;
  double sumSquare = 0.0;
  for (int i = 0; i < x.getNumEl(); ++i) {
    sum += data[i];
    sumSquare += data[i] * data[i];
  }

  double mean = sum / x.getNumEl();
  double stddev = sqrt(sumSquare / x.getNumEl() - mean * mean);
  CATCH_REQUIRE(fabs(mean) < 0.1);
  CATCH_REQUIRE(fabs(stddev - 1.0) < 0.1);
}

CATCH_TEST_CASE("test CUDA rand is uniform over the unit interval", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  std::vector<float> x = values(getOperators(Device::kCuda)->rand({8192}, DType::kFloat));

  // Sixteen even buckets. With 8192 draws each expects 512, and the bound is loose enough that a
  // fair generator will not trip it and tight enough to catch a stuck or clipped word.
  int buckets[16] = {0};
  double sum = 0.0;
  for (float v : x) {
    CATCH_REQUIRE(v > 0.0f);
    CATCH_REQUIRE(v <= 1.0f);
    sum += v;
    ++buckets[static_cast<int>(v * 16.0f) & 15];
  }

  CATCH_REQUIRE(fabs(sum / x.size() - 0.5) < 0.02);
  for (int count : buckets) {
    CATCH_REQUIRE(count > 400);
    CATCH_REQUIRE(count < 630);
  }
}

// What a caller asking for a seed is buying. Position is reset alongside the seed, so the run that
// draws a latent and the run that draws it again start from the same place in the same stream.
CATCH_TEST_CASE("test CUDA manualSeed repeats a draw", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  F::manualSeed(Device::getCuda(), 42);
  std::vector<float> first = values(getOperators(Device::kCuda)->randNormal({1024}));

  F::manualSeed(Device::getCuda(), 42);
  std::vector<float> again = values(getOperators(Device::kCuda)->randNormal({1024}));

  F::manualSeed(Device::getCuda(), 43);
  std::vector<float> other = values(getOperators(Device::kCuda)->randNormal({1024}));

  CATCH_REQUIRE(first == again);
  CATCH_REQUIRE(first != other);
}

// Two draws in a row are two different draws. A counter-based generator gets this wrong by the
// obvious route -- forgetting to carry the position forward -- and the result is a pair of
// identical latents rather than anything that looks broken.
CATCH_TEST_CASE("test CUDA rand advances between draws", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  F::manualSeed(Device::getCuda(), 7);
  std::vector<float> first = values(getOperators(Device::kCuda)->rand({257}, DType::kFloat));
  std::vector<float> second = values(getOperators(Device::kCuda)->rand({257}, DType::kFloat));

  CATCH_REQUIRE(first != second);

  // 257 values is 64 whole blocks and one value of a 65th, so the first draw ends a value into a
  // block. The second has to start after that block rather than on it: rounding the advance down
  // would put second[0] on the same position as first[256], and the two draws would share it.
  // These are float32 draws, so a match is an overlap rather than a coincidence.
  for (int i = 0; i < 4; ++i) {
    CATCH_REQUIRE(first[first.size() - 1 - i] != second[i]);
  }
}

}  // namespace fl
