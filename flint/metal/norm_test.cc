#include <cmath>
#include <vector>

#include "catch2/catch_amalgamated.hpp"
#include "flint/device.h"
#include "flint/operators.h"

namespace fl {

namespace {

/// The Metal operators, which the calls here that run on Metal are asked of.
Operators *metalOps() {
  return getOperators(Device::kMetal);
}

/// The CPU operators, which the calls here that run on CPU are asked of.
Operators *cpuOps() {
  return getOperators(Device::kCpu);
}

Tensor toMetal(const Tensor &a) {
  return metalOps()->cast(metalOps()->toDevice(Device::getMetal(), a), DType::kFloat16);
}

Tensor toCpu(const Tensor &a) {
  return metalOps()->toDevice(Device::getCpu(), metalOps()->cast(a, DType::kFloat));
}

std::vector<float> readFloats(const Tensor &a) {
  Tensor c = cpuOps()->contiguous(toCpu(a));
  const float *data = c.getInternalData()->getData<float>(c.getInternalOffset());
  return std::vector<float>(data, data + c.getNumEl());
}

}  // namespace

CATCH_TEST_CASE("test Metal layerNorm", "[op][metal]") {
  if (!isOperatorsAvailable(Device::kMetal)) CATCH_SKIP("metal device not available");

  for (int cols : {16, 48, 640, 1280}) {
    CATCH_INFO("cols = " << cols);
    constexpr int kRows = 4;
    constexpr float kEps = 1e-5f;

    Tensor a = cpuOps()->rand({kRows, cols}, DType::kFloat);
    Tensor weight = cpuOps()->rand({cols}, DType::kFloat);
    Tensor bias = cpuOps()->rand({cols}, DType::kFloat);

    std::vector<float> x = readFloats(a);
    std::vector<float> w = readFloats(weight);
    std::vector<float> b = readFloats(bias);

    std::vector<float> expected(x.size());
    for (int row = 0; row < kRows; ++row) {
      float mean = 0.0f;
      for (int i = 0; i < cols; ++i) mean += x[row * cols + i];
      mean /= cols;

      float var = 0.0f;
      for (int i = 0; i < cols; ++i) var += (x[row * cols + i] - mean) * (x[row * cols + i] - mean);
      var /= cols;

      float scale = 1.0f / std::sqrt(var + kEps);
      for (int i = 0; i < cols; ++i) {
        expected[row * cols + i] = (x[row * cols + i] - mean) * scale * w[i] + b[i];
      }
    }

    Tensor got = metalOps()->layerNorm(toMetal(a), toMetal(weight), toMetal(bias), kEps);
    std::vector<float> actual = readFloats(got);
    for (size_t i = 0; i < expected.size(); ++i) {
      CATCH_INFO("element " << i << ": " << actual[i] << " vs " << expected[i]);
      CATCH_REQUIRE(std::fabs(actual[i] - expected[i]) < 5e-3f);
    }
  }
}

CATCH_TEST_CASE("test Metal layerNorm (no weight, no bias)", "[op][metal]") {
  if (!isOperatorsAvailable(Device::kMetal)) CATCH_SKIP("metal device not available");

  constexpr int kHidden = 64;
  Tensor a = cpuOps()->rand({1, kHidden}, DType::kFloat);

  Tensor out = metalOps()->layerNorm(toMetal(a), Tensor(), Tensor(), 1e-5f);
  std::vector<float> data = readFloats(out);

  double mean = 0.0;
  double square = 0.0;
  for (int i = 0; i < kHidden; ++i) {
    mean += data[i];
    square += double(data[i]) * data[i];
  }
  CATCH_REQUIRE(std::abs(mean / kHidden) < 2e-2);
  CATCH_REQUIRE(std::abs(square / kHidden - 1.0) < 5e-2);
}

CATCH_TEST_CASE("test Metal layerNorm (3D input)", "[op][metal]") {
  if (!isOperatorsAvailable(Device::kMetal)) CATCH_SKIP("metal device not available");

  // The transformer path: [batch, tokens, hidden].
  Tensor a = cpuOps()->rand({2, 64, 640}, DType::kFloat);
  Tensor w = cpuOps()->rand({640}, DType::kFloat);
  Tensor b = cpuOps()->rand({640}, DType::kFloat);

  Tensor got = metalOps()->layerNorm(toMetal(a), toMetal(w), toMetal(b), 1e-5f);
  CATCH_REQUIRE(got.getShape() == std::vector<int>{2, 64, 640});

  std::vector<float> data = readFloats(got);
  int nanCount = 0;
  for (float v : data) {
    if (std::isnan(v)) ++nanCount;
  }
  CATCH_REQUIRE(nanCount == 0);
}

CATCH_TEST_CASE("test Metal groupNorm", "[op][metal]") {
  if (!isOperatorsAvailable(Device::kMetal)) CATCH_SKIP("metal device not available");

  constexpr int kBatch = 2;
  constexpr int kChannels = 8;
  constexpr int kGroups = 4;
  constexpr int kHeight = 3;
  constexpr int kWidth = 3;
  constexpr int kSpatial = kHeight * kWidth;
  constexpr int kPerGroup = kChannels / kGroups;
  constexpr float kEps = 1e-5f;

  Tensor a = cpuOps()->rand({kBatch, kChannels, kHeight, kWidth}, DType::kFloat);
  Tensor weight = cpuOps()->rand({kChannels}, DType::kFloat);
  Tensor bias = cpuOps()->rand({kChannels}, DType::kFloat);

  std::vector<float> x = readFloats(a);
  std::vector<float> w = readFloats(weight);
  std::vector<float> b = readFloats(bias);

  std::vector<float> expected(x.size());
  for (int n = 0; n < kBatch; ++n) {
    for (int g = 0; g < kGroups; ++g) {
      int base = n * kChannels * kSpatial + g * kPerGroup * kSpatial;
      int count = kPerGroup * kSpatial;

      float mean = 0.0f;
      for (int i = 0; i < count; ++i) mean += x[base + i];
      mean /= count;

      float var = 0.0f;
      for (int i = 0; i < count; ++i) var += (x[base + i] - mean) * (x[base + i] - mean);
      var /= count;

      float scale = 1.0f / std::sqrt(var + kEps);
      for (int i = 0; i < count; ++i) {
        int channel = g * kPerGroup + i / kSpatial;
        expected[base + i] = (x[base + i] - mean) * scale * w[channel] + b[channel];
      }
    }
  }

  Tensor got = metalOps()->groupNorm(toMetal(a), toMetal(weight), toMetal(bias), kGroups, kEps);
  std::vector<float> actual = readFloats(got);
  for (size_t i = 0; i < expected.size(); ++i) {
    CATCH_INFO("element " << i << ": " << actual[i] << " vs " << expected[i]);
    CATCH_REQUIRE(std::fabs(actual[i] - expected[i]) < 5e-3f);
  }
}

CATCH_TEST_CASE("test Metal groupNorm (one group, and one per channel)", "[op][metal]") {
  if (!isOperatorsAvailable(Device::kMetal)) CATCH_SKIP("metal device not available");

  Tensor input = toMetal(cpuOps()->rand({1, 4, 3, 3}, DType::kFloat));
  Tensor one = metalOps()->groupNorm(input, Tensor(), Tensor(), 1, 1e-5f);
  Tensor each = metalOps()->groupNorm(input, Tensor(), Tensor(), 4, 1e-5f);

  CATCH_REQUIRE(one.getShape() == std::vector<int>{1, 4, 3, 3});
  CATCH_REQUIRE(each.getShape() == std::vector<int>{1, 4, 3, 3});
}

CATCH_TEST_CASE("test Metal groupNorm (large plane, VAE scale)", "[op][metal]") {
  if (!isOperatorsAvailable(Device::kMetal)) CATCH_SKIP("metal device not available");

  // A VAE decoder normalizes 128 channels over 256x256 in 32 groups. The sum of squares is
  // over 260k elements per group, which is where fp16 runs out if the accumulation is not
  // widened.
  Tensor a = cpuOps()->rand({1, 128, 256, 256}, DType::kFloat);
  Tensor w = cpuOps()->rand({128}, DType::kFloat);
  Tensor b = cpuOps()->rand({128}, DType::kFloat);

  Tensor got = metalOps()->groupNorm(toMetal(a), toMetal(w), toMetal(b), 32, 1e-5);
  std::vector<float> data = readFloats(got);
  int nanCount = 0;
  for (float v : data) {
    if (std::isnan(v)) ++nanCount;
  }
  CATCH_INFO("groupNorm NaN count " << nanCount);
  CATCH_REQUIRE(nanCount == 0);
}

}  // namespace fl
