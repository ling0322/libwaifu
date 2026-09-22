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
  // Called on the CPU inputs as well as on what the card gave back, so a tensor already on the
  // CPU stays where it is: the Metal operators take only their own.
  Tensor host = a.getDevice().getType() == Device::kCpu ? a : toCpu(a);
  Tensor c = cpuOps()->contiguous(host);
  const float *data = c.getInternalData()->getData<float>(c.getInternalOffset());
  return std::vector<float>(data, data + c.getNumEl());
}

}  // namespace

CATCH_TEST_CASE("test Metal lookup", "[op][metal]") {
  if (!isOperatorsAvailable(Device::kMetal)) CATCH_SKIP("metal device not available");

  Tensor table = cpuOps()->rand({20, 8}, DType::kFloat);
  Tensor indices = Tensor::create<LongType>({2, 3}, {0, 5, 19, 3, 11, 7});

  CATCH_REQUIRE(
      cpuOps()->allClose(
          toCpu(metalOps()->lookup(
              toMetal(table),
              metalOps()->toDevice(Device::getMetal(), indices))),
          cpuOps()->lookup(table, indices),
          5e-3, 5e-3));
}

CATCH_TEST_CASE("test Metal lookup (single row)", "[op][metal]") {
  if (!isOperatorsAvailable(Device::kMetal)) CATCH_SKIP("metal device not available");

  Tensor table = cpuOps()->rand({10, 16}, DType::kFloat);
  Tensor indices = Tensor::create<LongType>({1}, {7});

  CATCH_REQUIRE(
      cpuOps()->allClose(
          toCpu(metalOps()->lookup(
              toMetal(table),
              metalOps()->toDevice(Device::getMetal(), indices))),
          cpuOps()->lookup(table, indices),
          5e-3, 5e-3));
}

CATCH_TEST_CASE("test Metal geglu", "[op][metal]") {
  if (!isOperatorsAvailable(Device::kMetal)) CATCH_SKIP("metal device not available");

  Tensor gated = cpuOps()->rand({3, 5, 16}, DType::kFloat);
  CATCH_REQUIRE(cpuOps()->allClose(
      toCpu(metalOps()->geglu(toMetal(gated))),
      cpuOps()->geglu(gated),
      5e-3,
      5e-3));
}

CATCH_TEST_CASE("test Metal geglu (SDXL width)", "[op][metal]") {
  if (!isOperatorsAvailable(Device::kMetal)) CATCH_SKIP("metal device not available");

  Tensor gated = cpuOps()->rand({1024, 10240}, DType::kFloat);
  CATCH_REQUIRE(cpuOps()->allClose(
      toCpu(metalOps()->geglu(toMetal(gated))),
      cpuOps()->geglu(gated),
      5e-3,
      5e-3));
}

CATCH_TEST_CASE("test Metal swiglu", "[op][metal]") {
  if (!isOperatorsAvailable(Device::kMetal)) CATCH_SKIP("metal device not available");

  Tensor gated = cpuOps()->rand({3, 5, 16}, DType::kFloat);
  CATCH_REQUIRE(cpuOps()->allClose(
      toCpu(metalOps()->swiglu(toMetal(gated))),
      cpuOps()->swiglu(gated),
      5e-3,
      5e-3));
}

CATCH_TEST_CASE("test Metal upsampleNearest2d", "[op][metal]") {
  if (!isOperatorsAvailable(Device::kMetal)) CATCH_SKIP("metal device not available");

  constexpr int kBatch = 2;
  constexpr int kChannels = 3;
  constexpr int kH = 4;
  constexpr int kW = 5;
  constexpr int kScale = 2;

  Tensor image = cpuOps()->rand({kBatch, kChannels, kH, kW}, DType::kFloat);
  std::vector<float> x = readFloats(image);

  std::vector<float> expected(kBatch * kChannels * kH * kScale * kW * kScale);
  for (int i = 0; i < kBatch * kChannels; ++i) {
    for (int oh = 0; oh < kH * kScale; ++oh) {
      for (int ow = 0; ow < kW * kScale; ++ow) {
        expected[(i * kH * kScale + oh) * kW * kScale + ow] =
            x[(i * kH + oh / kScale) * kW + ow / kScale];
      }
    }
  }

  Tensor got = metalOps()->upsampleNearest2d(toMetal(image), kScale);
  std::vector<float> actual = readFloats(got);
  for (size_t i = 0; i < expected.size(); ++i) {
    CATCH_INFO("element " << i << ": " << actual[i] << " vs " << expected[i]);
    CATCH_REQUIRE(std::fabs(actual[i] - expected[i]) < 5e-3f);
  }
}

CATCH_TEST_CASE("test Metal cast", "[op][metal]") {
  if (!isOperatorsAvailable(Device::kMetal)) CATCH_SKIP("metal device not available");

  Tensor a = cpuOps()->rand({4, 8}, DType::kFloat);
  Tensor metalA = metalOps()->toDevice(Device::getMetal(), a);
  Tensor half = metalOps()->cast(metalA, DType::kFloat16);
  Tensor back = metalOps()->cast(half, DType::kFloat);
  CATCH_REQUIRE(cpuOps()->allClose(toCpu(back), a, 5e-3, 5e-3));
}

CATCH_TEST_CASE("test Metal contiguous (transposed view)", "[op][metal]") {
  if (!isOperatorsAvailable(Device::kMetal)) CATCH_SKIP("metal device not available");

  Tensor a = cpuOps()->rand({2, 4, 6}, DType::kFloat);
  CATCH_REQUIRE(
      cpuOps()->allClose(
          toCpu(metalOps()->contiguous(toMetal(a).transpose(0, 2))),
          cpuOps()->contiguous(a.transpose(0, 2)),
          5e-3, 5e-3));
}

CATCH_TEST_CASE("test Metal cat", "[op][metal]") {
  if (!isOperatorsAvailable(Device::kMetal)) CATCH_SKIP("metal device not available");

  Tensor a = cpuOps()->rand({2, 4, 6}, DType::kFloat);
  Tensor b = cpuOps()->rand({2, 4, 6}, DType::kFloat);

  CATCH_REQUIRE(cpuOps()->allClose(
      toCpu(metalOps()->cat(toMetal(a), toMetal(b), -1)),
      cpuOps()->cat(a, b, -1),
      5e-3,
      5e-3));
  CATCH_REQUIRE(cpuOps()->allClose(
      toCpu(metalOps()->cat(toMetal(a), toMetal(b), 0)),
      cpuOps()->cat(a, b, 0),
      5e-3,
      5e-3));
  CATCH_REQUIRE(cpuOps()->allClose(
      toCpu(metalOps()->cat(toMetal(a), toMetal(b), 1)),
      cpuOps()->cat(a, b, 1),
      5e-3,
      5e-3));
}

CATCH_TEST_CASE("test Metal view and unsqueeze", "[op][metal]") {
  if (!isOperatorsAvailable(Device::kMetal)) CATCH_SKIP("metal device not available");

  Tensor a = cpuOps()->rand({2, 4, 6}, DType::kFloat);
  CATCH_REQUIRE(cpuOps()->allClose(toCpu(toMetal(a).view({2, 24})), a.view({2, 24}), 5e-3, 5e-3));
  CATCH_REQUIRE(cpuOps()->allClose(toCpu(toMetal(a).unsqueeze(1)), a.unsqueeze(1), 5e-3, 5e-3));
  CATCH_REQUIRE(cpuOps()->allClose(toCpu(toMetal(a).view({48})), a.view({48}), 5e-3, 5e-3));
}

CATCH_TEST_CASE("test Metal copy", "[op][metal]") {
  if (!isOperatorsAvailable(Device::kMetal)) CATCH_SKIP("metal device not available");

  Tensor src = cpuOps()->rand({3, 5}, DType::kFloat);
  Tensor metalSrc = toMetal(src);
  Tensor metalDst = metalOps()->zeros({3, 5}, DType::kFloat16);
  metalOps()->copy(metalSrc, metalDst);
  CATCH_REQUIRE(cpuOps()->allClose(toCpu(metalDst), src, 5e-3, 5e-3));
}

}  // namespace fl
