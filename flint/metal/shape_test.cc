#include "catch2/catch_amalgamated.hpp"
#include "flint/device.h"
#include "flint/functional.h"
#include "flint/operators.h"

namespace fl {
namespace {

Tensor toMetal(const Tensor &a) {
  return F::cast(F::toDevice(Device::getMetal(), a), DType::kFloat16);
}

Tensor toCpu(const Tensor &a) {
  return F::toDevice(Device::getCpu(), F::cast(a, DType::kFloat));
}

std::vector<float> readFloats(const Tensor &a) {
  Tensor c = F::contiguous(toCpu(a));
  const float *data = c.getInternalData()->getData<float>(c.getInternalOffset());
  return std::vector<float>(data, data + c.getNumEl());
}

}  // namespace

CATCH_TEST_CASE("test Metal lookup", "[op][metal]") {
  if (!isOperatorsAvailable(Device::kMetal)) CATCH_SKIP("metal device not available");

  Tensor table = F::rand({20, 8}, DType::kFloat);
  Tensor indices = Tensor::create<LongType>({2, 3}, {0, 5, 19, 3, 11, 7});

  CATCH_REQUIRE(
      F::allClose(
          toCpu(F::lookup(toMetal(table), F::toDevice(Device::getMetal(), indices))),
          F::lookup(table, indices),
          5e-3, 5e-3));
}

CATCH_TEST_CASE("test Metal lookup (single row)", "[op][metal]") {
  if (!isOperatorsAvailable(Device::kMetal)) CATCH_SKIP("metal device not available");

  Tensor table = F::rand({10, 16}, DType::kFloat);
  Tensor indices = Tensor::create<LongType>({1}, {7});

  CATCH_REQUIRE(
      F::allClose(
          toCpu(F::lookup(toMetal(table), F::toDevice(Device::getMetal(), indices))),
          F::lookup(table, indices),
          5e-3, 5e-3));
}

CATCH_TEST_CASE("test Metal geglu", "[op][metal]") {
  if (!isOperatorsAvailable(Device::kMetal)) CATCH_SKIP("metal device not available");

  Tensor gated = F::rand({3, 5, 16}, DType::kFloat);
  CATCH_REQUIRE(F::allClose(toCpu(F::geglu(toMetal(gated))), F::geglu(gated), 5e-3, 5e-3));
}

CATCH_TEST_CASE("test Metal geglu (SDXL width)", "[op][metal]") {
  if (!isOperatorsAvailable(Device::kMetal)) CATCH_SKIP("metal device not available");

  Tensor gated = F::rand({1024, 10240}, DType::kFloat);
  CATCH_REQUIRE(F::allClose(toCpu(F::geglu(toMetal(gated))), F::geglu(gated), 5e-3, 5e-3));
}

CATCH_TEST_CASE("test Metal swiglu", "[op][metal]") {
  if (!isOperatorsAvailable(Device::kMetal)) CATCH_SKIP("metal device not available");

  Tensor gated = F::rand({3, 5, 16}, DType::kFloat);
  CATCH_REQUIRE(F::allClose(toCpu(F::swiglu(toMetal(gated))), F::swiglu(gated), 5e-3, 5e-3));
}

CATCH_TEST_CASE("test Metal upsampleNearest2d", "[op][metal]") {
  if (!isOperatorsAvailable(Device::kMetal)) CATCH_SKIP("metal device not available");

  constexpr int kBatch = 2;
  constexpr int kChannels = 3;
  constexpr int kH = 4;
  constexpr int kW = 5;
  constexpr int kScale = 2;

  Tensor image = F::rand({kBatch, kChannels, kH, kW}, DType::kFloat);
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

  Tensor got = F::upsampleNearest2d(toMetal(image), kScale);
  std::vector<float> actual = readFloats(got);
  for (size_t i = 0; i < expected.size(); ++i) {
    CATCH_INFO("element " << i << ": " << actual[i] << " vs " << expected[i]);
    CATCH_REQUIRE(std::fabs(actual[i] - expected[i]) < 5e-3f);
  }
}

CATCH_TEST_CASE("test Metal cast", "[op][metal]") {
  if (!isOperatorsAvailable(Device::kMetal)) CATCH_SKIP("metal device not available");

  Tensor a = F::rand({4, 8}, DType::kFloat);
  Tensor metalA = F::toDevice(Device::getMetal(), a);
  Tensor half = F::cast(metalA, DType::kFloat16);
  Tensor back = F::cast(half, DType::kFloat);
  CATCH_REQUIRE(F::allClose(toCpu(back), a, 5e-3, 5e-3));
}

CATCH_TEST_CASE("test Metal contiguous (transposed view)", "[op][metal]") {
  if (!isOperatorsAvailable(Device::kMetal)) CATCH_SKIP("metal device not available");

  Tensor a = F::rand({2, 4, 6}, DType::kFloat);
  CATCH_REQUIRE(
      F::allClose(
          toCpu(F::contiguous(toMetal(a).transpose(0, 2))),
          F::contiguous(a.transpose(0, 2)),
          5e-3, 5e-3));
}

CATCH_TEST_CASE("test Metal cat", "[op][metal]") {
  if (!isOperatorsAvailable(Device::kMetal)) CATCH_SKIP("metal device not available");

  Tensor a = F::rand({2, 4, 6}, DType::kFloat);
  Tensor b = F::rand({2, 4, 6}, DType::kFloat);

  CATCH_REQUIRE(F::allClose(toCpu(F::cat(toMetal(a), toMetal(b), -1)), F::cat(a, b, -1), 5e-3, 5e-3));
  CATCH_REQUIRE(F::allClose(toCpu(F::cat(toMetal(a), toMetal(b), 0)), F::cat(a, b, 0), 5e-3, 5e-3));
  CATCH_REQUIRE(F::allClose(toCpu(F::cat(toMetal(a), toMetal(b), 1)), F::cat(a, b, 1), 5e-3, 5e-3));
}

CATCH_TEST_CASE("test Metal view and unsqueeze", "[op][metal]") {
  if (!isOperatorsAvailable(Device::kMetal)) CATCH_SKIP("metal device not available");

  Tensor a = F::rand({2, 4, 6}, DType::kFloat);
  CATCH_REQUIRE(F::allClose(toCpu(toMetal(a).view({2, 24})), a.view({2, 24}), 5e-3, 5e-3));
  CATCH_REQUIRE(F::allClose(toCpu(toMetal(a).unsqueeze(1)), a.unsqueeze(1), 5e-3, 5e-3));
  CATCH_REQUIRE(F::allClose(toCpu(toMetal(a).view({48})), a.view({48}), 5e-3, 5e-3));
}

CATCH_TEST_CASE("test Metal copy", "[op][metal]") {
  if (!isOperatorsAvailable(Device::kMetal)) CATCH_SKIP("metal device not available");

  Tensor src = F::rand({3, 5}, DType::kFloat);
  Tensor metalSrc = toMetal(src);
  Tensor metalDst = F::zeros({3, 5}, DType::kFloat16, Device::getMetal());
  F::copy(metalSrc, metalDst);
  CATCH_REQUIRE(F::allClose(toCpu(metalDst), src, 5e-3, 5e-3));
}

}  // namespace fl
