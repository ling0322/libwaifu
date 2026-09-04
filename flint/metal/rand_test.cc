#include <cmath>

#include "catch2/catch_amalgamated.hpp"
#include "flint/device.h"
#include "flint/functional.h"
#include "flint/operators.h"

namespace fl {
namespace {

Tensor toCpu(const Tensor &a) {
  return F::toDevice(Device::getCpu(), F::cast(a, DType::kFloat));
}

std::vector<float> readFloats(const Tensor &a) {
  Tensor c = F::contiguous(toCpu(a));
  const float *data = c.getInternalData()->getData<float>(c.getInternalOffset());
  return std::vector<float>(data, data + c.getNumEl());
}

}  // namespace

CATCH_TEST_CASE("test Metal rand", "[op][metal]") {
  if (!isOperatorsAvailable(Device::kMetal)) CATCH_SKIP("metal device not available");

  Tensor a = F::rand({100, 100}, DType::kFloat16, Device::getMetal());
  CATCH_REQUIRE(a.getDevice().getType() == Device::kMetal);

  std::vector<float> v = readFloats(a);
  float minVal = *std::min_element(v.begin(), v.end());
  float maxVal = *std::max_element(v.begin(), v.end());
  CATCH_REQUIRE(minVal >= 0.0f);
  CATCH_REQUIRE(maxVal <= 1.0f);
  CATCH_REQUIRE(maxVal > 0.5f);
}

CATCH_TEST_CASE("test Metal randn", "[op][metal]") {
  if (!isOperatorsAvailable(Device::kMetal)) CATCH_SKIP("metal device not available");

  F::manualSeed(Device::getMetal(), 42);
  Tensor a = F::randn({2, 3, 4}, Device::getMetal());
  CATCH_REQUIRE(a.getDevice().getType() == Device::kMetal);
  CATCH_REQUIRE(a.getNumEl() == 24);

  std::vector<float> v = readFloats(a);
  int nanCount = 0;
  for (float x : v) {
    if (std::isnan(x)) ++nanCount;
  }
  CATCH_REQUIRE(nanCount == 0);
}

CATCH_TEST_CASE("test Metal manualSeed reproducibility", "[op][metal]") {
  if (!isOperatorsAvailable(Device::kMetal)) CATCH_SKIP("metal device not available");

  F::manualSeed(Device::getMetal(), 123);
  Tensor a = F::randn({10, 10}, Device::getMetal());

  F::manualSeed(Device::getMetal(), 123);
  Tensor b = F::randn({10, 10}, Device::getMetal());

  CATCH_REQUIRE(F::allClose(a, b, 0, 0));
}

}  // namespace fl
