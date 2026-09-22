#include <cmath>

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

Tensor toCpu(const Tensor &a) {
  return metalOps()->toDevice(Device::getCpu(), metalOps()->cast(a, DType::kFloat));
}

std::vector<float> readFloats(const Tensor &a) {
  Tensor c = cpuOps()->contiguous(toCpu(a));
  const float *data = c.getInternalData()->getData<float>(c.getInternalOffset());
  return std::vector<float>(data, data + c.getNumEl());
}

}  // namespace

CATCH_TEST_CASE("test Metal rand", "[op][metal]") {
  if (!isOperatorsAvailable(Device::kMetal)) CATCH_SKIP("metal device not available");

  Tensor a = metalOps()->rand({100, 100}, DType::kFloat16);
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

  metalOps()->manualSeed(42);
  Tensor a = metalOps()->randNormal({2, 3, 4});
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

  metalOps()->manualSeed(123);
  Tensor a = metalOps()->randNormal({10, 10});

  metalOps()->manualSeed(123);
  Tensor b = metalOps()->randNormal({10, 10});

  CATCH_REQUIRE(metalOps()->allClose(a, b, 0, 0));
}

}  // namespace fl
