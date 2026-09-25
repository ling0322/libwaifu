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

Tensor toMetal(const Tensor &a) {
  return metalOps()->cast(metalOps()->toDevice(Device::getMetal(), a), DType::kFloat16);
}

Tensor toCpu(const Tensor &a) {
  return metalOps()->toDevice(Device::getCpu(), metalOps()->cast(a, DType::kFloat));
}

}  // namespace

CATCH_TEST_CASE("test Metal sum", "[op][metal]") {
  if (!isOperatorsAvailable(Device::kMetal)) CATCH_SKIP("metal device not available");

  Tensor a = cpuOps()->rand({4, 6, 8}, DType::kFloat);
  CATCH_REQUIRE(
      cpuOps()->allClose(toCpu(metalOps()->sum(toMetal(a), -1)), cpuOps()->sum(a, -1), 5e-2, 5e-2));
  CATCH_REQUIRE(
      cpuOps()->allClose(toCpu(metalOps()->sum(toMetal(a), 0)), cpuOps()->sum(a, 0), 5e-2, 5e-2));
  CATCH_REQUIRE(
      cpuOps()->allClose(toCpu(metalOps()->sum(toMetal(a), 1)), cpuOps()->sum(a, 1), 5e-2, 5e-2));
}

CATCH_TEST_CASE("test Metal max and min", "[op][metal]") {
  if (!isOperatorsAvailable(Device::kMetal)) CATCH_SKIP("metal device not available");

  Tensor a = cpuOps()->rand({3, 5, 7}, DType::kFloat);
  float metalMax = cpuOps()->elem(toCpu(metalOps()->max(toMetal(a))));
  float metalMin = cpuOps()->elem(toCpu(metalOps()->min(toMetal(a))));

  Tensor cpuFlat = a.view({3 * 5 * 7});
  float cpuMax = cpuOps()->elem(cpuOps()->max(cpuFlat));
  float cpuMin = cpuOps()->elem(cpuOps()->min(cpuFlat));
  CATCH_REQUIRE(std::fabs(metalMax - cpuMax) < 5e-3f);
  CATCH_REQUIRE(std::fabs(metalMin - cpuMin) < 5e-3f);
}

CATCH_TEST_CASE("test Metal allClose", "[op][metal]") {
  if (!isOperatorsAvailable(Device::kMetal)) CATCH_SKIP("metal device not available");

  Tensor a = cpuOps()->rand({4, 8}, DType::kFloat);
  CATCH_REQUIRE(metalOps()->allClose(toMetal(a), toMetal(a), 0, 0));
}

CATCH_TEST_CASE("test Metal cumsum", "[op][metal]") {
  if (!isOperatorsAvailable(Device::kMetal)) CATCH_SKIP("metal device not available");

  // Float16 on the card, so the tolerance is a half's at the ~150 a row of 300 sums to.
  Tensor a = cpuOps()->rand({3, 300, 5}, DType::kFloat);
  for (int dim : {0, 1, 2, -1}) {
    CATCH_REQUIRE(cpuOps()->allClose(
        toCpu(metalOps()->cumsum(toMetal(a), dim)),
        cpuOps()->cumsum(cpuOps()->cast(cpuOps()->cast(a, DType::kFloat16), DType::kFloat), dim),
        5e-3,
        0.2));
  }
}

}  // namespace fl
