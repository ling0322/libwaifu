#include <cmath>

#include "catch2/catch_amalgamated.hpp"
#include "flint/functional.h"
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
  return F::cast(metalOps(), F::toDevice(metalOps(), a, Device::getMetal()), DType::kFloat16);
}

Tensor toCpu(const Tensor &a) {
  return F::toDevice(metalOps(), F::cast(metalOps(), a, DType::kFloat), Device::getCpu());
}

}  // namespace

CATCH_TEST_CASE("test Metal sum", "[op][metal]") {
  if (!isOperatorsAvailable(Device::kMetal)) CATCH_SKIP("metal device not available");

  Tensor a = F::rand(Device::getCpu(), {4, 6, 8}, DType::kFloat);
  CATCH_REQUIRE(
      cpuOps()->allClose(
          toCpu(F::sum(metalOps(), toMetal(a), -1)), F::sum(cpuOps(), a, -1), 5e-2, 5e-2));
  CATCH_REQUIRE(
      cpuOps()->allClose(
          toCpu(F::sum(metalOps(), toMetal(a), 0)), F::sum(cpuOps(), a, 0), 5e-2, 5e-2));
  CATCH_REQUIRE(
      cpuOps()->allClose(
          toCpu(F::sum(metalOps(), toMetal(a), 1)), F::sum(cpuOps(), a, 1), 5e-2, 5e-2));
}

CATCH_TEST_CASE("test Metal max and min", "[op][metal]") {
  if (!isOperatorsAvailable(Device::kMetal)) CATCH_SKIP("metal device not available");

  Tensor a = F::rand(Device::getCpu(), {3, 5, 7}, DType::kFloat);

  // Over the last dimension, which is all max and min reduce now, on every device.
  CATCH_REQUIRE(cpuOps()->allClose(
      toCpu(F::max(metalOps(), toMetal(a))), F::max(cpuOps(), a), 5e-3, 5e-3));
  CATCH_REQUIRE(cpuOps()->allClose(
      toCpu(F::min(metalOps(), toMetal(a))), F::min(cpuOps(), a), 5e-3, 5e-3));

  // And over everything, through a view of it as one vector, which comes back as (1).
  float metalMax = cpuOps()->elem(toCpu(F::max(metalOps(), toMetal(a).view({3 * 5 * 7}))));
  float metalMin = cpuOps()->elem(toCpu(F::min(metalOps(), toMetal(a).view({3 * 5 * 7}))));

  Tensor cpuFlat = a.view({3 * 5 * 7});
  float cpuMax = cpuOps()->elem(F::max(cpuOps(), cpuFlat));
  float cpuMin = cpuOps()->elem(F::min(cpuOps(), cpuFlat));
  CATCH_REQUIRE(std::fabs(metalMax - cpuMax) < 5e-3f);
  CATCH_REQUIRE(std::fabs(metalMin - cpuMin) < 5e-3f);
}

CATCH_TEST_CASE("test Metal allClose", "[op][metal]") {
  if (!isOperatorsAvailable(Device::kMetal)) CATCH_SKIP("metal device not available");

  Tensor a = F::rand(Device::getCpu(), {4, 8}, DType::kFloat);
  CATCH_REQUIRE(metalOps()->allClose(toMetal(a), toMetal(a), 0, 0));
}

CATCH_TEST_CASE("test Metal cumsum", "[op][metal]") {
  if (!isOperatorsAvailable(Device::kMetal)) CATCH_SKIP("metal device not available");

  // Float16 on the card, so the tolerance is a half's at the ~150 a row of 300 sums to.
  Tensor a = F::rand(Device::getCpu(), {3, 300, 5}, DType::kFloat);
  for (int dim : {0, 1, 2, -1}) {
    CATCH_REQUIRE(cpuOps()->allClose(
        toCpu(F::cumsum(metalOps(), toMetal(a), dim)),
        F::cumsum(
            cpuOps(),
            F::cast(cpuOps(), F::cast(cpuOps(), a, DType::kFloat16), DType::kFloat),
            dim),
        5e-3,
        0.2));
  }
}

}  // namespace fl
