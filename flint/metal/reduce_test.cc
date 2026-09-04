#include <cmath>

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

}  // namespace

CATCH_TEST_CASE("test Metal sum", "[op][metal]") {
  if (!isOperatorsAvailable(Device::kMetal)) CATCH_SKIP("metal device not available");

  Tensor a = F::rand({4, 6, 8}, DType::kFloat);
  CATCH_REQUIRE(F::allClose(toCpu(F::sum(toMetal(a), -1)), F::sum(a, -1), 5e-2, 5e-2));
  CATCH_REQUIRE(F::allClose(toCpu(F::sum(toMetal(a), 0)), F::sum(a, 0), 5e-2, 5e-2));
  CATCH_REQUIRE(F::allClose(toCpu(F::sum(toMetal(a), 1)), F::sum(a, 1), 5e-2, 5e-2));
}

CATCH_TEST_CASE("test Metal max and min", "[op][metal]") {
  if (!isOperatorsAvailable(Device::kMetal)) CATCH_SKIP("metal device not available");

  Tensor a = F::rand({3, 5, 7}, DType::kFloat);
  float metalMax = F::elem(toCpu(F::max(toMetal(a))));
  float metalMin = F::elem(toCpu(F::min(toMetal(a))));

  Tensor cpuFlat = a.view({3 * 5 * 7});
  float cpuMax = F::elem(F::max(cpuFlat));
  float cpuMin = F::elem(F::min(cpuFlat));
  CATCH_REQUIRE(std::fabs(metalMax - cpuMax) < 5e-3f);
  CATCH_REQUIRE(std::fabs(metalMin - cpuMin) < 5e-3f);
}

CATCH_TEST_CASE("test Metal allClose", "[op][metal]") {
  if (!isOperatorsAvailable(Device::kMetal)) CATCH_SKIP("metal device not available");

  Tensor a = F::rand({4, 8}, DType::kFloat);
  CATCH_REQUIRE(F::allClose(toMetal(a), toMetal(a), 0, 0));
}

}  // namespace fl
