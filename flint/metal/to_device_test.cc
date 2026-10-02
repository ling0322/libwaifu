#include "catch2/catch_amalgamated.hpp"
#include "flint/test_functional.h"
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

}  // namespace


CATCH_TEST_CASE("test Metal device transfer round trip", "[op][metal]") {
  if (!isOperatorsAvailable(Device::kMetal)) CATCH_SKIP("metal device not available");

  Tensor a = F::rand(Device::getCpu(), {3, 7, 5}, DType::kFloat);
  CATCH_REQUIRE(cpuOps()->allClose(
      F::toDevice(metalOps(), F::toDevice(metalOps(), a, Device::getMetal()), Device::getCpu()),
      a));
}

CATCH_TEST_CASE("test Metal device transfer (fp32 -> fp16 -> fp32)", "[op][metal]") {
  if (!isOperatorsAvailable(Device::kMetal)) CATCH_SKIP("metal device not available");

  // The round trip every operator test relies on.
  Tensor a = F::rand(Device::getCpu(), {3, 7, 5}, DType::kFloat);
  Tensor metalHalf =
      F::cast(metalOps(), F::toDevice(metalOps(), a, Device::getMetal()), DType::kFloat16);
  Tensor back =
      F::toDevice(metalOps(), F::cast(metalOps(), metalHalf, DType::kFloat), Device::getCpu());
  CATCH_REQUIRE(cpuOps()->allClose(back, a, 5e-3, 5e-3));
}

CATCH_TEST_CASE("test Metal device transfer (various shapes)", "[op][metal]") {
  if (!isOperatorsAvailable(Device::kMetal)) CATCH_SKIP("metal device not available");

  for (auto shape : std::vector<std::vector<int>>{{1}, {100}, {4, 8}, {2, 3, 5, 7}}) {
    Tensor a = F::rand(Device::getCpu(), shape, DType::kFloat);
    CATCH_REQUIRE(cpuOps()->allClose(
        F::toDevice(metalOps(), F::toDevice(metalOps(), a, Device::getMetal()), Device::getCpu()),
        a));
  }
}

CATCH_TEST_CASE("test Metal device transfer (contiguous after transpose)", "[op][metal]") {
  if (!isOperatorsAvailable(Device::kMetal)) CATCH_SKIP("metal device not available");

  Tensor a = F::rand(Device::getCpu(), {4, 8, 16}, DType::kFloat);
  Tensor at = F::contiguous(cpuOps(), a.transpose(2, 1));
  Tensor metalT = F::toDevice(metalOps(), at, Device::getMetal());
  Tensor back = F::toDevice(metalOps(), metalT, Device::getCpu());
  CATCH_REQUIRE(cpuOps()->allClose(back, at, 1e-6, 1e-6));
}

CATCH_TEST_CASE("test Metal device transfer (a slice away from the start)", "[op][metal]") {
  if (!isOperatorsAvailable(Device::kMetal)) CATCH_SKIP("metal device not available");

  // Contiguous, but at an offset into a larger storage: exactly the slice's elements move, from
  // where it starts, both ways.
  Tensor a = F::rand(Device::getCpu(), {6, 8}, DType::kFloat);
  Tensor rows = a.slice(0, {2, 5});
  Tensor metalRows = F::toDevice(metalOps(), rows, Device::getMetal());
  CATCH_REQUIRE(metalRows.getNumEl() == 3 * 8);
  CATCH_REQUIRE(cpuOps()->allClose(F::toDevice(metalOps(), metalRows, Device::getCpu()), rows));

  Tensor metalA = F::toDevice(metalOps(), a, Device::getMetal());
  CATCH_REQUIRE(cpuOps()->allClose(
      F::toDevice(metalOps(), metalA.slice(0, {2, 5}), Device::getCpu()),
      rows));
}

}  // namespace fl
