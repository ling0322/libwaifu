#include "catch2/catch_amalgamated.hpp"
#include "flint/device.h"
#include "flint/functional.h"
#include "flint/operators.h"

namespace fl {

CATCH_TEST_CASE("test Metal device transfer round trip", "[op][metal]") {
  if (!isOperatorsAvailable(Device::kMetal)) CATCH_SKIP("metal device not available");

  Tensor a = F::rand({3, 7, 5}, DType::kFloat);
  CATCH_REQUIRE(F::allClose(F::toDevice(Device::getCpu(), F::toDevice(Device::getMetal(), a)), a));
}

CATCH_TEST_CASE("test Metal device transfer (fp32 -> fp16 -> fp32)", "[op][metal]") {
  if (!isOperatorsAvailable(Device::kMetal)) CATCH_SKIP("metal device not available");

  // The round trip every operator test relies on.
  Tensor a = F::rand({3, 7, 5}, DType::kFloat);
  Tensor metalHalf = F::cast(F::toDevice(Device::getMetal(), a), DType::kFloat16);
  Tensor back = F::toDevice(Device::getCpu(), F::cast(metalHalf, DType::kFloat));
  CATCH_REQUIRE(F::allClose(back, a, 5e-3, 5e-3));
}

CATCH_TEST_CASE("test Metal device transfer (various shapes)", "[op][metal]") {
  if (!isOperatorsAvailable(Device::kMetal)) CATCH_SKIP("metal device not available");

  for (auto shape : std::vector<std::vector<int>>{{1}, {100}, {4, 8}, {2, 3, 5, 7}}) {
    Tensor a = F::rand(shape, DType::kFloat);
    CATCH_REQUIRE(F::allClose(F::toDevice(Device::getCpu(), F::toDevice(Device::getMetal(), a)), a));
  }
}

CATCH_TEST_CASE("test Metal device transfer (contiguous after transpose)", "[op][metal]") {
  if (!isOperatorsAvailable(Device::kMetal)) CATCH_SKIP("metal device not available");

  Tensor a = F::rand({4, 8, 16}, DType::kFloat);
  Tensor at = F::contiguous(a.transpose(2, 1));
  Tensor metalT = F::toDevice(Device::getMetal(), at);
  Tensor back = F::toDevice(Device::getCpu(), metalT);
  CATCH_REQUIRE(F::allClose(back, at, 1e-6, 1e-6));
}

}  // namespace fl
