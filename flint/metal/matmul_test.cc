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

CATCH_TEST_CASE("test Metal matmul", "[op][metal]") {
  if (!isOperatorsAvailable(Device::kMetal)) CATCH_SKIP("metal device not available");

  Tensor a = F::rand(Device::getCpu(), {10, 20}, DType::kFloat);
  Tensor b = F::rand(Device::getCpu(), {20, 40}, DType::kFloat);
  CATCH_REQUIRE(
      cpuOps()->allClose(
          toCpu(F::matmul(metalOps(), toMetal(a), toMetal(b))),
          F::matmul(cpuOps(), a, b),
          5e-2,
          5e-2));
}

CATCH_TEST_CASE("test Metal matmul (transposed B)", "[op][metal]") {
  if (!isOperatorsAvailable(Device::kMetal)) CATCH_SKIP("metal device not available");

  // Linear layers store weight as (out, in) and multiply A @ W^T.
  Tensor a = F::rand(Device::getCpu(), {8, 64}, DType::kFloat);
  Tensor w = F::rand(Device::getCpu(), {128, 64}, DType::kFloat);
  CATCH_REQUIRE(
      cpuOps()->allClose(
          toCpu(F::matmul(metalOps(), toMetal(a), toMetal(w).transpose(-1, -2))),
          F::matmul(cpuOps(), a, w.transpose(-1, -2)),
          5e-2, 5e-2));
}

CATCH_TEST_CASE("test Metal matmul (batched)", "[op][metal]") {
  if (!isOperatorsAvailable(Device::kMetal)) CATCH_SKIP("metal device not available");

  Tensor c = F::rand(Device::getCpu(), {5, 10, 20}, DType::kFloat);
  Tensor d = F::rand(Device::getCpu(), {40, 20}, DType::kFloat);
  CATCH_REQUIRE(
      cpuOps()->allClose(
          toCpu(F::matmul(metalOps(), toMetal(c), toMetal(d).transpose(-1, -2))),
          F::matmul(cpuOps(), c, d.transpose(-1, -2)),
          5e-2, 5e-2));
}

CATCH_TEST_CASE("test Metal matmul (SDXL shapes)", "[op][metal]") {
  if (!isOperatorsAvailable(Device::kMetal)) CATCH_SKIP("metal device not available");

  struct Shape { int m, n, k; };
  const Shape shapes[] = {
      {1024, 10240, 1280},
      {1024, 1280, 5120},
      {4096, 5120, 640},
      {4096, 640, 2560},
      {77, 2560, 2048},
  };

  for (const Shape &s : shapes) {
    CATCH_INFO("shape " << s.m << "x" << s.n << "x" << s.k);
    Tensor a = F::rand(Device::getCpu(), {s.m, s.k}, DType::kFloat);
    Tensor b = F::rand(Device::getCpu(), {s.n, s.k}, DType::kFloat);
    CATCH_REQUIRE(
        cpuOps()->allClose(
            toCpu(F::matmul(metalOps(), toMetal(a), toMetal(b).transpose(-1, -2))),
            F::matmul(cpuOps(), a, b.transpose(-1, -2)),
            5e-2, 5e-2));
  }
}

}  // namespace fl
