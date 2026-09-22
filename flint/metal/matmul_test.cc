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

CATCH_TEST_CASE("test Metal matmul", "[op][metal]") {
  if (!isOperatorsAvailable(Device::kMetal)) CATCH_SKIP("metal device not available");

  Tensor a = cpuOps()->rand({10, 20}, DType::kFloat);
  Tensor b = cpuOps()->rand({20, 40}, DType::kFloat);
  CATCH_REQUIRE(
      cpuOps()->allClose(
          toCpu(metalOps()->matmul(toMetal(a), toMetal(b))),
          cpuOps()->matmul(a, b),
          5e-2,
          5e-2));
}

CATCH_TEST_CASE("test Metal matmul (transposed B)", "[op][metal]") {
  if (!isOperatorsAvailable(Device::kMetal)) CATCH_SKIP("metal device not available");

  // Linear layers store weight as (out, in) and multiply A @ W^T.
  Tensor a = cpuOps()->rand({8, 64}, DType::kFloat);
  Tensor w = cpuOps()->rand({128, 64}, DType::kFloat);
  CATCH_REQUIRE(
      cpuOps()->allClose(
          toCpu(metalOps()->matmul(toMetal(a), toMetal(w).transpose(-1, -2))),
          cpuOps()->matmul(a, w.transpose(-1, -2)),
          5e-2, 5e-2));
}

CATCH_TEST_CASE("test Metal matmul (batched)", "[op][metal]") {
  if (!isOperatorsAvailable(Device::kMetal)) CATCH_SKIP("metal device not available");

  Tensor c = cpuOps()->rand({5, 10, 20}, DType::kFloat);
  Tensor d = cpuOps()->rand({40, 20}, DType::kFloat);
  CATCH_REQUIRE(
      cpuOps()->allClose(
          toCpu(metalOps()->matmul(toMetal(c), toMetal(d).transpose(-1, -2))),
          cpuOps()->matmul(c, d.transpose(-1, -2)),
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
    Tensor a = cpuOps()->rand({s.m, s.k}, DType::kFloat);
    Tensor b = cpuOps()->rand({s.n, s.k}, DType::kFloat);
    CATCH_REQUIRE(
        cpuOps()->allClose(
            toCpu(metalOps()->matmul(toMetal(a), toMetal(b).transpose(-1, -2))),
            cpuOps()->matmul(a, b.transpose(-1, -2)),
            5e-2, 5e-2));
  }
}

}  // namespace fl
