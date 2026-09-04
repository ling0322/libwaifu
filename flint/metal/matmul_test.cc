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

CATCH_TEST_CASE("test Metal matmul", "[op][metal]") {
  if (!isOperatorsAvailable(Device::kMetal)) CATCH_SKIP("metal device not available");

  Tensor a = F::rand({10, 20}, DType::kFloat);
  Tensor b = F::rand({20, 40}, DType::kFloat);
  CATCH_REQUIRE(
      F::allClose(toCpu(F::matmul(toMetal(a), toMetal(b))), F::matmul(a, b), 5e-2, 5e-2));
}

CATCH_TEST_CASE("test Metal matmul (transposed B)", "[op][metal]") {
  if (!isOperatorsAvailable(Device::kMetal)) CATCH_SKIP("metal device not available");

  // Linear layers store weight as (out, in) and multiply A @ W^T.
  Tensor a = F::rand({8, 64}, DType::kFloat);
  Tensor w = F::rand({128, 64}, DType::kFloat);
  CATCH_REQUIRE(
      F::allClose(
          toCpu(F::matmul(toMetal(a), toMetal(w).transpose(-1, -2))),
          F::matmul(a, w.transpose(-1, -2)),
          5e-2, 5e-2));
}

CATCH_TEST_CASE("test Metal matmul (batched)", "[op][metal]") {
  if (!isOperatorsAvailable(Device::kMetal)) CATCH_SKIP("metal device not available");

  Tensor c = F::rand({5, 10, 20}, DType::kFloat);
  Tensor d = F::rand({40, 20}, DType::kFloat);
  CATCH_REQUIRE(
      F::allClose(
          toCpu(F::matmul(toMetal(c), toMetal(d).transpose(-1, -2))),
          F::matmul(c, d.transpose(-1, -2)),
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
    Tensor a = F::rand({s.m, s.k}, DType::kFloat);
    Tensor b = F::rand({s.n, s.k}, DType::kFloat);
    CATCH_REQUIRE(
        F::allClose(
            toCpu(F::matmul(toMetal(a), toMetal(b).transpose(-1, -2))),
            F::matmul(a, b.transpose(-1, -2)),
            5e-2, 5e-2));
  }
}

}  // namespace fl
