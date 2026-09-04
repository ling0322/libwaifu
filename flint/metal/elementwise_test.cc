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

CATCH_TEST_CASE("test Metal binary operators", "[op][metal]") {
  if (!isOperatorsAvailable(Device::kMetal)) CATCH_SKIP("metal device not available");

  Tensor a = F::rand({2, 5, 10}, DType::kFloat);
  Tensor b = F::rand({5}, DType::kFloat);

  // Non-contiguous views: a transposed and sliced tensor.
  Tensor at = a.transpose(2, 1).slice(1, {1, 9});
  Tensor xt = toMetal(a).transpose(2, 1).slice(1, {1, 9});
  Tensor y = toMetal(b);

  CATCH_REQUIRE(F::allClose(toCpu(F::add(xt, y)), F::add(at, b), 5e-3, 5e-3));
  CATCH_REQUIRE(F::allClose(toCpu(F::sub(xt, y)), F::sub(at, b), 5e-3, 5e-3));
  CATCH_REQUIRE(F::allClose(toCpu(F::mul(xt, y)), F::mul(at, b), 5e-3, 5e-3));
  CATCH_REQUIRE(F::allClose(toCpu(F::mul(xt, 0.1f)), F::mul(at, 0.1f), 1e-3, 1e-4));
  CATCH_REQUIRE(F::allClose(toCpu(F::div(xt, 4.0f)), F::div(at, 4.0f), 1e-3, 1e-4));
}

CATCH_TEST_CASE("test Metal binary operators (contiguous, larger)", "[op][metal]") {
  if (!isOperatorsAvailable(Device::kMetal)) CATCH_SKIP("metal device not available");

  Tensor a = F::rand({4, 32, 64}, DType::kFloat);
  Tensor b = F::rand({4, 32, 64}, DType::kFloat);

  CATCH_REQUIRE(F::allClose(toCpu(F::add(toMetal(a), toMetal(b))), F::add(a, b), 5e-3, 5e-3));
  CATCH_REQUIRE(F::allClose(toCpu(F::sub(toMetal(a), toMetal(b))), F::sub(a, b), 5e-3, 5e-3));
  CATCH_REQUIRE(F::allClose(toCpu(F::mul(toMetal(a), toMetal(b))), F::mul(a, b), 5e-3, 5e-3));
}

CATCH_TEST_CASE("test Metal unary operators", "[op][metal]") {
  if (!isOperatorsAvailable(Device::kMetal)) CATCH_SKIP("metal device not available");

  Tensor a = F::rand({2, 5, 10}, DType::kFloat);
  Tensor at = a.transpose(2, 1).slice(1, {1, 9});
  Tensor xt = toMetal(a).transpose(2, 1).slice(1, {1, 9});

  CATCH_REQUIRE(F::allClose(toCpu(F::neg(xt)), F::neg(at), 5e-3, 5e-3));
  CATCH_REQUIRE(F::allClose(toCpu(F::exp(xt)), F::exp(at), 5e-3, 5e-3));
  CATCH_REQUIRE(F::allClose(toCpu(F::sqrt(xt)), F::sqrt(at), 5e-3, 5e-3));
  CATCH_REQUIRE(F::allClose(toCpu(F::sigmoid(xt)), F::sigmoid(at), 5e-3, 5e-3));
  CATCH_REQUIRE(F::allClose(toCpu(F::tanh(xt)), F::tanh(at), 5e-3, 5e-3));
  CATCH_REQUIRE(F::allClose(toCpu(F::gelu(xt)), F::gelu(at), 5e-3, 5e-3));
  CATCH_REQUIRE(F::allClose(toCpu(F::silu(xt)), F::silu(at), 5e-3, 5e-3));
  CATCH_REQUIRE(F::allClose(toCpu(F::quickGelu(xt)), F::quickGelu(at), 5e-3, 5e-3));
}

CATCH_TEST_CASE("test Metal unary operators (larger contiguous)", "[op][metal]") {
  if (!isOperatorsAvailable(Device::kMetal)) CATCH_SKIP("metal device not available");

  Tensor a = F::rand({4, 32, 64}, DType::kFloat);
  CATCH_REQUIRE(F::allClose(toCpu(F::silu(toMetal(a))), F::silu(a), 5e-3, 5e-3));
  CATCH_REQUIRE(F::allClose(toCpu(F::gelu(toMetal(a))), F::gelu(a), 5e-3, 5e-3));
  CATCH_REQUIRE(F::allClose(toCpu(F::sigmoid(toMetal(a))), F::sigmoid(a), 5e-3, 5e-3));
}

CATCH_TEST_CASE("test Metal softmax", "[op][metal]") {
  if (!isOperatorsAvailable(Device::kMetal)) CATCH_SKIP("metal device not available");

  for (int lastDim : {8, 64, 1280}) {
    CATCH_INFO("lastDim = " << lastDim);
    Tensor a = F::rand({4, 6, lastDim}, DType::kFloat);
    CATCH_REQUIRE(F::allClose(toCpu(F::softmax(toMetal(a))), F::softmax(a), 5e-3, 5e-3));
  }
}

}  // namespace fl
