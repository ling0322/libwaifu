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

CATCH_TEST_CASE("test Metal binary operators", "[op][metal]") {
  if (!isOperatorsAvailable(Device::kMetal)) CATCH_SKIP("metal device not available");

  Tensor a = cpuOps()->rand({2, 5, 10}, DType::kFloat);
  Tensor b = cpuOps()->rand({5}, DType::kFloat);

  // Non-contiguous views: a transposed and sliced tensor.
  Tensor at = a.transpose(2, 1).slice(1, {1, 9});
  Tensor xt = toMetal(a).transpose(2, 1).slice(1, {1, 9});
  Tensor y = toMetal(b);

  CATCH_REQUIRE(
      cpuOps()->allClose(toCpu(metalOps()->add(xt, y)), cpuOps()->add(at, b), 5e-3, 5e-3));
  CATCH_REQUIRE(
      cpuOps()->allClose(toCpu(metalOps()->sub(xt, y)), cpuOps()->sub(at, b), 5e-3, 5e-3));
  CATCH_REQUIRE(
      cpuOps()->allClose(toCpu(metalOps()->mul(xt, y)), cpuOps()->mul(at, b), 5e-3, 5e-3));
  CATCH_REQUIRE(
      cpuOps()->allClose(toCpu(metalOps()->mul(xt, 0.1f)), cpuOps()->mul(at, 0.1f), 1e-3, 1e-4));
  CATCH_REQUIRE(
      cpuOps()->allClose(toCpu(metalOps()->div(xt, 4.0f)), cpuOps()->div(at, 4.0f), 1e-3, 1e-4));
}

CATCH_TEST_CASE("test Metal binary operators (contiguous, larger)", "[op][metal]") {
  if (!isOperatorsAvailable(Device::kMetal)) CATCH_SKIP("metal device not available");

  Tensor a = cpuOps()->rand({4, 32, 64}, DType::kFloat);
  Tensor b = cpuOps()->rand({4, 32, 64}, DType::kFloat);

  CATCH_REQUIRE(cpuOps()->allClose(
      toCpu(metalOps()->add(toMetal(a), toMetal(b))),
      cpuOps()->add(a, b),
      5e-3,
      5e-3));
  CATCH_REQUIRE(cpuOps()->allClose(
      toCpu(metalOps()->sub(toMetal(a), toMetal(b))),
      cpuOps()->sub(a, b),
      5e-3,
      5e-3));
  CATCH_REQUIRE(cpuOps()->allClose(
      toCpu(metalOps()->mul(toMetal(a), toMetal(b))),
      cpuOps()->mul(a, b),
      5e-3,
      5e-3));
}

CATCH_TEST_CASE("test Metal unary operators", "[op][metal]") {
  if (!isOperatorsAvailable(Device::kMetal)) CATCH_SKIP("metal device not available");

  Tensor a = cpuOps()->rand({2, 5, 10}, DType::kFloat);
  Tensor at = a.transpose(2, 1).slice(1, {1, 9});
  Tensor xt = toMetal(a).transpose(2, 1).slice(1, {1, 9});

  CATCH_REQUIRE(cpuOps()->allClose(toCpu(metalOps()->neg(xt)), cpuOps()->neg(at), 5e-3, 5e-3));
  CATCH_REQUIRE(cpuOps()->allClose(toCpu(metalOps()->exp(xt)), cpuOps()->exp(at), 5e-3, 5e-3));
  // Through exp first, so every input is at least one and log has no zero to fall off; half keeps
  // a relative error, which log turns into an absolute one of about the same size.
  CATCH_REQUIRE(cpuOps()->allClose(
      toCpu(metalOps()->log(metalOps()->exp(xt))),
      cpuOps()->log(cpuOps()->exp(at)),
      5e-3,
      5e-3));
  CATCH_REQUIRE(cpuOps()->allClose(toCpu(metalOps()->sqrt(xt)), cpuOps()->sqrt(at), 5e-3, 5e-3));
  CATCH_REQUIRE(
      cpuOps()->allClose(toCpu(metalOps()->sigmoid(xt)), cpuOps()->sigmoid(at), 5e-3, 5e-3));
  CATCH_REQUIRE(cpuOps()->allClose(toCpu(metalOps()->tanh(xt)), cpuOps()->tanh(at), 5e-3, 5e-3));
  CATCH_REQUIRE(cpuOps()->allClose(toCpu(metalOps()->gelu(xt)), cpuOps()->gelu(at), 5e-3, 5e-3));
  CATCH_REQUIRE(cpuOps()->allClose(toCpu(metalOps()->silu(xt)), cpuOps()->silu(at), 5e-3, 5e-3));
  CATCH_REQUIRE(
      cpuOps()->allClose(toCpu(metalOps()->quickGelu(xt)), cpuOps()->quickGelu(at), 5e-3, 5e-3));
}

CATCH_TEST_CASE("test Metal round and cast to int64", "[op][metal]") {
  if (!isOperatorsAvailable(Device::kMetal)) CATCH_SKIP("metal device not available");

  // Ties to even, as torch.round; half holds every value exactly, so the answers are exact.
  Tensor a = Tensor::create<float>({8}, {-2.5f, -1.5f, -0.5f, 0.5f, 1.5f, 2.5f, 0.6f, 3.7f});
  // Exact, element by element: allClose compares with a strict `<`, so no tolerance says "equal".
  Tensor rounded = metalOps()->round(toMetal(a));
  Tensor back = toCpu(rounded);
  const float *values = back.getInternalData()->getData<float>(back.getInternalOffset());
  const float wantRounded[] = {-2.0f, -2.0f, 0.0f, 0.0f, 2.0f, 2.0f, 1.0f, 4.0f};
  for (int i = 0; i < 8; ++i) CATCH_REQUIRE(values[i] == wantRounded[i]);

  // And to int64, through the same astype every Metal cast goes through.
  Tensor ids = metalOps()->toDevice(Device::getCpu(), metalOps()->cast(rounded, DType::kLong));
  CATCH_REQUIRE(ids.getDType() == DType::kLong);
  const LongType *data = ids.getInternalData()->getData<LongType>(ids.getInternalOffset());
  const LongType want[] = {-2, -2, 0, 0, 2, 2, 1, 4};
  for (int i = 0; i < 8; ++i) CATCH_REQUIRE(data[i] == want[i]);
}

CATCH_TEST_CASE("test Metal unary operators (larger contiguous)", "[op][metal]") {
  if (!isOperatorsAvailable(Device::kMetal)) CATCH_SKIP("metal device not available");

  Tensor a = cpuOps()->rand({4, 32, 64}, DType::kFloat);
  CATCH_REQUIRE(
      cpuOps()->allClose(toCpu(metalOps()->silu(toMetal(a))), cpuOps()->silu(a), 5e-3, 5e-3));
  CATCH_REQUIRE(
      cpuOps()->allClose(toCpu(metalOps()->gelu(toMetal(a))), cpuOps()->gelu(a), 5e-3, 5e-3));
  CATCH_REQUIRE(
      cpuOps()->allClose(toCpu(metalOps()->sigmoid(toMetal(a))), cpuOps()->sigmoid(a), 5e-3, 5e-3));
}

CATCH_TEST_CASE("test Metal softmax", "[op][metal]") {
  if (!isOperatorsAvailable(Device::kMetal)) CATCH_SKIP("metal device not available");

  for (int lastDim : {8, 64, 1280}) {
    CATCH_INFO("lastDim = " << lastDim);
    Tensor a = cpuOps()->rand({4, 6, lastDim}, DType::kFloat);
    CATCH_REQUIRE(cpuOps()->allClose(
        toCpu(metalOps()->softmax(toMetal(a))),
        cpuOps()->softmax(a),
        5e-3,
        5e-3));
  }
}

}  // namespace fl
