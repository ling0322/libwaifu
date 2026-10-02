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

CATCH_TEST_CASE("test Metal binary operators", "[op][metal]") {
  if (!isOperatorsAvailable(Device::kMetal)) CATCH_SKIP("metal device not available");

  Tensor a = F::rand(Device::getCpu(), {2, 5, 10}, DType::kFloat);
  Tensor b = F::rand(Device::getCpu(), {5}, DType::kFloat);

  // Non-contiguous views: a transposed and sliced tensor.
  Tensor at = a.transpose(2, 1).slice(1, {1, 9});
  Tensor xt = toMetal(a).transpose(2, 1).slice(1, {1, 9});
  Tensor y = toMetal(b);

  CATCH_REQUIRE(
      cpuOps()->allClose(toCpu(F::add(metalOps(), xt, y)), F::add(cpuOps(), at, b), 5e-3, 5e-3));
  CATCH_REQUIRE(
      cpuOps()->allClose(toCpu(F::sub(metalOps(), xt, y)), F::sub(cpuOps(), at, b), 5e-3, 5e-3));
  CATCH_REQUIRE(
      cpuOps()->allClose(toCpu(F::mul(metalOps(), xt, y)), F::mul(cpuOps(), at, b), 5e-3, 5e-3));
  CATCH_REQUIRE(cpuOps()->allClose(
      toCpu(F::mul(metalOps(), xt, 0.1f)),
      F::mul(cpuOps(), at, 0.1f),
      1e-3,
      1e-4));
  CATCH_REQUIRE(cpuOps()->allClose(
      toCpu(F::div(metalOps(), xt, 4.0f)),
      F::div(cpuOps(), at, 4.0f),
      1e-3,
      1e-4));
}

CATCH_TEST_CASE("test Metal binary operators (contiguous, larger)", "[op][metal]") {
  if (!isOperatorsAvailable(Device::kMetal)) CATCH_SKIP("metal device not available");

  Tensor a = F::rand(Device::getCpu(), {4, 32, 64}, DType::kFloat);
  Tensor b = F::rand(Device::getCpu(), {4, 32, 64}, DType::kFloat);

  CATCH_REQUIRE(cpuOps()->allClose(
      toCpu(F::add(metalOps(), toMetal(a), toMetal(b))),
      F::add(cpuOps(), a, b),
      5e-3,
      5e-3));
  CATCH_REQUIRE(cpuOps()->allClose(
      toCpu(F::sub(metalOps(), toMetal(a), toMetal(b))),
      F::sub(cpuOps(), a, b),
      5e-3,
      5e-3));
  CATCH_REQUIRE(cpuOps()->allClose(
      toCpu(F::mul(metalOps(), toMetal(a), toMetal(b))),
      F::mul(cpuOps(), a, b),
      5e-3,
      5e-3));
}

CATCH_TEST_CASE("test Metal unary operators", "[op][metal]") {
  if (!isOperatorsAvailable(Device::kMetal)) CATCH_SKIP("metal device not available");

  Tensor a = F::rand(Device::getCpu(), {2, 5, 10}, DType::kFloat);
  Tensor at = a.transpose(2, 1).slice(1, {1, 9});
  Tensor xt = toMetal(a).transpose(2, 1).slice(1, {1, 9});

  CATCH_REQUIRE(cpuOps()->allClose(
      toCpu(F::neg(metalOps(), xt)),
      F::neg(cpuOps(), at),
      5e-3,
      5e-3));
  CATCH_REQUIRE(cpuOps()->allClose(
      toCpu(F::exp(metalOps(), xt)),
      F::exp(cpuOps(), at),
      5e-3,
      5e-3));
  // Through exp first, so every input is at least one and log has no zero to fall off; half keeps
  // a relative error, which log turns into an absolute one of about the same size.
  CATCH_REQUIRE(cpuOps()->allClose(
      toCpu(F::log(metalOps(), F::exp(metalOps(), xt))),
      F::log(cpuOps(), F::exp(cpuOps(), at)),
      5e-3,
      5e-3));
  CATCH_REQUIRE(cpuOps()->allClose(
      toCpu(F::sqrt(metalOps(), xt)),
      F::sqrt(cpuOps(), at),
      5e-3,
      5e-3));
  CATCH_REQUIRE(
      cpuOps()->allClose(toCpu(F::sigmoid(metalOps(), xt)), F::sigmoid(cpuOps(), at), 5e-3, 5e-3));
  CATCH_REQUIRE(cpuOps()->allClose(
      toCpu(F::tanh(metalOps(), xt)),
      F::tanh(cpuOps(), at),
      5e-3,
      5e-3));
  CATCH_REQUIRE(cpuOps()->allClose(
      toCpu(F::gelu(metalOps(), xt)),
      F::gelu(cpuOps(), at),
      5e-3,
      5e-3));
  CATCH_REQUIRE(cpuOps()->allClose(
      toCpu(F::silu(metalOps(), xt)),
      F::silu(cpuOps(), at),
      5e-3,
      5e-3));
  CATCH_REQUIRE(cpuOps()->allClose(
      toCpu(F::quickGelu(metalOps(), xt)),
      F::quickGelu(cpuOps(), at),
      5e-3,
      5e-3));
}

CATCH_TEST_CASE("test Metal round and cast to int64", "[op][metal]") {
  if (!isOperatorsAvailable(Device::kMetal)) CATCH_SKIP("metal device not available");

  // Ties to even, as torch.round; half holds every value exactly, so the answers are exact.
  Tensor a = Tensor::create<float>({8}, {-2.5f, -1.5f, -0.5f, 0.5f, 1.5f, 2.5f, 0.6f, 3.7f});
  // Exact, element by element: allClose compares with a strict `<`, so no tolerance says "equal".
  Tensor rounded = F::round(metalOps(), toMetal(a));
  Tensor back = toCpu(rounded);
  const float *values = back.getInternalData()->getData<float>(back.getInternalOffset());
  const float wantRounded[] = {-2.0f, -2.0f, 0.0f, 0.0f, 2.0f, 2.0f, 1.0f, 4.0f};
  for (int i = 0; i < 8; ++i) CATCH_REQUIRE(values[i] == wantRounded[i]);

  // And to int64, through the same astype every Metal cast goes through.
  Tensor ids =
      F::toDevice(metalOps(), F::cast(metalOps(), rounded, DType::kLong), Device::getCpu());
  CATCH_REQUIRE(ids.getDType() == DType::kLong);
  const LongType *data = ids.getInternalData()->getData<LongType>(ids.getInternalOffset());
  const LongType want[] = {-2, -2, 0, 0, 2, 2, 1, 4};
  for (int i = 0; i < 8; ++i) CATCH_REQUIRE(data[i] == want[i]);
}

CATCH_TEST_CASE("test Metal unary operators (larger contiguous)", "[op][metal]") {
  if (!isOperatorsAvailable(Device::kMetal)) CATCH_SKIP("metal device not available");

  Tensor a = F::rand(Device::getCpu(), {4, 32, 64}, DType::kFloat);
  CATCH_REQUIRE(
      cpuOps()->allClose(toCpu(F::silu(metalOps(), toMetal(a))), F::silu(cpuOps(), a), 5e-3, 5e-3));
  CATCH_REQUIRE(
      cpuOps()->allClose(toCpu(F::gelu(metalOps(), toMetal(a))), F::gelu(cpuOps(), a), 5e-3, 5e-3));
  CATCH_REQUIRE(cpuOps()->allClose(
      toCpu(F::sigmoid(metalOps(), toMetal(a))),
      F::sigmoid(cpuOps(), a),
      5e-3,
      5e-3));
}

CATCH_TEST_CASE("test Metal softmax", "[op][metal]") {
  if (!isOperatorsAvailable(Device::kMetal)) CATCH_SKIP("metal device not available");

  for (int lastDim : {8, 64, 1280}) {
    CATCH_INFO("lastDim = " << lastDim);
    Tensor a = F::rand(Device::getCpu(), {4, 6, lastDim}, DType::kFloat);
    CATCH_REQUIRE(cpuOps()->allClose(
        toCpu(F::softmax(metalOps(), toMetal(a))),
        F::softmax(cpuOps(), a),
        5e-3,
        5e-3));
  }
}

}  // namespace fl
