#include <cmath>

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

std::vector<float> readFloats(const Tensor &a) {
  Tensor c = cpuOps()->contiguous(toCpu(a));
  const float *data = c.getInternalData()->getData<float>(c.getInternalOffset());
  return std::vector<float>(data, data + c.getNumEl());
}

}  // namespace

CATCH_TEST_CASE("test Metal attention", "[op][metal]") {
  if (!isOperatorsAvailable(Device::kMetal)) CATCH_SKIP("metal device not available");

  Tensor q = cpuOps()->rand({2, 4, 8, 16}, DType::kFloat);
  Tensor k = cpuOps()->rand({2, 4, 8, 16}, DType::kFloat);
  Tensor v = cpuOps()->rand({2, 4, 8, 16}, DType::kFloat);

  CATCH_REQUIRE(
      cpuOps()->allClose(
          toCpu(metalOps()->attention(toMetal(q), toMetal(k), toMetal(v), false)),
          cpuOps()->attention(q, k, v, false),
          5e-2, 5e-2));
}

CATCH_TEST_CASE("test Metal attention (head dims)", "[op][metal]") {
  if (!isOperatorsAvailable(Device::kMetal)) CATCH_SKIP("metal device not available");

  for (int headDim : {16, 32, 64, 128}) {
    CATCH_INFO("headDim = " << headDim);
    Tensor q = cpuOps()->rand({1, 4, 8, headDim}, DType::kFloat);
    Tensor k = cpuOps()->rand({1, 4, 8, headDim}, DType::kFloat);
    Tensor v = cpuOps()->rand({1, 4, 8, headDim}, DType::kFloat);

    CATCH_REQUIRE(
        cpuOps()->allClose(
            toCpu(metalOps()->attention(toMetal(q), toMetal(k), toMetal(v), false)),
            cpuOps()->attention(q, k, v, false),
            5e-2, 5e-2));
  }
}

CATCH_TEST_CASE("test Metal attention (SDXL shapes)", "[op][metal]") {
  if (!isOperatorsAvailable(Device::kMetal)) CATCH_SKIP("metal device not available");

  struct Shape { int heads, seqQ, seqKV, dim; };
  Shape shapes[] = {
      {10, 64, 64, 64},
      {20, 32, 32, 64},
      {10, 64, 77, 64},
      {20, 32, 77, 64},
  };

  for (auto &s : shapes) {
    CATCH_INFO("heads=" << s.heads << " seqQ=" << s.seqQ << " seqKV=" << s.seqKV);
    Tensor q = cpuOps()->rand({1, s.heads, s.seqQ, s.dim}, DType::kFloat);
    Tensor k = cpuOps()->rand({1, s.heads, s.seqKV, s.dim}, DType::kFloat);
    Tensor v = cpuOps()->rand({1, s.heads, s.seqKV, s.dim}, DType::kFloat);

    CATCH_REQUIRE(
        cpuOps()->allClose(
            toCpu(metalOps()->attention(toMetal(q), toMetal(k), toMetal(v), false)),
            cpuOps()->attention(q, k, v, false),
            5e-2, 5e-2));
  }
}

CATCH_TEST_CASE("test Metal attention (long sequence, NaN check)", "[op][metal]") {
  if (!isOperatorsAvailable(Device::kMetal)) CATCH_SKIP("metal device not available");

  Tensor q = cpuOps()->rand({1, 1, 4096, 64}, DType::kFloat);
  Tensor k = cpuOps()->rand({1, 1, 4096, 64}, DType::kFloat);
  Tensor v = cpuOps()->rand({1, 1, 4096, 64}, DType::kFloat);

  Tensor got = metalOps()->attention(toMetal(q), toMetal(k), toMetal(v), false);
  CATCH_REQUIRE(got.getShape() == std::vector<int>{1, 1, 4096, 64});

  std::vector<float> data = readFloats(got);
  int nanCount = 0;
  for (float x : data) {
    if (std::isnan(x)) ++nanCount;
  }
  CATCH_REQUIRE(nanCount == 0);
}

}  // namespace fl
