// The MIT License (MIT)
//
// Copyright (c) 2023 Xiaoyang Chen
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software
// and associated documentation files (the "Software"), to deal in the Software without
// restriction, including without limitation the rights to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all copies or
// substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
// BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
// DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include <algorithm>
#include <cmath>
#include <vector>

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

namespace op {
namespace metal {
namespace {

Tensor toMetal(const Tensor &a) {
  return F::cast(metalOps(), F::toDevice(metalOps(), a, Device::getMetal()), DType::kFloat16);
}

Tensor toCpu(const Tensor &a) {
  return F::toDevice(metalOps(), F::cast(metalOps(), a, DType::kFloat), Device::getCpu());
}

std::vector<float> readFloats(const Tensor &a) {
  Tensor c = F::contiguous(cpuOps(), toCpu(a));
  const float *data = c.getInternalData()->getData<float>(c.getInternalOffset());
  return std::vector<float>(data, data + c.getNumEl());
}

}  // namespace

CATCH_TEST_CASE("test Metal covers what SDXL calls", "[op][metal]") {
  if (!isOperatorsAvailable(Device::kMetal)) CATCH_SKIP("metal device not available");

  Tensor a = F::rand(Device::getCpu(), {2, 4, 6}, DType::kFloat);
  Tensor b = F::rand(Device::getCpu(), {2, 4, 6}, DType::kFloat);
  Tensor xa = toMetal(a);
  Tensor xb = toMetal(b);

  CATCH_SECTION("cat") {
    CATCH_REQUIRE(cpuOps()->allClose(
        toCpu(F::cat(metalOps(), xa, xb, -1)),
        F::cat(cpuOps(), a, b, -1),
        5e-3,
        5e-3));
    CATCH_REQUIRE(cpuOps()->allClose(
        toCpu(F::cat(metalOps(), xa, xb, 1)),
        F::cat(cpuOps(), a, b, 1),
        5e-3,
        5e-3));
  }

  CATCH_SECTION("contiguous of a view") {
    CATCH_REQUIRE(
        cpuOps()->allClose(
            toCpu(F::contiguous(metalOps(), xa.transpose(0, 2))),
            F::contiguous(cpuOps(), a.transpose(0, 2)),
            5e-3,
            5e-3));
  }

  CATCH_SECTION("view and unsqueeze") {
    CATCH_REQUIRE(cpuOps()->allClose(toCpu(xa.view({2, 24})), a.view({2, 24}), 5e-3, 5e-3));
    CATCH_REQUIRE(cpuOps()->allClose(toCpu(xa.unsqueeze(1)), a.unsqueeze(1), 5e-3, 5e-3));
  }

  CATCH_SECTION("randn and manualSeed") {
    metalOps()->manualSeed(42);
    Tensor noise = F::randNormal(Device(Device::kMetal), {2, 3, 4});
    CATCH_REQUIRE(noise.getDevice().getType() == Device::kMetal);
    CATCH_REQUIRE(noise.getNumEl() == 24);
  }
}

CATCH_TEST_CASE("test Metal at VAE scale in fp16", "[op][metal][probe]") {
  if (!isOperatorsAvailable(Device::kMetal)) CATCH_SKIP("metal device not available");

  auto countNaN = [](const Tensor &t) {
    std::vector<float> v = readFloats(t);
    return std::count_if(v.begin(), v.end(), [](float x) { return std::isnan(x); });
  };

  CATCH_SECTION("groupNorm over a large plane") {
    Tensor a = F::rand(Device::getCpu(), {1, 128, 256, 256}, DType::kFloat);
    Tensor w = F::rand(Device::getCpu(), {128}, DType::kFloat);
    Tensor b = F::rand(Device::getCpu(), {128}, DType::kFloat);

    Tensor got = F::groupNorm(metalOps(), toMetal(a), toMetal(w), toMetal(b), 32, 1e-5);
    CATCH_INFO("groupNorm NaN count " << countNaN(got));
    CATCH_REQUIRE(countNaN(got) == 0);
  }

  CATCH_SECTION("layerNorm over a wide row") {
    Tensor a = F::rand(Device::getCpu(), {2, 64, 8192}, DType::kFloat);
    Tensor w = F::rand(Device::getCpu(), {8192}, DType::kFloat);
    Tensor b = F::rand(Device::getCpu(), {8192}, DType::kFloat);
    Tensor got = F::layerNorm(metalOps(), toMetal(a), toMetal(w), toMetal(b), 1e-5);
    CATCH_INFO("layerNorm NaN count " << countNaN(got));
    CATCH_REQUIRE(countNaN(got) == 0);
  }

  CATCH_SECTION("attention over a long sequence") {
    Tensor q = F::rand(Device::getCpu(), {1, 1, 4096, 64}, DType::kFloat);
    Tensor k = F::rand(Device::getCpu(), {1, 1, 4096, 64}, DType::kFloat);
    Tensor v = F::rand(Device::getCpu(), {1, 1, 4096, 64}, DType::kFloat);

    Tensor got = F::attention(metalOps(), toMetal(q), toMetal(k), toMetal(v), false);
    CATCH_INFO("attention NaN count " << countNaN(got));
    CATCH_REQUIRE(countNaN(got) == 0);
  }
}

}  // namespace metal
}  // namespace op
}  // namespace fl
