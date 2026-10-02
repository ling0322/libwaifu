// The MIT License (MIT)
//
// Copyright (c) 2026 Xiaoyang Chen
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
// of the Software, and to permit persons to whom the Software is furnished to do
// so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <algorithm>
#include <cstddef>
#include <vector>

#include "catch2/catch_amalgamated.hpp"
#include "flint/test_functional.h"
#include "flint/device.h"
#include "flint/operators.h"

namespace fl {

namespace {

/// The CUDA operators, which the calls here that run on CUDA are asked of.
Operators *cudaOps() {
  return getOperators(Device::kCuda);
}

/// The CPU operators, which the calls here that run on CPU are asked of.
Operators *cpuOps() {
  return getOperators(Device::kCpu);
}

Tensor toCuda(const Tensor &a) {
  return F::cast(cudaOps(), F::toDevice(cudaOps(), a, Device::getCuda()), DType::kFloat16);
}

Tensor toCpu(const Tensor &a) {
  return F::toDevice(cudaOps(), F::cast(cudaOps(), a, DType::kFloat), Device::getCpu());
}

bool equalLong(Tensor a, Tensor b) {
  a.throwIfInvalidShape(b.getShape(), "equalLong");

  const LongType *pa = a.getInternalData()->getData<LongType>(a.getInternalOffset());
  const LongType *pb = b.getInternalData()->getData<LongType>(b.getInternalOffset());
  return std::equal(pa, pa + a.getNumEl(), pb);
}

}  // namespace

CATCH_TEST_CASE("test CUDA scalar operators", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  Tensor a = F::rand(Device::getCpu(), {2, 5, 10}, DType::kFloat);
  CATCH_REQUIRE(cpuOps()->allClose(
      toCpu(F::div(cudaOps(), toCuda(a), 8.0f)),
      F::mul(cpuOps(), a, 1.0f / 8.0f),
      5e-3));
  CATCH_REQUIRE(cpuOps()->allClose(toCpu(F::square(cudaOps(), toCuda(a))), F::mul(cpuOps(), a, a), 5e-3));

  Tensor ids = Tensor::create<LongType>({2, 3}, {0, 1, 2, 3, 4, 5});
  Tensor mod = F::toDevice(cudaOps(), F::mod(cudaOps(), F::toDevice(cudaOps(), ids, Device::getCuda()), 3), Device::getCpu());
  CATCH_REQUIRE(equalLong(mod, Tensor::create<LongType>({2, 3}, {0, 1, 2, 0, 1, 2})));

  CATCH_REQUIRE(cudaOps()->elem(toCuda(Tensor::create<float>({1}, {1.5f}))) == 1.5f);
}

CATCH_TEST_CASE("test CUDA scalar operators (strided ranks)", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  // The generic kernel is instantiated per rank, 1 through 4. A transposed view of each rank
  // walks all four instantiations; anything higher would fall off the dispatch chain.
  Tensor a4 = F::rand(Device::getCpu(), {2, 3, 4, 5}, DType::kFloat);
  Tensor a3 = F::rand(Device::getCpu(), {2, 3, 4}, DType::kFloat);
  Tensor a2 = F::rand(Device::getCpu(), {4, 5}, DType::kFloat);

  struct Case {
    Tensor host;
    Tensor device;
  };
  std::vector<Case> cases = {
      // rank 1: a row taken out of a transposed matrix keeps its parent's stride.
      {a2.transpose(0, 1).subtensor(0), toCuda(a2).transpose(0, 1).subtensor(0)},
      {a2.transpose(0, 1), toCuda(a2).transpose(0, 1)},
      {a3.transpose(0, 2), toCuda(a3).transpose(0, 2)},
      {a4.transpose(1, 3), toCuda(a4).transpose(1, 3)},
  };

  for (size_t i = 0; i < cases.size(); ++i) {
    const Case &c = cases[i];
    CATCH_INFO("case " << i << ", rank " << c.host.getDim());
    CATCH_REQUIRE(!c.device.isContiguous());
    CATCH_REQUIRE(cpuOps()->allClose(
        toCpu(F::mul(cudaOps(), c.device, 2.0f)),
        F::mul(cpuOps(), c.host, 2.0f),
        5e-3));
    CATCH_REQUIRE(cpuOps()->allClose(
        toCpu(F::div(cudaOps(), c.device, 4.0f)),
        F::mul(cpuOps(), c.host, 0.25f),
        5e-3));
    CATCH_REQUIRE(cpuOps()->allClose(
        toCpu(F::square(cudaOps(), c.device)),
        F::mul(cpuOps(), c.host, c.host),
        5e-3));
  }
}

CATCH_TEST_CASE("test CUDA scalar operators (identity and zero scalars)", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  Tensor a = F::rand(Device::getCpu(), {3, 7}, DType::kFloat);
  Tensor x = toCuda(a);

  CATCH_REQUIRE(cpuOps()->allClose(toCpu(F::mul(cudaOps(), x, 1.0f)), a, 5e-3));
  CATCH_REQUIRE(cpuOps()->allClose(toCpu(F::div(cudaOps(), x, 1.0f)), a, 5e-3));
  CATCH_REQUIRE(
      cpuOps()->allClose(toCpu(F::mul(cudaOps(), x, 0.0f)), F::zeros(Device::getCpu(), {3, 7}, DType::kFloat)));
  // negative scalars flip the sign rather than the magnitude.
  CATCH_REQUIRE(cpuOps()->allClose(toCpu(F::mul(cudaOps(), x, -1.0f)), F::mul(cpuOps(), a, -1.0f), 5e-3));
}

CATCH_TEST_CASE("test CUDA scalar mod", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  // Values on both sides of the divisor, plus exact multiples where the remainder is 0.
  Tensor ids = Tensor::create<LongType>({2, 4}, {0, 3, 4, 5, 7, 8, 99, 100});
  Tensor mod = F::toDevice(cudaOps(), F::mod(cudaOps(), F::toDevice(cudaOps(), ids, Device::getCuda()), 4), Device::getCpu());
  CATCH_REQUIRE(
      equalLong(mod, Tensor::create<LongType>({2, 4}, {0, 3, 0, 1, 3, 0, 3, 0})));

  // a divisor of 1 leaves nothing behind.
  Tensor one = F::toDevice(cudaOps(), F::mod(cudaOps(), F::toDevice(cudaOps(), ids, Device::getCuda()), 1), Device::getCpu());
  CATCH_REQUIRE(equalLong(one, Tensor::create<LongType>({2, 4}, {0, 0, 0, 0, 0, 0, 0, 0})));
}

CATCH_TEST_CASE("test CUDA elem", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  // elem reads a one-element tensor back to the host; the sign and zero must survive the trip.
  CATCH_REQUIRE(cudaOps()->elem(toCuda(Tensor::create<float>({1}, {0.0f}))) == 0.0f);
  CATCH_REQUIRE(cudaOps()->elem(toCuda(Tensor::create<float>({1}, {-2.5f}))) == -2.5f);

  // it also reads the single element of a 1x1 tensor that came out of an operator.
  Tensor scaled = F::mul(cudaOps(), toCuda(Tensor::create<float>({1}, {3.0f})), 0.5f);
  CATCH_REQUIRE(cudaOps()->elem(scaled) == 1.5f);
}

}  // namespace fl
