// The MIT License (MIT)
//
// Copyright (c) 2026 Xiaoyang Chen
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

#include "catch2/catch_amalgamated.hpp"
#include "flint/device.h"
#include "flint/operators.h"

namespace fl {

namespace {

Operators *cudaOps() {
  return getOperators(Device::kCuda);
}

Operators *cpuOps() {
  return getOperators(Device::kCpu);
}

Tensor toCuda(const Tensor &a, DType dtype) {
  return cudaOps()->cast(cudaOps()->toDevice(Device::getCuda(), a), dtype);
}

Tensor toCpu(const Tensor &a) {
  return cudaOps()->toDevice(Device::getCpu(), cudaOps()->cast(a, DType::kFloat));
}

}  // namespace

CATCH_TEST_CASE("test CUDA cumsum against the CPU", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  // Rows shorter than a tile, exactly one, and several with a ragged last one.
  for (int length : {1, 7, 256, 257, 1000, 4099}) {
    Tensor a = cpuOps()->rand({3, 5, length}, DType::kFloat);
    Tensor got = toCpu(cudaOps()->cumsum(toCuda(a, DType::kFloat), -1));
    CATCH_REQUIRE(cpuOps()->allClose(got, cpuOps()->cumsum(a, -1), 1e-4, 1e-4));
  }
}

CATCH_TEST_CASE("test CUDA cumsum along every dimension", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  Tensor a = cpuOps()->rand({4, 300, 9}, DType::kFloat);
  for (int dim : {0, 1, 2, -1, -2, -3}) {
    Tensor got = toCpu(cudaOps()->cumsum(toCuda(a, DType::kFloat), dim));
    CATCH_REQUIRE(cpuOps()->allClose(got, cpuOps()->cumsum(a, dim), 1e-4, 1e-4));
  }
}

CATCH_TEST_CASE("test CUDA cumsum in float16", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  Tensor a = cpuOps()->rand({2, 600}, DType::kFloat);
  Tensor scanned = cudaOps()->cumsum(toCuda(a, DType::kFloat16), -1);
  CATCH_REQUIRE(scanned.getDType() == DType::kFloat16);

  // Accumulated in float32 and rounded once, so the error is half a unit of the result's
  // precision -- about 0.1 at the ~300 the row sums to -- and not six hundred roundings of it.
  Tensor want = cpuOps()->cumsum(cpuOps()->cast(cpuOps()->cast(a, DType::kFloat16), DType::kFloat), -1);
  CATCH_REQUIRE(cpuOps()->allClose(toCpu(scanned), want, 1e-3, 1e-3));
}

}  // namespace fl
