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

#include <vector>

#include "catch2/catch_amalgamated.hpp"
#include "flint/cpu/common.h"
#include "flint/operators.h"
#include "flint/tensor.h"

namespace fl {

namespace {

Operators *cpuOps() {
  return getOperators(Device::kCpu);
}

/// The inclusive prefix sum of a contiguous (d0, d1, d2) float tensor along `dim`, written out
/// with three loops and nothing of the operator's: the reference the operator is checked against.
Tensor longhand(const Tensor &a, int dim) {
  std::vector<int> shape = a.getShape();
  int d0 = shape[0], d1 = shape[1], d2 = shape[2];
  const float *x = op::cpu::getDataPtrCpu<float>(a);

  std::vector<float> out(a.getNumEl());
  auto at = [&](int i, int j, int k) { return (i * d1 + j) * d2 + k; };
  for (int i = 0; i < d0; ++i) {
    for (int j = 0; j < d1; ++j) {
      for (int k = 0; k < d2; ++k) {
        int here = at(i, j, k);
        float previous = 0.0f;
        if (dim == 0 && i > 0) previous = out[at(i - 1, j, k)];
        if (dim == 1 && j > 0) previous = out[at(i, j - 1, k)];
        if (dim == 2 && k > 0) previous = out[at(i, j, k - 1)];
        out[here] = previous + x[here];
      }
    }
  }

  return Tensor::create<float>({d0, d1, d2}, out);
}

}  // namespace

CATCH_TEST_CASE("test CPU cumsum", "[core][operators]") {
  Tensor a = Tensor::create<float>({2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});

  CATCH_REQUIRE(cpuOps()->allClose(
      cpuOps()->cumsum(a, -1),
      Tensor::create<float>({2, 3}, {1.0f, 3.0f, 6.0f, 4.0f, 9.0f, 15.0f})));
  CATCH_REQUIRE(cpuOps()->allClose(
      cpuOps()->cumsum(a, 0),
      Tensor::create<float>({2, 3}, {1.0f, 2.0f, 3.0f, 5.0f, 7.0f, 9.0f})));
}

CATCH_TEST_CASE("test CPU cumsum along every dimension", "[core][operators]") {
  Tensor a = cpuOps()->rand({3, 7, 33}, DType::kFloat);

  for (int dim = 0; dim < 3; ++dim) {
    CATCH_REQUIRE(cpuOps()->allClose(cpuOps()->cumsum(a, dim), longhand(a, dim), 1e-5, 1e-5));
    CATCH_REQUIRE(cpuOps()->allClose(cpuOps()->cumsum(a, dim - 3), longhand(a, dim), 1e-5, 1e-5));
  }
}

CATCH_TEST_CASE("test CPU cumsum of a strided view", "[core][operators]") {
  Tensor a = cpuOps()->rand({5, 9, 4}, DType::kFloat);
  Tensor viewed = a.transpose(0, 2);

  // The same numbers scanned from a view and from a copy of it.
  CATCH_REQUIRE(cpuOps()->allClose(
      cpuOps()->cumsum(viewed, 1),
      cpuOps()->cumsum(cpuOps()->contiguous(viewed), 1)));
}

CATCH_TEST_CASE("test CPU cumsum in float16", "[core][operators]") {
  Tensor a = cpuOps()->rand({4, 50}, DType::kFloat);
  Tensor half = cpuOps()->cast(a, DType::kFloat16);

  Tensor scanned = cpuOps()->cumsum(half, -1);
  CATCH_REQUIRE(scanned.getDType() == DType::kFloat16);
  CATCH_REQUIRE(cpuOps()->allClose(
      cpuOps()->cast(scanned, DType::kFloat),
      cpuOps()->cumsum(cpuOps()->cast(half, DType::kFloat), -1),
      5e-3,
      5e-3));
}

CATCH_TEST_CASE("test CPU cumsum of a long row", "[core][operators]") {
  // A hundred thousand ones: exact in double, and what a float accumulator reaches too.
  std::vector<float> ones(100000, 1.0f);
  Tensor a = Tensor::create<float>({1, 100000}, ones);
  Tensor scanned = cpuOps()->cumsum(a, -1);

  CATCH_REQUIRE(op::cpu::getDataPtrCpu<float>(scanned)[99999] == 100000.0f);
}

}  // namespace fl
