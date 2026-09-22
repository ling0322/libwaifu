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

#include "catch2/catch_amalgamated.hpp"
#include "flint/operators.h"
#include "flint/tensor.h"

namespace fl {

namespace {

/// The CPU operators, which is what the calls in this file are asked of.
Operators *cpuOps() {
  return getOperators(Device::kCpu);
}

}  // namespace

namespace op {
namespace cpu {

namespace {

// the CPU fp16 kernels are checked against the fp32 kernels on the same device.
Tensor toFp16(const Tensor &a) {
  return cpuOps()->cast(a, DType::kFloat16);
}

Tensor toFp32(const Tensor &a) {
  return cpuOps()->cast(a, DType::kFloat);
}

}  // namespace

CATCH_TEST_CASE("test CPU fp16 binary operators", "[op][cpu][float16]") {
  Tensor a = cpuOps()->rand({2, 5, 10}, DType::kFloat);
  Tensor b = cpuOps()->rand({5}, DType::kFloat);
  Tensor at = a.transpose(2, 1).slice(1, {1, 9});
  Tensor xt = toFp16(a).transpose(2, 1).slice(1, {1, 9});
  Tensor y = toFp16(b);

  CATCH_REQUIRE(cpuOps()->allClose(toFp32(cpuOps()->add(xt, y)), cpuOps()->add(at, b), 5e-3));
  CATCH_REQUIRE(cpuOps()->allClose(toFp32(cpuOps()->mul(xt, y)), cpuOps()->mul(at, b), 5e-3));
}

CATCH_TEST_CASE("test CPU fp16 copy operators", "[op][cpu][float16]") {
  Tensor a = cpuOps()->rand({2, 10, 50}, DType::kFloat);
  Tensor x = toFp16(a).transpose(1, 0);
  Tensor dest = cpuOps()->tensorLike(x);
  cpuOps()->copy(x, dest);
  CATCH_REQUIRE(cpuOps()->allClose(toFp32(dest).transpose(1, 0), a));

  Tensor b = cpuOps()->rand({10, 2, 5, 20}, DType::kFloat);
  Tensor expanded = toFp16(b).unsqueeze(1).expand({10, 4, 2, 5, 20});
  Tensor dest5d = cpuOps()->tensorLike(expanded);
  cpuOps()->copy(expanded, dest5d);
  CATCH_REQUIRE(
      cpuOps()->allClose(
          toFp32(dest5d),
          cpuOps()->contiguous(b.unsqueeze(1).expand({10, 4, 2, 5, 20}))));
}

CATCH_TEST_CASE("test CPU fp16 matmul operators", "[op][cpu][float16]") {
  auto runCase = [](std::initializer_list<int> shapeA, std::initializer_list<int> shapeB) {
    Tensor a = cpuOps()->rand(shapeA, DType::kFloat);
    Tensor b = cpuOps()->rand(shapeB, DType::kFloat);
    Tensor xr = cpuOps()->matmul(a, b.slice(-1, {8, 28}).transpose(-1, -2));

    Tensor y = toFp16(b).slice(-1, {8, 28}).transpose(-1, -2);
    Tensor x = cpuOps()->matmul(toFp16(a), y);

    return cpuOps()->allClose(toFp32(x), xr, 5e-2);
  };

  CATCH_REQUIRE(runCase({10, 20}, {40, 30}));
  CATCH_REQUIRE(runCase({5, 10, 20}, {40, 30}));
  CATCH_REQUIRE(runCase({5, 10, 5, 20}, {10, 40, 30}));
}

CATCH_TEST_CASE("test CPU fp16 rmsNorm operator", "[op][cpu][float16]") {
  Tensor a = cpuOps()->rand({2, 5, 10}, DType::kFloat);
  Tensor w = cpuOps()->rand({10}, DType::kFloat);
  Tensor x = cpuOps()->rmsNorm(toFp16(a), toFp16(w), 1e-5);

  CATCH_REQUIRE(cpuOps()->allClose(toFp32(x), cpuOps()->rmsNorm(a, w, 1e-5), 5e-2));
}

CATCH_TEST_CASE("test CPU fp16 activation operators", "[op][cpu][float16]") {
  Tensor a = cpuOps()->rand({2, 5, 150}, DType::kFloat);

  CATCH_REQUIRE(
      cpuOps()->allClose(toFp32(cpuOps()->softmax(toFp16(a))), cpuOps()->softmax(a), 5e-2));
  CATCH_REQUIRE(cpuOps()->allClose(toFp32(cpuOps()->swiglu(toFp16(a))), cpuOps()->swiglu(a), 5e-2));
}

}  // namespace cpu
}  // namespace op
}  // namespace fl
