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

#include <memory>

#include "catch2/catch_amalgamated.hpp"
#include "flint/cuda/matmul.h"
#include "flint/device.h"
#include "flint/functional.h"
#include "flint/operators.h"

namespace fl {

CATCH_TEST_CASE("test matmul gemm (cutlass)", "[fl][op][cuda][cutlass]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  std::shared_ptr<op::cuda::MatMul> mm = op::cuda::MatMul::createCutlass();

  Tensor a = F::rand({10, 128}, DType::kFloat);
  Tensor b = F::rand({40, 256}, DType::kFloat);
  Tensor xr = F::matmul(a, b.slice(1, {128, 256}).transpose(1, 0));

  Tensor x = F::to(Device::getCuda(), a);
  Tensor y = F::to(Device::getCuda(), b);
  x = F::cast(x, DType::kFloat16);
  y = F::cast(y, DType::kFloat16);
  y = y.slice(1, {128, 256});
  y = y.transpose(1, 0);
  x = mm->apply(x, y);
  x = F::cast(x, DType::kFloat);
  x = F::to(Device::getCpu(), x);

  CATCH_REQUIRE(F::allClose(x, xr, 1e-2f));
}

CATCH_TEST_CASE("test matmul bmm (cutlass)", "[fl][op][cuda][cutlass]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  std::shared_ptr<op::cuda::MatMul> mm = op::cuda::MatMul::createCutlass();

  Tensor a = F::rand({5, 10, 8, 24}, DType::kFloat);
  Tensor b = F::rand({10, 64, 24}, DType::kFloat);
  Tensor xr = F::matmul(a, b.slice(1, {8, 32}).transpose(-1, -2));

  Tensor x = F::to(Device::getCuda(), a);
  Tensor y = F::to(Device::getCuda(), b);
  x = F::cast(x, DType::kFloat16);
  y = F::cast(y, DType::kFloat16);
  y = y.slice(1, {8, 32});
  y = y.transpose(-1, -2);
  x = mm->apply(x, y);
  x = F::cast(x, DType::kFloat);
  x = F::to(Device::getCpu(), x);

  CATCH_REQUIRE(F::allClose(x, xr, 5e-3f));
}

CATCH_TEST_CASE("test matmul gemm accumulates in float (cutlass)", "[fl][op][cuda][cutlass]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  // What the two cases above cannot see. At a K of 128 a half accumulator still holds the running
  // sum well enough to pass a loose comparison; at 2048 -- which is what SDXL's cross attention
  // contracts over -- the sum reaches a magnitude where half's step is larger than the products
  // being added to it, and most of each one is lost. The difference between accumulating in half
  // and in float is around thirty times at this shape, so the tolerance sits between them rather
  // than being tight for its own sake.
  constexpr int kM = 64;
  constexpr int kK = 2048;
  constexpr int kN = 128;

  std::shared_ptr<op::cuda::MatMul> mm = op::cuda::MatMul::createCutlass();

  Tensor a = F::rand({kM, kK}, DType::kFloat);
  Tensor b = F::rand({kK, kN}, DType::kFloat);

  // The reference is computed from the half values, not the float ones, so what this measures is
  // the accumulation rather than the rounding of the inputs.
  Tensor halfA = F::cast(F::cast(a, DType::kFloat16), DType::kFloat);
  Tensor halfB = F::cast(F::cast(b, DType::kFloat16), DType::kFloat);
  Tensor expected = F::matmul(halfA, halfB);

  Tensor x = F::cast(F::to(Device::getCuda(), a), DType::kFloat16);
  Tensor y = F::cast(F::to(Device::getCuda(), b), DType::kFloat16);
  Tensor actual = F::to(Device::getCpu(), F::cast(mm->apply(x, y), DType::kFloat));

  CATCH_REQUIRE(F::allClose(actual, expected, 2e-3f));
}

}  // namespace fl
