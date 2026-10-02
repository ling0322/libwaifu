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

Tensor toCpu(const Tensor &a) {
  return F::toDevice(cudaOps(), F::cast(cudaOps(), a, DType::kFloat), Device::getCpu());
}

}  // namespace

CATCH_TEST_CASE("test CUDA fill", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  Tensor filled = F::empty(Device::getCuda(), {2, 5, 10}, DType::kFloat16);
  cudaOps()->fill(filled, 1.5f);
  Tensor filledRef = F::empty(Device::getCpu(), {2, 5, 10}, DType::kFloat);
  cpuOps()->fill(filledRef, 1.5f);
  CATCH_REQUIRE(cpuOps()->allClose(toCpu(filled), filledRef));

  Tensor zeros = F::zeros(Device::getCuda(), {2, 5, 10}, DType::kFloat16);
  CATCH_REQUIRE(cpuOps()->allClose(toCpu(zeros), F::zeros(Device::getCpu(), {2, 5, 10}, DType::kFloat)));
}

CATCH_TEST_CASE("test CUDA fill (values)", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  // Negative and zero fills go through the same float-to-half conversion as any other value.
  for (float value : {0.0f, -1.5f, 1.0f, -0.0f, 100.0f}) {
    Tensor filled = F::empty(Device::getCuda(), {3, 4}, DType::kFloat16);
    cudaOps()->fill(filled, value);

    Tensor expected = F::empty(Device::getCpu(), {3, 4}, DType::kFloat);
    cpuOps()->fill(expected, value);

    CATCH_INFO("value = " << value);
    CATCH_REQUIRE(cpuOps()->allClose(toCpu(filled), expected));
  }
}

CATCH_TEST_CASE("test CUDA zeros (every dtype)", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  // zeros() used to hand back a half tensor whatever type it was asked for, and nothing said so
  // at the call: the first sign was something else refusing the tensor for its type, layers away.
  // It builds the type it was asked for now, which is a type fill has to know.
  for (DType dtype : {DType::kFloat16, DType::kFloat}) {
    CATCH_INFO("dtype = " << dtype.toString());

    Tensor zeroed = F::zeros(Device::getCuda(), {2, 5, 10}, dtype);
    CATCH_REQUIRE(zeroed.getDType() == dtype);
    CATCH_REQUIRE(cpuOps()->allClose(toCpu(zeroed), F::zeros(Device::getCpu(), {2, 5, 10}, DType::kFloat)));

    cudaOps()->fill(zeroed, 3.0f);
    Tensor expected = F::empty(Device::getCpu(), {2, 5, 10}, DType::kFloat);
    cpuOps()->fill(expected, 3.0f);
    CATCH_REQUIRE(cpuOps()->allClose(toCpu(zeroed), expected));
  }
}

CATCH_TEST_CASE("test CUDA fill (strided ranks)", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  // Filling a window of a larger buffer has to honour the strides and leave the rest alone. The
  // generic kernel is instantiated per rank, so walk ranks 1 through 4.
  constexpr int Rows = 4;
  constexpr int Cols = 10;
  Tensor dest = F::zeros(Device::getCuda(), {Rows, Cols}, DType::kFloat16);
  Tensor window = dest.slice(1, {2, 6});
  CATCH_REQUIRE(!window.isContiguous());
  cudaOps()->fill(window, 1.5f);

  Tensor host = toCpu(dest);
  Tensor filledRef = F::empty(Device::getCpu(), {Rows, 4}, DType::kFloat);
  cpuOps()->fill(filledRef, 1.5f);
  CATCH_REQUIRE(cpuOps()->allClose(F::contiguous(cpuOps(), host.slice(1, {2, 6})), filledRef));
  CATCH_REQUIRE(cpuOps()->allClose(
      F::contiguous(cpuOps(), host.slice(1, {0, 2})),
      F::zeros(Device::getCpu(), {Rows, 2}, DType::kFloat)));
  CATCH_REQUIRE(cpuOps()->allClose(
      F::contiguous(cpuOps(), host.slice(1, {6, 10})),
      F::zeros(Device::getCpu(), {Rows, 4}, DType::kFloat)));

  // rank 1 and rank 3/4 strided views of the same kind.
  Tensor cube = F::zeros(Device::getCuda(), {2, 3, 4}, DType::kFloat16);
  cudaOps()->fill(cube.transpose(0, 2), 2.0f);
  Tensor cubeRef = F::empty(Device::getCpu(), {2, 3, 4}, DType::kFloat);
  cpuOps()->fill(cubeRef, 2.0f);
  CATCH_REQUIRE(cpuOps()->allClose(toCpu(cube), cubeRef));

  Tensor row = dest.transpose(0, 1).subtensor(0);
  CATCH_REQUIRE(!row.isContiguous());
  cudaOps()->fill(row, 7.0f);
  Tensor rowBack = toCpu(dest).transpose(0, 1).subtensor(0);
  Tensor rowRef = F::empty(Device::getCpu(), {Rows}, DType::kFloat);
  cpuOps()->fill(rowRef, 7.0f);
  CATCH_REQUIRE(cpuOps()->allClose(F::contiguous(cpuOps(), rowBack), rowRef));
}

CATCH_TEST_CASE("test CUDA zeros (all ranks)", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  for (std::vector<int> shape : std::vector<std::vector<int>>{
           {1},
           {257},
           {3, 5},
           {2, 3, 4},
           {2, 3, 4, 5}}) {
    Tensor zeros = F::zeros(Device::getCuda(), shape, DType::kFloat16);
    CATCH_INFO("shape rank = " << shape.size());
    CATCH_REQUIRE(zeros.getShape() == shape);
    CATCH_REQUIRE(cpuOps()->allClose(toCpu(zeros), F::zeros(Device::getCpu(), shape, DType::kFloat)));
  }
}

}  // namespace fl
