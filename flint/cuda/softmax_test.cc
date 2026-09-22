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

#include <cmath>
#include <limits>
#include <vector>

#include "catch2/catch_amalgamated.hpp"
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
  return cudaOps()->cast(cudaOps()->toDevice(Device::getCuda(), a), DType::kFloat16);
}

Tensor toCpu(const Tensor &a) {
  return cudaOps()->toDevice(Device::getCpu(), cudaOps()->cast(a, DType::kFloat));
}

/// A float tensor moved to the device as it stands. The autoencoder's attention softmaxes in
/// float32, and that arm has no half2 fast path to fall into.
Tensor toCudaFloat(const Tensor &a) {
  return cudaOps()->toDevice(Device::getCuda(), a);
}

}  // namespace

CATCH_TEST_CASE("test CUDA softmax", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  for (int lastDim : {150, 151}) {
    Tensor a = cpuOps()->rand({2, 5, lastDim}, DType::kFloat);
    CATCH_REQUIRE(
        cpuOps()->allClose(toCpu(cudaOps()->softmax(toCuda(a))), cpuOps()->softmax(a), 5e-3));
  }
}

CATCH_TEST_CASE("test CUDA softmax (strided)", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  Tensor a = cpuOps()->rand({2, 3, 5}, DType::kFloat);
  Tensor x = cudaOps()->softmax(toCuda(a).transpose(1, 2));
  CATCH_REQUIRE(cpuOps()->allClose(toCpu(x), cpuOps()->softmax(a.transpose(1, 2)), 5e-3));
}

CATCH_TEST_CASE("test CUDA softmax (all ranks)", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  // Contiguous inputs of every rank go down the fused kernel, which flattens everything but the
  // last axis into rows.
  for (std::vector<int> shape : std::vector<std::vector<int>>{
           {8},
           {9},
           {3, 8},
           {2, 3, 8},
           {2, 3, 4, 8}}) {
    Tensor a = cpuOps()->rand(shape, DType::kFloat);
    CATCH_INFO("shape rank = " << shape.size());
    CATCH_REQUIRE(
        cpuOps()->allClose(toCpu(cudaOps()->softmax(toCuda(a))), cpuOps()->softmax(a), 5e-3));
  }
}

CATCH_TEST_CASE("test CUDA softmax (strided ranks)", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  // Each rank has its own entry point that reshapes into the strided 3D kernel; rank 1 and 2 in
  // particular are only reachable when the input is not contiguous.
  Tensor a2 = cpuOps()->rand({4, 6}, DType::kFloat);
  Tensor a3 = cpuOps()->rand({2, 3, 5}, DType::kFloat);
  Tensor a4 = cpuOps()->rand({2, 3, 4, 5}, DType::kFloat);

  // rank 1: one row of a transposed matrix.
  Tensor h1 = a2.transpose(0, 1).subtensor(0);
  Tensor d1 = toCuda(a2).transpose(0, 1).subtensor(0);
  CATCH_REQUIRE(!d1.isContiguous());
  CATCH_REQUIRE(cpuOps()->allClose(
      toCpu(cudaOps()->softmax(d1)),
      cpuOps()->softmax(cpuOps()->contiguous(h1)),
      5e-3));

  // rank 2.
  Tensor h2 = a2.transpose(0, 1);
  Tensor d2 = toCuda(a2).transpose(0, 1);
  CATCH_REQUIRE(cpuOps()->allClose(toCpu(cudaOps()->softmax(d2)), cpuOps()->softmax(h2), 5e-3));

  // rank 3.
  CATCH_REQUIRE(cpuOps()->allClose(
      toCpu(cudaOps()->softmax(toCuda(a3).transpose(0, 2))),
      cpuOps()->softmax(a3.transpose(0, 2)),
      5e-3));

  // rank 4 swaps the last two axes, the way attention scores are permuted. The rank-4 entry
  // point flattens the two leading axes with a view, so a permutation that separated those two
  // in memory would not be expressible and the operator rejects it instead of guessing.
  CATCH_REQUIRE(cpuOps()->allClose(
      toCpu(cudaOps()->softmax(toCuda(a4).transpose(2, 3))),
      cpuOps()->softmax(cpuOps()->contiguous(a4.transpose(2, 3))),
      5e-3));
}

CATCH_TEST_CASE("test CUDA softmax (row widths)", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  // A row up to 128 wide is reduced by one warp holding it in registers, and a wider one by a
  // whole 256-thread block, so the widths on either side of 128 take different kernels and the
  // ones below, at, and well above 256 take different numbers of loop iterations. Odd widths also
  // disable the half2 path, and a width that is not a multiple of 32 leaves a warp part idle.
  for (int width : {1, 2, 3, 31, 32, 33, 64, 77, 127, 128, 129, 255, 256, 257, 511, 512, 1024,
                    2048, 4096}) {
    Tensor a = cpuOps()->rand({2, width}, DType::kFloat);
    CATCH_INFO("width = " << width);
    CATCH_REQUIRE(
        cpuOps()->allClose(toCpu(cudaOps()->softmax(toCuda(a))), cpuOps()->softmax(a), 5e-3));
  }
}

CATCH_TEST_CASE("test CUDA softmax (degenerate rows)", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  // A single-element row normalises to exactly 1 no matter what the logit was.
  Tensor single = Tensor::create<float>({3, 1}, {-100.0f, 0.0f, 100.0f});
  CATCH_REQUIRE(cpuOps()->allClose(
      toCpu(cudaOps()->softmax(toCuda(single))),
      Tensor::create<float>({3, 1}, {1.0f, 1.0f, 1.0f}),
      5e-3));

  // Equal logits give a uniform distribution, which is where a missing max-subtraction still
  // looks right but a broken sum does not.
  Tensor flat = cpuOps()->zeros({2, 8}, DType::kFloat);
  Tensor uniform = cpuOps()->tensor({2, 8}, DType::kFloat);
  cpuOps()->fill(uniform, 0.125f);
  CATCH_REQUIRE(cpuOps()->allClose(toCpu(cudaOps()->softmax(toCuda(flat))), uniform, 5e-3));
}

CATCH_TEST_CASE("test CUDA softmax (extreme values)", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  Tensor a = Tensor::create<float>(
      {1, 1, 4},
      {-999.0f, -998.0f, -997.0f, -std::numeric_limits<float>::infinity()});

  CATCH_REQUIRE(cpuOps()->allClose(toCpu(cudaOps()->softmax(toCuda(a))), cpuOps()->softmax(a)));
}

CATCH_TEST_CASE("test CUDA softmax (large logits do not overflow)", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  // Without subtracting the row max, exp() of these overflows to infinity and the row comes back
  // as NaN. The result is the same distribution as the equivalent small logits.
  Tensor big = Tensor::create<float>({1, 4}, {1000.0f, 1001.0f, 1002.0f, 1003.0f});
  Tensor small = Tensor::create<float>({1, 4}, {0.0f, 1.0f, 2.0f, 3.0f});

  Tensor x = toCpu(cudaOps()->softmax(toCuda(big)));
  CATCH_REQUIRE(cpuOps()->allClose(x, cpuOps()->softmax(small), 5e-3));

  const float *data = x.getInternalData()->getData<float>(x.getInternalOffset());
  for (int i = 0; i < 4; ++i) {
    CATCH_INFO("i = " << i);
    CATCH_REQUIRE(!std::isnan(data[i]));
  }
}

CATCH_TEST_CASE("test CUDA softmax (float)", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  // Both entry points, at an even width and an odd one: the even one is where the half arm would
  // have vectorized, and the float arm has to answer the same whatever the width.
  for (int width : {1, 2, 255, 256, 257, 1024}) {
    Tensor a = cpuOps()->rand({3, width}, DType::kFloat);
    CATCH_INFO("width = " << width);
    Tensor x = cudaOps()->softmax(toCudaFloat(a));
    CATCH_REQUIRE(x.getDType() == DType::kFloat);
    CATCH_REQUIRE(
        cpuOps()->allClose(cudaOps()->toDevice(Device::getCpu(), x), cpuOps()->softmax(a), 1e-5f));
  }

  // Not contiguous, which is the strided kernel.
  Tensor a = cpuOps()->rand({2, 3, 5}, DType::kFloat);
  Tensor strided = cudaOps()->softmax(toCudaFloat(a).transpose(1, 2));
  CATCH_REQUIRE(cpuOps()->allClose(
      cudaOps()->toDevice(Device::getCpu(), strided),
      cpuOps()->softmax(a.transpose(1, 2)),
      1e-5f));

  // Scores a half softmax could not take: the shift by the row maximum keeps the exponentials in
  // range, but the input itself has to survive being read.
  Tensor wide = Tensor::create<float>({1, 3}, {70000.0f, 69999.0f, -70000.0f});
  CATCH_REQUIRE(cpuOps()->allClose(
      cudaOps()->toDevice(Device::getCpu(), cudaOps()->softmax(toCudaFloat(wide))),
      cpuOps()->softmax(wide),
      1e-5f));
}

}  // namespace fl
