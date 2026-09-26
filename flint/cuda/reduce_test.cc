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

}  // namespace

CATCH_TEST_CASE("test CUDA reductions", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  Tensor a = cpuOps()->rand({2, 5, 150}, DType::kFloat);
  CATCH_REQUIRE(
      cpuOps()->allClose(toCpu(cudaOps()->sum(toCuda(a), -1)), cpuOps()->sum(a, -1), 5e-3));
  CATCH_REQUIRE(cpuOps()->allClose(toCpu(cudaOps()->max(toCuda(a))), cpuOps()->max(a), 5e-3));
}

CATCH_TEST_CASE("test CUDA reductions (all ranks)", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  // Ranks 1 and 2 are reshaped to 3D on the way in and back again on the way out, so the shape
  // that comes out has to match what the CPU reduction produces for the same input.
  Tensor a1 = cpuOps()->rand({8}, DType::kFloat);
  Tensor s1 = toCpu(cudaOps()->sum(toCuda(a1), -1));
  CATCH_REQUIRE(s1.getShape() == cpuOps()->sum(a1, -1).getShape());
  CATCH_REQUIRE(cpuOps()->allClose(s1, cpuOps()->sum(a1, -1), 5e-3));
  CATCH_REQUIRE(cpuOps()->allClose(toCpu(cudaOps()->max(toCuda(a1))), cpuOps()->max(a1), 5e-3));

  Tensor a2 = cpuOps()->rand({3, 8}, DType::kFloat);
  Tensor s2 = toCpu(cudaOps()->sum(toCuda(a2), -1));
  CATCH_REQUIRE(s2.getShape() == cpuOps()->sum(a2, -1).getShape());
  CATCH_REQUIRE(cpuOps()->allClose(s2, cpuOps()->sum(a2, -1), 5e-3));
  CATCH_REQUIRE(cpuOps()->allClose(toCpu(cudaOps()->max(toCuda(a2))), cpuOps()->max(a2), 5e-3));

  Tensor a3 = cpuOps()->rand({2, 3, 8}, DType::kFloat);
  Tensor s3 = toCpu(cudaOps()->sum(toCuda(a3), -1));
  CATCH_REQUIRE(s3.getShape() == cpuOps()->sum(a3, -1).getShape());
  CATCH_REQUIRE(cpuOps()->allClose(s3, cpuOps()->sum(a3, -1), 5e-3));
}

CATCH_TEST_CASE("test CUDA sum over any dimension", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  // Anything but the last dimension used to come back as an empty tensor, with no error. Every
  // dimension of a rank-4 tensor, counted from either end, in float16 and in float32.
  Tensor a = cpuOps()->rand({2, 3, 4, 300}, DType::kFloat);
  for (int dim : {0, 1, 2, 3, -2, -3, -4}) {
    CATCH_INFO("dim = " << dim);
    Tensor want = cpuOps()->sum(a, dim);

    Tensor half = toCpu(cudaOps()->sum(toCuda(a), dim));
    CATCH_REQUIRE(half.getShape() == want.getShape());
    CATCH_REQUIRE(cpuOps()->allClose(half, want, 1e-2, 5e-2));

    Tensor single = cudaOps()->sum(cudaOps()->toDevice(Device::getCuda(), a), dim);
    CATCH_REQUIRE(single.getDType() == DType::kFloat);
    CATCH_REQUIRE(cpuOps()->allClose(
        cudaOps()->toDevice(Device::getCpu(), single), want, 1e-5, 1e-4));
  }

  // A strided input over a middle dimension.
  Tensor strided = cudaOps()->toDevice(Device::getCuda(), a).transpose(1, 3);
  CATCH_REQUIRE(cpuOps()->allClose(
      cudaOps()->toDevice(Device::getCpu(), cudaOps()->sum(strided, 2)),
      cpuOps()->sum(a.transpose(1, 3), 2),
      1e-5,
      1e-4));
}

CATCH_TEST_CASE("test CUDA reductions (row widths)", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  // One 256-thread block reduces a whole row, so a width below the block size leaves most
  // threads with nothing to contribute and a width above it makes every thread loop. Both have
  // to reach the same answer as the sequential CPU reduction.
  for (int width : {1, 2, 3, 255, 256, 257, 1000, 4096}) {
    Tensor a = cpuOps()->rand({2, 3, width}, DType::kFloat);
    CATCH_INFO("width = " << width);
    CATCH_REQUIRE(
        cpuOps()->allClose(toCpu(cudaOps()->sum(toCuda(a), -1)), cpuOps()->sum(a, -1), 1e-2));
    CATCH_REQUIRE(cpuOps()->allClose(toCpu(cudaOps()->max(toCuda(a))), cpuOps()->max(a), 5e-3));
  }
}

CATCH_TEST_CASE("test CUDA reductions (strided rows)", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  // The reduction reads through an accessor, so a row whose elements are not adjacent has to
  // work as well as a packed one.
  Tensor a = cpuOps()->rand({2, 3, 5}, DType::kFloat);
  Tensor strided = toCuda(a).transpose(0, 2);
  CATCH_REQUIRE(!strided.isContiguous());

  CATCH_REQUIRE(cpuOps()->allClose(
      toCpu(cudaOps()->sum(strided, -1)),
      cpuOps()->sum(a.transpose(0, 2), -1),
      5e-3));
  CATCH_REQUIRE(
      cpuOps()->allClose(toCpu(cudaOps()->max(strided)), cpuOps()->max(a.transpose(0, 2)), 5e-3));
}

CATCH_TEST_CASE("test CUDA reductions (known values)", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  // Fixed inputs, so a reduction that silently dropped or double-counted an element shows up as
  // a wrong total rather than as noise inside a tolerance.
  Tensor a = Tensor::create<float>({2, 4}, {1.0f, 2.0f, 3.0f, 4.0f, -1.0f, -2.0f, -3.0f, -4.0f});
  Tensor x = toCuda(a);

  CATCH_REQUIRE(cpuOps()->allClose(
      toCpu(cudaOps()->sum(x, -1)),
      Tensor::create<float>({2}, {10.0f, -10.0f}),
      5e-3));
  // max over an all-negative row must not fall back to the zero initial value.
  CATCH_REQUIRE(cpuOps()->allClose(
      toCpu(cudaOps()->max(x)),
      Tensor::create<float>({2}, {4.0f, -1.0f}),
      5e-3));

  // summing zeros stays zero rather than accumulating the initial value once per thread.
  Tensor zeros = cudaOps()->zeros({2, 300}, DType::kFloat16);
  CATCH_REQUIRE(
      cpuOps()->allClose(toCpu(cudaOps()->sum(zeros, -1)), cpuOps()->zeros({2}, DType::kFloat)));
}

}  // namespace fl
