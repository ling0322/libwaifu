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
#include <vector>

#include "catch2/catch_amalgamated.hpp"
#include "flint/functional.h"
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

}  // namespace

CATCH_TEST_CASE("test CUDA swiglu", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  for (int lastDim : {150, 152}) {
    Tensor a = F::rand(Device::getCpu(), {2, 5, lastDim}, DType::kFloat);
    CATCH_REQUIRE(
        cpuOps()->allClose(toCpu(F::swiglu(cudaOps(), toCuda(a))), F::swiglu(cpuOps(), a), 5e-3));
  }
}

CATCH_TEST_CASE("test CUDA geglu", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  // The same gating with a GELU, which shares every kernel with swiglu but the activation.
  for (int lastDim : {150, 152}) {
    Tensor a = F::rand(Device::getCpu(), {2, 5, lastDim}, DType::kFloat);
    CATCH_REQUIRE(cpuOps()->allClose(toCpu(F::geglu(cudaOps(), toCuda(a))), F::geglu(cpuOps(), a), 5e-3));
  }

  // The gate is the whole difference between the two, so one is not the other.
  Tensor a = F::rand(Device::getCpu(), {2, 5, 152}, DType::kFloat);
  CATCH_REQUIRE(!cpuOps()->allClose(F::geglu(cpuOps(), a), F::swiglu(cpuOps(), a), 5e-3));
}

CATCH_TEST_CASE("test CUDA swiglu (strided)", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  Tensor b = F::rand(Device::getCpu(), {2, 3, 152}, DType::kFloat);
  Tensor y = F::swiglu(cudaOps(), toCuda(b).transpose(0, 1));
  CATCH_REQUIRE(cpuOps()->allClose(toCpu(y), F::swiglu(cpuOps(), b.transpose(0, 1)), 5e-3));

  Tensor c = F::rand(Device::getCpu(), {2, 152, 3}, DType::kFloat);
  Tensor z = F::swiglu(cudaOps(), toCuda(c).transpose(1, 2));
  CATCH_REQUIRE(cpuOps()->allClose(toCpu(z), F::swiglu(cpuOps(), c.transpose(1, 2)), 5e-3));
}

CATCH_TEST_CASE("test CUDA swiglu (packed 2D batch)", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  // A packed batch arrives as [tokens, 2 * hidden]; the operator wraps it in a leading axis and
  // unwraps the result, so the output must still be 2D.
  for (int width : {4, 6, 512, 600}) {
    Tensor a = F::rand(Device::getCpu(), {5, width}, DType::kFloat);
    Tensor x = F::swiglu(cudaOps(), toCuda(a));

    CATCH_INFO("width = " << width);
    CATCH_REQUIRE(x.getShape() == std::vector<int>{5, width / 2});
    CATCH_REQUIRE(cpuOps()->allClose(toCpu(x), F::swiglu(cpuOps(), a), 5e-3));
  }

  // a single token is the decode-step shape.
  Tensor one = F::rand(Device::getCpu(), {1, 16}, DType::kFloat);
  CATCH_REQUIRE(
      cpuOps()->allClose(toCpu(F::swiglu(cudaOps(), toCuda(one))), F::swiglu(cpuOps(), one), 5e-3));
}

CATCH_TEST_CASE("test CUDA swiglu (output widths)", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  // The output half is split across 256-thread blocks in x, and an odd output width turns off
  // the half2 path. Walk both sides of the block boundary with each parity.
  for (int outputWidth : {1, 2, 3, 255, 256, 257, 512, 1000}) {
    Tensor a = F::rand(Device::getCpu(), {2, 3, outputWidth * 2}, DType::kFloat);
    Tensor x = F::swiglu(cudaOps(), toCuda(a));

    CATCH_INFO("outputWidth = " << outputWidth);
    CATCH_REQUIRE(x.getShape() == std::vector<int>{2, 3, outputWidth});
    CATCH_REQUIRE(cpuOps()->allClose(toCpu(x), F::swiglu(cpuOps(), a), 5e-3));
  }
}

CATCH_TEST_CASE("test CUDA swiglu (gate saturation)", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  // silu(gate) * value at the extremes: a large negative gate drives the output to zero and a
  // large positive one leaves the value essentially untouched. A zero gate contributes nothing.
  Tensor a = Tensor::create<float>({1, 8}, {-20.0f, 0.0f, 20.0f, 1.0f, 3.0f, 5.0f, 7.0f, 9.0f});
  Tensor x = toCpu(F::swiglu(cudaOps(), toCuda(a)));
  const float *data = x.getInternalData()->getData<float>(x.getInternalOffset());

  CATCH_REQUIRE(std::fabs(data[0]) < 1e-2f);          // silu(-20) * 3 ~= 0
  CATCH_REQUIRE(std::fabs(data[1]) < 1e-3f);          // silu(0) * 5 == 0
  CATCH_REQUIRE(std::fabs(data[2] - 20.0f * 7.0f) < 1.0f);  // silu(20) ~= 20
  CATCH_REQUIRE(cpuOps()->allClose(x, F::swiglu(cpuOps(), a), 5e-3));
}

CATCH_TEST_CASE("test CUDA swiglu and geglu in float32", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  // Float32 used to be read as if it were half: every value garbage, and no error. Now it is
  // computed in float32 and comes back in float32, so the tolerance is float32's.
  auto toCudaFloat = [](const Tensor &a) { return F::toDevice(cudaOps(), a, Device::getCuda()); };
  auto toCpuFloat = [](const Tensor &a) { return F::toDevice(cudaOps(), a, Device::getCpu()); };

  for (int outputWidth : {1, 3, 256, 257}) {
    Tensor a = F::rand(Device::getCpu(), {2, 3, outputWidth * 2}, DType::kFloat);
    Tensor x = F::swiglu(cudaOps(), toCudaFloat(a));
    Tensor y = F::geglu(cudaOps(), toCudaFloat(a));

    CATCH_INFO("outputWidth = " << outputWidth);
    CATCH_REQUIRE(x.getDType() == DType::kFloat);
    CATCH_REQUIRE(cpuOps()->allClose(toCpuFloat(x), F::swiglu(cpuOps(), a), 1e-5, 1e-6));
    CATCH_REQUIRE(cpuOps()->allClose(toCpuFloat(y), F::geglu(cpuOps(), a), 1e-5, 1e-6));
  }

  // Strided, through the rank-3 path and the rank-2 one.
  Tensor b = F::rand(Device::getCpu(), {2, 3, 152}, DType::kFloat);
  CATCH_REQUIRE(cpuOps()->allClose(
      toCpuFloat(F::swiglu(cudaOps(), toCudaFloat(b).transpose(0, 1))),
      F::swiglu(cpuOps(), F::contiguous(cpuOps(), b.transpose(0, 1))),
      1e-5,
      1e-6));
  Tensor c = F::rand(Device::getCpu(), {8, 6}, DType::kFloat);
  CATCH_REQUIRE(cpuOps()->allClose(
      toCpuFloat(F::swiglu(cudaOps(), toCudaFloat(c).transpose(0, 1))),
      F::swiglu(cpuOps(), F::contiguous(cpuOps(), c.transpose(0, 1))),
      1e-5,
      1e-6));
}

CATCH_TEST_CASE("test CUDA swiglu (any rank)", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  // Rank 1 and rank 4 used to end in NOT_IMPL; a strided rank 4 is copied first.
  Tensor one = F::rand(Device::getCpu(), {10}, DType::kFloat);
  CATCH_REQUIRE(
      cpuOps()->allClose(toCpu(F::swiglu(cudaOps(), toCuda(one))), F::swiglu(cpuOps(), one), 5e-3));

  Tensor four = F::rand(Device::getCpu(), {2, 3, 4, 20}, DType::kFloat);
  Tensor x = F::swiglu(cudaOps(), toCuda(four));
  CATCH_REQUIRE(x.getShape() == std::vector<int>{2, 3, 4, 10});
  CATCH_REQUIRE(cpuOps()->allClose(toCpu(x), F::swiglu(cpuOps(), four), 5e-3));

  Tensor strided = F::swiglu(cudaOps(), toCuda(four).transpose(0, 2));
  CATCH_REQUIRE(cpuOps()->allClose(
      toCpu(strided),
      F::swiglu(cpuOps(), F::contiguous(cpuOps(), four.transpose(0, 2))),
      5e-3));
}

CATCH_TEST_CASE("test CUDA swiglu (more rows than a grid axis holds)", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  // 70 000 rows is past the 65 535 a grid's y axis holds, so the rows spill into z, and the last
  // z slice is part empty -- which the kernel has to skip rather than write past the end.
  Tensor a = F::rand(Device::getCpu(), {70000, 4}, DType::kFloat);
  CATCH_REQUIRE(
      cpuOps()->allClose(toCpu(F::swiglu(cudaOps(), toCuda(a))), F::swiglu(cpuOps(), a), 5e-3));
}

}  // namespace fl
