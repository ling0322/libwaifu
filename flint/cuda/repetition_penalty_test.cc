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

CATCH_TEST_CASE("test CUDA repetitionPenalty", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  Tensor a = cpuOps()->rand({2, 16}, DType::kFloat);
  Tensor history = Tensor::create<LongType>({2, 4}, {1, 0, 1, 3, 0, 0, 0, 1});

  Tensor x = toCuda(a);
  cudaOps()->repetitionPenalty(x, cudaOps()->toDevice(Device::getCuda(), history), 1.5);
  cpuOps()->repetitionPenalty(a, history, 1.5);

  CATCH_REQUIRE(cpuOps()->allClose(toCpu(x), a, 1e-3));
}

CATCH_TEST_CASE("test CUDA repetitionPenalty (float, long, repeating history)", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  // What a speech model asks for: float logits over its own alphabet, and a history of every
  // token said so far -- far longer than the sixty-four the kernel once allowed, and full of
  // repeats. Each repeated token has to be penalized once, as the host does it, and not once per
  // appearance. Logits either side of zero, since the penalty divides one sign and multiplies
  // the other.
  const int vocabulary = 8194;
  const int said = 1000;

  Tensor a = cpuOps()->add(
      cpuOps()->rand({1, vocabulary}, DType::kFloat),
      Tensor::create<float>({1}, {-0.5f}));
  std::vector<LongType> values(said);
  for (int i = 0; i < said; ++i) values[i] = (i * 37) % 97;
  Tensor history = Tensor::create<LongType>({1, said}, values);

  Tensor x = cudaOps()->toDevice(Device::getCuda(), a);
  cudaOps()->repetitionPenalty(x, cudaOps()->toDevice(Device::getCuda(), history), 10.0);
  cpuOps()->repetitionPenalty(a, history, 10.0);

  CATCH_REQUIRE(x.getDType() == DType::kFloat);
  CATCH_REQUIRE(cpuOps()->allClose(toCpu(x), a, 1e-6, 1e-6));
}

CATCH_TEST_CASE("test CUDA repetitionPenalty (packed 1D logits)", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  // A single sequence arrives as 1D logits with a 1D history; the operator wraps both in a
  // leading axis, and the penalty has to land on the same positions as the 2D form.
  Tensor a = cpuOps()->rand({16}, DType::kFloat);
  Tensor history = Tensor::create<LongType>({3}, {2, 5, 11});

  Tensor x = toCuda(a);
  cudaOps()->repetitionPenalty(x, cudaOps()->toDevice(Device::getCuda(), history), 1.5);
  cpuOps()->repetitionPenalty(a, history, 1.5);

  CATCH_REQUIRE(cpuOps()->allClose(toCpu(x), a, 5e-3, 5e-3));
}

CATCH_TEST_CASE("test CUDA repetitionPenalty (sign and known values)", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  // A positive logit is divided by the weight and a negative one multiplied, so the penalty
  // always moves a score towards -inf. Zero and untouched positions stay exactly as they were.
  Tensor a = Tensor::create<float>({1, 5}, {2.0f, -2.0f, 0.0f, 4.0f, -4.0f});
  Tensor history = Tensor::create<LongType>({1, 3}, {0, 1, 2});

  Tensor x = toCuda(a);
  cudaOps()->repetitionPenalty(x, cudaOps()->toDevice(Device::getCuda(), history), 2.0f);

  Tensor host = toCpu(x);
  const float *data = host.getInternalData()->getData<float>(host.getInternalOffset());
  CATCH_REQUIRE(std::fabs(data[0] - 1.0f) < 1e-2f);   // 2 / 2
  CATCH_REQUIRE(std::fabs(data[1] + 4.0f) < 1e-2f);   // -2 * 2
  CATCH_REQUIRE(data[2] == 0.0f);                     // zero is left alone
  CATCH_REQUIRE(std::fabs(data[3] - 4.0f) < 1e-2f);   // not in the history
  CATCH_REQUIRE(std::fabs(data[4] + 4.0f) < 1e-2f);   // not in the history
}

CATCH_TEST_CASE("test CUDA repetitionPenalty (weight of one is a no-op)", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  Tensor a = cpuOps()->rand({2, 16}, DType::kFloat);
  Tensor history = Tensor::create<LongType>({2, 3}, {1, 2, 3, 4, 5, 6});

  Tensor x = toCuda(a);
  Tensor before = toCpu(x);
  cudaOps()->repetitionPenalty(x, cudaOps()->toDevice(Device::getCuda(), history), 1.0f);

  CATCH_REQUIRE(cpuOps()->allClose(toCpu(x), before, 5e-3, 5e-3));
}

CATCH_TEST_CASE("test CUDA repetitionPenalty (history lengths)", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  // The kernels launch a whole block of threads and return early past the end of the history, so
  // a short history must not penalise positions it does not name -- and a long one, past the
  // sixty-four a single block of the old kernel held, still has to reach every position it does.
  for (int length : {1, 2, 63, 64, 300}) {
    Tensor a = cpuOps()->rand({2, 64}, DType::kFloat);
    std::vector<LongType> ids(2 * length);
    for (int i = 0; i < 2 * length; ++i) ids[i] = i % 64;
    Tensor history = Tensor::create<LongType>({2, length}, ids);

    Tensor x = toCuda(a);
    cudaOps()->repetitionPenalty(x, cudaOps()->toDevice(Device::getCuda(), history), 1.5);
    cpuOps()->repetitionPenalty(a, history, 1.5);

    // The CUDA side is half and the reference is float, so the comparison has to leave room for
    // half's own round-off. A penalty that failed to apply would be off by the weight itself,
    // which is far outside this.
    CATCH_INFO("history length = " << length);
    CATCH_REQUIRE(cpuOps()->allClose(toCpu(x), a, 5e-3, 5e-3));
  }
}

}  // namespace fl
