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

Tensor toCpuFloat(Tensor tensor) {
  return F::toDevice(cudaOps(), F::cast(cudaOps(), tensor, DType::kFloat), Device::getCpu());
}

Tensor baselineRotaryEmbedding(Tensor input, Tensor roPE) {
  Tensor cos = roPE.subtensor(0);
  Tensor sin = roPE.subtensor(1);
  cos = cos.expand({cos.getShape(0), input.getShape(1), cos.getShape(2)});
  sin = sin.expand({sin.getShape(0), input.getShape(1), sin.getShape(2)});

  int halfDim = input.getShape(-1) / 2;
  Tensor rotated = F::emptyLike(input);
  Tensor first = input.slice(-1, {0, halfDim});
  Tensor second = F::mul(cudaOps(), input.slice(-1, {halfDim, None}), -1.0f);
  cudaOps()->copy(first, rotated.slice(-1, {halfDim, None}));
  cudaOps()->copy(second, rotated.slice(-1, {0, halfDim}));

  return F::add(cudaOps(), 
      F::mul(cudaOps(), input, F::contiguous(cudaOps(), cos)),
      F::mul(cudaOps(), rotated, F::contiguous(cudaOps(), sin)));
}

bool runCase(int numQueryHeads, int numKeyHeads, int headDim) {
  std::vector<LongType> positionValues = {7, 2, 31};
  int numTokens = static_cast<int>(positionValues.size());
  Device device = Device::getCuda();

  Tensor positions = F::toDevice(cudaOps(), Tensor::create<LongType>({numTokens}, positionValues), device);
  Tensor query = F::rand(Device::getCuda(), {numTokens, numQueryHeads, headDim}, DType::kFloat16);
  Tensor key = F::rand(Device::getCuda(), {numTokens, numKeyHeads, headDim}, DType::kFloat16);
  Tensor cache = F::rand(Device::getCuda(), {64, 2 * headDim}, DType::kFloat16);

  Tensor gathered = F::lookup(cudaOps(), cache, positions);
  gathered = gathered.view({numTokens, 2, 1, headDim}).transpose(0, 1);
  Tensor expectedQuery = baselineRotaryEmbedding(query, gathered);
  Tensor expectedKey = baselineRotaryEmbedding(key, gathered);

  Tensor actualQuery = F::contiguous(cudaOps(), query);
  Tensor actualKey = F::contiguous(cudaOps(), key);
  cudaOps()->rotaryEmbedding(positions, actualQuery, actualKey, cache);

  return cpuOps()->allClose(toCpuFloat(actualQuery), toCpuFloat(expectedQuery), 5e-3f) &&
         cpuOps()->allClose(toCpuFloat(actualKey), toCpuFloat(expectedKey), 5e-3f);
}

}  // namespace

CATCH_TEST_CASE("test CUDA rotaryEmbedding", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  CATCH_REQUIRE(runCase(4, 2, 64));
  CATCH_REQUIRE(runCase(24, 8, 128));
  CATCH_REQUIRE(runCase(8, 8, 256));
}

}  // namespace fl
