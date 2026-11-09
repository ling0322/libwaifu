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

#include <algorithm>
#include <cmath>
#include <memory>

#include "catch2/catch_amalgamated.hpp"
#include "flint/test_functional.h"
#include "flint/cuda/matmul.h"
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

/// A @ B through `mm`, into a result allocated the way fl::test::matmul would.
Tensor apply(op::cuda::MatMul *mm, const Tensor &A, const Tensor &B) {
  std::vector<int> shape = A.getShape();
  shape.back() = B.getShape(-1);
  Tensor C = F::empty(A.getDevice(), shape, A.getDType());
  mm->apply(A, B, C);
  return C;
}

}  // namespace


CATCH_TEST_CASE("test matmul gemm (cutlass)", "[fl][op][cuda][cutlass]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  std::shared_ptr<op::cuda::MatMul> mm = op::cuda::MatMul::createCutlass();

  Tensor a = fl::test::rand(Device::getCpu(), {10, 128}, DType::kFloat);
  Tensor b = fl::test::rand(Device::getCpu(), {40, 256}, DType::kFloat);
  Tensor xr = fl::test::matmul(cpuOps(), a, b.slice(1, {128, 256}).transpose(1, 0));

  Tensor x = fl::test::toDevice(cudaOps(), a, Device::getCuda());
  Tensor y = fl::test::toDevice(cudaOps(), b, Device::getCuda());
  x = fl::test::cast(cudaOps(), x, DType::kFloat16);
  y = fl::test::cast(cudaOps(), y, DType::kFloat16);
  y = y.slice(1, {128, 256});
  y = y.transpose(1, 0);
  x = apply(mm.get(), x, y);
  x = fl::test::cast(cudaOps(), x, DType::kFloat);
  x = fl::test::toDevice(cudaOps(), x, Device::getCpu());

  CATCH_REQUIRE(cpuOps()->allClose(x, xr, 1e-2f));
}

CATCH_TEST_CASE("test matmul bmm (cutlass)", "[fl][op][cuda][cutlass]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  std::shared_ptr<op::cuda::MatMul> mm = op::cuda::MatMul::createCutlass();

  Tensor a = fl::test::rand(Device::getCpu(), {5, 10, 8, 24}, DType::kFloat);
  Tensor b = fl::test::rand(Device::getCpu(), {10, 64, 24}, DType::kFloat);
  Tensor xr = fl::test::matmul(cpuOps(), a, b.slice(1, {8, 32}).transpose(-1, -2));

  Tensor x = fl::test::toDevice(cudaOps(), a, Device::getCuda());
  Tensor y = fl::test::toDevice(cudaOps(), b, Device::getCuda());
  x = fl::test::cast(cudaOps(), x, DType::kFloat16);
  y = fl::test::cast(cudaOps(), y, DType::kFloat16);
  y = y.slice(1, {8, 32});
  y = y.transpose(-1, -2);
  x = apply(mm.get(), x, y);
  x = fl::test::cast(cudaOps(), x, DType::kFloat);
  x = fl::test::toDevice(cudaOps(), x, Device::getCpu());

  CATCH_REQUIRE(cpuOps()->allClose(x, xr, 5e-3f));
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

  Tensor a = fl::test::rand(Device::getCpu(), {kM, kK}, DType::kFloat);
  Tensor b = fl::test::rand(Device::getCpu(), {kK, kN}, DType::kFloat);

  // The reference is computed from the half values, not the float ones, so what this measures is
  // the accumulation rather than the rounding of the inputs.
  Tensor halfA = fl::test::cast(cpuOps(), fl::test::cast(cpuOps(), a, DType::kFloat16), DType::kFloat);
  Tensor halfB = fl::test::cast(cpuOps(), fl::test::cast(cpuOps(), b, DType::kFloat16), DType::kFloat);
  Tensor expected = fl::test::matmul(cpuOps(), halfA, halfB);

  Tensor x = fl::test::cast(cudaOps(), fl::test::toDevice(cudaOps(), a, Device::getCuda()), DType::kFloat16);
  Tensor y = fl::test::cast(cudaOps(), fl::test::toDevice(cudaOps(), b, Device::getCuda()), DType::kFloat16);
  Tensor actual = fl::test::toDevice(cudaOps(), fl::test::cast(cudaOps(), apply(mm.get(), x, y), DType::kFloat), Device::getCpu());

  CATCH_REQUIRE(cpuOps()->allClose(actual, expected, 2e-3f));
}

/// A leading dimension that is not a multiple of eight, which is what Anima's patch embedder
/// contracts over: 68 channels in, 2048 out.
///
/// The wide kernel reads its operands eight halves at a time, so a row length of 68 puts every
/// row after the first eight bytes off a 16-byte boundary. CUTLASS does not notice on its own --
/// `initialize` builds the params and returns success, and only `can_implement` looks at the
/// strides -- so this used to launch anyway and fault the device, which surfaced as
/// `misaligned address` from whichever unrelated CUDA call next asked for a status.
CATCH_TEST_CASE("test matmul gemm (cutlass, unaligned K)", "[fl][op][cuda][cutlass]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  constexpr int kM = 128;
  constexpr int kK = 68;
  constexpr int kN = 256;

  std::shared_ptr<op::cuda::MatMul> mm = op::cuda::MatMul::createCutlass();

  Tensor a = fl::test::rand(Device::getCpu(), {kM, kK}, DType::kFloat);
  Tensor b = fl::test::rand(Device::getCpu(), {kK, kN}, DType::kFloat);
  Tensor expected = fl::test::matmul(cpuOps(), a, b);

  Tensor x = fl::test::cast(cudaOps(), fl::test::toDevice(cudaOps(), a, Device::getCuda()), DType::kFloat16);
  Tensor y = fl::test::cast(cudaOps(), fl::test::toDevice(cudaOps(), b, Device::getCuda()), DType::kFloat16);
  Tensor actual = fl::test::toDevice(cudaOps(), fl::test::cast(cudaOps(), apply(mm.get(), x, y), DType::kFloat), Device::getCpu());

  CATCH_REQUIRE(cpuOps()->allClose(actual, expected, 1e-2f));
}

CATCH_TEST_CASE("test matmul gemm (cutlass, odd leading dimension)", "[fl][op][cuda][cutlass]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  // Below the {8, 4, 2} ladder. A staged tensor-core mainloop moves its operands with `cp_async`,
  // which reads 4, 8 or 16 bytes and nothing else, so two halves is the narrowest access it has
  // and an odd leading dimension has no instantiation there at all. These are copied into zeroed
  // buffers at extents rounded up to eight and run there instead.
  //
  // What is checked is that they come back right rather than merely come back. The padding is
  // supposed to be exact -- zeros add nothing to a sum -- so these compare against the reference
  // at the same tolerance an aligned shape does.
  std::shared_ptr<op::cuda::MatMul> mm = op::cuda::MatMul::createCutlass();

  auto runCase = [&mm](int m, int k, int n) {
    Tensor a = fl::test::rand(Device::getCpu(), {m, k}, DType::kFloat);
    Tensor b = fl::test::rand(Device::getCpu(), {k, n}, DType::kFloat);
    Tensor expected = fl::test::matmul(cpuOps(), a, b);

    Tensor x = fl::test::cast(cudaOps(), fl::test::toDevice(cudaOps(), a, Device::getCuda()), DType::kFloat16);
    Tensor y = fl::test::cast(cudaOps(), fl::test::toDevice(cudaOps(), b, Device::getCuda()), DType::kFloat16);
    Tensor actual = fl::test::toDevice(cudaOps(), fl::test::cast(cudaOps(), apply(mm.get(), x, y), DType::kFloat), Device::getCpu());

    CATCH_INFO("m = " << m << ", k = " << k << ", n = " << n);
    CATCH_REQUIRE(actual.getShape() == std::vector<int>{m, n});
    return cpuOps()->allClose(actual, expected, 1e-2f);
  };

  // An odd N, which is the leading dimension of both B and the output at once.
  CATCH_REQUIRE(runCase(8, 16, 1));
  CATCH_REQUIRE(runCase(8, 16, 35));
  // An odd K, which is A's alone.
  CATCH_REQUIRE(runCase(8, 17, 16));

  // A K past one 32-wide step of the mainloop. This is the case that catches padding the leading
  // dimension without padding the extent to match: the kernel walks K a tile at a time, so a row
  // longer than the K it was given is read off the end of the last one. It faulted the device
  // while the three above passed.
  CATCH_REQUIRE(runCase(8, 33, 16));

  // Every extent odd at once, and each past a threadblock tile so there is more than one of them.
  CATCH_REQUIRE(runCase(129, 33, 131));
}

// Products with too few tiles to fill the card over a K of 4096, which is what splits K: the
// product is cut into slices of K, and each slice's sum is handed to the next. Summed
// in float and rounded into half once, nearly every element is the float sum rounded to half --
// all but the few a different order of summation puts on the other side of a rounding boundary,
// and those one step of half away. Handed from slice to slice in half, the running sum is rounded
// at every slice, and some tenths of the elements come out a step or two away. With a bias, which
// the epilogue adds, the same: the float sum and the bias, rounded once.
CATCH_TEST_CASE("test matmul gemm (cutlass, split K sums in float)", "[fl][op][cuda][cutlass]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  constexpr int kK = 4096;
  std::shared_ptr<op::cuda::MatMul> mm = op::cuda::MatMul::createCutlass();

  // Operands as half holds them, on both sides, so that only the summation is compared.
  auto asHalf = [](const Tensor &x) {
    return fl::test::cast(cpuOps(), fl::test::cast(cpuOps(), x, DType::kFloat16), DType::kFloat);
  };
  auto toCuda = [](const Tensor &x) {
    return fl::test::cast(cudaOps(), fl::test::toDevice(cudaOps(), x, Device::getCuda()), DType::kFloat16);
  };

  // How many of `got`'s elements are `sum` rounded to half, and the largest distance in steps of
  // half at that magnitude.
  auto agreement = [&](const Tensor &got, const Tensor &sum) {
    Tensor once = asHalf(sum);
    Tensor host = fl::test::toDevice(cudaOps(), fl::test::cast(cudaOps(), got, DType::kFloat), Device::getCpu());
    const float *g = host.getInternalData()->getData<float>(host.getInternalOffset());
    const float *w = once.getInternalData()->getData<float>(once.getInternalOffset());
    int64_t same = 0;
    float steps = 0.0f;
    for (int64_t i = 0; i < host.getNumEl(); ++i) {
      if (g[i] == w[i]) ++same;
      float ulp = std::ldexp(1.0f, std::ilogb(w[i]) - 10);
      steps = std::max(steps, std::fabs(g[i] - w[i]) / ulp);
    }
    return std::make_pair(double(same) / host.getNumEl(), steps);
  };

  // One tile, which is cut into slices of K as the serial split cut it, and sixteen, which
  // Stream-K shares out over the card on its own -- the two ways a half product that splits is run.
  for (auto [m, n] : {std::pair<int, int>{16, 128}, {128, 2048}}) {
    CATCH_INFO(m << " x " << n << " x " << kK);
    Tensor a = asHalf(fl::test::rand(Device::getCpu(), {m, kK}, DType::kFloat));
    Tensor b = asHalf(fl::test::rand(Device::getCpu(), {kK, n}, DType::kFloat));
    Tensor bias = asHalf(fl::test::rand(Device::getCpu(), {n}, DType::kFloat));

    Tensor sum = fl::test::matmul(cpuOps(), a, b);
    auto [same, steps] = agreement(apply(mm.get(), toCuda(a), toCuda(b)), sum);
    CATCH_INFO("without a bias: " << same * 100 << "% the float sum rounded once, at most " << steps << " steps away");
    CATCH_REQUIRE(same >= 0.95);
    CATCH_REQUIRE(steps <= 1.0f);

    Tensor c = F::empty(Device::getCuda(), {m, n}, DType::kFloat16);
    mm->apply(toCuda(a), toCuda(b), toCuda(bias), c);
    auto [sameBias, stepsBias] = agreement(c, fl::test::add(cpuOps(), sum, bias));
    CATCH_INFO("with a bias: " << sameBias * 100 << "% the float sum rounded once, at most " << stepsBias << " steps away");
    CATCH_REQUIRE(sameBias >= 0.95);
    CATCH_REQUIRE(stepsBias <= 1.0f);
  }
}

}  // namespace fl
