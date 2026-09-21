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

  Tensor x = F::toDevice(Device::getCuda(), a);
  Tensor y = F::toDevice(Device::getCuda(), b);
  x = F::cast(x, DType::kFloat16);
  y = F::cast(y, DType::kFloat16);
  y = y.slice(1, {128, 256});
  y = y.transpose(1, 0);
  x = mm->apply(x, y);
  x = F::cast(x, DType::kFloat);
  x = F::toDevice(Device::getCpu(), x);

  CATCH_REQUIRE(F::allClose(x, xr, 1e-2f));
}

CATCH_TEST_CASE("test matmul bmm (cutlass)", "[fl][op][cuda][cutlass]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  std::shared_ptr<op::cuda::MatMul> mm = op::cuda::MatMul::createCutlass();

  Tensor a = F::rand({5, 10, 8, 24}, DType::kFloat);
  Tensor b = F::rand({10, 64, 24}, DType::kFloat);
  Tensor xr = F::matmul(a, b.slice(1, {8, 32}).transpose(-1, -2));

  Tensor x = F::toDevice(Device::getCuda(), a);
  Tensor y = F::toDevice(Device::getCuda(), b);
  x = F::cast(x, DType::kFloat16);
  y = F::cast(y, DType::kFloat16);
  y = y.slice(1, {8, 32});
  y = y.transpose(-1, -2);
  x = mm->apply(x, y);
  x = F::cast(x, DType::kFloat);
  x = F::toDevice(Device::getCpu(), x);

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

  Tensor x = F::cast(F::toDevice(Device::getCuda(), a), DType::kFloat16);
  Tensor y = F::cast(F::toDevice(Device::getCuda(), b), DType::kFloat16);
  Tensor actual = F::toDevice(Device::getCpu(), F::cast(mm->apply(x, y), DType::kFloat));

  CATCH_REQUIRE(F::allClose(actual, expected, 2e-3f));
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

  Tensor a = F::rand({kM, kK}, DType::kFloat);
  Tensor b = F::rand({kK, kN}, DType::kFloat);
  Tensor expected = F::matmul(a, b);

  Tensor x = F::cast(F::toDevice(Device::getCuda(), a), DType::kFloat16);
  Tensor y = F::cast(F::toDevice(Device::getCuda(), b), DType::kFloat16);
  Tensor actual = F::toDevice(Device::getCpu(), F::cast(mm->apply(x, y), DType::kFloat));

  CATCH_REQUIRE(F::allClose(actual, expected, 1e-2f));
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
    Tensor a = F::rand({m, k}, DType::kFloat);
    Tensor b = F::rand({k, n}, DType::kFloat);
    Tensor expected = F::matmul(a, b);

    Tensor x = F::cast(F::toDevice(Device::getCuda(), a), DType::kFloat16);
    Tensor y = F::cast(F::toDevice(Device::getCuda(), b), DType::kFloat16);
    Tensor actual = F::toDevice(Device::getCpu(), F::cast(mm->apply(x, y), DType::kFloat));

    CATCH_INFO("m = " << m << ", k = " << k << ", n = " << n);
    CATCH_REQUIRE(actual.getShape() == std::vector<int>{m, n});
    return F::allClose(actual, expected, 1e-2f);
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

}  // namespace fl
