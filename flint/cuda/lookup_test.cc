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

Tensor toCuda(const Tensor &a) {
  return F::cast(cudaOps(), F::toDevice(cudaOps(), a, Device::getCuda()), DType::kFloat16);
}

Tensor toCpu(const Tensor &a) {
  return F::toDevice(cudaOps(), F::cast(cudaOps(), a, DType::kFloat), Device::getCpu());
}

}  // namespace

CATCH_TEST_CASE("test CUDA lookup", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  Tensor embd = F::rand(Device::getCpu(), {10, 32}, DType::kFloat);
  Tensor ids = Tensor::create<LongType>({2, 3}, {1, 2, 3, 4, 5, 6});
  Tensor xr = F::lookup(cpuOps(), embd, ids);

  Tensor x = F::lookup(cudaOps(), toCuda(embd), F::toDevice(cudaOps(), ids, Device::getCuda()));

  CATCH_REQUIRE(cpuOps()->allClose(toCpu(x), xr));

  // packed indices are 1D and give one embedding row per index.
  Tensor packedIds = Tensor::create<LongType>({3}, {1, 2, 3});
  Tensor packedRef = F::lookup(cpuOps(), embd, packedIds);
  Tensor packed = F::lookup(cudaOps(), 
      toCuda(embd),
      F::toDevice(cudaOps(), packedIds, Device::getCuda()));

  CATCH_REQUIRE(packed.getShape() == std::vector<int>{3, 32});
  CATCH_REQUIRE(cpuOps()->allClose(toCpu(packed), packedRef));
}

CATCH_TEST_CASE("test CUDA lookup (more ids than a grid axis holds)", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  // The kernel used to put the ids on the grid's y and z axes, which hold 65 535 blocks each, so
  // anything longer failed to launch. w2v-bert asks for one id per pair of frames: 250 000 for
  // ten seconds of speech. Both the packed and the batched layout, and a batch longer than an
  // axis as well as a sequence longer than one.
  const int many = 70000;
  std::vector<LongType> values(many);
  for (int i = 0; i < many; ++i) values[i] = (i * 7) % 73;

  Tensor embd = F::rand(Device::getCpu(), {73, 64}, DType::kFloat);
  Tensor cudaEmbd = F::toDevice(cudaOps(), embd, Device::getCuda());

  Tensor packed = Tensor::create<LongType>({many}, values);
  Tensor x = F::lookup(cudaOps(), cudaEmbd, F::toDevice(cudaOps(), packed, Device::getCuda()));
  CATCH_REQUIRE(x.getShape() == std::vector<int>{many, 64});
  CATCH_REQUIRE(cpuOps()->allClose(toCpu(x), F::lookup(cpuOps(), embd, packed)));

  Tensor tall = Tensor::create<LongType>({many, 1}, values);
  Tensor y = F::lookup(cudaOps(), cudaEmbd, F::toDevice(cudaOps(), tall, Device::getCuda()));
  CATCH_REQUIRE(y.getShape() == std::vector<int>{many, 1, 64});
  CATCH_REQUIRE(cpuOps()->allClose(toCpu(y), F::lookup(cpuOps(), embd, tall)));

  Tensor wide = Tensor::create<LongType>({1, many}, values);
  Tensor z = F::lookup(cudaOps(), cudaEmbd, F::toDevice(cudaOps(), wide, Device::getCuda()));
  CATCH_REQUIRE(z.getShape() == std::vector<int>{1, many, 64});
  CATCH_REQUIRE(cpuOps()->allClose(toCpu(z), F::lookup(cpuOps(), embd, wide)));
}

CATCH_TEST_CASE("test CUDA lookup (embedding widths)", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  // A row is copied by a grid that is 256 threads wide, so widths on both sides of the block
  // boundary decide whether the tail of a row is reached at all.
  for (int width : {1, 255, 256, 257, 1000}) {
    Tensor embd = F::rand(Device::getCpu(), {6, width}, DType::kFloat);
    Tensor ids = Tensor::create<LongType>({2, 3}, {0, 1, 2, 3, 4, 5});

    Tensor x = F::lookup(cudaOps(), toCuda(embd), F::toDevice(cudaOps(), ids, Device::getCuda()));
    CATCH_INFO("width = " << width);
    CATCH_REQUIRE(x.getShape() == std::vector<int>{2, 3, width});
    CATCH_REQUIRE(cpuOps()->allClose(toCpu(x), F::lookup(cpuOps(), embd, ids), 5e-3));
  }
}

CATCH_TEST_CASE("test CUDA lookup (index edges)", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  // The first and last rows of the table are where an off-by-one in the row offset shows up, and
  // a repeated index must return the same row every time rather than advancing.
  Tensor embd = F::rand(Device::getCpu(), {5, 8}, DType::kFloat);
  Tensor ids = Tensor::create<LongType>({2, 3}, {0, 4, 0, 4, 2, 2});

  Tensor x = F::lookup(cudaOps(), toCuda(embd), F::toDevice(cudaOps(), ids, Device::getCuda()));
  CATCH_REQUIRE(cpuOps()->allClose(toCpu(x), F::lookup(cpuOps(), embd, ids), 5e-3));

  // a table with a single row, so every index has to resolve to it.
  Tensor single = F::rand(Device::getCpu(), {1, 8}, DType::kFloat);
  Tensor zeros = Tensor::create<LongType>({3}, {0, 0, 0});
  Tensor y = F::lookup(cudaOps(), toCuda(single), F::toDevice(cudaOps(), zeros, Device::getCuda()));
  CATCH_REQUIRE(y.getShape() == std::vector<int>{3, 8});
  CATCH_REQUIRE(cpuOps()->allClose(toCpu(y), F::lookup(cpuOps(), single, zeros), 5e-3));
}

CATCH_TEST_CASE("test CUDA lookup (float table)", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  // The operator has a separate instantiation for a float table; moving the table across
  // without casting is what selects it.
  Tensor embd = F::rand(Device::getCpu(), {6, 16}, DType::kFloat);
  Tensor ids = Tensor::create<LongType>({2, 2}, {0, 5, 3, 1});

  Tensor table = F::toDevice(cudaOps(), embd, Device::getCuda());
  CATCH_REQUIRE(table.getDType() == DType::kFloat);

  Tensor x = F::lookup(cudaOps(), table, F::toDevice(cudaOps(), ids, Device::getCuda()));
  CATCH_REQUIRE(x.getDType() == DType::kFloat);
  CATCH_REQUIRE(
      cpuOps()->allClose(F::toDevice(cudaOps(), x, Device::getCpu()), F::lookup(cpuOps(), embd, ids)));

  // and the packed 1D form of the same table.
  Tensor packedIds = Tensor::create<LongType>({2}, {4, 2});
  Tensor packed = F::lookup(cudaOps(), table, F::toDevice(cudaOps(), packedIds, Device::getCuda()));
  CATCH_REQUIRE(packed.getShape() == std::vector<int>{2, 16});
  CATCH_REQUIRE(cpuOps()->allClose(
      F::toDevice(cudaOps(), packed, Device::getCpu()),
      F::lookup(cpuOps(), embd, packedIds)));
}

}  // namespace fl
