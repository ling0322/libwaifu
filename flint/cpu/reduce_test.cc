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
#include "flint/operators.h"
#include "flint/tensor.h"

namespace fl {

namespace {

/// The CPU operators, which is what the calls in this file are asked of.
Operators *cpuOps() {
  return getOperators(Device::kCpu);
}

}  // namespace

namespace op {
namespace cpu {

CATCH_TEST_CASE("test CPU reductions", "[core][nn][operators]") {
  Tensor a = Tensor::create<float>({2, 3}, {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f});

  CATCH_REQUIRE(cpuOps()->allClose(cpuOps()->sum(a, -1), Tensor::create<float>({2}, {0.6f, 1.5f})));
  CATCH_REQUIRE(cpuOps()->allClose(cpuOps()->max(a), Tensor::create<float>({2}, {0.3f, 0.6f})));
}

CATCH_TEST_CASE("test CPU sum over every dimension of a rank-4 tensor", "[core][nn][operators]") {
  // Each element is its own coordinates written as digits, so a sum that lands in the wrong place
  // -- the old transpose put the last dimension where the summed one had been -- is a wrong value
  // and not only a wrong shape.
  const int shape[4] = {2, 3, 4, 5};
  std::vector<float> values;
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 3; ++j)
      for (int k = 0; k < 4; ++k)
        for (int l = 0; l < 5; ++l) values.push_back(1000.0f * i + 100.0f * j + 10.0f * k + l);
  Tensor a = Tensor::create<float>({2, 3, 4, 5}, values);

  for (int dim = 0; dim < 4; ++dim) {
    std::vector<int> kept;
    for (int d = 0; d < 4; ++d)
      if (d != dim) kept.push_back(shape[d]);

    // The longhand sum: walk every element and add it into the slot it lands in without `dim`.
    std::vector<float> want(kept[0] * kept[1] * kept[2], 0.0f);
    int index[4];
    for (index[0] = 0; index[0] < 2; ++index[0])
      for (index[1] = 0; index[1] < 3; ++index[1])
        for (index[2] = 0; index[2] < 4; ++index[2])
          for (index[3] = 0; index[3] < 5; ++index[3]) {
            int slot = 0;
            for (int d = 0; d < 4; ++d)
              if (d != dim) slot = slot * shape[d] + index[d];
            int flat = ((index[0] * 3 + index[1]) * 4 + index[2]) * 5 + index[3];
            want[slot] += values[flat];
          }

    Tensor got = cpuOps()->sum(a, dim);
    CATCH_INFO("dim = " << dim);
    CATCH_REQUIRE(got.getShape() == kept);
    CATCH_REQUIRE(cpuOps()->allClose(got, Tensor::create<float>({kept[0], kept[1], kept[2]}, want)));
    CATCH_REQUIRE(cpuOps()->allClose(cpuOps()->sum(a, dim - 4), got));
  }
}

}  // namespace cpu
}  // namespace op
}  // namespace fl
