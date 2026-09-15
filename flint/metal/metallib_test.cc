// The MIT License (MIT)
//
// Copyright (c) 2023 Xiaoyang Chen
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software
// and associated documentation files (the "Software"), to deal in the Software without
// restriction, including without limitation the rights to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all copies or
// substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
// BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
// DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "catch2/catch_amalgamated.hpp"
#include "flint/metal/metallib.h"
#include "mlx/mlx.h"

namespace fl {
namespace op {
namespace metal {

// Proves the kernels really came from the binary rather than from a stray mlx.metallib on
// disk: the metallib built next to libmlx.a is compiled in as a fallback path, so a test that
// only checked "the GPU produced the right number" would pass either way. Pointing MLX at the
// embedded copy up front is what makes the fallback unreachable.
CATCH_TEST_CASE("MLX runs on Metal from the embedded metallib", "[core][metal]") {
  CATCH_REQUIRE(getEmbeddedMetallibSize() > 0);
  CATCH_REQUIRE_NOTHROW(useEmbeddedMetallib());

  constexpr int kDim = 256;
  mlx::core::array a = mlx::core::full({kDim, kDim}, 2.0f, mlx::core::float32);
  mlx::core::array b = mlx::core::full({kDim, kDim}, 3.0f, mlx::core::float32);
  mlx::core::array c = mlx::core::sum(mlx::core::matmul(a, b, mlx::core::Device::gpu));
  mlx::core::eval(c);

  CATCH_REQUIRE(c.item<float>() == Catch::Approx(2.0f * 3.0f * kDim * kDim * kDim));
}

}  // namespace metal
}  // namespace op
}  // namespace fl
