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

#include "lutil/error.h"
#include "lutil/log.h"
#include "flint/metal/common.h"
#include "flint/metal/ops.h"

namespace fl {
namespace op {
namespace metal {

void conv1d(
    const TensorView &input,
    const TensorView &weight,
    const TensorView &bias,
    int stride,
    int padding,
    int dilation,
    int groups,
    const TensorView &out) {
  CHECK(input.getDim() == 3) << "conv1d expects (N, C, L)";
  CHECK(weight.getDim() == 3) << "conv1d expects a (K, C / groups, R) weight";

  // A file of its own rather than beside conv2d, because the two share nothing here: each is a
  // transpose, one MLX call and a transpose back. The CPU's are one function because there they
  // share an im2col and a GEMM.
  //
  // flint is channels first, MLX's conv1d is channels last, so both operands turn inside out on
  // the way in and the result turns back on the way out. The transposes are real copies; if this
  // ever shows up in a profile the answer is to keep activations in NLC across the whole Metal
  // backend rather than to shave anything here.
  mlx::core::array x = mlx::core::transpose(toMlxArray(input), {0, 2, 1});
  mlx::core::array w = mlx::core::transpose(toMlxArray(weight), {0, 2, 1});

  mlx::core::array result = mlx::core::conv1d(
      x,
      w,
      /*stride=*/stride,
      /*padding=*/padding,
      /*dilation=*/dilation,
      groups);

  result = mlx::core::transpose(result, {0, 2, 1});

  if (!bias.empty()) {
    int k = weight.getShape(0);
    result = mlx::core::add(result, mlx::core::reshape(toMlxArray(bias), {1, k, 1}));
  }

  // The result is the input's type, as functional has it, even where a weight or a bias of
  // another type would have MLX promote it. A no-op when they agree, which is the usual case.
  writeInto(mlx::core::astype(result, toMlxDtype(out.getDType())), out);
}

}  // namespace metal
}  // namespace op
}  // namespace fl
