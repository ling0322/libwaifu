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

#include <optional>

#include "lutil/error.h"
#include "lutil/log.h"
#include "flint/metal/common.h"
#include "flint/metal/ops.h"

namespace fl {
namespace op {
namespace metal {

namespace {

/// flint spells "no weight" and "no bias" as an empty tensor; MLX wants an empty optional.
std::optional<mlx::core::array> optionalArray(const Tensor &tensor) {
  if (tensor.empty()) return std::nullopt;
  return toMlxArray(tensor);
}

}  // namespace

Tensor layerNorm(const Tensor &input, const Tensor &weight, const Tensor &bias, float eps) {
  return fromMlxArray(
      mlx::core::fast::layer_norm(
          toMlxArray(input),
          optionalArray(weight),
          optionalArray(bias),
          eps));
}

Tensor groupNorm(
    const Tensor &input,
    const Tensor &weight,
    const Tensor &bias,
    int groups,
    float eps) {
  CHECK(input.getDim() == 4) << "groupNorm expects (N, C, H, W)";
  int n = input.getShape(0);
  int c = input.getShape(1);
  int h = input.getShape(2);
  int w = input.getShape(3);
  CHECK(c % groups == 0) << "groupNorm: channels must divide into the groups";

  mlx::core::array x = toMlxArray(input);
  mlx::core::Dtype outType = x.dtype();

  // MLX has no group norm of its own. Folding each group's channels and the space they cover into
  // one axis turns it into a plain normalization over that axis, which is what group norm is.
  //
  // The statistics are taken in float32 whatever the input is. A group in a VAE decoder covers a
  // few hundred thousand elements, and the sum of squares behind the variance passes what half
  // precision can hold long before the reduction ends -- it reaches infinity, and the whole
  // tensor comes back NaN.
  mlx::core::array grouped = mlx::core::reshape(
      mlx::core::astype(x, mlx::core::float32),
      {n, groups, (c / groups) * h * w});

  mlx::core::array mean = mlx::core::mean(grouped, /*axis=*/-1, /*keepdims=*/true);
  mlx::core::array var = mlx::core::var(grouped, /*axis=*/-1, /*keepdims=*/true, /*ddof=*/0);
  mlx::core::array normalized = mlx::core::multiply(
      mlx::core::subtract(grouped, mean),
      mlx::core::rsqrt(mlx::core::add(var, mlx::core::array(eps, var.dtype()))));

  mlx::core::array result =
      mlx::core::astype(mlx::core::reshape(normalized, {n, c, h, w}), outType);

  // The scale and shift are per channel, so they broadcast against (N, C, H, W) once the trailing
  // spatial axes are there for them to spread over.
  if (!weight.empty()) {
    result = mlx::core::multiply(result, mlx::core::reshape(toMlxArray(weight), {1, c, 1, 1}));
  }
  if (!bias.empty()) {
    result = mlx::core::add(result, mlx::core::reshape(toMlxArray(bias), {1, c, 1, 1}));
  }

  return fromMlxArray(result);
}

}  // namespace metal
}  // namespace op
}  // namespace fl
