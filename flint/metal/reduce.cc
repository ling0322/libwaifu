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

void sum(const TensorView &input, int dim, const TensorView &out) {
  // `dim` is already non-negative and `out` is the input without it -- (1) for a vector, which
  // the scalar MLX gives back fills just the same. MLX promotes a sum of booleans or of small
  // integers; the result keeps the input's type, as functional says it does on every device.
  mlx::core::array x = toMlxArray(input);
  mlx::core::array result = mlx::core::sum(x, dim, /*keepdims=*/false);
  writeInto(mlx::core::astype(result, x.dtype()), out);
}

void cumsum(const TensorView &input, int dim, const TensorView &out) {
  // MLX's own scan: forward, and inclusive of the element itself, which is what torch.cumsum is.
  mlx::core::array x = toMlxArray(input);
  mlx::core::array result = mlx::core::cumsum(x, dim, /*reverse=*/false, /*inclusive=*/true);
  writeInto(mlx::core::astype(result, x.dtype()), out);
}

void max(const TensorView &input, const TensorView &out) {
  // Over the last dimension only, as functional asks of every device: `out` is the input without
  // it, (1) for a vector.
  writeInto(mlx::core::max(toMlxArray(input), /*axis=*/-1, /*keepdims=*/false), out);
}

void min(const TensorView &input, const TensorView &out) {
  writeInto(mlx::core::min(toMlxArray(input), /*axis=*/-1, /*keepdims=*/false), out);
}

bool all(const TensorView &input) {
  mlx::core::array result = mlx::core::all(toMlxArray(input));
  mlx::core::eval(result);
  return result.item<bool>();
}

bool allClose(const TensorView &a, const TensorView &b, float rtol, float atol) {
  mlx::core::array x = toMlxArray(a);
  mlx::core::array y = toMlxArray(b);

  // Comparing across dtypes is what a fp16 result checked against a fp32 reference needs, and
  // allclose refuses to broadcast two different types on its own.
  if (x.dtype() != y.dtype()) {
    y = mlx::core::astype(y, x.dtype());
  }

  mlx::core::array result = mlx::core::allclose(x, y, rtol, atol);
  mlx::core::eval(result);
  return result.item<bool>();
}

float elem(const TensorView &tensor) {
  CHECK(tensor.getNumEl() == 1) << "elem: expected a tensor of one element";

  mlx::core::array value = mlx::core::astype(toMlxArray(tensor), mlx::core::float32);
  mlx::core::eval(value);
  return value.item<float>();
}

bool elemBool(const TensorView &tensor) {
  CHECK(tensor.getNumEl() == 1) << "elemBool: expected a tensor of one element";

  mlx::core::array value = mlx::core::astype(toMlxArray(tensor), mlx::core::bool_);
  mlx::core::eval(value);
  return value.item<bool>();
}

}  // namespace metal
}  // namespace op
}  // namespace fl
