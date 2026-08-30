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

#include "flint/metal/common.h"
#include "flint/metal/ops.h"

namespace fl {
namespace op {
namespace metal {

namespace {

/// A scalar in the same dtype as the tensor it is combined with, so that a float16 tensor stays
/// float16 instead of being promoted the way a bare float literal would promote it.
mlx::core::array scalarLike(float value, const mlx::core::array &a) {
  return mlx::core::array(value, a.dtype());
}

}  // namespace

Tensor add(const Tensor &a, const Tensor &b) {
  return fromMlxArray(mlx::core::add(toMlxArray(a), toMlxArray(b)));
}

Tensor sub(const Tensor &a, const Tensor &b) {
  return fromMlxArray(mlx::core::subtract(toMlxArray(a), toMlxArray(b)));
}

Tensor mul(const Tensor &a, const Tensor &b) {
  return fromMlxArray(mlx::core::multiply(toMlxArray(a), toMlxArray(b)));
}

Tensor divTensor(const Tensor &a, const Tensor &b) {
  return fromMlxArray(mlx::core::divide(toMlxArray(a), toMlxArray(b)));
}

Tensor eq(const Tensor &a, const Tensor &b) {
  return fromMlxArray(mlx::core::equal(toMlxArray(a), toMlxArray(b)));
}

Tensor mulScalar(const Tensor &a, float other) {
  mlx::core::array x = toMlxArray(a);
  return fromMlxArray(mlx::core::multiply(x, scalarLike(other, x)));
}

Tensor divScalar(const Tensor &a, float other) {
  mlx::core::array x = toMlxArray(a);
  return fromMlxArray(mlx::core::divide(x, scalarLike(other, x)));
}

Tensor subScalar(const Tensor &a, float other) {
  mlx::core::array x = toMlxArray(a);
  return fromMlxArray(mlx::core::subtract(x, scalarLike(other, x)));
}

Tensor neg(const Tensor &a) {
  return fromMlxArray(mlx::core::negative(toMlxArray(a)));
}

Tensor abs(const Tensor &a) {
  return fromMlxArray(mlx::core::abs(toMlxArray(a)));
}

Tensor exp(const Tensor &a) {
  return fromMlxArray(mlx::core::exp(toMlxArray(a)));
}

Tensor sqrt(const Tensor &a) {
  return fromMlxArray(mlx::core::sqrt(toMlxArray(a)));
}

Tensor rsqrt(const Tensor &a) {
  return fromMlxArray(mlx::core::rsqrt(toMlxArray(a)));
}

Tensor square(const Tensor &a) {
  return fromMlxArray(mlx::core::square(toMlxArray(a)));
}

Tensor sigmoid(const Tensor &a) {
  return fromMlxArray(mlx::core::sigmoid(toMlxArray(a)));
}

Tensor tanh(const Tensor &a) {
  return fromMlxArray(mlx::core::tanh(toMlxArray(a)));
}

Tensor relu(const Tensor &a) {
  mlx::core::array x = toMlxArray(a);
  return fromMlxArray(mlx::core::maximum(x, scalarLike(0.0f, x)));
}

Tensor gelu(const Tensor &a) {
  // The exact form, 0.5x(1 + erf(x/sqrt(2))), rather than the tanh approximation: the CPU
  // reference these are tested against uses the exact one, and MLX's core has no gelu of its own.
  mlx::core::array x = toMlxArray(a);
  mlx::core::array half = scalarLike(0.5f, x);
  mlx::core::array one = scalarLike(1.0f, x);
  mlx::core::array invSqrt2 = scalarLike(0.7071067811865475f, x);

  return fromMlxArray(
      mlx::core::multiply(
          mlx::core::multiply(half, x),
          mlx::core::add(one, mlx::core::erf(mlx::core::multiply(x, invSqrt2)))));
}

Tensor silu(const Tensor &a) {
  mlx::core::array x = toMlxArray(a);
  return fromMlxArray(mlx::core::multiply(x, mlx::core::sigmoid(x)));
}

Tensor quickGelu(const Tensor &a) {
  mlx::core::array x = toMlxArray(a);
  return fromMlxArray(
      mlx::core::multiply(x, mlx::core::sigmoid(mlx::core::multiply(x, scalarLike(1.702f, x)))));
}

Tensor sin(const Tensor &a) {
  return fromMlxArray(mlx::core::sin(toMlxArray(a)));
}

Tensor cos(const Tensor &a) {
  return fromMlxArray(mlx::core::cos(toMlxArray(a)));
}

Tensor softmax(const Tensor &a) {
  return fromMlxArray(mlx::core::softmax(toMlxArray(a), -1));
}

}  // namespace metal
}  // namespace op
}  // namespace fl
