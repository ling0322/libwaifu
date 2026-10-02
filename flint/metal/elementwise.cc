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

void add(const TensorView &a, const TensorView &b, const TensorView &out) {
  writeInto(mlx::core::add(toMlxArray(a), toMlxArray(b)), out);
}

void sub(const TensorView &a, const TensorView &b, const TensorView &out) {
  writeInto(mlx::core::subtract(toMlxArray(a), toMlxArray(b)), out);
}

void mul(const TensorView &a, const TensorView &b, const TensorView &out) {
  writeInto(mlx::core::multiply(toMlxArray(a), toMlxArray(b)), out);
}

void divTensor(const TensorView &a, const TensorView &b, const TensorView &out) {
  writeInto(mlx::core::divide(toMlxArray(a), toMlxArray(b)), out);
}

void eq(const TensorView &a, const TensorView &b, const TensorView &out) {
  writeInto(mlx::core::equal(toMlxArray(a), toMlxArray(b)), out);
}

void mulScalar(const TensorView &a, float other, const TensorView &out) {
  mlx::core::array x = toMlxArray(a);
  writeInto(mlx::core::multiply(x, scalarLike(other, x)), out);
}

void divScalar(const TensorView &a, float other, const TensorView &out) {
  mlx::core::array x = toMlxArray(a);
  writeInto(mlx::core::divide(x, scalarLike(other, x)), out);
}

void subScalar(const TensorView &a, float other, const TensorView &out) {
  mlx::core::array x = toMlxArray(a);
  writeInto(mlx::core::subtract(x, scalarLike(other, x)), out);
}

void neg(const TensorView &a, const TensorView &out) {
  writeInto(mlx::core::negative(toMlxArray(a)), out);
}

void abs(const TensorView &a, const TensorView &out) {
  writeInto(mlx::core::abs(toMlxArray(a)), out);
}

void exp(const TensorView &a, const TensorView &out) {
  writeInto(mlx::core::exp(toMlxArray(a)), out);
}

void log(const TensorView &a, const TensorView &out) {
  writeInto(mlx::core::log(toMlxArray(a)), out);
}

void round(const TensorView &a, const TensorView &out) {
  // MLX's round at zero decimals is rint, ties to even, as torch.round is.
  writeInto(mlx::core::round(toMlxArray(a), 0), out);
}

void sqrt(const TensorView &a, const TensorView &out) {
  writeInto(mlx::core::sqrt(toMlxArray(a)), out);
}

void rsqrt(const TensorView &a, const TensorView &out) {
  writeInto(mlx::core::rsqrt(toMlxArray(a)), out);
}

void square(const TensorView &a, const TensorView &out) {
  writeInto(mlx::core::square(toMlxArray(a)), out);
}

void sigmoid(const TensorView &a, const TensorView &out) {
  writeInto(mlx::core::sigmoid(toMlxArray(a)), out);
}

void tanh(const TensorView &a, const TensorView &out) {
  writeInto(mlx::core::tanh(toMlxArray(a)), out);
}

void relu(const TensorView &a, const TensorView &out) {
  mlx::core::array x = toMlxArray(a);
  writeInto(mlx::core::maximum(x, scalarLike(0.0f, x)), out);
}

void gelu(const TensorView &a, const TensorView &out) {
  // The exact form, 0.5x(1 + erf(x/sqrt(2))), rather than the tanh approximation: the CPU
  // reference these are tested against uses the exact one, and MLX's core has no gelu of its own.
  mlx::core::array x = toMlxArray(a);
  mlx::core::array half = scalarLike(0.5f, x);
  mlx::core::array one = scalarLike(1.0f, x);
  mlx::core::array invSqrt2 = scalarLike(0.7071067811865475f, x);

  writeInto(
      mlx::core::multiply(
          mlx::core::multiply(half, x),
          mlx::core::add(one, mlx::core::erf(mlx::core::multiply(x, invSqrt2)))), out);
}

void silu(const TensorView &a, const TensorView &out) {
  mlx::core::array x = toMlxArray(a);
  writeInto(mlx::core::multiply(x, mlx::core::sigmoid(x)), out);
}

void quickGelu(const TensorView &a, const TensorView &out) {
  mlx::core::array x = toMlxArray(a);
  mlx::core::array gate = mlx::core::sigmoid(mlx::core::multiply(x, scalarLike(1.702f, x)));
  writeInto(mlx::core::multiply(x, gate), out);
}

void sin(const TensorView &a, const TensorView &out) {
  writeInto(mlx::core::sin(toMlxArray(a)), out);
}

void cos(const TensorView &a, const TensorView &out) {
  writeInto(mlx::core::cos(toMlxArray(a)), out);
}

void softmax(const TensorView &a, const TensorView &out) {
  writeInto(mlx::core::softmax(toMlxArray(a), -1), out);
}

}  // namespace metal
}  // namespace op
}  // namespace fl
