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

#include "flint/metal/metal_operators.h"

#include "lutil/error.h"
#include "lutil/log.h"
#include "flint/metal/common.h"
#include "flint/metal/metallib.h"
#include "flint/metal/ops.h"
#include "flint/metal/to_device.h"

namespace fl {
namespace op {
namespace metal {

bool MetalOperators::isAvailable() {
  return mlx::core::metal::is_available();
}

std::shared_ptr<Operators> MetalOperators::create() {
  // Before anything else touches MLX: the default library is built once, when MLX first
  // constructs its Metal device, and cached from then on. Setting it afterwards does nothing.
  useEmbeddedMetallib();

  return std::shared_ptr<MetalOperators>(new MetalOperators());
}

Tensor MetalOperators::lookup(Tensor table, Tensor indices) {
  return metal::lookup(table, indices);
}

Tensor MetalOperators::layerNorm(Tensor input, Tensor weight, Tensor bias, float eps) {
  return metal::layerNorm(input, weight, bias, eps);
}

Tensor MetalOperators::groupNorm(Tensor input, Tensor weight, Tensor bias, int groups, float eps) {
  return metal::groupNorm(input, weight, bias, groups, eps);
}

Tensor MetalOperators::upsampleNearest2d(Tensor input, int scale) {
  return metal::upsampleNearest2d(input, scale);
}

Tensor MetalOperators::geglu(Tensor input) {
  return metal::geglu(input);
}

Tensor MetalOperators::swiglu(Tensor input) {
  return metal::swiglu(input);
}

Tensor MetalOperators::matmul(Tensor A, Tensor B) {
  return metal::matmul(A, B);
}

Tensor MetalOperators::conv2d(
    Tensor input,
    Tensor weight,
    Tensor bias,
    int stride,
    int padding,
    int dilation,
    int groups) {
  return metal::conv2d(input, weight, bias, stride, padding, dilation, groups);
}

Tensor MetalOperators::softmax(Tensor input) {
  return metal::softmax(input);
}

Tensor MetalOperators::attention(Tensor q, Tensor k, Tensor v, bool causal) {
  return metal::attention(q, k, v, causal);
}

Tensor MetalOperators::add(Tensor input, Tensor other) {
  return metal::add(input, other);
}

Tensor MetalOperators::sub(Tensor input, Tensor other) {
  return metal::sub(input, other);
}

Tensor MetalOperators::subFloat(Tensor input, float other) {
  return metal::subScalar(input, other);
}

Tensor MetalOperators::mul(Tensor input, Tensor other) {
  return metal::mul(input, other);
}

Tensor MetalOperators::mul(Tensor input, float other) {
  return metal::mulScalar(input, other);
}

Tensor MetalOperators::div(Tensor input, float other) {
  return metal::divScalar(input, other);
}

Tensor MetalOperators::divTensor(Tensor input, Tensor other) {
  return metal::divTensor(input, other);
}

Tensor MetalOperators::eq(Tensor input, Tensor other) {
  return metal::eq(input, other);
}

Tensor MetalOperators::neg(Tensor input) {
  return metal::neg(input);
}

Tensor MetalOperators::abs(Tensor input) {
  return metal::abs(input);
}

Tensor MetalOperators::exp(Tensor input) {
  return metal::exp(input);
}

Tensor MetalOperators::sqrt(Tensor input) {
  return metal::sqrt(input);
}

Tensor MetalOperators::rsqrt(Tensor input) {
  return metal::rsqrt(input);
}

Tensor MetalOperators::square(Tensor input) {
  return metal::square(input);
}

Tensor MetalOperators::sigmoid(Tensor input) {
  return metal::sigmoid(input);
}

Tensor MetalOperators::tanh(Tensor input) {
  return metal::tanh(input);
}

Tensor MetalOperators::relu(Tensor input) {
  return metal::relu(input);
}

Tensor MetalOperators::gelu(Tensor input) {
  return metal::gelu(input);
}

Tensor MetalOperators::silu(Tensor input) {
  return metal::silu(input);
}

Tensor MetalOperators::quickGelu(Tensor input) {
  return metal::quickGelu(input);
}

Tensor MetalOperators::sin(Tensor input) {
  return metal::sin(input);
}

Tensor MetalOperators::cos(Tensor input) {
  return metal::cos(input);
}

Tensor MetalOperators::sum(Tensor input, int dim) {
  return metal::sum(input, dim);
}

Tensor MetalOperators::max(Tensor input) {
  return metal::max(input);
}

Tensor MetalOperators::min(Tensor input) {
  return metal::min(input);
}

bool MetalOperators::all(Tensor input) {
  return metal::all(input);
}

bool MetalOperators::allClose(Tensor A, Tensor B, float rtol, float atol) {
  return metal::allClose(A, B, rtol, atol);
}

Tensor MetalOperators::tensor(lut::Span<const int> shape, DType dtype) {
  return metal::createTensor(shape, dtype);
}

Tensor MetalOperators::tensorLike(Tensor input) {
  std::vector<int> shape;
  for (int d = 0; d < input.getDim(); ++d) {
    shape.push_back(input.getShape(d));
  }
  return metal::createTensor(lut::makeConstSpan(shape), input.getDType());
}

Tensor MetalOperators::zeros(lut::Span<const int> shape, DType dtype) {
  return metal::zeros(shape, dtype);
}

void MetalOperators::fill(Tensor input, float value) {
  metal::fill(input, value);
}

void MetalOperators::copy(Tensor src, Tensor dest) {
  metal::copy(src, dest);
}

Tensor MetalOperators::cast(Tensor tensor, DType dtype) {
  return metal::cast(tensor, dtype);
}

Tensor MetalOperators::to(Device device, Tensor tensor) {
  return metal::toDevice(device, tensor);
}

void MetalOperators::print(Tensor tensor) {
  metal::print(tensor);
}

float MetalOperators::elem(Tensor tensor) {
  return metal::elem(tensor);
}

bool MetalOperators::elemBool(Tensor tensor) {
  return metal::elemBool(tensor);
}

Tensor MetalOperators::rand(lut::Span<const int> shape, DType dtype) {
  return metal::rand(shape, dtype);
}

Tensor MetalOperators::randNormal(lut::Span<const int> shape) {
  return metal::randNormal(shape);
}

void MetalOperators::manualSeed(uint64_t seed) {
  metal::manualSeed(seed);
}

DType MetalOperators::getDefaultFloatType() {
  return DType::kFloat16;
}

}  // namespace metal
}  // namespace op
}  // namespace fl
