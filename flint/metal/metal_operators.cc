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

void MetalOperators::lookup(TensorView table, TensorView indices, TensorView out) {
  metal::lookup(table, indices, out);
}

void MetalOperators::layerNorm(
    TensorView input,
    TensorView weight,
    TensorView bias,
    float eps,
    TensorView out) {
  metal::layerNorm(input, weight, bias, eps, out);
}

void MetalOperators::rmsNorm(TensorView input, TensorView weight, float eps, TensorView out) {
  metal::rmsNorm(input, weight, eps, out);
}

void MetalOperators::groupNorm(
    TensorView input,
    TensorView weight,
    TensorView bias,
    int groups,
    float eps,
    TensorView out) {
  metal::groupNorm(input, weight, bias, groups, eps, out);
}

void MetalOperators::upsampleNearest2d(TensorView input, int scale, TensorView out) {
  metal::upsampleNearest2d(input, scale, out);
}

void MetalOperators::geglu(TensorView input, TensorView out) {
  metal::geglu(input, out);
}

void MetalOperators::swiglu(TensorView input, TensorView out) {
  metal::swiglu(input, out);
}

void MetalOperators::matmul(TensorView A, TensorView B, TensorView out) {
  metal::matmul(A, B, out);
}

void MetalOperators::conv2d(
    TensorView input,
    TensorView weight,
    TensorView bias,
    int stride,
    int padding,
    int dilation,
    int groups,
    TensorView out) {
  metal::conv2d(input, weight, bias, stride, padding, dilation, groups, out);
}

void MetalOperators::conv1d(
    TensorView input,
    TensorView weight,
    TensorView bias,
    int stride,
    int padding,
    int dilation,
    int groups,
    TensorView out) {
  metal::conv1d(input, weight, bias, stride, padding, dilation, groups, out);
}

void MetalOperators::softmax(TensorView input, TensorView out) {
  metal::softmax(input, out);
}

void MetalOperators::attention(
    TensorView q,
    TensorView k,
    TensorView v,
    bool causal,
    TensorView out) {
  metal::attention(q, k, v, causal, out);
}

void MetalOperators::add(TensorView input, TensorView other, TensorView out) {
  metal::add(input, other, out);
}

void MetalOperators::sub(TensorView input, TensorView other, TensorView out) {
  metal::sub(input, other, out);
}

void MetalOperators::mul(TensorView input, TensorView other, TensorView out) {
  metal::mul(input, other, out);
}

void MetalOperators::mul(TensorView input, float other, TensorView out) {
  metal::mulScalar(input, other, out);
}

void MetalOperators::div(TensorView input, float other, TensorView out) {
  metal::divScalar(input, other, out);
}

void MetalOperators::divTensor(TensorView input, TensorView other, TensorView out) {
  metal::divTensor(input, other, out);
}

void MetalOperators::eq(TensorView input, TensorView other, TensorView out) {
  metal::eq(input, other, out);
}

void MetalOperators::neg(TensorView input, TensorView out) {
  metal::neg(input, out);
}

void MetalOperators::abs(TensorView input, TensorView out) {
  metal::abs(input, out);
}

void MetalOperators::exp(TensorView input, TensorView out) {
  metal::exp(input, out);
}

void MetalOperators::log(TensorView input, TensorView out) {
  metal::log(input, out);
}

void MetalOperators::round(TensorView input, TensorView out) {
  metal::round(input, out);
}

void MetalOperators::sqrt(TensorView input, TensorView out) {
  metal::sqrt(input, out);
}

void MetalOperators::rsqrt(TensorView input, TensorView out) {
  metal::rsqrt(input, out);
}

void MetalOperators::square(TensorView input, TensorView out) {
  metal::square(input, out);
}

void MetalOperators::sigmoid(TensorView input, TensorView out) {
  metal::sigmoid(input, out);
}

void MetalOperators::tanh(TensorView input, TensorView out) {
  metal::tanh(input, out);
}

void MetalOperators::relu(TensorView input, TensorView out) {
  metal::relu(input, out);
}

void MetalOperators::gelu(TensorView input, TensorView out) {
  metal::gelu(input, out);
}

void MetalOperators::silu(TensorView input, TensorView out) {
  metal::silu(input, out);
}

void MetalOperators::quickGelu(TensorView input, TensorView out) {
  metal::quickGelu(input, out);
}

void MetalOperators::sin(TensorView input, TensorView out) {
  metal::sin(input, out);
}

void MetalOperators::cos(TensorView input, TensorView out) {
  metal::cos(input, out);
}

void MetalOperators::sum(TensorView input, int dim, TensorView out) {
  metal::sum(input, dim, out);
}

void MetalOperators::cumsum(TensorView input, int dim, TensorView out) {
  metal::cumsum(input, dim, out);
}

void MetalOperators::max(TensorView input, TensorView out) {
  metal::max(input, out);
}

void MetalOperators::min(TensorView input, TensorView out) {
  metal::min(input, out);
}

bool MetalOperators::all(TensorView input) {
  return metal::all(input);
}

bool MetalOperators::allClose(TensorView A, TensorView B, float rtol, float atol) {
  return metal::allClose(A, B, rtol, atol);
}

void MetalOperators::fill(TensorView input, float value) {
  metal::fill(input, value);
}

void MetalOperators::copy(TensorView src, TensorView dest) {
  metal::copy(src, dest);
}

void MetalOperators::cast(TensorView input, TensorView out) {
  metal::cast(input, out);
}

void MetalOperators::transfer(TensorView src, TensorView dest) {
  metal::transfer(src, dest);
}

void MetalOperators::print(TensorView tensor) {
  metal::print(tensor);
}

float MetalOperators::elem(TensorView tensor) {
  return metal::elem(tensor);
}

bool MetalOperators::elemBool(TensorView tensor) {
  return metal::elemBool(tensor);
}

void MetalOperators::rand(TensorView out) {
  metal::rand(out);
}

void MetalOperators::randNormal(TensorView out) {
  metal::randNormal(out);
}

void MetalOperators::manualSeed(uint64_t seed) {
  metal::manualSeed(seed);
}

DType MetalOperators::getDefaultFloatType() {
  return DType::kFloat16;
}

void MetalOperators::synchronize() {
  mlx::core::synchronize();
}

}  // namespace metal
}  // namespace op
}  // namespace fl
