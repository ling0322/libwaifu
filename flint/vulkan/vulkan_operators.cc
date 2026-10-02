// The MIT License (MIT)
//
// Copyright (c) 2026 Xiaoyang Chen
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

#include "flint/vulkan/vulkan_operators.h"

#include "lutil/error.h"
#include "lutil/log.h"
#include "flint/cpu/all_close.h"
#include "flint/cpu/print.h"
#include "flint/functional.h"
#include "flint/vulkan/common.h"
#include "flint/vulkan/to_device.h"

namespace fl {
namespace op {
namespace vulkan {

bool VulkanOperators::isAvailable() {
  return Context::isAvailable();
}

std::shared_ptr<Operators> VulkanOperators::create() {
  std::shared_ptr<VulkanOperators> operators(new VulkanOperators());
  operators->_context = Context::get();
  return operators;
}

void VulkanOperators::arangeLong(LongType begin, LongType step, TensorView out) {
  vulkan::arangeLong(begin, step, out);
}

void VulkanOperators::lookup(TensorView table, TensorView indices, TensorView out) {
  vulkan::lookup(table, indices, out);
}

void VulkanOperators::rotaryEmbedding(
    TensorView positions,
    TensorView query,
    TensorView key,
    TensorView rotaryCache) {
  vulkan::rotaryEmbedding(positions, query, key, rotaryCache);
}

void VulkanOperators::rmsNorm(TensorView input, TensorView weight, float eps, TensorView out) {
  vulkan::rmsNorm(input, weight, eps, out);
}

void VulkanOperators::layerNorm(
    TensorView input,
    TensorView weight,
    TensorView bias,
    float eps,
    TensorView out) {
  vulkan::layerNorm(input, weight, bias, eps, out);
}

void VulkanOperators::groupNorm(
    TensorView input,
    TensorView weight,
    TensorView bias,
    int groups,
    float eps,
    TensorView out) {
  vulkan::groupNorm(input, weight, bias, groups, eps, out);
}

void VulkanOperators::upsampleNearest2d(TensorView input, int scale, TensorView out) {
  vulkan::upsampleNearest2d(input, scale, out);
}

void VulkanOperators::upsampleNearest1d(TensorView input, TensorView out) {
  vulkan::upsampleNearest1d(input, out);
}

void VulkanOperators::geglu(TensorView input, TensorView out) {
  vulkan::geglu(input, out);
}

void VulkanOperators::swiglu(TensorView input, TensorView out) {
  vulkan::swiglu(input, out);
}

void VulkanOperators::matmul(TensorView A, TensorView B, TensorView out) {
  vulkan::matmul(A, B, out);
}

void VulkanOperators::conv2d(
    TensorView input,
    TensorView weight,
    TensorView bias,
    int stride,
    int padding,
    int dilation,
    int groups,
    TensorView out) {
  vulkan::conv2d(input, weight, bias, stride, padding, dilation, groups, out);
}

void VulkanOperators::conv1d(
    TensorView input,
    TensorView weight,
    TensorView bias,
    int stride,
    int padding,
    int dilation,
    int groups,
    TensorView out) {
  vulkan::conv1d(input, weight, bias, stride, padding, dilation, groups, out);
}

void VulkanOperators::softmax(TensorView input, TensorView out) {
  vulkan::softmax(input, out);
}

void VulkanOperators::add(TensorView input, TensorView other, TensorView out) {
  binary(BinaryOp::kAdd, input, other, out);
}

void VulkanOperators::sub(TensorView input, TensorView other, TensorView out) {
  binary(BinaryOp::kSub, input, other, out);
}

void VulkanOperators::mul(TensorView input, TensorView other, TensorView out) {
  binary(BinaryOp::kMul, input, other, out);
}

void VulkanOperators::mul(TensorView input, float other, TensorView out) {
  unary(UnaryOp::kMulScalar, input, other, out);
}

void VulkanOperators::div(TensorView input, float other, TensorView out) {
  unary(UnaryOp::kDivScalar, input, other, out);
}

void VulkanOperators::mod(TensorView input, LongType other, TensorView out) {
  vulkan::mod(input, other, out);
}

void VulkanOperators::divTensor(TensorView input, TensorView other, TensorView out) {
  binary(BinaryOp::kDiv, input, other, out);
}

void VulkanOperators::eq(TensorView input, TensorView other, TensorView out) {
  vulkan::eq(input, other, out);
}

void VulkanOperators::neg(TensorView input, TensorView out) {
  unary(UnaryOp::kNeg, input, 0.0f, out);
}

void VulkanOperators::abs(TensorView input, TensorView out) {
  unary(UnaryOp::kAbs, input, 0.0f, out);
}

void VulkanOperators::exp(TensorView input, TensorView out) {
  unary(UnaryOp::kExp, input, 0.0f, out);
}

void VulkanOperators::log(TensorView input, TensorView out) {
  unary(UnaryOp::kLog, input, 0.0f, out);
}

void VulkanOperators::round(TensorView input, TensorView out) {
  unary(UnaryOp::kRound, input, 0.0f, out);
}

void VulkanOperators::sqrt(TensorView input, TensorView out) {
  unary(UnaryOp::kSqrt, input, 0.0f, out);
}

void VulkanOperators::rsqrt(TensorView input, TensorView out) {
  unary(UnaryOp::kRsqrt, input, 0.0f, out);
}

void VulkanOperators::square(TensorView input, TensorView out) {
  unary(UnaryOp::kSquare, input, 0.0f, out);
}

void VulkanOperators::sigmoid(TensorView input, TensorView out) {
  unary(UnaryOp::kSigmoid, input, 0.0f, out);
}

void VulkanOperators::tanh(TensorView input, TensorView out) {
  unary(UnaryOp::kTanh, input, 0.0f, out);
}

void VulkanOperators::relu(TensorView input, TensorView out) {
  unary(UnaryOp::kRelu, input, 0.0f, out);
}

void VulkanOperators::gelu(TensorView input, TensorView out) {
  unary(UnaryOp::kGelu, input, 0.0f, out);
}

void VulkanOperators::silu(TensorView input, TensorView out) {
  unary(UnaryOp::kSilu, input, 0.0f, out);
}

void VulkanOperators::quickGelu(TensorView input, TensorView out) {
  unary(UnaryOp::kQuickGelu, input, 0.0f, out);
}

void VulkanOperators::sin(TensorView input, TensorView out) {
  unary(UnaryOp::kSin, input, 0.0f, out);
}

void VulkanOperators::cos(TensorView input, TensorView out) {
  unary(UnaryOp::kCos, input, 0.0f, out);
}

void VulkanOperators::sum(TensorView input, int dim, TensorView out) {
  vulkan::sum(input, dim, out);
}

void VulkanOperators::cumsum(TensorView input, int dim, TensorView out) {
  vulkan::cumsum(input, dim, out);
}

void VulkanOperators::max(TensorView input, TensorView out) {
  reduceLastDim(input, ReduceOp::kMax, out);
}

void VulkanOperators::min(TensorView input, TensorView out) {
  reduceLastDim(input, ReduceOp::kMin, out);
}

bool VulkanOperators::all(TensorView input) {
  return vulkan::all(input);
}

bool VulkanOperators::allClose(TensorView A, TensorView B, float rtol, float atol) {
  // A comparison for tests, so it is the CPU's, on copies brought over as float.
  auto toHostFloat = [](const TensorView &x) {
    if (x.getDType() == DType::kFloat) return toCpu(x);
    Tensor wide = createTensor(x.getShape(), DType::kFloat);
    vulkan::copy(x, wide);
    return toCpu(wide);
  };
  Tensor a = toHostFloat(A);
  Tensor b = toHostFloat(B);
  return cpu::allClose(a, b, rtol, atol);
}

void VulkanOperators::fill(TensorView input, float value) {
  vulkan::fill(input, value);
}

void VulkanOperators::copy(TensorView src, TensorView dest) {
  vulkan::copy(src, dest);
}

void VulkanOperators::cast(TensorView input, TensorView out) {
  vulkan::cast(input, out);
}

void VulkanOperators::transfer(TensorView src, TensorView dest) {
  vulkan::transfer(src, dest);
}

void VulkanOperators::causalMask(TensorView out) {
  vulkan::causalMask(out);
}

void VulkanOperators::print(TensorView tensor) {
  Tensor host = toCpu(tensor);
  cpu::print(host);
}

float VulkanOperators::elem(TensorView tensor) {
  return vulkan::elem(tensor);
}

bool VulkanOperators::elemBool(TensorView tensor) {
  return vulkan::elemBool(tensor);
}

void VulkanOperators::rand(TensorView out) {
  // Drawn in float32, as on CUDA, and converted when `out` is of another type.
  if (out.getDType() == DType::kFloat) {
    _rand.uniform(out);
    return;
  }

  Tensor wide = createTensor(out.getShape(), DType::kFloat);
  _rand.uniform(wide);
  vulkan::cast(wide, out);
}

void VulkanOperators::randNormal(TensorView out) {
  _rand.normal(out);
}

void VulkanOperators::manualSeed(uint64_t seed) {
  _rand.setSeed(seed);
}

MemorySnapshot VulkanOperators::captureMemorySnapshot() {
  return _context->captureMemorySnapshot();
}

void VulkanOperators::resetPeakMemoryStats() {
  _context->resetPeakMemoryStats();
}

void VulkanOperators::releaseUnusedMemory() {
  _context->releaseUnusedMemory();
}

DType VulkanOperators::getDefaultFloatType() {
  return DType::kFloat16;
}

void VulkanOperators::synchronize() {
  _context->synchronize();
}

}  // namespace vulkan
}  // namespace op
}  // namespace fl
