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

Tensor VulkanOperators::arangeLong(LongType begin, LongType end, LongType step) {
  return vulkan::arangeLong(begin, end, step);
}

Tensor VulkanOperators::lookup(Tensor table, Tensor indices) {
  return vulkan::lookup(table, indices);
}

void VulkanOperators::rotaryEmbedding(
    Tensor positions,
    Tensor query,
    Tensor key,
    Tensor rotaryCache) {
  vulkan::rotaryEmbedding(positions, query, key, rotaryCache);
}

Tensor VulkanOperators::rmsNorm(Tensor input, Tensor weight, float eps) {
  return vulkan::rmsNorm(input, weight, eps);
}

Tensor VulkanOperators::layerNorm(Tensor input, Tensor weight, Tensor bias, float eps) {
  return vulkan::layerNorm(input, weight, bias, eps);
}

Tensor VulkanOperators::groupNorm(Tensor input, Tensor weight, Tensor bias, int groups, float eps) {
  return vulkan::groupNorm(input, weight, bias, groups, eps);
}

Tensor VulkanOperators::upsampleNearest2d(Tensor input, int scale) {
  return vulkan::upsampleNearest2d(input, scale);
}

Tensor VulkanOperators::geglu(Tensor input) {
  return vulkan::geglu(input);
}

Tensor VulkanOperators::swiglu(Tensor input) {
  return vulkan::swiglu(input);
}

Tensor VulkanOperators::matmul(Tensor A, Tensor B) {
  return vulkan::matmul(A, B);
}

Tensor VulkanOperators::conv2d(
    Tensor input,
    Tensor weight,
    Tensor bias,
    int stride,
    int padding,
    int dilation,
    int groups) {
  return vulkan::conv2d(input, weight, bias, stride, padding, dilation, groups);
}

Tensor VulkanOperators::conv1d(
    Tensor input,
    Tensor weight,
    Tensor bias,
    int stride,
    int padding,
    int dilation,
    int groups) {
  return vulkan::conv1d(input, weight, bias, stride, padding, dilation, groups);
}

Tensor VulkanOperators::softmax(Tensor input) {
  return vulkan::softmax(input);
}

Tensor VulkanOperators::add(Tensor input, Tensor other) {
  return binary(BinaryOp::kAdd, input, other);
}

Tensor VulkanOperators::sub(Tensor input, Tensor other) {
  return binary(BinaryOp::kSub, input, other);
}

Tensor VulkanOperators::subFloat(Tensor input, float other) {
  return unary(UnaryOp::kAddScalar, input, -other);
}

Tensor VulkanOperators::mul(Tensor input, Tensor other) {
  return binary(BinaryOp::kMul, input, other);
}

Tensor VulkanOperators::mul(Tensor input, float other) {
  return unary(UnaryOp::kMulScalar, input, other);
}

Tensor VulkanOperators::div(Tensor input, float other) {
  return unary(UnaryOp::kDivScalar, input, other);
}

Tensor VulkanOperators::mod(Tensor input, LongType other) {
  return vulkan::mod(input, other);
}

Tensor VulkanOperators::divTensor(Tensor input, Tensor other) {
  return binary(BinaryOp::kDiv, input, other);
}

Tensor VulkanOperators::eq(Tensor input, Tensor other) {
  return vulkan::eq(input, other);
}

Tensor VulkanOperators::neg(Tensor input) {
  return unary(UnaryOp::kNeg, input);
}

Tensor VulkanOperators::abs(Tensor input) {
  return unary(UnaryOp::kAbs, input);
}

Tensor VulkanOperators::exp(Tensor input) {
  return unary(UnaryOp::kExp, input);
}

Tensor VulkanOperators::log(Tensor input) {
  return unary(UnaryOp::kLog, input);
}

Tensor VulkanOperators::round(Tensor input) {
  return unary(UnaryOp::kRound, input);
}

Tensor VulkanOperators::sqrt(Tensor input) {
  return unary(UnaryOp::kSqrt, input);
}

Tensor VulkanOperators::rsqrt(Tensor input) {
  return unary(UnaryOp::kRsqrt, input);
}

Tensor VulkanOperators::square(Tensor input) {
  return unary(UnaryOp::kSquare, input);
}

Tensor VulkanOperators::sigmoid(Tensor input) {
  return unary(UnaryOp::kSigmoid, input);
}

Tensor VulkanOperators::tanh(Tensor input) {
  return unary(UnaryOp::kTanh, input);
}

Tensor VulkanOperators::relu(Tensor input) {
  return unary(UnaryOp::kRelu, input);
}

Tensor VulkanOperators::gelu(Tensor input) {
  return unary(UnaryOp::kGelu, input);
}

Tensor VulkanOperators::silu(Tensor input) {
  return unary(UnaryOp::kSilu, input);
}

Tensor VulkanOperators::quickGelu(Tensor input) {
  return unary(UnaryOp::kQuickGelu, input);
}

Tensor VulkanOperators::sin(Tensor input) {
  return unary(UnaryOp::kSin, input);
}

Tensor VulkanOperators::cos(Tensor input) {
  return unary(UnaryOp::kCos, input);
}

Tensor VulkanOperators::sum(Tensor input, int dim) {
  return vulkan::sum(input, dim);
}

Tensor VulkanOperators::max(Tensor input) {
  return reduceLastDim(input, ReduceOp::kMax);
}

Tensor VulkanOperators::min(Tensor input) {
  return reduceLastDim(input, ReduceOp::kMin);
}

bool VulkanOperators::all(Tensor input) {
  return vulkan::all(input);
}

bool VulkanOperators::allClose(Tensor A, Tensor B, float rtol, float atol) {
  // A comparison for tests, so it is the CPU's, on copies brought over as float.
  auto toHostFloat = [](const Tensor &x) {
    return toCpu(x.getDType() == DType::kFloat ? x : vulkan::cast(x, DType::kFloat));
  };
  return cpu::allClose(toHostFloat(A), toHostFloat(B), rtol, atol);
}

Tensor VulkanOperators::tensor(lut::Span<const int> shape, DType dtype) {
  return createTensor(shape, dtype);
}

Tensor VulkanOperators::tensorLike(Tensor input) {
  return createTensor(input.getShape(), input.getDType());
}

Tensor VulkanOperators::zeros(lut::Span<const int> shape, DType dtype) {
  Tensor output = createTensor(shape, dtype);

  // Zero is zero bits in every type there is, so the buffer is cleared rather than filled. The
  // fill is in words, and every buffer is a whole number of them.
  int64_t bytes = dtype.getTotalSize(output.getNumEl());
  bytes = (bytes + 3) / 4 * 4;
  if (bytes > 0) _context->fillBuffer(getBuffer(output), getByteOffset(output), bytes, 0);
  return output;
}

void VulkanOperators::fill(Tensor input, float value) {
  vulkan::fill(input, value);
}

void VulkanOperators::copy(Tensor src, Tensor dest) {
  vulkan::copy(src, dest);
}

Tensor VulkanOperators::cast(Tensor tensor, DType dtype) {
  return vulkan::cast(tensor, dtype);
}

Tensor VulkanOperators::toDevice(Device device, Tensor tensor) {
  return vulkan::toDevice(device, tensor);
}

Tensor VulkanOperators::causalMask(int maxLen) {
  return vulkan::causalMask(maxLen, getDefaultFloatType());
}

void VulkanOperators::print(Tensor tensor) {
  cpu::print(toCpu(tensor));
}

float VulkanOperators::elem(Tensor tensor) {
  return vulkan::elem(tensor);
}

bool VulkanOperators::elemBool(Tensor tensor) {
  return vulkan::elemBool(tensor);
}

Tensor VulkanOperators::rand(lut::Span<const int> shape, DType dtype) {
  return vulkan::cast(_rand.uniform(shape), dtype);
}

Tensor VulkanOperators::randNormal(lut::Span<const int> shape) {
  return _rand.normal(shape);
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
