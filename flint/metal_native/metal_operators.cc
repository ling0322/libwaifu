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

#include "flint/metal_native/metal_operators.h"

#include "lutil/error.h"
#include "lutil/log.h"
#include "flint/cpu/all_close.h"
#include "flint/cpu/print.h"
#include "flint/functional.h"
#include "flint/metal_native/common.h"
#include "flint/metal_native/to_device.h"

namespace fl {
namespace op {
namespace metal_native {

bool MetalOperators::isAvailable() {
  return Context::isAvailable();
}

std::shared_ptr<Operators> MetalOperators::create() {
  std::shared_ptr<MetalOperators> operators(new MetalOperators());
  operators->_context = Context::get();
  return operators;
}

void MetalOperators::arangeLong(LongType begin, LongType step, TensorView out) {
  metal_native::arangeLong(begin, step, out);
}

void MetalOperators::lookup(TensorView table, TensorView indices, TensorView out) {
  metal_native::lookup(table, indices, out);
}

void MetalOperators::rotaryEmbedding(
    TensorView positions,
    TensorView query,
    TensorView key,
    TensorView rotaryCache) {
  metal_native::rotaryEmbedding(positions, query, key, rotaryCache);
}

void MetalOperators::rmsNorm(TensorView input, TensorView weight, float eps, TensorView out) {
  metal_native::rmsNorm(input, weight, eps, out);
}

void MetalOperators::layerNorm(
    TensorView input,
    TensorView weight,
    TensorView bias,
    float eps,
    TensorView out) {
  metal_native::layerNorm(input, weight, bias, eps, out);
}

void MetalOperators::groupNorm(
    TensorView input,
    TensorView weight,
    TensorView bias,
    int groups,
    float eps,
    TensorView out) {
  metal_native::groupNorm(input, weight, bias, groups, eps, out);
}

void MetalOperators::upsampleNearest2d(TensorView input, int scale, TensorView out) {
  metal_native::upsampleNearest2d(input, scale, out);
}

void MetalOperators::upsampleNearest1d(TensorView input, TensorView out) {
  metal_native::upsampleNearest1d(input, out);
}

void MetalOperators::geglu(TensorView input, TensorView out) {
  metal_native::geglu(input, out);
}

void MetalOperators::swiglu(TensorView input, TensorView out) {
  metal_native::swiglu(input, out);
}

void MetalOperators::matmul(TensorView A, TensorView B, TensorView out) {
  metal_native::matmul(A, B, out);
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
  metal_native::conv2d(input, weight, bias, stride, padding, dilation, groups, out);
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
  metal_native::conv1d(input, weight, bias, stride, padding, dilation, groups, out);
}

void MetalOperators::convTranspose1d(
    TensorView input,
    TensorView weight,
    TensorView bias,
    int stride,
    int padding,
    int outputPadding,
    int groups,
    TensorView out) {
  F::checkConvTranspose1d(input, weight, bias, stride, padding, outputPadding, groups, out);
  int groupOut = weight.getShape(1);

  // A matmul per group into columns, then the kernel gathering them into the output.
  for (int g = 0; g < groups; ++g) {
    Tensor columns = F::transposedConvolutionColumns(this, input, weight, groups, g);
    metal_native::overlapAdd1d(
        columns,
        bias.empty() ? bias : bias.slice(0, {g * groupOut, (g + 1) * groupOut}),
        input.getShape(2),
        stride,
        padding,
        weight.getShape(2),
        out.slice(1, {g * groupOut, (g + 1) * groupOut}));
  }
}

void MetalOperators::snake(
    TensorView input,
    TensorView alpha,
    TensorView beta,
    float eps,
    TensorView out) {
  metal_native::snake(input, alpha, beta, eps, out);
}

void MetalOperators::pad1d(TensorView input, int left, int right, PadMode mode, TensorView out) {
  // The kernel is built for the two float types; anything else is padded by composition.
  DType dtype = input.getDType();
  if (dtype != DType::kFloat && dtype != DType::kFloat16) {
    return Operators::pad1d(input, left, right, mode, out);
  }
  metal_native::pad1d(input, left, right, mode, out);
}

void MetalOperators::clamp(TensorView input, float lo, float hi, TensorView out) {
  metal_native::clamp(input, lo, hi, out);
}

void MetalOperators::attention(
    TensorView q,
    TensorView k,
    TensorView v,
    bool causal,
    TensorView out) {
  // The kernel where it takes these, and the composition where it does not.
  if (metal_native::attention(q, k, v, causal, out)) return;
  Operators::attention(q, k, v, causal, out);
}

void MetalOperators::softmax(TensorView input, TensorView out) {
  metal_native::softmax(input, out);
}

void MetalOperators::add(TensorView input, TensorView other, TensorView out) {
  binary(BinaryOp::kAdd, input, other, out);
}

void MetalOperators::sub(TensorView input, TensorView other, TensorView out) {
  binary(BinaryOp::kSub, input, other, out);
}

void MetalOperators::mul(TensorView input, TensorView other, TensorView out) {
  binary(BinaryOp::kMul, input, other, out);
}

void MetalOperators::mul(TensorView input, float other, TensorView out) {
  unary(UnaryOp::kMulScalar, input, other, out);
}

void MetalOperators::div(TensorView input, float other, TensorView out) {
  unary(UnaryOp::kDivScalar, input, other, out);
}

void MetalOperators::mod(TensorView input, LongType other, TensorView out) {
  metal_native::mod(input, other, out);
}

void MetalOperators::divTensor(TensorView input, TensorView other, TensorView out) {
  binary(BinaryOp::kDiv, input, other, out);
}

void MetalOperators::eq(TensorView input, TensorView other, TensorView out) {
  metal_native::eq(input, other, out);
}

void MetalOperators::neg(TensorView input, TensorView out) {
  unary(UnaryOp::kNeg, input, 0.0f, out);
}

void MetalOperators::abs(TensorView input, TensorView out) {
  unary(UnaryOp::kAbs, input, 0.0f, out);
}

void MetalOperators::exp(TensorView input, TensorView out) {
  unary(UnaryOp::kExp, input, 0.0f, out);
}

void MetalOperators::log(TensorView input, TensorView out) {
  unary(UnaryOp::kLog, input, 0.0f, out);
}

void MetalOperators::round(TensorView input, TensorView out) {
  unary(UnaryOp::kRound, input, 0.0f, out);
}

void MetalOperators::sqrt(TensorView input, TensorView out) {
  unary(UnaryOp::kSqrt, input, 0.0f, out);
}

void MetalOperators::rsqrt(TensorView input, TensorView out) {
  unary(UnaryOp::kRsqrt, input, 0.0f, out);
}

void MetalOperators::square(TensorView input, TensorView out) {
  unary(UnaryOp::kSquare, input, 0.0f, out);
}

void MetalOperators::sigmoid(TensorView input, TensorView out) {
  unary(UnaryOp::kSigmoid, input, 0.0f, out);
}

void MetalOperators::tanh(TensorView input, TensorView out) {
  unary(UnaryOp::kTanh, input, 0.0f, out);
}

void MetalOperators::relu(TensorView input, TensorView out) {
  unary(UnaryOp::kRelu, input, 0.0f, out);
}

void MetalOperators::gelu(TensorView input, TensorView out) {
  unary(UnaryOp::kGelu, input, 0.0f, out);
}

void MetalOperators::silu(TensorView input, TensorView out) {
  unary(UnaryOp::kSilu, input, 0.0f, out);
}

void MetalOperators::quickGelu(TensorView input, TensorView out) {
  unary(UnaryOp::kQuickGelu, input, 0.0f, out);
}

void MetalOperators::sin(TensorView input, TensorView out) {
  unary(UnaryOp::kSin, input, 0.0f, out);
}

void MetalOperators::cos(TensorView input, TensorView out) {
  unary(UnaryOp::kCos, input, 0.0f, out);
}

void MetalOperators::sum(TensorView input, int dim, TensorView out) {
  metal_native::sum(input, dim, out);
}

void MetalOperators::cumsum(TensorView input, int dim, TensorView out) {
  metal_native::cumsum(input, dim, out);
}

void MetalOperators::max(TensorView input, TensorView out) {
  reduceLastDim(input, ReduceOp::kMax, out);
}

void MetalOperators::min(TensorView input, TensorView out) {
  reduceLastDim(input, ReduceOp::kMin, out);
}

bool MetalOperators::all(TensorView input) {
  return metal_native::all(input);
}

bool MetalOperators::allClose(TensorView A, TensorView B, float rtol, float atol) {
  // A comparison for tests, so it is the CPU's, on copies brought over as float.
  auto toHostFloat = [](const TensorView &x) {
    if (x.getDType() == DType::kFloat) return toCpu(x);
    Tensor wide = createTensor(x.getShape(), DType::kFloat);
    metal_native::copy(x, wide);
    return toCpu(wide);
  };
  Tensor a = toHostFloat(A);
  Tensor b = toHostFloat(B);
  return cpu::allClose(a, b, rtol, atol);
}

void MetalOperators::fill(TensorView input, float value) {
  metal_native::fill(input, value);
}

void MetalOperators::copy(TensorView src, TensorView dest) {
  metal_native::copy(src, dest);
}

void MetalOperators::cast(TensorView input, TensorView out) {
  metal_native::cast(input, out);
}

void MetalOperators::transfer(TensorView src, TensorView dest) {
  metal_native::transfer(src, dest);
}

void MetalOperators::causalMask(TensorView out) {
  metal_native::causalMask(out);
}

void MetalOperators::print(TensorView tensor) {
  Tensor host = toCpu(tensor);
  cpu::print(host);
}

float MetalOperators::elem(TensorView tensor) {
  return metal_native::elem(tensor);
}

bool MetalOperators::elemBool(TensorView tensor) {
  return metal_native::elemBool(tensor);
}

void MetalOperators::rand(TensorView out) {
  // Drawn in float32, as on CUDA, and converted when `out` is of another type.
  if (out.getDType() == DType::kFloat) {
    _rand.uniform(out);
    return;
  }

  Tensor wide = createTensor(out.getShape(), DType::kFloat);
  _rand.uniform(wide);
  metal_native::cast(wide, out);
}

void MetalOperators::randNormal(TensorView out) {
  _rand.normal(out);
}

void MetalOperators::manualSeed(uint64_t seed) {
  _rand.setSeed(seed);
}

MemorySnapshot MetalOperators::captureMemorySnapshot() {
  return _context->captureMemorySnapshot();
}

void MetalOperators::resetPeakMemoryStats() {
  _context->resetPeakMemoryStats();
}

void MetalOperators::releaseUnusedMemory() {
  _context->releaseUnusedMemory();
}

void MetalOperators::setMemoryLimit(int64_t bytes) {
  _context->setMemoryLimit(bytes);
}

DType MetalOperators::getDefaultFloatType() {
  return DType::kFloat16;
}

void MetalOperators::synchronize() {
  _context->synchronize();
}

}  // namespace metal_native
}  // namespace op
}  // namespace fl
