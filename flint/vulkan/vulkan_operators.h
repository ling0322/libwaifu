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

#pragma once

#include <memory>

#include "flint/operators.h"
#include "flint/vulkan/context.h"
#include "flint/vulkan/ops.h"

namespace fl {
namespace op {
namespace vulkan {

/// Implementation of the Operators interface on a Vulkan device, with kernels of its own written
/// in GLSL.
///
/// Covers what the image models run on: the elementwise family, matmul, convolution, the norms,
/// softmax and attention -- the last composed by the base class out of the others. What is left
/// keeps the body it inherits, so an unimplemented operator says so rather than producing
/// something wrong.
class VulkanOperators : public Operators {
 public:
  /// @brief Whether the Vulkan loader can be opened and there is a device to run on.
  static bool isAvailable();

  /// @brief Create the operators, opening the device.
  static std::shared_ptr<Operators> create();

  // implement interface Operators
  void arangeLong(LongType begin, LongType step, TensorView out) override;
  void lookup(TensorView table, TensorView indices, TensorView out) override;
  void rotaryEmbedding(
      TensorView positions,
      TensorView query,
      TensorView key,
      TensorView rotaryCache) override;
  void rmsNorm(TensorView input, TensorView weight, float eps, TensorView out) override;
  void layerNorm(
      TensorView input,
      TensorView weight,
      TensorView bias,
      float eps,
      TensorView out) override;
  void groupNorm(
      TensorView input,
      TensorView weight,
      TensorView bias,
      int groups,
      float eps,
      TensorView out) override;
  void upsampleNearest2d(TensorView input, int scale, TensorView out) override;
  void upsampleNearest1d(TensorView input, TensorView out) override;
  void geglu(TensorView input, TensorView out) override;
  void swiglu(TensorView input, TensorView out) override;
  void matmul(TensorView A, TensorView B, TensorView out) override;
  void conv2d(
      TensorView input,
      TensorView weight,
      TensorView bias,
      int stride,
      int padding,
      int dilation,
      int groups,
      TensorView out) override;
  void conv1d(
      TensorView input,
      TensorView weight,
      TensorView bias,
      int stride,
      int padding,
      int dilation,
      int groups,
      TensorView out) override;
  void softmax(TensorView input, TensorView out) override;

  void add(TensorView input, TensorView other, TensorView out) override;
  void sub(TensorView input, TensorView other, TensorView out) override;
  void mul(TensorView input, TensorView other, TensorView out) override;
  void mul(TensorView input, float other, TensorView out) override;
  void div(TensorView input, float other, TensorView out) override;
  void mod(TensorView input, LongType other, TensorView out) override;
  void divTensor(TensorView input, TensorView other, TensorView out) override;
  void eq(TensorView input, TensorView other, TensorView out) override;

  void neg(TensorView input, TensorView out) override;
  void abs(TensorView input, TensorView out) override;
  void exp(TensorView input, TensorView out) override;
  void log(TensorView input, TensorView out) override;
  void round(TensorView input, TensorView out) override;
  void sqrt(TensorView input, TensorView out) override;
  void rsqrt(TensorView input, TensorView out) override;
  void square(TensorView input, TensorView out) override;
  void sigmoid(TensorView input, TensorView out) override;
  void tanh(TensorView input, TensorView out) override;
  void relu(TensorView input, TensorView out) override;
  void gelu(TensorView input, TensorView out) override;
  void silu(TensorView input, TensorView out) override;
  void quickGelu(TensorView input, TensorView out) override;
  void sin(TensorView input, TensorView out) override;
  void cos(TensorView input, TensorView out) override;

  void sum(TensorView input, int dim, TensorView out) override;
  void cumsum(TensorView input, int dim, TensorView out) override;
  void max(TensorView input, TensorView out) override;
  void min(TensorView input, TensorView out) override;
  bool all(TensorView input) override;
  bool allClose(TensorView A, TensorView B, float rtol, float atol) override;

  void fill(TensorView input, float value) override;
  void copy(TensorView src, TensorView dest) override;
  void cast(TensorView input, TensorView out) override;
  void transfer(TensorView src, TensorView dest) override;
  void causalMask(TensorView out) override;
  void print(TensorView tensor) override;
  float elem(TensorView tensor) override;
  bool elemBool(TensorView tensor) override;

  void rand(TensorView out) override;
  void randNormal(TensorView out) override;
  void manualSeed(uint64_t seed) override;

  MemorySnapshot captureMemorySnapshot() override;
  void resetPeakMemoryStats() override;
  void releaseUnusedMemory() override;

  DType getDefaultFloatType() override;
  void synchronize() override;

 private:
  std::shared_ptr<Context> _context;
  Rand _rand;

  VulkanOperators() = default;
};

}  // namespace vulkan
}  // namespace op
}  // namespace fl
