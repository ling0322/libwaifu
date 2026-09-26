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
  Tensor arangeLong(LongType begin, LongType end, LongType step) override;
  Tensor lookup(Tensor table, Tensor indices) override;
  void rotaryEmbedding(Tensor positions, Tensor query, Tensor key, Tensor rotaryCache) override;
  Tensor rmsNorm(Tensor input, Tensor weight, float eps) override;
  Tensor layerNorm(Tensor input, Tensor weight, Tensor bias, float eps) override;
  Tensor groupNorm(Tensor input, Tensor weight, Tensor bias, int groups, float eps) override;
  Tensor upsampleNearest2d(Tensor input, int scale) override;
  Tensor geglu(Tensor input) override;
  Tensor swiglu(Tensor input) override;
  Tensor matmul(Tensor A, Tensor B) override;
  Tensor conv2d(
      Tensor input,
      Tensor weight,
      Tensor bias,
      int stride,
      int padding,
      int dilation,
      int groups) override;
  Tensor conv1d(
      Tensor input,
      Tensor weight,
      Tensor bias,
      int stride,
      int padding,
      int dilation,
      int groups) override;
  Tensor softmax(Tensor input) override;

  Tensor add(Tensor input, Tensor other) override;
  Tensor sub(Tensor input, Tensor other) override;
  Tensor subFloat(Tensor input, float other) override;
  Tensor mul(Tensor input, Tensor other) override;
  Tensor mul(Tensor input, float other) override;
  Tensor div(Tensor input, float other) override;
  Tensor mod(Tensor input, LongType other) override;
  Tensor divTensor(Tensor input, Tensor other) override;
  Tensor eq(Tensor input, Tensor other) override;

  Tensor neg(Tensor input) override;
  Tensor abs(Tensor input) override;
  Tensor exp(Tensor input) override;
  Tensor sqrt(Tensor input) override;
  Tensor rsqrt(Tensor input) override;
  Tensor square(Tensor input) override;
  Tensor sigmoid(Tensor input) override;
  Tensor tanh(Tensor input) override;
  Tensor relu(Tensor input) override;
  Tensor gelu(Tensor input) override;
  Tensor silu(Tensor input) override;
  Tensor quickGelu(Tensor input) override;
  Tensor sin(Tensor input) override;
  Tensor cos(Tensor input) override;

  Tensor sum(Tensor input, int dim) override;
  Tensor max(Tensor input) override;
  Tensor min(Tensor input) override;
  bool all(Tensor input) override;
  bool allClose(Tensor A, Tensor B, float rtol, float atol) override;

  Tensor tensor(lut::Span<const int> shape, DType dtype) override;
  Tensor tensorLike(Tensor input) override;
  Tensor zeros(lut::Span<const int> shape, DType dtype) override;
  void fill(Tensor input, float value) override;
  void copy(Tensor src, Tensor dest) override;
  Tensor cast(Tensor tensor, DType dtype) override;
  Tensor toDevice(Device device, Tensor tensor) override;
  Tensor causalMask(int maxLen) override;
  void print(Tensor tensor) override;
  float elem(Tensor tensor) override;
  bool elemBool(Tensor tensor) override;

  Tensor rand(lut::Span<const int> shape, DType dtype) override;
  Tensor randNormal(lut::Span<const int> shape) override;
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
