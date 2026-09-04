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

#pragma once

#include "flint/operators.h"

namespace fl {
namespace op {
namespace metal {

/// @brief Implementation of the Operators interface on the Metal device, through MLX.
///
/// Covers what the SDXL pipeline calls plus the elementwise family, which MLX gives for free.
/// Everything else keeps the NOT_IMPL() body it inherits, so an unimplemented operator says so
/// rather than silently producing something wrong.
class MetalOperators : public Operators {
 public:
  ~MetalOperators() = default;

  /// @brief Whether this build can reach a Metal GPU.
  static bool isAvailable();

  /// @brief Create the operators. Hands MLX the metallib embedded in this binary before it
  ///        builds its Metal device, which has to happen before any other MLX call.
  static std::shared_ptr<Operators> create();

  // implement interface Operators
  Tensor lookup(Tensor table, Tensor indices) override;
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
  Tensor softmax(Tensor input) override;
  Tensor attention(Tensor q, Tensor k, Tensor v, bool causal) override;

  Tensor add(Tensor input, Tensor other) override;
  Tensor sub(Tensor input, Tensor other) override;
  Tensor subFloat(Tensor input, float other) override;
  Tensor mul(Tensor input, Tensor other) override;
  Tensor mul(Tensor input, float other) override;
  Tensor div(Tensor input, float other) override;
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
  void print(Tensor tensor) override;
  float elem(Tensor tensor) override;
  bool elemBool(Tensor tensor) override;

  Tensor rand(lut::Span<const int> shape, DType dtype) override;
  Tensor randNormal(lut::Span<const int> shape) override;
  void manualSeed(uint64_t seed) override;

  DType getDefaultFloatType() override;

 private:
  MetalOperators() = default;
};

}  // namespace metal
}  // namespace op
}  // namespace fl
