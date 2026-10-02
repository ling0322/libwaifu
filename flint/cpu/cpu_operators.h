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

#include <stdint.h>

#include <memory>

#include "lutil/random.h"
#include "flint/operators.h"
#include "flint/tensor.h"

namespace fl {
namespace op {
namespace cpu {

constexpr float Pi = 3.14159f;

class CPUOperators : public Operators {
 public:
  CPUOperators();

  // create a instance of CPUOperators
  static std::unique_ptr<Operators> create();
  static std::unique_ptr<Operators> createFp32Only();

  // implement interface Operators
  void add(TensorView a, TensorView b, TensorView out) override;
  void sub(TensorView a, TensorView b, TensorView out) override;
  bool allClose(TensorView A, TensorView B, float rtol, float atol) override;
  void cast(TensorView input, TensorView out) override;
  void causalMask(TensorView out) override;
  void copy(TensorView src, TensorView dest) override;
  void fill(TensorView input, float value) override;
  void lookup(TensorView table, TensorView indices, TensorView out) override;
  void matmul(TensorView a, TensorView b, TensorView out) override;
  void gatedDeltaNetPrefill(
      TensorView q,
      TensorView k,
      TensorView v,
      TensorView g,
      TensorView beta,
      TensorView cuSeqlens,
      TensorView stateSlots,
      TensorView state,
      TensorView out) override;
  void max(TensorView input, TensorView out) override;
  void min(TensorView input, TensorView out) override;
  void square(TensorView input, TensorView out) override;
  void divTensor(TensorView input, TensorView other, TensorView out) override;
  void neg(TensorView input, TensorView out) override;
  void abs(TensorView input, TensorView out) override;
  void exp(TensorView input, TensorView out) override;
  void log(TensorView input, TensorView out) override;
  void round(TensorView input, TensorView out) override;
  void sqrt(TensorView input, TensorView out) override;
  void rsqrt(TensorView input, TensorView out) override;
  void sigmoid(TensorView input, TensorView out) override;
  void tanh(TensorView input, TensorView out) override;
  void relu(TensorView input, TensorView out) override;
  void gelu(TensorView input, TensorView out) override;
  void silu(TensorView input, TensorView out) override;
  void sin(TensorView input, TensorView out) override;
  void cos(TensorView input, TensorView out) override;
  void quickGelu(TensorView input, TensorView out) override;
  void arangeLong(LongType begin, LongType step, TensorView out) override;
  void randNormal(TensorView out) override;
  void div(TensorView input, float other, TensorView out) override;
  float elem(TensorView tensor) override;
  bool elemBool(TensorView tensor) override;
  void mod(TensorView input, LongType other, TensorView out) override;
  void eq(TensorView input, TensorView other, TensorView out) override;
  bool all(TensorView A) override;
  void mul(TensorView input, float other, TensorView out) override;
  void mul(TensorView input, TensorView other, TensorView out) override;
  void print(TensorView tensor) override;
  void rand(TensorView out) override;
  void repetitionPenalty(TensorView logits, TensorView history, float weight) override;
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
  void sample(
      TensorView logits,
      TensorView temperatures,
      TensorView topKs,
      TensorView topPs,
      TensorView out) override;
  void softmax(TensorView input, TensorView out) override;
  void sum(TensorView input, int dim, TensorView out) override;
  void cumsum(TensorView input, int dim, TensorView out) override;
  void swiglu(TensorView A, TensorView out) override;
  void geglu(TensorView input, TensorView out) override;
  void manualSeed(uint64_t seed) override;

  DType getDefaultFloatType() override;

  MemorySnapshot captureMemorySnapshot() override;
  void resetPeakMemoryStats() override;

 private:
  typedef TensorShape::Elem Shape;
  lut::Random _rand;
};

}  // namespace cpu
}  // namespace op
}  // namespace fl
