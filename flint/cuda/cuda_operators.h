// The MIT License (MIT)
//
// Copyright (c) 2023-2025 Xiaoyang Chen
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
namespace cuda {

class MatMul;
class Rand;

/// @brief Implementation of Operator interface with cuda device.
class CudaOperators : public Operators {
 public:
  ~CudaOperators() = default;

  /// Pin the GEMM backend instead of letting `MatMul::create` choose. Nothing in the library
  /// passes these -- a plain run gets CUTLASS, or cuBLAS where FLINT_ENABLE_CUBLAS asks for it --
  /// and they exist for the benchmark, which builds both in the one process to put them in
  /// columns beside each other. Asking by name is what makes that comparison mean anything.
  static constexpr int OPT_CUTLASS_GEMM = 0x00000001;
  static constexpr int OPT_CUBLAS_GEMM = 0x00000002;

  /// @brief Returns true if the CudaOperators is available (CUDA device available in host).
  /// @return if CudaOperators available.
  static bool isAvailable();

  // create a instance of CudaOperators
  static std::shared_ptr<Operators> create(int options = 0);

  // implement interface Operators
  void arangeLong(LongType begin, LongType step, TensorView out) override;
  void cast(TensorView input, TensorView out) override;
  void add(TensorView input, TensorView other, TensorView out) override;
  void sub(TensorView input, TensorView other, TensorView out) override;
  void causalMask(TensorView out) override;
  void copy(TensorView src, TensorView dest) override;
  void transfer(TensorView src, TensorView dest) override;
  void fill(TensorView input, float value) override;
  void square(TensorView input, TensorView out) override;
  void lookup(TensorView table, TensorView indices, TensorView out) override;
  void rotaryEmbedding(
      TensorView positions,
      TensorView query,
      TensorView key,
      TensorView rotaryCache) override;
  void matmul(TensorView A, TensorView B, TensorView out) override;
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
  bool allClose(TensorView A, TensorView B, float rtol, float atol) override;
  void mul(TensorView input, float other, TensorView out) override;
  bool all(TensorView A) override;
  void div(TensorView input, float other, TensorView out) override;
  void mod(TensorView input, LongType other, TensorView out) override;
  void mul(TensorView input, TensorView other, TensorView out) override;
  void print(TensorView tensor) override;
  void repetitionPenalty(TensorView logits, TensorView history, float weight) override;
  void rmsNorm(TensorView input, TensorView weight, float eps, TensorView out) override;
  void sample(
      TensorView logits,
      TensorView temperatures,
      TensorView topKs,
      TensorView topPs,
      TensorView out) override;
  void softmax(TensorView input, TensorView out) override;
  void attention(TensorView q, TensorView k, TensorView v, bool causal, TensorView out) override;
  void pagedAttention(
      TensorView q,
      TensorView keyCache,
      TensorView valueCache,
      TensorView blockTable,
      TensorView cuSeqlensQ,
      TensorView seqlensK,
      int maxQLen,
      int maxKLen,
      bool causal,
      TensorView out) override;
  void storeKVCache(
      TensorView k,
      TensorView v,
      TensorView keyCache,
      TensorView valueCache,
      TensorView slotMapping) override;
  void sum(TensorView input, int dim, TensorView out) override;
  void cumsum(TensorView input, int dim, TensorView out) override;
  void swiglu(TensorView input, TensorView out) override;
  void randNormal(TensorView out) override;
  void rand(TensorView out) override;
  void manualSeed(uint64_t seed) override;
  float elem(TensorView tensor) override;
  bool elemBool(TensorView tensor) override;
  void eq(TensorView input, TensorView other, TensorView out) override;

  MemorySnapshot captureMemorySnapshot() override;
  void resetPeakMemoryStats() override;
  void releaseUnusedMemory() override;

  DType getDefaultFloatType() override;

 private:
  std::shared_ptr<MatMul> _matmul;
  std::shared_ptr<Rand> _rand;

  CudaOperators() = default;
};

}  // namespace cuda
}  // namespace op
}  // namespace fl
