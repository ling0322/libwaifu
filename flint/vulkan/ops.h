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

#include <stdint.h>

#include "lutil/span.h"
#include "flint/dtype.h"
#include "flint/tensor.h"
#include "flint/tensor_view.h"

namespace fl {
namespace op {
namespace vulkan {

// elementwise.cc
enum class UnaryOp {
  kNeg = 0,
  kAbs = 1,
  kExp = 2,
  kSqrt = 3,
  kRsqrt = 4,
  kSquare = 5,
  kSigmoid = 6,
  kTanh = 7,
  kRelu = 8,
  kGelu = 9,
  kSilu = 10,
  kSin = 11,
  kCos = 12,
  kQuickGelu = 13,
  kMulScalar = 14,
  kDivScalar = 15,
  kAddScalar = 16,
  kLog = 17,
  kRound = 18,
};

enum class BinaryOp {
  kAdd = 0,
  kSub = 1,
  kMul = 2,
  kDiv = 3,
};

void unary(UnaryOp op, const TensorView &input, float scalar, const TensorView &out);

/// `b` is already of `a`'s shape -- broadcast, if it was, as a view with strides of zero.
void binary(BinaryOp op, const TensorView &a, const TensorView &b, const TensorView &out);
void eq(const TensorView &a, const TensorView &b, const TensorView &out);
void mod(const TensorView &input, int64_t other, const TensorView &out);
void fill(const TensorView &tensor, float value);
void copy(const TensorView &src, const TensorView &dest);
void cast(const TensorView &input, const TensorView &out);
void arangeLong(int64_t begin, int64_t step, const TensorView &out);
void causalMask(const TensorView &out);

// reduce.cc
enum class ReduceOp {
  kSum = 0,
  kMax = 1,
  kMin = 2,
};

/// Reduce the last dimension into `out`, which is the input without it -- (1) for a vector.
void reduceLastDim(const TensorView &input, ReduceOp op, const TensorView &out);
void sum(const TensorView &input, int dim, const TensorView &out);
void cumsum(const TensorView &input, int dim, const TensorView &out);
void softmax(const TensorView &input, const TensorView &out);
bool all(const TensorView &input);
float elem(const TensorView &tensor);
bool elemBool(const TensorView &tensor);

// norm.cc
void layerNorm(
    const TensorView &input,
    const TensorView &weight,
    const TensorView &bias,
    float eps,
    const TensorView &out);
void rmsNorm(const TensorView &input, const TensorView &weight, float eps, const TensorView &out);
void groupNorm(
    const TensorView &input,
    const TensorView &weight,
    const TensorView &bias,
    int groups,
    float eps,
    const TensorView &out);

// shape.cc
void lookup(const TensorView &table, const TensorView &indices, const TensorView &out);
void upsampleNearest2d(const TensorView &input, int scale, const TensorView &out);
void upsampleNearest1d(const TensorView &input, const TensorView &out);
void geglu(const TensorView &input, const TensorView &out);
void swiglu(const TensorView &input, const TensorView &out);
void rotaryEmbedding(
    const TensorView &positions,
    const TensorView &query,
    const TensorView &key,
    const TensorView &rotaryCache);

// matmul.cc
void matmul(const TensorView &a, const TensorView &b, const TensorView &out);
void conv2d(
    const TensorView &input,
    const TensorView &weight,
    const TensorView &bias,
    int stride,
    int padding,
    int dilation,
    int groups,
    const TensorView &out);
void conv1d(
    const TensorView &input,
    const TensorView &weight,
    const TensorView &bias,
    int stride,
    int padding,
    int dilation,
    int groups,
    const TensorView &out);

// rand.cc
/// Philox4x32-10, drawing the same numbers for a seed as the CUDA operators do.
class Rand {
 public:
  /// Fill `out`, contiguous float32, from [0, 1) or from the standard normal distribution.
  void uniform(const TensorView &out);
  void normal(const TensorView &out);
  void setSeed(uint64_t seed);

 private:
  uint64_t _seed = 0;
  uint64_t _position = 0;

  void draw(const TensorView &out, bool normal);
};

}  // namespace vulkan
}  // namespace op
}  // namespace fl
