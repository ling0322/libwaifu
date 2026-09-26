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

Tensor unary(UnaryOp op, const Tensor &input, float scalar = 0.0f);
Tensor binary(BinaryOp op, const Tensor &a, const Tensor &b);
Tensor eq(const Tensor &a, const Tensor &b);
Tensor mod(const Tensor &input, int64_t other);
void fill(const Tensor &tensor, float value);
void copy(const Tensor &src, const Tensor &dest);
Tensor cast(const Tensor &input, DType dtype);
Tensor arangeLong(int64_t begin, int64_t end, int64_t step);
Tensor causalMask(int length, DType dtype);

// reduce.cc
enum class ReduceOp {
  kSum = 0,
  kMax = 1,
  kMin = 2,
};

/// Reduce the last dimension, which the result drops -- a vector becomes a vector of one.
Tensor reduceLastDim(const Tensor &input, ReduceOp op);
Tensor sum(const Tensor &input, int dim);
Tensor softmax(const Tensor &input);
bool all(const Tensor &input);
float elem(const Tensor &tensor);
bool elemBool(const Tensor &tensor);

// norm.cc
Tensor layerNorm(const Tensor &input, const Tensor &weight, const Tensor &bias, float eps);
Tensor rmsNorm(const Tensor &input, const Tensor &weight, float eps);
Tensor groupNorm(
    const Tensor &input,
    const Tensor &weight,
    const Tensor &bias,
    int groups,
    float eps);

// shape.cc
Tensor lookup(const Tensor &table, const Tensor &indices);
Tensor upsampleNearest2d(const Tensor &input, int scale);
Tensor geglu(const Tensor &input);
Tensor swiglu(const Tensor &input);
void rotaryEmbedding(
    const Tensor &positions,
    const Tensor &query,
    const Tensor &key,
    const Tensor &rotaryCache);

// matmul.cc
Tensor matmul(const Tensor &a, const Tensor &b);
Tensor conv2d(
    const Tensor &input,
    const Tensor &weight,
    const Tensor &bias,
    int stride,
    int padding,
    int dilation,
    int groups);
Tensor conv1d(
    const Tensor &input,
    const Tensor &weight,
    const Tensor &bias,
    int stride,
    int padding,
    int dilation,
    int groups);

// rand.cc
/// Philox4x32-10, drawing the same numbers for a seed as the CUDA operators do.
class Rand {
 public:
  Tensor uniform(lut::Span<const int> shape);
  Tensor normal(lut::Span<const int> shape);
  void setSeed(uint64_t seed);

 private:
  uint64_t _seed = 0;
  uint64_t _position = 0;

  Tensor draw(lut::Span<const int> shape, bool normal);
};

}  // namespace vulkan
}  // namespace op
}  // namespace fl
