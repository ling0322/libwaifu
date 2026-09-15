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

#include "lutil/span.h"
#include "flint/dtype.h"
#include "flint/tensor.h"

namespace fl {
namespace op {
namespace metal {

// elementwise.cc
Tensor add(const Tensor &a, const Tensor &b);
Tensor sub(const Tensor &a, const Tensor &b);
Tensor mul(const Tensor &a, const Tensor &b);
Tensor divTensor(const Tensor &a, const Tensor &b);
Tensor eq(const Tensor &a, const Tensor &b);
Tensor mulScalar(const Tensor &a, float other);
Tensor divScalar(const Tensor &a, float other);
Tensor subScalar(const Tensor &a, float other);
Tensor neg(const Tensor &a);
Tensor abs(const Tensor &a);
Tensor exp(const Tensor &a);
Tensor sqrt(const Tensor &a);
Tensor rsqrt(const Tensor &a);
Tensor square(const Tensor &a);
Tensor sigmoid(const Tensor &a);
Tensor tanh(const Tensor &a);
Tensor relu(const Tensor &a);
Tensor gelu(const Tensor &a);
Tensor silu(const Tensor &a);
Tensor quickGelu(const Tensor &a);
Tensor sin(const Tensor &a);
Tensor cos(const Tensor &a);
Tensor softmax(const Tensor &a);

// matmul.cc
Tensor matmul(const Tensor &a, const Tensor &b);

// norm.cc
Tensor layerNorm(const Tensor &input, const Tensor &weight, const Tensor &bias, float eps);
Tensor groupNorm(
    const Tensor &input,
    const Tensor &weight,
    const Tensor &bias,
    int groups,
    float eps);

// attention.cc
Tensor attention(const Tensor &q, const Tensor &k, const Tensor &v, bool causal);

// conv2d.cc
Tensor conv2d(
    const Tensor &input,
    const Tensor &weight,
    const Tensor &bias,
    int stride,
    int padding,
    int dilation,
    int groups);

// shape.cc
Tensor lookup(const Tensor &table, const Tensor &indices);
Tensor upsampleNearest2d(const Tensor &input, int scale);
Tensor geglu(const Tensor &input);
Tensor swiglu(const Tensor &input);
Tensor cast(const Tensor &input, DType dtype);
Tensor createTensor(lut::Span<const int> shape, DType dtype);
Tensor zeros(lut::Span<const int> shape, DType dtype);
void fill(Tensor input, float value);
void copy(const Tensor &src, Tensor dest);
void print(const Tensor &tensor);

// reduce.cc
Tensor sum(const Tensor &input, int dim);
Tensor max(const Tensor &input);
Tensor min(const Tensor &input);
bool all(const Tensor &input);
bool allClose(const Tensor &a, const Tensor &b, float rtol, float atol);
float elem(const Tensor &tensor);
bool elemBool(const Tensor &tensor);

// rand.cc
Tensor rand(lut::Span<const int> shape, DType dtype);
Tensor randNormal(lut::Span<const int> shape);
void manualSeed(uint64_t seed);

}  // namespace metal
}  // namespace op
}  // namespace fl
