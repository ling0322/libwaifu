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

#include "flint/dtype.h"
#include "flint/tensor_view.h"

namespace fl {
namespace op {
namespace metal {

// Every function here reads the views it is given and writes its result into `out`, which the
// caller (flint/functional.cc) has already allocated on the Metal device with the right shape and
// dtype. None of them allocates a tensor anyone else sees.

// elementwise.cc
void add(const TensorView &a, const TensorView &b, const TensorView &out);
void sub(const TensorView &a, const TensorView &b, const TensorView &out);
void mul(const TensorView &a, const TensorView &b, const TensorView &out);
void divTensor(const TensorView &a, const TensorView &b, const TensorView &out);
void eq(const TensorView &a, const TensorView &b, const TensorView &out);
void mulScalar(const TensorView &a, float other, const TensorView &out);
void divScalar(const TensorView &a, float other, const TensorView &out);
void subScalar(const TensorView &a, float other, const TensorView &out);
void neg(const TensorView &a, const TensorView &out);
void abs(const TensorView &a, const TensorView &out);
void exp(const TensorView &a, const TensorView &out);
void log(const TensorView &a, const TensorView &out);
void round(const TensorView &a, const TensorView &out);
void sqrt(const TensorView &a, const TensorView &out);
void rsqrt(const TensorView &a, const TensorView &out);
void square(const TensorView &a, const TensorView &out);
void sigmoid(const TensorView &a, const TensorView &out);
void tanh(const TensorView &a, const TensorView &out);
void relu(const TensorView &a, const TensorView &out);
void gelu(const TensorView &a, const TensorView &out);
void silu(const TensorView &a, const TensorView &out);
void quickGelu(const TensorView &a, const TensorView &out);
void sin(const TensorView &a, const TensorView &out);
void cos(const TensorView &a, const TensorView &out);
void softmax(const TensorView &a, const TensorView &out);

// matmul.cc
void matmul(const TensorView &a, const TensorView &b, const TensorView &out);

// norm.cc
void layerNorm(
    const TensorView &input,
    const TensorView &weight,
    const TensorView &bias,
    float eps,
    const TensorView &out);
void groupNorm(
    const TensorView &input,
    const TensorView &weight,
    const TensorView &bias,
    int groups,
    float eps,
    const TensorView &out);

// attention.cc
void attention(
    const TensorView &q,
    const TensorView &k,
    const TensorView &v,
    bool causal,
    const TensorView &out);

// conv1d.cc
void conv1d(
    const TensorView &input,
    const TensorView &weight,
    const TensorView &bias,
    int stride,
    int padding,
    int dilation,
    int groups,
    const TensorView &out);

// conv2d.cc
void conv2d(
    const TensorView &input,
    const TensorView &weight,
    const TensorView &bias,
    int stride,
    int padding,
    int dilation,
    int groups,
    const TensorView &out);

// shape.cc
void lookup(const TensorView &table, const TensorView &indices, const TensorView &out);
void upsampleNearest2d(const TensorView &input, int scale, const TensorView &out);
void geglu(const TensorView &input, const TensorView &out);
void swiglu(const TensorView &input, const TensorView &out);
void cast(const TensorView &input, const TensorView &out);
void fill(const TensorView &input, float value);
void copy(const TensorView &src, const TensorView &dest);
void print(const TensorView &tensor);

// reduce.cc
void sum(const TensorView &input, int dim, const TensorView &out);
void cumsum(const TensorView &input, int dim, const TensorView &out);
void max(const TensorView &input, const TensorView &out);
void min(const TensorView &input, const TensorView &out);
bool all(const TensorView &input);
bool allClose(const TensorView &a, const TensorView &b, float rtol, float atol);
float elem(const TensorView &tensor);
bool elemBool(const TensorView &tensor);

// rand.cc
void rand(const TensorView &out);
void randNormal(const TensorView &out);
void manualSeed(uint64_t seed);

}  // namespace metal
}  // namespace op
}  // namespace fl
