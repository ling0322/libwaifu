// The MIT License (MIT)
//
// Copyright (c) 2024 Xiaoyang Chen
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

#include "flint/tensor_view.h"

namespace fl {
namespace op {
namespace cpu {

/// @brief RMS-normalize over the last dimension and scale by `weight`, into `C` of the input's
///        shape and type.
void rmsNorm(const TensorView &tensor, const TensorView &weight, float eps, const TensorView &C);

/// @brief Normalize over the last dimension, subtracting the mean as well as dividing by the
///        spread. `weight` and `bias` are one value per position and either may be empty.
/// @param tensor <float>(..., hiddenSize), or <float16> where the CPU has it natively.
/// @param C where the result goes, of the input's shape and type.
void layerNorm(
    const TensorView &tensor,
    const TensorView &weight,
    const TensorView &bias,
    float eps,
    const TensorView &C);

/// @brief Normalize each image's group of channels over the channels and the space they cover,
///        which is what a diffusion model normalizes with.
/// @param tensor <float>(N, C, H, W), contiguous. `weight` and `bias` are one value per channel.
/// @param C where the result goes, contiguous, of the input's shape and type.
void groupNorm(
    const TensorView &tensor,
    const TensorView &weight,
    const TensorView &bias,
    int groups,
    float eps,
    const TensorView &C);

}  // namespace cpu
}  // namespace op
}  // namespace fl
