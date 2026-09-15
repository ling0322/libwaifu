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

#include "flint/tensor.h"

namespace fl {
namespace op {
namespace cuda {

/// @brief Scale each row of `input` by the reciprocal of its root mean square, then by `weight`.
///        This is the layerNorm below with the mean left at zero, and it shares its kernels.
/// @param input <half>(..., D).
/// @param weight <half>(D).
/// @return a tensor shaped like `input`.
Tensor rmsNorm(const Tensor &input, const Tensor &weight, float eps);

/// @brief Normalize each row of `input` to zero mean and unit variance, then scale and shift it.
///        Unlike an RMS norm this subtracts the mean, which is what every transformer outside the
///        Llama lineage does, CLIP and the diffusion U-Net among them.
/// @param input <half>(..., D). A tensor that is not contiguous is read through its strides.
/// @param weight <half>(D), or empty for no scaling.
/// @param bias <half>(D), or empty for no shift.
/// @return a tensor shaped like `input`.
Tensor layerNorm(const Tensor &input, const Tensor &weight, const Tensor &bias, float eps);

/// @brief Normalize over each group of channels together with the space they cover, then scale
///        and shift per channel. This is the normalization a diffusion U-Net and its VAE use,
///        where the batch is one image and normalizing across it would say nothing.
/// @param input <half>(N, C, H, W), contiguous, with C divisible by `groups`.
/// @param weight <half>(C), or empty for no scaling.
/// @param bias <half>(C), or empty for no shift.
/// @return a tensor shaped like `input`.
Tensor groupNorm(
    const Tensor &input,
    const Tensor &weight,
    const Tensor &bias,
    int groups,
    float eps);

}  // namespace cuda
}  // namespace op
}  // namespace fl
