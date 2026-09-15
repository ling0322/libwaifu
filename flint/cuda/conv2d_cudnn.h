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
// The cuDNN convolution, which is the reference the CUTLASS one is measured against and nothing
// else. The library convolves through it no longer: cuDNN is resolved by name at run time, so a
// machine without it silently got a different implementation than the one the numbers were taken
// on, which is the kind of difference that should be asked for rather than fallen into.

#pragma once

#include "flint/cuda/conv2d.h"
#include "flint/tensor.h"

namespace fl {
namespace op {
namespace cuda {

/// @brief Whether cuDNN is on this machine. Looked for by name at the first call, so a build with
///        cuDNN still answers where the library is absent.
bool isConv2dCudnnAvailable();

/// @brief The same convolution conv2d() performs, on cuDNN. Takes the groups CUTLASS will not.
///
/// Throws rather than falling back where cuDNN did not load: what this is for is being the other
/// implementation, and one that turns into the first is measuring nothing.
/// @param input <half|float>(N, C, H, W), contiguous.
/// @param weight <half|float>(K, C / groups, R, S), contiguous and of the same type as `input`.
/// @param bias <half|float>(K), or an empty tensor for no bias.
/// @return <half|float>(N, K, outH, outW).
Tensor conv2dCudnn(
    const Tensor &input,
    const Tensor &weight,
    const Tensor &bias,
    const Conv2dOptions &options);

}  // namespace cuda
}  // namespace op
}  // namespace fl
