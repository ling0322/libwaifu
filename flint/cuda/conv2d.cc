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
// Conv2d, on CUTLASS.
//
// cuDNN was the first choice here until it was not one at all. It is resolved by name at run
// time, so whether a convolution went through it depended on what happened to be installed, and
// two machines running the same build convolved differently without either saying so. What it is
// still good for is being the other implementation to check and to measure against, which is
// where it now lives: conv2d_cudnn.h, built into the benchmark alone.

#include "flint/cuda/conv2d.h"

#include "flint/cuda/conv2d_cutlass.h"

namespace fl {
namespace op {
namespace cuda {

Tensor conv2d(
    const Tensor &input,
    const Tensor &weight,
    const Tensor &bias,
    const Conv2dOptions &options) {
  return conv2dCutlass(input, weight, bias, options);
}

}  // namespace cuda
}  // namespace op
}  // namespace fl
