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

#include "flint/cpu/upsample.h"

#include "lutil/error.h"
#include "flint/cpu/common.h"
#include "flint/cpu/tensor.h"
#include "flint/tensor.h"

namespace fl {
namespace op {
namespace cpu {
namespace {

/// Driven from the output rather than the input, so the writes run straight down each row and
/// each input row is read `scale` times in a row while it is still warm. The value is only ever
/// copied, never arithmetic on, so the element type is carried through rather than widened.
template<typename T>
Tensor upsampleNearest2dKernel(const Tensor &input, int scale) {
  int batch = input.getShape(0);
  int channels = input.getShape(1);
  int inputH = input.getShape(2);
  int inputW = input.getShape(3);
  int outputH = inputH * scale;
  int outputW = inputW * scale;

  Tensor output = tensor({batch, channels, outputH, outputW}, input.getDType());
  const T *in = input.getInternalData()->getData<T>(input.getInternalOffset());
  T *out = output.getInternalData()->getData<T>(output.getInternalOffset());

  int planes = batch * channels;
#pragma omp parallel for schedule(dynamic, 1)
  for (int plane = 0; plane < planes; ++plane) {
    const T *inPlane = in + static_cast<int64_t>(plane) * inputH * inputW;
    T *outPlane = out + static_cast<int64_t>(plane) * outputH * outputW;

    for (int y = 0; y < outputH; ++y) {
      const T *inRow = inPlane + static_cast<int64_t>(y / scale) * inputW;
      T *outRow = outPlane + static_cast<int64_t>(y) * outputW;

      for (int x = 0; x < inputW; ++x) {
        T value = inRow[x];
        for (int j = 0; j < scale; ++j) outRow[x * scale + j] = value;
      }
    }
  }

  return output;
}

}  // namespace

Tensor upsampleNearest2d(const Tensor &input, int scale) {
  if (input.getDim() != 4) {
    THROW(InvalidArg, "upsampleNearest2d takes a 4-D input, as (N, C, H, W)");
  }
  if (!input.isContiguous()) THROW(InvalidArg, "upsampleNearest2d takes a contiguous input");
  if (scale < 1) THROW(InvalidArg, "upsampleNearest2d: the scale is below one");

  if (input.getDType() == DType::kFloat) return upsampleNearest2dKernel<float>(input, scale);
#if LUT_CPU_ARCH == LUT_AARCH64
  if (input.getDType() == DType::kFloat16) return upsampleNearest2dKernel<Float16>(input, scale);
#endif

  NOT_IMPL();
}

}  // namespace cpu
}  // namespace op
}  // namespace fl
