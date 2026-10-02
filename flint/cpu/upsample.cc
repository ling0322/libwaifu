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

#include <algorithm>
#include <cmath>
#include <vector>

#include "lutil/error.h"
#include "flint/cpu/common.h"

namespace fl {
namespace op {
namespace cpu {
namespace {

/// Driven from the output rather than the input, so the writes run straight down each row and
/// each input row is read `scale` times in a row while it is still warm. The value is only ever
/// copied, never arithmetic on, so the element type is carried through rather than widened.
template<typename T>
void upsampleNearest2dKernel(const TensorView &input, int scale, const TensorView &output) {
  int batch = input.getShape(0);
  int channels = input.getShape(1);
  int inputH = input.getShape(2);
  int inputW = input.getShape(3);
  int outputH = inputH * scale;
  int outputW = inputW * scale;

  output.throwIfInvalidShape({batch, channels, outputH, outputW}, "upsampleNearest2d");
  CHECK(output.isContiguous());
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
}

/// Each row read through one table of source positions, worked out once: the rows are many and
/// the table is the same for all of them. Only copied, so the element is moved as its bytes.
template<typename T>
void upsampleNearest1dKernel(const TensorView &input, const TensorView &output) {
  int length = input.getShape(-1);
  int size = output.getShape(-1);
  int64_t rows = input.getNumEl() / length;
  CHECK(output.isContiguous() && output.getNumEl() == rows * size);
  const T *in = reinterpret_cast<const T *>(input.getInternalData()->getData<void>(
      input.getInternalOffset()));
  T *out = reinterpret_cast<T *>(output.getInternalData()->getData<void>(
      output.getInternalOffset()));

  std::vector<int> source = nearestSources(length, size);

#pragma omp parallel for
  for (int64_t row = 0; row < rows; ++row) {
    const T *inRow = in + row * length;
    T *outRow = out + row * size;
    for (int j = 0; j < size; ++j) outRow[j] = inRow[source[j]];
  }
}

}  // namespace

std::vector<int> nearestSources(int length, int size) {
  // Volatile so that neither the division nor the product is folded into something wider or
  // fused: the whole point is to land on the float32 value torch lands on.
  volatile float scale = static_cast<float>(length) / static_cast<float>(size);

  std::vector<int> source(size);
  for (int j = 0; j < size; ++j) {
    volatile float position = static_cast<float>(j) * scale;
    source[j] = std::min(static_cast<int>(std::floor(position)), length - 1);
  }
  return source;
}

void upsampleNearest1d(const TensorView &input, const TensorView &output) {
  int size = output.getShape(-1);
  if (input.getDim() < 1) THROW(InvalidArg, "upsampleNearest1d takes at least one dimension");
  if (!input.isContiguous()) THROW(InvalidArg, "upsampleNearest1d takes a contiguous input");
  if (size < 1) THROW(InvalidArg, "upsampleNearest1d: the size is below one");
  if (input.getShape(-1) < 1) THROW(InvalidArg, "upsampleNearest1d: the input is empty");

  if (input.getDType() == DType::kFloat) return upsampleNearest1dKernel<uint32_t>(input, output);
  if (input.getDType() == DType::kFloat16) return upsampleNearest1dKernel<uint16_t>(input, output);

  NOT_IMPL();
}

void upsampleNearest2d(const TensorView &input, int scale, const TensorView &output) {
  if (input.getDim() != 4) {
    THROW(InvalidArg, "upsampleNearest2d takes a 4-D input, as (N, C, H, W)");
  }
  if (!input.isContiguous()) THROW(InvalidArg, "upsampleNearest2d takes a contiguous input");
  if (scale < 1) THROW(InvalidArg, "upsampleNearest2d: the scale is below one");

  if (input.getDType() == DType::kFloat) return upsampleNearest2dKernel<float>(input, scale, output);
#if LUT_CPU_ARCH == LUT_AARCH64
  if (input.getDType() == DType::kFloat16) return upsampleNearest2dKernel<Float16>(input, scale, output);
#endif

  NOT_IMPL();
}

}  // namespace cpu
}  // namespace op
}  // namespace fl
