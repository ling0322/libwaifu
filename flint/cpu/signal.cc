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

#include "flint/cpu/signal.h"

#include <math.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#include <algorithm>
#include <cstdint>

#include "lutil/error.h"
#include "flint/cpu/common.h"

namespace fl {
namespace op {
namespace cpu {

namespace {

/// The elements of `tensor` as `T`, whatever its type: for a kernel that moves bits.
template<typename T>
T *bitsOf(const TensorView &tensor) {
  return reinterpret_cast<T *>(tensor.getInternalData()->getData<void>(tensor.getInternalOffset()));
}

template<typename T>
void overlapAdd1dKernel(
    const TensorView &columns,
    const TensorView &bias,
    int length,
    int stride,
    int padding,
    int kernel,
    const TensorView &out) {
  int batch = out.getShape(0);
  int channels = out.getShape(1);
  int outLength = out.getShape(2);
  int64_t batchStride = out.getStride(0);
  int64_t channelStride = out.getStride(1);
  int64_t row = static_cast<int64_t>(channels) * kernel;

  const T *c = getDataPtrCpu<const T>(columns);
  const T *b = bias.empty() ? nullptr : getDataPtrCpu<const T>(bias);
  T *y = getDataPtrCpu<T>(out);

#pragma omp parallel for collapse(2) num_threads(omp_get_max_threads())
  for (int n = 0; n < batch; ++n) {
    for (int k = 0; k < channels; ++k) {
      T *target = y + n * batchStride + k * channelStride;
      const T *source = c + static_cast<int64_t>(n) * length * row + static_cast<int64_t>(k) * kernel;
      float shift = b ? static_cast<float>(b[k]) : 0.0f;

      for (int o = 0; o < outLength; ++o) {
        // Input position `l` lands tap `r` on `l * stride + r - padding`. The taps that reach `o`
        // are the ones congruent to `o + padding` modulo the stride, from the largest `l` down.
        int at = o + padding;
        int first = at % stride;
        float sum = shift;
        for (int r = first; r < kernel; r += stride) {
          int l = (at - r) / stride;
          if (l < 0) break;
          if (l < length) sum += static_cast<float>(source[static_cast<int64_t>(l) * row + r]);
        }
        target[o] = static_cast<T>(sum);
      }
    }
  }
}

template<typename T>
void snakeKernel(
    const TensorView &input,
    const TensorView &alpha,
    const TensorView &beta,
    float eps,
    const TensorView &out) {
  int rows = input.getShape(0) * input.getShape(1);
  int channels = input.getShape(1);
  int length = input.getShape(2);

  const T *x = getDataPtrCpu<const T>(input);
  const T *a = getDataPtrCpu<const T>(alpha);
  const T *b = beta.empty() ? a : getDataPtrCpu<const T>(beta);
  T *y = getDataPtrCpu<T>(out);

#pragma omp parallel for num_threads(omp_get_max_threads())
  for (int i = 0; i < rows; ++i) {
    int channel = i % channels;
    float scale = static_cast<float>(a[channel]);
    float inverse = 1.0f / (static_cast<float>(b[channel]) + eps);
    const T *source = x + static_cast<int64_t>(i) * length;
    T *target = y + static_cast<int64_t>(i) * length;

    for (int j = 0; j < length; ++j) {
      float v = static_cast<float>(source[j]);
      float s = sinf(scale * v);
      target[j] = static_cast<T>(v + s * s * inverse);
    }
  }
}

/// Bits, not numbers: a padding only moves elements, so it is written over unsigned integers of
/// the element's size and carries every type of that size alike.
template<typename T>
void pad1dKernel(const TensorView &input, int left, int right, PadMode mode, const TensorView &out) {
  int length = input.getShape(-1);
  int outLength = left + length + right;
  int64_t rows = input.getNumEl() / length;

  const T *x = bitsOf<const T>(input);
  T *y = bitsOf<T>(out);

#pragma omp parallel for num_threads(omp_get_max_threads())
  for (int64_t i = 0; i < rows; ++i) {
    const T *source = x + i * length;
    T *target = y + i * outLength;

    for (int j = 0; j < outLength; ++j) {
      int at = j - left;
      if (at >= 0 && at < length) {
        target[j] = source[at];
      } else if (mode == PadMode::kZero) {
        target[j] = T(0);
      } else if (mode == PadMode::kReplicate) {
        target[j] = source[std::min(std::max(at, 0), length - 1)];
      } else {
        target[j] = source[at < 0 ? -at : 2 * (length - 1) - at];
      }
    }
  }
}

template<typename T>
void clampKernel(const TensorView &input, float lo, float hi, const TensorView &out) {
  int64_t numel = input.getNumEl();
  const T *x = getDataPtrCpu<const T>(input);
  T *y = getDataPtrCpu<T>(out);

#pragma omp parallel for num_threads(omp_get_max_threads())
  for (int64_t i = 0; i < numel; ++i) {
    y[i] = static_cast<T>(std::min(std::max(static_cast<float>(x[i]), lo), hi));
  }
}

}  // namespace

void overlapAdd1d(
    const TensorView &columns,
    const TensorView &bias,
    int length,
    int stride,
    int padding,
    int kernel,
    const TensorView &out) {
  CHECK(columns.isContiguous() && out.getStride(2) == 1);
  CHECK(bias.empty() || bias.isContiguous());

  if (out.getDType() == DType::kFloat) {
    return overlapAdd1dKernel<float>(columns, bias, length, stride, padding, kernel, out);
  }
#if LUT_CPU_ARCH == LUT_AARCH64
  if (out.getDType() == DType::kFloat16) {
    return overlapAdd1dKernel<Float16>(columns, bias, length, stride, padding, kernel, out);
  }
#endif

  NOT_IMPL();
}

void snake(
    const TensorView &input,
    const TensorView &alpha,
    const TensorView &beta,
    float eps,
    const TensorView &out) {
  if (input.getDim() != 3) THROW(InvalidArg, "snake: the input must be (N, C, L)");
  if (!input.isContiguous() || !out.isContiguous()) {
    THROW(InvalidArg, "snake takes a contiguous input");
  }
  int channels = input.getShape(1);
  if (alpha.getNumEl() != channels || !alpha.isContiguous() ||
      (!beta.empty() && (beta.getNumEl() != channels || !beta.isContiguous()))) {
    THROW(InvalidArg, "snake: alpha and beta are one per channel, contiguous");
  }
  if (alpha.getDType() != input.getDType() ||
      (!beta.empty() && beta.getDType() != input.getDType())) {
    THROW(InvalidArg, "snake: alpha and beta are of the input's type");
  }
  out.throwIfInvalidShape(input.getShape(), "snake");

  if (input.getDType() == DType::kFloat) return snakeKernel<float>(input, alpha, beta, eps, out);
#if LUT_CPU_ARCH == LUT_AARCH64
  if (input.getDType() == DType::kFloat16) {
    return snakeKernel<Float16>(input, alpha, beta, eps, out);
  }
#endif

  NOT_IMPL();
}

void pad1d(const TensorView &input, int left, int right, PadMode mode, const TensorView &out) {
  if (input.getDim() < 1) THROW(InvalidArg, "pad1d: the input has no dimension to pad");
  if (!input.isContiguous() || !out.isContiguous()) {
    THROW(InvalidArg, "pad1d takes a contiguous input");
  }
  int length = input.getShape(-1);
  if (left < 0 || right < 0) THROW(InvalidArg, "pad1d: a padding cannot be negative");
  if (mode == PadMode::kReflect && (left >= length || right >= length)) {
    THROW(InvalidArg, "pad1d: a reflection pads by less than the input is long");
  }
  std::vector<int> shape = input.getShape();
  shape.back() = left + length + right;
  out.throwIfInvalidShape(shape, "pad1d");
  if (out.getDType() != input.getDType()) THROW(InvalidArg, "pad1d: out is of another type");

  switch (input.getDType().getTotalSize(1)) {
    case 1:
      return pad1dKernel<uint8_t>(input, left, right, mode, out);
    case 2:
      return pad1dKernel<uint16_t>(input, left, right, mode, out);
    case 4:
      return pad1dKernel<uint32_t>(input, left, right, mode, out);
    case 8:
      return pad1dKernel<uint64_t>(input, left, right, mode, out);
    default:
      NOT_IMPL();
  }
}

void clamp(const TensorView &input, float lo, float hi, const TensorView &out) {
  if (!input.isContiguous() || !out.isContiguous()) {
    THROW(InvalidArg, "clamp takes a contiguous input");
  }
  out.throwIfInvalidShape(input.getShape(), "clamp");

  if (input.getDType() == DType::kFloat) return clampKernel<float>(input, lo, hi, out);
#if LUT_CPU_ARCH == LUT_AARCH64
  if (input.getDType() == DType::kFloat16) return clampKernel<Float16>(input, lo, hi, out);
#endif

  NOT_IMPL();
}

}  // namespace cpu
}  // namespace op
}  // namespace fl
