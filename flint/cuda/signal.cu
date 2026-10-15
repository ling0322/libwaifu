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

#include <cuda_fp16.h>

#include <limits>
#include <vector>

#include "lutil/error.h"
#include "lutil/strings.h"
#include "flint/cuda/common.h"
#include "flint/cuda/signal.h"

namespace fl {
namespace op {
namespace cuda {
namespace {

constexpr int kBlockSize = 256;

/// The elements of `tensor` as `T`, whatever its type: for a kernel that moves bits.
template<typename T>
T *bitsOf(const TensorView &tensor) {
  return reinterpret_cast<T *>(tensor.getInternalData()->getData<void>(tensor.getInternalOffset()));
}

/// Every kernel here indexes with 32 bits, which is what keeps FastDivmod usable.
void checkSize(int64_t numel, const char *name) {
  if (numel > std::numeric_limits<int32_t>::max()) {
    THROW(InvalidArg, lut::sprintf("%s: the result is too large", name));
  }
}

/// Evaluated in float whatever the tensor's type, as the CPU kernels are.
template<typename T>
__device__ __forceinline__ float load(const T *p, int64_t i) {
  return static_cast<float>(p[i]);
}

template<typename T>
__device__ __forceinline__ T store(float x) {
  return static_cast<T>(x);
}

template<>
__device__ __forceinline__ half store<half>(float x) {
  return __float2half(x);
}

/// One thread per output element. Input position `l` lands tap `r` on `l * stride + r - padding`,
/// so the taps reaching output `o` are the ones congruent to `o + padding` modulo the stride --
/// ceil(R / stride) of them, two for every vocoder here -- and the thread reads exactly those.
template<typename T>
__global__ void overlapAdd1dKernel(
    const T *__restrict__ columns,
    const T *__restrict__ bias,
    T *__restrict__ out,
    int length,
    int stride,
    int padding,
    int kernel,
    int channels,
    int64_t batchStride,
    int64_t channelStride,
    FastDivmod outLengthDivmod,
    FastDivmod channelsDivmod,
    uint32_t numel) {
  int64_t row = static_cast<int64_t>(channels) * kernel;
  for (uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < numel;
       idx += blockDim.x * gridDim.x) {
    uint32_t plane, o, n, k;
    outLengthDivmod.divmod(idx, plane, o);
    channelsDivmod.divmod(plane, n, k);

    int at = static_cast<int>(o) + padding;
    const T *source = columns + static_cast<int64_t>(n) * length * row + static_cast<int64_t>(k) * kernel;
    float sum = bias ? load(bias, k) : 0.0f;
    for (int r = at % stride; r < kernel; r += stride) {
      int l = (at - r) / stride;
      if (l < 0) break;
      if (l < length) sum += load(source, static_cast<int64_t>(l) * row + r);
    }

    out[n * batchStride + k * channelStride + o] = store<T>(sum);
  }
}

template<typename T>
__global__ void snakeKernel(
    const T *__restrict__ input,
    const T *__restrict__ alpha,
    const T *__restrict__ beta,
    T *__restrict__ out,
    float eps,
    FastDivmod lengthDivmod,
    FastDivmod channelsDivmod,
    uint32_t numel) {
  for (uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < numel;
       idx += blockDim.x * gridDim.x) {
    uint32_t plane, j, n, channel;
    lengthDivmod.divmod(idx, plane, j);
    channelsDivmod.divmod(plane, n, channel);

    float x = load(input, idx);
    float s = sinf(load(alpha, channel) * x);
    out[idx] = store<T>(x + s * s / (load(beta, channel) + eps));
  }
}

/// Bits, not numbers, as on the CPU: a padding only moves elements.
template<typename T>
__global__ void pad1dKernel(
    const T *__restrict__ input,
    T *__restrict__ out,
    int length,
    int left,
    PadMode mode,
    FastDivmod outLengthDivmod,
    uint32_t numel) {
  for (uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < numel;
       idx += blockDim.x * gridDim.x) {
    uint32_t row, j;
    outLengthDivmod.divmod(idx, row, j);
    const T *source = input + static_cast<int64_t>(row) * length;

    int at = static_cast<int>(j) - left;
    if (at >= 0 && at < length) {
      out[idx] = source[at];
    } else if (mode == PadMode::kZero) {
      out[idx] = T(0);
    } else if (mode == PadMode::kReplicate) {
      out[idx] = source[min(max(at, 0), length - 1)];
    } else {
      out[idx] = source[at < 0 ? -at : 2 * (length - 1) - at];
    }
  }
}

template<typename T>
__global__ void clampKernel(
    const T *__restrict__ input,
    T *__restrict__ out,
    float lo,
    float hi,
    uint32_t numel) {
  for (uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < numel;
       idx += blockDim.x * gridDim.x) {
    // A NaN stays one, as torch.clamp and the CPU have it: fmaxf and fminf would return the bound.
    float x = load(input, idx);
    out[idx] = store<T>(isnan(x) ? x : fminf(fmaxf(x, lo), hi));
  }
}

template<typename T>
void overlapAdd1dImpl(
    const TensorView &columns,
    const TensorView &bias,
    int length,
    int stride,
    int padding,
    int kernel,
    const TensorView &out) {
  int64_t numel = out.getNumEl();
  checkSize(numel, "convTranspose1d");
  checkSize(columns.getNumEl(), "convTranspose1d");
  if (numel == 0) return;

  overlapAdd1dKernel<T><<<getGrid1D(static_cast<int>(numel), kBlockSize), kBlockSize>>>(
      getDataPtrCuda<T>(columns),
      bias.empty() ? nullptr : getDataPtrCuda<T>(bias),
      getDataPtrCuda<T>(out),
      length,
      stride,
      padding,
      kernel,
      out.getShape(1),
      out.getStride(0),
      out.getStride(1),
      FastDivmod(out.getShape(2)),
      FastDivmod(out.getShape(1)),
      static_cast<uint32_t>(numel));

  LL_CUDA_SYNCHRONIZE();
  LL_CHECK_CUDA_STATUS(cudaGetLastError());
}

template<typename T>
void snakeImpl(
    const TensorView &input,
    const TensorView &alpha,
    const TensorView &beta,
    float eps,
    const TensorView &out) {
  int64_t numel = input.getNumEl();
  checkSize(numel, "snake");
  if (numel == 0) return;

  const T *a = getDataPtrCuda<T>(alpha);
  snakeKernel<T><<<getGrid1D(static_cast<int>(numel), kBlockSize), kBlockSize>>>(
      getDataPtrCuda<T>(input),
      a,
      beta.empty() ? a : getDataPtrCuda<T>(beta),
      getDataPtrCuda<T>(out),
      eps,
      FastDivmod(input.getShape(2)),
      FastDivmod(input.getShape(1)),
      static_cast<uint32_t>(numel));

  LL_CUDA_SYNCHRONIZE();
  LL_CHECK_CUDA_STATUS(cudaGetLastError());
}

template<typename T>
void pad1dImpl(const TensorView &input, int left, PadMode mode, const TensorView &out) {
  int64_t numel = out.getNumEl();
  checkSize(numel, "pad1d");
  if (numel == 0) return;

  pad1dKernel<T><<<getGrid1D(static_cast<int>(numel), kBlockSize), kBlockSize>>>(
      bitsOf<const T>(input),
      bitsOf<T>(out),
      input.getShape(-1),
      left,
      mode,
      FastDivmod(out.getShape(-1)),
      static_cast<uint32_t>(numel));

  LL_CUDA_SYNCHRONIZE();
  LL_CHECK_CUDA_STATUS(cudaGetLastError());
}

template<typename T>
void clampImpl(const TensorView &input, float lo, float hi, const TensorView &out) {
  int64_t numel = input.getNumEl();
  checkSize(numel, "clamp");
  if (numel == 0) return;

  clampKernel<T><<<getGrid1D(static_cast<int>(numel), kBlockSize), kBlockSize>>>(
      getDataPtrCuda<T>(input), getDataPtrCuda<T>(out), lo, hi, static_cast<uint32_t>(numel));

  LL_CUDA_SYNCHRONIZE();
  LL_CHECK_CUDA_STATUS(cudaGetLastError());
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

  DType dtype = out.getDType();
  if (dtype == DType::kFloat16) {
    return overlapAdd1dImpl<half>(columns, bias, length, stride, padding, kernel, out);
  }
  if (dtype == DType::kFloat) {
    return overlapAdd1dImpl<float>(columns, bias, length, stride, padding, kernel, out);
  }

  THROW(InvalidArg, "convTranspose1d takes a <half> or <float> input");
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

  if (input.getDType() == DType::kFloat16) return snakeImpl<half>(input, alpha, beta, eps, out);
  if (input.getDType() == DType::kFloat) return snakeImpl<float>(input, alpha, beta, eps, out);

  THROW(InvalidArg, "snake takes a <half> or <float> input");
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
      return pad1dImpl<uint8_t>(input, left, mode, out);
    case 2:
      return pad1dImpl<uint16_t>(input, left, mode, out);
    case 4:
      return pad1dImpl<uint32_t>(input, left, mode, out);
    case 8:
      return pad1dImpl<uint64_t>(input, left, mode, out);
    default:
      THROW(InvalidArg, "pad1d: no kernel for an element of this size");
  }
}

void clamp(const TensorView &input, float lo, float hi, const TensorView &out) {
  if (!input.isContiguous() || !out.isContiguous()) {
    THROW(InvalidArg, "clamp takes a contiguous input");
  }
  out.throwIfInvalidShape(input.getShape(), "clamp");

  if (input.getDType() == DType::kFloat16) return clampImpl<half>(input, lo, hi, out);
  if (input.getDType() == DType::kFloat) return clampImpl<float>(input, lo, hi, out);

  THROW(InvalidArg, "clamp takes a <half> or <float> input");
}

}  // namespace cuda
}  // namespace op
}  // namespace fl
