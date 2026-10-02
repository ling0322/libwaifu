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

#include <cuda_fp16.h>

#include "flint/cuda/cast.h"
#include "flint/cuda/common.h"
#include "flint/functional.h"

namespace fl {
namespace op {
namespace cuda {

__global__ void castFloatToHalfKernel(int64_t n, const float *src, half *dest) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= n) return;

  dest[idx] = __float2half(src[idx]);
}

__global__ void castHalfToFloatKernel(int64_t n, const half *src, float *dest) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= n) return;

  dest[idx] = __half2float(src[idx]);
}

template<typename T>
__global__ void castToLongKernel(int64_t n, const T *src, LongType *dest) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= n) return;

  // Through float for half, which every half is exactly; the conversion truncates toward zero.
  dest[idx] = static_cast<LongType>(static_cast<float>(src[idx]));
}

void cast(const TensorView &input, const TensorView &out) {
  CHECK(input.getDevice().getType() == Device::kCuda);
  CHECK(out.getDevice().getType() == Device::kCuda);
  LL_CHECK_CONTIGUOUS(input);
  LL_CHECK_CONTIGUOUS(out);
  out.throwIfInvalidShape(input.getShape(), "cast");

  DType from = input.getDType();
  DType to = out.getDType();
  int64_t numel = input.getNumEl();
  if (numel == 0) return;

  if (from == to) {
    LL_CHECK_CUDA_STATUS(cudaMemcpy(
        getDataPtrCuda<void>(out),
        getDataPtrCuda<void>(input),
        from.getTotalSize(numel),
        cudaMemcpyDeviceToDevice));
    return;
  }

  constexpr int blockSize = 256;
  int64_t nb = (numel + blockSize - 1) / blockSize;
  if (from == DType::kFloat16 && to == DType::kFloat) {
    castHalfToFloatKernel<<<nb, blockSize>>>(
        numel,
        reinterpret_cast<const half *>(getDataPtrCuda<Float16>(input)),
        getDataPtrCuda<float>(out));
  } else if (from == DType::kFloat && to == DType::kFloat16) {
    castFloatToHalfKernel<<<nb, blockSize>>>(
        numel,
        getDataPtrCuda<float>(input),
        reinterpret_cast<half *>(getDataPtrCuda<Float16>(out)));
  } else if (from == DType::kFloat && to == DType::kLong) {
    castToLongKernel<float>
        <<<nb, blockSize>>>(numel, getDataPtrCuda<float>(input), getDataPtrCuda<LongType>(out));
  } else if (from == DType::kFloat16 && to == DType::kLong) {
    castToLongKernel<half><<<nb, blockSize>>>(
        numel,
        reinterpret_cast<const half *>(getDataPtrCuda<Float16>(input)),
        getDataPtrCuda<LongType>(out));
  } else {
    NOT_IMPL();
  }
  LL_CUDA_SYNCHRONIZE();
  LL_CHECK_CUDA_STATUS(cudaGetLastError());
}

Tensor castTo(const TensorView &input, DType dtype) {
  Tensor out = F::empty(input.getDevice(), input.getShape(), dtype);
  cast(input, out);
  return out;
}

}  // namespace cuda
}  // namespace op
}  // namespace fl
