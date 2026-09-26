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

namespace fl {
namespace op {
namespace cuda {

__global__ void castFloatToHalfKernel(int n, const float *src, half *dest) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;

  dest[idx] = __float2half(src[idx]);
}

__global__ void castHalfToFloatKernel(int n, const half *src, float *dest) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;

  dest[idx] = __half2float(src[idx]);
}

Tensor castFloatToHalf(const Tensor &tensor) {
  LL_CHECK_CONTIGUOUS(tensor);

  Tensor tgtTensor = createCudaTensorHalf(tensor.getShape());
  const float *src = getDataPtrCuda<float>(tensor);
  half *dest = (half *)getDataPtrCuda<Float16>(tgtTensor);

  int64_t numel = tensor.getNumEl();
  constexpr int blockSize = 256;
  int64_t nb = (numel + blockSize - 1) / blockSize;
  castFloatToHalfKernel<<<nb, blockSize>>>(numel, src, dest);
  LL_CUDA_SYNCHRONIZE();
  LL_CHECK_CUDA_STATUS(cudaGetLastError());

  return tgtTensor;
}

Tensor castHalfToFloat(const Tensor &tensor) {
  LL_CHECK_CONTIGUOUS(tensor);

  Tensor tgtTensor = createCudaTensorFloat(tensor.getShape());
  const half *src = (half *)getDataPtrCuda<Float16>(tensor);
  float *dest = getDataPtrCuda<float>(tgtTensor);

  int64_t numel = tensor.getNumEl();
  constexpr int blockSize = 256;
  int64_t nb = (numel + blockSize - 1) / blockSize;
  castHalfToFloatKernel<<<nb, blockSize>>>(numel, src, dest);
  LL_CUDA_SYNCHRONIZE();
  LL_CHECK_CUDA_STATUS(cudaGetLastError());

  return tgtTensor;
}

template<typename T>
__global__ void castToLongKernel(int64_t n, const T *src, LongType *dest) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= n) return;

  // Through float for half, which every half is exactly; the conversion truncates toward zero.
  dest[idx] = static_cast<LongType>(static_cast<float>(src[idx]));
}

Tensor castToLong(const Tensor &tensor) {
  LL_CHECK_CONTIGUOUS(tensor);

  Tensor tgtTensor = createCudaTensorLong(tensor.getShape());
  LongType *dest = getDataPtrCuda<LongType>(tgtTensor);

  int64_t numel = tensor.getNumEl();
  if (numel == 0) return tgtTensor;

  constexpr int blockSize = 256;
  int64_t nb = (numel + blockSize - 1) / blockSize;
  if (tensor.getDType() == DType::kFloat) {
    castToLongKernel<float><<<nb, blockSize>>>(numel, getDataPtrCuda<float>(tensor), dest);
  } else if (tensor.getDType() == DType::kFloat16) {
    castToLongKernel<half>
        <<<nb, blockSize>>>(numel, (const half *)getDataPtrCuda<Float16>(tensor), dest);
  } else {
    NOT_IMPL();
  }
  LL_CUDA_SYNCHRONIZE();
  LL_CHECK_CUDA_STATUS(cudaGetLastError());

  return tgtTensor;
}

Tensor cast(const Tensor &tensor, DType dtype) {
  if (tensor.getDType() == dtype) return tensor;
  if (tensor.getDType() == DType::kFloat16 && dtype == DType::kFloat)
    return castHalfToFloat(tensor);
  if (tensor.getDType() == DType::kFloat && dtype == DType::kFloat16)
    return castFloatToHalf(tensor);
  if ((tensor.getDType() == DType::kFloat || tensor.getDType() == DType::kFloat16) &&
      dtype == DType::kLong)
    return castToLong(tensor);

  NOT_IMPL();
}

}  // namespace cuda
}  // namespace op
}  // namespace fl
