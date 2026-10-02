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

#include "flint/cuda/common.h"

#include <cuda_fp16.h>

#include "flint/cuda/cuda_tensor_data.h"
#include "flint/tensor.h"

namespace fl {
namespace op {
namespace cuda {

int gMaxThreadsPerSM = 0;
int gNumSM = 0;

Tensor createCudaTensor(lut::Span<const int> shape, DType dtype) {
  auto tensorShape = std::make_shared<TensorShape>(shape);
  auto data = CudaTensorData::create(tensorShape->getNumEl(), dtype);

  return Tensor::create(tensorShape, std::move(data));
}

template<typename T>
float elemImpl(const TensorView &tensor) {
  T v;
  LL_CHECK_CUDA_STATUS(
      cudaMemcpy(&v, getDataPtrCuda<T>(tensor), sizeof(T), cudaMemcpyDeviceToHost));
  return v;
}
float elem(const TensorView &tensor) {
  CHECK(tensor.getDim() == 1 && tensor.getShape(0) == 1);

  if (tensor.getDType() == DType::kFloat16) return elemImpl<half>(tensor);
  if (tensor.getDType() == DType::kFloat) return elemImpl<float>(tensor);

  NOT_IMPL();
}

bool elemBool(const TensorView &tensor) {
  CHECK(tensor.getDim() == 1 && tensor.getShape(0) == 1);
  CHECK(tensor.getDType() == DType::kBool);

  bool v;
  LL_CHECK_CUDA_STATUS(
      cudaMemcpy(&v, getDataPtrCuda<BoolType>(tensor), sizeof(BoolType), cudaMemcpyDeviceToHost));

  return v;
}

int getCudaDeviceAttribute(cudaDeviceAttr attr) {
  int value;
  LL_CHECK_CUDA_STATUS(cudaDeviceGetAttribute(&value, attr, 0));
  return value;
}

int getCudaDeviceCount() {
  int value;

  cudaError_t status = cudaGetDeviceCount(&value);
  if (status != cudaSuccess) {
    return 0;
  }

  return value;
}

dim3 getGrid1D(int numel, int blockSize) {
  if (!gMaxThreadsPerSM) {
    gMaxThreadsPerSM = getCudaDeviceAttribute(cudaDevAttrMaxThreadsPerMultiProcessor);
  }
  if (!gNumSM) {
    gNumSM = getCudaDeviceAttribute(cudaDevAttrMultiProcessorCount);
  }

  int maxThreadsPerSM = gMaxThreadsPerSM;
  int numSM = gNumSM;
  int numBlock = (numel + blockSize - 1) / blockSize;
  int deviceNumBlock = maxThreadsPerSM * numSM / blockSize;
  CHECK(deviceNumBlock > 0);

  dim3 grid(std::min(numBlock, deviceNumBlock));

  return grid;
}

int getCudaArch() {
  // TODO: too slow
  cudaDeviceProp prop;
  LL_CHECK_CUDA_STATUS(cudaGetDeviceProperties(&prop, 0));
  return prop.major * 10 + prop.minor;
}

}  // namespace cuda
}  // namespace op
}  // namespace fl
