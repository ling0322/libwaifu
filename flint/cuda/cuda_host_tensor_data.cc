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

#include "flint/cuda/cuda_host_tensor_data.h"

#include <cuda_runtime.h>

#include "lutil/error.h"
#include "lutil/log.h"
#include "lutil/strings.h"
#include "flint/cuda/common.h"
#include "flint/device.h"
#include "flint/dtype.h"

namespace fl {
namespace op {
namespace cuda {

std::shared_ptr<TensorData> CudaHostTensorData::create(int64_t numel, DType dtype) {
  auto tensorData = std::make_shared<CudaHostTensorData>();

  CHECK(numel > 0);
  int64_t size = dtype.getTotalSize(numel);
  void *data = nullptr;

  // cudaHostAllocDefault rather than WriteCombined: write-combining would speed the copy engine's
  // read a little and make the CPU's read of the same memory an order of magnitude slower, and
  // this memory is meant to stay readable by both.
  cudaError_t err = cudaHostAlloc(&data, size, cudaHostAllocDefault);
  if (err != cudaSuccess) {
    throw lut::AbortedError(lut::sprintf(
        "could not page-lock %lld bytes of host memory: %s",
        static_cast<long long>(size),
        cudaGetErrorString(err)));
  }

  tensorData->_data = data;
  tensorData->_numel = numel;
  tensorData->_dtype = dtype;

  return tensorData;
}

CudaHostTensorData::CudaHostTensorData()
    : _data(nullptr) {
}

CudaHostTensorData::~CudaHostTensorData() {
  if (_data) {
    cudaError_t err = cudaFreeHost(_data);
    if (err != cudaSuccess) {
      LOG(ERROR) << "Error while freeing page-locked host memory: " << cudaGetErrorString(err);
    }
    _data = nullptr;
  }
}

Device CudaHostTensorData::getDevice() const {
  return Device(Device::Type::kCudaHost);
}

std::byte *CudaHostTensorData::getRawData() const {
  return reinterpret_cast<std::byte *>(_data);
}

Tensor createCudaHostTensor(lut::Span<const int> shape, DType dtype) {
  auto tensorShape = std::make_shared<TensorShape>(shape);
  auto data = CudaHostTensorData::create(tensorShape->getNumEl(), dtype);

  return Tensor::create(tensorShape, data);
}

}  // namespace cuda
}  // namespace op
}  // namespace fl
