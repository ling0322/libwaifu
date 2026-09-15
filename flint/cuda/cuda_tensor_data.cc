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

#include "flint/cuda/cuda_tensor_data.h"

#include <cuda_runtime.h>

#include "lutil/error.h"
#include "lutil/platform.h"
#include "lutil/span.h"
#include "lutil/strings.h"
#include "flint/cuda/common.h"
#include "flint/device.h"
#include "flint/dtype.h"

namespace fl {
namespace op {
namespace cuda {

std::shared_ptr<TensorData> CudaTensorData::create(
    int64_t numel,
    DType dtype,
    cudaStream_t stream) {
  auto tensorData = std::make_shared<CudaTensorData>();

  CHECK(numel > 0);
  int64_t size = dtype.getTotalSize(numel);
  void *data = nullptr;
  cudaError_t err = llynCudaMalloc(&data, size, stream);
  if (err != cudaSuccess) {
    throw lut::AbortedError(cudaGetErrorString(err));
  }

  tensorData->_data = data;
  data = nullptr;

  tensorData->_stream = stream;
  tensorData->_numel = numel;
  tensorData->_dtype = dtype;

  return tensorData;
}

CudaTensorData::CudaTensorData()
    : _data(nullptr),
      _stream(nullptr) {
}

CudaTensorData::~CudaTensorData() {
  if (_data) {
    // Given back in the order of the stream that last owned these bytes, which `_stream` tracks
    // for exactly this. Neither case needs the host to wait: a fetch nobody wanted is freed in
    // the copy stream, where the copy that may still be writing it runs, and one that was taken
    // is freed in the compute stream, which is ordered after that copy.
    llynCudaFree(_data, _stream);
    _data = nullptr;
  }
}

void CudaTensorData::setOwningStream(cudaStream_t stream) {
  _stream = stream;
}

Device CudaTensorData::getDevice() const {
  return Device(Device::Type::kCuda);
}

std::byte *CudaTensorData::getRawData() const {
  return reinterpret_cast<std::byte *>(_data);
}

}  // namespace cuda
}  // namespace op
}  // namespace fl
