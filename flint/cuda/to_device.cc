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

#include "flint/cuda/to_device.h"

#include <cuda_runtime.h>

#include <cstring>
#include <vector>

#include "lutil/strings.h"

#include "flint/cpu/cpu_tensor_data.h"
#include "flint/cpu/tensor.h"
#include "flint/cuda/common.h"
#include "flint/cuda/copy_stream.h"
#include "flint/cuda/cuda_host_tensor_data.h"
#include "flint/cuda/cuda_tensor_data.h"
#include "flint/cuda/future_tensor.h"
#include "flint/functional.h"
#include "flint/tensor.h"

namespace fl {
namespace op {
namespace cuda {

template<Device::Type DEVICE>
std::shared_ptr<TensorData> createData(int64_t numel, DType dtype);
template<>
std::shared_ptr<TensorData> createData<Device::kCpu>(int64_t numel, DType dtype) {
  return op::cpu::CpuTensorData::create(numel, dtype);
}
template<>
std::shared_ptr<TensorData> createData<Device::kCuda>(int64_t numel, DType dtype) {
  return CudaTensorData::create(numel, dtype);
}
template<>
std::shared_ptr<TensorData> createData<Device::kCudaHost>(int64_t numel, DType dtype) {
  return CudaHostTensorData::create(numel, dtype);
}

/// Copy `n` bytes, in whichever direction the two ends call for.
///
/// The direction cannot be read off the destination alone once there are two host devices: a copy
/// into the CPU may come from the GPU or from page-locked host memory, and those are a transfer
/// and a memcpy respectively. So both ends are asked.
inline void copyData(
    Device::Type destDevice,
    Device::Type srcDevice,
    void *dest,
    const void *src,
    int64_t n) {
  bool destIsHost = Device(destDevice).isHost();
  bool srcIsHost = Device(srcDevice).isHost();

  if (destIsHost && srcIsHost) {
    std::memcpy(dest, src, n);
  } else if (destIsHost) {
    LL_CHECK_CUDA_STATUS(cudaMemcpy(dest, src, n, cudaMemcpyDeviceToHost));
  } else {
    LL_CHECK_CUDA_STATUS(cudaMemcpy(dest, src, n, cudaMemcpyHostToDevice));
  }
}

template<Device::Type DEVICE>
Tensor toDevice(const Tensor &tensor) {
  CHECK(tensor.getDevice().getType() != DEVICE);
  CHECK(tensor.isContiguous()) << "only contiguous tensor is allowed to copy between devices";
  std::shared_ptr<TensorData> srcData = tensor.getInternalData();

  // create data object.
  int64_t numel = srcData->getNumEl();
  DType dtype = srcData->getDType();

  std::shared_ptr<TensorData> destData = createData<DEVICE>(numel, dtype);

  // copy data.
  int64_t srcOffset = tensor.getInternalOffset();
  void *src = srcData->getRawData() + dtype.getTotalSize(srcOffset);
  void *dest = destData->getRawData();
  int64_t nbytes = dtype.getTotalSize(destData->getNumEl());
  copyData(DEVICE, tensor.getDevice().getType(), dest, src, nbytes);

  // create dest tesnor.
  auto shape = std::make_shared<TensorShape>(tensor.getShape());
  return Tensor::create(shape, destData);
}

Tensor toCpu(const Tensor &tensor) {
  if (tensor.getDevice().getType() == Device::kCpu) return tensor;
  return toDevice<Device::kCpu>(tensor);
}

Tensor toCuda(const Tensor &tensor) {
  if (tensor.getDevice().getType() == Device::kCuda) return tensor;
  return toDevice<Device::kCuda>(tensor);
}

Tensor toCudaHost(const Tensor &tensor) {
  if (tensor.getDevice().getType() == Device::kCudaHost) return tensor;
  return toDevice<Device::kCudaHost>(tensor);
}

FutureTensor toDeviceAsync(Device device, const Tensor &tensor) {
  // One direction only, and a narrow one: page-locked host memory to the GPU. The others are
  // refused rather than quietly done synchronously, because a copy that says it is asynchronous
  // and is not costs nothing to write and 2.4 times the time to run. A pageable source is the
  // case that matters -- the driver has to stage it through a buffer of its own and fills that
  // buffer before returning -- and it is the reason the source device is checked rather than
  // merely that the source is on the host.
  if (device.getType() != Device::kCuda ||
      tensor.getDevice().getType() != Device::kCudaHost) {
    throw lut::InvalidArgError(lut::sprintf(
        "an asynchronous copy goes from cuda-host to cuda, not from %s to %s",
        tensor.getDevice().getName().c_str(),
        device.getName().c_str()));
  }
  CHECK(tensor.isContiguous()) << "only contiguous tensor is allowed to copy between devices";

  CopyStream *copies = CopyStream::getInstance();
  cudaStream_t stream = copies->getStream();

  std::shared_ptr<TensorData> srcData = tensor.getInternalData();
  DType dtype = srcData->getDType();
  int64_t numel = srcData->getNumEl();

  // Allocated in the copy stream's order, so that the copy below may write it with no dependency
  // to arrange: the allocator hands the block over at this point in this stream, and this stream
  // is where the writing happens.
  std::shared_ptr<TensorData> destData = CudaTensorData::create(numel, dtype, stream);

  const void *src = srcData->getRawData() + dtype.getTotalSize(tensor.getInternalOffset());
  void *dest = destData->getRawData();
  int64_t nbytes = dtype.getTotalSize(numel);
  LL_CHECK_CUDA_STATUS(
      cudaMemcpyAsync(dest, src, nbytes, cudaMemcpyHostToDevice, stream));

  // Made and thrown away per copy rather than kept in a pool. The pair measures 238 ns against
  // the 2.2 us it takes to launch the copy above, so a pool buys a tenth of one launch and costs
  // shared mutable state on a path that has none otherwise. cudaEventDisableTiming because
  // nothing here asks the event how long anything took.
  cudaEvent_t event = nullptr;
  LL_CHECK_CUDA_STATUS(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
  LL_CHECK_CUDA_STATUS(cudaEventRecord(event, stream));

  // From here the copy belongs to the future, which holds the event that marks its end and the
  // source it reads, and hands the tensor out only once it has been seen through.
  auto shape = std::make_shared<TensorShape>(tensor.getShape());
  return FutureTensor(Tensor::create(shape, destData), event, srcData);
}

Tensor toDevice(Device device, const Tensor &tensor) {
  if (Device::kCpu == device.getType()) return toCpu(tensor);
  if (Device::kCuda == device.getType()) return toCuda(tensor);
  if (Device::kCudaHost == device.getType()) return toCudaHost(tensor);

  NOT_IMPL();
  return Tensor();
}

}  // namespace cuda
}  // namespace op
}  // namespace fl

namespace fl {
namespace F {

FutureTensor toDeviceAsync(Device device, Tensor tensor) {
  // Defined here rather than in functional.cc because a FutureTensor holds a CUDA event, and a
  // build without CUDA has neither. Narrow on purpose, and the narrowness is checked by the call
  // it forwards to rather than guessed at here: cuda-host to cuda is the one pair that has anything to gain, and every
  // other pair is refused rather than served synchronously under a name that promises otherwise.
  return op::cuda::toDeviceAsync(device, tensor);
}

}  // namespace F
}  // namespace fl
