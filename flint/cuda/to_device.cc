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

#include <algorithm>
#include <cstring>
#include <vector>

#include "lutil/strings.h"

#include "flint/cuda/common.h"
#include "flint/cuda/copy_stream.h"
#include "flint/cuda/cuda_tensor_data.h"
#include "flint/cuda/future_tensor.h"
#include "flint/cuda/staged_upload.h"
#include "flint/functional.h"
#include "flint/tensor.h"

namespace fl {
namespace op {
namespace cuda {

namespace {

/// Copy `n` bytes, in whichever direction the two ends call for.
///
/// The direction cannot be read off the destination alone once there are two host devices: a copy
/// into the CPU may come from the GPU or from page-locked host memory, and those are a transfer
/// and a memcpy respectively. So both ends are asked.
void copyData(
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
  } else if (srcIsHost) {
    LL_CHECK_CUDA_STATUS(cudaMemcpy(dest, src, n, cudaMemcpyHostToDevice));
  } else {
    LL_CHECK_CUDA_STATUS(cudaMemcpy(dest, src, n, cudaMemcpyDeviceToDevice));
  }
}

/// Pageable host memory to the GPU through StagedUpload, on the copy stream.
///
/// The copy stream rather than the compute stream, and that is most of what this buys beside the
/// rate. A cudaMemcpy on the legacy stream waits for every kernel queued before it, so the host
/// stands still while the GPU drains and the bus and the GPU take turns. Here the host only waits
/// for the staging buffers, and the kernels already queued go on running while the next weight
/// crosses. The compute stream is made to wait for the copy instead, which is the one ordering the
/// reader needs.
///
/// `dest` was allocated in the compute stream's order -- `F::empty` allocates on the legacy
/// stream -- so the block is only ours where stream 0 has reached that allocation. The copy stream
/// is non-blocking and would otherwise write it while whatever last owned the block may still be
/// running, so it waits on an event recorded on stream 0 first. The block stays stream 0's to
/// give back, which is right: stream 0 waits for the copy below before anything else touches it.
void transferStaged(void *dest, const void *src, int64_t nbytes) {
  cudaStream_t stream = CopyStream::getInstance()->getStream();

  cudaEvent_t allocated = nullptr;
  LL_CHECK_CUDA_STATUS(cudaEventCreateWithFlags(&allocated, cudaEventDisableTiming));
  LL_CHECK_CUDA_STATUS(cudaEventRecord(allocated, 0));
  LL_CHECK_CUDA_STATUS(cudaStreamWaitEvent(stream, allocated, 0));
  LL_CHECK_CUDA_STATUS(cudaEventDestroy(allocated));

  StagedUpload::getInstance()->copy(dest, src, nbytes, stream);

  cudaEvent_t copied = nullptr;
  LL_CHECK_CUDA_STATUS(cudaEventCreateWithFlags(&copied, cudaEventDisableTiming));
  LL_CHECK_CUDA_STATUS(cudaEventRecord(copied, stream));
  LL_CHECK_CUDA_STATUS(cudaStreamWaitEvent(0, copied, 0));
  LL_CHECK_CUDA_STATUS(cudaEventDestroy(copied));
}

bool isCudaSide(Device::Type type) {
  return type == Device::kCpu || type == Device::kCuda || type == Device::kCudaHost;
}

}  // namespace

void transfer(const TensorView &src, const TensorView &dest) {
  Device::Type srcType = src.getDevice().getType();
  Device::Type destType = dest.getDevice().getType();
  CHECK(isCudaSide(srcType) && isCudaSide(destType));
  CHECK(srcType != destType);
  CHECK(srcType != Device::kCpu || destType != Device::kCpu);
  CHECK(src.getDType() == dest.getDType());
  CHECK(src.isContiguous() && dest.isContiguous())
      << "only contiguous tensor is allowed to copy between devices";
  dest.throwIfInvalidShape(src.getShape(), "transfer");

  DType dtype = src.getDType();
  int64_t nbytes = dtype.getTotalSize(src.getNumEl());
  if (nbytes == 0) return;

  const void *from = src.getInternalData()->getRawData() +
                     dtype.getTotalSize(src.getInternalOffset());
  void *to = dest.getInternalData()->getRawData() + dtype.getTotalSize(dest.getInternalOffset());

  if (srcType == Device::kCpu && destType == Device::kCuda && nbytes >= StagedUpload::MinBytes) {
    transferStaged(to, from, nbytes);
  } else {
    copyData(destType, srcType, to, from, nbytes);
  }
}

Tensor toCpu(const Tensor &tensor) {
  if (tensor.getDevice().getType() == Device::kCpu) return tensor;

  Tensor dest = F::empty(Device::getCpu(), tensor.getShape(), tensor.getDType());
  transfer(tensor, dest);
  return dest;
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

  // The tensor's own elements, not its whole storage: a view of part of a larger page-locked
  // block copies only what it sees.
  int64_t numel = tensor.getNumEl();

  // Allocated in the copy stream's order, so that the copy below may write it with no dependency
  // to arrange: the allocator hands the block over at this point in this stream, and this stream
  // is where the writing happens.
  std::shared_ptr<TensorData> destData =
      CudaTensorData::create(std::max<int64_t>(numel, 1), dtype, stream);

  const void *src = srcData->getRawData() + dtype.getTotalSize(tensor.getInternalOffset());
  void *dest = destData->getRawData();
  int64_t nbytes = dtype.getTotalSize(numel);
  if (nbytes > 0) {
    LL_CHECK_CUDA_STATUS(cudaMemcpyAsync(dest, src, nbytes, cudaMemcpyHostToDevice, stream));
  }

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

}  // namespace cuda
}  // namespace op
}  // namespace fl
