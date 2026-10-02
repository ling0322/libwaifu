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

#pragma once

#include <cuda_runtime.h>

#include <memory>

#include "flint/cuda/future_tensor.h"
#include "flint/tensor.h"
#include "flint/tensor_view.h"

namespace fl {
namespace op {
namespace cuda {

/// @brief Copy contiguous `src` into contiguous `dest`, of the same shape and type, between any
/// two of the CPU, the CUDA device and page-locked (CUDA host) memory.
///
/// Exactly `src.getNumEl()` elements are copied, starting at each view's own offset. A pageable
/// CPU source of at least `StagedUpload::MinBytes` going to the device is sent through
/// StagedUpload on the copy stream; the compute stream waits for it before anything after the
/// call runs.
void transfer(const TensorView &src, const TensorView &dest);

/// @brief A contiguous copy of contiguous `tensor` on the CPU, or `tensor` itself if it is there.
/// For a backend that reads a result on the host.
Tensor toCpu(const Tensor &tensor);

/// @brief Start a copy from page-locked host memory to the GPU and return before it is done.
///
/// What comes back is a tensor whose memory is allocated and whose bytes are on their way. It
/// is held in a FutureTensor until take() has been called on it, which orders the
/// compute stream after the copy without stopping the host.
FutureTensor toDeviceAsync(Device device, const Tensor &tensor);

/// @brief A copy issued and not yet seen through: the storage it fills and the event that marks
/// its end.
struct PendingTransfer {
  /// Allocated in the copy stream's order, and so owed to the copy stream until the copy has been
  /// seen through by completeTransfer().
  std::unique_ptr<TensorData> dest;
  cudaEvent_t event;
};

/// @brief What toDeviceAsync() does, without wrapping the result in a FutureTensor: start copying
/// contiguous `src` from page-locked host memory to `device`, which has to be the GPU, into fresh
/// storage of exactly its elements. The caller keeps `src`'s storage alive until the copy has been
/// completed or the event destroyed.
PendingTransfer startTransferAsync(Device device, const TensorView &src);

/// @brief See a transfer through: order the compute stream after it -- or, with `sync`, stop the
/// host until it is done -- give `dest`'s memory to the compute stream, and destroy `event`.
void completeTransfer(TensorData *dest, cudaEvent_t event, bool sync);

}  // namespace cuda
}  // namespace op
}  // namespace fl
