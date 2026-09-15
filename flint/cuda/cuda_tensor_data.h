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

#include <initializer_list>

#include "lutil/span.h"
#include "flint/device.h"
#include "flint/tensor.h"

namespace fl {
namespace op {
namespace cuda {

class CudaTensorData : public TensorData {
 public:
  /// @brief Allocate `numel` elements, in `stream`'s order.
  ///
  /// The stream matters because the allocator is stream-ordered: the block may still belong to
  /// work enqueued before it was freed, and it becomes this tensor's only where `stream` has
  /// reached the call. Pass the stream that will write it first. It is remembered, because
  /// giving the block back has the same requirement -- see `_stream`.
  static std::shared_ptr<TensorData> create(int64_t numel, DType dtype, cudaStream_t stream = 0);

  CudaTensorData();
  ~CudaTensorData();

  Device getDevice() const override;

  /// @brief The pointer to the memory. Whether the bytes in it are worth reading is not this
  /// class's to answer: a block still being filled by an asynchronous copy is held inside a
  /// FutureTensor until that copy has been seen through, and no Tensor reaches an operator before
  /// then.
  std::byte *getRawData() const override;

  /// @brief Say which stream owns these bytes from now on, which is the stream they will be
  /// given back in. FutureTensor calls this when it hands the tensor over, because that is the
  /// moment the compute stream becomes ordered after the copy and takes ownership from it.
  void setOwningStream(cudaStream_t stream);

 private:
  void *_data;

  /// The stream `_data` is given back in, which the stream-ordered allocator reads as "reusable
  /// once this stream reaches here". It has to be a stream the last use of these bytes was on, or
  /// the block goes to the next allocation while that use is still running and nothing says so.
  ///
  /// It starts as the stream the block was allocated in, which is also the stream that writes it
  /// first: zero for an ordinary tensor, the copy stream for one being fetched. It stays with the
  /// memory rather than with the pending copy because it outlives the copy -- the tensor handed
  /// out by FutureTensor::take() is freed long after, and by then this is the only record of
  /// where the block came from.
  cudaStream_t _stream;
};

}  // namespace cuda
}  // namespace op
}  // namespace fl
