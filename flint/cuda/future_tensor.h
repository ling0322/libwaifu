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

#pragma once

#include <cuda_runtime.h>

#include <memory>

#include "flint/tensor.h"

namespace fl {

/// @brief A tensor a copy is still filling, and everything that copy is owed.
///
/// The event that marks the end of the copy and the source it reads live here rather than in the
/// tensor data, because they belong to the copy and not to the memory: once the copy has been
/// seen through, neither has anything left to say, while the memory goes on. Holding them here
/// also means an ordinary CUDA tensor carries no trace of a mechanism it never uses.
///
/// F::toDeviceAsync() returns one of these rather than a Tensor, which is what keeps a tensor
/// that is not readable yet out of every operator: the tensor inside can only be had through
/// take(), and taking it is what discharges the obligation. Nothing else hands one out, so the
/// case an operator would have to check for cannot reach it.
///
/// Move-only, because it stands for one pending copy rather than for a value. One destroyed
/// without being taken is a fetch that turned out not to be wanted: the event is dropped, the
/// memory goes back in the copy stream's order, and the source is held until then. That costs
/// the bandwidth already spent and nothing else.
class FutureTensor {
 public:
  /// @brief Take charge of a copy already issued: `tensor` is the memory it fills, `event` marks
  /// where it ends, and `source` is what it reads, held so the copy engine cannot lose it.
  FutureTensor(Tensor tensor, cudaEvent_t event, std::shared_ptr<TensorData> source);

  FutureTensor(FutureTensor &&other) noexcept;
  FutureTensor &operator=(FutureTensor &&other) noexcept;
  FutureTensor(const FutureTensor &) = delete;
  FutureTensor &operator=(const FutureTensor &) = delete;

  ~FutureTensor();

  /// @brief Order the work that follows behind the copy, and hand the tensor over.
  ///
  /// The caller does not stop here. What is arranged is a dependency the compute stream carries
  /// from this point on, which is what an operator about to be enqueued needs and all it needs.
  /// Call it where the tensor is about to be used, not where the copy was started: the dependency
  /// lands at the stream position this call is made from, so taking early makes everything queued
  /// afterwards wait, which is the overlap the fetch was started early to buy.
  Tensor take();

  /// @brief The same, except that it does not return until the copy has finished.
  ///
  /// For a caller that is about to read the bytes with the CPU rather than enqueue work that
  /// reads them. Nothing in the library needs it today, because every host read of device memory
  /// goes through a synchronous cudaMemcpy that stops the host by itself.
  Tensor takeSync();

 private:
  /// Clear the pending state, having arranged whatever the caller asked for, and give the memory
  /// to the compute stream. Called by both takes and by nothing else.
  Tensor finish();

  Tensor _tensor;

  /// Non-null until taken. Also what says this future still holds something, which is what the
  /// destructor and the move operations read.
  cudaEvent_t _event;

  /// What the copy reads. Held so that it cannot be freed while the copy engine is reading it.
  std::shared_ptr<TensorData> _source;
};

}  // namespace fl
