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

#include "flint/cuda/future_tensor.h"

#include <utility>

#include "flint/cuda/common.h"
#include "flint/cuda/cuda_tensor_data.h"

namespace fl {

FutureTensor::FutureTensor(Tensor tensor, cudaEvent_t event, std::shared_ptr<TensorData> source)
    : _tensor(std::move(tensor)),
      _event(event),
      _source(std::move(source)) {
}

FutureTensor::FutureTensor(FutureTensor &&other) noexcept
    : _tensor(std::move(other._tensor)),
      _event(other._event),
      _source(std::move(other._source)) {
  other._event = nullptr;
}

FutureTensor &FutureTensor::operator=(FutureTensor &&other) noexcept {
  if (this != &other) {
    if (_event) cudaEventDestroy(_event);
    _tensor = std::move(other._tensor);
    _event = other._event;
    _source = std::move(other._source);
    other._event = nullptr;
  }
  return *this;
}

FutureTensor::~FutureTensor() {
  if (_event) {
    // A fetch nobody wanted. Unchecked because a destructor is no place to throw, and destroying
    // an event that is still pending is allowed: the call returns at once and the event is
    // reclaimed once the device has reached it. The memory the copy is writing goes back in the
    // copy stream's order, which the tensor data has recorded, so it cannot be handed to the next
    // allocation while that copy is still running.
    cudaEventDestroy(_event);
    _event = nullptr;
  }
}

Tensor FutureTensor::take() {
  if (!_event) return _tensor;

  // The compute stream waits, not the host. The dependency is what the reader needs; making the
  // host stand here as well would give up exactly the overlap the copy was issued early for.
  LL_CHECK_CUDA_STATUS(cudaStreamWaitEvent(0, _event, 0));
  return finish();
}

Tensor FutureTensor::takeSync() {
  if (!_event) return _tensor;

  // Waiting on the event itself rather than on the stream: the copy is what these bytes are owed,
  // and whatever the copy stream picked up after it is somebody else's fetch.
  LL_CHECK_CUDA_STATUS(cudaEventSynchronize(_event));
  return finish();
}

Tensor FutureTensor::finish() {
  // The compute stream is ordered after the copy now, so it is what owns these bytes and the
  // stream they have to be given back in. Done before the event is dropped so that a throw
  // anywhere leaves the copy stream as the answer, which is the safe one.
  auto *data = static_cast<op::cuda::CudaTensorData *>(_tensor.getInternalData().get());
  data->setOwningStream(0);

  // The wait above took the event's state as it stood, so destroying it here does not undo the
  // dependency that was arranged.
  LL_CHECK_CUDA_STATUS(cudaEventDestroy(_event));
  _event = nullptr;

  // The copy has been seen through, so the source has done its part and need not be kept alive by
  // this any longer. Note that dropping the last reference to page-locked memory drains the whole
  // device inside cudaFreeHost, so whoever wants the overlap holds its own reference to it.
  _source.reset();

  return _tensor;
}

}  // namespace fl
