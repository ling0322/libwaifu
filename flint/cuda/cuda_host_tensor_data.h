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

#include "lutil/span.h"
#include "flint/device.h"
#include "flint/dtype.h"
#include "flint/tensor.h"

namespace fl {
namespace op {
namespace cuda {

/// @brief Host memory that the CUDA driver page-locked, held as tensor data.
///
/// The same bytes an ordinary host allocation holds, with one property added: the pages cannot be
/// swapped out or moved, so the copy engine may read them by physical address. That is what makes
/// a copy out of here able to overlap with compute -- a copy from ordinary pageable memory has to
/// be staged through a buffer the driver owns, and it fills that buffer before it returns, which
/// is what makes such a copy synchronous however it was asked for.
///
/// Page-locking is not free to the machine: these pages stay resident whatever else needs memory,
/// and the allocation itself is slow enough to be worth doing once at load rather than per use.
class CudaHostTensorData : public TensorData {
 public:
  static std::shared_ptr<TensorData> create(int64_t numel, DType dtype);

  CudaHostTensorData();
  ~CudaHostTensorData();

  Device getDevice() const override;
  std::byte *getRawData() const override;

 private:
  void *_data;
};

/// @brief Create an uninitialized tensor in page-locked host memory.
/// @param shape the shape of the tensor.
/// @param dtype the element type.
/// @return the tensor created.
Tensor createCudaHostTensor(lut::Span<const int> shape, DType dtype);

}  // namespace cuda
}  // namespace op
}  // namespace fl
