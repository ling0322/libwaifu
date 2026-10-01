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

#include <memory>

#include "flint/device.h"
#include "flint/dtype.h"
#include "flint/tensor.h"

namespace fl {
namespace op {
namespace cpu {

/// @brief CPU storage that is somebody else's memory: the bytes are not copied in, and are given
/// back through `release` when the last tensor on them goes.
///
/// What it exists for is a weight file mapped into memory. The mapping is the storage, so reading
/// a package copies nothing, and the bytes are the page cache's -- shared with every other process
/// that has the file open, and dropped and read again from the file when memory runs short rather
/// than written out to swap.
///
/// Read-only, because a mapping of a file opened for reading is: a write through it is a fault.
/// isReadOnly() says so, and the one entry point that hands out a writable pointer refuses it.
class ExternalTensorData : public TensorData {
 public:
  typedef void (*ReleaseFn)(void *context);

  /// @brief Wrap `numel` elements of `dtype` at `data`. `release(context)` is called exactly once,
  /// when the last reference goes, and not at all if this throws.
  static std::shared_ptr<TensorData> create(
      const void *data,
      int64_t numel,
      DType dtype,
      ReleaseFn release,
      void *context);

  ExternalTensorData(const ExternalTensorData &) = delete;
  ExternalTensorData &operator=(const ExternalTensorData &) = delete;
  ~ExternalTensorData();

  Device getDevice() const override;
  std::byte *getRawData() const override;
  bool isReadOnly() const override {
    return true;
  }

 private:
  const std::byte *_data;
  ReleaseFn _release;
  void *_context;

  ExternalTensorData(const void *data, int64_t numel, DType dtype, ReleaseFn release, void *context);
};

}  // namespace cpu
}  // namespace op
}  // namespace fl
