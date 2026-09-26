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
#include "flint/tensor.h"
#include "flint/vulkan/context.h"

namespace fl {
namespace op {
namespace vulkan {

/// Tensor storage on the Vulkan device: one device buffer.
///
/// Unlike Metal's, this memory is not something the host can address. getRawData() says so
/// rather than returning a pointer; the bytes come and go through toDevice(), which stages them.
class VulkanTensorData : public TensorData {
 public:
  static std::shared_ptr<TensorData> create(int64_t numel, DType dtype);

  ~VulkanTensorData();

  Device getDevice() const override;
  std::byte *getRawData() const override;

  const Buffer &getBuffer() const {
    return _buffer;
  }

  Context *getContext() const {
    return _context.get();
  }

 private:
  // Held so that the device outlives every tensor on it, whatever order they go in.
  std::shared_ptr<Context> _context;
  Buffer _buffer;

  VulkanTensorData() = default;
};

}  // namespace vulkan
}  // namespace op
}  // namespace fl
