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

#include "flint/vulkan/vulkan_tensor_data.h"

#include "lutil/error.h"
#include "lutil/log.h"

namespace fl {
namespace op {
namespace vulkan {

std::shared_ptr<TensorData> VulkanTensorData::create(int64_t numel, DType dtype) {
  CHECK(numel > 0 && numel <= MaxNumEl);

  std::shared_ptr<VulkanTensorData> data(new VulkanTensorData());
  data->_context = Context::get();
  data->_buffer = data->_context->allocate(dtype.getTotalSize(numel));
  data->_numel = numel;
  data->_dtype = dtype;
  return data;
}

VulkanTensorData::~VulkanTensorData() {
  if (_buffer.buffer) _context->free(_buffer);
}

Device VulkanTensorData::getDevice() const {
  return Device(Device::kVulkan);
}

std::byte *VulkanTensorData::getRawData() const {
  throw lut::AbortedError(
      "a tensor on the Vulkan device has no address the host may touch; copy it to the CPU");
}

}  // namespace vulkan
}  // namespace op
}  // namespace fl
