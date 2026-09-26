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

#include "flint/vulkan/to_device.h"

#include "lutil/error.h"
#include "lutil/log.h"
#include "lutil/strings.h"
#include "flint/cpu/cpu_tensor_data.h"
#include "flint/operators.h"
#include "flint/vulkan/common.h"
#include "flint/vulkan/ops.h"
#include "flint/vulkan/vulkan_tensor_data.h"

namespace fl {
namespace op {
namespace vulkan {

namespace {

Tensor upload(const Tensor &tensor) {
  // Host memory is packed by the CPU before it goes: a strided copy is cheaper there than as a
  // second pass on the device, and it is one upload of exactly the bytes wanted either way.
  Tensor x = tensor.isContiguous() ? tensor : getOperators(Device::kCpu)->contiguous(tensor);

  Tensor output = createTensor(x.getShape(), x.getDType());
  int64_t bytes = x.getDType().getTotalSize(x.getNumEl());
  if (bytes > 0) {
    const void *src = x.getInternalData()->getData<void>(x.getInternalOffset());
    getContext(output)->upload(src, getBuffer(output), getByteOffset(output), bytes);
  }
  return output;
}

Tensor download(const Tensor &tensor) {
  Tensor x = makeContiguous(tensor);

  auto shape = std::make_shared<TensorShape>(x.getShape());
  int64_t numel = std::max<int64_t>(shape->getNumEl(), 1);
  std::shared_ptr<TensorData> data = cpu::CpuTensorData::create(numel, x.getDType());
  int64_t bytes = x.getDType().getTotalSize(x.getNumEl());
  if (bytes > 0) {
    getContext(x)->download(getBuffer(x), getByteOffset(x), data->getData<void>(0), bytes);
  }
  return Tensor::create(shape, data);
}

}  // namespace

Tensor toDevice(Device device, const Tensor &tensor) {
  Device from = tensor.getDevice();
  if (from.getType() == device.getType()) return tensor;

  if (device.getType() == Device::kVulkan && from.isHost()) return upload(tensor);
  if (device.getType() == Device::kCpu && from.getType() == Device::kVulkan) {
    return download(tensor);
  }

  throw lut::InvalidArgError(lut::sprintf(
      "the Vulkan operators do not copy from %s to %s",
      from.getName(),
      device.getName()));
}

Tensor toCpu(const Tensor &tensor) {
  return toDevice(Device(Device::kCpu), tensor);
}

Tensor toVulkan(const Tensor &tensor) {
  return toDevice(Device(Device::kVulkan), tensor);
}

}  // namespace vulkan
}  // namespace op
}  // namespace fl
