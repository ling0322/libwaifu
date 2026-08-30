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

#include "flint/metal/to_device.h"

#include <string.h>

#include "lutil/error.h"
#include "lutil/log.h"
#include "flint/cpu/cpu_tensor_data.h"
#include "flint/metal/common.h"
#include "flint/metal/metal_tensor_data.h"

namespace fl {
namespace op {
namespace metal {

namespace {

std::shared_ptr<TensorData> createData(Device::Type device, int64_t numel, DType dtype) {
  switch (device) {
    case Device::kCpu:
      return op::cpu::CpuTensorData::create(numel, dtype);
    case Device::kMetal:
      return MetalTensorData::create(numel, dtype);
    default:
      NOT_IMPL();
  }
}

}  // namespace

Tensor toDevice(Device device, const Tensor &tensor) {
  Device::Type destType = device.getType();
  if (tensor.getDevice().getType() == destType) return tensor;

  CHECK(tensor.isContiguous()) << "only contiguous tensor is allowed to copy between devices";

  std::shared_ptr<TensorData> srcData = tensor.getInternalData();
  DType dtype = srcData->getDType();
  int64_t numel = srcData->getNumEl();

  std::shared_ptr<TensorData> destData = createData(destType, numel, dtype);

  const std::byte *src = srcData->getRawData() + dtype.getTotalSize(tensor.getInternalOffset());
  std::byte *dest = destData->getRawData();
  memcpy(dest, src, static_cast<size_t>(dtype.getTotalSize(destData->getNumEl())));

  auto shape = std::make_shared<TensorShape>(tensor.getShape());
  return Tensor::create(shape, destData);
}

Tensor toCpu(const Tensor &tensor) {
  return toDevice(Device(Device::kCpu), tensor);
}

Tensor toMetal(const Tensor &tensor) {
  return toDevice(Device(Device::kMetal), tensor);
}

}  // namespace metal
}  // namespace op
}  // namespace fl
