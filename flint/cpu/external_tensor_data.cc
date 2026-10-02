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

#include "flint/cpu/external_tensor_data.h"

#include "lutil/log.h"

namespace fl {
namespace op {
namespace cpu {

ExternalTensorData::ExternalTensorData(const void *data, int64_t numel, DType dtype)
    : _data(static_cast<const std::byte *>(data)) {
  _numel = numel;
  _dtype = dtype;
}

std::unique_ptr<TensorData> ExternalTensorData::create(const void *data, int64_t numel, DType dtype) {
  CHECK(numel > 0);
  return std::unique_ptr<TensorData>(new ExternalTensorData(data, numel, dtype));
}

Device ExternalTensorData::getDevice() const {
  return Device(Device::Type::kCpu);
}

std::byte *ExternalTensorData::getRawData() const {
  // TensorData hands out a mutable pointer to every kind of storage; this one is not to be written
  // through, which isReadOnly() says.
  return const_cast<std::byte *>(_data);
}

}  // namespace cpu
}  // namespace op
}  // namespace fl
