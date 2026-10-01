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

ExternalTensorData::ExternalTensorData(
    const void *data,
    int64_t numel,
    DType dtype,
    ReleaseFn release,
    void *context)
    : _data(static_cast<const std::byte *>(data)),
      _release(release),
      _context(context) {
  _numel = numel;
  _dtype = dtype;
}

std::shared_ptr<TensorData> ExternalTensorData::create(
    const void *data,
    int64_t numel,
    DType dtype,
    ReleaseFn release,
    void *context) {
  CHECK(numel > 0);

  // Built without the release and handed it only once the shared_ptr owns the object: if making
  // the control block throws, the object is destroyed on the way out, and it must not give back
  // what the caller still thinks is the caller's.
  std::shared_ptr<ExternalTensorData> tensorData(
      new ExternalTensorData(data, numel, dtype, nullptr, nullptr));
  tensorData->_release = release;
  tensorData->_context = context;
  return tensorData;
}

ExternalTensorData::~ExternalTensorData() {
  if (_release) _release(_context);
  _release = nullptr;
}

Device ExternalTensorData::getDevice() const {
  return Device(Device::Type::kCpu);
}

std::byte *ExternalTensorData::getRawData() const {
  // TensorData hands out a mutable pointer to every kind of storage; this one is not to be written
  // through, which isReadOnly() says and fl_tensor_host_data() enforces.
  return const_cast<std::byte *>(_data);
}

}  // namespace cpu
}  // namespace op
}  // namespace fl
