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

namespace fl {
namespace op {
namespace metal {

void transfer(const TensorView &src, const TensorView &dest) {
  Device::Type from = src.getDevice().getType();
  Device::Type to = dest.getDevice().getType();
  CHECK((from == Device::kCpu && to == Device::kMetal) ||
        (from == Device::kMetal && to == Device::kCpu))
      << "transfer: Metal moves between the CPU and itself only, not from "
      << src.getDevice().getName() << " to " << dest.getDevice().getName();
  CHECK(src.isContiguous() && dest.isContiguous())
      << "only contiguous tensor is allowed to copy between devices";
  CHECK(src.getDType() == dest.getDType()) << "transfer: dtype mismatch";
  CHECK(src.getNumEl() == dest.getNumEl()) << "transfer: the two sides differ in size";

  // The tensor's own elements, from its own offset -- not the whole storage it sits in, which is
  // what this used to copy, reading past the end of a view of part of a buffer.
  DType dtype = src.getDType();
  const std::byte *source =
      src.getInternalData()->getRawData() + dtype.getTotalSize(src.getInternalOffset());
  std::byte *target =
      dest.getInternalData()->getRawData() + dtype.getTotalSize(dest.getInternalOffset());
  memcpy(target, source, static_cast<size_t>(dtype.getTotalSize(src.getNumEl())));
}

}  // namespace metal
}  // namespace op
}  // namespace fl
