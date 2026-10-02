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
#include "flint/functional.h"
#include "flint/vulkan/common.h"
#include "flint/vulkan/ops.h"

namespace fl {
namespace op {
namespace vulkan {

void transfer(const TensorView &src, const TensorView &dest) {
  if (src.getDType() != dest.getDType()) {
    throw lut::InvalidArgError("transfer: the source and the destination differ in dtype");
  }
  src.throwIfInvalidShape(dest.getShape(), "transfer");
  if (!src.isContiguous() || !dest.isContiguous()) {
    throw lut::InvalidArgError("transfer: the source and the destination must be contiguous");
  }

  // Exactly the elements of `src`, from where it starts: never the rest of its storage.
  int64_t bytes = src.getDType().getTotalSize(src.getNumEl());
  Device from = src.getDevice();
  Device to = dest.getDevice();
  if (to.getType() == Device::kVulkan && from.isHost()) {
    if (bytes == 0) return;
    const void *data = src.getInternalData()->getData<void>(src.getInternalOffset());
    getContext(dest)->upload(data, getBuffer(dest), getByteOffset(dest), bytes);
    return;
  }
  if (from.getType() == Device::kVulkan && to.isHost()) {
    if (bytes == 0) return;
    void *data = dest.getInternalData()->getData<void>(dest.getInternalOffset());
    getContext(src)->download(getBuffer(src), getByteOffset(src), data, bytes);
    return;
  }

  throw lut::InvalidArgError(lut::sprintf(
      "the Vulkan operators do not copy from %s to %s",
      from.getName(),
      to.getName()));
}

Tensor toCpu(const TensorView &tensor) {
  Tensor keep;
  TensorView x = makeContiguous(tensor, &keep);
  Tensor output = F::empty(Device::getCpu(), x.getShape(), x.getDType());
  transfer(x, output);
  return output;
}

}  // namespace vulkan
}  // namespace op
}  // namespace fl
