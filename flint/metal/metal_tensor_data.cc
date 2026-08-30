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

#include "flint/metal/metal_tensor_data.h"

#include "lutil/error.h"
#include "lutil/log.h"
#include "flint/metal/common.h"

namespace fl {
namespace op {
namespace metal {

MetalTensorData::MetalTensorData(mlx::core::array array)
    : _array(std::move(array)) {
  _numel = _array.size();
  _dtype = fromMlxDtype(_array.dtype());
}

std::shared_ptr<TensorData> MetalTensorData::create(int64_t numel, DType dtype) {
  CHECK(numel > 0);

  mlx::core::array array = mlx::core::zeros({static_cast<int>(numel)}, toMlxDtype(dtype));
  mlx::core::eval(array);

  return wrap(std::move(array));
}

std::shared_ptr<TensorData> MetalTensorData::wrap(mlx::core::array array) {
  CHECK(array.ndim() == 1);
  CHECK(array.flags().contiguous);

  // getRawData() is called without warning by anything holding the tensor, and data() on an
  // array that was never evaluated is undefined, so pay the eval here rather than leave a
  // landmine for the caller.
  mlx::core::eval(array);

  return std::shared_ptr<MetalTensorData>(new MetalTensorData(std::move(array)));
}

Device MetalTensorData::getDevice() const {
  return Device(Device::Type::kMetal);
}

std::byte *MetalTensorData::getRawData() const {
  // Unified memory: the buffer the GPU works on is the one we read here. const_cast because
  // MLX only hands out a mutable pointer, while TensorData promises the data itself is not
  // owned mutably by the caller.
  return reinterpret_cast<std::byte *>(
      const_cast<mlx::core::array &>(_array).data<std::byte>());
}

}  // namespace metal
}  // namespace op
}  // namespace fl
