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

#pragma once

#include "flint/device.h"
#include "flint/tensor.h"
#include "mlx/mlx.h"

namespace fl {
namespace op {
namespace metal {

/// @brief Tensor storage on the Metal device, held as a flat MLX array.
///
/// The array is one dimensional and always evaluated, which is what lets getRawData() hand out a
/// plain pointer: on Apple silicon MLX allocates in unified memory, so the GPU buffer is CPU
/// addressable and needs no staging copy to read. Shape and stride stay where flint keeps them,
/// in TensorShape; this object only owns the bytes.
class MetalTensorData : public TensorData {
 public:
  static std::shared_ptr<TensorData> create(int64_t numel, DType dtype);

  /// @brief Adopt `array` as tensor storage. It must be one dimensional, contiguous and already
  ///        evaluated -- fromMlxArray() is what normally arranges that.
  static std::shared_ptr<TensorData> wrap(mlx::core::array array);

  Device getDevice() const override;
  std::byte *getRawData() const override;

  /// @brief The underlying flat array, for building views onto it.
  const mlx::core::array &getArray() const {
    return _array;
  }

 private:
  mlx::core::array _array;

  explicit MetalTensorData(mlx::core::array array);
};

}  // namespace metal
}  // namespace op
}  // namespace fl
