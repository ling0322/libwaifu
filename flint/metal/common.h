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

#include "flint/dtype.h"
#include "flint/tensor.h"
#include "flint/tensor_view.h"
#include "mlx/mlx.h"

namespace fl {
namespace op {
namespace metal {

/// @brief The MLX dtype that `dtype` maps onto.
/// @throw lut::NotImplementedError for a dtype the Metal backend does not carry.
mlx::core::Dtype toMlxDtype(DType dtype);

/// @brief The flint dtype that `dtype` maps onto.
DType fromMlxDtype(mlx::core::Dtype dtype);

/// @brief View `tensor` as an MLX array, without copying.
///
/// flint's shape, stride and offset triple maps straight onto mlx::core::as_strided, so a
/// transposed or sliced tensor goes to MLX as it is rather than being made contiguous first.
/// @throw lut::AbortedError if `tensor` does not live on the Metal device.
mlx::core::array toMlxArray(const TensorView &tensor);

/// @brief Evaluate `result` and write its elements into `out`, in row-major order.
///
/// What every operator ends with: MLX builds a new array, and flint wants the result in the
/// tensor it was handed. `result` must have as many elements as `out` and be of its dtype -- no
/// conversion happens here, so a promotion an operator did not mean shows up as an error rather
/// than as a silently different result. The bytes go through unified memory, the same pointer
/// MetalTensorData::getRawData() hands out; `out` may be strided (a slice `F::cat` copies into).
/// Everything is evaluated before the write, so no GPU work is in flight over `out`'s buffer.
void writeInto(const mlx::core::array &result, const TensorView &out);

/// @brief The MLX shape of `tensor`, for ops that need the shape without the data.
mlx::core::Shape toMlxShape(const TensorView &tensor);

}  // namespace metal
}  // namespace op
}  // namespace fl
