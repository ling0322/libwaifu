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

#include "flint/metal/common.h"

#include <vector>

#include "lutil/error.h"
#include "lutil/log.h"
#include "lutil/span.h"
#include "lutil/strings.h"
#include "flint/metal/metal_tensor_data.h"

namespace fl {
namespace op {
namespace metal {

mlx::core::Dtype toMlxDtype(DType dtype) {
  switch (static_cast<int16_t>(dtype)) {
    case DType::kFloat:
      return mlx::core::float32;
    case DType::kFloat16:
      return mlx::core::float16;
    case DType::kLong:
      return mlx::core::int64;
    case DType::kInt32:
      return mlx::core::int32;
    case DType::kBool:
      return mlx::core::bool_;
    default:
      throw lut::NotImplementedError(
          lut::sprintf("the Metal backend has no dtype for %s", dtype.toString().c_str()));
  }
}

DType fromMlxDtype(mlx::core::Dtype dtype) {
  if (dtype == mlx::core::float32) return DType(DType::kFloat);
  if (dtype == mlx::core::float16) return DType(DType::kFloat16);
  if (dtype == mlx::core::int64) return DType(DType::kLong);
  if (dtype == mlx::core::int32) return DType(DType::kInt32);
  if (dtype == mlx::core::bool_) return DType(DType::kBool);

  throw lut::NotImplementedError("no flint dtype for this MLX dtype");
}

mlx::core::Shape toMlxShape(const Tensor &tensor) {
  mlx::core::Shape shape;
  for (int d = 0; d < tensor.getDim(); ++d) {
    shape.push_back(tensor.getShape(d));
  }
  return shape;
}

mlx::core::array toMlxArray(const Tensor &tensor) {
  if (tensor.getDevice().getType() != Device::kMetal) {
    throw lut::AbortedError(
        lut::sprintf(
            "expected a tensor on the metal device, got one on %s",
            tensor.getDevice().getName().c_str()));
  }

  const auto *data = static_cast<const MetalTensorData *>(tensor.getInternalData().get());

  mlx::core::Strides strides;
  for (int d = 0; d < tensor.getDim(); ++d) {
    strides.push_back(tensor.getStride(d));
  }

  // Both sides count strides in elements and measure the offset from the start of the buffer, so
  // a transposed or sliced flint tensor becomes the equivalent MLX view with nothing copied.
  return mlx::core::as_strided(
      data->getArray(),
      toMlxShape(tensor),
      strides,
      static_cast<size_t>(tensor.getInternalOffset()));
}

Tensor fromMlxArray(mlx::core::array array) {
  mlx::core::Shape shape = array.shape();

  // Storage is always a flat, contiguous buffer; the shape rides along in TensorShape. contiguous()
  // is a no-op when the array already is one, so an op that produced a dense result pays nothing.
  mlx::core::array flat = mlx::core::flatten(mlx::core::contiguous(array));
  mlx::core::eval(flat);

  std::vector<TensorShape::ShapeType> dims;
  for (mlx::core::ShapeElem dim : shape) {
    dims.push_back(static_cast<TensorShape::ShapeType>(dim));
  }
  // MLX reductions can drop to a scalar, which flint has no tensor for; the 1-element vector is
  // the same value and is what F::elem() expects to read.
  if (dims.empty()) dims.push_back(1);

  return Tensor::create(
      std::make_shared<TensorShape>(lut::makeConstSpan(dims)),
      MetalTensorData::wrap(std::move(flat)));
}

}  // namespace metal
}  // namespace op
}  // namespace fl
