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

#include <iostream>
#include <vector>

#include "lutil/error.h"
#include "lutil/log.h"
#include "flint/metal/common.h"
#include "flint/metal/ops.h"
#include "flint/metal/to_device.h"
#include "flint/operators.h"

namespace fl {
namespace op {
namespace metal {

namespace {

mlx::core::Shape toMlxShape(lut::Span<const int> shape) {
  mlx::core::Shape result;
  for (int dim : shape) {
    result.push_back(dim);
  }
  return result;
}

/// The gated linear units differ only in which activation gates the other half, so they share
/// everything but that one call.
template<typename Activation>
Tensor gatedLinearUnit(const Tensor &input, Activation activation) {
  mlx::core::array x = toMlxArray(input);
  CHECK(x.shape(-1) % 2 == 0) << "a gated linear unit needs an even last dimension";

  std::vector<mlx::core::array> halves = mlx::core::split(x, 2, -1);
  return fromMlxArray(mlx::core::multiply(activation(halves[0]), halves[1]));
}

}  // namespace

Tensor lookup(const Tensor &table, const Tensor &indices) {
  // take() along the vocabulary axis is the embedding lookup: the index tensor's shape becomes
  // the leading dimensions and the embedding width comes along on the end.
  return fromMlxArray(mlx::core::take(toMlxArray(table), toMlxArray(indices), /*axis=*/0));
}

Tensor upsampleNearest2d(const Tensor &input, int scale) {
  CHECK(input.getDim() == 4) << "upsampleNearest2d expects (N, C, H, W)";
  int n = input.getShape(0);
  int c = input.getShape(1);
  int h = input.getShape(2);
  int w = input.getShape(3);

  // Nearest-neighbour upsampling is a broadcast: give each pixel a pair of singleton axes, spread
  // them to `scale`, then fold them back into the spatial dimensions they belong to.
  mlx::core::array x = mlx::core::reshape(toMlxArray(input), {n, c, h, 1, w, 1});
  x = mlx::core::broadcast_to(x, {n, c, h, scale, w, scale});

  return fromMlxArray(mlx::core::reshape(x, {n, c, h * scale, w * scale}));
}

Tensor geglu(const Tensor &input) {
  return gatedLinearUnit(input, [](const mlx::core::array &a) {
    mlx::core::array half = mlx::core::array(0.5f, a.dtype());
    mlx::core::array one = mlx::core::array(1.0f, a.dtype());
    mlx::core::array invSqrt2 = mlx::core::array(0.7071067811865475f, a.dtype());
    return mlx::core::multiply(
        mlx::core::multiply(half, a),
        mlx::core::add(one, mlx::core::erf(mlx::core::multiply(a, invSqrt2))));
  });
}

Tensor swiglu(const Tensor &input) {
  return gatedLinearUnit(input, [](const mlx::core::array &a) {
    return mlx::core::multiply(a, mlx::core::sigmoid(a));
  });
}

Tensor cast(const Tensor &input, DType dtype) {
  return fromMlxArray(mlx::core::astype(toMlxArray(input), toMlxDtype(dtype)));
}

Tensor createTensor(lut::Span<const int> shape, DType dtype) {
  // MLX has no uninitialized allocation in its public API, and zeros costs one pass over memory
  // that a caller about to overwrite the tensor does not need. It is still the safe default.
  return zeros(shape, dtype);
}

Tensor zeros(lut::Span<const int> shape, DType dtype) {
  return fromMlxArray(mlx::core::zeros(toMlxShape(shape), toMlxDtype(dtype)));
}

void fill(Tensor input, float value) {
  // flint's fill mutates in place, while MLX only builds new arrays, so the value is written
  // through the raw pointer that unified memory already gives us.
  mlx::core::array filled = mlx::core::full(
      toMlxShape(input),
      mlx::core::array(value, toMlxDtype(input.getDType())),
      toMlxDtype(input.getDType()));
  mlx::core::eval(filled);

  copy(fromMlxArray(filled), input);
}

void copy(const Tensor &src, Tensor dest) {
  CHECK(src.getDType() == dest.getDType()) << "copy: dtype mismatch";

  // contiguous() resolves whatever view the source is, so a transposed or sliced tensor lands in
  // the destination in the destination's own layout rather than carrying its strides along.
  mlx::core::array resolved =
      mlx::core::contiguous(mlx::core::reshape(toMlxArray(src), toMlxShape(dest)));
  mlx::core::eval(resolved);

  DType dtype = dest.getDType();
  int64_t elemSize = dtype.getTotalSize(1);
  std::byte *base =
      dest.getInternalData()->getRawData() + dtype.getTotalSize(dest.getInternalOffset());
  const std::byte *from = resolved.data<std::byte>();

  if (dest.isContiguous()) {
    memcpy(base, from, static_cast<size_t>(dtype.getTotalSize(dest.getNumEl())));
    return;
  }

  // A strided destination is what F::cat asks for: it allocates the joined tensor and copies each
  // half into a slice of it. MLX arrays are values and cannot be written through, so the scatter
  // goes through the pointer that unified memory already exposes. Everything is evaluated by the
  // time we get here, so there is no GPU work in flight over this buffer.
  int ndim = dest.getDim();
  std::vector<int> shape(ndim);
  std::vector<int64_t> stride(ndim);
  for (int d = 0; d < ndim; ++d) {
    shape[d] = dest.getShape(d);
    stride[d] = dest.getStride(d);
  }

  // Whole rows move at once whenever the last dimension is dense, which is the case for a slice
  // taken along any earlier axis -- so concatenating feature maps stays memcpy-shaped.
  int64_t rowLen = (ndim > 0 && stride[ndim - 1] == 1) ? shape[ndim - 1] : 1;
  int64_t rows = dest.getNumEl() / rowLen;

  std::vector<int> index(ndim, 0);
  for (int64_t row = 0; row < rows; ++row) {
    int64_t offset = 0;
    for (int d = 0; d < ndim; ++d) {
      offset += static_cast<int64_t>(index[d]) * stride[d];
    }
    memcpy(
        base + offset * elemSize,
        from + row * rowLen * elemSize,
        static_cast<size_t>(rowLen * elemSize));

    // Step the odometer over every axis the row does not already cover.
    for (int d = (rowLen > 1 ? ndim - 2 : ndim - 1); d >= 0; --d) {
      if (++index[d] < shape[d]) break;
      index[d] = 0;
    }
  }
}

void print(const Tensor &tensor) {
  F::print(toCpu(tensor));
}

}  // namespace metal
}  // namespace op
}  // namespace fl
