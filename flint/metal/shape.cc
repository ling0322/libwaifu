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

#include <vector>

#include "lutil/error.h"
#include "lutil/log.h"
#include "flint/functional.h"
#include "flint/metal/common.h"
#include "flint/metal/ops.h"
#include "flint/metal/to_device.h"
#include "flint/operators.h"

namespace fl {
namespace op {
namespace metal {

namespace {

/// The gated linear units differ only in which activation gates the other half, so they share
/// everything but that one call.
template<typename Activation>
void gatedLinearUnit(const TensorView &input, Activation activation, const TensorView &out) {
  mlx::core::array x = toMlxArray(input);
  CHECK(x.shape(-1) % 2 == 0) << "a gated linear unit needs an even last dimension";

  std::vector<mlx::core::array> halves = mlx::core::split(x, 2, -1);
  writeInto(mlx::core::multiply(activation(halves[0]), halves[1]), out);
}

}  // namespace

void lookup(const TensorView &table, const TensorView &indices, const TensorView &out) {
  // take() along the vocabulary axis is the embedding lookup: the index tensor's shape becomes
  // the leading dimensions and the embedding width comes along on the end.
  writeInto(mlx::core::take(toMlxArray(table), toMlxArray(indices), /*axis=*/0), out);
}

void upsampleNearest2d(const TensorView &input, int scale, const TensorView &out) {
  CHECK(input.getDim() == 4) << "upsampleNearest2d expects (N, C, H, W)";
  int n = input.getShape(0);
  int c = input.getShape(1);
  int h = input.getShape(2);
  int w = input.getShape(3);

  // Nearest-neighbour upsampling is a broadcast: give each pixel a pair of singleton axes, spread
  // them to `scale`, then fold them back into the spatial dimensions they belong to.
  mlx::core::array x = mlx::core::reshape(toMlxArray(input), {n, c, h, 1, w, 1});
  x = mlx::core::broadcast_to(x, {n, c, h, scale, w, scale});

  writeInto(mlx::core::reshape(x, {n, c, h * scale, w * scale}), out);
}

void geglu(const TensorView &input, const TensorView &out) {
  gatedLinearUnit(
      input,
      [](const mlx::core::array &a) {
        mlx::core::array half = mlx::core::array(0.5f, a.dtype());
        mlx::core::array one = mlx::core::array(1.0f, a.dtype());
        mlx::core::array invSqrt2 = mlx::core::array(0.7071067811865475f, a.dtype());
        return mlx::core::multiply(
            mlx::core::multiply(half, a),
            mlx::core::add(one, mlx::core::erf(mlx::core::multiply(a, invSqrt2))));
      },
      out);
}

void swiglu(const TensorView &input, const TensorView &out) {
  gatedLinearUnit(
      input,
      [](const mlx::core::array &a) { return mlx::core::multiply(a, mlx::core::sigmoid(a)); },
      out);
}

void cast(const TensorView &input, const TensorView &out) {
  // The target type is `out`'s: functional decides it, and never asks for a cast to the type the
  // input already has.
  writeInto(mlx::core::astype(toMlxArray(input), toMlxDtype(out.getDType())), out);
}

void fill(const TensorView &input, float value) {
  // flint's fill mutates in place, while MLX only builds new arrays, so the value is written
  // through the raw pointer that unified memory already gives us -- into whatever view `input`
  // is, a slice included.
  mlx::core::Dtype dtype = toMlxDtype(input.getDType());
  writeInto(mlx::core::full(toMlxShape(input), mlx::core::array(value, dtype), dtype), input);
}

void copy(const TensorView &src, const TensorView &dest) {
  CHECK(src.getDType() == dest.getDType()) << "copy: dtype mismatch";
  CHECK(src.getNumEl() == dest.getNumEl()) << "copy: the two sides differ in size";

  // writeInto resolves whatever view the source is, so a transposed or sliced tensor lands in the
  // destination in the destination's own layout rather than carrying its strides along.
  writeInto(toMlxArray(src), dest);
}

void print(const TensorView &tensor) {
  // Packed on the card first if it is a strided view, since a transfer is one run of bytes, then
  // printed by the CPU operators, which know how.
  Tensor packed;
  TensorView source = tensor;
  if (!tensor.isContiguous()) {
    packed = F::emptyLike(tensor);
    copy(tensor, packed);
    source = packed;
  }

  Tensor host = F::empty(Device::getCpu(), source.getShape(), source.getDType());
  transfer(source, host);
  getOperators(Device::kCpu)->print(host);
}

}  // namespace metal
}  // namespace op
}  // namespace fl
