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
#include "flint/metal/ops.h"

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

}  // namespace

Tensor rand(lut::Span<const int> shape, DType dtype) {
  return fromMlxArray(mlx::core::random::uniform(toMlxShape(shape), toMlxDtype(dtype)));
}

Tensor randNormal(lut::Span<const int> shape) {
  return fromMlxArray(
      mlx::core::random::normal(toMlxShape(shape), mlx::core::float32, 0.0f, 1.0f));
}

void manualSeed(uint64_t seed) {
  mlx::core::random::seed(seed);
}

}  // namespace metal
}  // namespace op
}  // namespace fl
