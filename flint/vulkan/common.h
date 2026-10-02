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

#pragma once

#include <stdint.h>

#include <initializer_list>
#include <string>
#include <vector>

#include "lutil/span.h"
#include "flint/dtype.h"
#include "flint/tensor.h"
#include "flint/tensor_view.h"
#include "flint/vulkan/context.h"

namespace fl {
namespace op {
namespace vulkan {

/// The most dimensions a strided kernel takes, after collapse() has merged what it can.
constexpr int kMaxDims = 6;

/// A contiguous, uninitialized tensor on the Vulkan device, for a kernel's own temporaries.
Tensor createTensor(lut::Span<const int> shape, DType dtype);

/// A view of `base`'s storage, from its first element, laid out as `elems` say.
TensorView restride(const TensorView &base, lut::Span<const TensorShape::Elem> elems);

/// Throws unless `out` is contiguous, of `shape` and of `dtype`: what every kernel here writes.
void checkOutput(const TensorView &out, lut::Span<const int> shape, DType dtype, const char *op);

/// The device `tensor` lives on.
/// @throw lut::AbortedError if it is not a Vulkan tensor.
Context *getContext(const TensorView &tensor);

/// The device address of `tensor`'s first element, which is what a kernel is handed.
uint64_t getAddress(const TensorView &tensor);

/// The buffer holding `tensor`, and the byte offset of its first element in it.
const Buffer &getBuffer(const TensorView &tensor);
int64_t getByteOffset(const TensorView &tensor);

/// The suffix that names `dtype` in a kernel's name: f32, f16, i64, i32, u8, i8 or bool.
/// @throw lut::NotImplementedError for a dtype no kernel is built for.
std::string getTypeSuffix(DType dtype);

/// The name of the kernel `base` built for `dtype`, as in "softmax_f16".
std::string kernelName(const char *base, DType dtype);

/// Throws unless `dtype` is float or float16, naming `op` in the message.
void checkFloat(DType dtype, const char *op);

/// A shape and the strides of each of several tensors over it, with the dimensions merged that
/// every one of those tensors lays out as one, and dimensions of size one dropped: the shape as a
/// strided kernel wants it. Merging is what gets a tensor of any rank below kMaxDims.
struct Layout {
  int ndim;
  uint32_t shape[kMaxDims];
  uint32_t strides[3][kMaxDims];
};

/// Collapse `shape` with the strides of up to three tensors. Returns false when even the merged
/// shape has more than kMaxDims dimensions.
bool collapse(
    const std::vector<int> &shape,
    std::initializer_list<std::vector<int>> strides,
    Layout *layout);

/// The strides of `tensor`, as a vector.
std::vector<int> getStrides(const TensorView &tensor);

/// `tensor`'s elements as one contiguous run: `tensor` itself if they already are one, and
/// otherwise a packed copy, which this holds for as long as it lives.
///
/// It is a TensorView, so it goes wherever one is expected, a temporary one included -- it lives
/// to the end of the call it was made for. What it must not be is copied into a plain TensorView
/// that outlives it, which keeps the view and drops the copy. An empty view stays empty.
class Contiguous : public TensorView {
 public:
  Contiguous() = default;
  explicit Contiguous(const TensorView &tensor);

 private:
  Tensor _copy;
};

}  // namespace vulkan
}  // namespace op
}  // namespace fl
