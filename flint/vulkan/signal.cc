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

#include <limits>
#include <vector>

#include "lutil/error.h"
#include "lutil/strings.h"
#include "flint/vulkan/common.h"
#include "flint/vulkan/ops.h"

namespace fl {
namespace op {
namespace vulkan {

namespace {

struct SnakePush {
  uint64_t c;
  uint64_t a;
  uint64_t alpha;
  uint64_t beta;
  uint32_t numel;
  uint32_t width;
  uint32_t channels;
  float eps;
};

struct ClampPush {
  uint64_t c;
  uint64_t a;
  uint32_t numel;
  float lo;
  float hi;
};

struct Pad1dPush {
  uint64_t c;
  uint64_t a;
  uint32_t numel;
  uint32_t width;
  uint32_t outWidth;
  uint32_t left;
  uint32_t mode;
};

struct OverlapAdd1dPush {
  uint64_t c;
  uint64_t a;
  uint64_t bias;
  uint32_t numel;
  uint32_t inWidth;
  uint32_t stride;
  uint32_t padding;
  uint32_t kernel;
  uint32_t channels;
  uint32_t outWidth;
  uint32_t batchStride;
  uint32_t channelStride;
};

/// Every kernel here indexes with 32 bits.
void checkSize(int64_t numel, const char *op) {
  if (numel > std::numeric_limits<uint32_t>::max()) {
    throw lut::InvalidArgError(lut::sprintf("%s: the tensor is too large", op));
  }
}

/// A per-channel parameter of `channels` elements of `dtype`, packed. An empty one stays empty.
Contiguous parameter(const TensorView &tensor, DType dtype, int channels, const char *what) {
  if (tensor.empty()) return Contiguous();
  if (tensor.getDType() != dtype || tensor.getNumEl() != channels) {
    throw lut::InvalidArgError(
        lut::sprintf("%s: %s is not one of the input's type per channel", "snake", what));
  }
  return Contiguous(tensor);
}

}  // namespace

void overlapAdd1d(
    const TensorView &columns,
    const TensorView &bias,
    int length,
    int stride,
    int padding,
    int kernel,
    const TensorView &out) {
  checkFloat(out.getDType(), "convTranspose1d");
  if (!columns.isContiguous() || out.getStride(2) != 1) {
    throw lut::InvalidArgError("convTranspose1d: the columns or the output is not laid out in rows");
  }
  checkSize(columns.getNumEl(), "convTranspose1d");
  checkSize(out.getNumEl(), "convTranspose1d");
  if (out.getNumEl() == 0) return;

  Contiguous b(bias);
  OverlapAdd1dPush push{};
  push.c = getAddress(out);
  push.a = getAddress(columns);
  push.bias = b.empty() ? 0 : getAddress(b);
  push.numel = static_cast<uint32_t>(out.getNumEl());
  push.inWidth = static_cast<uint32_t>(length);
  push.stride = static_cast<uint32_t>(stride);
  push.padding = static_cast<uint32_t>(padding);
  push.kernel = static_cast<uint32_t>(kernel);
  push.channels = static_cast<uint32_t>(out.getShape(1));
  push.outWidth = static_cast<uint32_t>(out.getShape(2));
  push.batchStride = static_cast<uint32_t>(out.getStride(0));
  push.channelStride = static_cast<uint32_t>(out.getStride(1));
  getContext(out)->dispatchLinear(
      kernelName("overlap_add1d", out.getDType()).c_str(),
      &push,
      sizeof(push),
      push.numel);
}

void snake(
    const TensorView &input,
    const TensorView &alpha,
    const TensorView &beta,
    float eps,
    const TensorView &out) {
  checkFloat(input.getDType(), "snake");
  if (input.getDim() != 3) throw lut::InvalidArgError("snake: the input must be (N, C, L)");
  checkOutput(out, input.getShape(), input.getDType(), "snake");
  checkSize(input.getNumEl(), "snake");
  if (input.getNumEl() == 0) return;

  int channels = input.getShape(1);
  Contiguous x(input);
  Contiguous a = parameter(alpha, x.getDType(), channels, "alpha");
  Contiguous b = parameter(beta, x.getDType(), channels, "beta");
  if (a.empty()) throw lut::InvalidArgError("snake: alpha is empty");

  SnakePush push{};
  push.c = getAddress(out);
  push.a = getAddress(x);
  push.alpha = getAddress(a);
  push.beta = b.empty() ? 0 : getAddress(b);
  push.numel = static_cast<uint32_t>(x.getNumEl());
  push.width = static_cast<uint32_t>(x.getShape(2));
  push.channels = static_cast<uint32_t>(channels);
  push.eps = eps;
  getContext(x)->dispatchLinear(
      kernelName("snake", x.getDType()).c_str(),
      &push,
      sizeof(push),
      push.numel);
}

void pad1d(const TensorView &input, int left, int right, PadMode mode, const TensorView &out) {
  checkFloat(input.getDType(), "pad1d");
  if (input.getDim() < 1) throw lut::InvalidArgError("pad1d: the input has no dimension to pad");
  int length = input.getShape(-1);
  if (left < 0 || right < 0) throw lut::InvalidArgError("pad1d: a padding cannot be negative");
  if (length < 1) throw lut::InvalidArgError("pad1d: the input is empty");
  if (mode == PadMode::kReflect && (left >= length || right >= length)) {
    throw lut::InvalidArgError("pad1d: a reflection pads by less than the input is long");
  }
  std::vector<int> shape = input.getShape();
  shape.back() = left + length + right;
  checkOutput(out, shape, input.getDType(), "pad1d");
  checkSize(out.getNumEl(), "pad1d");
  if (out.getNumEl() == 0) return;

  Contiguous x(input);
  Pad1dPush push{};
  push.c = getAddress(out);
  push.a = getAddress(x);
  push.numel = static_cast<uint32_t>(out.getNumEl());
  push.width = static_cast<uint32_t>(length);
  push.outWidth = static_cast<uint32_t>(shape.back());
  push.left = static_cast<uint32_t>(left);
  push.mode = static_cast<uint32_t>(mode);
  getContext(x)->dispatchLinear(
      kernelName("pad1d", x.getDType()).c_str(),
      &push,
      sizeof(push),
      push.numel);
}

void clamp(const TensorView &input, float lo, float hi, const TensorView &out) {
  checkFloat(input.getDType(), "clamp");
  checkOutput(out, input.getShape(), input.getDType(), "clamp");
  checkSize(input.getNumEl(), "clamp");
  if (input.getNumEl() == 0) return;

  Contiguous x(input);
  ClampPush push{};
  push.c = getAddress(out);
  push.a = getAddress(x);
  push.numel = static_cast<uint32_t>(x.getNumEl());
  push.lo = lo;
  push.hi = hi;
  getContext(x)->dispatchLinear(
      kernelName("clamp", x.getDType()).c_str(),
      &push,
      sizeof(push),
      push.numel);
}

}  // namespace vulkan
}  // namespace op
}  // namespace fl
