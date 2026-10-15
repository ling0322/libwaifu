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
#include "flint/metal_native/common.h"
#include "flint/metal_native/ops.h"

namespace fl {
namespace op {
namespace metal_native {

namespace {

// Keep each of these in step with the struct of the same name in kernels/signal.metal.

struct OverlapAdd1dParams {
  uint32_t numel;
  uint32_t inWidth;
  uint32_t stride;
  uint32_t padding;
  uint32_t taps;
  uint32_t channels;
  uint32_t outWidth;
  uint32_t batchStride;
  uint32_t channelStride;
  uint32_t hasBias;
};

struct SnakeParams {
  uint32_t numel;
  uint32_t width;
  uint32_t channels;
  uint32_t hasBeta;
  float eps;
};

struct Pad1dParams {
  uint32_t numel;
  uint32_t width;
  uint32_t outWidth;
  uint32_t left;
  uint32_t mode;
};

struct ClampParams {
  uint32_t numel;
  float lo;
  float hi;
};

/// Every kernel here indexes with 32 bits.
void checkSize(int64_t numel, const char *op) {
  if (numel > std::numeric_limits<uint32_t>::max()) {
    throw lut::InvalidArgError(lut::sprintf("%s: the tensor is too large", op));
  }
}

/// A per-channel operand of `channels` elements of `dtype`, packed. An empty one stays empty.
Contiguous parameter(
    const TensorView &tensor,
    DType dtype,
    int channels,
    const char *op,
    const char *what) {
  if (tensor.empty()) return Contiguous();
  if (tensor.getDType() != dtype || tensor.getNumEl() != channels) {
    throw lut::InvalidArgError(
        lut::sprintf("%s: %s is not one of the input's type per channel", op, what));
  }
  return Contiguous(tensor);
}

/// What a kernel is handed for an operand that may be empty: the operand, or `x` in its place,
/// which the kernel is told not to read.
Arg argOr(const TensorView &operand, const TensorView &x) {
  return operand.empty() ? getArg(x) : getArg(operand);
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

  Contiguous b = parameter(bias, out.getDType(), out.getShape(1), "convTranspose1d", "bias");
  OverlapAdd1dParams params{};
  params.numel = static_cast<uint32_t>(out.getNumEl());
  params.inWidth = static_cast<uint32_t>(length);
  params.stride = static_cast<uint32_t>(stride);
  params.padding = static_cast<uint32_t>(padding);
  params.taps = static_cast<uint32_t>(kernel);
  params.channels = static_cast<uint32_t>(out.getShape(1));
  params.outWidth = static_cast<uint32_t>(out.getShape(2));
  params.batchStride = static_cast<uint32_t>(out.getStride(0));
  params.channelStride = static_cast<uint32_t>(out.getStride(1));
  params.hasBias = b.empty() ? 0 : 1;
  getContext(out)->dispatch(
      kernelName("overlap_add1d", out.getDType()).c_str(),
      {getArg(out), getArg(columns), argOr(b, columns)},
      &params,
      sizeof(params),
      params.numel);
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
  Contiguous a = parameter(alpha, x.getDType(), channels, "snake", "alpha");
  Contiguous b = parameter(beta, x.getDType(), channels, "snake", "beta");
  if (a.empty()) throw lut::InvalidArgError("snake: alpha is empty");

  SnakeParams params{};
  params.numel = static_cast<uint32_t>(x.getNumEl());
  params.width = static_cast<uint32_t>(x.getShape(2));
  params.channels = static_cast<uint32_t>(channels);
  params.hasBeta = b.empty() ? 0 : 1;
  params.eps = eps;
  getContext(x)->dispatch(
      kernelName("snake", x.getDType()).c_str(),
      {getArg(out), getArg(x), getArg(a), argOr(b, a)},
      &params,
      sizeof(params),
      params.numel);
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
  Pad1dParams params{};
  params.numel = static_cast<uint32_t>(out.getNumEl());
  params.width = static_cast<uint32_t>(length);
  params.outWidth = static_cast<uint32_t>(shape.back());
  params.left = static_cast<uint32_t>(left);
  params.mode = static_cast<uint32_t>(mode);
  getContext(x)->dispatch(
      kernelName("pad1d", x.getDType()).c_str(),
      {getArg(out), getArg(x)},
      &params,
      sizeof(params),
      params.numel);
}

void clamp(const TensorView &input, float lo, float hi, const TensorView &out) {
  checkFloat(input.getDType(), "clamp");
  checkOutput(out, input.getShape(), input.getDType(), "clamp");
  checkSize(input.getNumEl(), "clamp");
  if (input.getNumEl() == 0) return;

  Contiguous x(input);
  ClampParams params{};
  params.numel = static_cast<uint32_t>(x.getNumEl());
  params.lo = lo;
  params.hi = hi;
  getContext(x)->dispatch(
      kernelName("clamp", x.getDType()).c_str(),
      {getArg(out), getArg(x)},
      &params,
      sizeof(params),
      params.numel);
}

}  // namespace metal_native
}  // namespace op
}  // namespace fl
