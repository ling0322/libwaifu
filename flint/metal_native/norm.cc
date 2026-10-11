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

#include "lutil/error.h"
#include "lutil/log.h"
#include "lutil/strings.h"
#include "flint/metal_native/common.h"
#include "flint/metal_native/ops.h"

namespace fl {
namespace op {
namespace metal_native {

namespace {

// As kRowThreads in kernels/rows.h.
constexpr uint32_t kRowThreads = 256;

// In step with NormParams in kernels/norm.metal.
struct NormParams {
  uint32_t numRows;
  uint32_t rowLength;
  float eps;
  uint32_t hasWeight;
  uint32_t hasBias;
};

// In step with GroupNormParams in kernels/norm.metal.
struct GroupNormParams {
  uint32_t numGroups;
  uint32_t groupLength;
  uint32_t spatial;
  uint32_t channelsPerGroup;
  uint32_t groups;
  float eps;
  uint32_t hasWeight;
  uint32_t hasBias;
};

// A weight or bias, checked to be `size` elements of `dtype` and made contiguous when it is not,
// to be held until the kernel is recorded. An empty one stays empty.
Contiguous getOperand(
    const TensorView &operand,
    DType dtype,
    int size,
    const char *op,
    const char *what) {
  if (operand.empty()) return Contiguous();

  if (operand.getDType() != dtype) {
    throw lut::InvalidArgError(lut::sprintf(
        "the %s of %s is %s, not %s like its input",
        what,
        op,
        operand.getDType().toString(),
        dtype.toString()));
  }
  if (operand.getNumEl() != size) {
    throw lut::InvalidArgError(lut::sprintf(
        "the %s of %s has %d elements, expected %d",
        what,
        op,
        operand.getNumEl(),
        size));
  }

  return Contiguous(operand);
}

/// The operand to bind for one a kernel may be given or not: `x` stands in for one that is not
/// there, and the kernel is told by its flag not to read it.
Arg argOr(const TensorView &operand, const TensorView &x) {
  return operand.empty() ? getArg(x) : getArg(operand);
}

void normOverLastDim(
    const char *base,
    const char *op,
    const TensorView &input,
    const TensorView &weight,
    const TensorView &bias,
    float eps,
    const TensorView &output) {
  checkFloat(input.getDType(), op);
  CHECK(input.getDim() >= 1);
  checkOutput(output, input.getShape(), input.getDType(), op);
  if (input.getNumEl() == 0) return;

  Contiguous x(input);
  int rowLength = x.getShape(-1);
  Contiguous w = getOperand(weight, x.getDType(), rowLength, op, "weight");
  Contiguous b = getOperand(bias, x.getDType(), rowLength, op, "bias");

  NormParams params{};
  params.numRows = static_cast<uint32_t>(x.getNumEl() / rowLength);
  params.rowLength = static_cast<uint32_t>(rowLength);
  params.eps = eps;
  params.hasWeight = w.empty() ? 0 : 1;
  params.hasBias = b.empty() ? 0 : 1;

  getContext(x)->dispatchGroups(
      kernelName(base, x.getDType()).c_str(),
      {getArg(output), getArg(x), argOr(w, x), argOr(b, x)},
      &params,
      sizeof(params),
      params.numRows,
      1,
      1,
      kRowThreads);
}

}  // namespace

void layerNorm(
    const TensorView &input,
    const TensorView &weight,
    const TensorView &bias,
    float eps,
    const TensorView &out) {
  normOverLastDim("layer_norm", "layerNorm", input, weight, bias, eps, out);
}

void rmsNorm(const TensorView &input, const TensorView &weight, float eps, const TensorView &out) {
  if (weight.empty()) throw lut::InvalidArgError("rmsNorm needs a weight");
  normOverLastDim("rms_norm", "rmsNorm", input, weight, TensorView(), eps, out);
}

void groupNorm(
    const TensorView &input,
    const TensorView &weight,
    const TensorView &bias,
    int groups,
    float eps,
    const TensorView &output) {
  checkFloat(input.getDType(), "groupNorm");
  if (input.getDim() != 4) throw lut::InvalidArgError("groupNorm takes (N, C, H, W)");

  int channels = input.getShape(1);
  if (groups < 1 || channels % groups != 0) {
    throw lut::InvalidArgError(lut::sprintf(
        "groupNorm: %d channels do not split into %d groups",
        channels,
        groups));
  }

  checkOutput(output, input.getShape(), input.getDType(), "groupNorm");
  if (input.getNumEl() == 0) return;

  Contiguous x(input);
  int spatial = x.getShape(2) * x.getShape(3);
  int channelsPerGroup = channels / groups;
  Contiguous w = getOperand(weight, x.getDType(), channels, "groupNorm", "weight");
  Contiguous b = getOperand(bias, x.getDType(), channels, "groupNorm", "bias");

  GroupNormParams params{};
  params.numGroups = static_cast<uint32_t>(x.getShape(0) * groups);
  params.groupLength = static_cast<uint32_t>(channelsPerGroup * spatial);
  params.spatial = static_cast<uint32_t>(spatial);
  params.channelsPerGroup = static_cast<uint32_t>(channelsPerGroup);
  params.groups = static_cast<uint32_t>(groups);
  params.eps = eps;
  params.hasWeight = w.empty() ? 0 : 1;
  params.hasBias = b.empty() ? 0 : 1;

  getContext(x)->dispatchGroups(
      kernelName("group_norm", x.getDType()).c_str(),
      {getArg(output), getArg(x), argOr(w, x), argOr(b, x)},
      &params,
      sizeof(params),
      params.numGroups,
      1,
      1,
      kRowThreads);
}

}  // namespace metal_native
}  // namespace op
}  // namespace fl
