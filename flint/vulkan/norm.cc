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
#include "flint/vulkan/common.h"
#include "flint/vulkan/ops.h"

namespace fl {
namespace op {
namespace vulkan {

namespace {

constexpr int kRowThreads = 256;

struct NormPush {
  uint64_t c;
  uint64_t a;
  uint64_t weight;
  uint64_t bias;
  uint32_t numRows;
  uint32_t rowLength;
  float eps;
};

struct GroupNormPush {
  uint64_t c;
  uint64_t a;
  uint64_t weight;
  uint64_t bias;
  uint32_t numGroups;
  uint32_t groupLength;
  uint32_t spatial;
  uint32_t channelsPerGroup;
  uint32_t groups;
  float eps;
};

// The address of a weight or bias, which must be `size` elements of `dtype` -- or zero when it is
// empty, which the kernels take to mean there is none. Made contiguous first when it is not, and
// kept alive through `keep` until the kernel is recorded.
uint64_t getOperand(
    const Tensor &operand,
    DType dtype,
    int size,
    const char *op,
    const char *what,
    Tensor *keep) {
  if (operand.empty()) return 0;

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

  *keep = makeContiguous(operand);
  return getAddress(*keep);
}

Tensor normOverLastDim(
    const char *base,
    const char *op,
    const Tensor &input,
    const Tensor &weight,
    const Tensor &bias,
    float eps) {
  checkFloat(input.getDType(), op);
  CHECK(input.getDim() >= 1);

  Tensor x = makeContiguous(input);
  int rowLength = x.getShape(-1);
  Tensor output = createTensor(x.getShape(), x.getDType());
  if (x.getNumEl() == 0) return output;

  Tensor keepWeight;
  Tensor keepBias;
  NormPush push{};
  push.c = getAddress(output);
  push.a = getAddress(x);
  push.weight = getOperand(weight, x.getDType(), rowLength, op, "weight", &keepWeight);
  push.bias = getOperand(bias, x.getDType(), rowLength, op, "bias", &keepBias);
  push.numRows = static_cast<uint32_t>(x.getNumEl() / rowLength);
  push.rowLength = static_cast<uint32_t>(rowLength);
  push.eps = eps;

  getContext(x)->dispatchLinear(
      kernelName(base, x.getDType()).c_str(),
      &push,
      sizeof(push),
      static_cast<int64_t>(push.numRows) * kRowThreads,
      kRowThreads);
  return output;
}

}  // namespace

Tensor layerNorm(const Tensor &input, const Tensor &weight, const Tensor &bias, float eps) {
  return normOverLastDim("layer_norm", "layerNorm", input, weight, bias, eps);
}

Tensor rmsNorm(const Tensor &input, const Tensor &weight, float eps) {
  if (weight.empty()) throw lut::InvalidArgError("rmsNorm needs a weight");
  return normOverLastDim("rms_norm", "rmsNorm", input, weight, Tensor(), eps);
}

Tensor groupNorm(
    const Tensor &input,
    const Tensor &weight,
    const Tensor &bias,
    int groups,
    float eps) {
  checkFloat(input.getDType(), "groupNorm");
  if (input.getDim() != 4) throw lut::InvalidArgError("groupNorm takes (N, C, H, W)");

  int channels = input.getShape(1);
  if (groups < 1 || channels % groups != 0) {
    throw lut::InvalidArgError(lut::sprintf(
        "groupNorm: %d channels do not split into %d groups",
        channels,
        groups));
  }

  Tensor x = makeContiguous(input);
  Tensor output = createTensor(x.getShape(), x.getDType());
  if (x.getNumEl() == 0) return output;

  int spatial = x.getShape(2) * x.getShape(3);
  int channelsPerGroup = channels / groups;

  Tensor keepWeight;
  Tensor keepBias;
  GroupNormPush push{};
  push.c = getAddress(output);
  push.a = getAddress(x);
  push.weight = getOperand(weight, x.getDType(), channels, "groupNorm", "weight", &keepWeight);
  push.bias = getOperand(bias, x.getDType(), channels, "groupNorm", "bias", &keepBias);
  push.numGroups = static_cast<uint32_t>(x.getShape(0) * groups);
  push.groupLength = static_cast<uint32_t>(channelsPerGroup * spatial);
  push.spatial = static_cast<uint32_t>(spatial);
  push.channelsPerGroup = static_cast<uint32_t>(channelsPerGroup);
  push.groups = static_cast<uint32_t>(groups);
  push.eps = eps;

  getContext(x)->dispatchLinear(
      kernelName("group_norm", x.getDType()).c_str(),
      &push,
      sizeof(push),
      static_cast<int64_t>(push.numGroups) * kRowThreads,
      kRowThreads);
  return output;
}

}  // namespace vulkan
}  // namespace op
}  // namespace fl
