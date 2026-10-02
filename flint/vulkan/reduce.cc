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

#include <vector>

#include "lutil/error.h"
#include "lutil/log.h"
#include "lutil/strings.h"
#include "flint/vulkan/common.h"
#include "flint/vulkan/ops.h"

namespace fl {
namespace op {
namespace vulkan {

namespace {

struct RowPush {
  uint64_t c;
  uint64_t a;
  uint32_t numRows;
  uint32_t rowLength;
  uint32_t op;
};

// A kernel that gives each row a workgroup of this many threads, as rows.glsl does.
constexpr int kRowThreads = 256;

void dispatchRows(
    const char *kernel,
    const TensorView &input,
    const TensorView &output,
    uint32_t op) {
  int rowLength = input.getShape(-1);
  RowPush push{};
  push.c = getAddress(output);
  push.a = getAddress(input);
  push.numRows = static_cast<uint32_t>(input.getNumEl() / rowLength);
  push.rowLength = static_cast<uint32_t>(rowLength);
  push.op = op;

  getContext(input)->dispatchLinear(
      kernel,
      &push,
      sizeof(push),
      static_cast<int64_t>(push.numRows) * kRowThreads,
      kRowThreads);
}

std::vector<int> withoutLastDim(const TensorView &input) {
  std::vector<int> shape = input.getShape();
  shape.pop_back();
  if (shape.empty()) shape.push_back(1);
  return shape;
}

int realDim(const TensorView &input, int dim) {
  int rank = input.getDim();
  if (dim < 0) dim += rank;
  CHECK(dim >= 0 && dim < rank);
  return dim;
}

}  // namespace

void reduceLastDim(const TensorView &input, ReduceOp op, const TensorView &out) {
  checkFloat(input.getDType(), "a reduction");
  CHECK(input.getDim() >= 1 && input.getShape(-1) > 0);
  checkOutput(out, withoutLastDim(input), input.getDType(), "a reduction");
  if (input.getNumEl() == 0) return;

  Contiguous x(input);
  dispatchRows(kernelName("reduce", x.getDType()).c_str(), x, out, static_cast<uint32_t>(op));
}

void sum(const TensorView &input, int dim, const TensorView &out) {
  dim = realDim(input, dim);
  if (dim == input.getDim() - 1) {
    reduceLastDim(input, ReduceOp::kSum, out);
    return;
  }

  // Any other dimension is moved to the end and summed there, as the CPU does it, and the
  // result is turned back as it is written out.
  TensorView moved = input.transpose(dim, -1);
  Tensor summed = createTensor(withoutLastDim(moved), input.getDType());
  reduceLastDim(moved, ReduceOp::kSum, summed);
  TensorView restored = TensorView(summed).unsqueeze(summed.getDim()).transpose(dim, -1).squeeze(
      dim);
  copy(restored, out);
}

void cumsum(const TensorView &input, int dim, const TensorView &out) {
  checkFloat(input.getDType(), "cumsum");
  CHECK(input.getDim() >= 1);
  checkOutput(out, input.getShape(), input.getDType(), "cumsum");
  if (input.getNumEl() == 0) return;

  // The scan runs along the last dimension; any other is moved there and back, which leaves the
  // shape as it was.
  dim = realDim(input, dim);
  bool moved = dim != input.getDim() - 1;
  Contiguous x(moved ? input.transpose(dim, -1) : input);

  if (!moved) {
    dispatchRows(kernelName("scan", x.getDType()).c_str(), x, out, 0);
    return;
  }

  Tensor scanned = createTensor(x.getShape(), x.getDType());
  dispatchRows(kernelName("scan", x.getDType()).c_str(), x, scanned, 0);
  copy(TensorView(scanned).transpose(dim, -1), out);
}

void softmax(const TensorView &input, const TensorView &out) {
  checkFloat(input.getDType(), "softmax");
  CHECK(input.getDim() >= 1);
  checkOutput(out, input.getShape(), input.getDType(), "softmax");
  if (input.getNumEl() == 0) return;

  Contiguous x(input);
  dispatchRows(kernelName("softmax", x.getDType()).c_str(), x, out, 0);
}

bool all(const TensorView &input) {
  if (input.getDType() != DType::kBool) throw lut::InvalidArgError("all takes a bool tensor");
  if (input.getNumEl() == 0) return true;

  Contiguous packed(input);
  TensorView x = packed.view({1, -1});
  Tensor output = createTensor({1}, DType::kBool);
  dispatchRows("reduce_bool", x, output, 0);
  return elemBool(output);
}

float elem(const TensorView &tensor) {
  if (tensor.getNumEl() != 1) {
    throw lut::InvalidArgError("elem takes a tensor of exactly one element");
  }

  Tensor wide;
  TensorView x = tensor;
  if (tensor.getDType() != DType::kFloat) {
    wide = createTensor(tensor.getShape(), DType::kFloat);
    copy(tensor, wide);
    x = wide;
  }
  float value = 0.0f;
  getContext(x)->download(getBuffer(x), getByteOffset(x), &value, sizeof(float));
  return value;
}

bool elemBool(const TensorView &tensor) {
  if (tensor.getNumEl() != 1 || tensor.getDType() != DType::kBool) {
    throw lut::InvalidArgError("elemBool takes a bool tensor of exactly one element");
  }

  uint8_t value = 0;
  getContext(tensor)->download(getBuffer(tensor), getByteOffset(tensor), &value, 1);
  return value != 0;
}

}  // namespace vulkan
}  // namespace op
}  // namespace fl
