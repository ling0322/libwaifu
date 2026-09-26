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

struct RowPush {
  uint64_t c;
  uint64_t a;
  uint32_t numRows;
  uint32_t rowLength;
  uint32_t op;
};

// A kernel that gives each row a workgroup of this many threads, as rows.glsl does.
constexpr int kRowThreads = 256;

void dispatchRows(const char *kernel, const Tensor &input, const Tensor &output, uint32_t op) {
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

std::vector<int> withoutLastDim(const Tensor &input) {
  std::vector<int> shape = input.getShape();
  shape.pop_back();
  if (shape.empty()) shape.push_back(1);
  return shape;
}

}  // namespace

Tensor reduceLastDim(const Tensor &input, ReduceOp op) {
  checkFloat(input.getDType(), "a reduction");
  CHECK(input.getDim() >= 1 && input.getShape(-1) > 0);

  Tensor x = makeContiguous(input);
  Tensor output = createTensor(withoutLastDim(x), x.getDType());
  if (x.getNumEl() == 0) return output;

  dispatchRows(kernelName("reduce", x.getDType()).c_str(), x, output, static_cast<uint32_t>(op));
  return output;
}

Tensor sum(const Tensor &input, int dim) {
  if (dim == None) return reduceLastDim(makeContiguous(input).view({1, -1}), ReduceOp::kSum);

  int realDim = input.getInternalShape()->getRealDim(dim);
  if (realDim == input.getDim() - 1) return reduceLastDim(input, ReduceOp::kSum);

  // Any other dimension is moved to the end and summed there, as the CPU does it.
  Tensor moved = input.transpose(realDim, -1);
  Tensor summed = reduceLastDim(moved, ReduceOp::kSum);
  if (input.getDim() == 1) return summed;
  Tensor restored = summed.unsqueeze(summed.getDim()).transpose(realDim, -1).squeeze(realDim);
  return makeContiguous(restored);
}

Tensor softmax(const Tensor &input) {
  checkFloat(input.getDType(), "softmax");
  CHECK(input.getDim() >= 1);

  Tensor x = makeContiguous(input);
  Tensor output = createTensor(x.getShape(), x.getDType());
  if (x.getNumEl() == 0) return output;

  dispatchRows(kernelName("softmax", x.getDType()).c_str(), x, output, 0);
  return output;
}

bool all(const Tensor &input) {
  if (input.getDType() != DType::kBool) throw lut::InvalidArgError("all takes a bool tensor");
  if (input.getNumEl() == 0) return true;

  Tensor x = makeContiguous(input).view({1, -1});
  Tensor output = createTensor({1}, DType::kBool);
  dispatchRows("reduce_bool", x, output, 0);
  return elemBool(output);
}

float elem(const Tensor &tensor) {
  if (tensor.getNumEl() != 1) {
    throw lut::InvalidArgError("elem takes a tensor of exactly one element");
  }

  Tensor x = cast(tensor, DType::kFloat);
  float value = 0.0f;
  getContext(x)->download(getBuffer(x), getByteOffset(x), &value, sizeof(float));
  return value;
}

bool elemBool(const Tensor &tensor) {
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
