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

#include <math.h>

#include "lutil/error.h"
#include "lutil/log.h"
#include "lutil/strings.h"
#include "flint/vulkan/common.h"
#include "flint/vulkan/ops.h"

namespace fl {
namespace op {
namespace vulkan {

namespace {

// Keep each of these in step with the push constant block of the shader it is named after.

struct UnaryPush {
  uint64_t c;
  uint64_t a;
  uint32_t numel;
  uint32_t ndim;
  uint32_t op;
  float scalar;
  uint32_t shape[kMaxDims];
  uint32_t strideA[kMaxDims];
};

struct BinaryPush {
  uint64_t c;
  uint64_t a;
  uint64_t b;
  uint32_t numel;
  uint32_t ndim;
  uint32_t op;
  uint32_t pad;
  uint32_t shape[kMaxDims];
  uint32_t strideA[kMaxDims];
  uint32_t strideB[kMaxDims];
};

struct CopyPush {
  uint64_t c;
  uint64_t a;
  uint32_t numel;
  uint32_t ndim;
  uint32_t shape[kMaxDims];
  uint32_t strideA[kMaxDims];
  uint32_t strideC[kMaxDims];
};

struct FillPush {
  uint64_t c;
  uint32_t numel;
  uint32_t ndim;
  float value;
  uint32_t shape[kMaxDims];
  uint32_t strideC[kMaxDims];
};

struct ModPush {
  uint64_t c;
  uint64_t a;
  int64_t other;
  uint32_t numel;
};

struct ArangePush {
  uint64_t c;
  int64_t begin;
  int64_t step;
  uint32_t numel;
};

uint32_t checkedNumel(const Tensor &tensor) {
  int64_t numel = tensor.getNumEl();
  CHECK(numel >= 0 && numel <= TensorData::MaxNumEl);
  return static_cast<uint32_t>(numel);
}

// `other` given A's rank, by prepending broadcast dimensions, and then expanded to A's shape:
// the one-directional broadcast the other devices do, where only the right operand grows.
Tensor broadcastTo(const Tensor &other, const Tensor &a) {
  if (other.getDim() > a.getDim()) {
    throw lut::InvalidArgError(lut::sprintf(
        "unable to broadcast %s into %s",
        other.getShapeString(),
        a.getShapeString()));
  }

  std::vector<TensorShape::Elem> elems;
  for (int d = 0; d < a.getDim() - other.getDim(); ++d) elems.push_back({a.getShape(d), 0});
  for (int d = 0; d < other.getDim(); ++d) {
    elems.push_back({other.getShape(d), other.getStride(d)});
  }

  Tensor expanded = Tensor::create(
      std::make_shared<TensorShape>(lut::makeConstSpan(elems)),
      other.getInternalData(),
      other.getInternalOffset());
  expanded = expanded.expand(a.getShape());
  expanded.throwIfInvalidShape(a.getShape(), "broadcast");
  return expanded;
}

}  // namespace

Tensor unary(UnaryOp op, const Tensor &input, float scalar) {
  checkFloat(input.getDType(), "an elementwise operator");

  Layout layout;
  if (!collapse(input.getShape(), {getStrides(input)}, &layout)) {
    return unary(op, makeContiguous(input), scalar);
  }

  Tensor output = createTensor(input.getShape(), input.getDType());
  UnaryPush push{};
  push.c = getAddress(output);
  push.a = getAddress(input);
  push.numel = checkedNumel(input);
  push.ndim = layout.ndim;
  push.op = static_cast<uint32_t>(op);
  push.scalar = scalar;
  for (int d = 0; d < kMaxDims; ++d) {
    push.shape[d] = layout.shape[d];
    push.strideA[d] = layout.strides[0][d];
  }

  getContext(input)->dispatchLinear(
      kernelName("unary", input.getDType()).c_str(),
      &push,
      sizeof(push),
      push.numel);
  return output;
}

namespace {

Tensor binaryKernel(const char *base, DType outputType, int op, const Tensor &a, const Tensor &b) {
  if (a.getDType() != b.getDType()) {
    throw lut::InvalidArgError(lut::sprintf(
        "an elementwise operator on the Vulkan device takes two of one dtype, not %s and %s",
        a.getDType().toString(),
        b.getDType().toString()));
  }

  Tensor expanded = broadcastTo(b, a);
  Layout layout;
  if (!collapse(a.getShape(), {getStrides(a), getStrides(expanded)}, &layout)) {
    return binaryKernel(base, outputType, op, makeContiguous(a), makeContiguous(expanded));
  }

  Tensor output = createTensor(a.getShape(), outputType);
  BinaryPush push{};
  push.c = getAddress(output);
  push.a = getAddress(a);
  push.b = getAddress(expanded);
  push.numel = checkedNumel(a);
  push.ndim = layout.ndim;
  push.op = static_cast<uint32_t>(op);
  for (int d = 0; d < kMaxDims; ++d) {
    push.shape[d] = layout.shape[d];
    push.strideA[d] = layout.strides[0][d];
    push.strideB[d] = layout.strides[1][d];
  }

  getContext(a)->dispatchLinear(
      kernelName(base, a.getDType()).c_str(),
      &push,
      sizeof(push),
      push.numel);
  return output;
}

}  // namespace

Tensor binary(BinaryOp op, const Tensor &a, const Tensor &b) {
  if (a.getDType() != DType::kLong) checkFloat(a.getDType(), "an elementwise operator");
  return binaryKernel("binary", a.getDType(), static_cast<int>(op), a, b);
}

Tensor eq(const Tensor &a, const Tensor &b) {
  DType dtype = a.getDType();
  if (dtype != DType::kFloat && dtype != DType::kFloat16 && dtype != DType::kLong &&
      dtype != DType::kUInt8) {
    throw lut::InvalidArgError(lut::sprintf("eq on the Vulkan device does not take %s",
                                            dtype.toString()));
  }
  return binaryKernel("equal", DType::kBool, 0, a, b);
}

Tensor mod(const Tensor &input, int64_t other) {
  if (input.getDType() != DType::kLong) {
    throw lut::InvalidArgError("mod on the Vulkan device takes int64 only");
  }
  if (other == 0) throw lut::InvalidArgError("mod by zero");

  Tensor x = makeContiguous(input);
  Tensor output = createTensor(x.getShape(), DType::kLong);
  ModPush push{};
  push.c = getAddress(output);
  push.a = getAddress(x);
  push.other = other;
  push.numel = checkedNumel(x);
  getContext(x)->dispatchLinear("mod_i64", &push, sizeof(push), push.numel);
  return output;
}

void fill(const Tensor &tensor, float value) {
  if (tensor.getNumEl() == 0) return;

  Layout layout;
  if (!collapse(tensor.getShape(), {getStrides(tensor)}, &layout)) {
    // Too many dimensions that do not merge: fill the leading one a slice at a time.
    for (int i = 0; i < tensor.getShape(0); ++i) fill(tensor.subtensor(i), value);
    return;
  }

  FillPush push{};
  push.c = getAddress(tensor);
  push.numel = checkedNumel(tensor);
  push.ndim = layout.ndim;
  push.value = value;
  for (int d = 0; d < kMaxDims; ++d) {
    push.shape[d] = layout.shape[d];
    push.strideC[d] = layout.strides[0][d];
  }

  getContext(tensor)->dispatchLinear(
      kernelName("fill", tensor.getDType()).c_str(),
      &push,
      sizeof(push),
      push.numel);
}

namespace {

void copyKernel(const Tensor &src, const Tensor &dest) {
  // Contiguous both ends and of one type is a plain buffer copy, which is as fast as the device
  // moves memory -- and the only copy there is for a type no kernel is built for.
  if (src.getDType() == dest.getDType() && src.isContiguous() && dest.isContiguous()) {
    int64_t bytes = src.getDType().getTotalSize(src.getNumEl());
    getContext(src)->copyBuffer(
        getBuffer(src),
        getByteOffset(src),
        getBuffer(dest),
        getByteOffset(dest),
        bytes);
    return;
  }

  Layout layout;
  if (!collapse(src.getShape(), {getStrides(src), getStrides(dest)}, &layout)) {
    for (int i = 0; i < src.getShape(0); ++i) copyKernel(src.subtensor(i), dest.subtensor(i));
    return;
  }

  CopyPush push{};
  push.c = getAddress(dest);
  push.a = getAddress(src);
  push.numel = checkedNumel(src);
  push.ndim = layout.ndim;
  for (int d = 0; d < kMaxDims; ++d) {
    push.shape[d] = layout.shape[d];
    push.strideA[d] = layout.strides[0][d];
    push.strideC[d] = layout.strides[1][d];
  }

  std::string name = "copy_" + getTypeSuffix(src.getDType()) + "_" +
                     getTypeSuffix(dest.getDType());
  getContext(src)->dispatchLinear(name.c_str(), &push, sizeof(push), push.numel);
}

}  // namespace

void copy(const Tensor &src, const Tensor &dest) {
  src.throwIfInvalidShape(dest.getShape(), "copy");
  if (src.getNumEl() == 0) return;
  copyKernel(src, dest);
}

Tensor cast(const Tensor &input, DType dtype) {
  if (input.getDType() == dtype) return input;

  Tensor output = createTensor(input.getShape(), dtype);
  copy(input, output);
  return output;
}

Tensor arangeLong(int64_t begin, int64_t end, int64_t step) {
  if (step == 0) throw lut::InvalidArgError("arange with a step of zero");
  int64_t numel = std::max<int64_t>(0, (end - begin) / step);
  CHECK(numel <= TensorData::MaxNumEl);

  Tensor output = createTensor({static_cast<int>(numel)}, DType::kLong);
  ArangePush push{};
  push.c = getAddress(output);
  push.begin = begin;
  push.step = step;
  push.numel = static_cast<uint32_t>(numel);
  getContext(output)->dispatchLinear("arange_i64", &push, sizeof(push), push.numel);
  return output;
}

Tensor causalMask(int length, DType dtype) {
  // Built once on the host and copied up: it is small, asked for rarely, and a kernel that wrote
  // it would be one more thing to keep in step with the CPU's.
  std::vector<float> mask(static_cast<size_t>(length) * length);
  for (int y = 0; y < length; ++y) {
    for (int x = 0; x < length; ++x) {
      mask[static_cast<size_t>(y) * length + x] = x > y ? -INFINITY : 0.0f;
    }
  }

  Tensor output = createTensor({length, length}, DType::kFloat);
  getContext(output)->upload(
      mask.data(),
      getBuffer(output),
      getByteOffset(output),
      static_cast<int64_t>(mask.size() * sizeof(float)));
  return cast(output, dtype);
}

}  // namespace vulkan
}  // namespace op
}  // namespace fl
