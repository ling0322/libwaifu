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

#include <string>
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

uint32_t checkedNumel(const TensorView &tensor) {
  int64_t numel = tensor.getNumEl();
  CHECK(numel >= 0 && numel <= TensorData::MaxNumEl);
  return static_cast<uint32_t>(numel);
}

}  // namespace

void unary(UnaryOp op, const TensorView &input, float scalar, const TensorView &out) {
  checkFloat(input.getDType(), "an elementwise operator");
  checkOutput(out, input.getShape(), input.getDType(), "an elementwise operator");
  if (input.getNumEl() == 0) return;

  Layout layout;
  if (!collapse(input.getShape(), {getStrides(input)}, &layout)) {
    Tensor keep;
    unary(op, makeContiguous(input, &keep), scalar, out);
    return;
  }

  UnaryPush push{};
  push.c = getAddress(out);
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
}

namespace {

void binaryKernel(
    const char *base,
    DType outputType,
    int op,
    const TensorView &a,
    const TensorView &b,
    const TensorView &out) {
  if (a.getDType() != b.getDType()) {
    throw lut::InvalidArgError(lut::sprintf(
        "an elementwise operator on the Vulkan device takes two of one dtype, not %s and %s",
        a.getDType().toString(),
        b.getDType().toString()));
  }
  b.throwIfInvalidShape(a.getShape(), "an elementwise operator");
  checkOutput(out, a.getShape(), outputType, "an elementwise operator");
  if (a.getNumEl() == 0) return;

  Layout layout;
  if (!collapse(a.getShape(), {getStrides(a), getStrides(b)}, &layout)) {
    Tensor keepA, keepB;
    binaryKernel(
        base,
        outputType,
        op,
        makeContiguous(a, &keepA),
        makeContiguous(b, &keepB),
        out);
    return;
  }

  BinaryPush push{};
  push.c = getAddress(out);
  push.a = getAddress(a);
  push.b = getAddress(b);
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
}

}  // namespace

void binary(BinaryOp op, const TensorView &a, const TensorView &b, const TensorView &out) {
  if (a.getDType() != DType::kLong) checkFloat(a.getDType(), "an elementwise operator");
  binaryKernel("binary", a.getDType(), static_cast<int>(op), a, b, out);
}

void eq(const TensorView &a, const TensorView &b, const TensorView &out) {
  DType dtype = a.getDType();
  if (dtype != DType::kFloat && dtype != DType::kFloat16 && dtype != DType::kLong &&
      dtype != DType::kUInt8) {
    throw lut::InvalidArgError(lut::sprintf("eq on the Vulkan device does not take %s",
                                            dtype.toString()));
  }
  binaryKernel("equal", DType::kBool, 0, a, b, out);
}

void mod(const TensorView &input, int64_t other, const TensorView &out) {
  if (input.getDType() != DType::kLong) {
    throw lut::InvalidArgError("mod on the Vulkan device takes int64 only");
  }
  if (other == 0) throw lut::InvalidArgError("mod by zero");
  checkOutput(out, input.getShape(), DType::kLong, "mod");
  if (input.getNumEl() == 0) return;

  Tensor keep;
  TensorView x = makeContiguous(input, &keep);
  ModPush push{};
  push.c = getAddress(out);
  push.a = getAddress(x);
  push.other = other;
  push.numel = checkedNumel(x);
  getContext(x)->dispatchLinear("mod_i64", &push, sizeof(push), push.numel);
}

void fill(const TensorView &tensor, float value) {
  if (tensor.getNumEl() == 0) return;

  // Zero is zero bits in every type there is, so a contiguous run of whole words is cleared
  // rather than filled.
  if (value == 0.0f && tensor.isContiguous()) {
    int64_t bytes = tensor.getDType().getTotalSize(tensor.getNumEl());
    int64_t offset = getByteOffset(tensor);
    if (bytes % 4 == 0 && offset % 4 == 0) {
      getContext(tensor)->fillBuffer(getBuffer(tensor), offset, bytes, 0);
      return;
    }
  }

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

void copyKernel(const TensorView &src, const TensorView &dest) {
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

void copy(const TensorView &src, const TensorView &dest) {
  src.throwIfInvalidShape(dest.getShape(), "copy");
  if (src.getNumEl() == 0) return;
  copyKernel(src, dest);
}

void cast(const TensorView &input, const TensorView &out) {
  checkOutput(out, input.getShape(), out.getDType(), "cast");
  copy(input, out);
}

void arangeLong(int64_t begin, int64_t step, const TensorView &out) {
  if (out.getDim() != 1) throw lut::InvalidArgError("arangeLong writes a vector");
  checkOutput(out, out.getShape(), DType::kLong, "arangeLong");
  if (out.getNumEl() == 0) return;

  ArangePush push{};
  push.c = getAddress(out);
  push.begin = begin;
  push.step = step;
  push.numel = checkedNumel(out);
  getContext(out)->dispatchLinear("arange_i64", &push, sizeof(push), push.numel);
}

void causalMask(const TensorView &out) {
  if (out.getDim() != 2 || out.getShape(0) != out.getShape(1)) {
    throw lut::InvalidArgError("causalMask writes a square matrix");
  }
  checkFloat(out.getDType(), "causalMask");
  checkOutput(out, out.getShape(), out.getDType(), "causalMask");
  int length = out.getShape(0);

  // Built once on the host and copied up: it is small, asked for rarely, and a kernel that wrote
  // it would be one more thing to keep in step with the CPU's.
  std::vector<float> mask(static_cast<size_t>(length) * length);
  for (int y = 0; y < length; ++y) {
    for (int x = 0; x < length; ++x) {
      mask[static_cast<size_t>(y) * length + x] = x > y ? -INFINITY : 0.0f;
    }
  }

  // Uploaded as float, and converted on the device when `out` is of another type.
  Tensor wide;
  TensorView target = out;
  if (out.getDType() != DType::kFloat) {
    wide = createTensor(out.getShape(), DType::kFloat);
    target = wide;
  }
  getContext(target)->upload(
      mask.data(),
      getBuffer(target),
      getByteOffset(target),
      static_cast<int64_t>(mask.size() * sizeof(float)));
  if (!wide.empty()) copy(wide, out);
}

}  // namespace vulkan
}  // namespace op
}  // namespace fl
