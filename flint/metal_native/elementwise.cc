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
#include "flint/metal_native/common.h"
#include "flint/metal_native/ops.h"

namespace fl {
namespace op {
namespace metal_native {

namespace {

// Keep each of these in step with the struct of the same name in kernels/elementwise.metal.

struct UnaryParams {
  uint32_t numel;
  uint32_t ndim;
  uint32_t op;
  float scalar;
  uint32_t shape[kMaxDims];
  uint32_t strideA[kMaxDims];
};

struct BinaryParams {
  uint32_t numel;
  uint32_t ndim;
  uint32_t op;
  uint32_t pad;
  uint32_t shape[kMaxDims];
  uint32_t strideA[kMaxDims];
  uint32_t strideB[kMaxDims];
};

struct CopyParams {
  uint32_t numel;
  uint32_t ndim;
  uint32_t shape[kMaxDims];
  uint32_t strideA[kMaxDims];
  uint32_t strideC[kMaxDims];
};

struct FillParams {
  uint32_t numel;
  uint32_t ndim;
  float value;
  uint32_t shape[kMaxDims];
  uint32_t strideC[kMaxDims];
};

struct ModParams {
  int64_t other;
  uint32_t numel;
};

struct ArangeParams {
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
    unary(op, Contiguous(input), scalar, out);
    return;
  }

  UnaryParams params{};
  params.numel = checkedNumel(input);
  params.ndim = layout.ndim;
  params.op = static_cast<uint32_t>(op);
  params.scalar = scalar;
  for (int d = 0; d < kMaxDims; ++d) {
    params.shape[d] = layout.shape[d];
    params.strideA[d] = layout.strides[0][d];
  }

  getContext(input)->dispatch(
      kernelName("unary", input.getDType()).c_str(),
      {getArg(out), getArg(input)},
      &params,
      sizeof(params),
      params.numel);
}

namespace {

void binaryKernel(
    const std::string &name,
    DType outputType,
    int op,
    const TensorView &a,
    const TensorView &b,
    const TensorView &out) {
  if (a.getDType() != b.getDType()) {
    throw lut::InvalidArgError(lut::sprintf(
        "an elementwise operator on the Metal device takes two of one dtype, not %s and %s",
        a.getDType().toString(),
        b.getDType().toString()));
  }
  b.throwIfInvalidShape(a.getShape(), "an elementwise operator");
  checkOutput(out, a.getShape(), outputType, "an elementwise operator");
  if (a.getNumEl() == 0) return;

  Layout layout;
  if (!collapse(a.getShape(), {getStrides(a), getStrides(b)}, &layout)) {
    binaryKernel(name, outputType, op, Contiguous(a), Contiguous(b), out);
    return;
  }

  BinaryParams params{};
  params.numel = checkedNumel(a);
  params.ndim = layout.ndim;
  params.op = static_cast<uint32_t>(op);
  for (int d = 0; d < kMaxDims; ++d) {
    params.shape[d] = layout.shape[d];
    params.strideA[d] = layout.strides[0][d];
    params.strideB[d] = layout.strides[1][d];
  }

  getContext(a)->dispatch(
      name.c_str(),
      {getArg(out), getArg(a), getArg(b)},
      &params,
      sizeof(params),
      params.numel);
}

}  // namespace

void binary(BinaryOp op, const TensorView &a, const TensorView &b, const TensorView &out) {
  if (a.getDType() != DType::kLong) checkFloat(a.getDType(), "an elementwise operator");
  binaryKernel(kernelName("binary", a.getDType()), a.getDType(), static_cast<int>(op), a, b, out);
}

void eq(const TensorView &a, const TensorView &b, const TensorView &out) {
  DType dtype = a.getDType();
  if (dtype != DType::kFloat && dtype != DType::kFloat16 && dtype != DType::kLong &&
      dtype != DType::kUInt8) {
    throw lut::InvalidArgError(
        lut::sprintf("eq on the Metal device does not take %s", dtype.toString()));
  }
  binaryKernel(kernelName("equal", dtype), DType::kBool, 0, a, b, out);
}

void mod(const TensorView &input, int64_t other, const TensorView &out) {
  if (input.getDType() != DType::kLong) {
    throw lut::InvalidArgError("mod on the Metal device takes int64 only");
  }
  if (other == 0) throw lut::InvalidArgError("mod by zero");
  checkOutput(out, input.getShape(), DType::kLong, "mod");
  if (input.getNumEl() == 0) return;

  Contiguous x(input);
  ModParams params{};
  params.other = other;
  params.numel = checkedNumel(x);
  getContext(x)->dispatch("mod_i64", {getArg(out), getArg(x)}, &params, sizeof(params), params.numel);
}

void fill(const TensorView &tensor, float value) {
  if (tensor.getNumEl() == 0) return;

  Layout layout;
  if (!collapse(tensor.getShape(), {getStrides(tensor)}, &layout)) {
    // Too many dimensions that do not merge: fill the leading one a slice at a time.
    for (int i = 0; i < tensor.getShape(0); ++i) fill(tensor.subtensor(i), value);
    return;
  }

  FillParams params{};
  params.numel = checkedNumel(tensor);
  params.ndim = layout.ndim;
  params.value = value;
  for (int d = 0; d < kMaxDims; ++d) {
    params.shape[d] = layout.shape[d];
    params.strideC[d] = layout.strides[0][d];
  }

  getContext(tensor)->dispatch(
      kernelName("fill", tensor.getDType()).c_str(),
      {getArg(tensor)},
      &params,
      sizeof(params),
      params.numel);
}

namespace {

void copyKernel(const TensorView &src, const TensorView &dest) {
  // Contiguous both ends and of one type is a run of bytes, copied a word at a time where both
  // ends start on one -- and the only copy there is for a type no kernel is built for.
  if (src.getDType() == dest.getDType() && src.isContiguous() && dest.isContiguous()) {
    int64_t bytes = src.getDType().getTotalSize(src.getNumEl());
    Arg from = getArg(src);
    Arg to = getArg(dest);
    bool words = bytes % 4 == 0 && from.offset % 4 == 0 && to.offset % 4 == 0;
    uint32_t count = static_cast<uint32_t>(words ? bytes / 4 : bytes);
    getContext(src)->dispatch(
        words ? "copy_words" : "copy_bytes",
        {to, from},
        &count,
        sizeof(count),
        count);
    return;
  }

  Layout layout;
  if (!collapse(src.getShape(), {getStrides(src), getStrides(dest)}, &layout)) {
    for (int i = 0; i < src.getShape(0); ++i) copyKernel(src.subtensor(i), dest.subtensor(i));
    return;
  }

  CopyParams params{};
  params.numel = checkedNumel(src);
  params.ndim = layout.ndim;
  for (int d = 0; d < kMaxDims; ++d) {
    params.shape[d] = layout.shape[d];
    params.strideA[d] = layout.strides[0][d];
    params.strideC[d] = layout.strides[1][d];
  }

  std::string name = "copy_" + getTypeSuffix(src.getDType()) + "_" +
                     getTypeSuffix(dest.getDType());
  getContext(src)->dispatch(
      name.c_str(),
      {getArg(dest), getArg(src)},
      &params,
      sizeof(params),
      params.numel);
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

  ArangeParams params{};
  params.begin = begin;
  params.step = step;
  params.numel = checkedNumel(out);
  getContext(out)->dispatch("arange_i64", {getArg(out)}, &params, sizeof(params), params.numel);
}

void causalMask(const TensorView &out) {
  if (out.getDim() != 2 || out.getShape(0) != out.getShape(1)) {
    throw lut::InvalidArgError("causalMask writes a square matrix");
  }
  checkFloat(out.getDType(), "causalMask");
  checkOutput(out, out.getShape(), out.getDType(), "causalMask");
  int length = out.getShape(0);

  // Built on the host and copied in: it is small, asked for rarely, and a kernel that wrote it
  // would be one more thing to keep in step with the CPU's.
  std::vector<float> mask(static_cast<size_t>(length) * length);
  for (int y = 0; y < length; ++y) {
    for (int x = 0; x < length; ++x) {
      mask[static_cast<size_t>(y) * length + x] = x > y ? -INFINITY : 0.0f;
    }
  }

  // Copied in as float, and converted on the device when `out` is of another type.
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

}  // namespace metal_native
}  // namespace op
}  // namespace fl
