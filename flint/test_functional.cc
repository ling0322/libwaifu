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

#include "flint/test_functional.h"

#include <limits>
#include <string>
#include <vector>

#include "lutil/error.h"
#include "lutil/strings.h"
#include "flint/operators.h"

namespace fl {
namespace F {

namespace {

/// `input`'s shape with dimension `dim` -- already non-negative -- dropped. A tensor with nothing
/// left is (1), which is what every reduction of a vector has always given.
std::vector<int> withoutDim(TensorView input, int dim) {
  std::vector<int> shape = input.getShape();
  shape.erase(shape.begin() + dim);
  if (shape.empty()) shape.push_back(1);
  return shape;
}

int realDim(TensorView input, int dim, const char *name) {
  int rank = input.getDim();
  if (dim < -rank || dim >= rank) {
    THROW(InvalidArg, lut::sprintf("%s: no dimension %d in a %d-D tensor", name, dim, rank));
  }
  return dim < 0 ? dim + rank : dim;
}

/// `other` seen at `input`'s shape: `other` may have fewer dimensions, and a dimension of one
/// where `input` has more. Only `other` grows; `input`'s shape is the result's.
TensorView broadcastTo(TensorView other, TensorView input, const char *name) {
  if (other.getDim() > input.getDim()) {
    THROW(
        InvalidArg,
        lut::sprintf(
            "%s: %s cannot be broadcast to %s",
            name,
            other.getShapeString(),
            input.getShapeString()));
  }

  while (other.getDim() < input.getDim()) other = other.unsqueeze(0);
  for (int d = 0; d < input.getDim(); ++d) {
    if (other.getShape(d) != input.getShape(d) && other.getShape(d) != 1) {
      THROW(
          InvalidArg,
          lut::sprintf(
              "%s: %s cannot be broadcast to %s",
              name,
              other.getShapeString(),
              input.getShapeString()));
    }
  }

  std::vector<int> shape = input.getShape();
  return other.expand(shape);
}

/// A result of `input`'s shape and type, the way every elementwise operator makes one.
Tensor likeInput(TensorView input) {
  return emptyLike(input);
}

/// The length a convolution leaves, refused below one.
int convolvedLength(int length, int kernel, int stride, int padding, int dilation, const char *name) {
  int out = (length + 2 * padding - dilation * (kernel - 1) - 1) / stride + 1;
  if (out < 1) {
    THROW(InvalidArg, lut::sprintf("%s: the output would be %d long", name, out));
  }
  return out;
}

void checkConvolution(int stride, int padding, int dilation, int groups, const char *name) {
  if (groups < 1) THROW(InvalidArg, lut::sprintf("%s: groups must be at least 1", name));
  if (stride < 1 || dilation < 1) {
    THROW(InvalidArg, lut::sprintf("%s: stride and dilation must be at least 1", name));
  }
  if (padding < 0) THROW(InvalidArg, lut::sprintf("%s: padding cannot be negative", name));
}

}  // namespace

Tensor zeros(Device device, lut::Span<const int> shape, DType dtype) {
  Tensor tensor = empty(device, shape, dtype);
  getOperators(device.getType())->fill(tensor, 0.0f);
  return tensor;
}

Tensor rand(Device device, lut::Span<const int> shape, DType dtype) {
  Tensor tensor = empty(device, shape, dtype);
  getOperators(device.getType())->rand(tensor);
  return tensor;
}

Tensor randNormal(Device device, lut::Span<const int> shape) {
  Tensor tensor = empty(device, shape, DType::kFloat);
  getOperators(device.getType())->randNormal(tensor);
  return tensor;
}

Tensor arangeLong(Device device, LongType begin, LongType end, LongType step) {
  if (step == 0) THROW(InvalidArg, "arange: the step cannot be zero");
  LongType numel = (end - begin) / step;
  if (numel < 0 || numel >= std::numeric_limits<int32_t>::max()) {
    THROW(
        InvalidArg,
        lut::sprintf("arange: %d elements from %d to %d by %d", numel, begin, end, step));
  }

  Tensor tensor = empty(device, {static_cast<int>(numel)}, DType::kLong);
  if (numel > 0) getOperators(device.getType())->arangeLong(begin, step, tensor);
  return tensor;
}

Tensor causalMask(Device device, int n) {
  if (n < 1) THROW(InvalidArg, lut::sprintf("causalMask: a mask of %d positions", n));
  Operators *op = getOperators(device.getType());
  Tensor mask = empty(device, {n, n}, op->getDefaultFloatType());
  op->causalMask(mask);
  return mask;
}

// --- Moving and reshaping -------------------------------------------------------------------

Tensor toDevice(Operators *op, const Tensor &input, Device device) {
  if (input.getDevice().getType() == device.getType()) return input;

  // A transfer is a plain copy of one run of bytes, so a strided input is packed first, on the
  // host side when it is on the host -- page-locked memory has no operators of its own, and the
  // CPU's can read it.
  Tensor source = input;
  if (!source.isContiguous()) {
    Device::Type side =
        input.getDevice().isHost() ? Device::kCpu : input.getDevice().getType();
    source = contiguous(getOperators(side), source);
  }

  Tensor dest = empty(device, source.getShape(), source.getDType());
  op->transfer(source, dest);
  return dest;
}

Tensor cast(Operators *op, const Tensor &input, DType dtype) {
  if (input.getDType() == dtype) return input;

  Tensor out = empty(input.getDevice(), input.getShape(), dtype);
  op->cast(input, out);
  return out;
}

Tensor contiguous(Operators *op, const Tensor &input) {
  if (input.isContiguous()) return input;

  Tensor out = emptyLike(input);
  op->copy(input, out);
  return out;
}

Tensor cat(Operators *op, TensorView A, TensorView B, int dim) {
  if (A.getDType() != B.getDType()) THROW(InvalidArg, "cat: the two halves differ in type");
  if (A.getDim() != B.getDim()) THROW(InvalidArg, "cat: the two halves differ in rank");
  dim = realDim(A, dim, "cat");
  for (int d = 0; d < A.getDim(); ++d) {
    if (d != dim && A.getShape(d) != B.getShape(d)) {
      THROW(
          InvalidArg,
          lut::sprintf(
              "cat: %s and %s differ in more than dimension %d",
              A.getShapeString(),
              B.getShapeString(),
              dim));
    }
  }

  std::vector<int> shape = A.getShape();
  int dA = A.getShape(dim);
  int dB = B.getShape(dim);
  shape[dim] = dA + dB;

  Tensor C = empty(A.getDevice(), shape, A.getDType());
  op->copy(A, C.slice(dim, {0, dA}));
  op->copy(B, C.slice(dim, {dA, dA + dB}));
  return C;
}

// --- Elementwise ----------------------------------------------------------------------------

#define FL_BINARY(name)                                                \
  Tensor name(Operators *op, TensorView input, TensorView other) {     \
    if (input.getDType() != other.getDType()) {                        \
      THROW(InvalidArg, #name ": the two operands differ in type");    \
    }                                                                  \
    TensorView broadcast = broadcastTo(other, input, #name);           \
    Tensor out = likeInput(input);                                     \
    op->name(input, broadcast, out);                                   \
    return out;                                                        \
  }

FL_BINARY(add)
FL_BINARY(sub)
FL_BINARY(mul)
FL_BINARY(divTensor)

#undef FL_BINARY

Tensor eq(Operators *op, TensorView input, TensorView other) {
  if (input.getDType() != other.getDType()) THROW(InvalidArg, "eq: the two operands differ in type");
  other.throwIfInvalidShape(input.getShape(), "eq");

  Tensor out = empty(input.getDevice(), input.getShape(), DType::kBool);
  op->eq(input, other, out);
  return out;
}

#define FL_SCALAR(name, type)                                \
  Tensor name(Operators *op, TensorView input, type other) { \
    Tensor out = likeInput(input);                           \
    op->name(input, other, out);                             \
    return out;                                              \
  }

FL_SCALAR(mul, float)
FL_SCALAR(div, float)

#undef FL_SCALAR

Tensor mod(Operators *op, TensorView input, LongType other) {
  if (other == 0) THROW(InvalidArg, "mod: by zero");
  Tensor out = likeInput(input);
  op->mod(input, other, out);
  return out;
}

#define FL_UNARY(name)                               \
  Tensor name(Operators *op, TensorView input) {     \
    Tensor out = likeInput(input);                   \
    op->name(input, out);                            \
    return out;                                      \
  }

FL_UNARY(square)
FL_UNARY(neg)
FL_UNARY(abs)
FL_UNARY(exp)
FL_UNARY(log)
FL_UNARY(round)
FL_UNARY(sqrt)
FL_UNARY(rsqrt)
FL_UNARY(sigmoid)
FL_UNARY(tanh)
FL_UNARY(relu)
FL_UNARY(gelu)
FL_UNARY(silu)
FL_UNARY(sin)
FL_UNARY(cos)
FL_UNARY(quickGelu)

#undef FL_UNARY

namespace {

/// The half a gated linear unit leaves: the last dimension halved.
Tensor gatedHalf(TensorView input, const char *name) {
  if (input.getDim() < 1 || input.getShape(-1) % 2 != 0) {
    THROW(
        InvalidArg,
        lut::sprintf("%s: the last dimension of %s is not even", name, input.getShapeString()));
  }
  std::vector<int> shape = input.getShape();
  shape.back() /= 2;
  return empty(input.getDevice(), shape, input.getDType());
}

}  // namespace

Tensor swiglu(Operators *op, TensorView input) {
  Tensor out = gatedHalf(input, "swiglu");
  op->swiglu(input, out);
  return out;
}

Tensor geglu(Operators *op, TensorView input) {
  Tensor out = gatedHalf(input, "geglu");
  op->geglu(input, out);
  return out;
}

Tensor snake(Operators *op, TensorView input, TensorView alpha, TensorView beta, float eps) {
  Tensor out = likeInput(input);
  op->snake(input, alpha, beta, eps, out);
  return out;
}

// --- Reductions -----------------------------------------------------------------------------

Tensor sum(Operators *op, TensorView input, int dim) {
  dim = realDim(input, dim, "sum");
  Tensor out = empty(input.getDevice(), withoutDim(input, dim), input.getDType());
  op->sum(input, dim, out);
  return out;
}

Tensor cumsum(Operators *op, TensorView input, int dim) {
  dim = realDim(input, dim, "cumsum");
  Tensor out = likeInput(input);
  op->cumsum(input, dim, out);
  return out;
}

Tensor max(Operators *op, TensorView input) {
  Tensor out = empty(input.getDevice(), withoutDim(input, input.getDim() - 1), input.getDType());
  op->max(input, out);
  return out;
}

Tensor min(Operators *op, TensorView input) {
  Tensor out = empty(input.getDevice(), withoutDim(input, input.getDim() - 1), input.getDType());
  op->min(input, out);
  return out;
}

Tensor softmax(Operators *op, TensorView input) {
  Tensor out = likeInput(input);
  op->softmax(input, out);
  return out;
}

// --- Normalization and resampling -----------------------------------------------------------

Tensor rmsNorm(Operators *op, TensorView input, TensorView weight, float eps) {
  Tensor out = likeInput(input);
  op->rmsNorm(input, weight, eps, out);
  return out;
}

Tensor layerNorm(Operators *op, TensorView input, TensorView weight, TensorView bias, float eps) {
  Tensor out = likeInput(input);
  op->layerNorm(input, weight, bias, eps, out);
  return out;
}

Tensor groupNorm(
    Operators *op,
    TensorView input,
    TensorView weight,
    TensorView bias,
    int groups,
    float eps) {
  Tensor out = likeInput(input);
  op->groupNorm(input, weight, bias, groups, eps, out);
  return out;
}

Tensor upsampleNearest2d(Operators *op, TensorView input, int scale) {
  if (input.getDim() != 4) THROW(InvalidArg, "upsampleNearest2d: the input is not 4-D");
  if (scale < 1) THROW(InvalidArg, "upsampleNearest2d: the scale must be at least 1");

  Tensor out = empty(
      input.getDevice(),
      {input.getShape(0), input.getShape(1), input.getShape(2) * scale, input.getShape(3) * scale},
      input.getDType());
  op->upsampleNearest2d(input, scale, out);
  return out;
}

Tensor upsampleNearest1d(Operators *op, TensorView input, int size) {
  if (input.getDim() < 1) THROW(InvalidArg, "upsampleNearest1d: the input has no dimensions");
  if (size < 1) THROW(InvalidArg, "upsampleNearest1d: the size must be at least 1");

  std::vector<int> shape = input.getShape();
  shape.back() = size;
  Tensor out = empty(input.getDevice(), shape, input.getDType());
  op->upsampleNearest1d(input, out);
  return out;
}

// --- Products and convolutions --------------------------------------------------------------

Tensor matmul(Operators *op, TensorView A, TensorView B) {
  if (A.getDim() < 2 || B.getDim() < 2) THROW(InvalidArg, "matmul: both operands need 2 dimensions");
  if (A.getShape(-1) != B.getShape(-2)) {
    THROW(
        InvalidArg,
        lut::sprintf("matmul: %s by %s", A.getShapeString(), B.getShapeString()));
  }

  // A batch of B, if B has one, is A's: B may have fewer dimensions, and is seen at A's batch.
  if (B.getDim() > 2) {
    if (A.getDim() < B.getDim()) {
      THROW(
          InvalidArg,
          lut::sprintf("matmul: %s by %s", A.getShapeString(), B.getShapeString()));
    }
    int offset = A.getDim() - B.getDim();
    for (int d = 0; d < B.getDim() - 2; ++d) {
      if (B.getShape(d) != A.getShape(d + offset)) {
        THROW(
            InvalidArg,
            lut::sprintf("matmul: the batches of %s and %s differ", A.getShapeString(),
                         B.getShapeString()));
      }
    }
  }

  std::vector<int> shape = A.getShape();
  shape.back() = B.getShape(-1);
  Tensor out = empty(A.getDevice(), shape, A.getDType());
  op->matmul(A, B, out);
  return out;
}

Tensor conv2d(
    Operators *op,
    TensorView input,
    TensorView weight,
    TensorView bias,
    int stride,
    int padding,
    int dilation,
    int groups) {
  if (input.getDim() != 4 || weight.getDim() != 4) {
    THROW(InvalidArg, "conv2d: the input and the weight must be 4-D");
  }
  checkConvolution(stride, padding, dilation, groups, "conv2d");

  int outH = convolvedLength(input.getShape(2), weight.getShape(2), stride, padding, dilation,
                             "conv2d");
  int outW = convolvedLength(input.getShape(3), weight.getShape(3), stride, padding, dilation,
                             "conv2d");
  Tensor out = empty(
      input.getDevice(),
      {input.getShape(0), weight.getShape(0), outH, outW},
      input.getDType());
  op->conv2d(input, weight, bias, stride, padding, dilation, groups, out);
  return out;
}

Tensor conv1d(
    Operators *op,
    TensorView input,
    TensorView weight,
    TensorView bias,
    int stride,
    int padding,
    int dilation,
    int groups) {
  if (input.getDim() != 3 || weight.getDim() != 3) {
    THROW(InvalidArg, "conv1d: the input and the weight must be 3-D");
  }
  checkConvolution(stride, padding, dilation, groups, "conv1d");

  int length = convolvedLength(input.getShape(2), weight.getShape(2), stride, padding, dilation,
                               "conv1d");
  Tensor out =
      empty(input.getDevice(), {input.getShape(0), weight.getShape(0), length}, input.getDType());
  op->conv1d(input, weight, bias, stride, padding, dilation, groups, out);
  return out;
}

Tensor convTranspose1d(
    Operators *op,
    TensorView input,
    TensorView weight,
    TensorView bias,
    int stride,
    int padding,
    int outputPadding,
    int groups) {
  if (input.getDim() != 3 || weight.getDim() != 3) {
    THROW(InvalidArg, "convTranspose1d: the input and the weight must be 3-D");
  }
  checkConvolution(stride, padding, 1, groups, "convTranspose1d");

  int length = (input.getShape(2) - 1) * stride - 2 * padding + weight.getShape(2) + outputPadding;
  if (length < 1) {
    THROW(InvalidArg, lut::sprintf("convTranspose1d: the output would be %d long", length));
  }
  Tensor out = empty(
      input.getDevice(),
      {input.getShape(0), weight.getShape(1) * groups, length},
      input.getDType());
  op->convTranspose1d(input, weight, bias, stride, padding, outputPadding, groups, out);
  return out;
}

Tensor stft(Operators *op, TensorView input, TensorView window, int nFft, int hop, bool centered) {
  if (input.getDim() != 3) THROW(InvalidArg, "stft: the input must be (N, 1, L)");
  if (nFft < 1 || hop < 1) THROW(InvalidArg, "stft: nFft and hop must be at least 1");

  int length = input.getShape(2) + (centered ? 2 * (nFft / 2) : 0);
  if (length < nFft) THROW(InvalidArg, "stft: the input is shorter than one window");
  int frames = (length - nFft) / hop + 1;
  Tensor out = empty(
      input.getDevice(),
      {input.getShape(0), 2 * (nFft / 2 + 1), frames},
      input.getDType());
  op->stft(input, window, nFft, hop, centered, out);
  return out;
}

Tensor istft(
    Operators *op,
    TensorView spectrum,
    TensorView window,
    int nFft,
    int hop,
    bool centered) {
  if (spectrum.getDim() != 3) THROW(InvalidArg, "istft: the spectrum must be (N, bins, frames)");
  if (nFft < 1 || hop < 1) THROW(InvalidArg, "istft: nFft and hop must be at least 1");

  int length = nFft + hop * (spectrum.getShape(2) - 1) - (centered ? 2 * (nFft / 2) : 0);
  if (length < 1) THROW(InvalidArg, "istft: the spectrum is too short to invert");
  Tensor out = empty(spectrum.getDevice(), {spectrum.getShape(0), 1, length}, spectrum.getDType());
  op->istft(spectrum, window, nFft, hop, centered, out);
  return out;
}

Tensor lookup(Operators *op, TensorView table, TensorView indices) {
  if (table.getDim() != 2) THROW(InvalidArg, "lookup: the table must be 2-D");

  std::vector<int> shape = indices.getShape();
  shape.push_back(table.getShape(1));
  Tensor out = empty(table.getDevice(), shape, table.getDType());
  op->lookup(table, indices, out);
  return out;
}

// --- Attention and sampling -----------------------------------------------------------------

Tensor attention(Operators *op, TensorView q, TensorView k, TensorView v, bool causal) {
  if (q.getDim() != 4 || k.getDim() != 4 || v.getDim() != 4) {
    THROW(InvalidArg, "attention: q, k and v must be 4-D");
  }
  if (q.getShape(1) % k.getShape(1) != 0) {
    THROW(InvalidArg, "attention: the query heads are not a multiple of the key-value heads");
  }

  Tensor out = empty(
      q.getDevice(),
      {q.getShape(0), q.getShape(1), q.getShape(2), v.getShape(3)},
      q.getDType());
  op->attention(q, k, v, causal, out);
  return out;
}

Tensor pagedAttention(
    Operators *op,
    TensorView q,
    TensorView keyCache,
    TensorView valueCache,
    TensorView blockTable,
    TensorView cuSeqlensQ,
    TensorView seqlensK,
    int maxQLen,
    int maxKLen,
    bool causal) {
  if (q.getDim() != 3) THROW(InvalidArg, "pagedAttention: q must be (tokens, heads, headDim)");

  Tensor out = emptyLike(q);
  op->pagedAttention(
      q,
      keyCache,
      valueCache,
      blockTable,
      cuSeqlensQ,
      seqlensK,
      maxQLen,
      maxKLen,
      causal,
      out);
  return out;
}

Tensor gatedDeltaNetPrefill(
    Operators *op,
    TensorView q,
    TensorView k,
    TensorView v,
    TensorView g,
    TensorView beta,
    TensorView cuSeqlens,
    TensorView stateSlots,
    TensorView state) {
  if (q.getDim() != 3 || v.getDim() != 3) {
    THROW(InvalidArg, "gatedDeltaNetPrefill: q and v must be (tokens, heads, headDim)");
  }

  Tensor out = empty(v.getDevice(), {q.getShape(0), v.getShape(1), v.getShape(2)}, v.getDType());
  op->gatedDeltaNetPrefill(q, k, v, g, beta, cuSeqlens, stateSlots, state, out);
  return out;
}

Tensor sample(
    Operators *op,
    TensorView logits,
    TensorView temperatures,
    TensorView topKs,
    TensorView topPs) {
  if (logits.getDim() != 2) THROW(InvalidArg, "sample: the logits must be (rows, vocabulary)");

  Tensor out = empty(logits.getDevice(), {logits.getShape(0)}, DType::kLong);
  op->sample(logits, temperatures, topKs, topPs, out);
  return out;
}

}  // namespace F
}  // namespace fl
