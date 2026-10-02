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

#include "flint/functional.h"

#include <math.h>

#include <algorithm>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "lutil/error.h"
#include "lutil/strings.h"
#include "flint/cpu/cpu_tensor_data.h"
#include "flint/operators.h"
#ifdef LIBWAIFU_CUDA_ENABLED
#include "flint/cuda/cuda_host_tensor_data.h"
#include "flint/cuda/cuda_tensor_data.h"
#endif
#ifdef LIBWAIFU_MLX_ENABLED
#include "flint/metal/metal_tensor_data.h"
#endif
#ifdef LIBWAIFU_VULKAN_ENABLED
#include "flint/vulkan/vulkan_tensor_data.h"
#endif

namespace fl {
namespace F {

namespace {

int64_t numelOf(lut::Span<const int> shape) {
  int64_t numel = 1;
  for (int size : shape) {
    if (size < 0) THROW(InvalidArg, lut::sprintf("a tensor cannot have a dimension of %d", size));
    numel *= size;
  }
  return numel;
}


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

/// `input` <(B, KVH, L, D)> with each key-value head repeated for the query heads that share it,
/// so that grouped-query attention can be done as plain attention.
Tensor expandKeyValueHeads(Operators *op, TensorView input, int numHeads) {
  int batchSize = input.getShape(0);
  int numKeyValueHeads = input.getShape(1);
  int length = input.getShape(2);
  int headDim = input.getShape(3);
  int groupSize = numHeads / numKeyValueHeads;

  TensorView expanded =
      input.unsqueeze(2).expand({batchSize, numKeyValueHeads, groupSize, length, headDim});
  Tensor output = emptyLike(expanded);
  op->copy(expanded, output);

  return output.view({batchSize, numHeads, length, headDim});
}

/// How many score elements one block of queries may hold at once, counting every head and every
/// sequence in the batch -- the whole tensor the two matmuls hand each other, not one head's
/// share of it. 128M of them is 256MB in half, and the softmax needs a second copy of it.
///
/// This is a bound on memory and nothing else. Blocking was measured on every shape SDXL runs and
/// a bigger block was faster every time, cache or no cache: the score matrix is streamed once
/// either way, and what the block size really moves is which kernel cuBLAS picks for the second
/// matmul, which is not something to steer by. So the budget is set where it stops an allocation
/// nobody meant to make, and left clear of everything that runs.
constexpr int64_t kAttentionScoreLimit = 128 * 1024 * 1024;

}  // namespace

// --- Allocation -----------------------------------------------------------------------------

std::unique_ptr<TensorData> allocate(Device device, int64_t numel, DType dtype) {
  if (numel > TensorData::MaxNumEl) {
    THROW(InvalidArg, lut::sprintf("a tensor of %d elements is too large", numel));
  }
  numel = std::max<int64_t>(numel, 1);

  switch (device.getType()) {
    case Device::kCpu:
      return op::cpu::CpuTensorData::create(numel, dtype);
#ifdef LIBWAIFU_CUDA_ENABLED
    case Device::kCuda:
      return op::cuda::CudaTensorData::create(numel, dtype);
    case Device::kCudaHost:
      return op::cuda::CudaHostTensorData::create(numel, dtype);
#endif
#ifdef LIBWAIFU_MLX_ENABLED
    case Device::kMetal:
      return op::metal::MetalTensorData::create(numel, dtype);
#endif
#ifdef LIBWAIFU_VULKAN_ENABLED
    case Device::kVulkan:
      return op::vulkan::VulkanTensorData::create(numel, dtype);
#endif
    default:
      THROW(
          NotImplemented,
          lut::sprintf("this build cannot allocate on the %s device", device.getName()));
  }
}

Tensor empty(Device device, lut::Span<const int> shape, DType dtype) {
  int64_t numel = numelOf(shape);
  return Tensor::create(std::make_shared<TensorShape>(shape), allocate(device, numel, dtype));
}

Tensor emptyLike(const TensorView &input) {
  std::vector<int> shape = input.getShape();
  return empty(input.getDevice(), shape, input.getDType());
}

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
FL_SCALAR(subFloat, float)

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

void composedAttention(
    Operators *op,
    TensorView q,
    TensorView k,
    TensorView v,
    bool causal,
    TensorView out) {
  int numHeads = q.getShape(1);
  int numKeyValueHeads = k.getShape(1);
  int queryLength = q.getShape(2);
  int keyValueLength = k.getShape(2);
  int headDim = q.getShape(3);

  Tensor expandedK, expandedV;
  if (numHeads != numKeyValueHeads) {
    expandedK = expandKeyValueHeads(op, k, numHeads);
    expandedV = expandKeyValueHeads(op, v, numHeads);
    k = expandedK;
    v = expandedV;
  }

  // Scaling both q and k keeps the scores in range for half precision.
  float scale = sqrtf(1.0f / sqrtf(1.0f * headDim));
  Tensor scaledK = mul(op, k, scale).transpose(-2, -1);

  // The score matrix is the whole cost of doing it this way: a VAE decoding a 1024 by 1024 image
  // attends over 16384 positions, and holding all of that at once is half a gigabyte before the
  // softmax needs a second copy. Since each output row depends only on its own row of scores, the
  // queries are taken a block at a time instead once that gets out of hand. The answer is the
  // same to the bit -- no running maximum is needed, because a block still sees every key.
  //
  // What a row of queries costs is a row of scores in every head of every sequence, so that is
  // what the budget is divided by.
  int64_t scoresPerQuery = static_cast<int64_t>(q.getShape(0)) * numHeads * keyValueLength;

  // What the budget affords is rounded down to a power of two rather than taken where the
  // division landed. The second matmul takes its kernel from the number of query rows handed to
  // it, and the choice is not monotone in that number: 640 rows of scores against the values
  // measured 837us where 512 measured 219, for a quarter more work. The powers of two were
  // uniformly among the good ones.
  int blockSize = queryLength;
  if (scoresPerQuery * queryLength > kAttentionScoreLimit) {
    int64_t budget = kAttentionScoreLimit / scoresPerQuery;
    blockSize = 1;
    while (blockSize * 2 <= budget) blockSize *= 2;
  }

  // The probabilities of one block of queries, against every key.
  auto probabilities = [&](TensorView blockQ, int begin, int end) {
    Tensor scores = matmul(op, mul(op, blockQ, scale), scaledK);
    // A single query attends to the whole history, so it needs no mask. A block of them is masked
    // against where it sits, not where the whole query is.
    if (causal && queryLength > 1) {
      Tensor mask = causalMask(q.getDevice(), keyValueLength)
                        .slice(0, {keyValueLength - queryLength + begin,
                                   keyValueLength - queryLength + end});
      // The mask comes in the device's *default* float type, which is not always the type the
      // scores are in: on aarch64 that default is half, and a model running in float32 has
      // float32 scores.
      mask = cast(op, mask, scores.getDType());
      scores = add(op, scores, mask);
    }
    return softmax(op, scores);
  };

  if (blockSize >= queryLength) {
    op->matmul(probabilities(q, 0, queryLength), v, out);
    return;
  }

  // Each block's answer is written into the rows of `out` it belongs to. Joining the blocks as they
  // arrive would copy everything finished so far on every pass, which is quadratic in the number
  // of blocks: the VAE's 64 of them spent 1.7ms of an 18ms call doing nothing else.
  for (int begin = 0; begin < queryLength; begin += blockSize) {
    int end = std::min(begin + blockSize, queryLength);
    Tensor blockOutput = matmul(op, probabilities(q.slice(-2, {begin, end}), begin, end), v);
    op->copy(blockOutput, out.slice(-2, {begin, end}));
  }
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
